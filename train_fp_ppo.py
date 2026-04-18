from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Callable, List

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from fp_gym_env import FeasibilityPumpRLEnv, FPGymConfig
from mmp_fp_core import FPRunConfig


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def read_instance_list(file_path: str) -> List[str]:
    """
    Read a plain text file containing one .npz path per line.
    Blank lines are ignored.
    """
    paths: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if path:
                paths.append(path)

    if not paths:
        raise ValueError(f"No instance paths found in: {file_path}")
    return paths


def resolve_device(device_name: str) -> str:
    """
    Resolve SB3/PyTorch device.
    """
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device_name


def make_env_fn(
    instance_paths: List[str],
    seed: int,
    max_iterations: int,
    time_limit: float,
    stall_threshold: int,
    max_stalls: int,
    cplex_threads: int,
    max_reset_resamples: int,
) -> Callable[[], FeasibilityPumpRLEnv]:
    """
    Factory function for one environment instance.
    """

    def _init():
        fp_cfg = FPRunConfig(
            max_iterations=max_iterations,
            time_limit=time_limit,
            stall_threshold=stall_threshold,
            max_stalls=max_stalls,
            cplex_threads=cplex_threads,
        )

        env_cfg = FPGymConfig(
            instance_paths=instance_paths,
            fp_config=fp_cfg,
            max_reset_resamples=max_reset_resamples,
            seed=seed,
        )

        env = FeasibilityPumpRLEnv(env_cfg)
        env = Monitor(env)
        return env

    return _init


# -----------------------------------------------------------------------------
# Training callback
# -----------------------------------------------------------------------------
class PPOTrainingLogger(BaseCallback):
    """
    Simple training logger for SB3.

    Tracks recent:
    - episode rewards
    - feasible solve rate
    - final distance
    - FP time
    """

    def __init__(self, print_every_episodes: int = 5, rolling_window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.print_every_episodes = max(1, print_every_episodes)
        self.rolling_window = max(1, rolling_window)

        self.episode_count = 0
        self.reward_hist = deque(maxlen=self.rolling_window)
        self.feasible_hist = deque(maxlen=self.rolling_window)
        self.distance_hist = deque(maxlen=self.rolling_window)
        self.time_hist = deque(maxlen=self.rolling_window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_count += 1

            ep_reward = None
            if "episode" in info and isinstance(info["episode"], dict):
                ep_reward = float(info["episode"].get("r", 0.0))
            else:
                ep_reward = float(info.get("reward", 0.0))

            feasible = int(bool(info.get("feasible_found", False)))
            final_distance = float(info.get("current_distance", 0.0))
            elapsed_seconds = float(info.get("elapsed_seconds", 0.0))

            self.reward_hist.append(ep_reward)
            self.feasible_hist.append(feasible)
            self.distance_hist.append(final_distance)
            self.time_hist.append(elapsed_seconds)

            if self.episode_count % self.print_every_episodes == 0:
                avg_reward = sum(self.reward_hist) / max(1, len(self.reward_hist))
                avg_feasible = sum(self.feasible_hist) / max(1, len(self.feasible_hist))
                avg_distance = sum(self.distance_hist) / max(1, len(self.distance_hist))
                avg_time = sum(self.time_hist) / max(1, len(self.time_hist))

                print(
                    f"[train] episodes={self.episode_count} "
                    f"timesteps={self.num_timesteps} "
                    f"avg_reward={avg_reward:.4f} "
                    f"feasible_rate={avg_feasible:.3f} "
                    f"avg_final_distance={avg_distance:.4f} "
                    f"avg_fp_time={avg_time:.2f}s",
                    flush=True,
                )

        return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO on FeasibilityPumpRLEnv.")

    # Data
    parser.add_argument(
        "--instance-list",
        required=True,
        help="Text file containing one .npz instance path per line.",
    )

    # Run / output
    parser.add_argument("--run-dir", default="runs/fp_ppo_step3")
    parser.add_argument("--run-name", default="trial_1")
    parser.add_argument("--save-name", default="ppo_fp_model")

    # Environment / FP
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--stall-threshold", type=int, default=3)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--max-reset-resamples", type=int, default=5)

    # PPO
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    # Parallelism / device
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=10)

    # Debug / checks
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--print-every-episodes", type=int, default=5)

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    instance_paths = read_instance_list(args.instance_list)

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / args.save_name
    config_path = run_dir / "run_config.json"

    device = resolve_device(args.device)

    # Save run configuration
    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    run_config["num_instances"] = len(instance_paths)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[info] loaded {len(instance_paths)} instances", flush=True)
    print(f"[info] run directory: {run_dir}", flush=True)
    print(f"[info] device: {device}", flush=True)

    # -------------------------------------------------------------------------
    # Single probe env for env checking if needed
    # -------------------------------------------------------------------------
    if args.check_env:
        probe_env = make_env_fn(
            instance_paths=instance_paths,
            seed=args.seed,
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_threshold=args.stall_threshold,
            max_stalls=args.max_stalls,
            cplex_threads=args.cplex_threads,
            max_reset_resamples=args.max_reset_resamples,
        )()
        print("[info] running check_env ...", flush=True)
        check_env(probe_env.unwrapped, warn=True)
        probe_env.close()
        print("[info] check_env passed", flush=True)

    # -------------------------------------------------------------------------
    # Vectorized env
    # -------------------------------------------------------------------------
    if args.num_envs == 1:
        env = DummyVecEnv(
            [
                make_env_fn(
                    instance_paths=instance_paths,
                    seed=args.seed,
                    max_iterations=args.max_iterations,
                    time_limit=args.time_limit,
                    stall_threshold=args.stall_threshold,
                    max_stalls=args.max_stalls,
                    cplex_threads=args.cplex_threads,
                    max_reset_resamples=args.max_reset_resamples,
                )
            ]
        )
    else:
        env_fns = []
        for i in range(args.num_envs):
            env_fns.append(
                make_env_fn(
                    instance_paths=instance_paths,
                    seed=args.seed + i,
                    max_iterations=args.max_iterations,
                    time_limit=args.time_limit,
                    stall_threshold=args.stall_threshold,
                    max_stalls=args.max_stalls,
                    cplex_threads=args.cplex_threads,
                    max_reset_resamples=args.max_reset_resamples,
                )
            )
        env = SubprocVecEnv(env_fns, start_method="fork")

    env = VecMonitor(env)

    # -------------------------------------------------------------------------
    # PPO model
    # -------------------------------------------------------------------------
    # MultiInputPolicy is required because our observation is a Dict.
    # PPO supports MultiDiscrete action spaces, so this is the correct choice.
    # -------------------------------------------------------------------------
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Callback
    # -------------------------------------------------------------------------
    callback = CallbackList(
        [
            PPOTrainingLogger(
                print_every_episodes=args.print_every_episodes,
                rolling_window=50,
                verbose=0,
            )
        ]
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("[info] starting PPO training ...", flush=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"[info] saving model to: {model_path}.zip", flush=True)
    model.save(str(model_path))

    env.close()
    print("[info] training complete", flush=True)


if __name__ == "__main__":
    main()