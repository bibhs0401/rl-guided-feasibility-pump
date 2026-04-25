from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from set_packing.spp_fp_core import SPFPRunConfig, load_spp_instance
from set_packing.spp_fp_gym_env import SPPGymConfig, SetPackingFPRLEnv


def read_instance_list(file_path: str) -> List[str]:
    out: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(p)
    if not out:
        raise ValueError(f"No instance paths found in {file_path}")
    return out


def read_instance_dir(instance_dir: str, pattern: str) -> List[str]:
    base = Path(instance_dir)
    if not base.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    if not base.is_dir():
        raise ValueError(f"--instance-dir must point to a directory: {instance_dir}")
    out = sorted(str(p.resolve()) for p in base.glob(pattern) if p.is_file())
    if not out:
        raise ValueError(f"No instance files matched pattern {pattern!r} in {instance_dir}")
    return out


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device_name


class _Tee:
    """Duplicate stdout to a file."""

    def __init__(self, log_path: str):
        self._console = sys.stdout
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, data: str) -> None:
        self._console.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._console
        self._file.close()


_CSV_FIELDS = [
    "episode",
    "timesteps",
    "wall_time",
    "avg_reward",
    "avg_return",
    "feasible_rate",
    "avg_time_to_feasible_s",
    "avg_stall_events",
    "avg_iterations",
    "avg_perturbations",
    "avg_total_flips",
    "avg_final_distance",
    "avg_best_distance",
    "avg_objective_quality",
    "avg_best_objective_quality",
    "avg_fp_time_s",
    "avg_r_frac",
    "avg_r_best",
    "avg_r_feas",
    "avg_r_time",
    "avg_r_flip",
    "avg_r_stall",
    "m",
    "n",
    "p",
]

_PER_EP_CSV_FIELDS = [
    "episode",
    "timesteps",
    "wall_time",
    "instance_name",
    "m",
    "n",
    "p",
    "reward",
    "feasible",
    "failed",
    "termination_reason",
    "iterations",
    "perturbations",
    "stall_events",
    "total_flips",
    "initial_distance",
    "final_distance",
    "best_distance",
    "objective_quality",
    "best_objective_quality",
    "time_to_feasible_s",
    "fp_time_s",
    "initial_lp_solve_seconds",
    "reset_seconds",
    "r_frac",
    "r_best",
    "r_feas",
    "r_time",
    "r_flip",
    "r_stall",
]


class SPPTrainingLogger(BaseCallback):
    def __init__(
        self,
        print_every_episodes: int = 5,
        rolling_window: int = 50,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "",
        csv_log_path: str = "",
        per_episode_csv_path: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.print_every_episodes = max(1, print_every_episodes)
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self._last_checkpoint_step = 0

        self._csv_path = csv_log_path
        self._csv_file = None
        self._csv_writer = None
        self._per_ep_csv_path = per_episode_csv_path
        self._per_ep_csv_file = None
        self._per_ep_csv_writer = None

        self.episode_count = 0
        W = max(1, rolling_window)
        self._hist: Dict[str, deque] = {
            k: deque(maxlen=W)
            for k in [
                "reward",
                "feasible",
                "time_to_feasible",
                "stall_events",
                "iterations",
                "perturbations",
                "total_flips",
                "final_distance",
                "best_distance",
                "objective_quality",
                "best_objective_quality",
                "fp_time",
                "r_frac",
                "r_best",
                "r_feas",
                "r_time",
                "r_flip",
                "r_stall",
            ]
        }
        self._last_m = 0
        self._last_n = 0
        self._last_p = 1
        self._ep_acc: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {k: 0.0 for k in ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]}
        )

    def _avg_hist(self, key: str) -> float:
        h = self._hist[key]
        return sum(h) / max(1, len(h))

    def _log_aggregates(self) -> None:
        if not self._hist["reward"]:
            return
        a_rew = self._avg_hist("reward")
        a_feas = self._avg_hist("feasible")
        a_ttf = self._avg_hist("time_to_feasible")
        a_stall = self._avg_hist("stall_events")
        a_iter = self._avg_hist("iterations")
        a_pert = self._avg_hist("perturbations")
        a_flips = self._avg_hist("total_flips")
        a_fdist = self._avg_hist("final_distance")
        a_bdist = self._avg_hist("best_distance")
        a_obj = self._avg_hist("objective_quality")
        a_best_obj = self._avg_hist("best_objective_quality")
        a_time = self._avg_hist("fp_time")
        a_rfrac = self._avg_hist("r_frac")
        a_rbest = self._avg_hist("r_best")
        a_rfeas = self._avg_hist("r_feas")
        a_rtime = self._avg_hist("r_time")
        a_rflip = self._avg_hist("r_flip")
        a_rstall = self._avg_hist("r_stall")

        print(
            f"\n[spp-train] ep={self.episode_count:>6d} ts={self.num_timesteps:>8d} m={self._last_m} n={self._last_n} p={self._last_p}\n"
            f"  avg_return={a_rew:+.4f} feasible_rate={a_feas:.3f} time_to_feasible={a_ttf:.2f}s\n"
            f"  iters={a_iter:.1f} perturbations={a_pert:.1f} flips={a_flips:.1f} stalls={a_stall:.2f}\n"
            f"  distance final={a_fdist:.4f} best={a_bdist:.4f} objective={a_obj:.4f} best_obj={a_best_obj:.4f} fp_time={a_time:.2f}s\n"
            f"  reward split: r_frac={a_rfrac:+.4f} r_best={a_rbest:+.4f} r_feas={a_rfeas:+.4f} "
            f"r_time={a_rtime:+.4f} r_flip={a_rflip:+.4f} r_stall={a_rstall:+.4f}",
            flush=True,
        )

        if self._csv_writer is not None:
            self._csv_writer.writerow(
                {
                    "episode": self.episode_count,
                    "timesteps": self.num_timesteps,
                    "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "avg_reward": round(a_rew, 6),
                    "avg_return": round(a_rew, 6),
                    "feasible_rate": round(a_feas, 4),
                    "avg_time_to_feasible_s": round(a_ttf, 4),
                    "avg_stall_events": round(a_stall, 3),
                    "avg_iterations": round(a_iter, 2),
                    "avg_perturbations": round(a_pert, 2),
                    "avg_total_flips": round(a_flips, 2),
                    "avg_final_distance": round(a_fdist, 4),
                    "avg_best_distance": round(a_bdist, 4),
                    "avg_objective_quality": round(a_obj, 4),
                    "avg_best_objective_quality": round(a_best_obj, 4),
                    "avg_fp_time_s": round(a_time, 3),
                    "avg_r_frac": round(a_rfrac, 6),
                    "avg_r_best": round(a_rbest, 6),
                    "avg_r_feas": round(a_rfeas, 6),
                    "avg_r_time": round(a_rtime, 6),
                    "avg_r_flip": round(a_rflip, 6),
                    "avg_r_stall": round(a_rstall, 6),
                    "m": self._last_m,
                    "n": self._last_n,
                    "p": self._last_p,
                }
            )
            self._csv_file.flush()

    def _on_training_start(self) -> None:
        if self._csv_path:
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = _csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
            self._csv_writer.writeheader()
            self._csv_file.flush()
        if self._per_ep_csv_path:
            self._per_ep_csv_file = open(self._per_ep_csv_path, "w", newline="", encoding="utf-8")
            self._per_ep_csv_writer = _csv.DictWriter(self._per_ep_csv_file, fieldnames=_PER_EP_CSV_FIELDS)
            self._per_ep_csv_writer.writeheader()
            self._per_ep_csv_file.flush()
        super()._on_training_start()

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        if self._per_ep_csv_file is not None:
            self._per_ep_csv_file.flush()
            self._per_ep_csv_file.close()
        super()._on_training_end()

    def _on_step(self) -> bool:
        if (
            self.checkpoint_every > 0
            and self.checkpoint_dir
            and self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_every
        ):
            ckpt_path = os.path.join(self.checkpoint_dir, f"ppo_step_{self.num_timesteps}")
            self.model.save(ckpt_path)
            print(f"[checkpoint] saved -> {ckpt_path}.zip", flush=True)
            self._last_checkpoint_step = self.num_timesteps

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            for k in ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]:
                self._ep_acc[env_idx][k] += float(info.get(k, 0.0))
            if not done:
                continue
            self.episode_count += 1

            ep_raw = info.get("episode")
            if isinstance(ep_raw, dict) and "r" in ep_raw:
                ep_reward = float(ep_raw["r"])
            else:
                ep_reward = float(info.get("reward", 0.0))
            feasible = int(bool(info.get("feasible_found", False)))
            failed = int(bool(info.get("failed", False)))
            time_to_feasible = (
                float(info.get("time_to_feasible_seconds"))
                if info.get("time_to_feasible_seconds") is not None
                else 0.0
            )
            acc = self._ep_acc.pop(env_idx, {})

            self._hist["reward"].append(ep_reward)
            self._hist["feasible"].append(feasible)
            self._hist["time_to_feasible"].append(time_to_feasible if feasible else 0.0)
            self._hist["stall_events"].append(int(info.get("stall_events", 0)))
            self._hist["iterations"].append(int(info.get("iterations", 0)))
            self._hist["perturbations"].append(int(info.get("perturbation_events", 0)))
            self._hist["total_flips"].append(int(info.get("total_flips", 0)))
            self._hist["final_distance"].append(float(info.get("current_distance", 0.0)))
            self._hist["best_distance"].append(float(info.get("best_distance", 0.0)))
            self._hist["objective_quality"].append(float(info.get("objective_quality", 0.0)))
            self._hist["best_objective_quality"].append(float(info.get("best_objective_quality", 0.0)))
            self._hist["fp_time"].append(float(info.get("elapsed_seconds", 0.0)))
            for k in ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]:
                self._hist[k].append(acc.get(k, 0.0))

            self._last_m = int(info.get("m", 0))
            self._last_n = int(info.get("n", 0))
            self._last_p = int(info.get("p", 1))

            if self._per_ep_csv_writer is not None:
                self._per_ep_csv_writer.writerow(
                    {
                        "episode": self.episode_count,
                        "timesteps": self.num_timesteps,
                        "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "instance_name": str(info.get("instance_name", "")),
                        "m": self._last_m,
                        "n": self._last_n,
                        "p": self._last_p,
                        "reward": round(ep_reward, 6),
                        "feasible": feasible,
                        "failed": failed,
                        "termination_reason": str(info.get("termination_reason", "")),
                        "iterations": int(info.get("iterations", 0)),
                        "perturbations": int(info.get("perturbation_events", 0)),
                        "stall_events": int(info.get("stall_events", 0)),
                        "total_flips": int(info.get("total_flips", 0)),
                        "initial_distance": round(float(info.get("initial_distance", 0.0) or 0.0), 6),
                        "final_distance": round(float(info.get("current_distance", 0.0)), 6),
                        "best_distance": round(float(info.get("best_distance", 0.0)), 6),
                        "objective_quality": round(float(info.get("objective_quality", 0.0)), 6),
                        "best_objective_quality": round(float(info.get("best_objective_quality", 0.0)), 6),
                        "time_to_feasible_s": round(time_to_feasible, 3) if feasible else "",
                        "fp_time_s": round(float(info.get("elapsed_seconds", 0.0)), 3),
                        "initial_lp_solve_seconds": round(float(info.get("initial_lp_solve_seconds", 0.0) or 0.0), 3),
                        "reset_seconds": round(float(info.get("reset_seconds", 0.0) or 0.0), 3),
                        "r_frac": round(acc.get("r_frac", 0.0), 6),
                        "r_best": round(acc.get("r_best", 0.0), 6),
                        "r_feas": round(acc.get("r_feas", 0.0), 6),
                        "r_time": round(acc.get("r_time", 0.0), 6),
                        "r_flip": round(acc.get("r_flip", 0.0), 6),
                        "r_stall": round(acc.get("r_stall", 0.0), 6),
                    }
                )
                self._per_ep_csv_file.flush()

            if self.episode_count % self.print_every_episodes == 0:
                self._log_aggregates()
        return True


def make_env_fn(
    instance_paths: List[str],
    seed: int,
    max_iterations: int,
    time_limit: float,
    initial_lp_time_limit: float | None,
    stall_threshold: int,
    max_stalls: int,
    cplex_threads: int,
    max_reset_resamples: int,
) -> Callable[[], SetPackingFPRLEnv]:
    def _init():
        fp_cfg = SPFPRunConfig(
            max_iterations=max_iterations,
            time_limit=time_limit,
            initial_lp_time_limit=initial_lp_time_limit,
            stall_threshold=stall_threshold,
            max_stalls=max_stalls,
            cplex_threads=cplex_threads,
        )
        env_cfg = SPPGymConfig(
            instance_paths=instance_paths,
            fp_config=fp_cfg,
            max_reset_resamples=max_reset_resamples,
            seed=seed,
        )
        return SetPackingFPRLEnv(env_cfg)

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on set-packing FP env.")
    parser.add_argument(
        "--instance-list",
        default="",
        help="Text file with one .npz or .lp path per line.",
    )
    parser.add_argument(
        "--instance-dir",
        default="",
        help="Directory to scan for instances (alternative to --instance-list).",
    )
    parser.add_argument(
        "--instance-pattern",
        default="*.lp",
        help="Glob pattern used with --instance-dir, e.g. '*.lp' or '*.npz'.",
    )
    parser.add_argument("--pool-size", type=int, default=0, help="Use first N instances (0=all)")
    parser.add_argument("--run-dir", default="runs/spp_fp_ppo")
    parser.add_argument("--run-name", default="trial_1")
    parser.add_argument("--save-name", default="ppo_spp_fp_model")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--initial-lp-time-limit", type=float, default=60.0)
    parser.add_argument("--initial-lp-optimal", action="store_true")
    parser.add_argument("--stall-threshold", type=int, default=1)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--max-reset-resamples", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--print-every-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    initial_lp_limit = None if args.initial_lp_optimal else args.initial_lp_time_limit
    if bool(args.instance_list) == bool(args.instance_dir):
        raise ValueError("Provide exactly one of --instance-list or --instance-dir.")
    if args.instance_list:
        instance_paths = read_instance_list(args.instance_list)
    else:
        instance_paths = read_instance_dir(args.instance_dir, args.instance_pattern)
    if args.pool_size and args.pool_size < len(instance_paths):
        instance_paths = instance_paths[: args.pool_size]

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    probe = load_spp_instance(instance_paths[0])
    shape_tag = f"m{probe.m}_n{probe.n}"
    csv_log_path = str(run_dir / f"training_log_{shape_tag}_{timestamp}.csv")
    per_ep_csv_path = str(run_dir / f"training_episodes_{shape_tag}_{timestamp}.csv")

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else str(run_dir / "checkpoints")
    if args.checkpoint_every > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if args.log_file:
        log_file = args.log_file
    elif args.tensorboard:
        log_file = str(run_dir / f"train_{shape_tag}_{timestamp}.log")
    else:
        log_file = ""
    _tee: Optional[_Tee] = _Tee(log_file) if log_file else None

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "resolved_device": device,
                "num_instances": len(instance_paths),
                "shape_tag": shape_tag,
                "timestamp": timestamp,
                "csv_log_path": csv_log_path,
                "per_ep_csv_path": per_ep_csv_path,
            },
            f,
            indent=2,
        )

    if args.check_env:
        check_env(
            make_env_fn(
                instance_paths=instance_paths,
                seed=args.seed,
                max_iterations=args.max_iterations,
                time_limit=args.time_limit,
                initial_lp_time_limit=initial_lp_limit,
                stall_threshold=args.stall_threshold,
                max_stalls=args.max_stalls,
                cplex_threads=args.cplex_threads,
                max_reset_resamples=args.max_reset_resamples,
            )().unwrapped,
            warn=True,
        )

    env_kwargs = dict(
        instance_paths=instance_paths,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        initial_lp_time_limit=initial_lp_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
        max_reset_resamples=args.max_reset_resamples,
    )
    if args.num_envs == 1:
        vec_env = DummyVecEnv([make_env_fn(seed=args.seed, **env_kwargs)])
    else:
        vec_env = SubprocVecEnv(
            [make_env_fn(seed=args.seed + i, **env_kwargs) for i in range(args.num_envs)],
            start_method="fork",
        )
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
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
        tensorboard_log=str(run_dir / "tb_logs") if args.tensorboard else None,
    )
    callback = CallbackList(
        [
            SPPTrainingLogger(
                print_every_episodes=args.print_every_episodes,
                rolling_window=50,
                checkpoint_every=args.checkpoint_every,
                checkpoint_dir=checkpoint_dir,
                csv_log_path=csv_log_path,
                per_episode_csv_path=per_ep_csv_path,
            )
        ]
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    final_model = str(run_dir / f"{args.save_name}_{shape_tag}_{timestamp}")
    model.save(final_model)
    vec_env.close()
    print(f"Saved model: {final_model}.zip")
    print(f"Training log CSV: {csv_log_path}")
    print(f"Per-episode CSV: {per_ep_csv_path}")
    if log_file:
        print(f"Stdout log: {log_file}")
    if _tee is not None:
        _tee.close()


if __name__ == "__main__":
    main()
