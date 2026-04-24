from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

from fp_baseline_spp import FPConfig
from spp_model import find_instance_files
from spp_rl_env import SPPRLEnvConfig, SPPFeasibilityPumpEnv, heuristic_action_from_observation


AGGREGATE_LOG_FIELDS = [
    "episode",
    "timesteps",
    "wall_time_seconds",
    "window_size",
    "avg_return",
    "success_rate",
    "avg_stalls",
    "avg_rl_interventions",
    "avg_iterations",
    "avg_runtime_seconds",
    "avg_final_violation",
    "avg_num_violated_constraints",
    "avg_final_objective",
    "failures",
]


def _load_paths(args: argparse.Namespace) -> list[str]:
    if args.instances:
        paths = [str(Path(p).resolve()) for p in args.instances]
    else:
        paths = find_instance_files([args.instance_dir])
    if args.max_instances:
        paths = paths[: args.max_instances]
    if not paths:
        raise ValueError("No .npz or .lp set-packing instances found for training.")
    return paths


def create_env(instance_paths: Sequence[str], args: argparse.Namespace) -> SPPFeasibilityPumpEnv:
    fp_cfg = FPConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_length=args.stall_length,
        random_seed=args.seed,
        cplex_threads=args.cplex_threads,
        verbose=args.verbose,
    )
    env_cfg = SPPRLEnvConfig(
        instance_paths=list(instance_paths),
        fp_config=fp_cfg,
        seed=args.seed,
        continuation_steps_after_action=args.continuation_steps,
    )
    return SPPFeasibilityPumpEnv(env_cfg)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def build_aggregate_logger_class(base_callback_cls):
    class SPPAggregateLogger(base_callback_cls):
        def __init__(
            self,
            csv_path: str | Path,
            print_every_episodes: int = 5,
            rolling_window: int = 20,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.csv_path = Path(csv_path)
            self.print_every_episodes = max(1, int(print_every_episodes))
            self.history = {
                "return": deque(maxlen=max(1, int(rolling_window))),
                "success": deque(maxlen=max(1, int(rolling_window))),
                "stalls": deque(maxlen=max(1, int(rolling_window))),
                "interventions": deque(maxlen=max(1, int(rolling_window))),
                "iterations": deque(maxlen=max(1, int(rolling_window))),
                "runtime": deque(maxlen=max(1, int(rolling_window))),
                "violation": deque(maxlen=max(1, int(rolling_window))),
                "violated_constraints": deque(maxlen=max(1, int(rolling_window))),
                "objective": deque(maxlen=max(1, int(rolling_window))),
                "failure": deque(maxlen=max(1, int(rolling_window))),
            }
            self.episode_count = 0
            self.start_time = time.time()
            self._file = None
            self._writer = None

        def _on_training_start(self) -> None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=AGGREGATE_LOG_FIELDS)
            self._writer.writeheader()
            self._file.flush()
            print(f"[train-log] aggregate CSV: {self.csv_path.resolve()}", flush=True)

        def _row(self) -> dict:
            window_size = len(self.history["return"])
            return {
                "episode": self.episode_count,
                "timesteps": self.num_timesteps,
                "wall_time_seconds": f"{time.time() - self.start_time:.3f}",
                "window_size": window_size,
                "avg_return": f"{_mean(self.history['return']):.6f}",
                "success_rate": f"{_mean(self.history['success']):.6f}",
                "avg_stalls": f"{_mean(self.history['stalls']):.6f}",
                "avg_rl_interventions": f"{_mean(self.history['interventions']):.6f}",
                "avg_iterations": f"{_mean(self.history['iterations']):.6f}",
                "avg_runtime_seconds": f"{_mean(self.history['runtime']):.6f}",
                "avg_final_violation": f"{_mean(self.history['violation']):.10g}",
                "avg_num_violated_constraints": f"{_mean(self.history['violated_constraints']):.6f}",
                "avg_final_objective": f"{_mean(self.history['objective']):.10g}",
                "failures": int(sum(self.history["failure"])),
            }

        def _write_and_print(self) -> None:
            row = self._row()
            if self._writer is not None:
                self._writer.writerow(row)
                self._file.flush()
            print(
                "[train-log] "
                f"ep={row['episode']} steps={row['timesteps']} "
                f"avg_return={float(row['avg_return']):+.4f} "
                f"success={float(row['success_rate']):.3f} "
                f"stalls={float(row['avg_stalls']):.2f} "
                f"rl_int={float(row['avg_rl_interventions']):.2f} "
                f"iters={float(row['avg_iterations']):.1f} "
                f"viol={float(row['avg_final_violation']):.4g}",
                flush=True,
            )

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if not done:
                    continue
                self.episode_count += 1
                self.history["return"].append(float(info.get("average_return", info.get("reward", 0.0)) or 0.0))
                self.history["success"].append(float(bool(info.get("success", 0))))
                self.history["stalls"].append(float(info.get("num_stalls", 0) or 0))
                self.history["interventions"].append(float(info.get("num_rl_interventions", 0) or 0))
                self.history["iterations"].append(float(info.get("iterations", 0) or 0))
                self.history["runtime"].append(float(info.get("runtime_seconds", 0.0) or 0.0))
                self.history["violation"].append(float(info.get("final_violation", 0.0) or 0.0))
                self.history["violated_constraints"].append(
                    float(info.get("num_violated_constraints", 0) or 0)
                )
                self.history["objective"].append(float(info.get("final_objective", 0.0) or 0.0))
                self.history["failure"].append(0.0 if bool(info.get("success", 0)) else 1.0)
                if self.episode_count % self.print_every_episodes == 0:
                    self._write_and_print()
            return True

        def _on_training_end(self) -> None:
            if self.episode_count and self.episode_count % self.print_every_episodes != 0:
                self._write_and_print()
            if self._file is not None:
                self._file.close()

    return SPPAggregateLogger


def train_or_create_policy(
    instance_paths: Sequence[str],
    out_dir: str | Path = "results/rl_training",
    algorithm: str = "PPO",
    timesteps: int = 500,
    seed: int = 0,
    max_iterations: int = 80,
    time_limit: float = 10.0,
    stall_length: int = 3,
    continuation_steps: int | None = None,
    cplex_threads: int = 1,
    log_every_episodes: int = 5,
    rolling_window: int = 20,
    aggregate_log_csv: str | Path | None = None,
    verbose: bool = False,
    allow_heuristic_fallback: bool = True,
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / f"{algorithm.lower()}_spp_fp_policy"

    class _Args:
        pass

    args = _Args()
    args.max_iterations = max_iterations
    args.time_limit = time_limit
    args.stall_length = stall_length
    args.cplex_threads = cplex_threads
    args.seed = seed
    args.verbose = verbose
    args.continuation_steps = continuation_steps

    try:
        from stable_baselines3 import A2C, PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as exc:
        if not allow_heuristic_fallback:
            raise RuntimeError(
                "stable_baselines3 is not installed. Install it to train PPO/A2C, "
                "or rerun with heuristic fallback enabled."
            ) from exc
        fallback_path = out / "heuristic_spp_fp_policy.json"
        fallback_path.write_text(
            json.dumps(
                {
                    "policy_type": "heuristic_stall_perturbation",
                    "algorithm_requested": algorithm,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "stable_baselines3_not_installed",
                    "action_space": {
                        "0": "skip",
                        "1": "flip_1_percent",
                        "2": "flip_5_percent",
                        "3": "flip_10_percent",
                        "4": "flip_20_percent",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "[train] stable_baselines3 is not installed; wrote heuristic fallback policy "
            f"to {fallback_path}"
        )
        return str(fallback_path.resolve())

    def make_env():
        return Monitor(create_env(instance_paths, args))

    env = DummyVecEnv([make_env])
    algo_name = algorithm.upper()
    model_cls = PPO if algo_name == "PPO" else A2C
    model = model_cls("MlpPolicy", env, verbose=1 if verbose else 0, seed=seed)
    aggregate_log_path = Path(aggregate_log_csv) if aggregate_log_csv else out / "training_aggregate_logs.csv"
    AggregateLogger = build_aggregate_logger_class(BaseCallback)
    callbacks = CallbackList(
        [
            AggregateLogger(
                aggregate_log_path,
                print_every_episodes=log_every_episodes,
                rolling_window=rolling_window,
            )
        ]
    )
    model.learn(total_timesteps=int(timesteps), callback=callbacks)
    model.save(str(policy_path))
    saved = str(policy_path.with_suffix(".zip").resolve())
    print(f"[train] saved {algo_name} model to {saved}")
    return saved


def smoke_test_policy(policy_path: str, instance_paths: Sequence[str], args: argparse.Namespace) -> None:
    env = create_env(instance_paths, args)
    obs, info = env.reset()
    total_return = 0.0
    for _ in range(args.smoke_steps):
        if policy_path.endswith(".json"):
            action = heuristic_action_from_observation(obs)
        else:
            try:
                from stable_baselines3 import A2C, PPO
            except ModuleNotFoundError:
                action = heuristic_action_from_observation(obs)
            else:
                model_cls = PPO if args.algorithm.upper() == "PPO" else A2C
                model = model_cls.load(policy_path)
                action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_return += reward
        if terminated or truncated:
            break
    print(
        "[train] smoke episode "
        f"instance={info.get('instance_name')} return={total_return:.4f} "
        f"success={info.get('success')} stalls={info.get('num_stalls')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/A2C for stall-only SPP FP perturbations.")
    parser.add_argument("--instance-dir", default=".", help="Directory containing .npz or .lp instances.")
    parser.add_argument("--instances", nargs="*", default=None, help="Explicit instance paths.")
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--algorithm", choices=["PPO", "A2C"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--out-dir", default="results/rl_training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--stall-length", type=int, default=3)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--continuation-steps", type=int, default=None)
    parser.add_argument("--log-every-episodes", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--aggregate-log-csv",
        default=None,
        help="CSV path for online aggregate training logs. Defaults to OUT_DIR/training_aggregate_logs.csv.",
    )
    parser.add_argument("--no-heuristic-fallback", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _load_paths(args)
    print(f"[train] number of instances found: {len(paths)}")
    for path in paths:
        print(f"[train] instance: {Path(path).name}")
    policy_path = train_or_create_policy(
        paths,
        out_dir=args.out_dir,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        seed=args.seed,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_length=args.stall_length,
        continuation_steps=args.continuation_steps,
        cplex_threads=args.cplex_threads,
        log_every_episodes=args.log_every_episodes,
        rolling_window=args.rolling_window,
        aggregate_log_csv=args.aggregate_log_csv,
        verbose=args.verbose,
        allow_heuristic_fallback=not args.no_heuristic_fallback,
    )
    smoke_test_policy(policy_path, paths, args)
    print(f"[train] policy path: {policy_path}")


if __name__ == "__main__":
    main()
