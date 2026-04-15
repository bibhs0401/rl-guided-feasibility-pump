from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from fp_ppo import (
    DEFAULT_CPLEX_THREADS,
    DEFAULT_K_CHOICES,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_TIME_LIMIT,
    FeasibilityPumpKEnv,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO to choose perturbation size k (flip count) each FP step on sparse .npz instances."
    )
    parser.add_argument(
        "--instances",
        required=True,
        help="Glob pattern for sparse .npz instances, for example C:/.../instances/*.npz",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs/train_ppo",
        help="Base directory where each run gets its own timestamped folder.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. If omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional explicit model path/prefix. If omitted, model is saved under the run folder.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20000,
        help="Total PPO timesteps.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum FP distance-model solves per episode.",
    )
    parser.add_argument(
        "--k-choices",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_CHOICES),
        help="Comma-separated perturbation sizes used as discrete actions, e.g. 1,2,5,10,20,50.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Per-episode time limit in seconds.",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=DEFAULT_STALL_THRESHOLD,
        help="How many consecutive no-change FP steps count as a stall in the observation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="PPO rollout size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO batch size.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per update.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Gymnasium/SB3 environment checks before training.",
    )
    parser.add_argument(
        "--tensorboard-log",
        default=None,
        help="Optional TensorBoard log directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cuda", "cpu", "auto"],
        # Bug fix 6: was "cuda" — raised RuntimeError on any CPU-only machine.
        # "auto" selects CUDA when available and falls back to CPU silently.
        help="Torch device for PPO. 'auto' picks CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Helpful on Linux servers for long runs.",
    )
    parser.add_argument(
        "--progress-log-steps",
        type=int,
        default=100,
        help="How often to log training progress in environment steps.",
    )
    parser.add_argument(
        "--dashboard-window",
        type=int,
        default=100,
        help="Rolling episode window used by dashboard metrics.",
    )
    parser.add_argument(
        "--cplex-threads",
        type=int,
        default=DEFAULT_CPLEX_THREADS,
        help="Number of CPLEX threads per solve. Use 0 for automatic.",
    )
    parser.add_argument(
        "--curve-csv",
        default=None,
        help="Optional path for per-checkpoint learning curve CSV. Defaults to <run_dir>/learning_curve_<m,n>.csv.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments for training (uses SubprocVecEnv when >1).",
    )
    parser.add_argument(
        "--vec-start-method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method for SubprocVecEnv. 'auto' uses fork on POSIX and spawn on Windows.",
    )
    return parser.parse_args()


def setup_logging(level_name: str, log_file: str | None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and make sure your GPU driver is working."
        )

    return device_name


def parse_k_choices(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))

    if not values:
        raise ValueError("--k-choices must include at least one integer")
    if any(k <= 0 for k in values):
        raise ValueError("--k-choices must contain strictly positive integers")
    return tuple(values)


def make_env_factory(
    instance_paths: list[str],
    k_choices: tuple[int, ...],
    max_iterations: int,
    time_limit: float,
    stall_threshold: int,
    cplex_threads: int,
):
    def _init():
        env = FeasibilityPumpKEnv(
            instance_paths=instance_paths,
            k_choices=k_choices,
            max_iterations=max_iterations,
            time_limit=time_limit,
            stall_threshold=stall_threshold,
            cplex_threads=cplex_threads,
        )
        return Monitor(env)

    return _init


def resolve_vec_start_method(method: str) -> str:
    if method != "auto":
        return method
    return "fork" if os.name == "posix" else "spawn"


def _read_scalar(value) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(float(array.item()))
    return int(float(array.reshape(-1)[0]))


def infer_instance_shape_tag(instance_paths: list[str]) -> str:
    if not instance_paths:
        return "mNA_nNA"

    m_values: list[int] = []
    n_values: list[int] = []
    for instance_path in instance_paths:
        try:
            with np.load(instance_path, allow_pickle=False) as archive:
                m_values.append(_read_scalar(archive["m"]))
                n_values.append(_read_scalar(archive["n"]))
        except Exception:
            logger.warning("Could not read m/n metadata from %s", instance_path)

    if not m_values or not n_values:
        return "mNA_nNA"

    min_m, max_m = min(m_values), max(m_values)
    min_n, max_n = min(n_values), max(n_values)
    if min_m == max_m and min_n == max_n:
        return f"m{min_m}_n{min_n}"
    return f"m{min_m}-{max_m}_n{min_n}-{max_n}"


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every_steps: int, rolling_window: int, curve_csv: str | None = None):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_every_steps = max(1, log_every_steps)
        self.next_progress_log = self.log_every_steps
        self.rolling_window = max(1, rolling_window)
        self.curve_csv = curve_csv
        self._curve_rows: list[dict] = []
        self.success_history: deque[float] = deque(maxlen=self.rolling_window)
        self.return_history: deque[float] = deque(maxlen=self.rolling_window)
        self.final_distance_history: deque[float] = deque(maxlen=self.rolling_window)
        self.failure_history: deque[float] = deque(maxlen=self.rolling_window)
        self.steps_to_success_history: deque[float] = deque(maxlen=self.rolling_window)
        self.stall_recovery_history: deque[float] = deque(maxlen=self.rolling_window)
        self.flips_per_step_history: deque[float] = deque(maxlen=self.rolling_window)
        self.solve_seconds_history: deque[float] = deque(maxlen=self.rolling_window)

    def _on_training_start(self) -> None:
        logger.info("SB3 training loop started")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done or not info:
                continue

            monitor_episode = info.get("episode")
            if isinstance(monitor_episode, dict) and "r" in monitor_episode:
                self.return_history.append(float(monitor_episode["r"]))

            integer_found = bool(info.get("integer_found", False))
            failed = bool(info.get("failed", False))
            final_distance = float(info.get("distance", 0.0))
            iterations = float(info.get("iterations", 0.0))
            perturb_steps = float(info.get("perturb_steps", 0.0))
            total_flips = float(info.get("total_flips", 0.0))
            stall_recoveries = float(info.get("stall_recoveries", 0.0))
            stall_events = float(info.get("stall_events", 0.0))
            last_solve_seconds = float(info.get("last_distance_solve_seconds", 0.0))

            self.success_history.append(1.0 if integer_found else 0.0)
            self.final_distance_history.append(final_distance)
            self.failure_history.append(1.0 if failed else 0.0)
            self.solve_seconds_history.append(last_solve_seconds)

            if integer_found:
                self.steps_to_success_history.append(iterations)
            if stall_events > 0:
                self.stall_recovery_history.append(stall_recoveries / stall_events)
            if iterations > 0:
                self.flips_per_step_history.append(total_flips / iterations)
            elif perturb_steps > 0:
                self.flips_per_step_history.append(total_flips / perturb_steps)

            instance_name = Path(str(info.get("instance_path", "unknown"))).name
            env_episode = info.get("env_episode", "?")
            logger.info(
                "Episode %s finished: instance=%s iterations=%s decisions=%s integer_found=%s "
                "failed=%s distance=%.4f load=%.2fs reset=%.2fs last_solve=%.2fs elapsed=%.2fs",
                env_episode,
                instance_name,
                info.get("iterations", "?"),
                info.get("decisions", "?"),
                info.get("integer_found", False),
                info.get("failed", False),
                float(info.get("distance", 0.0)),
                float(info.get("load_seconds", 0.0)),
                float(info.get("reset_seconds", 0.0)),
                float(info.get("last_distance_solve_seconds", 0.0)),
                float(info.get("elapsed_seconds", 0.0)),
            )

        if self.num_timesteps >= self.next_progress_log:
            progress = 100.0 * self.num_timesteps / max(1, self.total_timesteps)
            self.next_progress_log += self.log_every_steps

            def mean_or_nan(values: deque[float]) -> float:
                if not values:
                    return float("nan")
                return sum(values) / len(values)

            if self.success_history:
                logger.info(
                    "Dashboard t=%d/%d (%.1f%%) | window=%d | sr=%.3f | ret=%.3f | dist=%.3f | "
                    "fail=%.3f | flips/step=%.3f | "
                    "stall_recover=%.3f | succ_steps=%.2f | solve_s=%.3f",
                    self.num_timesteps,
                    self.total_timesteps,
                    progress,
                    len(self.success_history),
                    mean_or_nan(self.success_history),
                    mean_or_nan(self.return_history),
                    mean_or_nan(self.final_distance_history),
                    mean_or_nan(self.failure_history),
                    mean_or_nan(self.flips_per_step_history),
                    mean_or_nan(self.stall_recovery_history),
                    mean_or_nan(self.steps_to_success_history),
                    mean_or_nan(self.solve_seconds_history),
                )
                self._curve_rows.append({
                    "timestep": self.num_timesteps,
                    "episodes_in_window": len(self.success_history),
                    "success_rate": mean_or_nan(self.success_history),
                    "mean_return": mean_or_nan(self.return_history),
                    "mean_final_distance": mean_or_nan(self.final_distance_history),
                    "failure_rate": mean_or_nan(self.failure_history),
                    "mean_flips_per_step": mean_or_nan(self.flips_per_step_history),
                    "mean_stall_recovery_rate": mean_or_nan(self.stall_recovery_history),
                    "mean_steps_to_success": mean_or_nan(self.steps_to_success_history),
                    "mean_solve_seconds": mean_or_nan(self.solve_seconds_history),
                })
            else:
                logger.info(
                    "Dashboard t=%d/%d (%.1f%%) | waiting for completed episodes",
                    self.num_timesteps,
                    self.total_timesteps,
                    progress,
                )

        return True

    def _on_training_end(self) -> None:
        logger.info("SB3 training loop finished")
        if self.curve_csv and self._curve_rows:
            curve_path = Path(self.curve_csv)
            curve_path.parent.mkdir(parents=True, exist_ok=True)
            with curve_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._curve_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._curve_rows)
            logger.info("Learning curve written to: %s (%d rows)", curve_path, len(self._curve_rows))

    def summary(self) -> dict:
        def mean_or_nan(values: deque[float]) -> float:
            if not values:
                return float("nan")
            return sum(values) / len(values)

        return {
            "episodes_in_window": len(self.success_history),
            "success_rate": mean_or_nan(self.success_history),
            "mean_return": mean_or_nan(self.return_history),
            "mean_final_distance": mean_or_nan(self.final_distance_history),
            "failure_rate": mean_or_nan(self.failure_history),
            "mean_flips_per_step": mean_or_nan(self.flips_per_step_history),
            "mean_stall_recovery_rate": mean_or_nan(self.stall_recovery_history),
            "mean_steps_to_success": mean_or_nan(self.steps_to_success_history),
            "mean_solve_seconds": mean_or_nan(self.solve_seconds_history),
            "checkpoints_written": len(self._curve_rows),
        }


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    device = resolve_device(args.device)
    k_choices = parse_k_choices(args.k_choices)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    vec_start_method = resolve_vec_start_method(args.vec_start_method)

    instance_paths = sorted(glob.glob(args.instances))
    if not instance_paths:
        raise FileNotFoundError(f"No .npz instance files matched: {args.instances}")
    shape_tag = infer_instance_shape_tag(instance_paths)

    run_started = time.time()
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = Path(args.save_path) if args.save_path else (run_dir / "model")
    curve_csv_path = Path(args.curve_csv) if args.curve_csv else (run_dir / f"learning_curve_{shape_tag}.csv")
    summary_json_path = run_dir / "run_summary.json"
    summary_csv_path = run_dir / f"run_summary_{shape_tag}.csv"
    args_json_path = run_dir / "run_args.json"

    single_env = Monitor(
        FeasibilityPumpKEnv(
            instance_paths=instance_paths,
            k_choices=k_choices,
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_threshold=args.stall_threshold,
            cplex_threads=args.cplex_threads,
        )
    )
    env = single_env

    if args.check_env:
        check_env(single_env.unwrapped, warn=True)

    if args.num_envs > 1:
        env = SubprocVecEnv(
            [
                make_env_factory(
                    instance_paths=instance_paths,
                    k_choices=k_choices,
                    max_iterations=args.max_iterations,
                    time_limit=args.time_limit,
                    stall_threshold=args.stall_threshold,
                    cplex_threads=args.cplex_threads,
                )
                for _ in range(args.num_envs)
            ],
            start_method=vec_start_method,
        )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Training on %d instances", len(instance_paths))
    logger.info("Instance shape tag: %s", shape_tag)
    logger.info("Run directory: %s", run_dir)
    logger.info("k choices: %s", list(k_choices))
    logger.info("Time limit per episode: %.1f seconds", args.time_limit)
    logger.info("Stall threshold: %d", args.stall_threshold)
    logger.info("Parallel environments: %d", args.num_envs)
    if args.num_envs > 1:
        logger.info("SubprocVecEnv start method: %s", vec_start_method)
    logger.info("CPLEX threads per solve: %d", args.cplex_threads)
    logger.info("Using torch device: %s", device)
    logger.info("Saving model to: %s", model_save_path)
    if args.log_file:
        logger.info("Writing logs to: %s", args.log_file)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs={"net_arch": [64, 64]},
        device=device,
    )

    callback = TrainingLoggerCallback(
        total_timesteps=args.total_timesteps,
        log_every_steps=args.progress_log_steps,
        rolling_window=args.dashboard_window,
        curve_csv=str(curve_csv_path),
    )
    with args_json_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(str(model_save_path))

    run_summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_path": str(model_save_path) + ".zip",
        "curve_csv_path": str(curve_csv_path),
        "args_path": str(args_json_path),
        "instances_glob": args.instances,
        "num_instances": len(instance_paths),
        "k_choices": list(k_choices),
        "total_timesteps": args.total_timesteps,
        "elapsed_seconds": time.time() - run_started,
        "training_window_metrics": callback.summary(),
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, sort_keys=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_summary.keys()))
        writer.writeheader()
        writer.writerow(run_summary)

    logger.info("Training complete")
    logger.info("Run summary JSON: %s", summary_json_path)
    logger.info("Run summary CSV: %s", summary_csv_path)


if __name__ == "__main__":
    main()
