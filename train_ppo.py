from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch

from fp_ppo import (
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_TIME_LIMIT,
    FeasibilityPumpFlipEnv,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO to decide when to perturb and which candidate variables to flip from sparse .npz instances."
    )
    parser.add_argument(
        "--instances",
        required=True,
        help="Glob pattern for sparse .npz instances, for example C:/.../instances/*.npz",
    )
    parser.add_argument(
        "--save-path",
        default="models/ppo_fp_k",
        help="Path prefix for the saved PPO model.",
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
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help="How many top candidate variables are exposed to the policy at each decision point.",
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
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Torch device for PPO. Defaults to cuda.",
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


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every_steps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.log_every_steps = max(1, log_every_steps)
        self.next_progress_log = self.log_every_steps

    def _on_training_start(self) -> None:
        logger.info("SB3 training loop started")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_progress_log:
            progress = 100.0 * self.num_timesteps / max(1, self.total_timesteps)
            logger.info(
                "Training progress: %d / %d timesteps (%.1f%%)",
                self.num_timesteps,
                self.total_timesteps,
                progress,
            )
            self.next_progress_log += self.log_every_steps

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done or not info:
                continue

            instance_name = Path(str(info.get("instance_path", "unknown"))).name
            logger.info(
                "Episode %s finished: instance=%s iterations=%s decisions=%s integer_found=%s "
                "failed=%s distance=%.4f load=%.2fs reset=%.2fs last_solve=%.2fs elapsed=%.2fs",
                info.get("episode", "?"),
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

        return True

    def _on_training_end(self) -> None:
        logger.info("SB3 training loop finished")


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    device = resolve_device(args.device)

    instance_paths = sorted(glob.glob(args.instances))
    if not instance_paths:
        raise FileNotFoundError(f"No .npz instance files matched: {args.instances}")

    env = Monitor(
        FeasibilityPumpFlipEnv(
            instance_paths=instance_paths,
            num_candidates=args.num_candidates,
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_threshold=args.stall_threshold,
        )
    )

    if args.check_env:
        check_env(env.unwrapped, warn=True)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Training on %d instances", len(instance_paths))
    logger.info("Candidate variables per decision: %d", args.num_candidates)
    logger.info("Time limit per episode: %.1f seconds", args.time_limit)
    logger.info("Stall threshold: %d", args.stall_threshold)
    logger.info("Using torch device: %s", device)
    logger.info("Saving model to: %s", save_path)
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
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(str(save_path))

    logger.info("Training complete")


if __name__ == "__main__":
    main()
