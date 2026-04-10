from __future__ import annotations

import argparse
import glob
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from fp_ppo import (
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_TIME_LIMIT,
    FeasibilityPumpFlipEnv,
)


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
    return parser.parse_args()


def main():
    args = parse_args()

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

    print(f"Training on {len(instance_paths)} instances")
    print(f"candidate variables per decision: {args.num_candidates}")
    print(f"time limit per episode: {args.time_limit} seconds")
    print(f"stall threshold: {args.stall_threshold}")
    print(f"Saving model to: {save_path}")

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
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(str(save_path))

    print("Training complete.")


if __name__ == "__main__":
    main()
