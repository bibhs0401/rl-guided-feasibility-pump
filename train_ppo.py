from __future__ import annotations

import argparse
import glob
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from fp_ppo import DEFAULT_K_VALUES, FeasibilityPumpKEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO to choose the FP perturbation size k.")
    parser.add_argument(
        "--instances",
        required=True,
        help="Glob pattern for instance files, for example C:/.../instances/*.npz",
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
        "--k-values",
        default="0,2,5,10,15,20",
        help="Comma-separated list of k actions.",
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


def parse_k_values(value: str) -> list[int]:
    k_values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not k_values:
        return list(DEFAULT_K_VALUES)
    return k_values


def main():
    args = parse_args()

    instance_paths = sorted(glob.glob(args.instances))
    if not instance_paths:
        raise FileNotFoundError(f"No instance files matched: {args.instances}")

    k_values = parse_k_values(args.k_values)

    env = Monitor(
        FeasibilityPumpKEnv(
            instance_paths=instance_paths,
            k_values=k_values,
            max_iterations=args.max_iterations,
        )
    )

    if args.check_env:
        check_env(env.unwrapped, warn=True)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {len(instance_paths)} instances")
    print(f"k actions: {k_values}")
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
