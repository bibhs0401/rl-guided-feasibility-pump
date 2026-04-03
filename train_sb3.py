import argparse
import glob
import logging
import os

import main_phase1_sb3 as phase1_sb3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sb3")
DEFAULT_INSTANCE_DIR = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SB3 policy for phase-1 feasibility-pump stall decisions.")
    parser.add_argument(
        "--instances",
        default=os.path.join(DEFAULT_INSTANCE_DIR, "instance*.csv"),
        help="Glob pattern for training instance CSV files.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total SB3 training timesteps.",
    )
    parser.add_argument(
        "--model-path",
        default="phase1_maskable_ppo",
        help="Path prefix used when saving the trained model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate flip variables exposed to the policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for SB3 training.",
    )
    parser.add_argument(
        "--no-masking",
        action="store_true",
        help="Disable action masking and fall back to plain PPO behavior.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip stable-baselines3 environment validation before training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instance_files = sorted(glob.glob(args.instances))

    if not instance_files:
        raise FileNotFoundError(f"No instance files matched pattern: {args.instances}")

    logger.info("Training on %s instances", len(instance_files))
    logger.info("First few instances: %s", instance_files[:5])

    phase1_sb3.train_phase1_sb3_model(
        instance_files=instance_files,
        total_timesteps=args.timesteps,
        model_path=args.model_path,
        use_masking=not args.no_masking,
        top_k=args.top_k,
        seed=args.seed,
        check_environment=not args.skip_env_check,
    )

    logger.info("Finished training. Saved model to %s", args.model_path)


if __name__ == "__main__":
    main()
