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
DEFAULT_INSTANCE_DIR = os.path.join(os.path.expanduser("~"), "1")
DEFAULT_FOLDERS_ROOT = os.path.expanduser("~")
DEFAULT_INSTANCE_PATTERN = "instance*.csv"


def discover_folder_names(root_path):
    folder_names = [
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    numeric_names = [name for name in folder_names if name.isdigit()]
    if numeric_names:
        return sorted(numeric_names, key=int)
    return sorted(folder_names)


def parse_folder_names(folder_names_arg, root_path):
    if folder_names_arg:
        return [name.strip() for name in folder_names_arg.split(",") if name.strip()]
    return discover_folder_names(root_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SB3 policy for phase-1 feasibility-pump stall decisions.")
    parser.add_argument(
        "--instances",
        default=None,
        help="Glob pattern for training instance CSV files. If omitted, the trainer iterates through numbered folders.",
    )
    parser.add_argument(
        "--folders-root",
        default=DEFAULT_FOLDERS_ROOT,
        help="Root directory containing folders such as 1, 2, 3, ... used for sequential training.",
    )
    parser.add_argument(
        "--folder-names",
        default=None,
        help="Comma-separated folder names to process in order. Defaults to auto-discovered folders under --folders-root.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total SB3 training timesteps per folder/stage.",
    )
    parser.add_argument(
        "--model-path",
        default="phase1_maskable_ppo",
        help="Path prefix used when saving the trained model.",
    )
    parser.add_argument(
        "--load-model-path",
        default=None,
        help="Optional existing SB3 model path to resume training from.",
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
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit for the entire SB3 training run.",
    )
    parser.add_argument(
        "--episode-time-limit",
        type=float,
        default=None,
        help="Optional per-episode time cap in seconds inside the FP environment.",
    )
    parser.add_argument(
        "--min-fp-time-limit",
        type=float,
        default=10.0,
        help="Minimum FP time budget in seconds for each training episode.",
    )
    parser.add_argument(
        "--no-skip-trivial-episodes",
        action="store_true",
        help="Do not skip episodes that end before the agent gets to make a decision.",
    )
    parser.add_argument(
        "--no-stage-checkpoints",
        action="store_true",
        help="Do not save an extra checkpoint after each folder stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.instances:
        instance_files = sorted(glob.glob(args.instances))
        if not instance_files:
            raise FileNotFoundError(f"No instance files matched pattern: {args.instances}")

        logger.info("Training on %s instances", len(instance_files))
        logger.info("First few instances: %s", instance_files[:5])

        phase1_sb3.train_phase1_sb3_model(
            instance_files=instance_files,
            total_timesteps=args.timesteps,
            model_path=args.model_path,
            load_model_path=args.load_model_path,
            use_masking=not args.no_masking,
            top_k=args.top_k,
            seed=args.seed,
            check_environment=not args.skip_env_check,
            max_train_seconds=args.max_train_seconds,
            episode_time_limit=args.episode_time_limit,
            min_fp_time_limit=args.min_fp_time_limit,
            skip_trivial_episodes=not args.no_skip_trivial_episodes,
        )

        logger.info("Finished training. Saved model to %s", args.model_path)
        return

    folder_names = parse_folder_names(args.folder_names, args.folders_root)
    if not folder_names:
        raise FileNotFoundError(f"No folders found under {args.folders_root}")

    current_model_path = args.load_model_path
    stage_check_environment = not args.skip_env_check

    logger.info("Sequential folder training enabled")
    logger.info("Folders root: %s", args.folders_root)
    logger.info("Folders to process: %s", folder_names)

    for folder_name in folder_names:
        instance_pattern = os.path.join(args.folders_root, folder_name, DEFAULT_INSTANCE_PATTERN)
        instance_files = sorted(glob.glob(instance_pattern))
        if not instance_files:
            logger.warning("Skipping folder %s because no instances matched %s", folder_name, instance_pattern)
            continue

        logger.info("=" * 80)
        logger.info("Training on folder %s with %s instances", folder_name, len(instance_files))
        logger.info("First few instances: %s", instance_files[:5])

        model = phase1_sb3.train_phase1_sb3_model(
            instance_files=instance_files,
            total_timesteps=args.timesteps,
            model_path=args.model_path,
            load_model_path=current_model_path,
            use_masking=not args.no_masking,
            top_k=args.top_k,
            seed=args.seed,
            check_environment=stage_check_environment,
            max_train_seconds=args.max_train_seconds,
            episode_time_limit=args.episode_time_limit,
            min_fp_time_limit=args.min_fp_time_limit,
            skip_trivial_episodes=not args.no_skip_trivial_episodes,
        )

        if not args.no_stage_checkpoints:
            stage_checkpoint_path = f"{args.model_path}_folder{folder_name}"
            model.save(stage_checkpoint_path)
            logger.info("Saved stage checkpoint to %s", stage_checkpoint_path)

        current_model_path = args.model_path
        stage_check_environment = False

    logger.info("Finished sequential training. Final model saved to %s", args.model_path)


if __name__ == "__main__":
    main()
