from __future__ import annotations

import argparse
import csv
from datetime import datetime
import glob
import logging
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
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
        description="Evaluate a trained PPO policy on held-out sparse .npz instances."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved PPO model (.zip or path prefix).",
    )
    parser.add_argument(
        "--instances",
        required=True,
        help="Glob pattern for held-out sparse .npz instances.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum FP distance-model solves per episode.",
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
        "--cplex-threads",
        type=int,
        default=DEFAULT_CPLEX_THREADS,
        help="Number of CPLEX threads per solve. Use 0 for automatic.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Torch device used for PPO inference.",
    )
    parser.add_argument(
        "--k-choices",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_CHOICES),
        help="Comma-separated perturbation sizes used by the trained model action mapping.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of held-out instances to evaluate.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs/eval_ppo",
        help="Base directory where each evaluation run gets its own timestamped folder.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional evaluation run folder name. If omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--per-instance-csv",
        default=None,
        help="Optional CSV path for per-instance results. Defaults to <run_dir>/eval_instances.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV path for aggregate summary results. Defaults to <run_dir>/eval_summary.csv.",
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
        help="Optional log file path.",
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


def validate_action_mapping(model: PPO, k_choices: tuple[int, ...]) -> None:
    from gymnasium import spaces as gym_spaces

    action_space = model.action_space
    if not isinstance(action_space, gym_spaces.Discrete):
        raise ValueError(
            f"Expected a Discrete action space (choose a k-choice index) "
            f"but got {type(action_space).__name__}. Load a model trained with FeasibilityPumpKEnv."
        )
    if int(action_space.n) != len(k_choices):
        raise ValueError(
            f"Model action space has {int(action_space.n)} actions, but --k-choices has {len(k_choices)} values."
        )


def evaluate_instance(env: FeasibilityPumpKEnv, model: PPO, instance_path: str) -> dict:
    started = time.time()
    observation, info = env.reset(options={"instance_path": instance_path})

    episode_reward = 0.0
    step_count = 0
    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        step_count += 1
        done = bool(terminated or truncated)

    wall_time_seconds = time.time() - started

    result = {
        "instance": Path(instance_path).name,
        "instance_path": instance_path,
        "integer_found": bool(info.get("integer_found", False)),
        "failed": bool(info.get("failed", False)),
        "iterations": int(info.get("iterations", 0)),
        "decisions": int(info.get("decisions", 0)),
        "steps_taken": step_count,
        "perturb_steps": int(info.get("perturb_steps", 0)),
        "total_flips": int(info.get("total_flips", 0)),
        "stall_events": int(info.get("stall_events", 0)),
        "stall_recoveries": int(info.get("stall_recoveries", 0)),
        "final_distance": float(info.get("distance", 0.0)),
        "load_seconds": float(info.get("load_seconds", 0.0)),
        "reset_seconds": float(info.get("reset_seconds", 0.0)),
        "initial_solve_seconds": float(info.get("initial_solve_seconds", 0.0)),
        "last_distance_solve_seconds": float(info.get("last_distance_solve_seconds", 0.0)),
        "elapsed_seconds": float(info.get("elapsed_seconds", 0.0)),
        "wall_time_seconds": wall_time_seconds,
        "episode_reward": float(episode_reward),
    }
    return result


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write for {output_path}")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(results: list[dict], model_path: str, instances_glob: str) -> dict:
    success_count = sum(1 for result in results if result["integer_found"])

    return {
        "model_path": model_path,
        "instances_glob": instances_glob,
        "num_instances": len(results),
        "num_successes": success_count,
        "success_rate": success_count / max(1, len(results)),
        "mean_iterations": mean([float(result["iterations"]) for result in results]),
        "mean_time_seconds": mean([float(result["wall_time_seconds"]) for result in results]),
        "mean_final_distance": mean([float(result["final_distance"]) for result in results]),
        "mean_perturb_steps": mean([float(result["perturb_steps"]) for result in results]),
        "mean_total_flips": mean([float(result["total_flips"]) for result in results]),
        "mean_episode_reward": mean([float(result["episode_reward"]) for result in results]),
    }


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    device = resolve_device(args.device)
    k_choices = parse_k_choices(args.k_choices)

    instance_paths = sorted(glob.glob(args.instances))
    if args.limit is not None:
        instance_paths = instance_paths[: args.limit]

    if not instance_paths:
        raise FileNotFoundError(f"No .npz instance files matched: {args.instances}")

    run_name = args.run_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_instance_csv = Path(args.per_instance_csv) if args.per_instance_csv else (run_dir / "eval_instances.csv")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (run_dir / "eval_summary.csv")

    model = PPO.load(args.model_path, device=device)
    validate_action_mapping(model, k_choices)

    env = FeasibilityPumpKEnv(
        instance_paths=instance_paths,
        k_choices=k_choices,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_threshold=args.stall_threshold,
        cplex_threads=args.cplex_threads,
    )

    logger.info("Loaded model from: %s", args.model_path)
    logger.info("Evaluating on %d held-out instances", len(instance_paths))
    logger.info("Evaluation run directory: %s", run_dir)
    logger.info("Using k choices: %s", list(k_choices))
    logger.info("Using torch device: %s", device)
    logger.info("Using CPLEX threads per solve: %d", args.cplex_threads)

    results: list[dict] = []
    for index, instance_path in enumerate(instance_paths, start=1):
        logger.info("[%d/%d] Evaluating %s", index, len(instance_paths), Path(instance_path).name)
        result = evaluate_instance(env, model, instance_path)
        results.append(result)
        logger.info(
            "[%d/%d] Result: success=%s iterations=%d time=%.2fs distance=%.4f",
            index,
            len(instance_paths),
            result["integer_found"],
            result["iterations"],
            result["wall_time_seconds"],
            result["final_distance"],
        )

    summary = build_summary(results, args.model_path, args.instances)
    write_csv(per_instance_csv, results)
    write_csv(summary_csv, [summary])

    logger.info("Wrote per-instance results to: %s", per_instance_csv)
    logger.info("Wrote summary results to: %s", summary_csv)
    logger.info(
        "Summary: success_rate=%.3f mean_iterations=%.2f mean_time=%.2fs",
        summary["success_rate"],
        summary["mean_iterations"],
        summary["mean_time_seconds"],
    )


if __name__ == "__main__":
    main()
