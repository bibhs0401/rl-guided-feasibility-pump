from __future__ import annotations

import argparse
import csv
import glob
import logging
import random
import sys
import time
from pathlib import Path

from fp_ppo import (
    DEFAULT_CPLEX_THREADS,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_TIME_LIMIT,
    FeasibilityPumpRunner,
    load_problem,
    select_flip_candidates,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a simple classical-FP-style perturbation baseline on sparse .npz instances."
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
        help="How many consecutive no-change FP steps count as a stall.",
    )
    parser.add_argument(
        "--cplex-threads",
        type=int,
        default=DEFAULT_CPLEX_THREADS,
        help="Number of CPLEX threads per solve. Use 0 for automatic.",
    )
    parser.add_argument(
        "--flip-k",
        type=int,
        default=10,
        help="How many top disagreement variables to flip when a stall is detected.",
    )
    parser.add_argument(
        "--random-flip-k",
        action="store_true",
        help="If set, sample k uniformly from [flip-k-min, flip-k-max] at each perturbation.",
    )
    parser.add_argument(
        "--flip-k-min",
        type=int,
        default=5,
        help="Minimum k when --random-flip-k is enabled.",
    )
    parser.add_argument(
        "--flip-k-max",
        type=int,
        default=15,
        help="Maximum k when --random-flip-k is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for the optional random-k baseline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of held-out instances to evaluate.",
    )
    parser.add_argument(
        "--per-instance-csv",
        default="results/eval_baseline_instances.csv",
        help="CSV path for per-instance results.",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/eval_baseline_summary.csv",
        help="CSV path for aggregate summary results.",
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


def choose_flip_indices(runner: FeasibilityPumpRunner, args) -> list[int]:
    if not runner.is_stalled():
        return []

    if args.random_flip_k:
        flip_k = random.randint(args.flip_k_min, args.flip_k_max)
    else:
        flip_k = args.flip_k

    return select_flip_candidates(runner, flip_k)


def evaluate_instance(instance_path: str, args) -> dict:
    started = time.time()
    problem = load_problem(instance_path)
    runner = FeasibilityPumpRunner(
        problem=problem,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_threshold=args.stall_threshold,
        cplex_threads=args.cplex_threads,
    )
    runner.reset()

    while not runner.done:
        was_stalled_before_action = runner.is_stalled()
        selected_indices = choose_flip_indices(runner, args)
        if selected_indices and not was_stalled_before_action:
            runner.off_stall_perturb_steps += 1
        runner.run_one_iteration(selected_indices)

    wall_time_seconds = time.time() - started

    result = {
        "policy_name": "baseline_fp_random_k" if args.random_flip_k else "baseline_fp_topk",
        "instance": Path(instance_path).name,
        "instance_path": instance_path,
        "integer_found": bool(runner.integer_found),
        "failed": bool(runner.failed),
        "iterations": int(runner.iteration),
        "decisions": int(runner.decision_count),
        "perturb_steps": int(runner.perturb_steps),
        "off_stall_perturb_steps": int(runner.off_stall_perturb_steps),
        "no_flip_steps": int(runner.no_flip_steps),
        "total_flips": int(runner.total_flips),
        "stall_events": int(runner.stall_events),
        "stall_recoveries": int(runner.stall_recoveries),
        "final_distance": float(runner.current_distance()),
        "reset_seconds": float(runner.reset_seconds),
        "initial_solve_seconds": float(runner.relaxation_solve_seconds),
        "last_distance_solve_seconds": float(runner.last_distance_solve_seconds),
        "elapsed_seconds": 0.0 if runner.start_time is None else time.time() - runner.start_time,
        "wall_time_seconds": wall_time_seconds,
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


def build_summary(results: list[dict], args) -> dict:
    success_count = sum(1 for result in results if result["integer_found"])
    total_perturb_steps = sum(int(result["perturb_steps"]) for result in results)
    total_off_stall_perturb_steps = sum(int(result["off_stall_perturb_steps"]) for result in results)

    return {
        "policy_name": "baseline_fp_random_k" if args.random_flip_k else "baseline_fp_topk",
        "instances_glob": args.instances,
        "num_instances": len(results),
        "num_successes": success_count,
        "success_rate": success_count / max(1, len(results)),
        "mean_iterations": mean([float(result["iterations"]) for result in results]),
        "mean_time_seconds": mean([float(result["wall_time_seconds"]) for result in results]),
        "mean_final_distance": mean([float(result["final_distance"]) for result in results]),
        "mean_perturb_steps": mean([float(result["perturb_steps"]) for result in results]),
        "mean_off_stall_perturb_steps": mean([float(result["off_stall_perturb_steps"]) for result in results]),
        "off_stall_perturb_ratio": total_off_stall_perturb_steps / max(1, total_perturb_steps),
        "mean_total_flips": mean([float(result["total_flips"]) for result in results]),
        "flip_k": args.flip_k,
        "random_flip_k": args.random_flip_k,
        "flip_k_min": args.flip_k_min,
        "flip_k_max": args.flip_k_max,
    }


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    random.seed(args.seed)

    instance_paths = sorted(glob.glob(args.instances))
    if args.limit is not None:
        instance_paths = instance_paths[: args.limit]

    if not instance_paths:
        raise FileNotFoundError(f"No .npz instance files matched: {args.instances}")

    logger.info("Evaluating baseline on %d held-out instances", len(instance_paths))
    logger.info("Using CPLEX threads per solve: %d", args.cplex_threads)
    if args.random_flip_k:
        logger.info("Using random k in [%d, %d]", args.flip_k_min, args.flip_k_max)
    else:
        logger.info("Using fixed flip_k=%d", args.flip_k)

    results: list[dict] = []
    for index, instance_path in enumerate(instance_paths, start=1):
        logger.info("[%d/%d] Evaluating %s", index, len(instance_paths), Path(instance_path).name)
        result = evaluate_instance(instance_path, args)
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

    summary = build_summary(results, args)
    write_csv(args.per_instance_csv, results)
    write_csv(args.summary_csv, [summary])

    logger.info("Wrote per-instance results to: %s", args.per_instance_csv)
    logger.info("Wrote summary results to: %s", args.summary_csv)
    logger.info(
        "Summary: success_rate=%.3f mean_iterations=%.2f mean_time=%.2fs",
        summary["success_rate"],
        summary["mean_iterations"],
        summary["mean_time_seconds"],
    )


if __name__ == "__main__":
    main()
