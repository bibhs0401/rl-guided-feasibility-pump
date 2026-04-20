from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from mmp_fp_core import FPRunConfig, run_single_fp_episode


def infer_shape_tag(rows: list[dict]) -> str:
    """
    Build a filename-friendly shape tag from the screened instances.

    If all instances share the same (p, m, n), return:
        p3_m9000_n3000

    Otherwise return a mixed-range tag like:
        pmixed_m4000-12000_n2000-4000
    """
    if not rows:
        return "empty"

    p_values = sorted({int(r["p"]) for r in rows})
    m_values = sorted({int(r["m"]) for r in rows})
    n_values = sorted({int(r["n"]) for r in rows})

    if len(p_values) == 1 and len(m_values) == 1 and len(n_values) == 1:
        return f"p{p_values[0]}_m{m_values[0]}_n{n_values[0]}"

    p_tag = f"p{p_values[0]}" if len(p_values) == 1 else "pmixed"
    m_tag = f"m{m_values[0]}-{m_values[-1]}"
    n_tag = f"n{n_values[0]}-{n_values[-1]}"
    return f"{p_tag}_{m_tag}_{n_tag}"


def hardness_sort_key(row: dict):
    """
    Sort hardest instances first.

    Priority:
    1. failed instances first
    2. nontrivial FP cases before trivial initial-relaxation cases
    3. longer runtime
    4. more iterations
    5. more stall events
    """
    failed = 1 if row["failed"] else 0
    nontrivial = 0 if row["terminated_in_initial_relaxation"] else 1
    elapsed = float(row["elapsed_seconds"])
    iterations = int(row["iterations"])
    stall_events = int(row["stall_events"])

    return (
        failed,          # failed = hardest
        nontrivial,      # nontrivial FP cases ahead of trivial ones
        elapsed,
        iterations,
        stall_events,
    )


def classify_instance(row: dict) -> str:
    """
    Simple rule-based class label for easy / medium / hard.

    You can refine this later, but this is a good first version.
    """
    if row["failed"]:
        return "hard"

    if row["terminated_in_initial_relaxation"]:
        return "easy"

    if int(row["iterations"]) == 0 and float(row["elapsed_seconds"]) < 5.0:
        return "easy"

    if int(row["stall_events"]) >= 3:
        return "hard"

    if int(row["iterations"]) >= 20:
        return "hard"

    if float(row["elapsed_seconds"]) >= 20.0:
        return "hard"

    if int(row["iterations"]) >= 5 or float(row["elapsed_seconds"]) >= 8.0:
        return "medium"

    return "easy"


def write_txt_list(path: Path, rows: list[dict]) -> None:
    """
    Write one instance path per line.
    """
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["instance_path"]).strip() + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Screen .npz instances with baseline FP and identify hard instances."
    )
    parser.add_argument(
        "--instances",
        required=True,
        help="Glob pattern for .npz instances, for example 'instances_n3000/*.npz'",
    )
    parser.add_argument(
        "--out-dir",
        default="screening_results",
        help="Directory where CSV/TXT/JSON outputs will be saved.",
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--stall-threshold", type=int, default=3)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument(
        "--initial-lp-time-limit",
        type=float,
        default=200.0,
        help="Time limit in seconds for initial LP solve per instance.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log level for backend modules.",
    )
    parser.add_argument("--seed-tag", default="seed10", help="Tag to include in output filenames.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("mmp_fp_core").setLevel(getattr(logging, args.log_level))

    instance_paths: List[str] = sorted(glob.glob(args.instances))
    if not instance_paths:
        raise FileNotFoundError(f"No .npz files matched: {args.instances}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_cfg = FPRunConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        initial_lp_time_limit=args.initial_lp_time_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
    )

    rows: list[dict] = []

    print(f"[screen] found {len(instance_paths)} instances", flush=True)
    print(f"[screen] starting baseline FP screening...", flush=True)

    for idx, instance_path in enumerate(instance_paths, start=1):
        print(f"[screen] ({idx}/{len(instance_paths)}) {instance_path}", flush=True)
        t0 = time.time()
        summary = run_single_fp_episode(instance_path, fp_cfg)
        summary["instance_name"] = Path(instance_path).name
        summary["class_label"] = classify_instance(summary)

        rows.append(summary)

        print(
            "[screen] done "
            f"name={summary['instance_name']} "
            f"p={summary['p']} m={summary['m']} n={summary['n']} "
            f"integer_found={summary['integer_found']} "
            f"failed={summary['failed']} "
            f"init_relax={summary['terminated_in_initial_relaxation']} "
            f"iters={summary['iterations']} "
            f"stalls={summary['stall_events']} "
            f"time={summary['elapsed_seconds']:.2f}s "
            f"class={summary['class_label']} "
            f"wall={time.time() - t0:.2f}s",
            flush=True,
        )

    # Sort hardest first
    rows_sorted = sorted(rows, key=hardness_sort_key, reverse=True)

    # Split by class
    hard_rows = [r for r in rows_sorted if r["class_label"] == "hard"]
    medium_rows = [r for r in rows_sorted if r["class_label"] == "medium"]
    easy_rows = [r for r in rows_sorted if r["class_label"] == "easy"]

    shape_tag = infer_shape_tag(rows_sorted)
    seed_tag = args.seed_tag

    # Filenames include p/m/n information when possible
    csv_path = out_dir / f"instance_screening_{shape_tag}_{seed_tag}.csv"
    hard_txt = out_dir / f"hard_instances_{shape_tag}_{seed_tag}.txt"
    medium_txt = out_dir / f"medium_instances_{shape_tag}_{seed_tag}.txt"
    easy_txt = out_dir / f"easy_instances_{shape_tag}_{seed_tag}.txt"
    summary_json = out_dir / f"screening_summary_{shape_tag}_{seed_tag}.json"

    # Write CSV
    fieldnames = [
        "instance_path",
        "instance_name",
        "m",
        "n",
        "p",
        "iterations",
        "stall_events",
        "total_flips",
        "integer_found",
        "failed",
        "terminated_in_initial_relaxation",
        "initial_solution_was_integer",
        "initial_distance",
        "final_distance",
        "initial_lp_objective",
        "elapsed_seconds",
        "class_label",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # Write TXT lists
    write_txt_list(hard_txt, hard_rows)
    write_txt_list(medium_txt, medium_rows)
    write_txt_list(easy_txt, easy_rows)

    # Write summary JSON
    summary = {
        "num_instances": len(rows_sorted),
        "num_hard": len(hard_rows),
        "num_medium": len(medium_rows),
        "num_easy": len(easy_rows),
        "shape_tag": shape_tag,
        "seed_tag": seed_tag,
        "csv_path": str(csv_path),
        "hard_txt": str(hard_txt),
        "medium_txt": str(medium_txt),
        "easy_txt": str(easy_txt),
        "screening_config": {
            "instances_glob": args.instances,
            "time_limit": args.time_limit,
            "initial_lp_time_limit": args.initial_lp_time_limit,
            "max_iterations": args.max_iterations,
            "stall_threshold": args.stall_threshold,
            "max_stalls": args.max_stalls,
            "cplex_threads": args.cplex_threads,
            "log_level": args.log_level,
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[screen] completed", flush=True)
    print(f"[screen] csv: {csv_path}", flush=True)
    print(f"[screen] hard list: {hard_txt}", flush=True)
    print(f"[screen] medium list: {medium_txt}", flush=True)
    print(f"[screen] easy list: {easy_txt}", flush=True)
    print(
        f"[screen] counts -> hard={len(hard_rows)}, medium={len(medium_rows)}, easy={len(easy_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()