"""
build_trainable_pool.py
-----------------------
Read a screening CSV produced by screen_hard_instances.py and
write a trainable-pool .txt file containing only instances that
meet the minimum RL-usability criteria.

Usage
-----
    python build_trainable_pool.py \
        --csv screening_results/instance_screening_p3_m1000_n500_seed10.csv \
        --out-dir screening_results \
        --min-stalls 1 \
        --max-time 120

Output
------
    screening_results/trainable_pool_<shape_tag>_<seed_tag>.txt
    screening_results/trainable_pool_<shape_tag>_<seed_tag>_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Screening CSV from screen_hard_instances.py")
    parser.add_argument("--out-dir", default="screening_results")
    parser.add_argument(
        "--min-stalls", type=int, default=1,
        help="Minimum stall_events required (default 1 = at least one RL decision point).",
    )
    parser.add_argument(
        "--min-iters", type=int, default=1,
        help="Minimum FP iterations required (default 1).",
    )
    parser.add_argument(
        "--max-time", type=float, default=120.0,
        help="Maximum elapsed_seconds allowed per episode (default 120 s).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    # ── per-instance verdict ───────────────────────────────────────────────
    reasons: dict[str, list[str]] = {}  # instance_path -> list of rejection reasons

    trainable = []
    rejected = []

    for row in rows:
        path = row["instance_path"]
        why = []

        if str(row.get("failed", "False")).strip().lower() in ("true", "1"):
            why.append("failed=True (LP/model crash)")

        if str(row.get("terminated_in_initial_relaxation", "False")).strip().lower() in ("true", "1"):
            why.append("terminated_in_initial_relaxation=True (no FP loop entered)")

        stalls = int(row.get("stall_events", 0) or 0)
        if stalls < args.min_stalls:
            why.append(f"stall_events={stalls} < min {args.min_stalls} (no RL decision points)")

        iters = int(row.get("iterations", 0) or 0)
        if iters < args.min_iters:
            why.append(f"iterations={iters} < min {args.min_iters}")

        elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
        if elapsed > args.max_time:
            why.append(f"elapsed_seconds={elapsed:.1f} > max {args.max_time:.1f} s")

        if why:
            reasons[path] = why
            rejected.append(row)
        else:
            trainable.append(row)

    # ── rejection breakdown ───────────────────────────────────────────────
    tag_failed          = sum(1 for r in rejected if str(r.get("failed","False")).lower() in ("true","1"))
    tag_init_relax      = sum(1 for r in rejected if str(r.get("terminated_in_initial_relaxation","False")).lower() in ("true","1"))
    tag_no_stall        = sum(1 for r in rejected if int(r.get("stall_events",0) or 0) < args.min_stalls)
    tag_too_slow        = sum(1 for r in rejected if float(r.get("elapsed_seconds",0.0) or 0.0) > args.max_time)

    # ── derive shape/seed tag from CSV filename ───────────────────────────
    stem = csv_path.stem  # e.g. instance_screening_p3_m1000_n500_seed10
    parts = stem.split("_")
    # strip leading "instance_screening_"
    tag = "_".join(p for p in parts if p not in ("instance", "screening"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_txt  = out_dir / f"trainable_pool_{tag}.txt"
    summary_json = out_dir / f"trainable_pool_{tag}_summary.json"

    # ── write trainable list ──────────────────────────────────────────────
    with pool_txt.open("w", encoding="utf-8") as f:
        for row in trainable:
            f.write(row["instance_path"].strip() + "\n")

    # ── stall / iteration stats for trainable set ─────────────────────────
    if trainable:
        stall_list  = [int(r.get("stall_events", 0) or 0) for r in trainable]
        iter_list   = [int(r.get("iterations",   0) or 0) for r in trainable]
        time_list   = [float(r.get("elapsed_seconds", 0.0) or 0.0) for r in trainable]
        stats = {
            "stall_events": {
                "min": min(stall_list), "max": max(stall_list),
                "mean": round(sum(stall_list) / len(stall_list), 2),
            },
            "iterations": {
                "min": min(iter_list), "max": max(iter_list),
                "mean": round(sum(iter_list) / len(iter_list), 2),
            },
            "elapsed_seconds": {
                "min": round(min(time_list), 2), "max": round(max(time_list), 2),
                "mean": round(sum(time_list) / len(time_list), 2),
            },
        }
    else:
        stats = {}

    summary = {
        "total_screened": total,
        "trainable": len(trainable),
        "rejected": len(rejected),
        "rejection_breakdown": {
            "failed":                    tag_failed,
            "terminated_in_init_relax":  tag_init_relax,
            "too_few_stall_events":      tag_no_stall,
            "too_slow":                  tag_too_slow,
        },
        "filter_criteria": {
            "min_stalls":  args.min_stalls,
            "min_iters":   args.min_iters,
            "max_time_s":  args.max_time,
        },
        "trainable_stats": stats,
        "pool_txt": str(pool_txt),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── console report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Trainable pool filter report")
    print(f"{'='*60}")
    print(f"  Total screened       : {total}")
    print(f"  Trainable            : {len(trainable)}  ({100*len(trainable)/max(1,total):.1f}%)")
    print(f"  Rejected             : {len(rejected)}")
    print(f"    └─ failed          : {tag_failed}")
    print(f"    └─ init relax only : {tag_init_relax}")
    print(f"    └─ no stall events : {tag_no_stall}")
    print(f"    └─ too slow        : {tag_too_slow}")
    if stats:
        print(f"\n  Trainable instance stats:")
        print(f"    stall_events  : min={stats['stall_events']['min']}  "
              f"max={stats['stall_events']['max']}  "
              f"mean={stats['stall_events']['mean']}")
        print(f"    iterations    : min={stats['iterations']['min']}  "
              f"max={stats['iterations']['max']}  "
              f"mean={stats['iterations']['mean']}")
        print(f"    elapsed (s)   : min={stats['elapsed_seconds']['min']}  "
              f"max={stats['elapsed_seconds']['max']}  "
              f"mean={stats['elapsed_seconds']['mean']}")
    print(f"\n  Pool file  : {pool_txt}")
    print(f"  Summary    : {summary_json}")
    print(f"{'='*60}\n")

    if len(trainable) == 0:
        print("WARNING: No trainable instances found. Check instance size and FP time limit.")
    elif len(trainable) < 20:
        print(f"WARNING: Only {len(trainable)} trainable instances — consider generating more or relaxing filter criteria.")


if __name__ == "__main__":
    main()
