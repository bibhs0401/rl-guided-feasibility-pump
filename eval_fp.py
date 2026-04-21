from __future__ import annotations

# -----------------------------------------------------------------------------
# eval_fp.py
#
# Evaluate and compare feasibility-pump variants on a fixed pool of instances.
#
# Supported methods:
#   baseline   — fixed flip count + fixed continuation bin (no RL).
#   random     — random (flip_bin, continuation_bin) per RL decision window.
#   rl         — load a trained SB3 PPO zip and run with deterministic=True.
#
# For each (method, instance, seed) we run one full episode and append one row
# to a tidy long-format CSV.  A second CSV aggregates per-method statistics.
#
# This is the canonical harness for producing publication-ready tables /
# performance profiles / cactus plots.
# -----------------------------------------------------------------------------

import argparse
import csv as _csv
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fp_gym_env import FeasibilityPumpRLEnv, FPGymConfig
from mmp_fp_core import FPRunConfig


# -----------------------------------------------------------------------------
# CSV schemas
# -----------------------------------------------------------------------------
_PER_EPISODE_FIELDS = [
    # provenance
    "method", "method_tag", "seed", "episode_idx",
    "instance_path", "instance_name", "m", "n", "p",
    # wall-clock breakdown
    "wall_seconds",                   # full harness wall-time (incl. build)
    "initial_lp_solve_seconds",       # one-time LP solve (cached after first)
    "reset_seconds",
    "fp_time_s",                      # time inside FP loop only
    # outcome
    "integer_found", "failed",
    "terminated_in_initial_relaxation", "termination_reason",
    # FP counters
    "iterations", "stall_events", "total_flips",
    # FP quality
    "initial_distance", "final_distance", "best_distance",
    # RL aggregates
    "rl_steps", "episode_return",
    "mean_flip_bin", "mean_cont_bin",
    # config echoes
    "time_limit", "initial_lp_time_limit", "max_iterations",
    "stall_threshold", "max_stalls", "cplex_threads",
]


_SUMMARY_FIELDS = [
    "method", "method_tag", "num_runs", "num_instances", "num_seeds",
    "success_rate",
    "median_wall_seconds", "mean_wall_seconds",
    "median_fp_time_s",    "mean_fp_time_s",
    "median_iterations",   "mean_iterations",
    "median_total_flips",  "mean_total_flips",
    "median_stall_events", "mean_stall_events",
    "median_final_distance", "mean_final_distance",
    "mean_episode_return",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def read_instance_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not paths:
        raise ValueError(f"No instance paths found in: {path}")
    return paths


def make_env(
    instance_paths: List[str],
    fp_cfg: FPRunConfig,
    max_reset_resamples: int,
    seed: int,
) -> FeasibilityPumpRLEnv:
    env_cfg = FPGymConfig(
        instance_paths=instance_paths,
        fp_config=fp_cfg,
        max_reset_resamples=max_reset_resamples,
        seed=seed,
    )
    return FeasibilityPumpRLEnv(env_cfg)


def _median(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def _mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


# -----------------------------------------------------------------------------
# Action policies
# -----------------------------------------------------------------------------
class _BaselinePolicy:
    """Fixed (flip_bin, continuation_bin) at every decision window."""

    def __init__(self, flip_bin: int, cont_bin: int):
        self.flip_bin = int(flip_bin)
        self.cont_bin = int(cont_bin)

    def __call__(self, obs, rng) -> Tuple[int, int]:
        return self.flip_bin, self.cont_bin


class _RandomPolicy:
    """Uniform random action per decision window."""

    def __call__(self, obs, rng) -> Tuple[int, int]:
        return int(rng.integers(0, 6)), int(rng.integers(0, 5))


class _RLPolicy:
    """Wrap an SB3 PPO model and emit deterministic actions."""

    def __init__(self, model_path: str, deterministic: bool = True):
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path, device="auto")
        self._deterministic = bool(deterministic)

    def __call__(self, obs, rng) -> Tuple[int, int]:
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        action = np.asarray(action).reshape(-1)
        return int(action[0]), int(action[1])


# -----------------------------------------------------------------------------
# Run one episode
# -----------------------------------------------------------------------------
def run_one_episode(
    env: FeasibilityPumpRLEnv,
    policy,
    seed: int,
    episode_idx: int,
    instance_path: str,
    method: str,
    method_tag: str,
    fp_cfg: FPRunConfig,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + episode_idx)

    wall_t0 = time.time()
    obs, reset_info = env.reset(
        seed=seed + episode_idx,
        options={"instance_path": instance_path},
    )

    flip_bins: List[int] = []
    cont_bins: List[int] = []
    ep_return = 0.0
    rl_steps = 0
    last_info: Dict[str, Any] = dict(reset_info)

    done = truncated = False
    while not (done or truncated):
        flip_bin, cont_bin = policy(obs, rng)
        action = np.array([flip_bin, cont_bin], dtype=np.int64)
        obs, reward, done, truncated, info = env.step(action)
        ep_return += float(reward)
        flip_bins.append(int(flip_bin))
        cont_bins.append(int(cont_bin))
        rl_steps += 1
        last_info = info

    wall_seconds = time.time() - wall_t0

    row: Dict[str, Any] = {
        "method": method,
        "method_tag": method_tag,
        "seed": seed,
        "episode_idx": episode_idx,
        "instance_path": instance_path,
        "instance_name": Path(instance_path).name,
        "m": last_info.get("m"),
        "n": last_info.get("n"),
        "p": last_info.get("p"),

        "wall_seconds": round(wall_seconds, 4),
        "initial_lp_solve_seconds": round(
            float(reset_info.get("initial_lp_solve_seconds", 0.0) or 0.0), 4
        ),
        "reset_seconds": round(float(reset_info.get("reset_seconds", 0.0) or 0.0), 4),
        "fp_time_s": round(float(last_info.get("elapsed_seconds", 0.0) or 0.0), 4),

        "integer_found": bool(last_info.get("feasible_found", False)
                              or reset_info.get("integer_found", False)),
        "failed": bool(last_info.get("failed", False)
                       or reset_info.get("failed", False)),
        "terminated_in_initial_relaxation":
            bool(reset_info.get("terminated_in_initial_relaxation", False)),
        "termination_reason": str(last_info.get("termination_reason", "")),

        "iterations": int(last_info.get("iterations", 0) or 0),
        "stall_events": int(last_info.get("stall_events", 0) or 0),
        "total_flips": int(last_info.get("total_flips", 0) or 0),

        "initial_distance": round(
            float(reset_info.get("initial_distance", 0.0) or 0.0), 6
        ),
        "final_distance": round(
            float(last_info.get("current_distance", 0.0) or 0.0), 6
        ),
        "best_distance": round(
            float(last_info.get("best_distance", 0.0) or 0.0), 6
        ),

        "rl_steps": rl_steps,
        "episode_return": round(ep_return, 6),
        "mean_flip_bin": round(float(np.mean(flip_bins)) if flip_bins else 0.0, 3),
        "mean_cont_bin": round(float(np.mean(cont_bins)) if cont_bins else 0.0, 3),

        "time_limit": fp_cfg.time_limit,
        "initial_lp_time_limit": fp_cfg.initial_lp_time_limit,
        "max_iterations": fp_cfg.max_iterations,
        "stall_threshold": fp_cfg.stall_threshold,
        "max_stalls": fp_cfg.max_stalls,
        "cplex_threads": fp_cfg.cplex_threads,
    }
    return row


# -----------------------------------------------------------------------------
# Summarize
# -----------------------------------------------------------------------------
def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-episode rows into one summary row per (method, method_tag)."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["method_tag"]), []).append(r)

    summary: List[Dict[str, Any]] = []
    for (method, tag), rs in groups.items():
        success = [1.0 if r["integer_found"] else 0.0 for r in rs]
        wall    = [r["wall_seconds"] for r in rs]
        fp_t    = [r["fp_time_s"] for r in rs]
        iters   = [r["iterations"] for r in rs]
        flips   = [r["total_flips"] for r in rs]
        stalls  = [r["stall_events"] for r in rs]
        final_d = [r["final_distance"] for r in rs]
        rets    = [r["episode_return"] for r in rs]

        insts = {r["instance_path"] for r in rs}
        seeds = {r["seed"] for r in rs}

        summary.append({
            "method": method,
            "method_tag": tag,
            "num_runs": len(rs),
            "num_instances": len(insts),
            "num_seeds": len(seeds),
            "success_rate": round(_mean(success), 4),
            "median_wall_seconds": round(_median(wall), 4),
            "mean_wall_seconds": round(_mean(wall), 4),
            "median_fp_time_s": round(_median(fp_t), 4),
            "mean_fp_time_s": round(_mean(fp_t), 4),
            "median_iterations": round(_median(iters), 3),
            "mean_iterations": round(_mean(iters), 3),
            "median_total_flips": round(_median(flips), 3),
            "mean_total_flips": round(_mean(flips), 3),
            "median_stall_events": round(_median(stalls), 3),
            "mean_stall_events": round(_mean(stalls), 3),
            "median_final_distance": round(_median(final_d), 6),
            "mean_final_distance": round(_mean(final_d), 6),
            "mean_episode_return": round(_mean(rets), 6),
        })
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FP variants (baseline / random / RL) on a fixed instance pool."
    )
    parser.add_argument("--instances", required=True,
                        help="Path to a .txt file with one .npz path per line.")
    parser.add_argument("--methods", default="baseline,random,rl",
                        help="Comma-separated methods from {baseline,random,rl}.")
    parser.add_argument("--rl-model", default="",
                        help="Path to an SB3 PPO .zip (required if 'rl' in methods).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds used for BOTH the env and action sampling.")

    # FP config
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--initial-lp-time-limit", type=float, default=200.0,
        help="Wall-clock cap for the initial LP (feasible, not necessarily optimal). "
             "Ignored if --initial-lp-optimal is set.",
    )
    parser.add_argument("--initial-lp-optimal", action="store_true",
                        help="Solve initial LP to optimality (no time limit).")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--stall-threshold", type=int, default=3)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--max-reset-resamples", type=int, default=1,
                        help="Keep this at 1 for eval so each requested instance is actually used.")

    # Baseline-policy parameters
    parser.add_argument("--baseline-flip-bin", type=int, default=3,
                        help="Fixed flip bin for the baseline policy (0..5).")
    parser.add_argument("--baseline-cont-bin", type=int, default=2,
                        help="Fixed continuation bin for the baseline policy (0..4).")

    parser.add_argument("--deterministic-rl", action="store_true", default=True,
                        help="Use PPO.predict(deterministic=True). Default True.")
    parser.add_argument("--stochastic-rl", dest="deterministic_rl",
                        action="store_false",
                        help="Use PPO.predict(deterministic=False) for stochastic evaluation.")

    # IO
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--tag", default="",
                        help="Optional filename suffix for all outputs.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    allowed = {"baseline", "random", "rl"}
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Allowed: {sorted(allowed)}")
    if "rl" in methods and not args.rl_model:
        raise ValueError("--rl-model is required when 'rl' is in --methods.")

    instance_paths = read_instance_list(args.instances)
    print(f"[eval] instances : {len(instance_paths)}", flush=True)
    print(f"[eval] methods   : {methods}", flush=True)
    print(f"[eval] seeds     : {args.seeds}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    per_ep_csv = out_dir / f"eval_per_episode_{timestamp}{tag_suffix}.csv"
    summary_csv = out_dir / f"eval_summary_{timestamp}{tag_suffix}.csv"
    config_json = out_dir / f"eval_config_{timestamp}{tag_suffix}.json"

    initial_lp_limit = None if args.initial_lp_optimal else args.initial_lp_time_limit

    fp_cfg = FPRunConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        initial_lp_time_limit=initial_lp_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
    )

    run_config = {
        "instances_file": args.instances,
        "instance_count": len(instance_paths),
        "methods": methods,
        "seeds": list(args.seeds),
        "rl_model": args.rl_model if "rl" in methods else "",
        "baseline_flip_bin": args.baseline_flip_bin,
        "baseline_cont_bin": args.baseline_cont_bin,
        "deterministic_rl": bool(args.deterministic_rl),
        "fp_config": asdict(fp_cfg),
        "max_reset_resamples": args.max_reset_resamples,
        "per_episode_csv": str(per_ep_csv),
        "summary_csv": str(summary_csv),
        "timestamp": timestamp,
    }
    with config_json.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"[eval] config saved -> {config_json}", flush=True)

    # One env per (method, seed) so LP caches are shared across instances.
    # max_reset_resamples=1 so the env uses the exact requested instance.
    policies: Dict[str, Tuple[Any, str]] = {}
    if "baseline" in methods:
        policies["baseline"] = (
            _BaselinePolicy(args.baseline_flip_bin, args.baseline_cont_bin),
            f"fbin={args.baseline_flip_bin};cbin={args.baseline_cont_bin}",
        )
    if "random" in methods:
        policies["random"] = (_RandomPolicy(), "uniform")
    if "rl" in methods:
        policies["rl"] = (
            _RLPolicy(args.rl_model, deterministic=args.deterministic_rl),
            Path(args.rl_model).name
            + (";det" if args.deterministic_rl else ";stoch"),
        )

    rows: List[Dict[str, Any]] = []
    with per_ep_csv.open("w", newline="", encoding="utf-8") as ep_file:
        writer = _csv.DictWriter(ep_file, fieldnames=_PER_EPISODE_FIELDS)
        writer.writeheader()

        for method, (policy, tag) in policies.items():
            for seed in args.seeds:
                env = make_env(
                    instance_paths=instance_paths,
                    fp_cfg=fp_cfg,
                    max_reset_resamples=args.max_reset_resamples,
                    seed=seed,
                )
                try:
                    for idx, inst_path in enumerate(instance_paths):
                        print(
                            f"[eval] method={method:<8s} seed={seed} "
                            f"({idx + 1}/{len(instance_paths)}) {Path(inst_path).name}",
                            flush=True,
                        )
                        row = run_one_episode(
                            env=env,
                            policy=policy,
                            seed=seed,
                            episode_idx=idx,
                            instance_path=inst_path,
                            method=method,
                            method_tag=tag,
                            fp_cfg=fp_cfg,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        ep_file.flush()
                finally:
                    env.close()

    summary_rows = summarize_rows(rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as sf:
        sw = _csv.DictWriter(sf, fieldnames=_SUMMARY_FIELDS)
        sw.writeheader()
        for r in summary_rows:
            sw.writerow(r)

    print("\n[eval] done", flush=True)
    print(f"[eval] per-episode csv : {per_ep_csv}", flush=True)
    print(f"[eval] summary csv     : {summary_csv}", flush=True)
    print(f"[eval] config          : {config_json}", flush=True)

    # Console summary table
    print("\n[eval] method summary:")
    for r in summary_rows:
        print(
            f"  method={r['method']:<8s}  tag={r['method_tag']}"
            f"  runs={r['num_runs']}"
            f"  success={r['success_rate']*100:.1f}%"
            f"  median_wall={r['median_wall_seconds']:.2f}s"
            f"  median_iters={r['median_iterations']:.1f}"
            f"  median_flips={r['median_total_flips']:.1f}"
            f"  mean_ret={r['mean_episode_return']:+.3f}"
        )


if __name__ == "__main__":
    main()
