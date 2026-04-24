from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

from fp_baseline_spp import FPConfig, FPResult, run_baseline_fp, write_results_csv
from spp_model import find_instance_files, load_spp_instance
from spp_rl_env import SPPRLEnvConfig, SPPFeasibilityPumpEnv, heuristic_action_from_observation


SUMMARY_COLUMNS = [
    "method",
    "num_instances",
    "success_rate",
    "average_runtime",
    "average_final_violation",
    "average_objective_successful",
    "average_number_of_stalls",
    "number_of_failures",
]


class EvaluationPolicy:
    def __init__(self, policy_path: str | None, algorithm: str = "PPO", fixed_action: int = 2, seed: int = 0):
        self.policy_path = policy_path
        self.algorithm = algorithm.upper()
        self.fixed_action = fixed_action
        self.rng = random.Random(seed)
        self.model = None
        self.policy_type = "heuristic"
        if policy_path:
            path = Path(policy_path)
            if path.suffix.lower() == ".json":
                self.policy_type = "heuristic"
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    self.policy_type = meta.get("policy_type", "heuristic")
                except Exception:
                    pass
            elif path.exists():
                try:
                    from stable_baselines3 import A2C, PPO
                except ModuleNotFoundError:
                    print("[eval] stable_baselines3 missing; using heuristic policy instead.")
                    self.policy_type = "heuristic"
                else:
                    model_cls = PPO if self.algorithm == "PPO" else A2C
                    self.model = model_cls.load(str(path))
                    self.policy_type = self.algorithm.lower()

    def predict(self, obs, mode: str = "heuristic") -> int:
        if mode == "random":
            return self.rng.randrange(5)
        if mode == "fixed":
            return int(max(0, min(4, self.fixed_action)))
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        return int(heuristic_action_from_observation(obs))


def _load_paths(args: argparse.Namespace) -> list[str]:
    if args.instances:
        paths = [str(Path(p).resolve()) for p in args.instances]
    else:
        paths = find_instance_files([args.instance_dir])
    if args.max_instances:
        paths = paths[: args.max_instances]
    if not paths:
        raise ValueError("No .npz or .lp set-packing instances found.")
    return paths


def run_rl_guided_fp(
    instance_path: str,
    fp_config: FPConfig,
    policy: EvaluationPolicy,
    policy_mode: str = "heuristic",
    max_actions: int = 20,
) -> FPResult:
    env_cfg = SPPRLEnvConfig(
        instance_paths=[instance_path],
        fp_config=fp_config,
        seed=fp_config.random_seed,
    )
    env = SPPFeasibilityPumpEnv(env_cfg)
    obs, info = env.reset(options={"instance_path": instance_path})
    total_return = 0.0
    terminated = bool(env.runner is not None and env.runner.done and info.get("success", 0))
    truncated = False
    actions = 0
    while not (terminated or truncated) and actions < max_actions:
        action = policy.predict(obs, mode=policy_mode)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        actions += 1
    if actions >= max_actions and not (terminated or truncated):
        info["notes_error_status"] = f"{info.get('notes_error_status', '')};max_rl_actions"
    return FPResult(
        instance_name=str(info.get("instance_name", Path(instance_path).name)),
        method="rl_guided_fp",
        success=int(info.get("success", 0)),
        final_objective=float(info.get("final_objective", 0.0)),
        final_violation=float(info.get("final_violation", 0.0)),
        num_violated_constraints=int(info.get("num_violated_constraints", 0)),
        iterations=int(info.get("iterations", 0)),
        runtime_seconds=float(info.get("runtime_seconds", 0.0)),
        num_stalls=int(info.get("num_stalls", 0)),
        num_rl_interventions=int(info.get("num_rl_interventions", 0)),
        average_return=f"{total_return:.6f}",
        notes_error_status=str(info.get("notes_error_status", "")),
    )


def summarize_results(rows: Sequence[FPResult]) -> list[dict[str, Any]]:
    by_method: dict[str, list[FPResult]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    summary: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        n = len(method_rows)
        successes = [r for r in method_rows if r.success]
        failures = n - len(successes)
        avg_runtime = sum(r.runtime_seconds for r in method_rows) / max(1, n)
        avg_violation = sum(r.final_violation for r in method_rows) / max(1, n)
        avg_success_obj = sum(r.final_objective for r in successes) / max(1, len(successes))
        avg_stalls = sum(r.num_stalls for r in method_rows) / max(1, n)
        summary.append(
            {
                "method": method,
                "num_instances": n,
                "success_rate": len(successes) / max(1, n),
                "average_runtime": avg_runtime,
                "average_final_violation": avg_violation,
                "average_objective_successful": avg_success_obj,
                "average_number_of_stalls": avg_stalls,
                "number_of_failures": failures,
            }
        )
    return summary


def write_summary_csv(summary: Sequence[dict[str, Any]], path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "method": row["method"],
                    "num_instances": row["num_instances"],
                    "success_rate": f"{row['success_rate']:.6f}",
                    "average_runtime": f"{row['average_runtime']:.6f}",
                    "average_final_violation": f"{row['average_final_violation']:.10g}",
                    "average_objective_successful": f"{row['average_objective_successful']:.10g}",
                    "average_number_of_stalls": f"{row['average_number_of_stalls']:.6f}",
                    "number_of_failures": row["number_of_failures"],
                }
            )
    return str(out.resolve())


def print_summary_table(summary: Sequence[dict[str, Any]]) -> None:
    print("\nSummary table")
    print("method | success_rate | avg_runtime | avg_violation | avg_success_obj | avg_stalls | failures")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in summary:
        print(
            f"{row['method']} | "
            f"{row['success_rate']:.3f} | "
            f"{row['average_runtime']:.3f} | "
            f"{row['average_final_violation']:.6g} | "
            f"{row['average_objective_successful']:.6g} | "
            f"{row['average_number_of_stalls']:.3f} | "
            f"{row['number_of_failures']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and RL-guided FP on set packing.")
    parser.add_argument("--instance-dir", default=".", help="Directory containing .npz or .lp instances.")
    parser.add_argument("--instances", nargs="*", default=None, help="Explicit instance paths.")
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--policy-path", default=None, help="SB3 .zip model or heuristic JSON policy.")
    parser.add_argument("--policy-mode", choices=["heuristic", "random", "fixed"], default="heuristic")
    parser.add_argument("--algorithm", choices=["PPO", "A2C"], default="PPO")
    parser.add_argument("--fixed-action", type=int, default=2, choices=range(5))
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--stall-length", type=int, default=3)
    parser.add_argument("--baseline-action", type=int, default=2, choices=range(5))
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/spp_comparison.csv")
    parser.add_argument("--summary-output", default="results/spp_summary.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _load_paths(args)
    print(f"[eval] number of instances found: {len(paths)}")
    policy = EvaluationPolicy(args.policy_path, algorithm=args.algorithm, fixed_action=args.fixed_action, seed=args.seed)
    results: list[FPResult] = []
    for path in paths:
        print(f"[eval] current instance: {Path(path).name}")
        instance = load_spp_instance(path)
        baseline_cfg = FPConfig(
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_length=args.stall_length,
            random_seed=args.seed,
            baseline_action=args.baseline_action,
            cplex_threads=args.cplex_threads,
            verbose=not args.quiet,
        )
        results.append(run_baseline_fp(instance, baseline_cfg, perturb_on_stall=True))

        rl_cfg = FPConfig(
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_length=args.stall_length,
            random_seed=args.seed,
            baseline_action=args.baseline_action,
            cplex_threads=args.cplex_threads,
            verbose=not args.quiet,
        )
        results.append(
            run_rl_guided_fp(
                path,
                rl_cfg,
                policy,
                policy_mode=args.policy_mode,
                max_actions=args.max_actions,
            )
        )
    out = write_results_csv(results, args.output)
    summary = summarize_results(results)
    summary_out = write_summary_csv(summary, args.summary_output)
    print(f"[eval] wrote results CSV: {out}")
    print(f"[eval] wrote summary CSV: {summary_out}")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
