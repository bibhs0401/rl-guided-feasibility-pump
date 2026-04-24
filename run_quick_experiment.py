from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_spp import EvaluationPolicy, print_summary_table, run_rl_guided_fp, summarize_results, write_summary_csv
from fp_baseline_spp import FPConfig, FPResult, run_baseline_fp, write_results_csv
from spp_model import (
    find_instance_files,
    generate_random_set_packing_instance,
    load_spp_instance,
    save_npz_instance,
    write_instance_list,
)
from train_rl_spp import train_or_create_policy


def prepare_instances(args: argparse.Namespace) -> list[str]:
    paths = find_instance_files([args.instance_dir]) if args.instance_dir else []
    paths = paths[: args.max_instances]
    print(f"[quick] number of existing instances found: {len(paths)}")
    if paths:
        for path in paths:
            print(f"[quick] using instance: {Path(path).name}")
        return paths

    out_dir = Path(args.output_dir) / "quick_instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    print("[quick] no .npz/.lp instances found; generating small preliminary SPP instances")
    for i in range(args.max_instances):
        inst = generate_random_set_packing_instance(
            name=f"quick_spp_{i + 1}.npz",
            n=args.n,
            m=args.m,
            density=args.density,
            seed=args.seed + i,
        )
        generated.append(save_npz_instance(inst, out_dir / inst.name))
    write_instance_list(generated, Path(args.output_dir) / "quick_instance_list.txt")
    return generated


def run_quick(args: argparse.Namespace) -> tuple[list[FPResult], list[dict]]:
    paths = prepare_instances(args)
    policy_path = train_or_create_policy(
        paths,
        out_dir=Path(args.output_dir) / "rl_training",
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        seed=args.seed,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_length=args.stall_length,
        verbose=args.verbose,
        allow_heuristic_fallback=True,
    )
    print(f"[quick] policy path: {policy_path}")

    policy = EvaluationPolicy(policy_path, algorithm=args.algorithm, fixed_action=args.fixed_action, seed=args.seed)
    results: list[FPResult] = []
    for path in paths:
        print(f"[quick] current instance: {Path(path).name}")
        instance = load_spp_instance(path)
        baseline_cfg = FPConfig(
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_length=args.stall_length,
            random_seed=args.seed,
            baseline_action=args.baseline_action,
            verbose=args.verbose,
        )
        results.append(run_baseline_fp(instance, baseline_cfg, perturb_on_stall=True))

        rl_cfg = FPConfig(
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_length=args.stall_length,
            random_seed=args.seed,
            baseline_action=args.baseline_action,
            verbose=args.verbose,
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

    results_path = write_results_csv(results, Path(args.output_dir) / "quick_results.csv")
    summary = summarize_results(results)
    summary_path = write_summary_csv(summary, Path(args.output_dir) / "quick_summary.csv")
    print(f"[quick] wrote results CSV: {results_path}")
    print(f"[quick] wrote summary CSV: {summary_path}")
    print_summary_table(summary)
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick preliminary SPP FP vs RL-guided FP experiment.")
    parser.add_argument(
        "--instance-dir",
        default=r"C:\Users\bibhushaojha\Desktop\MMP\SPP\SPP",
        help="Directory containing .npz or .lp instances. Synthetic instances are generated if none are found.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-instances", type=int, default=3)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument("--density", type=float, default=0.06)
    parser.add_argument("--algorithm", choices=["PPO", "A2C"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--policy-mode", choices=["heuristic", "random", "fixed"], default="heuristic")
    parser.add_argument("--fixed-action", type=int, default=2, choices=range(5))
    parser.add_argument("--baseline-action", type=int, default=2, choices=range(5))
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--stall-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_quick(parse_args())


if __name__ == "__main__":
    main()
