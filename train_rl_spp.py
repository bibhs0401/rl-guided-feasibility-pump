from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from fp_baseline_spp import FPConfig
from spp_model import find_instance_files
from spp_rl_env import SPPRLEnvConfig, SPPFeasibilityPumpEnv, heuristic_action_from_observation


def _load_paths(args: argparse.Namespace) -> list[str]:
    if args.instances:
        paths = [str(Path(p).resolve()) for p in args.instances]
    else:
        paths = find_instance_files([args.instance_dir])
    if args.max_instances:
        paths = paths[: args.max_instances]
    if not paths:
        raise ValueError("No .npz or .lp set-packing instances found for training.")
    return paths


def create_env(instance_paths: Sequence[str], args: argparse.Namespace) -> SPPFeasibilityPumpEnv:
    fp_cfg = FPConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_length=args.stall_length,
        random_seed=args.seed,
        verbose=args.verbose,
    )
    env_cfg = SPPRLEnvConfig(
        instance_paths=list(instance_paths),
        fp_config=fp_cfg,
        seed=args.seed,
        continuation_steps_after_action=args.continuation_steps,
    )
    return SPPFeasibilityPumpEnv(env_cfg)


def train_or_create_policy(
    instance_paths: Sequence[str],
    out_dir: str | Path = "results/rl_training",
    algorithm: str = "PPO",
    timesteps: int = 500,
    seed: int = 0,
    max_iterations: int = 80,
    time_limit: float = 10.0,
    stall_length: int = 3,
    continuation_steps: int | None = None,
    verbose: bool = False,
    allow_heuristic_fallback: bool = True,
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / f"{algorithm.lower()}_spp_fp_policy"

    class _Args:
        pass

    args = _Args()
    args.max_iterations = max_iterations
    args.time_limit = time_limit
    args.stall_length = stall_length
    args.seed = seed
    args.verbose = verbose
    args.continuation_steps = continuation_steps

    try:
        from stable_baselines3 import A2C, PPO
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        if not allow_heuristic_fallback:
            raise RuntimeError(
                "stable_baselines3 is not installed. Install it to train PPO/A2C, "
                "or rerun with heuristic fallback enabled."
            ) from exc
        fallback_path = out / "heuristic_spp_fp_policy.json"
        fallback_path.write_text(
            json.dumps(
                {
                    "policy_type": "heuristic_stall_perturbation",
                    "algorithm_requested": algorithm,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "stable_baselines3_not_installed",
                    "action_space": {
                        "0": "skip",
                        "1": "flip_1_percent",
                        "2": "flip_5_percent",
                        "3": "flip_10_percent",
                        "4": "flip_20_percent",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "[train] stable_baselines3 is not installed; wrote heuristic fallback policy "
            f"to {fallback_path}"
        )
        return str(fallback_path.resolve())

    def make_env():
        return Monitor(create_env(instance_paths, args))

    env = make_env()
    algo_name = algorithm.upper()
    model_cls = PPO if algo_name == "PPO" else A2C
    model = model_cls("MlpPolicy", env, verbose=1 if verbose else 0, seed=seed)
    model.learn(total_timesteps=int(timesteps))
    model.save(str(policy_path))
    saved = str(policy_path.with_suffix(".zip").resolve())
    print(f"[train] saved {algo_name} model to {saved}")
    return saved


def smoke_test_policy(policy_path: str, instance_paths: Sequence[str], args: argparse.Namespace) -> None:
    env = create_env(instance_paths, args)
    obs, info = env.reset()
    total_return = 0.0
    for _ in range(args.smoke_steps):
        if policy_path.endswith(".json"):
            action = heuristic_action_from_observation(obs)
        else:
            try:
                from stable_baselines3 import A2C, PPO
            except ModuleNotFoundError:
                action = heuristic_action_from_observation(obs)
            else:
                model_cls = PPO if args.algorithm.upper() == "PPO" else A2C
                model = model_cls.load(policy_path)
                action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_return += reward
        if terminated or truncated:
            break
    print(
        "[train] smoke episode "
        f"instance={info.get('instance_name')} return={total_return:.4f} "
        f"success={info.get('success')} stalls={info.get('num_stalls')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/A2C for stall-only SPP FP perturbations.")
    parser.add_argument("--instance-dir", default=".", help="Directory containing .npz or .lp instances.")
    parser.add_argument("--instances", nargs="*", default=None, help="Explicit instance paths.")
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--algorithm", choices=["PPO", "A2C"], default="PPO")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--out-dir", default="results/rl_training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--stall-length", type=int, default=3)
    parser.add_argument("--continuation-steps", type=int, default=None)
    parser.add_argument("--no-heuristic-fallback", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _load_paths(args)
    print(f"[train] number of instances found: {len(paths)}")
    for path in paths:
        print(f"[train] instance: {Path(path).name}")
    policy_path = train_or_create_policy(
        paths,
        out_dir=args.out_dir,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        seed=args.seed,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_length=args.stall_length,
        continuation_steps=args.continuation_steps,
        verbose=args.verbose,
        allow_heuristic_fallback=not args.no_heuristic_fallback,
    )
    smoke_test_policy(policy_path, paths, args)
    print(f"[train] policy path: {policy_path}")


if __name__ == "__main__":
    main()
