from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from fp_gym_env import FeasibilityPumpRLEnv, FPGymConfig
from mmp_fp_core import FPRunConfig


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
class _Tee:
    """Duplicate stdout writes to a log file (line-buffered)."""

    def __init__(self, log_path: str):
        self._console = sys.stdout
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, data: str) -> None:
        self._console.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._console
        self._file.close()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def read_instance_list(file_path: str) -> List[str]:
    paths: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    if not paths:
        raise ValueError(f"No instance paths found in: {file_path}")
    return paths


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device_name


def make_env_fn(
    instance_paths: List[str],
    seed: int,
    max_iterations: int,
    time_limit: float,
    stall_threshold: int,
    max_stalls: int,
    cplex_threads: int,
    max_reset_resamples: int,
) -> Callable[[], FeasibilityPumpRLEnv]:
    def _init():
        fp_cfg = FPRunConfig(
            max_iterations=max_iterations,
            time_limit=time_limit,
            stall_threshold=stall_threshold,
            max_stalls=max_stalls,
            cplex_threads=cplex_threads,
        )
        env_cfg = FPGymConfig(
            instance_paths=instance_paths,
            fp_config=fp_cfg,
            max_reset_resamples=max_reset_resamples,
            seed=seed,
        )
        # VecMonitor wraps the vectorized env outside — no inner Monitor needed.
        return FeasibilityPumpRLEnv(env_cfg)

    return _init


# ---------------------------------------------------------------------------
# CSV column list (one row per print interval)
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "episode", "timesteps", "wall_time",
    # rolling-average episode metrics
    "avg_reward", "feasible_rate",
    "avg_stall_events", "avg_iterations", "avg_total_flips",
    "avg_final_distance", "avg_best_distance",
    "avg_fp_time_s",
    # termination breakdown (fraction of window)
    "frac_feasible", "frac_budget", "frac_failed",
    # reward component averages (accumulated across steps, averaged per episode)
    "avg_r_frac", "avg_r_best", "avg_r_feas",
    "avg_r_time", "avg_r_flip", "avg_r_stall",
    # instance shape (from the most recent episode)
    "m", "n", "p",
]


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------
class PPOTrainingLogger(BaseCallback):
    """
    Per-episode logger, CSV writer, and periodic checkpointer.

    Console output (every ``print_every_episodes`` episodes):
      episode | timesteps | m n p
      reward | feasible_rate | termination breakdown
      stall_events | iterations | total_flips
      final_distance | best_distance | fp_time
      reward components: r_frac r_best r_feas r_time r_flip r_stall

    All metrics are rolling averages over the last ``rolling_window`` episodes.
    The CSV is flushed after every write so it can be read mid-run.
    """

    def __init__(
        self,
        print_every_episodes: int = 5,
        rolling_window: int = 50,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "",
        csv_log_path: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.print_every_episodes = max(1, print_every_episodes)
        W = max(1, rolling_window)

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self._last_checkpoint_step = 0

        self._csv_path = csv_log_path
        self._csv_file = None
        self._csv_writer = None

        self.episode_count = 0

        # One deque per metric
        self._hist: Dict[str, deque] = {k: deque(maxlen=W) for k in [
            "reward", "feasible", "stall_events", "iterations", "total_flips",
            "final_distance", "best_distance", "fp_time",
            "r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall",
            "term_feasible", "term_budget", "term_failed",
        ]}

        self._last_m: int = 0
        self._last_n: int = 0
        self._last_p: int = 0

        # Per-env reward-component accumulators (reset on episode done)
        self._ep_acc: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {k: 0.0 for k in
                     ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]}
        )

        # Last episode count written to console/CSV (for end-of-run tail flush).
        self._last_logged_episode: int = 0

    def _avg_hist(self, key: str) -> float:
        h = self._hist[key]
        return sum(h) / max(1, len(h))

    def _log_aggregates(self, *, is_final: bool = False) -> None:
        """Print and CSV one row from current rolling-window deques."""
        if not self._hist["reward"]:
            return

        a_rew = self._avg_hist("reward")
        a_feas = self._avg_hist("feasible")
        a_stall = self._avg_hist("stall_events")
        a_iter = self._avg_hist("iterations")
        a_flips = self._avg_hist("total_flips")
        a_fdist = self._avg_hist("final_distance")
        a_bdist = self._avg_hist("best_distance")
        a_time = self._avg_hist("fp_time")
        a_rfrac = self._avg_hist("r_frac")
        a_rbest = self._avg_hist("r_best")
        a_rfeas = self._avg_hist("r_feas")
        a_rtime = self._avg_hist("r_time")
        a_rflip = self._avg_hist("r_flip")
        a_rstall = self._avg_hist("r_stall")
        a_tf = self._avg_hist("term_feasible")
        a_tb = self._avg_hist("term_budget")
        a_tfail = self._avg_hist("term_failed")

        tag = "[train] FINAL" if is_final else "[train]"
        print(
            f"\n{tag} ep={self.episode_count:>6d}  ts={self.num_timesteps:>8d}"
            f"  m={self._last_m} n={self._last_n} p={self._last_p}\n"
            f"  reward       : {a_rew:+.4f}\n"
            f"  feasible_rate: {a_feas:.3f}"
            f"  (feasible={a_tf:.2f}  budget={a_tb:.2f}  failed={a_tfail:.2f})\n"
            f"  stall_events : {a_stall:.2f}   iters={a_iter:.1f}   flips={a_flips:.1f}\n"
            f"  distance     : final={a_fdist:.4f}   best={a_bdist:.4f}\n"
            f"  fp_time      : {a_time:.2f}s\n"
            f"  reward split : r_frac={a_rfrac:+.4f}  r_best={a_rbest:+.4f}"
            f"  r_feas={a_rfeas:+.4f}  r_time={a_rtime:+.4f}"
            f"  r_flip={a_rflip:+.4f}  r_stall={a_rstall:+.4f}",
            flush=True,
        )

        if self._csv_writer is not None:
            self._csv_writer.writerow({
                "episode":            self.episode_count,
                "timesteps":          self.num_timesteps,
                "wall_time":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "avg_reward":         round(a_rew, 6),
                "feasible_rate":      round(a_feas, 4),
                "avg_stall_events":   round(a_stall, 3),
                "avg_iterations":     round(a_iter, 2),
                "avg_total_flips":    round(a_flips, 2),
                "avg_final_distance": round(a_fdist, 4),
                "avg_best_distance":  round(a_bdist, 4),
                "avg_fp_time_s":      round(a_time, 3),
                "frac_feasible":      round(a_tf, 4),
                "frac_budget":        round(a_tb, 4),
                "frac_failed":        round(a_tfail, 4),
                "avg_r_frac":         round(a_rfrac, 6),
                "avg_r_best":         round(a_rbest, 6),
                "avg_r_feas":         round(a_rfeas, 6),
                "avg_r_time":         round(a_rtime, 6),
                "avg_r_flip":         round(a_rflip, 6),
                "avg_r_stall":        round(a_rstall, 6),
                "m":                  self._last_m,
                "n":                  self._last_n,
                "p":                  self._last_p,
            })
            self._csv_file.flush()

        self._last_logged_episode = self.episode_count

    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        super()._on_training_start()
        if self._csv_path:
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = _csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _on_training_end(self) -> None:
        if self.episode_count > self._last_logged_episode:
            self._log_aggregates(is_final=True)
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None
        super()._on_training_end()

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # ── periodic checkpoint ────────────────────────────────────────
        if (
            self.checkpoint_every > 0
            and self.checkpoint_dir
            and self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_every
        ):
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"ppo_step_{self.num_timesteps}"
            )
            self.model.save(ckpt_path)
            print(f"[checkpoint] saved -> {ckpt_path}.zip", flush=True)
            self._last_checkpoint_step = self.num_timesteps

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            # Accumulate reward components every step
            for k in ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]:
                self._ep_acc[env_idx][k] += float(info.get(k, 0.0))

            if not done:
                continue

            # ── episode boundary ───────────────────────────────────────
            self.episode_count += 1

            ep_raw = info.get("episode")
            if isinstance(ep_raw, dict) and "r" in ep_raw:
                ep_reward = float(ep_raw["r"])
            else:
                ep_reward = float(info.get("reward", 0.0))

            stall_events = int(info.get("stall_events", 0))
            iterations   = int(info.get("iterations", 0))
            total_flips  = int(info.get("total_flips", 0))
            final_dist   = float(info.get("current_distance", 0.0))
            best_dist    = float(info.get("best_distance", 0.0))
            fp_time      = float(info.get("elapsed_seconds", 0.0))
            feasible     = int(bool(info.get("feasible_found", False)))
            failed       = int(bool(info.get("failed", False)))
            term_reason  = str(info.get("termination_reason", ""))

            acc = self._ep_acc.pop(env_idx, {})

            self._hist["reward"].append(ep_reward)
            self._hist["feasible"].append(feasible)
            self._hist["stall_events"].append(stall_events)
            self._hist["iterations"].append(iterations)
            self._hist["total_flips"].append(total_flips)
            self._hist["final_distance"].append(final_dist)
            self._hist["best_distance"].append(best_dist)
            self._hist["fp_time"].append(fp_time)
            self._hist["term_feasible"].append(
                1 if term_reason == "feasible_found" else 0)
            self._hist["term_budget"].append(
                1 if term_reason == "time_or_iteration_or_stall_budget" else 0)
            self._hist["term_failed"].append(failed)
            for k in ["r_frac", "r_best", "r_feas", "r_time", "r_flip", "r_stall"]:
                self._hist[k].append(acc.get(k, 0.0))

            self._last_m = int(info.get("m", 0))
            self._last_n = int(info.get("n", 0))
            self._last_p = int(info.get("p", 0))

            # ── print + CSV every N episodes ───────────────────────────
            if self.episode_count % self.print_every_episodes == 0:
                self._log_aggregates(is_final=False)

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO on FeasibilityPumpRLEnv.")

    # Data
    parser.add_argument(
        "--instance-list", required=True,
        help="Text file with one .npz path per line.",
    )

    # Run / output
    parser.add_argument("--run-dir",   default="runs/fp_ppo_step3")
    parser.add_argument("--run-name",  default="trial_1")
    parser.add_argument("--save-name", default="ppo_fp_model")

    # Environment / FP
    parser.add_argument("--max-iterations",     type=int,   default=100)
    parser.add_argument("--time-limit",         type=float, default=30.0)
    parser.add_argument("--stall-threshold",    type=int,   default=3)
    parser.add_argument("--max-stalls",         type=int,   default=50)
    parser.add_argument("--cplex-threads",      type=int,   default=1)
    parser.add_argument("--max-reset-resamples",type=int,   default=20)

    # PPO
    parser.add_argument("--total-timesteps", type=int,   default=200000)
    parser.add_argument("--learning-rate",   type=float, default=3e-4)
    parser.add_argument("--n-steps",         type=int,   default=512)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--n-epochs",        type=int,   default=10)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--gae-lambda",      type=float, default=0.95)
    parser.add_argument("--ent-coef",        type=float, default=0.01)

    # Parallelism / device
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device",   choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed",     type=int, default=10)

    # Debug
    parser.add_argument("--check-env",            action="store_true")
    parser.add_argument("--print-every-episodes", type=int, default=5)

    # Checkpointing
    parser.add_argument("--checkpoint-every", type=int, default=10000,
                        help="Save checkpoint every N timesteps (0 = off).")
    parser.add_argument("--checkpoint-dir",   default="",
                        help="Checkpoint dir (default: <run-dir>/checkpoints).")

    # Logging
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard events to <run-dir>/tb_logs/.")
    parser.add_argument("--log-file", default="",
                        help="Tee stdout to this file (default: <run-dir>/train_<tag>_<ts>.log "
                             "when --tensorboard is set).")

    args = parser.parse_args()

    # ── paths ────────────────────────────────────────────────────────────
    instance_paths = read_instance_list(args.instance_list)

    run_dir = Path(args.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── logging setup ────────────────────────────────────────────────────
    # Show INFO from fp_gym_env (instance progress) and mmp_fp_core (model build).
    # SB3/torch stay at WARNING to avoid noise.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("fp_gym_env").setLevel(logging.INFO)
    logging.getLogger("mmp_fp_core").setLevel(logging.INFO)

    checkpoint_dir = (
        args.checkpoint_dir if args.checkpoint_dir
        else str(run_dir / "checkpoints")
    )
    if args.checkpoint_every > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── timestamp + shape tag ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import numpy as _np
    _probe = _np.load(instance_paths[0])
    _m = int(_probe["b"].shape[0]) if "b" in _probe else 0
    _c = _probe["c"] if "c" in _probe else None
    _n = int(_c.shape[1] if (_c is not None and _c.ndim > 1) else (_c.shape[0] if _c is not None else 0))
    _p = int(_c.shape[0] if (_c is not None and _c.ndim > 1) else 1)
    shape_tag = f"p{_p}_m{_m}_n{_n}"

    csv_log_path = str(run_dir / f"training_log_{shape_tag}_{timestamp}.csv")

    # ── stdout tee ───────────────────────────────────────────────────────
    if args.log_file:
        log_file = args.log_file
    elif args.tensorboard:
        log_file = str(run_dir / f"train_{shape_tag}_{timestamp}.log")
    else:
        log_file = ""
    _tee: Optional[_Tee] = _Tee(log_file) if log_file else None

    device = resolve_device(args.device)

    # ── save run config ──────────────────────────────────────────────────
    run_config = vars(args).copy()
    run_config.update({
        "resolved_device": device,
        "num_instances":   len(instance_paths),
        "shape_tag":       shape_tag,
        "timestamp":       timestamp,
        "csv_log_path":    csv_log_path,
    })
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[info] instances  : {len(instance_paths)}", flush=True)
    print(f"[info] shape tag  : {shape_tag}", flush=True)
    print(f"[info] run dir    : {run_dir}", flush=True)
    print(f"[info] csv log    : {csv_log_path}", flush=True)
    print(f"[info] device     : {device}", flush=True)

    # ── optional env check ───────────────────────────────────────────────
    if args.check_env:
        probe = make_env_fn(
            instance_paths=instance_paths, seed=args.seed,
            max_iterations=args.max_iterations, time_limit=args.time_limit,
            stall_threshold=args.stall_threshold, max_stalls=args.max_stalls,
            cplex_threads=args.cplex_threads,
            max_reset_resamples=args.max_reset_resamples,
        )()
        print("[info] running check_env ...", flush=True)
        check_env(probe.unwrapped, warn=True)
        probe.close()
        print("[info] check_env passed", flush=True)

    # ── vectorized env ───────────────────────────────────────────────────
    env_kwargs = dict(
        instance_paths=instance_paths,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
        max_reset_resamples=args.max_reset_resamples,
    )
    if args.num_envs == 1:
        vec_env = DummyVecEnv([make_env_fn(seed=args.seed, **env_kwargs)])
    else:
        vec_env = SubprocVecEnv(
            [make_env_fn(seed=args.seed + i, **env_kwargs) for i in range(args.num_envs)],
            start_method="fork",
        )
    vec_env = VecMonitor(vec_env)

    # ── PPO ──────────────────────────────────────────────────────────────
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=str(run_dir / "tb_logs") if args.tensorboard else None,
    )

    # ── callback ─────────────────────────────────────────────────────────
    callback = CallbackList([
        PPOTrainingLogger(
            print_every_episodes=args.print_every_episodes,
            rolling_window=50,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            csv_log_path=csv_log_path,
            verbose=0,
        )
    ])

    # ── train ────────────────────────────────────────────────────────────
    print("[info] starting PPO training ...", flush=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # ── save final model (name includes shape + timestamp) ───────────────
    final_model = str(run_dir / f"{args.save_name}_{shape_tag}_{timestamp}")
    print(f"[info] saving model -> {final_model}.zip", flush=True)
    model.save(final_model)

    vec_env.close()
    print("[info] training complete", flush=True)
    print(f"[info] csv log    : {csv_log_path}", flush=True)
    if log_file:
        print(f"[info] stdout log : {log_file}", flush=True)
    if _tee is not None:
        _tee.close()


if __name__ == "__main__":
    main()
