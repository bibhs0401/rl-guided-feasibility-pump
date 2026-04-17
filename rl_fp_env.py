"""
rl_fp_env.py  (v2 – real-solver edition)
==========================================

Gymnasium / SB3 environment that wraps the real Feasibility Pump from
fp_ppo.py (docplex / CPLEX backend) and lets a PPO agent control two
decisions at every stall event:

    1. flip-level  — how many binary variables to perturb
    2. continuation-level — how patient FP should be before calling RL again
       (implemented as a local stall-threshold override on the runner)

Architecture
------------
Layer A  Generic SB3 / Gymnasium logic — fully implemented here.
Layer B  FP backend hooks.  Two interchangeable implementations:

    RealFPBackend   Wraps FeasibilityPumpRunner + docplex. Used automatically
                    when fp_ppo is importable AND instance_paths is non-empty.

    MockFPBackend   Lightweight noise-based simulator.  Used as a fallback for
                    offline testing / check_env without CPLEX.

Both backends expose the **same typed interface**, so FeasibilityPumpEnv never
needs to know which one it is talking to.

Key dimensions
--------------
    N_PROGRESS = 12   10 features from build_observation_k + elapsed/remaining
    N_HISTORY  =  8   Lag-1/lag-2 distance deltas, rewards, action flags
    N_INSTANCE = 16   Paper Figure-1 features from build_instance_features

Action space
------------
    MultiDiscrete([6, 5])
        axis 0 – flip-level (relative to n_integer)
        axis 1 – continuation-level (stall-threshold multiplier)

Compatible with SB3 PPO + MultiInputPolicy.

Usage (real solver)
-------------------
    from rl_fp_env import FeasibilityPumpEnv, FPConfig
    cfg = FPConfig(instance_paths=["inst.npz", ...], seed=0)
    env = FeasibilityPumpEnv(config=cfg)     # → RealFPBackend
    obs, info = env.reset()

Usage (mock / no CPLEX)
-----------------------
    cfg = FPConfig(instance_paths=[], seed=0)
    env = FeasibilityPumpEnv(config=cfg)     # → MockFPBackend

Python ≥ 3.10.  Requires: numpy, gymnasium, stable_baselines3.
With real solver: fp_ppo.py on PYTHONPATH + docplex + CPLEX.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# fp_ppo compatibility shim
# ---------------------------------------------------------------------------
# When fp_ppo.py + docplex are available the real CPLEX backend is unlocked.
# The environment degrades gracefully to MockFPBackend when they are absent.
# ---------------------------------------------------------------------------
try:
    from fp_ppo import (
        ProblemData,
        FeasibilityPumpRunner,
        load_problem,
        build_instance_features,
        build_observation_k,
        select_flip_candidates,
        DEFAULT_CPLEX_THREADS,
        DEFAULT_STALL_THRESHOLD,
        DEFAULT_MAX_STALLS,
        DEFAULT_TIME_LIMIT,
        DYNAMIC_FEATURE_DIM,    # 10
        INSTANCE_FEATURE_DIM,   # 16
        INTEGER_TOLERANCE,
    )
    _FP_PPO_AVAILABLE = True
except ImportError:
    _FP_PPO_AVAILABLE = False
    # Fallback values so module-level constants work without fp_ppo.
    DYNAMIC_FEATURE_DIM  = 10
    INSTANCE_FEATURE_DIM = 16
    DEFAULT_CPLEX_THREADS   = 1
    DEFAULT_STALL_THRESHOLD = 3
    DEFAULT_MAX_STALLS      = 50
    DEFAULT_TIME_LIMIT      = 30.0
    INTEGER_TOLERANCE       = 1e-6

# SB3 callback base (soft dependency — only needed for training)
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object  # type: ignore


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# ── Reward coefficients ────────────────────────────────────────────────────
# Exposed at module scope so they are easy to sweep without editing the class.
# FPConfig copies these as defaults; the env always reads from cfg.
R_FEASIBLE_BONUS:    float =  10.0   # one-shot bonus for first feasible solution
R_FRAC_REDUCTION:    float =   5.0   # weight on normalised distance reduction
R_BEST_IMPROVEMENT:  float =   2.0   # extra weight when best-so-far improves
R_STALL_PENALTY:     float =  -0.5   # flat penalty when this step stalled
R_CYCLE_PENALTY:     float =  -0.3   # flat penalty when a cycle was detected
R_TIME_PENALTY:      float =  -0.2   # proportional to normalised step runtime
R_FLIP_PENALTY:      float =  -0.1   # proportional to flip_count / n_integer
R_STEP_COST:         float =  -0.01  # constant per-decision cost

# ── Flip menu ──────────────────────────────────────────────────────────────
# Six levels: absolute count (index 0) or relative fractions of n_integer.
# Each entry is a callable (n_integer: int) -> int.
FLIP_BINS: Tuple = (
    lambda n: 1,
    lambda n: max(2, math.ceil(0.01 * n)),
    lambda n: max(1, math.ceil(0.02 * n)),
    lambda n: max(1, math.ceil(0.05 * n)),
    lambda n: max(1, math.ceil(0.10 * n)),
    lambda n: max(1, math.ceil(0.20 * n)),
)
N_FLIP_BINS: int = len(FLIP_BINS)   # 6

# ── Continuation / stall-tolerance menu ───────────────────────────────────
# Each level is a multiplier on cfg.stall_threshold.
# local_thresh = max(1, round(mult * cfg.stall_threshold))
# Higher patience  → more natural FP iterations before RL is called again.
# Lower patience   → RL gets to decide more frequently.
#
# This maps cleanly onto FeasibilityPumpRunner.stall_threshold: the env
# temporarily overrides it for each advance_to_decision_point call, then
# restores it afterwards.  No single global time limit is set once and
# locked in.
CONT_STALL_MULTS: Tuple[float, ...] = (0.33, 0.67, 1.0, 1.5, 2.5)
CONT_LABELS: Tuple[str, ...]        = (
    "very_short", "short", "medium", "long", "very_long"
)
N_CONT_BINS: int = len(CONT_STALL_MULTS)   # 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FPConfig:
    """All tuneable parameters for the environment and training run.

    Serialisable to JSON for reproducibility.
    """

    # ── Instance pool ────────────────────────────────────────────────────
    # List of .npz file paths understood by fp_ppo.load_problem().
    # When empty the environment falls back to MockFPBackend.
    instance_paths: List[str] = field(default_factory=list)

    # ── FP solver limits (per episode) ───────────────────────────────────
    max_iterations:    int   = 100           # hard cap on FP LP solves
    max_stalls:        int   = DEFAULT_MAX_STALLS    # stall-event budget
    stall_threshold:   int   = DEFAULT_STALL_THRESHOLD  # base patience
    fp_time_limit_max: float = DEFAULT_TIME_LIMIT    # wall-clock seconds

    # ── CPLEX settings ───────────────────────────────────────────────────
    cplex_threads: int = DEFAULT_CPLEX_THREADS   # 0 = automatic

    # ── Reward coefficients ──────────────────────────────────────────────
    r_feasible_bonus:   float = R_FEASIBLE_BONUS
    r_frac_reduction:   float = R_FRAC_REDUCTION
    r_best_improvement: float = R_BEST_IMPROVEMENT
    r_stall_penalty:    float = R_STALL_PENALTY
    r_cycle_penalty:    float = R_CYCLE_PENALTY
    r_time_penalty:     float = R_TIME_PENALTY
    r_flip_penalty:     float = R_FLIP_PENALTY
    r_step_cost:        float = R_STEP_COST

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir:               str = "logs"
    log_prefix:            str = ""
    flush_every_n_steps:   int = 500
    flush_every_n_episodes: int = 10

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: Optional[int] = 0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# RealFPBackend
# ---------------------------------------------------------------------------

class RealFPBackend:
    """FP backend backed by FeasibilityPumpRunner (docplex / CPLEX).

    Requires fp_ppo.py to be importable and at least one instance path
    to be present in cfg.instance_paths.

    Interface contract (shared with MockFPBackend)
    -----------------------------------------------
    Properties:
        n_integer, n_vars, n_constraints, instance_id
        distance, best_distance, prev_distance
        is_feasible, is_done, is_failed
        elapsed, initial_lp_obj
        last_flip_count, total_flips, iteration
        consecutive_no_change, stall_events

    Methods:
        load_and_initialize(instance_id=None)
        get_progress_features(k_max) -> (DYNAMIC_FEATURE_DIM,) float32
        get_instance_features()      -> (INSTANCE_FEATURE_DIM,) float32
        advance_to_decision_point(local_stall_threshold) -> float (runtime)
        apply_flips(flip_count)      -> (cycle_flag: bool, stall_flag: bool)
        check_feasibility()          -> bool
        distance_normalized()        -> float in [0, 1]
    """

    def __init__(self, cfg: FPConfig, rng: np.random.Generator) -> None:
        if not _FP_PPO_AVAILABLE:
            raise RuntimeError(
                "fp_ppo is not importable. Install docplex and ensure "
                "fp_ppo.py is on the Python path, or leave instance_paths "
                "empty to use MockFPBackend."
            )
        self.cfg  = cfg
        self.rng  = rng

        self.runner: Optional[FeasibilityPumpRunner] = None
        self.problem: Optional[ProblemData]          = None
        self._instance_features: np.ndarray = np.zeros(
            INSTANCE_FEATURE_DIM, dtype=np.float32
        )
        self._instance_id_str: str  = "none"
        self._best_distance:   float = 0.0
        self._prev_distance:   float = 0.0
        self._last_flip_count_val: int = 0
        self._episode_count:   int  = 0

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def n_integer(self) -> int:
        return len(self.problem.integer_indices) if self.problem else 1

    @property
    def n_vars(self) -> int:
        return self.problem.n if self.problem else 1

    @property
    def n_constraints(self) -> int:
        return self.problem.m if self.problem else 1

    @property
    def instance_id(self) -> str:
        return self._instance_id_str

    @property
    def distance(self) -> float:
        return self.runner.current_distance() if self.runner else 0.0

    @property
    def best_distance(self) -> float:
        return self._best_distance

    @property
    def prev_distance(self) -> float:
        return self._prev_distance

    @property
    def is_feasible(self) -> bool:
        return bool(self.runner.integer_found) if self.runner else False

    @property
    def is_done(self) -> bool:
        return bool(self.runner.done) if self.runner else True

    @property
    def is_failed(self) -> bool:
        return bool(self.runner.failed) if self.runner else False

    @property
    def elapsed(self) -> float:
        if self.runner is None or self.runner.start_time is None:
            return 0.0
        return time.time() - self.runner.start_time

    @property
    def initial_lp_obj(self) -> float:
        return float(self.runner.initial_lp_objective) if self.runner else 0.0

    @property
    def last_flip_count(self) -> int:
        return self._last_flip_count_val

    @property
    def total_flips(self) -> int:
        return self.runner.total_flips if self.runner else 0

    @property
    def iteration(self) -> int:
        return self.runner.iteration if self.runner else 0

    @property
    def consecutive_no_change(self) -> int:
        return self.runner.consecutive_no_change if self.runner else 0

    @property
    def stall_events(self) -> int:
        return self.runner.stall_events if self.runner else 0

    # ── Interface methods ─────────────────────────────────────────────────

    def load_and_initialize(self, instance_id: Optional[Any] = None) -> None:
        """Load one instance from disk and build + solve the initial LP.

        Parameters
        ----------
        instance_id : str (explicit path) | int (index into cfg.instance_paths)
                      | None (sample randomly)
        """
        if isinstance(instance_id, str):
            path = instance_id
        elif isinstance(instance_id, int):
            path = self.cfg.instance_paths[instance_id % len(self.cfg.instance_paths)]
        else:
            idx  = int(self.rng.integers(0, len(self.cfg.instance_paths)))
            path = self.cfg.instance_paths[idx]

        self._instance_id_str = Path(path).stem
        self.problem = load_problem(path)

        # build_instance_features is O(n * m) — compute once per episode.
        self._instance_features = build_instance_features(self.problem)

        self._episode_count += 1
        self.runner = FeasibilityPumpRunner(
            problem=self.problem,
            max_iterations=self.cfg.max_iterations,
            time_limit=self.cfg.fp_time_limit_max,
            stall_threshold=self.cfg.stall_threshold,
            max_stalls=self.cfg.max_stalls,
            cplex_threads=self.cfg.cplex_threads,
        )
        self.runner.episode_index = self._episode_count
        # reset() builds CPLEX models and solves the initial argmax LP
        # (Algorithm 2, lines 1-6).
        self.runner.reset()

        init_dist = self.runner.current_distance()
        self._best_distance = init_dist
        self._prev_distance = init_dist
        self._last_flip_count_val = 0

    def get_progress_features(self, k_max: int) -> np.ndarray:
        """Return the 10-dim dynamic FP state from build_observation_k."""
        if self.runner is None:
            return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)
        return build_observation_k(self.runner, k_max)

    def get_instance_features(self) -> np.ndarray:
        """Return the 16-dim static instance descriptor."""
        return self._instance_features.copy()

    def advance_to_decision_point(self, local_stall_threshold: int) -> float:
        """Run free FP iterations (no flip) until stall or episode end.

        Temporarily overrides runner.stall_threshold with local_stall_threshold
        so the agent's continuation choice takes effect.  The original
        threshold is always restored, even if an exception propagates.

        Returns wall-clock seconds consumed by this call.
        """
        if self.runner is None or self.runner.done:
            return 0.0
        t0 = time.time()
        orig_threshold = self.runner.stall_threshold
        self.runner.stall_threshold = max(1, local_stall_threshold)
        try:
            while not self.runner.done and not self.runner.is_stalled():
                executed = self.runner.run_one_iteration([])
                if not executed:
                    break
        finally:
            self.runner.stall_threshold = orig_threshold
        return time.time() - t0

    def apply_flips(self, flip_count: int) -> Tuple[bool, bool]:
        """Select top-k candidates by fractionality gap and run one FP solve.

        This corresponds to Algorithm 2 lines 14-15 (the stall branch):
        x̃^I ← Flip(x̃^I), then solve the projection LP once.

        Returns
        -------
        (cycle_flag, stall_flag)
            cycle_flag  True when consecutive_no_change ≥ 2× stall_threshold
                        (a proxy for cycling, since the runner tracks no-change
                        streak but not fingerprints).
            stall_flag  True when runner.is_stalled() after the iteration.
        """
        if self.runner is None or self.runner.done:
            return False, False

        selected = select_flip_candidates(self.runner, flip_count)
        self._last_flip_count_val = len(selected)
        self._prev_distance = self.runner.current_distance()

        self.runner.run_one_iteration(selected)

        cur = self.runner.current_distance()
        if cur < self._best_distance:
            self._best_distance = cur

        stall = self.runner.is_stalled()
        cycle = (
            self.runner.consecutive_no_change
            >= 2 * max(1, self.cfg.stall_threshold)
        )
        return cycle, stall

    def check_feasibility(self) -> bool:
        """True iff runner declared an integer solution."""
        return bool(self.runner.integer_found) if self.runner else False

    def distance_normalized(self) -> float:
        """FP distance normalised to [0, 1] by n_integer."""
        if self.runner is None:
            return 0.0
        return min(1.0, self.runner.current_distance() / max(1, self.n_integer))


# ---------------------------------------------------------------------------
# MockFPBackend
# ---------------------------------------------------------------------------

class MockFPBackend:
    """Lightweight FP simulator for CPLEX-free smoke testing.

    Exposes the same interface as RealFPBackend so FeasibilityPumpEnv
    is completely agnostic to which backend is in use.

    Dynamics:
        * Per-episode distance starts at a random initial fractionality × n_int.
        * advance_to_decision_point runs natural steps with noisy drift until a
          stall (consecutive_no_change >= local_stall_threshold) is reached.
        * apply_flips applies a flip-count-dependent perturbation with noise.
        * Feasibility is declared when distance drops to zero.
    """

    def __init__(self, cfg: FPConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        # Per-episode state (reset in load_and_initialize).
        self._n_integer:    int   = 100
        self._n_vars:       int   = 120
        self._n_constraints: int  = 80
        self._hardness:     float = 0.5
        self._distance:     float = 0.0
        self._best_distance: float = 0.0
        self._prev_distance: float = 0.0
        self._elapsed:      float = 0.0
        self._initial_lp_obj: float = 0.0
        self._last_flip_count_val: int = 0
        self._total_flips:  int   = 0
        self._iteration:    int   = 0
        self._consecutive_no_change: int = 0
        self._stall_events: int   = 0
        self._is_feasible:  bool  = False
        self._is_failed:    bool  = False
        self._instance_features: np.ndarray = np.zeros(
            INSTANCE_FEATURE_DIM, dtype=np.float32
        )
        self._recent_deltas: List[float] = []
        self._episode_count: int = 0
        self._flips_since_last_improvement: int = 0
        self._last_k: int = 0

    # ── Properties (mirroring RealFPBackend) ─────────────────────────────

    @property
    def n_integer(self) -> int:
        return self._n_integer

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def n_constraints(self) -> int:
        return self._n_constraints

    @property
    def instance_id(self) -> str:
        return f"mock_{self._episode_count}"

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def best_distance(self) -> float:
        return self._best_distance

    @property
    def prev_distance(self) -> float:
        return self._prev_distance

    @property
    def is_feasible(self) -> bool:
        return self._is_feasible

    @property
    def is_done(self) -> bool:
        return self._is_feasible or self._is_failed

    @property
    def is_failed(self) -> bool:
        return self._is_failed

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def initial_lp_obj(self) -> float:
        return self._initial_lp_obj

    @property
    def last_flip_count(self) -> int:
        return self._last_flip_count_val

    @property
    def total_flips(self) -> int:
        return self._total_flips

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def consecutive_no_change(self) -> int:
        return self._consecutive_no_change

    @property
    def stall_events(self) -> int:
        return self._stall_events

    # ── Interface methods ─────────────────────────────────────────────────

    def load_and_initialize(self, instance_id: Optional[Any] = None) -> None:
        """Randomly generate a mock MIP instance and initialise FP state."""
        self._episode_count += 1
        n_b = int(self.rng.integers(50, 301))
        self._n_integer     = n_b
        self._n_vars        = n_b + int(self.rng.integers(0, max(1, n_b // 4) + 1))
        self._n_constraints = int(self.rng.integers(40, 401))
        self._hardness      = float(self.rng.uniform(0.05, 0.95))
        init_frac           = float(self.rng.uniform(0.05, 0.45))
        self._distance      = init_frac * n_b
        self._best_distance = self._distance
        self._prev_distance = self._distance
        self._elapsed       = 0.0
        self._initial_lp_obj = float(self.rng.uniform(-1.0, 1.0))
        self._last_flip_count_val = 0
        self._total_flips   = 0
        self._iteration     = 0
        self._consecutive_no_change = 0
        self._stall_events  = 0
        self._is_feasible   = False
        self._is_failed     = False
        self._recent_deltas = []
        self._flips_since_last_improvement = 0
        self._last_k        = 0
        # Plausible instance features in the expected [-1, 1] range.
        self._instance_features = np.clip(
            self.rng.uniform(-0.5, 1.0, INSTANCE_FEATURE_DIM), -1.0, 1.0
        ).astype(np.float32)

    def get_progress_features(self, k_max: int) -> np.ndarray:
        """Compute DYNAMIC_FEATURE_DIM features matching build_observation_k."""
        n = max(1, self._n_integer)
        # Approximate fractionality from distance.
        scale        = min(1.0, self._distance / max(1.0, n))
        mean_frac    = min(1.0, scale)
        max_frac     = min(1.0, scale * 1.5)
        frac_ratio   = min(1.0, scale)
        dist_ratio   = min(1.0, self._distance / n)
        stall_ratio  = min(1.0, self._consecutive_no_change
                          / max(1, self.cfg.stall_threshold))
        iter_ratio   = min(1.0, self._iteration
                          / max(1, self.cfg.max_iterations))
        recent_delta = float(np.clip(
            np.mean(self._recent_deltas) / max(1.0, n)
            if self._recent_deltas else 0.0, -1.0, 1.0
        ))
        stall_depth  = min(1.0, self._stall_events
                          / max(1, self.cfg.max_stalls))
        last_k_ratio = min(1.0, self._last_k / max(1, k_max))
        flips_since  = min(1.0, self._flips_since_last_improvement
                          / max(1, k_max))
        return np.array([
            mean_frac, max_frac, frac_ratio, dist_ratio,
            stall_ratio, iter_ratio, recent_delta, stall_depth,
            last_k_ratio, flips_since,
        ], dtype=np.float32)

    def get_instance_features(self) -> np.ndarray:
        return self._instance_features.copy()

    def advance_to_decision_point(self, local_stall_threshold: int) -> float:
        """Run natural FP steps until stall or done. Returns simulated runtime."""
        if self.is_done:
            return 0.0
        t0 = self._elapsed
        threshold = max(1, local_stall_threshold)
        while not self.is_done:
            self._run_natural_step()
            if self._consecutive_no_change >= threshold:
                break
        return self._elapsed - t0

    def _run_natural_step(self) -> None:
        """One natural (no-flip) FP iteration with noisy fractionality drift."""
        if self.is_done:
            return
        self._iteration += 1
        self._elapsed  += 0.002 + 0.00002 * self._n_vars
        self._prev_distance = self._distance

        drift    = self.rng.normal(
            -0.3 * (1.0 - self._hardness),
            0.5 + 0.5 * self._hardness,
        )
        new_dist = float(np.clip(self._distance + drift, 0.0, self._n_integer))
        delta    = self._distance - new_dist
        self._distance = new_dist

        self._recent_deltas.append(delta)
        if len(self._recent_deltas) > 5:
            self._recent_deltas.pop(0)

        if new_dist < self._best_distance:
            self._best_distance = new_dist
            self._consecutive_no_change = 0
            self._flips_since_last_improvement = 0
        else:
            self._consecutive_no_change += 1

        # Feasibility when distance collapses below tolerance.
        if self._distance <= 0.5:
            self._distance    = 0.0
            self._is_feasible = True
            return

        # Stall event book-keeping.
        if self._consecutive_no_change == self.cfg.stall_threshold:
            self._stall_events += 1

        # Hard termination conditions.
        if (
            self._iteration >= self.cfg.max_iterations
            or self._elapsed >= self.cfg.fp_time_limit_max
            or self._stall_events >= self.cfg.max_stalls
        ):
            self._is_failed = True

    def apply_flips(self, flip_count: int) -> Tuple[bool, bool]:
        """Apply noisy flip perturbation. Returns (cycle_flag, stall_flag)."""
        self._last_flip_count_val = flip_count
        self._total_flips        += flip_count
        self._last_k              = flip_count
        self._prev_distance       = self._distance

        expected = flip_count * (1.0 - 0.8 * self._hardness) * 0.5
        noise    = self.rng.normal(0.0, 1.0 + 0.02 * flip_count)
        self._distance = float(
            np.clip(self._distance - expected + noise, 0.0, self._n_integer)
        )
        if self._distance < self._best_distance:
            self._best_distance = self._distance
            self._flips_since_last_improvement = 0
        else:
            self._flips_since_last_improvement += flip_count

        # Flip always resets the no-change streak.
        self._consecutive_no_change = 0

        stall = (self._consecutive_no_change >= self.cfg.stall_threshold)
        cycle = False   # flipping makes a cycle unlikely immediately
        return cycle, stall

    def check_feasibility(self) -> bool:
        return self._is_feasible

    def distance_normalized(self) -> float:
        return min(1.0, self._distance / max(1.0, float(self._n_integer)))


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class FPLogger:
    """Research-friendly step + episode logger.

    Buffers rows in memory and flushes them periodically to:
        <log_dir>/step_logs.csv        <log_dir>/step_logs.jsonl
        <log_dir>/episode_logs.csv     <log_dir>/episode_logs.jsonl
        <log_dir>/config.json          (written once at construction)

    All CSV columns are flat scalars for clean pandas ingestion.
    """

    STEP_FIELDS: Tuple[str, ...] = (
        "episode_id", "instance_id", "step_id",
        "nloops", "nstallloops",
        "distance", "best_distance",
        "flip_bin", "flip_count",
        "continuation_bin", "continuation_value",
        "stall_flag", "cycle_flag", "feasible_found",
        "step_runtime", "elapsed_fp_time", "remaining_fp_time",
        "reward",
        "r_frac", "r_best", "r_feas", "r_stall",
        "r_cycle", "r_time", "r_flip", "r_step",
        "terminated", "truncated", "termination_reason",
    )

    EPISODE_FIELDS: Tuple[str, ...] = (
        "episode_id", "instance_id",
        "total_steps", "total_reward",
        "feasible_found", "success_flag",
        "final_distance", "best_distance",
        "total_fp_time", "total_flips_used",
        "num_stalls", "num_cycles",
        "termination_reason",
    )

    def __init__(self, cfg: FPConfig) -> None:
        self.cfg     = cfg
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        prefix = cfg.log_prefix.strip()
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"

        self.step_csv    = self.log_dir / f"{prefix}step_logs.csv"
        self.ep_csv      = self.log_dir / f"{prefix}episode_logs.csv"
        self.step_jsonl  = self.log_dir / f"{prefix}step_logs.jsonl"
        self.ep_jsonl    = self.log_dir / f"{prefix}episode_logs.jsonl"
        self.config_json = self.log_dir / f"{prefix}config.json"

        self._step_buf: List[Dict[str, Any]] = []
        self._ep_buf:   List[Dict[str, Any]] = []
        self._step_n = 0
        self._ep_n   = 0

        self._ensure_header(self.step_csv, self.STEP_FIELDS)
        self._ensure_header(self.ep_csv,   self.EPISODE_FIELDS)

        with self.config_json.open("w") as f:
            json.dump(cfg.to_json(), f, indent=2, default=str)

    @staticmethod
    def _ensure_header(path: Path, fields: Tuple[str, ...]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(fields)).writeheader()

    def log_step(self, row: Dict[str, Any]) -> None:
        self._step_buf.append({k: row.get(k) for k in self.STEP_FIELDS})
        self._step_n += 1
        if self._step_n % max(1, self.cfg.flush_every_n_steps) == 0:
            self._flush_step()

    def log_episode(self, row: Dict[str, Any]) -> None:
        self._ep_buf.append({k: row.get(k) for k in self.EPISODE_FIELDS})
        self._ep_n += 1
        if self._ep_n % max(1, self.cfg.flush_every_n_episodes) == 0:
            self._flush_episode()

    def save(self) -> None:
        """Flush all remaining in-memory rows to disk."""
        self._flush_step()
        self._flush_episode()

    def _flush_step(self) -> None:
        if not self._step_buf:
            return
        with self.step_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(self.STEP_FIELDS)).writerows(
                self._step_buf
            )
        with self.step_jsonl.open("a") as f:
            for r in self._step_buf:
                f.write(json.dumps(r, default=str) + "\n")
        self._step_buf.clear()

    def _flush_episode(self) -> None:
        if not self._ep_buf:
            return
        with self.ep_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(self.EPISODE_FIELDS)).writerows(
                self._ep_buf
            )
        with self.ep_jsonl.open("a") as f:
            for r in self._ep_buf:
                f.write(json.dumps(r, default=str) + "\n")
        self._ep_buf.clear()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FeasibilityPumpEnv(gym.Env):
    """Gymnasium environment wrapping one FP run.

    Observation (Dict):
        "progress"  (12,) float32  10 dynamic FP-state features + elapsed + remaining
        "history"   ( 8,) float32  Lag-1/lag-2 deltas, reward lags, action flags
        "instance"  (16,) float32  Static instance descriptors (paper Fig. 1)

    Action (MultiDiscrete([6, 5])):
        [flip_bin, continuation_bin]

    Episode lifecycle:
        reset()   → load instance, build LP models, solve initial LP,
                    advance to first decision point.
        step()    → safety advance → apply flip → advance to next stall.
        close()   → flush logs.

    Termination:
        terminated = True  when the FP finds a feasible integer solution
                           (runner.integer_found) or the solver fails.
        truncated  = True  when the FP hits max_iterations / max_stalls /
                           time limit without a feasible solution.
    """

    metadata = {"render_modes": []}

    # Sub-observation sizes.
    N_PROGRESS: int = DYNAMIC_FEATURE_DIM + 2   # 12
    N_HISTORY:  int = 8
    N_INSTANCE: int = INSTANCE_FEATURE_DIM      # 16

    def __init__(
        self,
        config: Optional[FPConfig] = None,
        backend=None,
        logger: Optional[FPLogger] = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else FPConfig()

        self._np_random: np.random.Generator = np.random.default_rng(self.cfg.seed)

        if backend is not None:
            self.backend = backend
        elif _FP_PPO_AVAILABLE and self.cfg.instance_paths:
            self.backend = RealFPBackend(self.cfg, self._np_random)
        else:
            self.backend = MockFPBackend(self.cfg, self._np_random)

        self.logger: Optional[FPLogger] = logger

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete([N_FLIP_BINS, N_CONT_BINS])
        self.observation_space = spaces.Dict({
            # progress uses [-1, 1] because the signed distance-delta can be
            # negative when fractionality temporarily increases.
            "progress": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_PROGRESS,), dtype=np.float32,
            ),
            "history": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_HISTORY,), dtype=np.float32,
            ),
            "instance": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_INSTANCE,), dtype=np.float32,
            ),
        })

        # ── Per-episode state ─────────────────────────────────────────────
        self._episode_id: int = -1
        self._k_max:      int = 1
        self._step_id     = 0
        self._nloops      = 0
        self._nstallloops = 0
        self._total_reward        = 0.0
        self._prev_reward         = 0.0
        self._prev_prev_reward    = 0.0
        self._prev_dist_delta     = 0.0
        self._prev_prev_dist_delta = 0.0
        self._last_action_caused_stall = False
        self._last_action_broke_stall  = False
        self._cum_flips           = 0
        self._consec_non_improving = 0
        self._num_stalls          = 0
        self._num_cycles          = 0
        self._termination_reason  = ""

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.backend.rng = self._np_random

        # Reset per-episode bookkeeping.
        self._episode_id += 1
        self._step_id     = 0
        self._nloops      = 0
        self._nstallloops = 0
        self._total_reward = 0.0
        self._prev_reward  = 0.0
        self._prev_prev_reward    = 0.0
        self._prev_dist_delta     = 0.0
        self._prev_prev_dist_delta = 0.0
        self._last_action_caused_stall = False
        self._last_action_broke_stall  = False
        self._cum_flips           = 0
        self._consec_non_improving = 0
        self._num_stalls          = 0
        self._num_cycles          = 0
        self._termination_reason  = ""

        instance_id = None if options is None else options.get("instance_id")

        # Load instance + initialise LP models + solve initial LP.
        # For the real backend this involves docplex model construction and
        # the argmax LP solve (Algorithm 2, lines 1-6).
        self.backend.load_and_initialize(instance_id=instance_id)

        # k_max = largest possible flip count for this instance.
        self._k_max = max(1, FLIP_BINS[-1](self.backend.n_integer))

        # Advance to first stall using base (medium) patience.
        # After this call the env is at the first decision point where
        # the agent should act.
        base_thresh = self.cfg.stall_threshold
        if not self.backend.is_done:
            self.backend.advance_to_decision_point(base_thresh)

        obs  = self._build_observation()
        info = {
            "episode_id":      self._episode_id,
            "instance_id":     self.backend.instance_id,
            "n_integer":       self.backend.n_integer,
            "initial_distance": self.backend.distance,
            "backend_type":    type(self.backend).__name__,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

        # 1. Decode action ─────────────────────────────────────────────────
        flip_bin = int(np.clip(int(action[0]), 0, N_FLIP_BINS - 1))
        cont_bin = int(np.clip(int(action[1]), 0, N_CONT_BINS - 1))

        flip_count   = FLIP_BINS[flip_bin](self.backend.n_integer)
        cont_mult    = CONT_STALL_MULTS[cont_bin]
        cont_label   = CONT_LABELS[cont_bin]
        local_thresh = max(1, int(round(cont_mult * self.cfg.stall_threshold)))

        # 2. Snapshot state before this decision ───────────────────────────
        prev_distance = self.backend.distance
        prev_best     = self.backend.best_distance
        elapsed_before = self.backend.elapsed

        # 3. Safety advance (ensures we are at a stall before flipping) ────
        #    Handles the edge-case where the last step or reset left the
        #    runner not-quite-stalled (e.g., very last iteration hit
        #    max_iterations without triggering is_stalled()).
        if not self.backend.is_done:
            self.backend.advance_to_decision_point(local_thresh)

        # 4. Apply the flip chosen by the agent ────────────────────────────
        #    This is Algorithm 2 line 15: x̃^I ← Flip(x̃^I), then one LP solve.
        cycle_flag, stall_flag = False, False
        if not self.backend.is_done:
            cycle_flag, stall_flag = self.backend.apply_flips(flip_count)

        # 5. Advance to next decision point ────────────────────────────────
        #    Algorithm 2 outer loop continues until the next stall.
        if not self.backend.is_done:
            self.backend.advance_to_decision_point(local_thresh)

        # 6. Collect post-step metrics ──────────────────────────────────────
        feasible     = self.backend.check_feasibility()
        failed       = self.backend.is_failed
        new_distance = self.backend.distance
        new_best     = self.backend.best_distance
        elapsed      = self.backend.elapsed
        step_runtime = max(0.0, elapsed - elapsed_before)
        remaining    = max(0.0, self.cfg.fp_time_limit_max - elapsed)

        # 7. Update episode counters ────────────────────────────────────────
        self._step_id += 1
        self._nloops   = self.backend.iteration
        self._cum_flips += flip_count
        was_stalled_before = self._last_action_caused_stall

        if stall_flag:
            self._num_stalls  += 1
            self._nstallloops += 1
        else:
            self._nstallloops = 0
        if cycle_flag:
            self._num_cycles += 1

        improved = new_distance < prev_distance - 1e-9
        self._consec_non_improving = 0 if improved else self._consec_non_improving + 1
        self._last_action_caused_stall = stall_flag
        self._last_action_broke_stall  = was_stalled_before and (not stall_flag)

        # 8. Reward ─────────────────────────────────────────────────────────
        reward, comps = self._compute_reward(
            prev_distance=prev_distance,
            prev_best=prev_best,
            new_distance=new_distance,
            new_best=new_best,
            feasible=feasible,
            stall=stall_flag,
            cycle=cycle_flag,
            step_runtime=step_runtime,
            flip_count=flip_count,
        )
        self._total_reward += reward

        # 9. Termination / truncation ───────────────────────────────────────
        terminated = feasible or failed
        truncated  = False
        if not terminated and self.backend.is_done:
            truncated = True
            self._termination_reason = "maxloops_or_stalls_or_time"
        elif feasible:
            self._termination_reason = "feasible_found"
        elif failed:
            self._termination_reason = "solver_failed"

        # 10. Update history buffers ────────────────────────────────────────
        dist_delta = prev_distance - new_distance
        self._prev_prev_dist_delta = self._prev_dist_delta
        self._prev_dist_delta      = dist_delta
        self._prev_prev_reward     = self._prev_reward
        self._prev_reward          = reward

        # 11. Build observation and info ────────────────────────────────────
        obs = self._build_observation()
        info: Dict[str, Any] = {
            # ── Core FP state ──────────────────────────────────────────
            "episode_id":    self._episode_id,
            "instance_id":   self.backend.instance_id,
            "step_id":       self._step_id,
            "nloops":        self._nloops,
            "nstallloops":   self._nstallloops,
            "distance":      float(new_distance),
            "best_distance": float(new_best),
            # ── Action taken ───────────────────────────────────────────
            "flip_bin":          flip_bin,
            "flip_count":        int(flip_count),
            "continuation_bin":  cont_bin,
            "continuation_value": float(cont_mult),
            "continuation_label": cont_label,
            # ── Diagnostic flags ───────────────────────────────────────
            "stall_flag":    int(stall_flag),
            "cycle_flag":    int(cycle_flag),
            "feasible_found": int(feasible),
            "failed":        int(failed),
            # ── Timing ─────────────────────────────────────────────────
            "step_runtime":     float(step_runtime),
            "elapsed_fp_time":  float(elapsed),
            "remaining_fp_time": float(remaining),
            # ── Reward breakdown ───────────────────────────────────────
            "reward":   float(reward),
            "r_frac":   comps["frac"],
            "r_best":   comps["best"],
            "r_feas":   comps["feas"],
            "r_stall":  comps["stall"],
            "r_cycle":  comps["cycle"],
            "r_time":   comps["time"],
            "r_flip":   comps["flip"],
            "r_step":   comps["step"],
            # ── Episode status ─────────────────────────────────────────
            "terminated":        int(terminated),
            "truncated":         int(truncated),
            "termination_reason": self._termination_reason,
            # ── Runner stats (keep TrainingLoggerCallback compatible) ──
            "integer_found":                    int(feasible),
            "iterations":                       self._nloops,
            "decisions":                        self.backend.iteration,
            "total_flips":                      self.backend.total_flips,
            "stall_events":                     self.backend.stall_events,
            "elapsed_seconds":                  float(elapsed),
            "last_distance_solve_seconds":      float(step_runtime),
            "load_seconds":                     0.0,
            "reset_seconds":                    0.0,
        }

        if self.logger is not None:
            self.logger.log_step(info)
            if terminated or truncated:
                self.logger.log_episode(self._build_episode_log(feasible))

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.logger is not None:
            self.logger.save()

    def save_logs(self) -> None:
        if self.logger is not None:
            self.logger.save()

    # ──────────────────────────────────────────────────────────────────────
    # Observation, reward, episode log
    # ──────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, np.ndarray]:
        n_int    = max(1, self.backend.n_integer)
        elapsed  = self.backend.elapsed
        remaining = max(0.0, self.cfg.fp_time_limit_max - elapsed)

        # ── Progress: 10 dynamic features + 2 time features ───────────────
        dynamic10 = self.backend.get_progress_features(self._k_max)
        time_feats = np.array([
            min(1.0, elapsed   / max(1e-8, self.cfg.fp_time_limit_max)),
            min(1.0, remaining / max(1e-8, self.cfg.fp_time_limit_max)),
        ], dtype=np.float32)
        progress = np.clip(
            np.concatenate([dynamic10, time_feats]).astype(np.float32),
            -1.0, 1.0,
        )

        # ── History: lag-1/lag-2 + reward lags + action meta ──────────────
        history = np.array([
            float(np.clip(self._prev_dist_delta      / n_int, -1.0, 1.0)),
            float(np.clip(self._prev_prev_dist_delta  / n_int, -1.0, 1.0)),
            float(np.clip(self._prev_reward      / 10.0, -1.0, 1.0)),
            float(np.clip(self._prev_prev_reward  / 10.0, -1.0, 1.0)),
            1.0 if self._last_action_caused_stall else 0.0,
            1.0 if self._last_action_broke_stall  else 0.0,
            float(np.clip(self._cum_flips / max(1, 10 * n_int), 0.0, 1.0)),
            float(np.clip(
                self._consec_non_improving / max(1, self.cfg.max_stalls),
                0.0, 1.0,
            )),
        ], dtype=np.float32)
        history = np.clip(history, -1.0, 1.0)

        # ── Instance: 16 static features from build_instance_features ─────
        instance = np.clip(
            self.backend.get_instance_features().astype(np.float32),
            -1.0, 1.0,
        )

        return {"progress": progress, "history": history, "instance": instance}

    def _compute_reward(
        self,
        prev_distance: float,
        prev_best: float,
        new_distance: float,
        new_best: float,
        feasible: bool,
        stall: bool,
        cycle: bool,
        step_runtime: float,
        flip_count: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Named-coefficient reward matching the paper's FP objectives.

        Components
        ----------
        r_frac  Rewards distance reduction (normalised by n_integer).
        r_best  Extra reward when best-so-far improves.
        r_feas  Large one-shot bonus on first feasible integer solution.
        r_stall Flat penalty when this step stalled.
        r_cycle Flat penalty when a cycle was detected.
        r_time  Penalty proportional to normalised step runtime.
        r_flip  Mild penalty proportional to flip_count / n_integer
                (discourages unnecessarily aggressive flips).
        r_step  Constant per-decision cost.
        """
        cfg   = self.cfg
        n_int = max(1, self.backend.n_integer)

        delta     = (prev_distance - new_distance) / n_int
        best_gain = max(0.0, prev_best - new_best)   / n_int
        rt_norm   = float(np.clip(step_runtime / max(1e-8, cfg.fp_time_limit_max), 0.0, 1.0))
        flip_frac = float(np.clip(flip_count / n_int, 0.0, 1.0))

        r_frac  = cfg.r_frac_reduction   * float(delta)
        r_best  = cfg.r_best_improvement * float(best_gain)
        r_feas  = cfg.r_feasible_bonus   if feasible else 0.0
        r_stall = cfg.r_stall_penalty    if stall     else 0.0
        r_cycle = cfg.r_cycle_penalty    if cycle     else 0.0
        r_time  = cfg.r_time_penalty     * rt_norm
        r_flip  = cfg.r_flip_penalty     * flip_frac
        r_step  = cfg.r_step_cost

        total = r_frac + r_best + r_feas + r_stall + r_cycle + r_time + r_flip + r_step
        comps = dict(
            frac=r_frac, best=r_best, feas=r_feas, stall=r_stall,
            cycle=r_cycle, time=r_time, flip=r_flip, step=r_step,
        )
        return float(total), comps

    def _build_episode_log(self, feasible: bool) -> Dict[str, Any]:
        return {
            "episode_id":       self._episode_id,
            "instance_id":      self.backend.instance_id,
            "total_steps":      self._step_id,
            "total_reward":     float(self._total_reward),
            "feasible_found":   int(feasible),
            "success_flag":     int(feasible),
            "final_distance":   float(self.backend.distance),
            "best_distance":    float(self.backend.best_distance),
            "total_fp_time":    float(self.backend.elapsed),
            "total_flips_used": int(self.backend.total_flips),
            "num_stalls":       int(self._num_stalls),
            "num_cycles":       int(self._num_cycles),
            "termination_reason": self._termination_reason or "unknown",
        }


# ---------------------------------------------------------------------------
# SB3 callback
# ---------------------------------------------------------------------------

class TrainingStatsCallback(BaseCallback):
    """Accumulate high-level training metrics and flush to CSV periodically.

    Reads the `info` dicts passed by SB3's VecEnv at each step.
    Also compatible with the existing TrainingLoggerCallback pattern in
    train_ppo.py — the field names in info match both.

    Output: <out_dir>/training_stats.csv
    """

    COLUMNS: Tuple[str, ...] = (
        "episode_idx", "timesteps",
        "episode_reward_100", "feasible_rate_100",
        "avg_final_distance_100", "avg_fp_time_100",
        "avg_flips_per_step_100", "no_flip_ratio_100",
        "stall_recovery_count", "failure_rate_100",
    )

    def __init__(
        self,
        out_dir: str = "logs",
        file_name: str = "training_stats.csv",
        flush_every: int = 10,
        print_episode_logs: bool = True,
        print_every_episodes: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.out_dir    = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path   = self.out_dir / file_name
        self.flush_every = max(1, flush_every)
        self.print_episode_logs = bool(print_episode_logs)
        self.print_every_episodes = max(1, int(print_every_episodes))

        self._ep_rewards:  List[float] = []
        self._ep_feasible: List[int]   = []
        self._ep_dist:     List[float] = []
        self._ep_time:     List[float] = []
        self._ep_fps:      List[float] = []
        self._ep_noflip:   List[float] = []
        self._stall_recovery_count: int = 0

        self._cur: Dict[int, Dict[str, Any]] = {}
        self._episode_idx: int = 0
        self._buf: List[Dict[str, Any]] = []

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=list(self.COLUMNS)).writeheader()

    def _reset_env(self, idx: int) -> None:
        self._cur[idx] = {
            "reward": 0.0, "steps": 0,
            "flips": 0, "noflip": 0,
            "prev_stall": False,
        }

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos",   []) or []
        rewards = self.locals.get("rewards", []) or []
        dones   = self.locals.get("dones",   []) or []

        for i, info in enumerate(infos):
            if i not in self._cur:
                self._reset_env(i)
            cur = self._cur[i]

            r = float(rewards[i]) if i < len(rewards) else 0.0
            cur["reward"] += r
            cur["steps"]  += 1
            fc = int(info.get("flip_count", 0))
            cur["flips"] += fc
            if fc <= 1:
                cur["noflip"] += 1

            stall_now = bool(info.get("stall_flag", 0))
            if cur["prev_stall"] and not stall_now:
                self._stall_recovery_count += 1
            cur["prev_stall"] = stall_now

            done = bool(dones[i]) if i < len(dones) else False
            if done or info.get("terminated") or info.get("truncated"):
                self._episode_idx += 1
                self._ep_rewards.append(cur["reward"])
                self._ep_feasible.append(int(info.get("feasible_found", 0)))
                self._ep_dist.append(float(info.get("distance", 0.0)))
                self._ep_time.append(float(info.get("elapsed_fp_time", 0.0)))
                s = max(1, cur["steps"])
                self._ep_fps.append(cur["flips"] / s)
                self._ep_noflip.append(cur["noflip"] / s)
                self._buf.append(self._summary_row())
                if self.print_episode_logs and (self._episode_idx % self.print_every_episodes == 0):
                    print(
                        "[EP] "
                        f"idx={self._episode_idx} "
                        f"inst={info.get('instance_id', 'unknown')} "
                        f"feasible={int(info.get('feasible_found', 0))} "
                        f"failed={int(info.get('failed', 0))} "
                        f"steps={s} "
                        f"iters={int(info.get('nloops', info.get('iterations', 0)))} "
                        f"dist={float(info.get('distance', 0.0)):.4f} "
                        f"r={cur['reward']:.4f} "
                        f"time={float(info.get('elapsed_fp_time', info.get('elapsed_seconds', 0.0))):.2f}s"
                    )
                if len(self._buf) >= self.flush_every:
                    self._flush()
                self._reset_env(i)
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _summary_row(self) -> Dict[str, Any]:
        def tail_mean(xs: List, n: int = 100) -> float:
            t = xs[-n:]
            return float(sum(t) / max(1, len(t))) if t else float("nan")

        feas = tail_mean(self._ep_feasible)
        return {
            "episode_idx":          self._episode_idx,
            "timesteps":            int(self.num_timesteps),
            "episode_reward_100":   tail_mean(self._ep_rewards),
            "feasible_rate_100":    feas,
            "avg_final_distance_100": tail_mean(self._ep_dist),
            "avg_fp_time_100":      tail_mean(self._ep_time),
            "avg_flips_per_step_100": tail_mean(self._ep_fps),
            "no_flip_ratio_100":    tail_mean(self._ep_noflip),
            "stall_recovery_count": self._stall_recovery_count,
            "failure_rate_100":     1.0 - feas,
        }

    def _flush(self) -> None:
        if not self._buf:
            return
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(self.COLUMNS)).writerows(self._buf)
        self._buf.clear()


# ---------------------------------------------------------------------------
# __main__: smoke test + short training (real or mock)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob as _glob

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    # ── CLI ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train PPO on FeasibilityPumpEnv (real or mock backend)."
    )
    parser.add_argument(
        "--instances", default="",
        help="Glob pattern for .npz instance files, e.g. data/*.npz. "
             "Leave empty to use MockFPBackend.",
    )
    parser.add_argument("--log-dir",      default="logs")
    parser.add_argument(
        "--log-prefix",
        default="",
        help="Optional prefix added to output log files (e.g. run1_seed42).",
    )
    parser.add_argument(
        "--auto-log-suffix",
        action="store_true",
        help="Append a timestamp suffix to --log-prefix for unique per-run filenames.",
    )
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--n-steps",      type=int, default=1024)
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--gae-lambda",   type=float, default=0.95)
    parser.add_argument("--ent-coef",     type=float, default=0.01)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--time-limit",   type=float,
                        default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--cplex-threads", type=int,
                        default=DEFAULT_CPLEX_THREADS)
    parser.add_argument("--check",        action="store_true",
                        help="Run check_env before training.")
    parser.add_argument(
        "--print-episode-logs",
        action="store_true",
        default=True,
        help="Print per-episode summaries to terminal during training.",
    )
    parser.add_argument(
        "--no-print-episode-logs",
        action="store_true",
        help="Disable per-episode terminal logs.",
    )
    parser.add_argument(
        "--print-every-episodes",
        type=int,
        default=1,
        help="Print one episode summary every N completed episodes.",
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    resolved_prefix = args.log_prefix.strip()
    if args.auto_log_suffix:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_prefix = f"{resolved_prefix}_{stamp}" if resolved_prefix else f"run_{stamp}"

    file_prefix = resolved_prefix
    if file_prefix and not file_prefix.endswith("_"):
        file_prefix = f"{file_prefix}_"

    # ── Resolve instance paths ────────────────────────────────────────────
    instance_paths: List[str] = sorted(_glob.glob(args.instances)) \
        if args.instances else []

    if instance_paths:
        print(f"[INFO] Real backend — {len(instance_paths)} instance(s) found.")
    else:
        print("[INFO] No instances supplied or fp_ppo unavailable — using MockFPBackend.")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = FPConfig(
        instance_paths=instance_paths,
        fp_time_limit_max=args.time_limit,
        cplex_threads=args.cplex_threads,
        log_dir=args.log_dir,
        log_prefix=resolved_prefix,
        seed=args.seed,
    )
    fp_logger = FPLogger(cfg)

    # Save full reproducibility snapshot.
    ppo_hparams = dict(
        policy="MultiInputPolicy",
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
    )
    run_config = {
        "env_config": cfg.to_json(),
        "ppo_hparams": ppo_hparams,
        "flip_bins": [
            "1", "max(2,ceil(0.01*n))", "ceil(0.02*n)",
            "ceil(0.05*n)", "ceil(0.10*n)", "ceil(0.20*n)",
        ],
        "cont_bins": list(CONT_LABELS),
        "cont_stall_mults": list(CONT_STALL_MULTS),
        "reward_coefficients": {
            "R_FEASIBLE_BONUS":   R_FEASIBLE_BONUS,
            "R_FRAC_REDUCTION":   R_FRAC_REDUCTION,
            "R_BEST_IMPROVEMENT": R_BEST_IMPROVEMENT,
            "R_STALL_PENALTY":    R_STALL_PENALTY,
            "R_CYCLE_PENALTY":    R_CYCLE_PENALTY,
            "R_TIME_PENALTY":     R_TIME_PENALTY,
            "R_FLIP_PENALTY":     R_FLIP_PENALTY,
            "R_STEP_COST":        R_STEP_COST,
        },
        "backend": "RealFPBackend" if (instance_paths and _FP_PPO_AVAILABLE)
                   else "MockFPBackend",
    }
    with open(os.path.join(args.log_dir, f"{file_prefix}run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ── Build env ─────────────────────────────────────────────────────────
    raw_env = FeasibilityPumpEnv(config=cfg, logger=fp_logger)

    if args.check:
        print("[INFO] Running check_env …")
        check_env(raw_env, warn=True)
        print("[INFO] check_env: OK")

    env = Monitor(raw_env)

    # ── PPO with MultiInputPolicy (required for Dict obs) ─────────────────
    model = PPO(
        ppo_hparams["policy"],
        env,
        verbose=1,
        n_steps=ppo_hparams["n_steps"],
        batch_size=ppo_hparams["batch_size"],
        learning_rate=ppo_hparams["learning_rate"],
        gamma=ppo_hparams["gamma"],
        gae_lambda=ppo_hparams["gae_lambda"],
        ent_coef=ppo_hparams["ent_coef"],
        seed=ppo_hparams["seed"],
    )

    stats_cb = TrainingStatsCallback(
        out_dir=args.log_dir,
        file_name=f"{file_prefix}training_stats.csv" if file_prefix else "training_stats.csv",
        flush_every=5,
        print_episode_logs=(args.print_episode_logs and not args.no_print_episode_logs),
        print_every_episodes=args.print_every_episodes,
        verbose=1,
    )
    model.learn(
        total_timesteps=ppo_hparams["total_timesteps"],
        callback=stats_cb,
    )

    # ── Flush logs ────────────────────────────────────────────────────────
    raw_env.save_logs()
    env.close()

    # ── Short rollout diagnostic ──────────────────────────────────────────
    print("\n[INFO] Rollout diagnostic (10 steps, deterministic policy):")
    diag_env = FeasibilityPumpEnv(config=cfg)
    obs, _ = diag_env.reset(seed=args.seed + 1)
    total_r = 0.0
    for t in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = diag_env.step(action)
        total_r += r
        print(
            f"  t={t:02d} | act=({int(action[0])},{int(action[1])}) | "
            f"dist={info['distance']:7.3f} | best={info['best_distance']:7.3f} | "
            f"flip={info['flip_count']:3d} | cont={info['continuation_label']:<10s} | "
            f"r={r:+.3f} | stall={info['stall_flag']} cycle={info['cycle_flag']}"
        )
        if term or trunc:
            print(f"  → episode end: {info['termination_reason']}  "
                  f"total_r={total_r:+.3f}")
            break
    diag_env.close()

    # ── Load logs with pandas ─────────────────────────────────────────────
    try:
        import pandas as pd
        step_name = f"{file_prefix}step_logs.csv"
        ep_name = f"{file_prefix}episode_logs.csv"
        stats_name = f"{file_prefix}training_stats.csv" if file_prefix else "training_stats.csv"
        step_df  = pd.read_csv(os.path.join(args.log_dir, step_name))
        ep_df    = pd.read_csv(os.path.join(args.log_dir, ep_name))
        stats_df = pd.read_csv(os.path.join(args.log_dir, stats_name))
        print(f"\n{step_name:<18}: {step_df.shape}")
        print(f"{ep_name:<18}: {ep_df.shape}")
        print(f"{stats_name:<18}: {stats_df.shape}")
        print("\n--- episode summary ---")
        print(ep_df.head())
    except Exception as exc:
        print(f"(pandas not available or logs empty: {exc})")
