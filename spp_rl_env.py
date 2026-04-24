from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fp_baseline_spp import (
    ACTION_PROPORTIONS,
    CONTINUATION_STEPS,
    FPConfig,
    SPPFeasibilityPump,
)
from spp_model import (
    SPPInstance,
    feasibility_metrics,
    load_spp_instance,
    objective_value,
)

# ---------------------------------------------------------------------------
# Reward weights (all tunable)
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights for each reward component (see slide: What the Agent Decides)."""
    frac_improve: float = 5.0      # +r_frac  weight
    best_improve: float = 10.0     # +r_best  weight
    feasible_bonus: float = 50.0   # fixed bonus for finding a feasible solution
    time_penalty: float = 0.5      # -r_time  weight
    flip_penalty: float = 0.1      # -r_flip  weight
    stall_penalty: float = 1.0     # -r_stall penalty (still stalled after action)


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@dataclass
class SPPRLEnvConfig:
    instance_paths: list[str]
    fp_config: FPConfig = field(default_factory=FPConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    seed: Optional[int] = 0
    max_reset_attempts: int = 10


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SPPFeasibilityPumpEnv(gym.Env):
    """Stall-point RL environment for SPP feasibility pump perturbations.

    Observation space (Dict — requires MultiInputPolicy in SB3):
        "progress"  : Box(-1, 1, shape=(7,))  — current FP progress features
        "history"   : Box(-1, 1, shape=(8,))  — recent step history
        "instance"  : Box( 0, 1, shape=(7,))  — static per-instance features

    Action space: MultiDiscrete([6, 5])
        dim 0 — flip bin   (6 choices): matches ACTION_PROPORTIONS
        dim 1 — cont bin   (5 choices): matches CONTINUATION_STEPS

    Reward (per slide):
        +r_frac   : frac_improve   * (prev_dist - new_dist)  / n
        +r_best   : best_improve   * max(0, prev_best - new_best) / n
        +50       : feasible_bonus if feasible solution found
        -r_time   : time_penalty   * step_runtime / time_limit
        -r_flip   : flip_penalty   * flip_count   / n
        -r_stall  : stall_penalty  if FP is still stalled after the action
    """

    metadata = {"render_modes": []}

    N_FLIP_BINS: int = len(ACTION_PROPORTIONS)   # 6
    N_CONT_BINS: int = len(CONTINUATION_STEPS)   # 5

    def __init__(self, config: SPPRLEnvConfig):
        super().__init__()
        if not config.instance_paths:
            raise ValueError("SPPRLEnvConfig.instance_paths must not be empty")
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # --- Action & observation spaces ---------------------------------
        self.action_space = spaces.MultiDiscrete([self.N_FLIP_BINS, self.N_CONT_BINS])
        self.observation_space = spaces.Dict({
            # Progress features can include signed deltas → [-1, 1]
            "progress": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            # History features include signed reward history → [-1, 1]
            "history":  spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
            # Instance features are ratios/fractions → [0, 1]
            "instance": spaces.Box(low=0.0,  high=1.0, shape=(7,), dtype=np.float32),
        })

        # --- Runtime state -----------------------------------------------
        self.problem: Optional[SPPInstance] = None
        self.runner:  Optional[SPPFeasibilityPump] = None
        self.episode_id = -1

        # Per-episode trackers (reset in _reset_episode_state)
        self.episode_return: float = 0.0
        self.episode_steps:  int   = 0

        # History buffers (3-step distance window for delta features)
        self._dist_history: deque[float] = deque([math.inf, math.inf, math.inf], maxlen=3)
        self._prev_reward:       float = 0.0
        self._prev_prev_reward:  float = 0.0
        self._prev_nfracs:       float = 1.0
        self._best_nfracs:       float = 1.0
        self._last_flip_count:   int   = 0
        self._cumulative_flips:  int   = 0
        self._caused_stall:      bool  = False
        self._broke_stall:       bool  = False

        # Static per-instance features (computed once per reset)
        self._inst: np.ndarray = np.zeros(7, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_path(self, requested: Optional[str] = None) -> str:
        if requested:
            return requested
        return self.config.instance_paths[
            int(self.rng.integers(0, len(self.config.instance_paths)))
        ]

    def _build_runner(self, path: str) -> SPPFeasibilityPump:
        problem = load_spp_instance(path)
        self.problem = problem
        cfg = FPConfig(
            max_iterations=self.config.fp_config.max_iterations,
            time_limit=self.config.fp_config.time_limit,
            stall_length=self.config.fp_config.stall_length,
            tolerance=self.config.fp_config.tolerance,
            random_seed=int(self.rng.integers(0, 2**31 - 1)),
            baseline_action=self.config.fp_config.baseline_action,
            stop_on_repaired_incumbent=False,
            cplex_threads=self.config.fp_config.cplex_threads,
            verbose=self.config.fp_config.verbose,
        )
        runner = SPPFeasibilityPump(problem, cfg)
        runner.reset()
        return runner

    def _reset_episode_state(self) -> None:
        """Zero out all per-episode accumulators."""
        self.episode_return = 0.0
        self.episode_steps  = 0
        self._dist_history  = deque([math.inf, math.inf, math.inf], maxlen=3)
        self._prev_reward       = 0.0
        self._prev_prev_reward  = 0.0
        self._prev_nfracs       = 1.0
        self._best_nfracs       = 1.0
        self._last_flip_count   = 0
        self._cumulative_flips  = 0
        self._caused_stall      = False
        self._broke_stall       = False

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _nfracs(self) -> float:
        """Fraction of LP variables that are strictly fractional."""
        if self.runner is None or self.runner.x_lp is None:
            return 0.0
        tol = self.runner.config.tolerance
        return float(np.mean(np.abs(self.runner.x_lp - np.round(self.runner.x_lp)) > tol))

    def _compute_instance_features(self) -> None:
        """Compute static per-instance features from the initial LP solution.

        Features (7-dim, all in [0, 1]):
            0  binary_fraction      — always 1.0 for pure SPP
            1  constraints_per_var  — m / n (clipped to 1)
            2  lp_obj_norm          — LP objective / (n * max_profit)
            3  initial_frac         — mean |x_lp - round(x_lp)| at initial LP
            4  sparsity             — 1 - nnz/(m*n)  (higher = sparser)
            5  unfixed_fraction     — fraction of LP vars strictly in (tol, 1-tol)
            6  p_norm               — 1/3 for SPP (single objective; 2/3 for bi-, 1.0 for tri-)
        """
        assert self.runner is not None and self.problem is not None
        p   = self.problem
        tol = self.runner.config.tolerance

        binary_fraction     = 1.0                                           # SPP: all binary
        constraints_per_var = min(1.0, p.m / max(1, p.n))

        if self.runner.x_lp is not None:
            max_profit  = float(np.max(np.abs(p.profits))) if p.n > 0 else 1.0
            lp_obj      = float(np.dot(p.profits, self.runner.x_lp))
            lp_obj_norm = min(1.0, max(0.0, lp_obj / max(1e-9, p.n * max_profit)))

            frac             = np.abs(self.runner.x_lp - np.round(self.runner.x_lp))
            initial_frac     = float(np.mean(frac))
            unfixed_fraction = float(np.mean(
                (self.runner.x_lp > tol) & (self.runner.x_lp < 1.0 - tol)
            ))
        else:
            lp_obj_norm = unfixed_fraction = initial_frac = 0.0

        nnz       = p.A.nnz
        sparsity  = 1.0 - min(1.0, nnz / max(1, p.m * p.n))   # 1 = fully sparse
        p_norm    = 1.0 / 3.0                                   # p=1 for SPP

        self._inst = np.asarray([
            binary_fraction,
            constraints_per_var,
            lp_obj_norm,
            initial_frac,
            sparsity,
            unfixed_fraction,
            p_norm,
        ], dtype=np.float32)

    def _progress_obs(self) -> np.ndarray:
        """7-dim current-progress features.

        Fields (slide order):
            0  nfracs_ratio          — fraction of LP vars that are fractional
            1  best_nfracs_ratio     — lowest nfracs seen this episode
            2  nfracs_improve        — 1 if nfracs improved since last step
            3  current_distance_norm — FP distance / n
            4  dist_improve          — (prev_dist - cur_dist) / n  [signed]
            5  elapsed_ratio         — elapsed time / time_limit
            6  last_flip_ratio       — last flip count / n
        """
        if self.runner is None or self.problem is None:
            return np.zeros(7, dtype=np.float32)

        n        = max(1, self.problem.n)
        nfracs   = self._nfracs()
        cur_dist = self.runner.current_distance()

        dist_norm    = min(1.0, cur_dist / n) if math.isfinite(cur_dist) else 1.0
        prev_dist    = self._dist_history[-1]
        dist_improve = float(np.clip((prev_dist - cur_dist) / n, -1.0, 1.0)) \
                       if (math.isfinite(prev_dist) and math.isfinite(cur_dist)) else 0.0

        nfracs_improve = 1.0 if nfracs < self._prev_nfracs - 1e-6 else 0.0

        elapsed       = time.time() - self.runner.start_time if self.runner.start_time else 0.0
        elapsed_ratio = min(1.0, elapsed / max(1e-9, self.runner.config.time_limit))
        flip_ratio    = min(1.0, self._last_flip_count / n)

        return np.asarray([
            nfracs,
            self._best_nfracs,
            nfracs_improve,
            dist_norm,
            dist_improve,
            elapsed_ratio,
            flip_ratio,
        ], dtype=np.float32)

    def _history_obs(self) -> np.ndarray:
        """8-dim recent-history features.

        Fields (slide order):
            0  prev_dist_delta         — improvement in last step  / n  [signed]
            1  prev_prev_dist_delta    — improvement two steps ago  / n  [signed]
            2  prev_reward             — reward at last step (normalised)
            3  prev_prev_reward        — reward two steps ago (normalised)
            4  last_action_caused_stall — 1 if FP re-stalled after last action
            5  last_action_broke_stall  — 1 if last action broke the stall
            6  cumulative_flips        — total flips this episode / n
            7  consecutive_non_improving — runner's no_improvement_count / max_iters
        """
        if self.runner is None or self.problem is None:
            return np.zeros(8, dtype=np.float32)

        n         = max(1, self.problem.n)
        d0, d1, d2 = list(self._dist_history)   # oldest … newest

        # Signed per-step distance improvement (positive = distance decreased = good)
        prev_dd      = float(np.clip((d1 - d2) / n, -1.0, 1.0)) \
                       if (math.isfinite(d1) and math.isfinite(d2)) else 0.0
        prev_prev_dd = float(np.clip((d0 - d1) / n, -1.0, 1.0)) \
                       if (math.isfinite(d0) and math.isfinite(d1)) else 0.0

        bonus = max(1.0, self.config.reward_config.feasible_bonus)
        r_norm      = float(np.clip(self._prev_reward      / bonus, -1.0, 1.0))
        r_prev_norm = float(np.clip(self._prev_prev_reward / bonus, -1.0, 1.0))

        cum_flips    = min(1.0, self._cumulative_flips / n)
        max_iters    = max(1, self.runner.config.max_iterations)
        consec_noimpr = min(1.0, self.runner.no_improvement_count / max_iters)

        return np.asarray([
            prev_dd,
            prev_prev_dd,
            r_norm,
            r_prev_norm,
            1.0 if self._caused_stall else 0.0,
            1.0 if self._broke_stall  else 0.0,
            cum_flips,
            consec_noimpr,
        ], dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "progress": self._progress_obs(),
            "history":  self._history_obs(),
            "instance": self._inst.copy(),
        }

    def _info(self, reward: float = 0.0, termination_reason: str = "") -> dict[str, Any]:
        assert self.runner is not None and self.problem is not None
        result  = self.runner.result("rl_guided_fp", average_return=self.episode_return)
        metrics = feasibility_metrics(self.problem, result.final_solution)
        elapsed = time.time() - self.runner.start_time if self.runner.start_time else 0.0
        return {
            "instance_name":           self.problem.name,
            "instance_path":           self.problem.path,
            "m":                       self.problem.m,
            "n":                       self.problem.n,
            "success":                 result.success,
            "final_objective":         result.final_objective,
            "final_violation":         metrics.total_violation,
            "num_violated_constraints": metrics.num_violated_constraints,
            "iterations":              self.runner.iterations,
            "runtime_seconds":         elapsed,
            "num_stalls":              self.runner.num_stalls,
            "num_rl_interventions":    self.runner.num_rl_interventions,
            "average_return":          self.episode_return,
            "reward":                  reward,
            "termination_reason":      termination_reason,
            "notes_error_status":      result.notes_error_status,
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_id += 1
        self._reset_episode_state()

        requested_path = options.get("instance_path") if options else None
        last_runner: Optional[SPPFeasibilityPump] = None

        for _ in range(max(1, self.config.max_reset_attempts)):
            path   = self._choose_path(requested_path)
            runner = self._build_runner(path)

            # Skip instances that failed during the LP relaxation
            if runner.done and runner.failed:
                continue

            runner.run_until_stall_or_done()
            last_runner = runner

            # Trivially-solved: LP-rounded solution was already feasible before any stall.
            # These episodes offer no RL learning signal — skip them (unless caller pinned
            # a specific instance via options).
            trivially_solved = runner.done and runner.num_stalls == 0 and not runner.failed
            if not requested_path and trivially_solved:
                continue

            # Accept: either a real stall was detected, or the budget was hit (done by
            # max_iterations / time_limit with no stall) — both give an RL decision point.
            self.runner = runner
            break

        if self.runner is None:
            # Last resort: use whatever runner we managed to build
            self.runner = last_runner
        if self.runner is None:
            raise RuntimeError("Could not initialise a set-packing FP runner")

        # Sync problem reference (may have been re-assigned inside _build_runner)
        self.problem = self.runner.instance

        # Compute static instance features from the initial LP solution
        self._compute_instance_features()

        # Seed the distance history with the current distance
        cur_dist = self.runner.current_distance()
        self._dist_history = deque([math.inf, math.inf, cur_dist], maxlen=3)
        self._prev_nfracs  = self._nfracs()
        self._best_nfracs  = self._prev_nfracs

        return self._observation(), self._info(termination_reason="reset")

    def step(self, action):
        assert self.runner is not None and self.problem is not None

        # If runner already finished (shouldn't normally reach here after the fixed reset)
        if self.runner.done:
            return (
                self._observation(), 0.0, True, False,
                self._info(termination_reason="already_done"),
            )

        # Advance to a genuine stall if we somehow entered step() before one
        if self.runner.num_stalls == 0 or not self.runner.current_metrics().num_violated_constraints:
            self.runner.run_until_stall_or_done()
            if self.runner.done:
                return (
                    self._observation(), 0.0, True, False,
                    self._info(termination_reason="done_before_action"),
                )

        # --- Parse MultiDiscrete action -----------------------------------
        action      = np.asarray(action, dtype=int).reshape(-1)
        flip_action = int(np.clip(action[0], 0, self.N_FLIP_BINS - 1))
        cont_action = int(np.clip(action[1], 0, self.N_CONT_BINS - 1))
        max_cont    = CONTINUATION_STEPS[cont_action]

        # --- Snapshot state before perturbation ---------------------------
        prev_dist  = self.runner.current_distance()
        prev_best  = self.runner.best_distance
        prev_nfracs = self._nfracs()
        step_t0    = time.time()

        # --- Apply perturbation -------------------------------------------
        perturb_info = self.runner.apply_perturbation(flip_action, rl_intervention=True)
        flip_count   = perturb_info["flip_count"]
        self._last_flip_count   = flip_count
        self._cumulative_flips += flip_count

        # --- Run FP until next stall or done (bounded by continuation bin) -
        outcome      = self.runner.run_until_stall_or_done(max_steps=max_cont)
        step_runtime = time.time() - step_t0

        # --- Post-action metrics ------------------------------------------
        new_dist  = self.runner.current_distance()
        new_best  = self.runner.best_distance
        new_nfracs = self._nfracs()

        # Update stall flags for history observation
        self._caused_stall = (not self.runner.done) and outcome.stalled
        self._broke_stall  = not self._caused_stall

        # --- Reward (matching slide exactly) ------------------------------
        n  = max(1, self.problem.n)
        rc = self.config.reward_config

        # +r_frac: distance improved this step
        if math.isfinite(prev_dist) and math.isfinite(new_dist):
            r_frac = rc.frac_improve * (prev_dist - new_dist) / n
        else:
            r_frac = 0.0

        # +r_best: new best distance reached
        if math.isfinite(prev_best) and math.isfinite(new_best):
            r_best = rc.best_improve * max(0.0, prev_best - new_best) / n
        else:
            r_best = 0.0

        # +50: feasible solution found
        feasible = bool(self.runner.result("rl_guided_fp").success)
        r_feasible = rc.feasible_bonus if feasible else 0.0

        # -r_time: time consumed in this window
        r_time = -rc.time_penalty * step_runtime / max(1e-9, self.runner.config.time_limit)

        # -r_flip: aggressiveness of the perturbation
        r_flip = -rc.flip_penalty * flip_count / n

        # -r_stall: still stuck after action
        r_stall = -rc.stall_penalty if self._caused_stall else 0.0

        reward = r_frac + r_best + r_feasible + r_time + r_flip + r_stall

        # --- Update history buffers ---------------------------------------
        self._dist_history.append(new_dist)
        self._prev_prev_reward = self._prev_reward
        self._prev_reward      = reward
        self._prev_nfracs      = new_nfracs
        self._best_nfracs      = min(self._best_nfracs, new_nfracs)

        self.episode_steps  += 1
        self.episode_return += float(reward)

        # --- Termination --------------------------------------------------
        terminated = bool(self.runner.done and feasible)
        truncated  = bool(self.runner.done and not terminated)
        reason = (
            "feasible_success"    if terminated else
            "budget_or_failure"   if truncated  else
            "stall"
        )
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(reward, reason),
        )


# ---------------------------------------------------------------------------
# Heuristic fallback (used when SB3 is unavailable)
# ---------------------------------------------------------------------------

def heuristic_action_from_observation(obs: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """Simple rule-based policy that returns a MultiDiscrete([6,5]) action array."""
    if isinstance(obs, dict):
        progress = np.asarray(obs.get("progress", np.zeros(7)), dtype=float)
    else:
        progress = np.asarray(obs, dtype=float)

    # distance_norm is index 3; elapsed_ratio is index 5
    dist_norm    = float(progress[3]) if len(progress) > 3 else 0.5
    elapsed_ratio = float(progress[5]) if len(progress) > 5 else 0.0

    # Flip action: more aggressive when far from integer
    if dist_norm > 0.5:
        flip_action = 5        # 20%
    elif dist_norm > 0.3:
        flip_action = 4        # 10%
    elif dist_norm > 0.15:
        flip_action = 3        # 5%
    elif dist_norm > 0.05:
        flip_action = 2        # 2%
    else:
        flip_action = 1        # 1%

    # Continuation: shorter as time runs out
    cont_action = max(0, 2 - int(elapsed_ratio * 3))

    return np.array([flip_action, cont_action], dtype=np.int64)
