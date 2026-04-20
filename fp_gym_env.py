from __future__ import annotations

# -----------------------------------------------------------------------------
# fp_gym_env.py
#
# Step 2:
# Real Gymnasium environment wrapping the real FP backend from mmp_fp_core.py
#
# Design:
# - one episode = one instance
# - one RL step = one FP decision window
# - action = (flip_level, continuation_level)
# - Dict observation for SB3 MultiInputPolicy
# -----------------------------------------------------------------------------

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mmp_fp_core import (
    FPRunConfig,
    FeasibilityPumpCore,
    ProblemInstance,
    load_npz_instance,
    fp_distance,
)


# -----------------------------------------------------------------------------
# Action menus
# -----------------------------------------------------------------------------
# We keep the action space exactly as planned:
#
# MultiDiscrete([6, 5])
#
# axis 0 -> flip level
# axis 1 -> continuation / patience level
#
# Flip levels are relative to the number of integer variables.
# Continuation levels scale the stall threshold temporarily for the next
# decision window.
# -----------------------------------------------------------------------------

def flip_bin_to_count(bin_index: int, n_integer: int) -> int:
    """
    Map discrete flip bin to an actual number of variables to flip.

    Bins:
        0 -> flip 1 variable
        1 -> flip max(2, ceil(0.01 * n_integer))
        2 -> flip ceil(0.02 * n_integer)
        3 -> flip ceil(0.05 * n_integer)
        4 -> flip ceil(0.10 * n_integer)
        5 -> flip ceil(0.20 * n_integer)
    """
    if bin_index == 0:
        return 1
    if bin_index == 1:
        return max(2, math.ceil(0.01 * n_integer))
    if bin_index == 2:
        return max(1, math.ceil(0.02 * n_integer))
    if bin_index == 3:
        return max(1, math.ceil(0.05 * n_integer))
    if bin_index == 4:
        return max(1, math.ceil(0.10 * n_integer))
    if bin_index == 5:
        return max(1, math.ceil(0.20 * n_integer))
    raise ValueError(f"Invalid flip bin: {bin_index}")


# Continuation bins map to a maximum number of natural FP iterations that
# may run before control returns to the agent.  A larger value lets FP run
# longer between interventions; a smaller value returns control sooner.
# These are genuine action semantics: advance_until_stall_or_done(max_steps)
# now respects this limit, so the action dimension actually affects behaviour.
CONTINUATION_MAX_STEPS = (1, 3, 5, 10, 20)
CONTINUATION_LABELS = ("very_short", "short", "medium", "long", "very_long")


# -----------------------------------------------------------------------------
# Environment config
# -----------------------------------------------------------------------------
@dataclass
class FPGymConfig:
    """
    Environment-level configuration.

    This wraps the solver-level config and also adds environment choices such as:
    - instance paths
    - how many trivial instances to skip before giving up
    - reward coefficients
    """
    instance_paths: List[str]

    # Real FP backend config
    fp_config: FPRunConfig = field(default_factory=FPRunConfig)

    # If reset() samples an instance that terminates in the initial LP relaxation,
    # skip it and sample another one. This helps avoid trivial episodes.
    max_reset_resamples: int = 20

    # Reward coefficients
    reward_frac_improve: float = 5.0
    reward_best_improve: float = 8.0
    reward_feasible_bonus: float = 50.0
    reward_time_penalty: float = 0.05
    reward_flip_penalty: float = 0.5
    reward_stall_penalty: float = 2.0

    # Random seed
    seed: Optional[int] = 0


# -----------------------------------------------------------------------------
# Main Gymnasium environment
# -----------------------------------------------------------------------------
class FeasibilityPumpRLEnv(gym.Env):
    """
    Real Gymnasium environment for RL-guided Feasibility Pump.

    Episode:
        One episode = one MMP instance

    Step:
        One step = one FP decision window
        Agent chooses:
            - how many variables to flip
            - how patient FP should be before handing control back again

    Observation space:
        Dict with three components:
            progress : current FP metrics
            history  : short memory of recent behavior
            instance : static instance descriptors

    Action space:
        MultiDiscrete([6, 5])
    """

    metadata = {"render_modes": []}

    def __init__(self, config: FPGymConfig):
        super().__init__()

        if not config.instance_paths:
            raise ValueError("config.instance_paths must contain at least one .npz file")

        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # ---------------------------------------------------------------------
        # Action space
        # ---------------------------------------------------------------------
        # [flip_bin, continuation_bin]
        # flip_bin: 0..5
        # continuation_bin: 0..4
        # ---------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete([6, 5])

        # ---------------------------------------------------------------------
        # Observation space
        # ---------------------------------------------------------------------
        # Dict observation, consistent with the design we discussed:
        #
        # progress: current FP status
        # history : recent step history
        # instance: static instance descriptors
        #
        # All are numeric vectors, float32, scaled roughly to [-1, 1] or [0, 1].
        # ---------------------------------------------------------------------
        self.observation_space = spaces.Dict(
            {
                "progress": spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32),
                "history": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                "instance": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            }
        )

        # Current episode objects
        self.problem: Optional[ProblemInstance] = None
        self.runner: Optional[FeasibilityPumpCore] = None

        # Episode bookkeeping
        self.episode_id = -1
        self.step_id = 0
        self.total_reward = 0.0
        self.termination_reason = ""

        # Best-so-far distance
        self.best_distance = 0.0

        # History features
        self.prev_reward = 0.0
        self.prev_prev_reward = 0.0
        self.prev_dist_delta = 0.0
        self.prev_prev_dist_delta = 0.0
        self.last_action_caused_stall = False
        self.last_action_broke_stall = False
        self.cumulative_flips = 0
        self.consecutive_non_improving = 0

        # ---------------------------------------------------------------------
        # Pre-build CPLEX models for every instance in the pool (Fix 3).
        # build_models() is slow but only runs once per instance at startup.
        # Each episode calls runner.reset_state() which reuses these models,
        # eliminating the per-episode rebuild cost.
        # ---------------------------------------------------------------------
        logger.info(
            "[init] pre-building CPLEX models for %d instances …",
            len(config.instance_paths),
        )
        self._instance_cache: List[tuple] = []
        for i, path in enumerate(config.instance_paths):
            problem = load_npz_instance(path)
            runner = FeasibilityPumpCore(problem, self.config.fp_config)
            runner.build_models()
            self._instance_cache.append((problem, runner))
            logger.info(
                "[init] built %d/%d  %s  (relax=%.1fs  dist=%.1fs)",
                i + 1, len(config.instance_paths), path,
                runner.relaxation_build_seconds, runner.distance_build_seconds,
            )
        logger.info("[init] all models built — env ready")

    # -------------------------------------------------------------------------
    # Reset helpers
    # -------------------------------------------------------------------------
    def _sample_cache_index(self) -> int:
        """Randomly sample one index from the pre-built instance cache."""
        return int(self.rng.integers(0, len(self._instance_cache)))

    def _load_nontrivial_instance(
        self,
        requested_path: Optional[str] = None,
    ) -> Tuple[ProblemInstance, FeasibilityPumpCore]:
        """
        Pick a cached (problem, runner) pair, reset its FP state, and advance
        to the first usable decision point.

        Uses pre-built CPLEX models (Fix 3): runner.reset_state() re-solves the
        initial LP without rebuilding the model, so this is fast.

        Fallback behaviour
        ------------------
        If no instance reaches a usable decision point after max_reset_resamples
        attempts, a ValueError is raised rather than silently returning a runner
        in a done/failed state (which would give the agent a zero-length episode
        and waste training timesteps).
        """
        last_problem: Optional[ProblemInstance] = None
        last_runner: Optional[FeasibilityPumpCore] = None

        for attempt in range(self.config.max_reset_resamples):
            # Pick from cache
            if attempt == 0 and requested_path is not None:
                matches = [
                    i for i, (p, _) in enumerate(self._instance_cache)
                    if p.instance_path == requested_path
                ]
                idx = matches[0] if matches else self._sample_cache_index()
            else:
                idx = self._sample_cache_index()

            problem, runner = self._instance_cache[idx]

            logger.debug(
                "[reset] attempt=%d/%d  instance=%s",
                attempt + 1, self.config.max_reset_resamples, problem.instance_path,
            )

            # Fast state reset — reuses pre-built CPLEX models
            runner.reset_state()

            logger.debug(
                "[reset] after reset_state: failed=%s done=%s "
                "trivial=%s lp_solve=%.2fs",
                runner.failed, runner.done,
                runner.terminated_in_initial_relaxation,
                runner.initial_lp_solve_seconds,
            )

            last_problem = problem
            last_runner = runner

            if runner.failed:
                logger.debug("[reset] skipping: reset_state failed")
                continue

            if runner.terminated_in_initial_relaxation:
                logger.debug("[reset] skipping: solved in initial relaxation")
                continue

            # Advance naturally to the first decision point
            if not runner.done:
                runner.advance_until_stall_or_done()
                logger.debug(
                    "[reset] after natural advance: done=%s stalled=%s "
                    "iters=%d dist=%.4f",
                    runner.done, runner.is_stalled(),
                    runner.iteration, runner.current_distance(),
                )

            if (not runner.done) and (not runner.failed) and runner.is_stalled():
                logger.debug("[reset] accepted: decision point reached")
                return problem, runner

            logger.debug("[reset] rejected: no usable RL decision point")

        # Fallback: use the last sampled runner rather than looping forever.
        # Log a warning so the user knows the pool quality is poor.
        logger.warning(
            "[reset] all %d resamples exhausted without finding a usable "
            "decision point.  Returning last runner (may be in done/failed "
            "state).  Consider screening the instance pool.",
            self.config.max_reset_resamples,
        )
        return last_problem, last_runner

    # -------------------------------------------------------------------------
    # Feature builders
    # -------------------------------------------------------------------------
    def _build_progress_features(self) -> np.ndarray:
        """
        Build the 12-dim 'progress' feature vector.

        Chosen to match the experiment design:
        - current fractionality / distance status
        - stall status
        - iteration progress
        - time usage
        - last perturbation info
        """
        assert self.runner is not None
        assert self.problem is not None

        n_integer = max(1, len(self.problem.integer_indices))

        current_distance = self.runner.current_distance()
        current_distance_norm = min(1.0, current_distance / n_integer)
        best_distance_norm = min(1.0, self.best_distance / n_integer)

        # Improvement from previous step
        dist_improve = self.prev_dist_delta / n_integer
        dist_improve = float(np.clip(dist_improve, -1.0, 1.0))

        # Binary flag: is FP currently at a decision point (stalled)?
        # With the 1-event stall semantics, consecutive_no_change is 0 or 1,
        # so this is more informative than dividing by stall_threshold.
        at_decision_point = 1.0 if self.runner.is_stalled() else 0.0

        iter_ratio = min(
            1.0,
            self.runner.iteration / max(1, self.runner.config.max_iterations),
        )

        elapsed = 0.0 if self.runner.start_time is None else (self.runner.config.time_limit - max(0.0, self.runner.remaining_time() or 0.0))
        elapsed_ratio = min(1.0, elapsed / max(1e-8, self.runner.config.time_limit))
        remaining_ratio = min(
            1.0,
            max(0.0, self.runner.remaining_time() or 0.0) / max(1e-8, self.runner.config.time_limit),
        )

        last_flip_ratio = min(1.0, self.runner.last_k / n_integer)
        # How many stall events have accumulated relative to the budget?
        # Replaces current_cycle_flag which could never be True with 1-event stalls.
        stall_event_ratio = min(1.0, self.runner.stall_events / max(1, self.runner.config.max_stalls))
        feasible_found_flag = 1.0 if self.runner.integer_found else 0.0

        # Use the runner's recent distance deltas as a compact progress signal
        recent_delta = 0.0
        if self.runner.recent_distance_deltas:
            recent_delta = float(np.mean(self.runner.recent_distance_deltas)) / n_integer
            recent_delta = float(np.clip(recent_delta, -1.0, 1.0))

        objective_degradation_from_lp = 0.0
        if self.runner.initial_lp_objective != 0.0 and self.runner.y_values is not None:
            current_sum_y = float(np.sum(self.runner.y_values))
            objective_degradation_from_lp = (self.runner.initial_lp_objective - current_sum_y) / (abs(self.runner.initial_lp_objective) + 1.0)
            objective_degradation_from_lp = float(np.clip(objective_degradation_from_lp, -1.0, 1.0))

        progress = np.array(
            [
                current_distance_norm,          # 0  current dist / n_int
                best_distance_norm,             # 1  best dist / n_int
                dist_improve,                   # 2  improvement this step
                at_decision_point,              # 3  1 = stalled, 0 = running  (Fix 2)
                iter_ratio,                     # 4  iterations used
                elapsed_ratio,                  # 5  time used
                remaining_ratio,                # 6  time remaining
                last_flip_ratio,                # 7  flip size / n_int
                stall_event_ratio,              # 8  stall count / max_stalls  (Fix 2)
                feasible_found_flag,            # 9  1 = integer solution found
                recent_delta,                   # 10 mean recent dist delta
                objective_degradation_from_lp,  # 11 LP obj degradation
            ],
            dtype=np.float32,
        )

        return np.clip(progress, -1.0, 1.0)

    def _build_history_features(self) -> np.ndarray:
        """
        Build the 8-dim 'history' feature vector.

        This gives the agent short-term memory without requiring an RNN.
        """
        assert self.problem is not None
        n_integer = max(1, len(self.problem.integer_indices))

        history = np.array(
            [
                float(np.clip(self.prev_dist_delta / n_integer, -1.0, 1.0)),
                float(np.clip(self.prev_prev_dist_delta / n_integer, -1.0, 1.0)),
                float(np.clip(self.prev_reward / 10.0, -1.0, 1.0)),
                float(np.clip(self.prev_prev_reward / 10.0, -1.0, 1.0)),
                1.0 if self.last_action_caused_stall else 0.0,
                1.0 if self.last_action_broke_stall else 0.0,
                float(np.clip(self.cumulative_flips / max(1, 10 * n_integer), 0.0, 1.0)),
                float(np.clip(self.consecutive_non_improving / max(1, self.config.fp_config.max_stalls), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.clip(history, -1.0, 1.0)

    def _build_instance_features(self) -> np.ndarray:
        """
        Build the 7-dim 'instance' feature vector.

        These are static descriptors so the policy can generalize across
        multiple instances.
        """
        assert self.problem is not None

        n = max(1, self.problem.n)
        m = max(1, self.problem.m)
        n_integer = max(1, len(self.problem.integer_indices))

        binary_fraction = len(self.problem.integer_indices) / n
        constraints_per_var = m / n

        # LP relaxation objective normalized
        lp_obj_norm = self.runner.initial_lp_objective / (abs(self.runner.initial_lp_objective) + 1.0) if self.runner is not None else 0.0

        # Initial average fractionality
        initial_frac = 0.0
        if self.runner is not None and self.runner.x_relaxed is not None:
            fracs = [abs(self.runner.x_relaxed[idx] - round(self.runner.x_relaxed[idx])) for idx in self.problem.integer_indices]
            if fracs:
                initial_frac = float(np.mean(fracs)) * 2.0
                initial_frac = min(1.0, initial_frac)

        # Matrix sparsity
        density = self.problem.A.nnz / max(1, self.problem.m * self.problem.n)
        sparsity = 1.0 - density
        sparsity = float(np.clip(sparsity, 0.0, 1.0))

        # Fraction of binaries currently unfixed
        unfixed_fraction = 1.0
        if self.runner is not None and self.runner.x_relaxed is not None:
            non_integral = sum(
                abs(self.runner.x_relaxed[idx] - round(self.runner.x_relaxed[idx])) > 1e-6
                for idx in self.problem.integer_indices
            )
            unfixed_fraction = non_integral / n_integer

        # Normalized number of objectives (we mainly care about p=2 or p=3 now)
        p_norm = self.problem.p / 5.0

        instance = np.array(
            [
                float(np.clip(binary_fraction, 0.0, 1.0)),
                float(np.clip(constraints_per_var / 10.0, 0.0, 1.0)),
                float(np.clip(lp_obj_norm, -1.0, 1.0)),
                float(np.clip(initial_frac, 0.0, 1.0)),
                sparsity,
                float(np.clip(unfixed_fraction, 0.0, 1.0)),
                float(np.clip(p_norm, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

        return np.clip(instance, -1.0, 1.0)

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Assemble the full Dict observation."""
        return {
            "progress": self._build_progress_features(),
            "history": self._build_history_features(),
            "instance": self._build_instance_features(),
        }

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------
    def _compute_reward(
        self,
        prev_distance: float,
        prev_best_distance: float,
        new_distance: float,
        new_best_distance: float,
        feasible_found: bool,
        step_runtime: float,
        flip_count: int,
        still_stalled: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Reward design from our experiment plan:

        positive:
        - reduce distance / fractionality
        - improve best-so-far distance
        - find first feasible solution

        negative:
        - spend time
        - flip too aggressively
        - remain stalled
        """
        assert self.problem is not None

        n_integer = max(1, len(self.problem.integer_indices))

        delta = (prev_distance - new_distance) / n_integer
        best_gain = max(0.0, prev_best_distance - new_best_distance) / n_integer

        r_frac = self.config.reward_frac_improve * float(delta)
        r_best = self.config.reward_best_improve * float(best_gain)
        r_feas = self.config.reward_feasible_bonus if feasible_found else 0.0
        # Normalize by time_limit so the penalty scale is instance-size-independent (Fix 5)
        r_time = -self.config.reward_time_penalty * (
            step_runtime / max(1e-8, self.runner.config.time_limit)
        )
        r_flip = -self.config.reward_flip_penalty * (flip_count / n_integer)
        r_stall = -self.config.reward_stall_penalty if still_stalled else 0.0

        total = r_frac + r_best + r_feas + r_time + r_flip + r_stall

        components = {
            "r_frac": r_frac,
            "r_best": r_best,
            "r_feas": r_feas,
            "r_time": r_time,
            "r_flip": r_flip,
            "r_stall": r_stall,
        }
        return float(total), components

    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Start a new episode.

        We load one nontrivial instance, initialize the real FP backend,
        and advance until the first decision point.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_id += 1
        self.step_id = 0
        self.total_reward = 0.0
        self.termination_reason = ""

        self.prev_reward = 0.0
        self.prev_prev_reward = 0.0
        self.prev_dist_delta = 0.0
        self.prev_prev_dist_delta = 0.0
        self.last_action_caused_stall = False
        self.last_action_broke_stall = False
        self.cumulative_flips = 0
        self.consecutive_non_improving = 0

        requested_path = None
        if options is not None:
            requested_path = options.get("instance_path")

        self.problem, self.runner = self._load_nontrivial_instance(requested_path=requested_path)

        # _load_nontrivial_instance() already advances to the first usable
        # decision point when possible.
        self.best_distance = self.runner.current_distance()

        observation = self._build_observation()
        logger.info(
            "[episode %d] instance=%s  m=%d n=%d p=%d  dist0=%.4f  stalls_so_far=%d",
            self.episode_id,
            Path(self.problem.instance_path).name,
            self.problem.m, self.problem.n, self.problem.p,
            self.runner.current_distance(),
            self.runner.stall_events,
        )
        info = {
            "episode_id": self.episode_id,
            "instance_path": self.problem.instance_path if self.problem is not None else None,
            "instance_name": Path(self.problem.instance_path).name if self.problem is not None else None,
            "m": self.problem.m if self.problem is not None else None,
            "n": self.problem.n if self.problem is not None else None,
            "p": self.problem.p if self.problem is not None else None,

            "failed": self.runner.failed if self.runner is not None else None,
            "integer_found": self.runner.integer_found if self.runner is not None else None,
            "terminated_in_initial_relaxation": self.runner.terminated_in_initial_relaxation if self.runner is not None else None,
            "initial_solution_was_integer": self.runner.initial_solution_was_integer if self.runner is not None else None,

            "initial_distance": self.runner.initial_distance if self.runner is not None else None,
            "current_distance": self.runner.current_distance() if self.runner is not None else None,

            "iterations": self.runner.iteration if self.runner is not None else None,
            "stall_events": self.runner.stall_events if self.runner is not None else None,

            "relaxation_build_seconds": getattr(self.runner, "relaxation_build_seconds", 0.0) if self.runner is not None else None,
            "distance_build_seconds": getattr(self.runner, "distance_build_seconds", 0.0) if self.runner is not None else None,
            "initial_lp_solve_seconds": getattr(self.runner, "initial_lp_solve_seconds", 0.0) if self.runner is not None else None,
            "reset_seconds": getattr(self.runner, "reset_seconds", 0.0) if self.runner is not None else None,
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        One RL step = one FP decision window.

        Flow:
        1. Decode action
        2. Apply chosen flip count
        3. Temporarily adjust patience / stall threshold
        4. Advance FP naturally until next decision point or termination
        5. Compute reward
        6. Return Gymnasium tuple
        """
        assert self.problem is not None
        assert self.runner is not None

        if self.runner.done:
            obs = self._build_observation()
            return obs, 0.0, True, False, {"reason": "runner_already_done"}

        # ---------------------------------------------------------------------
        # 1. Decode action
        # ---------------------------------------------------------------------
        flip_bin = int(np.clip(int(action[0]), 0, 5))
        cont_bin = int(np.clip(int(action[1]), 0, 4))

        n_integer = max(1, len(self.problem.integer_indices))
        flip_count = flip_bin_to_count(flip_bin, n_integer)

        continuation_max_steps = CONTINUATION_MAX_STEPS[cont_bin]
        continuation_label = CONTINUATION_LABELS[cont_bin]

        # Save state before acting
        prev_distance = self.runner.current_distance()
        prev_best_distance = self.best_distance
        prev_stalled = self.runner.is_stalled()

        remaining_before = self.runner.remaining_time()
        elapsed_before = (
            0.0
            if remaining_before is None
            else self.runner.config.time_limit - max(0.0, remaining_before)
        )

        # ---------------------------------------------------------------------
        # 2. Apply the chosen perturbation
        # ---------------------------------------------------------------------
        self.runner.apply_flip_count(flip_count)
        self.cumulative_flips += flip_count

        # ---------------------------------------------------------------------
        # 3. Advance FP naturally for up to continuation_max_steps iterations
        #    before returning control to the agent (Fix 1).
        #    This makes the continuation action genuinely affect behaviour:
        #    the agent decides how long FP runs between interventions.
        # ---------------------------------------------------------------------
        if not self.runner.done:
            self.runner.advance_until_stall_or_done(max_steps=continuation_max_steps)

        # ---------------------------------------------------------------------
        # 4. Collect new state
        # ---------------------------------------------------------------------
        new_distance = self.runner.current_distance()
        self.best_distance = min(self.best_distance, new_distance)

        remaining_after = self.runner.remaining_time()
        elapsed_after = (
            0.0
            if remaining_after is None
            else self.runner.config.time_limit - max(0.0, remaining_after)
        )
        step_runtime = max(0.0, elapsed_after - elapsed_before)

        feasible_found = self.runner.integer_found
        failed = self.runner.failed
        still_stalled = self.runner.is_stalled()
        dist_delta = prev_distance - new_distance

        # ---------------------------------------------------------------------
        # 5. Reward
        # ---------------------------------------------------------------------
        reward, reward_components = self._compute_reward(
            prev_distance=prev_distance,
            prev_best_distance=prev_best_distance,
            new_distance=new_distance,
            new_best_distance=self.best_distance,
            feasible_found=feasible_found,
            step_runtime=step_runtime,
            flip_count=flip_count,
            still_stalled=still_stalled,
        )
        self.total_reward += reward

        # Update short-term history
        self.prev_prev_dist_delta = self.prev_dist_delta
        self.prev_dist_delta = dist_delta
        self.prev_prev_reward = self.prev_reward
        self.prev_reward = reward

        improved = new_distance < prev_distance - 1e-9
        self.consecutive_non_improving = 0 if improved else self.consecutive_non_improving + 1

        self.last_action_caused_stall = still_stalled
        self.last_action_broke_stall = prev_stalled and (not still_stalled)

        self.step_id += 1

        # ---------------------------------------------------------------------
        # 6. Termination / truncation
        # ---------------------------------------------------------------------
        terminated = False
        truncated = False

        if feasible_found:
            terminated = True
            self.termination_reason = "feasible_found"
        elif failed:
            terminated = True
            self.termination_reason = "solver_failed"
        elif self.runner.done:
            truncated = True
            self.termination_reason = "time_or_iteration_or_stall_budget"

        # ---------------------------------------------------------------------
        # 7. Build outputs
        # ---------------------------------------------------------------------
        observation = self._build_observation()

        info = {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "instance_path": self.problem.instance_path,
            "instance_name": Path(self.problem.instance_path).name,
            "m": self.problem.m,
            "n": self.problem.n,
            "p": self.problem.p,

            "flip_bin": flip_bin,
            "flip_count": flip_count,
            "continuation_bin": cont_bin,
            "continuation_max_steps": continuation_max_steps,
            "continuation_label": continuation_label,

            "iterations": self.runner.iteration,
            "stall_events": self.runner.stall_events,
            "total_flips": self.runner.total_flips,
            "current_distance": new_distance,
            "best_distance": self.best_distance,

            "feasible_found": feasible_found,
            "failed": failed,
            "still_stalled": still_stalled,

            "reward": reward,
            **reward_components,

            "step_runtime": step_runtime,
            "elapsed_seconds": elapsed_after,
            "remaining_seconds": max(0.0, self.runner.remaining_time() or 0.0),

            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": self.termination_reason,
        }

        return observation, float(reward), terminated, truncated, infoaining_time() or 0.0),

            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": self.termination_reason,
        }

        return observation, float(reward), terminated, truncated, info
