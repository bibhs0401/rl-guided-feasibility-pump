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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# Continuation bins scale the stall threshold for the next decision window.
CONTINUATION_MULTIPLIERS = (0.33, 0.67, 1.0, 1.5, 2.5)
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

    # -------------------------------------------------------------------------
    # Reset helpers
    # -------------------------------------------------------------------------
    def _sample_instance_path(self) -> str:
        """Randomly sample one instance path from the configured pool."""
        idx = int(self.rng.integers(0, len(self.config.instance_paths)))
        return self.config.instance_paths[idx]

    def _load_nontrivial_instance(self, requested_path: Optional[str] = None) -> Tuple[ProblemInstance, FeasibilityPumpCore]:
        """
        Load an instance and initialize the FP runner.

        We skip instances that are not useful for RL, including:
        1. instances that terminate in the initial LP relaxation
        2. instances that get solved before the first decision point

        A valid training episode should leave the runner at a real decision point
        where the agent can take an action.
        """
        last_problem = None
        last_runner = None

        for attempt in range(self.config.max_reset_resamples):
            instance_path = requested_path if (attempt == 0 and requested_path is not None) else self._sample_instance_path()

            problem = load_npz_instance(instance_path)
            runner = FeasibilityPumpCore(problem, self.config.fp_config)
            print("[reset] building runner and solving initial LP...", flush=True)
            runner.reset()
            print(
                f"[reset] after runner.reset(): "
                f"terminated_in_initial_relaxation={runner.terminated_in_initial_relaxation}, "
                f"done={runner.done}",
                flush=True,
            )

            last_problem = problem
            last_runner = runner

            # Skip if solved immediately in the initial LP relaxation
            if runner.terminated_in_initial_relaxation:
                print("[reset] skipping instance: solved in initial relaxation", flush=True)
                continue

            # Advance naturally until first stall or done
            if not runner.done:
                print("[reset] advancing until first stall or done...", flush=True)
                runner.advance_until_stall_or_done()
                print(
                    f"[reset] after natural advance: done={runner.done}, stalled={runner.is_stalled()}, "
                    f"iterations={runner.iteration}, current_distance={runner.current_distance()}",
                    flush=True,
                )

            # Accept only if the runner is now at a real decision point
            # (not done, and stalled)
            if (not runner.done) and runner.is_stalled():
                print("[reset] accepted instance: reached real decision point", flush=True)
                return problem, runner
            print("[reset] rejected instance: no usable decision point reached", flush=True)
        # Fallback: return the last sampled runner even if it was not ideal.
        # This avoids infinite reset loops when the pool is mostly easy.
        print("[reset] fallback: returning last sampled runner", flush=True)
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

        stall_ratio = min(
            1.0,
            self.runner.consecutive_no_change / max(1, self.runner.config.stall_threshold),
        )
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
        current_cycle_flag = 1.0 if self.runner.consecutive_no_change >= 2 * max(1, self.runner.config.stall_threshold) else 0.0
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
                current_distance_norm,     # 0
                best_distance_norm,        # 1
                dist_improve,              # 2
                stall_ratio,               # 3
                iter_ratio,                # 4
                elapsed_ratio,             # 5
                remaining_ratio,           # 6
                last_flip_ratio,           # 7
                current_cycle_flag,        # 8
                feasible_found_flag,       # 9
                recent_delta,              # 10
                objective_degradation_from_lp,  # 11
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
        r_time = -self.config.reward_time_penalty * step_runtime
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
        print(f"[reset] sampling instance: {self.problem.instance_path}", flush=True)
        info = {
            "episode_id": self.episode_id,
            "instance_path": self.problem.instance_path,
            "instance_name": Path(self.problem.instance_path).name,
            "m": self.problem.m,
            "n": self.problem.n,
            "p": self.problem.p,
            "terminated_in_initial_relaxation": self.runner.terminated_in_initial_relaxation,
            "initial_solution_was_integer": self.runner.initial_solution_was_integer,
            "initial_distance": self.runner.initial_distance,
            "current_distance": self.runner.current_distance(),
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

        continuation_mult = CONTINUATION_MULTIPLIERS[cont_bin]
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
        # 3. Temporarily modify patience for the next natural FP window
        # ---------------------------------------------------------------------
        original_threshold = self.runner.config.stall_threshold
        local_threshold = max(1, int(round(original_threshold * continuation_mult)))
        self.runner.config.stall_threshold = local_threshold

        try:
            if not self.runner.done:
                self.runner.advance_until_stall_or_done()
        finally:
            # Always restore the original threshold
            self.runner.config.stall_threshold = original_threshold

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
            "continuation_mult": continuation_mult,
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

        return observation, float(reward), terminated, truncated, info