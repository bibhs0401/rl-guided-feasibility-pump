from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from set_packing.spp_fp_core import (
    SPFPRunConfig,
    SPPProblemInstance,
    SetPackingFPCore,
    load_spp_instance,
)

logger = logging.getLogger(__name__)
_FRAC_EPS = 1e-6


def flip_bin_to_count(bin_index: int, n_integer: int) -> int:
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


CONTINUATION_MAX_STEPS = (1, 3, 5, 10, 20)
CONTINUATION_LABELS = ("very_short", "short", "medium", "long", "very_long")


@dataclass
class SPPGymConfig:
    instance_paths: List[str]
    fp_config: SPFPRunConfig = field(default_factory=SPFPRunConfig)
    max_reset_resamples: int = 20
    reward_frac_improve: float = 5.0
    reward_best_improve: float = 8.0
    reward_feasible_bonus: float = 50.0
    reward_time_penalty: float = 0.05
    reward_flip_penalty: float = 0.5
    reward_stall_penalty: float = 2.0
    seed: Optional[int] = 0


class SetPackingFPRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: SPPGymConfig):
        super().__init__()
        if not config.instance_paths:
            raise ValueError("config.instance_paths must contain at least one .npz or .lp path")
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.action_space = spaces.MultiDiscrete([6, 5])
        self.observation_space = spaces.Dict(
            {
                "progress": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "history": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                "instance": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            }
        )

        self.problem: Optional[SPPProblemInstance] = None
        self.runner: Optional[SetPackingFPCore] = None

        self.episode_id = -1
        self.step_id = 0
        self.total_reward = 0.0
        self.termination_reason = ""
        self.best_distance = 0.0
        self.best_objective = 0.0
        self.best_nfracs = 0
        self.prev_reward = 0.0
        self.prev_prev_reward = 0.0
        self.prev_dist_delta = 0.0
        self.prev_prev_dist_delta = 0.0
        self.prev_nfracs = 0
        self.last_action_caused_stall = False
        self.last_action_broke_stall = False
        self.cumulative_flips = 0
        self.consecutive_non_improving = 0

        self._instance_paths = list(config.instance_paths)
        self._instance_cache: List[Optional[tuple]] = [None] * len(self._instance_paths)

    def _ensure_built(self, idx: int) -> tuple:
        if self._instance_cache[idx] is not None:
            return self._instance_cache[idx]
        path = self._instance_paths[idx]
        problem = load_spp_instance(path)
        runner = SetPackingFPCore(problem, self.config.fp_config)
        runner.build_models()
        self._instance_cache[idx] = (problem, runner)
        return self._instance_cache[idx]

    def _sample_cache_index(self) -> int:
        return int(self.rng.integers(0, len(self._instance_paths)))

    def _load_nontrivial_instance(self, requested_path: Optional[str] = None):
        last_problem, last_runner = None, None
        for attempt in range(self.config.max_reset_resamples):
            if attempt == 0 and requested_path is not None:
                matches = [i for i, p in enumerate(self._instance_paths) if p == requested_path]
                idx = matches[0] if matches else self._sample_cache_index()
            else:
                idx = self._sample_cache_index()
            problem, runner = self._ensure_built(idx)
            runner.reset_state()
            last_problem, last_runner = problem, runner
            if runner.failed or runner.terminated_in_initial_relaxation:
                continue
            if not runner.done:
                runner.advance_until_stall_or_done()
            if (not runner.done) and (not runner.failed) and runner.is_stalled():
                return problem, runner
        if last_problem is None or last_runner is None:
            raise RuntimeError(
                "Could not load any set-packing instance. Check instance paths and formats."
            )
        raise RuntimeError(
            "Could not sample a non-trivial set-packing instance within "
            f"max_reset_resamples={self.config.max_reset_resamples}. "
            f"Last candidate: {Path(last_problem.instance_path).name}, "
            f"failed={last_runner.failed}, "
            f"terminated_in_initial_relaxation={last_runner.terminated_in_initial_relaxation}, "
            f"initial_lp_solve_seconds={last_runner.initial_lp_solve_seconds:.3f}. "
            "Likely causes: initial LP time limit too small, incompatible instance files, "
            "or all sampled instances are trivial."
        )

    def _count_fractional_binaries(self) -> int:
        if self.runner is None or self.problem is None or self.runner.x_relaxed is None:
            return 0
        return int(
            sum(
                abs(self.runner.x_relaxed[idx] - round(self.runner.x_relaxed[idx])) > _FRAC_EPS
                for idx in self.problem.integer_indices
            )
        )

    def _build_progress_features(self) -> np.ndarray:
        n_integer = max(1, len(self.problem.integer_indices))
        nfracs = self._count_fractional_binaries()
        nfracs_ratio = nfracs / n_integer
        best_nfracs_ratio = self.best_nfracs / n_integer
        nfracs_improve = float(np.clip((self.prev_nfracs - nfracs) / n_integer, -1.0, 1.0))
        current_distance = self.runner.current_distance()
        current_distance_norm = min(1.0, current_distance / n_integer)
        dist_improve = float(np.clip(self.prev_dist_delta / n_integer, -1.0, 1.0))
        elapsed = 0.0 if self.runner.start_time is None else self.runner.config.time_limit - max(0.0, self.runner.remaining_time() or 0.0)
        elapsed_ratio = min(1.0, elapsed / max(1e-8, self.runner.config.time_limit))
        last_flip_ratio = min(1.0, self.runner.last_k / n_integer)
        return np.array(
            [
                float(np.clip(nfracs_ratio, 0.0, 1.0)),
                float(np.clip(best_nfracs_ratio, 0.0, 1.0)),
                nfracs_improve,
                float(np.clip(current_distance_norm, 0.0, 1.0)),
                dist_improve,
                float(np.clip(elapsed_ratio, 0.0, 1.0)),
                float(np.clip(last_flip_ratio, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _build_history_features(self) -> np.ndarray:
        n_integer = max(1, len(self.problem.integer_indices))
        return np.array(
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

    def _build_instance_features(self) -> np.ndarray:
        n = max(1, self.problem.n)
        m = max(1, self.problem.m)
        n_integer = max(1, len(self.problem.integer_indices))
        binary_fraction = len(self.problem.integer_indices) / n
        constraints_per_var = m / n
        lp_obj_norm = self.runner.initial_lp_objective / (abs(self.runner.initial_lp_objective) + 1.0)
        initial_frac = 0.0
        if self.runner.x_relaxed is not None:
            fracs = [abs(self.runner.x_relaxed[idx] - round(self.runner.x_relaxed[idx])) for idx in self.problem.integer_indices]
            if fracs:
                initial_frac = min(1.0, float(np.mean(fracs)) * 2.0)
        density = self.problem.A.nnz / max(1, self.problem.m * self.problem.n)
        sparsity = float(np.clip(1.0 - density, 0.0, 1.0))
        unfixed_fraction = 1.0
        if self.runner.x_relaxed is not None:
            non_integral = sum(abs(self.runner.x_relaxed[idx] - round(self.runner.x_relaxed[idx])) > 1e-6 for idx in self.problem.integer_indices)
            unfixed_fraction = non_integral / n_integer
        # Encode number of objectives; paper uses p in {3,4,5}, normalise to [0,1]
        p_norm = min(1.0, self.problem.n_objectives / 5.0) if hasattr(self.problem, "n_objectives") else min(1.0, self.problem.p / 5.0)
        return np.array(
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

    def _build_observation(self) -> Dict[str, np.ndarray]:
        return {
            "progress": self._build_progress_features(),
            "history": self._build_history_features(),
            "instance": self._build_instance_features(),
        }

    def _current_integer_objective(self) -> float:
        if self.runner is None or self.problem is None:
            return 0.0
        # Use tracked y_values (sum of objectives) when available
        if self.runner.y_values is not None and self.problem.p > 1:
            return float(sum(self.runner.y_values))
        if self.runner.x_rounded is None:
            return 0.0
        # Fall back to profits·x for single-objective or when y unavailable
        return float(
            sum(
                float(self.problem.profits[j]) * float(self.runner.x_rounded[j])
                for j in self.problem.integer_indices
            )
        )

    def _compute_reward(self, prev_distance, prev_best_distance, new_distance, new_best_distance, feasible_found, step_runtime, flip_count, still_stalled):
        n_integer = max(1, len(self.problem.integer_indices))
        delta = (prev_distance - new_distance) / n_integer
        best_gain = max(0.0, prev_best_distance - new_best_distance) / n_integer
        r_frac = self.config.reward_frac_improve * float(delta)
        r_best = self.config.reward_best_improve * float(best_gain)
        r_feas = self.config.reward_feasible_bonus if feasible_found else 0.0
        r_time = -self.config.reward_time_penalty * (step_runtime / max(1e-8, self.runner.config.time_limit))
        r_flip = -self.config.reward_flip_penalty * (flip_count / n_integer)
        r_stall = -self.config.reward_stall_penalty if still_stalled else 0.0
        total = r_frac + r_best + r_feas + r_time + r_flip + r_stall
        return float(total), {
            "r_frac": r_frac, "r_best": r_best, "r_feas": r_feas, "r_time": r_time, "r_flip": r_flip, "r_stall": r_stall
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
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
        self.prev_nfracs = 0
        self.last_action_caused_stall = False
        self.last_action_broke_stall = False
        self.cumulative_flips = 0
        self.consecutive_non_improving = 0
        requested_path = options.get("instance_path") if options else None
        self.problem, self.runner = self._load_nontrivial_instance(requested_path=requested_path)
        self.best_distance = self.runner.current_distance()
        self.best_objective = self._current_integer_objective()
        self.best_nfracs = self._count_fractional_binaries()
        self.prev_nfracs = self.best_nfracs
        obs = self._build_observation()
        info = {
            "episode_id": self.episode_id,
            "instance_path": self.problem.instance_path,
            "instance_name": Path(self.problem.instance_path).name,
            "m": self.problem.m,
            "n": self.problem.n,
            "p": self.problem.p,
            "failed": self.runner.failed,
            "integer_found": self.runner.integer_found,
            "terminated_in_initial_relaxation": self.runner.terminated_in_initial_relaxation,
            "initial_solution_was_integer": self.runner.initial_solution_was_integer,
            "initial_distance": self.runner.initial_distance,
            "current_distance": self.runner.current_distance(),
            "iterations": self.runner.iteration,
            "stall_events": self.runner.stall_events,
            "perturbation_events": self.runner.perturbation_events,
            "objective_quality": self._current_integer_objective(),
            "best_objective_quality": self.best_objective,
            "relaxation_build_seconds": self.runner.relaxation_build_seconds,
            "distance_build_seconds": self.runner.distance_build_seconds,
            "initial_lp_solve_seconds": self.runner.initial_lp_solve_seconds,
            "reset_seconds": self.runner.reset_seconds,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self.runner.done:
            info = {
                "reason": "runner_already_done",
                "instance_path": self.problem.instance_path if self.problem is not None else "",
                "instance_name": (
                    Path(self.problem.instance_path).name if self.problem is not None else ""
                ),
                "m": self.problem.m if self.problem is not None else 0,
                "n": self.problem.n if self.problem is not None else 0,
                "p": self.problem.p if self.problem is not None else 1,
                "failed": self.runner.failed if self.runner is not None else True,
                "terminated_in_initial_relaxation": (
                    self.runner.terminated_in_initial_relaxation
                    if self.runner is not None
                    else False
                ),
            }
            return self._build_observation(), 0.0, True, False, info
        flip_bin = int(np.clip(int(action[0]), 0, 5))
        cont_bin = int(np.clip(int(action[1]), 0, 4))
        n_integer = max(1, len(self.problem.integer_indices))
        flip_count = flip_bin_to_count(flip_bin, n_integer)
        continuation_max_steps = CONTINUATION_MAX_STEPS[cont_bin]
        prev_distance = self.runner.current_distance()
        prev_best_distance = self.best_distance
        prev_nfracs = self._count_fractional_binaries()
        prev_stalled = self.runner.is_stalled()
        rem_before = self.runner.remaining_time()
        elapsed_before = 0.0 if rem_before is None else self.runner.config.time_limit - max(0.0, rem_before)

        self.runner.apply_flip_count(flip_count)
        self.cumulative_flips += flip_count
        if not self.runner.done:
            self.runner.advance_until_stall_or_done(max_steps=continuation_max_steps)

        new_distance = self.runner.current_distance()
        self.best_distance = min(self.best_distance, new_distance)
        current_objective = self._current_integer_objective()
        self.best_objective = max(self.best_objective, current_objective)
        new_nfracs = self._count_fractional_binaries()
        self.best_nfracs = min(self.best_nfracs, new_nfracs)
        rem_after = self.runner.remaining_time()
        elapsed_after = 0.0 if rem_after is None else self.runner.config.time_limit - max(0.0, rem_after)
        step_runtime = max(0.0, elapsed_after - elapsed_before)
        feasible_found = self.runner.integer_found
        failed = self.runner.failed
        still_stalled = self.runner.is_stalled()
        dist_delta = prev_distance - new_distance

        reward, reward_components = self._compute_reward(
            prev_distance, prev_best_distance, new_distance, self.best_distance, feasible_found, step_runtime, flip_count, still_stalled
        )
        self.total_reward += reward
        self.prev_prev_dist_delta = self.prev_dist_delta
        self.prev_dist_delta = dist_delta
        self.prev_nfracs = prev_nfracs
        self.prev_prev_reward = self.prev_reward
        self.prev_reward = reward
        improved = new_distance < prev_distance - 1e-9
        self.consecutive_non_improving = 0 if improved else self.consecutive_non_improving + 1
        self.last_action_caused_stall = still_stalled
        self.last_action_broke_stall = prev_stalled and (not still_stalled)
        self.step_id += 1

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
            "continuation_label": CONTINUATION_LABELS[cont_bin],
            "iterations": self.runner.iteration,
            "stall_events": self.runner.stall_events,
            "perturbation_events": self.runner.perturbation_events,
            "total_flips": self.runner.total_flips,
            "current_distance": new_distance,
            "best_distance": self.best_distance,
            "objective_quality": current_objective,
            "best_objective_quality": self.best_objective,
            "feasible_found": feasible_found,
            "time_to_feasible_seconds": elapsed_after if feasible_found else None,
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
        return self._build_observation(), float(reward), terminated, truncated, info
