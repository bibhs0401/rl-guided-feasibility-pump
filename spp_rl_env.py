from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fp_baseline_spp import ACTION_PROPORTIONS, FPConfig, SPPFeasibilityPump
from spp_model import SPPInstance, feasibility_metrics, load_spp_instance, objective_value


@dataclass
class SPPRLEnvConfig:
    instance_paths: list[str]
    fp_config: FPConfig = field(default_factory=FPConfig)
    seed: Optional[int] = 0
    max_reset_attempts: int = 10
    continuation_steps_after_action: Optional[int] = None


class SPPFeasibilityPumpEnv(gym.Env):
    """Stall-only RL environment for set-packing feasibility pump perturbations."""

    metadata = {"render_modes": []}

    def __init__(self, config: SPPRLEnvConfig):
        super().__init__()
        if not config.instance_paths:
            raise ValueError("SPPRLEnvConfig.instance_paths must not be empty")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_space = spaces.Discrete(len(ACTION_PROPORTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self.problem: Optional[SPPInstance] = None
        self.runner: Optional[SPPFeasibilityPump] = None
        self.previous_action = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.episode_id = -1

    def _choose_instance_path(self, requested_path: Optional[str] = None) -> str:
        if requested_path:
            return requested_path
        return self.config.instance_paths[int(self.rng.integers(0, len(self.config.instance_paths)))]

    def _build_runner(self, path: str) -> SPPFeasibilityPump:
        self.problem = load_spp_instance(path)
        cfg = FPConfig(
            max_iterations=self.config.fp_config.max_iterations,
            time_limit=self.config.fp_config.time_limit,
            stall_length=self.config.fp_config.stall_length,
            tolerance=self.config.fp_config.tolerance,
            random_seed=int(self.rng.integers(0, 2**31 - 1)),
            baseline_action=self.config.fp_config.baseline_action,
            stop_on_repaired_incumbent=False,
            verbose=self.config.fp_config.verbose,
        )
        runner = SPPFeasibilityPump(self.problem, cfg)
        runner.reset()
        return runner

    def _fractionality(self) -> tuple[float, float, float]:
        if self.runner is None or self.runner.x_lp is None or self.problem is None:
            return 0.0, 0.0, 0.0
        frac = np.abs(self.runner.x_lp - np.round(self.runner.x_lp))
        if frac.size == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(frac)),
            float(np.max(frac)),
            float(np.mean(frac > self.runner.config.tolerance)),
        )

    def _observation(self) -> np.ndarray:
        if self.runner is None or self.problem is None:
            return np.zeros(10, dtype=np.float32)
        mean_frac, max_frac, frac_fractional = self._fractionality()
        metrics = self.runner.current_metrics()
        distance_norm = min(1.0, self.runner.current_distance() / max(1, self.problem.n))
        iteration_ratio = min(1.0, self.runner.iterations / max(1, self.runner.config.max_iterations))
        violated_norm = min(1.0, metrics.num_violated_constraints / max(1, self.problem.m))
        violation_norm = min(1.0, metrics.total_violation / max(1, self.problem.m))
        stall_flag = 1.0 if metrics.num_violated_constraints > 0 and not self.runner.done else 0.0
        previous_action_norm = self.previous_action / max(1, len(ACTION_PROPORTIONS) - 1)
        no_improve_norm = min(1.0, self.runner.no_improvement_count / max(1, self.runner.config.stall_length))
        return np.asarray(
            [
                mean_frac,
                max_frac,
                frac_fractional,
                distance_norm,
                iteration_ratio,
                violated_norm,
                violation_norm,
                stall_flag,
                previous_action_norm,
                no_improve_norm,
            ],
            dtype=np.float32,
        )

    def _info(self, reward: float = 0.0, termination_reason: str = "") -> dict[str, Any]:
        assert self.runner is not None and self.problem is not None
        result = self.runner.result("rl_guided_fp", average_return=self.episode_return)
        metrics = feasibility_metrics(self.problem, result.final_solution)
        elapsed = time.time() - self.runner.start_time if self.runner.start_time else 0.0
        return {
            "instance_name": self.problem.name,
            "instance_path": self.problem.path,
            "m": self.problem.m,
            "n": self.problem.n,
            "success": result.success,
            "final_objective": result.final_objective,
            "final_violation": metrics.total_violation,
            "num_violated_constraints": metrics.num_violated_constraints,
            "iterations": self.runner.iterations,
            "runtime_seconds": elapsed,
            "num_stalls": self.runner.num_stalls,
            "num_rl_interventions": self.runner.num_rl_interventions,
            "average_return": self.episode_return,
            "reward": reward,
            "termination_reason": termination_reason,
            "notes_error_status": result.notes_error_status,
        }

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
        self.episode_return = 0.0
        self.episode_steps = 0
        self.previous_action = 0

        requested_path = options.get("instance_path") if options else None
        last_runner: Optional[SPPFeasibilityPump] = None
        for _ in range(max(1, self.config.max_reset_attempts)):
            path = self._choose_instance_path(requested_path=requested_path)
            runner = self._build_runner(path)
            runner.run_until_stall_or_done()
            last_runner = runner
            if requested_path or runner.done or runner.num_stalls > 0:
                self.runner = runner
                break
        if self.runner is None:
            self.runner = last_runner
        if self.runner is None:
            raise RuntimeError("Could not initialize a set-packing FP runner")

        return self._observation(), self._info(termination_reason="reset")

    def step(self, action: int):
        assert self.runner is not None and self.problem is not None
        if self.runner.done:
            return self._observation(), 0.0, True, False, self._info(termination_reason="already_done")

        # Safety guard: even if a caller steps early, the environment advances FP
        # naturally first. RL still acts only after this point.
        if self.runner.num_stalls == 0 or not self.runner.current_metrics().num_violated_constraints:
            self.runner.run_until_stall_or_done()
            if self.runner.done:
                return self._observation(), 0.0, True, False, self._info(termination_reason="done_before_action")

        action = int(np.clip(action, 0, len(ACTION_PROPORTIONS) - 1))
        prev_metrics = self.runner.current_metrics()
        prev_distance = self.runner.current_distance()
        prev_violation = prev_metrics.total_violation
        self.runner.apply_perturbation(action, rl_intervention=True)
        self.previous_action = action

        self.runner.run_until_stall_or_done(max_steps=self.config.continuation_steps_after_action)

        new_metrics = self.runner.current_metrics()
        new_distance = self.runner.current_distance()
        reduction_in_violation = (prev_violation - new_metrics.total_violation) / max(1, self.problem.m)
        if not math.isfinite(prev_distance):
            reduction_in_distance = 0.0
        else:
            reduction_in_distance = (prev_distance - new_distance) / max(1, self.problem.n)
        failure_flag = 1.0 if self.runner.done and not self.runner.result("rl_guided_fp").success else 0.0
        feasibility_success = 1.0 if self.runner.result("rl_guided_fp").success else 0.0
        worse_penalty = max(0.0, -reduction_in_violation)
        reward = (
            10.0 * feasibility_success
            + 2.0 * reduction_in_violation
            + reduction_in_distance
            - 0.01 * (self.episode_steps + 1)
            - 1.0 * failure_flag
            - worse_penalty
        )
        self.episode_steps += 1
        self.episode_return += float(reward)

        terminated = bool(self.runner.done and self.runner.result("rl_guided_fp").success)
        truncated = bool(self.runner.done and not terminated)
        reason = "feasible_success" if terminated else ("budget_or_failure" if truncated else "stall")
        return self._observation(), float(reward), terminated, truncated, self._info(reward, reason)


def heuristic_action_from_observation(obs: np.ndarray) -> int:
    violation_norm = float(obs[6])
    no_improve_norm = float(obs[9])
    if violation_norm <= 1e-9 and no_improve_norm < 0.5:
        return 0
    if violation_norm < 0.02:
        return 1
    if violation_norm < 0.08:
        return 2
    if violation_norm < 0.20:
        return 3
    return 4
