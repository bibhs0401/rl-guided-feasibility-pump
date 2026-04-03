import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import main_phase1_rl as base

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    PPO = None
    BaseCallback = None
    check_env = None

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    MaskablePPO = None
    ActionMasker = None


random.seed(10)
np.random.seed(10)


logger = logging.getLogger("phase1_sb3")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


GymEnvBase = gym.Env if gym is not None else object


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required to load CSV instances for the SB3 phase-1 pipeline")


def _require_gymnasium():
    if gym is None or spaces is None:
        raise ImportError("gymnasium is required for the SB3 phase-1 environment")


def _require_sb3(require_maskable: bool = False):
    if PPO is None:
        raise ImportError("stable-baselines3 is required for the SB3 phase-1 pipeline")
    if require_maskable and MaskablePPO is None:
        raise ImportError("sb3-contrib is required for MaskablePPO support")


def build_phase1_observation(
    x_list: Sequence[float],
    x_tilde: Sequence[float],
    integer_indices: Sequence[int],
    nIT: int,
    delta: Optional[float],
    top_k: int = 10,
) -> Tuple[np.ndarray, List[int]]:
    if len(integer_indices) == 0:
        candidates: List[int] = []
    else:
        candidates = base.candidate_variables(x_list, x_tilde, integer_indices, top_k=top_k)

    fractional_distances = [abs(x_list[j] - round(x_list[j])) for j in integer_indices]
    candidate_distances = [abs(x_list[j] - x_tilde[j]) for j in candidates]
    candidate_values = [float(x_tilde[j]) for j in candidates]
    candidate_fractionality = [abs(x_list[j] - round(x_list[j])) for j in candidates]

    def _pad(values: Sequence[float]) -> List[float]:
        padded = list(values[:top_k])
        padded.extend([0.0] * (top_k - len(padded)))
        return padded

    mean_frac = float(np.mean(fractional_distances)) if fractional_distances else 0.0
    max_frac = float(np.max(fractional_distances)) if fractional_distances else 0.0
    mismatch_ratio = (
        float(np.mean([1.0 if round(x_list[j]) != x_tilde[j] else 0.0 for j in integer_indices]))
        if integer_indices
        else 0.0
    )
    delta_feature = 0.0 if delta is None else float(np.tanh(delta))

    features = [
        mean_frac,
        max_frac,
        mismatch_ratio,
        min(1.0, float(nIT) / 100.0),
        delta_feature,
        float(len(candidates)) / float(max(1, top_k)),
    ]
    features.extend(_pad(candidate_distances))
    features.extend(_pad(candidate_values))
    features.extend(_pad(candidate_fractionality))

    return np.array(features, dtype=np.float32), candidates


def build_action_mask(candidate_count: int, top_k: int, terminal_state: bool = False) -> np.ndarray:
    mask = np.zeros(top_k + 1, dtype=bool)
    if terminal_state:
        mask[0] = True
    else:
        mask[1:1 + min(candidate_count, top_k)] = True
    return mask


def action_to_variable(action: int, candidates: Sequence[int]) -> Optional[int]:
    if not candidates:
        return None
    if action <= 0:
        return candidates[0]
    rank = action - 1
    if rank >= len(candidates):
        return candidates[0]
    return candidates[rank]


class SB3Phase1Agent:
    def __init__(self, model: Any, top_k: int = 10, deterministic: bool = True):
        self.model = model
        self.top_k = top_k
        self.deterministic = deterministic
        self.uses_masking = MaskablePPO is not None and isinstance(model, MaskablePPO)

    @classmethod
    def load(cls, model_path: str, top_k: int = 10, deterministic: bool = True):
        _require_sb3()

        load_errors = []
        if MaskablePPO is not None:
            try:
                return cls(MaskablePPO.load(model_path), top_k=top_k, deterministic=deterministic)
            except Exception as exc:
                load_errors.append(exc)

        try:
            return cls(PPO.load(model_path), top_k=top_k, deterministic=deterministic)
        except Exception as exc:
            load_errors.append(exc)

        raise RuntimeError(f"Unable to load SB3 model from {model_path!r}: {load_errors}")

    def choose_variable(
        self,
        x_list: Sequence[float],
        x_tilde: Sequence[float],
        integer_indices: Sequence[int],
        nIT: int,
        delta: Optional[float],
    ) -> Optional[int]:
        observation, candidates = build_phase1_observation(
            x_list,
            x_tilde,
            integer_indices,
            nIT,
            delta,
            top_k=self.top_k,
        )
        if not candidates:
            return None

        action_mask = build_action_mask(len(candidates), self.top_k, terminal_state=False)
        if self.uses_masking:
            action, _ = self.model.predict(
                observation,
                deterministic=self.deterministic,
                action_masks=action_mask,
            )
        else:
            action, _ = self.model.predict(observation, deterministic=self.deterministic)

        chosen_variable = action_to_variable(int(action), candidates)
        logger.info(
            "SB3 chose variable %s at nIT=%s from candidates=%s",
            chosen_variable,
            nIT,
            candidates,
        )
        return chosen_variable


class Phase1FeasibilityPumpEnv(GymEnvBase):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        instance_files: Sequence[str],
        top_k: int = 10,
        integer_bonus: float = 10.0,
        invalid_action_penalty: float = -0.25,
        failure_penalty: float = -5.0,
        log_progress: bool = True,
        episode_time_limit: Optional[float] = None,
    ):
        _require_gymnasium()
        _require_pandas()

        super().__init__()

        if not instance_files:
            raise ValueError("instance_files must contain at least one CSV path")

        self.instance_files = list(instance_files)
        self.top_k = top_k
        self.integer_bonus = integer_bonus
        self.invalid_action_penalty = invalid_action_penalty
        self.failure_penalty = failure_penalty
        self.log_progress = log_progress
        self.episode_time_limit = episode_time_limit

        obs_size = 6 + (3 * top_k)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(top_k + 1)

        self.current_instance_path: Optional[str] = None
        self.awaiting_action = False
        self.terminated = False
        self.truncated = False
        self.result_reason = "uninitialized"
        self.current_candidates: List[int] = []
        self.current_observation = np.zeros(obs_size, dtype=np.float32)
        self.delta: Optional[float] = None
        self.last_reward = 0.0
        self.nIT = 0
        self.episode_counter = 0
        self.decision_counter = 0
        self.episode_start_time = None

    def _select_instance(self, options: Optional[Dict[str, Any]]) -> str:
        if options and options.get("instance_path"):
            return str(options["instance_path"])
        index = int(self.np_random.integers(len(self.instance_files)))
        return self.instance_files[index]

    def _load_instance(self, instance_path: str):
        data = pd.read_csv(instance_path)
        A, b, c, m, n, d, p = base.required_data(data)

        self.A = A
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.d = d
        self.p = p
        self.I, self.not_I = base.get_integer_index(n)
        self.TL, self.T, self.TT, self.R = base.FP_parameters(self.I, m, n)
        if self.episode_time_limit is not None:
            self.TL = min(self.TL, float(self.episode_time_limit))
        self.m1, self.multiplicative_y_m1 = base.first_linear_model(A, b, c, d, n, p, self.I)
        self.f, self.objvar, self.z, self.multiplicative_y_f = base.second_model_FP(n, A, b, c, d, p, self.I)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _set_terminal(self, reason: str):
        self.awaiting_action = False
        self.terminated = reason not in {"timeout"}
        self.truncated = reason == "timeout"
        self.result_reason = reason
        self.current_candidates = []
        self.current_observation = self._terminal_observation()
        if self.log_progress:
            elapsed = 0.0 if self.episode_start_time is None else (time.time() - self.episode_start_time)
            logger.info(
                "Episode %s finished | reason=%s | nIT=%s | delta=%s | decisions=%s | elapsed=%.2fs | instance=%s",
                self.episode_counter,
                reason,
                self.nIT,
                self.delta,
                self.decision_counter,
                elapsed,
                self.current_instance_path,
            )

    def _solve_until_decision_or_stop(self):
        while (time.time() - self.fp_start_time) < self.TL:
            self.nIT += 1

            objfun = self.f.add_constraint(
                self.objvar >= sum(self.z[0, j] for j in self.I if self.x_tilde[j] == 0)
                + sum(1 - self.z[0, j] for j in self.I if self.x_tilde[j] == 1)
            )
            solved_model, delta, x_list, y_values = base.solve_second_model_FP(self.f, self.z, self.n)
            self.f.remove_constraint(objfun)

            if y_values == "none":
                self.delta = delta if delta != "none" else self.delta
                self._set_terminal("infeasible")
                return

            self.f = solved_model
            self.delta = float(delta)
            self.x_list = x_list
            self.y_values = y_values

            if base.check_integer(self.I, x_list):
                self.solution_value = sum((np.dot(self.c[i], x_list) + self.d[i]) for i in range(len(self.c)))
                self._set_terminal("integer")
                return

            if self.nIT <= 100:
                mismatch_found = False
                for index in self.I:
                    if round(x_list[index]) != self.x_tilde[index]:
                        mismatch_found = True
                        break

                if mismatch_found:
                    self.x_tilde = base.rounding(x_list, self.I)
                    continue

                self.current_observation, self.current_candidates = build_phase1_observation(
                    x_list,
                    self.x_tilde,
                    self.I,
                    self.nIT,
                    self.delta,
                    top_k=self.top_k,
                )
                self.awaiting_action = True
                self.terminated = False
                self.truncated = False
                self.result_reason = "awaiting_action"
                if self.log_progress:
                    logger.info(
                        "Episode %s awaiting action | nIT=%s | delta=%.6f | candidates=%s | instance=%s",
                        self.episode_counter,
                        self.nIT,
                        self.delta,
                        len(self.current_candidates),
                        self.current_instance_path,
                    )
                return

            for j in self.I:
                ro = np.random.uniform(-0.3, 0.7)
                if (abs(x_list[j] - self.x_tilde[j]) + max(ro, 0)) > 0.5:
                    if self.x_tilde[j] == 0:
                        self.x_tilde[j] = 1
                    elif self.x_tilde[j] == 1:
                        self.x_tilde[j] = 0

        self._set_terminal("timeout")

    def _initialize_episode(self, instance_path: str):
        self.episode_counter += 1
        self.current_instance_path = instance_path
        self.last_reward = 0.0
        self.awaiting_action = False
        self.terminated = False
        self.truncated = False
        self.result_reason = "initializing"
        self.current_candidates = []
        self.delta = None
        self.nIT = 0
        self.decision_counter = 0
        self.solution_value = None

        self._load_instance(instance_path)
        self.fp_start_time = time.time()
        self.episode_start_time = self.fp_start_time
        if self.log_progress:
            logger.info(
                "Episode %s started | instance=%s | m=%s | n=%s | p=%s | TL=%.2fs",
                self.episode_counter,
                instance_path,
                self.m,
                self.n,
                self.p,
                self.TL,
            )

        self.m1, self.z_lp, self.x_relaxed, self.y_values = base.solve_first_linear_model(self.m1, self.n)
        if self.y_values == "none":
            self._set_terminal("infeasible")
            return

        if base.check_integer(self.I, self.x_relaxed):
            self.delta = 0.0
            self.solution_value = self.z_lp
            self._set_terminal("integer")
            return

        self.x_tilde = base.rounding(self.x_relaxed, self.I)
        self._solve_until_decision_or_stop()

    def action_masks(self) -> np.ndarray:
        if self.awaiting_action:
            return build_action_mask(len(self.current_candidates), self.top_k, terminal_state=False)
        return build_action_mask(0, self.top_k, terminal_state=True)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        max_tries = max(1, len(self.instance_files))
        for _ in range(max_tries):
            instance_path = self._select_instance(options)
            self._initialize_episode(instance_path)
            if self.awaiting_action or (options and options.get("instance_path")):
                break
            if self.result_reason not in {"integer", "infeasible"}:
                break

        return self.current_observation.copy(), self._info()

    def _info(self) -> Dict[str, Any]:
        return {
            "instance_path": self.current_instance_path,
            "action_mask": self.action_masks(),
            "nIT": self.nIT,
            "delta": self.delta,
            "result_reason": self.result_reason,
            "candidate_count": len(self.current_candidates),
        }

    def step(self, action: int):
        if self.terminated or self.truncated:
            return self.current_observation.copy(), 0.0, self.terminated, self.truncated, self._info()

        reward = 0.0
        action_mask = self.action_masks()
        action = int(action)

        if action < 0 or action >= len(action_mask) or not action_mask[action]:
            reward += self.invalid_action_penalty

        previous_delta = self.delta
        chosen_variable = action_to_variable(action, self.current_candidates)
        self.decision_counter += 1
        if self.log_progress:
            logger.info(
                "Episode %s decision %s | action=%s | chosen_var=%s | prev_delta=%s | instance=%s",
                self.episode_counter,
                self.decision_counter,
                action,
                chosen_variable,
                previous_delta,
                self.current_instance_path,
            )
        self.x_tilde = base.flip_one(self.x_tilde, chosen_variable)
        self.awaiting_action = False
        self.current_candidates = []

        self._solve_until_decision_or_stop()

        if previous_delta is not None and self.delta is not None:
            reward += previous_delta - self.delta

        if self.result_reason == "integer":
            reward += self.integer_bonus
        elif self.result_reason in {"infeasible", "timeout"}:
            reward += self.failure_penalty

        self.last_reward = float(reward)
        if self.log_progress:
            logger.info(
                "Episode %s post-step | reward=%.6f | delta=%s | reason=%s | nIT=%s",
                self.episode_counter,
                self.last_reward,
                self.delta,
                self.result_reason,
                self.nIT,
            )
        return self.current_observation.copy(), float(reward), self.terminated, self.truncated, self._info()

    def render(self):
        logger.info(
            "instance=%s nIT=%s delta=%s reason=%s candidates=%s",
            self.current_instance_path,
            self.nIT,
            self.delta,
            self.result_reason,
            self.current_candidates,
        )

    def close(self):
        return None


def make_phase1_env(instance_files: Sequence[str], top_k: int = 10, use_masking: bool = True, **env_kwargs):
    env = Phase1FeasibilityPumpEnv(instance_files, top_k=top_k, **env_kwargs)
    if use_masking and ActionMasker is not None:
        return ActionMasker(env, lambda inner_env: inner_env.action_masks())
    return env


class WallClockLimitCallback(BaseCallback):
    def __init__(self, max_seconds: Optional[float], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_seconds = max_seconds
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.max_seconds is not None:
            logger.info("Training wall-clock limit set to %.2fs", self.max_seconds)

    def _on_step(self) -> bool:
        if self.max_seconds is None or self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            logger.warning(
                "Stopping training because wall-clock limit was reached | elapsed=%.2fs | limit=%.2fs",
                elapsed,
                self.max_seconds,
            )
            return False
        return True


def train_phase1_sb3_model(
    instance_files: Sequence[str],
    total_timesteps: int,
    model_path: Optional[str] = None,
    use_masking: bool = True,
    top_k: int = 10,
    policy: str = "MlpPolicy",
    seed: int = 10,
    check_environment: bool = True,
    verbose: int = 1,
    max_train_seconds: Optional[float] = None,
    episode_time_limit: Optional[float] = None,
    **model_kwargs,
):
    _require_gymnasium()
    _require_sb3(require_maskable=False)

    if use_masking and MaskablePPO is None:
        logger.warning("sb3-contrib is not installed, falling back to plain PPO without action masking")
        use_masking = False

    logger.info(
        "Starting SB3 training | instances=%s | timesteps=%s | use_masking=%s | top_k=%s | check_env=%s | max_train_seconds=%s | episode_time_limit=%s",
        len(instance_files),
        total_timesteps,
        use_masking,
        top_k,
        check_environment,
        max_train_seconds,
        episode_time_limit,
    )

    base_env = Phase1FeasibilityPumpEnv(
        instance_files,
        top_k=top_k,
        episode_time_limit=episode_time_limit,
    )
    if check_environment and check_env is not None:
        check_env(base_env, warn=True)

    train_env = base_env
    if use_masking and ActionMasker is not None:
        train_env = ActionMasker(base_env, lambda inner_env: inner_env.action_masks())
        model = MaskablePPO(policy, train_env, verbose=verbose, seed=seed, **model_kwargs)
    else:
        model = PPO(policy, train_env, verbose=verbose, seed=seed, **model_kwargs)

    callback = None
    if max_train_seconds is not None:
        if BaseCallback is None:
            raise ImportError("stable-baselines3 callback support is unavailable in this environment")
        callback = WallClockLimitCallback(max_train_seconds, verbose=verbose)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_path:
        model.save(model_path)
        logger.info("Saved SB3 phase-1 model to %s", model_path)

    return model


def feasibility_pump(
    f,
    I,
    c,
    d,
    TT,
    n,
    objvar,
    z,
    time_limit,
    x_tilde,
    y_values,
    phase1_agent: Optional[SB3Phase1Agent] = None,
):
    if phase1_agent is None:
        raise ValueError("phase1_agent must be provided for the SB3-guided feasibility pump")

    nIT = 0
    FP_start_time = time.time()
    delta = None

    logger.info("SB3-guided FP started | time_limit=%.2fs", time_limit)

    while (time.time() - FP_start_time) < time_limit:
        nIT += 1

        objfun = f.add_constraint(
            objvar >= sum(z[0, j] for j in I if x_tilde[j] == 0)
            + sum(1 - z[0, j] for j in I if x_tilde[j] == 1)
        )
        solved_model, delta, x_list, y_values = base.solve_second_model_FP(f, z, n)
        f.remove_constraint(objfun)

        if y_values == "none":
            logger.warning("SB3-guided FP stopped: no y_values at nIT=%s", nIT)
            break

        f = solved_model
        if base.check_integer(I, x_list):
            z_ip = sum((np.dot(c[i], x_list) + d[i]) for i in range(len(c)))
            logger.info(
                "SB3-guided FP found integer solution | nIT=%s | delta=%.6f",
                nIT,
                delta,
            )
            return f, x_list, y_values, z_ip, delta, time.time() - FP_start_time, nIT

        if nIT <= 100:
            mismatch_found = False
            for items in I:
                if round(x_list[items]) != x_tilde[items]:
                    mismatch_found = True
                    break

            if mismatch_found:
                x_tilde = base.rounding(x_list, I)
            else:
                chosen_var = phase1_agent.choose_variable(x_list, x_tilde, I, nIT, float(delta))
                x_tilde = base.flip_one(x_tilde, chosen_var)
        else:
            for j in I:
                ro = np.random.uniform(-0.3, 0.7)
                if (abs(x_list[j] - x_tilde[j]) + max(ro, 0)) > 0.5:
                    if x_tilde[j] == 0:
                        x_tilde[j] = 1
                    elif x_tilde[j] == 1:
                        x_tilde[j] = 0

    logger.warning("SB3-guided FP timed out | nIT=%s", nIT)
    return f, "none", "none", "none", delta, time.time() - FP_start_time, nIT


def multiplicative_FP(
    start_time,
    z,
    n,
    I,
    c,
    d,
    TT,
    TL,
    previous_y_values,
    m1,
    f,
    objvar,
    phase1_agent: Optional[SB3Phase1Agent] = None,
):
    nIT_list = []
    cut_iteration = 0

    for _ in range(1):
        m1, z_lp, x_relaxed, y_values = base.solve_first_linear_model(m1, n)

        if y_values == "none":
            delta = "none"
            decision_variables = x_relaxed
            algorithm_objective_value = z_lp
            solution_time = time.time() - start_time
            return (
                m1,
                f,
                decision_variables,
                previous_y_values,
                algorithm_objective_value,
                delta,
                solution_time,
                nIT_list,
                cut_iteration,
            )

        if base.check_integer(I, x_relaxed):
            delta = 0
            decision_variables = x_relaxed
            algorithm_objective_value = z_lp
            solution_time = time.time() - start_time
        else:
            x_tilde = base.rounding(x_relaxed, I)
            f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT = feasibility_pump(
                f,
                I,
                c,
                d,
                TT,
                n,
                objvar,
                z,
                TL,
                x_tilde,
                y_values,
                phase1_agent=phase1_agent,
            )
            nIT_list.append(nIT)

            if decision_variables == "none":
                return (
                    m1,
                    f,
                    decision_variables,
                    previous_y_values,
                    algorithm_objective_value,
                    delta,
                    solution_time,
                    nIT_list,
                    cut_iteration,
                )

        previous_y_values = y_values

    return (
        m1,
        f,
        decision_variables,
        previous_y_values,
        algorithm_objective_value,
        delta,
        solution_time,
        nIT_list,
        cut_iteration,
    )


def main_function(
    A,
    b,
    c,
    m,
    n,
    d,
    p,
    phase1_agent: Optional[SB3Phase1Agent] = None,
):
    if phase1_agent is None:
        raise ValueError("phase1_agent must be provided to main_phase1_sb3.main_function")

    epsilon, multiplier = base.MILMMP_parameters()
    I, not_I = base.get_integer_index(n)
    TL, T, TT, R = base.FP_parameters(I, m, n)
    y_values = [0 for _ in range(p)]

    logger.info("=" * 80)
    logger.info("SB3 main_function started | m=%s | n=%s | p=%s | TL=%.2f", m, n, p, TL)

    m1, multiplicative_y_m1 = base.first_linear_model(A, b, c, d, n, p, I)
    f, objvar, z, multiplicative_y_f = base.second_model_FP(n, A, b, c, d, p, I)

    start_time = time.time()
    m1, f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = multiplicative_FP(
        start_time,
        z,
        n,
        I,
        c,
        d,
        TT,
        TL,
        y_values,
        m1,
        f,
        objvar,
        phase1_agent=phase1_agent,
    )

    solution_time = time.time() - start_time
    logger.info("SB3 main_function finished | time=%.2fs | obj=%s", solution_time, algorithm_objective_value)

    return decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration
