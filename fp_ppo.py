from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from docplex.mp.model import Model
from gymnasium import spaces
from scipy import sparse

logger = logging.getLogger(__name__)


# Maximum number of variables the RL policy may flip in one FP step (action is k in {0,…,k_max}).
DEFAULT_K_MAX = 20
INSTANCE_FEATURE_DIM = 16
DEFAULT_TIME_LIMIT = 30.0
DEFAULT_STALL_THRESHOLD = 3
DEFAULT_LOG_INTERVAL = 10
DEFAULT_CPLEX_THREADS = 1
DEFAULT_OFF_STALL_PERTURB_PENALTY = 0.25
INTEGER_VARIABLE_FRACTION = 0.8
INTEGER_TOLERANCE = 1e-6


@dataclass
class ProblemData:
    instance_path: str
    A: sparse.csr_matrix
    b: np.ndarray
    c: list[np.ndarray]
    d: list[float]
    m: int
    n: int
    p: int
    integer_indices: list[int]


def _to_int(value: str) -> int:
    return int(float(value))


def _read_scalar(value) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        return _to_int(array.item())
    return _to_int(array.reshape(-1)[0])


def load_problem(instance_path: str | Path) -> ProblemData:
    path = Path(instance_path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a sparse .npz instance file, got: {path}")

    archive = np.load(path, allow_pickle=False)

    required_keys = {"data", "indices", "indptr", "shape", "b", "c", "d", "n", "m", "p"}
    missing_keys = sorted(required_keys - set(archive.files))
    if missing_keys:
        raise KeyError(f"Missing keys in {path}: {missing_keys}")

    m = _read_scalar(archive["m"])
    n = _read_scalar(archive["n"])
    p = _read_scalar(archive["p"])

    A = sparse.csr_matrix(
        (archive["data"], archive["indices"], archive["indptr"]),
        shape=tuple(archive["shape"]),
        dtype=float,
    )
    b = np.asarray(archive["b"], dtype=float)
    c_matrix = np.asarray(archive["c"], dtype=float).reshape(p, n)
    d = np.asarray(archive["d"], dtype=float).reshape(p).tolist()
    c = [c_matrix[objective_index].copy() for objective_index in range(p)]

    integer_indices = list(range(int(INTEGER_VARIABLE_FRACTION * n)))

    return ProblemData(
        instance_path=str(path),
        A=A,
        b=b,
        c=c,
        d=d,
        m=m,
        n=n,
        p=p,
        integer_indices=integer_indices,
    )


def round_integer_values(values: Sequence[float], integer_indices: Sequence[int]) -> list[float]:
    rounded = list(values)
    for index in integer_indices:
        rounded[index] = round(rounded[index])
    return rounded


def is_integer_solution(
    values: Sequence[float],
    integer_indices: Sequence[int],
    tolerance: float = INTEGER_TOLERANCE,
) -> bool:
    for index in integer_indices:
        if abs(values[index] - round(values[index])) > tolerance:
            return False
    return True


def rounding_changed(
    values: Sequence[float],
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
) -> bool:
    for index in integer_indices:
        if round(values[index]) != rounded_values[index]:
            return True
    return False


def fp_distance(
    values: Sequence[float],
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
) -> float:
    return float(sum(abs(values[index] - rounded_values[index]) for index in integer_indices))


def flip_selected_variables(
    rounded_values: Sequence[float],
    selected_indices: Sequence[int],
) -> list[float]:
    updated = list(rounded_values)
    for index in selected_indices:
        updated[index] = 1 - updated[index]
    return updated


def iter_matrix_row_entries(matrix: sparse.csr_matrix, row_index: int):
    start = matrix.indptr[row_index]
    end = matrix.indptr[row_index + 1]
    indices = matrix.indices[start:end]
    values = matrix.data[start:end]
    for col_index, value in zip(indices, values):
        yield int(col_index), float(value)


def iter_vector_entries(vector: Sequence[float]):
    array = np.asarray(vector).reshape(-1)
    nonzero_indices = np.nonzero(array)[0]
    for index in nonzero_indices:
        yield int(index), float(array[index])


def apply_cplex_threads(model: Model, cplex_threads: int) -> None:
    # CPLEX uses 0 to mean "automatic".
    model.context.cplex_parameters.threads = max(0, int(cplex_threads))


def build_relaxation_model(problem: ProblemData, cplex_threads: int = DEFAULT_CPLEX_THREADS):
    model = Model(name="fp_relaxation")
    apply_cplex_threads(model, cplex_threads)

    x = model.continuous_var_list(problem.n, lb=0, name="x")
    y = model.continuous_var_list(problem.p, name="y")

    for row_index in range(problem.m):
        model.add_constraint(
            model.sum(value * x[col_index] for col_index, value in iter_matrix_row_entries(problem.A, row_index))
            <= problem.b[row_index]
        )

    for index in problem.integer_indices:
        model.add_constraint(x[index] <= 1)

    for objective_index in range(problem.p):
        expr = model.sum(value * x[col_index] for col_index, value in iter_vector_entries(problem.c[objective_index]))
        model.add_constraint(expr + problem.d[objective_index] == y[objective_index])

    model.maximize(model.sum(y))
    return model, x, y


def build_distance_model(problem: ProblemData, cplex_threads: int = DEFAULT_CPLEX_THREADS):
    model = Model(name="fp_distance")
    apply_cplex_threads(model, cplex_threads)

    z = model.continuous_var_list(problem.n, lb=0, name="z")
    y = model.continuous_var_list(problem.p, name="y")
    distance_var = model.continuous_var(lb=0, name="distance")

    for row_index in range(problem.m):
        model.add_constraint(
            model.sum(value * z[col_index] for col_index, value in iter_matrix_row_entries(problem.A, row_index))
            <= problem.b[row_index]
        )

    for index in problem.integer_indices:
        model.add_constraint(z[index] <= 1)

    for objective_index in range(problem.p):
        expr = model.sum(value * z[col_index] for col_index, value in iter_vector_entries(problem.c[objective_index]))
        model.add_constraint(expr + problem.d[objective_index] == y[objective_index])

    model.minimize(distance_var)
    return model, z, y, distance_var


def solve_with_time_limit(model: Model, max_seconds: float | None):
    if max_seconds is not None:
        model.set_time_limit(max(0.01, float(max_seconds)))
    return model.solve(log_output=False)


def solve_relaxation_model(model: Model, x_vars, y_vars, max_seconds: float | None = None):
    # docplex Model.solve() returns True/False (not None) on failure in typical versions.
    ok = solve_with_time_limit(model, max_seconds)
    if not ok:
        status = getattr(model, "solve_status", None)
        details = getattr(model, "solve_details", None)
        logger.warning("Relaxation solve failed: status=%s details=%s", status, details)
        return None

    x_values = [float(var.solution_value) for var in x_vars]
    y_values = [float(var.solution_value) for var in y_vars]
    return x_values, y_values, float(model.objective_value)


def solve_distance_model(
    model: Model,
    z_vars,
    y_vars,
    distance_var,
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
    max_seconds: float | None = None,
):
    distance_constraint = model.add_constraint(
        distance_var
        >= model.sum(z_vars[index] for index in integer_indices if rounded_values[index] == 0)
        + model.sum(1 - z_vars[index] for index in integer_indices if rounded_values[index] == 1)
    )

    try:
        ok = solve_with_time_limit(model, max_seconds)
        if not ok:
            status = getattr(model, "solve_status", None)
            details = getattr(model, "solve_details", None)
            logger.warning("Distance solve failed: status=%s details=%s", status, details)
            return None

        x_values = [float(var.solution_value) for var in z_vars]
        y_values = [float(var.solution_value) for var in y_vars]
        return x_values, y_values, float(model.objective_value)
    finally:
        model.remove_constraint(distance_constraint)


class FeasibilityPumpRunner:
    def __init__(
        self,
        problem: ProblemData,
        max_iterations: int = 100,
        time_limit: float = DEFAULT_TIME_LIMIT,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        cplex_threads: int = DEFAULT_CPLEX_THREADS,
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold
        self.cplex_threads = int(cplex_threads)

        self.relaxation_model = None
        self.relaxation_x = None
        self.relaxation_y = None
        self.distance_model = None
        self.distance_z = None
        self.distance_y = None
        self.distance_var = None
        self.episode_index = 0

        self.start_time = None
        self.iteration = 0
        self.decision_count = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.consecutive_no_change = 0
        self.last_rounding_changed = True
        self.last_flip_indices: list[int] = []
        self.total_flips = 0
        self.no_flip_steps = 0
        self.perturb_steps = 0
        self.off_stall_perturb_steps = 0
        self.stall_events = 0
        self.stall_recoveries = 0
        self._was_stalled_last_step = False
        self.relaxation_build_seconds = 0.0
        self.distance_build_seconds = 0.0
        self.relaxation_solve_seconds = 0.0
        self.last_distance_solve_seconds = 0.0
        self.reset_seconds = 0.0

        self.x_list = None
        self.x_tilde = None
        self.y_values = None

    def remaining_time(self) -> float | None:
        if self.start_time is None:
            return None
        return self.time_limit - (time.time() - self.start_time)

    def reset(self) -> None:
        reset_started = time.time()
        instance_name = Path(self.problem.instance_path).name

        logger.info(
            "Episode %d building FP models for %s (n=%d, m=%d, p=%d, threads=%d)",
            self.episode_index,
            instance_name,
            self.problem.n,
            self.problem.m,
            self.problem.p,
            self.cplex_threads,
        )

        build_started = time.time()
        self.relaxation_model, self.relaxation_x, self.relaxation_y = build_relaxation_model(
            self.problem,
            cplex_threads=self.cplex_threads,
        )
        self.relaxation_build_seconds = time.time() - build_started

        build_started = time.time()
        self.distance_model, self.distance_z, self.distance_y, self.distance_var = build_distance_model(
            self.problem,
            cplex_threads=self.cplex_threads,
        )
        self.distance_build_seconds = time.time() - build_started

        logger.info(
            "Episode %d built FP models for %s in %.2fs (relaxation=%.2fs, distance=%.2fs)",
            self.episode_index,
            instance_name,
            self.relaxation_build_seconds + self.distance_build_seconds,
            self.relaxation_build_seconds,
            self.distance_build_seconds,
        )

        self.iteration = 0
        self.decision_count = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.consecutive_no_change = 0
        self.last_rounding_changed = True
        self.last_flip_indices = []
        self.total_flips = 0
        self.no_flip_steps = 0
        self.perturb_steps = 0
        self.off_stall_perturb_steps = 0
        self.stall_events = 0
        self.stall_recoveries = 0
        self._was_stalled_last_step = False
        self.relaxation_solve_seconds = 0.0
        self.last_distance_solve_seconds = 0.0
        self.x_list = None
        self.x_tilde = None
        self.y_values = None

        # Match the baseline FP semantics more closely:
        # the FP time limit starts after model construction.
        self.start_time = time.time()
        logger.info("Episode %d solving initial LP relaxation for %s", self.episode_index, instance_name)

        try:
            remaining_time = self.remaining_time()
            if remaining_time is not None and remaining_time <= 0:
                self.done = True
                logger.warning(
                    "Episode %d ran out of time before the initial LP relaxation for %s",
                    self.episode_index,
                    instance_name,
                )
                return

            solve_started = time.time()
            relaxation_result = solve_relaxation_model(
                self.relaxation_model,
                self.relaxation_x,
                self.relaxation_y,
                max_seconds=remaining_time,
            )
            self.relaxation_solve_seconds = time.time() - solve_started

            if relaxation_result is None:
                self.failed = True
                self.done = True
                logger.warning(
                    "Episode %d initial LP relaxation failed for %s after %.2fs",
                    self.episode_index,
                    instance_name,
                    self.relaxation_solve_seconds,
                )
                return

            self.x_list, self.y_values, _ = relaxation_result

            if is_integer_solution(self.x_list, self.problem.integer_indices):
                self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
                self.integer_found = True
                self.done = True
                logger.info(
                    "Episode %d initial LP relaxation already found an integer solution for %s in %.2fs",
                    self.episode_index,
                    instance_name,
                    self.relaxation_solve_seconds,
                )
                return

            self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
            self.consecutive_no_change = 0
            logger.info(
                "Episode %d reset complete for %s in %.2fs (initial_solve=%.2fs, distance=%.4f)",
                self.episode_index,
                instance_name,
                self.relaxation_build_seconds + self.distance_build_seconds + self.relaxation_solve_seconds,
                self.relaxation_solve_seconds,
                self.current_distance(),
            )
        finally:
            self.reset_seconds = time.time() - reset_started

    def is_stalled(self) -> bool:
        return self.consecutive_no_change >= self.stall_threshold

    def run_one_iteration(self, selected_indices: Sequence[int]) -> bool:
        if self.done or self.x_tilde is None or self.x_list is None:
            return False

        instance_name = Path(self.problem.instance_path).name
        previous_distance = self.current_distance()
        num_flips = len(selected_indices)

        # ── Bug fix 1 & 2: guard termination BEFORE mutating state or updating
        # counters.  Previously x_tilde was flipped and counters incremented even
        # when the episode was about to exit without solving, leaving x_tilde and
        # x_list inconsistent and inflating per-episode statistics by one phantom
        # decision.
        if self.iteration >= self.max_iterations:
            self.done = True
            return False

        remaining_time = self.remaining_time()
        if remaining_time is not None and remaining_time <= 0:
            self.done = True
            return False

        # Safe to update state: we know we will actually run the solve.
        self.decision_count += 1
        self.last_flip_indices = list(selected_indices)
        self.total_flips += num_flips
        if num_flips > 0:
            self.perturb_steps += 1
        else:
            self.no_flip_steps += 1

        if selected_indices:
            self.x_tilde = flip_selected_variables(self.x_tilde, selected_indices)

        should_log_iteration = num_flips > 0
        self.last_rounding_changed = False

        if should_log_iteration:
            logger.debug(
                "Episode %d iteration %d starting for %s: flips=%d distance=%.4f stalled=%s",
                self.episode_index,
                self.iteration + 1,
                instance_name,
                num_flips,
                previous_distance,
                self.is_stalled(),
            )

        solve_started = time.time()
        distance_result = solve_distance_model(
            self.distance_model,
            self.distance_z,
            self.distance_y,
            self.distance_var,
            self.x_tilde,
            self.problem.integer_indices,
            max_seconds=remaining_time,
        )
        self.last_distance_solve_seconds = time.time() - solve_started
        self.iteration += 1

        if distance_result is None:
            self.failed = True
            self.done = True
            logger.warning(
                "Episode %d distance solve failed for %s at iteration %d after %.2fs",
                self.episode_index,
                instance_name,
                self.iteration,
                self.last_distance_solve_seconds,
            )
            return True

        self.x_list, self.y_values, _ = distance_result

        if is_integer_solution(self.x_list, self.problem.integer_indices):
            self.integer_found = True
            self.done = True
            logger.info(
                "Episode %d integer solution found for %s at iteration %d after %.2fs",
                self.episode_index,
                instance_name,
                self.iteration,
                self.last_distance_solve_seconds,
            )
            return True

        self.last_rounding_changed = rounding_changed(
            self.x_list,
            self.x_tilde,
            self.problem.integer_indices,
        )

        if self.last_rounding_changed:
            self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
            self.consecutive_no_change = 0
        elif num_flips > 0:
            # Bug fix 3: we deliberately changed x_tilde via perturbation, so the
            # "consecutive no-change" streak is broken regardless of whether the LP
            # solution happens to round back to the flipped target.  Without this
            # reset, the stall counter keeps growing after every perturbation that
            # the LP immediately absorbs, causing the agent to see an ever-rising
            # stall signal even though it is actively perturbing.
            self.consecutive_no_change = 0
        else:
            self.consecutive_no_change += 1

        currently_stalled = self.is_stalled()
        if currently_stalled and not self._was_stalled_last_step:
            self.stall_events += 1
        if self._was_stalled_last_step and not currently_stalled:
            self.stall_recoveries += 1
        self._was_stalled_last_step = currently_stalled

        remaining_time = self.remaining_time()
        if self.iteration >= self.max_iterations:
            self.done = True
        elif remaining_time is not None and remaining_time <= 0:
            self.done = True

        heartbeat_iteration = self.iteration <= 3 or self.iteration % DEFAULT_LOG_INTERVAL == 0
        should_log_iteration = (
            should_log_iteration
            or heartbeat_iteration
            or self.last_distance_solve_seconds >= 5.0
            or self.is_stalled()
            or self.done
        )

        if should_log_iteration:
            message = (
                "Episode %d iteration %d complete for %s: flips=%d solve=%.2fs distance=%.4f changed=%s stalled=%s done=%s"
            )
            args = (
                self.episode_index,
                self.iteration,
                instance_name,
                num_flips,
                self.last_distance_solve_seconds,
                self.current_distance(),
                self.last_rounding_changed,
                self.is_stalled(),
                self.done,
            )

            if (
                num_flips > 0
                or heartbeat_iteration
                or self.last_distance_solve_seconds >= 5.0
                or self.is_stalled()
                or self.done
            ):
                logger.info(message, *args)
            else:
                logger.debug(message, *args)

        return True

    def current_distance(self) -> float:
        if self.integer_found:
            return 0.0
        if self.x_list is None or self.x_tilde is None:
            return 0.0
        return fp_distance(self.x_list, self.x_tilde, self.problem.integer_indices)


def select_flip_candidates(
    runner: FeasibilityPumpRunner,
    num_candidates: int,
) -> list[int]:
    if runner.x_list is None or runner.x_tilde is None:
        return []

    scored_indices = sorted(
        runner.problem.integer_indices,
        key=lambda index: abs(runner.x_list[index] - runner.x_tilde[index]),
        reverse=True,
    )
    return scored_indices[: min(num_candidates, len(scored_indices))]


def _three_stats(values: np.ndarray) -> np.ndarray:
    """Mean / std / max-abs scaled into ~[0, 1] for a coefficient array."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        return np.zeros(3, dtype=np.float32)
    mean_v = float(np.mean(array))
    std_v = float(np.std(array))
    max_abs = float(np.max(np.abs(array)))
    scale = max(max_abs, 1e-9)
    return np.array(
        [
            float(np.tanh(mean_v / scale) * 0.5 + 0.5),
            float(min(1.0, std_v / scale)),
            float(min(1.0, max_abs / (max_abs + 1.0))),
        ],
        dtype=np.float32,
    )


def build_instance_features(problem: ProblemData) -> np.ndarray:
    """Static instance features β (paper-style): size, density, coefficient summaries."""
    n = float(problem.n)
    m = float(problem.m)
    p = float(problem.p)
    nnz = float(problem.A.nnz)
    denom = max(1.0, m * n)
    density = float(min(1.0, nnz / denom))
    log_cap = np.log1p(100_000.0)

    pieces: list[float] = [
        float(min(1.0, np.log1p(n) / log_cap)),
        float(min(1.0, np.log1p(m) / log_cap)),
        float(min(1.0, np.log1p(p) / log_cap)),
        density,
    ]
    a_data = np.asarray(problem.A.data, dtype=np.float64).ravel()
    b_vec = np.asarray(problem.b, dtype=np.float64).ravel()
    if problem.c:
        c_stack = np.concatenate([np.asarray(row, dtype=np.float64).ravel() for row in problem.c])
    else:
        c_stack = np.array([0.0], dtype=np.float64)
    d_vec = np.asarray(problem.d, dtype=np.float64).ravel()

    pieces.extend(_three_stats(a_data).tolist())
    pieces.extend(_three_stats(b_vec).tolist())
    pieces.extend(_three_stats(c_stack).tolist())
    pieces.extend(_three_stats(d_vec).tolist())

    features = np.asarray(pieces, dtype=np.float32)
    if features.shape[0] != INSTANCE_FEATURE_DIM:
        raise RuntimeError(
            f"INSTANCE_FEATURE_DIM={INSTANCE_FEATURE_DIM} but build_instance_features produced {features.shape[0]}"
        )
    return features


def build_observation_k(
    runner: FeasibilityPumpRunner,
    instance_features: np.ndarray,
    k_max: int,
) -> np.ndarray:
    """β (static) plus a short dynamic vector from the current FP state."""
    static = np.asarray(instance_features, dtype=np.float32).reshape(-1)
    if static.shape[0] != INSTANCE_FEATURE_DIM:
        raise ValueError(f"Expected {INSTANCE_FEATURE_DIM} static features, got {static.shape[0]}")

    if runner.x_list is None or runner.x_tilde is None:
        return np.concatenate([static, np.zeros(8, dtype=np.float32)])

    integer_indices = runner.problem.integer_indices
    if not integer_indices:
        return np.concatenate([static, np.zeros(8, dtype=np.float32)])

    fractionality = np.array(
        [abs(runner.x_list[index] - round(runner.x_list[index])) for index in integer_indices],
        dtype=np.float32,
    )

    mean_fractionality = float(np.mean(fractionality)) * 2.0
    max_fractionality = float(np.max(fractionality)) * 2.0
    fractional_ratio = float(np.mean(fractionality > INTEGER_TOLERANCE))
    distance_ratio = runner.current_distance() / max(1, len(integer_indices))
    iteration_ratio = runner.iteration / max(1, runner.max_iterations)
    decision_ratio = min(1.0, runner.decision_count / max(1, runner.max_iterations))
    stall_ratio = min(1.0, runner.consecutive_no_change / max(1, runner.stall_threshold))
    last_k_ratio = min(1.0, len(runner.last_flip_indices) / max(1, k_max))

    dynamic = np.array(
        [
            min(1.0, mean_fractionality),
            min(1.0, max_fractionality),
            min(1.0, fractional_ratio),
            min(1.0, distance_ratio),
            min(1.0, iteration_ratio),
            decision_ratio,
            stall_ratio,
            last_k_ratio,
        ],
        dtype=np.float32,
    )
    return np.concatenate([static, dynamic])


class FeasibilityPumpKEnv(gym.Env):
    """
    PPO chooses perturbation size k each FP iteration (ML-paper-style), not a full bitmask.

    Action:
        k in {0, ..., k_max}: flip the k integer coordinates with largest |x_j - x_tilde_j| (ties by index order).

    Observation:
        Normalized instance features (paper-style beta) plus eight dynamic Feasibility Pump statistics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: Sequence[str | Path],
        k_max: int = DEFAULT_K_MAX,
        max_iterations: int = 100,
        time_limit: float = DEFAULT_TIME_LIMIT,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        cplex_threads: int = DEFAULT_CPLEX_THREADS,
    ):
        super().__init__()

        self.instance_paths = [str(path) for path in instance_paths]
        if not self.instance_paths:
            raise ValueError("instance_paths must contain at least one .npz file")

        self.k_max = int(k_max)
        if self.k_max < 0:
            raise ValueError("k_max must be non-negative")

        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold
        self.cplex_threads = int(cplex_threads)

        self.action_space = spaces.Discrete(self.k_max + 1)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(INSTANCE_FEATURE_DIM + 8,),
            dtype=np.float32,
        )

        self.problem = None
        self.runner = None
        self._instance_features: np.ndarray | None = None
        self.episode_index = 0
        self.last_load_seconds = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        requested_path = None if options is None else options.get("instance_path")
        self.episode_index += 1

        if requested_path is not None:
            chosen_path = requested_path
        else:
            chosen_index = int(self.np_random.integers(0, len(self.instance_paths)))
            chosen_path = self.instance_paths[chosen_index]

        logger.info("Episode %d loading %s", self.episode_index, Path(chosen_path).name)
        load_started = time.time()
        self.problem = load_problem(chosen_path)
        self.last_load_seconds = time.time() - load_started
        logger.info(
            "Episode %d loaded %s in %.2fs",
            self.episode_index,
            Path(chosen_path).name,
            self.last_load_seconds,
        )
        self._instance_features = build_instance_features(self.problem)
        self.runner = FeasibilityPumpRunner(
            problem=self.problem,
            max_iterations=self.max_iterations,
            time_limit=self.time_limit,
            stall_threshold=self.stall_threshold,
            cplex_threads=self.cplex_threads,
        )
        self.runner.episode_index = self.episode_index
        self.runner.reset()

        observation = build_observation_k(self.runner, self._instance_features, self.k_max)
        info = self._build_info()
        return observation, info

    def step(self, action):
        if self.runner is None:
            raise RuntimeError("Call reset() before step().")

        if self.runner.done:
            return (
                build_observation_k(self.runner, self._instance_features or np.zeros(INSTANCE_FEATURE_DIM), self.k_max),
                0.0,
                True,
                False,
                self._build_info(),
            )

        k_action = int(np.asarray(action).reshape(-1)[0])
        k_action = int(np.clip(k_action, 0, self.k_max))
        selected_indices = select_flip_candidates(self.runner, k_action)

        num_flips = len(selected_indices)
        previous_distance = self.runner.current_distance()
        was_stalled_before_action = self.runner.is_stalled()

        step_executed = self.runner.run_one_iteration(selected_indices)
        if step_executed and num_flips > 0 and not was_stalled_before_action:
            self.runner.off_stall_perturb_steps += 1

        if not step_executed:
            observation = build_observation_k(self.runner, self._instance_features, self.k_max)
            info = self._build_info()
            info["num_flips"] = 0
            info["k_action"] = k_action
            return observation, 0.0, self.runner.done, False, info

        next_distance = self.runner.current_distance()
        reward = previous_distance - next_distance
        reward -= 0.02 * num_flips
        reward -= 0.1

        if num_flips == 0 and was_stalled_before_action:
            reward -= 1.0

        if num_flips > 0 and not was_stalled_before_action:
            reward -= DEFAULT_OFF_STALL_PERTURB_PENALTY

        if num_flips > 0 and self.runner.last_rounding_changed:
            reward += 1.0

        if self.runner.integer_found:
            reward += 50.0
        elif self.runner.failed:
            reward -= 25.0
        elif self.runner.done:
            reward -= 5.0

        observation = build_observation_k(self.runner, self._instance_features, self.k_max)
        info = self._build_info()
        info["num_flips"] = num_flips
        info["k_action"] = k_action

        terminated = self.runner.integer_found or self.runner.failed
        truncated = self.runner.done and not terminated
        return observation, float(reward), terminated, truncated, info

    def _build_info(self) -> dict:
        if self.runner is None or self.problem is None:
            return {}

        return {
            "env_episode": self.episode_index,
            "instance_path": self.problem.instance_path,
            "k_max": self.k_max,
            "iterations": self.runner.iteration,
            "decisions": self.runner.decision_count,
            "no_flip_steps": self.runner.no_flip_steps,
            "perturb_steps": self.runner.perturb_steps,
            "off_stall_perturb_steps": self.runner.off_stall_perturb_steps,
            "total_flips": self.runner.total_flips,
            "stall_events": self.runner.stall_events,
            "stall_recoveries": self.runner.stall_recoveries,
            "stalled": self.runner.is_stalled(),
            "integer_found": self.runner.integer_found,
            "failed": self.runner.failed,
            "distance": self.runner.current_distance(),
            "consecutive_no_change": self.runner.consecutive_no_change,
            "last_flip_indices": list(self.runner.last_flip_indices),
            "load_seconds": self.last_load_seconds,
            "reset_seconds": self.runner.reset_seconds,
            "relaxation_build_seconds": self.runner.relaxation_build_seconds,
            "distance_build_seconds": self.runner.distance_build_seconds,
            "initial_solve_seconds": self.runner.relaxation_solve_seconds,
            "last_distance_solve_seconds": self.runner.last_distance_solve_seconds,
            "elapsed_seconds": 0.0 if self.runner.start_time is None else time.time() - self.runner.start_time,
        }
