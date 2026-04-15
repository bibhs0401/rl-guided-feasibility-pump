from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from docplex.mp.model import Model
from gymnasium import spaces
from scipy import sparse

logger = logging.getLogger(__name__)


# Discrete perturbation sizes chosen by the policy at stall events.
DEFAULT_K_CHOICES = (1, 2, 5, 10, 20, 50)
DYNAMIC_FEATURE_DIM = 10
# Kept for backward compatibility with legacy feature builders.
INSTANCE_FEATURE_DIM = 16
DEFAULT_TIME_LIMIT = 30.0
DEFAULT_STALL_THRESHOLD = 3
DEFAULT_MAX_STALLS = 50
DEFAULT_RECENT_DELTA_WINDOW = 5
DEFAULT_LOG_INTERVAL = 10
DEFAULT_CPLEX_THREADS = 1
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
        max_stalls: int = DEFAULT_MAX_STALLS,
        recent_delta_window: int = DEFAULT_RECENT_DELTA_WINDOW,
        cplex_threads: int = DEFAULT_CPLEX_THREADS,
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold
        self.max_stalls = int(max_stalls)
        self.recent_delta_window = max(1, int(recent_delta_window))
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
        self.last_k = 0
        self.total_flips = 0
        self.perturb_steps = 0
        self.flips_since_last_improvement = 0
        self.stall_events = 0
        self.stall_recoveries = 0
        self._was_stalled_last_step = False
        self.last_distance_delta = 0.0
        self.recent_distance_deltas: deque[float] = deque(maxlen=self.recent_delta_window)
        self.relaxation_build_seconds = 0.0
        self.distance_build_seconds = 0.0
        self.relaxation_solve_seconds = 0.0
        self.last_distance_solve_seconds = 0.0
        self.reset_seconds = 0.0
        self.initial_lp_objective = 0.0

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
        self.last_k = 0
        self.total_flips = 0
        self.perturb_steps = 0
        self.flips_since_last_improvement = 0
        self.stall_events = 0
        self.stall_recoveries = 0
        self._was_stalled_last_step = False
        self.last_distance_delta = 0.0
        self.recent_distance_deltas.clear()
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

            self.x_list, self.y_values, lp_obj = relaxation_result
            self.initial_lp_objective = lp_obj

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
        self.last_k = num_flips
        self.total_flips += num_flips
        if num_flips > 0:
            self.perturb_steps += 1

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
        next_distance = self.current_distance()
        self.last_distance_delta = previous_distance - next_distance
        self.recent_distance_deltas.append(self.last_distance_delta)

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
            self.flips_since_last_improvement = 0
        elif num_flips > 0:
            self.consecutive_no_change = 0
            self.flips_since_last_improvement += num_flips
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
        elif self.stall_events >= self.max_stalls:
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


def build_instance_features(problem: ProblemData) -> np.ndarray:
    """
    Static instance features matching the paper's 16 selected features (Figure 1):
      [0]     numVariables  (log-normalised)
      [1]     numConstraints (log-normalised)
      [2]     nonZeroCost   (fraction of non-zero objective coefficients)
      [3-10]  Column-wise A statistics:
                Min(col_min), Min(col_max), Min(col_std),
                Max(col_min), Max(col_max), Max(col_std),
                Std(col_min), Std(col_std)
      [11]    bMin  (normalised by max |b|)
      [12]    bMax  (normalised by max |b|)
      [13]    bStd  (normalised by max |b|)
      [14]    cMin  (normalised by max |c|)
      [15]    cMax  (normalised by max |c|)

    LP objective and LP solve time are dynamic — they appear in build_observation_k.
    nIT (iteration count) is already captured by iteration_ratio in the dynamic features.
    """
    n = problem.n
    m = problem.m

    # ── Column-wise statistics of A (including implicit zeros) ──────────────
    # For each column j, compute min, max, std over all m rows.
    # Implicit zeros are included so that statistics reflect the true density.
    A_csc = problem.A.tocsc()
    col_mins = np.zeros(n, dtype=np.float64)
    col_maxs = np.zeros(n, dtype=np.float64)
    col_stds = np.zeros(n, dtype=np.float64)

    for j in range(n):
        start = int(A_csc.indptr[j])
        end   = int(A_csc.indptr[j + 1])
        nnz_j = end - start
        if nnz_j == 0:
            pass  # col_min = col_max = col_std = 0 already
        else:
            col_data = np.asarray(A_csc.data[start:end], dtype=np.float64)
            n_zeros  = m - nnz_j
            col_mins[j] = min(float(np.min(col_data)), 0.0) if n_zeros > 0 else float(np.min(col_data))
            col_maxs[j] = max(float(np.max(col_data)), 0.0) if n_zeros > 0 else float(np.max(col_data))
            mean_j = float(np.sum(col_data)) / m
            var_j  = float(np.dot(col_data, col_data)) / m - mean_j ** 2
            col_stds[j] = float(np.sqrt(max(0.0, var_j)))

    # Normalise by the known A-coefficient range [-30, 30].
    A_SCALE = 30.0
    min_of_mins = float(np.min(col_mins)) / A_SCALE
    min_of_maxs = float(np.min(col_maxs)) / A_SCALE
    min_of_stds = float(np.min(col_stds)) / A_SCALE
    max_of_mins = float(np.max(col_mins)) / A_SCALE
    max_of_maxs = float(np.max(col_maxs)) / A_SCALE
    max_of_stds = float(np.max(col_stds)) / A_SCALE
    std_of_mins = float(np.std(col_mins))  / A_SCALE
    std_of_stds = float(np.std(col_stds))  / A_SCALE

    # ── b statistics ─────────────────────────────────────────────────────────
    b_vec   = np.asarray(problem.b, dtype=np.float64)
    b_scale = max(float(np.max(np.abs(b_vec))), 1.0)
    b_min   = float(np.min(b_vec)) / b_scale
    b_max   = float(np.max(b_vec)) / b_scale
    b_std   = float(np.std(b_vec)) / b_scale

    # ── c statistics (all objective coefficients flattened) ──────────────────
    if problem.c:
        c_all = np.concatenate([np.asarray(row, dtype=np.float64) for row in problem.c])
    else:
        c_all = np.zeros(1, dtype=np.float64)
    c_scale = max(float(np.max(np.abs(c_all))), 1.0)
    c_min   = float(np.min(c_all)) / c_scale
    c_max   = float(np.max(c_all)) / c_scale

    # ── Instance size ─────────────────────────────────────────────────────────
    log_cap     = np.log1p(10_000.0)
    n_feat      = float(min(1.0, np.log1p(n) / log_cap))
    m_feat      = float(min(1.0, np.log1p(m) / log_cap))
    nonzero_cost = float(np.count_nonzero(c_all)) / max(1, len(c_all))

    features = np.array([
        n_feat, m_feat, nonzero_cost,          # [0-2]  instance size
        min_of_mins, min_of_maxs, min_of_stds, # [3-5]  Min(col_min/max/std)
        max_of_mins, max_of_maxs, max_of_stds, # [6-8]  Max(col_min/max/std)
        std_of_mins, std_of_stds,              # [9-10] Std(col_min/std)  — 8 A-stats total [3-10]
        b_min, b_max, b_std,                   # [11-13] b statistics
        c_min, c_max,                          # [14-15] c statistics
    ], dtype=np.float32)

    if features.shape[0] != INSTANCE_FEATURE_DIM:
        raise RuntimeError(
            f"INSTANCE_FEATURE_DIM={INSTANCE_FEATURE_DIM} but build_instance_features produced {features.shape[0]}"
        )
    return features


def build_observation_k(
    runner: FeasibilityPumpRunner,
    k_max: int,
) -> np.ndarray:
    """
    Option-A compact state used only at stall decision points:
      [0] mean fractionality
      [1] max fractionality
      [2] fraction still fractional
      [3] normalized FP distance
      [4] normalized consecutive no-change streak
      [5] normalized iteration progress
      [6] recent normalized distance delta (window mean)
      [7] normalized stall depth
      [8] normalized previous k
      [9] normalized flips since last improvement
    """
    if runner.x_list is None or runner.x_tilde is None:
        return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

    integer_indices = runner.problem.integer_indices
    if not integer_indices:
        return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

    fractionality = np.array(
        [abs(runner.x_list[index] - round(runner.x_list[index])) for index in integer_indices],
        dtype=np.float32,
    )

    num_integer = max(1, len(integer_indices))
    mean_fractionality = float(np.mean(fractionality)) * 2.0
    max_fractionality = float(np.max(fractionality)) * 2.0
    fractional_ratio = float(np.mean(fractionality > INTEGER_TOLERANCE))
    distance_ratio = runner.current_distance() / num_integer
    stall_ratio = min(1.0, runner.consecutive_no_change / max(1, runner.stall_threshold))
    iteration_ratio = min(1.0, runner.iteration / max(1, runner.max_iterations))
    recent_distance_delta = 0.0
    if runner.recent_distance_deltas:
        recent_distance_delta = float(np.mean(runner.recent_distance_deltas)) / num_integer
    recent_distance_delta = float(np.clip(recent_distance_delta, -1.0, 1.0))
    stall_depth_ratio = min(1.0, runner.stall_events / max(1, runner.max_stalls))
    last_k_ratio = min(1.0, runner.last_k / max(1, k_max))
    flips_since_last_improvement_ratio = min(1.0, runner.flips_since_last_improvement / max(1, k_max))

    dynamic = np.array(
        [
            min(1.0, mean_fractionality),
            min(1.0, max_fractionality),
            min(1.0, fractional_ratio),
            min(1.0, distance_ratio),
            stall_ratio,
            iteration_ratio,
            recent_distance_delta,
            stall_depth_ratio,
            last_k_ratio,
            flips_since_last_improvement_ratio,
        ],
        dtype=np.float32,
    )
    return dynamic


class FeasibilityPumpKEnv(gym.Env):
    """
    Option A environment:
    FP runs as a black box between stall events, and PPO acts only at stalls.

    Action:
        index into K_CHOICES; chosen k flips top-k heuristic candidates.

    Observation:
        Compact dynamic state (DYNAMIC_FEATURE_DIM).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: Sequence[str | Path],
        k_choices: Sequence[int] = DEFAULT_K_CHOICES,
        max_iterations: int = 100,
        time_limit: float = DEFAULT_TIME_LIMIT,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        max_stalls: int = DEFAULT_MAX_STALLS,
        recent_delta_window: int = DEFAULT_RECENT_DELTA_WINDOW,
        cplex_threads: int = DEFAULT_CPLEX_THREADS,
    ):
        super().__init__()

        self.instance_paths = [str(path) for path in instance_paths]
        if not self.instance_paths:
            raise ValueError("instance_paths must contain at least one .npz file")

        self.k_choices = tuple(int(k) for k in k_choices)
        if not self.k_choices:
            raise ValueError("k_choices must contain at least one action")
        if any(k <= 0 for k in self.k_choices):
            raise ValueError("k_choices must contain strictly positive values")
        self.k_max = max(self.k_choices)

        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold
        self.max_stalls = int(max_stalls)
        self.recent_delta_window = int(recent_delta_window)
        self.cplex_threads = int(cplex_threads)

        self.action_space = spaces.Discrete(len(self.k_choices))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(DYNAMIC_FEATURE_DIM,),
            dtype=np.float32,
        )

        self.problem = None
        self.runner = None
        self.episode_index = 0
        self.last_load_seconds = 0.0
        self._distance_before_stall = 0.0

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
        self.runner = FeasibilityPumpRunner(
            problem=self.problem,
            max_iterations=self.max_iterations,
            time_limit=self.time_limit,
            stall_threshold=self.stall_threshold,
            max_stalls=self.max_stalls,
            recent_delta_window=self.recent_delta_window,
            cplex_threads=self.cplex_threads,
        )
        self.runner.episode_index = self.episode_index
        self.runner.reset()
        self._advance_to_decision_point()

        observation = build_observation_k(self.runner, self.k_max)
        info = self._build_info()
        return observation, info

    def step(self, action):
        if self.runner is None:
            raise RuntimeError("Call reset() before step().")

        if self.runner.done:
            return (
                build_observation_k(self.runner, self.k_max),
                0.0,
                True,
                False,
                self._build_info(),
            )

        self._advance_to_decision_point()
        if self.runner.done:
            observation = build_observation_k(self.runner, self.k_max)
            return observation, -5.0, False, True, self._build_info()

        action_index = int(np.asarray(action).reshape(-1)[0])
        action_index = int(np.clip(action_index, 0, len(self.k_choices) - 1))
        chosen_k = int(self.k_choices[action_index])
        previous_distance = self.runner.current_distance()
        selected_indices = select_flip_candidates(self.runner, chosen_k)
        num_flips = len(selected_indices)

        step_executed = self.runner.run_one_iteration(selected_indices)
        stall_broken = bool(self.runner.last_rounding_changed)
        still_stalled_after_perturb = self.runner.is_stalled()

        if not step_executed:
            observation = build_observation_k(self.runner, self.k_max)
            info = self._build_info()
            info["num_flips"] = 0
            info["k_action"] = chosen_k
            return observation, 0.0, self.runner.done, False, info

        self._advance_to_decision_point()
        next_distance = self.runner.current_distance()
        num_integer = max(1, len(self.runner.problem.integer_indices))
        reward = (previous_distance - next_distance) / num_integer
        reward -= 0.01 * num_flips

        if stall_broken:
            reward += 2.0
        if still_stalled_after_perturb:
            reward -= 0.5

        terminated = self.runner.integer_found or self.runner.failed
        truncated = self.runner.done and not terminated
        if self.runner.integer_found:
            reward += 20.0
        elif terminated or truncated:
            reward -= 5.0

        observation = build_observation_k(self.runner, self.k_max)
        info = self._build_info()
        info["num_flips"] = num_flips
        info["k_action"] = chosen_k
        info["stall_broken"] = stall_broken
        info["still_stalled_after_perturb"] = still_stalled_after_perturb

        return observation, float(reward), terminated, truncated, info

    def _advance_to_decision_point(self) -> None:
        if self.runner is None:
            return

        while not self.runner.done and not self.runner.is_stalled():
            executed = self.runner.run_one_iteration([])
            if not executed:
                break

    def _build_info(self) -> dict:
        if self.runner is None or self.problem is None:
            return {}

        return {
            "env_episode": self.episode_index,
            "instance_path": self.problem.instance_path,
            "k_choices": list(self.k_choices),
            "iterations": self.runner.iteration,
            "decisions": self.runner.decision_count,
            "perturb_steps": self.runner.perturb_steps,
            "total_flips": self.runner.total_flips,
            "last_k": self.runner.last_k,
            "flips_since_last_improvement": self.runner.flips_since_last_improvement,
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
