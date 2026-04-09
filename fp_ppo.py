from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from docplex.mp.model import Model
from gymnasium import spaces
from scipy import sparse


DEFAULT_K_VALUES = [0, 2, 5, 10, 15, 20]
INTEGER_VARIABLE_FRACTION = 0.8
INTEGER_TOLERANCE = 1e-6


@dataclass
class ProblemData:
    instance_path: str
    A: np.ndarray | sparse.csr_matrix
    b: np.ndarray
    c: list[np.ndarray]
    d: list[float]
    m: int
    n: int
    p: int
    integer_indices: list[int]


def _to_int(value: str) -> int:
    return int(float(value))


def load_problem_from_csv(csv_path: str | Path) -> ProblemData:
    csv_path = str(csv_path)
    with open(csv_path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"Instance file is empty: {csv_path}")

    m = _to_int(rows[0]["m"])
    n = _to_int(rows[0]["n"])
    p = _to_int(rows[0]["p"])

    d = [float(rows[i]["d"]) for i in range(p)]
    c = []
    for objective_index in range(p):
        start = objective_index * n
        end = (objective_index + 1) * n
        c.append(np.array([float(rows[i]["c"]) for i in range(start, end)], dtype=float))

    b = np.array([float(rows[i]["b"]) for i in range(m)], dtype=float)
    A = np.array([float(row["A"]) for row in rows], dtype=float).reshape(m, n)
    integer_indices = list(range(int(INTEGER_VARIABLE_FRACTION * n)))

    return ProblemData(
        instance_path=csv_path,
        A=A,
        b=b,
        c=c,
        d=d,
        m=m,
        n=n,
        p=p,
        integer_indices=integer_indices,
    )


def _read_scalar(value) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        return _to_int(array.item())
    return _to_int(array.reshape(-1)[0])


def load_problem_from_npz(npz_path: str | Path) -> ProblemData:
    npz_path = str(npz_path)
    archive = np.load(npz_path, allow_pickle=False)

    required_keys = {"data", "indices", "indptr", "shape", "b", "c", "d", "n", "m", "p"}
    missing_keys = sorted(required_keys - set(archive.files))
    if missing_keys:
        raise KeyError(f"Missing keys in {npz_path}: {missing_keys}")

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
        instance_path=npz_path,
        A=A,
        b=b,
        c=c,
        d=d,
        m=m,
        n=n,
        p=p,
        integer_indices=integer_indices,
    )


def load_problem(instance_path: str | Path) -> ProblemData:
    path = Path(instance_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return load_problem_from_npz(path)
    if suffix == ".csv":
        return load_problem_from_csv(path)

    raise ValueError(f"Unsupported instance format: {path}")


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


def flip_top_k(
    rounded_values: Sequence[float],
    relaxed_values: Sequence[float],
    integer_indices: Sequence[int],
    k: int,
) -> list[float]:
    updated = list(rounded_values)
    if k <= 0 or not integer_indices:
        return updated

    scored_indices = sorted(
        integer_indices,
        key=lambda index: abs(relaxed_values[index] - rounded_values[index]),
        reverse=True,
    )

    for index in scored_indices[: min(k, len(scored_indices))]:
        updated[index] = 1 - updated[index]

    return updated


def iter_matrix_row_entries(matrix: np.ndarray | sparse.csr_matrix, row_index: int):
    if sparse.isspmatrix_csr(matrix):
        start = matrix.indptr[row_index]
        end = matrix.indptr[row_index + 1]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        for col_index, value in zip(indices, values):
            yield int(col_index), float(value)
        return

    row = np.asarray(matrix[row_index]).reshape(-1)
    nonzero_indices = np.nonzero(row)[0]
    for col_index in nonzero_indices:
        yield int(col_index), float(row[col_index])


def iter_vector_entries(vector: Sequence[float]):
    array = np.asarray(vector).reshape(-1)
    nonzero_indices = np.nonzero(array)[0]
    for index in nonzero_indices:
        yield int(index), float(array[index])


def build_relaxation_model(problem: ProblemData):
    model = Model(name="fp_relaxation")
    model.context.cplex_parameters.threads = 1

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


def build_distance_model(problem: ProblemData):
    model = Model(name="fp_distance")
    model.context.cplex_parameters.threads = 1

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


def solve_relaxation_model(model: Model, x_vars, y_vars):
    solution = model.solve(log_output=False)
    if solution is None:
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
):
    distance_constraint = model.add_constraint(
        distance_var
        >= model.sum(z_vars[index] for index in integer_indices if rounded_values[index] == 0)
        + model.sum(1 - z_vars[index] for index in integer_indices if rounded_values[index] == 1)
    )

    try:
        solution = model.solve(log_output=False)
        if solution is None:
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
        time_limit: float | None = None,
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.time_limit = time_limit if time_limit is not None else float(np.log(problem.m + problem.n) / 4.0)

        self.relaxation_model = None
        self.relaxation_x = None
        self.relaxation_y = None
        self.distance_model = None
        self.distance_z = None
        self.distance_y = None
        self.distance_var = None

        self.start_time = None
        self.iteration = 0
        self.decision_count = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.stalled = False
        self.last_k = 0

        self.x_list = None
        self.x_tilde = None
        self.y_values = None

    def reset(self) -> None:
        self.relaxation_model, self.relaxation_x, self.relaxation_y = build_relaxation_model(self.problem)
        self.distance_model, self.distance_z, self.distance_y, self.distance_var = build_distance_model(self.problem)

        self.start_time = time.time()
        self.iteration = 0
        self.decision_count = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.stalled = False
        self.last_k = 0
        self.x_list = None
        self.x_tilde = None
        self.y_values = None

        relaxation_result = solve_relaxation_model(self.relaxation_model, self.relaxation_x, self.relaxation_y)
        if relaxation_result is None:
            self.failed = True
            self.done = True
            return

        self.x_list, self.y_values, _ = relaxation_result

        if is_integer_solution(self.x_list, self.problem.integer_indices):
            self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
            self.integer_found = True
            self.done = True
            return

        self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
        self.advance_until_stall_or_done()

    def advance_until_stall_or_done(self) -> None:
        self.stalled = False

        while not self.done:
            if self.iteration >= self.max_iterations:
                self.done = True
                return

            if self.start_time is not None and (time.time() - self.start_time) >= self.time_limit:
                self.done = True
                return

            distance_result = solve_distance_model(
                self.distance_model,
                self.distance_z,
                self.distance_y,
                self.distance_var,
                self.x_tilde,
                self.problem.integer_indices,
            )
            self.iteration += 1

            if distance_result is None:
                self.failed = True
                self.done = True
                return

            self.x_list, self.y_values, _ = distance_result

            if is_integer_solution(self.x_list, self.problem.integer_indices):
                self.integer_found = True
                self.done = True
                return

            if rounding_changed(self.x_list, self.x_tilde, self.problem.integer_indices):
                self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
                continue

            self.stalled = True
            return

    def apply_perturbation(self, k: int) -> None:
        if self.done or self.x_tilde is None or self.x_list is None:
            return

        self.decision_count += 1
        self.last_k = k
        self.stalled = False
        self.x_tilde = flip_top_k(self.x_tilde, self.x_list, self.problem.integer_indices, k)

    def current_distance(self) -> float:
        if self.integer_found:
            return 0.0
        if self.x_list is None or self.x_tilde is None:
            return 0.0
        return fp_distance(self.x_list, self.x_tilde, self.problem.integer_indices)


def build_observation(runner: FeasibilityPumpRunner) -> np.ndarray:
    if runner.x_list is None or runner.x_tilde is None:
        return np.zeros(6, dtype=np.float32)

    integer_indices = runner.problem.integer_indices
    if not integer_indices:
        return np.zeros(6, dtype=np.float32)

    fractionality = np.array(
        [abs(runner.x_list[index] - round(runner.x_list[index])) for index in integer_indices],
        dtype=np.float32,
    )

    mean_fractionality = float(np.mean(fractionality)) * 2.0
    max_fractionality = float(np.max(fractionality)) * 2.0
    fractional_ratio = float(np.mean(fractionality > INTEGER_TOLERANCE))
    distance_ratio = runner.current_distance() / max(1, len(integer_indices))
    iteration_ratio = runner.iteration / max(1, runner.max_iterations)
    decision_ratio = min(1.0, runner.decision_count / 10.0)

    observation = np.array(
        [
            min(1.0, mean_fractionality),
            min(1.0, max_fractionality),
            min(1.0, fractional_ratio),
            min(1.0, distance_ratio),
            min(1.0, iteration_ratio),
            decision_ratio,
        ],
        dtype=np.float32,
    )
    return observation


class FeasibilityPumpKEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: Sequence[str | Path],
        k_values: Sequence[int] | None = None,
        max_iterations: int = 100,
        time_limit: float | None = None,
    ):
        super().__init__()

        self.instance_paths = [str(path) for path in instance_paths]
        if not self.instance_paths:
            raise ValueError("instance_paths must contain at least one CSV file")

        self.k_values = list(k_values or DEFAULT_K_VALUES)
        self.max_iterations = max_iterations
        self.time_limit = time_limit

        self.action_space = spaces.Discrete(len(self.k_values))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self.problem = None
        self.runner = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        requested_path = None if options is None else options.get("instance_path")

        for _ in range(10):
            if requested_path is not None:
                chosen_path = requested_path
            else:
                chosen_index = int(self.np_random.integers(0, len(self.instance_paths)))
                chosen_path = self.instance_paths[chosen_index]

            self.problem = load_problem(chosen_path)
            self.runner = FeasibilityPumpRunner(
                problem=self.problem,
                max_iterations=self.max_iterations,
                time_limit=self.time_limit,
            )
            self.runner.reset()

            if not self.runner.done or requested_path is not None or len(self.instance_paths) == 1:
                break

        observation = build_observation(self.runner)
        info = self._build_info()
        return observation, info

    def step(self, action):
        if self.runner is None:
            raise RuntimeError("Call reset() before step().")

        if self.runner.done:
            return build_observation(self.runner), 0.0, True, False, self._build_info()

        k = self.k_values[int(action)]
        previous_distance = self.runner.current_distance()

        self.runner.apply_perturbation(k)
        self.runner.advance_until_stall_or_done()

        next_distance = self.runner.current_distance()
        reward = previous_distance - next_distance
        reward -= 0.02 * k
        reward -= 0.1

        if k == 0 and self.runner.stalled:
            reward -= 1.0

        if self.runner.integer_found:
            reward += 50.0
        elif self.runner.failed:
            reward -= 25.0
        elif self.runner.done:
            reward -= 5.0

        observation = build_observation(self.runner)
        info = self._build_info()
        info["k"] = k

        return observation, float(reward), self.runner.done, False, info

    def _build_info(self) -> dict:
        if self.runner is None or self.problem is None:
            return {}

        return {
            "instance_path": self.problem.instance_path,
            "iterations": self.runner.iteration,
            "decisions": self.runner.decision_count,
            "stalled": self.runner.stalled,
            "integer_found": self.runner.integer_found,
            "failed": self.runner.failed,
            "distance": self.runner.current_distance(),
            "last_k": self.runner.last_k,
        }
