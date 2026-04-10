from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from docplex.mp.model import Model
from gymnasium import spaces
from scipy import sparse


DEFAULT_NUM_CANDIDATES = 20
DEFAULT_TIME_LIMIT = 30.0
DEFAULT_STALL_THRESHOLD = 3
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
        time_limit: float = DEFAULT_TIME_LIMIT,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold

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
        self.consecutive_no_change = 0
        self.last_rounding_changed = True
        self.last_flip_indices: list[int] = []

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
        self.consecutive_no_change = 0
        self.last_rounding_changed = True
        self.last_flip_indices = []
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
        self.consecutive_no_change = 0

    def is_stalled(self) -> bool:
        return self.consecutive_no_change >= self.stall_threshold

    def run_one_iteration(self, selected_indices: Sequence[int]) -> None:
        if self.done or self.x_tilde is None or self.x_list is None:
            return

        self.decision_count += 1
        self.last_flip_indices = list(selected_indices)

        if selected_indices:
            self.x_tilde = flip_selected_variables(self.x_tilde, selected_indices)

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

        self.last_rounding_changed = rounding_changed(
            self.x_list,
            self.x_tilde,
            self.problem.integer_indices,
        )

        if self.last_rounding_changed:
            self.x_tilde = round_integer_values(self.x_list, self.problem.integer_indices)
            self.consecutive_no_change = 0
        else:
            self.consecutive_no_change += 1

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


def build_observation(
    runner: FeasibilityPumpRunner,
    candidate_indices: Sequence[int],
    num_candidates: int,
) -> np.ndarray:
    if runner.x_list is None or runner.x_tilde is None:
        return np.zeros(8 + 3 * num_candidates, dtype=np.float32)

    integer_indices = runner.problem.integer_indices
    if not integer_indices:
        return np.zeros(8 + 3 * num_candidates, dtype=np.float32)

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
    stall_ratio = min(1.0, runner.consecutive_no_change / max(1, runner.stall_threshold))
    last_flip_ratio = min(1.0, len(runner.last_flip_indices) / max(1, num_candidates))

    observation = np.array(
        [
            min(1.0, mean_fractionality),
            min(1.0, max_fractionality),
            min(1.0, fractional_ratio),
            min(1.0, distance_ratio),
            min(1.0, iteration_ratio),
            decision_ratio,
            stall_ratio,
            last_flip_ratio,
        ],
        dtype=np.float32,
    )

    candidate_relaxed = []
    candidate_distance = []
    candidate_rounded = []

    for index in candidate_indices[:num_candidates]:
        candidate_relaxed.append(float(runner.x_list[index]))
        candidate_distance.append(float(abs(runner.x_list[index] - runner.x_tilde[index])))
        candidate_rounded.append(float(runner.x_tilde[index]))

    while len(candidate_relaxed) < num_candidates:
        candidate_relaxed.append(0.0)
        candidate_distance.append(0.0)
        candidate_rounded.append(0.0)

    return np.concatenate(
        [
            observation,
            np.array(candidate_relaxed, dtype=np.float32),
            np.array(candidate_distance, dtype=np.float32),
            np.array(candidate_rounded, dtype=np.float32),
        ]
    )


class FeasibilityPumpFlipEnv(gym.Env):
    """
    PPO acts at every FP iteration.

    Action:
    - one bit per candidate variable
    - all zeros means "do not perturb now"
    - selected bits mean "flip these candidate variables now"

    This single action lets the policy learn:
    - when to perturb
    - which variables to flip
    - how many variables to flip
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_paths: Sequence[str | Path],
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        max_iterations: int = 100,
        time_limit: float = DEFAULT_TIME_LIMIT,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
    ):
        super().__init__()

        self.instance_paths = [str(path) for path in instance_paths]
        if not self.instance_paths:
            raise ValueError("instance_paths must contain at least one .npz file")

        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        self.time_limit = float(time_limit)
        self.stall_threshold = stall_threshold

        # Each action bit corresponds to one candidate variable.
        self.action_space = spaces.MultiBinary(self.num_candidates)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8 + 3 * self.num_candidates,),
            dtype=np.float32,
        )

        self.problem = None
        self.runner = None
        self.candidate_indices: list[int] = []

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
                stall_threshold=self.stall_threshold,
            )
            self.runner.reset()
            self.candidate_indices = select_flip_candidates(self.runner, self.num_candidates)

            if not self.runner.done or requested_path is not None or len(self.instance_paths) == 1:
                break

        observation = build_observation(self.runner, self.candidate_indices, self.num_candidates)
        info = self._build_info()
        return observation, info

    def step(self, action):
        if self.runner is None:
            raise RuntimeError("Call reset() before step().")

        if self.runner.done:
            return (
                build_observation(self.runner, self.candidate_indices, self.num_candidates),
                0.0,
                True,
                False,
                self._build_info(),
            )

        action_array = np.asarray(action).reshape(-1)
        selected_indices = []
        for position, raw_value in enumerate(action_array[: self.num_candidates]):
            if position >= len(self.candidate_indices):
                break
            if int(raw_value) == 1:
                selected_indices.append(self.candidate_indices[position])

        num_flips = len(selected_indices)
        previous_distance = self.runner.current_distance()

        # All-zero action means "do not perturb now".
        # Otherwise, flip the selected candidate variables before the next FP solve.
        self.runner.run_one_iteration(selected_indices)
        self.candidate_indices = select_flip_candidates(self.runner, self.num_candidates)

        next_distance = self.runner.current_distance()
        reward = previous_distance - next_distance
        reward -= 0.02 * num_flips
        reward -= 0.1

        if num_flips == 0 and self.runner.is_stalled():
            reward -= 1.0

        if num_flips > 0 and self.runner.last_rounding_changed:
            reward += 1.0

        if self.runner.integer_found:
            reward += 50.0
        elif self.runner.failed:
            reward -= 25.0
        elif self.runner.done:
            reward -= 5.0

        observation = build_observation(self.runner, self.candidate_indices, self.num_candidates)
        info = self._build_info()
        info["num_flips"] = num_flips

        return observation, float(reward), self.runner.done, False, info

    def _build_info(self) -> dict:
        if self.runner is None or self.problem is None:
            return {}

        return {
            "instance_path": self.problem.instance_path,
            "iterations": self.runner.iteration,
            "decisions": self.runner.decision_count,
            "stalled": self.runner.is_stalled(),
            "integer_found": self.runner.integer_found,
            "failed": self.runner.failed,
            "distance": self.runner.current_distance(),
            "num_candidates": len(self.candidate_indices),
            "consecutive_no_change": self.runner.consecutive_no_change,
            "last_flip_indices": list(self.runner.last_flip_indices),
        }
