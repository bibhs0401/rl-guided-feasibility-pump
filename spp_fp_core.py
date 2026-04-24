from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from docplex.mp.model import Model
from scipy import sparse

logger = logging.getLogger(__name__)
INTEGER_TOLERANCE = 1e-6


@dataclass
class SPPProblemInstance:
    instance_path: str
    A: sparse.csr_matrix
    b: np.ndarray
    profits: np.ndarray
    m: int
    n: int
    integer_indices: list[int]


@dataclass
class SPFPRunConfig:
    max_iterations: int = 100
    time_limit: float = 30.0
    initial_lp_time_limit: float | None = 60.0
    stall_threshold: int = 1
    max_stalls: int = 50
    cplex_threads: int = 1


def _iter_csr_row(A: sparse.csr_matrix, i: int):
    start, end = A.indptr[i], A.indptr[i + 1]
    for j, v in zip(A.indices[start:end], A.data[start:end]):
        yield int(j), float(v)


def _parse_objective_terms(obj_text: str) -> dict[int, float]:
    one_line = re.sub(r"\s+", " ", obj_text).strip()
    term_re = re.compile(r"([+-]?\s*\d*\.?\d*)\s*x(\d+)", re.IGNORECASE)
    out: dict[int, float] = {}
    for coef_s, var_s in term_re.findall(one_line):
        coef_t = coef_s.replace(" ", "")
        if coef_t in ("", "+"):
            coef = 1.0
        elif coef_t == "-":
            coef = -1.0
        else:
            coef = float(coef_t)
        j = int(var_s) - 1
        out[j] = out.get(j, 0.0) + coef
    return out


def load_spp_lp_instance(instance_path: str | Path) -> SPPProblemInstance:
    path = Path(instance_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    obj_match = re.search(r"(?is)Maximize\s*(.*?)\s*Subject To", text)
    if not obj_match:
        raise ValueError(f"Could not parse objective from {path}")
    obj_terms = _parse_objective_terms(obj_match.group(1))

    st_match = re.search(r"(?is)Subject To\s*(.*?)(?=^\s*(?:Binary|Bounds|End)\s*$)", text, re.MULTILINE)
    if not st_match:
        raise ValueError(f"Could not parse Subject To block from {path}")
    st_body = st_match.group(1)

    rows: list[list[int]] = []
    rhs_vals: list[float] = []
    max_x = max(obj_terms.keys(), default=-1) + 1
    for raw in st_body.splitlines():
        line = raw.strip()
        if not line or ":" not in line or "<=" not in line:
            continue
        _, body = line.split(":", 1)
        lhs, rhs_s = body.rsplit("<=", 1)
        rhs_vals.append(float(rhs_s.strip()))
        cols = [int(v) - 1 for v in re.findall(r"x(\d+)", lhs, flags=re.IGNORECASE)]
        if cols:
            max_x = max(max_x, max(cols) + 1)
        rows.append(cols)

    if max_x <= 0:
        raise ValueError(f"No variables found in {path}")

    m = len(rows)
    n = max_x
    data, r_idx, c_idx = [], [], []
    for i, cols in enumerate(rows):
        for j in cols:
            data.append(1.0)
            r_idx.append(i)
            c_idx.append(j)
    A = sparse.coo_matrix((data, (r_idx, c_idx)), shape=(m, n), dtype=float).tocsr()
    b = np.asarray(rhs_vals, dtype=float)

    profits = np.ones(n, dtype=float)
    for j, c in obj_terms.items():
        if 0 <= j < n:
            profits[j] = c

    return SPPProblemInstance(
        instance_path=str(path),
        A=A,
        b=b,
        profits=profits,
        m=m,
        n=n,
        integer_indices=list(range(n)),
    )


def load_spp_npz_instance(instance_path: str | Path) -> SPPProblemInstance:
    path = Path(instance_path)
    arc = np.load(path, allow_pickle=False)
    if "A" not in arc.files or "b" not in arc.files:
        raise KeyError(f"{path} must contain at least A and b arrays")
    A = sparse.csr_matrix(np.asarray(arc["A"], dtype=float))
    b = np.asarray(arc["b"], dtype=float).reshape(-1)
    n = int(A.shape[1])
    profits = np.ones(n, dtype=float)
    if "c" in arc.files:
        c = np.asarray(arc["c"], dtype=float)
        if c.ndim == 2 and c.shape[1] == n:
            profits = c[0].copy()
    return SPPProblemInstance(
        instance_path=str(path),
        A=A,
        b=b,
        profits=profits,
        m=int(A.shape[0]),
        n=n,
        integer_indices=list(range(n)),
    )


def load_spp_instance(instance_path: str | Path) -> SPPProblemInstance:
    path = Path(instance_path)
    suf = path.suffix.lower()
    if suf == ".lp":
        return load_spp_lp_instance(path)
    if suf == ".npz":
        return load_spp_npz_instance(path)
    raise ValueError(f"Unsupported extension {path.suffix!r} for {path}")


def fp_distance(values: Sequence[float], rounded_values: Sequence[float], integer_indices: Sequence[int]) -> float:
    return float(sum(abs(values[idx] - rounded_values[idx]) for idx in integer_indices))


def round_integer_values(values: Sequence[float], integer_indices: Sequence[int]) -> list[float]:
    out = list(values)
    for idx in integer_indices:
        out[idx] = float(round(values[idx]))
    return out


def is_integer_solution(values: Sequence[float], integer_indices: Sequence[int], tol: float = INTEGER_TOLERANCE) -> bool:
    for idx in integer_indices:
        if abs(values[idx] - round(values[idx])) > tol:
            return False
    return True


class SetPackingFPCore:
    def __init__(self, problem: SPPProblemInstance, config: SPFPRunConfig):
        self.problem = problem
        self.config = config

        self.relaxation_model = None
        self.relaxation_x = None
        self.distance_model = None
        self.distance_z = None

        self.start_time: Optional[float] = None
        self.iteration = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.consecutive_no_change = 0
        self.stall_events = 0
        self.total_flips = 0
        self.perturbation_events = 0
        self.last_k = 0
        self.last_flip_indices: list[int] = []

        self.initial_lp_objective = 0.0
        self.initial_solution_was_integer = False
        self.terminated_in_initial_relaxation = False
        self.initial_distance = 0.0
        self.relaxation_build_seconds = 0.0
        self.distance_build_seconds = 0.0
        self.initial_lp_solve_seconds = 0.0
        self.reset_seconds = 0.0

        self.x_relaxed: Optional[list[float]] = None
        self.x_rounded: Optional[list[float]] = None
        self._cached_lp_result = None

    def remaining_time(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return self.config.time_limit - (time.time() - self.start_time)

    def current_distance(self) -> float:
        if self.integer_found or self.x_relaxed is None or self.x_rounded is None:
            return 0.0
        return fp_distance(self.x_relaxed, self.x_rounded, self.problem.integer_indices)

    def build_models(self) -> None:
        t0 = time.time()
        rm = Model(name="spp_relaxation")
        rm.context.cplex_parameters.threads = self.config.cplex_threads
        x = rm.continuous_var_list(self.problem.n, lb=0.0, ub=1.0, name="x")
        for i in range(self.problem.m):
            rm.add_constraint(rm.sum(v * x[j] for j, v in _iter_csr_row(self.problem.A, i)) <= float(self.problem.b[i]))
        rm.maximize(rm.sum(float(self.problem.profits[j]) * x[j] for j in range(self.problem.n)))
        self.relaxation_model, self.relaxation_x = rm, x
        self.relaxation_build_seconds = time.time() - t0

        t0 = time.time()
        dm = Model(name="spp_distance")
        dm.context.cplex_parameters.threads = self.config.cplex_threads
        z = dm.continuous_var_list(self.problem.n, lb=0.0, ub=1.0, name="z")
        for i in range(self.problem.m):
            dm.add_constraint(dm.sum(v * z[j] for j, v in _iter_csr_row(self.problem.A, i)) <= float(self.problem.b[i]))
        self.distance_model, self.distance_z = dm, z
        self.distance_build_seconds = time.time() - t0

        t0 = time.time()
        if self.config.initial_lp_time_limit is not None:
            self.relaxation_model.set_time_limit(max(0.01, float(self.config.initial_lp_time_limit)))
        sol = self.relaxation_model.solve(log_output=False)
        self.initial_lp_solve_seconds = time.time() - t0
        if sol is None:
            self._cached_lp_result = None
            return
        x_relaxed = [float(sol.get_value(self.relaxation_x[j])) for j in range(self.problem.n)]
        self._cached_lp_result = (x_relaxed, float(self.relaxation_model.objective_value))

    def reset_state(self) -> None:
        t0 = time.time()
        self.iteration = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.consecutive_no_change = 0
        self.stall_events = 0
        self.total_flips = 0
        self.perturbation_events = 0
        self.last_k = 0
        self.last_flip_indices = []
        self.initial_solution_was_integer = False
        self.terminated_in_initial_relaxation = False
        self.initial_distance = 0.0
        self.x_relaxed = None
        self.x_rounded = None
        self.start_time = None

        if self._cached_lp_result is None:
            self.done = True
            self.failed = True
            self.reset_seconds = time.time() - t0
            return

        x_relaxed, obj = self._cached_lp_result
        self.x_relaxed = list(x_relaxed)
        self.initial_lp_objective = float(obj)
        self.x_rounded = round_integer_values(self.x_relaxed, self.problem.integer_indices)
        self.initial_distance = fp_distance(self.x_relaxed, self.x_rounded, self.problem.integer_indices)
        self.initial_solution_was_integer = is_integer_solution(self.x_relaxed, self.problem.integer_indices)
        if self.initial_solution_was_integer:
            self.integer_found = True
            self.done = True
            self.terminated_in_initial_relaxation = True
        else:
            self.start_time = time.time()
        self.reset_seconds = time.time() - t0

    def reset(self) -> None:
        self.build_models()
        self.reset_state()

    def is_stalled(self) -> bool:
        return self.consecutive_no_change >= self.config.stall_threshold

    def _select_flip_indices(self, k: int) -> list[int]:
        if self.x_relaxed is None:
            return []
        idxs = self.problem.integer_indices
        dist = np.array([abs(self.x_relaxed[i] - self.x_rounded[i]) for i in idxs], dtype=float)
        if len(dist) == 0:
            return []
        top = np.argsort(-dist)[: max(1, min(k, len(dist)))]
        return [idxs[int(t)] for t in top]

    def apply_flip_count(self, k: int) -> None:
        if self.done or self.x_rounded is None:
            return
        sel = self._select_flip_indices(k)
        for idx in sel:
            self.x_rounded[idx] = 1.0 - self.x_rounded[idx]
        self.last_flip_indices = sel
        self.last_k = len(sel)
        self.total_flips += len(sel)
        if sel:
            self.perturbation_events += 1
        self.consecutive_no_change = 0

    def run_one_iteration(self, flip_indices: Sequence[int]) -> bool:
        if self.done or self.x_rounded is None:
            return False
        if self.iteration >= self.config.max_iterations:
            self.done = True
            return False
        rem = self.remaining_time()
        if rem is not None and rem <= 0:
            self.done = True
            return False

        if flip_indices:
            for idx in flip_indices:
                self.x_rounded[idx] = 1.0 - self.x_rounded[idx]
            self.last_flip_indices = list(flip_indices)
            self.last_k = len(flip_indices)
            self.total_flips += len(flip_indices)
            self.perturbation_events += 1

        obj = self.distance_model.sum(
            self.distance_z[j] if self.x_rounded[j] < 0.5 else (1 - self.distance_z[j])
            for j in self.problem.integer_indices
        )
        self.distance_model.minimize(obj)
        if rem is not None:
            self.distance_model.set_time_limit(max(0.01, rem))
        sol = self.distance_model.solve(log_output=False)
        if sol is None:
            self.done = True
            self.failed = True
            return False

        prev = list(self.x_rounded)
        self.x_relaxed = [float(sol.get_value(self.distance_z[j])) for j in range(self.problem.n)]
        if is_integer_solution(self.x_relaxed, self.problem.integer_indices):
            self.x_rounded = round_integer_values(self.x_relaxed, self.problem.integer_indices)
            self.integer_found = True
            self.done = True
            self.iteration += 1
            return True

        new_round = round_integer_values(self.x_relaxed, self.problem.integer_indices)
        changed = any(new_round[i] != prev[i] for i in self.problem.integer_indices)
        self.x_rounded = new_round
        self.consecutive_no_change = 0 if changed else (self.consecutive_no_change + 1)
        if self.is_stalled():
            self.stall_events += 1
            if self.stall_events >= self.config.max_stalls:
                self.done = True

        self.iteration += 1
        return True

    def advance_until_stall_or_done(self, max_steps: Optional[int] = None) -> None:
        steps = 0
        while not self.done:
            if max_steps is not None and steps >= max_steps:
                break
            moved = self.run_one_iteration([])
            if not moved:
                break
            steps += 1
            if self.is_stalled():
                break
