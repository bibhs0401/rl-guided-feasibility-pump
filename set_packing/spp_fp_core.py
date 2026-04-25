from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
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
    profits: np.ndarray          # = c[0] for backward compat
    m: int
    n: int
    integer_indices: list[int]
    c: list = field(default_factory=list)   # list of p np.ndarray objective vectors
    d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))  # shape (p,)
    p: int = 1                              # number of objectives


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

    obj_match = re.search(r"(?is)\b(Maximize|Minimize)\b(.*?)\bSubject\s+To\b", text)
    if not obj_match:
        raise ValueError(f"Could not parse objective from {path}")
    obj_raw_text = obj_match.group(2)

    # Detect multi-obj (y variables in objective) vs single-obj (x variables)
    has_y_obj = bool(re.search(r"\by\d+\b", obj_raw_text, re.IGNORECASE))

    st_match = re.search(
        r"(?is)\bSubject\s+To\b(.*?)(?=^\s*(?:Binary|Binaries|Bounds|Generals|End)\b)",
        text, re.MULTILINE,
    )
    if not st_match:
        raise ValueError(f"Could not parse Subject To block from {path}")
    st_body = st_match.group(1)

    le_rows_cols: list[list[int]] = []
    le_rhs: list[float] = []
    eq_rows: dict[int, dict[int, float]] = {}   # k -> {col: coef}
    eq_d: dict[int, float] = {}                  # k -> d_k

    for raw in st_body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            _, line = line.split(":", 1)
            line = line.strip()
        if "<=" in line:
            lhs, rhs_s = line.rsplit("<=", 1)
            cols = [int(v) - 1 for v in re.findall(r"x(\d+)", lhs, re.IGNORECASE)]
            le_rows_cols.append(cols)
            le_rhs.append(float(rhs_s.strip()))
        elif "=" in line:
            lhs, rhs_s = line.rsplit("=", 1)
            y_m = re.findall(r"([+-]?\s*\d*\.?\d*)\s*y(\d+)", lhs, re.IGNORECASE)
            if y_m:
                ki = int(y_m[0][1]) - 1
                x_terms: dict[int, float] = {}
                for coef_s, xidx_s in re.findall(
                    r"([+-]?\s*\d*\.?\d*)\s*x(\d+)", lhs, re.IGNORECASE
                ):
                    tok = coef_s.replace(" ", "")
                    coef = 1.0 if tok in ("", "+") else (-1.0 if tok == "-" else float(tok))
                    x_terms[int(xidx_s) - 1] = coef
                eq_rows[ki] = x_terms
                eq_d[ki] = -float(rhs_s.strip())

    all_cols: list[int] = [j for cols in le_rows_cols for j in cols]
    for row in eq_rows.values():
        all_cols.extend(row.keys())
    if not has_y_obj:
        all_cols.extend(int(v) - 1 for v in re.findall(r"x(\d+)", obj_raw_text, re.IGNORECASE))
    n = max(all_cols, default=0) + 1
    m = len(le_rows_cols)

    data_v, r_idx, c_idx_v = [], [], []
    for i, cols in enumerate(le_rows_cols):
        for j in cols:
            data_v.append(1.0); r_idx.append(i); c_idx_v.append(j)
    A = sparse.coo_matrix((data_v, (r_idx, c_idx_v)), shape=(m, n), dtype=float).tocsr()
    b = np.asarray(le_rhs, dtype=float)

    if has_y_obj and eq_rows:
        num_obj = max(eq_rows.keys()) + 1
        c_list: list[np.ndarray] = []
        d_arr = np.zeros(num_obj, dtype=float)
        for k in range(num_obj):
            ck = np.zeros(n, dtype=float)
            if k in eq_rows:
                for j, v in eq_rows[k].items():
                    if 0 <= j < n:
                        ck[j] = v
                d_arr[k] = eq_d.get(k, 0.0)
            c_list.append(ck)
        profits = c_list[0].copy()
        p_val = num_obj
    else:
        profits = np.ones(n, dtype=float)
        obj_terms = _parse_objective_terms(obj_raw_text)
        for j, cv in obj_terms.items():
            if 0 <= j < n:
                profits[j] = cv
        c_list = [profits.copy()]
        d_arr = np.zeros(1, dtype=float)
        p_val = 1

    return SPPProblemInstance(
        instance_path=str(path),
        A=A, b=b, profits=profits, m=m, n=n,
        integer_indices=list(range(n)),
        c=c_list, d=d_arr, p=p_val,
    )


def load_spp_npz_instance(instance_path: str | Path) -> SPPProblemInstance:
    path = Path(instance_path)
    arc = np.load(path, allow_pickle=False)
    if "A" not in arc.files or "b" not in arc.files:
        raise KeyError(f"{path} must contain at least A and b arrays")
    A = sparse.csr_matrix(np.asarray(arc["A"], dtype=float))
    b = np.asarray(arc["b"], dtype=float).reshape(-1)
    n = int(A.shape[1])

    c_list: list[np.ndarray] = []
    d_arr = np.zeros(1, dtype=float)
    if "profits" in arc.files:
        profits = np.asarray(arc["profits"], dtype=float).reshape(-1)
        c_list = [profits.copy()]
    elif "c" in arc.files:
        c_raw = np.asarray(arc["c"], dtype=float)
        if c_raw.ndim == 1:
            c_list = [c_raw.reshape(-1)]
        elif c_raw.ndim == 2 and c_raw.shape[1] == n:
            c_list = [c_raw[k].copy() for k in range(c_raw.shape[0])]
        else:
            c_list = [np.ones(n, dtype=float)]
        if "d" in arc.files:
            d_raw = np.asarray(arc["d"], dtype=float).reshape(-1)
            if d_raw.shape[0] < len(c_list):
                d_arr = np.concatenate([d_raw, np.zeros(len(c_list) - d_raw.shape[0])])
            else:
                d_arr = d_raw[: len(c_list)]
        else:
            d_arr = np.zeros(len(c_list), dtype=float)
    else:
        c_list = [np.ones(n, dtype=float)]

    profits = c_list[0]
    return SPPProblemInstance(
        instance_path=str(path),
        A=A, b=b, profits=profits,
        m=int(A.shape[0]), n=n,
        integer_indices=list(range(n)),
        c=c_list, d=d_arr, p=len(c_list),
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
        self.y_values: Optional[list[float]] = None   # per-objective values (p,)
        self._cached_lp_result = None
        # Placeholders set by build_models(); kept here for type-checker clarity
        self.relaxation_y = None
        self.distance_y = None

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
        if self.problem.p > 1 and len(self.problem.c) == self.problem.p:
            ry = rm.continuous_var_list(self.problem.p, name="y")
            for k in range(self.problem.p):
                ck = self.problem.c[k]
                rm.add_constraint(
                    rm.sum(float(ck[j]) * x[j] for j in range(self.problem.n) if ck[j] != 0.0)
                    + float(self.problem.d[k]) == ry[k]
                )
            rm.maximize(rm.sum(ry))
            self.relaxation_y = ry
        else:
            rm.maximize(rm.sum(float(self.problem.profits[j]) * x[j] for j in range(self.problem.n)))
            self.relaxation_y = None
        self.relaxation_model, self.relaxation_x = rm, x
        self.relaxation_build_seconds = time.time() - t0

        t0 = time.time()
        dm = Model(name="spp_distance")
        dm.context.cplex_parameters.threads = self.config.cplex_threads
        z = dm.continuous_var_list(self.problem.n, lb=0.0, ub=1.0, name="z")
        for i in range(self.problem.m):
            dm.add_constraint(dm.sum(v * z[j] for j, v in _iter_csr_row(self.problem.A, i)) <= float(self.problem.b[i]))
        if self.problem.p > 1 and len(self.problem.c) == self.problem.p:
            dy = dm.continuous_var_list(self.problem.p, name="y")
            for k in range(self.problem.p):
                ck = self.problem.c[k]
                dm.add_constraint(
                    dm.sum(float(ck[j]) * z[j] for j in range(self.problem.n) if ck[j] != 0.0)
                    + float(self.problem.d[k]) == dy[k]
                )
            self.distance_y = dy
        else:
            self.distance_y = None
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
        y_relaxed = (
            [float(sol.get_value(self.relaxation_y[k])) for k in range(self.problem.p)]
            if self.relaxation_y is not None else None
        )
        self._cached_lp_result = (x_relaxed, float(self.relaxation_model.objective_value), y_relaxed)

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

        cached = self._cached_lp_result
        x_relaxed, obj = cached[0], cached[1]
        y_relaxed = cached[2] if len(cached) > 2 else None
        self.x_relaxed = list(x_relaxed)
        self.initial_lp_objective = float(obj)
        self.y_values = list(y_relaxed) if y_relaxed is not None else None
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
        if self.distance_y is not None:
            self.y_values = [float(sol.get_value(self.distance_y[k])) for k in range(self.problem.p)]
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
