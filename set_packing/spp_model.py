from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


DEFAULT_TOLERANCE = 1e-6

@dataclass
class SPPInstance:
    name: str
    path: str
    A: sparse.csr_matrix
    b: np.ndarray
    profits: np.ndarray          # first objective vector (c[0]); kept for backward compat
    c: list = field(default_factory=list)   # list of p np.ndarray objective vectors
    d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))  # shape (p,)
    p: int = 1                   # number of objectives

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])


@dataclass
class FeasibilityMetrics:
    is_binary: bool
    is_feasible: bool
    total_violation: float
    num_violated_constraints: int
    max_violation: float
    binary_residual: float


@dataclass
class RepairInfo:
    applied: bool
    removed_indices: list[int]
    initial_total_violation: float
    final_total_violation: float
    final_num_violated_constraints: int


def _as_csr(A: np.ndarray | sparse.spmatrix) -> sparse.csr_matrix:
    if sparse.issparse(A):
        return A.astype(float).tocsr()
    return sparse.csr_matrix(np.asarray(A, dtype=float))


def _parse_linear_terms(text: str) -> dict[int, float]:
    """Parse x-variable terms only; returns {0-based col index: coef}."""
    one_line = re.sub(r"\s+", " ", text).strip()
    term_re = re.compile(r"([+-]?\s*\d*\.?\d*)\s*x(\d+)", re.IGNORECASE)
    terms: dict[int, float] = {}
    for coef_s, var_s in term_re.findall(one_line):
        token = coef_s.replace(" ", "")
        if token in ("", "+"):
            coef = 1.0
        elif token == "-":
            coef = -1.0
        else:
            coef = float(token)
        j = int(var_s) - 1
        terms[j] = terms.get(j, 0.0) + coef
    return terms


def _parse_xy_terms(text: str) -> dict[str, float]:
    """Parse x and y variable terms; returns {'x3': coef, 'y1': coef, ...}."""
    one_line = re.sub(r"\s+", " ", text).strip()
    term_re = re.compile(r"([+-]?\s*\d*\.?\d*)\s*([xy]\d+)", re.IGNORECASE)
    out: dict[str, float] = {}
    for coef_s, vname in term_re.findall(one_line):
        token = coef_s.replace(" ", "")
        if token in ("", "+"):
            coef = 1.0
        elif token == "-":
            coef = -1.0
        else:
            coef = float(token)
        key = vname.lower()
        out[key] = out.get(key, 0.0) + coef
    return out


def load_spp_lp_instance(path: str | Path) -> SPPInstance:
    """Load a set-packing LP instance.

    Handles two formats:
    - *Single-objective* (legacy): ``Maximize obj: x1 + x2 + ...`` with only
      ``<=`` rows in Subject To.
    - *Multi-objective*: ``Maximize obj: y1 + y2 + ...`` with both ``<=`` rows
      (set-packing) and ``=`` rows of the form ``c_k x − y_k = −d_k``.
    """
    lp_path = Path(path)
    text = lp_path.read_text(encoding="utf-8", errors="replace")

    obj_match = re.search(r"(?is)\b(Maximize|Minimize)\b(.*?)\bSubject\s+To\b", text)
    if not obj_match:
        raise ValueError(f"Could not find a Maximize or Minimize block in {lp_path}")
    objective_sense = obj_match.group(1).lower()
    obj_raw = _parse_xy_terms(obj_match.group(2))
    # Detect whether objective is over y variables (multi-obj) or x variables (single-obj)
    is_multi_obj = any(k.startswith("y") for k in obj_raw)

    st_match = re.search(
        r"(?is)\bSubject\s+To\b(.*?)(?=^\s*(?:Binary|Binaries|Bounds|Generals|End)\b)",
        text,
        re.MULTILINE,
    )
    if not st_match:
        raise ValueError(f"Could not find a Subject To block in {lp_path}")

    # Parse <= rows (Ax <= b) and = rows (c_k x - y_k = -d_k)
    le_rows: list[dict[int, float]] = []
    le_rhs: list[float] = []
    eq_rows: dict[int, dict[int, float]] = {}   # k (0-based) -> {col: coef}
    eq_d: dict[int, float] = {}                  # k -> d_k

    for raw_line in st_match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            _, line = line.split(":", 1)
            line = line.strip()
        if "<=" in line:
            lhs, rhs_s = line.rsplit("<=", 1)
            x_terms = _parse_linear_terms(lhs)
            if x_terms:
                le_rows.append(x_terms)
                le_rhs.append(float(rhs_s.strip()))
        elif "=" in line:
            lhs, rhs_s = line.rsplit("=", 1)
            mixed = _parse_xy_terms(lhs)
            y_vars = {k: v for k, v in mixed.items() if k.startswith("y")}
            x_vars = {}
            for k, v in mixed.items():
                m_x = re.match(r"^x(\d+)$", k)
                if m_x:
                    x_vars[int(m_x.group(1)) - 1] = v
            if y_vars and x_vars:
                for yname in y_vars:
                    m_y = re.match(r"^y(\d+)$", yname)
                    if m_y:
                        ki = int(m_y.group(1)) - 1
                        eq_rows[ki] = x_vars
                        # constraint is c_k x - y_k = -d_k  =>  d_k = -rhs
                        eq_d[ki] = -float(rhs_s.strip())

    if not le_rows:
        raise ValueError(f"No Ax <= b rows were parsed from {lp_path}")

    # Determine n from all parsed column indices
    all_cols: list[int] = []
    for row in le_rows:
        all_cols.extend(row.keys())
    for row in eq_rows.values():
        all_cols.extend(row.keys())
    if not is_multi_obj:
        x_obj_terms = {int(re.match(r"^x(\d+)$", k).group(1)) - 1: v
                       for k, v in obj_raw.items() if re.match(r"^x(\d+)$", k)}
        all_cols.extend(x_obj_terms.keys())
    n = max(all_cols, default=0) + 1

    # Build A (sparse) and b
    r_idx, c_idx, vals = [], [], []
    for i, row in enumerate(le_rows):
        for j, v in row.items():
            r_idx.append(i); c_idx.append(j); vals.append(float(v))
    A = sparse.coo_matrix((vals, (r_idx, c_idx)), shape=(len(le_rows), n), dtype=float).tocsr()
    b = np.asarray(le_rhs, dtype=float)

    # Build objective arrays
    if is_multi_obj and eq_rows:
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
        # Single-objective: profits from objective line
        profits = np.ones(n, dtype=float)
        x_obj_terms = {int(re.match(r"^x(\d+)$", k).group(1)) - 1: v
                       for k, v in obj_raw.items() if re.match(r"^x(\d+)$", k)}
        for j, value in x_obj_terms.items():
            if 0 <= j < n:
                profits[j] = -float(value) if objective_sense == "minimize" else float(value)
        c_list = [profits.copy()]
        d_arr = np.zeros(1, dtype=float)
        p_val = 1

    return SPPInstance(lp_path.name, str(lp_path), A, b, profits, c=c_list, d=d_arr, p=p_val)


def load_spp_npz_instance(path: str | Path) -> SPPInstance:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as arc:
        if "A" not in arc.files:
            raise KeyError(f"{npz_path} does not contain an A matrix")
        A = _as_csr(arc["A"])
        n = int(A.shape[1])
        b = np.asarray(arc["b"], dtype=float).reshape(-1) if "b" in arc.files else np.ones(A.shape[0])

        # Build full c list and d array
        c_list: list[np.ndarray] = []
        d_arr: np.ndarray = np.zeros(1, dtype=float)
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
                d_arr = np.asarray(arc["d"], dtype=float).reshape(-1)
                # Pad / trim d to match number of objectives
                if d_arr.shape[0] < len(c_list):
                    d_arr = np.concatenate([d_arr, np.zeros(len(c_list) - d_arr.shape[0])])
                elif d_arr.shape[0] > len(c_list):
                    d_arr = d_arr[: len(c_list)]
            else:
                d_arr = np.zeros(len(c_list), dtype=float)
        else:
            c_list = [np.ones(n, dtype=float)]

        profits = c_list[0]

    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b has length {b.shape[0]}, but A has {A.shape[0]} rows in {npz_path}")
    if profits.shape[0] != n:
        raise ValueError(f"profits has length {profits.shape[0]}, but A has {n} columns in {npz_path}")

    return SPPInstance(
        npz_path.name, str(npz_path), A, b, profits,
        c=c_list, d=d_arr, p=len(c_list),
    )


def load_spp_instance(path: str | Path) -> SPPInstance:
    instance_path = Path(path)
    suffix = instance_path.suffix.lower()
    if suffix == ".npz":
        return load_spp_npz_instance(instance_path)
    if suffix == ".lp":
        return load_spp_lp_instance(instance_path)
    raise ValueError(f"Unsupported set-packing instance format: {instance_path}")


def validate_set_packing_instance(instance: SPPInstance, tol: float = DEFAULT_TOLERANCE) -> list[str]:
    warnings: list[str] = []
    A = instance.A
    if A.shape != (instance.m, instance.n):
        warnings.append("A shape is inconsistent with instance dimensions.")
    if np.any(instance.b < -tol):
        warnings.append("Some RHS values are negative; zero solution may not be feasible.")
    if not np.allclose(instance.b, 1.0, atol=tol):
        warnings.append("RHS is not exactly all ones; code will still use Ax <= b.")
    if A.nnz and np.min(A.data) < -tol:
        warnings.append("A contains negative coefficients; this is not standard set packing.")
    if A.nnz and not np.allclose(A.data, np.round(A.data), atol=tol):
        warnings.append("A contains non-integer coefficients; expected 0/1 set-packing rows.")
    if A.nnz and np.max(A.data) > 1.0 + tol:
        warnings.append("A contains coefficients larger than one; expected 0/1 set-packing rows.")
    return warnings


def find_instance_files(
    roots: Sequence[str | Path],
    patterns: Sequence[str] = ("*.npz", "*.lp"),
    recursive: bool = True,
) -> list[str]:
    found: list[Path] = []
    for root in roots:
        base = Path(root)
        if base.is_file() and base.suffix.lower() in {".npz", ".lp"}:
            found.append(base)
        elif base.is_dir():
            for pattern in patterns:
                iterator = base.rglob(pattern) if recursive else base.glob(pattern)
                found.extend(p for p in iterator if p.is_file())
    return [str(p.resolve()) for p in sorted(set(found))]


def round_binary(values: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr >= threshold).astype(float)


def objective_value(instance: SPPInstance, x: Sequence[float]) -> float:
    """Scalarized objective: sum of all p individual objective values y_k = c_k·x + d_k."""
    x_arr = np.asarray(x, dtype=float)
    if instance.p > 1 and len(instance.c) == instance.p:
        return float(sum(
            float(np.dot(instance.c[k], x_arr)) + float(instance.d[k])
            for k in range(instance.p)
        ))
    return float(np.dot(instance.profits, x_arr))


def objective_values_per_obj(instance: SPPInstance, x: Sequence[float]) -> np.ndarray:
    """Return a (p,) array of individual objective values y_k = c_k·x + d_k."""
    x_arr = np.asarray(x, dtype=float)
    if instance.p > 1 and len(instance.c) == instance.p:
        return np.array([
            float(np.dot(instance.c[k], x_arr)) + float(instance.d[k])
            for k in range(instance.p)
        ], dtype=float)
    return np.array([float(np.dot(instance.profits, x_arr))], dtype=float)


def feasibility_metrics(
    instance: SPPInstance,
    x: Sequence[float],
    tol: float = DEFAULT_TOLERANCE,
) -> FeasibilityMetrics:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    if x_arr.shape[0] != instance.n:
        raise ValueError(f"x has length {x_arr.shape[0]}, expected {instance.n}")
    rounded = np.round(x_arr)
    binary_residual = float(np.max(np.abs(x_arr - rounded))) if x_arr.size else 0.0
    is_binary = bool(binary_residual <= tol and np.all((rounded >= -tol) & (rounded <= 1.0 + tol)))
    activity = instance.A @ x_arr
    violation = np.maximum(activity - instance.b, 0.0)
    total_violation = float(np.sum(violation))
    num_violated = int(np.sum(violation > tol))
    max_violation = float(np.max(violation)) if violation.size else 0.0
    return FeasibilityMetrics(
        is_binary=is_binary,
        is_feasible=bool(is_binary and num_violated == 0),
        total_violation=total_violation,
        num_violated_constraints=num_violated,
        max_violation=max_violation,
        binary_residual=binary_residual,
    )


def is_feasible_binary(instance: SPPInstance, x: Sequence[float], tol: float = DEFAULT_TOLERANCE) -> bool:
    return feasibility_metrics(instance, x, tol).is_feasible


def violation_pattern(instance: SPPInstance, x: Sequence[float], tol: float = DEFAULT_TOLERANCE) -> tuple[int, ...]:
    activity = instance.A @ np.asarray(x, dtype=float)
    return tuple(int(i) for i in np.flatnonzero(activity > instance.b + tol))


def repair_set_packing_solution(
    instance: SPPInstance,
    x: Sequence[float],
    tol: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, RepairInfo]:
    repaired = round_binary(x)
    initial_metrics = feasibility_metrics(instance, repaired, tol)
    removed: list[int] = []

    if initial_metrics.num_violated_constraints == 0:
        return repaired, RepairInfo(False, removed, initial_metrics.total_violation, 0.0, 0)

    max_profit_scale = max(1.0, float(np.max(np.abs(instance.profits))) if instance.profits.size else 1.0)
    for _ in range(instance.n + 1):
        activity = instance.A @ repaired
        violation = np.maximum(activity - instance.b, 0.0)
        violated_rows = np.flatnonzero(violation > tol)
        if violated_rows.size == 0:
            break

        active_cols = np.flatnonzero(repaired > 0.5)
        if active_cols.size == 0:
            break

        conflict_counts = np.asarray(instance.A[violated_rows][:, active_cols].sum(axis=0)).reshape(-1)
        if np.max(conflict_counts) <= tol:
            j_remove = int(active_cols[np.argmin(instance.profits[active_cols])])
        else:
            profit_penalty = instance.profits[active_cols] / max_profit_scale
            remove_score = conflict_counts - 0.25 * profit_penalty
            j_remove = int(active_cols[int(np.argmax(remove_score))])
        repaired[j_remove] = 0.0
        removed.append(j_remove)

    final_metrics = feasibility_metrics(instance, repaired, tol)
    return repaired, RepairInfo(
        applied=True,
        removed_indices=removed,
        initial_total_violation=initial_metrics.total_violation,
        final_total_violation=final_metrics.total_violation,
        final_num_violated_constraints=final_metrics.num_violated_constraints,
    )


def generate_random_set_packing_instance(
    name: str,
    n: int = 80,
    m: int | None = None,
    density: float = 0.04,
    p: int = 1,
    seed: int = 0,
) -> SPPInstance:
    rng = np.random.default_rng(seed)
    if m is None:
        m = 2 * n
    density = float(np.clip(density, 1.0 / max(1, n), 0.5))
    A_dense = (rng.random((m, n)) < density).astype(float)
    for i in range(m):
        if not A_dense[i].any():
            A_dense[i, int(rng.integers(0, n))] = 1.0
    # First objective: uniform profits; additional objectives: random integer weights
    c_list: list[np.ndarray] = [rng.integers(1, 101, size=n).astype(float)]
    for _ in range(p - 1):
        c_list.append(rng.integers(1, 101, size=n).astype(float))
    profits = c_list[0]
    d_arr = np.zeros(p, dtype=float)
    return SPPInstance(
        name=name, path="",
        A=sparse.csr_matrix(A_dense), b=np.ones(m),
        profits=profits, c=c_list, d=d_arr, p=p,
    )


def save_npz_instance(instance: SPPInstance, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    c_mat = np.stack(instance.c, axis=0).astype(float) if instance.c else instance.profits.reshape(1, -1)
    np.savez_compressed(
        out,
        A=instance.A.toarray().astype(np.int8),
        b=instance.b.astype(float),
        profits=instance.profits.astype(float),
        c=c_mat,
        d=instance.d.astype(float),
    )
    return str(out.resolve())


def write_instance_list(paths: Iterable[str], out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return str(out.resolve())
