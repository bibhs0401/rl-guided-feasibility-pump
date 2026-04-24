from __future__ import annotations

import re
from dataclasses import dataclass
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
    profits: np.ndarray

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


def load_spp_lp_instance(path: str | Path) -> SPPInstance:
    lp_path = Path(path)
    text = lp_path.read_text(encoding="utf-8", errors="replace")

    obj_match = re.search(r"(?is)\b(Maximize|Minimize)\b(.*?)\bSubject\s+To\b", text)
    if not obj_match:
        raise ValueError(f"Could not find a Maximize or Minimize block in {lp_path}")
    objective_sense = obj_match.group(1).lower()
    objective_terms = _parse_linear_terms(obj_match.group(2))

    st_match = re.search(
        r"(?is)\bSubject\s+To\b(.*?)(?=^\s*(?:Binary|Binaries|Bounds|Generals|End)\b)",
        text,
        re.MULTILINE,
    )
    if not st_match:
        raise ValueError(f"Could not find a Subject To block in {lp_path}")

    rows: list[dict[int, float]] = []
    rhs_values: list[float] = []
    max_col = max(objective_terms.keys(), default=-1)
    for raw_line in st_match.group(1).splitlines():
        line = raw_line.strip()
        if not line or "<=" not in line:
            continue
        if ":" in line:
            _, line = line.split(":", 1)
        lhs, rhs_s = line.rsplit("<=", 1)
        terms = _parse_linear_terms(lhs)
        if not terms:
            continue
        max_col = max(max_col, max(terms.keys()))
        rows.append(terms)
        rhs_values.append(float(rhs_s.strip()))

    if not rows:
        raise ValueError(f"No Ax <= b rows were parsed from {lp_path}")

    n = max_col + 1
    r_idx: list[int] = []
    c_idx: list[int] = []
    data: list[float] = []
    for i, row in enumerate(rows):
        for j, value in row.items():
            r_idx.append(i)
            c_idx.append(j)
            data.append(float(value))
    A = sparse.coo_matrix((data, (r_idx, c_idx)), shape=(len(rows), n), dtype=float).tocsr()
    b = np.asarray(rhs_values, dtype=float)
    profits = np.ones(n, dtype=float)
    for j, value in objective_terms.items():
        if 0 <= j < n:
            profits[j] = -float(value) if objective_sense == "minimize" else float(value)

    return SPPInstance(lp_path.name, str(lp_path), A, b, profits)


def load_spp_npz_instance(path: str | Path) -> SPPInstance:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as arc:
        if "A" not in arc.files:
            raise KeyError(f"{npz_path} does not contain an A matrix")
        A = _as_csr(arc["A"])
        b = np.asarray(arc["b"], dtype=float).reshape(-1) if "b" in arc.files else np.ones(A.shape[0])
        profits = np.ones(A.shape[1], dtype=float)
        if "profits" in arc.files:
            profits = np.asarray(arc["profits"], dtype=float).reshape(-1)
        elif "c" in arc.files:
            c = np.asarray(arc["c"], dtype=float)
            if c.ndim == 1:
                profits = c.reshape(-1)
            elif c.ndim == 2 and c.shape[1] == A.shape[1]:
                profits = c[0].reshape(-1)
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b has length {b.shape[0]}, but A has {A.shape[0]} rows in {npz_path}")
    if profits.shape[0] != A.shape[1]:
        raise ValueError(
            f"profits has length {profits.shape[0]}, but A has {A.shape[1]} columns in {npz_path}"
        )
    return SPPInstance(npz_path.name, str(npz_path), A, b, profits)


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
    return float(np.dot(instance.profits, np.asarray(x, dtype=float)))


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
    profits = rng.integers(1, 101, size=n).astype(float)
    return SPPInstance(name=name, path="", A=sparse.csr_matrix(A_dense), b=np.ones(m), profits=profits)


def save_npz_instance(instance: SPPInstance, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        A=instance.A.toarray().astype(np.int8),
        b=instance.b.astype(float),
        profits=instance.profits.astype(float),
    )
    return str(out.resolve())


def write_instance_list(paths: Iterable[str], out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return str(out.resolve())
