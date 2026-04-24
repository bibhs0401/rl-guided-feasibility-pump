from __future__ import annotations

# Standard library imports
import logging
import pickle
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# Third-party imports
import numpy as np
from docplex.mp.model import Model
from scipy import sparse


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
# We keep logging simple here. Later, when we build training code, we can plug
# this into a richer logging pipeline.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# Optional: CPLEX low-level API for fast bulk constraint building.
# This module ships with every CPLEX installation (academic or commercial).
# If it is not importable for some reason, we fall back to the docplex
# row-by-row approach automatically.
try:
    import cplex as _cplex_module
    _FAST_CONSTRAINTS = True
except ImportError:
    _FAST_CONSTRAINTS = False
    logger.warning(
        "cplex low-level module not importable. "
        "Falling back to row-by-row constraint building (slow for large instances). "
        "Make sure the CPLEX installation directory is on PYTHONPATH."
    )


# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------
# INTEGER_VARIABLE_FRACTION:
#   For now, we follow the same assumption used in your current code:
#   the first 80% of variables are treated as integer/binary variables.
#
# INTEGER_TOLERANCE:
#   Small tolerance used when checking whether a relaxed value is effectively
#   integer. This avoids floating-point issues like 0.999999999 vs 1.0.
# -----------------------------------------------------------------------------
INTEGER_VARIABLE_FRACTION = 0.8
INTEGER_TOLERANCE = 1e-6


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
# ProblemInstance:
#   Holds one instance in matrix/vector form after reading a .npz or .lp file.
#
# FPRunConfig:
#   Holds the solver-facing FP configuration for one run.
# -----------------------------------------------------------------------------
@dataclass
class ProblemInstance:
    """
    One MMP instance loaded from a sparse .npz archive or a CPLEX .lp file.

    Fields
    ------
    instance_path : str
        Original file path of the instance.
    A : sparse.csr_matrix
        Constraint matrix in Ax <= b.
    b : np.ndarray
        Right-hand side vector.
    c : list[np.ndarray]
        Objective coefficient vectors. There are p such vectors, one per
        objective function.
    d : np.ndarray
        Constant terms for the p objective functions.
    m : int
        Number of constraints.
    n : int
        Number of decision variables.
    p : int
        Number of objectives.
    integer_indices : list[int]
        Indices of variables treated as integer/binary variables.
    """
    instance_path: str
    A: sparse.csr_matrix
    b: np.ndarray
    c: list[np.ndarray]
    d: np.ndarray
    m: int
    n: int
    p: int
    integer_indices: list[int]


@dataclass
class FPRunConfig:
    """
    Configuration for one real Feasibility Pump run.

    Notes
    -----
    time_limit:
        FP-loop time budget after the initial LP relaxation has already been
        solved. This is the quantity that should match the baseline semantics
        in main_phase1.py.

    initial_lp_time_limit:
        Wall-clock cap in seconds for the initial LP relaxation solve in
        build_models().  CPLEX may stop early with a feasible solution that is
        not proven optimal — sufficient as an FP starting point.  Use None to
        disable the cap and let the solver run to optimality (slow on large
        instances).
    """
    max_iterations: int = 100
    time_limit: float = 30.0
    initial_lp_time_limit: float | None = 200.0
    stall_threshold: int = 3
    max_stalls: int = 50
    recent_delta_window: int = 5
    cplex_threads: int = 1


def _initial_lp_cache_tag(initial_lp_time_limit: float | None) -> str:
    """
    Build a short, filesystem-safe token for disk LP caches.

    Distinguishes optimal (no limit) vs time-capped initial solves so cache
    files do not collide when initial_lp_time_limit changes.
    """
    if initial_lp_time_limit is None:
        return "ilp_opt"
    s = f"{float(initial_lp_time_limit):g}".replace(".", "p").replace("-", "m")
    return f"ilp_t{s}"


def initial_lp_disk_cache_path(
    instance_path: str | Path,
    m: int,
    n: int,
    initial_lp_time_limit: float | None,
) -> Path:
    """Path for the pickled initial-LP solution next to the instance .npz."""
    inst_path = Path(instance_path)
    tag = _initial_lp_cache_tag(initial_lp_time_limit)
    return inst_path.parent / f"{inst_path.stem}_m{m}_n{n}_{tag}.lp_cache.pkl"


def _legacy_lp_disk_cache_path(instance_path: str | Path, m: int, n: int) -> Path:
    """Pre-tag cache filename (same instance may have been cached before tags)."""
    inst_path = Path(instance_path)
    return inst_path.parent / f"{inst_path.stem}_m{m}_n{n}.lp_cache.pkl"


# -----------------------------------------------------------------------------
# Small helper functions
# -----------------------------------------------------------------------------
# These utility functions keep the rest of the code cleaner.
# -----------------------------------------------------------------------------
def _read_scalar(value) -> int:
    """
    Read a scalar value from a numpy-loaded archive entry.

    Some archive values may come as:
    - true scalars
    - size-1 arrays

    This helper makes both cases behave the same.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(float(arr.item()))
    return int(float(arr.reshape(-1)[0]))


def load_npz_instance(instance_path: str | Path) -> ProblemInstance:
    """
    Load one sparse MMP instance from .npz and return it as a ProblemInstance.

    Expected keys inside the archive:
        data, indices, indptr, shape, b, c, d, n, m, p

    Matrix format:
        A is reconstructed as a CSR sparse matrix from:
            data, indices, indptr, shape
    """
    path = Path(instance_path)

    # Basic file extension validation
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz instance file, got: {path}")

    archive = np.load(path, allow_pickle=False)

    # Check that all required arrays are present
    required = {"data", "indices", "indptr", "shape", "b", "c", "d", "n", "m", "p"}
    missing = sorted(required - set(archive.files))
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    # Read scalar metadata
    m = _read_scalar(archive["m"])
    n = _read_scalar(archive["n"])
    p = _read_scalar(archive["p"])

    # Rebuild sparse constraint matrix A in CSR format
    A = sparse.csr_matrix(
        (archive["data"], archive["indices"], archive["indptr"]),
        shape=tuple(archive["shape"]),
        dtype=float,
    )

    # Read vectors / matrices
    b = np.asarray(archive["b"], dtype=float).reshape(m)
    c_matrix = np.asarray(archive["c"], dtype=float).reshape(p, n)
    d = np.asarray(archive["d"], dtype=float).reshape(p)

    # Convert objective matrix into a list of objective vectors
    c = [c_matrix[i].copy() for i in range(p)]

    # Follow the same 80% integer-variable convention used in your current code
    integer_indices = list(range(int(INTEGER_VARIABLE_FRACTION * n)))

    return ProblemInstance(
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


# -----------------------------------------------------------------------------
# CPLEX .lp import (format produced by instance_generator_sparse.save_instance_lp)
# -----------------------------------------------------------------------------


def _lp_extract_subject_to_body(text: str) -> str:
    m = re.search(
        r"(?is)Subject To\s*(.*?)(?=^\s*(?:Bounds|Binary|End)\s*$)",
        text,
        re.MULTILINE,
    )
    if not m:
        raise ValueError("Could not find a Subject To / Bounds (or End) block in the .lp file.")
    return m.group(1).strip()


def _lp_iter_constraint_blocks(subject_body: str) -> list[str]:
    """
    Split the Subject To body into one string per named constraint, keeping
    generator-style line continuations.
    """
    lines = subject_body.splitlines()
    blocks: list[str] = []
    cur: list[str] = []
    for line in lines:
        if re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", line) and cur:
            blocks.append("\n".join(cur))
            cur = [line]
        else:
            if not cur and not line.strip():
                continue
            if not cur and line.strip() and re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", line
            ) is None:
                continue
            if cur or line.strip():
                cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _lp_normalize_unary_y(lhs: str) -> str:
    """Map CPLEX '- y1' to '-1.0 y1' (the generator does not print an explicit 1)."""
    s = re.sub(r"(?<![\d.])\s*-\s*y([0-9]+)", r" -1.0 y\1", lhs)
    s = re.sub(r"(?<![\d.])\s*\+\s*y([0-9]+)", r" 1.0 y\1", s)
    return s


def _lp_parse_linear_terms(lhs: str) -> dict[str, float]:
    """
    Parse a CPLEX-style linear expression into variable -> coefficient.
    Coefficients are always explicit in our generator, except for unary '- yk'.
    """
    lhs_1 = _lp_normalize_unary_y(lhs)
    one_line = re.sub(r"[\r\n\t]+", " ", lhs_1)
    one_line = re.sub(r"\s+", " ", one_line).strip()

    if not one_line or one_line in ("0", "0.0"):
        return {}

    # Term pattern: (signed float) (varname)
    term_re = re.compile(
        r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    out: dict[str, float] = {}
    for coef_s, vname in term_re.findall(one_line):
        c = float(coef_s)
        if vname not in out:
            out[vname] = 0.0
        out[vname] += c

    return out


def _lp_parse_one_constraint_block(block: str) -> tuple[str, str, float, dict[str, float]]:
    """
    Return (name, sense, rhs, terms) where sense is 'L' (<=) or 'E' (=).
    """
    m = re.match(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:(.*)\Z",
        block,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"Invalid constraint block (expected name: ...): {block!r}")
    cname, body = m.group(1), m.group(2)
    btxt = re.sub(r"[\n\r\t]+", " ", body)
    btxt = re.sub(r"\s+", " ", btxt).strip()

    if "<=" in btxt:
        lhs, rhs_s = btxt.rsplit("<=", 1)
        sense = "L"
    elif "=" in btxt:
        lhs, rhs_s = btxt.rsplit("=", 1)
        sense = "E"
    else:
        raise ValueError(f"Constraint has no <= or = : {btxt!r}")

    terms = _lp_parse_linear_terms(lhs)
    return cname, sense, float(rhs_s.strip()), terms


def load_lp_instance(instance_path: str | Path) -> ProblemInstance:
    """
    Load an instance from a CPLEX .lp file (same MMP layout as the .npz path).

    Supported layout matches instance_generator_sparse.save_instance_lp:
        Ax <= b on rows c1..cm; objective links obj_y1..obj_yp with
        c_k @ x - y_k = -d_k.
    """
    path = Path(instance_path)
    if path.suffix.lower() != ".lp":
        raise ValueError(f"Expected a .lp instance file, got: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    n_meta = m_meta = p_meta = None
    cm = re.search(r"\\\s*n\s*=\s*(\d+)\s*,\s*m\s*=\s*(\d+)\s*,\s*p\s*=\s*(\d+)", text)
    if cm:
        n_meta, m_meta, p_meta = int(cm.group(1)), int(cm.group(2)), int(cm.group(3))

    subject_body = _lp_extract_subject_to_body(text)
    blocks = _lp_iter_constraint_blocks(subject_body)
    if not blocks:
        raise ValueError(f"Empty Subject To block in: {path}")

    ax: dict[int, dict[str, float]] = {}
    bvec: dict[int, float] = {}
    oby: dict[int, dict[str, float]] = {}
    drhs: dict[int, float] = {}

    for b in blocks:
        name, sense, rhs, terms = _lp_parse_one_constraint_block(b)
        mc = re.match(r"^c(\d+)$", name, re.IGNORECASE)
        mo = re.match(r"^obj_y(\d+)$", name, re.IGNORECASE)
        if mc:
            idx = int(mc.group(1)) - 1
            if sense != "L":
                raise ValueError(f"Expected Ax<=b (<=) for {name!r} in {path}, got {sense!r}.")
            ax[idx] = terms
            bvec[idx] = rhs
        elif mo:
            k = int(mo.group(1)) - 1
            if sense != "E":
                raise ValueError(f"Expected linear equality for {name!r} in {path}, got {sense!r}.")
            yk = f"y{mo.group(1)}"
            if yk not in terms:
                raise ValueError(
                    f"Expected {yk} in {name!r} in {path}, got terms {sorted(terms.keys())!r}."
                )
            if abs(terms[yk] + 1.0) > 1e-4:
                raise ValueError(
                    f"Expected {yk} coeff -1.0 in {name!r} in {path}, got {terms[yk]!r}."
                )
            tcopy = {k0: v0 for k0, v0 in terms.items() if not k0.startswith("y")}
            oby[k] = tcopy
            # yk = tcopy @ x + d_k, row is tcopy @ x - yk = -d_k, so rhs = -d_k
            drhs[k] = -float(rhs)
        else:
            raise ValueError(
                f"Unrecognized constraint name {name!r} in {path} "
                f"(expected c1..cM and obj_y1..obj_yP)"
            )

    m_parsed, p_parsed = len(bvec), len(drhs)
    if m_meta is not None and m_parsed != m_meta:
        raise ValueError(
            f"Header says m={m_meta} but found {m_parsed} Ax<=b rows in: {path}"
        )
    if p_meta is not None and p_parsed != p_meta:
        raise ValueError(
            f"Header says p={p_meta} but found {p_parsed} obj_y* rows in: {path}"
        )
    m, p = m_parsed, p_parsed

    if set(range(m)) != set(bvec.keys()) or set(range(m)) != set(ax.keys()):
        raise ValueError(
            f"Missing or non-contiguous c1..c{m} rows when parsing: {path} "
            f"(found rows {sorted(bvec.keys())!r})"
        )
    if set(range(p)) != set(drhs.keys()) or set(range(p)) != set(oby.keys()):
        raise ValueError(
            f"Missing or non-contiguous obj_y1..obj_y{p} when parsing: {path} "
            f"(found {sorted(drhs.keys())!r})"
        )

    # Infer n as max x index present (names are x1..x{n})
    x_indices: list[int] = []
    for tmap in [ax[i] for i in range(m)] + [oby[i] for i in range(p)]:
        for vname, val in tmap.items():
            if vname.startswith("x"):
                if val != 0.0:
                    x_indices.append(int(vname[1:]))
    if n_meta is not None:
        n = n_meta
    else:
        if not x_indices:
            raise ValueError(f"Could not infer n: no x variables in constraints in: {path}")
        n = max(x_indices)
    n_max_seen = max(x_indices) if x_indices else 0
    if n < n_max_seen:
        raise ValueError(
            f"Header or inferred n={n} but constraints reference x{max(x_indices)} in: {path}"
        )

    b_arr = np.array([bvec[i] for i in range(m)], dtype=float)
    d = np.array([drhs[i] for i in range(p)], dtype=float)

    a_rows, a_cols, a_data = [], [], []
    c_rows, c_cols, c_data = [], [], []
    for i in range(m):
        tmap = ax[i]
        for vname, val in tmap.items():
            if vname.startswith("x"):
                j = int(vname[1:]) - 1
            else:
                raise ValueError(
                    f"Non-x variable {vname!r} in Ax<=b row c{i+1} of {path}"
                )
            if not (0 <= j < n):
                raise ValueError(f"Out-of-range {vname!r} (n={n}) in c{i+1} of {path}")
            if val != 0.0:
                a_rows.append(i)
                a_cols.append(j)
                a_data.append(float(val))
    for k in range(p):
        tmap = oby[k]
        for vname, val in tmap.items():
            if vname.startswith("x"):
                j = int(vname[1:]) - 1
            else:
                raise ValueError(
                    f"Non-x variable {vname!r} in obj_y row {k+1} of {path}"
                )
            if not (0 <= j < n):
                raise ValueError(f"Out-of-range {vname!r} (n={n}) in obj_y{k+1} of {path}")
            if val != 0.0:
                c_rows.append(k)
                c_cols.append(j)
                c_data.append(float(val))

    A = sparse.coo_matrix(
        (a_data, (a_rows, a_cols)), shape=(m, n), dtype=float
    ).tocsr()
    c_mat = sparse.coo_matrix(
        (c_data, (c_rows, c_cols)), shape=(p, n), dtype=float
    ).toarray()
    c = [c_mat[i].copy() for i in range(p)]

    integer_indices = list(range(int(INTEGER_VARIABLE_FRACTION * n)))

    return ProblemInstance(
        instance_path=str(path),
        A=A,
        b=b_arr,
        c=c,
        d=d,
        m=m,
        n=n,
        p=p,
        integer_indices=integer_indices,
    )


def load_instance(instance_path: str | Path) -> ProblemInstance:
    """
    Load a problem instance from .npz (sparse archive) or .lp (CPLEX LP).
    """
    path = Path(instance_path)
    suf = path.suffix.lower()
    if suf == ".npz":
        return load_npz_instance(path)
    if suf == ".lp":
        return load_lp_instance(path)
    raise ValueError(
        f"Unsupported instance file type {path.suffix!r} "
        f"(expected .npz or .lp): {path}"
    )


def apply_cplex_threads(model: Model, cplex_threads: int) -> None:
    """
    Set the number of CPLEX threads for a docplex model.

    Notes
    -----
    CPLEX uses:
        0 -> automatic thread selection
    """
    model.context.cplex_parameters.threads = max(0, int(cplex_threads))


def iter_csr_row(matrix: sparse.csr_matrix, row_index: int):
    """
    Yield (column_index, value) pairs for one row of a CSR sparse matrix.

    We use this so we only build model expressions over nonzero entries,
    which is much more efficient than iterating over all n columns.
    """
    start = matrix.indptr[row_index]
    end = matrix.indptr[row_index + 1]
    cols = matrix.indices[start:end]
    vals = matrix.data[start:end]
    for j, v in zip(cols, vals):
        yield int(j), float(v)


def iter_vector_nonzero(vector: Sequence[float]):
    """
    Yield (index, value) pairs for nonzero entries of a dense vector.

    This is useful for objective vectors c[k] so we only add nonzero terms
    into the docplex expression.
    """
    arr = np.asarray(vector).reshape(-1)
    nz = np.nonzero(arr)[0]
    for i in nz:
        yield int(i), float(arr[i])


# -----------------------------------------------------------------------------
# FP math helpers
# -----------------------------------------------------------------------------
# These functions implement the standard FP logic:
# - rounding integer variables
# - checking integrality
# - measuring distance between relaxed and rounded points
# - flipping chosen binary variables
# -----------------------------------------------------------------------------
def round_integer_values(values: Sequence[float], integer_indices: Sequence[int]) -> list[float]:
    """
    Round only the integer variables.

    Continuous variables are left unchanged.
    """
    rounded = list(values)
    for idx in integer_indices:
        rounded[idx] = round(rounded[idx])
    return rounded


def is_integer_solution(
    values: Sequence[float],
    integer_indices: Sequence[int],
    tolerance: float = INTEGER_TOLERANCE,
) -> bool:
    """
    Check whether all integer-designated variables are effectively integral.

    Returns True if every integer variable is within tolerance of a rounded value.
    """
    for idx in integer_indices:
        if abs(values[idx] - round(values[idx])) > tolerance:
            return False
    return True


def rounding_changed(
    values: Sequence[float],
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
) -> bool:
    """
    Check whether rounding the current relaxed point would change the current
    rounded point.

    This is important for stall detection:
    if rounding no longer changes anything, FP is usually considered stalled.
    """
    for idx in integer_indices:
        if round(values[idx]) != rounded_values[idx]:
            return True
    return False


def fp_distance(
    values: Sequence[float],
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
) -> float:
    """
    Compute the FP distance between:
    - current relaxed solution
    - current rounded/integer guide point

    We use the standard L1-like distance over integer variables only.
    """
    return float(sum(abs(values[idx] - rounded_values[idx]) for idx in integer_indices))


def flip_selected_variables(
    rounded_values: Sequence[float],
    selected_indices: Sequence[int],
) -> list[float]:
    """
    Flip the selected binary variables in the rounded point.

    0 -> 1
    1 -> 0

    This is the perturbation step in FP.
    """
    updated = list(rounded_values)
    for idx in selected_indices:
        updated[idx] = 1 - updated[idx]
    return updated


# -----------------------------------------------------------------------------
# Solver wrappers
# -----------------------------------------------------------------------------
# These keep the actual docplex solve calls small and reusable.
# -----------------------------------------------------------------------------
def solve_with_time_limit(model: Model, max_seconds: Optional[float]):
    """
    Solve a docplex model with an optional time limit.
    """
    if max_seconds is not None:
        model.set_time_limit(max(0.01, float(max_seconds)))
    return model.solve(log_output=False)


# -----------------------------------------------------------------------------
# Fast bulk constraint helpers
# -----------------------------------------------------------------------------
# These two helpers replace the slow Python-level row-by-row loops in the
# model builders below.  Both require the cplex low-level module.
#
# Why this is faster:
#   docplex builds a Python expression tree for every constraint and then
#   translates it into CPLEX API calls one at a time.  For a 9000-row sparse
#   matrix that means ~9000 individual Python -> C boundary crossings.
#   The CPLEX low-level API accepts the entire sparse matrix as a single list
#   of SparsePair objects and performs one bulk C-level call, which is
#   typically 20-50x faster for large instances.
#
# Safety note on mixing docplex and the low-level API:
#   Constraints added via cpx.linear_constraints.add() are invisible to
#   docplex's internal constraint registry, but that is fine here because
#   (a) we never ask docplex to remove or look up the Ax<=b rows, and
#   (b) the dynamic distance constraint in solve_distance_model() is added
#       and removed exclusively through docplex, so its internal IDs are
#       unaffected by the bulk-added rows.
# -----------------------------------------------------------------------------

def _add_Axb_bulk(model: Model, A: sparse.csr_matrix, b: np.ndarray) -> None:
    """
    Add all Ax <= b constraints to a docplex model in a single bulk call
    using the CPLEX low-level Python API.

    Precondition
    ------------
    The decision variables (x or z) must be the FIRST n columns of the model,
    so that A's CSR column indices (0 .. n-1) map directly to CPLEX variable
    indices.  Both build_relaxation_model and build_distance_model satisfy this
    because they add x / z before anything else.
    """
    cpx = model.cplex
    m = A.shape[0]

    lin_expr = [
        _cplex_module.SparsePair(
            ind=A.indices[A.indptr[i] : A.indptr[i + 1]].tolist(),
            val=A.data[A.indptr[i] : A.indptr[i + 1]].tolist(),
        )
        for i in range(m)
    ]

    cpx.linear_constraints.add(
        lin_expr=lin_expr,
        senses=["L"] * m,
        rhs=b.tolist(),
    )


def _set_integer_upper_bounds(model: Model, integer_indices: Sequence[int]) -> None:
    """
    Set x[i] <= 1 for all integer-designated variables by updating variable
    upper bounds directly via the CPLEX low-level API.

    Using variable bounds instead of explicit <= constraints removes 0.8*n
    extra rows from the LP, which makes each solve slightly faster too.

    Precondition
    ------------
    Variable j must have CPLEX column index j, which holds when the x / z
    variable list is created as the first block of variables in the model.
    """
    cpx = model.cplex
    cpx.variables.set_upper_bounds([(int(idx), 1.0) for idx in integer_indices])


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
# We build two models:
#
# 1. Relaxation model
#    - maximize sum(y_i)
#    - this gives the initial relaxed point
#
# 2. Distance model
#    - minimize distance to the current rounded point
#    - this is the main FP projection step
# -----------------------------------------------------------------------------
def build_relaxation_model(problem: ProblemInstance, cplex_threads: int = 1):
    """
    Build the initial LP relaxation model.

    Mathematical form:
        maximize sum(y_i)
        subject to
            A x <= b
            y_i = c_i x + d_i
            0 <= x_j <= 1 for integer variables
            x_j >= 0 for all variables

    Solver configuration
    --------------------
    This model is configured to solve with the **barrier algorithm and no
    crossover**.  The rationale is documented in-line at the parameter-set
    call below.  The distance-projection model in build_distance_model is
    intentionally NOT configured this way — see its docstring.
    """
    model = Model(name="fp_relaxation")
    apply_cplex_threads(model, cplex_threads)

    # -------------------------------------------------------------------------
    # LP algorithm: barrier without crossover (initial LP only)
    # -------------------------------------------------------------------------
    # The initial LP relaxation is the slowest single solve in the whole
    # pipeline: on our typical instances (m ~ 9000 rows, n ~ 3000 cols,
    # sparse A) dual simplex — CPLEX's default choice here — can take on the
    # order of minutes and is the reason FPRunConfig exposes a 200s wall-clock
    # cap via `initial_lp_time_limit`.
    #
    # Why barrier (lpmethod = 4):
    #   * Barrier / interior-point methods scale much better than simplex on
    #     large sparse LPs.  A single Cholesky factorisation of (A A^T), plus
    #     a small number of Newton-style iterations, replaces the millions of
    #     pivots that simplex would otherwise perform.
    #   * Barrier is internally parallel — its factorisation and back-solves
    #     scale with `cplex_threads`.  Simplex is effectively serial, so
    #     giving simplex more threads would not help here.
    #   * This LP is solved only ONCE per instance.  The result is then
    #     cached to disk (see initial_lp_disk_cache_path in build_models),
    #     so subsequent process restarts skip the solve entirely.  Making
    #     the first-ever solve faster is pure upside; there is no repeated
    #     warm-start workflow that we would hurt by switching algorithms.
    #
    # Why crossover is disabled (barrier.crossover = -1):
    #   * CPLEX's default after a barrier solve is to run *crossover*, which
    #     pushes the interior-point solution to a nearby vertex (basic)
    #     solution.  Crossover can cost as much wall time as the barrier
    #     phase itself on large LPs.
    #   * Feasibility Pump only needs a *feasible* relaxed point to start
    #     rounding from — it does not need a basic solution, an optimal
    #     vertex, or an LP basis that would later be warm-started.  This
    #     matches the code in solve_relaxation_model, which already accepts
    #     time-limited / non-optimal feasible points as the FP start.
    #   * Skipping crossover therefore saves the vertex-recovery phase at
    #     zero cost to downstream FP behaviour.
    #
    # Accepted values for barrier.crossover:
    #     -1  no crossover (return the interior-point solution as-is)  <-- us
    #      0  automatic (CPLEX default, usually runs crossover)
    #      1  primal crossover
    #      2  dual crossover
    #
    # IMPORTANT: these two parameter lines apply ONLY to the initial
    # relaxation LP.  The distance-projection LP (build_distance_model)
    # is solved O(100) times per episode with a one-row change each time;
    # it benefits enormously from dual-simplex basis warm-starts and must
    # stay on CPLEX defaults.  Do NOT copy these settings into
    # build_distance_model.
    # -------------------------------------------------------------------------
    model.parameters.lpmethod = 4             # 4 = barrier
    # model.parameters.barrier.crossover = -1   # -1 = no crossover

    # Decision variables
    x = model.continuous_var_list(problem.n, lb=0, name="x")
    y = model.continuous_var_list(problem.p, name="y")

    # Constraint system Ax <= b
    # Fast path: one bulk CPLEX call instead of m individual docplex calls.
    # Fall back to the row-by-row docplex loop only if the cplex module is
    # not available (should not happen when CPLEX is properly installed).
    if _FAST_CONSTRAINTS:
        _add_Axb_bulk(model, problem.A, problem.b)
    else:
        for row_idx in range(problem.m):
            model.add_constraint(
                model.sum(val * x[col_idx] for col_idx, val in iter_csr_row(problem.A, row_idx))
                <= problem.b[row_idx]
            )

    # Integer-designated variables are relaxed to [0, 1].
    # Fast path: set as variable upper bounds rather than explicit constraints,
    # which keeps the LP smaller and is faster to set.
    if _FAST_CONSTRAINTS:
        _set_integer_upper_bounds(model, problem.integer_indices)
    else:
        for idx in problem.integer_indices:
            model.add_constraint(x[idx] <= 1)

    # Objective-image constraints: y_i = c_i x + d_i
    # Only p = 2 or 3 of these, so the docplex loop is negligible.
    for obj_idx in range(problem.p):
        expr = model.sum(val * x[col_idx] for col_idx, val in iter_vector_nonzero(problem.c[obj_idx]))
        model.add_constraint(expr + float(problem.d[obj_idx]) == y[obj_idx])

    # Initial LP objective used in the FP paper/code family
    model.maximize(model.sum(y))
    return model, x, y


def solve_relaxation_model(model: Model, x_vars, y_vars, max_seconds: Optional[float]):
    """
    Solve the relaxation model and return:
        x_values, y_values, objective_value

    Returns None only if no feasible solution was found at all.

    Accepts time-limited feasible solutions — FP only needs a starting relaxed
    point, not a proven LP optimum.  For large instances (m=9000, n=3000)
    CPLEX finds a good feasible LP solution in seconds but can take hours to
    close the optimality gap, so requiring "optimal" status would force an
    unbounded solve even though the FP starting point is already good enough.
    """
    # Note: solve() may return None when CPLEX stops at a time limit even though
    # a feasible LP basis exists on the model.  We therefore do NOT early-return
    # on a falsy solve() result; instead, we check model.solution directly so
    # time-limited feasible solutions are accepted as the FP starting point.
    solve_with_time_limit(model, max_seconds)

    if model.solution is None:
        status = str(getattr(model.solve_details, "status", "unknown"))
        logger.warning(
            "Initial LP relaxation produced no solution (status=%s, limit=%s). "
            "Marking instance as failed.",
            status,
            f"{max_seconds:.2f}s" if max_seconds is not None else "none",
        )
        return None

    status = str(getattr(model.solve_details, "status", "")).lower()
    if "optimal" not in status:
        logger.info(
            "Initial LP relaxation stopped before proven optimality with a feasible "
            "solution (status=%s, limit=%s). Using feasible point as FP start.",
            getattr(model.solve_details, "status", "unknown"),
            f"{float(max_seconds):.2f}s" if max_seconds is not None else "none",
        )

    x_values = [float(v.solution_value) for v in x_vars]
    y_values = [float(v.solution_value) for v in y_vars]
    obj = float(model.objective_value)
    return x_values, y_values, obj


def build_distance_model(problem: ProblemInstance, cplex_threads: int = 1):
    """
    Build the distance-projection model.

    Mathematical form:
        minimize distance_var
        subject to
            A z <= b
            y_i = c_i z + d_i
            0 <= z_j <= 1 for integer variables
            distance_var >= current FP distance expression

    The actual distance constraint depends on the current rounded point,
    so we add/remove that constraint dynamically during each solve.

    Solver configuration
    --------------------
    This model is intentionally left on CPLEX defaults (dual simplex).
    Unlike the initial relaxation, the distance model is solved O(100)
    times per FP episode with only a single row added/removed each time
    (see solve_distance_model).  Dual simplex re-optimises from the
    previous optimal basis in a handful of pivots after such a small
    change, which is by far the fastest option for this workload.

    Do NOT switch this model to barrier: barrier has no warm start and
    would re-do the Cholesky factorisation from scratch on every
    iteration, erasing the basis-reuse speedup.  The barrier / no-crossover
    setting in build_relaxation_model is deliberately confined to the
    one-shot initial LP.
    """
    model = Model(name="fp_distance")
    apply_cplex_threads(model, cplex_threads)

    # Projection variables
    z = model.continuous_var_list(problem.n, lb=0, name="z")
    y = model.continuous_var_list(problem.p, name="y")
    distance_var = model.continuous_var(lb=0, name="distance")

    # Constraint system A z <= b — same fast/fallback pattern as the relaxation model
    if _FAST_CONSTRAINTS:
        _add_Axb_bulk(model, problem.A, problem.b)
    else:
        for row_idx in range(problem.m):
            model.add_constraint(
                model.sum(val * z[col_idx] for col_idx, val in iter_csr_row(problem.A, row_idx))
                <= problem.b[row_idx]
            )

    # Integer-designated variables are still relaxed in [0, 1]
    if _FAST_CONSTRAINTS:
        _set_integer_upper_bounds(model, problem.integer_indices)
    else:
        for idx in problem.integer_indices:
            model.add_constraint(z[idx] <= 1)

    # Objective-image constraints
    for obj_idx in range(problem.p):
        expr = model.sum(val * z[col_idx] for col_idx, val in iter_vector_nonzero(problem.c[obj_idx]))
        model.add_constraint(expr + float(problem.d[obj_idx]) == y[obj_idx])

    model.minimize(distance_var)
    return model, z, y, distance_var


def solve_distance_model(
    model: Model,
    z_vars,
    y_vars,
    distance_var,
    rounded_values: Sequence[float],
    integer_indices: Sequence[int],
    max_seconds: Optional[float],
):
    """
    Solve one FP distance-projection step.

    We temporarily add the current distance constraint:
        distance_var >= distance(z, rounded_values)

    Then solve the model, and finally remove that temporary constraint so the
    model can be reused in the next iteration.
    """
    dist_ct = model.add_constraint(
        distance_var >=
        model.sum(z_vars[idx] for idx in integer_indices if rounded_values[idx] == 0) +
        model.sum(1 - z_vars[idx] for idx in integer_indices if rounded_values[idx] == 1)
    )

    try:
        ok = solve_with_time_limit(model, max_seconds)
        if not ok:
            return None

        z_values = [float(v.solution_value) for v in z_vars]
        y_values = [float(v.solution_value) for v in y_vars]
        obj = float(model.objective_value)
        return z_values, y_values, obj
    finally:
        # Important: remove the temporary distance constraint so the model
        # stays reusable for the next rounded point.
        model.remove_constraint(dist_ct)


# -----------------------------------------------------------------------------
# Flip candidate selection
# -----------------------------------------------------------------------------
# For Step 1, we use a simple rule:
# select the variables with the largest disagreement between relaxed and rounded.
#
# Later, the RL agent will decide HOW MANY variables to flip.
# The backend can still decide WHICH ones to flip using this heuristic.
# -----------------------------------------------------------------------------
def select_flip_candidates(
    x_relaxed: Sequence[float],
    x_rounded: Sequence[float],
    integer_indices: Sequence[int],
    num_candidates: int,
) -> list[int]:
    """
    Pick the most fractional / most disagreeing integer variables first.
    """
    scored = sorted(
        integer_indices,
        key=lambda idx: abs(x_relaxed[idx] - x_rounded[idx]),
        reverse=True,
    )
    return scored[: min(num_candidates, len(scored))]


# -----------------------------------------------------------------------------
# Core FP runner
# -----------------------------------------------------------------------------
# This is the main backend object.
#
# It does NOT know anything about Gymnasium yet.
# It only knows how to:
# - initialize FP
# - advance FP naturally until stall
# - apply a perturbation
# - track the quantities we will need later for RL
# -----------------------------------------------------------------------------
class FeasibilityPumpCore:
    """
    Real FP backend for one instance.

    This is the object that Step 2 will wrap inside a Gymnasium environment.
    """

    def __init__(self, problem: ProblemInstance, config: FPRunConfig):
        self.problem = problem
        self.config = config

        # Models and solver variables
        self.relaxation_model = None
        self.relaxation_x = None
        self.relaxation_y = None

        self.distance_model = None
        self.distance_z = None
        self.distance_y = None
        self.distance_var = None

        # Timing / run status
        self.start_time: Optional[float] = None
        self.iteration = 0
        self.done = False
        self.failed = False
        self.integer_found = False

        # Stall and perturbation bookkeeping
        self.consecutive_no_change = 0
        self.stall_events = 0
        self.total_flips = 0
        self.last_k = 0
        self.last_flip_indices: list[int] = []

        # Progress tracking
        self.last_rounding_changed = True
        self.last_distance_delta = 0.0
        self.recent_distance_deltas: deque[float] = deque(maxlen=config.recent_delta_window)

        # Initial LP diagnostics
        self.initial_lp_objective = 0.0
        self.initial_solution_was_integer = False
        self.terminated_in_initial_relaxation = False
        self.initial_distance = 0.0

        # Timing diagnostics
        self.relaxation_build_seconds = 0.0
        self.distance_build_seconds = 0.0
        self.initial_lp_solve_seconds = 0.0
        self.reset_seconds = 0.0

        # Current FP state
        self.x_relaxed: Optional[list[float]] = None
        self.x_rounded: Optional[list[float]] = None
        self.y_values: Optional[list[float]] = None

    def remaining_time(self) -> Optional[float]:
        """
        Return the remaining episode time budget in seconds.
        """
        if self.start_time is None:
            return None
        return self.config.time_limit - (time.time() - self.start_time)

    def current_distance(self) -> float:
        """
        Return current FP distance between:
        - relaxed point
        - rounded point
        """
        if self.integer_found:
            return 0.0
        if self.x_relaxed is None or self.x_rounded is None:
            return 0.0
        return fp_distance(self.x_relaxed, self.x_rounded, self.problem.integer_indices)

    # ------------------------------------------------------------------
    # Two-phase initialisation
    # ------------------------------------------------------------------
    # build_models()  — build the CPLEX LP objects once per instance.
    #                   Call this at pool-construction time so the cost
    #                   is paid once rather than every episode.
    #
    # reset_state()   — reset episode counters and re-solve the initial
    #                   LP using the already-built models.  Fast: no
    #                   model rebuild.
    #
    # reset()         — convenience wrapper that calls both in sequence.
    #                   Kept for backward-compatibility with code that
    #                   creates a fresh runner per episode.
    # ------------------------------------------------------------------

    def build_models(self) -> None:
        """
        Build the two reusable CPLEX LP models and solve the initial LP once.

        This is the slow, one-time-per-instance step.  For a fixed
        training pool, call this once at environment initialisation and
        reuse the models (and the cached LP solution) across all episodes
        via reset_state().

        The initial relaxation uses config.initial_lp_time_limit (None =
        optimality).  A feasible starting point suffices for FP; a time cap
        avoids spending minutes proving LP optimality on huge instances.

        The initial LP solution is cached so that reset_state() can restore
        it instantly without re-solving.
        """
        build_started = time.time()
        self.relaxation_model, self.relaxation_x, self.relaxation_y = build_relaxation_model(
            self.problem,
            cplex_threads=self.config.cplex_threads,
        )
        self.relaxation_build_seconds = time.time() - build_started

        build_started = time.time()
        self.distance_model, self.distance_z, self.distance_y, self.distance_var = build_distance_model(
            self.problem,
            cplex_threads=self.config.cplex_threads,
        )
        self.distance_build_seconds = time.time() - build_started

        # Solve the initial LP once and cache the result.
        # reset_state() copies from this cache instead of re-solving.
        #
        # Disk cache: on first run the solution is written to a .pkl file next
        # to the .npz instance so that subsequent process restarts skip the
        # LP solve entirely.  Filename includes initial_lp_time_limit so
        # optimal vs time-capped caches do not collide.
        inst_path = Path(self.problem.instance_path)
        lp_cache_path = initial_lp_disk_cache_path(
            self.problem.instance_path,
            self.problem.m,
            self.problem.n,
            self.config.initial_lp_time_limit,
        )
        legacy_cache_path = _legacy_lp_disk_cache_path(
            self.problem.instance_path,
            self.problem.m,
            self.problem.n,
        )

        for candidate_path in (lp_cache_path, legacy_cache_path):
            if not candidate_path.exists():
                continue
            try:
                with open(candidate_path, "rb") as _f:
                    lp_result = pickle.load(_f)
                self.initial_lp_solve_seconds = 0.0
                self._cached_lp_result = lp_result
                logger.info(
                    "LP cache hit  — loaded %s (skipped solve)", candidate_path.name
                )
                return
            except Exception as _e:
                logger.warning(
                    "LP cache file %s unreadable (%s) — trying next / re-solving.",
                    candidate_path.name, _e,
                )

        lp_started = time.time()
        lp_result = solve_relaxation_model(
            self.relaxation_model,
            self.relaxation_x,
            self.relaxation_y,
            max_seconds=self.config.initial_lp_time_limit,
        )
        self.initial_lp_solve_seconds = time.time() - lp_started
        # Store as a tuple (x_relaxed, y_values, lp_obj) or None if infeasible.
        self._cached_lp_result = lp_result

        if lp_result is not None:
            try:
                with open(lp_cache_path, "wb") as _f:
                    pickle.dump(lp_result, _f)
                logger.info(
                    "LP cache saved — %s (%.1f KB)",
                    lp_cache_path.name,
                    lp_cache_path.stat().st_size / 1024,
                )
            except Exception as _e:
                logger.warning("Could not write LP cache %s: %s", lp_cache_path.name, _e)

    def reset_state(self) -> None:
        """
        Reset all FP episode state and restore the cached initial LP solution.

        Assumes build_models() has already been called.  Does NOT rebuild
        the CPLEX models, so it is fast enough to call at the start of
        every training episode.

        Baseline-matching behavior
        --------------------------
        The initial LP is obtained in build_models() before the FP-loop timer
        starts, matching the semantics of main_phase1.py.
        """
        reset_started = time.time()

        # Clear all episode counters and solution state
        self.iteration = 0
        self.done = False
        self.failed = False
        self.integer_found = False
        self.consecutive_no_change = 0
        self.stall_events = 0
        self.total_flips = 0
        self.last_k = 0
        self.last_flip_indices = []
        self.last_rounding_changed = True
        self.last_distance_delta = 0.0
        self.recent_distance_deltas.clear()

        self.initial_solution_was_integer = False
        self.terminated_in_initial_relaxation = False
        self.initial_distance = 0.0

        # Build times are not reset here — they reflect the one-time cost
        # from build_models() and remain valid across episodes.
        self.initial_lp_solve_seconds = 0.0
        self.reset_seconds = 0.0

        self.x_relaxed = None
        self.x_rounded = None
        self.y_values = None

        # Restore the initial LP solution from the cache built in build_models().
        # This avoids re-solving the LP on every episode — for large instances
        # (m=9000, n=3000) that solve takes 200+ seconds; the cache makes it
        # near-instant by copying the already-computed arrays.
        self.start_time = None
        result = getattr(self, "_cached_lp_result", None)

        if result is None:
            self.failed = True
            self.done = True
            self.reset_seconds = time.time() - reset_started
            return

        # Copy the cached lists so FP iterations can mutate x_relaxed freely
        # without corrupting the stored initial solution.
        cached_x, cached_y, cached_obj = result
        self.x_relaxed = list(cached_x)
        self.y_values  = list(cached_y)
        self.initial_lp_objective = cached_obj
        self.x_rounded = round_integer_values(self.x_relaxed, self.problem.integer_indices)
        self.initial_distance = fp_distance(
            self.x_relaxed, self.x_rounded, self.problem.integer_indices,
        )
        self.initial_solution_was_integer = is_integer_solution(
            self.x_relaxed, self.problem.integer_indices,
        )

        if self.initial_solution_was_integer:
            self.integer_found = True
            self.done = True
            self.terminated_in_initial_relaxation = True
            self.reset_seconds = time.time() - reset_started
            return

        # FP-loop timer starts only after the initial LP is solved
        self.start_time = time.time()
        self.reset_seconds = time.time() - reset_started

    def reset(self) -> None:
        """
        Full reset: build CPLEX models AND reset episode state.

        Convenience wrapper kept for backward compatibility.
        For a fixed training pool, prefer calling build_models() once
        at startup and reset_state() at the start of each episode.
        """
        self.build_models()
        self.reset_state()

    def is_stalled(self) -> bool:
        """
        Baseline-matching interpretation of a decision point.

        In main_phase1.py, the algorithm flips immediately when rounding no
        longer changes. It does not wait for several no-change rounds.

        So here, one no-change event is enough to say that FP has reached the
        point where a flip decision is needed.
        """
        return self.consecutive_no_change >= 1

    def run_one_iteration(self, flip_indices: Sequence[int]) -> bool:
        """
        Run exactly one FP distance-projection iteration.

        Baseline-matching intent
        ------------------------
        This is closer to main_phase1.py:

        - solve one distance-projection step
        - if the new relaxed point is integer, stop
        - if rounding the new relaxed point changes the current rounded point,
          update the rounded point and continue naturally
        - otherwise, mark an immediate decision/stall point

        Parameters
        ----------
        flip_indices : Sequence[int]
            If non-empty, these binary variables are flipped in the rounded point
            before solving the distance model.

        Returns
        -------
        bool
            True if an iteration was actually executed, False if the run had
            already ended.
        """
        if self.done or self.x_rounded is None:
            return False

        # Stop if iteration budget is exhausted
        if self.iteration >= self.config.max_iterations:
            self.done = True
            return False

        # Stop if FP-loop time budget is exhausted
        remaining = self.remaining_time()
        if remaining is not None and remaining <= 0:
            self.done = True
            return False

        num_flips = len(flip_indices)
        prev_stalled = self.is_stalled()

        # Apply perturbation to the rounded point if requested
        if flip_indices:
            self.x_rounded = flip_selected_variables(self.x_rounded, flip_indices)

        prev_distance = self.current_distance()

        # Record perturbation metadata
        self.last_flip_indices = list(flip_indices)
        self.last_k = num_flips
        self.total_flips += num_flips

        # Solve one distance-projection step against the current rounded point
        result = solve_distance_model(
            self.distance_model,
            self.distance_z,
            self.distance_y,
            self.distance_var,
            self.x_rounded,
            self.problem.integer_indices,
            max_seconds=remaining,
        )
        self.iteration += 1

        # If the solve fails, terminate the run
        if result is None:
            self.failed = True
            self.done = True
            return True

        # Update the current relaxed point and objective image
        self.x_relaxed, self.y_values, _ = result

        # Track distance improvement
        new_distance = self.current_distance()
        self.last_distance_delta = prev_distance - new_distance
        self.recent_distance_deltas.append(self.last_distance_delta)

        # If projection landed on an integer point, FP succeeded
        if is_integer_solution(self.x_relaxed, self.problem.integer_indices):
            self.integer_found = True
            self.done = True
            return True

        # -------------------------------------------------------------
        # Core baseline-matching logic:
        # if rounding changes, update the rounded guide point;
        # otherwise, this is immediately a flip / decision point.
        # -------------------------------------------------------------
        self.last_rounding_changed = rounding_changed(
            self.x_relaxed,
            self.x_rounded,
            self.problem.integer_indices,
        )

        if self.last_rounding_changed:
            self.x_rounded = round_integer_values(
                self.x_relaxed,
                self.problem.integer_indices,
            )
            self.consecutive_no_change = 0

        elif num_flips > 0:
            # A manual flip was just applied. Reset the no-change streak after
            # that perturbation step.
            self.consecutive_no_change = 0

        else:
            # One no-change event is already the decision point.
            self.consecutive_no_change = 1

        # Count a stall event only when entering stalled state
        currently_stalled = self.is_stalled()
        if currently_stalled and not prev_stalled:
            self.stall_events += 1

        # Global stopping conditions
        remaining = self.remaining_time()
        if self.iteration >= self.config.max_iterations:
            self.done = True
        elif remaining is not None and remaining <= 0:
            self.done = True
        elif self.stall_events >= self.config.max_stalls:
            self.done = True

        return True

    def advance_until_stall_or_done(self, max_steps: int = 999_999) -> None:
        """
        Run natural FP steps (no manual flip) until:
        - FP is done, or
        - a no-change event occurs (decision point), or
        - max_steps natural iterations have been executed.

        Parameters
        ----------
        max_steps : int
            Maximum number of natural FP iterations to run before returning
            control to the caller, even if no stall has been detected.
            Defaults to a large sentinel so existing call-sites that omit the
            argument are unaffected.

            The RL agent uses this to implement the continuation / patience
            action: a larger value lets FP run longer before the next
            intervention; a smaller value returns control sooner.

        Baseline-matching intent
        ------------------------
        Stall detection (is_stalled) fires after a single no-change event,
        matching the flip-immediately semantics of main_phase1.py.
        max_steps is an additional early-exit for the RL agent and does not
        alter the stall definition.
        """
        steps_taken = 0
        while not self.done and steps_taken < max_steps:
            executed = self.run_one_iteration([])
            if not executed:
                break
            steps_taken += 1

            # Return control as soon as a no-change / stall event is detected.
            if self.is_stalled():
                break

    def apply_flip_count(self, flip_count: int) -> None:
        """
        Apply one perturbation decision using a flip count.

        For Step 1, this is a simple baseline action:
        - choose how many variables to flip
        - select candidates by largest disagreement
        - run one FP iteration after the flip
        """
        if self.done:
            return

        if self.x_relaxed is None or self.x_rounded is None:
            return

        flip_indices = select_flip_candidates(
            self.x_relaxed,
            self.x_rounded,
            self.problem.integer_indices,
            flip_count,
        )
        self.run_one_iteration(flip_indices)


# -----------------------------------------------------------------------------
# Simple single-instance runner
# -----------------------------------------------------------------------------
# This is the easiest way to test that the backend works on a real instance
# before we build the Gymnasium environment around it.
# -----------------------------------------------------------------------------
def run_single_fp_episode(instance_path: str | Path, config: Optional[FPRunConfig] = None) -> dict:
    """
    Run one full real FP episode on one instance.

    For Step 1 we use a simple baseline perturbation rule:
        whenever FP stalls, flip 10 variables.

    This is not RL yet. It is just a real working backend run.
    """
    cfg = config or FPRunConfig()
    problem = load_instance(instance_path)
    runner = FeasibilityPumpCore(problem, cfg)

    # Initialize FP
    runner.reset()

    # Run until first stall or termination
    if not runner.done:
        runner.advance_until_stall_or_done()

    # Keep going until the run ends
    while not runner.done:
        # Baseline decision rule for Step 1
        runner.apply_flip_count(10)

        # After perturbation, run naturally again until next stall or termination
        if not runner.done:
            runner.advance_until_stall_or_done()

    # Return a simple JSON-friendly summary
    return {
        "instance_path": problem.instance_path,
        "m": problem.m,
        "n": problem.n,
        "p": problem.p,
        "iterations": runner.iteration,
        "stall_events": runner.stall_events,
        "total_flips": runner.total_flips,
        "integer_found": runner.integer_found,
        "failed": runner.failed,

        # Initial diagnostics
        "terminated_in_initial_relaxation": runner.terminated_in_initial_relaxation,
        "initial_solution_was_integer": runner.initial_solution_was_integer,
        "initial_distance": runner.initial_distance,
        "initial_lp_objective": runner.initial_lp_objective,

        # Timing diagnostics
        "relaxation_build_seconds": runner.relaxation_build_seconds,
        "distance_build_seconds": runner.distance_build_seconds,
        "initial_lp_solve_seconds": runner.initial_lp_solve_seconds,
        "reset_seconds": runner.reset_seconds,

        "final_distance": runner.current_distance(),
        "elapsed_seconds": 0.0 if runner.start_time is None else (time.time() - runner.start_time),
    }


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
# This lets you run the backend directly on one instance from the terminal.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run real FP core on one .npz or .lp instance."
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="Path to one .npz or .lp instance",
    )
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--stall-threshold", type=int, default=3)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument(
        "--initial-lp-time-limit",
        type=float,
        default=200.0,
        help=(
            "Wall-clock cap (seconds) for the initial LP relaxation in build_models. "
            "Feasible solution may be non-optimal; enough for FP. "
            "Ignored if --initial-lp-optimal is set."
        ),
    )
    parser.add_argument(
        "--initial-lp-optimal",
        action="store_true",
        help="Solve the initial LP to optimality (no time limit); slow on large instances.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    initial_lp_limit = None if args.initial_lp_optimal else args.initial_lp_time_limit

    cfg = FPRunConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
        initial_lp_time_limit=initial_lp_limit,
    )

    summary = run_single_fp_episode(args.instance, cfg)
    print(json.dumps(summary, indent=2))
