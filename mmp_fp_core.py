from __future__ import annotations

# Standard library imports
import logging
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
#   Holds one instance in matrix/vector form after reading the .npz file.
#
# FPRunConfig:
#   Holds the solver-facing FP configuration for one run.
# -----------------------------------------------------------------------------
@dataclass
class ProblemInstance:
    """
    One MMP instance loaded from a sparse .npz file.

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
        Separate protective cap for the initial LP relaxation solve.
        This is an engineering safeguard so reset() does not hang for a very
        long time on difficult instances.
    """
    max_iterations: int = 100
    time_limit: float = 30.0
    initial_lp_time_limit: float | None = 200.0
    stall_threshold: int = 3
    max_stalls: int = 50
    recent_delta_window: int = 5
    cplex_threads: int = 1


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
    """
    model = Model(name="fp_relaxation")
    apply_cplex_threads(model, cplex_threads)

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

    Returns None if the solve fails.
    """
    ok = solve_with_time_limit(model, max_seconds)
    if not ok:
        return None

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
        Build the two reusable CPLEX LP models for this instance.

        This is the slow, one-time-per-instance step.  For a fixed
        training pool, call this once at environment initialisation and
        reuse the models across all episodes via reset_state().
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

    def reset_state(self) -> None:
        """
        Reset all FP episode state and solve the initial LP relaxation.

        Assumes build_models() has already been called.  Does NOT rebuild
        the CPLEX models, so it is fast enough to call at the start of
        every training episode.

        Baseline-matching behavior
        --------------------------
        The initial LP is solved BEFORE the FP-loop timer starts, matching
        the semantics of main_phase1.py.
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

        # Solve the initial LP relaxation using the pre-built model
        self.start_time = None

        solve_started = time.time()
        result = solve_relaxation_model(
            self.relaxation_model,
            self.relaxation_x,
            self.relaxation_y,
            max_seconds=self.config.initial_lp_time_limit,
        )
        self.initial_lp_solve_seconds = time.time() - solve_started

        if result is None:
            self.failed = True
            self.done = True
            self.reset_seconds = time.time() - reset_started
            return

        self.x_relaxed, self.y_values, self.initial_lp_objective = result
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
    problem = load_npz_instance(instance_path)
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
# This lets you run the backend directly on one .npz instance from the terminal.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run real FP core on one .npz instance.")
    parser.add_argument("--instance", required=True, help="Path to one .npz instance")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--stall-threshold", type=int, default=3)
    parser.add_argument("--max-stalls", type=int, default=50)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument(
        "--initial-lp-time-limit",
        type=float,
        default=200.0,
        help="Separate time limit in seconds for the initial LP relaxation solve.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = FPRunConfig(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        stall_threshold=args.stall_threshold,
        max_stalls=args.max_stalls,
        cplex_threads=args.cplex_threads,
        initial_lp_time_limit=args.initial_lp_time_limit,
    )

    summary = run_single_fp_episode(args.instance, cfg)
    print(json.dumps(summary, indent=2))
