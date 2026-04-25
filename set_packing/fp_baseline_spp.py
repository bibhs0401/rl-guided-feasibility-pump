from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from set_packing.spp_model import (
    DEFAULT_TOLERANCE,
    SPPInstance,
    feasibility_metrics,
    find_instance_files,
    load_spp_instance,
    objective_value,
    objective_values_per_obj,
    repair_set_packing_solution,
    round_binary,
    validate_set_packing_instance,
    violation_pattern,
)


# 6 flip bins: None = exactly 1 variable, others are proportions of n.
# Matches slide: bin0=1var, bin1=1%, bin2=2%, bin3=5%, bin4=10%, bin5=20%
ACTION_PROPORTIONS: tuple = (None, 0.01, 0.02, 0.05, 0.10, 0.20)

# 5 continuation bins: max FP iterations after a perturbation before re-intervening.
# Matches slide: very short, short, medium, long, very long
CONTINUATION_STEPS: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass
class FPConfig:
    max_iterations: int = 100
    time_limit: float = 30.0
    stall_length: int = 3
    tolerance: float = DEFAULT_TOLERANCE
    random_seed: int = 0
    baseline_action: int = 2
    stop_on_repaired_incumbent: bool = False
    # Quality gate for repair-found incumbents.
    # A repaired solution is only accepted as best_feasible if its objective
    # exceeds  lp_obj * repair_quality_threshold.  Set to 0.0 to accept any
    # feasible solution (old behaviour).  Set to e.g. 0.30 to require that
    # the repaired solution is at least 30% of the LP optimal — this prevents
    # the trivial "remove all conflicting variables" repair from short-circuiting
    # the RL training signal on hard instances.
    repair_quality_threshold: float = 0.0
    # LP objective stored at reset time for quality-gate comparisons.
    # Set automatically; do not configure manually.
    _lp_obj: float = 0.0
    cplex_threads: int = 1
    verbose: bool = True


@dataclass
class FPResult:
    instance_name: str
    method: str
    success: int
    final_objective: float
    final_violation: float
    num_violated_constraints: int
    iterations: int
    runtime_seconds: float
    num_stalls: int
    num_rl_interventions: int
    average_return: str | float
    notes_error_status: str
    final_solution: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class StepOutcome:
    moved: bool
    stalled: bool
    feasible_found: bool
    message: str = ""


@dataclass
class CplexSolveResult:
    success: bool
    x: Optional[np.ndarray]
    message: str
    objective_value: Optional[float] = None
    y: Optional[np.ndarray] = None   # per-objective values y_k (p,), when p > 1


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def _iter_csr_row(instance: SPPInstance, row_index: int):
    A = instance.A
    start, end = A.indptr[row_index], A.indptr[row_index + 1]
    for j, value in zip(A.indices[start:end], A.data[start:end]):
        yield int(j), float(value)


def _new_cplex_model(name: str, time_limit: Optional[float], cplex_threads: int):
    try:
        from docplex.mp.model import Model
    except ModuleNotFoundError:
        return None, (
            "DOCplex/CPLEX is not installed in this Python environment. "
            "Install with: python -m pip install docplex cplex"
        )

    model = Model(name=name)
    model.context.cplex_parameters.threads = max(1, int(cplex_threads))
    model.parameters.simplex.tolerances.feasibility = 1e-7
    model.parameters.simplex.tolerances.optimality = 1e-7
    if time_limit is not None:
        model.set_time_limit(max(0.01, float(time_limit)))
    return model, ""


def _solve_docplex_model(model, variables: Sequence, log_output: bool = False) -> CplexSolveResult:
    solution = model.solve(log_output=log_output)
    status = str(model.solve_details.status) if model.solve_details is not None else "unknown"
    if solution is None:
        return CplexSolveResult(False, None, f"CPLEX solve failed: {status}")
    values = np.asarray([float(solution.get_value(var)) for var in variables], dtype=float)
    return CplexSolveResult(True, values, f"CPLEX status: {status}", float(model.objective_value))


def solve_lp_relaxation(
    instance: SPPInstance,
    time_limit: Optional[float] = None,
    cplex_threads: int = 1,
) -> CplexSolveResult:
    model, error = _new_cplex_model("spp_lp_relaxation", time_limit, cplex_threads)
    if model is None:
        return CplexSolveResult(False, None, error)
    x = model.continuous_var_list(instance.n, lb=0.0, ub=1.0, name="x")
    # Set-packing constraints Ax <= b
    for i in range(instance.m):
        model.add_constraint(
            model.sum(value * x[j] for j, value in _iter_csr_row(instance, i)) <= float(instance.b[i]),
            ctname=f"packing_{i}",
        )
    if instance.p > 1 and len(instance.c) == instance.p:
        # Multi-objective: maximize sum(y_k) with y_k = c_k·x + d_k
        y = model.continuous_var_list(instance.p, name="y")
        for k in range(instance.p):
            ck = instance.c[k]
            model.add_constraint(
                model.sum(float(ck[j]) * x[j] for j in range(instance.n) if ck[j] != 0.0)
                + float(instance.d[k]) == y[k],
                ctname=f"obj_y{k + 1}",
            )
        model.maximize(model.sum(y))
        base = _solve_docplex_model(model, x)
        if base.success:
            y_vals = np.asarray([float(yv.solution_value) for yv in y], dtype=float)
            return CplexSolveResult(True, base.x, base.message, base.objective_value, y=y_vals)
        return base
    else:
        # Single-objective: maximize profits·x
        model.maximize(model.sum(float(instance.profits[j]) * x[j] for j in range(instance.n)))
        return _solve_docplex_model(model, x)


def solve_distance_projection(
    instance: SPPInstance,
    rounded: Sequence[float],
    time_limit: Optional[float] = None,
    cplex_threads: int = 1,
) -> CplexSolveResult:
    target = np.asarray(rounded, dtype=float)
    model, error = _new_cplex_model("spp_distance_projection", time_limit, cplex_threads)
    if model is None:
        return CplexSolveResult(False, None, error)
    z = model.continuous_var_list(instance.n, lb=0.0, ub=1.0, name="z")
    # Set-packing constraints Az <= b
    for i in range(instance.m):
        model.add_constraint(
            model.sum(value * z[j] for j, value in _iter_csr_row(instance, i)) <= float(instance.b[i]),
            ctname=f"packing_{i}",
        )
    # Add objective-image constraints when multi-objective (y_k = c_k·z + d_k)
    y_vars = None
    if instance.p > 1 and len(instance.c) == instance.p:
        y_vars = model.continuous_var_list(instance.p, name="y")
        for k in range(instance.p):
            ck = instance.c[k]
            model.add_constraint(
                model.sum(float(ck[j]) * z[j] for j in range(instance.n) if ck[j] != 0.0)
                + float(instance.d[k]) == y_vars[k],
                ctname=f"obj_y{k + 1}",
            )
    distance_objective = model.sum(
        z[j] if target[j] < 0.5 else (1.0 - z[j])
        for j in range(instance.n)
    )
    model.minimize(distance_objective)
    base = _solve_docplex_model(model, z)
    if base.success and y_vars is not None:
        y_vals = np.asarray([float(yv.solution_value) for yv in y_vars], dtype=float)
        return CplexSolveResult(True, base.x, base.message, base.objective_value, y=y_vals)
    return base


def fp_distance(lp_solution: Sequence[float], rounded: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.asarray(lp_solution, dtype=float) - np.asarray(rounded, dtype=float))))


def action_to_flip_count(action: int, n: int) -> int:
    """Map a flip-bin action to an integer flip count.

    Bin 0  → exactly 1 variable (the minimum meaningful perturbation).
    Bin 1  → max(2, ceil(1%  * n))  — slide specifies min-2 for the 1% bin.
    Bin 2+ → ceil(proportion * n),  minimum 1.
    """
    action = int(np.clip(action, 0, len(ACTION_PROPORTIONS) - 1))
    proportion = ACTION_PROPORTIONS[action]
    if proportion is None:          # bin 0: exactly 1 variable
        return 1
    k = int(math.ceil(proportion * max(1, n)))
    if action == 1:                 # 1% bin: minimum 2 flips
        return max(2, k)
    return max(1, k)


class SPPFeasibilityPump:
    def __init__(self, instance: SPPInstance, config: FPConfig):
        self.instance = instance
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.start_time = 0.0
        self.iterations = 0
        self.num_stalls = 0
        self.num_rl_interventions = 0
        self.failed = False
        self.done = False
        self.success = False
        self.notes: list[str] = []

        self.x_lp: Optional[np.ndarray] = None
        self.x_binary: Optional[np.ndarray] = None
        self.y_values: Optional[np.ndarray] = None   # per-objective values (p,) from last LP/projection
        self.best_feasible: Optional[np.ndarray] = None
        self.best_feasible_objective = -math.inf
        self.best_distance = math.inf
        self.no_improvement_count = 0
        self.seen_rounded: set[tuple[int, ...]] = set()
        self.pattern_counts: dict[tuple[int, ...], int] = {}
        self.last_flip_indices: list[int] = []
        self.last_distance = math.inf
        self.last_lp_status = ""

    def remaining_time(self) -> float:
        return max(0.0, self.config.time_limit - (time.time() - self.start_time))

    def _current_signature(self) -> tuple[int, ...]:
        if self.x_binary is None:
            return ()
        return tuple(int(v > 0.5) for v in self.x_binary)

    def current_distance(self) -> float:
        if self.x_lp is None or self.x_binary is None:
            return math.inf
        return fp_distance(self.x_lp, self.x_binary)

    def current_metrics(self):
        if self.x_binary is None:
            zeros = np.zeros(self.instance.n)
            return feasibility_metrics(self.instance, zeros, self.config.tolerance)
        return feasibility_metrics(self.instance, self.x_binary, self.config.tolerance)

    def current_objective(self) -> float:
        if self.instance.p > 1 and self.y_values is not None:
            return float(np.sum(self.y_values))
        if self.x_binary is None:
            return 0.0
        return objective_value(self.instance, self.x_binary)

    def reset(self) -> None:
        self.start_time = time.time()
        _log(self.config.verbose, f"[{self.instance.name}] solving LP relaxation")
        res = solve_lp_relaxation(
            self.instance,
            self.config.time_limit,
            cplex_threads=self.config.cplex_threads,
        )
        self.last_lp_status = str(res.message)
        _log(self.config.verbose, f"[{self.instance.name}] LP relaxation status: {res.message}")
        if not res.success:
            self.failed = True
            self.done = True
            self.notes.append(f"initial_lp_failed: {res.message}")
            return

        self.x_lp = np.asarray(res.x, dtype=float)
        self.x_binary = round_binary(self.x_lp)
        if res.y is not None:
            self.y_values = res.y.copy()
        self.last_distance = self.current_distance()
        # Cache LP objective for repair quality gate
        if res.objective_value is not None:
            self.config._lp_obj = float(res.objective_value)
        else:
            self.config._lp_obj = float(objective_value(self.instance, self.x_lp))
        self.best_distance = self.last_distance
        _log(
            self.config.verbose,
            (
                f"[{self.instance.name}] initial distance={self.last_distance:.6g} "
                f"violation={self.current_metrics().total_violation:.6g}"
            ),
        )

    def _update_repaired_incumbent(self) -> None:
        if self.x_binary is None:
            return
        repaired, info = repair_set_packing_solution(self.instance, self.x_binary, self.config.tolerance)
        metrics = feasibility_metrics(self.instance, repaired, self.config.tolerance)
        if info.applied:
            _log(
                self.config.verbose,
                (
                    f"[{self.instance.name}] repair applied: removed={len(info.removed_indices)} "
                    f"violation {info.initial_total_violation:.6g}->{info.final_total_violation:.6g}"
                ),
            )
        if metrics.is_feasible:
            obj = objective_value(self.instance, repaired)
            # Quality gate: only accept repair incumbent if it clears the threshold
            # relative to the LP optimal.  This prevents the trivial "remove all
            # conflicting variables" repair from masking the RL training signal on
            # hard instances where the repair finds a near-empty (low-quality) solution.
            lp_obj = self.config._lp_obj
            threshold = self.config.repair_quality_threshold
            quality_ok = (threshold <= 0.0) or (lp_obj <= 0.0) or (obj >= threshold * lp_obj)
            if quality_ok and obj > self.best_feasible_objective + self.config.tolerance:
                self.best_feasible = repaired.copy()
                self.best_feasible_objective = obj
                self.notes.append("repaired_incumbent")
            if self.config.stop_on_repaired_incumbent:
                self.success = True
                self.done = True
                self.x_binary = repaired

    def _mark_success(self, solution: np.ndarray, note: str) -> None:
        self.success = True
        self.done = True
        self.x_binary = solution.copy()
        obj = objective_value(self.instance, solution)
        if obj > self.best_feasible_objective:
            self.best_feasible = solution.copy()
            self.best_feasible_objective = obj
        self.notes.append(note)
        _log(self.config.verbose, f"[{self.instance.name}] final success: {note}, objective={obj:.6g}")

    def _detect_stall(self, distance: float, signature: tuple[int, ...], pattern: tuple[int, ...]) -> bool:
        repeated_round = signature in self.seen_rounded
        self.seen_rounded.add(signature)

        if distance < self.best_distance - self.config.tolerance:
            self.best_distance = distance
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
        repeated_pattern = self.pattern_counts[pattern] >= max(2, self.config.stall_length)
        no_improvement = self.no_improvement_count >= self.config.stall_length
        return bool(repeated_round or no_improvement or repeated_pattern)

    def run_one_iteration(self) -> StepOutcome:
        if self.done or self.x_binary is None:
            return StepOutcome(False, False, self.success, "already_done")
        if self.iterations >= self.config.max_iterations:
            self.done = True
            self.notes.append("max_iterations")
            return StepOutcome(False, False, self.success, "max_iterations")
        if self.remaining_time() <= 0:
            self.done = True
            self.notes.append("time_limit")
            return StepOutcome(False, False, self.success, "time_limit")

        metrics_before = self.current_metrics()
        distance_before = self.current_distance()
        _log(
            self.config.verbose,
            (
                f"[{self.instance.name}] iter={self.iterations} "
                f"distance={distance_before:.6g} "
                f"violation={metrics_before.total_violation:.6g} "
                f"violated_rows={metrics_before.num_violated_constraints}"
            ),
        )
        if metrics_before.is_feasible:
            self._mark_success(self.x_binary, "rounded_feasible")
            return StepOutcome(True, False, True, "rounded_feasible")

        self._update_repaired_incumbent()
        if self.done:
            return StepOutcome(True, False, self.success, "repair_feasible")

        res = solve_distance_projection(
            self.instance,
            self.x_binary,
            self.remaining_time(),
            cplex_threads=self.config.cplex_threads,
        )
        _log(self.config.verbose, f"[{self.instance.name}] projection LP status: {res.message}")
        if not res.success:
            self.failed = True
            self.done = True
            self.notes.append(f"projection_failed: {res.message}")
            return StepOutcome(False, False, False, "projection_failed")

        self.x_lp = np.asarray(res.x, dtype=float)
        if res.y is not None:
            self.y_values = res.y.copy()
        new_binary = round_binary(self.x_lp)
        self.x_binary = new_binary
        self.iterations += 1

        metrics_after = self.current_metrics()
        distance_after = self.current_distance()
        self.last_distance = distance_after
        if metrics_after.is_feasible:
            self._mark_success(new_binary, "projection_rounded_feasible")
            return StepOutcome(True, False, True, "projection_rounded_feasible")

        signature = self._current_signature()
        pattern = violation_pattern(self.instance, self.x_binary, self.config.tolerance)
        stalled = self._detect_stall(distance_after, signature, pattern)
        if stalled:
            self.num_stalls += 1
            _log(self.config.verbose, f"[{self.instance.name}] stall detected at iter={self.iterations}")
        return StepOutcome(True, stalled, False, "stalled" if stalled else "continue")

    def run_until_stall_or_done(self, max_steps: Optional[int] = None) -> StepOutcome:
        steps = 0
        last = StepOutcome(False, False, self.success, "not_started")
        while not self.done:
            if max_steps is not None and steps >= max_steps:
                return last
            last = self.run_one_iteration()
            steps += 1
            if last.stalled:
                return last
            if not last.moved:
                return last
        return last

    def _candidate_scores(self) -> np.ndarray:
        if self.x_lp is None or self.x_binary is None:
            return np.zeros(self.instance.n)
        fractionality = np.abs(self.x_lp - np.round(self.x_lp))
        metrics = self.current_metrics()
        activity = self.instance.A @ self.x_binary
        violated = np.flatnonzero(activity > self.instance.b + self.config.tolerance)
        conflict = np.zeros(self.instance.n, dtype=float)
        if violated.size:
            conflict = np.asarray(self.instance.A[violated].sum(axis=0)).reshape(-1)
        recent = np.zeros(self.instance.n, dtype=float)
        for idx in self.last_flip_indices:
            if 0 <= idx < self.instance.n:
                recent[idx] = 0.25
        profit_scale = max(1.0, float(np.max(np.abs(self.instance.profits))))
        profit_term = self.instance.profits / profit_scale
        active_bonus = np.where(self.x_binary > 0.5, 0.15, 0.0)
        return 2.0 * fractionality + conflict + recent + 0.10 * profit_term + active_bonus + metrics.total_violation * 0.0

    def select_flip_indices(self, k: int) -> list[int]:
        if k <= 0 or self.x_binary is None:
            return []
        scores = self._candidate_scores()
        if not np.any(scores > self.config.tolerance):
            scores = self.rng.random(self.instance.n)
        order = np.argsort(-scores)
        return [int(j) for j in order[: min(k, self.instance.n)]]

    def apply_perturbation(self, action: int, rl_intervention: bool = False) -> dict:
        if self.done or self.x_binary is None:
            return {"action": action, "flip_count": 0, "repair_applied": False}

        action = int(np.clip(action, 0, len(ACTION_PROPORTIONS) - 1))
        k = action_to_flip_count(action, self.instance.n)
        selected = self.select_flip_indices(k)
        _log(
            self.config.verbose,
            f"[{self.instance.name}] {'RL' if rl_intervention else 'baseline'} action={action} flips={len(selected)}",
        )
        for idx in selected:
            self.x_binary[idx] = 1.0 - self.x_binary[idx]
        self.last_flip_indices = selected
        if rl_intervention:
            self.num_rl_interventions += 1

        repair_applied = False
        if selected:
            repaired, info = repair_set_packing_solution(self.instance, self.x_binary, self.config.tolerance)
            self.x_binary = repaired
            repair_applied = info.applied
            _log(
                self.config.verbose,
                (
                    f"[{self.instance.name}] repair after perturbation="
                    f"{'yes' if repair_applied else 'no'} "
                    f"final_violation={info.final_total_violation:.6g}"
                ),
            )
        self.no_improvement_count = 0
        self.seen_rounded.clear()
        self.pattern_counts.clear()
        return {"action": action, "flip_count": len(selected), "repair_applied": repair_applied}

    def result(self, method: str, average_return: str | float = "") -> FPResult:
        if self.best_feasible is not None:
            final = self.best_feasible
            success = 1
        elif self.x_binary is not None:
            final = self.x_binary
            success = 0
        else:
            final = np.zeros(self.instance.n)
            success = 0
        metrics = feasibility_metrics(self.instance, final, self.config.tolerance)
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        notes = ";".join(dict.fromkeys(self.notes))
        if self.failed:
            notes = f"{notes};failed" if notes else "failed"
        return FPResult(
            instance_name=self.instance.name,
            method=method,
            success=success,
            final_objective=objective_value(self.instance, final),
            final_violation=metrics.total_violation,
            num_violated_constraints=metrics.num_violated_constraints,
            iterations=self.iterations,
            runtime_seconds=elapsed,
            num_stalls=self.num_stalls,
            num_rl_interventions=self.num_rl_interventions,
            average_return=average_return,
            notes_error_status=notes,
            final_solution=final.copy(),
        )


def run_baseline_fp(
    instance: SPPInstance,
    config: FPConfig,
    perturb_on_stall: bool = True,
) -> FPResult:
    warnings = validate_set_packing_instance(instance, config.tolerance)
    for warning in warnings:
        _log(config.verbose, f"[{instance.name}] warning: {warning}")

    runner = SPPFeasibilityPump(instance, config)
    runner.reset()
    while not runner.done:
        outcome = runner.run_until_stall_or_done()
        if runner.done:
            break
        if outcome.stalled and perturb_on_stall:
            runner.apply_perturbation(config.baseline_action, rl_intervention=False)
            continue
        break
    result = runner.result("baseline_fp")
    _log(
        config.verbose,
        (
            f"[{instance.name}] final success/failure={result.success} "
            f"obj={result.final_objective:.6g} violation={result.final_violation:.6g}"
        ),
    )
    return result


RESULT_COLUMNS = [
    "instance_name",
    "method",
    "success",
    "final_objective",
    "final_violation",
    "num_violated_constraints",
    "iterations",
    "runtime_seconds",
    "num_stalls",
    "num_rl_interventions",
    "average_return",
    "notes_error_status",
]


def write_results_csv(results: Sequence[FPResult], path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "instance_name": row.instance_name,
                    "method": row.method,
                    "success": row.success,
                    "final_objective": f"{row.final_objective:.10g}",
                    "final_violation": f"{row.final_violation:.10g}",
                    "num_violated_constraints": row.num_violated_constraints,
                    "iterations": row.iterations,
                    "runtime_seconds": f"{row.runtime_seconds:.6f}",
                    "num_stalls": row.num_stalls,
                    "num_rl_interventions": row.num_rl_interventions,
                    "average_return": row.average_return,
                    "notes_error_status": row.notes_error_status,
                }
            )
    return str(out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline feasibility pump for set packing.")
    parser.add_argument("--instance-dir", default=".", help="Directory containing .npz or .lp instances.")
    parser.add_argument("--instances", nargs="*", default=None, help="Explicit instance paths.")
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--stall-length", type=int, default=3)
    parser.add_argument("--baseline-action", type=int, default=2, choices=range(5))
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--output", default="results/baseline_fp_results.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.instances or find_instance_files([args.instance_dir])
    paths = paths[: args.max_instances]
    print(f"[baseline] number of instances found: {len(paths)}")
    if not paths:
        raise SystemExit("No .npz or .lp set-packing instances found.")

    results: list[FPResult] = []
    for path in paths:
        instance = load_spp_instance(path)
        print(f"[baseline] current instance: {instance.name}")
        cfg = FPConfig(
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            stall_length=args.stall_length,
            baseline_action=args.baseline_action,
            cplex_threads=args.cplex_threads,
            verbose=not args.quiet,
        )
        results.append(run_baseline_fp(instance, cfg, perturb_on_stall=not args.no_perturb))
    out = write_results_csv(results, args.output)
    print(f"[baseline] wrote {out}")


if __name__ == "__main__":
    main()
