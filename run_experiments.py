import csv
import time
import logging
import random
import os
import pandas as pd
import main_phase1
import main_phase1_rl
try:
    import main_phase1_sb3
except ImportError:
    main_phase1_sb3 = None

try:
    import main_phase1_phase2_rl
except ImportError:
    main_phase1_phase2_rl = None

try:
    import socp
except ImportError:
    socp = None

random.seed(10)
DEFAULT_INSTANCE_DIR = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("runner")


def algorithm_y_multiply(values):
    result = 1
    for v in values:
        result *= v
    return result


def cplex_gamma_multiply(gamma_value, p):
    return gamma_value ** p


def run_one_mode(mode, total_instances=10, sb3_model_path=None):
    output_file = f"output_{mode}.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "mode", "y_values_algorithm", "objective_value_algorithm",
            "solution_time", "cut_iterations", "fp_iterations",
            "cplex_gamma", "best_bound", "cplex_socp_optimal",
            "cplex_solution_time", "optimality_gap_(%)",
            "optimality_gap_best_bound", "gap_cplex"
        ])

    phase1_agent = None
    phase2_agent = None

    if mode == "phase1":
        phase1_agent = main_phase1_rl.Phase1FlipAgent()
    elif mode == "phase1_sb3":
        if main_phase1_sb3 is None:
            raise ImportError("phase1_sb3 mode requested, but main_phase1_sb3.py is not available in this workspace")
        if not sb3_model_path:
            raise ValueError("phase1_sb3 mode requires sb3_model_path")
        phase1_agent = main_phase1_sb3.SB3Phase1Agent.load(sb3_model_path)
    elif mode == "phase2":
        if main_phase1_phase2_rl is None:
            raise ImportError("phase2 mode requested, but main_phase1_phase2_rl.py is not available in this workspace")
        phase2_agent = main_phase1_phase2_rl.Phase2FlipAgent()

    times = []
    total_start = time.time()

    for i in range(total_instances):
        inst_file = os.path.join(DEFAULT_INSTANCE_DIR, f"instance{i+1}.csv")
        start = time.time()

        logger.info(f"Starting {inst_file} | mode={mode}")

        data = pd.read_csv(inst_file)

        if mode == "baseline":
            A, b, c, m, n, d, p = main_phase1.required_data(data)
            decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = \
                main_phase1.main_function(A, b, c, m, n, d, p)
            I, not_I = main_phase1.get_integer_index(n)

        elif mode == "phase1":
            A, b, c, m, n, d, p = main_phase1_rl.required_data(data)
            decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = \
                main_phase1_rl.main_function(A, b, c, m, n, d, p, phase1_agent=phase1_agent)
            I, not_I = main_phase1_rl.get_integer_index(n)

        elif mode == "phase1_sb3":
            A, b, c, m, n, d, p = main_phase1_sb3.base.required_data(data)
            decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = \
                main_phase1_sb3.main_function(A, b, c, m, n, d, p, phase1_agent=phase1_agent)
            I, not_I = main_phase1_sb3.base.get_integer_index(n)

        elif mode == "phase2":
            A, b, c, m, n, d, p = main_phase1_phase2_rl.required_data(data)
            decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = \
                main_phase1_phase2_rl.main_function(A, b, c, m, n, d, p, phase2_agent=phase2_agent)
            I, not_I = main_phase1_phase2_rl.get_integer_index(n)

        optimal_value_algorithm = "none"
        if y_values not in ["none", None]:
            optimal_value_algorithm = algorithm_y_multiply(y_values)

        if socp is None:
            variables_value, cplex_solution_time, best_bound, gamma, gap_cplex = ("none", "none", "none", "none", "none")
        else:
            variables_value, cplex_solution_time, best_bound, gamma, gap_cplex = socp.main(A, b, d, p, n, c, I)

        try:
            optimal_value_cplex = cplex_gamma_multiply(gamma, p)
        except:
            optimal_value_cplex = "none"

        try:
            optimality_gap = ((optimal_value_cplex - optimal_value_algorithm) * 100 / optimal_value_cplex)
        except:
            optimality_gap = "none"

        try:
            best_bound_multiplied = cplex_gamma_multiply(best_bound, p)
        except:
            best_bound_multiplied = "none"

        try:
            optimality_gap_best_bound = ((best_bound_multiplied - optimal_value_algorithm) * 100 / best_bound_multiplied)
        except:
            optimality_gap_best_bound = "none"

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i + 1, mode, y_values, optimal_value_algorithm, solution_time,
                cut_iteration, nIT_list, gamma, best_bound, optimal_value_cplex,
                cplex_solution_time, optimality_gap, optimality_gap_best_bound, gap_cplex
            ])

        elapsed = time.time() - start
        times.append(elapsed)
        avg = sum(times) / len(times)
        eta = avg * (total_instances - (i + 1))

        logger.info(
            f"Finished {inst_file} | mode={mode} | "
            f"instance_time={elapsed:.2f}s | avg={avg:.2f}s | ETA={eta:.2f}s"
        )

    logger.info(f"Completed mode={mode} | total_time={time.time() - total_start:.2f}s")

if __name__ == "__main__":
    run_one_mode("baseline")
    run_one_mode("phase1")
    if main_phase1_phase2_rl is not None:
        run_one_mode("phase2")
    else:
        logger.info("Skipping phase2 mode because main_phase1_phase2_rl.py is not available")
