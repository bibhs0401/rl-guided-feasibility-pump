import argparse
import csv
import glob
import logging
import os
from statistics import mean

import pandas as pd

import main_phase1
import main_phase1_rl

try:
    import socp
except ImportError:
    socp = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_report")


def algorithm_y_multiply(values):
    result = 1
    for value in values:
        result *= value
    return result


def cplex_gamma_multiply(gamma_value, p):
    return gamma_value ** p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and RL-based FP by subclass and save a paper-style summary table."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing subclass folders like 1, 2, ..., 9.",
    )
    parser.add_argument(
        "--output-file",
        default="paper_summary.csv",
        help="CSV file for the aggregated paper-style table.",
    )
    parser.add_argument(
        "--instances-pattern",
        default="instance*.csv",
        help="Filename pattern used inside each subclass folder.",
    )
    return parser.parse_args()


def sorted_subclass_dirs(root_path):
    subdirs = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    def key_fn(path):
        name = os.path.basename(path)
        return (0, int(name)) if name.isdigit() else (1, name)

    return sorted(subdirs, key=key_fn)


def get_subclass_label(instance_path):
    data = pd.read_csv(instance_path)
    _, _, _, m, n, _, _ = main_phase1.required_data(data)
    return f"{m:,} x {n:,}"


def run_baseline_instance(instance_path):
    data = pd.read_csv(instance_path)
    A, b, c, m, n, d, p = main_phase1.required_data(data)
    decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = (
        main_phase1.main_function(A, b, c, m, n, d, p)
    )

    algorithm_value = "none"
    if y_values not in ["none", None]:
        algorithm_value = algorithm_y_multiply(y_values)

    socp_time = "none"
    socp_gap = "none"
    baseline_gap = "none"

    if socp is not None:
        I, _ = main_phase1.get_integer_index(n)
        _, cplex_solution_time, best_bound, gamma, gap_cplex = socp.main(A, b, d, p, n, c, I)
        socp_time = cplex_solution_time
        socp_gap = gap_cplex
        try:
            optimal_value_cplex = cplex_gamma_multiply(gamma, p)
            baseline_gap = ((optimal_value_cplex - algorithm_value) * 100 / optimal_value_cplex)
        except Exception:
            baseline_gap = "none"

    return {
        "solution_time": solution_time,
        "gap": baseline_gap,
        "socp_time": socp_time,
        "socp_gap": socp_gap,
    }


def run_phase1_instance(instance_path):
    data = pd.read_csv(instance_path)
    A, b, c, m, n, d, p = main_phase1_rl.required_data(data)
    agent = main_phase1_rl.Phase1FlipAgent()
    decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = (
        main_phase1_rl.main_function(A, b, c, m, n, d, p, phase1_agent=agent)
    )

    algorithm_value = "none"
    if y_values not in ["none", None]:
        algorithm_value = algorithm_y_multiply(y_values)

    rl_gap = "none"
    if socp is not None:
        I, _ = main_phase1_rl.get_integer_index(n)
        _, cplex_solution_time, best_bound, gamma, gap_cplex = socp.main(A, b, d, p, n, c, I)
        try:
            optimal_value_cplex = cplex_gamma_multiply(gamma, p)
            rl_gap = ((optimal_value_cplex - algorithm_value) * 100 / optimal_value_cplex)
        except Exception:
            rl_gap = "none"

    return {
        "solution_time": solution_time,
        "gap": rl_gap,
    }


def average_numeric(values):
    numeric_values = [float(value) for value in values if value not in ["none", None]]
    if not numeric_values:
        return "none"
    return mean(numeric_values)


def format_value(value):
    if value == "none":
        return value
    return f"{float(value):.1f}"


def main():
    args = parse_args()
    subclass_dirs = sorted_subclass_dirs(args.root)
    if not subclass_dirs:
        raise FileNotFoundError(f"No subclass directories found under {args.root}")

    rows = []

    for subclass_dir in subclass_dirs:
        instance_files = sorted(glob.glob(os.path.join(subclass_dir, args.instances_pattern)))
        if not instance_files:
            logger.warning("Skipping %s because no instances matched %s", subclass_dir, args.instances_pattern)
            continue

        subclass_name = os.path.basename(subclass_dir)
        subclass_label = get_subclass_label(instance_files[0])
        logger.info("Processing subclass %s (%s) with %s instances", subclass_name, subclass_label, len(instance_files))

        baseline_results = [run_baseline_instance(path) for path in instance_files]
        rl_results = [run_phase1_instance(path) for path in instance_files]

        row = {
            "subclass_folder": subclass_name,
            "subclass": subclass_label,
            "socp_time_sec": format_value(average_numeric([result["socp_time"] for result in baseline_results])),
            "socp_gap_pct": format_value(average_numeric([result["socp_gap"] for result in baseline_results])),
            "baseline_time_sec": format_value(average_numeric([result["solution_time"] for result in baseline_results])),
            "baseline_gap_pct": format_value(average_numeric([result["gap"] for result in baseline_results])),
            "rl_phase1_time_sec": format_value(average_numeric([result["solution_time"] for result in rl_results])),
            "rl_phase1_gap_pct": format_value(average_numeric([result["gap"] for result in rl_results])),
        }
        rows.append(row)

    with open(args.output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "subclass_folder",
                "subclass",
                "socp_time_sec",
                "socp_gap_pct",
                "baseline_time_sec",
                "baseline_gap_pct",
                "rl_phase1_time_sec",
                "rl_phase1_gap_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved paper-style summary to %s", args.output_file)


if __name__ == "__main__":
    main()
