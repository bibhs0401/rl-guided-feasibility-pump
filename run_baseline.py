import argparse
import csv
import glob
import logging
import os

import pandas as pd

import main_phase1 as baseline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baseline")
DEFAULT_INSTANCE_DIR = os.path.join(os.path.expanduser("~"), "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the original baseline feasibility-pump code.")
    parser.add_argument(
        "--instance",
        help="Single CSV instance file to run.",
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="Glob pattern for CSV instance files. If provided, runs each match in order.",
    )
    parser.add_argument(
        "--output-file",
        default="output_baseline.csv",
        help="CSV file where baseline results will be saved.",
    )
    return parser.parse_args()


def run_instance(instance_path: str):
    logger.info("Running baseline on %s", instance_path)
    data = pd.read_csv(instance_path)
    A, b, c, m, n, d, p = baseline.required_data(data)
    result = baseline.main_function(A, b, c, m, n, d, p)
    print(f"{instance_path}: {result}")
    return result


def main():
    args = parse_args()

    with open(args.output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "instance",
            "decision_variables",
            "y_values",
            "objective_value",
            "delta",
            "solution_time",
            "fp_iterations",
            "cut_iterations",
        ])

        if args.instance:
            result = run_instance(args.instance)
            writer.writerow([args.instance, *result])
            logger.info("Saved baseline result to %s", args.output_file)
            return

        pattern = args.instances or os.path.join(DEFAULT_INSTANCE_DIR, "instance*.csv")
        instance_files = sorted(glob.glob(pattern))
        if not instance_files:
            raise FileNotFoundError(f"No instance files matched pattern: {pattern}")

        for instance_path in instance_files:
            result = run_instance(instance_path)
            writer.writerow([instance_path, *result])

    logger.info("Saved baseline results to %s", args.output_file)


if __name__ == "__main__":
    main()
