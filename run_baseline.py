import argparse
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
    return parser.parse_args()


def run_instance(instance_path: str):
    logger.info("Running baseline on %s", instance_path)
    data = pd.read_csv(instance_path)
    A, b, c, m, n, d, p = baseline.required_data(data)
    result = baseline.main_function(A, b, c, m, n, d, p)
    print(f"{instance_path}: {result}")


def main():
    args = parse_args()

    if args.instance:
        run_instance(args.instance)
        return

    pattern = args.instances or os.path.join(DEFAULT_INSTANCE_DIR, "instance*.csv")
    instance_files = sorted(glob.glob(pattern))
    if not instance_files:
        raise FileNotFoundError(f"No instance files matched pattern: {pattern}")

    for instance_path in instance_files:
        run_instance(instance_path)


if __name__ == "__main__":
    main()
