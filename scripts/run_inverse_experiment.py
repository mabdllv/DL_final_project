from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inverse.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    defaults = ExperimentConfig(checkpoint_path="", case_file="", output_dir="", method="unconditional")
    parser = argparse.ArgumentParser(description="Run one inverse-problem sampler on a shared case file.")
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, type=str, default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required")
    if not args.case_file:
        raise SystemExit("--case-file is required")
    if not args.output_dir:
        raise SystemExit("--output-dir is required")
    metrics_path = run_experiment(ExperimentConfig(**vars(args)))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
