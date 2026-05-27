from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inverse.cases import CaseConfig, create_case_file


def parse_args() -> argparse.Namespace:
    defaults = CaseConfig(data_source="", output_path="", operator="sparse_grid")
    parser = argparse.ArgumentParser(description="Create a shared inverse-problem case file from validation data.")
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
    if not args.data_source:
        raise SystemExit("--data-source is required")
    if not args.output_path:
        raise SystemExit("--output-path is required")
    path = create_case_file(CaseConfig(**vars(args)))
    print(f"Saved inverse cases: {path}")


if __name__ == "__main__":
    main()
