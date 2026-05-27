from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolmogorov_dataset import KolmogorovConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate forced 2D Kolmogorov-flow vorticity snapshots."
    )
    defaults = KolmogorovConfig()
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
        elif isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, type=str, default=value)
    return parser.parse_args()


def main() -> None:
    config = KolmogorovConfig(**vars(parse_args()))
    paths = generate_dataset(config)
    print("Saved chunks:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
