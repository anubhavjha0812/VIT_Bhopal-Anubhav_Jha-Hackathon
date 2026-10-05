"""One-command demo. For now it loads the config and shows which version of each part is active."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from riskengine.config import DEFAULT_PATH, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--smoke", action="store_true", help="fast run for CI")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Config: {args.config}")
    for name, value in asdict(config).items():
        print(f"  {name:<15}{value}")
    print("Pipeline not built yet. Exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
