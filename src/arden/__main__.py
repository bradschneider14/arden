"""Command-line entry point: ``python -m arden``."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from arden import __version__
from arden.config import Settings, load_config, make_rng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arden")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the simulation")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to arden.toml. Defaults to ./arden.toml when that file exists.",
    )
    return parser


def run_simulation(settings: Settings, rng: random.Random) -> None:
    """Run one simulation.

    ``settings`` and the seeded ``rng`` are the inputs for the world loop.
    That loop is a later ticket. Until then this command only confirms the
    wiring: config is loaded, one generator is created, and both are passed here.
    """
    print("not implemented yet")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        settings = load_config(args.config)
        run_simulation(settings, make_rng(settings))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
