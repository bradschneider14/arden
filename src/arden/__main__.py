"""Command-line entry point: ``python -m arden``."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from arden import __version__
from arden.config import Settings, load_config, make_rng
from arden.llm.factory import make_backend
from arden.llm.prompts import build_decide_prompt
from arden.llm.schemas import Intent

if TYPE_CHECKING:
    from arden.world.world import World


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
    run_parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help="Path to a world definition file (e.g. worlds/first_run.toml).",
    )
    run_parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Number of simulation ticks to run when --world is given.",
    )

    smoke_parser = subparsers.add_parser(
        "llm-smoke",
        help="Send one decide prompt to the configured LLM backend and print the parsed intent and latency",
    )
    smoke_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to arden.toml. Defaults to ./arden.toml when that file exists.",
    )
    return parser


def run_simulation(
    settings: Settings,
    rng: random.Random,
    *,
    world_path: Path | str | None = None,
    ticks: int | None = None,
) -> World | None:
    """Run one simulation.

    When ``world_path`` is provided (e.g. ``--world worlds/first_run.toml``),
    loads and validates the world definition and optionally steps ``ticks`` ticks.
    When called without ``world_path``, preserves the placeholder wiring output.
    """
    if world_path is None and ticks is None:
        print("not implemented yet")
        return None

    from arden.world.loader import DEFAULT_WORLD_PATH, load_world, resolve_world_path, run_world_ticks

    target_path = resolve_world_path(DEFAULT_WORLD_PATH if world_path is None else world_path)
    world = load_world(target_path, rng=rng, settings=settings)
    if ticks is not None and ticks > 0:
        run_world_ticks(world, ticks=ticks, settings=settings)
    print(f"loaded world: {target_path.name} objects={len(world.objects)} tick={world.tick}")
    return world


def run_llm_smoke(settings: Settings) -> Intent:
    """Send one decide prompt to the configured backend and print the parsed intent and latency."""
    backend = make_backend(settings)
    prompt = build_decide_prompt(
        percept=(
            "tick: 1\n"
            "body: pos=(0,0) level=ground energy=0.8 comfort=0.7 holding=none\n"
            "objects:\n"
            "- object_1 dist=1 dir=N pos=(0,1) level=ground shape=sphere h=0.2 vol=0.004 w=0.3 rig=0.4 color=red\n"
            "affordances:\n"
            "- object_1: grab, push, strike\n"
            "events: none"
        ),
        self_model="none",
        memories=[],
        body_state={"position": {"x": 0.0, "y": 0.0}, "level": "ground", "energy": 0.8, "comfort": 0.7, "holding": None},
        previous_intent_result=None,
    )
    start = time.perf_counter()
    raw_intent = backend.complete_json(
        "decide",
        prompt.system,
        prompt.user,
        Intent.model_json_schema(),
        settings.llm.timeout_s,
    )
    elapsed_s = time.perf_counter() - start
    intent = Intent.model_validate(raw_intent)
    print(f"backend: {settings.llm.backend} (model: {settings.llm.model})")
    print(f"latency_ms: {elapsed_s * 1000.0:.2f} ({elapsed_s:.3f}s)")
    print(f"intent: {json.dumps(intent.model_dump(), sort_keys=True)}")
    return intent


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        settings = load_config(args.config)
        if args.world is None and args.ticks is None:
            run_simulation(settings, make_rng(settings))
        else:
            run_simulation(
                settings,
                make_rng(settings),
                world_path=args.world,
                ticks=args.ticks,
            )
        return 0
    if args.command == "llm-smoke":
        settings = load_config(args.config)
        run_llm_smoke(settings)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
