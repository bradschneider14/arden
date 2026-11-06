"""Command-line entry point: ``python -m arden``."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path

from arden import __version__
from arden.agent.loop import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_RUN_TICKS,
    AgentLoop,
    run_headless,
)
from arden.config import Settings, load_config, make_rng
from arden.llm.factory import make_backend
from arden.llm.prompts import build_decide_prompt
from arden.llm.schemas import Intent


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
        "--ticks",
        type=int,
        default=None,
        help=f"Number of simulation ticks to run (default: {DEFAULT_RUN_TICKS}).",
    )
    run_parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="Override the configured LLM backend (e.g. 'mock' or 'ollama').",
    )
    run_parser.add_argument(
        "--heartbeat",
        type=int,
        default=None,
        help=f"Heartbeat gate interval in ticks (default: {DEFAULT_HEARTBEAT_INTERVAL}).",
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
    ticks: int = DEFAULT_RUN_TICKS,
    heartbeat_interval: int | None = None,
) -> AgentLoop:
    """Run one headless simulation and print a one-line summary per wake."""
    return run_headless(
        settings=settings,
        rng=rng,
        ticks=ticks,
        heartbeat_interval=heartbeat_interval,
        print_wakes=True,
    )


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
        if args.llm is not None:
            settings.llm.backend = args.llm
        rng = make_rng(settings)
        kwargs: dict[str, int] = {}
        sig = inspect.signature(run_simulation)
        if args.ticks is not None and "ticks" in sig.parameters:
            kwargs["ticks"] = args.ticks
        if args.heartbeat is not None and "heartbeat_interval" in sig.parameters:
            kwargs["heartbeat_interval"] = args.heartbeat
        run_simulation(settings, rng, **kwargs)
        return 0
    if args.command == "llm-smoke":
        settings = load_config(args.config)
        run_llm_smoke(settings)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
