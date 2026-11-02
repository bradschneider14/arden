"""Data-driven world loader, property-coverage verification, and simulation runner."""

from __future__ import annotations

from collections import deque
import json
import random
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from arden.config import Settings, make_rng
from arden.world.objects import WorldObject
from arden.world.world import World

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORLD_PATH = Path("worlds/first_run.toml")
FIRST_RUN_WORLD_PATH = REPO_ROOT / DEFAULT_WORLD_PATH

REQUIRED_MATERIAL_PROPERTIES: tuple[str, ...] = (
    "rigid",
    "binding",
    "hollow",
    "heavy",
    "light",
    "round",
    "fragile",
    "wet",
)

RIGID_MIN_THRESHOLD: float = 0.7
HEAVY_MIN_WEIGHT: float = 2.0
LIGHT_MAX_WEIGHT: float = 0.5
ROUND_SHAPES: frozenset[str] = frozenset({"sphere", "cylinder"})

_OPAQUE_VERB_RE = re.compile(r"^action_[0-9]+$")


def resolve_world_path(path: Path | str | None = None) -> Path:
    """Resolve a world file path relative to the current working directory or repository root."""
    candidate = Path(DEFAULT_WORLD_PATH if path is None else path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate
    return candidate


def load_world_data(path: Path | str | None = None) -> dict[str, Any]:
    """Read and parse a TOML or JSON world definition file into a dictionary."""
    resolved = resolve_world_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"world file not found: {resolved}")

    raw_text = resolved.read_text(encoding="utf-8")
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        parsed = json.loads(raw_text)
    else:
        parsed = tomllib.loads(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError(f"world definition in {resolved} must decode to a mapping")
    return parsed


def _object_has_property(obj: WorldObject, prop: str) -> bool:
    """Return True if ``obj`` exhibits material property ``prop``."""
    if prop == "rigid":
        return obj.rigidity >= RIGID_MIN_THRESHOLD
    if prop == "binding":
        return bool(obj.binding)
    if prop == "hollow":
        return bool(obj.hollow)
    if prop == "heavy":
        return obj.weight >= HEAVY_MIN_WEIGHT
    if prop == "light":
        return obj.weight <= LIGHT_MAX_WEIGHT
    if prop == "round":
        return obj.shape in ROUND_SHAPES
    if prop == "fragile":
        return bool(obj.fragile)
    if prop == "wet":
        return bool(obj.state.get("wet", False))
    raise ValueError(f"unknown material property: {prop!r}")


def property_coverage(world: World) -> dict[str, bool]:
    """Return a mapping of each required material property to whether it is present in ``world``."""
    objects = list(world.objects.values())
    coverage: dict[str, bool] = {}
    for prop in REQUIRED_MATERIAL_PROPERTIES:
        present = any(_object_has_property(obj, prop) for obj in objects)
        if not present and prop == "wet":
            present = any(tile.wet for tile in world.terrain._tiles.values())
        coverage[prop] = present
    return coverage


def check_property_coverage(world: World) -> dict[str, bool]:
    """Verify that every required material property is represented at least once in ``world``.

    Raises ``ValueError`` listing any missing properties.
    """
    coverage = property_coverage(world)
    missing = [prop for prop, ok in coverage.items() if not ok]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"world is missing required material properties: {joined}")
    return coverage


def _register_placeholder_opaque_actions(world: World) -> None:
    """Register no-op placeholder handlers for any ``action_<n>`` behavior rules on loaded objects."""
    for obj in world.objects.values():
        for rule in obj.behavior:
            if _OPAQUE_VERB_RE.fullmatch(rule.verb):
                reg = world._opaque_actions.get(obj.id, {})
                if rule.verb not in reg:
                    world.register_opaque_action(
                        obj.id,
                        rule.verb,
                        lambda *_args, **_kwargs: {"ok": True, "placeholder": True},
                    )


def load_world(
    path: Path | str | Mapping[str, Any] | None = None,
    *,
    rng: random.Random | int | None = None,
    seed: int | None = None,
    config: Any = None,
    settings: Settings | None = None,
    validate_coverage: bool = True,
) -> World:
    """Load a ``World`` from a TOML/JSON file (defaulting to ``worlds/first_run.toml``) or mapping."""
    if isinstance(path, Mapping):
        data: dict[str, Any] = dict(path)
    else:
        data = load_world_data(path)

    effective_config = settings if settings is not None else config
    if rng is None and seed is None and isinstance(effective_config, Settings):
        rng = make_rng(effective_config)

    body_data = data.get("body")
    terrain_data = data.get("terrain")
    objects_data = data.get("objects")
    daynight_data = data.get("daynight")
    weather_data = data.get("weather")
    body_config_data = data.get("body_config")
    drives_data = data.get("drives")
    preset = data.get("preset")

    world = World(
        rng=rng,
        seed=seed,
        body=body_data,
        objects=objects_data,
        terrain=terrain_data,
        daynight=daynight_data,
        weather=weather_data,
        config=effective_config,
        body_config=body_config_data,
        preset=preset,
        drives=drives_data,
    )
    _register_placeholder_opaque_actions(world)

    if validate_coverage:
        check_property_coverage(world)

    return world


def load_first_run_world(
    *,
    rng: random.Random | int | None = None,
    seed: int | None = None,
    config: Any = None,
    settings: Settings | None = None,
) -> World:
    """Load the first-run clearing world from ``worlds/first_run.toml``."""
    return load_world(
        FIRST_RUN_WORLD_PATH,
        rng=rng,
        seed=seed,
        config=config,
        settings=settings,
        validate_coverage=True,
    )


def run_world_ticks(
    world: World,
    ticks: int = 2000,
    *,
    settings: Settings | None = None,
    backend: Any = None,
    heartbeat_ticks: int = 25,
) -> World:
    """Run ``world`` for ``ticks`` ticks using the configured or mock LLM backend."""
    if ticks < 0:
        raise ValueError("ticks must be non-negative")

    from arden.actions.verbs import execute_action
    from arden.llm.factory import make_backend
    from arden.llm.mock import MockLLM, derive_llm_seed
    from arden.llm.prompts import build_decide_prompt
    from arden.llm.schemas import Intent, Step
    from arden.perception.feed import PerceptionTracker

    cfg = settings if isinstance(settings, Settings) else (
        world.config if isinstance(getattr(world, "config", None), Settings) else Settings()
    )
    if backend is not None:
        llm_backend = backend
    elif cfg.llm.backend == "mock":
        llm_backend = make_backend(cfg)
    else:
        llm_backend = MockLLM(seed=derive_llm_seed(cfg.world.seed))

    tracker = PerceptionTracker(world, config=cfg)
    intent_schema = Intent.model_json_schema()
    step_queue: deque[Step] = deque()
    last_result: str | None = None

    for step_idx in range(ticks):
        percept = tracker.perceive()
        should_decide = (
            step_idx == 0
            or (
                not step_queue
                and (
                    bool(percept.events)
                    or world.should_wake_gate
                    or (heartbeat_ticks > 0 and step_idx % heartbeat_ticks == 0)
                )
            )
        )
        if should_decide:
            prompt = build_decide_prompt(
                percept=percept.to_text(),
                self_model="none",
                memories=[],
                body_state=percept.body.to_dict(),
                previous_intent_result=last_result,
            )
            raw_intent = llm_backend.complete_json(
                "decide",
                prompt.system,
                prompt.user,
                intent_schema,
                cfg.llm.timeout_s,
            )
            intent = Intent.model_validate(raw_intent)
            step_queue.clear()
            step_queue.extend(intent.steps)

        verb_name: str | None = None
        if step_queue:
            current_step = step_queue.popleft()
            verb_name = current_step.verb
            res = execute_action(
                world,
                current_step.verb,
                current_step.target,
                current_step.params,
            )
            last_result = "ok" if res.ok else f"failed:{res.reason}"

        world.step(verb=verb_name)

    return world
