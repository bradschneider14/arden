"""Data-driven world loader and material-property coverage checks.

Loads a ``World`` from a TOML or JSON file (such as ``worlds/first_run.toml``)
containing ``[terrain]`` (with ``regions`` and/or ``tiles``), optional
``[body]`` / ``[agent]`` / ``[daynight]`` / ``[weather]`` tables, and
``[[objects]]`` built purely from physical properties, hidden ``behavior``
rules, and optional ``creator_label`` metadata for the web client and observer
log.
"""

from __future__ import annotations

import json
import random
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from arden.environment.weather import WIND_MAX_OBJECT_WEIGHT
from arden.world.objects import WorldObject, validate_world_object
from arden.world.world import World

DEFAULT_FIRST_RUN_WORLD_PATH = Path("worlds/first_run.toml")

MATERIAL_PROPERTIES: tuple[str, ...] = (
    "rigid",
    "binding",
    "hollow",
    "heavy",
    "light",
    "round",
    "fragile",
    "wet",
)

# Thresholds used to verify material property representation across a world's objects.
RIGID_MIN_RIGIDITY: float = 0.8
HEAVY_MIN_WEIGHT: float = 2.0
LIGHT_MAX_WEIGHT: float = 0.5
VERY_LIGHT_MAX_WEIGHT: float = WIND_MAX_OBJECT_WEIGHT
ROUND_SHAPES: frozenset[str] = frozenset({"sphere", "cylinder"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_world_path(path: Path | str | None = None) -> Path:
    """Resolve ``path`` (defaulting to ``worlds/first_run.toml``) to an existing file path."""
    target = DEFAULT_FIRST_RUN_WORLD_PATH if path is None else Path(path)
    if target.is_file():
        return target
    if not target.is_absolute():
        fallback = _repo_root() / target
        if fallback.is_file():
            return fallback
    return target


def load_world_data(path: Path | str | None = None) -> dict[str, Any]:
    """Read and parse a TOML or JSON world definition file into a dictionary."""
    resolved = resolve_world_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"world file not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    else:
        with resolved.open("rb") as fh:
            raw = tomllib.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"world file {resolved} must contain a top-level mapping")
    return raw


def load_world(
    path: Path | str | Mapping[str, Any] | World | None = None,
    *,
    rng: random.Random | int | None = None,
    seed: int | None = None,
    settings: Any = None,
    config: Any = None,
) -> World:
    """Load and return a ``World`` from a file path, mapping, or existing ``World``."""
    if isinstance(path, World):
        return path

    if isinstance(path, Mapping):
        data: dict[str, Any] = dict(path)
    else:
        data = load_world_data(path)

    effective_config = settings if settings is not None else config
    if rng is None and seed is None:
        if effective_config is not None and hasattr(effective_config, "seed"):
            seed = int(effective_config.seed)
        elif "seed" in data and isinstance(data["seed"], int) and not isinstance(data["seed"], bool):
            seed = int(data["seed"])

    body_data = data.get("body", data.get("agent"))
    terrain_data = data.get("terrain")
    daynight_data = data.get("daynight")
    weather_data = data.get("weather")
    body_config_data = data.get("body_config")
    drives_data = data.get("drives")
    raw_objects = data.get("objects", ())

    objects_list: list[WorldObject] = []
    if isinstance(raw_objects, Mapping):
        for obj_id, obj_payload in raw_objects.items():
            if isinstance(obj_payload, WorldObject):
                objects_list.append(obj_payload)
            elif isinstance(obj_payload, Mapping):
                merged = {"id": obj_id, **dict(obj_payload)}
                objects_list.append(validate_world_object(merged))
            else:
                raise TypeError("objects mapping values must be mappings or WorldObject instances")
    else:
        for item in raw_objects:
            if isinstance(item, WorldObject):
                objects_list.append(item)
            elif isinstance(item, Mapping):
                objects_list.append(validate_world_object(item))
            else:
                raise TypeError("objects list entries must be mappings or WorldObject instances")

    world = World(
        rng=rng,
        seed=seed,
        body=body_data,
        objects=objects_list,
        terrain=terrain_data,
        daynight=daynight_data,
        weather=weather_data,
        config=effective_config,
        body_config=body_config_data,
        drives=drives_data,
    )
    return world


def load_first_run_world(
    *,
    rng: random.Random | int | None = None,
    seed: int | None = None,
    settings: Any = None,
    config: Any = None,
) -> World:
    """Load the default first-run clearing world from ``worlds/first_run.toml``."""
    return load_world(
        DEFAULT_FIRST_RUN_WORLD_PATH,
        rng=rng,
        seed=seed,
        settings=settings,
        config=config,
    )


build_first_run_world = load_first_run_world


def object_has_material_property(obj: WorldObject, prop: str, world: World | None = None) -> bool:
    """Return True if ``obj`` exhibits material property ``prop``."""
    if prop == "rigid":
        return obj.rigidity >= RIGID_MIN_RIGIDITY
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
        if obj.state.get("wet") is True:
            return True
        if world is not None and world.terrain.is_water(obj.position):
            return True
        return False
    raise ValueError(f"unknown material property: {prop!r}")


def check_material_property_coverage(world: World) -> dict[str, bool]:
    """Return a mapping of each required material property to whether ``world`` represents it."""
    objects = list(world.objects.values())
    coverage: dict[str, bool] = {}
    for prop in MATERIAL_PROPERTIES:
        if prop == "wet":
            has_wet_obj = any(object_has_material_property(obj, "wet", world) for obj in objects)
            has_water_tile = any(tile.wet for tile in world.terrain._tiles.values())
            coverage[prop] = has_wet_obj or has_water_tile
        else:
            coverage[prop] = any(object_has_material_property(obj, prop, world) for obj in objects)
    return coverage


def verify_material_properties(world: World) -> bool:
    """Return True if all eight material properties are represented at least once in ``world``."""
    return all(check_material_property_coverage(world).values())


__all__ = [
    "DEFAULT_FIRST_RUN_WORLD_PATH",
    "HEAVY_MIN_WEIGHT",
    "LIGHT_MAX_WEIGHT",
    "MATERIAL_PROPERTIES",
    "RIGID_MIN_RIGIDITY",
    "ROUND_SHAPES",
    "VERY_LIGHT_MAX_WEIGHT",
    "build_first_run_world",
    "check_material_property_coverage",
    "load_first_run_world",
    "load_world",
    "load_world_data",
    "object_has_material_property",
    "resolve_world_path",
    "verify_material_properties",
]
