"""Deterministic 2.5D property-driven physics: terrain, movement, collisions, stacking, and occlusion."""

from __future__ import annotations

import math
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arden.world.objects import Position, Velocity, WorldObject, object_radius

if TYPE_CHECKING:
    from arden.world.world import AgentBody, World

TileType = Literal["grass", "water", "ledge"]
TILE_TYPES: frozenset[str] = frozenset({"grass", "water", "ledge"})

# Physics constants (simple property-driven rules).
BASE_IMPULSE: float = 1.0
STOP_SPEED_EPSILON: float = 0.005
MAX_SUBSTEP_DISTANCE: float = 0.20
MAX_SUBSTEPS: int = 16
COLLISION_RESTITUTION: float = 0.35
COLLISION_ITERATIONS: int = 30
SEPARATION_SLOP: float = 1e-6

# Water and ledge constants.
DEFAULT_WATER_FLOW: tuple[float, float] = (1.0, 0.0)
WATER_FLOAT_MAX_WEIGHT: float = 1.0
WATER_DRIFT_STRENGTH: float = 0.25
WATER_SINK_DRAG: float = 0.50
LEDGE_HEIGHT: float = 1.0
LEDGE_CLIMB_MIN_HEIGHT: float = 0.15

# Default proximity tolerance for approximate line-of-sight occlusion.
DEFAULT_OCCLUSION_TOLERANCE: float = 0.5

# Shape-dependent impulse mobility multipliers (round shapes start faster and roll).
SHAPE_IMPULSE_FACTOR: dict[str, float] = {
    "sphere": 1.20,
    "cylinder": 1.10,
    "long": 0.85,
    "cube": 0.70,
    "flat": 0.60,
}

# Per-tick velocity retention (1 - friction); round shapes roll much farther than flat/cube.
SHAPE_RETENTION: dict[str, float] = {
    "sphere": 0.85,
    "cylinder": 0.80,
    "long": 0.65,
    "cube": 0.55,
    "flat": 0.50,
}

SHAPE_FRICTION: dict[str, float] = {
    shape: round(1.0 - retention, 4) for shape, retention in SHAPE_RETENTION.items()
}

# Shape stability when acting as the lower support in a stack.
SHAPE_STABILITY: dict[str, float] = {
    "flat": 1.00,
    "cube": 0.90,
    "cylinder": 0.75,
    "long": 0.60,
    "sphere": 0.00,
}

_DIRECTION_NAMES: dict[str, tuple[float, float]] = {
    "e": (1.0, 0.0),
    "east": (1.0, 0.0),
    "right": (1.0, 0.0),
    "w": (-1.0, 0.0),
    "west": (-1.0, 0.0),
    "left": (-1.0, 0.0),
    "n": (0.0, 1.0),
    "north": (0.0, 1.0),
    "up": (0.0, 1.0),
    "s": (0.0, -1.0),
    "south": (0.0, -1.0),
    "down": (0.0, -1.0),
    "ne": (math.sqrt(0.5), math.sqrt(0.5)),
    "northeast": (math.sqrt(0.5), math.sqrt(0.5)),
    "nw": (-math.sqrt(0.5), math.sqrt(0.5)),
    "northwest": (-math.sqrt(0.5), math.sqrt(0.5)),
    "se": (math.sqrt(0.5), -math.sqrt(0.5)),
    "southeast": (math.sqrt(0.5), -math.sqrt(0.5)),
    "sw": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "southwest": (-math.sqrt(0.5), -math.sqrt(0.5)),
}

_last_active_world: weakref.ReferenceType[World] | None = None


def _set_active_world(world: World) -> None:
    global _last_active_world
    _last_active_world = weakref.ref(world)


def _get_active_world() -> World | None:
    if _last_active_world is None:
        return None
    return _last_active_world()


def normalize_direction(
    direction: Position | Velocity | Mapping[str, Any] | tuple[float, float] | list[float] | str | int | float,
) -> tuple[float, float]:
    """Convert a direction vector, cardinal name, or radian angle into a unit (ux, uy) pair."""
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key not in _DIRECTION_NAMES:
            raise ValueError(f"unknown direction string: {direction!r}")
        return _DIRECTION_NAMES[key]

    if isinstance(direction, int | float) and not isinstance(direction, bool):
        angle = float(direction)
        if not math.isfinite(angle):
            raise ValueError("direction angle must be finite")
        return (math.cos(angle), math.sin(angle))

    if isinstance(direction, Position | Velocity):
        dx, dy = direction.x, direction.y
    elif isinstance(direction, Mapping):
        if "x" in direction and "y" in direction:
            dx, dy = float(direction["x"]), float(direction["y"])
        elif "dx" in direction and "dy" in direction:
            dx, dy = float(direction["dx"]), float(direction["dy"])
        elif "vx" in direction and "vy" in direction:
            dx, dy = float(direction["vx"]), float(direction["vy"])
        else:
            raise ValueError("direction mapping must contain ('x', 'y') or ('dx', 'dy')")
    elif isinstance(direction, tuple | list) and len(direction) == 2:
        dx, dy = float(direction[0]), float(direction[1])
    else:
        raise TypeError(
            "direction must be a (dx, dy) pair, Position, mapping, direction string, or angle"
        )

    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError("direction components must be finite")
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def extract_xy(
    pos: Position | Velocity | WorldObject | AgentBody | Mapping[str, Any] | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Extract finite (x, y) coordinates from a position-like value or entity."""
    if isinstance(pos, Position | Velocity):
        return (pos.x, pos.y)
    if hasattr(pos, "position") and isinstance(pos.position, Position):
        return (pos.position.x, pos.position.y)
    if isinstance(pos, Mapping):
        return (float(pos["x"]), float(pos["y"]))
    if isinstance(pos, tuple | list) and len(pos) == 2:
        return (float(pos[0]), float(pos[1]))
    raise TypeError("expected a Position, {'x', 'y'} mapping, (x, y) pair, WorldObject, or AgentBody")


class Tile(BaseModel):
    """One terrain grid cell."""

    model_config = ConfigDict(extra="forbid")

    type: TileType = "grass"
    flow: Velocity = Field(default_factory=Velocity)

    @model_validator(mode="before")
    @classmethod
    def _coerce_tile(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = {"type": data}
        if isinstance(data, Mapping):
            d = dict(data)
            if d.get("type") == "water" and "flow" not in d:
                d["flow"] = {"x": DEFAULT_WATER_FLOW[0], "y": DEFAULT_WATER_FLOW[1]}
            return d
        return data

    @property
    def kind(self) -> TileType:
        """Alias for ``type``."""
        return self.type

    @property
    def tile_type(self) -> TileType:
        """Alias for ``type``."""
        return self.type

    @property
    def wet(self) -> bool:
        """Water tiles are wet."""
        return self.type == "water"

    @property
    def level(self) -> str:
        """Ledge tiles are at the ``raised`` height level; other tiles are ``ground``."""
        return "raised" if self.type == "ledge" else "ground"

    @property
    def elevation(self) -> float:
        """Base elevation of this terrain tile."""
        return LEDGE_HEIGHT if self.type == "ledge" else 0.0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.type == other
        return super().__eq__(other)


class TerrainGrid:
    """2D tile grid supporting at least ``grass``, ``water`` (with flow), and ``ledge``."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        tile_size: float = 1.0,
        default_type: TileType = "grass",
        tiles: Mapping[tuple[int, int] | str, Tile | TileType | Mapping[str, Any]] | None = None,
    ) -> None:
        if tile_size <= 0 or not math.isfinite(tile_size):
            raise ValueError("tile_size must be a finite number greater than 0")
        if default_type not in TILE_TYPES:
            raise ValueError(f"default_type must be one of {sorted(TILE_TYPES)}")
        if width is not None and width <= 0:
            raise ValueError("width must be positive when specified")
        if height is not None and height <= 0:
            raise ValueError("height must be positive when specified")

        self.width = width
        self.height = height
        self.tile_size = float(tile_size)
        self.default_type: TileType = default_type
        self._tiles: dict[tuple[int, int], Tile] = {}

        if tiles is not None:
            for key, value in tiles.items():
                if isinstance(key, str):
                    parts = key.split(",")
                    if len(parts) != 2:
                        raise ValueError(f"invalid tile coordinate key: {key!r}")
                    coord = (int(parts[0]), int(parts[1]))
                elif isinstance(key, tuple) and len(key) == 2:
                    coord = (int(key[0]), int(key[1]))
                else:
                    raise TypeError("tile keys must be (tx, ty) tuples or 'tx,ty' strings")
                self._tiles[coord] = Tile.model_validate(value)

    def world_to_cell(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> tuple[int, int]:
        """Convert world coordinates ``(x, y)`` into integer grid cell indices ``(tx, ty)``."""
        if y is None:
            wx, wy = extract_xy(x_or_pos)  # type: ignore[arg-type]
        else:
            wx, wy = float(x_or_pos), float(y)  # type: ignore[arg-type]
        return (math.floor(wx / self.tile_size), math.floor(wy / self.tile_size))

    def set_tile(
        self,
        x_or_pos: int | float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y_or_tile: int | float | Tile | TileType | Mapping[str, Any] | None = None,
        tile: Tile | TileType | Mapping[str, Any] | None = None,
        *,
        flow: Velocity | Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
    ) -> Tile:
        """Set the tile at ``(x, y)`` or ``pos`` and return the validated ``Tile``."""
        if tile is None:
            if y_or_tile is None:
                raise ValueError("must provide tile type or Tile")
            cell = self.world_to_cell(x_or_pos)  # type: ignore[arg-type]
            raw_tile = y_or_tile
        else:
            if not isinstance(x_or_pos, int | float) or not isinstance(y_or_tile, int | float):
                raise TypeError("when passing 3 positional args, first two must be x and y")
            cell = self.world_to_cell(float(x_or_pos), float(y_or_tile))
            raw_tile = tile

        if isinstance(raw_tile, Tile):
            validated = raw_tile.model_copy(deep=True)
            if flow is not None:
                validated = validated.model_copy(update={"flow": Velocity.model_validate(flow)})
        elif isinstance(raw_tile, str):
            payload: dict[str, Any] = {"type": raw_tile}
            if flow is not None:
                payload["flow"] = Velocity.model_validate(flow)
            validated = Tile.model_validate(payload)
        elif isinstance(raw_tile, Mapping):
            payload = dict(raw_tile)
            if flow is not None:
                payload["flow"] = Velocity.model_validate(flow)
            validated = Tile.model_validate(payload)
        else:
            raise TypeError("tile must be a Tile, tile type string, or mapping")

        if validated.type == self.default_type and validated.flow.speed == 0.0:
            self._tiles.pop(cell, None)
        else:
            self._tiles[cell] = validated
        return validated

    def set_rect(
        self,
        x0: int | float,
        y0: int | float,
        x1: int | float,
        y1: int | float,
        tile: Tile | TileType | Mapping[str, Any],
        *,
        flow: Velocity | Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
    ) -> None:
        """Fill a rectangular region of grid cells ``[x0..x1, y0..y1]`` inclusive."""
        cx0, cy0 = self.world_to_cell(float(x0), float(y0))
        cx1, cy1 = self.world_to_cell(float(x1), float(y1))
        for tx in range(min(cx0, cx1), max(cx0, cx1) + 1):
            for ty in range(min(cy0, cy1), max(cy0, cy1) + 1):
                self.set_tile(
                    (tx + 0.5) * self.tile_size,
                    (ty + 0.5) * self.tile_size,
                    tile,
                    flow=flow,
                )

    def get_tile(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> Tile:
        """Return the ``Tile`` at ``(x, y)`` or ``pos``."""
        cell = self.world_to_cell(x_or_pos, y)
        existing = self._tiles.get(cell)
        if existing is not None:
            return existing
        return Tile(type=self.default_type)

    def tile_at(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> Tile:
        """Alias for ``get_tile``."""
        return self.get_tile(x_or_pos, y)

    def tile_type_at(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> TileType:
        """Return the tile type string (``grass``, ``water``, or ``ledge``) at ``pos``."""
        return self.get_tile(x_or_pos, y).type

    def flow_at(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> Velocity:
        """Return the flow velocity vector at ``pos``."""
        return self.get_tile(x_or_pos, y).flow

    def is_water(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> bool:
        """Return True when the tile at ``pos`` is ``water``."""
        return self.get_tile(x_or_pos, y).type == "water"

    def is_ledge(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> bool:
        """Return True when the tile at ``pos`` is ``ledge``."""
        return self.get_tile(x_or_pos, y).type == "ledge"

    def can_reach(
        self,
        target_pos: Position | Mapping[str, Any] | tuple[float, float] | list[float],
        *,
        from_pos: Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
        from_level: str = "ground",
        objects: Mapping[str, WorldObject] | Iterable[WorldObject] | None = None,
    ) -> bool:
        """Return True if ``target_pos`` is reachable from ``from_pos`` and ``from_level``.

        A ``ledge`` tile (raised level) is reachable only when:
        - Already at ``raised`` level (e.g. on an adjacent raised/ledge tile), or
        - Climbing from ``on_top_of:object_<n>`` where the supporting object has
          sufficient height and lies adjacent to the ledge.
        """
        dest_tile = self.get_tile(target_pos)
        if dest_tile.type != "ledge":
            return True

        # Reaching a ledge tile:
        if from_level == "raised":
            if from_pos is None:
                return True
            OriginCell = self.world_to_cell(from_pos)
            TargetCell = self.world_to_cell(target_pos)
            # Reachable from the same ledge or an adjacent raised tile.
            if max(abs(OriginCell[0] - TargetCell[0]), abs(OriginCell[1] - TargetCell[1])) <= 1:
                if self.get_tile(from_pos).type == "ledge" or from_level == "raised":
                    return True
            return self.get_tile(from_pos).type == "ledge"

        if from_level.startswith("on_top_of:"):
            support_id = from_level.removeprefix("on_top_of:")
            if objects is None:
                return True
            obj_map: dict[str, WorldObject]
            if isinstance(objects, Mapping):
                obj_map = dict(objects)
            else:
                obj_map = {o.id: o for o in objects}
            support = obj_map.get(support_id)
            if support is None:
                return False
            if support.height < LEDGE_CLIMB_MIN_HEIGHT:
                return False
            tx, ty = extract_xy(target_pos)
            dist = math.hypot(support.position.x - tx, support.position.y - ty)
            return dist <= self.tile_size * 1.75 + support.radius

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the terrain grid for ``World.snapshot()``."""
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "default_type": self.default_type,
            "tiles": {
                f"{tx},{ty}": tile.model_dump(mode="json")
                for (tx, ty), tile in sorted(self._tiles.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerrainGrid:
        """Restore a ``TerrainGrid`` from ``to_dict()`` output."""
        return cls(
            width=data.get("width"),
            height=data.get("height"),
            tile_size=float(data.get("tile_size", 1.0)),
            default_type=data.get("default_type", "grass"),
            tiles=data.get("tiles"),
        )


def objects_overlap(a: WorldObject, b: WorldObject, *, tolerance: float = 1e-5) -> bool:
    """Return True when two objects' collision circles overlap on the 2D plane."""
    if a.id == b.id:
        return False
    dist = math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)
    return dist < (object_radius(a) + object_radius(b) - tolerance)


def is_stack_stable(upper: WorldObject, lower: WorldObject) -> bool:
    """Return True if ``upper`` can stably rest ``on_top_of`` ``lower`` at rest."""
    if upper.id == lower.id:
        return False
    if SHAPE_STABILITY.get(lower.shape, 0.0) <= 0.0:
        return False
    if upper.radius > lower.radius * 1.5 + 1e-6:
        return False
    if not lower.fixed and upper.weight > max(0.05, lower.weight) * 2.0:
        return False
    if lower.rigidity < 0.2 and upper.weight > lower.weight:
        return False
    return True


def max_carry_step(upper: WorldObject, lower: WorldObject) -> float:
    """Return the maximum distance ``lower`` can move in one step while carrying ``upper``."""
    if not is_stack_stable(upper, lower):
        return 0.0
    base_stability = SHAPE_STABILITY.get(lower.shape, 0.5)
    upper_shape_factor = 0.65 if upper.shape in ("sphere", "cylinder") else 1.0
    if lower.fixed:
        weight_factor = 1.0
    else:
        weight_factor = min(1.0, max(0.1, lower.weight) / max(0.1, upper.weight))
    return base_stability * upper_shape_factor * weight_factor * 1.5


def compute_impulse_delta(
    obj: WorldObject,
    direction: Position | Velocity | Mapping[str, Any] | tuple[float, float] | list[float] | str | int | float,
    force: float,
) -> Velocity:
    """Compute the velocity change resulting from ``apply_impulse`` on ``obj``."""
    if isinstance(force, bool) or not isinstance(force, int | float):
        raise TypeError("force must be a number")
    force_val = float(force)
    if not math.isfinite(force_val) or force_val < 0.0:
        raise ValueError("force must be a finite non-negative number")

    ux, uy = normalize_direction(direction)
    if obj.fixed or force_val == 0.0 or (ux == 0.0 and uy == 0.0):
        return Velocity(x=0.0, y=0.0)

    shape_factor = SHAPE_IMPULSE_FACTOR.get(obj.shape, 0.75)
    # Check if the object has a push BehaviorRule with a custom scale
    rule_scale = 1.0
    for rule in obj.behavior:
        if rule.verb == "push":
            extras = rule.__pydantic_extra__ or {}
            scale_val = extras.get("scale")
            if isinstance(scale_val, int | float) and not isinstance(scale_val, bool) and scale_val > 0:
                # Normalize around default push scale of 4 so standard rules stay consistent
                rule_scale = float(scale_val) / 4.0
                break

    dv = (force_val * BASE_IMPULSE * shape_factor * rule_scale) / (1.0 + obj.weight)
    return Velocity(x=ux * dv, y=uy * dv)


def apply_impulse(
    world_or_id: World | str,
    id_or_direction: Any,
    direction_or_force: Any = None,
    force: float | None = None,
) -> WorldObject:
    """Apply an impulse to an object.

    Supports both ``apply_impulse(world, object_id, direction, force)`` and
    ``apply_impulse(object_id, direction, force)`` (operating on the active ``World``).
    """
    if isinstance(world_or_id, str):
        world = _get_active_world()
        if world is None:
            raise RuntimeError("no active World found; call world.apply_impulse(...) or pass world")
        if direction_or_force is None:
            raise TypeError("apply_impulse(id, direction, force) requires 3 arguments")
        return world.apply_impulse(world_or_id, id_or_direction, float(direction_or_force))

    if force is None:
        raise TypeError("apply_impulse(world, id, direction, force) requires 4 arguments")
    return world_or_id.apply_impulse(str(id_or_direction), direction_or_force, float(force))


def _object_elevation(
    obj: WorldObject,
    objects_by_id: Mapping[str, WorldObject],
    terrain: TerrainGrid | None = None,
    _visited: set[str] | None = None,
) -> float:
    """Compute the base elevation of ``obj`` from its level and parent stack."""
    if _visited is None:
        _visited = set()
    if obj.id in _visited:
        return 0.0
    _visited.add(obj.id)

    if obj.level == "raised":
        return LEDGE_HEIGHT
    if obj.level == "ground":
        if terrain is not None and terrain.is_ledge(obj.position):
            return LEDGE_HEIGHT
        return 0.0
    if obj.level.startswith("on_top_of:"):
        parent_id = obj.level.removeprefix("on_top_of:")
        parent = objects_by_id.get(parent_id)
        if parent is None:
            return 0.0
        return _object_elevation(parent, objects_by_id, terrain, _visited) + parent.height
    if obj.level.startswith("inside:"):
        parent_id = obj.level.removeprefix("inside:")
        parent = objects_by_id.get(parent_id)
        if parent is None:
            return 0.0
        return _object_elevation(parent, objects_by_id, terrain, _visited)
    return 0.0


def object_top_height(
    obj: WorldObject,
    objects_by_id: Mapping[str, WorldObject] | None = None,
    terrain: TerrainGrid | None = None,
) -> float:
    """Return the top elevation (base elevation + object height) of ``obj``."""
    obj_map = objects_by_id if objects_by_id is not None else {}
    return _object_elevation(obj, obj_map, terrain) + obj.height


def is_occluded(
    viewer_or_world: Any,
    target_or_viewer: Any,
    obstacles_or_target: Any = None,
    *,
    tolerance: float = DEFAULT_OCCLUSION_TOLERANCE,
    world: World | None = None,
) -> bool:
    """Approximate line-of-sight occlusion helper.

    Returns ``True`` when a taller object lies close to the line segment between
    ``viewer_pos`` and ``target``.

    Supports calling conventions:
    - ``is_occluded(viewer_pos, target)`` (uses active ``World``)
    - ``is_occluded(viewer_pos, target, obstacles)``
    - ``is_occluded(world, viewer_pos, target)``
    - ``world.is_occluded(viewer_pos, target)``
    """
    from arden.world.world import World as WorldCls

    if isinstance(viewer_or_world, WorldCls):
        return viewer_or_world.is_occluded(target_or_viewer, obstacles_or_target, tolerance=tolerance)

    viewer_pos = viewer_or_world
    target = target_or_viewer
    obstacles = obstacles_or_target

    resolved_world: World | None = world
    if isinstance(obstacles, WorldCls):
        resolved_world = obstacles
        obstacles = obstacles.objects

    if resolved_world is None and obstacles is None:
        resolved_world = _get_active_world()

    obj_map: dict[str, WorldObject] = {}
    if isinstance(obstacles, Mapping):
        for k, v in obstacles.items():
            if isinstance(v, WorldObject):
                obj_map[v.id] = v
            elif isinstance(v, Mapping):
                parsed = WorldObject.model_validate(v)
                obj_map[parsed.id] = parsed
    elif isinstance(obstacles, Iterable) and not isinstance(obstacles, str | bytes):
        for item in obstacles:
            if isinstance(item, WorldObject):
                obj_map[item.id] = item
            elif isinstance(item, Mapping):
                parsed = WorldObject.model_validate(item)
                obj_map[parsed.id] = parsed
    elif resolved_world is not None:
        obj_map = dict(resolved_world.objects)

    terrain = resolved_world.terrain if resolved_world is not None else None
    held_id = resolved_world.body.holding if resolved_world is not None else None

    # Resolve viewer coordinates and elevation
    viewer_id: str | None = None
    viewer_elevation: float = 0.0
    if isinstance(viewer_pos, str):
        if viewer_pos not in obj_map:
            raise KeyError(f"unknown viewer object id: {viewer_pos}")
        viewer_obj = obj_map[viewer_pos]
        viewer_id = viewer_obj.id
        vx, vy = viewer_obj.position.x, viewer_obj.position.y
        viewer_elevation = _object_elevation(viewer_obj, obj_map, terrain)
    elif isinstance(viewer_pos, WorldObject):
        viewer_id = viewer_pos.id
        vx, vy = viewer_pos.position.x, viewer_pos.position.y
        viewer_elevation = _object_elevation(viewer_pos, obj_map, terrain)
    else:
        vx, vy = extract_xy(viewer_pos)
        if hasattr(viewer_pos, "level") and isinstance(viewer_pos.level, str):
            if viewer_pos.level == "raised":
                viewer_elevation = LEDGE_HEIGHT
        elif terrain is not None and terrain.is_ledge((vx, vy)):
            viewer_elevation = LEDGE_HEIGHT

    # Resolve target coordinates and top height
    target_id: str | None = None
    target_obj: WorldObject | None = None
    if isinstance(target, str):
        if target not in obj_map:
            raise KeyError(f"unknown target object id: {target}")
        target_obj = obj_map[target]
        target_id = target_obj.id
        tx, ty = target_obj.position.x, target_obj.position.y
        target_height = object_top_height(target_obj, obj_map, terrain)
    elif isinstance(target, WorldObject):
        target_obj = target
        target_id = target.id
        obj_map.setdefault(target.id, target)
        tx, ty = target.position.x, target.position.y
        target_height = object_top_height(target, obj_map, terrain)
    else:
        tx, ty = extract_xy(target)
        # If an object sits at this exact target position, use its height
        target_height = 0.0
        for candidate in obj_map.values():
            if math.hypot(candidate.position.x - tx, candidate.position.y - ty) <= 1e-6:
                target_obj = candidate
                target_id = candidate.id
                target_height = object_top_height(candidate, obj_map, terrain)
                break

    seg_x = tx - vx
    seg_y = ty - vy
    seg_len_sq = seg_x * seg_x + seg_y * seg_y
    if seg_len_sq <= 1e-12:
        return False

    for candidate in obj_map.values():
        if candidate.id == target_id or candidate.id == viewer_id:
            continue
        if held_id is not None and candidate.id == held_id:
            continue
        if target_obj is not None:
            if target_obj.level in (f"on_top_of:{candidate.id}", f"inside:{candidate.id}"):
                continue
            if candidate.level in (f"on_top_of:{target_obj.id}", f"inside:{target_obj.id}"):
                continue

        cand_top = object_top_height(candidate, obj_map, terrain)
        # Must be strictly taller than the target (and taller than viewer base elevation)
        if cand_top <= target_height or cand_top <= viewer_elevation:
            continue

        cx, cy = candidate.position.x, candidate.position.y
        # Projection parameter t along viewer -> target segment
        t = ((cx - vx) * seg_x + (cy - vy) * seg_y) / seg_len_sq
        if t <= 0.0 or t >= 1.0:
            continue

        closest_x = vx + t * seg_x
        closest_y = vy + t * seg_y
        perp_dist = math.hypot(cx - closest_x, cy - closest_y)
        line_threshold = max(object_radius(candidate), tolerance)
        if perp_dist <= line_threshold:
            return True

    return False
