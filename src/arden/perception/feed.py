"""Structured perception feed, delta tracker, look focus, and prompt text rendering."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer

from arden.config import PerceptionSettings, Settings, load_config
from arden.llm.schemas import VERBS
from arden.world.objects import PERCEPTIBLE_STATE_KEYS, Position, WorldObject
from arden.world.physics import _get_active_world, _set_active_world
from arden.world.world import AgentBody, World, WorldEvent

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_OPAQUE_ACTION = re.compile(r"^action_[0-9]+$")
_BODY_VERBS: frozenset[str] = frozenset(VERBS)

_COMPASS_BEARINGS: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

_PERCEPTIBLE_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "position",
        "level",
        "shape",
        "height",
        "volume",
        "weight",
        "rigidity",
        "binding",
        "hollow",
        "fragile",
        "color",
        "fixed",
        "state",
    }
)

PerceivedEventType = Literal[
    "object_spawned",
    "object_despawned",
    "object_moved",
    "object_changed",
    "appeared",
    "disappeared",
    "moved",
    "changed",
]


def _fmt_num(value: float | int) -> str:
    """Format a finite number cleanly without trailing zeros or ``-0``."""
    rounded = round(float(value), 4)
    if rounded == 0.0:
        return "0"
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _fmt_pos(pos: Position) -> str:
    """Format a 2D position as ``(x,y)``."""
    return f"({_fmt_num(pos.x)},{_fmt_num(pos.y)})"


def _object_sort_key(object_id: str) -> tuple[int, str]:
    if object_id.startswith("object_"):
        suffix = object_id.removeprefix("object_")
        if suffix.isdigit():
            return (int(suffix), object_id)
    return (10**9, object_id)


def compass_bearing(from_pos: Position, to_pos: Position) -> str:
    """Return the 8-wind compass bearing (or ``here``) from ``from_pos`` to ``to_pos``."""
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y
    if math.hypot(dx, dy) <= 1e-9:
        return "here"
    bearing_deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    sector = int((bearing_deg + 22.5) // 45.0) % 8
    return _COMPASS_BEARINGS[sector]


class AffordanceMap(dict[str, list[str]]):
    """Mapping of ``object_id -> list[verb_or_action]`` within reach.

    Supports both ``"object_7" in affordances`` and ``"action_3" in affordances``.
    """

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str):
            return any(item in actions for actions in self.values())
        return False


def _extract_object_affordances(
    obj: WorldObject,
    body: AgentBody,
    opaque_registry: dict[tuple[str, str], str] | None = None,
) -> list[str]:
    """Derive perceptible affordances for an object within reach.

    Standard body verbs and opaque ``action_<n>`` names are returned as bare
    tokens with no descriptions or hidden rule parameters.
    """
    actions: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        if token not in seen:
            seen.add(token)
            actions.append(token)

    # Physical body affordances from perceptible physical properties
    if body.holding == obj.id:
        for verb in ("drop", "place", "stack"):
            _add(verb)
    elif not obj.fixed:
        if body.holding is None:
            _add("grab")
        _add("push")
        _add("strike")

    if obj.binding:
        _add("bind")
    if obj.hollow:
        _add("insert")

    # Affordances declared in behavior rules (opaque actions appear as action_<n>)
    next_opaque_idx = 1
    if opaque_registry:
        for mapped in opaque_registry.values():
            if _OPAQUE_ACTION.fullmatch(mapped):
                num = int(mapped.removeprefix("action_"))
                if num >= next_opaque_idx:
                    next_opaque_idx = num + 1

    for rule in obj.behavior:
        verb = rule.verb
        extras = rule.__pydantic_extra__ or {}
        explicit_opaque: str | None = None
        for key in ("action", "action_id", "opaque_id", "opaque_action"):
            candidate = extras.get(key)
            if isinstance(candidate, str) and _OPAQUE_ACTION.fullmatch(candidate):
                explicit_opaque = candidate
                break

        if explicit_opaque is not None:
            _add(explicit_opaque)
        elif _OPAQUE_ACTION.fullmatch(verb) or verb in _BODY_VERBS:
            _add(verb)
        else:
            # Non-physical verb with no physical analogue: expose only as action_<n>
            reg_key = (obj.id, verb)
            if opaque_registry is not None and reg_key in opaque_registry:
                opaque_token = opaque_registry[reg_key]
            else:
                opaque_token = f"action_{next_opaque_idx}"
                next_opaque_idx += 1
                if opaque_registry is not None:
                    opaque_registry[reg_key] = opaque_token
            _add(opaque_token)

    return sorted(actions)


class BodyPercept(BaseModel):
    """Perceptible body state as plain numbers and ids, with no evaluative labels."""

    model_config = ConfigDict(extra="forbid")

    position: Position
    level: str = "ground"
    energy: float = Field(ge=0.0, le=1.0)
    comfort: float = Field(ge=0.0, le=1.0)
    holding: str | None = None

    @property
    def held_item(self) -> str | None:
        """Alias for ``holding``."""
        return self.holding

    def to_dict(self) -> dict[str, Any]:
        """Serialize body state to a plain dictionary."""
        return {
            "position": {"x": self.position.x, "y": self.position.y},
            "level": self.level,
            "energy": self.energy,
            "comfort": self.comfort,
            "holding": self.holding,
        }


BodyState = BodyPercept


class PerceivedObject(BaseModel):
    """Perceptible view of a single object, with distance and compass direction."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_OBJECT_ID.pattern)
    distance: float = Field(ge=0.0)
    direction: str
    position: Position
    level: str = "ground"
    shape: Literal["sphere", "cube", "cylinder", "flat", "long"]
    height: float = Field(ge=0.0)
    volume: float = Field(ge=0.0)
    weight: float = Field(ge=0.0)
    rigidity: float = Field(ge=0.0, le=1.0)
    binding: bool = False
    hollow: bool = False
    fragile: bool = False
    color: str | None = None
    fixed: bool = False
    state: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    affordances: list[str] = Field(default_factory=list)

    @property
    def objects(self) -> list[PerceivedObject]:
        """Single-element list containing this object percept."""
        return [self]

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler: Any) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("color") is None:
            data.pop("color", None)
        if isinstance(data.get("properties"), dict) and data["properties"].get("color") is None:
            data["properties"].pop("color", None)
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict, omitting ``color`` when None/dropped."""
        data = self.model_dump(mode="json")
        if data.get("color") is None:
            data.pop("color", None)
        if isinstance(data.get("properties"), dict) and data["properties"].get("color") is None:
            data["properties"].pop("color", None)
        return data

    def to_json(self) -> str:
        """Serialize this object percept to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_text(self, *, detailed: bool = False) -> str:
        """Render a compact, deterministic text line for this object."""
        parts = [
            f"{self.id}",
            f"dist={_fmt_num(self.distance)}",
            f"dir={self.direction}",
            f"pos={_fmt_pos(self.position)}",
            f"level={self.level}",
            f"shape={self.shape}",
            f"h={_fmt_num(self.height)}",
            f"vol={_fmt_num(self.volume)}",
            f"w={_fmt_num(self.weight)}",
            f"rig={_fmt_num(self.rigidity)}",
        ]
        if self.color is not None:
            parts.append(f"color={self.color}")
        if detailed:
            parts.append(f"binding={str(self.binding).lower()}")
            parts.append(f"hollow={str(self.hollow).lower()}")
            parts.append(f"fragile={str(self.fragile).lower()}")
            parts.append(f"fixed={str(self.fixed).lower()}")
        else:
            if self.binding:
                parts.append("binding=true")
            if self.hollow:
                parts.append("hollow=true")
            if self.fragile:
                parts.append("fragile=true")
            if self.fixed:
                parts.append("fixed=true")
        if self.state or detailed:
            parts.append(f"state={json.dumps(self.state, sort_keys=True, separators=(',', ':'))}")
        return " ".join(parts)

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        return self.properties[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.to_dict() or key in self.properties

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        return self.properties.get(key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.to_dict().keys())


ObjectPercept = PerceivedObject


class PerceivedEvent(BaseModel):
    """Perceptible world or delta event, stripped of hidden rules and creator labels."""

    model_config = ConfigDict(extra="forbid")

    type: PerceivedEventType
    tick: int = Field(default=0, ge=0)
    object_id: str = Field(pattern=_OBJECT_ID.pattern)
    position: Position | None = None
    previous_position: Position | None = None
    distance: float | None = None
    direction: str | None = None
    changes: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None

    @property
    def kind(self) -> PerceivedEventType:
        """Alias for ``type``."""
        return self.type

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a clean dictionary omitting unset optional fields."""
        data = self.model_dump(mode="json", exclude_none=True)
        if "properties" in data and isinstance(data["properties"], dict):
            if data["properties"].get("color") is None:
                data["properties"].pop("color", None)
        return data

    def to_text(self) -> str:
        """Render a compact one-line summary of the event."""
        kind = self.type.removeprefix("object_")
        parts = [kind, self.object_id]
        if self.previous_position is not None and self.position is not None:
            parts.append(f"from={_fmt_pos(self.previous_position)}")
            parts.append(f"to={_fmt_pos(self.position)}")
        elif self.position is not None:
            parts.append(f"pos={_fmt_pos(self.position)}")
        elif self.previous_position is not None:
            parts.append(f"pos={_fmt_pos(self.previous_position)}")
        if self.distance is not None:
            parts.append(f"dist={_fmt_num(self.distance)}")
        if self.direction is not None:
            parts.append(f"dir={self.direction}")
        if self.changes:
            parts.append(json.dumps(self.changes, sort_keys=True, separators=(",", ":")))
        return " ".join(parts)


EventPercept = PerceivedEvent


class Percept(BaseModel):
    """Structured perception feed delivered to the agent."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tick: int = Field(default=0, ge=0)
    body: BodyPercept
    objects: list[PerceivedObject] = Field(default_factory=list)
    events: list[PerceivedEvent] = Field(default_factory=list)
    affordances: dict[str, list[str]] = Field(default_factory=AffordanceMap)
    is_delta: bool = False
    appeared: list[PerceivedObject] = Field(default_factory=list)
    disappeared: list[str] = Field(default_factory=list)
    moved: list[PerceivedObject] = Field(default_factory=list)
    changed: list[PerceivedObject] = Field(default_factory=list)

    _empty_delta: bool = PrivateAttr(default=False)

    @field_validator("affordances", mode="after")
    @classmethod
    def _wrap_affordances(cls, value: dict[str, list[str]]) -> AffordanceMap:
        if isinstance(value, AffordanceMap):
            return value
        return AffordanceMap(value)

    @property
    def nearby_objects(self) -> list[PerceivedObject]:
        """Alias for ``objects``."""
        return self.objects

    @property
    def body_state(self) -> BodyPercept:
        """Alias for ``body``."""
        return self.body

    @property
    def is_empty(self) -> bool:
        """Return True when this is a delta percept with no changes since the last call."""
        if self.is_delta:
            return self._empty_delta or (
                not self.objects
                and not self.events
                and not self.appeared
                and not self.disappeared
                and not self.moved
                and not self.changed
            )
        return False

    def __bool__(self) -> bool:
        if self.is_delta:
            return not self.is_empty
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the percept to a JSON-ready dictionary."""
        result: dict[str, Any] = {
            "tick": self.tick,
            "body": self.body.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "events": [ev.to_dict() for ev in self.events],
            "affordances": {k: list(v) for k, v in sorted(self.affordances.items(), key=lambda kv: _object_sort_key(kv[0]))},
        }
        if self.is_delta:
            result["is_delta"] = True
            result["appeared"] = [obj.to_dict() for obj in self.appeared]
            result["disappeared"] = list(self.disappeared)
            result["moved"] = [obj.to_dict() for obj in self.moved]
            result["changed"] = [obj.to_dict() for obj in self.changed]
        return result

    def to_json(self) -> str:
        """Serialize the percept to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_text(self) -> str:
        """Render a short, stable, token-cheap text representation for LLM prompts."""
        if self.is_delta and self.is_empty:
            return ""

        held_str = self.body.holding if self.body.holding is not None else "none"
        lines: list[str] = [
            f"tick: {self.tick}",
            (
                f"body: pos={_fmt_pos(self.body.position)} "
                f"level={self.body.level} "
                f"energy={_fmt_num(self.body.energy)} "
                f"comfort={_fmt_num(self.body.comfort)} "
                f"holding={held_str}"
            ),
        ]

        if self.objects:
            lines.append("objects:")
            for obj in self.objects:
                lines.append(f"- {obj.to_text()}")
        else:
            lines.append("objects: none")

        if self.affordances:
            lines.append("affordances:")
            for obj_id in sorted(self.affordances, key=_object_sort_key):
                verbs = self.affordances[obj_id]
                if verbs:
                    lines.append(f"- {obj_id}: {', '.join(verbs)}")
        else:
            lines.append("affordances: none")

        if self.events:
            lines.append("events:")
            for ev in self.events:
                lines.append(f"- {ev.to_text()}")
        else:
            lines.append("events: none")

        return "\n".join(lines)


def _resolve_perception_settings(
    config: Settings | PerceptionSettings | Mapping[str, Any] | None = None,
    world: World | None = None,
    *,
    sensing_radius: float | None = None,
    reach_radius: float | None = None,
    move_threshold: float | None = None,
    color_visibility_threshold: float | None = None,
) -> PerceptionSettings:
    """Resolve effective perception settings from explicit args, world, or config."""
    base: PerceptionSettings | None = None

    if config is not None:
        if isinstance(config, PerceptionSettings):
            base = config.model_copy()
        elif isinstance(config, Settings):
            base = config.perception.model_copy()
            if config.world.sensing_radius is not None:
                base.sensing_radius = config.world.sensing_radius
        elif isinstance(config, Mapping):
            if "perception" in config and isinstance(config["perception"], Mapping):
                base = PerceptionSettings.model_validate(config["perception"])
            else:
                base = PerceptionSettings.model_validate(dict(config))
        else:
            raise TypeError("config must be Settings, PerceptionSettings, a mapping, or None")

    if base is None and world is not None:
        world_cfg = getattr(world, "config", None) or getattr(world, "settings", None)
        if isinstance(world_cfg, Settings):
            base = world_cfg.perception.model_copy()
            if world_cfg.world.sensing_radius is not None:
                base.sensing_radius = world_cfg.world.sensing_radius
        elif isinstance(world_cfg, PerceptionSettings):
            base = world_cfg.model_copy()

    if base is None:
        try:
            loaded = load_config()
            base = loaded.perception.model_copy()
            if loaded.world.sensing_radius is not None:
                base.sensing_radius = loaded.world.sensing_radius
        except Exception:
            base = PerceptionSettings()

    if world is not None:
        world_sr = getattr(world, "sensing_radius", None)
        if isinstance(world_sr, int | float) and not isinstance(world_sr, bool):
            base.sensing_radius = float(world_sr)

    overrides: dict[str, Any] = {}
    if sensing_radius is not None:
        overrides["sensing_radius"] = float(sensing_radius)
    if reach_radius is not None:
        overrides["reach_radius"] = float(reach_radius)
    if move_threshold is not None:
        overrides["move_threshold"] = float(move_threshold)
    if color_visibility_threshold is not None:
        overrides["color_visibility_threshold"] = float(color_visibility_threshold)

    if overrides:
        merged = {**base.model_dump(), **overrides}
        return PerceptionSettings.model_validate(merged)
    return base


def _validate_visibility_factor(visibility_factor: float) -> float:
    if isinstance(visibility_factor, bool) or not isinstance(visibility_factor, int | float):
        raise TypeError("visibility_factor must be a number")
    vf = float(visibility_factor)
    if not math.isfinite(vf) or vf < 0.0 or vf > 1.0:
        raise ValueError("visibility_factor must be a finite number in [0.0, 1.0]")
    return vf


def _should_keep_color(
    distance: float,
    *,
    base_sensing_radius: float,
    visibility_factor: float,
    color_threshold: float,
) -> bool:
    """Return True if fine detail such as color is perceptible at ``distance`` and ``visibility_factor``."""
    if visibility_factor >= 1.0:
        return True
    if visibility_factor <= color_threshold:
        return False
    # Between color_threshold and 1.0, distant objects near the edge of the
    # shrunk radius lose fine detail (color) before closer objects do.
    detail_radius = base_sensing_radius * (visibility_factor * visibility_factor)
    return distance <= detail_radius + 1e-9


def _build_perceived_object(
    obj: WorldObject,
    body: AgentBody,
    *,
    keep_color: bool,
    affordances: list[str] | None = None,
) -> PerceivedObject:
    """Build a ``PerceivedObject`` from ``obj`` relative to ``body``."""
    raw_props = obj.perceptible()
    if not keep_color:
        raw_props.pop("color", None)
    color_val = raw_props.get("color")

    dist = round(
        math.hypot(obj.position.x - body.position.x, obj.position.y - body.position.y),
        6,
    )
    bearing = compass_bearing(body.position, obj.position)

    return PerceivedObject(
        id=obj.id,
        distance=dist,
        direction=bearing,
        position=obj.position.model_copy(),
        level=obj.level,
        shape=obj.shape,
        height=obj.height,
        volume=obj.volume,
        weight=obj.weight,
        rigidity=obj.rigidity,
        binding=obj.binding,
        hollow=obj.hollow,
        fragile=obj.fragile,
        color=color_val,
        fixed=obj.fixed,
        state=dict(raw_props.get("state", {})),
        properties=raw_props,
        affordances=list(affordances) if affordances is not None else [],
    )


def _sanitize_world_event(
    event: WorldEvent | PerceivedEvent | Mapping[str, Any],
    *,
    world: World,
    body: AgentBody,
    effective_radius: float,
    base_sensing_radius: float,
    visibility_factor: float,
    color_threshold: float,
) -> PerceivedEvent | None:
    """Convert a ``WorldEvent`` into a sanitized ``PerceivedEvent`` if perceptible."""
    if isinstance(event, PerceivedEvent):
        return event

    if isinstance(event, Mapping):
        ev_type = event.get("type")
        if ev_type == "tick" or ev_type is None:
            return None
        obj_id = event.get("object_id")
        if not isinstance(obj_id, str) or not _OBJECT_ID.fullmatch(obj_id):
            return None
        tick = int(event.get("tick", world.tick))
        pos_raw = event.get("position")
        prev_raw = event.get("previous_position")
        pos = Position.model_validate(pos_raw) if pos_raw is not None else None
        prev_pos = Position.model_validate(prev_raw) if prev_raw is not None else None
        raw_changes = event.get("changes")
    elif isinstance(event, WorldEvent):
        if event.type == "tick" or event.object_id is None:
            return None
        ev_type = event.type
        obj_id = event.object_id
        tick = event.tick
        pos = event.position.model_copy() if event.position is not None else None
        prev_pos = event.previous_position.model_copy() if event.previous_position is not None else None
        raw_changes = event.changes
    else:
        return None

    current_obj = world.objects.get(obj_id)
    if pos is None and current_obj is not None:
        pos = current_obj.position.model_copy()

    # Check whether the event occurred within sensing range (at either current or previous position)
    in_range = False
    dist: float | None = None
    direction: str | None = None

    if pos is not None:
        d_curr = math.hypot(pos.x - body.position.x, pos.y - body.position.y)
        if d_curr <= effective_radius + 1e-9:
            if current_obj is None or not world.is_occluded(body, current_obj):
                in_range = True
                dist = round(d_curr, 6)
                direction = compass_bearing(body.position, pos)

    if not in_range and prev_pos is not None:
        d_prev = math.hypot(prev_pos.x - body.position.x, prev_pos.y - body.position.y)
        if d_prev <= effective_radius + 1e-9:
            in_range = True
            dist = round(d_prev, 6)
            direction = compass_bearing(body.position, prev_pos)

    if not in_range:
        return None

    clean_changes: dict[str, Any] | None = None
    if isinstance(raw_changes, Mapping):
        keep_color = _should_keep_color(
            dist if dist is not None else 0.0,
            base_sensing_radius=base_sensing_radius,
            visibility_factor=visibility_factor,
            color_threshold=color_threshold,
        )
        filtered: dict[str, Any] = {}
        for k, v in raw_changes.items():
            if k not in _PERCEPTIBLE_KEYS or (k == "color" and not keep_color):
                continue
            if k == "state" and isinstance(v, Mapping):
                allowed_state = {
                    sk: sv for sk, sv in v.items() if sk in PERCEPTIBLE_STATE_KEYS
                }
                if v and not allowed_state:
                    continue
                filtered[k] = allowed_state
            else:
                filtered[k] = v
        if not filtered:
            return None
        clean_changes = filtered

    return PerceivedEvent(
        type=ev_type,  # type: ignore[arg-type]
        tick=tick,
        object_id=obj_id,
        position=pos,
        previous_position=prev_pos,
        distance=dist,
        direction=direction,
        changes=clean_changes,
    )


def perceive(
    world: World,
    body: AgentBody | Mapping[str, Any] | None = None,
    *,
    visibility_factor: float = 1.0,
    sensing_radius: float | None = None,
    reach_radius: float | None = None,
    color_visibility_threshold: float | None = None,
    config: Settings | PerceptionSettings | Mapping[str, Any] | None = None,
    events: Iterable[WorldEvent | PerceivedEvent | Mapping[str, Any]] | None = None,
) -> Percept:
    """Turn ``world`` and ``body`` state into a structured ``Percept``.

    - Nearby objects within ``sensing_radius * visibility_factor`` that are not
      occluded are included with perceptible properties, distance, and compass
      bearing.
    - When ``visibility_factor`` is lowered, the effective sensing radius shrinks
      and fine detail such as ``color`` is dropped before coarse geometry.
    - Body state is reported as plain numbers with no evaluative labels.
    - Affordances are included for perceived objects within ``reach_radius``,
      with opaque actions formatted as ``action_<n>`` with no description.
    """
    _set_active_world(world)
    vf = _validate_visibility_factor(visibility_factor)

    if body is None:
        agent_body = world.body
    elif isinstance(body, AgentBody):
        agent_body = body
    elif isinstance(body, Mapping):
        agent_body = AgentBody.model_validate(dict(body))
    else:
        raise TypeError("body must be an AgentBody, mapping, or None")

    cfg = _resolve_perception_settings(
        config,
        world,
        sensing_radius=sensing_radius,
        reach_radius=reach_radius,
        color_visibility_threshold=color_visibility_threshold,
    )
    effective_radius = cfg.sensing_radius * vf

    perceived_objects: list[PerceivedObject] = []
    affordances = AffordanceMap()
    opaque_registry: dict[tuple[str, str], str] = {}

    if vf > 0.0:
        for obj_id in sorted(world.objects, key=_object_sort_key):
            obj = world.objects[obj_id]
            dist = math.hypot(
                obj.position.x - agent_body.position.x,
                obj.position.y - agent_body.position.y,
            )
            is_held = agent_body.holding == obj.id
            if not is_held and dist > effective_radius + 1e-9:
                continue
            if not is_held and world.is_occluded(agent_body, obj):
                continue

            keep_color = _should_keep_color(
                dist,
                base_sensing_radius=cfg.sensing_radius,
                visibility_factor=vf,
                color_threshold=cfg.color_visibility_threshold,
            )

            in_reach = is_held or (dist <= cfg.reach_radius + 1e-9)
            obj_affordances: list[str] = []
            if in_reach:
                obj_affordances = _extract_object_affordances(obj, agent_body, opaque_registry)
                if obj_affordances:
                    affordances[obj.id] = obj_affordances

            perceived_objects.append(
                _build_perceived_object(
                    obj,
                    agent_body,
                    keep_color=keep_color,
                    affordances=obj_affordances,
                )
            )

    perceived_objects.sort(key=lambda o: (round(o.distance, 6), _object_sort_key(o.id)))

    # Collect and sanitize events since the last percept
    if events is not None:
        raw_events = list(events)
    else:
        raw_events = list(world._perception_event_log)
        world._perception_event_log.clear()
    world._has_been_perceived = True

    perceived_events: list[PerceivedEvent] = []
    for raw_ev in raw_events:
        sanitized = _sanitize_world_event(
            raw_ev,
            world=world,
            body=agent_body,
            effective_radius=effective_radius,
            base_sensing_radius=cfg.sensing_radius,
            visibility_factor=vf,
            color_threshold=cfg.color_visibility_threshold,
        )
        if sanitized is not None:
            perceived_events.append(sanitized)

    body_percept = BodyPercept(
        position=agent_body.position.model_copy(),
        level=agent_body.level,
        energy=agent_body.energy,
        comfort=agent_body.comfort,
        holding=agent_body.holding,
    )

    return Percept(
        tick=world.tick,
        body=body_percept,
        objects=perceived_objects,
        events=perceived_events,
        affordances=affordances,
        is_delta=False,
    )


def look(
    target_or_world: World | WorldObject | str,
    target: WorldObject | str | None = None,
    body: AgentBody | Mapping[str, Any] | None = None,
    *,
    world: World | None = None,
) -> PerceivedObject:
    """Return a detailed percept for a single object with all perceptible properties.

    Supports calling conventions:
    - ``look(target)`` (where ``target`` is a ``WorldObject`` or ``"object_<n>"`` in the active ``World``)
    - ``look(world, target)``
    - ``look(target, world=world)``
    - ``look(world, target, body)``
    """
    resolved_world: World | None = world
    resolved_target: WorldObject | str

    if isinstance(target_or_world, World):
        resolved_world = target_or_world
        if target is None:
            raise TypeError("look(world, target) requires a target argument")
        resolved_target = target
    else:
        resolved_target = target_or_world
        if isinstance(target, World):
            resolved_world = target
        elif isinstance(target, AgentBody | Mapping) and body is None:
            body = target

    if resolved_world is None:
        resolved_world = _get_active_world()

    if isinstance(resolved_target, WorldObject):
        obj = resolved_target
        if resolved_world is not None and obj.id in resolved_world.objects:
            obj = resolved_world.objects[obj.id]
    elif isinstance(resolved_target, str):
        if resolved_world is None:
            raise RuntimeError("no active World found to resolve object id; pass world or a WorldObject")
        if resolved_target not in resolved_world.objects:
            raise KeyError(f"unknown object id: {resolved_target}")
        obj = resolved_world.objects[resolved_target]
    else:
        raise TypeError("target must be an object_<n> id string or a WorldObject")

    if body is not None:
        agent_body = body if isinstance(body, AgentBody) else AgentBody.model_validate(dict(body))
    elif resolved_world is not None:
        agent_body = resolved_world.body
    else:
        agent_body = AgentBody(position=obj.position.model_copy())

    obj_affordances = _extract_object_affordances(obj, agent_body)
    return _build_perceived_object(
        obj,
        agent_body,
        keep_color=True,
        affordances=obj_affordances,
    )


class PerceptionTracker:
    """Stateful perception tracker supporting delta mode and on-demand full refreshes.

    In delta mode, returns only what appeared, disappeared, moved beyond
    ``move_threshold``, or changed perceptible properties since the last call.
    """

    def __init__(
        self,
        world: World | None = None,
        *,
        config: Settings | PerceptionSettings | Mapping[str, Any] | None = None,
        sensing_radius: float | None = None,
        reach_radius: float | None = None,
        move_threshold: float | None = None,
        color_visibility_threshold: float | None = None,
        visibility_factor: float = 1.0,
    ) -> None:
        self.world = world
        self.config = config
        self.sensing_radius = sensing_radius
        self.reach_radius = reach_radius
        self.move_threshold = move_threshold
        self.color_visibility_threshold = color_visibility_threshold
        self.visibility_factor = _validate_visibility_factor(visibility_factor)

        self._initialized: bool = False
        self._known_objects: dict[str, PerceivedObject] = {}
        self._last_body: BodyPercept | None = None
        self._last_affordances: AffordanceMap = AffordanceMap()

    def _resolve_world(self, world: World | None) -> World:
        if world is not None:
            self.world = world
            return world
        if self.world is not None:
            return self.world
        active = _get_active_world()
        if active is not None:
            self.world = active
            return active
        raise RuntimeError("PerceptionTracker requires a World instance")

    def full_refresh(
        self,
        world: World | None = None,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        visibility_factor: float | None = None,
    ) -> Percept:
        """Return a complete (non-delta) ``Percept`` and update the tracker baseline."""
        w = self._resolve_world(world)
        vf = self.visibility_factor if visibility_factor is None else _validate_visibility_factor(visibility_factor)
        self.visibility_factor = vf

        full = perceive(
            w,
            body,
            visibility_factor=vf,
            sensing_radius=self.sensing_radius,
            reach_radius=self.reach_radius,
            color_visibility_threshold=self.color_visibility_threshold,
            config=self.config,
        )
        self._known_objects = {obj.id: obj.model_copy(deep=True) for obj in full.objects}
        self._last_body = full.body.model_copy(deep=True)
        self._last_affordances = AffordanceMap({k: list(v) for k, v in full.affordances.items()})
        self._initialized = True
        return full

    refresh = full_refresh

    def perceive(
        self,
        world: World | None = None,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        full: bool = False,
        refresh: bool = False,
        visibility_factor: float | None = None,
    ) -> Percept:
        """Return a delta ``Percept`` since the last call (or a full percept on first call / ``full=True``)."""
        if full or refresh or not self._initialized:
            return self.full_refresh(world, body, visibility_factor=visibility_factor)
        return self.delta(world, body, visibility_factor=visibility_factor)

    def delta(
        self,
        world: World | None = None,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        visibility_factor: float | None = None,
    ) -> Percept:
        """Compute only what appeared, disappeared, moved beyond threshold, or changed since the last call."""
        w = self._resolve_world(world)
        vf = self.visibility_factor if visibility_factor is None else _validate_visibility_factor(visibility_factor)
        self.visibility_factor = vf

        cfg = _resolve_perception_settings(
            self.config,
            w,
            sensing_radius=self.sensing_radius,
            reach_radius=self.reach_radius,
            move_threshold=self.move_threshold,
            color_visibility_threshold=self.color_visibility_threshold,
        )
        threshold = cfg.move_threshold

        # Capture current full state without consuming tracker state yet
        current = perceive(
            w,
            body,
            visibility_factor=vf,
            sensing_radius=cfg.sensing_radius,
            reach_radius=cfg.reach_radius,
            color_visibility_threshold=cfg.color_visibility_threshold,
            config=self.config,
            events=(),
        )
        current_by_id = {obj.id: obj for obj in current.objects}

        appeared: list[PerceivedObject] = []
        disappeared: list[str] = []
        moved: list[PerceivedObject] = []
        changed: list[PerceivedObject] = []
        delta_objects_by_id: dict[str, PerceivedObject] = {}
        delta_events: list[PerceivedEvent] = []

        # 1. Check disappeared objects (in _known_objects but no longer in current_by_id)
        for obj_id in sorted(list(self._known_objects.keys()), key=_object_sort_key):
            if obj_id not in current_by_id:
                prev_obj = self._known_objects.pop(obj_id)
                disappeared.append(obj_id)
                delta_events.append(
                    PerceivedEvent(
                        type="disappeared",
                        tick=w.tick,
                        object_id=obj_id,
                        previous_position=prev_obj.position.model_copy(),
                    )
                )

        # 2. Check appeared, moved beyond threshold, and changed objects
        for obj_id in sorted(current_by_id, key=_object_sort_key):
            curr_obj = current_by_id[obj_id]
            if obj_id not in self._known_objects:
                appeared.append(curr_obj)
                delta_objects_by_id[obj_id] = curr_obj
                self._known_objects[obj_id] = curr_obj.model_copy(deep=True)
                delta_events.append(
                    PerceivedEvent(
                        type="appeared",
                        tick=w.tick,
                        object_id=obj_id,
                        position=curr_obj.position.model_copy(),
                        distance=curr_obj.distance,
                        direction=curr_obj.direction,
                        properties=dict(curr_obj.properties),
                    )
                )
                continue

            prev_obj = self._known_objects[obj_id]
            move_dist = math.hypot(
                curr_obj.position.x - prev_obj.position.x,
                curr_obj.position.y - prev_obj.position.y,
            )
            did_move = (move_dist > threshold) if threshold > 0.0 else (move_dist > 0.0)

            prev_props = {k: v for k, v in prev_obj.properties.items() if k != "position"}
            curr_props = {k: v for k, v in curr_obj.properties.items() if k != "position"}
            prop_changes: dict[str, Any] = {}
            for k, v in curr_props.items():
                if k not in prev_props or prev_props[k] != v:
                    prop_changes[k] = v
            for k in prev_props:
                if k not in curr_props:
                    prop_changes[k] = None

            did_change = bool(prop_changes)

            if did_move:
                moved.append(curr_obj)
                delta_objects_by_id[obj_id] = curr_obj
                delta_events.append(
                    PerceivedEvent(
                        type="moved",
                        tick=w.tick,
                        object_id=obj_id,
                        previous_position=prev_obj.position.model_copy(),
                        position=curr_obj.position.model_copy(),
                        distance=curr_obj.distance,
                        direction=curr_obj.direction,
                    )
                )

            if did_change:
                changed.append(curr_obj)
                delta_objects_by_id[obj_id] = curr_obj
                delta_events.append(
                    PerceivedEvent(
                        type="changed",
                        tick=w.tick,
                        object_id=obj_id,
                        position=curr_obj.position.model_copy(),
                        distance=curr_obj.distance,
                        direction=curr_obj.direction,
                        changes=prop_changes,
                        properties=dict(curr_obj.properties),
                    )
                )

            if did_move and did_change:
                self._known_objects[obj_id] = curr_obj.model_copy(deep=True)
            elif did_move:
                self._known_objects[obj_id] = curr_obj.model_copy(deep=True)
            elif did_change:
                # Update properties while keeping the baseline position for cumulative move threshold
                kept_pos = prev_obj.position.model_copy()
                updated_copy = curr_obj.model_copy(deep=True)
                updated_copy.position = kept_pos
                self._known_objects[obj_id] = updated_copy

        delta_objects = sorted(
            delta_objects_by_id.values(),
            key=lambda o: (round(o.distance, 6), _object_sort_key(o.id)),
        )

        body_changed = self._last_body != current.body
        affordances_changed = self._last_affordances != current.affordances
        has_any_object_delta = bool(appeared or disappeared or moved or changed or delta_events)

        self._last_body = current.body.model_copy(deep=True)
        self._last_affordances = AffordanceMap({k: list(v) for k, v in current.affordances.items()})
        self._initialized = True

        is_completely_empty = not has_any_object_delta and not body_changed and not affordances_changed
        delta_affordances = (
            current.affordances
            if (has_any_object_delta or affordances_changed or body_changed)
            else AffordanceMap()
        )

        result = Percept(
            tick=w.tick,
            body=current.body,
            objects=delta_objects,
            events=delta_events,
            affordances=delta_affordances,
            is_delta=True,
            appeared=appeared,
            disappeared=disappeared,
            moved=moved,
            changed=changed,
        )
        result._empty_delta = is_completely_empty
        return result

    def __call__(
        self,
        world: World | None = None,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        full: bool = False,
        refresh: bool = False,
        visibility_factor: float | None = None,
    ) -> Percept:
        return self.perceive(
            world,
            body,
            full=full,
            refresh=refresh,
            visibility_factor=visibility_factor,
        )
