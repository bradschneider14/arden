"""World objects: perceptible physical properties plus hidden verb rules.

Objects are data. There is no subclass per kind of thing. A human name such as
"ball" is not a field; callers refer to an object by its ``object_<n>`` id.
``Composite`` is a ``WorldObject`` whose physical properties are derived from
its constituent parts and how they are joined.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_LEVEL = re.compile(r"^(?:ground|raised|on_top_of:object_[0-9]+|inside:object_[0-9]+)$")
_VERB = re.compile(r"^[a-z][a-z0-9_]*$")

# Fields a perceiver may see. `behavior`, `velocity`, effect-revealing
# properties (`fragile`, `fixed`, `binding`, `rigidity`), and internal
# composite structure fields are intentionally absent.
_PERCEPTIBLE_FIELDS = frozenset(
    {
        "id",
        "position",
        "level",
        "shape",
        "height",
        "volume",
        "weight",
        "hollow",
        "color",
        "state",
    }
)

# Only raw, directly sensed physical observations belong here (never internal
# simulation parameters like `radius`, derived labels like `floating`, or human
# names). Future tickets must add keys here deliberately so arbitrary `state`
# entries never leak to the agent through `perceptible()`.
PERCEPTIBLE_STATE_KEYS: frozenset[str] = frozenset(
    {
        "wet", "made_by_agent",
    }
)
_PERCEPTIBLE_STATE_KEYS = PERCEPTIBLE_STATE_KEYS
_HIDDEN_OBJECT_FIELDS = frozenset(
    {
        "behavior",
        "velocity",
        "fragile",
        "fixed",
        "binding",
        "rigidity",
        "parts",
        "binder",
        "join_type",
        "joint_strength",
        "center_of_mass_offset",
        "base_object",
        "made_by_agent_flag",
    }
)

_COMPOSITE_EXTRA_KEYS = frozenset(
    {
        "parts",
        "binder",
        "join_type",
        "joint_strength",
        "center_of_mass_offset",
        "center_of_mass",
        "base_object",
        "made_by_agent_flag",
        "made_by_agent",
        "length",
    }
)


def _require_json_value(value: object, path: str) -> None:
    """Reject values that ``json.dumps`` cannot round-trip as data."""
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                # Pydantic turns ValueError into ValidationError. TypeError would escape.
                raise ValueError(f"{path} keys must be strings")  # noqa: TRY004
            _require_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must be a JSON value")


class CenterOfMass(float):
    """Normalized axial center-of-mass ratio in ``[0, 1]`` with 2D coordinate access.

    Behaves as a ``float`` (``0.5`` when balanced, ``> 0.5`` when mass is
    concentrated toward the far/head end) and also exposes ``.x``, ``.y``,
    ``.offset``, and ``['x']`` / ``['y']`` / ``[0]`` / ``[1]``.
    """

    _x: float
    _y: float

    def __new__(
        cls,
        offset: float = 0.5,
        *,
        x: float = 0.0,
        y: float = 0.0,
    ) -> CenterOfMass:
        val = float(offset)
        instance = super().__new__(cls, val)
        instance._x = float(x)
        instance._y = float(y)
        return instance

    @property
    def offset(self) -> float:
        """Normalized axial offset in ``[0, 1]``."""
        return float(self)

    @property
    def ratio(self) -> float:
        """Alias for ``offset``."""
        return float(self)

    @property
    def x(self) -> float:
        """2D world x coordinate of the center of mass."""
        return self._x

    @property
    def y(self) -> float:
        """2D world y coordinate of the center of mass."""
        return self._y

    def __getitem__(self, key: str | int) -> float:
        if key in ("x", 0):
            return self._x
        if key in ("y", 1):
            return self._y
        if key in ("offset", "ratio", "z", 2):
            return float(self)
        raise KeyError(key)

    def to_dict(self) -> dict[str, float]:
        return {"offset": float(self), "x": self._x, "y": self._y}


class Position(BaseModel):
    """A point on the 2D movement plane."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def coordinates_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("coordinates must be finite")
        return value

    def distance_to(self, other: Position | tuple[float, float] | list[float]) -> float:
        """Return the Euclidean distance to ``other`` on the 2D plane."""
        if isinstance(other, Position):
            return math.hypot(self.x - other.x, self.y - other.y)
        return math.hypot(self.x - float(other[0]), self.y - float(other[1]))


class Velocity(BaseModel):
    """2D velocity vector on the movement plane (meters per tick)."""

    model_config = ConfigDict(extra="forbid")

    x: float = 0.0
    y: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _coerce_velocity(cls, data: Any) -> Any:
        if isinstance(data, tuple | list) and len(data) == 2:
            return {"x": data[0], "y": data[1]}
        if isinstance(data, Mapping):
            d = dict(data)
            if "vx" in d or "vy" in d:
                if "x" not in d and "vx" in d:
                    d["x"] = d.pop("vx")
                if "y" not in d and "vy" in d:
                    d["y"] = d.pop("vy")
            return d
        if hasattr(data, "x") and hasattr(data, "y") and not isinstance(data, cls):
            return {"x": data.x, "y": data.y}
        return data

    @field_validator("x", "y")
    @classmethod
    def components_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("velocity components must be finite")
        return value

    @property
    def vx(self) -> float:
        """Alias for ``x``."""
        return self.x

    @property
    def vy(self) -> float:
        """Alias for ``y``."""
        return self.y

    @property
    def speed(self) -> float:
        """Euclidean speed ``hypot(x, y)``."""
        return math.hypot(self.x, self.y)


MIN_OBJECT_RADIUS: float = 0.05
MAX_OBJECT_RADIUS: float = 2.0


def object_radius(obj: WorldObject) -> float:
    """Compute the 2D collision circle radius for ``obj`` from its physical properties."""
    explicit_r = obj.state.get("radius")
    if isinstance(explicit_r, int | float) and not isinstance(explicit_r, bool):
        r_val = float(explicit_r)
        if math.isfinite(r_val) and r_val > 0.0:
            return r_val

    if obj.shape == "sphere":
        r_h = obj.height / 2.0 if obj.height > 0.0 else 0.0
        r_v = (
            (3.0 * obj.volume / (4.0 * math.pi)) ** (1.0 / 3.0)
            if obj.volume > 0.0
            else 0.0
        )
        raw = max(r_h, r_v)
    elif obj.volume > 0.0 and obj.height > 0.0:
        raw = math.sqrt(obj.volume / (math.pi * obj.height))
    elif obj.volume > 0.0:
        raw = (obj.volume / math.pi) ** (1.0 / 3.0)
    elif obj.height > 0.0:
        raw = obj.height * 0.5
    else:
        raw = 0.0

    return min(MAX_OBJECT_RADIUS, max(MIN_OBJECT_RADIUS, raw))


class BehaviorRule(BaseModel):
    """One hidden response to a body verb.

    ``verb`` is the key. Every other field is rule data (a push scale, a bounce
    response, a force threshold). Rules are not subclasses and do not carry code.
    """

    model_config = ConfigDict(extra="allow")

    verb: str = Field(pattern=_VERB.pattern)

    @model_validator(mode="after")
    def parameters_are_json(self) -> BehaviorRule:
        extras = self.__pydantic_extra__ or {}
        for key, value in extras.items():
            _require_json_value(value, f"behavior.{key}")
        return self


class WorldObject(BaseModel):
    """One simulated thing, addressed by ``object_<n>`` and never by a name."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_OBJECT_ID.pattern)
    position: Position
    level: str = "ground"
    shape: Literal["sphere", "cube", "cylinder", "flat", "long"]
    height: float = Field(ge=0)
    volume: float = Field(ge=0)
    weight: float = Field(ge=0)
    rigidity: float = Field(ge=0, le=1)
    binding: bool = False
    hollow: bool = False
    fragile: bool = False
    color: str | None = None
    fixed: bool = False
    velocity: Velocity = Field(default_factory=Velocity)
    state: dict[str, Any] = Field(default_factory=dict)
    behavior: list[BehaviorRule] = Field(default_factory=list)

    @property
    def radius(self) -> float:
        """2D collision circle radius derived from physical properties."""
        return object_radius(self)

    @property
    def length(self) -> float:
        """Primary extent / length of the object (defaults to ``height``)."""
        raw_len = self.state.get("length")
        if isinstance(raw_len, int | float) and not isinstance(raw_len, bool):
            return float(raw_len)
        return self.height

    @property
    def center_of_mass(self) -> CenterOfMass:
        """Normalized center of mass along the object (``0.5`` for uniform objects)."""
        raw_com = self.state.get("center_of_mass", 0.5)
        offset = (
            float(raw_com)
            if isinstance(raw_com, int | float) and not isinstance(raw_com, bool)
            else 0.5
        )
        return CenterOfMass(offset, x=self.position.x, y=self.position.y)

    @property
    def made_by_agent(self) -> bool:
        """True if this object was built/assembled by the agent."""
        return bool(self.state.get("made_by_agent", False))

    @field_validator("level")
    @classmethod
    def level_is_known(cls, value: str) -> str:
        if _LEVEL.fullmatch(value) is None:
            raise ValueError(
                "level must be ground, raised, on_top_of:object_<n>, or inside:object_<n>"
            )
        return value

    @field_validator("height", "volume", "weight", "rigidity")
    @classmethod
    def magnitudes_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("color")
    @classmethod
    def color_is_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("color must be a non-empty string")
        return stripped

    @field_validator("state")
    @classmethod
    def state_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, "state")
        return value

    def perceptible(self) -> dict[str, Any]:
        """Return properties a perceiver can sense.

        Hidden ``behavior`` rules are omitted. Color is omitted when unset.
        Only allowlisted keys in ``state`` are included.
        The id is the referent (``object_7``), not a name derived from it.
        """
        data = self.model_dump(mode="json", exclude=set(_HIDDEN_OBJECT_FIELDS))
        if data.get("color") is None:
            del data["color"]
        data["state"] = {
            key: value
            for key, value in data["state"].items()
            if key in PERCEPTIBLE_STATE_KEYS
        }
        unexpected = set(data) - _PERCEPTIBLE_FIELDS
        if unexpected:
            joined = ", ".join(sorted(unexpected))
            raise RuntimeError(f"perceptible() leaked fields: {joined}")
        return data

    def to_json(self) -> str:
        """Serialize the object, including hidden behavior rules."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> WorldObject:
        """Build an object from ``to_json`` output."""
        if cls is WorldObject:
            raw = json.loads(payload)
            if isinstance(raw, Mapping) and _is_composite_mapping(raw):
                return Composite.model_validate(raw)
        return cls.model_validate_json(payload)


def _is_composite_mapping(data: Mapping[str, Any]) -> bool:
    if any(k in data for k in ("parts", "binder", "join_type", "joint_strength", "base_object")):
        return True
    state = data.get("state")
    if isinstance(state, Mapping) and state.get("composite") is True:
        return True
    return False


def validate_world_object(data: WorldObject | Mapping[str, Any]) -> WorldObject:
    """Validate ``data`` into a ``Composite`` (when composite data is present) or ``WorldObject``."""
    if isinstance(data, Composite):
        return data
    if isinstance(data, WorldObject):
        return data
    if isinstance(data, Mapping) and _is_composite_mapping(data):
        return Composite.model_validate(data)
    return WorldObject.model_validate(data)


def extract_original_parts(parts: Sequence[WorldObject]) -> list[WorldObject]:
    """Flatten nested bound ``Composite`` objects into their original leaf ``WorldObject`` parts."""
    leaf_parts: list[WorldObject] = []
    for part in parts:
        if isinstance(part, Composite) and part.parts and part.join_type == "bind":
            leaf_parts.extend(extract_original_parts(part.parts))
        elif isinstance(part, Composite) and part.base_object is not None:
            leaf_parts.append(WorldObject.model_validate(part.base_object))
        else:
            clean_data = part.model_dump(exclude=set(_COMPOSITE_EXTRA_KEYS))
            leaf_parts.append(WorldObject.model_validate(clean_data))
    return leaf_parts


def compute_composite_rigidity(parts: Sequence[WorldObject]) -> float:
    """Compute derived rigidity in ``[0, 1]`` from ``parts`` (weight-weighted mean)."""
    if not parts:
        return 0.5
    total_weight = sum(max(0.0, p.weight) for p in parts)
    if total_weight > 1e-9:
        weighted = sum(p.rigidity * max(0.0, p.weight) for p in parts) / total_weight
    else:
        weighted = sum(p.rigidity for p in parts) / len(parts)
    return min(1.0, max(0.0, round(weighted, 6)))


def compute_composite_shape(
    parts: Sequence[WorldObject],
    join_type: str = "bind",
) -> Literal["sphere", "cube", "cylinder", "flat", "long"]:
    """Derive composite shape from its constituent ``parts`` and ``join_type``."""
    if not parts:
        return "cube"
    if join_type == "insert":
        return parts[0].shape
    if any(p.shape == "long" for p in parts):
        return "long"
    if len(parts) > 1:
        total_height = sum(p.height for p in parts)
        max_rad = max((object_radius(p) for p in parts), default=0.05)
        if total_height > 2.2 * max_rad:
            return "long"
    first_shape = parts[0].shape
    if all(p.shape == first_shape for p in parts):
        return first_shape
    dominant = max(parts, key=lambda p: (p.volume, p.height, p.weight))
    return dominant.shape


def compute_center_of_mass(
    parts: Sequence[WorldObject],
    position: Position | None = None,
) -> CenterOfMass:
    """Derive center of mass from ``parts`` along the composite axis and on the 2D plane.

    Returns a ``CenterOfMass(float)`` whose float value is in ``[0, 1]`` (``0.5``
    when balanced, ``> 0.5`` when mass is shifted toward the heavier end, e.g. a
    light stick bound to a heavy stone).
    """
    base_x = position.x if position is not None else (parts[0].position.x if parts else 0.0)
    base_y = position.y if position is not None else (parts[0].position.y if parts else 0.0)
    if len(parts) <= 1:
        return CenterOfMass(0.5, x=base_x, y=base_y)

    total_weight = sum(max(0.0, p.weight) for p in parts)
    if total_weight <= 1e-9:
        return CenterOfMass(0.5, x=base_x, y=base_y)

    # Order parts from lightest (grip/handle end at 0) to heaviest (striking head at 1)
    ordered = sorted(parts, key=lambda p: (p.weight, -p.height))
    weights = [max(0.0, p.weight) for p in ordered]
    if max(weights) - min(weights) <= 1e-9:
        offset = 0.5
    else:
        total_len = sum(max(0.05, p.height) for p in ordered)
        cursor = 0.0
        weighted_pos = 0.0
        for p, w in zip(ordered, weights, strict=True):
            seg = max(0.05, p.height)
            mid_ratio = (cursor + 0.5 * seg) / total_len
            cursor += seg
            weighted_pos += w * mid_ratio
        axial_ratio = weighted_pos / total_weight

        n = len(ordered)
        index_ratio = sum(w * ((idx + 0.5) / n) for idx, w in enumerate(weights)) / total_weight
        offset = min(0.98, max(0.02, round(max(axial_ratio, index_ratio), 6)))

    cx = sum(max(0.0, p.weight) * p.position.x for p in parts) / total_weight
    cy = sum(max(0.0, p.weight) * p.position.y for p in parts) / total_weight
    if math.hypot(cx - base_x, cy - base_y) <= 1e-9 and offset != 0.5:
        total_h = sum(p.height for p in parts)
        cx = base_x + (offset - 0.5) * max(0.1, total_h)
    return CenterOfMass(offset, x=round(cx, 6), y=round(cy, 6))


def compute_joint_strength(
    parts: Sequence[WorldObject],
    binder: WorldObject | None = None,
) -> float:
    """Derive joint strength from ``binder`` and ``parts``.

    A bound joint can fail under load when struck or used to strike with force
    exceeding its ``joint_strength``.
    """
    if binder is not None:
        for key in ("joint_strength", "strength"):
            explicit = binder.state.get(key)
            if isinstance(explicit, int | float) and not isinstance(explicit, bool):
                return max(0.05, float(explicit))

        binder_base = (
            0.72
            + 0.20 * binder.rigidity
            + 0.10 * min(1.0, max(0.0, binder.weight))
        )
        if binder.fragile:
            binder_base *= 0.55
    else:
        binder_base = 0.75

    total_part_weight = sum(max(0.0, p.weight) for p in parts)
    avg_rigidity = (
        sum(p.rigidity for p in parts) / len(parts) if parts else 0.5
    )
    strength = (binder_base + 0.05 * avg_rigidity) / (1.0 + 0.04 * total_part_weight)
    return round(min(0.95, max(0.15, strength)), 6)


class Composite(WorldObject):
    """A ``WorldObject`` built by joining parts (via ``bind``, ``stack``, or ``insert``).

    Its physical properties (``weight``, ``height``/``length``, ``Volume``,
    ``rigidity``, ``center_of_mass``, and ``shape``) are derived from its
    constituent parts and join type. Records ``made_by_agent=True`` for memory.
    """

    model_config = ConfigDict(extra="forbid")

    parts: list[WorldObject] = Field(default_factory=list)
    binder: WorldObject | None = None
    join_type: Literal["bind", "stack", "insert"] = "bind"
    joint_strength: float = Field(default=1.0, ge=0.0)
    center_of_mass_offset: float = Field(default=0.5, ge=0.0, le=1.0)
    base_object: dict[str, Any] | None = None
    made_by_agent_flag: bool = True

    @model_validator(mode="before")
    @classmethod
    def _derive_from_parts(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        d = dict(data)

        raw_parts = d.get("parts")
        parsed_parts: list[WorldObject] = []
        if isinstance(raw_parts, Sequence) and not isinstance(raw_parts, str | bytes):
            for item in raw_parts:
                parsed_parts.append(validate_world_object(item))
            d["parts"] = parsed_parts

        raw_binder = d.get("binder")
        binder_obj: WorldObject | None = None
        if raw_binder is not None:
            binder_obj = validate_world_object(raw_binder)
            d["binder"] = binder_obj

        join_type = str(d.get("join_type", "bind"))
        if join_type not in ("bind", "stack", "insert"):
            join_type = "bind"
        d["join_type"] = join_type

        if "id" not in d or d["id"] is None:
            d["id"] = "object_0"

        if "position" not in d or d["position"] is None:
            if parsed_parts:
                d["position"] = parsed_parts[-1].position.model_copy()
            else:
                d["position"] = Position(x=0.0, y=0.0)

        pos_obj = (
            d["position"]
            if isinstance(d["position"], Position)
            else Position.model_validate(d["position"])
        )

        explicit_len = d.pop("length", None)
        if parsed_parts:
            binder_weight = (
                binder_obj.weight
                if (binder_obj is not None and join_type == "bind")
                else 0.0
            )
            binder_vol = (
                binder_obj.volume
                if (binder_obj is not None and join_type == "bind")
                else 0.0
            )
            if "weight" not in d or d["weight"] is None:
                d["weight"] = round(sum(p.weight for p in parsed_parts) + binder_weight, 6)
            if "volume" not in d or d["volume"] is None:
                if join_type == "insert":
                    d["volume"] = parsed_parts[0].volume
                else:
                    d["volume"] = round(sum(p.volume for p in parsed_parts) + binder_vol, 6)
            if "height" not in d or d["height"] is None:
                if explicit_len is not None:
                    d["height"] = float(explicit_len)
                elif join_type == "insert":
                    d["height"] = parsed_parts[0].height
                else:
                    d["height"] = round(sum(p.height for p in parsed_parts), 6)
            if "rigidity" not in d or d["rigidity"] is None:
                if join_type == "insert":
                    d["rigidity"] = parsed_parts[0].rigidity
                else:
                    d["rigidity"] = compute_composite_rigidity(parsed_parts)
            if "shape" not in d or d["shape"] is None:
                d["shape"] = compute_composite_shape(parsed_parts, join_type=join_type)
            if "hollow" not in d or d["hollow"] is None:
                if join_type == "insert":
                    d["hollow"] = parsed_parts[0].hollow
                else:
                    d["hollow"] = any(p.hollow for p in parsed_parts)
            if "fragile" not in d or d["fragile"] is None:
                d["fragile"] = all(p.fragile for p in parsed_parts)
            if "color" not in d:
                dominant = max(parsed_parts, key=lambda p: (p.weight, p.volume, p.height))
                d["color"] = dominant.color or next((p.color for p in parsed_parts if p.color), None)
        else:
            if explicit_len is not None and ("height" not in d or d["height"] is None):
                d["height"] = float(explicit_len)

        raw_com = d.pop("center_of_mass", None)
        if "center_of_mass_offset" not in d or d["center_of_mass_offset"] is None:
            if isinstance(raw_com, int | float) and not isinstance(raw_com, bool):
                d["center_of_mass_offset"] = float(raw_com)
            elif isinstance(raw_com, Mapping) and "offset" in raw_com:
                d["center_of_mass_offset"] = float(raw_com["offset"])
            elif parsed_parts:
                d["center_of_mass_offset"] = float(
                    compute_center_of_mass(parsed_parts, position=pos_obj)
                )
            else:
                d["center_of_mass_offset"] = 0.5

        if "joint_strength" not in d or d["joint_strength"] is None:
            d["joint_strength"] = compute_joint_strength(parsed_parts, binder_obj)

        made_flag = d.pop("made_by_agent", d.get("made_by_agent_flag", True))
        d["made_by_agent_flag"] = bool(made_flag)

        state = dict(d.get("state") or {})
        state["made_by_agent"] = bool(made_flag)
        d["state"] = state
        return d

    @property
    def center_of_mass(self) -> CenterOfMass:
        """Center of mass derived from the composite's parts."""
        if self.parts:
            com = compute_center_of_mass(self.parts, position=self.position)
            return CenterOfMass(self.center_of_mass_offset, x=com.x, y=com.y)
        return CenterOfMass(
            self.center_of_mass_offset,
            x=self.position.x,
            y=self.position.y,
        )

    @property
    def length(self) -> float:
        """Derived length/height of the composite."""
        raw_len = self.state.get("length")
        if isinstance(raw_len, int | float) and not isinstance(raw_len, bool):
            return float(raw_len)
        return self.height

    @property
    def made_by_agent(self) -> bool:
        """Composites built by the agent record ``made_by_agent=True``."""
        return bool(self.state.get("made_by_agent", self.made_by_agent_flag))

    @property
    def part_ids(self) -> list[str]:
        """IDs of the direct parts in this composite."""
        return [p.id for p in self.parts]

    @property
    def original_parts(self) -> list[WorldObject]:
        """Original non-composite leaf parts that make up this composite."""
        return extract_original_parts(self.parts)

    @classmethod
    def from_parts(
        cls,
        parts: Sequence[WorldObject | Mapping[str, Any]],
        *,
        binder: WorldObject | Mapping[str, Any] | None = None,
        id: str = "object_0",
        position: Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
        level: str = "ground",
        join_type: Literal["bind", "stack", "insert"] = "bind",
        made_by_agent: bool = True,
        state: Mapping[str, Any] | None = None,
        behavior: list[BehaviorRule | Mapping[str, Any]] | None = None,
    ) -> Composite:
        """Construct a ``Composite`` whose properties are derived from ``parts`` and ``binder``."""
        parsed_parts = [validate_world_object(p) for p in parts]
        if not parsed_parts:
            raise ValueError("Composite.from_parts requires at least one part")
        if position is None:
            pos = parsed_parts[-1].position.model_copy()
        elif isinstance(position, Position):
            pos = position.model_copy()
        elif isinstance(position, Mapping):
            pos = Position.model_validate(dict(position))
        elif isinstance(position, tuple | list) and len(position) == 2:
            pos = Position(x=float(position[0]), y=float(position[1]))
        else:
            raise TypeError("position must be Position, mapping, (x, y) pair, or None")

        payload: dict[str, Any] = {
            "id": id,
            "position": pos,
            "level": level,
            "parts": parsed_parts,
            "binder": binder,
            "join_type": join_type,
            "made_by_agent": made_by_agent,
            "state": dict(state or {}),
        }
        if behavior is not None:
            payload["behavior"] = behavior
        return cls.model_validate(payload)


class ObjectFactory:
    """Assign ``object_<n>`` ids and build objects from property dicts.

    One factory corresponds to one world. First-run content can omit ids and
    stay pure data. An explicit id is kept when it matches ``object_<n>``.
    """

    def __init__(self) -> None:
        self._next = 1
        self._used: set[str] = set()

    def create(self, properties: Mapping[str, Any] | WorldObject) -> WorldObject:
        """Validate ``properties`` and return a new object.

        Unknown keys are rejected, so a human label cannot be stored on the object.
        """
        if isinstance(properties, Composite):
            obj: WorldObject = properties.model_copy(deep=True)
            if obj.id in self._used:
                raise ValueError(f"duplicate object id: {obj.id}")
            self._claim(obj.id)
            return obj
        if isinstance(properties, WorldObject):
            data = properties.model_dump()
        elif isinstance(properties, Mapping):
            data = dict(properties)
        else:
            raise TypeError("properties must be a mapping")
        if data.get("id") is None:
            data["id"] = self._peek_next()
        obj = validate_world_object(data)
        if obj.id in self._used:
            raise ValueError(f"duplicate object id: {obj.id}")
        self._claim(obj.id)
        return obj

    def _peek_next(self) -> str:
        return f"object_{self._next}"

    def _claim(self, object_id: str) -> None:
        number = int(object_id.removeprefix("object_"))
        self._used.add(object_id)
        if number >= self._next:
            self._next = number + 1
