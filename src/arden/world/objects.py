"""World objects: perceptible physical properties plus hidden verb rules.

Objects are data. There is no subclass per kind of thing. A human name such as
"ball" is not a field; callers refer to an object by its ``object_<n>`` id.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_LEVEL = re.compile(r"^(?:ground|raised|on_top_of:object_[0-9]+|inside:object_[0-9]+)$")
_VERB = re.compile(r"^[a-z][a-z0-9_]*$")

# Fields a perceiver may see. `behavior` is intentionally absent.
_PERCEPTIBLE_FIELDS = frozenset(
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

# Only raw, directly sensed physical observations belong here (never internal
# simulation parameters like `radius`, derived labels like `floating`, or human
# names). Future tickets must add keys here deliberately so arbitrary `state`
# entries never leak to the agent through `perceptible()`.
PERCEPTIBLE_STATE_KEYS: frozenset[str] = frozenset(
    {
        "wet",
    }
)
_PERCEPTIBLE_STATE_KEYS = PERCEPTIBLE_STATE_KEYS


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
        data = self.model_dump(mode="json", exclude={"behavior", "velocity"})
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
        return cls.model_validate_json(payload)


class ObjectFactory:
    """Assign ``object_<n>`` ids and build objects from property dicts.

    One factory corresponds to one world. First-run content can omit ids and
    stay pure data. An explicit id is kept when it matches ``object_<n>``.
    """

    def __init__(self) -> None:
        self._next = 1
        self._used: set[str] = set()

    def create(self, properties: Mapping[str, Any]) -> WorldObject:
        """Validate ``properties`` and return a new object.

        Unknown keys are rejected, so a human label cannot be stored on the object.
        """
        if not isinstance(properties, Mapping):
            raise TypeError("properties must be a mapping")
        data = dict(properties)
        if data.get("id") is None:
            data["id"] = self._peek_next()
        obj = WorldObject.model_validate(data)
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
