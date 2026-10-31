"""Weather system: probabilistic automation, manual/scheduled overrides, shelter, and world effects."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from arden.world.objects import Position, WorldObject
    from arden.world.world import AgentBody, World

WeatherType = Literal["clear", "rain", "wind", "cold", "storm"]
WEATHER_TYPES: tuple[WeatherType, ...] = ("clear", "rain", "wind", "cold", "storm")

WeatherSource = Literal["manual", "automated"]
WEATHER_SOURCES: tuple[WeatherSource, ...] = ("manual", "automated")

DEFAULT_WEATHER_SEED: int = 0
DEFAULT_TICKS_PER_HOUR: int = 25

DEFAULT_WEATHER_CHANCE: dict[WeatherType, float] = {
    "clear": 0.45,
    "rain": 0.30,
    "wind": 0.25,
    "cold": 0.20,
    "storm": 0.15,
}

DEFAULT_WEATHER_DURATION: dict[WeatherType, tuple[int, int]] = {
    "clear": (60, 150),
    "rain": (40, 100),
    "wind": (30, 90),
    "cold": (40, 100),
    "storm": (25, 70),
}

DEFAULT_WEATHER_INTENSITY: dict[WeatherType, tuple[float, float]] = {
    "clear": (0.0, 0.0),
    "rain": (0.25, 0.85),
    "wind": (0.25, 0.85),
    "cold": (0.25, 0.85),
    "storm": (0.50, 1.00),
}

DEFAULT_WEATHER_TRANSITIONS: dict[WeatherType, list[WeatherType]] = {
    "clear": ["rain", "wind", "cold"],
    "rain": ["clear", "wind", "cold", "storm"],
    "wind": ["clear", "rain", "cold", "storm"],
    "cold": ["clear", "rain", "wind", "storm"],
    "storm": ["rain", "wind", "cold", "clear"],
}

# Physical and environmental effect constants
WIND_MAX_OBJECT_WEIGHT: float = 0.20
WIND_PUSH_STRENGTH: float = 0.15
DEFAULT_WIND_DIRECTION: tuple[float, float] = (1.0, 0.0)
RAIN_SLIPPERINESS_FACTOR: float = 0.50
RAIN_IMPULSE_SLIP_BONUS: float = 0.25
RAIN_WET_THRESHOLD: float = 0.25
COLD_TEMPERATURE_DROP: float = 0.35
STORM_TEMPERATURE_DROP: float = 0.30
STORM_LIGHT_REDUCTION: float = 0.50

# Shelter constants
TREE_MIN_HEIGHT: float = 1.2
TREE_MIN_VOLUME: float = 0.10
TREE_SHELTER_EXPOSURE: float = 0.20
HOLLOW_SHELTER_EXPOSURE: float = 0.0


def validate_weather_type(value: Any) -> WeatherType:
    """Validate that ``value`` is one of ``clear``, ``rain``, ``wind``, ``cold``, ``storm``."""
    if not isinstance(value, str):
        raise TypeError(f"weather type must be a string, got {type(value).__name__}")
    norm = value.strip().lower()
    if norm not in WEATHER_TYPES:
        allowed = ", ".join(WEATHER_TYPES)
        raise ValueError(f"unknown weather type {value!r}; expected one of: {allowed}")
    return norm  # type: ignore[return-value]


def validate_intensity(value: Any) -> float:
    """Validate that ``value`` is a finite float in ``[0.0, 1.0]``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("weather intensity must be a finite number in [0.0, 1.0]")
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
        raise ValueError("weather intensity must be a finite number in [0.0, 1.0]")
    return fval


def _coerce_duration_pair(value: Any, key: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        lo = value.get("min", value.get("min_duration", value.get("min_ticks")))
        hi = value.get("max", value.get("max_duration", value.get("max_ticks")))
        pair: Sequence[Any] = [lo, hi]
    elif isinstance(value, tuple | list) and len(value) == 2:
        pair = value
    elif isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    else:
        raise ValueError(f"duration for {key!r} must be a (min, max) integer pair")

    a, b = pair[0], pair[1]
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int | float) or not isinstance(b, int | float):
        raise ValueError(f"duration bounds for {key!r} must be positive integers")
    if isinstance(a, float) and (not math.isfinite(a) or not a.is_integer()):
        raise ValueError(f"duration bounds for {key!r} must be integers")
    if isinstance(b, float) and (not math.isfinite(b) or not b.is_integer()):
        raise ValueError(f"duration bounds for {key!r} must be integers")
    ia, ib = int(a), int(b)
    if ia < 1 or ib < ia:
        raise ValueError(f"duration bounds for {key!r} must satisfy 1 <= min <= max")
    return (ia, ib)


def _coerce_intensity_pair(value: Any, key: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        lo = value.get("min", value.get("min_intensity"))
        hi = value.get("max", value.get("max_intensity"))
        pair: Sequence[Any] = [lo, hi]
    elif isinstance(value, tuple | list) and len(value) == 2:
        pair = value
    elif isinstance(value, int | float) and not isinstance(value, bool):
        pair = (value, value)
    else:
        raise ValueError(f"intensity for {key!r} must be a (min, max) float pair in [0.0, 1.0]")

    a, b = pair[0], pair[1]
    fa = validate_intensity(a)
    fb = validate_intensity(b)
    if fb < fa:
        raise ValueError(f"intensity bounds for {key!r} must satisfy 0.0 <= min <= max <= 1.0")
    return (fa, fb)


class WeatherConfig(BaseModel):
    """Configuration for automated probabilistic weather and its seeded RNG."""

    model_config = ConfigDict(extra="forbid")

    seed: int = DEFAULT_WEATHER_SEED
    ticks_per_hour: int = Field(default=DEFAULT_TICKS_PER_HOUR, gt=0)
    paused: bool = False
    chance: dict[WeatherType, float] = Field(
        default_factory=lambda: dict(DEFAULT_WEATHER_CHANCE)
    )
    duration: dict[WeatherType, tuple[int, int]] = Field(
        default_factory=lambda: dict(DEFAULT_WEATHER_DURATION)
    )
    intensity: dict[WeatherType, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_WEATHER_INTENSITY)
    )
    transitions: dict[WeatherType, list[WeatherType]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_WEATHER_TRANSITIONS.items()}
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        raw = dict(data)

        # Normalize aliases
        if "chances" in raw and "chance" not in raw:
            raw["chance"] = raw.pop("chances")
        if "durations" in raw and "duration" not in raw:
            raw["duration"] = raw.pop("durations")
        if "duration_range" in raw and "duration" not in raw:
            raw["duration"] = raw.pop("duration_range")
        if "duration_ranges" in raw and "duration" not in raw:
            raw["duration"] = raw.pop("duration_ranges")
        if "intensities" in raw and "intensity" not in raw:
            raw["intensity"] = raw.pop("intensities")
        if "intensity_range" in raw and "intensity" not in raw:
            raw["intensity"] = raw.pop("intensity_range")
        if "intensity_ranges" in raw and "intensity" not in raw:
            raw["intensity"] = raw.pop("intensity_ranges")
        if "allowed_transitions" in raw and "transitions" not in raw:
            raw["transitions"] = raw.pop("allowed_transitions")
        if "hour_ticks" in raw and "ticks_per_hour" not in raw:
            raw["ticks_per_hour"] = raw.pop("hour_ticks")
        if "hour_length" in raw and "ticks_per_hour" not in raw:
            raw["ticks_per_hour"] = raw.pop("hour_length")
        if "automation_paused" in raw and "paused" not in raw:
            raw["paused"] = raw.pop("automation_paused")
        if "automated" in raw and "paused" not in raw:
            raw["paused"] = not bool(raw.pop("automated"))

        chance_map: dict[WeatherType, float] = dict(DEFAULT_WEATHER_CHANCE)
        duration_map: dict[WeatherType, tuple[int, int]] = dict(DEFAULT_WEATHER_DURATION)
        intensity_map: dict[WeatherType, tuple[float, float]] = dict(DEFAULT_WEATHER_INTENSITY)
        transitions_map: dict[WeatherType, list[WeatherType]] = {
            k: list(v) for k, v in DEFAULT_WEATHER_TRANSITIONS.items()
        }

        # Support optional `types` field (either list of weather type names or per-type spec dict)
        if "types" in raw:
            raw_types = raw.pop("types")
            if isinstance(raw_types, Mapping):
                for wt_raw, spec in raw_types.items():
                    wt = validate_weather_type(wt_raw)
                    if not isinstance(spec, Mapping):
                        raise ValueError(f"type specification for {wt!r} must be a mapping")
                    if "chance" in spec:
                        chance_map[wt] = float(spec["chance"])
                    for d_key in ("duration", "duration_range"):
                        if d_key in spec:
                            duration_map[wt] = _coerce_duration_pair(spec[d_key], wt)
                    for i_key in ("intensity", "intensity_range"):
                        if i_key in spec:
                            intensity_map[wt] = _coerce_intensity_pair(spec[i_key], wt)
                    for t_key in ("transitions", "allowed_transitions"):
                        if t_key in spec:
                            transitions_map[wt] = [validate_weather_type(x) for x in spec[t_key]]
            elif isinstance(raw_types, Sequence | set | frozenset) and not isinstance(raw_types, str | bytes):
                for item in raw_types:
                    validate_weather_type(item)
            else:
                raise ValueError("types must be a sequence of weather type names or a per-type mapping")

        if "chance" in raw:
            raw_chance = raw["chance"]
            if not isinstance(raw_chance, Mapping):
                raise ValueError("chance must be a mapping of weather_type -> probability")
            for k, v in raw_chance.items():
                wt = validate_weather_type(k)
                if isinstance(v, bool) or not isinstance(v, int | float):
                    raise ValueError(f"chance for {wt!r} must be a number in [0.0, 1.0]")
                fv = float(v)
                if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
                    raise ValueError(f"chance for {wt!r} must be in [0.0, 1.0]")
                chance_map[wt] = fv
            raw["chance"] = chance_map

        if "duration" in raw:
            raw_dur = raw["duration"]
            if not isinstance(raw_dur, Mapping):
                raise ValueError("duration must be a mapping of weather_type -> (min, max)")
            for k, v in raw_dur.items():
                wt = validate_weather_type(k)
                duration_map[wt] = _coerce_duration_pair(v, wt)
            raw["duration"] = duration_map

        if "intensity" in raw:
            raw_int = raw["intensity"]
            if not isinstance(raw_int, Mapping):
                raise ValueError("intensity must be a mapping of weather_type -> (min, max)")
            for k, v in raw_int.items():
                wt = validate_weather_type(k)
                intensity_map[wt] = _coerce_intensity_pair(v, wt)
            raw["intensity"] = intensity_map

        if "transitions" in raw:
            raw_trans = raw["transitions"]
            if not isinstance(raw_trans, Mapping):
                raise ValueError("transitions must be a mapping of weather_type -> list[weather_type]")
            for k, v in raw_trans.items():
                wt = validate_weather_type(k)
                if isinstance(v, str | bytes) or not isinstance(v, Sequence | set | frozenset):
                    raise ValueError(f"transitions for {wt!r} must be a list of weather types")
                transitions_map[wt] = [validate_weather_type(dest) for dest in v]
            raw["transitions"] = transitions_map

        return raw

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("weather seed must be an integer")
        return value

    @field_validator("ticks_per_hour", mode="before")
    @classmethod
    def _validate_ticks_per_hour(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("ticks_per_hour must be a positive integer")
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError("ticks_per_hour must be an integer")
            return int(value)
        return value

    @property
    def types(self) -> tuple[WeatherType, ...]:
        """Supported weather types."""
        return WEATHER_TYPES

    @property
    def allowed_transitions(self) -> dict[WeatherType, list[WeatherType]]:
        """Alias for ``transitions``."""
        return self.transitions

    def can_transition(self, from_type: str, to_type: str) -> bool:
        """Return True if ``to_type`` is an allowed transition from ``from_type``."""
        ft = validate_weather_type(from_type)
        tt = validate_weather_type(to_type)
        return tt in self.transitions.get(ft, [])


WeatherSettings = WeatherConfig


class WindDirection(BaseModel):
    """2D wind direction vector."""

    model_config = ConfigDict(extra="forbid")

    x: float = DEFAULT_WIND_DIRECTION[0]
    y: float = DEFAULT_WIND_DIRECTION[1]

    @property
    def speed(self) -> float:
        return math.hypot(self.x, self.y)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple | list) and len(other) == 2:
            return math.isclose(self.x, float(other[0]), abs_tol=1e-9) and math.isclose(
                self.y, float(other[1]), abs_tol=1e-9
            )
        if hasattr(other, "x") and hasattr(other, "y"):
            return math.isclose(self.x, float(getattr(other, "x")), abs_tol=1e-9) and math.isclose(
                self.y, float(getattr(other, "y")), abs_tol=1e-9
            )
        return super().__eq__(other)


def _normalize_wind_direction(direction: Any) -> WindDirection:
    from arden.world.physics import normalize_direction

    ux, uy = normalize_direction(direction)
    return WindDirection(x=ux, y=uy)


class ScheduledWeather(BaseModel):
    """A manual weather event scheduled to start at a future world tick."""

    model_config = ConfigDict(extra="forbid")

    type: WeatherType
    intensity: float = Field(ge=0.0, le=1.0)
    duration: int = Field(gt=0)
    start_at: int = Field(ge=0)
    wind_direction: WindDirection = Field(default_factory=WindDirection)

    @property
    def weather_type(self) -> WeatherType:
        """Alias for ``type``."""
        return self.type


class WeatherState(BaseModel):
    """Snapshot of the active weather conditions."""

    model_config = ConfigDict(extra="forbid")

    type: WeatherType = "clear"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    duration: int = Field(default=60, gt=0)
    remaining_ticks: int = Field(default=60, ge=0)
    source: WeatherSource = "automated"
    wind_direction: WindDirection = Field(default_factory=WindDirection)

    @property
    def weather_type(self) -> WeatherType:
        """Alias for ``type``."""
        return self.type

    @property
    def kind(self) -> WeatherType:
        """Alias for ``type``."""
        return self.type

    @property
    def remaining_duration(self) -> int:
        """Alias for ``remaining_ticks``."""
        return self.remaining_ticks

    @property
    def ticks_remaining(self) -> int:
        """Alias for ``remaining_ticks``."""
        return self.remaining_ticks

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.type == other
        return super().__eq__(other)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


class WeatherSystem:
    """Combined automated and manual weather controller with a dedicated seeded RNG."""

    def __init__(
        self,
        config: WeatherConfig | Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        ticks_per_hour: int | None = None,
        paused: bool | None = None,
        wind_direction: Any = None,
    ) -> None:
        if config is None:
            raw_cfg: dict[str, Any] = {}
        elif isinstance(config, WeatherConfig):
            raw_cfg = config.model_dump()
        elif isinstance(config, Mapping):
            raw_cfg = dict(config)
        else:
            raise TypeError("config must be a WeatherConfig, mapping, or None")

        if seed is not None:
            raw_cfg["seed"] = seed
        if ticks_per_hour is not None:
            raw_cfg["ticks_per_hour"] = ticks_per_hour
        if paused is not None:
            raw_cfg["paused"] = paused

        self.config: WeatherConfig = WeatherConfig.model_validate(raw_cfg)
        self.rng: random.Random = random.Random(self.config.seed)
        self._paused: bool = self.config.paused

        if wind_direction is not None:
            self._wind_direction = _normalize_wind_direction(wind_direction)
        else:
            self._wind_direction = WindDirection()

        init_dur = self._sample_duration("clear")
        init_int = self._sample_intensity("clear")
        self._type: WeatherType = "clear"
        self._intensity: float = init_int
        self._duration: int = init_dur
        self._remaining_ticks: int = init_dur
        self._source: WeatherSource = "automated"

        self._manual_active: bool = False
        self._just_activated_tick: int | None = None
        self._saved_auto: dict[str, Any] | None = None
        self._scheduled: list[ScheduledWeather] = []
        self._world_ref: World | None = None
        self._tick_counter: int = 0
        self.events: list[Any] = []
        self._subscribers: list[Any] = []

    def _bind_world(self, world: World) -> None:
        self._world_ref = world

    @property
    def types(self) -> tuple[WeatherType, ...]:
        """All valid weather types."""
        return WEATHER_TYPES

    @property
    def seed(self) -> int:
        """Configured weather RNG seed."""
        return self.config.seed

    @property
    def type(self) -> WeatherType:
        """Current active weather type."""
        return self._type

    @property
    def weather_type(self) -> WeatherType:
        """Alias for ``type``."""
        return self._type

    @property
    def kind(self) -> WeatherType:
        """Alias for ``type``."""
        return self._type

    @property
    def intensity(self) -> float:
        """Current active weather intensity in ``[0.0, 1.0]``."""
        return self._intensity

    @property
    def duration(self) -> int:
        """Total duration in ticks of the current weather spell."""
        return self._duration

    @property
    def remaining_ticks(self) -> int:
        """Remaining duration in ticks of the current weather spell."""
        return self._remaining_ticks

    @property
    def remaining_duration(self) -> int:
        """Alias for ``remaining_ticks``."""
        return self._remaining_ticks

    @property
    def ticks_remaining(self) -> int:
        """Alias for ``remaining_ticks``."""
        return self._remaining_ticks

    @property
    def source(self) -> WeatherSource:
        """Source of the current weather (``'manual'`` or ``'automated'``)."""
        return self._source

    @property
    def mode(self) -> WeatherSource:
        """Current control mode (``'manual'`` during a manual override, else ``'automated'``)."""
        return "manual" if self._manual_active else "automated"

    @property
    def is_manual(self) -> bool:
        """True when a manual weather override is currently active."""
        return self._manual_active

    @property
    def is_automated(self) -> bool:
        """True when automated weather is currently in control."""
        return not self._manual_active

    @property
    def paused(self) -> bool:
        """True when automated weather transitions are paused."""
        return self._paused

    @property
    def is_paused(self) -> bool:
        """Alias for ``paused``."""
        return self._paused

    @property
    def automation_paused(self) -> bool:
        """Alias for ``paused``."""
        return self._paused

    @property
    def automation_enabled(self) -> bool:
        """True when automated mode is enabled (not paused)."""
        return not self._paused

    @property
    def wind_direction(self) -> WindDirection:
        """Current unit wind direction vector."""
        return self._wind_direction

    @property
    def wind_speed(self) -> float:
        """Effective wind intensity in ``[0.0, 1.0]`` (active during ``wind`` and ``storm``)."""
        if self._type in ("wind", "storm"):
            return self._intensity
        return 0.0

    @property
    def rain_intensity(self) -> float:
        """Effective rain intensity in ``[0.0, 1.0]`` (active during ``rain`` and ``storm``)."""
        if self._type in ("rain", "storm"):
            return self._intensity
        return 0.0

    @property
    def cold_intensity(self) -> float:
        """Effective cold intensity in ``[0.0, 1.0]`` (active during ``cold`` and ``storm``)."""
        if self._type in ("cold", "storm"):
            return self._intensity
        return 0.0

    @property
    def surface_slipperiness(self) -> float:
        """Base surface slipperiness in ``[0.0, 1.0]`` caused by rain or storm."""
        return self.rain_intensity

    @property
    def is_slippery(self) -> bool:
        """True when rain or storm makes exposed surfaces slippery."""
        return self.surface_slipperiness > 0.0

    @property
    def temperature_drop(self) -> float:
        """Ambient temperature drop caused by ``cold`` or ``storm`` weather."""
        if self._type == "cold":
            return COLD_TEMPERATURE_DROP * self._intensity
        if self._type == "storm":
            return STORM_TEMPERATURE_DROP * self._intensity
        return 0.0

    @property
    def temperature_offset(self) -> float:
        """Signed temperature offset (``<= 0.0``) applied to ambient temperature."""
        return -self.temperature_drop

    @property
    def light_multiplier(self) -> float:
        """Multiplicative light factor in ``(0.0, 1.0]`` (lowered during ``storm``)."""
        if self._type == "storm":
            return max(0.0, 1.0 - STORM_LIGHT_REDUCTION * self._intensity)
        return 1.0

    @property
    def light_drop(self) -> float:
        """Fraction of light blocked by ``storm`` weather."""
        return 1.0 - self.light_multiplier

    @property
    def state(self) -> WeatherState:
        """Return a ``WeatherState`` snapshot of the current weather."""
        return WeatherState(
            type=self._type,
            intensity=self._intensity,
            duration=self._duration,
            remaining_ticks=self._remaining_ticks,
            source=self._source,
            wind_direction=self._wind_direction.model_copy(),
        )

    @property
    def current(self) -> WeatherState:
        """Alias for ``state``."""
        return self.state

    @property
    def scheduled(self) -> list[ScheduledWeather]:
        """List of pending scheduled manual weather events."""
        return list(self._scheduled)

    def _sample_duration(self, weather_type: WeatherType) -> int:
        lo, hi = self.config.duration[weather_type]
        if lo == hi:
            return lo
        return self.rng.randint(lo, hi)

    def _sample_intensity(self, weather_type: WeatherType) -> float:
        lo, hi = self.config.intensity[weather_type]
        if lo == hi:
            return lo
        return round(self.rng.uniform(lo, hi), 4)

    def _weighted_choice(self, candidates: Sequence[WeatherType]) -> WeatherType:
        weights = [max(0.0, self.config.chance.get(t, 0.0)) for t in candidates]
        total = sum(weights)
        if total <= 0.0:
            return candidates[0]
        pick = self.rng.uniform(0.0, total)
        upto = 0.0
        for candidate, weight in zip(candidates, weights, strict=True):
            upto += weight
            if pick <= upto:
                return candidate
        return candidates[-1]

    def subscribe(self, handler: Any) -> Any:
        """Subscribe a callback to receive ``WeatherChangedEvent`` notifications."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._subscribers.append(handler)

        def _unsub() -> None:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

        return _unsub

    def _emit_weather_changed(
        self,
        *,
        tick: int,
        source: WeatherSource,
        previous_type: WeatherType,
        previous_intensity: float,
    ) -> None:
        from arden.world.world import WeatherChangedEvent

        ev = WeatherChangedEvent(
            tick=tick,
            source=source,
            previous_weather_type=previous_type,
            weather_type=self._type,
            previous_intensity=previous_intensity,
            intensity=self._intensity,
            duration=self._duration,
            light_level=(
                self._world_ref.light_level
                if self._world_ref is not None
                else self.light_multiplier
            ),
            ambient_temperature=(
                self._world_ref.ambient_temperature
                if self._world_ref is not None
                else 1.0 - self.temperature_drop
            ),
        )
        self.events.append(ev)
        for sub in list(self._subscribers):
            sub(ev)
        if self._world_ref is not None:
            self._world_ref._emit_weather_changed_event(
                tick=tick,
                source=source,
                previous_weather_type=previous_type,
                weather_type=self._type,
                previous_intensity=previous_intensity,
                intensity=self._intensity,
                duration=self._duration,
            )

    def pause_automation(self, paused: bool = True) -> bool:
        """Pause or unpause automated weather transitions."""
        if not isinstance(paused, bool):
            raise TypeError("paused must be a bool")
        self._paused = paused
        self.config.paused = paused
        return self._paused

    def resume_automation(self) -> bool:
        """Resume automated weather transitions."""
        return self.pause_automation(False)

    def set_automation(self, enabled: bool) -> bool:
        """Enable or disable automated weather transitions."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.pause_automation(not enabled)
        return not self._paused

    pause = pause_automation
    resume = resume_automation

    def set_wind_direction(self, direction: Any) -> WindDirection:
        """Set the wind direction to a normalized 2D unit vector."""
        w_dir = _normalize_wind_direction(direction)
        if w_dir.x == 0.0 and w_dir.y == 0.0:
            raise ValueError("wind direction must be non-zero")
        self._wind_direction = w_dir
        return self._wind_direction

    def set_weather(
        self,
        type: WeatherType | None = None,
        intensity: float | None = None,
        duration: int | None = None,
        start_at: int | None = None,
        *,
        weather_type: WeatherType | None = None,
        wind_direction: Any = None,
        current_tick: int | None = None,
    ) -> WeatherState:
        """Set or schedule a manual weather override: ``set_weather(type, intensity, duration, start_at=None)``."""
        raw_type = weather_type if type is None else type
        if raw_type is None:
            raise ValueError("weather type is required")
        w_type = validate_weather_type(raw_type)

        if intensity is None:
            int_val = 0.0 if w_type == "clear" else 0.5
        else:
            int_val = validate_intensity(intensity)

        if duration is None:
            dur_val = self.config.duration[w_type][0]
        else:
            if isinstance(duration, bool) or not isinstance(duration, int | float):
                raise TypeError("duration must be a positive integer")
            if isinstance(duration, float) and (not math.isfinite(duration) or not duration.is_integer()):
                raise ValueError("duration must be an integer number of ticks")
            dur_val = int(duration)
            if dur_val <= 0:
                raise ValueError("duration must be > 0")

        if wind_direction is not None:
            w_dir = _normalize_wind_direction(wind_direction)
        else:
            w_dir = self._wind_direction.model_copy()

        now_tick = (
            current_tick
            if current_tick is not None
            else (self._world_ref.tick if self._world_ref is not None else 0)
        )

        if start_at is not None:
            if isinstance(start_at, bool) or not isinstance(start_at, int | float):
                raise TypeError("start_at must be an integer tick or None")
            if isinstance(start_at, float) and (not math.isfinite(start_at) or not start_at.is_integer()):
                raise ValueError("start_at must be an integer tick")
            start_tick = int(start_at)
            if start_tick < 0 or start_tick < now_tick:
                raise ValueError(f"start_at ({start_tick}) must be >= current tick ({now_tick})")
            if start_tick > now_tick:
                scheduled_ev = ScheduledWeather(
                    type=w_type,
                    intensity=int_val,
                    duration=dur_val,
                    start_at=start_tick,
                    wind_direction=w_dir,
                )
                self._scheduled.append(scheduled_ev)
                self._scheduled.sort(key=lambda s: s.start_at)
                return WeatherState(
                    type=w_type,
                    intensity=int_val,
                    duration=dur_val,
                    remaining_ticks=dur_val,
                    source="manual",
                    wind_direction=w_dir,
                )

        # Activate manual override immediately
        self._activate_manual(
            w_type=w_type,
            intensity=int_val,
            duration=dur_val,
            wind_direction=w_dir,
            tick=now_tick,
            from_schedule=False,
        )
        return self.state

    def _activate_manual(
        self,
        *,
        w_type: WeatherType,
        intensity: float,
        duration: int,
        wind_direction: WindDirection,
        tick: int,
        from_schedule: bool,
    ) -> None:
        if not self._manual_active:
            self._saved_auto = {
                "type": self._type,
                "intensity": self._intensity,
                "duration": self._duration,
                "remaining_ticks": max(1, self._remaining_ticks),
                "wind_direction": self._wind_direction.model_dump(mode="json"),
            }

        prev_type = self._type
        prev_intensity = self._intensity

        self._type = w_type
        self._intensity = intensity
        self._duration = duration
        self._remaining_ticks = duration
        self._source = "manual"
        self._manual_active = True
        self._wind_direction = wind_direction
        self._just_activated_tick = tick if from_schedule else None

        self._emit_weather_changed(
            tick=tick,
            source="manual",
            previous_type=prev_type,
            previous_intensity=prev_intensity,
        )

    def _resume_automated_after_manual(self, tick: int) -> None:
        prev_type = self._type
        prev_intensity = self._intensity
        self._manual_active = False
        self._just_activated_tick = None

        saved = self._saved_auto
        self._saved_auto = None

        allowed = list(self.config.transitions.get(prev_type, []))
        positive_allowed = [t for t in allowed if self.config.chance.get(t, 0.0) > 0.0]

        if (
            saved is not None
            and saved["type"] != prev_type
            and (not allowed or saved["type"] in allowed)
        ):
            self._type = validate_weather_type(saved["type"])
            self._intensity = float(saved["intensity"])
            self._duration = int(saved["duration"])
            self._remaining_ticks = max(1, int(saved["remaining_ticks"]))
            self._source = "automated"
        else:
            distinct_pool = [t for t in positive_allowed if t != prev_type]
            pool = distinct_pool or positive_allowed or allowed or ["clear"]
            if self._paused:
                next_type: WeatherType = (
                    "clear" if "clear" in pool else validate_weather_type(pool[0])
                )
                self._type = next_type
                self._intensity = self.config.intensity[next_type][0]
                self._duration = self.config.duration[next_type][0]
                self._remaining_ticks = self._duration
                self._source = "automated"
            else:
                next_type = self._weighted_choice(pool)
                self._type = next_type
                self._duration = self._sample_duration(next_type)
                self._remaining_ticks = self._duration
                self._intensity = self._sample_intensity(next_type)
                self._source = "automated"

        self._emit_weather_changed(
            tick=tick,
            source="automated",
            previous_type=prev_type,
            previous_intensity=prev_intensity,
        )

    def _transition_automated(self, next_type: WeatherType, *, tick: int) -> None:
        prev_type = self._type
        prev_intensity = self._intensity
        self._type = next_type
        self._duration = self._sample_duration(next_type)
        self._remaining_ticks = self._duration
        self._intensity = self._sample_intensity(next_type)
        self._source = "automated"
        self._manual_active = False
        self._emit_weather_changed(
            tick=tick,
            source="automated",
            previous_type=prev_type,
            previous_intensity=prev_intensity,
        )

    def begin_tick(self, tick: int) -> None:
        """Activate any scheduled manual weather event due at ``tick``."""
        due: list[ScheduledWeather] = []
        remaining: list[ScheduledWeather] = []
        for item in self._scheduled:
            if item.start_at <= tick:
                due.append(item)
            else:
                remaining.append(item)
        self._scheduled = remaining

        if due:
            chosen = due[-1]
            self._activate_manual(
                w_type=chosen.type,
                intensity=chosen.intensity,
                duration=chosen.duration,
                wind_direction=chosen.wind_direction,
                tick=tick,
                from_schedule=True,
            )

    def end_tick(self, tick: int) -> None:
        """Advance weather duration counters and run automated transitions at the end of ``tick``."""
        if self._manual_active:
            if self._just_activated_tick == tick:
                self._just_activated_tick = None
                return
            self._remaining_ticks = max(0, self._remaining_ticks - 1)
            if self._remaining_ticks <= 0:
                self._resume_automated_after_manual(tick)
            return

        if self._paused:
            return

        self._remaining_ticks -= 1
        if self._remaining_ticks > 0:
            return

        current = self._type
        allowed = list(self.config.transitions.get(current, []))
        candidates = [t for t in allowed if self.config.chance.get(t, 0.0) > 0.0]

        if not candidates:
            self._duration = self.config.ticks_per_hour
            self._remaining_ticks = self._duration
            return

        if current == "clear":
            non_clear = [t for t in candidates if t != "clear"]
            if non_clear:
                p_start = 1.0 - math.prod(
                    1.0 - min(1.0, max(0.0, self.config.chance[t]))
                    for t in non_clear
                )
                if "clear" not in allowed or self.rng.random() < p_start:
                    next_type = self._weighted_choice(non_clear)
                    self._transition_automated(next_type, tick=tick)
                else:
                    self._duration = self.config.ticks_per_hour
                    self._remaining_ticks = self._duration
            else:
                self._duration = self._sample_duration("clear")
                self._remaining_ticks = self._duration
            return

        distinct_candidates = [t for t in candidates if t != current]
        pool = distinct_candidates if distinct_candidates else candidates
        next_type = self._weighted_choice(pool)
        if next_type != current:
            self._transition_automated(next_type, tick=tick)
        else:
            self._duration = self._sample_duration(current)
            self._remaining_ticks = self._duration

    def step(self, tick: int | None = None) -> WeatherState:
        """Advance the weather system by one tick (for standalone testing without a ``World``)."""
        if tick is not None:
            resolved_tick = tick
            self._tick_counter = tick
        elif self._world_ref is not None:
            resolved_tick = self._world_ref.tick
        else:
            self._tick_counter += 1
            resolved_tick = self._tick_counter
        self.begin_tick(resolved_tick)
        self.end_tick(resolved_tick)
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Serialize full weather system state (including its RNG and scheduled events)."""
        version, internal_state, gauss_next = self.rng.getstate()
        return {
            "config": self.config.model_dump(mode="json"),
            "type": self._type,
            "intensity": self._intensity,
            "duration": self._duration,
            "remaining_ticks": self._remaining_ticks,
            "source": self._source,
            "paused": self._paused,
            "manual_active": self._manual_active,
            "just_activated_tick": self._just_activated_tick,
            "saved_auto": self._saved_auto,
            "wind_direction": self._wind_direction.model_dump(mode="json"),
            "scheduled": [s.model_dump(mode="json") for s in self._scheduled],
            "rng_state": [version, list(internal_state), gauss_next],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WeatherSystem:
        """Restore a ``WeatherSystem`` from ``to_dict()`` output."""
        if not isinstance(data, Mapping):
            raise TypeError("weather snapshot data must be a mapping")
        system = cls(config=data.get("config", {}))
        system._type = validate_weather_type(data.get("type", "clear"))
        system._intensity = validate_intensity(data.get("intensity", 0.0))
        system._duration = max(1, int(data.get("duration", 60)))
        system._remaining_ticks = max(0, int(data.get("remaining_ticks", system._duration)))
        raw_source = str(data.get("source", "automated"))
        system._source = "manual" if raw_source == "manual" else "automated"
        system._paused = bool(data.get("paused", system.config.paused))
        system.config.paused = system._paused
        system._manual_active = bool(data.get("manual_active", False))
        raw_jat = data.get("just_activated_tick")
        system._just_activated_tick = int(raw_jat) if isinstance(raw_jat, int) and not isinstance(raw_jat, bool) else None
        raw_saved = data.get("saved_auto")
        system._saved_auto = dict(raw_saved) if isinstance(raw_saved, Mapping) else None
        if "wind_direction" in data and data["wind_direction"] is not None:
            system._wind_direction = WindDirection.model_validate(data["wind_direction"])
        raw_sched = data.get("scheduled", [])
        if isinstance(raw_sched, Sequence):
            system._scheduled = [ScheduledWeather.model_validate(item) for item in raw_sched]

        rng_state = data.get("rng_state")
        if isinstance(rng_state, list | tuple) and len(rng_state) == 3:
            version, internal_state, gauss_next = rng_state
            if isinstance(internal_state, list | tuple):
                system.rng.setstate((int(version), tuple(int(x) for x in internal_state), gauss_next))
        return system

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._type == other
        return super().__eq__(other)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


def compute_shelter_exposure(
    world: World,
    target: Position | WorldObject | AgentBody | Mapping[str, Any] | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Return ``(rain_exposure, wind_exposure)`` in ``[0.0, 1.0]`` for ``target`` in ``world``.

    - ``(1.0, 1.0)`` when exposed in the open.
    - ``(0.0, 0.0)`` when inside a hollow object (``level == 'inside:object_<n>'``
      or located inside a hollow object's footprint).
    - ``(TREE_SHELTER_EXPOSURE, TREE_SHELTER_EXPOSURE)`` (``0.20``) when under a
      large fixed object such as a tree (``fixed=True`` and ``height >= 1.2`` or ``volume >= 0.10``).
    """
    from arden.world.objects import WorldObject
    from arden.world.physics import _object_elevation, extract_xy, object_top_height

    entity_id: str | None = None
    entity_level: str = "ground"

    if isinstance(target, WorldObject):
        entity_id = target.id
        entity_level = target.level
        tx, ty = target.position.x, target.position.y
    elif hasattr(target, "position") and hasattr(target, "level"):
        entity_level = str(getattr(target, "level", "ground"))
        tx, ty = extract_xy(target)  # type: ignore[arg-type]
    else:
        tx, ty = extract_xy(target)

    # 1. Explicit inside:object_<n> level
    if entity_level.startswith("inside:"):
        container_id = entity_level.removeprefix("inside:")
        container = world.objects.get(container_id)
        if container is not None and container.hollow:
            return (HOLLOW_SHELTER_EXPOSURE, HOLLOW_SHELTER_EXPOSURE)

    exposure = 1.0

    for obj in world.objects.values():
        if entity_id is not None and obj.id == entity_id:
            continue

        # Check if target is listed in a hollow object's contents
        if obj.hollow:
            contents = obj.state.get("contents")
            if entity_id is not None and isinstance(contents, list) and entity_id in contents:
                return (HOLLOW_SHELTER_EXPOSURE, HOLLOW_SHELTER_EXPOSURE)
            # If the agent or a queried position is within the hollow object's radius
            dist = math.hypot(tx - obj.position.x, ty - obj.position.y)
            hollow_shelter_radius = max(obj.radius, 0.35)
            if entity_id is None and dist <= hollow_shelter_radius + 1e-6:
                return (HOLLOW_SHELTER_EXPOSURE, HOLLOW_SHELTER_EXPOSURE)

        # Check if target is under a large fixed object (tree)
        if obj.fixed and (obj.height >= TREE_MIN_HEIGHT or obj.volume >= TREE_MIN_VOLUME):
            if entity_level == f"on_top_of:{obj.id}":
                continue
            dist = math.hypot(tx - obj.position.x, ty - obj.position.y)
            canopy_radius = max(obj.radius * 2.5, obj.height * 0.75, 1.5)
            if dist <= canopy_radius + 1e-6:
                top_h = object_top_height(obj, world.objects, world.terrain)
                if entity_id is not None and entity_id in world.objects:
                    ent_elev = _object_elevation(world.objects[entity_id], world.objects, world.terrain)
                else:
                    ent_elev = 0.0
                if ent_elev < top_h:
                    exposure = min(exposure, TREE_SHELTER_EXPOSURE)

    return (exposure, exposure)


def set_weather(
    world_or_type: World | WeatherSystem | str,
    type_or_intensity: Any = None,
    intensity_or_duration: Any = None,
    duration_or_start_at: Any = None,
    start_at: int | None = None,
    *,
    world: World | None = None,
    intensity: float | None = None,
    duration: int | None = None,
    wind_direction: Any = None,
) -> WeatherState:
    """Set or schedule weather on ``world`` (or the active ``World``).

    Supports both:
    - ``set_weather(type, intensity, duration, start_at=None)``
    - ``set_weather(world, type, intensity, duration, start_at=None)``
    """
    if isinstance(world_or_type, str):
        from arden.world.physics import _get_active_world

        target_world = world if world is not None else _get_active_world()
        if target_world is None:
            raise RuntimeError("no active World found; pass world explicitly or instantiate a World first")
        w_type = world_or_type
        w_intensity = intensity if intensity is not None else type_or_intensity
        w_duration = duration if duration is not None else intensity_or_duration
        w_start_at = start_at if start_at is not None else duration_or_start_at
        return target_world.set_weather(
            w_type,
            w_intensity,
            w_duration,
            start_at=w_start_at,
            wind_direction=wind_direction,
        )

    if isinstance(world_or_type, WeatherSystem):
        w_type = str(type_or_intensity)
        w_intensity = intensity if intensity is not None else intensity_or_duration
        w_duration = duration if duration is not None else duration_or_start_at
        return world_or_type.set_weather(
            w_type,  # type: ignore[arg-type]
            w_intensity,
            w_duration,
            start_at=start_at,
            wind_direction=wind_direction,
        )

    if hasattr(world_or_type, "set_weather"):
        w_type = str(type_or_intensity)
        w_intensity = intensity if intensity is not None else intensity_or_duration
        w_duration = duration if duration is not None else duration_or_start_at
        return world_or_type.set_weather(
            w_type,
            w_intensity,
            w_duration,
            start_at=start_at,
            wind_direction=wind_direction,
        )

    raise TypeError("set_weather expects (type, intensity, duration, start_at=None) or (world, ...)")
