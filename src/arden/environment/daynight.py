"""Continuous day and night cycle driving light, visibility, and ambient temperature."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIDNIGHT: float = 0.0
DAWN: float = 0.25
NOON: float = 0.5
DUSK: float = 0.75

DEFAULT_DAY_LENGTH: int = 600
DEFAULT_DAYLIGHT_SHARE: float = 0.5
DEFAULT_MIN_VISIBILITY: float = 0.25
DEFAULT_TEMPERATURE_SWING: float = 0.4
DEFAULT_BASE_TEMPERATURE: float = 1.0
COARSE_LIGHT_DECIMALS: int = 2


def normalize_time_of_day(fraction: float) -> float:
    """Validate and normalize a time-of-day fraction in ``[0.0, 1.0]`` into ``[0.0, 1.0)``."""
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise TypeError("time_of_day fraction must be a finite number in [0.0, 1.0]")
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("time_of_day fraction must be a finite number in [0.0, 1.0]")
    if value >= 1.0:
        return 0.0
    return 0.0 if value == 0.0 else value


def compute_light_level(
    time_of_day: float,
    daylight_share: float = DEFAULT_DAYLIGHT_SHARE,
) -> float:
    """Return the smooth light level in ``[0.0, 1.0]`` for ``time_of_day``.

    - ``0.0`` (midnight) is darkest (``light = 0.0``).
    - ``0.25`` (dawn) and ``0.75`` (dusk) are smooth transitions.
    - ``0.5`` (noon) is brightest (``light = 1.0``).
    - ``daylight_share`` shapes the curve exponent so that the fraction of the
      cycle with ``light >= 0.5`` equals ``daylight_share`` with no hard switches.
    """
    t = normalize_time_of_day(time_of_day)
    if isinstance(daylight_share, bool) or not isinstance(daylight_share, int | float):
        raise TypeError("daylight_share must be a finite number in [0.0, 1.0]")
    share = float(daylight_share)
    if not math.isfinite(share) or share < 0.0 or share > 1.0:
        raise ValueError("daylight_share must be a finite number in [0.0, 1.0]")

    if share <= 0.0:
        return 0.0
    if share >= 1.0:
        return 1.0

    raw = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    if raw <= 0.0:
        return 0.0
    if raw >= 1.0:
        return 1.0

    edge_raw = math.cos(0.5 * math.pi * share) ** 2
    gamma = math.log(0.5) / math.log(edge_raw)
    light = raw**gamma
    return max(0.0, min(1.0, float(light)))


def coarse_light_level(light: float, decimals: int = COARSE_LIGHT_DECIMALS) -> float:
    """Quantize ``light`` to a coarse numeric value in ``[0.0, 1.0]`` with no labels."""
    if isinstance(light, bool) or not isinstance(light, int | float):
        raise TypeError("light must be a finite number in [0.0, 1.0]")
    val = float(light)
    if not math.isfinite(val):
        raise ValueError("light must be a finite number")
    clamped = max(0.0, min(1.0, val))
    rounded = round(clamped, decimals)
    return 0.0 if rounded == 0.0 else rounded


def compute_visibility_factor(
    light: float,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> float:
    """Map ``light`` in ``[0.0, 1.0]`` to perception ``visibility_factor`` in ``[min_visibility, 1.0]``."""
    if isinstance(light, bool) or not isinstance(light, int | float):
        raise TypeError("light must be a finite number in [0.0, 1.0]")
    if isinstance(min_visibility, bool) or not isinstance(min_visibility, int | float):
        raise TypeError("min_visibility must be a finite number in [0.0, 1.0]")
    l_val = float(light)
    mv_val = float(min_visibility)
    if not math.isfinite(l_val) or l_val < 0.0 or l_val > 1.0:
        raise ValueError("light must be a finite number in [0.0, 1.0]")
    if not math.isfinite(mv_val) or mv_val < 0.0 or mv_val > 1.0:
        raise ValueError("min_visibility must be a finite number in [0.0, 1.0]")
    vf = mv_val + (1.0 - mv_val) * l_val
    return max(0.0, min(1.0, vf))


def compute_ambient_temperature(
    light: float,
    temperature_swing: float = DEFAULT_TEMPERATURE_SWING,
    *,
    base_temperature: float = DEFAULT_BASE_TEMPERATURE,
) -> float:
    """Compute smooth ambient temperature from ``light`` and ``temperature_swing``."""
    if isinstance(light, bool) or not isinstance(light, int | float):
        raise TypeError("light must be a finite number in [0.0, 1.0]")
    if isinstance(temperature_swing, bool) or not isinstance(temperature_swing, int | float):
        raise TypeError("temperature_swing must be a finite non-negative number")
    if isinstance(base_temperature, bool) or not isinstance(base_temperature, int | float):
        raise TypeError("base_temperature must be a finite number")
    l_val = float(light)
    swing_val = float(temperature_swing)
    base_val = float(base_temperature)
    if not math.isfinite(l_val) or l_val < 0.0 or l_val > 1.0:
        raise ValueError("light must be a finite number in [0.0, 1.0]")
    if not math.isfinite(swing_val) or swing_val < 0.0:
        raise ValueError("temperature_swing must be a finite non-negative number")
    if not math.isfinite(base_val):
        raise ValueError("base_temperature must be a finite number")
    return base_val - (1.0 - l_val) * swing_val


class DayNightConfig(BaseModel):
    """Configuration for the continuous day and night cycle."""

    model_config = ConfigDict(extra="forbid")

    day_length: int = Field(default=DEFAULT_DAY_LENGTH, gt=0)
    daylight_share: float = Field(default=DEFAULT_DAYLIGHT_SHARE, ge=0.0, le=1.0)
    min_visibility: float = Field(default=DEFAULT_MIN_VISIBILITY, ge=0.0, le=1.0)
    temperature_swing: float = Field(default=DEFAULT_TEMPERATURE_SWING, ge=0.0)
    base_temperature: float = Field(default=DEFAULT_BASE_TEMPERATURE)

    @field_validator("day_length", mode="before")
    @classmethod
    def _validate_day_length(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("day_length must be a positive integer")
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError("day_length must be an integer number of ticks")
            return int(value)
        return value

    @field_validator("daylight_share", "min_visibility", "temperature_swing", "base_temperature")
    @classmethod
    def _validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("day/night configuration values must be finite")
        return value


DayNightSettings = DayNightConfig


class DayNightCycle:
    """Continuous day and night cycle state and smooth environmental curves."""

    def __init__(
        self,
        config: DayNightConfig | Mapping[str, Any] | None = None,
        *,
        day_length: int | None = None,
        daylight_share: float | None = None,
        min_visibility: float | None = None,
        temperature_swing: float | None = None,
        base_temperature: float | None = None,
        time_of_day: float = NOON,
        frozen: bool = False,
    ) -> None:
        if config is None:
            raw_cfg: dict[str, Any] = {}
        elif isinstance(config, DayNightConfig):
            raw_cfg = config.model_dump()
        elif isinstance(config, Mapping):
            raw_cfg = dict(config)
        else:
            raise TypeError("config must be a DayNightConfig, mapping, or None")

        if day_length is not None:
            raw_cfg["day_length"] = day_length
        if daylight_share is not None:
            raw_cfg["daylight_share"] = daylight_share
        if min_visibility is not None:
            raw_cfg["min_visibility"] = min_visibility
        if temperature_swing is not None:
            raw_cfg["temperature_swing"] = temperature_swing
        if base_temperature is not None:
            raw_cfg["base_temperature"] = base_temperature

        self.config: DayNightConfig = DayNightConfig.model_validate(raw_cfg)
        self._time_of_day: float = normalize_time_of_day(time_of_day)
        if not isinstance(frozen, bool):
            raise TypeError("frozen must be a bool")
        self._frozen: bool = frozen

    @property
    def day_length(self) -> int:
        """Cycle length in ticks."""
        return self.config.day_length

    @property
    def daylight_share(self) -> float:
        """Fraction of the cycle that is daytime."""
        return self.config.daylight_share

    @property
    def min_visibility(self) -> float:
        """Minimum visibility factor at midnight."""
        return self.config.min_visibility

    @property
    def temperature_swing(self) -> float:
        """Temperature drop between noon and midnight."""
        return self.config.temperature_swing

    @property
    def base_temperature(self) -> float:
        """Daytime peak ambient temperature at noon."""
        return self.config.base_temperature

    @property
    def time_of_day(self) -> float:
        """Current cycle fraction in ``[0.0, 1.0)`` (0.0=midnight, 0.5=noon)."""
        return self._time_of_day

    @property
    def fraction(self) -> float:
        """Alias for ``time_of_day``."""
        return self._time_of_day

    @property
    def frozen(self) -> bool:
        """Return True when the cycle is frozen."""
        return self._frozen

    @property
    def is_frozen(self) -> bool:
        """Alias for ``frozen``."""
        return self._frozen

    @property
    def light(self) -> float:
        """Current smooth light level in ``[0.0, 1.0]``."""
        return compute_light_level(self._time_of_day, self.config.daylight_share)

    @property
    def light_level(self) -> float:
        """Alias for ``light``."""
        return self.light

    @property
    def coarse_light(self) -> float:
        """Coarse numeric light level in ``[0.0, 1.0]`` for perception."""
        return coarse_light_level(self.light)

    @property
    def coarse_light_level(self) -> float:
        """Alias for ``coarse_light``."""
        return self.coarse_light

    @property
    def visibility_factor(self) -> float:
        """Current perception ``visibility_factor`` in ``[min_visibility, 1.0]``."""
        return compute_visibility_factor(self.light, self.config.min_visibility)

    @property
    def ambient_temperature(self) -> float:
        """Current smooth ambient temperature."""
        return compute_ambient_temperature(
            self.light,
            self.config.temperature_swing,
            base_temperature=self.config.base_temperature,
        )

    @property
    def temperature(self) -> float:
        """Alias for ``ambient_temperature``."""
        return self.ambient_temperature

    def light_at(self, time_of_day: float) -> float:
        """Evaluate the smooth light curve at ``time_of_day``."""
        return compute_light_level(time_of_day, self.config.daylight_share)

    def visibility_at(self, time_of_day: float) -> float:
        """Evaluate ``visibility_factor`` at ``time_of_day``."""
        return compute_visibility_factor(self.light_at(time_of_day), self.config.min_visibility)

    def temperature_at(self, time_of_day: float) -> float:
        """Evaluate smooth ambient temperature at ``time_of_day``."""
        return compute_ambient_temperature(
            self.light_at(time_of_day),
            self.config.temperature_swing,
            base_temperature=self.config.base_temperature,
        )

    def effective_sensing_radius(self, day_radius: float) -> float:
        """Return the effective sensing radius for a daytime ``day_radius``."""
        if isinstance(day_radius, bool) or not isinstance(day_radius, int | float):
            raise TypeError("day_radius must be a positive number")
        r = float(day_radius)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError("day_radius must be a finite non-negative number")
        return r * self.visibility_factor

    def set_time_of_day(self, fraction: float) -> float:
        """Directly set ``time_of_day`` to ``fraction`` in ``[0.0, 1.0]``."""
        self._time_of_day = normalize_time_of_day(fraction)
        return self._time_of_day

    def freeze_cycle(self, frozen: bool = True) -> bool:
        """Freeze or unfreeze the day/night cycle."""
        if not isinstance(frozen, bool):
            raise TypeError("frozen must be a bool")
        self._frozen = frozen
        return self._frozen

    freeze = freeze_cycle

    def step(self, ticks: int = 1) -> float:
        """Advance the cycle by ``ticks`` when not frozen."""
        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise TypeError("ticks must be an integer")
        if ticks < 0:
            raise ValueError("ticks must be >= 0")
        if self._frozen or ticks == 0:
            return self._time_of_day
        advanced = (self._time_of_day + (ticks / self.config.day_length)) % 1.0
        self._time_of_day = 0.0 if advanced == 0.0 else advanced
        return self._time_of_day

    def to_dict(self) -> dict[str, Any]:
        """Serialize cycle configuration and state to a JSON-compatible dictionary."""
        return {
            "config": self.config.model_dump(mode="json"),
            "time_of_day": self._time_of_day,
            "frozen": self._frozen,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DayNightCycle:
        """Restore a ``DayNightCycle`` from ``to_dict()`` output."""
        if not isinstance(data, Mapping):
            raise TypeError("daynight snapshot data must be a mapping")
        raw_cfg = data.get("config", {})
        time_of_day = data.get("time_of_day", NOON)
        frozen = data.get("frozen", False)
        return cls(config=raw_cfg, time_of_day=time_of_day, frozen=frozen)
