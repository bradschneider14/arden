"""Environment systems: day and night cycle, light, visibility, and temperature."""

from __future__ import annotations

from arden.environment.daynight import (
    COARSE_LIGHT_DECIMALS,
    DAWN,
    DEFAULT_BASE_TEMPERATURE,
    DEFAULT_DAY_LENGTH,
    DEFAULT_DAYLIGHT_SHARE,
    DEFAULT_MIN_VISIBILITY,
    DEFAULT_TEMPERATURE_SWING,
    DUSK,
    MIDNIGHT,
    NOON,
    DayNightConfig,
    DayNightCycle,
    DayNightSettings,
    coarse_light_level,
    compute_ambient_temperature,
    compute_light_level,
    compute_visibility_factor,
    normalize_time_of_day,
)

__all__ = [
    "COARSE_LIGHT_DECIMALS",
    "DAWN",
    "DEFAULT_BASE_TEMPERATURE",
    "DEFAULT_DAY_LENGTH",
    "DEFAULT_DAYLIGHT_SHARE",
    "DEFAULT_MIN_VISIBILITY",
    "DEFAULT_TEMPERATURE_SWING",
    "DUSK",
    "MIDNIGHT",
    "NOON",
    "DayNightConfig",
    "DayNightCycle",
    "DayNightSettings",
    "coarse_light_level",
    "compute_ambient_temperature",
    "compute_light_level",
    "compute_visibility_factor",
    "normalize_time_of_day",
]
