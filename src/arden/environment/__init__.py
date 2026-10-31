"""Environment systems: day/night cycle, light, visibility, temperature, body state, and drives."""

from __future__ import annotations

from typing import Any

from arden.environment.body import (
    BODY_PRESETS,
    DEFAULT_BODY_CONFIG,
    DEFAULT_VERB_ENERGY_WEIGHTS,
    GENTLE_BODY_CONFIG,
    HARSH_BODY_CONFIG,
    BodyConfig,
    BodyModel,
    BodySettings,
    BodyStateModel,
    BodyStateSystem,
    DriveState,
    DriveThresholds,
    Drives,
    PresetName,
    activity_energy_cost,
    compute_body_modifiers,
    compute_memory_salience,
    compute_shelter_level,
    eat_object,
    extract_weather_conditions,
    format_numeric_body_state,
    is_edible_object,
    memory_salience,
    resolve_body_config,
)
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


def __getattr__(name: str) -> Any:
    if name == "BodyModifiers":
        from arden.actions.verbs import BodyModifiers

        return BodyModifiers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BODY_PRESETS",
    "COARSE_LIGHT_DECIMALS",
    "DAWN",
    "DEFAULT_BASE_TEMPERATURE",
    "DEFAULT_BODY_CONFIG",
    "DEFAULT_DAY_LENGTH",
    "DEFAULT_DAYLIGHT_SHARE",
    "DEFAULT_MIN_VISIBILITY",
    "DEFAULT_TEMPERATURE_SWING",
    "DEFAULT_VERB_ENERGY_WEIGHTS",
    "DUSK",
    "GENTLE_BODY_CONFIG",
    "HARSH_BODY_CONFIG",
    "MIDNIGHT",
    "NOON",
    "BodyConfig",
    "BodyModel",
    "BodyModifiers",
    "BodySettings",
    "BodyStateModel",
    "BodyStateSystem",
    "DayNightConfig",
    "DayNightCycle",
    "DayNightSettings",
    "DriveState",
    "DriveThresholds",
    "Drives",
    "PresetName",
    "activity_energy_cost",
    "coarse_light_level",
    "compute_ambient_temperature",
    "compute_body_modifiers",
    "compute_light_level",
    "compute_memory_salience",
    "compute_shelter_level",
    "compute_visibility_factor",
    "eat_object",
    "extract_weather_conditions",
    "format_numeric_body_state",
    "is_edible_object",
    "memory_salience",
    "normalize_time_of_day",
    "resolve_body_config",
]
