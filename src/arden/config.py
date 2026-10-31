"""Load typed settings from arden.toml."""

from __future__ import annotations

import random
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
DEFAULT_CONFIG_PATH = Path("arden.toml")


class WorldSettings(BaseModel):
    """Fixed-tick world clock and the seed for the simulation RNG."""

    model_config = ConfigDict(extra="forbid")

    tick_rate: float = Field(default=10.0, gt=0)
    seed: int = 0
    sensing_radius: float | None = Field(default=None, gt=0)


class PerceptionSettings(BaseModel):
    """Sensing radius, reach distance, and delta thresholds for perception."""

    model_config = ConfigDict(extra="forbid")

    sensing_radius: float = Field(default=10.0, gt=0)
    reach_radius: float = Field(default=1.5, gt=0)
    move_threshold: float = Field(default=0.1, ge=0.0)
    color_visibility_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class LlmSettings(BaseModel):
    """Which model backend the cognition layer should call."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "mock"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)

    @field_validator("backend", "base_url", "model")
    @classmethod
    def string_must_be_non_empty(cls, value: str, info: object) -> str:
        if not value.strip():
            field_name = getattr(info, "field_name", "field")
            raise ValueError(f"{field_name} must be a non-empty string")
        return value


class LoggingSettings(BaseModel):
    """Process logging. This is not the ground-truth observer log."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def level_must_be_known(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _LOG_LEVELS:
            allowed = ", ".join(sorted(_LOG_LEVELS))
            raise ValueError(f"level must be one of: {allowed}")
        return normalized


class Settings(BaseModel):
    """Top-level Arden settings."""

    model_config = ConfigDict(extra="forbid")

    world: WorldSettings = Field(default_factory=WorldSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_config(path: Path | None = None) -> Settings:
    """Load settings from TOML.

    With no ``path``, read ``arden.toml`` in the current working directory when
    it exists, and return defaults when it does not. An explicit path that is
    missing raises ``FileNotFoundError``.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
        if not path.is_file():
            return Settings()
    else:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return Settings.model_validate(raw)


def make_rng(settings: Settings) -> random.Random:
    """Return the single RNG for a run.

    The generator is seeded from ``settings.world.seed`` and does not read or
    write the process-wide ``random`` module. Callers pass the instance into
    the simulation. Each call returns a new generator at the start of the seed,
    so two runs with the same config replay the same sequence.
    """
    return random.Random(settings.world.seed)
