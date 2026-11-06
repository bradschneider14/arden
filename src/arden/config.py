"""Load typed settings from arden.toml."""

from __future__ import annotations

import random
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arden.agent.gate import GateConfig, GateSettings
from arden.environment.body import BodyConfig, BodySettings
from arden.environment.daynight import DayNightConfig, DayNightSettings
from arden.environment.weather import WeatherConfig, WeatherSettings

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
DEFAULT_CONFIG_PATH = Path("arden.toml")
DEFAULT_BUFFER_SIZE: int = 40
DEFAULT_REST_IDLE_TICKS: int = 10
DEFAULT_DB_PATH: Path = Path("data/memory.db")
DEFAULT_DECAY_RATE: float = 0.005
DEFAULT_SALIENCE_FLOOR: float = 0.10
DEFAULT_RECALL_BOOST: float = 0.15
DEFAULT_DISCOMFORT_SALIENCE_BOOST: float = 0.25
DEFAULT_SURPRISE_SALIENCE_BOOST: float = 0.35


class MemoryConfig(BaseModel):
    """Configuration for the short-term buffer, reflection, and episodic store under ``[memory]``."""

    model_config = ConfigDict(extra="forbid")

    buffer_size: int = Field(default=DEFAULT_BUFFER_SIZE, gt=0)
    rest_idle_ticks: int = Field(default=DEFAULT_REST_IDLE_TICKS, gt=0)
    db_path: Path = Field(default=DEFAULT_DB_PATH)
    decay_rate: float = Field(default=DEFAULT_DECAY_RATE, ge=0.0, le=1.0)
    salience_floor: float = Field(default=DEFAULT_SALIENCE_FLOOR, ge=0.0, le=1.0)
    recall_boost: float = Field(default=DEFAULT_RECALL_BOOST, ge=0.0, le=1.0)
    discomfort_salience_boost: float = Field(
        default=DEFAULT_DISCOMFORT_SALIENCE_BOOST, ge=0.0, le=1.0
    )
    surprise_salience_boost: float = Field(
        default=DEFAULT_SURPRISE_SALIENCE_BOOST, ge=0.0, le=1.0
    )

    @field_validator("db_path", mode="before")
    @classmethod
    def _validate_db_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            if not str(value).strip():
                raise ValueError("db_path must be a non-empty path")
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("db_path must be a non-empty string")
            return Path(stripped)
        raise TypeError("db_path must be a Path or str")

    @property
    def capacity(self) -> int:
        """Alias for ``buffer_size``."""
        return self.buffer_size

    @property
    def short_term_size(self) -> int:
        """Alias for ``buffer_size``."""
        return self.buffer_size

    @property
    def max_size(self) -> int:
        """Alias for ``buffer_size``."""
        return self.buffer_size

    @property
    def idle_ticks(self) -> int:
        """Alias for ``rest_idle_ticks``."""
        return self.rest_idle_ticks

    @property
    def rest_ticks(self) -> int:
        """Alias for ``rest_idle_ticks``."""
        return self.rest_idle_ticks

    @property
    def floor(self) -> float:
        """Alias for ``salience_floor``."""
        return self.salience_floor

    @property
    def min_salience(self) -> float:
        """Alias for ``salience_floor``."""
        return self.salience_floor


MemorySettings = MemoryConfig


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
    daynight: DayNightSettings = Field(default_factory=DayNightSettings)
    body: BodySettings = Field(default_factory=BodySettings)
    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    gate: GateSettings = Field(default_factory=GateSettings)
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


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "BodyConfig",
    "BodySettings",
    "DayNightConfig",
    "DayNightSettings",
    "GateConfig",
    "GateSettings",
    "LlmSettings",
    "LoggingSettings",
    "MemoryConfig",
    "MemorySettings",
    "PerceptionSettings",
    "Settings",
    "WeatherConfig",
    "WeatherSettings",
    "WorldSettings",
    "load_config",
    "make_rng",
]
