"""Config defaults, file overrides, validation, and the seeded RNG."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from arden.config import Settings, load_config, make_rng

_GLOBAL_RANDOM_CALLS = (
    "random.seed(",
    "random.random(",
    "random.choice(",
    "random.randint(",
    "random.randrange(",
    "random.uniform(",
    "random.shuffle(",
)


def test_defaults_when_config_file_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_config()
    assert settings == Settings()
    assert settings.world.tick_rate == 10.0
    assert settings.world.seed == 0
    assert settings.llm.backend == "mock"
    assert settings.logging.level == "INFO"


def test_file_override(tmp_path: Path) -> None:
    config_path = tmp_path / "arden.toml"
    config_path.write_text(
        '[world]\ntick_rate = 2.5\nseed = 42\n\n[llm]\nbackend = "mock"\n\n[logging]\nlevel = "warning"\n',
        encoding="utf-8",
    )
    settings = load_config(config_path)
    assert settings.world.tick_rate == 2.5
    assert settings.world.seed == 42
    assert settings.llm.backend == "mock"
    assert settings.logging.level == "WARNING"


def test_partial_file_keeps_other_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text("[world]\nseed = 7\n", encoding="utf-8")
    settings = load_config(config_path)
    assert settings.world.seed == 7
    assert settings.world.tick_rate == 10.0
    assert settings.llm.backend == "mock"
    assert settings.logging.level == "INFO"


def test_invalid_tick_rate(tmp_path: Path) -> None:
    config_path = tmp_path / "arden.toml"
    config_path.write_text("[world]\ntick_rate = 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(config_path)


def test_example_toml_matches_defaults() -> None:
    example = Path(__file__).resolve().parents[1] / "arden.example.toml"
    assert load_config(example) == Settings()


def test_make_rng_is_reproducible_and_ignores_global_random() -> None:
    settings = Settings(world={"seed": 1234})
    before = random.getstate()
    first = make_rng(settings)
    second = make_rng(settings)
    assert first.getstate() == second.getstate()
    assert [first.random() for _ in range(4)] == [second.random() for _ in range(4)]
    assert random.getstate() == before

    other = make_rng(Settings(world={"seed": 1235}))
    assert other.getstate() != make_rng(settings).getstate()


def test_package_does_not_call_global_random() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "arden"
    sources = [path.read_text(encoding="utf-8") for path in package.rglob("*.py")]
    combined = "\n".join(sources)
    for call in _GLOBAL_RANDOM_CALLS:
        assert call not in combined
