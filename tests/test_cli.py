"""CLI entry point: version, help, and the placeholder run command."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from arden import __version__
from arden.__main__ import main, run_simulation
from arden.config import Settings, make_rng


def test_version_flag() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arden", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == __version__


def test_help_lists_run() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arden", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run" in completed.stdout


def test_run_prints_not_implemented(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arden", "run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.stdout.strip() == "not implemented yet"


def test_run_passes_seeded_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "arden.toml"
    config_path.write_text("[world]\nseed = 99\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def spy(settings: Settings, rng: random.Random) -> None:
        captured["settings"] = settings
        captured["rng"] = rng
        print("not implemented yet")

    monkeypatch.setattr("arden.__main__.run_simulation", spy)
    code = main(["run", "--config", str(config_path)])
    assert code == 0
    settings = captured["settings"]
    rng = captured["rng"]
    assert isinstance(settings, Settings)
    assert isinstance(rng, random.Random)
    assert settings.world.seed == 99
    assert rng.getstate() == make_rng(settings).getstate()
    assert capsys.readouterr().out.strip() == "not implemented yet"


def test_run_simulation_placeholder_prints_message(capsys: pytest.CaptureFixture[str]) -> None:
    run_simulation(Settings(), make_rng(Settings()))
    assert capsys.readouterr().out.strip() == "not implemented yet"


def test_llm_smoke_with_mock_backend_prints_intent_and_latency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "arden.toml"
    config_path.write_text('[llm]\nbackend = "mock"\n', encoding="utf-8")
    code = main(["llm-smoke", "--config", str(config_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "backend: mock" in out
    assert "latency_ms:" in out
    assert "intent:" in out
