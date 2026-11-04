"""CLI entry point: version, help, headless run, and llm-smoke commands."""

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


def test_run_cli_headless_500_ticks_with_mock_llm(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arden", "run", "--ticks", "500", "--llm", "mock"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    lines = [line.strip() for line in completed.stdout.strip().splitlines() if line.strip()]
    assert len(lines) >= 2
    assert lines[0].startswith("tick=0 reasons=[start]")
    for line in lines:
        assert "tick=" in line
        assert "reasons=[" in line
        assert "goal=" in line
        assert "thought=" in line
    assert any("heartbeat" in line for line in lines)


def test_run_passes_seeded_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "arden.toml"
    config_path.write_text("[world]\nseed = 99\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def spy(settings: Settings, rng: random.Random) -> None:
        captured["settings"] = settings
        captured["rng"] = rng
        print("spy called")

    monkeypatch.setattr("arden.__main__.run_simulation", spy)
    code = main(["run", "--config", str(config_path)])
    assert code == 0
    settings = captured["settings"]
    rng = captured["rng"]
    assert isinstance(settings, Settings)
    assert isinstance(rng, random.Random)
    assert settings.world.seed == 99
    assert rng.getstate() == make_rng(settings).getstate()
    assert capsys.readouterr().out.strip() == "spy called"


def test_run_simulation_prints_wake_summaries(capsys: pytest.CaptureFixture[str]) -> None:
    settings = Settings()
    loop = run_simulation(settings, make_rng(settings), ticks=20)
    out = capsys.readouterr().out.strip()
    lines = [line for line in out.splitlines() if line.strip()]
    assert loop.world.tick == 20
    assert len(lines) == len(loop.wake_summaries)
    assert lines[0].startswith("tick=0 reasons=[start]")


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
