"""Pytest fixtures and cleanup hooks."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _cleanup_test_runs_dir() -> Iterator[None]:
    """Remove any ``data/runs/<run_id>`` directories created during a test."""
    runs_dir = Path("data/runs")
    data_dir = Path("data")
    data_existed = data_dir.exists()
    runs_existed = runs_dir.exists()
    existing_runs = set(runs_dir.iterdir()) if runs_existed else set()

    yield

    if runs_dir.exists():
        for entry in list(runs_dir.iterdir()):
            if entry not in existing_runs:
                shutil.rmtree(entry, ignore_errors=True)
        if not runs_existed and not any(runs_dir.iterdir()):
            shutil.rmtree(runs_dir, ignore_errors=True)
    if not data_existed and data_dir.exists() and not any(data_dir.iterdir()):
        shutil.rmtree(data_dir, ignore_errors=True)
