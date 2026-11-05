"""Mock LLM backend: canned JSON, no network and no model."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from arden.llm.base import parse_json_output, require_role

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REFLECT_FIXTURES_DIR = FIXTURES_DIR / "reflect"
REFLECT_FIXTURE_PATH = REFLECT_FIXTURES_DIR / "reflect.json"
LLM_SEED_OFFSET: int = 1


def derive_llm_seed(world_seed: int) -> int:
    """Derive a distinct deterministic seed for ``MockLLM`` from ``world_seed``."""
    return world_seed + LLM_SEED_OFFSET


def load_reflect_fixture() -> str:
    """Return the raw JSON text of the built-in canned ``reflect`` fixture."""
    return REFLECT_FIXTURE_PATH.read_text(encoding="utf-8")


def _is_intent_schema(schema: dict) -> bool:
    """Return True if ``schema`` is the ``Intent`` JSON Schema (requiring ``goal`` and ``steps``)."""
    if not isinstance(schema, dict):
        return False
    props = schema.get("properties")
    return isinstance(props, dict) and "goal" in props and "steps" in props


class MockLLM:
    """Deterministic backend for tests and for ``backend = "mock"``.

    With no scripted responses, each ``decide`` (or ``name``) call draws one
    intent fixture with ``random.Random(seed)``, while ``reflect`` calls draw
    from the canned ``reflect`` fixture(s). The same seed replays the same
    sequence. The draw does not use the simulation RNG from ``make_rng``.

    ``script`` is a queue of raw response strings. Calls take from the front
    of that queue first, including strings that are not valid JSON. The seed
    is not advanced for a scripted response. When the queue is empty, the
    next call draws a fixture.

    ``system``, ``user``, and ``timeout_s`` are accepted so the signature
    matches ``LLMBackend``. They do not change which text is returned, and
    the mock returns immediately.
    """

    def __init__(
        self,
        seed: int = LLM_SEED_OFFSET,
        script: Sequence[str] | None = None,
        fixtures_dir: Path | None = None,
    ) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._queue: deque[str] = deque(script or ())
        directory = FIXTURES_DIR if fixtures_dir is None else Path(fixtures_dir)
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No LLM fixtures in {directory}")
        self._fixtures = [path.read_text(encoding="utf-8") for path in paths]

        reflect_dir = directory / "reflect"
        reflect_paths = sorted(reflect_dir.glob("*.json")) if reflect_dir.is_dir() else []
        if not reflect_paths and REFLECT_FIXTURES_DIR.is_dir():
            reflect_paths = sorted(REFLECT_FIXTURES_DIR.glob("*.json"))
        self._reflect_fixtures = [path.read_text(encoding="utf-8") for path in reflect_paths]

    @property
    def reflect_fixtures(self) -> list[str]:
        """Raw JSON strings of the canned ``reflect`` fixtures."""
        return list(self._reflect_fixtures)

    def enqueue(self, raw: str) -> None:
        """Append one raw response to the script queue."""
        self._queue.append(raw)

    def complete_json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict,
        timeout_s: float,
    ) -> dict:
        require_role(role)
        del system, user, timeout_s
        if self._queue:
            raw = self._queue.popleft()
        elif role == "reflect" and self._reflect_fixtures and not _is_intent_schema(schema):
            raw = self._rng.choice(self._reflect_fixtures)
        else:
            raw = self._rng.choice(self._fixtures)
        return parse_json_output(raw, schema)
