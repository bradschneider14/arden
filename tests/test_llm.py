"""Mock LLM backend, intent schema, and output validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arden.config import Settings, load_config
from arden.llm import (
    VERBS,
    Intent,
    LLMBackend,
    LLMOutputError,
    MockLLM,
    Step,
    make_backend,
)
from arden.llm.mock import FIXTURES_DIR

_EXAMPLE = Path(__file__).resolve().parents[1] / "arden.example.toml"


def _schema() -> dict:
    return Intent.model_json_schema()


def _calls(backend: MockLLM, count: int, role: str = "decide") -> list[dict]:
    return [backend.complete_json(role, "system", "user", _schema(), 1.0) for _ in range(count)]


def test_intent_schema_lists_design_verbs() -> None:
    schema = _schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["goal", "steps", "wake_if", "expect", "thought", "confidence"]
    assert schema["$defs"]["Step"]["properties"]["verb"]["enum"] == list(VERBS)
    Intent.model_validate(
        {
            "goal": "wait",
            "steps": [{"verb": "look", "target": None, "params": {}}],
            "wake_if": [],
            "expect": "nothing changes",
            "thought": "Staying put.",
            "confidence": 0,
        }
    )
    with pytest.raises(ValidationError):
        Step.model_validate({"verb": "fly"})
    with pytest.raises(ValidationError):
        Intent.model_validate(
            {
                "goal": "wait",
                "steps": [],
                "wake_if": [],
                "expect": "nothing",
                "thought": "No.",
                "confidence": 1.1,
            }
        )


def test_valid_fixture_parses() -> None:
    paths = sorted(FIXTURES_DIR.glob("*.json"))
    assert paths
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        backend = MockLLM(script=[raw])
        parsed = backend.complete_json("decide", "system", "user", _schema(), 1.0)
        assert parsed == json.loads(raw)
        assert Intent.model_validate(parsed).model_dump() == parsed


def test_scripted_queue_order() -> None:
    raws = [
        (FIXTURES_DIR / "push.json").read_text(encoding="utf-8"),
        (FIXTURES_DIR / "explore.json").read_text(encoding="utf-8"),
        (FIXTURES_DIR / "speak.json").read_text(encoding="utf-8"),
    ]
    backend = MockLLM(seed=99, script=list(raws))
    assert _calls(backend, 3) == [json.loads(raw) for raw in raws]


def test_invalid_json_raises_with_raw_text() -> None:
    malformed = "{not json"
    schema_invalid = '{"goal": "look", "steps": [{"verb": "fly"}]}'
    backend = MockLLM(script=[malformed, schema_invalid, "[1]"])

    with pytest.raises(LLMOutputError) as malformed_error:
        backend.complete_json("reflect", "system", "user", _schema(), 1.0)
    assert malformed_error.value.raw == malformed

    with pytest.raises(LLMOutputError) as schema_error:
        backend.complete_json("name", "system", "user", _schema(), 1.0)
    assert schema_error.value.raw == schema_invalid

    with pytest.raises(LLMOutputError) as array_error:
        backend.complete_json("decide", "system", "user", _schema(), 1.0)
    assert array_error.value.raw == "[1]"


def test_same_seed_gives_the_same_sequence() -> None:
    first = _calls(MockLLM(seed=7), 12)
    second = _calls(MockLLM(seed=7), 12)
    other = _calls(MockLLM(seed=8), 12)
    assert first == second
    assert first != other
    assert len({item["goal"] for item in first}) > 1


def test_make_backend_uses_config_and_rejects_unknown_names() -> None:
    settings = load_config(_EXAMPLE)
    assert settings.llm.backend == "mock"
    backend = make_backend(settings)
    assert isinstance(backend, MockLLM)
    assert isinstance(backend, LLMBackend)
    again = make_backend(Settings())
    assert _calls(backend, 4, role="name") == _calls(again, 4, role="name")

    with pytest.raises(ValueError, match="ollama"):
        make_backend(Settings.model_validate({"llm": {"backend": "ollama"}}))


def test_unknown_role_does_not_consume_the_queue() -> None:
    backend = MockLLM(script=['{"goal": "kept"}'])
    with pytest.raises(ValueError, match="role"):
        backend.complete_json("plan", "system", "user", _schema(), 1.0)
    with pytest.raises(LLMOutputError) as error:
        backend.complete_json("decide", "system", "user", _schema(), 1.0)
    assert error.value.raw == '{"goal": "kept"}'
