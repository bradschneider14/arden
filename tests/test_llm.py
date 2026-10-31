"""Mock LLM backend, OpenAI-compatible/Ollama backend, prompt builder, and output validation."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any, Self

import pytest
from pydantic import ValidationError

from arden.actions.verbs import ActionResult
from arden.config import Settings, load_config
from arden.llm import (
    VERBS,
    Intent,
    LLMBackend,
    LLMOutputError,
    MockLLM,
    OpenAICompatLLM,
    Step,
    build_decide_prompt,
    build_name_prompt,
    build_reflect_prompt,
    estimate_tokens,
    make_backend,
)
from arden.llm.mock import FIXTURES_DIR
from arden.perception.feed import perceive
from arden.world.world import World

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

    ollama_settings = Settings.model_validate(
        {
            "llm": {
                "backend": "ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen2.5:7b-instruct",
                "temperature": 0.1,
                "timeout_s": 12.0,
                "max_retries": 3,
            }
        }
    )
    ollama_backend = make_backend(ollama_settings)
    assert isinstance(ollama_backend, OpenAICompatLLM)
    assert isinstance(ollama_backend, LLMBackend)
    assert ollama_backend.base_url == "http://127.0.0.1:11434/v1"
    assert ollama_backend.model == "qwen2.5:7b-instruct"
    assert ollama_backend.temperature == 0.1
    assert ollama_backend.timeout_s == 12.0
    assert ollama_backend.max_retries == 3

    with pytest.raises(ValueError, match="Unknown LLM backend"):
        make_backend(Settings.model_validate({"llm": {"backend": "unsupported_backend"}}))


def test_unknown_role_does_not_consume_the_queue() -> None:
    backend = MockLLM(script=['{"goal": "kept"}'])
    with pytest.raises(ValueError, match="role"):
        backend.complete_json("plan", "system", "user", _schema(), 1.0)
    with pytest.raises(LLMOutputError) as error:
        backend.complete_json("decide", "system", "user", _schema(), 1.0)
    assert error.value.raw == '{"goal": "kept"}'


def test_openai_compat_valid_response() -> None:
    valid_raw = (FIXTURES_DIR / "push.json").read_text(encoding="utf-8")
    captured_calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        captured_calls.append((url, payload, timeout_s))
        return {"choices": [{"message": {"role": "assistant", "content": valid_raw}}]}

    backend = OpenAICompatLLM(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b-instruct",
        temperature=0.3,
        timeout_s=15.0,
        max_retries=2,
        transport=fake_transport,
    )
    result = backend.complete_json("decide", "sys prompt", "user prompt", _schema(), 10.0)
    assert result == json.loads(valid_raw)
    assert len(captured_calls) == 1
    url, payload, timeout_s = captured_calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert timeout_s == 10.0
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["temperature"] == 0.3
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == _schema()
    assert payload["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_openai_compat_ollama_format_mode() -> None:
    valid_raw = (FIXTURES_DIR / "explore.json").read_text(encoding="utf-8")
    captured_calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        captured_calls.append((url, payload, timeout_s))
        return {"message": {"role": "assistant", "content": valid_raw}}

    backend = OpenAICompatLLM(
        base_url="http://localhost:11434/api",
        model="qwen2.5:7b-instruct",
        temperature=0.25,
        transport=fake_transport,
    )
    result = backend.complete_json("decide", "sys", "usr", _schema(), 5.0)
    assert result == json.loads(valid_raw)
    assert len(captured_calls) == 1
    url, payload, _ = captured_calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["format"] == _schema()
    assert payload["options"] == {"temperature": 0.25}


def test_openai_compat_invalid_then_valid_retries_with_repair_message() -> None:
    valid_raw = (FIXTURES_DIR / "speak.json").read_text(encoding="utf-8")
    responses = [
        {"choices": [{"message": {"content": "{broken json"}}]},
        {"choices": [{"message": {"content": '{"goal": "bad", "steps": [{"verb": "fly"}]}'}}]},
        {"choices": [{"message": {"content": valid_raw}}]},
    ]
    payload_history: list[dict[str, Any]] = []

    def fake_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        del url, timeout_s
        payload_history.append(json.loads(json.dumps(payload)))
        return responses.pop(0)

    backend = OpenAICompatLLM(max_retries=2, transport=fake_transport)
    result = backend.complete_json("decide", "sys", "usr", _schema(), 5.0)
    assert result == json.loads(valid_raw)
    assert len(payload_history) == 3
    # Second call includes the first broken assistant output and a repair message
    second_msgs = payload_history[1]["messages"]
    assert len(second_msgs) == 4
    assert second_msgs[2] == {"role": "assistant", "content": "{broken json"}
    assert "invalid" in second_msgs[3]["content"].lower()
    # Third call includes the schema validation failure repair message
    third_msgs = payload_history[2]["messages"]
    assert len(third_msgs) == 6
    assert "schema" in third_msgs[5]["content"].lower()


def test_openai_compat_always_invalid_raises_llm_output_error() -> None:
    bad_raw = '{"goal": "only goal"}'
    calls = 0

    def fake_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        nonlocal calls
        del url, payload, timeout_s
        calls += 1
        return {"choices": [{"message": {"content": bad_raw}}]}

    backend = OpenAICompatLLM(max_retries=2, transport=fake_transport)
    with pytest.raises(LLMOutputError) as exc_info:
        backend.complete_json("decide", "sys", "usr", _schema(), 5.0)
    assert calls == 3  # 1 initial + 2 retries
    assert exc_info.value.raw == bad_raw


def test_openai_compat_urllib_transport_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_raw = (FIXTURES_DIR / "explore.json").read_text(encoding="utf-8")
    response_bytes = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": valid_raw}}]}
    ).encode("utf-8")

    class _FakeHTTPResponse:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return response_bytes

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> _FakeHTTPResponse:
        assert request.full_url == "http://127.0.0.1:11434/v1/chat/completions"
        assert timeout == 7.5
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatLLM()
    parsed = backend.complete_json("decide", "sys", "usr", _schema(), 7.5)
    assert parsed == json.loads(valid_raw)


def test_prompts_build_decide_reflect_name_and_avoid_semantic_or_evaluative_labels() -> None:
    world = World(seed=1)
    world.body.energy = 0.25
    world.body.comfort = 0.15
    obj_id = world.spawn(
        {
            "shape": "sphere",
            "height": 0.2,
            "volume": 0.004,
            "weight": 0.3,
            "rigidity": 0.5,
            "color": "red",
        },
        (0.0, 1.0),
    )
    percept = perceive(world)
    prev_res = ActionResult(ok=False, reason="out_of_reach", events=[], data={"target": obj_id})

    decide_p = build_decide_prompt(
        percept=percept,
        self_model="I push round objects to see how far they roll.",
        memories=["tick 2: pushed object_1, it rolled north"],
        body_state=percept.body,
        previous_intent_result=prev_res,
    )
    reflect_p = build_reflect_prompt(
        percept=percept,
        self_model="I push round objects.",
        memories=["tick 2: pushed object_1"],
        body_state=percept.body,
        previous_intent_result=prev_res,
    )
    name_p = build_name_prompt(
        percept=percept,
        self_model="I push round objects.",
        memories=[],
        body_state=percept.body,
        previous_intent_result=prev_res,
        target=obj_id,
    )

    for built, expected_role in ((decide_p, "decide"), (reflect_p, "reflect"), (name_p, "name")):
        assert built.role == expected_role
        sys_text, usr_text = built
        assert sys_text == built.system
        assert usr_text == built.user
        assert f"role: {expected_role}" in usr_text
        assert "energy=0.25" in usr_text
        assert "comfort=0.15" in usr_text
        assert obj_id.id in usr_text
        assert "out_of_reach" in usr_text

        combined = f"{sys_text}\n{usr_text}".lower()
        for forbidden in (
            "good",
            "bad",
            "healthy",
            "unhealthy",
            "hungry",
            "starving",
            "tired",
            "exhausted",
            "comfortable",
            "uncomfortable",
            "ball",
            "tree",
            "mailbox",
            "berry",
        ):
            assert forbidden not in combined


def test_prompt_token_budget_truncates_memories_first() -> None:
    memories = [
        f"memory_{i}: observed object_{i} move across the ground for several meters"
        for i in range(1, 25)
    ]
    full_prompt = build_decide_prompt(
        percept="tick: 5\nbody: pos=(0,0) level=ground energy=0.5 comfort=0.5 holding=none\nobjects: none\naffordances: none\nevents: none",
        self_model="I test how objects move when pushed.",
        memories=memories,
        body_state="pos=(0,0) level=ground energy=0.5 comfort=0.5 holding=none",
        previous_intent_result="ok=true",
        max_tokens=10_000,
    )
    full_tokens = estimate_tokens(full_prompt.system) + estimate_tokens(full_prompt.user)
    budget = full_tokens - 60

    budgeted_prompt = build_decide_prompt(
        percept="tick: 5\nbody: pos=(0,0) level=ground energy=0.5 comfort=0.5 holding=none\nobjects: none\naffordances: none\nevents: none",
        self_model="I test how objects move when pushed.",
        memories=memories,
        body_state="pos=(0,0) level=ground energy=0.5 comfort=0.5 holding=none",
        previous_intent_result="ok=true",
        max_tokens=budget,
    )
    budgeted_tokens = estimate_tokens(budgeted_prompt.system) + estimate_tokens(budgeted_prompt.user)
    assert budgeted_tokens <= budget
    # Core sections remain intact while later memories were truncated first
    assert "self_model:\nI test how objects move when pushed." in budgeted_prompt.user
    assert "percept_delta:\ntick: 5" in budgeted_prompt.user
    assert "previous_intent_result:\nok=true" in budgeted_prompt.user
    assert "memory_1:" in budgeted_prompt.user
    assert "memory_24:" not in budgeted_prompt.user

