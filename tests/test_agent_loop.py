"""Tests for the headless agent loop and HeartbeatGate (M4 / T12)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from arden.agent import (
    AgentLoop,
    Gate,
    GateContext,
    HeartbeatGate,
    IntentSetEvent,
    LLMErrorEvent,
    WakeDecision,
    WakeEvent,
    build_test_world,
    format_wake_summary,
)
from arden.config import Settings, make_rng
from arden.controller import IDLE, RUNNING, SUCCEEDED, Controller, IntentEndedEvent
from arden.llm import Intent, LLMOutputError, MockLLM, Step, derive_llm_seed
from arden.world import World, WorldClock, WorldEvent


def test_heartbeat_gate_wakes_on_start_intent_ended_and_every_n_ticks() -> None:
    gate = HeartbeatGate(interval=10)
    assert isinstance(gate, Gate)

    # Tick 0: wakes at start
    d0 = gate.should_wake(GateContext(tick=0, is_start=True))
    assert d0 == WakeDecision(wake=True, reasons=["start"])
    assert bool(d0) is True

    # Ticks 1..3: no wake
    for t in (1, 2, 3):
        dt = gate.should_wake(GateContext(tick=t, is_start=False, intent_ended=False))
        assert dt == WakeDecision(wake=False, reasons=[])
        assert bool(dt) is False

    # Tick 4: intent_ended wakes the gate
    d4 = gate.should_wake(GateContext(tick=4, is_start=False, intent_ended=True))
    assert d4 == WakeDecision(wake=True, reasons=["intent_ended"])

    # Ticks 5..9: no wake
    for t in range(5, 10):
        dt = gate.should_wake(GateContext(tick=t, is_start=False, intent_ended=False))
        assert dt.wake is False
        assert dt.reasons == []

    # Tick 10: heartbeat fires
    d10 = gate.should_wake(GateContext(tick=10, is_start=False, intent_ended=False))
    assert d10 == WakeDecision(wake=True, reasons=["heartbeat"])

    # Tick 20 with simultaneous intent_ended and heartbeat reports both reasons
    for t in range(11, 20):
        assert not gate.should_wake(GateContext(tick=t)).wake
    d20 = gate.should_wake(GateContext(tick=20, intent_ended=True))
    assert d20.wake is True
    assert d20.reasons == ["intent_ended", "heartbeat"]


def test_heartbeat_gate_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HeartbeatGate(0)
    with pytest.raises(ValueError, match="positive integer"):
        HeartbeatGate(-5)


def test_each_tick_orders_perceive_gate_pause_decide_set_intent_controller_and_world_step() -> None:
    """Verify the exact per-tick execution sequence and that the world pauses while the LLM thinks."""
    settings = Settings()
    world = build_test_world(make_rng(settings), settings)
    clock = WorldClock(world, tick_rate=10.0)
    controller = Controller(world=world)
    trace: list[str] = []

    class TracingGate:
        def should_wake(self, context: GateContext) -> WakeDecision:
            assert context.percept is not None
            assert context.percept.tick == world.tick
            trace.append(f"gate:tick={context.tick}")
            return WakeDecision(wake=(context.tick == 0), reasons=["start"] if context.tick == 0 else [])

    class InspectingLLM:
        def complete_json(
            self,
            role: str,
            system: str,
            user: str,
            schema: dict[str, Any],
            timeout_s: float,
        ) -> dict[str, Any]:
            assert role == "decide"
            # World and clock must be paused while the LLM thinks
            assert clock.paused is True
            assert world.paused is True
            assert loop.paused is True
            trace.append(f"llm_decide:tick={world.tick}:paused={clock.paused}")
            return {
                "goal": "speak hello",
                "steps": [{"verb": "speak", "target": None, "params": {"text": "hi"}}],
                "wake_if": [],
                "expect": "spoken",
                "thought": "test greeting",
                "confidence": 0.9,
            }

    loop = AgentLoop(
        world=world,
        llm=InspectingLLM(),
        gate=TracingGate(),
        controller=controller,
        clock=clock,
        settings=settings,
    )

    loop.subscribe("wake", lambda ev: trace.append(f"event:wake:tick={ev.tick}"))
    loop.subscribe("intent_set", lambda ev: trace.append(f"event:intent_set:tick={ev.tick}"))
    loop.subscribe("spoke", lambda ev: trace.append(f"event:spoke:tick={ev.tick}"))
    loop.subscribe("intent_ended", lambda ev: trace.append(f"event:intent_ended:tick={ev.tick}"))
    loop.subscribe("tick", lambda ev: trace.append(f"event:world_tick:tick={ev.tick}"))

    result = loop.tick()
    assert result.woke is True
    assert clock.paused is False
    assert world.paused is False
    assert loop.paused is False
    assert world.tick == 1

    assert trace == [
        "gate:tick=0",
        "event:wake:tick=0",
        "llm_decide:tick=0:paused=True",
        "event:intent_set:tick=0",
        "event:spoke:tick=0",
        "event:intent_ended:tick=0",
        "event:world_tick:tick=1",
    ]


def test_llm_failure_while_idle_or_running_never_crashes_and_emits_llm_error() -> None:
    """LLMOutputError and TimeoutError are survived: agent idles or keeps its current intent."""
    settings = Settings()
    world = build_test_world(make_rng(settings), settings)

    # 1. Start with an invalid JSON response -> agent has no intent, so it idles and emits llm_error
    long_move_intent = {
        "goal": "push object_7 toward the tree",
        "steps": [
            {"verb": "move", "target": None, "params": {"direction": "toward object_7", "speed": 0.5}},
            {"verb": "push", "target": "object_7", "params": {"force": 0.5}},
        ],
        "wake_if": [],
        "expect": "object_7 moves",
        "thought": "Moving to object_7.",
        "confidence": 0.8,
    }
    mock_llm = MockLLM(
        seed=derive_llm_seed(settings.world.seed),
        script=[
            "not valid json at all",
            json.dumps(long_move_intent),
            '{"goal": "missing required fields"}',
        ],
    )

    # Heartbeat every 2 ticks so:
    # - tick 0: wakes on start -> fails ("not valid json at all") -> idles
    # - tick 1: continues idling
    # - tick 2: wakes on heartbeat -> succeeds (sets long_move_intent, begins moving)
    # - tick 3: continues moving (running)
    # - tick 4: wakes on heartbeat while still running -> fails (schema mismatch) -> keeps long_move_intent!
    gate = HeartbeatGate(interval=2)
    loop = AgentLoop(world=world, llm=mock_llm, gate=gate, settings=settings)

    wake_events: list[WakeEvent] = []
    intent_events: list[IntentSetEvent] = []
    error_events: list[LLMErrorEvent] = []
    loop.subscribe("wake", wake_events.append)
    loop.subscribe("intent_set", intent_events.append)
    loop.subscribe("llm_error", error_events.append)

    # Tick 0: start wake -> LLMOutputError -> controller idles
    r0 = loop.tick()
    assert r0.woke is True
    assert r0.reasons == ["start"]
    assert isinstance(r0.llm_error, LLMOutputError)
    assert r0.controller_status == IDLE
    assert loop.intent is None
    assert len(error_events) == 1
    assert error_events[0].type == "llm_error"
    assert error_events[0].raw == "not valid json at all"
    assert error_events[0].kept_intent is None

    # Tick 1: no wake -> continues idling
    r1 = loop.tick()
    assert r1.woke is False
    assert r1.controller_status == IDLE

    # Tick 2: heartbeat wake -> valid intent -> controller starts moving (running)
    r2 = loop.tick()
    assert r2.woke is True
    assert r2.reasons == ["heartbeat"]
    assert r2.llm_error is None
    assert len(intent_events) == 1
    assert intent_events[0].goal == "push object_7 toward the tree"
    assert r2.controller_status == RUNNING
    assert loop.intent is not None
    assert loop.intent.goal == "push object_7 toward the tree"

    # Tick 3: no wake -> still running the multi-step move+push intent
    r3 = loop.tick()
    assert r3.woke is False
    assert r3.controller_status == RUNNING

    # Tick 4: heartbeat wake while running -> invalid schema -> keeps active intent and keeps running!
    r4 = loop.tick()
    assert r4.woke is True
    assert "heartbeat" in r4.reasons
    assert isinstance(r4.llm_error, LLMOutputError)
    assert len(error_events) == 2
    assert error_events[1].kept_intent is not None
    assert error_events[1].kept_intent.goal == "push object_7 toward the tree"
    assert loop.intent is not None
    assert loop.intent.goal == "push object_7 toward the tree"

    # 2. Also verify TimeoutError is caught and emitted as llm_error without crashing
    class TimeoutLLM:
        def complete_json(
            self,
            role: str,
            system: str,
            user: str,
            schema: dict[str, Any],
            timeout_s: float,
        ) -> dict[str, Any]:
            raise TimeoutError("simulated LLM request timeout")

    loop.llm = TimeoutLLM()
    r5 = loop.tick()
    r6 = loop.tick()  # tick 6: heartbeat wake -> TimeoutError
    assert r6.woke is True
    assert isinstance(r6.llm_error, TimeoutError)
    assert len(error_events) == 3
    assert error_events[2].error_type == "TimeoutError"
    assert "timeout" in error_events[2].error


def test_end_to_end_500_ticks_with_mock_llm_executes_intents_wakes_on_heartbeat_and_survives_invalid_output() -> None:
    """End-to-end test: 500 ticks complete, >=1 heartbeat wake, mock intents execute, invalid output survived."""
    settings = Settings()
    rng = make_rng(settings)
    world = build_test_world(rng=rng, settings=settings)
    initial_obj7_pos = world.objects["object_7"].position.model_copy()

    # Inject invalid LLM outputs into the script queue followed by normal mock fixture draws
    mock_llm = MockLLM(
        seed=derive_llm_seed(settings.world.seed),
        script=[
            "{malformed json",
            '{"goal": "bad verb", "steps": [{"verb": "fly"}], "wake_if": [], "expect": "x", "thought": "y", "confidence": 0.5}',
        ],
    )

    wake_events: list[WakeEvent] = []
    intent_set_events: list[IntentSetEvent] = []
    llm_error_events: list[LLMErrorEvent] = []
    intent_ended_events: list[IntentEndedEvent] = []
    spoke_events: list[WorldEvent] = []

    world.subscribe("wake", wake_events.append)
    world.subscribe("intent_set", intent_set_events.append)
    world.subscribe("llm_error", llm_error_events.append)
    world.subscribe("intent_ended", intent_ended_events.append)
    world.subscribe("spoke", spoke_events.append)

    loop = AgentLoop(
        world=world,
        llm=mock_llm,
        gate=HeartbeatGate(interval=50),
        settings=settings,
        rng=rng,
    )

    results = loop.run(500)

    # 1. All 500 ticks completed
    assert len(results) == 500
    assert loop.ticks_run == 500
    assert world.tick == 500

    # 2. Injected invalid LLM responses were survived and emitted as llm_error events
    assert len(llm_error_events) == 2
    assert llm_error_events[0].tick == 0
    assert llm_error_events[0].raw == "{malformed json"
    assert llm_error_events[1].tick == 50

    # 3. At least one heartbeat wake happened (including pure heartbeat wakes while idling/running)
    heartbeat_wakes = [e for e in wake_events if "heartbeat" in e.reasons]
    assert len(heartbeat_wakes) >= 1
    assert any(e.reasons == ["heartbeat"] for e in wake_events)

    # 4. Controller executed the mock intents (look, speak, move+push object_7)
    assert len(intent_set_events) >= 10
    assert len(intent_ended_events) >= 10
    assert any(e.succeeded for e in intent_ended_events)
    assert len(spoke_events) >= 1
    assert world.objects["object_7"].position != initial_obj7_pos

    # 5. Every wake produced a one-line summary with tick, reasons, goal, and thought
    assert len(loop.wake_summaries) == len(wake_events)
    for summary in loop.wake_summaries:
        assert "\n" not in summary
        assert "tick=" in summary
        assert "reasons=[" in summary
        assert "goal=" in summary
        assert "thought=" in summary
