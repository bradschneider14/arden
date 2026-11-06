"""Table-driven and integration tests for RuleGate, wake_if conditions, and gate_decision events (T15)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from arden.agent import (
    AgentLoop,
    Gate,
    GateConfig,
    GateContext,
    GateDecisionEvent,
    HeartbeatGate,
    MailArrivedEvent,
    RuleGate,
    WakeCondition,
    WakeDecision,
    build_test_world,
    parse_wake_conditions,
    parse_wake_if,
)
from arden.config import Settings, load_config, make_rng
from arden.llm import Intent, MockLLM, Step
from arden.perception import BodyPercept, PerceivedEvent, PerceivedObject, Percept, perceive
from arden.world import (
    ObjectChangedEvent,
    ObjectDespawnedEvent,
    ObjectMovedEvent,
    ObjectSpawnedEvent,
    Position,
    World,
)


def _make_perceived_object(
    obj_id: str = "object_1",
    *,
    x: float = 2.0,
    y: float = 0.0,
    shape: str = "sphere",
    height: float = 0.3,
    volume: float = 0.01,
    weight: float = 0.5,
    rigidity: float = 0.6,
    hollow: bool = False,
    color: str | None = "red",
    state: dict | None = None,
    affordances: list[str] | None = None,
) -> PerceivedObject:
    st = dict(state or {})
    props = {
        "id": obj_id,
        "position": {"x": x, "y": y},
        "level": "ground",
        "shape": shape,
        "height": height,
        "volume": volume,
        "weight": weight,
        "hollow": hollow,
        "state": st,
    }
    if color is not None:
        props["color"] = color
    return PerceivedObject(
        id=obj_id,
        distance=round((x * x + y * y) ** 0.5, 4),
        direction="E",
        position=Position(x=x, y=y),
        level="ground",
        shape=shape,  # type: ignore[arg-type]
        height=height,
        volume=volume,
        weight=weight,
        rigidity=rigidity,
        hollow=hollow,
        color=color,
        state=st,
        properties=props,
        affordances=list(affordances or []),
    )


def _make_percept(
    *,
    tick: int = 1,
    energy: float = 1.0,
    comfort: float = 1.0,
    objects: list[PerceivedObject] | None = None,
    affordances: dict[str, list[str]] | None = None,
    events: list[PerceivedEvent] | None = None,
    is_delta: bool = False,
    appeared: list[PerceivedObject] | None = None,
    disappeared: list[str] | None = None,
    moved: list[PerceivedObject] | None = None,
    changed: list[PerceivedObject] | None = None,
) -> Percept:
    objs = list(objects) if objects is not None else [_make_perceived_object("object_1")]
    affs = dict(affordances) if affordances is not None else {"object_1": ["grab", "push", "strike"]}
    return Percept(
        tick=tick,
        light=1.0,
        body=BodyPercept(
            position=Position(x=0.0, y=0.0),
            level="ground",
            energy=energy,
            comfort=comfort,
            holding=None,
        ),
        objects=objs,
        events=list(events or []),
        affordances=affs,
        is_delta=is_delta,
        appeared=list(appeared or []),
        disappeared=list(disappeared or []),
        moved=list(moved or []),
        changed=list(changed or []),
    )


def _make_seeded_gate(**kwargs: object) -> RuleGate:
    """Create a RuleGate that has already observed the baseline `object_1` percept at tick 0."""
    baseline = _make_percept(tick=0)
    gate = RuleGate(baseline=baseline, **kwargs)  # type: ignore[arg-type]
    return gate


def _build_trigger_cases() -> list[tuple[str, Callable[[], tuple[RuleGate, GateDecisionEvent, WakeDecision]], bool, list[str]]]:
    """Build table-driven test cases: one case per trigger, plus continue and unparseable wake_if."""

    def case_start() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = RuleGate()
        decision = gate.should_wake(GateContext(tick=0, is_start=True, percept=_make_percept(tick=0)))
        return gate, gate.decisions[-1], decision

    def case_surprise() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(surprise_threshold=0.25)
        decision = gate.should_wake(GateContext(tick=1, percept=_make_percept(tick=1), surprise=0.40))
        return gate, gate.decisions[-1], decision

    def case_novelty_never_seen_object() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        # object_2 has the exact same properties and affordances as object_1, so only its ID is novel
        obj1 = _make_perceived_object("object_1", x=2.0, y=0.0)
        obj2 = _make_perceived_object("object_2", x=3.0, y=0.0)
        percept = _make_percept(tick=1, objects=[obj1, obj2])
        decision = gate.should_wake(GateContext(tick=1, percept=percept))
        return gate, gate.decisions[-1], decision

    def case_novelty_never_seen_property_combination() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        # Same object_1 ID and same affordances, but a never-seen property combination (cube + blue + hollow)
        modified_obj1 = _make_perceived_object(
            "object_1",
            x=2.0,
            y=0.0,
            shape="cube",
            color="blue",
            hollow=True,
        )
        percept = _make_percept(tick=1, objects=[modified_obj1])
        decision = gate.should_wake(GateContext(tick=1, percept=percept))
        return gate, gate.decisions[-1], decision

    def case_novelty_never_seen_affordance() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        # Same object_1 ID and same properties, but a never-seen opaque affordance `action_1` appears in reach
        obj1 = _make_perceived_object("object_1", x=1.0, y=0.0, affordances=["grab", "push", "strike", "action_1"])
        percept = _make_percept(
            tick=1,
            objects=[obj1],
            affordances={"object_1": ["grab", "push", "strike", "action_1"]},
        )
        decision = gate.should_wake(GateContext(tick=1, percept=percept))
        return gate, gate.decisions[-1], decision

    def case_intent_ended() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        decision = gate.should_wake(
            GateContext(tick=1, percept=_make_percept(tick=1), intent_ended=True)
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_object_moves() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        moved_obj1 = _make_perceived_object("object_1", x=2.8, y=0.0)
        percept = _make_percept(tick=1, objects=[moved_obj1], moved=[moved_obj1])
        decision = gate.should_wake(
            GateContext(tick=1, percept=percept, wake_if=["object_1 moves"])
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_object_stops_moving() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(moving_objects=["object_1"])
        stationary_obj1 = _make_perceived_object("object_1", x=2.0, y=0.0)
        percept = _make_percept(tick=2, objects=[stationary_obj1])
        decision = gate.should_wake(
            GateContext(tick=2, percept=percept, wake_if=["object_1 stops moving"])
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_object_disappears() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        percept = _make_percept(tick=1, objects=[], affordances={}, disappeared=["object_1"])
        decision = gate.should_wake(
            GateContext(tick=1, percept=percept, wake_if=["object_1 disappears"])
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_new_object_in_range() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        # Pre-seed object_2 into seen_objects/seen_property_combos so novelty doesn't fire, isolating wake_if
        obj1 = _make_perceived_object("object_1", x=2.0, y=0.0)
        obj2 = _make_perceived_object("object_2", x=3.0, y=0.0)
        gate = _make_seeded_gate(seen_objects=["object_1", "object_2"])
        gate._last_visible_objects = {"object_1"}
        percept = _make_percept(tick=1, objects=[obj1, obj2], appeared=[obj2])
        decision = gate.should_wake(
            GateContext(tick=1, percept=percept, wake_if=["new object in range"])
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_energy_below() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(low_energy_threshold=0.25)
        # Energy 0.55 is well above the gate drive threshold (0.25) but below the LLM's `energy below 0.6`
        percept = _make_percept(tick=1, energy=0.55)
        decision = gate.should_wake(
            GateContext(tick=1, percept=percept, wake_if=["energy below 0.6"])
        )
        return gate, gate.decisions[-1], decision

    def case_wake_if_comfort_below() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(low_comfort_threshold=0.25)
        # Comfort 0.55 is above the gate drive threshold (0.25) but below the LLM's `comfort below 0.6`
        percept = _make_percept(tick=1, comfort=0.55)
        decision = gate.should_wake(
            GateContext(tick=1, percept=percept, wake_if=["comfort below 0.6"])
        )
        return gate, gate.decisions[-1], decision

    def case_drive_low_energy() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(low_energy_threshold=0.30)
        percept = _make_percept(tick=1, energy=0.20)
        decision = gate.should_wake(GateContext(tick=1, percept=percept))
        return gate, gate.decisions[-1], decision

    def case_drive_low_comfort() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(low_comfort_threshold=0.30)
        percept = _make_percept(tick=1, comfort=0.18)
        decision = gate.should_wake(GateContext(tick=1, percept=percept))
        return gate, gate.decisions[-1], decision

    def case_drive_boredom() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(boredom_threshold=0.50)
        percept = _make_percept(tick=1)
        decision = gate.should_wake(GateContext(tick=1, percept=percept, boredom=0.65))
        return gate, gate.decisions[-1], decision

    def case_creator_spawn_nearby() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        spawn_ev = ObjectSpawnedEvent(
            tick=1,
            source="creator",
            object_id="object_9",
            position=Position(x=1.5, y=0.5),
            object={"id": "object_9", "position": {"x": 1.5, "y": 0.5}},
        )
        decision = gate.should_wake(
            GateContext(tick=1, percept=_make_percept(tick=1), events=[spawn_ev])
        )
        return gate, gate.decisions[-1], decision

    def case_creator_mail_arrives() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate()
        mail_ev = MailArrivedEvent(tick=1, source="creator", text="hello from creator")
        decision = gate.should_wake(
            GateContext(tick=1, percept=_make_percept(tick=1), events=[mail_ev])
        )
        return gate, gate.decisions[-1], decision

    def case_heartbeat() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(heartbeat_interval=10)
        decision = gate.should_wake(
            GateContext(tick=10, percept=_make_percept(tick=10), ticks_since_wake=10)
        )
        return gate, gate.decisions[-1], decision

    def case_continue_when_nothing_changes() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(heartbeat_interval=30)
        # Surprise is 0, no new object/combo/affordance, no intent_ended, no wake_if match, normal drives
        decision = gate.should_wake(
            GateContext(tick=1, percept=_make_percept(tick=1), surprise=0.0, boredom=0.1)
        )
        return gate, gate.decisions[-1], decision

    def case_unparseable_wake_if_wakes_at_next_heartbeat() -> tuple[RuleGate, GateDecisionEvent, WakeDecision]:
        gate = _make_seeded_gate(heartbeat_interval=5)
        # Tick 1: unparseable condition is set -> kept as text, does not wake early
        d1 = gate.should_wake(
            GateContext(
                tick=1,
                percept=_make_percept(tick=1),
                wake_if=["the red thing dances a jig"],
            )
        )
        assert d1.wake is False
        assert gate.unparseable_wake_if == ["the red thing dances a jig"]
        # Ticks 2..4: still no wake
        for t in (2, 3, 4):
            dt = gate.should_wake(GateContext(tick=t, percept=_make_percept(tick=t)))
            assert dt.wake is False
        # Tick 5 (next heartbeat): unparseable condition causes a wake with wake_if + heartbeat
        d5 = gate.should_wake(GateContext(tick=5, percept=_make_percept(tick=5)))
        return gate, gate.decisions[-1], d5

    return [
        ("start", case_start, True, ["start"]),
        ("surprise", case_surprise, True, ["surprise"]),
        ("novelty_object", case_novelty_never_seen_object, True, ["novelty"]),
        ("novelty_property_combo", case_novelty_never_seen_property_combination, True, ["novelty"]),
        ("novelty_affordance", case_novelty_never_seen_affordance, True, ["novelty"]),
        ("intent_ended", case_intent_ended, True, ["intent_ended"]),
        ("wake_if_moves", case_wake_if_object_moves, True, ["wake_if"]),
        ("wake_if_stops_moving", case_wake_if_object_stops_moving, True, ["wake_if"]),
        ("wake_if_disappears", case_wake_if_object_disappears, True, ["wake_if"]),
        ("wake_if_new_object_in_range", case_wake_if_new_object_in_range, True, ["wake_if"]),
        ("wake_if_energy_below", case_wake_if_energy_below, True, ["wake_if"]),
        ("wake_if_comfort_below", case_wake_if_comfort_below, True, ["wake_if"]),
        ("drive_low_energy", case_drive_low_energy, True, ["low_energy"]),
        ("drive_low_comfort", case_drive_low_comfort, True, ["low_comfort"]),
        ("drive_boredom", case_drive_boredom, True, ["boredom"]),
        ("creator_spawn_nearby", case_creator_spawn_nearby, True, ["creator_spawn"]),
        ("creator_mail_arrives", case_creator_mail_arrives, True, ["mail_arrived"]),
        ("heartbeat", case_heartbeat, True, ["heartbeat"]),
        ("continue_nothing_changes", case_continue_when_nothing_changes, False, []),
        (
            "unparseable_wake_if_next_heartbeat",
            case_unparseable_wake_if_wakes_at_next_heartbeat,
            True,
            ["wake_if", "heartbeat"],
        ),
    ]


@pytest.mark.parametrize(
    ("case_name", "run_case", "expected_wake", "expected_reasons"),
    _build_trigger_cases(),
    ids=[c[0] for c in _build_trigger_cases()],
)
def test_rule_gate_triggers_table(
    case_name: str,
    run_case: Callable[[], tuple[RuleGate, GateDecisionEvent, WakeDecision]],
    expected_wake: bool,
    expected_reasons: list[str],
) -> None:
    """Table-driven verification of every RuleGate trigger, continue, and unparseable wake_if."""
    del case_name
    gate, event, decision = run_case()
    assert isinstance(gate, Gate)
    assert decision.wake is expected_wake
    assert bool(decision) is expected_wake
    assert decision.reasons == expected_reasons

    # Every decision is logged as a gate_decision event with matching wake flag and reasons
    assert event.type == "gate_decision"
    assert event.wake is expected_wake
    assert event.reasons == expected_reasons
    assert event.outcome == ("wake" if expected_wake else "continue")


def test_rule_gate_replaces_heartbeat_gate_by_default_and_reads_config(tmp_path: Path) -> None:
    """RuleGate implements Gate, replaces HeartbeatGate by default in AgentLoop, and loads from [gate]."""
    loop = AgentLoop()
    assert isinstance(loop.gate, RuleGate)
    assert isinstance(loop.gate, Gate)
    assert not isinstance(loop.gate, HeartbeatGate)

    # Verify [gate] configuration loading from TOML
    cfg_path = tmp_path / "arden.toml"
    cfg_path.write_text(
        "[gate]\n"
        "heartbeat_interval = 12\n"
        "surprise_threshold = 0.15\n"
        "low_energy_threshold = 0.4\n"
        "low_comfort_threshold = 0.35\n"
        "boredom_threshold = 0.45\n"
        "move_threshold = 0.02\n",
        encoding="utf-8",
    )
    settings = load_config(cfg_path)
    assert settings.gate == GateConfig(
        heartbeat_interval=12,
        surprise_threshold=0.15,
        low_energy_threshold=0.4,
        low_comfort_threshold=0.35,
        boredom_threshold=0.45,
        move_threshold=0.02,
    )

    configured_loop = AgentLoop(settings=settings)
    assert isinstance(configured_loop.gate, RuleGate)
    assert configured_loop.gate.heartbeat_interval == 12
    assert configured_loop.gate.surprise_threshold == 0.15
    assert configured_loop.gate.low_energy_threshold == 0.4
    assert configured_loop.gate.low_comfort_threshold == 0.35
    assert configured_loop.gate.boredom_threshold == 0.45
    assert configured_loop.gate.move_threshold == 0.02


def test_wake_if_condition_language_parsing_and_unparseable_preservation() -> None:
    """Test parsing of all documented wake_if conditions and preservation of unparseable strings."""
    c_moves = parse_wake_if("object_7 moves")
    assert c_moves == WakeCondition(
        raw="object_7 moves",
        kind="object_moves",
        target_id="object_7",
        is_valid=True,
    )

    c_stops = parse_wake_if("object_7 stops moving")
    assert c_stops == WakeCondition(
        raw="object_7 stops moving",
        kind="object_stops_moving",
        target_id="object_7",
        is_valid=True,
    )

    c_disappears = parse_wake_if("object_3 disappears")
    assert c_disappears == WakeCondition(
        raw="object_3 disappears",
        kind="object_disappears",
        target_id="object_3",
        is_valid=True,
    )

    c_new = parse_wake_if("new object in range")
    assert c_new == WakeCondition(
        raw="new object in range",
        kind="new_object_in_range",
        is_valid=True,
    )

    c_energy = parse_wake_if("energy below 0.35")
    assert c_energy == WakeCondition(
        raw="energy below 0.35",
        kind="energy_below",
        threshold=0.35,
        inclusive=False,
        is_valid=True,
    )

    c_comfort = parse_wake_if("comfort below 0.4")
    assert c_comfort == WakeCondition(
        raw="comfort below 0.4",
        kind="comfort_below",
        threshold=0.4,
        inclusive=False,
        is_valid=True,
    )

    # Unparseable conditions are kept as text with is_valid=False rather than raising or being dropped
    for raw_unparseable in (
        "a sound answers",
        "the red thing moves",
        "energy below 1.5",
        "comfort below -0.1",
        "",
    ):
        parsed = parse_wake_if(raw_unparseable)
        assert parsed.is_valid is False
        assert parsed.is_unparseable is True
        assert parsed.kind == "unparseable"
        assert parsed.raw == raw_unparseable

    batch = parse_wake_conditions(["object_1 moves", "a sound answers"])
    assert len(batch) == 2
    assert batch[0].is_valid is True
    assert batch[1].is_valid is False
    assert batch[1].raw == "a sound answers"


def test_creator_spawn_distance_filtering_and_mailbox_state_change() -> None:
    """Creator spawn outside sensing radius does not wake; spawn nearby and mailbox changes do wake."""
    world = World(seed=10)
    gate = _make_seeded_gate(world=world)

    # 1. Creator spawn far away (at x=50, well outside sensing_radius=10) does not wake
    far_spawn = ObjectSpawnedEvent(
        tick=1,
        source="creator",
        object_id="object_99",
        position=Position(x=50.0, y=50.0),
        object={"id": "object_99", "position": {"x": 50.0, "y": 50.0}},
    )
    d_far = gate.should_wake(GateContext(tick=1, world=world, percept=_make_percept(tick=1), events=[far_spawn]))
    assert d_far.wake is False
    assert d_far.reasons == []

    # 2. Mail arriving via an object_changed event on a mailbox object wakes with mail_arrived
    mail_change = ObjectChangedEvent(
        tick=2,
        source="creator",
        object_id="object_1",
        changes={"state": {"mail": ["letter 1"]}},
        object={"id": "object_1", "state": {"mail": ["letter 1"]}},
    )
    d_mail = gate.should_wake(
        GateContext(tick=2, world=world, percept=_make_percept(tick=2), events=[mail_change])
    )
    assert d_mail.wake is True
    assert "mail_arrived" in d_mail.reasons


def test_every_decision_emits_gate_decision_event_in_agent_loop() -> None:
    """Every tick in AgentLoop emits a gate_decision event (wake or continue) with reasons."""
    settings = Settings()
    rng = make_rng(settings)
    world = build_test_world(rng=rng, settings=settings)

    # Multi-step intent with an unparseable wake_if so the agent runs across several continue ticks
    # before waking at the next heartbeat
    multi_step_intent = {
        "goal": "move toward object_1",
        "steps": [
            {"verb": "move", "target": None, "params": {"direction": "toward object_1", "speed": 0.3}},
            {"verb": "look", "target": "object_1", "params": {}},
        ],
        "wake_if": ["something unexpected happens"],
        "expect": "closer to object_1",
        "thought": "Walking slowly toward the tall cylinder.",
        "confidence": 0.8,
    }
    mock_llm = MockLLM(
        seed=1,
        script=[json.dumps(multi_step_intent), json.dumps(multi_step_intent)],
    )

    decision_events: list[GateDecisionEvent] = []
    world.subscribe("gate_decision", decision_events.append)

    loop = AgentLoop(
        world=world,
        llm=mock_llm,
        settings=settings,
        rng=rng,
        heartbeat_interval=5,
    )
    results = loop.run(6)

    assert len(results) == 6
    assert len(decision_events) == 6
    assert len(loop.gate_decision_events) == 6

    # Tick 0: woke on start
    assert decision_events[0].tick == 0
    assert decision_events[0].wake is True
    assert decision_events[0].reasons == ["start"]

    # Ticks 1..4: continue decisions emitted with wake=False and reasons=[] while preserving unparseable wake_if
    for t in (1, 2, 3, 4):
        assert decision_events[t].tick == t
        assert decision_events[t].wake is False
        assert decision_events[t].reasons == []
        assert decision_events[t].continued is True
        assert decision_events[t].unparseable_wake_if == ["something unexpected happens"]

    # Tick 5: heartbeat fires and unparseable wake_if is included in reasons
    assert decision_events[5].tick == 5
    assert decision_events[5].wake is True
    assert "wake_if" in decision_events[5].reasons
    assert "heartbeat" in decision_events[5].reasons
    assert decision_events[5].matched_wake_if == ["something unexpected happens"]
