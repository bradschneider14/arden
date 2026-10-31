"""Unit tests for the scripted low-level controller (T10)."""

from __future__ import annotations

from typing import Any

import pytest

from arden.actions.verbs import BodyModifiers, in_reach
from arden.controller import (
    FAILED,
    IDLE,
    NO_PATH,
    OUT_OF_REACH,
    RUNNING,
    SUCCEEDED,
    Controller,
    ControllerStatus,
    IntentEndedEvent,
)
from arden.llm.schemas import Intent, Step
from arden.perception import Percept
from arden.world import Position, TerrainGrid, World, WorldEvent

_BALL_PROPS: dict[str, Any] = {
    "shape": "sphere",
    "height": 0.2,
    "volume": 0.004,
    "weight": 0.3,
    "rigidity": 0.5,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "red",
    "fixed": False,
    "state": {"wet": False},
    "behavior": [{"verb": "push", "scale": 6}],
}

_ROCK_PROPS: dict[str, Any] = {
    "shape": "cube",
    "height": 0.8,
    "volume": 0.4,
    "weight": 25.0,
    "rigidity": 0.9,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "gray",
    "fixed": True,
}


def _make_intent(steps: list[Step], *, goal: str = "test goal", expect: str = "test outcome") -> Intent:
    return Intent(
        goal=goal,
        steps=steps,
        wake_if=["new object in range"],
        expect=expect,
        thought="testing controller",
        confidence=0.8,
    )


def test_idle_when_no_intent_is_set() -> None:
    world = World(seed=1, body={"position": {"x": 0.0, "y": 0.0}})
    controller = Controller()

    assert controller.intent is None
    assert controller.is_idle is True
    assert controller.is_resting is True

    status = controller.tick(world)
    assert isinstance(status, ControllerStatus)
    assert status == IDLE
    assert status == ControllerStatus.idle()
    assert status.state == "idle"
    assert status.is_idle is True
    assert status.is_resting is True
    assert controller.is_resting is True
    assert world.body.position == Position(x=0.0, y=0.0)


def test_walk_to_target_then_push_it_succeeds_and_emits_intent_ended() -> None:
    world = World(seed=2, body={"position": {"x": 0.0, "y": 0.0}})
    ball = world.spawn(_BALL_PROPS, {"x": 4.0, "y": 0.0})

    ended_events: list[WorldEvent] = []
    world.subscribe("intent_ended", ended_events.append)

    controller_events: list[IntentEndedEvent] = []
    controller = Controller()
    controller.subscribe(controller_events.append)

    intent = _make_intent(
        [
            Step(verb="move", target=ball.id, params={"speed": 1.0}),
            Step(verb="push", target=ball.id, params={"force": 0.6}),
        ],
        goal=f"walk to {ball.id} and push it",
        expect=f"{ball.id} moves east",
    )
    controller.set_intent(intent)

    statuses: list[ControllerStatus] = []
    for _ in range(10):
        st = controller.tick(world)
        statuses.append(st)
        world.step()
        if st != RUNNING:
            break

    # Several running ticks followed by succeeded
    assert len(statuses) >= 3
    assert all(s == RUNNING for s in statuses[:-1])
    final_status = statuses[-1]
    assert final_status == SUCCEEDED
    assert final_status == ControllerStatus.succeeded()
    assert final_status.is_succeeded is True

    # Agent walked into reach of the ball and pushed it east
    assert world.objects[ball.id].position.x > 4.0

    # Emitted exactly one intent_ended event with the final observation
    assert len(ended_events) == 1
    assert len(controller_events) == 1
    ev = ended_events[0]
    assert isinstance(ev, IntentEndedEvent)
    assert ev.type == "intent_ended"
    assert ev.status == "succeeded"
    assert ev.ok is True
    assert ev.failed is False
    assert ev.reason is None
    assert ev.intent == intent
    assert isinstance(ev.observation, Percept)
    assert any(obj.id == ball.id for obj in ev.observation.objects)

    # Subsequent tick with no new intent idles (agent rests)
    after_status = controller.tick(world)
    assert after_status == IDLE
    assert after_status.is_resting is True
    assert len(ended_events) == 1


def test_navigation_avoids_obstacles_on_tile_grid_toward_target_and_position() -> None:
    terrain = TerrainGrid()
    # Place a ledge wall at cell (2, 0) and (2, -1) blocking the direct y=0 path
    terrain.set_tile(2, 0, "ledge")
    terrain.set_tile(2, -1, "ledge")
    world = World(seed=3, terrain=terrain, body={"position": {"x": 0.0, "y": 0.5}})

    # Also place a fixed boulder obstacle at (1.5, 0.5)
    boulder = world.spawn(_ROCK_PROPS, {"x": 1.5, "y": 0.5})
    ball = world.spawn(_BALL_PROPS, {"x": 4.5, "y": 0.5})

    controller = Controller()
    intent = _make_intent(
        [
            Step(verb="move", params={"direction": f"toward {ball.id}", "speed": 1.0}),
            Step(verb="push", target=ball.id, params={"force": 0.5}),
        ]
    )
    controller.set_intent(intent)

    visited: list[Position] = [world.body.position.model_copy()]
    final_status: ControllerStatus | None = None
    for _ in range(25):
        st = controller.tick(world)
        visited.append(world.body.position.model_copy())
        world.step()
        if st != RUNNING:
            final_status = st
            break

    assert final_status is not None
    assert final_status == SUCCEEDED
    assert in_reach(world, ball.id) or world.objects[ball.id].position.x > 4.5

    # Verify the agent detoured around the boulder and ledge tiles
    for pos in visited:
        assert not world.terrain.is_ledge(pos)
        dist_boulder = ((pos.x - boulder.position.x) ** 2 + (pos.y - boulder.position.y) ** 2) ** 0.5
        assert dist_boulder >= boulder.radius

    # Now test navigating to an explicit (x, y) position around the obstacle
    pos_intent = _make_intent(
        [Step(verb="move", params={"position": {"x": 0.0, "y": 0.5}, "speed": 1.0})]
    )
    controller.set_intent(pos_intent)
    for _ in range(25):
        st = controller.tick(world)
        world.step()
        if st != RUNNING:
            final_status = st
            break

    assert final_status == SUCCEEDED
    assert world.body.position.x == pytest.approx(0.0, abs=1e-2)
    assert world.body.position.y == pytest.approx(0.5, abs=1e-2)


def test_move_steps_respect_body_modifiers() -> None:
    world_normal = World(seed=4, body={"position": {"x": 0.0, "y": 0.0}})
    world_slow = World(seed=4, body={"position": {"x": 0.0, "y": 0.0}})
    world_slow.body_modifiers = BodyModifiers(speed=0.5)

    intent = _make_intent(
        [Step(verb="move", params={"to": {"x": 3.0, "y": 0.0}, "speed": 1.0})]
    )

    ctrl_normal = Controller(intent=intent)
    ctrl_slow = Controller(intent=intent)

    # Single tick comparison: slow agent moves half as far as normal agent
    st_norm_1 = ctrl_normal.tick(world_normal)
    st_slow_1 = ctrl_slow.tick(world_slow)
    assert st_norm_1 == RUNNING
    assert st_slow_1 == RUNNING
    assert world_normal.body.position.x == pytest.approx(1.0)
    assert world_slow.body.position.x == pytest.approx(0.5)

    ticks_normal = 1
    while True:
        ticks_normal += 1
        if ctrl_normal.tick(world_normal) != RUNNING:
            break

    ticks_slow = 1
    while True:
        ticks_slow += 1
        if ctrl_slow.tick(world_slow) != RUNNING:
            break

    assert ticks_normal == 3
    assert ticks_slow == 6
    assert world_normal.body.position.x == pytest.approx(3.0)
    assert world_slow.body.position.x == pytest.approx(3.0)


def test_unreachable_target_fails_with_no_path_or_out_of_reach_and_emits_event() -> None:
    # 1. Target enclosed by a ring of impassable ledge tiles -> fails with no_path
    terrain = TerrainGrid()
    for tx in range(3, 8):
        for ty in range(-2, 3):
            if (tx, ty) != (5, 0):
                terrain.set_tile(tx, ty, "ledge")

    world = World(seed=5, terrain=terrain, body={"position": {"x": 0.0, "y": 0.5}})
    trapped_ball = world.spawn(_BALL_PROPS, {"x": 5.5, "y": 0.5})

    events: list[WorldEvent] = []
    world.subscribe("intent_ended", events.append)

    controller = Controller()
    move_step = Step(verb="move", target=trapped_ball.id, params={"speed": 1.0})
    push_step = Step(verb="push", target=trapped_ball.id, params={"force": 0.5})
    controller.set_intent(_make_intent([move_step, push_step]))

    # Attempt 1 fails -> retries once (returns running)
    st1 = controller.tick(world)
    assert st1 == RUNNING
    assert controller.retries == 1
    assert len(events) == 0

    # Attempt 2 (retry) also fails -> returns failed(reason) and emits intent_ended
    st2 = controller.tick(world)
    assert st2 == FAILED
    assert st2.is_failed is True
    assert st2.reason in (NO_PATH, OUT_OF_REACH)
    assert st2 == f"failed({st2.reason})"
    assert st2 == ControllerStatus.failed(st2.reason)
    assert st2.step == move_step
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, IntentEndedEvent)
    assert ev.status == "failed"
    assert ev.failed is True
    assert ev.reason == st2.reason
    assert ev.step == move_step
    assert isinstance(ev.observation, Percept)

    # 2. Target on a raised ledge with no way up -> fails with out_of_reach
    terrain2 = TerrainGrid()
    terrain2.set_tile(4, 0, "ledge")
    world2 = World(seed=6, terrain=terrain2, body={"position": {"x": 0.0, "y": 0.5}})
    ledge_ball = world2.spawn({**_BALL_PROPS, "level": "raised"}, {"x": 4.5, "y": 0.5})

    ctrl2 = Controller()
    ctrl2.set_intent(_make_intent([Step(verb="move", target=ledge_ball.id)]))
    assert ctrl2.tick(world2) == RUNNING
    st_ledge = ctrl2.tick(world2)
    assert st_ledge == FAILED
    assert st_ledge.reason in (OUT_OF_REACH, NO_PATH)


def test_retries_failed_step_at_most_once_and_succeeds_if_retry_works() -> None:
    world = World(seed=7, body={"position": {"x": 0.0, "y": 0.0}})
    # Ball starts out of reach at x=2.2, moving west toward the agent at (-1.0, 0.0)
    ball = world.spawn(_BALL_PROPS, {"x": 2.2, "y": 0.0})
    world.apply_impulse(ball.id, (-1.0, 0.0), 1.5)

    controller = Controller()
    grab_step = Step(verb="grab", target=ball.id)
    speak_step = Step(verb="speak", params={"text": "got it"})
    controller.set_intent(_make_intent([grab_step, speak_step]))

    # Tick 1: ball is at 2.2 (out_of_reach), grab fails on attempt 1 -> scheduled for retry
    st1 = controller.tick(world)
    assert st1 == RUNNING
    assert controller.retries == 1
    assert controller.step_index == 0

    # World steps: ball rolls into reach (< 1.5m from origin)
    world.step()
    assert in_reach(world, ball.id)

    # Tick 2: retry of grab_step succeeds! Advances to speak_step (index 1)
    st2 = controller.tick(world)
    assert st2 == RUNNING
    assert controller.retries == 0
    assert controller.step_index == 1
    assert world.body.holding == ball.id

    # Tick 3: speak_step succeeds -> intent finishes with succeeded
    st3 = controller.tick(world)
    assert st3 == SUCCEEDED
    assert controller.last_event is not None
    assert controller.last_event.status == "succeeded"


def test_set_intent_interrupts_running_intent() -> None:
    world = World(seed=8, body={"position": {"x": 0.0, "y": 0.0}})
    ball_a = world.spawn(_BALL_PROPS, {"x": 5.0, "y": 0.0})
    ball_b = world.spawn(_BALL_PROPS, {"x": 0.0, "y": 1.0})

    controller = Controller()
    intent_a = _make_intent([Step(verb="move", target=ball_a.id)])
    intent_b = _make_intent([Step(verb="push", target=ball_b.id, params={"force": 0.5})])

    controller.set_intent(intent_a)
    assert controller.tick(world) == RUNNING
    assert world.body.position.x > 0.0

    # Interrupt with intent_b while intent_a is still running
    controller.set_intent(intent_b)
    assert controller.intent == intent_b
    assert controller.step_index == 0

    st = controller.tick(world)
    assert st == SUCCEEDED
    assert controller.last_event is not None
    assert controller.last_event.intent == intent_b
