"""Tests for the scripted low-level intent controller (M3 / T10).

All tests use hand-built ``Intent`` and ``Step`` instances (no LLM, GPU, or network).
"""

from __future__ import annotations

import math

import pytest

from arden.actions.verbs import NO_PATH, OUT_OF_REACH, in_reach
from arden.controller import (
    FAILED,
    IDLE,
    RUNNING,
    SUCCEEDED,
    Controller,
    ControllerStatus,
    IntentEndedEvent,
    find_grid_path,
)
from arden.llm.schemas import Intent, Step
from arden.perception.feed import Percept
from arden.world import Position, TerrainGrid, World


def obj_props(
    *,
    shape: str = "cube",
    weight: float = 1.0,
    height: float = 0.5,
    volume: float = 0.2,
    rigidity: float = 0.8,
    color: str = "brown",
    fixed: bool = False,
    level: str = "ground",
) -> dict[str, object]:
    """Return a valid WorldObject property dict for world.spawn()."""
    return {
        "shape": shape,
        "weight": weight,
        "height": height,
        "volume": volume,
        "rigidity": rigidity,
        "color": color,
        "fixed": fixed,
        "level": level,
    }


def make_intent(
    steps: list[Step],
    *,
    goal: str = "test goal",
    expect: str = "expected outcome",
    wake_if: list[str] | None = None,
    thought: str = "hand-built test intent",
    confidence: float = 0.85,
) -> Intent:
    """Build a valid ``Intent`` without calling any LLM."""
    return Intent(
        goal=goal,
        steps=steps,
        wake_if=wake_if or [],
        expect=expect,
        thought=thought,
        confidence=confidence,
    )


def test_idle_with_no_intent_rests_and_recovers_energy() -> None:
    """With no active intent, Controller.tick idles and the agent rests."""
    world = World()
    world.body.energy = 0.50
    world.body.comfort = 0.60

    controller = Controller()
    assert controller.intent is None
    assert controller.status == IDLE

    status = controller.tick(world)
    assert status == IDLE
    assert status == ControllerStatus.IDLE
    assert status.is_idle
    assert status.state == "idle"
    assert status.reason is None
    assert world.body_state.resting is True
    assert world.body.energy > 0.50
    assert world.body.comfort > 0.60


def test_walk_to_target_then_push_it_succeeds() -> None:
    """Controller walks across the grid to a target object and then pushes it."""
    world = World()
    world.body.position = Position(x=0.5, y=0.5)
    crate = world.spawn(obj_props(shape="cube", weight=1.0), pos=(4.5, 0.5))
    initial_crate_x = crate.position.x

    events: list[IntentEndedEvent] = []
    world.subscribe("intent_ended", events.append)  # type: ignore[arg-type]

    intent = make_intent(
        [
            Step(verb="move", target=crate.id, params={"speed": 1.0}),
            Step(
                verb="push",
                target=crate.id,
                params={"direction": "east", "force": 0.7},
            ),
        ],
        goal=f"walk to {crate.id} and push it east",
        expect=f"{crate.id} slides east",
    )

    controller = Controller()
    controller.set_intent(intent)
    assert controller.intent == intent

    statuses: list[ControllerStatus] = []
    for _ in range(20):
        st = controller.tick(world)
        statuses.append(st)
        if st.done:
            break

    assert len(statuses) >= 2
    assert all(s == RUNNING for s in statuses[:-1])
    final_status = statuses[-1]
    assert final_status == SUCCEEDED
    assert final_status == ControllerStatus.SUCCEEDED
    assert final_status.is_succeeded
    assert final_status.reason is None

    # Crate should have been pushed east
    updated_crate = world.objects[crate.id]
    assert updated_crate.position.x > initial_crate_x
    assert in_reach(world, crate.id) or math.hypot(
        world.body.position.x - initial_crate_x,
        world.body.position.y - 0.5,
    ) <= 1.5

    # IntentEndedEvent emitted with final Percept observation
    assert len(events) == 1
    ended = events[0]
    assert ended.type == "intent_ended"
    assert ended.status == "succeeded"
    assert ended.ok is True
    assert ended.succeeded is True
    assert ended.failed is False
    assert ended.goal == intent.goal
    assert ended.expect == intent.expect
    assert isinstance(ended.observation, Percept)
    assert final_status.observation == ended.observation
    assert any(o.id == crate.id for o in ended.observation.objects)

    # Subsequent tick with no new intent idles
    next_status = controller.tick(world)
    assert next_status == IDLE


def test_move_supports_toward_object_direction_and_position_targets() -> None:
    """Controller handles 'toward object_<n>' direction strings and coordinate targets."""
    world = World()
    world.body.position = Position(x=0.5, y=0.5)
    stone = world.spawn(obj_props(shape="sphere", weight=0.8), pos=(3.5, 0.5))

    intent = make_intent(
        [
            Step(
                verb="move",
                target=None,
                params={"direction": f"toward {stone.id}", "speed": 0.9},
            ),
            Step(verb="grab", target=stone.id, params={}),
            Step(verb="move", target=None, params={"position": [1.5, 2.5], "speed": 1.0}),
            Step(verb="drop", target=stone.id, params={}),
        ]
    )

    controller = Controller(intent)
    for _ in range(30):
        status = controller.tick(world)
        if status.done:
            break

    assert status == SUCCEEDED
    assert world.body.holding is None
    assert math.hypot(world.body.position.x - 1.5, world.body.position.y - 2.5) <= 0.4
    dropped = world.objects[stone.id]
    assert math.hypot(dropped.position.x - 1.5, dropped.position.y - 2.5) <= 0.4


def test_navigation_avoids_ledge_and_object_obstacles() -> None:
    """Controller navigates around impassable ledge tiles and blocking objects on the grid."""
    terrain = TerrainGrid(width=8, height=6)
    # Build a vertical ledge wall at x=2, y=0..3 so the agent must route around y=4
    for ty in range(4):
        terrain.set_tile(2.5, ty + 0.5, "ledge")

    world = World(terrain=terrain)
    world.body.position = Position(x=0.5, y=1.5)

    # Also place a fixed boulder obstacle at (3.5, 4.5) so path must curve cleanly
    boulder = world.spawn(
        obj_props(shape="sphere", weight=20.0, volume=0.5, fixed=True),
        pos=(3.5, 3.5),
    )
    target = world.spawn(
        obj_props(shape="cube", weight=1.0),
        pos=(5.5, 1.5),
    )

    path = find_grid_path(
        world,
        target.position,
        target_object=target,
        exclude_ids={target.id},
    )
    assert path is not None
    assert len(path) >= 2

    intent = make_intent(
        [
            Step(verb="move", target=target.id, params={"speed": 1.0}),
            Step(
                verb="push",
                target=target.id,
                params={"direction": "east", "force": 0.5},
            ),
        ]
    )
    controller = Controller(intent)

    visited_cells: list[tuple[int, int]] = []
    for _ in range(40):
        status = controller.tick(world)
        cell = world.terrain.world_to_cell(world.body.position)
        visited_cells.append(cell)
        assert not world.terrain.is_ledge(world.body.position)
        assert (
            math.hypot(
                world.body.position.x - boulder.position.x,
                world.body.position.y - boulder.position.y,
            )
            >= boulder.radius
        )
        if status.done:
            break

    assert status == SUCCEEDED
    assert world.objects[target.id].position.x > 5.5


def test_move_steps_respect_body_modifiers() -> None:
    """Fatigued/cold agent with lower BodyModifiers.speed moves slower and takes more ticks."""
    # Fresh agent (energy=1.0, comfort=1.0 -> speed modifier = 1.0)
    world_fast = World()
    world_fast.body.position = Position(x=0.5, y=0.5)
    intent_fast = make_intent(
        [Step(verb="move", target=None, params={"position": [6.5, 0.5], "speed": 1.0})]
    )
    ctrl_fast = Controller(intent_fast)
    ticks_fast = 0
    first_step_fast = 0.0
    for i in range(40):
        prev_x = world_fast.body.position.x
        st = ctrl_fast.tick(world_fast)
        if i == 0:
            first_step_fast = world_fast.body.position.x - prev_x
        ticks_fast += 1
        if st.done:
            break
    assert st == SUCCEEDED

    # Fatigued agent (energy=0.0 -> speed modifier = min_speed_modifier < 1.0)
    world_slow = World()
    world_slow.body.position = Position(x=0.5, y=0.5)
    world_slow.body.energy = 0.0
    world_slow.body_state.cold = 1.0
    assert world_slow.body_modifiers.speed < world_fast.body_modifiers.speed

    intent_slow = make_intent(
        [Step(verb="move", target=None, params={"position": [6.5, 0.5], "speed": 1.0})]
    )
    ctrl_slow = Controller(intent_slow)
    ticks_slow = 0
    first_step_slow = 0.0
    for i in range(60):
        prev_x = world_slow.body.position.x
        st_slow = ctrl_slow.tick(world_slow)
        if i == 0:
            first_step_slow = world_slow.body.position.x - prev_x
        ticks_slow += 1
        if st_slow.done:
            break

    assert st_slow == SUCCEEDED
    assert first_step_slow < first_step_fast
    assert ticks_slow > ticks_fast


def test_unreachable_target_on_ledge_fails_with_out_of_reach() -> None:
    """Moving to or interacting with an unreachable ledge target fails with out_of_reach."""
    terrain = TerrainGrid(width=6, height=6)
    terrain.set_tile(4.5, 4.5, "ledge")
    world = World(terrain=terrain)
    world.body.position = Position(x=0.5, y=0.5)

    gem = world.spawn(
        obj_props(shape="sphere", weight=0.5, level="raised"),
        pos=(4.5, 4.5),
    )

    events: list[IntentEndedEvent] = []
    world.subscribe("intent_ended", events.append)  # type: ignore[arg-type]

    move_step = Step(verb="move", target=gem.id, params={"speed": 1.0})
    intent = make_intent(
        [
            move_step,
            Step(verb="grab", target=gem.id, params={}),
        ]
    )
    controller = Controller(intent)

    # First tick fails attempt 1 and schedules 1 retry (status=running)
    status_1 = controller.tick(world)
    assert status_1 == RUNNING
    assert status_1.retries == 1

    # Second tick retries once and transitions to failed(out_of_reach)
    status_2 = controller.tick(world)
    assert status_2 == FAILED
    assert status_2 == "failed(out_of_reach)"
    assert status_2.is_failed
    assert status_2.reason == OUT_OF_REACH
    assert status_2.step == move_step
    assert status_2.failed_step == move_step
    assert status_2.retries == 1

    assert len(events) == 1
    assert events[0].status == "failed"
    assert events[0].ok is False
    assert events[0].failed is True
    assert events[0].reason == OUT_OF_REACH
    assert events[0].step == move_step
    assert isinstance(events[0].observation, Percept)


def test_walled_off_target_fails_with_no_path() -> None:
    """A target completely enclosed by obstacles fails with no_path after 1 retry."""
    terrain = TerrainGrid(width=8, height=8)
    # Surround cell (5, 5) with ledge walls on all 4 cardinal sides
    for cx, cy in ((4, 5), (6, 5), (5, 4), (5, 6)):
        terrain.set_tile(cx + 0.5, cy + 0.5, "ledge")

    world = World(terrain=terrain)
    world.body.position = Position(x=1.5, y=1.5)
    box = world.spawn(
        obj_props(shape="cube", weight=1.0),
        pos=(5.5, 5.5),
    )

    move_step = Step(verb="move", target=box.id, params={"speed": 1.0})
    controller = Controller(make_intent([move_step]))

    s1 = controller.tick(world)
    assert s1 == RUNNING
    assert s1.retries == 1

    s2 = controller.tick(world)
    assert s2 == FAILED
    assert s2.reason in (NO_PATH, OUT_OF_REACH)
    assert s2.reason == NO_PATH
    assert s2.step == move_step


def test_step_retry_succeeds_when_transient_failure_clears() -> None:
    """A step that fails once and succeeds on its single retry completes the intent."""
    world = World()
    world.body.position = Position(x=0.5, y=0.5)
    # Initially place target out of reach for a direct push (no move step)
    ball = world.spawn(
        obj_props(shape="sphere", weight=0.5),
        pos=(4.5, 0.5),
    )

    push_step = Step(
        verb="push",
        target=ball.id,
        params={"direction": "east", "force": 0.6},
    )
    controller = Controller(make_intent([push_step]))

    # Attempt 1 fails with out_of_reach -> controller stays running for 1 retry
    first = controller.tick(world)
    assert first == RUNNING
    assert first.retries == 1

    # Move the ball into reach before the retry tick
    world.move(ball.id, (1.0, 0.5))

    # Retry attempt succeeds!
    second = controller.tick(world)
    assert second == SUCCEEDED
    assert world.objects[ball.id].position.x > 1.0


def test_missing_object_and_out_of_bounds_position_fail_with_out_of_reach() -> None:
    """Missing target object id and out-of-bounds target coordinates fail with out_of_reach."""
    terrain = TerrainGrid(width=5, height=5)
    world = World(terrain=terrain)
    world.body.position = Position(x=1.5, y=1.5)

    # 1. Non-existent object id
    missing_step = Step(verb="move", target="object_99", params={"speed": 1.0})
    ctrl = Controller(make_intent([missing_step]))
    assert ctrl.tick(world) == RUNNING
    res_missing = ctrl.tick(world)
    assert res_missing == FAILED
    assert res_missing.reason == OUT_OF_REACH
    assert res_missing.step == missing_step

    # 2. Out-of-bounds coordinate on bounded TerrainGrid(5, 5)
    oob_step = Step(verb="move", target=None, params={"position": [12.0, 12.0], "speed": 1.0})
    ctrl.set_intent(make_intent([oob_step]))
    assert ctrl.tick(world) == RUNNING
    res_oob = ctrl.tick(world)
    assert res_oob == FAILED
    assert res_oob.reason == OUT_OF_REACH
    assert res_oob.step == oob_step


def test_second_step_failure_retries_once_and_reports_failed_step_and_reason() -> None:
    """When step 0 succeeds and step 1 fails twice, failed status reports step 1 and its reason."""
    world = World()
    world.body.position = Position(x=0.5, y=0.5)
    heavy_rock = world.spawn(
        obj_props(shape="sphere", weight=25.0, fixed=True),
        pos=(2.5, 0.5),
    )

    step_move = Step(verb="move", target=heavy_rock.id, params={"speed": 1.0})
    step_grab = Step(verb="grab", target=heavy_rock.id, params={})
    intent = make_intent([step_move, step_grab])

    events: list[IntentEndedEvent] = []
    controller = Controller(intent)
    controller.subscribe("intent_ended", events.append)

    for _ in range(15):
        st = controller.tick(world)
        if st.done:
            break

    assert st == FAILED
    assert st.reason == "too_heavy"
    assert st.step == step_grab
    assert st.step_index == 1
    assert st.retries == 1
    assert len(events) == 1
    assert events[0].status == "failed"
    assert events[0].step == step_grab
    assert events[0].reason == "too_heavy"


def test_directional_move_uses_greedy_avoidance_around_blocking_object() -> None:
    """A pure directional move steers around an obstacle directly ahead."""
    world = World()
    world.body.position = Position(x=0.5, y=0.5)
    pillar = world.spawn(
        obj_props(shape="cylinder", weight=15.0, volume=0.4, fixed=True),
        pos=(1.3, 0.5),
    )

    move_east = Step(verb="move", target=None, params={"direction": "east", "speed": 1.0})
    controller = Controller(make_intent([move_east]))

    st = controller.tick(world)
    assert st == SUCCEEDED
    assert world.body.position.x > 0.5
    assert (
        math.hypot(
            world.body.position.x - pillar.position.x,
            world.body.position.y - pillar.position.y,
        )
        >= pillar.radius
    )

