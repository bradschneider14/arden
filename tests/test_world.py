"""Tests for World state, event bus, creator operations, clock, and save/load."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from arden.world import (
    AgentBody,
    Clock,
    ObjectChangedEvent,
    ObjectDespawnedEvent,
    ObjectMovedEvent,
    ObjectSpawnedEvent,
    Position,
    TickEvent,
    World,
    WorldClock,
    WorldEvent,
)

_SAMPLE_PROPS: dict[str, Any] = {
    "shape": "sphere",
    "height": 0.2,
    "volume": 0.004,
    "weight": 0.3,
    "rigidity": 0.4,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "red",
    "fixed": False,
    "state": {"wet": False},
    "behavior": [{"verb": "push", "scale": 4}],
}


def test_world_holds_objects_body_tick_and_rng_and_steps_one_tick() -> None:
    rng = random.Random(42)
    world = World(
        rng=rng,
        body={"position": {"x": 1.0, "y": 2.0}, "energy": 0.8, "comfort": 0.9},
    )

    assert world.tick == 0
    assert world.rng is rng
    assert isinstance(world.body, AgentBody)
    assert world.body.position == Position(x=1.0, y=2.0)
    assert world.body.energy == 0.8
    assert world.body.comfort == 0.9
    assert world.body.holding is None
    assert world.objects == {}

    obj = world.spawn(_SAMPLE_PROPS, {"x": 3.0, "y": 4.0})
    assert obj.id == "object_1"
    assert world.objects["object_1"] == obj

    new_tick = world.step()
    assert new_tick == 1
    assert world.tick == 1


def test_creator_operations_emit_typed_events_marked_creator() -> None:
    world = World(seed=7)
    all_events: list[WorldEvent] = []
    spawn_events: list[WorldEvent] = []
    move_events: list[WorldEvent] = []
    despawn_events: list[WorldEvent] = []

    world.subscribe(all_events.append)
    world.subscribe("object_spawned", spawn_events.append)
    world.subscribe("object_moved", move_events.append)
    world.subscribe("object_despawned", despawn_events.append)

    obj = world.spawn(_SAMPLE_PROPS, Position(x=1.0, y=2.0))
    assert obj.id == "object_1"
    assert obj.position == Position(x=1.0, y=2.0)

    world.body.holding = obj.id
    moved = world.move(obj.id, {"x": 5.0, "y": -1.0})
    assert moved.position == Position(x=5.0, y=-1.0)
    assert world.objects[obj.id].position == Position(x=5.0, y=-1.0)

    removed = world.despawn(obj.id)
    assert removed.id == obj.id
    assert obj.id not in world.objects
    assert world.body.holding is None

    assert len(spawn_events) == 1
    spawn_ev = spawn_events[0]
    assert isinstance(spawn_ev, ObjectSpawnedEvent)
    assert spawn_ev.type == "object_spawned"
    assert spawn_ev.source == "creator"
    assert spawn_ev.object_id == "object_1"
    assert spawn_ev.position == Position(x=1.0, y=2.0)
    assert spawn_ev.object["id"] == "object_1"

    assert len(move_events) == 1
    move_ev = move_events[0]
    assert isinstance(move_ev, ObjectMovedEvent)
    assert move_ev.type == "object_moved"
    assert move_ev.source == "creator"
    assert move_ev.object_id == "object_1"
    assert move_ev.previous_position == Position(x=1.0, y=2.0)
    assert move_ev.position == Position(x=5.0, y=-1.0)

    assert len(despawn_events) == 1
    despawn_ev = despawn_events[0]
    assert isinstance(despawn_ev, ObjectDespawnedEvent)
    assert despawn_ev.type == "object_despawned"
    assert despawn_ev.source == "creator"
    assert despawn_ev.object_id == "object_1"
    assert despawn_ev.position == Position(x=5.0, y=-1.0)

    assert [e.type for e in all_events] == [
        "object_spawned",
        "object_moved",
        "object_despawned",
    ]


def test_event_bus_emits_object_changed_and_tick_and_supports_unsubscribe() -> None:
    world = World(seed=11)
    events: list[WorldEvent] = []
    unsub = world.subscribe(events.append)

    obj = world.spawn(_SAMPLE_PROPS, (0.0, 0.0))
    updated = world.update_object(obj.id, {"state": {"wet": True}, "color": "blue"})
    assert updated.state == {"wet": True}
    assert updated.color == "blue"

    world.step()

    assert len(events) == 3
    assert isinstance(events[1], ObjectChangedEvent)
    assert events[1].type == "object_changed"
    assert events[1].object_id == obj.id
    assert events[1].changes == {"state": {"wet": True}, "color": "blue"}
    assert events[1].object["color"] == "blue"

    assert isinstance(events[2], TickEvent)
    assert events[2].type == "tick"
    assert events[2].tick == 1
    assert events[2].source == "world"

    unsub()
    world.step()
    assert len(events) == 3


def test_despawn_and_move_reject_unknown_object_id() -> None:
    world = World(seed=0)
    with pytest.raises(KeyError, match="unknown object id"):
        world.despawn("object_99")
    with pytest.raises(KeyError, match="unknown object id"):
        world.move("object_99", {"x": 0.0, "y": 0.0})
    with pytest.raises(KeyError, match="unknown object id"):
        world.update_object("object_99", {"color": "green"})


def test_clock_runs_synchronously_with_pause_resume_speed_and_step_once() -> None:
    world = World(seed=3)
    clock = WorldClock(world, tick_rate=10.0)
    assert Clock is WorldClock
    assert clock.tick_interval == pytest.approx(0.1)

    # 1.0s at 10 Hz -> 10 ticks
    assert clock.advance(1.0) == 10
    assert world.tick == 10

    # Pause stops advance() and run(), but step_once() still steps 1 tick
    clock.pause()
    assert clock.paused is True
    assert clock.advance(5.0) == 0
    assert clock.run(10) == 0
    assert world.tick == 10

    assert clock.step_once() == 11
    assert world.tick == 11

    # Resume and change speed to 0.5x (5 ticks per second)
    clock.resume()
    clock.set_speed(0.5)
    assert clock.effective_tick_rate == pytest.approx(5.0)
    assert clock.tick_interval == pytest.approx(0.2)
    assert clock.advance(1.0) == 5
    assert world.tick == 16

    # Increase speed to 2.0x (20 ticks per second)
    clock.set_speed(2.0)
    assert clock.advance(0.5) == 10
    assert world.tick == 26


def test_clock_can_pause_from_tick_subscriber_while_llm_thinks() -> None:
    world = World(seed=5)
    clock = WorldClock(world, tick_rate=10.0)

    def pause_at_tick_3(event: WorldEvent) -> None:
        if event.tick == 3:
            clock.pause()

    world.subscribe("tick", pause_at_tick_3)
    executed = clock.run(10)
    assert executed == 3
    assert world.tick == 3
    assert clock.paused is True

    clock.resume()
    assert clock.run(4) == 4
    assert world.tick == 7


def test_clock_validates_parameters() -> None:
    world = World(seed=0)
    with pytest.raises(ValueError, match="tick_rate"):
        WorldClock(world, tick_rate=0)
    with pytest.raises(ValueError, match="speed"):
        WorldClock(world, tick_rate=10.0, speed=-1.0)

    clock = WorldClock(world, tick_rate=10.0)
    with pytest.raises(ValueError, match="speed"):
        clock.set_speed(0.0)
    with pytest.raises(ValueError, match="dt"):
        clock.advance(-0.1)


def _apply_scripted_inputs(world: World, tick: int) -> None:
    """Apply deterministic creator and simulation inputs at selected ticks."""
    if tick == 1:
        world.spawn(_SAMPLE_PROPS, {"x": 1.0, "y": 1.0})
        world.spawn(
            {**_SAMPLE_PROPS, "shape": "cube", "color": "green", "weight": 2.5},
            {"x": -2.0, "y": 3.5},
        )
    elif tick == 50:
        world.move("object_1", {"x": 4.25, "y": -0.5})
        world.body.position = Position(x=0.5, y=0.5)
        world.body.energy = 0.85
    elif tick == 125:
        world.update_object("object_2", {"state": {"wet": True, "roll": world.rng.random()}})
        world.spawn({**_SAMPLE_PROPS, "shape": "long", "color": None}, {"x": 0.0, "y": 0.0})
    elif tick == 200:
        world.despawn("object_1")
    elif tick == 300:
        dx = world.rng.uniform(-1.0, 1.0)
        dy = world.rng.uniform(-1.0, 1.0)
        world.move("object_3", {"x": dx, "y": dy})
        world.body.comfort = 0.75
        world.body.holding = "object_3"
    elif tick == 420:
        world.spawn(_SAMPLE_PROPS, {"x": world.rng.random(), "y": world.rng.random()})


def test_determinism_two_worlds_same_seed_and_inputs_match_after_500_ticks() -> None:
    world_a = World(seed=12345)
    world_b = World(seed=12345)
    clock_a = WorldClock(world_a, tick_rate=10.0)
    clock_b = WorldClock(world_b, tick_rate=10.0)

    for _ in range(500):
        clock_a.step_once()
        _apply_scripted_inputs(world_a, world_a.tick)

        clock_b.step_once()
        _apply_scripted_inputs(world_b, world_b.tick)

    assert world_a.tick == 500
    assert world_b.tick == 500
    snap_a = world_a.snapshot()
    snap_b = world_b.snapshot()
    assert snap_a == snap_b


def test_snapshot_and_restore_then_stepping_matches_never_saving() -> None:
    continuous = World(seed=98765)
    clock_continuous = WorldClock(continuous, tick_rate=10.0)

    for _ in range(250):
        clock_continuous.step_once()
        _apply_scripted_inputs(continuous, continuous.tick)

    saved_json = continuous.snapshot()
    parsed = json.loads(saved_json)
    assert parsed["tick"] == 250
    assert "rng_state" in parsed
    assert "objects" in parsed
    assert "body" in parsed

    # Restore via both instance method (mutating a differently-seeded world)
    # and class method, then step all three worlds for 250 more ticks.
    restored_instance = World(seed=1)
    restored_instance.restore(saved_json)
    restored_cls = World.restore(saved_json)

    for _ in range(250):
        clock_continuous.step_once()
        _apply_scripted_inputs(continuous, continuous.tick)

        restored_instance.step()
        _apply_scripted_inputs(restored_instance, restored_instance.tick)

        restored_cls.step()
        _apply_scripted_inputs(restored_cls, restored_cls.tick)

    assert continuous.tick == 500
    assert restored_instance.tick == 500
    assert restored_cls.tick == 500
    assert restored_instance.snapshot() == continuous.snapshot()
    assert restored_cls.snapshot() == continuous.snapshot()
    assert restored_instance.rng.random() == continuous.rng.random()
