"""Unit tests for body-level verbs, parameter validation, body modifiers, and opaque actions."""

from __future__ import annotations

from typing import Any

import pytest

from arden.actions.verbs import (
    HANDS_FULL,
    INVALID_PARAMS,
    NOTHING_HAPPENED,
    OUT_OF_REACH,
    TOO_HEAVY,
    ActionResult,
    BodyModifiers,
    available_opaque_actions,
    drop,
    execute_action,
    grab,
    invoke_opaque_action,
    look,
    move,
    place,
    push,
    register_opaque_action,
    speak,
    strike,
)
from arden.world import (
    ObjectChangedEvent,
    ObjectDespawnedEvent,
    ObjectMovedEvent,
    ObjectSpawnedEvent,
    Position,
    SpokeEvent,
    TerrainGrid,
    World,
    WorldEvent,
)

_BASE_PROPS: dict[str, Any] = {
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
}


def test_move_verb_validates_speed_applies_modifiers_and_checks_ledge_reachability() -> None:
    terrain = TerrainGrid()
    terrain.set_tile(2, 0, "ledge")
    world = World(seed=1, terrain=terrain, body={"position": {"x": 0.0, "y": 0.0}})

    # Move east at speed 0.5
    res = move(world, "east", 0.5)
    assert isinstance(res, ActionResult)
    assert res.ok is True
    assert res.failed is False
    assert res.reason is None
    assert world.body.position == Position(x=0.5, y=0.0)

    # Active-world calling convention move(direction, speed) with BodyModifiers(speed=0.5)
    res_mod = move("north", 0.8, modifiers=BodyModifiers(speed=0.5))
    assert res_mod.ok is True
    assert world.body.position == Position(x=0.5, y=0.4)

    # Speed outside 0..1 fails parameter validation
    assert move(world, "east", -0.1).ok is False
    assert move(world, "east", 1.2).ok is False
    assert move(world, "east", 0.0).reason == NOTHING_HAPPENED

    # Trying to move from ground onto a raised ledge tile (cell 2, 0 -> x in 2..3) fails with out_of_reach
    world.body.position = Position(x=1.8, y=0.5)
    res_ledge = move(world, "east", 0.8)
    assert res_ledge.ok is False
    assert res_ledge.failed is True
    assert res_ledge.reason == OUT_OF_REACH
    assert world.body.position == Position(x=1.8, y=0.5)


def test_look_verb_returns_perceptible_properties_and_fails_when_out_of_reach() -> None:
    world = World(seed=2)
    near = world.spawn(
        {
            **_BASE_PROPS,
            "color": "blue",
            "behavior": [{"verb": "push", "scale": 4}, {"verb": "look", "state": {"inspected": True}}],
        },
        {"x": 1.0, "y": 0.0},
    )
    far = world.spawn(_BASE_PROPS, {"x": 5.0, "y": 0.0})

    res_near = look(world, near.id)
    assert res_near.ok is True
    assert res_near.reason is None
    assert res_near.data["id"] == near.id
    assert res_near.data["color"] == "blue"
    assert world.objects[near.id].state["inspected"] is True
    assert "inspected" not in res_near.data["state"]
    assert "behavior" not in res_near.data
    assert any(isinstance(e, ObjectChangedEvent) for e in res_near.events)

    # Reach failure on distant object
    res_far = look(world, far.id)
    assert res_far.ok is False
    assert res_far.reason == OUT_OF_REACH


def test_push_verb_uses_apply_impulse_and_respects_force_validation_reach_and_modifiers() -> None:
    world = World(seed=3)
    ball = world.spawn(
        {**_BASE_PROPS, "shape": "sphere", "weight": 0.3, "behavior": [{"verb": "push", "scale": 8}]},
        {"x": 1.0, "y": 0.0},
    )
    far_ball = world.spawn(_BASE_PROPS, {"x": 6.0, "y": 0.0})
    fixed_tree = world.spawn(
        {**_BASE_PROPS, "shape": "cylinder", "weight": 20.0, "fixed": True},
        {"x": 0.0, "y": 1.0},
    )

    # Parameter validation: force outside 0..1 fails
    assert push(world, ball.id, -0.2).ok is False
    assert push(world, ball.id, 1.5).ok is False
    assert push(world, ball.id, 0.0).reason == NOTHING_HAPPENED

    # Out of reach fails with out_of_reach
    res_far = push(world, far_ball.id, 0.5)
    assert res_far.ok is False
    assert res_far.reason == OUT_OF_REACH

    # Heavy fixed tree fails with too_heavy
    res_tree = push(world, fixed_tree.id, 0.8)
    assert res_tree.ok is False
    assert res_tree.reason == TOO_HEAVY

    # Push ball in reach: sets velocity via apply_impulse and emits ObjectChangedEvent
    res_push = push(world, ball.id, 0.5)
    assert res_push.ok is True
    assert res_push.reason is None
    assert len(res_push.events) == 1
    assert isinstance(res_push.events[0], ObjectChangedEvent)
    v_normal = world.objects[ball.id].velocity.x
    assert v_normal > 0.0

    world.step()
    assert world.objects[ball.id].position.x > 1.0

    # BodyModifiers(force=0.5) halves the impulse compared to neutral modifiers
    ball_weaker = world.spawn(
        {**_BASE_PROPS, "shape": "sphere", "weight": 0.3, "behavior": [{"verb": "push", "scale": 8}]},
        {"x": 0.0, "y": -1.0},
    )
    res_weak = push(world, ball_weaker.id, 0.5, modifiers=BodyModifiers(force=0.5))
    assert res_weak.ok is True
    assert abs(world.objects[ball_weaker.id].velocity.y) == pytest.approx(v_normal * 0.5)


def test_strike_verb_breaks_fragile_object_above_threshold_and_bounces_below() -> None:
    world = World(seed=4)
    fragile_jar = world.spawn(
        {
            **_BASE_PROPS,
            "shape": "cylinder",
            "height": 0.3,
            "volume": 0.02,
            "weight": 0.6,
            "fragile": True,
            "behavior": [
                {"verb": "strike", "response": "bounce"},
                {"verb": "strike", "response": "break", "above_force": 0.6},
            ],
        },
        {"x": 1.0, "y": 0.0},
    )

    # Strike below threshold (0.4 <= 0.6): bounces, does not break
    res_gentle = strike(world, fragile_jar.id, 0.4)
    assert res_gentle.ok is True
    assert res_gentle.data["broken"] is False
    assert fragile_jar.id in world.objects
    assert world.objects[fragile_jar.id].state.get("bounced") is True
    assert world.objects[fragile_jar.id].velocity.x > 0.0

    # Reset position within reach and strike above threshold (0.8 > 0.6): breaks into smaller pieces
    world.move(fragile_jar.id, {"x": 1.0, "y": 0.0})
    res_hard = strike(world, fragile_jar.id, 0.8)
    assert res_hard.ok is True
    assert res_hard.data["broken"] is True
    assert fragile_jar.id not in world.objects
    assert any(isinstance(e, ObjectDespawnedEvent) and e.object_id == fragile_jar.id for e in res_hard.events)
    assert any(isinstance(e, ObjectSpawnedEvent) for e in res_hard.events)

    piece_ids = res_hard.data["piece_ids"]
    assert len(piece_ids) == 2
    for pid in piece_ids:
        piece = world.objects[pid]
        assert piece.weight < fragile_jar.weight
        assert piece.volume < fragile_jar.volume
        assert piece.fragile is False

    # Fragile object with no explicit BehaviorRule also breaks when struck hard, or removes when pieces=0
    berry = world.spawn(
        {
            **_BASE_PROPS,
            "fragile": True,
            "state": {"pieces": 0},
        },
        {"x": 0.5, "y": 0.5},
    )
    res_berry = strike(world, berry.id, 0.9)
    assert res_berry.ok is True
    assert berry.id not in world.objects
    assert res_berry.data["piece_ids"] == []

    # Out-of-reach strike fails with out_of_reach
    distant = world.spawn(_BASE_PROPS, {"x": 8.0, "y": 0.0})
    res_distant = strike(world, distant.id, 0.5)
    assert res_distant.ok is False
    assert res_distant.reason == OUT_OF_REACH


def test_grab_drop_and_place_verbs_enforce_single_held_item_weight_reach_and_grab_fail_chance() -> None:
    world = World(seed=5)
    item_a = world.spawn(_BASE_PROPS, {"x": 0.8, "y": 0.0})
    item_b = world.spawn(_BASE_PROPS, {"x": 0.0, "y": 0.8})
    heavy_boulder = world.spawn({**_BASE_PROPS, "weight": 15.0}, {"x": -0.8, "y": 0.0})
    distant_item = world.spawn(_BASE_PROPS, {"x": 5.0, "y": 5.0})

    # Dropping or placing when holding nothing fails with nothing_happened
    assert drop(world).ok is False
    assert drop(world).reason == NOTHING_HAPPENED
    assert place(world, (0.5, 0.0)).ok is False
    assert place(world, (0.5, 0.0)).reason == NOTHING_HAPPENED

    # Grabbing out-of-reach object fails with out_of_reach
    res_far = grab(world, distant_item.id)
    assert res_far.ok is False
    assert res_far.reason == OUT_OF_REACH

    # Grabbing heavy object fails with too_heavy
    res_heavy = grab(world, heavy_boulder.id)
    assert res_heavy.ok is False
    assert res_heavy.reason == TOO_HEAVY

    # Grabbing when grab_fail_chance=1.0 fails and leaves hands empty
    res_fumble = grab(world, item_a.id, modifiers=BodyModifiers(grab_fail_chance=1.0))
    assert res_fumble.ok is False
    assert res_fumble.failed is True
    assert res_fumble.reason == NOTHING_HAPPENED
    assert world.body.holding is None

    # Normal grab succeeds
    res_grab = grab(world, item_a.id)
    assert res_grab.ok is True
    assert res_grab.reason is None
    assert world.body.holding == item_a.id
    assert len(res_grab.events) >= 1

    # Grabbing a second item while holding item_a fails with hands_full
    res_full = grab(world, item_b.id)
    assert res_full.ok is False
    assert res_full.reason == HANDS_FULL
    assert world.body.holding == item_a.id

    # Placing at an out-of-reach location fails with out_of_reach and keeps item held
    res_place_far = place(world, (10.0, 0.0))
    assert res_place_far.ok is False
    assert res_place_far.reason == OUT_OF_REACH
    assert world.body.holding == item_a.id

    # Placing at an in-reach location succeeds and clears holding
    res_place = place(world, (1.0, 0.5))
    assert res_place.ok is True
    assert world.body.holding is None
    assert world.objects[item_a.id].position == Position(x=1.0, y=0.5)
    assert any(isinstance(e, ObjectMovedEvent) for e in res_place.events)

    # Grab item_b, move agent, then drop() at agent's new position
    assert grab(world, item_b.id).ok is True
    assert move(world, "west", 0.5).ok is True
    res_drop = drop(world)
    assert res_drop.ok is True
    assert world.body.holding is None
    assert world.objects[item_b.id].position == world.body.position


def test_speak_verb_emits_spoke_event_with_no_listener_required() -> None:
    world = World(seed=6, body={"position": {"x": 2.0, "y": -1.0}})

    # Nobody subscribed to world.events: speak still succeeds and returns SpokeEvent in res.events
    res = speak(world, "hello forest")
    assert res.ok is True
    assert res.reason is None
    assert len(res.events) == 1
    ev = res.events[0]
    assert isinstance(ev, SpokeEvent)
    assert ev.type == "spoke"
    assert ev.text == "hello forest"
    assert ev.position == Position(x=2.0, y=-1.0)

    # Subscriber on "spoke" also receives the event
    received: list[WorldEvent] = []
    world.subscribe("spoke", received.append)
    res2 = speak("is anyone there?")
    assert res2.ok is True
    assert len(received) == 1
    assert received[0].text == "is anyone there?"


def test_opaque_actions_are_offered_and_invoked_only_within_reach() -> None:
    world = World(seed=7, body={"position": {"x": 0.0, "y": 0.0}})
    mailbox = world.spawn(
        {**_BASE_PROPS, "shape": "cube", "fixed": True},
        {"x": 4.0, "y": 0.0},
    )

    messages: list[str] = []

    def handle_mail(w: World, obj: Any, text: str = "ping") -> dict[str, Any]:
        messages.append(f"{obj.id}:{text}")
        w.update_object(obj.id, {"state": {"last_message": text}}, source="agent")
        return {"sent": text}

    action_name = register_opaque_action(world, mailbox.id, "action_3", handle_mail)
    assert action_name == "action_3"

    # Mailbox is at (4.0, 0.0), out of reach from (0.0, 0.0)
    assert available_opaque_actions(world) == []
    res_far = invoke_opaque_action(world, "action_3", text="hello creator")
    assert res_far.ok is False
    assert res_far.reason == OUT_OF_REACH
    assert messages == []

    # Move agent within reach of the mailbox
    for _ in range(3):
        assert move(world, "east", 1.0).ok is True
    assert world.body.position == Position(x=3.0, y=0.0)

    # Now action_3 is offered and can be executed
    assert available_opaque_actions(world) == ["action_3"]
    res_near = execute_action(world, "action_3", text="hello creator")
    assert res_near.ok is True
    assert res_near.reason is None
    assert messages == [f"{mailbox.id}:hello creator"]
    assert world.objects[mailbox.id].state["last_message"] == "hello creator"
    assert len(res_near.events) == 1
    assert isinstance(res_near.events[0], ObjectChangedEvent)

    # Invalid opaque action name is rejected
    with pytest.raises(ValueError, match="action_<n>"):
        register_opaque_action(world, mailbox.id, "open_mailbox", handle_mail)
