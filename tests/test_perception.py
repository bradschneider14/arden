"""Tests for the perception feed, visibility factor, delta tracker, look focus, and golden text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arden.config import PerceptionSettings, Settings
from arden.perception.feed import (
    Percept,
    PerceptionTracker,
    compass_bearing,
    look,
    perceive,
)
from arden.world import AgentBody, Position, World

_GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "percept_golden.txt"

_BASE_OBJ: dict[str, Any] = {
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
}


def test_perceive_includes_nearby_objects_events_body_state_and_affordances() -> None:
    world = World(
        seed=42,
        body={"position": {"x": 0.0, "y": 0.0}, "energy": 0.65, "comfort": 0.25},
    )
    ball = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_7",
            "behavior": [
                {"verb": "push", "scale": 4, "note": "secret_ball_rule"},
                {"verb": "strike", "response": "bounce"},
            ],
        },
        {"x": 0.0, "y": 1.0},
    )
    mailbox = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_9",
            "shape": "cube",
            "height": 0.9,
            "volume": 0.12,
            "weight": 8.0,
            "rigidity": 0.95,
            "color": None,
            "fixed": True,
            "state": {},
            "behavior": [
                {
                    "verb": "action_3",
                    "description": "Send message to creator",
                    "internal_name": "mailbox",
                }
            ],
        },
        {"x": 1.2, "y": 0.0},
    )
    distant_stone = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_4",
            "shape": "flat",
            "height": 0.1,
            "volume": 0.02,
            "weight": 3.5,
            "rigidity": 1.0,
            "color": "gray",
            "state": {},
        },
        {"x": 3.0, "y": 3.0},
    )

    initial = perceive(world, world.body)
    assert [o.id for o in initial.objects] == ["object_7", "object_9", "object_4"]
    assert initial.events == []

    # Trigger an event since the last percept
    world.move(ball.id, {"x": 0.0, "y": 1.2})
    percept = perceive(world, world.body)

    # Nearby objects have perceptible properties, distance, and compass direction
    by_id = {o.id: o for o in percept.objects}
    assert set(by_id) == {"object_7", "object_9", "object_4"}
    assert by_id["object_7"].distance == pytest.approx(1.2)
    assert by_id["object_7"].direction == "N"
    assert by_id["object_7"].shape == "sphere"
    assert by_id["object_7"].color == "red"

    assert by_id["object_9"].distance == pytest.approx(1.2)
    assert by_id["object_9"].direction == "E"

    assert by_id["object_4"].distance == pytest.approx((18.0) ** 0.5)
    assert by_id["object_4"].direction == "NE"

    # Body state has plain numbers and no good/bad labels
    assert percept.body.position == Position(x=0.0, y=0.0)
    assert percept.body.energy == 0.65
    assert percept.body.comfort == 0.25
    assert percept.body.holding is None

    # Affordances only for objects within reach (<= 1.5m), opaque as action_<n>
    assert "object_7" in percept.affordances
    assert "object_9" in percept.affordances
    assert distant_stone.id not in percept.affordances
    assert percept.affordances["object_7"] == ["grab", "push", "strike"]
    assert percept.affordances[mailbox.id] == ["action_3"]
    assert "action_3" in percept.affordances

    # Events since the last percept
    assert len(percept.events) == 1
    assert percept.events[0].type == "object_moved"
    assert percept.events[0].object_id == "object_7"
    assert percept.events[0].previous_position == Position(x=0.0, y=1.0)
    assert percept.events[0].position == Position(x=0.0, y=1.2)


def test_no_hidden_rules_human_names_or_evaluative_labels_ever_appear() -> None:
    world = World(
        seed=7,
        body={"position": {"x": 0.0, "y": 0.0}, "energy": 0.15, "comfort": 0.1},
    )
    world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_7",
            "behavior": [
                {"verb": "push", "scale": 4, "human_name": "red_ball", "response": "roll_fast"},
                {"verb": "strike", "response": "bounce", "above_force": 0.7},
                {"verb": "send_creator_mail", "description": "secret_mailbox_explanation"},
            ],
        },
        {"x": 0.5, "y": 0.0},
    )
    world.update_object("object_7", {"state": {"wet": True}})

    percept = perceive(world, world.body)
    text_out = percept.to_text()
    json_out = percept.to_json()

    # Non-body verb in behavior is anonymized to action_1 with no description
    assert "action_1" in percept.affordances["object_7"]

    forbidden_words = (
        "behavior",
        "scale",
        "human_name",
        "red_ball",
        "roll_fast",
        "bounce",
        "above_force",
        "send_creator_mail",
        "secret_mailbox_explanation",
        "description",
        "creator",
        "good",
        "bad",
        "low",
        "high",
        "tired",
        "cold",
        "hungry",
    )
    import re

    text_words = set(re.findall(r"[a-z0-9_]+", text_out.lower()))
    json_words = set(re.findall(r"[a-z0-9_]+", json_out.lower()))
    for word in forbidden_words:
        assert word not in text_words
        assert word not in json_words


def test_sensing_radius_from_config_and_occluded_objects_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "arden.toml"
    config_file.write_text("[perception]\nsensing_radius = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    world = World(seed=1)
    near_obj = world.spawn({**_BASE_OBJ, "id": "object_1"}, {"x": 2.0, "y": 0.0})
    outside_config_radius = world.spawn({**_BASE_OBJ, "id": "object_2"}, {"x": 6.0, "y": 0.0})
    tall_blocker = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_3",
            "shape": "cylinder",
            "height": 2.5,
            "volume": 0.5,
            "weight": 20.0,
            "fixed": True,
        },
        {"x": 0.0, "y": 2.0},
    )
    occluded_behind_tree = world.spawn(
        {**_BASE_OBJ, "id": "object_4", "height": 0.2},
        {"x": 0.0, "y": 4.0},
    )

    assert world.is_occluded(world.body, occluded_behind_tree) is True

    percept = perceive(world, world.body)
    seen_ids = [o.id for o in percept.objects]
    assert near_obj.id in seen_ids
    assert tall_blocker.id in seen_ids
    assert outside_config_radius.id not in seen_ids
    assert occluded_behind_tree.id not in seen_ids

    # Passing a larger config sensing_radius reveals object_2 (at 6.0m) but still omits occluded object_4
    wide_settings = Settings(perception=PerceptionSettings(sensing_radius=8.0))
    wide_percept = perceive(world, world.body, config=wide_settings)
    wide_ids = [o.id for o in wide_percept.objects]
    assert outside_config_radius.id in wide_ids
    assert occluded_behind_tree.id not in wide_ids


def test_lowering_visibility_factor_removes_far_objects_and_color_first() -> None:
    world = World(seed=3)
    near = world.spawn(
        {**_BASE_OBJ, "id": "object_1", "color": "crimson", "shape": "sphere", "height": 0.3},
        {"x": 2.0, "y": 0.0},
    )
    mid = world.spawn(
        {**_BASE_OBJ, "id": "object_2", "color": "cobalt", "shape": "cube", "height": 0.5},
        {"x": 6.5, "y": 0.0},
    )
    far = world.spawn(
        {**_BASE_OBJ, "id": "object_3", "color": "emerald", "shape": "cylinder", "height": 0.8},
        {"x": 9.0, "y": 0.0},
    )

    # Full visibility (1.0): all three objects and their colors are perceived
    p_full = perceive(world, world.body, sensing_radius=10.0, visibility_factor=1.0)
    full_by_id = {o.id: o for o in p_full.objects}
    assert set(full_by_id) == {near.id, mid.id, far.id}
    assert full_by_id[near.id].color == "crimson"
    assert full_by_id[mid.id].color == "cobalt"
    assert full_by_id[far.id].color == "emerald"

    # Slightly lowered visibility (0.8 -> radius 8.0, detail radius 6.4):
    # - far object (9.0m) is removed
    # - mid object (6.5m) is still visible with shape/height, but loses color first
    # - near object (2.0m) keeps color
    p_dusk = perceive(world, world.body, sensing_radius=10.0, visibility_factor=0.8)
    dusk_by_id = {o.id: o for o in p_dusk.objects}
    assert set(dusk_by_id) == {near.id, mid.id}
    assert dusk_by_id[near.id].color == "crimson"
    assert dusk_by_id[mid.id].color is None
    assert "color" not in dusk_by_id[mid.id].properties
    assert dusk_by_id[mid.id].shape == "cube"
    assert dusk_by_id[mid.id].height == 0.5

    # Low visibility (0.5 -> radius 5.0):
    # - mid and far objects are removed
    # - near object (2.0m) remains visible with coarse properties (shape, height, volume)
    #   while its color is dropped
    p_night = perceive(world, world.body, sensing_radius=10.0, visibility_factor=0.5)
    night_by_id = {o.id: o for o in p_night.objects}
    assert set(night_by_id) == {near.id}
    assert night_by_id[near.id].color is None
    assert "color" not in night_by_id[near.id].properties
    assert "crimson" not in p_night.to_text()
    assert "color" not in p_night.to_json()
    assert night_by_id[near.id].shape == "sphere"
    assert night_by_id[near.id].height == 0.3


def test_delta_mode_perception_tracker_and_full_refresh() -> None:
    world = World(seed=9)
    obj1 = world.spawn({**_BASE_OBJ, "id": "object_1", "color": "red"}, {"x": 2.0, "y": 0.0})
    obj2 = world.spawn({**_BASE_OBJ, "id": "object_2", "color": "blue"}, {"x": 0.0, "y": 3.0})

    tracker = PerceptionTracker(world, sensing_radius=10.0, move_threshold=0.2)

    # First call initializes the baseline and returns the full view
    first = tracker.perceive()
    assert first.is_delta is False
    assert [o.id for o in first.objects] == ["object_1", "object_2"]

    # Second call with no changes in the world: delta is completely empty
    empty_delta = tracker.perceive()
    assert empty_delta.is_delta is True
    assert empty_delta.is_empty is True
    assert not empty_delta
    assert empty_delta.objects == []
    assert empty_delta.events == []
    assert empty_delta.appeared == []
    assert empty_delta.disappeared == []
    assert empty_delta.moved == []
    assert empty_delta.changed == []
    assert empty_delta.to_text() == ""

    # Movement smaller than move_threshold (0.05 <= 0.2) is ignored in delta mode
    world.move(obj1.id, {"x": 2.05, "y": 0.0})
    sub_threshold = tracker.perceive()
    assert sub_threshold.is_empty is True
    assert sub_threshold.objects == []
    assert sub_threshold.moved == []

    # Movement beyond move_threshold (from 2.0 to 2.5 -> 0.5 > 0.2) is reported
    world.move(obj1.id, {"x": 2.5, "y": 0.0})
    move_delta = tracker.perceive()
    assert move_delta.is_empty is False
    assert [o.id for o in move_delta.moved] == ["object_1"]
    assert [o.id for o in move_delta.objects] == ["object_1"]
    assert any(ev.type == "moved" and ev.object_id == "object_1" for ev in move_delta.events)

    # Property change, new object spawn (appeared), and despawn (disappeared)
    world.update_object(obj1.id, {"state": {"wet": True}})
    obj3 = world.spawn({**_BASE_OBJ, "id": "object_3", "color": "yellow"}, {"x": 1.0, "y": 1.0})
    world.despawn(obj2.id)

    mixed_delta = tracker.perceive()
    assert [o.id for o in mixed_delta.appeared] == [obj3.id]
    assert mixed_delta.disappeared == [obj2.id]
    assert [o.id for o in mixed_delta.changed] == [obj1.id]

    # Full refresh on demand returns all currently visible objects even when nothing changed since mixed_delta
    assert tracker.perceive().is_empty is True
    refreshed = tracker.full_refresh()
    assert refreshed.is_delta is False
    assert [o.id for o in refreshed.objects] == ["object_3", "object_1"]
    assert tracker.perceive(full=True).objects == refreshed.objects


def test_look_returns_detailed_percept_for_one_object() -> None:
    world = World(seed=12, body={"position": {"x": 0.0, "y": 0.0}})
    target = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_7",
            "shape": "cylinder",
            "height": 0.4,
            "volume": 0.01,
            "weight": 1.2,
            "rigidity": 0.8,
            "binding": False,
            "hollow": True,
            "fragile": True,
            "color": "amber",
            "fixed": False,
            "state": {"wet": False, "contents": []},
            "behavior": [{"verb": "push", "scale": 2, "secret": "never_shown"}],
        },
        {"x": 0.0, "y": -2.0},
    )

    detail_by_id = look("object_7")
    detail_by_world = look(world, target.id)
    detail_by_obj = look(target)

    assert detail_by_id == detail_by_world == detail_by_obj
    assert detail_by_id.id == "object_7"
    assert detail_by_id.distance == pytest.approx(2.0)
    assert detail_by_id.direction == "S"
    assert detail_by_id.shape == "cylinder"
    assert detail_by_id.height == 0.4
    assert detail_by_id.volume == 0.01
    assert detail_by_id.weight == 1.2
    assert detail_by_id.rigidity == 0.8
    assert detail_by_id.hollow is True
    assert detail_by_id.fragile is True
    assert detail_by_id.color == "amber"
    assert detail_by_id.state == {"wet": False}
    assert "contents" not in detail_by_id.state
    assert detail_by_id.properties == target.perceptible()
    assert "behavior" not in detail_by_id
    assert "secret" not in detail_by_id.to_json()

    with pytest.raises(KeyError, match="unknown object id"):
        look(world, "object_404")


def test_compass_bearings_cover_all_eight_directions_and_here() -> None:
    origin = Position(x=0.0, y=0.0)
    assert compass_bearing(origin, Position(x=0.0, y=0.0)) == "here"
    assert compass_bearing(origin, Position(x=0.0, y=2.0)) == "N"
    assert compass_bearing(origin, Position(x=2.0, y=2.0)) == "NE"
    assert compass_bearing(origin, Position(x=2.0, y=0.0)) == "E"
    assert compass_bearing(origin, Position(x=2.0, y=-2.0)) == "SE"
    assert compass_bearing(origin, Position(x=0.0, y=-2.0)) == "S"
    assert compass_bearing(origin, Position(x=-2.0, y=-2.0)) == "SW"
    assert compass_bearing(origin, Position(x=-2.0, y=0.0)) == "W"
    assert compass_bearing(origin, Position(x=-2.0, y=2.0)) == "NW"


def test_percept_to_text_matches_golden_file() -> None:
    world = World(
        seed=10,
        body={
            "position": {"x": 0.0, "y": 0.0},
            "level": "ground",
            "energy": 0.75,
            "comfort": 0.3,
            "holding": "object_2",
        },
    )
    world.tick = 1

    # Held stick at (0, 0)
    world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_2",
            "shape": "long",
            "height": 0.8,
            "volume": 0.002,
            "weight": 0.4,
            "rigidity": 0.9,
            "color": "brown",
            "state": {},
        },
        {"x": 0.0, "y": 0.0},
    )
    # Red sphere at (0, 0.5), then moved to (0, 1.0)
    world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_7",
            "shape": "sphere",
            "height": 0.2,
            "volume": 0.004,
            "weight": 0.3,
            "rigidity": 0.4,
            "color": "red",
            "state": {"wet": False},
            "behavior": [{"verb": "push", "scale": 4}, {"verb": "strike", "response": "bounce"}],
        },
        {"x": 0.0, "y": 0.5},
    )
    # Fixed mailbox at (1.2, 0) with opaque action_3
    world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_9",
            "shape": "cube",
            "height": 0.9,
            "volume": 0.12,
            "weight": 8.0,
            "rigidity": 0.95,
            "color": None,
            "fixed": True,
            "state": {},
            "behavior": [{"verb": "action_3", "note": "mailbox"}],
        },
        {"x": 1.2, "y": 0.0},
    )
    # Distant flat stone at SW, distance 4.0
    d = -(8.0**0.5)
    world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_4",
            "shape": "flat",
            "height": 0.1,
            "volume": 0.02,
            "weight": 3.5,
            "rigidity": 1.0,
            "color": "gray",
            "state": {},
        },
        {"x": d, "y": d},
    )

    world.move("object_7", {"x": 0.0, "y": 1.0})

    percept = perceive(world, world.body)
    expected = _GOLDEN_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    assert percept.to_text() == expected

    # Verify JSON serialization is also valid and deterministic
    parsed = json.loads(percept.to_json())
    assert parsed["tick"] == 1
    assert parsed["body"]["holding"] == "object_2"
    assert [o["id"] for o in parsed["objects"]] == ["object_2", "object_7", "object_9", "object_4"]


def test_perception_filters_unallowed_state_keys_from_objects_and_events() -> None:
    world = World(seed=21, body={"position": {"x": 0.0, "y": 0.0}})
    world.set_tile(0, 1, "water", flow=(1.0, 0.0))
    leaf = world.spawn(
        {
            **_BASE_OBJ,
            "id": "object_1",
            "shape": "flat",
            "weight": 0.1,
            "state": {"wet": False, "radius": 0.35, "secret_label": "maple_leaf"},
        },
        {"x": 0.5, "y": 1.5},
    )

    # Step physics so water sets state["wet"] = True and state["floating"] = True
    world.step()
    assert world.objects[leaf.id].state["wet"] is True
    assert world.objects[leaf.id].state["floating"] is True
    assert world.objects[leaf.id].state["radius"] == 0.35

    percept = perceive(world, world.body)
    obj_view = percept.objects[0]
    assert obj_view.state == {"wet": True}
    assert obj_view.properties["state"] == {"wet": True}
    assert "floating" not in percept.to_text()
    assert "floating" not in percept.to_json()
    assert "radius" not in percept.to_text()
    assert "radius" not in percept.to_json()
    assert "secret_label" not in percept.to_text()
    assert "maple_leaf" not in percept.to_json()

    # Updating only unallowed state keys produces no perceptible state event
    world.update_object(leaf.id, {"state": {**world.objects[leaf.id].state, "radius": 0.5}})
    percept_after_hidden = perceive(world, world.body)
    assert percept_after_hidden.objects[0].state == {"wet": True}
    for ev in percept_after_hidden.events:
        if ev.changes and "state" in ev.changes:
            assert "radius" not in ev.changes["state"]
            assert "floating" not in ev.changes["state"]
