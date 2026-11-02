"""Tests for Ticket #20: first-run clearing world definition, loader, and 2,000-tick mock run."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from arden.__main__ import main
from arden.actions.verbs import BASE_MOVE_STEP, OUT_OF_REACH, grab, in_reach, look, move, stack
from arden.config import Settings, make_rng
from arden.llm.mock import MockLLM, derive_llm_seed
from arden.llm.prompts import build_decide_prompt
from arden.perception.feed import PerceptionTracker
from arden.world import (
    DEFAULT_WORLD_PATH,
    FIRST_RUN_WORLD_PATH,
    REQUIRED_MATERIAL_PROPERTIES,
    World,
    WorldObject,
    check_property_coverage,
    load_first_run_world,
    load_world,
    load_world_data,
    property_coverage,
    run_world_ticks,
)


def test_first_run_toml_loads_and_defines_terrain_and_twelve_types() -> None:
    world = load_world(DEFAULT_WORLD_PATH)
    assert isinstance(world, World)
    assert world.tick == 0
    assert world.terrain.width == 16
    assert world.terrain.height == 16
    assert world.terrain.default_type == "grass"

    # Grass in the main clearing
    assert world.terrain.tile_type_at(5.5, 7.5) == "grass"

    # Stream along the eastern edge with a flow direction
    stream_tile = world.terrain.get_tile(11.5, 7.5)
    assert stream_tile.type == "water"
    assert stream_tile.wet is True
    assert stream_tile.flow.speed > 0.0
    assert stream_tile.flow.y > 0.0

    # Raised rock ledge in the northwest corner
    ledge_tile = world.terrain.get_tile(2.5, 13.0)
    assert ledge_tile.type == "ledge"
    assert ledge_tile.level == "raised"

    objs = list(world.objects.values())
    assert len(objs) >= 12
    assert all(type(o) is WorldObject for o in objs)

    # Large trees (tall, fixed, shade)
    trees = [
        o for o in objs
        if o.fixed and o.height >= 3.0 and o.state.get("shade") is True
    ]
    assert len(trees) >= 2
    assert world.is_sheltered(trees[0].position)

    # Sticks (long, rigid, light/medium)
    sticks = [
        o for o in objs
        if o.shape == "long" and not o.binding and o.rigidity >= 0.7
    ]
    assert len(sticks) >= 2

    # Flat stones (flat, heavy, rigid)
    flat_stones = [
        o for o in objs
        if o.shape == "flat" and o.weight >= 2.0 and o.rigidity >= 0.8
    ]
    assert len(flat_stones) >= 3

    # Vines (binding)
    vines = [o for o in objs if o.binding]
    assert len(vines) >= 1

    # Leaves (very light)
    leaves = [o for o in objs if o.shape == "flat" and o.weight <= 0.05]
    assert len(leaves) >= 1

    # Ball (round sphere, light, push & bounce rules)
    balls = [
        o for o in objs
        if o.shape == "sphere"
        and not o.fixed
        and any(r.verb == "push" for r in o.behavior)
        and any(r.verb == "strike" for r in o.behavior)
    ]
    assert len(balls) >= 1

    # Hollow log (hollow, heavy, cylinder)
    hollow_logs = [
        o for o in objs
        if o.hollow and not o.fixed and o.weight >= 2.0 and o.shape == "cylinder"
    ]
    assert len(hollow_logs) >= 1

    # Berry bush and fragile edible berries
    edible_objects = [
        o for o in objs
        if any(r.verb == "edible" for r in o.behavior)
    ]
    assert len(edible_objects) >= 2
    assert any(o.fixed for o in edible_objects)  # berry bush
    assert any(o.fragile and not o.fixed for o in edible_objects)  # loose fragile berries

    # Mailbox placeholder at the edge of the path
    mailboxes = [
        o for o in objs
        if o.fixed and any(r.verb == "action_1" for r in o.behavior)
    ]
    assert len(mailboxes) == 1
    assert mailboxes[0].position.y <= 2.0


def test_property_coverage_passes_for_first_run_and_detects_missing() -> None:
    world = load_first_run_world()
    coverage = check_property_coverage(world)
    assert set(coverage.keys()) == set(REQUIRED_MATERIAL_PROPERTIES)
    assert all(coverage.values()), f"missing properties: {coverage}"

    # Removing all binding objects makes check_property_coverage fail
    for oid, obj in list(world.objects.items()):
        if obj.binding:
            world.despawn(oid)
    partial = property_coverage(world)
    assert partial["binding"] is False
    with pytest.raises(ValueError, match="binding"):
        check_property_coverage(world)


def test_creator_label_present_for_observer_and_never_leaks_into_perception() -> None:
    raw_data = load_world_data(FIRST_RUN_WORLD_PATH)
    for raw_obj in raw_data["objects"]:
        assert "name" not in raw_obj
        assert "label" not in raw_obj
        assert "creator_label" in raw_obj

    world = load_first_run_world()
    labels = world.creator_labels
    assert len(labels) == len(world.objects)
    label_values = set(labels.values())
    assert "ball" in label_values
    assert "hollow_log" in label_values
    assert "mailbox" in label_values

    # Observer event bus receives creator_label on object events
    emitted_events = []
    world.subscribe(emitted_events.append)
    world.move("object_7", (5.6, 8.4))
    assert any(e.object_id == "object_7" and e.creator_label == "ball" for e in emitted_events)

    # Check every object's perceptible() dict
    for obj in world.objects.values():
        p_dict = obj.perceptible()
        assert "creator_label" not in p_dict
        serialized = json.dumps(p_dict)
        for label in label_values:
            assert f'"{label}"' not in serialized

    # Check full-range percept (sensing_radius=50 to include every object in the world)
    tracker = PerceptionTracker(world, sensing_radius=50.0)
    percept = tracker.perceive()
    assert len(percept.objects) >= 10

    percept_dict_str = json.dumps(percept.to_dict())
    percept_json_str = percept.to_json()
    percept_text_str = percept.to_text()
    prompt = build_decide_prompt(
        percept=percept_text_str,
        self_model="none",
        memories=[],
        body_state=percept.body.to_dict(),
        previous_intent_result=None,
    )

    for payload in (percept_dict_str, percept_json_str, percept_text_str, prompt.system, prompt.user):
        assert "creator_label" not in payload
        for label in label_values:
            assert label not in payload

    # Trigger an object change & move within sensing range and verify delta events do not leak creator_label
    world.update_object("object_7", {"state": {"wet": True}})
    world.move("object_7", (5.8, 8.4))
    delta_percept = tracker.perceive()
    assert len(delta_percept.events) >= 1
    delta_json = delta_percept.to_json()
    delta_text = delta_percept.to_text()
    assert "creator_label" not in delta_json
    assert "creator_label" not in delta_text
    for label in label_values:
        assert label not in delta_json
        assert label not in delta_text

    # Also check the `look` verb on a nearby object
    look_res = look(world, "object_3")
    assert look_res.ok
    assert "creator_label" not in look_res.data
    assert "stick" not in json.dumps(look_res.data)

    # Move next to every object in the world and verify neither perceive() nor look() leaks creator_label
    for obj in world.objects.values():
        world.body.position = obj.position.model_copy()
        world.body.level = obj.level
        close_percept = tracker.perceive(full=True)
        assert any(po.id == obj.id for po in close_percept.objects)
        for text_form in (
            json.dumps(close_percept.to_dict()),
            close_percept.to_json(),
            close_percept.to_text(),
        ):
            assert "creator_label" not in text_form
            for label in label_values:
                assert label not in text_form
        res = look(world, obj.id)
        assert res.ok
        res_json = json.dumps(res.data)
        assert "creator_label" not in res_json
        for label in label_values:
            assert f'"{label}"' not in res_json


def test_deliberate_friction_ledge_out_of_reach_and_far_side_of_stream() -> None:
    world = load_first_run_world()

    # 1. At least one object is on the raised rock ledge and out of reach from the ground
    ledge_objs = [
        o for o in world.objects.values()
        if o.level == "raised" and world.terrain.is_ledge(o.position)
    ]
    assert len(ledge_objs) >= 1
    ledge_target = world.objects["object_13"]
    assert ledge_target.level == "raised"

    # Place agent on the ground right next to the ledge tile
    world.body.position = world.objects["object_4"].position.model_copy()
    world.body.level = "ground"
    assert not world.can_reach(ledge_target.position)
    assert not in_reach(world, ledge_target)
    res_ground_grab = grab(world, ledge_target.id)
    assert not res_ground_grab.ok
    assert res_ground_grab.reason == OUT_OF_REACH

    # Stack the two flat stones (object_5 on object_4) to climb onto the ledge
    world.body.position = world.objects["object_5"].position.model_copy()
    assert grab(world, "object_5").ok
    world.body.position = world.objects["object_4"].position.model_copy()
    stack_res = stack(world, "object_5", "object_4")
    assert stack_res.ok
    assert world.object_top_height("object_5") >= 0.5
    assert world.climb_onto("object_5")
    assert world.move_agent(ledge_target.position)
    assert world.body.level == "raised"
    assert in_reach(world, ledge_target)
    assert grab(world, ledge_target.id).ok

    # 2. Loose materials exist on the far side of the stream (x > 13.0, east of stream x=11..12)
    far_side_loose = [
        o for o in world.objects.values()
        if not o.fixed and o.position.x > 13.0
    ]
    assert len(far_side_loose) >= 3
    # Verify the stream lies between the agent's start x (5.5) and the far-bank objects
    for obj in far_side_loose:
        assert world.terrain.is_water(11.5, obj.position.y)


def test_world_size_reachable_within_a_few_days_at_default_speed() -> None:
    world = load_first_run_world()
    day_length_ticks = world.daynight.config.day_length
    max_allowed_ticks = day_length_ticks * 2  # well under a few in-world days

    # Compute total ticks needed to walk from start across the entire clearing to all objects
    curr = world.body.position
    total_distance = 0.0
    for obj in world.objects.values():
        total_distance += math.hypot(obj.position.x - curr.x, obj.position.y - curr.y)
        curr = obj.position

    ticks_to_visit_all = math.ceil(total_distance / BASE_MOVE_STEP)
    assert ticks_to_visit_all < max_allowed_ticks
    assert ticks_to_visit_all < 300

    # Verify the agent can actively walk across the stream to the far bank in < 20 ticks
    steps = 0
    while world.body.position.x < 14.0 and steps < 30:
        res = move(world, "E", 1.0)
        assert res.ok
        steps += 1
    assert world.body.position.x >= 14.0
    assert steps <= 15


def test_cli_run_with_world_flag_loads_first_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["run", "--world", str(FIRST_RUN_WORLD_PATH)])
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert "loaded world: first_run.toml" in out
    assert "objects=19" in out

    completed = subprocess.run(
        [sys.executable, "-m", "arden", "run", "--world", str(FIRST_RUN_WORLD_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "loaded world: first_run.toml" in completed.stdout


def test_mock_llm_run_of_2000_ticks_completes() -> None:
    settings = Settings()
    rng = make_rng(settings)
    world = load_first_run_world(rng=rng, settings=settings)

    class SpyMockLLM(MockLLM):
        def __init__(self, seed: int) -> None:
            super().__init__(seed=seed)
            self.prompts: list[tuple[str, str]] = []

        def complete_json(
            self,
            role: str,
            system: str,
            user: str,
            schema: dict[str, Any],
            timeout_s: float,
        ) -> dict[str, Any]:
            self.prompts.append((system, user))
            return super().complete_json(role, system, user, schema, timeout_s)

    backend = SpyMockLLM(seed=derive_llm_seed(settings.world.seed))
    run_world_ticks(world, ticks=2000, settings=settings, backend=backend, heartbeat_ticks=25)

    assert world.tick == 2000
    assert len(backend.prompts) >= 80
    # Verify no creator_label leaked into any prompt sent to the mock LLM
    labels = set(world.creator_labels.values())
    for sys_prompt, user_prompt in backend.prompts:
        assert "creator_label" not in sys_prompt
        assert "creator_label" not in user_prompt
        for label in labels:
            assert label not in user_prompt
