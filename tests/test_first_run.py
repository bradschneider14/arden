"""Tests for the first-run clearing world definition (Ticket #20)."""

from __future__ import annotations

import json
from pathlib import Path

from arden.__main__ import main
from arden.actions.verbs import in_reach, look
from arden.agent.loop import AgentLoop, run_headless
from arden.config import Settings
from arden.environment.weather import compute_shelter_exposure
from arden.perception.feed import PerceptionTracker, perceive
from arden.world import (
    DEFAULT_FIRST_RUN_WORLD_PATH,
    MATERIAL_PROPERTIES,
    VERY_LIGHT_MAX_WEIGHT,
    World,
    check_material_property_coverage,
    load_first_run_world,
    load_world,
    verify_material_properties,
)


def test_first_run_world_loads_terrain_and_twelve_object_types() -> None:
    world = load_world(DEFAULT_FIRST_RUN_WORLD_PATH, seed=42)

    assert world.terrain.width == 16
    assert world.terrain.height == 16
    assert world.terrain.tile_size == 1.0
    assert world.terrain.default_type == "grass"

    # Agent starts in the grass clearing and can reach any corner within a fraction of one day
    # (max diagonal ~22.6m at 0.5m/tick is ~46 ticks, well under 1,200 ticks/day).
    assert world.terrain.tile_type_at(world.body.position) == "grass"
    max_diagonal = (
        (world.terrain.width * world.terrain.tile_size) ** 2
        + (world.terrain.height * world.terrain.tile_size) ** 2
    ) ** 0.5
    ticks_to_cross_diagonal = max_diagonal / 0.5
    assert ticks_to_cross_diagonal < world.daynight.config.day_length * 0.1

    # Stream along y = 12 with eastward flow direction
    for tx in range(world.terrain.width):
        tile = world.terrain.get_tile(tx + 0.5, 12.5)
        assert tile.type == "water"
        assert tile.wet is True
        assert tile.flow.x > 0.0
        assert tile.flow.speed > 0.0

    # Raised rock ledge at x = 1..2, y = 5..8
    for tx in (1, 2):
        for ty in (5, 6, 7, 8):
            tile = world.terrain.get_tile(tx + 0.5, ty + 0.5)
            assert tile.type == "ledge"
            assert tile.level == "raised"
            assert tile.elevation == 1.0

    # Verify all required object archetypes exist and are built purely from properties
    objs = list(world.objects.values())
    assert len(objs) == 20

    # 1. Large trees: tall, fixed, cylindrical, provide shade/shelter
    trees = [o for o in objs if o.fixed and o.shape == "cylinder" and o.height >= 2.8]
    assert len(trees) >= 2
    rain_exp, wind_exp = compute_shelter_exposure(world, trees[0].position)
    assert rain_exp < 0.5
    assert wind_exp < 0.5

    # 2. Sticks: long, rigid, light, movable
    sticks = [o for o in objs if o.shape == "long" and not o.binding and o.rigidity >= 0.8]
    assert len(sticks) >= 3

    # 3. Flat stones: flat, heavy, rigid, movable
    stones = [o for o in objs if o.shape == "flat" and o.weight >= 2.0 and o.rigidity >= 0.9]
    assert len(stones) >= 4

    # 4. Vines: long, binding, flexible
    vines = [o for o in objs if o.binding and o.shape == "long"]
    assert len(vines) >= 2

    # 5. Leaves: flat, very light (pushed by wind), including a wet leaf in the stream
    leaves = [o for o in objs if o.shape == "flat" and o.weight <= VERY_LIGHT_MAX_WEIGHT]
    assert len(leaves) >= 2
    assert any(o.state.get("wet") is True for o in leaves)

    # 6. Ball: round sphere, light, with push scale and bounce on strike
    balls = [
        o
        for o in objs
        if o.shape == "sphere"
        and not o.fragile
        and any(r.verb == "push" for r in o.behavior)
        and any(r.verb == "strike" and getattr(r, "response", None) == "bounce" for r in o.behavior)
    ]
    assert len(balls) == 1

    # 7. Hollow log: hollow, heavy cylinder
    hollow_logs = [o for o in objs if o.shape == "cylinder" and o.hollow and not o.fixed and o.weight >= 5.0]
    assert len(hollow_logs) == 1

    # 8. Berry bush and fragile edible berries
    edible_fragile = [
        o
        for o in objs
        if o.fragile and any(r.verb == "edible" for r in o.behavior)
    ]
    assert len(edible_fragile) >= 3
    bush = [o for o in edible_fragile if o.fixed]
    loose_berries = [o for o in edible_fragile if not o.fixed]
    assert len(bush) == 1
    assert len(loose_berries) >= 2

    # 9. Mailbox placeholder at the edge of the path offering an opaque action
    mailboxes = [
        o
        for o in objs
        if o.fixed and o.hollow and any(r.verb == "action_1" for r in o.behavior)
    ]
    assert len(mailboxes) == 1


def test_first_run_material_property_coverage() -> None:
    world = load_first_run_world(seed=7)
    coverage = check_material_property_coverage(world)

    assert set(coverage.keys()) == set(MATERIAL_PROPERTIES)
    for prop in MATERIAL_PROPERTIES:
        assert coverage[prop] is True, f"material property {prop!r} not represented in first_run.toml"
    assert verify_material_properties(world) is True


def test_first_run_deliberate_friction_ledge_and_far_side_of_stream() -> None:
    world = load_first_run_world(seed=11)

    # 1. Objects on the raised rock ledge are out of reach from ground level
    ledge_objects = [
        obj
        for obj in world.objects.values()
        if obj.level == "raised" and world.terrain.is_ledge(obj.position)
    ]
    assert len(ledge_objects) >= 2

    # Even when the agent walks right next to the ledge on the ground (x = 3.1, y = 6.0),
    # the ledge tile and objects on it cannot be reached directly from the ground.
    assert world.move_agent((3.1, 6.0)) is True
    assert world.body.level == "ground"
    for ledge_obj in ledge_objects:
        assert world.can_reach(ledge_obj.position) is False
        assert in_reach(world, ledge_obj.id) is False

    # Stacking the two nearby flat stones (object_9 and object_10) next to the ledge
    # gives combined height 0.56m >= 0.5m, allowing the agent to climb onto the ledge!
    world.move("object_9", (3.2, 6.0))
    world.move("object_10", (3.2, 6.6))
    stacked = world.stack("object_10", "object_9")
    assert stacked.level == "on_top_of:object_9"
    assert world.climb_onto("object_10") is True
    assert world.can_reach((2.5, 6.0)) is True
    assert world.move_agent((2.5, 6.0)) is True
    assert world.body.level == "raised"
    assert world.move_agent((1.8, 6.0)) is True
    assert in_reach(world, "object_16") is True

    # 2. Loose materials on the far side of the stream (y > 13.0, across stream at y = 12)
    stream_y = 12.5
    assert world.terrain.tile_type_at((6.5, stream_y)) == "water"
    far_bank_materials = [
        obj
        for obj in world.objects.values()
        if not obj.fixed and obj.position.y > 13.0
    ]
    assert len(far_bank_materials) >= 3
    far_shapes = {obj.shape for obj in far_bank_materials}
    assert "long" in far_shapes
    assert "flat" in far_shapes
    assert any(obj.binding for obj in far_bank_materials)


def test_creator_label_never_leaks_into_perception(tmp_path: Path) -> None:
    world = load_first_run_world(seed=19)

    # Every object in first_run.toml has a creator_label for the web client and observer log
    labels = world.creator_labels
    assert len(labels) == len(world.objects)
    for obj_id, obj in world.objects.items():
        assert obj.creator_label is not None
        assert world.creator_label_for(obj_id) == obj.creator_label
        # Observer / web client serialization includes creator_label
        dumped = obj.model_dump(mode="json")
        assert dumped["creator_label"] == obj.creator_label

    # Collect all forbidden label strings and words that are exclusively human names
    forbidden_labels = set(labels.values())
    forbidden_words = {
        "tree",
        "berry",
        "bush",
        "hollow log",
        "ball",
        "stick",
        "stone",
        "vine",
        "leaf",
        "mailbox",
    }

    # 1. WorldObject.perceptible() never contains creator_label
    for obj in world.objects.values():
        perceptible_dict = obj.perceptible()
        assert "creator_label" not in perceptible_dict
        serialized = json.dumps(perceptible_dict)
        for label in forbidden_labels:
            assert label not in serialized
        for word in forbidden_words:
            assert word not in serialized.lower()

    # 2. Full-world Percept (to_dict, to_json, to_text) never contains creator_label
    percept = perceive(world, sensing_radius=100.0)
    percept_dict = percept.to_dict()
    percept_json = percept.to_json()
    percept_text = percept.to_text()
    assert "creator_label" not in percept_json
    assert "creator_label" not in percept_text
    for obj_entry in percept_dict["objects"]:
        assert "creator_label" not in obj_entry
    for label in forbidden_labels:
        assert label not in percept_json
        assert label not in percept_text
    for word in forbidden_words:
        assert word not in percept_json.lower()
        assert word not in percept_text.lower()

    # 3. look() action observation never contains creator_label
    look_res = look(world, "object_3")
    assert look_res.ok is True
    look_json = json.dumps(look_res.data)
    assert "creator_label" not in look_json
    for label in forbidden_labels:
        assert label not in look_json

    # 4. PerceptionTracker and AgentLoop decide prompts never contain creator_label
    tracker = PerceptionTracker(sensing_radius=100.0)
    tracked_percept = tracker.perceive(world)
    assert "creator_label" not in tracked_percept.to_json()

    loop = AgentLoop(world=world, settings=Settings(), print_wakes=False)
    tick_res = loop.tick()
    assert tick_res.woke is True
    assert tick_res.prompt is not None
    assert "creator_label" not in tick_res.prompt.system
    assert "creator_label" not in tick_res.prompt.user
    for label in forbidden_labels:
        assert label not in tick_res.prompt.user

    # Also verify JSON world file loading works identically
    json_world_path = tmp_path / "first_run.json"
    from arden.world.loader import load_world_data

    json_world_path.write_text(json.dumps(load_world_data(DEFAULT_FIRST_RUN_WORLD_PATH)), encoding="utf-8")
    world_from_json = World.from_file(json_world_path, seed=19)
    assert set(world_from_json.objects.keys()) == set(world.objects.keys())
    assert world_from_json.creator_labels == world.creator_labels


def test_first_run_mock_llm_2000_ticks_and_cli_world_flag(capsys: object) -> None:
    rc = main(
        [
            "run",
            "--world",
            str(DEFAULT_FIRST_RUN_WORLD_PATH),
            "--ticks",
            "2000",
            "--llm",
            "mock",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert len(lines) >= 20
    assert lines[0].startswith("tick=0 reasons=[")

    # Also verify via run_headless directly to inspect final world state
    loop = run_headless(
        world=DEFAULT_FIRST_RUN_WORLD_PATH,
        ticks=200,
        llm="mock",
        print_wakes=False,
    )
    assert loop.world.tick == 200
    assert len(loop.wake_events) >= 2
    assert loop.last_decide_prompt is not None
    assert "creator_label" not in loop.last_decide_prompt.user
