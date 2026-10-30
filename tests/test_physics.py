"""Tests for 2.5D physics: terrain grid, impulse, rolling/friction, collisions, stacking, and occlusion."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from arden.world import (
    Position,
    TerrainGrid,
    Velocity,
    World,
    apply_impulse,
    is_occluded,
    objects_overlap,
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


def _step_until_rest(world: World, max_ticks: int = 100) -> int:
    """Step the world until all objects have zero velocity or ``max_ticks`` is reached."""
    for tick in range(1, max_ticks + 1):
        world.step()
        if all(obj.velocity.speed == 0.0 for obj in world.objects.values()):
            return tick
    return max_ticks


def test_light_sphere_pushed_with_force_0_5_travels_farther_than_flat_stone() -> None:
    world = World(seed=1)
    sphere = world.spawn(
        {**_BASE_PROPS, "shape": "sphere", "weight": 0.3},
        {"x": 0.0, "y": 0.0},
    )
    flat_stone = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "height": 0.1, "volume": 0.01, "weight": 2.5},
        {"x": 0.0, "y": 5.0},
    )

    world.apply_impulse(sphere.id, (1.0, 0.0), 0.5)
    world.apply_impulse(flat_stone.id, (1.0, 0.0), 0.5)

    # First tick: both have positive velocity and slow down each subsequent tick
    assert world.objects[sphere.id].velocity.x > world.objects[flat_stone.id].velocity.x > 0.0
    v0_sphere = world.objects[sphere.id].velocity.speed
    world.step()
    v1_sphere = world.objects[sphere.id].velocity.speed
    assert 0.0 < v1_sphere < v0_sphere

    _step_until_rest(world)

    sphere_dist = world.objects[sphere.id].position.x
    stone_dist = world.objects[flat_stone.id].position.x
    assert sphere_dist > stone_dist > 0.0
    assert sphere_dist > stone_dist * 3.0


def test_round_shapes_roll_farther_than_cube_and_flat_with_same_weight_and_impulse() -> None:
    world = World(seed=2)
    shapes = ["sphere", "cylinder", "cube", "flat"]
    ids: dict[str, str] = {}
    for idx, shape in enumerate(shapes):
        obj = world.spawn(
            {**_BASE_PROPS, "shape": shape, "weight": 0.5},
            {"x": 0.0, "y": float(idx * 4)},
        )
        ids[shape] = obj.id
        apply_impulse(world, obj.id, (1.0, 0.0), 0.6)

    _step_until_rest(world)

    dist = {shape: world.objects[obj_id].position.x for shape, obj_id in ids.items()}
    assert dist["sphere"] > dist["cylinder"] > dist["cube"] > dist["flat"] > 0.0


def test_apply_impulse_distance_depends_on_force_and_weight() -> None:
    world = World(seed=3)
    light = world.spawn({**_BASE_PROPS, "shape": "cube", "weight": 0.2}, {"x": 0.0, "y": 0.0})
    heavy = world.spawn({**_BASE_PROPS, "shape": "cube", "weight": 3.0}, {"x": 0.0, "y": 4.0})
    strong = world.spawn({**_BASE_PROPS, "shape": "cube", "weight": 0.2}, {"x": 0.0, "y": 8.0})
    fixed_obj = world.spawn(
        {**_BASE_PROPS, "shape": "cylinder", "weight": 10.0, "fixed": True},
        {"x": 0.0, "y": 12.0},
    )

    world.apply_impulse(light.id, (1.0, 0.0), 0.4)
    world.apply_impulse(heavy.id, (1.0, 0.0), 0.4)
    world.apply_impulse(strong.id, (1.0, 0.0), 0.9)
    world.apply_impulse(fixed_obj.id, (1.0, 0.0), 1.0)

    _step_until_rest(world)

    assert world.objects[strong.id].position.x > world.objects[light.id].position.x
    assert world.objects[light.id].position.x > world.objects[heavy.id].position.x > 0.0
    assert world.objects[fixed_obj.id].position == Position(x=0.0, y=12.0)


def test_terrain_water_makes_objects_wet_and_drifts_light_objects_downstream() -> None:
    terrain = TerrainGrid()
    for tx in range(0, 10):
        terrain.set_tile(tx, 0, "water", flow=(1.0, 0.2))

    world = World(seed=4, terrain=terrain)
    assert world.tile_at((0.5, 0.5)) == "water"
    assert world.tile_at((0.5, 0.5)).wet is True
    assert world.tile_at((0.5, 5.5)) == "grass"

    leaf = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "weight": 0.1, "state": {"wet": False}},
        {"x": 0.5, "y": 0.5},
    )
    heavy_rock = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "weight": 4.0, "state": {"wet": False}},
        {"x": 0.5, "y": 0.5},
    )
    world.move(heavy_rock.id, {"x": 2.5, "y": 0.5})

    for _ in range(15):
        world.step()

    updated_leaf = world.objects[leaf.id]
    updated_rock = world.objects[heavy_rock.id]

    assert updated_leaf.state["wet"] is True
    assert updated_leaf.state["floating"] is True
    assert updated_leaf.position.x > 1.5
    assert updated_leaf.position.y > 0.5

    # Heavy rock gets wet in water but does not float or drift with the current
    assert updated_rock.state["wet"] is True
    assert updated_rock.state["floating"] is False
    assert updated_rock.position == Position(x=2.5, y=0.5)


def test_terrain_ledge_reachability_by_climbing_or_adjacent_raised_tile() -> None:
    terrain = TerrainGrid()
    terrain.set_tile(2, 0, "ledge")
    terrain.set_tile(3, 0, "ledge")

    world = World(seed=5, terrain=terrain, body={"position": {"x": 1.5, "y": 0.5}})
    assert world.tile_at((2.5, 0.5)).type == "ledge"
    assert world.tile_at((2.5, 0.5)).level == "raised"

    # Cannot walk directly from ground (1.5, 0.5) onto the ledge (2.5, 0.5)
    assert world.can_reach((2.5, 0.5)) is False
    assert world.move_agent((2.5, 0.5)) is False
    assert world.body.position == Position(x=1.5, y=0.5)
    assert world.body.level == "ground"

    # Spawn a climbable crate next to the ledge and climb onto it
    crate = world.spawn(
        {**_BASE_PROPS, "shape": "cube", "height": 0.5, "volume": 0.1, "weight": 2.0},
        {"x": 1.6, "y": 0.5},
    )
    assert world.climb_onto(crate.id) is True
    assert world.body.level == f"on_top_of:{crate.id}"

    # From on_top_of the crate, the adjacent ledge is now reachable
    assert world.can_reach((2.5, 0.5)) is True
    assert world.move_agent((2.5, 0.5)) is True
    assert world.body.position == Position(x=2.5, y=0.5)
    assert world.body.level == "raised"

    # From a raised ledge tile, an adjacent raised ledge tile (3.5, 0.5) is reachable
    assert world.can_reach((3.5, 0.5)) is True
    assert world.move_agent((3.5, 0.5)) is True
    assert world.body.level == "raised"

    # Stepping off the ledge onto grass drops the agent back to ground
    assert world.move_agent((3.5, 1.5)) is True
    assert world.body.level == "ground"

    # Ground objects pushed toward a ledge are blocked from entering the ledge tile
    ball = world.spawn({**_BASE_PROPS, "shape": "sphere", "weight": 0.2}, {"x": 1.2, "y": 0.5})
    world.apply_impulse(ball.id, (1.0, 0.0), 1.0)
    for _ in range(10):
        world.step()
        assert not world.terrain.is_ledge(world.objects[ball.id].position)


def test_circle_collision_with_movable_and_fixed_objects() -> None:
    world = World(seed=6)
    cue = world.spawn({**_BASE_PROPS, "shape": "sphere", "weight": 0.5}, {"x": 0.0, "y": 0.0})
    target = world.spawn({**_BASE_PROPS, "shape": "sphere", "weight": 0.5}, {"x": 0.35, "y": 0.0})
    tree = world.spawn(
        {**_BASE_PROPS, "shape": "cylinder", "height": 2.0, "volume": 0.4, "weight": 20.0, "fixed": True},
        {"x": 1.2, "y": 0.0},
    )

    world.apply_impulse(cue.id, (1.0, 0.0), 0.8)
    for _ in range(15):
        world.step()
        assert not objects_overlap(world.objects[cue.id], world.objects[target.id])
        assert not objects_overlap(world.objects[target.id], world.objects[tree.id])
        assert not objects_overlap(world.objects[cue.id], world.objects[tree.id])

    # Target was knocked forward by cue, and fixed tree never moved
    assert world.objects[target.id].position.x > 0.35
    assert world.objects[tree.id].position == Position(x=1.2, y=0.0)


def test_on_top_of_carries_when_stable_and_topples_when_unstable() -> None:
    world = World(seed=7)
    base = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "height": 0.2, "volume": 0.08, "weight": 3.0},
        {"x": 0.0, "y": 0.0},
    )
    upper = world.spawn(
        {**_BASE_PROPS, "shape": "cube", "height": 0.15, "volume": 0.01, "weight": 0.4},
        {"x": 1.0, "y": 0.0},
    )

    stacked = world.stack(upper.id, base.id)
    assert stacked.level == f"on_top_of:{base.id}"
    assert stacked.position == base.position
    assert world.object_top_height(upper.id) == pytest.approx(base.height + upper.height)

    # Gentle push on the heavy flat base carries the light cube on top of it
    world.apply_impulse(base.id, (1.0, 0.0), 0.5)
    _step_until_rest(world)

    moved_base = world.objects[base.id]
    carried_upper = world.objects[upper.id]
    assert moved_base.position.x > 0.0
    assert carried_upper.level == f"on_top_of:{base.id}"
    assert carried_upper.position == moved_base.position

    # Violent push / sudden large movement topples the upper object to the ground
    world.apply_impulse(base.id, (1.0, 0.0), 30.0)
    world.step()

    toppled_upper = world.objects[upper.id]
    assert toppled_upper.level == "ground"
    assert not objects_overlap(world.objects[base.id], toppled_upper)

    # Stacking onto a sphere is statically unstable and topples immediately
    ball = world.spawn(
        {**_BASE_PROPS, "shape": "sphere", "height": 0.3, "volume": 0.014, "weight": 0.5},
        {"x": -3.0, "y": 0.0},
    )
    block = world.spawn(
        {**_BASE_PROPS, "shape": "cube", "height": 0.2, "volume": 0.008, "weight": 0.4},
        {"x": -2.0, "y": 0.0},
    )
    after_stack = world.stack(block.id, ball.id)
    assert after_stack.level == "ground"
    assert not objects_overlap(world.objects[ball.id], after_stack)


def test_is_occluded_detects_taller_object_near_line_of_sight() -> None:
    world = World(seed=8)
    small_target = world.spawn(
        {**_BASE_PROPS, "shape": "sphere", "height": 0.2, "volume": 0.004},
        {"x": 6.0, "y": 0.0},
    )
    tall_tree = world.spawn(
        {
            **_BASE_PROPS,
            "shape": "cylinder",
            "height": 2.5,
            "volume": 0.6,
            "weight": 15.0,
            "fixed": True,
        },
        {"x": 3.0, "y": 0.1},
    )

    viewer = Position(x=0.0, y=0.0)
    assert world.is_occluded(viewer, small_target) is True
    assert is_occluded(viewer, small_target) is True
    assert is_occluded(viewer, small_target.id) is True

    # Move the tall tree far off the line between viewer and target -> not occluded
    world.move(tall_tree.id, {"x": 3.0, "y": 3.0})
    assert world.is_occluded(viewer, small_target) is False

    # Move the tall tree behind the target -> not occluded
    world.move(tall_tree.id, {"x": 8.0, "y": 0.0})
    assert world.is_occluded(viewer, small_target) is False

    # Replace the blocker between viewer and target with a shorter object -> not occluded
    short_pebble = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "height": 0.05, "volume": 0.001},
        {"x": 3.0, "y": 0.0},
    )
    assert short_pebble.height < small_target.height
    assert world.is_occluded(viewer, small_target) is False


def test_creator_move_and_despawn_carry_or_topple_stacked_objects() -> None:
    world = World(seed=9)
    base = world.spawn(
        {**_BASE_PROPS, "shape": "cube", "height": 0.3, "volume": 0.05, "weight": 2.0},
        {"x": 0.0, "y": 0.0},
    )
    upper = world.spawn(
        {**_BASE_PROPS, "shape": "flat", "height": 0.1, "volume": 0.01, "weight": 0.4},
        {"x": 1.0, "y": 0.0},
    )
    world.stack(upper.id, base.id)

    # Small move carries upper object
    world.move(base.id, {"x": 0.5, "y": 0.0})
    assert world.objects[upper.id].level == f"on_top_of:{base.id}"
    assert world.objects[upper.id].position == Position(x=0.5, y=0.0)

    # Large sudden move topples upper object
    world.move(base.id, {"x": 10.0, "y": 0.0})
    assert world.objects[upper.id].level == "ground"
    assert not objects_overlap(world.objects[base.id], world.objects[upper.id])

    # Restack and despawn base -> upper drops to ground
    world.stack(upper.id, base.id)
    assert world.objects[upper.id].level == f"on_top_of:{base.id}"
    world.despawn(base.id)
    assert world.objects[upper.id].level == "ground"


def test_snapshot_and_restore_preserves_terrain_and_midflight_physics() -> None:
    terrain = TerrainGrid()
    terrain.set_rect(0, 2, 5, 2, "water", flow=(0.6, 0.0))
    terrain.set_tile(4, 0, "ledge")

    world_a = World(seed=101, terrain=terrain)
    ball = world_a.spawn({**_BASE_PROPS, "shape": "sphere", "weight": 0.3}, {"x": 0.0, "y": 0.0})
    leaf = world_a.spawn({**_BASE_PROPS, "shape": "flat", "weight": 0.1}, {"x": 0.5, "y": 2.5})
    world_a.apply_impulse(ball.id, "east", 0.7)

    for _ in range(4):
        world_a.step()

    assert world_a.objects[ball.id].velocity.speed > 0.0
    snap = world_a.snapshot()
    world_b = World.restore(snap)

    assert world_b.tile_at((1.5, 2.5)) == "water"
    assert world_b.tile_at((1.5, 2.5)).flow == Velocity(x=0.6, y=0.0)
    assert world_b.tile_at((4.5, 0.5)) == "ledge"

    for _ in range(20):
        world_a.step()
        world_b.step()

    assert world_a.snapshot() == world_b.snapshot()
    assert world_a.objects[leaf.id].position.x > 0.5



def test_no_overlaps_after_1000_random_ticks_with_fixed_seed() -> None:
    rng = random.Random(20261030)
    terrain = TerrainGrid()
    for tx in range(-2, 3):
        terrain.set_tile(tx, 2, "water", flow=(0.8, -0.2))
    terrain.set_tile(4, 0, "ledge")
    terrain.set_tile(4, 1, "ledge")

    world = World(seed=42, terrain=terrain)
    shapes = ("sphere", "cylinder", "cube", "flat", "long")

    # Spawn fixed obstacles at non-overlapping locations
    fixed_positions = [(-1.5, -1.5), (1.5, -1.0), (0.0, 1.2)]
    for fx, fy in fixed_positions:
        world.spawn(
            {
                **_BASE_PROPS,
                "shape": "cylinder",
                "height": 1.5,
                "volume": 0.12,
                "weight": 10.0,
                "fixed": True,
            },
            {"x": fx, "y": fy},
        )

    # Spawn 12 movable objects with varied shapes, weights, and sizes (some initially overlapping)
    movable_ids: list[str] = []
    for _ in range(12):
        shape = rng.choice(shapes)
        height = rng.uniform(0.1, 0.5)
        volume = rng.uniform(0.003, 0.04)
        weight = rng.uniform(0.1, 3.0)
        obj = world.spawn(
            {
                **_BASE_PROPS,
                "shape": shape,
                "height": height,
                "volume": volume,
                "weight": weight,
                "fixed": False,
            },
            {"x": rng.uniform(-2.0, 2.0), "y": rng.uniform(-2.0, 2.0)},
        )
        movable_ids.append(obj.id)

    for _ in range(1000):
        # Apply random impulses to a few objects each tick
        for _ in range(rng.randint(0, 3)):
            oid = rng.choice(movable_ids)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            force = rng.uniform(0.05, 1.5)
            world.apply_impulse(oid, (math.cos(angle), math.sin(angle)), force)

        world.step()

        # Verify no pair of ground objects overlaps after every single step
        ground_objs = [o for o in world.objects.values() if o.level == "ground"]
        for i in range(len(ground_objs)):
            a = ground_objs[i]
            for j in range(i + 1, len(ground_objs)):
                b = ground_objs[j]
                assert not objects_overlap(a, b), (
                    f"Overlap at tick {world.tick} between {a.id} at {a.position} (r={a.radius}) "
                    f"and {b.id} at {b.position} (r={b.radius})"
                )
