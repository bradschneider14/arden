"""Tests for T09 building verbs (stack, bind, insert, separate) and Composite physics."""

from __future__ import annotations

import math

import pytest

from arden.actions import (
    HANDS_FULL,
    INVALID_PARAMS,
    NOTHING_HAPPENED,
    OUT_OF_REACH,
    TOO_HEAVY,
    Composite,
    bind,
    drop,
    execute_action,
    grab,
    insert,
    move,
    place,
    push,
    separate,
    stack,
    strike,
)
from arden.perception import perceive
from arden.world import (
    STACK_STABILITY_THRESHOLD,
    World,
    WorldObject,
    compute_center_of_mass,
    compute_joint_strength,
    compute_strike_force,
    is_stack_stable,
    stack_stability_score,
    strike_impact_multiplier,
)


def _stick_props() -> dict[str, object]:
    return {
        "shape": "long",
        "height": 0.8,
        "volume": 0.002,
        "weight": 0.4,
        "rigidity": 0.85,
        "color": "brown",
    }


def _vine_props(*, weight: float = 0.1, rigidity: float = 0.35, binding: bool = True) -> dict[str, object]:
    return {
        "shape": "long",
        "height": 0.5,
        "volume": 0.0005,
        "weight": weight,
        "rigidity": rigidity,
        "binding": binding,
        "color": "green",
    }


def _flat_stone_props(*, weight: float = 1.8, height: float = 0.12, volume: float = 0.015) -> dict[str, object]:
    return {
        "shape": "flat",
        "height": height,
        "volume": volume,
        "weight": weight,
        "rigidity": 0.95,
        "color": "gray",
    }


def _sphere_props(*, weight: float = 0.6, height: float = 0.25, volume: float = 0.008) -> dict[str, object]:
    return {
        "shape": "sphere",
        "height": height,
        "volume": volume,
        "weight": weight,
        "rigidity": 0.9,
        "color": "gray",
    }


class TestBindAndCompositeProperties:
    """Tests for bind(held, target, binder) and Composite derived properties."""

    def test_bind_stick_vine_flat_stone_creates_composite_and_consumes_binder(self) -> None:
        world = World(seed=7)
        stick = world.spawn(_stick_props(), (0.2, 0.0))
        stone = world.spawn(_flat_stone_props(), (0.4, 0.0))
        vine = world.spawn(_vine_props(), (0.3, 0.2))

        assert grab(world, stick.id).ok
        res = bind(world, stick.id, stone.id, vine.id)
        assert res.ok
        assert res.reason is None

        composite_id = res.data["composite_id"]
        comp = world.objects[composite_id]
        assert isinstance(comp, Composite)
        assert isinstance(comp, WorldObject)

        # Original parts and binder are removed from active world objects (vine is consumed into joint)
        assert stick.id not in world.objects
        assert stone.id not in world.objects
        assert vine.id not in world.objects
        assert res.data["binder_id"] == vine.id

        # Derived physical properties
        expected_weight = stick.weight + stone.weight + vine.weight
        assert math.isclose(comp.weight, expected_weight, rel_tol=1e-6)
        assert comp.length > stick.height
        assert math.isclose(comp.height, comp.length, rel_tol=1e-6)
        assert comp.shape == "long"
        assert 0.0 < comp.rigidity <= 1.0
        # Stone at tip shifts center of mass above 0.5 toward the head
        assert comp.center_of_mass > 0.5
        assert comp.joint_strength > 0.0

        # Records made_by_agent=True for memory and perception
        assert comp.made_by_agent is True
        assert comp.state.get("made_by_agent") is True
        perceptible = comp.perceptible()
        assert perceptible["state"]["made_by_agent"] is True
        assert "parts" not in perceptible
        assert "binder" not in perceptible

    def test_bind_requires_binder_with_binding_true(self) -> None:
        world = World(seed=11)
        stick = world.spawn(_stick_props(), (0.2, 0.0))
        stone = world.spawn(_flat_stone_props(), (0.4, 0.0))
        non_binder = world.spawn(_vine_props(binding=False), (0.3, 0.1))

        grab(world, stick.id)
        res = bind(world, stick.id, stone.id, non_binder.id)
        assert not res.ok
        assert res.reason == NOTHING_HAPPENED
        assert stick.id in world.objects
        assert stone.id in world.objects
        assert non_binder.id in world.objects

    def test_stick_vine_flat_stone_composite_hits_harder_than_stick_alone(self) -> None:
        # 1. Compare strike force and impulse on an identical target in two parallel worlds
        world_stick = World(seed=21)
        stick_a = world_stick.spawn(_stick_props(), (0.2, 0.0))
        crate_a = world_stick.spawn(
            {"shape": "cube", "height": 0.4, "volume": 0.03, "weight": 2.5, "rigidity": 0.9},
            (0.8, 0.0),
        )
        grab(world_stick, stick_a.id)
        res_stick = strike(world_stick, crate_a.id, 0.4)
        assert res_stick.ok
        world_stick.step()
        dist_stick = crate_a.position.distance_to(world_stick.objects[crate_a.id].position)

        world_comp = World(seed=21)
        stick_b = world_comp.spawn(_stick_props(), (0.2, 0.0))
        stone_b = world_comp.spawn(_flat_stone_props(weight=2.0), (0.3, 0.0))
        vine_b = world_comp.spawn(_vine_props(weight=0.25, rigidity=0.7), (0.25, 0.1))
        crate_b = world_comp.spawn(
            {"shape": "cube", "height": 0.4, "volume": 0.03, "weight": 2.5, "rigidity": 0.9},
            (0.8, 0.0),
        )
        grab(world_comp, stick_b.id)
        bind_res = bind(world_comp, stick_b.id, stone_b.id, vine_b.id)
        assert bind_res.ok
        comp = world_comp.objects[bind_res.data["composite_id"]]

        assert strike_impact_multiplier(comp) > strike_impact_multiplier(stick_a)
        assert compute_strike_force(0.4, comp) > compute_strike_force(0.4, stick_a)

        res_comp = strike(world_comp, crate_b.id, 0.4)
        assert res_comp.ok
        assert res_comp.data["effective_force"] > res_stick.data["effective_force"]
        world_comp.step()
        dist_comp = crate_b.position.distance_to(world_comp.objects[crate_b.id].position)
        assert dist_comp > dist_stick

        # 2. A rock with a high break threshold (above_force=0.75) does not break when struck
        # with the stick at force=0.45, but breaks when struck with the composite at force=0.45.
        rock_props = {
            "shape": "cube",
            "height": 0.35,
            "volume": 0.02,
            "weight": 3.0,
            "rigidity": 0.95,
            "behavior": [{"verb": "strike", "response": "break", "above_force": 0.75}],
        }
        rock_a = world_stick.spawn(rock_props, (0.6, 0.4))
        strike_rock_stick = strike(world_stick, rock_a.id, 0.45)
        assert strike_rock_stick.ok
        assert strike_rock_stick.data["broken"] is False
        assert rock_a.id in world_stick.objects

        rock_b = world_comp.spawn(rock_props, (0.6, 0.4))
        strike_rock_comp = strike(world_comp, rock_b.id, 0.45)
        assert strike_rock_comp.ok
        assert strike_rock_comp.data["broken"] is True
        assert rock_b.id not in world_comp.objects

    def test_joint_fails_under_high_load_when_striking_above_joint_strength(self) -> None:
        world = World(seed=33)
        stick = world.spawn(_stick_props(), (0.2, 0.0))
        heavy_stone = world.spawn(_flat_stone_props(weight=2.4), (0.3, 0.0))
        # Weak, thin vine produces a joint that fails under a heavy full-force strike
        weak_vine = world.spawn(_vine_props(weight=0.02, rigidity=0.1), (0.25, 0.1))
        anvil = world.spawn(
            {"shape": "cube", "height": 0.5, "volume": 0.08, "weight": 20.0, "rigidity": 1.0, "fixed": True},
            (0.7, 0.0),
        )

        grab(world, stick.id)
        bind_res = bind(world, stick.id, heavy_stone.id, weak_vine.id)
        assert bind_res.ok
        comp_id = bind_res.data["composite_id"]
        comp = world.objects[comp_id]
        assert comp.joint_strength < compute_strike_force(1.0, comp)

        strike_res = strike(world, anvil.id, 1.0)
        assert strike_res.ok
        assert strike_res.data["joint_failed"] is True
        assert comp_id not in world.objects
        assert stick.id in world.objects
        assert heavy_stone.id in world.objects
        # Binder remains consumed even after joint failure
        assert weak_vine.id not in world.objects


class TestCompositePersistenceRejoinAndSeparate:
    """Tests for composite persistence, re-joining, and separate(target)."""

    def test_composite_persists_across_steps_snapshots_and_can_be_joined_again_and_separated(self) -> None:
        world = World(seed=42)
        stick = world.spawn(_stick_props(), (0.2, 0.0))
        stone = world.spawn(_flat_stone_props(weight=1.2), (0.3, 0.0))
        vine1 = world.spawn(_vine_props(), (0.2, 0.2))
        counterweight = world.spawn(
            {"shape": "cube", "height": 0.15, "volume": 0.003, "weight": 0.6, "rigidity": 0.9},
            (0.4, 0.1),
        )
        vine2 = world.spawn(_vine_props(), (0.3, 0.2))

        grab(world, stick.id)
        res1 = bind(world, stick.id, stone.id, vine1.id)
        assert res1.ok
        comp1_id = res1.data["composite_id"]

        # Persist across ticks and snapshot/restore
        world.step()
        snap = world.snapshot()
        restored = World.restore(snap)
        assert comp1_id in restored.objects
        restored_comp1 = restored.objects[comp1_id]
        assert isinstance(restored_comp1, Composite)
        assert restored_comp1.made_by_agent is True
        assert math.isclose(restored_comp1.weight, world.objects[comp1_id].weight, rel_tol=1e-6)

        # Join the composite again with another part using a second binder
        res2 = bind(restored, comp1_id, counterweight.id, vine2.id)
        assert res2.ok
        comp2_id = res2.data["composite_id"]
        comp2 = restored.objects[comp2_id]
        assert isinstance(comp2, Composite)
        assert comp2.made_by_agent is True
        assert comp1_id not in restored.objects
        assert counterweight.id not in restored.objects
        assert vine2.id not in restored.objects
        expected_total_weight = (
            stick.weight + stone.weight + vine1.weight + counterweight.weight + vine2.weight
        )
        assert math.isclose(comp2.weight, expected_total_weight, rel_tol=1e-6)

        # Perception feed sees made_by_agent and offers 'separate' affordance without leaking hidden rules
        percept = perceive(restored)
        assert "separate" in percept.affordances[comp2_id]
        assert "behavior" not in percept.to_json()

        # Separate restores all original non-binder parts
        sep_res = separate(restored, comp2_id)
        assert sep_res.ok
        assert comp2_id not in restored.objects
        assert stick.id in restored.objects
        assert stone.id in restored.objects
        assert counterweight.id in restored.objects
        assert vine1.id not in restored.objects
        assert vine2.id not in restored.objects

        restored_stick = restored.objects[stick.id]
        restored_stone = restored.objects[stone.id]
        restored_cw = restored.objects[counterweight.id]
        assert math.isclose(restored_stick.weight, stick.weight, rel_tol=1e-6)
        assert restored_stick.shape == stick.shape
        assert math.isclose(restored_stone.weight, stone.weight, rel_tol=1e-6)
        assert restored_stone.shape == stone.shape
        assert math.isclose(restored_cw.weight, counterweight.weight, rel_tol=1e-6)


class TestStackStabilityAndToppling:
    """Tests for stack(held, target), stability scoring, and deterministic toppling."""

    def test_unstable_four_stack_of_spheres_topples_deterministically(self) -> None:
        def _run_sphere_stack(seed: int) -> dict[str, tuple[float, float, str]]:
            world = World(seed=seed)
            spheres = [
                world.spawn(_sphere_props(), (0.4, 0.0)),
                world.spawn(_sphere_props(), (0.2, 0.2)),
                world.spawn(_sphere_props(), (0.2, -0.2)),
                world.spawn(_sphere_props(), (0.0, 0.3)),
            ]
            # Stability score on a sphere base is below threshold
            assert stack_stability_score(spheres[1], spheres[0]) < STACK_STABILITY_THRESHOLD

            for s in spheres[1:]:
                assert grab(world, s.id).ok
                res = stack(world, s.id, spheres[0].id)
                assert res.ok
                assert res.data["stable"] is False
                assert res.data["toppled"] is True

            world.step()
            for s in spheres:
                obj = world.objects[s.id]
                assert obj.level == "ground"

            return {
                s.id: (
                    round(world.objects[s.id].position.x, 6),
                    round(world.objects[s.id].position.y, 6),
                    world.objects[s.id].level,
                )
                for s in spheres
            }

        # Deterministic given the same seed
        run_a = _run_sphere_stack(seed=99)
        run_b = _run_sphere_stack(seed=99)
        assert run_a == run_b

    def test_four_stack_of_flat_stones_stacks_stably(self) -> None:
        world = World(seed=99)
        stones = [
            world.spawn(_flat_stone_props(weight=2.0, height=0.15, volume=0.02), (0.5, 0.0)),
            world.spawn(_flat_stone_props(weight=1.8, height=0.15, volume=0.018), (0.2, 0.2)),
            world.spawn(_flat_stone_props(weight=1.6, height=0.15, volume=0.016), (0.2, -0.2)),
            world.spawn(_flat_stone_props(weight=1.4, height=0.15, volume=0.014), (0.0, 0.3)),
        ]

        assert stack_stability_score(stones[1], stones[0]) >= STACK_STABILITY_THRESHOLD

        for idx in range(1, 4):
            stone = stones[idx]
            assert grab(world, stone.id).ok
            # Stacking onto the base stone stacks onto the top of the existing column
            res = stack(world, stone.id, stones[0].id)
            assert res.ok
            assert res.data["stable"] is True
            assert res.data["toppled"] is False

        world.step()
        assert world.objects[stones[0].id].level == "ground"
        assert world.objects[stones[1].id].level == f"on_top_of:{stones[0].id}"
        assert world.objects[stones[2].id].level == f"on_top_of:{stones[1].id}"
        assert world.objects[stones[3].id].level == f"on_top_of:{stones[2].id}"
        assert is_stack_stable(stones[3].id, world=world)
        assert math.isclose(world.object_top_height(stones[3].id), 0.60, rel_tol=1e-6)

        # Separating the top stone unstacks it back to ground
        sep_res = separate(world, stones[3].id)
        assert sep_res.ok
        assert world.objects[stones[3].id].level == "ground"


class TestInsertAndContainerMovement:
    """Tests for insert(held, target) and container movement propagation."""

    def test_insert_fails_for_non_hollow_target_and_insufficient_volume(self) -> None:
        world = World(seed=5)
        pebble = world.spawn(
            {"shape": "sphere", "height": 0.1, "volume": 0.002, "weight": 0.2, "rigidity": 0.9},
            (0.2, 0.0),
        )
        solid_block = world.spawn(
            {"shape": "cube", "height": 0.4, "volume": 0.05, "weight": 2.0, "rigidity": 0.9, "hollow": False},
            (0.4, 0.0),
        )
        tiny_cup = world.spawn(
            {"shape": "cylinder", "height": 0.1, "volume": 0.001, "weight": 0.1, "rigidity": 0.8, "hollow": True},
            (0.3, 0.2),
        )

        assert grab(world, pebble.id).ok
        # Non-hollow target fails
        res_solid = insert(world, pebble.id, solid_block.id)
        assert not res_solid.ok
        assert res_solid.reason == NOTHING_HAPPENED
        assert world.body.holding == pebble.id

        # Hollow target with volume < pebble.volume fails
        res_tiny = insert(world, pebble.id, tiny_cup.id)
        assert not res_tiny.ok
        assert res_tiny.reason == NOTHING_HAPPENED
        assert world.body.holding == pebble.id

    def test_inserted_objects_move_with_container_and_separate_restores_them(self) -> None:
        world = World(seed=15)
        box = world.spawn(
            {
                "shape": "cube",
                "height": 0.4,
                "volume": 0.02,
                "weight": 1.0,
                "rigidity": 0.9,
                "hollow": True,
            },
            (0.4, 0.0),
        )
        pebble1 = world.spawn(
            {"shape": "sphere", "height": 0.1, "volume": 0.005, "weight": 0.3, "rigidity": 0.9},
            (0.2, 0.1),
        )
        pebble2 = world.spawn(
            {"shape": "sphere", "height": 0.1, "volume": 0.005, "weight": 0.4, "rigidity": 0.9},
            (0.2, -0.1),
        )
        oversized = world.spawn(
            {"shape": "cube", "height": 0.3, "volume": 0.015, "weight": 0.5, "rigidity": 0.9},
            (0.1, 0.2),
        )

        # Insert pebble1 and pebble2 into box
        assert grab(world, pebble1.id).ok
        res1 = insert(world, pebble1.id, box.id)
        assert res1.ok
        assert world.body.holding is None
        assert world.objects[pebble1.id].level == f"inside:{box.id}"
        assert world.objects[pebble1.id].state.get("made_by_agent") is True
        assert world.objects[box.id].state.get("made_by_agent") is True

        assert grab(world, pebble2.id).ok
        res2 = execute_action(world, "insert", target=box.id, params={"held": pebble2.id})
        assert res2.ok
        assert world.objects[pebble2.id].level == f"inside:{box.id}"
        assert math.isclose(world.objects[box.id].weight, 1.0 + 0.3 + 0.4, rel_tol=1e-6)

        # Remaining capacity is 0.02 - 0.01 = 0.01, so oversized (0.015) does not fit
        assert grab(world, oversized.id).ok
        res_over = insert(world, oversized.id, box.id)
        assert not res_over.ok
        assert res_over.reason == NOTHING_HAPPENED
        assert drop(world).ok

        # 1. Pushing the container moves inserted objects with it across ticks
        assert push(world, box.id, 0.8, direction=(1.0, 0.0)).ok
        world.step()
        box_pos = world.objects[box.id].position
        assert box_pos.x > 0.4
        assert world.objects[pebble1.id].position == box_pos
        assert world.objects[pebble2.id].position == box_pos

        # 2. Grabbing, carrying, and placing the container moves inserted objects with it
        assert move(world, (1.0, 0.0), 0.5).ok
        assert grab(world, box.id).ok
        assert move(world, (0.0, 1.0), 1.0).ok
        assert place(world, (world.body.position.x + 0.3, world.body.position.y)).ok
        placed_box_pos = world.objects[box.id].position
        assert world.objects[pebble1.id].position == placed_box_pos
        assert world.objects[pebble2.id].position == placed_box_pos

        # 3. Separating the container extracts the inserted objects and restores box weight
        sep = separate(world, box.id)
        assert sep.ok
        assert world.objects[pebble1.id].level == "ground"
        assert world.objects[pebble2.id].level == "ground"
        assert math.isclose(world.objects[box.id].weight, 1.0, rel_tol=1e-6)
        assert world.objects[box.id].state.get("contents") is None
