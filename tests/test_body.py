"""Tests for T19: body state (energy, comfort), physical side effects (BodyModifiers), and drives."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arden.actions import (
    BodyModifiers,
    bind,
    eat,
    execute_action,
    grab,
    invoke_opaque_action,
    look,
    move,
    push,
    rest,
    strike,
)
from arden.config import BodyConfig, BodySettings, Settings, load_config
from arden.environment import (
    DEFAULT_BODY_CONFIG,
    GENTLE_BODY_CONFIG,
    HARSH_BODY_CONFIG,
    MIDNIGHT,
    NOON,
    BodyStateSystem,
    DriveState,
    activity_energy_cost,
    compute_body_modifiers,
    compute_memory_salience,
    eat_object,
    format_numeric_body_state,
)
from arden.perception import perceive
from arden.world import World

_BASE_PROPS = {
    "shape": "sphere",
    "height": 0.25,
    "volume": 0.008,
    "weight": 0.5,
    "rigidity": 0.8,
}

_FORBIDDEN_EVALUATIVE_WORDS = (
    "good",
    "bad",
    "hungry",
    "tired",
    "starving",
    "exhausted",
    "freezing",
    "soaked",
    "comfortable",
    "uncomfortable",
    "bored",
    "curious",
    "sleepy",
    "dead",
)


def test_energy_drains_slowly_over_time_and_faster_when_pushing_than_idle() -> None:
    """Energy drains slowly over time, faster with activity weighted by verb and force."""
    idle_world = World(seed=10)
    push_light_world = World(seed=10)
    push_heavy_world = World(seed=10)

    for w in (push_light_world, push_heavy_world):
        w.spawn(
            {**_BASE_PROPS, "shape": "cube", "weight": 2.0},
            {"x": 0.8, "y": 0.0},
        )

    # 1. Passive time drain vs pushing
    idle_world.step()
    assert idle_world.body.energy < 1.0
    passive_drop = 1.0 - idle_world.body.energy
    assert 0.0 < passive_drop <= 0.01

    res_light = push(push_light_world, "object_1", 0.2)
    assert res_light.ok is True
    push_light_world.step()

    res_heavy = push(push_heavy_world, "object_1", 1.0)
    assert res_heavy.ok is True
    push_heavy_world.step()

    # Pushing drains more energy than being idle, and higher force drains more than lower force
    assert push_light_world.body.energy < idle_world.body.energy
    assert push_heavy_world.body.energy < push_light_world.body.energy

    # Heavy physical verbs (push/strike) cost more than light verbs (look/speak)
    assert activity_energy_cost("push", force=0.8) > activity_energy_cost("move", speed=0.8)
    assert activity_energy_cost("strike", force=0.8) > activity_energy_cost("grab")
    assert activity_energy_cost("grab") > activity_energy_cost("look")
    assert activity_energy_cost("idle") == 0.0
    assert activity_energy_cost("rest") == 0.0


def test_energy_restored_by_rest_and_by_eating_edible_objects() -> None:
    """Rest (idle) and eating objects with an edible rule (berries) restore energy."""
    world = World(seed=11, body={"energy": 0.4, "comfort": 0.5})

    # Rest via verb or step(resting=True) restores energy
    before_rest = world.body.energy
    res_rest = rest(world, ticks=3)
    assert res_rest.ok is True
    assert world.body.energy > before_rest
    assert world.body.comfort > 0.5

    after_verb_rest = world.body.energy
    world.step(resting=True)
    assert world.body.energy > after_verb_rest

    # Spawn a berry object with an edible rule and an inedible rock
    world.body.energy = 0.5
    berries = world.spawn(
        {
            **_BASE_PROPS,
            "shape": "sphere",
            "weight": 0.1,
            "color": "red",
            "behavior": [{"verb": "edible", "energy": 0.25}],
        },
        {"x": 0.5, "y": 0.0},
    )
    rock = world.spawn(
        {**_BASE_PROPS, "weight": 1.0},
        {"x": 0.0, "y": 0.5},
    )

    # Inedible object cannot be eaten
    res_rock = eat(world, rock.id)
    assert res_rock.ok is False

    # Eating berries restores energy and consumes the berry object
    res_eat = eat(world, berries.id)
    assert res_eat.ok is True
    assert world.body.energy == pytest.approx(0.75)
    assert berries.id not in world.objects

    # Berry bush with multiple charges / opaque action_1 also restores energy
    bush = world.spawn(
        {
            **_BASE_PROPS,
            "fixed": True,
            "state": {"berries": 2},
            "behavior": [{"verb": "edible", "restore_energy": 0.15}],
        },
        {"x": 0.0, "y": -0.5},
    )
    res_bush = invoke_opaque_action(world, "action_1", target=bush.id)
    assert res_bush.ok is True
    assert world.body.energy == pytest.approx(0.90)
    assert world.objects[bush.id].state["berries"] == 1


def test_comfort_lowered_by_cold_wet_night_storms_and_low_energy() -> None:
    """Comfort is lowered by cold, wet, night temperature, storms, and low energy."""
    # Baseline warm daytime dry high-energy world stays at 1.0 comfort
    base = World(seed=20)
    base.set_time_of_day(NOON)
    base.step()
    assert base.body.comfort == pytest.approx(1.0)

    # 1. Cold lowers comfort
    cold_w = World(seed=20)
    cold_w.set_time_of_day(NOON)
    cold_w.step(cold=1.0)
    assert cold_w.body.comfort < base.body.comfort

    # 2. Wet (either explicit or standing in water tile) lowers comfort
    wet_w = World(seed=20)
    wet_w.set_time_of_day(NOON)
    wet_w.set_tile(0, 0, "water")
    wet_w.step()
    assert wet_w.body.comfort < base.body.comfort

    # 3. Night temperature lowers comfort compared to noon
    night_w = World(seed=20)
    night_w.set_time_of_day(MIDNIGHT)
    night_w.step()
    assert night_w.body.comfort < base.body.comfort

    # 4. Storm lowers comfort faster than clear weather
    storm_w = World(seed=20, weather="storm")
    storm_w.set_time_of_day(NOON)
    storm_w.step()
    assert storm_w.body.comfort < cold_w.body.comfort

    # 5. Low energy lowers comfort
    low_energy_w = World(seed=20, body={"energy": 0.1, "comfort": 1.0})
    low_energy_w.set_time_of_day(NOON)
    low_energy_w.step()
    assert low_energy_w.body.comfort < base.body.comfort


def test_comfort_restored_by_shelter_dryness_warmth_and_rest() -> None:
    """Comfort is restored by shelter, dryness, warmth, and rest."""
    # Shelter protects from storm/night and restores comfort
    exposed = World(seed=21, body={"comfort": 0.5}, weather="storm")
    sheltered = World(seed=21, body={"comfort": 0.5}, weather="storm")
    # Build an agent-made shelter next to the agent in `sheltered`
    sheltered.spawn(
        {
            **_BASE_PROPS,
            "shape": "flat",
            "height": 1.2,
            "volume": 0.2,
            "hollow": True,
            "state": {"made_by_agent": True},
        },
        {"x": 0.3, "y": 0.0},
    )
    exposed.step()
    sheltered.step()
    assert sheltered.body.comfort > exposed.body.comfort

    # Dryness + warmth + rest restore comfort from a low level
    recover_w = World(seed=22, body={"energy": 0.8, "comfort": 0.4})
    recover_w.set_time_of_day(NOON)
    before = recover_w.body.comfort
    recover_w.step(sheltered=True, dry=True, warm=True, resting=True)
    assert recover_w.body.comfort > before


def test_side_effects_gradual_curves_for_low_energy_cold_wet_and_low_comfort() -> None:
    """Side effects fill BodyModifiers with gradual curves across energy, cold, wet, and comfort."""
    # 1. Low energy -> gradual reduction in speed and push/strike force
    mods_high_e = compute_body_modifiers(energy=1.0, comfort=1.0)
    mods_mid_e = compute_body_modifiers(energy=0.5, comfort=1.0)
    mods_low_e = compute_body_modifiers(energy=0.1, comfort=1.0)
    assert mods_high_e.speed > mods_mid_e.speed > mods_low_e.speed > 0.0
    assert mods_high_e.force > mods_mid_e.force > mods_low_e.force > 0.0

    # Verify in World: low-energy agent moves slower and pushes/strikes weaker than full-energy agent
    w_full = World(seed=30, body={"energy": 1.0})
    w_low = World(seed=30, body={"energy": 0.1})
    move(w_full, "east", 1.0)
    move(w_low, "east", 1.0)
    assert w_full.body.position.x > w_low.body.position.x > 0.0

    b_full = w_full.spawn({**_BASE_PROPS, "weight": 0.5}, {"x": w_full.body.position.x + 0.5, "y": 0.0})
    b_low = w_low.spawn({**_BASE_PROPS, "weight": 0.5}, {"x": w_low.body.position.x + 0.5, "y": 0.0})
    push(w_full, b_full.id, 0.8)
    push(w_low, b_low.id, 0.8)
    assert w_full.objects[b_full.id].velocity.x > w_low.objects[b_low.id].velocity.x > 0.0

    # 2. Cold -> gradual increase in grab and bind failure chance (lower dexterity)
    mods_warm = compute_body_modifiers(cold=0.0)
    mods_chilly = compute_body_modifiers(cold=0.5)
    mods_freezing = compute_body_modifiers(cold=1.0)
    assert 0.0 == mods_warm.grab_fail_chance < mods_chilly.grab_fail_chance < mods_freezing.grab_fail_chance
    assert 0.0 == mods_warm.bind_fail_chance < mods_chilly.bind_fail_chance < mods_freezing.bind_fail_chance
    assert 1.0 == mods_warm.dexterity > mods_chilly.dexterity > mods_freezing.dexterity

    # 3. Wet -> slower movement and held items can slip
    mods_dry = compute_body_modifiers(wet=0.0)
    mods_damp = compute_body_modifiers(wet=0.5)
    mods_soaked = compute_body_modifiers(wet=1.0)
    assert mods_dry.speed > mods_damp.speed > mods_soaked.speed
    assert 0.0 == mods_dry.slip_chance < mods_damp.slip_chance < mods_soaked.slip_chance

    # Verify wet held item slipping over multiple moves
    slip_world = World(seed=31, preset="harsh")
    slip_world.wet = 1.0
    item = slip_world.spawn({**_BASE_PROPS, "weight": 0.3}, {"x": 0.2, "y": 0.0})
    slip_world.body.holding = item.id
    slipped = False
    for _ in range(25):
        res = move(slip_world, "east", 0.5)
        if res.data.get("slipped") == item.id or slip_world.body.holding is None:
            slipped = True
            break
    assert slipped is True
    assert slip_world.body.holding is None

    # 4. Very low comfort -> smaller sensing radius in perceive()
    mods_comfy = compute_body_modifiers(comfort=0.9)
    mods_uncomfy = compute_body_modifiers(comfort=0.2)
    mods_miserable = compute_body_modifiers(comfort=0.0)
    assert mods_comfy.sensing_radius_scale == 1.0
    assert 1.0 > mods_uncomfy.sensing_radius_scale > mods_miserable.sensing_radius_scale > 0.0

    perc_world = World(seed=32)
    perc_world.set_time_of_day(NOON)
    # Place an object at 8.5m (within 10m normal sensing radius, outside shrunk radius at 0 comfort)
    far_obj = perc_world.spawn(_BASE_PROPS, {"x": 8.5, "y": 0.0})
    perc_normal = perceive(perc_world)
    assert any(o.id == far_obj.id for o in perc_normal.objects)

    perc_world.body.comfort = 0.0
    perc_low_comfort = perceive(perc_world)
    assert all(o.id != far_obj.id for o in perc_low_comfort.objects)


def test_wet_and_cold_raises_grab_and_bind_failures() -> None:
    """Wet + cold raises grab failure rate compared to dry/warm, cold-only, or wet-only."""
    mods_dry_warm = compute_body_modifiers(cold=0.0, wet=0.0)
    mods_cold_only = compute_body_modifiers(cold=0.8, wet=0.0)
    mods_wet_only = compute_body_modifiers(cold=0.0, wet=0.8)
    mods_wet_and_cold = compute_body_modifiers(cold=0.8, wet=0.8)

    assert mods_wet_and_cold.grab_fail_chance > mods_cold_only.grab_fail_chance > mods_dry_warm.grab_fail_chance
    assert mods_wet_and_cold.grab_fail_chance > mods_wet_only.grab_fail_chance > mods_dry_warm.grab_fail_chance

    # Empirical grab failure check across deterministic trials
    def _count_grab_failures(cold_val: float, wet_val: float, trials: int = 60) -> int:
        world = World(seed=42)
        world.cold = cold_val
        world.wet = wet_val
        ball = world.spawn({**_BASE_PROPS, "weight": 0.4}, {"x": 0.5, "y": 0.0})
        failures = 0
        for _ in range(trials):
            world.body.holding = None
            res = grab(world, ball.id)
            if not res.ok:
                failures += 1
        return failures

    assert _count_grab_failures(0.0, 0.0) == 0
    cold_Visual_failures = _count_grab_failures(0.9, 0.0)
    wet_cold_failures = _count_grab_failures(0.9, 0.9)
    assert cold_Visual_failures > 0
    assert wet_cold_failures > cold_Visual_failures

    # Cold also causes bind failures
    bind_world = World(seed=43)
    bind_world.cold = 1.0
    a = bind_world.spawn({**_BASE_PROPS, "shape": "long"}, {"x": 0.2, "y": 0.0})
    b = bind_world.spawn({**_BASE_PROPS, "shape": "sphere"}, {"x": 0.3, "y": 0.0})
    vine = bind_world.spawn({**_BASE_PROPS, "binding": True}, {"x": 0.2, "y": 0.2})
    bind_failures = 0
    for _ in range(40):
        if a.id not in bind_world.objects:
            break
        res = bind(bind_world, a.id, b.id, vine.id)
        if not res.ok:
            bind_failures += 1
    assert bind_failures > 0


def test_gentle_and_harsh_presets_and_never_forces_sleep_or_death(tmp_path: Path) -> None:
    """Gentle preset produces smaller effects than harsh, and nothing ever forces sleep or death."""
    gentle_mods = compute_body_modifiers(
        energy=0.1,
        comfort=0.05,
        cold=0.9,
        wet=0.9,
        preset="gentle",
    )
    default_mods = compute_body_modifiers(
        energy=0.1,
        comfort=0.05,
        cold=0.9,
        wet=0.9,
        preset="default",
    )
    harsh_mods = compute_body_modifiers(
        energy=0.1,
        comfort=0.05,
        cold=0.9,
        wet=0.9,
        preset="harsh",
    )

    # Gentle produces smaller side effects (higher speed/force/dexterity/sensing, lower fail/slip) than harsh
    assert gentle_mods.speed > default_mods.speed > harsh_mods.speed > 0.0
    assert gentle_mods.force > default_mods.force > harsh_mods.force > 0.0
    assert gentle_mods.dexterity > default_mods.dexterity > harsh_mods.dexterity > 0.0
    assert gentle_mods.grab_fail_chance < default_mods.grab_fail_chance < harsh_mods.grab_fail_chance
    assert gentle_mods.bind_fail_chance < default_mods.bind_fail_chance < harsh_mods.bind_fail_chance
    assert gentle_mods.slip_chance < default_mods.slip_chance < harsh_mods.slip_chance
    assert gentle_mods.sensing_radius_scale > default_mods.sensing_radius_scale > harsh_mods.sensing_radius_scale > 0.0

    # TOML loading supports [body] preset = "gentle" and "harsh"
    cfg_path = tmp_path / "arden.toml"
    cfg_path.write_text('[body]\npreset = "gentle"\n', encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.body.preset == "gentle"
    assert loaded.body.low_energy_speed_effect == GENTLE_BODY_CONFIG.low_energy_speed_effect
    assert loaded.body.low_energy_speed_effect < HARSH_BODY_CONFIG.low_energy_speed_effect

    # Nothing ever forces sleep or death even at 0 energy and 0 comfort in harsh mode
    extreme_world = World(
        seed=50,
        preset="harsh",
        body={"energy": 0.0, "comfort": 0.0},
        weather="storm",
    )
    extreme_world.set_time_of_day(MIDNIGHT)
    for _ in range(20):
        extreme_world.step(cold=1.0, wet=1.0, storm=1.0)

    assert extreme_world.body.energy == 0.0
    assert extreme_world.body.comfort == 0.0
    # Agent can still move and act at 0 energy and 0 comfort
    res_move = move(extreme_world, "east", 1.0)
    assert res_move.ok is True
    assert extreme_world.body.position.x > 0.0


def test_drives_boredom_and_curiosity_gate_thresholds_and_memory_salience() -> None:
    """Boredom rises without novelty/surprise and falls with it; curiosity accumulates and feeds gate & salience."""
    world = World(seed=60)
    assert world.drives.boredom == 0.0
    assert world.drives.curiosity == 0.0

    # Stepping with no novelty or surprise raises boredom
    for _ in range(10):
        world.step(novelty=0.0, surprise=0.0)
    boredom_after_idle = world.drives.boredom
    assert boredom_after_idle > 0.15
    assert world.drives.curiosity == 0.0

    # Novelty and surprise reduce boredom and accumulate curiosity reward
    world.step(novelty=0.5, surprise=0.6)
    assert world.drives.boredom < boredom_after_idle
    assert world.drives.curiosity > 0.0
    assert world.drives.last_curiosity_reward > 0.0

    # Gate wake thresholds expose boredom, curiosity, low_energy, low_comfort
    thresholds = world.drives.thresholds(world.body_config)
    assert thresholds.boredom == world.body_config.boredom_wake_threshold
    assert thresholds.curiosity == world.body_config.curiosity_wake_threshold

    world.drives.boredom = thresholds.boredom + 0.05
    assert "boredom" in world.gate_triggers()
    assert world.should_wake_gate is True

    world.drives.boredom = 0.0
    world.drives.curiosity = thresholds.curiosity + 0.1
    assert "curiosity" in world.gate_triggers()
    assert world.should_wake_gate is True

    world.drives.reset_curiosity()
    assert world.drives.curiosity == 0.0
    world.body.energy = 0.1
    world.body.comfort = 0.1
    assert "low_energy" in world.gate_triggers()
    assert "low_comfort" in world.gate_triggers()

    # Memory salience rises with curiosity/surprise/novelty and with low comfort
    baseline_salience = compute_memory_salience(
        drives=DriveState(boredom=0.0, curiosity=0.0),
        body={"energy": 1.0, "comfort": 1.0},
    )
    curious_salience = compute_memory_salience(
        drives=DriveState(boredom=0.0, curiosity=0.8),
        body={"energy": 1.0, "comfort": 1.0},
        surprise=0.7,
        novelty=0.5,
    )
    uncomfortable_salience = compute_memory_salience(
        drives=DriveState(boredom=0.0, curiosity=0.0),
        body={"energy": 0.2, "comfort": 0.1},
    )
    assert curious_salience > baseline_salience
    assert uncomfortable_salience > baseline_salience
    assert 0.0 <= world.memory_salience <= 1.0


def test_percept_and_body_text_contain_plain_numbers_only_and_no_forbidden_words() -> None:
    """Percept and body formatting output plain numbers only and never words like good, bad, hungry, or tired."""
    world = World(seed=70, body={"energy": 0.12, "comfort": 0.08})
    world.set_time_of_day(MIDNIGHT)
    world.spawn(
        {**_BASE_PROPS, "color": "blue", "behavior": [{"verb": "edible", "energy": 0.2}]},
        {"x": 0.5, "y": 0.0},
    )

    percept = perceive(world)
    assert percept.body.energy == pytest.approx(0.12)
    assert percept.body.comfort == pytest.approx(0.08)

    outputs = [
        percept.to_text(),
        percept.to_json(),
        format_numeric_body_state(world.body),
    ]
    for text in outputs:
        lower = text.lower()
        for word in _FORBIDDEN_EVALUATIVE_WORDS:
            assert re.search(rf"\b{re.escape(word)}\b", lower) is None, (
                f"Forbidden word {word!r} found in output: {text!r}"
            )
    assert "energy=0.12" in percept.to_text()
    assert "comfort=0.08" in percept.to_text()
