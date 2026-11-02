"""Tests for T18: Weather manual events, probabilistic automation, shelter, and world effects."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from arden.config import load_config
from arden.environment import (
    DEFAULT_WEATHER_TRANSITIONS,
    HOLLOW_SHELTER_EXPOSURE,
    TREE_SHELTER_EXPOSURE,
    WEATHER_TYPES,
    WeatherChangedEvent,
    WeatherConfig,
    WeatherSystem,
    compute_shelter_exposure,
    set_weather,
)
from arden.perception import perceive
from arden.world import ObjectChangedEvent, ObjectMovedEvent, Position, World

_LEAF_PROPS: dict[str, Any] = {
    "shape": "flat",
    "height": 0.02,
    "volume": 0.0002,
    "weight": 0.01,
    "rigidity": 0.2,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "green",
    "fixed": False,
    "state": {"wet": False},
    "behavior": [],
}

_STONE_PROPS: dict[str, Any] = {
    "shape": "sphere",
    "height": 0.2,
    "volume": 0.004,
    "weight": 0.6,
    "rigidity": 0.9,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "gray",
    "fixed": False,
    "state": {"wet": False},
    "behavior": [],
}

_TREE_PROPS: dict[str, Any] = {
    "shape": "long",
    "height": 3.0,
    "volume": 0.5,
    "weight": 9.5,
    "rigidity": 0.9,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "brown",
    "fixed": True,
    "state": {"wet": False},
    "behavior": [],
}

_HOLLOW_TRUNK_PROPS: dict[str, Any] = {
    "shape": "cube",
    "height": 1.0,
    "volume": 0.25,
    "weight": 4.0,
    "rigidity": 0.85,
    "binding": False,
    "hollow": True,
    "fragile": False,
    "color": "brown",
    "fixed": True,
    "state": {"wet": False},
    "behavior": [],
}


def test_weather_types_and_intensity_validation() -> None:
    assert set(WEATHER_TYPES) == {"clear", "rain", "wind", "cold", "storm"}

    world = World(seed=1)
    for w_type in WEATHER_TYPES:
        state = world.set_weather(w_type, intensity=0.65, duration=10)
        assert state.type == w_type
        assert world.weather.type == w_type
        assert world.weather_type == w_type
        assert world.weather.intensity == pytest.approx(0.65)
        assert 0.0 <= world.weather.intensity <= 1.0

    # Boundary intensities 0.0 and 1.0 are valid
    world.set_weather("rain", intensity=0.0, duration=5)
    assert world.weather.intensity == 0.0
    world.set_weather("storm", intensity=1.0, duration=5)
    assert world.weather.intensity == 1.0

    # Invalid types and intensities are rejected
    with pytest.raises(ValueError, match="unknown weather type"):
        world.set_weather("blizzard", intensity=0.5, duration=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="intensity"):
        world.set_weather("rain", intensity=-0.1, duration=10)
    with pytest.raises(ValueError, match="intensity"):
        world.set_weather("rain", intensity=1.1, duration=10)
    with pytest.raises(ValueError, match="duration"):
        world.set_weather("rain", intensity=0.5, duration=0)


def test_automated_weather_config_loads_from_toml_and_validates(tmp_path: Path) -> None:
    toml_file = tmp_path / "weather.toml"
    toml_file.write_text(
        """
[world]
seed = 10

[weather]
seed = 777
ticks_per_hour = 20
paused = false

[weather.chance]
clear = 0.5
rain = 0.4
wind = 0.3
cold = 0.2
storm = 0.25

[weather.duration]
clear = [20, 40]
rain = [15, 30]
wind = [10, 25]
cold = [15, 30]
storm = [10, 20]

[weather.intensity]
clear = [0.0, 0.0]
rain = [0.3, 0.8]
wind = [0.2, 0.7]
cold = [0.4, 0.9]
storm = [0.6, 1.0]

[weather.transitions]
clear = ["rain", "wind", "cold"]
rain = ["clear", "storm", "wind"]
wind = ["clear", "rain"]
cold = ["clear", "storm"]
storm = ["rain", "clear"]
""",
        encoding="utf-8",
    )
    settings = load_config(toml_file)
    assert settings.weather.seed == 777
    assert settings.weather.ticks_per_hour == 20
    assert settings.weather.chance["storm"] == pytest.approx(0.25)
    assert settings.weather.duration["rain"] == (15, 30)
    assert settings.weather.intensity["storm"] == (0.6, 1.0)
    assert settings.weather.can_transition("rain", "storm") is True
    assert settings.weather.can_transition("clear", "storm") is False


def test_same_seed_gives_same_weather_sequence_over_10000_ticks() -> None:
    cfg = WeatherConfig(
        seed=424242,
        ticks_per_hour=10,
        duration={
            "clear": (8, 20),
            "rain": (6, 15),
            "wind": (5, 14),
            "cold": (6, 15),
            "storm": (5, 12),
        },
    )
    sys_a = WeatherSystem(config=cfg)
    sys_b = WeatherSystem(config=cfg)
    sys_diff = WeatherSystem(config=cfg.model_copy(update={"seed": 999999}))

    seq_a: list[tuple[str, float, int]] = []
    seq_b: list[tuple[str, float, int]] = []
    seq_diff: list[tuple[str, float, int]] = []

    for tick in range(1, 10_001):
        sa = sys_a.step(tick)
        sb = sys_b.step(tick)
        sd = sys_diff.step(tick)
        seq_a.append((sa.type, sa.intensity, sa.remaining_ticks))
        seq_b.append((sb.type, sb.intensity, sb.remaining_ticks))
        seq_diff.append((sd.type, sd.intensity, sd.remaining_ticks))

    assert seq_a == seq_b
    assert seq_a != seq_diff
    # Ensure all 5 weather types appeared over 10,000 ticks
    seen_types = {item[0] for item in seq_a}
    assert seen_types == {"clear", "rain", "wind", "cold", "storm"}


def test_forbidden_transitions_never_occur_over_10000_ticks() -> None:
    cfg = WeatherConfig(
        seed=1337,
        ticks_per_hour=8,
        duration={
            "clear": (4, 10),
            "rain": (4, 10),
            "wind": (4, 10),
            "cold": (4, 10),
            "storm": (4, 10),
        },
        transitions=DEFAULT_WEATHER_TRANSITIONS,
    )
    assert "storm" not in cfg.transitions["clear"]
    assert "storm" in cfg.transitions["rain"]

    world = World(seed=1, weather=cfg)
    weather_events: list[WeatherChangedEvent] = []
    world.subscribe("weather_changed", weather_events.append)  # type: ignore[arg-type]

    prev_type = world.weather.type
    observed_transitions: set[tuple[str, str]] = set()

    for _ in range(10_000):
        world.step()
        curr_type = world.weather.type
        if curr_type != prev_type:
            observed_transitions.add((prev_type, curr_type))
            assert curr_type in cfg.transitions[prev_type], (
                f"Forbidden transition {prev_type} -> {curr_type} occurred"
            )
            prev_type = curr_type

    assert ("clear", "storm") not in observed_transitions
    assert ("rain", "storm") in observed_transitions
    assert len(weather_events) > 100
    for ev in weather_events:
        assert ev.source == "automated"
        assert ev.weather_type in cfg.transitions[ev.previous_weather_type]
        lo_dur, hi_dur = cfg.duration[ev.weather_type]
        assert lo_dur <= ev.duration <= hi_dur
        lo_int, hi_int = cfg.intensity[ev.weather_type]
        assert lo_int <= ev.intensity <= hi_int


def test_manual_event_overrides_automation_then_automation_resumes_and_emits_events() -> None:
    cfg = WeatherConfig(
        seed=99,
        ticks_per_hour=5,
        duration={
            "clear": (4, 6),
            "rain": (4, 6),
            "wind": (4, 6),
            "cold": (4, 6),
            "storm": (4, 6),
        },
    )
    world = World(seed=99, weather=cfg)
    events: list[WeatherChangedEvent] = []
    world.subscribe("weather_changed", events.append)  # type: ignore[arg-type]

    # Step until an automated transition happens
    for _ in range(15):
        world.step()
    assert len(events) >= 1
    assert all(e.source == "automated" for e in events)

    count_before = len(events)
    # Trigger immediate manual override for 5 ticks via top-level set_weather(...)
    state = set_weather("storm", 0.9, 5)
    assert state.type == "storm"
    assert state.intensity == pytest.approx(0.9)
    assert state.source == "manual"
    assert world.weather.is_manual is True
    assert len(events) == count_before + 1
    manual_ev = events[-1]
    assert manual_ev.type == "weather_changed"
    assert manual_ev.source == "manual"
    assert manual_ev.weather_type == "storm"
    assert manual_ev.intensity == pytest.approx(0.9)
    assert manual_ev.duration == 5

    # During ticks 1..4 of the manual event, weather remains storm (manual)
    for step_idx in range(4):
        world.step()
        assert world.weather.type == "storm", f"failed at step {step_idx}"
        assert world.weather.intensity == pytest.approx(0.9)
        assert world.weather.source == "manual"
        assert world.weather.is_manual is True

    # On the 5th tick, the manual override finishes at end of tick and automation resumes
    world.step()
    assert world.weather.is_automated is True
    assert world.weather.source == "automated"
    assert len(events) == count_before + 2
    resume_ev = events[-1]
    assert resume_ev.type == "weather_changed"
    assert resume_ev.source == "automated"
    assert resume_ev.previous_weather_type == "storm"


def test_scheduled_manual_weather_and_pausing_automation() -> None:
    cfg = WeatherConfig(
        seed=7,
        ticks_per_hour=5,
        duration={
            "clear": (3, 4),
            "rain": (3, 4),
            "wind": (3, 4),
            "cold": (3, 4),
            "storm": (3, 4),
        },
    )
    world = World(seed=7, weather=cfg)
    events: list[WeatherChangedEvent] = []
    world.subscribe("weather_changed", events.append)  # type: ignore[arg-type]

    # Pause automation: weather should not change on its own even after many ticks
    world.pause_weather_automation()
    assert world.weather.paused is True
    for _ in range(20):
        world.step()
    assert world.weather.type == "clear"
    assert len(events) == 0

    # Schedule a manual rain event to start at tick 25 for 3 ticks
    scheduled = world.set_weather("rain", intensity=0.75, duration=3, start_at=25)
    assert scheduled.type == "rain"
    assert world.weather.type == "clear"
    assert len(world.weather.scheduled) == 1
    assert len(events) == 0

    # Step ticks 21..24: still clear
    for _ in range(4):
        world.step()
    assert world.tick == 24
    assert world.weather.type == "clear"
    assert len(events) == 0

    # Tick 25: scheduled manual weather activates at start of tick 25!
    world.step()
    assert world.tick == 25
    assert world.weather.type == "rain"
    assert world.weather.intensity == pytest.approx(0.75)
    assert world.weather.source == "manual"
    assert len(events) == 1
    assert events[0].source == "manual"
    assert events[0].tick == 25
    assert events[0].weather_type == "rain"

    # Ticks 26 and 27 remain in manual rain
    world.step()
    assert world.tick == 26
    assert world.weather.type == "rain"
    world.step()
    assert world.tick == 27
    assert world.weather.type == "rain"

    # Tick 28: 3 ticks of manual rain have elapsed, so automation resumes
    world.step()
    assert world.tick == 28
    assert world.weather.source == "automated"
    assert len(events) == 2
    assert events[1].source == "automated"

    # Resume automation and verify automated transitions happen again
    world.resume_weather_automation()
    assert world.weather.paused is False
    for _ in range(30):
        world.step()
    assert len(events) > 2


def test_rain_wets_exposed_objects_and_agent_and_makes_surfaces_slippery() -> None:
    # World 1: Clear weather baseline for push distance
    world_clear = World(seed=10)
    stone_clear = world_clear.spawn(_STONE_PROPS, (0.0, 0.0))
    world_clear.push(stone_clear.id, (1.0, 0.0), force=0.6)
    for _ in range(15):
        world_clear.step()
    clear_dist = world_clear.objects[stone_clear.id].position.x
    assert world_clear.body.wet is False
    assert world_clear.objects[stone_clear.id].state["wet"] is False
    assert world_clear.is_slippery is False
    assert world_clear.surface_slipperiness == 0.0

    # World 2: Rain weather makes exposed objects and agent wet, and surfaces slippery
    world_rain = World(seed=10)
    stone_rain = world_rain.spawn(_STONE_PROPS, (0.0, 0.0))
    changed_events: list[ObjectChangedEvent] = []
    world_rain.subscribe("object_changed", changed_events.append)  # type: ignore[arg-type]

    world_rain.set_weather("rain", intensity=0.8, duration=30)
    assert world_rain.is_slippery is True
    assert world_rain.surface_slipperiness == pytest.approx(0.8)
    assert world_rain.surface_friction_at(stone_rain) < world_clear.surface_friction_at(stone_clear)

    world_rain.push(stone_rain.id, (1.0, 0.0), force=0.6)
    for _ in range(15):
        world_rain.step()

    rain_dist = world_rain.objects[stone_rain.id].position.x
    assert world_rain.body.wet is True
    assert world_rain.body.wetness > 0.0
    assert world_rain.objects[stone_rain.id].state["wet"] is True
    assert any(
        ev.object_id == stone_rain.id and ev.changes.get("state", {}).get("wet") is True
        for ev in changed_events
    )
    # Stone slides significantly farther on rain-slicked surface than on dry surface
    assert rain_dist > clear_dist


def test_wind_moves_leaf_but_not_stone_each_tick() -> None:
    world = World(seed=21)
    leaf = world.spawn(_LEAF_PROPS, (0.0, 0.0))
    stone = world.spawn(_STONE_PROPS, (0.0, 3.0))
    moved_events: list[ObjectMovedEvent] = []
    world.subscribe("object_moved", moved_events.append)  # type: ignore[arg-type]

    world.set_weather("wind", intensity=0.7, duration=10, wind_direction=(1.0, 0.0))

    prev_leaf_x = world.objects[leaf.id].position.x
    for _ in range(5):
        world.step()
        curr_leaf = world.objects[leaf.id]
        curr_stone = world.objects[stone.id]
        assert curr_leaf.position.x > prev_leaf_x
        assert curr_stone.position == Position(x=0.0, y=3.0)
        prev_leaf_x = curr_leaf.position.x

    moved_ids = {ev.object_id for ev in moved_events}
    assert leaf.id in moved_ids
    assert stone.id not in moved_ids


def test_cold_and_storm_lower_temperature_and_storm_lowers_light_stacking_with_night() -> None:
    world = World(seed=5)
    world.set_time_of_day(0.5)  # Noon
    noon_clear_temp = world.ambient_temperature
    noon_clear_light = world.light_level
    assert noon_clear_light == pytest.approx(1.0)

    # Cold lowers ambient temperature without reducing light
    world.set_weather("cold", intensity=0.8, duration=10)
    noon_cold_temp = world.ambient_temperature
    assert noon_cold_temp < noon_clear_temp
    assert world.light_level == pytest.approx(noon_clear_light)

    # Storm lowers both ambient temperature and light at noon
    world.set_weather("storm", intensity=0.8, duration=10)
    noon_storm_temp = world.ambient_temperature
    noon_storm_light = world.light_level
    assert noon_storm_temp < noon_clear_temp
    assert noon_storm_light < noon_clear_light

    # At dusk/night, storm light reduction stacks with night darkness
    world.set_weather("clear", intensity=0.0, duration=10)
    world.set_time_of_day(0.25)  # Dawn/dusk transition
    dusk_clear_light = world.light_level
    assert dusk_clear_light < noon_clear_light

    world.set_weather("storm", intensity=0.8, duration=10)
    dusk_storm_light = world.light_level
    assert dusk_storm_light < dusk_clear_light
    assert dusk_storm_light < noon_storm_light

    # At midnight, storm stacks with night light as well
    world.set_weather("clear", intensity=0.0, duration=10)
    world.set_time_of_day(0.15)
    night_clear_light = world.light_level
    world.set_weather("storm", intensity=0.8, duration=10)
    night_storm_light = world.light_level
    assert night_storm_light < night_clear_light
    assert night_storm_light == pytest.approx(night_clear_light * world.weather.light_multiplier)


def test_shelter_under_tree_and_inside_hollow_object_reduces_rain_and_wind_exposure() -> None:
    world = World(seed=33, body={"position": {"x": 0.0, "y": 0.0}})
    tree = world.spawn(_TREE_PROPS, (10.0, 0.0))
    hollow = world.spawn(_HOLLOW_TRUNK_PROPS, (20.0, 0.0))

    exposed_leaf = world.spawn(_LEAF_PROPS, (0.0, 2.0))
    tree_leaf = world.spawn(_LEAF_PROPS, (10.8, 0.0))
    inside_leaf = world.spawn(_LEAF_PROPS, (0.0, 0.0))
    # Place inside_leaf inside the hollow object
    world.move_agent((20.0, 0.0))
    world.body.holding = inside_leaf.id
    res = world.insert(inside_leaf.id, hollow.id)
    assert res.ok is True
    assert world.objects[inside_leaf.id].level == f"inside:{hollow.id}"

    # Verify exposure metrics
    exp_open = compute_shelter_exposure(world, world.objects[exposed_leaf.id])
    exp_tree = compute_shelter_exposure(world, world.objects[tree_leaf.id])
    exp_hollow = compute_shelter_exposure(world, world.objects[inside_leaf.id])

    assert exp_open == (1.0, 1.0)
    assert exp_tree == (TREE_SHELTER_EXPOSURE, TREE_SHELTER_EXPOSURE)
    assert exp_tree[0] < exp_open[0]
    assert exp_hollow == (HOLLOW_SHELTER_EXPOSURE, HOLLOW_SHELTER_EXPOSURE)
    assert exp_hollow[0] < exp_tree[0]

    # Put agent under the tree and step during a storm (both rain and wind active)
    world.move_agent((10.5, 0.0))
    assert world.is_sheltered(world.body) is True
    assert world.is_sheltered(exposed_leaf.id) is False
    assert world.is_sheltered(tree_leaf.id) is True
    assert world.is_sheltered(inside_leaf.id) is True

    world.set_weather("storm", intensity=0.8, duration=10, wind_direction=(0.0, 1.0))
    world.step()

    # Exposed leaf gets wet and is pushed strongly by wind;
    # tree-sheltered leaf stays dry and moves much less;
    # hollow-sheltered leaf stays completely dry and does not move at all.
    leaf_open_after = world.objects[exposed_leaf.id]
    leaf_tree_after = world.objects[tree_leaf.id]
    leaf_inside_after = world.objects[inside_leaf.id]

    assert leaf_open_after.state["wet"] is True
    assert leaf_tree_after.state["wet"] is False
    assert leaf_inside_after.state["wet"] is False
    assert world.body.wet is False

    open_dy = leaf_open_after.position.y - 2.0
    tree_dy = leaf_tree_after.position.y - 0.0
    inside_dy = leaf_inside_after.position.y - 0.0

    assert open_dy > 0.05
    assert 0.0 < tree_dy < open_dy * 0.35
    assert math.isclose(inside_dy, 0.0, abs_tol=1e-9)


def test_perception_never_leaks_weather_type_or_source_labels() -> None:
    world = World(seed=12)
    leaf = world.spawn(_LEAF_PROPS, (0.5, 0.0))
    world.set_weather("storm", intensity=0.8, duration=10)
    world.step()

    percept = perceive(world)
    text = percept.to_text().lower()
    # The agent perceives the wet object state and moved object, and darker light,
    # but never sees raw weather type names or "manual"/"automated".
    for forbidden in ("storm", "weather_changed", "automated", "manual"):
        assert forbidden not in text
    perceived_leaf = next(o for o in percept.objects if o.id == leaf.id)
    assert perceived_leaf.state.get("wet") is True
