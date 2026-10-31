"""Tests for the continuous day/night cycle: light, visibility, temperature, and creator controls."""

from __future__ import annotations

import re
from typing import Any

import pytest

from arden.environment.daynight import (
    DAWN,
    DUSK,
    MIDNIGHT,
    NOON,
    DayNightConfig,
    DayNightCycle,
    coarse_light_level,
    compute_ambient_temperature,
    compute_light_level,
    compute_visibility_factor,
)
from arden.perception import PerceptionTracker, perceive
from arden.world import CycleFrozenEvent, TimeOfDayChangedEvent, World, WorldEvent

_SAMPLE_OBJ: dict[str, Any] = {
    "shape": "sphere",
    "height": 0.3,
    "volume": 0.008,
    "weight": 0.5,
    "rigidity": 0.5,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "crimson",
    "fixed": False,
    "state": {"wet": False},
}


def test_light_and_temperature_follow_smooth_curves_and_noon_gt_dusk_gt_midnight() -> None:
    cfg = DayNightConfig(
        day_length=200,
        daylight_share=0.6,
        min_visibility=0.3,
        temperature_swing=0.45,
        base_temperature=1.0,
    )
    cycle = DayNightCycle(cfg)

    light_noon = cycle.light_at(NOON)
    light_dusk = cycle.light_at(DUSK)
    light_dawn = cycle.light_at(DAWN)
    light_midnight = cycle.light_at(MIDNIGHT)

    assert light_noon == pytest.approx(1.0)
    assert light_midnight == pytest.approx(0.0)
    assert light_noon > light_dusk > light_midnight
    assert light_dawn == pytest.approx(light_dusk)

    # Ambient temperature is exposed and follows the smooth curve
    temp_noon = cycle.temperature_at(NOON)
    temp_dusk = cycle.temperature_at(DUSK)
    temp_midnight = cycle.temperature_at(MIDNIGHT)

    assert temp_noon == pytest.approx(1.0)
    assert temp_midnight == pytest.approx(1.0 - 0.45)
    assert temp_noon - temp_midnight == pytest.approx(cfg.temperature_swing)
    assert temp_noon > temp_dusk > temp_midnight

    # Smoothness across a full cycle: no abrupt jumps (hard switches) between consecutive ticks
    cycle.set_time_of_day(MIDNIGHT)
    lights: list[float] = []
    temps: list[float] = []
    for _ in range(cfg.day_length):
        lights.append(cycle.light)
        temps.append(cycle.ambient_temperature)
        cycle.step()

    max_light_step = max(
        abs(lights[(i + 1) % len(lights)] - lights[i]) for i in range(len(lights))
    )
    max_temp_step = max(
        abs(temps[(i + 1) % len(temps)] - temps[i]) for i in range(len(temps))
    )
    assert max_light_step < 0.05
    assert max_temp_step < 0.05 * cfg.temperature_swing

    # Strictly monotonic rise from midnight to noon and fall from noon to midnight
    half = cfg.day_length // 2
    for i in range(half):
        assert lights[i + 1] > lights[i]
        assert temps[i + 1] > temps[i]
    for i in range(half, cfg.day_length - 1):
        assert lights[i + 1] < lights[i]
        assert temps[i + 1] < temps[i]


def test_daylight_share_scales_daytime_fraction_smoothly() -> None:
    short_day = DayNightCycle(day_length=400, daylight_share=0.3)
    equal_day = DayNightCycle(day_length=400, daylight_share=0.5)
    long_day = DayNightCycle(day_length=400, daylight_share=0.7)

    # At dusk (0.75), longer daylight_share produces higher light
    assert short_day.light_at(DUSK) < equal_day.light_at(DUSK) < long_day.light_at(DUSK)
    assert equal_day.light_at(DUSK) == pytest.approx(0.5)

    # Share of ticks with light >= 0.5 matches daylight_share
    for cyc, expected_share in ((short_day, 0.3), (equal_day, 0.5), (long_day, 0.7)):
        bright_ticks = sum(
            1 for i in range(cyc.day_length) if cyc.light_at(i / cyc.day_length) >= 0.5 - 1e-9
        )
        assert (bright_ticks / cyc.day_length) == pytest.approx(expected_share, abs=0.01)


def test_midnight_sensing_radius_equals_min_visibility_times_day_radius() -> None:
    day_radius = 12.0
    min_vis = 0.35
    world = World(
        seed=5,
        daynight=DayNightConfig(day_length=100, min_visibility=min_vis),
    )

    # Spawn objects around the midnight boundary (0.35 * 12.0 = 4.2m) and daytime boundary (12.0m)
    inside_midnight = world.spawn({**_SAMPLE_OBJ, "id": "object_1"}, {"x": 4.2, "y": 0.0})
    just_outside_midnight = world.spawn({**_SAMPLE_OBJ, "id": "object_2"}, {"x": 4.25, "y": 0.0})
    inside_day = world.spawn({**_SAMPLE_OBJ, "id": "object_3"}, {"x": 11.9, "y": 0.0})

    # At noon (0.5): visibility_factor == 1.0, effective radius == day_radius (12.0m)
    world.set_time_of_day(NOON)
    assert world.visibility_factor == pytest.approx(1.0)
    assert world.daynight.effective_sensing_radius(day_radius) == pytest.approx(day_radius)
    p_day = perceive(world, sensing_radius=day_radius)
    day_ids = {o.id for o in p_day.objects}
    assert day_ids == {inside_midnight.id, just_outside_midnight.id, inside_day.id}

    # At midnight (0.0): visibility_factor == min_visibility, effective radius == min_visibility * day_radius
    world.set_time_of_day(MIDNIGHT)
    assert world.visibility_factor == pytest.approx(min_vis)
    assert world.daynight.effective_sensing_radius(day_radius) == pytest.approx(min_vis * day_radius)
    p_midnight = perceive(world, sensing_radius=day_radius)
    midnight_ids = {o.id for o in p_midnight.objects}
    assert midnight_ids == {inside_midnight.id}


def test_color_fades_before_shape_and_size_as_light_falls_and_at_night() -> None:
    world = World(
        seed=8,
        daynight=DayNightConfig(day_length=200, min_visibility=0.3),
    )
    near = world.spawn(
        {**_SAMPLE_OBJ, "id": "object_1", "color": "crimson", "shape": "sphere", "height": 0.3, "volume": 0.01},
        {"x": 2.0, "y": 0.0},
    )
    far_in_twilight = world.spawn(
        {**_SAMPLE_OBJ, "id": "object_2", "color": "cobalt", "shape": "cube", "height": 0.5, "volume": 0.04},
        {"x": 6.8, "y": 0.0},
    )
    very_far = world.spawn(
        {**_SAMPLE_OBJ, "id": "object_3", "color": "emerald", "shape": "cylinder", "height": 0.8, "volume": 0.06},
        {"x": 9.2, "y": 0.0},
    )

    # 1. Noon: all three objects and their colors are perceived
    world.set_time_of_day(NOON)
    noon_by_id = {o.id: o for o in perceive(world, sensing_radius=10.0).objects}
    assert set(noon_by_id) == {near.id, far_in_twilight.id, very_far.id}
    assert noon_by_id[near.id].color == "crimson"
    assert noon_by_id[far_in_twilight.id].color == "cobalt"
    assert noon_by_id[very_far.id].color == "emerald"

    # 2. Evening twilight (time_of_day=0.66 -> light ~0.77, visibility_factor ~0.84,
    #    radius ~8.4m, detail_radius ~7.0m > 6.8m? Set time_of_day=0.68 -> vf ~0.80, radius ~8.0m, detail_radius ~6.4m):
    #    - very_far (9.2m) is beyond shrunk sensing radius and disappears
    #    - far_in_twilight (6.8m) is still within sensing radius with shape/height/volume, but loses color
    #    - near (2.0m) keeps color
    world.set_time_of_day(0.68)
    assert 0.75 < world.visibility_factor < 0.82
    twilight_by_id = {o.id: o for o in perceive(world, sensing_radius=10.0).objects}
    assert set(twilight_by_id) == {near.id, far_in_twilight.id}
    assert twilight_by_id[near.id].color == "crimson"
    assert twilight_by_id[far_in_twilight.id].color is None
    assert "color" not in twilight_by_id[far_in_twilight.id].properties
    assert twilight_by_id[far_in_twilight.id].shape == "cube"
    assert twilight_by_id[far_in_twilight.id].height == 0.5
    assert twilight_by_id[far_in_twilight.id].volume == 0.04

    # 3. Night / Midnight (time_of_day=0.0 -> visibility_factor = 0.3, radius = 3.0m):
    #    - far objects are outside the shrunk radius
    #    - near object (2.0m) remains visible with shape and size, while color is dropped
    world.set_time_of_day(MIDNIGHT)
    night_percept = perceive(world, sensing_radius=10.0)
    night_by_id = {o.id: o for o in night_percept.objects}
    assert set(night_by_id) == {near.id}
    assert night_by_id[near.id].color is None
    assert "color" not in night_by_id[near.id].properties
    assert night_by_id[near.id].shape == "sphere"
    assert night_by_id[near.id].height == 0.3
    assert night_by_id[near.id].volume == 0.01


def test_creator_controls_set_time_of_day_and_freeze_cycle_emit_creator_events_and_stop_change() -> None:
    world = World(
        seed=14,
        daynight=DayNightConfig(day_length=100, min_visibility=0.25, temperature_swing=0.5),
    )
    all_events: list[WorldEvent] = []
    tod_events: list[WorldEvent] = []
    freeze_events: list[WorldEvent] = []

    world.subscribe(all_events.append)
    world.subscribe("time_of_day_changed", tod_events.append)
    world.subscribe("cycle_frozen", freeze_events.append)

    # Set time of day to dusk (0.75)
    returned_tod = world.set_time_of_day(DUSK)
    assert returned_tod == pytest.approx(DUSK)
    assert world.time_of_day == pytest.approx(DUSK)
    assert len(tod_events) == 1
    tod_ev = tod_events[0]
    assert isinstance(tod_ev, TimeOfDayChangedEvent)
    assert tod_ev.source == "creator"
    assert tod_ev.previous_time_of_day == pytest.approx(NOON)
    assert tod_ev.time_of_day == pytest.approx(DUSK)
    assert tod_ev.light_level == pytest.approx(0.5)
    assert tod_ev.ambient_temperature == pytest.approx(world.ambient_temperature)

    # Freeze the cycle: stepping the world must not change time_of_day, light, visibility, or temperature
    assert world.freeze_cycle(True) is True
    assert world.cycle_frozen is True
    assert len(freeze_events) == 1
    freeze_ev = freeze_events[0]
    assert isinstance(freeze_ev, CycleFrozenEvent)
    assert freeze_ev.source == "creator"
    assert freeze_ev.frozen is True
    assert freeze_ev.time_of_day == pytest.approx(DUSK)

    frozen_tod = world.time_of_day
    frozen_light = world.light_level
    frozen_vis = world.visibility_factor
    frozen_temp = world.ambient_temperature

    for _ in range(25):
        world.step()

    assert world.time_of_day == pytest.approx(frozen_tod)
    assert world.light_level == pytest.approx(frozen_light)
    assert world.visibility_factor == pytest.approx(frozen_vis)
    assert world.ambient_temperature == pytest.approx(frozen_temp)

    # Unfreeze the cycle: stepping now advances time_of_day, light, visibility, and temperature
    assert world.freeze_cycle(False) is False
    assert world.cycle_frozen is False
    assert len(freeze_events) == 2
    assert freeze_events[1].source == "creator"
    assert freeze_events[1].frozen is False

    world.step()
    assert world.time_of_day != pytest.approx(frozen_tod)
    assert world.light_level != pytest.approx(frozen_light)
    assert world.visibility_factor != pytest.approx(frozen_vis)
    assert world.ambient_temperature != pytest.approx(frozen_temp)


def test_percept_exposes_coarse_light_number_with_no_labels() -> None:
    world = World(seed=20, daynight=DayNightConfig(day_length=120))

    world.set_time_of_day(NOON)
    p_noon = perceive(world)
    assert p_noon.light == 1.0
    assert p_noon.light_level == 1.0

    world.set_time_of_day(DUSK)
    p_dusk = perceive(world)
    assert p_dusk.light == 0.5
    assert p_dusk.light_level == 0.5

    world.set_time_of_day(MIDNIGHT)
    p_midnight = perceive(world)
    assert p_midnight.light == 0.0
    assert p_midnight.light_level == 0.0

    assert p_noon.light > p_dusk.light > p_midnight.light

    # Intermediate time produces a coarse 2-decimal number and never any time/light labels
    world.set_time_of_day(0.61)
    p_mid = perceive(world)
    assert isinstance(p_mid.light, float)
    assert p_mid.light == round(p_mid.light, 2)
    assert p_mid.to_dict()["light"] == p_mid.light
    assert f"light: {p_mid.light}" in p_mid.to_text()

    forbidden_labels = (
        "day",
        "night",
        "noon",
        "dusk",
        "dawn",
        "midnight",
        "twilight",
        "bright",
        "dark",
        "dim",
        "warm",
        "cold",
    )
    for percept in (p_noon, p_dusk, p_midnight, p_mid):
        tokens = set(re.findall(r"[a-z0-9_]+", percept.to_text().lower()))
        json_tokens = set(re.findall(r"[a-z0-9_]+", percept.to_json().lower()))
        for label in forbidden_labels:
            assert label not in tokens
            assert label not in json_tokens


def test_perception_tracker_follows_world_daynight_visibility_and_light() -> None:
    world = World(seed=33, daynight=DayNightConfig(day_length=100, min_visibility=0.3))
    far_obj = world.spawn({**_SAMPLE_OBJ, "id": "object_1", "color": "blue"}, {"x": 6.0, "y": 0.0})

    tracker = PerceptionTracker(world, sensing_radius=10.0)
    first = tracker.perceive()
    assert first.light == 1.0
    assert [o.id for o in first.objects] == [far_obj.id]

    # Moving time to midnight shrinks visibility so far_obj disappears and light drops to 0
    world.set_time_of_day(MIDNIGHT)
    delta = tracker.perceive()
    assert delta.is_delta is True
    assert delta.is_empty is False
    assert delta.light == 0.0
    assert delta.disappeared == [far_obj.id]


def test_daynight_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        DayNightConfig(day_length=0)
    with pytest.raises(ValueError):
        DayNightConfig(daylight_share=1.5)
    with pytest.raises(ValueError):
        DayNightConfig(min_visibility=-0.1)
    with pytest.raises(ValueError):
        DayNightConfig(temperature_swing=-1.0)

    world = World(seed=1)
    with pytest.raises(ValueError):
        world.set_time_of_day(-0.1)
    with pytest.raises(ValueError):
        world.set_time_of_day(1.1)
    with pytest.raises(TypeError):
        world.freeze_cycle("yes")  # type: ignore[arg-type]

    assert coarse_light_level(0.756) == 0.76
    assert compute_visibility_factor(0.0, 0.25) == pytest.approx(0.25)
    assert compute_ambient_temperature(0.0, 0.4, base_temperature=1.0) == pytest.approx(0.6)
    assert compute_light_level(1.0, 0.5) == pytest.approx(0.0)
