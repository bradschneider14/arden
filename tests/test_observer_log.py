"""Tests for T25: Observer log (append-only ground truth, query API, replay, and compare helper)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arden.__main__ import main
from arden.agent import AgentLoop, HeartbeatGate, RuleGate, build_test_world
from arden.config import Settings, load_config, make_rng
from arden.environment.weather import WeatherConfig, WeatherSystem
from arden.llm.mock import MockLLM, derive_llm_seed
from arden.memory import MemoryManager
from arden.observer import (
    OBSERVER_KINDS,
    BeliefComparison,
    ComparisonResult,
    Divergence,
    LogRecord,
    ObserverLog,
    ReplayResult,
    compare,
    compare_beliefs,
    replay,
    replay_run,
)
from arden.server import create_app
from arden.world import World


def test_short_mock_run_produces_all_record_kinds_with_schema_config_seed_and_size_rotation() -> None:
    """A short mock run writes all 8 record kinds to rotated JSONL files under data/runs/<run_id>/ with config and seed."""
    settings = Settings()
    settings.world.seed = 12345
    settings.memory.buffer_size = 3
    settings.weather.ticks_per_hour = 2

    rng = make_rng(settings)
    world = build_test_world(rng=rng, settings=settings)
    world.seed = settings.world.seed
    world.weather = WeatherSystem(
        config=WeatherConfig(
            ticks_per_hour=2,
            paused=False,
        ),
        seed=settings.world.seed,
    )
    world.weather._bind_world(world)

    run_id = "test_short_mock_run_t25"
    # Small max_bytes to exercise size-based JSONL rotation across multiple files
    obs_log = ObserverLog(
        world,
        run_id=run_id,
        max_bytes=1200,
        settings=settings,
        seed=settings.world.seed,
    )

    assert obs_log.run_dir == Path("data/runs") / run_id
    assert obs_log.run_dir.is_dir()

    mock_llm = MockLLM(seed=derive_llm_seed(settings.world.seed))
    memory = MemoryManager(
        llm=mock_llm,
        world=world,
        settings=settings,
        db_path=":memory:",
        buffer_size=3,
    )
    loop = AgentLoop(
        world=world,
        llm=mock_llm,
        gate=HeartbeatGate(interval=3),
        memory=memory,
        settings=settings,
        rng=rng,
        observer_log=obs_log,
    )

    # Creator action 1: spawn a new object at tick 0
    spawned = world.spawn(
        {
            "shape": "sphere",
            "position": {"x": 2.0, "y": 2.0},
            "level": "ground",
            "height": 0.25,
            "volume": 0.01,
            "weight": 0.4,
            "rigidity": 0.6,
            "color": "blue",
        },
        source="creator",
    )

    # Creator action 2: manual weather change at tick 0 (clear -> rain)
    world.set_weather("rain", intensity=0.6, duration=2)

    # Step 6 ticks:
    # - ticks 0..2 execute actions, filling the 3-entry short-term buffer -> triggers reflection & belief_change
    # - manual rain expires and automated weather triggers at tick 6
    loop.run(6)

    # Creator action 3: move spawned object at tick 6
    world.move(spawned.id, {"x": 2.5, "y": 2.0}, source="creator")

    # Ensure config and seed are saved alongside the JSONL files
    assert (obs_log.run_dir / "config.json").is_file()
    assert (obs_log.run_dir / "config.toml").is_file()
    assert (obs_log.run_dir / "seed.txt").is_file()
    assert int((obs_log.run_dir / "seed.txt").read_text(encoding="utf-8").strip()) == 12345

    saved_meta = json.loads((obs_log.run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_meta["run_id"] == run_id
    assert saved_meta["seed"] == 12345
    assert saved_meta["config"]["world"]["seed"] == 12345

    loaded_toml = load_config(obs_log.run_dir / "config.toml")
    assert loaded_toml.world.seed == 12345

    # Check size-based rotation produced multiple log_NNNNN.jsonl files
    jsonl_files = obs_log.files
    assert len(jsonl_files) >= 3
    assert jsonl_files[0].name == "log_00001.jsonl"
    assert jsonl_files[1].name == "log_00002.jsonl"

    records = obs_log.read_all()
    assert len(records) > 10

    # Verify every record has run_id, tick, timestamp (wall-clock ISO), kind, and payload
    observed_kinds: set[str] = set()
    for rec in records:
        assert isinstance(rec, LogRecord)
        raw = rec.to_dict()
        assert set(raw.keys()) == {"run_id", "tick", "timestamp", "kind", "payload"}
        assert rec.run_id == run_id
        assert isinstance(rec.tick, int) and rec.tick >= 0
        parsed_ts = datetime.fromisoformat(rec.timestamp)
        assert parsed_ts.tzinfo is not None
        assert rec.kind in OBSERVER_KINDS
        assert isinstance(rec.payload, dict)
        observed_kinds.add(rec.kind)

    # All 8 required kinds must be present
    assert observed_kinds == set(OBSERVER_KINDS) == {
        "perception",
        "gate_decision",
        "intent",
        "action_result",
        "reflection",
        "belief_change",
        "weather_change",
        "creator_action",
    }

    # Check gate_decision records include reasons
    gate_records = [r for r in records if r.kind == "gate_decision"]
    assert len(gate_records) == 6
    assert gate_records[0].payload["wake"] is True
    assert "start" in gate_records[0].payload["reasons"]

    # Check weather_change includes both manual and automated sources
    weather_records = [r for r in records if r.kind == "weather_change"]
    weather_sources = {r.payload["source"] for r in weather_records}
    assert "manual" in weather_sources
    assert "automated" in weather_sources


def test_query_api_filters_and_paging() -> None:
    """GET /api/log?kind=&from_tick=&to_tick=&object_id= supports filtering and paging."""
    settings = Settings()
    settings.memory.buffer_size = 2
    app = create_app(
        world="worlds/first_run.toml",
        settings=settings,
        creator_token="secret-token",
        auto_start=False,
        run_id="test_api_log_query_t25",
    )
    headers = {"Authorization": "Bearer secret-token"}

    with TestClient(app) as client:
        # Creator spawn & move + simulation steps + manual weather + reflection
        spawn_res = client.post(
            "/spawn",
            json={"shape": "sphere", "position": {"x": 1.0, "y": 1.0}, "color": "yellow"},
            headers=headers,
        )
        assert spawn_res.status_code == 200
        spawned_id = spawn_res.json()["object_id"]

        client.post(
            "/weather",
            json={"type": "fog", "intensity": 0.5, "duration": 5},
            headers=headers,
        )
        step_res = client.post("/step", json={"ticks": 4}, headers=headers)
        assert step_res.status_code == 200

        client.post(
            "/move",
            json={"object_id": spawned_id, "position": {"x": 1.5, "y": 1.2}},
            headers=headers,
        )

        # 1. Empty query parameters work cleanly (GET /api/log?kind=&from_tick=&to_tick=&object_id=)
        empty_filter_resp = client.get("/api/log?kind=&from_tick=&to_tick=&object_id=")
        assert empty_filter_resp.status_code == 200
        all_data = empty_filter_resp.json()
        assert all_data["run_id"] == "test_api_log_query_t25"
        assert all_data["total"] >= 10
        assert len(all_data["records"]) == all_data["count"]

        # 2. Filter by kind
        gate_resp = client.get("/api/log", params={"kind": "gate_decision"})
        assert gate_resp.status_code == 200
        gate_data = gate_resp.json()
        assert gate_data["total"] == 4
        assert all(r["kind"] == "gate_decision" for r in gate_data["records"])
        assert all("reasons" in r["payload"] for r in gate_data["records"])

        # Multiple kinds (comma-separated)
        multi_resp = client.get("/api/log", params={"kind": "intent,action_result"})
        assert multi_resp.status_code == 200
        multi_data = multi_resp.json()
        assert multi_data["total"] >= 2
        assert all(r["kind"] in ("intent", "action_result") for r in multi_data["records"])

        # 3. Filter by from_tick and to_tick
        tick_resp = client.get("/api/log", params={"from_tick": 1, "to_tick": 2})
        assert tick_resp.status_code == 200
        tick_data = tick_resp.json()
        assert tick_data["total"] >= 2
        assert all(1 <= r["tick"] <= 2 for r in tick_data["records"])

        # 4. Filter by object_id
        obj_resp = client.get("/api/log", params={"object_id": spawned_id})
        assert obj_resp.status_code == 200
        obj_data = obj_resp.json()
        assert obj_data["total"] >= 2
        obj_kinds = {r["kind"] for r in obj_data["records"]}
        assert "creator_action" in obj_kinds
        for rec in obj_data["records"]:
            assert spawned_id in json.dumps(rec["payload"])

        # 5. Paging with limit/offset and page/page_size
        page1 = client.get("/api/log", params={"limit": 3, "offset": 0}).json()
        page2 = client.get("/api/log", params={"limit": 3, "offset": 3}).json()
        assert len(page1["records"]) == 3
        assert len(page2["records"]) == 3
        assert page1["has_more"] is True
        assert page1["next_offset"] == 3
        assert page1["records"] != page2["records"]

        combined = client.get("/api/log", params={"limit": 6, "offset": 0}).json()
        assert page1["records"] + page2["records"] == combined["records"]

        # Page/page_size style
        p2_by_page = client.get("/api/log", params={"page": 2, "page_size": 3}).json()
        assert p2_by_page["records"] == page2["records"]
        assert p2_by_page["page"] == 2
        assert p2_by_page["page_size"] == 3


def test_replay_of_mock_run_reports_no_divergence_and_detects_tampered_divergence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m arden replay data/runs/<run_id>` replays saved seed, creator actions, and intents with no divergence."""
    settings = Settings()
    settings.world.seed = 777
    settings.memory.buffer_size = 3
    rng = make_rng(settings)
    world = build_test_world(rng=rng, settings=settings)
    world.seed = settings.world.seed

    run_id = "test_replay_mock_run_t25"
    obs_log = ObserverLog(world, run_id=run_id, settings=settings, seed=settings.world.seed)
    mock_llm = MockLLM(seed=derive_llm_seed(settings.world.seed))
    memory = MemoryManager(
        llm=mock_llm,
        world=world,
        settings=settings,
        db_path=":memory:",
        buffer_size=3,
    )
    loop = AgentLoop(
        world=world,
        llm=mock_llm,
        gate=HeartbeatGate(interval=4),
        memory=memory,
        settings=settings,
        rng=rng,
        observer_log=obs_log,
    )

    # Perform creator actions and run simulation ticks
    spawned = world.spawn(
        {
            "shape": "cube",
            "position": {"x": 1.5, "y": -0.5},
            "level": "ground",
            "height": 0.3,
            "volume": 0.02,
            "weight": 0.6,
            "rigidity": 0.8,
            "color": "orange",
        },
        source="creator",
    )
    loop.run(3)
    world.move(spawned.id, {"x": 2.0, "y": -0.5}, source="creator")
    world.set_weather("wind", intensity=0.5, duration=4)
    loop.run(5)

    run_path = Path("data/runs") / run_id
    assert run_path.is_dir()

    # 1. Programmatic replay_run reports no divergence
    result = replay_run(run_path)
    assert isinstance(result, ReplayResult)
    assert result.ok is True
    assert result.diverged is False
    assert result.first_divergence is None
    assert result.ticks_replayed == world.tick == 8
    assert result.records_checked == len(obs_log.read_all())

    # 2. CLI `python -m arden replay data/runs/<run_id>` exits 0 and prints "no divergence"
    exit_code = main(["replay", str(run_path)])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "no divergence" in captured.lower()

    # Also verify subprocess invocation `python -m arden replay data/runs/<run_id>`
    proc = subprocess.run(
        [sys.executable, "-m", "arden", "replay", str(run_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "no divergence" in proc.stdout.lower()

    # 3. Tamper with a logged record and verify replay reports the exact first divergence
    log_file = obs_log.files[0]
    lines = log_file.read_text(encoding="utf-8").splitlines()
    tampered_lines: list[str] = []
    tampered_tick: int | None = None
    for line in lines:
        rec_dict = json.loads(line)
        if rec_dict["kind"] == "action_result" and tampered_tick is None:
            tampered_tick = rec_dict["tick"]
            rec_dict["payload"]["ok"] = not bool(rec_dict["payload"].get("ok", True))
            rec_dict["payload"]["reason"] = "tampered_failure"
        tampered_lines.append(json.dumps(rec_dict))
    log_file.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")

    div_result = replay(run_path)
    assert div_result.ok is False
    assert div_result.diverged is True
    assert isinstance(div_result.first_divergence, Divergence)
    assert div_result.first_divergence.kind == "action_result"
    assert div_result.first_divergence.tick == tampered_tick

    div_exit_code = main(["replay", str(run_path)])
    div_out = capsys.readouterr().out
    assert div_exit_code == 1
    assert "divergence" in div_out.lower()


def test_compare_helper_lists_beliefs_next_to_forming_and_contradicting_ground_truth_events() -> None:
    """`compare` lists beliefs alongside ground-truth events that formed or contradicted them (memory vs reality)."""
    world = build_test_world()
    obs_log = ObserverLog(world, run_id="test_compare_helper_t25")
    mock_llm = MockLLM(seed=42)
    memory = MemoryManager(llm=mock_llm, world=world, db_path=":memory:")
    bstore = memory.belief_store

    # Step 1: Agent looks at object_3 (red sphere at (0.8, 0.4)) and pushes object_7
    from arden.actions.verbs import look, push
    from arden.perception import perceive

    perceive(world)
    look(world, "object_3")
    push(world, "object_7", force=0.5)
    world.step()

    # Form beliefs:
    # - b1: accurate belief about object_3 being red and at (0.8, 0.4)
    b1 = bstore.add(
        statement="Object_3 is red and rests at (0.8, 0.4).",
        tags=["object_3", "red"],
        confidence=0.80,
        tick=world.tick,
    )
    # - b2: belief about object_1 moving when pushed (which will be contradicted by a failed push on fixed object_1)
    b2 = bstore.add(
        statement="Object_1 rolls when pushed.",
        tags=["object_1", "push"],
        confidence=0.65,
        tick=world.tick,
    )

    # Now reality diverges from b1 and b2:
    # 1) Creator moves object_3 away from (0.8, 0.4) to (5.0, 5.0) and changes its color to blue
    world.move("object_3", {"x": 5.0, "y": 5.0}, source="creator")
    world.update_object("object_3", {"color": "blue"}, source="creator")

    # 2) Agent tries to push fixed object_1 -> fails with target_is_fixed, and belief b2 is contradicted
    push(world, "object_1", force=0.5)
    bstore.contradict(b2.id, tick=world.tick)

    comp = compare(obs_log, bstore, world=world)
    assert isinstance(comp, ComparisonResult)
    assert len(comp) == 2

    by_id: dict[int | None, BeliefComparison] = {item.belief_id: item for item in comp}
    c1 = by_id[b1.id]
    c2 = by_id[b2.id]

    # c1 was formed when object_3 was at (0.8, 0.4) and red, then contradicted by creator move/color change
    assert len(c1.forming_events) >= 1
    assert any(ev.kind in ("perception", "action_result", "belief_change") for ev in c1.forming_events)
    assert len(c1.contradicting_events) >= 1
    assert any(ev.kind == "creator_action" for ev in c1.contradicting_events)
    assert c1.reality_status in ("contradicted", "outdated")
    assert c1.reality_gap is not None
    assert "object_3" in c1.reality_gap.lower()

    # c2 was formed at tick 1 and contradicted by the failed push action_result + contradict belief_change
    assert len(c2.forming_events) >= 1
    assert len(c2.contradicting_events) >= 1
    assert any(
        ev.kind == "action_result" and ev.payload.get("ok") is False
        for ev in c2.contradicting_events
    )
    assert c2.reality_status == "contradicted"
    assert c2.reality_gap is not None

    # Verify compare_beliefs alias and dict serialization
    comp_dict = compare_beliefs(obs_log, bstore, world=world).to_dict()
    assert comp_dict["count"] == 2
    assert len(comp_dict["comparisons"]) == 2
