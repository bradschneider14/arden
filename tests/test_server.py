"""Tests for T22: FastAPI simulation service, creator API, WebSocket stream, and auth."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from arden.__main__ import main
from arden.actions.verbs import grab, speak
from arden.config import CREATOR_TOKEN_ENV_VAR, DEFAULT_SERVER_HOST, Settings
from arden.llm.mock import MockLLM
from arden.server import (
    SCHEMA_VERSION,
    SimulationService,
    create_app,
    is_local_host,
    validate_server_binding,
)
from arden.world import World


def _drain_ws_messages(ws: Any, count: int) -> list[dict[str, Any]]:
    """Receive ``count`` JSON messages from a TestClient WebSocket session."""
    return [ws.receive_json() for _ in range(count)]


def test_default_host_is_loopback_and_non_local_without_token_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default bind is 127.0.0.1; non-local binding without a token refuses to start."""
    monkeypatch.delenv(CREATOR_TOKEN_ENV_VAR, raising=False)

    cfg = Settings()
    assert cfg.server.host == DEFAULT_SERVER_HOST == "127.0.0.1"
    assert is_local_host("127.0.0.1") is True
    assert is_local_host("localhost") is True
    assert is_local_host("::1") is True
    assert is_local_host("0.0.0.0") is False
    assert is_local_host("192.168.1.50") is False

    # Local binding starts without a token
    local_app = create_app(host="127.0.0.1", auto_start=False)
    assert local_app.service.host == "127.0.0.1"

    # Non-local binding without a token refuses to start
    with pytest.raises(ValueError, match="Refusing to bind non-local host"):
        validate_server_binding("0.0.0.0", None)

    with pytest.raises(ValueError, match="Refusing to bind non-local host"):
        create_app(host="0.0.0.0", auto_start=False)

    with pytest.raises(ValueError, match="Refusing to bind non-local host"):
        main(["serve", "--world", "worlds/first_run.toml", "--host", "0.0.0.0"])

    # Non-local binding succeeds when token is provided in config or ARDEN_CREATOR_TOKEN
    app_with_config_token = create_app(
        host="0.0.0.0",
        creator_token="secret-config-token",
        auto_start=False,
    )
    assert app_with_config_token.service.creator_token == "secret-config-token"

    monkeypatch.setenv(CREATOR_TOKEN_ENV_VAR, "secret-env-token")
    app_with_env_token = create_app(host="0.0.0.0", auto_start=False)
    assert app_with_env_token.service.creator_token == "secret-env-token"


def test_cli_serve_starts_uvicorn_on_127_0_0_1_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`python -m arden serve --world worlds/first_run.toml` starts uvicorn on 127.0.0.1 by default."""
    captured: dict[str, Any] = {}

    def fake_uvicorn_run(app: Any, *, host: str, port: int, log_level: str = "info") -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("arden.server.app.uvicorn.run", fake_uvicorn_run)
    code = main(["serve", "--world", "worlds/first_run.toml"])
    assert code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    app = captured["app"]
    assert len(app.service.world.objects) == 20
    assert "object_15" in app.service.world.objects


def test_read_endpoints_snapshot_agent_inbox_beliefs_self_model_weather_and_time() -> None:
    """All required read endpoints return structured state for first_run.toml."""
    app = create_app(
        world="worlds/first_run.toml",
        creator_token="test-token",
        auto_start=False,
    )
    auth = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        # 1. World snapshot
        snap_resp = client.get("/snapshot")
        assert snap_resp.status_code == 200
        snap = snap_resp.json()
        assert snap["tick"] == 0
        assert snap["schema_version"] == SCHEMA_VERSION
        assert "object_1" in snap["objects"]
        assert "object_15" in snap["objects"]
        assert snap["creator_labels"]["object_15"] == "mailbox"
        restored = World.restore(snap)
        assert len(restored.objects) == len(snap["objects"])

        # 2. Agent state (initially before any tick, then after 1 step)
        agent_resp0 = client.get("/agent")
        assert agent_resp0.status_code == 200
        agent0 = agent_resp0.json()
        assert "body" in agent0
        assert agent0["body"]["energy"] == pytest.approx(1.0)
        assert agent0["body"]["comfort"] == pytest.approx(1.0)
        assert agent0["current_intent"] is None
        assert agent0["last_thought"] is None

        # Step 1 tick so the agent wakes at start and sets an intent + thought
        step_resp = client.post("/step", json={"ticks": 1}, headers=auth)
        assert step_resp.status_code == 200

        agent_resp1 = client.get("/agent")
        assert agent_resp1.status_code == 200
        agent1 = agent_resp1.json()
        assert agent1["tick"] == 1
        assert agent1["current_intent"] is not None
        assert "goal" in agent1["current_intent"]
        assert isinstance(agent1["last_thought"], str) and len(agent1["last_thought"]) > 0

        # Trigger reflection so beliefs and self-model are populated from MockLLM reflect fixture
        ref_resp = client.post("/reflect", headers=auth)
        assert ref_resp.status_code == 200

        # 3. Beliefs
        beliefs_resp = client.get("/beliefs")
        assert beliefs_resp.status_code == 200
        beliefs_data = beliefs_resp.json()
        assert beliefs_data["count"] >= 1
        assert len(beliefs_data["beliefs"]) == beliefs_data["count"]
        assert "statement" in beliefs_data["beliefs"][0]
        assert "confidence" in beliefs_data["beliefs"][0]
        assert "history" in beliefs_data["beliefs"][0]

        # 4. Self-model
        sm_resp = client.get("/self-model")
        assert sm_resp.status_code == 200
        sm_data = sm_resp.json()
        assert isinstance(sm_data["self_model"], str) and len(sm_data["self_model"]) > 0
        assert sm_data["text"] == sm_data["self_model"]
        assert sm_data["version"] >= 1
        assert len(sm_data["history"]) >= 1

        # 5. Inbox
        inbox_resp = client.get("/inbox")
        assert inbox_resp.status_code == 200
        inbox_data = inbox_resp.json()
        assert "inbox" in inbox_data
        assert "messages" in inbox_data
        assert inbox_data["count"] == 0

        # 6. Weather
        weather_resp = client.get("/weather")
        assert weather_resp.status_code == 200
        weather_data = weather_resp.json()
        assert weather_data["type"] == "clear"
        assert weather_data["automation_enabled"] is True

        # 7. Time
        time_resp = client.get("/time")
        assert time_resp.status_code == 200
        time_data = time_resp.json()
        assert time_data["tick"] == 1
        assert 0.0 <= time_data["time_of_day"] <= 1.0
        assert time_data["frozen"] is False
        assert "light_level" in time_data
        assert "ambient_temperature" in time_data


def test_creator_endpoints_and_websocket_stream() -> None:
    """Spawn via API appears in snapshot and on WebSocket; all creator endpoints mutate and stream."""
    app = create_app(
        world="worlds/first_run.toml",
        creator_token="creator-secret",
        auto_start=False,
    )
    headers = {"Authorization": "Bearer creator-secret"}

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        # 1. Spawn object via API -> appears in snapshot and on WebSocket
        spawn_resp = client.post(
            "/spawn",
            json={
                "shape": "sphere",
                "position": {"x": 6.0, "y": 6.0},
                "height": 0.3,
                "volume": 0.02,
                "weight": 0.4,
                "color": "blue",
            },
            headers=headers,
        )
        assert spawn_resp.status_code == 200
        spawn_data = spawn_resp.json()
        spawned_id = spawn_data["object_id"]
        assert spawned_id.startswith("object_")

        # Appears in snapshot
        snap = client.get("/snapshot").json()
        assert spawned_id in snap["objects"]
        assert snap["objects"][spawned_id]["color"] == "blue"
        assert snap["objects"][spawned_id]["position"] == {"x": 6.0, "y": 6.0}

        # Appears on WebSocket (world_delta + creator_action)
        ws_msgs = _drain_ws_messages(ws, 2)
        for msg in ws_msgs:
            assert msg["schema_version"] == SCHEMA_VERSION
            assert msg["object_id"] == spawned_id
            assert msg["object"]["id"] == spawned_id
        msg_types = {m["type"] for m in ws_msgs}
        assert "world_delta" in msg_types
        assert "creator_action" in msg_types

        # 2. Move object via API
        move_resp = client.post(
            "/move",
            json={"object_id": spawned_id, "position": {"x": 7.5, "y": 8.25}},
            headers=headers,
        )
        assert move_resp.status_code == 200
        assert move_resp.json()["position"] == {"x": 7.5, "y": 8.25}
        snap_after_move = client.get("/snapshot").json()
        assert snap_after_move["objects"][spawned_id]["position"] == {"x": 7.5, "y": 8.25}
        move_ws_msgs = _drain_ws_messages(ws, 2)
        assert any(
            m["type"] == "world_delta" and m["event"] == "object_moved"
            for m in move_ws_msgs
        )
        assert any(
            m["type"] == "creator_action" and m["action"] == "move"
            for m in move_ws_msgs
        )

        # 3. Despawn object via API
        despawn_resp = client.post(
            "/despawn",
            json={"object_id": spawned_id},
            headers=headers,
        )
        assert despawn_resp.status_code == 200
        snap_after_despawn = client.get("/snapshot").json()
        assert spawned_id not in snap_after_despawn["objects"]
        despawn_ws_msgs = _drain_ws_messages(ws, 2)
        assert any(
            m["type"] == "world_delta" and m["event"] == "object_despawned"
            for m in despawn_ws_msgs
        )
        assert any(
            m["type"] == "creator_action" and m["action"] == "despawn"
            for m in despawn_ws_msgs
        )

        # 4. Reply to mail and spawn-in-response
        reply_resp = client.post(
            "/inbox/reply",
            json={"text": "try stacking flat stones"},
            headers=headers,
        )
        assert reply_resp.status_code == 200
        letter_id = reply_resp.json()["object_id"]
        snap_after_reply = client.get("/snapshot").json()
        assert letter_id in snap_after_reply["objects"]
        assert snap_after_reply["objects"][letter_id]["level"] == "inside:object_15"
        assert snap_after_reply["objects"][letter_id]["state"]["text"] == "try stacking flat stones"

        # Drain WS messages from reply (object_spawned, mail_arrived, creator_action)
        reply_ws_msgs = _drain_ws_messages(ws, 3)
        assert any(
            m["type"] == "creator_action" and m["action"] == "reply"
            for m in reply_ws_msgs
        )

        sir_resp = client.post(
            "/inbox/spawn-in-response",
            json={
                "props": {
                    "shape": "long",
                    "height": 0.9,
                    "volume": 0.03,
                    "weight": 0.4,
                    "rigidity": 0.9,
                }
            },
            headers=headers,
        )
        assert sir_resp.status_code == 200
        sir_id = sir_resp.json()["object_id"]
        snap_after_sir = client.get("/snapshot").json()
        assert snap_after_sir["objects"][sir_id]["level"] == "inside:object_15"
        sir_ws_msgs = _drain_ws_messages(ws, 3)
        assert any(
            m["type"] == "creator_action" and m["action"] == "spawn_in_response"
            for m in sir_ws_msgs
        )

        # 5. Weather: set immediate, schedule future, and toggle automation
        weather_set = client.post(
            "/weather",
            json={"type": "rain", "intensity": 0.7, "duration": 40},
            headers=headers,
        )
        assert weather_set.status_code == 200
        assert weather_set.json()["type"] == "rain"
        assert weather_set.json()["intensity"] == pytest.approx(0.7)
        assert client.get("/weather").json()["type"] == "rain"
        _drain_ws_messages(ws, 2)  # weather_changed + creator_action

        weather_sched = client.post(
            "/weather",
            json={"type": "storm", "intensity": 0.9, "duration": 25, "start_at": 10},
            headers=headers,
        )
        assert weather_sched.status_code == 200
        assert len(weather_sched.json()["scheduled"]) == 1
        assert weather_sched.json()["scheduled"][0]["start_at"] == 10
        _drain_ws_messages(ws, 1)  # creator_action

        auto_toggle = client.post(
            "/weather/automation",
            json={"enabled": False},
            headers=headers,
        )
        assert auto_toggle.status_code == 200
        assert auto_toggle.json()["automation_enabled"] is False
        assert auto_toggle.json()["paused"] is True
        _drain_ws_messages(ws, 1)

        # 6. Time of day: set fraction and freeze/unfreeze cycle
        time_set = client.post(
            "/time",
            json={"time_of_day": 0.0, "frozen": True},
            headers=headers,
        )
        assert time_set.status_code == 200
        assert time_set.json()["time_of_day"] == pytest.approx(0.0)
        assert time_set.json()["frozen"] is True
        assert client.get("/time").json()["frozen"] is True
        _drain_ws_messages(ws, 3)  # time_of_day_changed, cycle_frozen, creator_action

        # 7. Pause, step (while paused), set speed, and resume
        pause_resp = client.post("/pause", headers=headers)
        assert pause_resp.status_code == 200
        assert pause_resp.json()["paused"] is True
        _drain_ws_messages(ws, 1)

        speed_resp = client.post("/speed", json={"speed": 2.5}, headers=headers)
        assert speed_resp.status_code == 200
        assert speed_resp.json()["speed"] == pytest.approx(2.5)
        _drain_ws_messages(ws, 1)

        step_resp = client.post("/step", json={"ticks": 1}, headers=headers)
        assert step_resp.status_code == 200
        assert step_resp.json()["tick"] == 1
        assert step_resp.json()["paused"] is True  # stays paused after single-step

        resume_resp = client.post("/resume", headers=headers)
        assert resume_resp.status_code == 200
        assert resume_resp.json()["paused"] is False


def test_websocket_streams_all_agent_and_world_event_types() -> None:
    """WebSocket /ws streams gate_decision, wake, intent, thought, world_delta, reflection, and creator_action."""
    app = create_app(
        world="worlds/first_run.toml",
        creator_token="secret",
        auto_start=False,
    )
    headers = {"Authorization": "Bearer secret"}

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # Step 1 tick -> wakes at tick 0 ('start') -> emits gate_decision, wake, intent, thought, world_delta, creator_action
            r_step = client.post("/step", json={"ticks": 1}, headers=headers)
            assert r_step.status_code == 200

            # Trigger reflection -> emits reflection
            r_ref = client.post("/reflect", headers=headers)
            assert r_ref.status_code == 200

            received: list[dict[str, Any]] = []
            while True:
                msg = ws.receive_json()
                received.append(msg)
                seen_types = {m["type"] for m in received}
                if {
                    "gate_decision",
                    "wake",
                    "intent",
                    "thought",
                    "world_delta",
                    "creator_action",
                    "reflection",
                }.issubset(seen_types):
                    break

            for m in received:
                assert m["schema_version"] == SCHEMA_VERSION
                assert "tick" in m

            by_type = {m["type"]: m for m in received}
            assert by_type["gate_decision"]["wake"] is True
            assert "start" in by_type["gate_decision"]["reasons"]
            assert "start" in by_type["wake"]["reasons"]
            assert isinstance(by_type["intent"]["goal"], str)
            assert isinstance(by_type["thought"]["thought"], str)
            assert len(by_type["reflection"]["episodes"]) >= 1
            assert by_type["creator_action"]["action"] == "step"
            assert "delta" in by_type["world_delta"]


def test_auth_rejects_missing_or_wrong_token_and_spectator_mode_cannot_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creator endpoints reject missing/wrong tokens; spectator mode can read when enabled but never mutate."""
    monkeypatch.setenv(CREATOR_TOKEN_ENV_VAR, "valid-token")
    app = create_app(
        world="worlds/first_run.toml",
        spectator_enabled=True,
        auto_start=False,
    )

    creator_calls = [
        ("POST", "/spawn", {"shape": "sphere", "position": {"x": 1.0, "y": 1.0}}),
        ("POST", "/despawn", {"object_id": "object_3"}),
        ("POST", "/move", {"object_id": "object_3", "position": {"x": 2.0, "y": 2.0}}),
        ("POST", "/inbox/reply", {"text": "hello"}),
        ("POST", "/inbox/spawn-in-response", {"props": {"shape": "sphere"}}),
        ("POST", "/weather", {"type": "rain", "intensity": 0.5, "duration": 20}),
        ("POST", "/weather/automation", {"enabled": False}),
        ("POST", "/time", {"time_of_day": 0.25}),
        ("POST", "/time/freeze", {"frozen": True}),
        ("POST", "/pause", {}),
        ("POST", "/resume", {}),
        ("POST", "/step", {"ticks": 1}),
        ("POST", "/speed", {"speed": 2.0}),
    ]

    with TestClient(app) as client:
        # Spectators (no token) can read all GET endpoints when spectator_enabled=True
        for read_path in (
            "/snapshot",
            "/agent",
            "/inbox",
            "/beliefs",
            "/self-model",
            "/weather",
            "/time",
        ):
            r = client.get(read_path)
            assert r.status_code == 200, f"Expected 200 on {read_path}, got {r.status_code}"

        # Missing token rejected on every creator endpoint (401)
        for method, path, payload in creator_calls:
            r_missing = client.request(method, path, json=payload)
            assert r_missing.status_code == 401, f"Expected 401 on {path}"

        # Wrong token rejected on every creator endpoint (403)
        for method, path, payload in creator_calls:
            r_wrong = client.request(
                method,
                path,
                json=payload,
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert r_wrong.status_code == 403, f"Expected 403 on {path}"

        # Explicit spectator flag/role cannot mutate even if a valid token is sent (403)
        r_spec = client.post(
            "/spawn",
            json={"shape": "sphere", "position": {"x": 1.0, "y": 1.0}},
            headers={"Authorization": "Bearer valid-token", "X-Role": "spectator"},
        )
        assert r_spec.status_code == 403

        # WebSocket spectator cannot mutate via WS frames
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "spawn", "shape": "sphere"})
            err_msg = ws.receive_json()
            assert err_msg["schema_version"] == SCHEMA_VERSION
            assert err_msg["type"] == "error"
            assert err_msg["error"] == "forbidden"

        # Disabling read-only spectator access blocks unauthenticated GET and /ws
        app.service.spectator_enabled = False
        assert client.get("/snapshot").status_code == 401
        assert (
            client.get(
                "/snapshot",
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/snapshot",
                headers={"Authorization": "Bearer valid-token"},
            ).status_code
            == 200
        )


def test_inbox_delivery_visible_on_inbox_endpoint() -> None:
    """Messages and objects delivered through the mailbox appear in GET /inbox."""
    app = create_app(
        world="worlds/first_run.toml",
        creator_token="secret",
        auto_start=False,
    )
    world = app.service.world
    mailbox = world.objects["object_15"]
    # Move agent next to the mailbox and open lid
    world.body.position = mailbox.position
    grab(world, "object_15")
    speak(world, "need a long stick")
    grab(world, "object_15")  # closing lid delivers the message

    with TestClient(app) as client:
        inbox_data = client.get("/inbox").json()
        assert inbox_data["count"] == 1
        assert inbox_data["messages"][0]["text"] == "need a long stick"


def test_background_service_runs_without_watchers_and_llm_does_not_block_loop() -> None:
    """SimulationService background task advances ticks without clients and offloads slow LLM calls."""
    slow_calls = 0

    class SlowMockLLM(MockLLM):
        def complete_json(
            self,
            role: str,
            system: str,
            user: str,
            schema: dict,
            timeout_s: float,
        ) -> dict:
            nonlocal slow_calls
            slow_calls += 1
            time.sleep(0.08)
            return super().complete_json(role, system, user, schema, timeout_s)

    settings = Settings()
    settings.world.tick_rate = 50.0
    service = SimulationService(
        world="worlds/first_run.toml",
        settings=settings,
        llm=SlowMockLLM(seed=1),
        creator_token="secret",
        auto_start=True,
    )
    app = create_app(service=service)

    with TestClient(app) as client:
        # While the background task is running (and calling SlowMockLLM on a worker thread),
        # HTTP read endpoints remain responsive and ticks advance with no WebSocket watchers.
        t0 = time.perf_counter()
        while service.world.tick < 2 and (time.perf_counter() - t0) < 2.0:
            resp = client.get("/time")
            assert resp.status_code == 200
            time.sleep(0.02)

        assert service.world.tick >= 2
        assert slow_calls >= 1
