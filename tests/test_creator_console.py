"""Tests for T24: Creator console and observer views (login, inbox, controls, timeline, beliefs, names)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from arden.server import (
    FIRST_RUN_PRESETS,
    create_app,
    extract_object_names,
    get_spawn_presets,
)
from arden.actions.verbs import grab, speak
from arden.llm.mock import MockLLM
from arden.web import STATIC_DIR


def test_login_auth_endpoint_and_spectator_mode_status(tmp_path: Path) -> None:
    """POST /auth validates the creator token and GET /status reports auth_required and authenticated."""
    app = create_app(
        world="worlds/first_run.toml",
        db_path=tmp_path / "auth_console.sqlite",
        creator_token="console-secret-token",
        auto_start=False,
    )

    with TestClient(app) as client:
        # Unauthenticated status check (spectator view)
        status_unauth = client.get("/status")
        assert status_unauth.status_code == 200
        status_body = status_unauth.json()
        assert status_body["auth_required"] is True
        assert status_body["authenticated"] is False

        # Wrong token returns 401 on POST /auth
        bad_login = client.post("/auth", json={"token": "wrong-token"})
        assert bad_login.status_code == 401
        assert bad_login.json()["authenticated"] is False

        # Valid token in JSON body returns 200 and creator mode
        good_login = client.post("/auth", json={"token": "console-secret-token"})
        assert good_login.status_code == 200
        login_data = good_login.json()
        assert login_data["authenticated"] is True
        assert login_data["auth_required"] is True
        assert login_data["mode"] == "creator"

        # Authenticated status check with Bearer header
        headers = {"Authorization": "Bearer console-secret-token"}
        status_auth = client.get("/status", headers=headers)
        assert status_auth.status_code == 200
        assert status_auth.json()["authenticated"] is True


def test_presets_endpoint_and_spawn_with_preset(tmp_path: Path) -> None:
    """GET /presets returns first-run presets and POST /spawn expands preset keys."""
    presets = get_spawn_presets()
    preset_keys = {p["key"] for p in presets}
    assert {
        "ball",
        "stick",
        "flat_stone",
        "vine",
        "leaf",
        "berry",
        "berry_bush",
        "hollow_log",
        "large_tree",
        "mailbox",
    }.issubset(preset_keys)
    assert len(FIRST_RUN_PRESETS) == len(presets)

    app = create_app(
        world="worlds/first_run.toml",
        db_path=tmp_path / "presets.sqlite",
        auto_start=False,
    )

    with TestClient(app) as client:
        res = client.get("/presets")
        assert res.status_code == 200
        body = res.json()
        assert body["count"] >= 10
        assert any(p["key"] == "ball" and p["color"] == "red" for p in body["presets"])

        # Spawn using a preset key with custom coordinates
        spawn_res = client.post(
            "/spawn",
            json={"preset": "vine", "x": 8.5, "y": 7.5},
        )
        assert spawn_res.status_code == 200
        spawned_obj = spawn_res.json()["object"]
        assert spawned_obj["shape"] == "long"
        assert spawned_obj["color"] == "green"
        assert spawned_obj["binding"] is True
        assert spawned_obj["creator_label"] == "vine"
        assert spawned_obj["position"] == {"x": 8.5, "y": 7.5}


def test_inbox_private_note_is_silent_while_reply_and_spawn_wake_agent(tmp_path: Path) -> None:
    """Private notes save without sending any world event or mail_arrived; reply and spawn-in-response do."""
    app = create_app(
        world="worlds/first_run.toml",
        db_path=tmp_path / "inbox_notes.sqlite",
        auto_start=False,
    )
    world = app.service.world
    mailbox = world.objects["object_15"]

    with TestClient(app) as client:
        # Agent moves to mailbox, opens lid, speaks, and closes lid to deliver message
        world.body.position = mailbox.position
        grab(world, "object_15")  # opens lid
        speak(world, "Here is the red berry (object_6)")
        grab(world, "object_15")  # closes lid -> delivers to creator inbox

        inbox_res = client.get("/inbox")
        assert inbox_res.status_code == 200
        inbox_data = inbox_res.json()
        assert inbox_data["count"] == 1
        entry = inbox_data["messages"][0]
        entry_id = entry["id"]
        assert entry["message"] == "Here is the red berry (object_6)"
        assert entry["note"] is None

        # Record event count & objects before saving notes to verify saving a note is silent
        events_before = len(app.service.event_history)
        objects_before = set(world.objects.keys())

        # Save per-entry private note
        note_res = client.post(
            f"/inbox/{entry_id}/note",
            json={"note": "Agent offered a berry; wait before replying."},
        )
        assert note_res.status_code == 200
        note_body = note_res.json()
        assert note_body["entry"]["note"] == "Agent offered a berry; wait before replying."

        # Also save experiment-level private note
        exp_note_res = client.post(
            "/inbox/note",
            json={"note": "General experiment observation note"},
        )
        assert exp_note_res.status_code == 200
        assert exp_note_res.json()["inbox"]["creator_note"] == "General experiment observation note"

        # Confirm NO world/WS events or objects were spawned when saving notes
        assert len(app.service.event_history) == events_before
        assert set(world.objects.keys()) == objects_before

        # Now creator explicitly acts: spawn-in-response using a preset
        spawn_resp = client.post(
            "/inbox/spawn-in-response",
            json={"entry_id": entry_id, "object": {"preset": "ball"}},
        )
        assert spawn_resp.status_code == 200
        spawned_in_box = spawn_resp.json()["object"]
        assert spawned_in_box["level"] == "inside:object_15"
        assert spawned_in_box["shape"] == "sphere"
        assert spawned_in_box["color"] == "red"


def test_timeline_streams_and_filters_wakes_thoughts_intents_and_actions(tmp_path: Path) -> None:
    """GET /timeline and /ws stream wake, thought, intent, and action entries and filter by kind."""
    app = create_app(
        world="worlds/first_run.toml",
        db_path=tmp_path / "timeline.sqlite",
        auto_start=False,
    )

    with TestClient(app) as client:
        # Step simulation so agent wakes, plans (thought + intent), and executes actions
        step_res = client.post("/step", json={"ticks": 4})
        assert step_res.status_code == 200

        # Creator actions: weather change and inbox reply
        assert client.post("/weather", json={"type": "rain", "intensity": 0.7, "duration": 60}).status_code == 200
        assert client.post("/inbox/reply", json={"text": "Welcome to the clearing"}).status_code == 200

        # Fetch full timeline
        tl_all = client.get("/timeline").json()
        kinds_present = {entry["kind"] for entry in tl_all["entries"]}
        assert {"wake", "thought", "intent", "action"}.issubset(kinds_present)

        # Verify filtering by each kind
        for kind in ("wake", "thought", "intent", "action"):
            tl_filtered = client.get(f"/timeline?kind={kind}").json()
            assert tl_filtered["kind"] == kind
            assert len(tl_filtered["entries"]) >= 1
            assert all(e["kind"] == kind for e in tl_filtered["entries"])

        # Verify wake entries include reasons
        wake_entries = client.get("/timeline?kind=wake").json()["entries"]
        assert any("reasons" in e["details"] and len(e["details"]["reasons"]) >= 1 for e in wake_entries)


def test_beliefs_history_and_self_model_versions_and_names_extraction(tmp_path: Path) -> None:
    """Beliefs include confidence history, self-model tracks versions, and /names maps words to object IDs."""
    llm = MockLLM(seed=1)
    app = create_app(
        world="worlds/first_run.toml",
        db_path=tmp_path / "beliefs_names.sqlite",
        llm=llm,
        auto_start=False,
    )
    service = app.service

    with TestClient(app) as client:
        # Step 1 tick so short-term memory has an entry, then trigger first reflection
        assert client.post("/step", json={"ticks": 1}).status_code == 200

        speak(service.world, "I like this red ball (object_1) and long stick (object_2)")
        service._append_timeline_entry(
            {
                "kind": "thought",
                "tick": 1,
                "thought": "Maybe the flat stone (object_3) is heavy.",
                "goal": "Inspect flat stone (object_3)",
                "summary": "Maybe the flat stone (object_3) is heavy.",
            }
        )

        r1 = client.post("/reflect")
        assert r1.status_code == 200
        first_belief_id = client.get("/beliefs").json()["beliefs"][0]["id"]

        # Add a buffer entry and enqueue a second reflection output that updates first_belief_id and self-model v2
        service.memory.buffer.add(
            tick=2,
            observation="Inspected object_15 the blue mailbox",
            action="look",
            outcome="ok",
        )
        llm.enqueue(
            json.dumps(
                {
                    "episodes": [
                        {
                            "summary": "Tested object_15 the blue mailbox.",
                            "salience": 0.8,
                            "tags": ["object_15", "mailbox"],
                            "surprise_kind": None,
                            "made_by_agent": [],
                        }
                    ],
                    "beliefs": [
                        {
                            "action": "support",
                            "statement": "Pushing sphere object_7 causes it to roll across flat ground toward object_1.",
                            "confidence": 0.85,
                            "tags": ["object_7", "object_1", "push", "sphere"],
                        },
                        {
                            "action": "propose",
                            "statement": "object_15 is the blue mailbox.",
                            "confidence": 0.80,
                            "tags": ["object_15", "mailbox"],
                        },
                    ],
                    "self_model": "I am an experienced explorer of the clearing and its mailbox.",
                }
            )
        )
        r2 = client.post("/reflect")
        assert r2.status_code == 200

        # Check /beliefs has confidence history with >= 2 points for first_belief_id
        beliefs_data = client.get("/beliefs").json()
        assert beliefs_data["count"] >= 2
        b1 = next(b for b in beliefs_data["beliefs"] if b["id"] == first_belief_id)
        assert len(b1["history"]) >= 2

        # Check /self-model has version 2 and history of both v1 and v2
        sm_data = client.get("/self-model").json()
        assert sm_data["version"] == 2
        assert len(sm_data["history"]) == 2
        assert sm_data["history"][0]["version"] == 1
        assert sm_data["history"][1]["version"] == 2

        # Check /names maps words from speak, thoughts, and beliefs to object_<n>
        names_data = client.get("/names").json()
        by_obj = names_data["by_object"]
        assert "object_1" in by_obj
        assert "object_2" in by_obj
        assert "object_3" in by_obj
        assert "object_15" in by_obj

        assert any("ball" in w for w in by_obj["object_1"]["words"])
        assert "speak" in by_obj["object_1"]["sources"]
        assert by_obj["object_1"]["creator_label"] == "large tree (east)"

        assert any("stick" in w for w in by_obj["object_2"]["words"])
        assert "speak" in by_obj["object_2"]["sources"]

        assert any("stone" in w for w in by_obj["object_3"]["words"])
        assert "thought" in by_obj["object_3"]["sources"]

        assert any("mailbox" in w for w in by_obj["object_15"]["words"])
        assert "belief" in by_obj["object_15"]["sources"]
        assert by_obj["object_15"]["creator_label"] == "mailbox"


def test_extract_object_names_unit() -> None:
    """Direct unit test for `extract_object_names` across speak, thought, and belief inputs."""
    extracted = extract_object_names(
        speeches=[{"tick": 2, "text": "Hello shiny berry (object_6)!"}],
        thoughts=[{"tick": 3, "text": "I can use the long vine (object_4) to tie things."}],
        beliefs=[
            {
                "id": "belief_1",
                "statement": "object_8 (hollow log) can hold small items.",
                "updated_tick": 5,
            }
        ],
    )
    by_obj = extracted["by_object"]
    assert "object_6" in by_obj
    assert any("berry" in w for w in by_obj["object_6"]["words"])
    assert "object_4" in by_obj
    assert any("vine" in w for w in by_obj["object_4"]["words"])
    assert "object_8" in by_obj
    assert any("log" in w for w in by_obj["object_8"]["words"])


def test_static_console_html_and_js_cover_all_t24_views() -> None:
    """Static HTML/JS/CSS include sessionStorage login, spectator mode hiding, inbox, controls, timeline, beliefs, and names."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    state_js = (STATIC_DIR / "state.js").read_text(encoding="utf-8")
    ws_js = (STATIC_DIR / "ws.js").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    # 1. Login with sessionStorage & spectator mode hiding controls
    assert "sessionStorage" in ws_js
    assert "arden_creator_token" in ws_js
    assert 'id="creator-token-input"' in html
    assert 'id="btn-spectator-toggle"' in html
    assert "data-creator-only" in html
    assert 'data-spectator="true"' in css

    # 2. Inbox panel with reply, spawn-in-response, and private note
    assert 'id="panel-inbox"' in html
    assert 'id="inbox-private-note"' in html
    assert 'id="inbox-reply-text"' in html
    assert 'id="btn-inbox-spawn-response"' in html
    assert "/inbox/note" in app_js
    assert "/inbox/reply" in app_js
    assert "/inbox/spawn-in-response" in app_js

    # 3. Controls: preset palette, property form, despawn, drag-to-move, weather, time slider/freeze, pause/step/speed
    assert 'id="preset-palette"' in html
    assert 'id="btn-spawn-object"' in html
    assert 'id="btn-despawn-selected"' in html
    assert "draggingObjectId" in app_js
    assert 'id="weather-type-select"' in html
    assert 'id="weather-automation-checkbox"' in html
    assert 'id="time-slider"' in html
    assert 'id="freeze-time-checkbox"' in html
    assert 'id="btn-pause-resume"' in html
    assert 'id="btn-step"' in html
    assert 'id="btn-set-speed"' in html

    # 4. Timeline panel with live stream and kind filters
    assert 'id="panel-timeline"' in html
    assert 'id="timeline-filters"' in html
    for kind in ("all", "wake", "thought", "intent", "action"):
        assert f'data-timeline-filter="{kind}"' in html
    assert "filterTimelineEntries" in state_js

    # 5. Belief browser with confidence history chart & self-model versions
    assert 'id="panel-beliefs"' in html
    assert 'id="self-model-versions"' in html
    assert "renderBeliefHistoryChart" in app_js
    assert "belief-history-chart" in app_js

    # 6. Names panel mapping agent words to internal object IDs
    assert 'id="panel-names"' in html
    assert 'id="names-list"' in html
    assert "extractNamesFromText" in state_js
