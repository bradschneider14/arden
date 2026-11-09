from __future__ import annotations

from pathlib import Path

from arden.actions.verbs import (
    NOTHING_HAPPENED,
    OUT_OF_REACH,
    grab,
    in_reach,
    insert,
    look,
    separate,
    speak,
)
from arden.agent.gate import RuleGate
from arden.config import MailboxConfig, Settings, load_config
from arden.mailbox import CreatorInbox, reply, spawn_in_response
from arden.perception import look as feed_look
from arden.perception import perceive
from arden.world import (
    PERCEPTIBLE_STATE_KEYS,
    MailArrivedEvent,
    World,
    load_first_run_world,
)


def _make_mailbox_world(
    *,
    db_path: Path | str = ":memory:",
    lid_drift_chance: float = 0.0,
    delivery_delay_s: float = 10.0,
    creator_note: str | None = None,
    seed: int = 42,
) -> tuple[World, str]:
    cfg = MailboxConfig(
        lid_drift_chance=lid_drift_chance,
        delivery_delay_s=delivery_delay_s,
        db_path=Path(db_path),
        creator_note=creator_note,
    )
    world = World(seed=seed, mailbox=cfg)
    world.move_agent((8.0, 2.0))
    mb = world.spawn(
        {
            "id": "object_15",
            "creator_label": "mailbox",
            "level": "ground",
            "shape": "cube",
            "height": 0.9,
            "volume": 0.25,
            "weight": 15.0,
            "rigidity": 0.9,
            "hollow": True,
            "fixed": True,
            "color": "gray",
            "state": {"lid": "closed"},
            "behavior": [
                {"verb": "grab", "response": "toggle_lid"},
                {"verb": "separate", "response": "toggle_lid"},
                {"verb": "speak", "response": "capture_mail"},
                {"verb": "insert", "response": "deliver_mail"},
            ],
        },
        (8.5, 1.5),
    )
    world.clear_events()
    return world, mb.id


def test_lid_state_is_perceptible_and_toggled_by_grab_and_separate() -> None:
    assert "lid" in PERCEPTIBLE_STATE_KEYS
    assert "text" in PERCEPTIBLE_STATE_KEYS

    world, mb_id = _make_mailbox_world()
    mb = world.get_object(mb_id)
    assert mb.lid == "closed"
    assert mb.perceptible()["state"]["lid"] == "closed"

    p0 = perceive(world)
    mb_percept = next(o for o in p0.objects if o.id == mb_id)
    assert mb_percept.state == {"lid": "closed"}

    # grab on the mailbox toggles lid from closed -> open (does not pick up the fixed mailbox)
    res_open = grab(world, mb_id)
    assert res_open.ok is True
    assert res_open.data["lid"] == "open"
    assert res_open.data["object_id"] == mb_id
    assert world.body.holding is None
    assert world.get_object(mb_id).state["lid"] == "open"

    p1 = perceive(world)
    mb_percept_open = next(o for o in p1.objects if o.id == mb_id)
    assert mb_percept_open.state == {"lid": "open"}

    # separate on the mailbox toggles lid from open -> closed
    res_close = separate(world, mb_id)
    assert res_close.ok is True
    assert res_close.data["lid"] == "closed"
    assert res_close.data["object_id"] == mb_id
    assert world.get_object(mb_id).state["lid"] == "closed"

    # separate again toggles closed -> open, and grab again toggles open -> closed
    assert separate(world, mb_id).data["lid"] == "open"
    assert grab(world, mb_id).data["lid"] == "closed"


def test_closed_lid_hides_contents_from_perception_and_reach() -> None:
    world, mb_id = _make_mailbox_world()
    # Place an item inside the closed mailbox
    item = world.spawn(
        {
            "shape": "sphere",
            "height": 0.1,
            "volume": 0.01,
            "weight": 0.2,
            "rigidity": 0.8,
            "color": "red",
            "level": f"inside:{mb_id}",
        },
        (8.5, 1.5),
    )

    # While lid is closed: item is neither perceptible nor reachable
    p_closed = perceive(world)
    assert item.id not in {o.id for o in p_closed.objects}
    assert item.id not in p_closed.affordances
    assert in_reach(world, item.id) is False
    assert look(world, item.id).ok is False
    assert look(world, item.id).reason == OUT_OF_REACH
    assert grab(world, item.id).ok is False
    assert grab(world, item.id).reason == OUT_OF_REACH

    # Inserting into a closed mailbox also fails
    pebble = world.spawn(
        {"shape": "sphere", "height": 0.1, "volume": 0.01, "weight": 0.1, "rigidity": 0.9},
        (8.0, 2.0),
    )
    assert grab(world, pebble.id).ok is True
    ins_closed = insert(world, pebble.id, mb_id)
    assert ins_closed.ok is False
    assert ins_closed.reason == NOTHING_HAPPENED

    # Drop pebble and open the mailbox lid
    world.body.holding = None
    assert separate(world, mb_id).ok is True
    assert world.get_object(mb_id).lid == "open"

    # With lid open: item inside is perceptible and reachable
    p_open = perceive(world)
    assert item.id in {o.id for o in p_open.objects}
    assert in_reach(world, item.id) is True
    assert look(world, item.id).ok is True
    grab_item = grab(world, item.id)
    assert grab_item.ok is True
    assert world.body.holding == item.id


def test_speak_with_lid_closed_or_out_of_reach_is_not_captured() -> None:
    world, mb_id = _make_mailbox_world()

    # 1. Speak while in reach but lid is closed
    res1 = speak(world, "hello closed box")
    assert res1.ok is True
    assert world.mailbox.pending_speech(mb_id) == []
    for _ in range(150):
        world.step()
    assert world.inbox.list_entries() == []

    # 2. Open lid, walk out of reach, and speak
    assert grab(world, mb_id).ok is True
    assert world.get_object(mb_id).lid == "open"
    world.move_agent((2.0, 2.0))
    assert in_reach(world, mb_id) is False

    res2 = speak(world, "hello from far away")
    assert res2.ok is True
    assert world.mailbox.pending_speech(mb_id) == []
    for _ in range(150):
        world.step()
    assert world.inbox.list_entries() == []


def test_speak_with_lid_open_delivers_on_close_or_after_timeout(tmp_path: Path) -> None:
    db_file = tmp_path / "inbox.db"
    world, mb_id = _make_mailbox_world(
        db_path=db_file,
        delivery_delay_s=10.0,
        creator_note="experiment-alpha",
    )

    # Open the lid and speak twice within reach
    assert grab(world, mb_id).ok is True
    assert speak(world, "first line").ok is True
    assert speak(world, "second line").ok is True
    assert world.mailbox.pending_speech(mb_id) == ["first line", "second line"]
    assert world.inbox.list_entries() == []

    # Close the lid -> delivers immediately to SQLite inbox
    close_res = separate(world, mb_id)
    assert close_res.ok is True
    # Nothing in ActionResult or Percept acknowledges delivery to the agent
    assert "delivered" not in close_res.data
    assert "inbox" not in close_res.data
    percept_after_close = perceive(world)
    assert "first line" not in percept_after_close.to_text()

    entries = world.inbox.list_entries()
    assert len(entries) == 1
    entry1 = entries[0]
    assert entry1.mailbox_id == mb_id
    assert entry1.text == "first line\nsecond line"
    assert entry1.texts == ["first line", "second line"]
    assert entry1.object_id is None
    assert entry1.delivery_reason == "lid_closed"
    assert entry1.tick == world.tick
    assert entry1.timestamp.endswith("+00:00")
    assert entry1.creator_note == "experiment-alpha"

    # Re-open lid, speak, and wait for the 10s (100 ticks at 10 Hz) timeout without closing lid
    assert grab(world, mb_id).ok is True
    assert speak(world, "timed message").ok is True
    for _ in range(99):
        world.step()
    assert len(world.inbox.list_entries()) == 1

    world.step()  # 100th tick -> timeout delivery fires
    entries = world.inbox.list_entries()
    assert len(entries) == 2
    entry2 = entries[1]
    assert entry2.text == "timed message"
    assert entry2.delivery_reason == "timeout"
    assert entry2.tick == world.tick

    # Verify persistence by opening a fresh CreatorInbox on the same SQLite file
    reopened = CreatorInbox(db_file)
    persisted = reopened.list_entries()
    assert len(persisted) == 2
    assert persisted[0].text == "first line\nsecond line"
    assert persisted[1].text == "timed message"
    reopened.close()


def test_insert_into_open_mailbox_removes_object_and_delivers_with_speech() -> None:
    world, mb_id = _make_mailbox_world()

    token = world.spawn(
        {
            "id": "object_42",
            "shape": "flat",
            "height": 0.05,
            "volume": 0.01,
            "weight": 0.3,
            "rigidity": 0.9,
            "color": "blue",
        },
        (8.2, 1.8),
    )

    # Open mailbox, grab token, speak, and insert token into mailbox
    assert grab(world, mb_id).ok is True
    assert grab(world, token.id).ok is True
    assert world.body.holding == token.id

    assert speak(world, "here is a blue flat thing").ok is True
    ins_res = insert(world, token.id, mb_id)
    assert ins_res.ok is True
    assert world.body.holding is None

    # Object is removed from the world
    assert token.id not in world.objects

    # Delivered to the inbox together with the speech captured in the same window
    entries = world.inbox.list_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.text == "here is a blue flat thing"
    assert entry.object_id == token.id
    assert entry.delivery_reason == "insert"
    assert entry.object is not None
    assert entry.object.id == token.id
    assert entry.object.color == "blue"
    assert entry.object.shape == "flat"
    assert world.mailbox.pending_speech(mb_id) == []

    # Creator can return the preserved object into the mailbox
    returned = world.return_object(entry)
    assert returned.id == token.id
    assert returned.id in world.objects
    assert returned.level == f"inside:{mb_id}"
    assert returned.color == "blue"


def test_reply_letter_text_visible_only_via_look_with_lid_open() -> None:
    world, mb_id = _make_mailbox_world()
    assert world.get_object(mb_id).lid == "closed"

    # Creator sends a reply while the mailbox lid is closed
    letter = reply(world, "seek the stone by the water")
    assert letter.shape == "flat"
    assert letter.level == f"inside:{mb_id}"
    assert letter.text == "seek the stone by the water"

    # 1. With lid closed: letter is invisible to perceive() and unreachable for look()
    p_closed = perceive(world)
    assert letter.id not in {o.id for o in p_closed.objects}
    assert "seek the stone" not in p_closed.to_text()
    assert "seek the stone" not in p_closed.to_json()

    look_closed = look(world, letter.id)
    assert look_closed.ok is False
    assert look_closed.reason == OUT_OF_REACH
    assert "text" not in feed_look(world, letter.id).state

    # 2. Open the lid: ambient perceive() sees the flat letter object inside,
    #    but does NOT reveal its text until look() is called!
    assert grab(world, mb_id).ok is True
    p_open = perceive(world)
    perceived_letter = next(o for o in p_open.objects if o.id == letter.id)
    assert perceived_letter.shape == "flat"
    assert perceived_letter.level == f"inside:{mb_id}"
    assert perceived_letter.state == {}
    assert "text" not in perceived_letter.to_dict().get("state", {})
    assert "seek the stone" not in p_open.to_text()
    assert "seek the stone" not in p_open.to_json()

    # 3. Calling look() on the letter with the lid open reveals its text
    look_open = look(world, letter.id)
    assert look_open.ok is True
    assert look_open.data["text"] == "seek the stone by the water"
    assert look_open.data["state"]["text"] == "seek the stone by the water"

    feed_looked = feed_look(world, letter.id)
    assert feed_looked is not None
    assert feed_looked.state["text"] == "seek the stone by the water"


def test_spawn_in_response_places_object_inside_with_no_note() -> None:
    world, mb_id = _make_mailbox_world()

    spawned = spawn_in_response(
        world,
        {
            "shape": "long",
            "height": 0.8,
            "volume": 0.04,
            "weight": 0.9,
            "rigidity": 0.95,
            "color": "brown",
        },
    )
    assert spawned.id in world.objects
    assert spawned.level == f"inside:{mb_id}"
    assert spawned.text is None
    assert "text" not in spawned.state

    # Open the lid and verify the spawned object is inside and reachable
    assert grab(world, mb_id).ok is True
    p_open = perceive(world)
    assert spawned.id in {o.id for o in p_open.objects}
    assert grab(world, spawned.id).ok is True
    assert world.body.holding == spawned.id


def test_no_extra_affordance_or_mailbox_specific_novelty_in_first_run() -> None:
    world = load_first_run_world(seed=5)
    # Move agent next to the mailbox (object_15 at 6.5, 1.2)
    world.move_agent((6.5, 1.8))
    assert in_reach(world, "object_15") is True

    p = perceive(world)
    mb_affordances = p.affordances.get("object_15", [])
    # Only "insert" (standard for any hollow object in reach) is listed;
    # no opaque action_1, no speak, no mailbox-specific hint.
    assert mb_affordances == ["insert"]
    assert not any(a.startswith("action_") for a in mb_affordances)
    assert "speak" not in mb_affordances

    # Compare with the other hollow object in first_run.toml (object_6, hollow log):
    # both expose "insert" as their hollow container affordance.
    world.move_agent((5.5, 4.8))
    p_log = perceive(world)
    assert "insert" in p_log.affordances.get("object_6", [])


def test_arrivals_in_mailbox_wake_rule_gate_with_mail_arrived() -> None:
    world, mb_id = _make_mailbox_world()
    gate = RuleGate(heartbeat_interval=50)

    # Initial tick wakes on start/novelty of the mailbox itself
    p0 = perceive(world)
    d0 = gate.should_wake(world=world, percept=p0)
    assert d0.wake is True

    # Next tick with no changes does not wake
    world.step()
    p1 = perceive(world)
    d1 = gate.should_wake(world=world, percept=p1)
    assert d1.wake is False

    # Agent speaking into the open mailbox (outgoing mail) does NOT trigger mail_arrived
    assert grab(world, mb_id).ok is True
    assert speak(world, "anyone out there?").ok is True
    assert separate(world, mb_id).ok is True
    assert len(world.inbox.list_entries()) == 1
    p_after_send = perceive(world)
    d_send = gate.should_wake(world=world, percept=p_after_send)
    assert "mail_arrived" not in d_send.reasons

    # Creator reply arrives in the closed mailbox -> emits MailArrivedEvent and wakes gate
    events_seen: list[MailArrivedEvent] = []
    world.subscribe("mail_arrived", lambda e: events_seen.append(e))  # type: ignore[arg-type]

    letter = world.reply("hello from creator")
    assert len(events_seen) == 1
    assert events_seen[0].mailbox_id == mb_id
    assert events_seen[0].object_id == letter.id
    assert events_seen[0].type == "mail_arrived"
    assert events_seen[0].arrival_kind == "reply"

    p2 = perceive(world)
    d2 = gate.should_wake(world=world, percept=p2)
    assert d2.wake is True
    assert "mail_arrived" in d2.reasons

    # Subsequent tick without new arrivals does not re-trigger mail_arrived
    world.step()
    p3 = perceive(world)
    d3 = gate.should_wake(world=world, percept=p3)
    assert d3.wake is False
    assert "mail_arrived" not in d3.reasons

    # spawn_in_response also triggers mail_arrived
    gift = world.spawn_in_response({"shape": "sphere", "height": 0.1, "volume": 0.01, "weight": 0.2})
    assert gift.level == f"inside:{mb_id}"
    p4 = perceive(world)
    d4 = gate.should_wake(world=world, percept=p4)
    assert d4.wake is True
    assert "mail_arrived" in d4.reasons


def test_lid_drift_chance_zero_vs_positive() -> None:
    # 1. Default lid_drift_chance = 0.0 never opens a closed lid
    world_zero, mb_id_zero = _make_mailbox_world(lid_drift_chance=0.0, seed=123)
    for _ in range(500):
        world_zero.step()
    assert world_zero.get_object(mb_id_zero).lid == "closed"

    # 2. High lid_drift_chance swings the closed lid open on its own
    world_drift, mb_id_drift = _make_mailbox_world(lid_drift_chance=0.95, seed=123)
    opened_at_tick: int | None = None
    for _ in range(100):
        world_drift.step()
        if world_drift.get_object(mb_id_drift).lid == "open":
            opened_at_tick = world_drift.tick
            break
    assert opened_at_tick is not None


def test_creator_and_mail_logging_with_private_experiment_note() -> None:
    world, mb_id = _make_mailbox_world(creator_note="exp-silent-then-gift")
    assert world.creator_note == "exp-silent-then-gift"

    assert grab(world, mb_id).ok is True
    assert speak(world, "testing log").ok is True
    assert separate(world, mb_id).ok is True

    world.reply("logged reply")
    world.set_creator_note("exp-phase-2")
    world.spawn_in_response({"shape": "cube", "height": 0.1, "volume": 0.01, "weight": 0.4})

    logs = world.mailbox_logs
    kinds = [entry.kind for entry in logs]
    assert "send" in kinds
    assert "delivery" in kinds
    assert "reply" in kinds
    assert "creator_action" in kinds
    assert "spawn_in_response" in kinds

    for entry in logs:
        assert isinstance(entry.tick, int)
        assert entry.timestamp.endswith("+00:00")
        assert entry.unix_timestamp > 0

    reply_log = next(e for e in logs if e.kind == "reply")
    assert reply_log.text == "logged reply"
    assert reply_log.creator_note == "exp-silent-then-gift"

    spawn_log = next(e for e in logs if e.kind == "spawn_in_response")
    assert spawn_log.creator_note == "exp-phase-2"


def test_example_config_includes_mailbox_settings() -> None:
    settings = load_config(Path("arden.example.toml"))
    assert isinstance(settings, Settings)
    assert settings.mailbox.lid_drift_chance == 0.0
    assert settings.mailbox.delivery_delay_s == 10.0
    assert settings.mailbox.db_path == Path("data/mailbox.db")
