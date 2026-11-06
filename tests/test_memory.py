"""Tests for T13 & T14: short-term buffer, episodic memory store (SQLite), reflection, decay, semantic beliefs, self-model, and relevance retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arden.actions.verbs import bind, grab, move
from arden.agent import AgentLoop, HeartbeatGate, build_test_world
from arden.config import MemoryConfig, MemorySettings, Settings, load_config
from arden.llm import Intent, MockLLM, Step, build_decide_prompt
from arden.memory import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_HOOK,
    DEFAULT_MAX_SELF_MODEL_WORDS,
    Belief,
    BeliefEvictedEvent,
    BeliefStore,
    BufferEntry,
    BufferFullEvent,
    ConfidenceChange,
    Episode,
    EpisodeStore,
    MemoryManager,
    ReflectionBelief,
    ReflectionEvent,
    ReflectionOutput,
    SelfModel,
    SelfModelVersion,
    ShortTermBuffer,
    compute_episode_salience,
    compute_recency,
    reflect,
    retrieve,
    truncate_self_model,
)
from arden.perception import perceive
from arden.world import World


def test_short_term_buffer_defaults_and_emits_buffer_full() -> None:
    """ShortTermBuffer defaults to capacity 40 from config and emits buffer_full when filled."""
    default_buf = ShortTermBuffer()
    assert default_buf.capacity == DEFAULT_BUFFER_SIZE == 40
    assert Settings().memory.buffer_size == 40
    assert len(default_buf) == 0
    assert default_buf.is_full is False

    world = World(seed=1)
    buf = ShortTermBuffer(capacity=4, world=world)
    buf_events: list[BufferFullEvent] = []
    world_buf_events: list[BufferFullEvent] = []

    buf.subscribe("buffer_full", buf_events.append)
    world.subscribe("buffer_full", world_buf_events.append)

    for i in range(3):
        buf.add(
            tick=i,
            observation=f"tick={i} saw object_3",
            action=Step(verb="look", target="object_3"),
            outcome={"ok": True},
        )
        assert buf.is_full is False
        assert len(buf_events) == 0

    # Adding the 4th item fills the buffer and emits buffer_full on both buses
    buf.add(
        tick=3,
        observation="tick=3 pushed object_3",
        action=Step(verb="push", target="object_3", params={"force": 0.5}),
        outcome={"ok": True},
    )
    assert len(buf) == 4
    assert buf.is_full is True
    assert len(buf_events) == 1
    assert len(world_buf_events) == 1
    assert buf_events[0].type == "buffer_full"
    assert buf_events[0].tick == 3
    assert buf_events[0].count == 4
    assert buf_events[0].capacity == 4

    # Fixed-size window: adding a 5th item without clearing keeps the most recent 4 entries
    buf.add(tick=4, observation="tick=4 extra")
    assert len(buf) == 4
    assert [e.tick for e in buf] == [1, 2, 3, 4]


def test_buffer_fills_triggers_reflection_and_saves_episodes_to_sqlite(tmp_path: Path) -> None:
    """Full cycle: buffer fills -> reflection runs (MockLLM reflect fixture) -> episodes saved to SQLite."""
    db_file = tmp_path / "data" / "episodes.db"
    mock_llm = MockLLM(seed=7)

    with EpisodeStore(db_path=db_file) as store:
        buf = ShortTermBuffer(
            capacity=5,
            llm=mock_llm,
            store=store,
        )
        full_events: list[BufferFullEvent] = []
        reflect_events: list[ReflectionEvent] = []
        buf.subscribe("buffer_full", full_events.append)
        buf.subscribe("reflection", reflect_events.append)

        for t in range(5):
            buf.add(
                tick=t + 1,
                observation=f"tick={t + 1} object_7 dist=1.2 dir=E",
                action=Step(verb="push", target="object_7", params={"force": 0.5}),
                outcome={"ok": True},
                comfort=0.8,
            )

        # Filling the 5-item buffer emitted buffer_full, ran reflection, saved episodes, and cleared buffer
        assert len(full_events) == 1
        assert len(reflect_events) == 1
        assert reflect_events[0].trigger == "buffer_full"
        assert len(buf) == 0

        saved = store.all()
        assert len(saved) == 2
        for ep in saved:
            assert isinstance(ep, Episode)
            assert ep.id is not None
            assert ep.tick == 5
            assert ep.last_recalled_tick == 5
            assert ep.summary
            assert 0.0 <= ep.salience <= 1.0
            assert len(ep.tags) > 0

        # Check tags (object ids and verbs), surprise_kind, and made_by_agent from canned fixture
        ep1, ep2 = saved
        assert "object_7" in ep1.tags
        assert "push" in ep1.tags
        assert ep1.surprise_kind == "physical"
        assert ep1.made_by_agent == []

        assert "object_8" in ep2.tags
        assert "bind" in ep2.tags
        assert ep2.surprise_kind is None
        assert ep2.made_by_agent == ["object_8"]

    # Re-open the SQLite file in a new EpisodeStore instance to verify disk persistence
    with EpisodeStore(db_path=db_file) as reopened:
        persisted = reopened.all()
        assert len(persisted) == 2
        assert persisted[0].summary == ep1.summary
        assert persisted[0].salience == pytest.approx(ep1.salience)
        assert persisted[0].tick == 5
        assert persisted[0].last_recalled_tick == 5
        assert persisted[1].made_by_agent == ["object_8"]


def test_reflection_triggered_by_rest_idle_for_m_ticks(tmp_path: Path) -> None:
    """Resting (idle for M ticks) triggers reflection even when the buffer is not full."""
    db_file = tmp_path / "rest_memory.db"
    mock_llm = MockLLM(seed=11)

    with EpisodeStore(db_path=db_file) as store:
        world = World(seed=2)
        buf = ShortTermBuffer(
            capacity=40,
            rest_idle_ticks=4,
            world=world,
            llm=mock_llm,
            store=store,
        )

        # Record 2 active steps (far below capacity 40)
        buf.add(
            tick=1,
            observation=perceive(world),
            action=Step(verb="look", target="object_1"),
            outcome={"ok": True},
        )
        buf.add(
            tick=2,
            observation=perceive(world),
            action=Step(verb="move", params={"direction": "N", "speed": 0.5}),
            outcome={"ok": True},
        )
        assert len(buf) == 2
        assert buf.idle_ticks == 0
        assert store.count() == 0

        # Rest for 3 ticks (< M=4): reflection has not run yet
        for _ in range(3):
            world.step(resting=True)
        assert buf.idle_ticks == 3
        assert len(buf) == 2
        assert store.count() == 0

        # 4th resting tick reaches M=4 -> triggers reflection and saves episodes
        world.step(resting=True)
        assert len(buf) == 0
        assert len(buf.reflection_events) == 1
        assert buf.reflection_events[0].trigger == "rest"
        assert store.count() == 2


def test_decay_deletes_low_salience_episode_and_recall_refreshes_it(tmp_path: Path) -> None:
    """Decay lowers salience over world time and deletes episodes below floor; recall refreshes them."""
    db_file = tmp_path / "decay_test.db"

    with EpisodeStore(
        db_path=db_file,
        decay_rate=0.005,
        salience_floor=0.10,
        recall_boost=0.15,
    ) as store:
        ep_low_forgotten = store.add(
            summary="Observed object_2 sitting still on the grass.",
            tags=["object_2", "look"],
            salience=0.22,
            tick=10,
        )
        ep_low_recalled = store.add(
            summary="Pushed object_7 and saw it roll toward object_1.",
            tags=["object_7", "object_1", "push"],
            salience=0.22,
            tick=10,
        )
        ep_high = store.add(
            summary="Bound object_4 and object_5 into composite object_8.",
            tags=["object_4", "object_5", "object_8", "bind"],
            salience=0.80,
            made_by_agent=["object_8"],
            tick=10,
        )

        # Advance to tick 25 (15 ticks elapsed): 0.22 - 15 * 0.005 = 0.145 >= 0.10
        surviving_25 = store.decay(current_tick=25)
        assert len(surviving_25) == 3

        # Recall ep_low_recalled at tick 25: refreshes last_recalled_tick and boosts salience
        refreshed = store.recall(ep_low_recalled.id, tick=25)
        assert isinstance(refreshed, Episode)
        assert refreshed.last_recalled_tick == 25
        assert refreshed.recall_count == 1
        assert refreshed.salience == pytest.approx(0.22 + 0.15)

        # Advance to tick 45 (20 more ticks; 35 total since tick 10):
        # - ep_low_forgotten decays to 0.145 - 20 * 0.005 = 0.045 < 0.10 -> DELETED (real forgetting)
        # - ep_low_recalled decays from 0.37 by 20 * 0.005 = 0.27 >= 0.10 -> SURVIVES
        # - ep_high decays to 0.80 - 35 * 0.005 = 0.625 >= 0.10 -> SURVIVES
        surviving_45 = store.decay(current_tick=45)
        surviving_ids = {ep.id for ep in surviving_45}

        assert ep_low_forgotten.id not in surviving_ids
        assert store.get(ep_low_forgotten.id) is None
        assert len(store.forgotten_episodes) == 1
        assert store.forgotten_episodes[0].id == ep_low_forgotten.id

        kept_recalled = store.get(ep_low_recalled.id)
        assert kept_recalled is not None
        assert kept_recalled.last_recalled_tick == 25
        assert kept_recalled.salience == pytest.approx(0.27)

        kept_high = store.get(ep_high.id)
        assert kept_high is not None
        assert kept_high.salience == pytest.approx(0.625)

        # Also verify recall by tag refreshes matching episodes
        tag_recalled = store.recall_by_tag("object_8", tick=50)
        assert len(tag_recalled) == 1
        assert tag_recalled[0].id == ep_high.id
        assert tag_recalled[0].last_recalled_tick == 50


def test_comfort_and_surprise_raise_salience(tmp_path: Path) -> None:
    """Uncomfortable (low comfort) and surprising episodes receive higher salience."""
    # 1. Direct salience computation checks
    s_neutral = compute_episode_salience(0.40, comfort=1.0, surprise=0.0, surprise_kind=None)
    s_uncomfortable = compute_episode_salience(0.40, comfort=0.15, surprise=0.0, surprise_kind=None)
    s_surprising_num = compute_episode_salience(0.40, comfort=1.0, surprise=0.75, surprise_kind=None)
    s_surprising_phys = compute_episode_salience(0.40, comfort=1.0, surprise=0.0, surprise_kind="physical")
    s_surprising_exp = compute_episode_salience(0.40, comfort=1.0, surprise=0.0, surprise_kind="expectation")
    s_surprising_belief = compute_episode_salience(0.40, comfort=1.0, surprise=0.0, surprise_kind="belief")
    s_both = compute_episode_salience(0.40, comfort=0.15, surprise=0.75, surprise_kind="expectation")

    assert s_neutral == pytest.approx(0.40)
    assert s_uncomfortable > s_neutral
    assert s_surprising_num > s_neutral
    assert s_surprising_phys > s_neutral
    # Expectation and belief surprises outrank physical ones (docs/DESIGN.md 10.4)
    assert s_surprising_exp > s_surprising_phys
    assert s_surprising_belief >= s_surprising_exp
    assert s_both > s_uncomfortable
    assert s_both > s_surprising_exp
    assert 0.0 <= s_both <= 1.0

    # 2. Full reflection cycle checks with comfortable vs uncomfortable and calm vs surprising buffers
    mock_llm = MockLLM(seed=3)

    buf_comfortable = ShortTermBuffer(capacity=4)
    buf_uncomfortable = ShortTermBuffer(capacity=4)
    buf_surprised = ShortTermBuffer(capacity=4)

    for t in range(3):
        buf_comfortable.add(tick=t, observation="obs", action="look", comfort=1.0, surprise=0.0)
        buf_uncomfortable.add(tick=t, observation="obs", action="look", comfort=0.1, surprise=0.0)
        buf_surprised.add(
            tick=t,
            observation="obs",
            action="push",
            comfort=1.0,
            surprise=0.85,
            surprise_kind="expectation",
        )

    eps_comfortable = reflect(buf_comfortable, mock_llm)
    eps_uncomfortable = reflect(buf_uncomfortable, mock_llm)
    eps_surprised = reflect(buf_surprised, mock_llm)

    # Every episode from the uncomfortable buffer has strictly higher salience than from the comfortable buffer
    for ep_c, ep_u in zip(eps_comfortable, eps_uncomfortable, strict=True):
        assert ep_u.salience > ep_c.salience

    # Every episode from the surprised buffer has strictly higher salience than from the calm buffer
    for ep_c, ep_s in zip(eps_comfortable, eps_surprised, strict=True):
        assert ep_s.salience > ep_c.salience


def test_made_by_agent_references_captured_from_world_and_buffer(tmp_path: Path) -> None:
    """Built composites with made_by_agent=True are tracked in buffer entries and reflected episodes."""
    world = World(seed=5)
    s1 = world.spawn(
        {"shape": "long", "height": 0.8, "volume": 0.02, "weight": 0.4, "rigidity": 0.8},
        (0.5, 0.0),
    )
    s2 = world.spawn(
        {"shape": "flat", "height": 0.2, "volume": 0.05, "weight": 1.2, "rigidity": 0.9},
        (0.6, 0.0),
    )
    vine = world.spawn(
        {"shape": "long", "height": 0.5, "volume": 0.01, "weight": 0.1, "rigidity": 0.2, "binding": True},
        (0.4, 0.0),
    )

    assert grab(world, s1.id).ok
    res = bind(world, s1.id, s2.id, vine.id)
    assert res.ok
    composite_id = res.data["composite"].id

    buf = ShortTermBuffer(capacity=5, world=world)
    entry = buf.add(
        tick=world.tick,
        observation=perceive(world),
        action=Step(verb="bind", target=s2.id, params={"held": s1.id, "binder": vine.id}),
        outcome=res,
    )
    assert composite_id in entry.made_by_agent
    assert composite_id in entry.tags

    custom_reflect = {
        "episodes": [
            {
                "summary": f"Bound {s1.id} and {s2.id} with {vine.id} to build {composite_id}.",
                "tags": [s1.id, s2.id, composite_id, "bind"],
                "salience": 0.6,
                "surprise_kind": None,
                "made_by_agent": [],
            }
        ]
    }
    llm = MockLLM(script=[json.dumps(custom_reflect)])
    with EpisodeStore(db_path=tmp_path / "built.db") as store:
        episodes = reflect(buf, llm, store=store, world=world)
        assert len(episodes) == 1
        assert composite_id in episodes[0].made_by_agent
        assert store.query(made_by_agent=composite_id) == episodes


def test_store_uses_config_db_path_under_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default DB path comes from Settings.memory.db_path under data/."""
    monkeypatch.chdir(tmp_path)
    settings = load_config()
    assert settings.memory.db_path == Path("data/memory.db") == DEFAULT_DB_PATH

    with EpisodeStore(settings=settings) as store:
        assert store.db_path == Path("data/memory.db")
        assert (tmp_path / "data" / "memory.db").is_file()
        ep = store.add(summary="Tested default data/memory.db path.", tick=1, salience=0.5)
        assert ep.id == 1


def test_memory_manager_coordinates_buffer_store_rest_and_decay(tmp_path: Path) -> None:
    """MemoryManager wires ShortTermBuffer, EpisodeStore, MockLLM, rest triggers, and decay."""
    db_file = tmp_path / "manager.db"
    llm = MockLLM(seed=19)
    mgr = MemoryManager(
        llm,
        db_path=db_file,
        buffer_size=3,
        rest_idle_ticks=2,
    )
    try:
        mgr.record(tick=1, observation="saw object_3", action="look object_3", outcome="ok")
        mgr.record(tick=2, observation="pushed object_3", action="push object_3", outcome="rolled")
        # Trigger via rest (2 idle ticks)
        reflected = mgr.record_idle(2, tick=4)
        assert len(reflected) == 2
        assert mgr.store.count() == 2
        assert len(mgr.buffer) == 0
    finally:
        mgr.store.close()


def test_belief_schema_and_scripted_reflection_confidence_goes_up_then_down(tmp_path: Path) -> None:
    """Beliefs record statement, tags, confidence 0..1, evidence count, created/updated tick, and confidence history; confidence rises on support and falls on contradiction."""
    db_file = tmp_path / "beliefs.db"

    # Scripted reflection 1 (tick 10): propose a new belief at confidence 0.50
    reflect_1 = {
        "episodes": [
            {
                "summary": "Pushed object_7 and saw it roll east.",
                "tags": ["object_7", "push"],
                "salience": 0.5,
            }
        ],
        "beliefs": [
            {
                "statement": "Pushing sphere object_7 causes it to roll.",
                "tags": ["object_7", "push", "sphere"],
                "action": "propose",
                "confidence": 0.50,
            }
        ],
    }
    # Scripted reflection 2 (tick 20): supporting evidence raises confidence
    reflect_2 = {
        "episodes": [
            {
                "summary": "Pushed object_7 again; it rolled smoothly.",
                "tags": ["object_7", "push"],
                "salience": 0.6,
            }
        ],
        "beliefs": [
            {
                "statement": "Pushing sphere object_7 causes it to roll.",
                "tags": ["object_7", "push"],
                "action": "support",
            }
        ],
    }
    # Scripted reflection 3 (tick 35): contradicting evidence lowers confidence
    reflect_3 = {
        "episodes": [
            {
                "summary": "Pushed object_7 against a ledge and it did not roll.",
                "tags": ["object_7", "push"],
                "salience": 0.7,
                "surprise_kind": "belief",
            }
        ],
        "beliefs": [
            {
                "statement": "Pushing sphere object_7 causes it to roll.",
                "tags": ["object_7", "push", "ledge"],
                "action": "contradict",
            }
        ],
    }

    llm = MockLLM(script=[json.dumps(reflect_1), json.dumps(reflect_2), json.dumps(reflect_3)])
    with EpisodeStore(db_path=db_file) as store:
        buf = ShortTermBuffer(capacity=5, llm=llm, store=store, auto_reflect=False)

        # 1. Propose initial belief at tick 10
        buf.add(tick=10, observation="saw object_7", action="push object_7", outcome="rolled")
        res1 = buf.reflect(tick=10)
        assert len(res1.beliefs) == 1
        b1 = store.belief_store.all()[0]
        assert isinstance(b1, Belief)
        assert b1.statement == "Pushing sphere object_7 causes it to roll."
        assert "object_7" in b1.tags
        assert "push" in b1.tags
        assert b1.confidence == pytest.approx(0.50)
        assert b1.evidence_count == 1
        assert b1.created_tick == 10
        assert b1.updated_tick == 10
        assert len(b1.history) == 1
        assert b1.history[0].tick == 10
        assert b1.history[0].confidence == pytest.approx(0.50)
        assert b1.history[0].outcome == "created"

        # 2. Support existing belief at tick 20 -> confidence rises
        buf.add(tick=20, observation="pushed object_7", action="push object_7", outcome="rolled")
        buf.reflect(tick=20)
        b2 = store.belief_store.all()[0]
        assert b2.id == b1.id
        assert b2.confidence > b1.confidence
        assert b2.confidence == pytest.approx(0.65)
        assert b2.evidence_count == 2
        assert b2.created_tick == 10
        assert b2.updated_tick == 20
        assert len(b2.history) == 2
        assert b2.history[1].tick == 20
        assert b2.history[1].previous_confidence == pytest.approx(0.50)
        assert b2.history[1].confidence == pytest.approx(0.65)
        assert b2.history[1].delta > 0.0
        assert b2.history[1].outcome == "support"

        # 3. Contradict existing belief at tick 35 -> confidence falls
        buf.add(tick=35, observation="pushed object_7", action="push object_7", outcome="blocked")
        buf.reflect(tick=35)
        b3 = store.belief_store.all()[0]
        assert b3.id == b1.id
        assert b3.confidence < b2.confidence
        assert b3.confidence == pytest.approx(0.45)
        assert b3.evidence_count == 3
        assert b3.created_tick == 10
        assert b3.updated_tick == 35
        assert "ledge" in b3.tags
        assert len(b3.history) == 3
        assert [h.outcome for h in b3.history] == ["created", "support", "contradict"]
        assert b3.history[2].tick == 35
        assert b3.history[2].previous_confidence == pytest.approx(0.65)
        assert b3.history[2].confidence == pytest.approx(0.45)
        assert b3.history[2].delta < 0.0


def test_hard_cap_on_beliefs_evicts_lowest_confidence_times_recency_and_logs_event() -> None:
    """BeliefStore enforces max_beliefs by evicting the lowest confidence * recency belief and emitting BeliefEvictedEvent."""
    world = World(seed=3)
    bstore = BeliefStore(max_beliefs=3, recency_decay_rate=0.05, world=world)
    store_evictions: list[BeliefEvictedEvent] = []
    world_evictions: list[BeliefEvictedEvent] = []
    bstore.subscribe("belief_evicted", store_evictions.append)
    world.subscribe("belief_evicted", world_evictions.append)

    # Add 3 beliefs at different ticks and confidences:
    # - b_old_moderate: tick=0, confidence=0.55. At tick=100, recency = 1 / (1 + 0.05 * 100) = 1/6 ≈ 0.1667 -> score ≈ 0.0917 (LOWEST!)
    # - b_recent_low:   tick=98, confidence=0.40. At tick=100, recency = 1 / (1 + 0.05 * 2) = 1/1.1 ≈ 0.9091 -> score ≈ 0.3636
    # - b_old_high:     tick=20, confidence=0.95. At tick=100, recency = 1 / (1 + 0.05 * 80) = 1/5 = 0.2000 -> score = 0.1900
    # Notice b_old_moderate has HIGHER confidence than b_recent_low (0.55 > 0.40),
    # and b_old_high is also old, so only `confidence * recency` picks b_old_moderate as the lowest!
    b_old_moderate = bstore.add(
        statement="Object_2 is slightly damp.",
        tags=["object_2"],
        confidence=0.55,
        tick=0,
    )
    b_old_high = bstore.add(
        statement="Object_1 is fixed and does not move when pushed.",
        tags=["object_1", "push"],
        confidence=0.95,
        tick=20,
    )
    b_recent_low = bstore.add(
        statement="Object_5 might bend when pulled.",
        tags=["object_5"],
        confidence=0.40,
        tick=98,
    )

    assert bstore.count() == 3
    assert len(store_evictions) == 0

    # Adding a 4th belief at tick=100 exceeds cap=3 -> evicts b_old_moderate (lowest confidence * recency)
    b_new = bstore.add(
        statement="Object_7 rolls when pushed.",
        tags=["object_7", "push"],
        confidence=0.70,
        tick=100,
    )

    assert bstore.count() == 3
    remaining_ids = {b.id for b in bstore.all()}
    assert b_old_moderate.id not in remaining_ids
    assert b_old_high.id in remaining_ids
    assert b_recent_low.id in remaining_ids
    assert b_new.id in remaining_ids

    assert len(store_evictions) == 1
    assert len(world_evictions) == 1
    ev = store_evictions[0]
    assert ev.type == "belief_evicted"
    assert ev.tick == 100
    assert ev.belief_id == b_old_moderate.id
    assert ev.statement == b_old_moderate.statement
    assert ev.cap == 3
    assert ev.score == pytest.approx(
        b_old_moderate.confidence * compute_recency(100, 0, decay_rate=0.05),
        rel=1e-4,
    )


def test_self_model_rewritten_during_reflection_capped_at_80_words_and_stores_all_versions(
    tmp_path: Path,
) -> None:
    """Self-model is rewritten during reflection, truncated to max ~80 words, with every version stored."""
    overlong_self_model = " ".join(f"word_{i}" for i in range(1, 110))
    assert len(overlong_self_model.split()) == 109

    script = [
        json.dumps(
            {
                "episodes": [{"summary": "Pushed object_3 across the ground.", "tags": ["object_3"], "salience": 0.5}],
                "beliefs": [],
                "self_model": "I test small round objects by pushing them across the clearing.",
            }
        ),
        json.dumps(
            {
                "episodes": [{"summary": "Bound object_4 and object_5 together.", "tags": ["object_4", "object_5"], "salience": 0.6}],
                "beliefs": [],
                "self_model": overlong_self_model,
            }
        ),
    ]
    llm = MockLLM(script=script)
    mgr = MemoryManager(llm, db_path=tmp_path / "self_model.db", buffer_size=4)
    try:
        mgr.record(tick=5, observation="saw object_3", action="push object_3", outcome="ok")
        res1 = mgr.reflect(tick=5)
        assert res1.self_model == "I test small round objects by pushing them across the clearing."
        assert mgr.self_model == "I test small round objects by pushing them across the clearing."
        assert len(mgr.self_model_history) == 1

        mgr.record(tick=15, observation="bound object_4", action="bind object_4", outcome="ok")
        res2 = mgr.reflect(tick=15)
        assert res2.self_model is not None
        assert len(res2.self_model.split()) == DEFAULT_MAX_SELF_MODEL_WORDS == 80
        assert mgr.self_model.split()[0] == "word_1"
        assert mgr.self_model.split()[-1] == "word_80"

        # Every version is stored in chronological order
        history = mgr.self_model_history
        assert len(history) == 2
        assert history[0].version == 1
        assert history[0].tick == 5
        assert history[0].text == "I test small round objects by pushing them across the clearing."
        assert history[1].version == 2
        assert history[1].tick == 15
        assert history[1].word_count == 80
    finally:
        mgr.store.close()


def test_retrieve_ranks_belief_tagged_with_nearby_object_above_unrelated_ones_and_supports_embedding_hook() -> None:
    """retrieve(context_tags, k) ranks a belief tagged with the nearby object above unrelated beliefs and episodes."""
    assert DEFAULT_EMBEDDING_HOOK is None

    bstore = BeliefStore(max_beliefs=10)
    b_unrelated_1 = bstore.add(
        statement="Object_2 is heavy and flat.",
        tags=["object_2", "flat"],
        confidence=0.85,
        tick=20,
    )
    b_unrelated_2 = bstore.add(
        statement="Object_6 is hollow and can hold smaller items.",
        tags=["object_6", "insert"],
        confidence=0.80,
        tick=22,
    )
    b_nearby = bstore.add(
        statement="Object_7 rolls easily toward object_1 when pushed.",
        tags=["object_7", "push"],
        confidence=0.65,
        tick=15,
    )

    with EpisodeStore(db_path=":memory:") as ep_store:
        ep_unrelated = ep_store.add(
            summary="Walked north across the grass near object_2.",
            tags=["object_2", "move"],
            salience=0.75,
            tick=21,
        )
        ep_nearby = ep_store.add(
            summary="Pushed object_7 and watched it roll.",
            tags=["object_7", "push"],
            salience=0.70,
            tick=18,
        )

        # Retrieve top-2 for context_tags=["object_7"]
        top_2 = retrieve(["object_7"], 2)
        assert len(top_2) == 2
        # Both items tagged with object_7 outrank the higher-confidence unrelated items
        top_texts = [item.text for item in top_2]
        assert b_nearby.statement in top_texts
        assert ep_nearby.summary in top_texts
        assert b_unrelated_1.statement not in top_texts
        assert b_unrelated_2.statement not in top_texts
        assert ep_unrelated.summary not in top_texts

        # Retrieve beliefs only: b_nearby is ranked #1 above b_unrelated_1 and b_unrelated_2
        ranked_beliefs = retrieve(["object_7"], 3, beliefs=bstore.all(), episodes=[])
        assert len(ranked_beliefs) == 3
        assert ranked_beliefs[0].id == b_nearby.id
        assert ranked_beliefs[0].relevance > 0.0
        assert ranked_beliefs[1].relevance == 0.0
        assert ranked_beliefs[0].score > ranked_beliefs[1].score

        # Optional embedding hook is unused by default, but boosts semantic match when provided
        called_texts: list[str] = []

        def custom_embed_hook(text: str) -> list[float]:
            called_texts.append(text)
            if "hollow" in text.lower() or "container" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]

        sem_results = retrieve(
            ["container"],
            1,
            beliefs=bstore.all(),
            episodes=[],
            embedding_fn=custom_embed_hook,
        )
        assert len(called_texts) > 0
        assert len(sem_results) == 1
        assert sem_results[0].id == b_unrelated_2.id


def test_decide_prompt_and_agent_loop_include_self_model_and_top_k_retrieved_memories(
    tmp_path: Path,
) -> None:
    """build_decide_prompt and AgentLoop include the current self-model and top-k retrieved memories."""
    settings = Settings()
    world = build_test_world(settings=settings)
    llm = MockLLM(seed=5)
    mgr = MemoryManager(llm, db_path=tmp_path / "loop_mem.db", world=world)
    try:
        mgr.self_model_store.update(
            "I investigate round spheres like object_3 and object_7 by pushing them.",
            tick=1,
        )
        mgr.belief_store.add(
            statement="Sphere object_3 rolls smoothly when pushed.",
            tags=["object_3", "push", "sphere"],
            confidence=0.85,
            tick=2,
        )
        mgr.belief_store.add(
            statement="Distant object_99 is completely unrelated.",
            tags=["object_99"],
            confidence=0.30,
            tick=1,
        )
        mgr.store.add(
            summary="Previously pushed object_7 toward object_1.",
            tags=["object_7", "object_1", "push"],
            salience=0.80,
            tick=2,
        )

        loop = AgentLoop(
            world=world,
            llm=llm,
            gate=HeartbeatGate(interval=10),
            memory=mgr,
            retrieval_k=2,
            settings=settings,
        )
        tick_res = loop.tick()
        assert tick_res.woke is True
        assert tick_res.prompt is not None
        user_prompt = tick_res.prompt.user

        # Self-model is included in the decide prompt
        assert "self_model:\nI investigate round spheres like object_3 and object_7 by pushing them." in user_prompt
        # Top-2 relevant memories (for perceived object_3 and object_7) are included, while unrelated object_99 is excluded
        assert "Sphere object_3 rolls smoothly when pushed." in user_prompt
        assert "Previously pushed object_7 toward object_1." in user_prompt
        assert "Distant object_99 is completely unrelated." not in user_prompt
    finally:
        mgr.store.close()
