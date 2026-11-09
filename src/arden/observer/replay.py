"""Deterministic replay of a recorded run from its saved seed, creator actions, and logged intents."""

from __future__ import annotations

import json
import math
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arden.agent.gate import GateConfig, HeartbeatGate, RuleGate
from arden.config import Settings, load_config
from arden.controller.controller import Controller
from arden.llm.mock import MockLLM, derive_llm_seed
from arden.memory.short_term import MemoryManager
from arden.observer.log import LogRecord, ObserverLog
from arden.perception.feed import PerceptionTracker, perceive
from arden.surprise.scorer import SurpriseScorer
from arden.world.world import World


@dataclass(frozen=True)
class Divergence:
    """First divergence detected between a recorded observer log and a replayed run."""

    tick: int
    index: int
    kind: str
    reason: str
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "index": self.index,
            "kind": self.kind,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
        }

    def __str__(self) -> str:
        return f"tick={self.tick} kind={self.kind} index={self.index}: {self.reason}"


@dataclass(frozen=True)
class ReplayResult:
    """Summary of replaying a run directory."""

    run_id: str
    run_dir: Path
    ok: bool
    diverged: bool
    first_divergence: Divergence | None
    ticks_replayed: int
    records_checked: int
    replayed_records: list[LogRecord]

    @property
    def divergence(self) -> Divergence | None:
        """Alias for ``first_divergence``."""
        return self.first_divergence

    @property
    def matched(self) -> bool:
        """True when no divergence was detected."""
        return self.ok and not self.diverged

    def __bool__(self) -> bool:
        return self.ok and not self.diverged

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir.as_posix(),
            "ok": self.ok,
            "diverged": self.diverged,
            "first_divergence": self.first_divergence.to_dict() if self.first_divergence else None,
            "ticks_replayed": self.ticks_replayed,
            "records_checked": self.records_checked,
        }


class ReplayMockLLM(MockLLM):
    """``MockLLM`` subclass that replays logged ``decide`` intents and ``reflect`` outputs in order."""

    def __init__(
        self,
        *,
        seed: int = 0,
        decide_intents: Sequence[Mapping[str, Any] | str] = (),
        reflect_outputs: Sequence[Mapping[str, Any] | str] = (),
    ) -> None:
        script_strs = [
            item if isinstance(item, str) else json.dumps(dict(item))
            for item in decide_intents
        ]
        super().__init__(seed=seed, script=script_strs)
        self._reflect_queue: list[str] = [
            item if isinstance(item, str) else json.dumps(dict(item))
            for item in reflect_outputs
        ]

    def complete_json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        if role == "reflect" and self._reflect_queue:
            raw = self._reflect_queue.pop(0)
            return json.loads(raw)
        return super().complete_json(role, system, user, schema, timeout_s)


def _approx_equal(a: Any, b: Any, tol: float = 1e-4) -> bool:
    if type(a) is bool or type(b) is bool:
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return False
        return all(_approx_equal(x, y, tol=tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_approx_equal(a[k], b[k], tol=tol) for k in a)
    return a == b


def _compare_record_pair(
    index: int,
    expected: LogRecord,
    actual: LogRecord,
) -> Divergence | None:
    """Compare one logged record against one replayed record and return a ``Divergence`` if they differ."""
    if expected.tick != actual.tick or expected.kind != actual.kind:
        return Divergence(
            tick=min(expected.tick, actual.tick),
            index=index,
            kind=expected.kind,
            reason=f"expected {expected.kind!r} at tick {expected.tick}, got {actual.kind!r} at tick {actual.tick}",
            expected=expected.to_dict(),
            actual=actual.to_dict(),
        )

    kind = expected.kind
    ep = expected.payload
    ap = actual.payload

    if kind == "perception":
        for field in ("light", "is_delta"):
            if field in ep and field in ap and not _approx_equal(ep[field], ap[field]):
                return Divergence(
                    tick=expected.tick,
                    index=index,
                    kind=kind,
                    reason=f"perception.{field} diverged: expected {ep[field]!r}, got {ap[field]!r}",
                    expected=ep,
                    actual=ap,
                )
        if "body" in ep and "body" in ap and not _approx_equal(ep["body"], ap["body"]):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=f"perception.body diverged: expected {ep['body']!r}, got {ap['body']!r}",
                expected=ep,
                actual=ap,
            )
        e_objs = ep.get("objects", ep.get("percept", {}).get("objects", []))
        a_objs = ap.get("objects", ap.get("percept", {}).get("objects", []))
        if not _approx_equal(e_objs, a_objs):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason="perception.objects diverged",
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "gate_decision":
        if bool(ep.get("wake")) != bool(ap.get("wake")) or list(ep.get("reasons", [])) != list(ap.get("reasons", [])):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=(
                    f"gate_decision diverged: expected wake={ep.get('wake')} reasons={ep.get('reasons')}, "
                    f"got wake={ap.get('wake')} reasons={ap.get('reasons')}"
                ),
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "intent":
        for field in ("goal", "steps", "wake_if", "expect"):
            if ep.get(field) != ap.get(field):
                return Divergence(
                    tick=expected.tick,
                    index=index,
                    kind=kind,
                    reason=f"intent.{field} diverged: expected {ep.get(field)!r}, got {ap.get(field)!r}",
                    expected=ep,
                    actual=ap,
                )
        return None

    if kind == "action_result":
        for field in ("verb", "target", "ok", "reason"):
            if ep.get(field) != ap.get(field):
                return Divergence(
                    tick=expected.tick,
                    index=index,
                    kind=kind,
                    reason=f"action_result.{field} diverged: expected {ep.get(field)!r}, got {ap.get(field)!r}",
                    expected=ep,
                    actual=ap,
                )
        if not _approx_equal(ep.get("data", {}), ap.get("data", {})):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=f"action_result.data diverged: expected {ep.get('data')!r}, got {ap.get('data')!r}",
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "weather_change":
        for field in ("source", "weather_type", "previous_weather_type", "duration"):
            if ep.get(field) != ap.get(field):
                return Divergence(
                    tick=expected.tick,
                    index=index,
                    kind=kind,
                    reason=f"weather_change.{field} diverged: expected {ep.get(field)!r}, got {ap.get(field)!r}",
                    expected=ep,
                    actual=ap,
                )
        if not _approx_equal(ep.get("intensity", 0.0), ap.get("intensity", 0.0)):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=f"weather_change.intensity diverged: expected {ep.get('intensity')!r}, got {ap.get('intensity')!r}",
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "belief_change":
        for field in ("action", "statement"):
            if ep.get(field) != ap.get(field):
                return Divergence(
                    tick=expected.tick,
                    index=index,
                    kind=kind,
                    reason=f"belief_change.{field} diverged: expected {ep.get(field)!r}, got {ap.get(field)!r}",
                    expected=ep,
                    actual=ap,
                )
        if not _approx_equal(ep.get("confidence", 0.0), ap.get("confidence", 0.0)):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=f"belief_change.confidence diverged: expected {ep.get('confidence')!r}, got {ap.get('confidence')!r}",
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "reflection":
        if ep.get("trigger") != ap.get("trigger") or ep.get("self_model") != ap.get("self_model"):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason="reflection trigger or self_model diverged",
                expected=ep,
                actual=ap,
            )
        return None

    if kind == "creator_action":
        if ep.get("action") != ap.get("action") or ep.get("object_id") != ap.get("object_id"):
            return Divergence(
                tick=expected.tick,
                index=index,
                kind=kind,
                reason=f"creator_action diverged: expected {ep.get('action')!r}, got {ap.get('action')!r}",
                expected=ep,
                actual=ap,
            )
        return None

    return None


def _apply_creator_action(world: World, payload: Mapping[str, Any]) -> None:
    """Re-apply a logged ``creator_action`` payload to ``world`` during replay."""
    action = str(payload.get("action", "")).strip().lower()
    if action == "spawn":
        obj_props = dict(payload.get("props") or payload.get("object") or {})
        pos = payload.get("position") or obj_props.get("position") or {"x": 0.0, "y": 0.0}
        world.spawn(obj_props, pos, source="creator")
        return
    if action == "move":
        oid = str(payload["object_id"])
        pos = payload["position"]
        world.move(oid, pos, source="creator")
        return
    if action == "despawn":
        oid = str(payload["object_id"])
        world.despawn(oid, source="creator")
        return
    if action == "update_object":
        oid = str(payload["object_id"])
        changes = dict(payload.get("changes") or {})
        world.update_object(oid, changes, source="creator")
        return
    if action in ("set_weather", "weather"):
        w_type = payload.get("weather_type", payload.get("type", "clear"))
        intensity = payload.get("intensity")
        duration = payload.get("duration")
        start_at = payload.get("start_at")
        world.set_weather(w_type, intensity=intensity, duration=duration, start_at=start_at)
        return
    if action == "schedule_weather":
        w_type = payload.get("weather_type", payload.get("type", "clear"))
        intensity = payload.get("intensity")
        duration = payload.get("duration")
        start_at = payload.get("start_at")
        world.set_weather(w_type, intensity=intensity, duration=duration, start_at=start_at)
        return
    if action == "set_weather_automation":
        enabled = bool(payload.get("enabled", not payload.get("paused", False)))
        world.set_weather_automation(enabled)
        return
    if action == "set_time_of_day":
        tod = float(payload["time_of_day"])
        world.set_time_of_day(tod, source="creator")
        return
    if action == "freeze_cycle":
        frozen = bool(payload.get("frozen", True))
        world.freeze_cycle(frozen, source="creator")
        return
    if action == "reply":
        text = str(payload.get("text") or "")
        mb_id = payload.get("mailbox_id")
        world.reply(text, mailbox_id=mb_id)
        return
    if action == "spawn_in_response":
        props = dict(payload.get("props") or payload.get("object") or {})
        mb_id = payload.get("mailbox_id")
        world.spawn_in_response(props, mailbox_id=mb_id)
        return
    if action == "return_object":
        obj_data = payload.get("object") or payload.get("props") or payload.get("object_id")
        mb_id = payload.get("mailbox_id")
        if obj_data is not None:
            world.return_object(obj_data, mailbox_id=mb_id)
        return


def _reconstruct_reflect_output(rec: LogRecord) -> dict[str, Any]:
    p = rec.payload
    episodes_out: list[dict[str, Any]] = []
    for ep in p.get("episodes", []) or []:
        if isinstance(ep, Mapping):
            episodes_out.append(
                {
                    "summary": str(ep.get("summary", "episode")),
                    "tags": list(ep.get("tags", [])),
                    "salience": float(ep.get("salience", 0.5)),
                    **(
                        {"surprise_kind": ep["surprise_kind"]}
                        if ep.get("surprise_kind")
                        else {}
                    ),
                    **(
                        {"made_by_agent": ep["made_by_agent"]}
                        if ep.get("made_by_agent")
                        else {}
                    ),
                }
            )
    beliefs_out: list[dict[str, Any]] = []
    for b in p.get("beliefs", []) or []:
        if isinstance(b, Mapping):
            action = "propose"
            hist = b.get("history")
            if isinstance(hist, list) and hist:
                last_h = hist[-1]
                if isinstance(last_h, Mapping) and last_h.get("outcome") in ("support", "contradict"):
                    action = str(last_h["outcome"])
            beliefs_out.append(
                {
                    "statement": str(b.get("statement", "")),
                    "tags": list(b.get("tags", [])),
                    "action": action,
                    "confidence": float(b.get("confidence", 0.5)),
                }
            )
    return {
        "episodes": episodes_out,
        "beliefs": beliefs_out,
        "self_model": p.get("self_model"),
    }


def _resolve_run_dir(run_dir: Path | str | ObserverLog) -> Path:
    if isinstance(run_dir, ObserverLog):
        return run_dir.run_dir
    p = Path(run_dir)
    if p.is_dir():
        return p
    alt = Path("data/runs") / str(run_dir)
    if alt.is_dir():
        return alt
    raise FileNotFoundError(f"Run directory not found: {run_dir}")


def replay_run(run_dir: Path | str | ObserverLog) -> ReplayResult:
    """Re-run a world from its saved seed and creator actions with MockLLM replaying logged intents.

    Returns a ``ReplayResult`` reporting ``diverged=False`` if the replay matches
    the ground-truth log, or ``diverged=True`` with ``first_divergence`` populated.
    """
    from arden.agent.loop import AgentLoop, build_test_world

    resolved_dir = _resolve_run_dir(run_dir)
    orig_log = ObserverLog.open(resolved_dir)
    logged_records = orig_log.read_all()

    meta_raw: dict[str, Any] = {}
    config_json = resolved_dir / "config.json"
    if config_json.is_file():
        meta_raw = json.loads(config_json.read_text(encoding="utf-8"))

    settings = orig_log.settings
    seed = int(meta_raw.get("seed", orig_log.seed))
    if (resolved_dir / "config.toml").is_file():
        try:
            toml_settings = load_config(resolved_dir / "config.toml")
            if toml_settings.world.seed != orig_log.seed and seed == orig_log.seed:
                seed = int(toml_settings.world.seed)
        except Exception:
            pass

    initial_world = meta_raw.get("initial_world")
    runner_meta = dict(meta_raw.get("runner_meta") or {})

    # Reconstruct initial World
    if isinstance(initial_world, Mapping):
        world = World(seed=seed)
        world.restore(initial_world)
        world.seed = seed
        orig_init_seed = meta_raw.get("config", {}).get("world", {}).get("seed", seed)
        if seed != orig_init_seed:
            world.rng = random.Random(seed)
            world.weather.rng = random.Random(seed)
    else:
        rng = random.Random(seed)
        world = build_test_world(rng=rng, settings=settings)

    decide_intents = [
        rec.payload.get("intent")
        or {
            "goal": rec.payload.get("goal", "goal"),
            "steps": rec.payload.get("steps", []),
            "wake_if": rec.payload.get("wake_if", []),
            "expect": rec.payload.get("expect", ""),
            "thought": rec.payload.get("thought", ""),
            "confidence": float(rec.payload.get("confidence", 0.5)),
        }
        for rec in logged_records
        if rec.kind == "intent"
    ]
    reflect_outputs = [
        _reconstruct_reflect_output(rec)
        for rec in logged_records
        if rec.kind == "reflection"
    ]

    mock_llm = ReplayMockLLM(
        seed=derive_llm_seed(seed),
        decide_intents=decide_intents,
        reflect_outputs=reflect_outputs,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        replay_log = ObserverLog(
            world,
            run_id=orig_log.run_id,
            run_dir=Path(tmp_str) / orig_log.run_id,
            settings=settings,
            seed=seed,
        )

        has_gate_decisions = any(r.kind == "gate_decision" for r in logged_records)
        if has_gate_decisions or runner_meta.get("runner") in ("agent_loop", "simulation_service"):
            gate_type = runner_meta.get("gate_type", "RuleGate")
            hb_interval = runner_meta.get("heartbeat_interval")
            if gate_type == "HeartbeatGate":
                interval = int(hb_interval if hb_interval is not None else 50)
                gate: Any = HeartbeatGate(interval=max(1, interval))
            else:
                gate_cfg = GateConfig.model_validate(
                    runner_meta.get("gate_config", settings.gate.model_dump())
                )
                gate = RuleGate(
                    config=gate_cfg,
                    heartbeat_interval=int(hb_interval) if hb_interval is not None else None,
                    world=world,
                )

            use_tracker = bool(runner_meta.get("use_perception_tracker", False)) or any(
                r.kind == "perception" and r.payload.get("is_delta") for r in logged_records
            )
            use_scorer = bool(runner_meta.get("use_surprise_scorer", False))
            has_mem = bool(runner_meta.get("has_memory", False)) or any(
                r.kind in ("reflection", "belief_change") for r in logged_records
            )

            controller = Controller(world=world)
            memory = MemoryManager(
                config=settings.memory,
                world=world,
                llm=mock_llm,
                db_path=":memory:",
                buffer_size=runner_meta.get("buffer_size"),
                rest_idle_ticks=runner_meta.get("rest_idle_ticks"),
            )
            loop = AgentLoop(
                world=world,
                llm=mock_llm,
                gate=gate,
                controller=controller,
                settings=settings,
                memory=memory if has_mem else None,
                perception_tracker=PerceptionTracker() if use_tracker else None,
                surprise_scorer=SurpriseScorer(config=settings.surprise, llm=mock_llm) if use_scorer else None,
                observer_log=replay_log,
            )

            # Group logged records relative to each tick's perception event
            # so creator_actions and manual reflections execute at the exact phase they occurred.
            idx = 0
            n = len(logged_records)
            while idx < n:
                rec = logged_records[idx]
                if rec.kind == "creator_action":
                    while world.tick < rec.tick:
                        loop.tick()
                    _apply_creator_action(world, rec.payload)
                    idx += 1
                    # If this creator_action was set_weather, the next logged record is the manual weather_change
                    if (
                        rec.payload.get("action") in ("set_weather", "weather")
                        and idx < n
                        and logged_records[idx].kind == "weather_change"
                        and logged_records[idx].payload.get("source") == "manual"
                    ):
                        idx += 1
                    continue

                if rec.kind == "perception":
                    target_tick = rec.tick
                    while world.tick < target_tick:
                        loop.tick()
                    # Advance idx past all records produced within this loop.tick()
                    # (perception, gate_decision, intent, action_result, and any auto reflection/belief_change at target_tick)
                    idx += 1
                    had_manual_reflection_at_tick = False
                    while idx < n and logged_records[idx].tick == target_tick and logged_records[idx].kind != "perception":
                        sub_rec = logged_records[idx]
                        if sub_rec.kind == "creator_action":
                            break
                        if sub_rec.kind == "reflection":
                            had_manual_reflection_at_tick = True
                        idx += 1

                    replayed_reflections_before = sum(1 for r in replay_log.read_all() if r.kind == "reflection" and r.tick == target_tick)
                    loop.tick()
                    replayed_reflections_after = sum(1 for r in replay_log.read_all() if r.kind == "reflection" and r.tick == target_tick)
                    if had_manual_reflection_at_tick and replayed_reflections_after == replayed_reflections_before:
                        memory.reflect(tick=target_tick)
                    continue

                if rec.kind == "reflection":
                    while world.tick < rec.tick:
                        loop.tick()
                    replayed_now = replay_log.read_all()
                    logged_refl_so_far = sum(1 for r in logged_records[: idx + 1] if r.kind == "reflection")
                    replayed_refl_so_far = sum(1 for r in replayed_now if r.kind == "reflection")
                    if replayed_refl_so_far < logged_refl_so_far:
                        memory.reflect(tick=rec.tick, trigger=str(rec.payload.get("trigger", "rest")))
                    idx += 1
                    continue

                if rec.kind == "belief_change":
                    # Check if this belief_change is part of an upcoming reflection at the same tick
                    upcoming_refl = any(
                        logged_records[j].kind == "reflection" and logged_records[j].tick == rec.tick
                        for j in range(idx + 1, min(n, idx + 8))
                    )
                    if not upcoming_refl:
                        replayed_bc = sum(1 for r in replay_log.read_all() if r.kind == "belief_change")
                        logged_bc = sum(1 for r in logged_records[:idx] if r.kind == "belief_change")
                        if replayed_bc <= logged_bc:
                            act = str(rec.payload.get("action", "created"))
                            stmt = str(rec.payload.get("statement", ""))
                            tags = list(rec.payload.get("tags", []))
                            if act == "created":
                                memory.belief_store.add(
                                    statement=stmt,
                                    tags=tags,
                                    confidence=float(rec.payload.get("confidence", 0.5)),
                                    tick=rec.tick,
                                )
                            elif act == "support":
                                memory.belief_store.support(stmt, tick=rec.tick, tags=tags)
                            elif act == "contradict":
                                memory.belief_store.contradict(stmt, tick=rec.tick, tags=tags)
                    idx += 1
                    continue

                if rec.kind == "weather_change" and rec.payload.get("source") == "automated":
                    while world.tick < rec.tick:
                        loop.tick()
                    idx += 1
                    continue

                idx += 1
        else:
            # Direct World replay (when run without AgentLoop)
            tracker = PerceptionTracker(world, config=settings)
            controller = Controller(world=world)
            memory = MemoryManager(
                config=settings.memory,
                world=world,
                llm=mock_llm,
                db_path=":memory:",
            )
            idx = 0
            n = len(logged_records)
            while idx < n:
                rec = logged_records[idx]
                while world.tick < rec.tick:
                    world.step()
                if rec.kind == "creator_action":
                    _apply_creator_action(world, rec.payload)
                    idx += 1
                    if (
                        rec.payload.get("action") in ("set_weather", "weather")
                        and idx < n
                        and logged_records[idx].kind == "weather_change"
                        and logged_records[idx].payload.get("source") == "manual"
                    ):
                        idx += 1
                elif rec.kind == "perception":
                    if rec.payload.get("is_delta"):
                        tracker.perceive(world)
                    else:
                        perceive(world)
                    idx += 1
                elif rec.kind == "intent":
                    intent_data = rec.payload.get("intent") or rec.payload
                    controller.set_intent(intent_data, tick=world.tick)
                    world.events.emit(
                        _build_intent_event_from_payload(world.tick, rec.payload)
                    )
                    idx += 1
                elif rec.kind == "action_result":
                    from arden.actions.verbs import execute_action

                    verb = str(rec.payload.get("verb", ""))
                    target = rec.payload.get("target")
                    params = dict(rec.payload.get("params") or {})
                    execute_action(world, verb, target=target, params=params)
                    idx += 1
                elif rec.kind == "reflection":
                    memory.reflect(tick=rec.tick, trigger=str(rec.payload.get("trigger", "rest")))
                    idx += 1
                elif rec.kind == "belief_change":
                    upcoming_refl = any(
                        logged_records[j].kind == "reflection" and logged_records[j].tick == rec.tick
                        for j in range(idx + 1, min(n, idx + 8))
                    )
                    if not upcoming_refl:
                        act = str(rec.payload.get("action", "created"))
                        stmt = str(rec.payload.get("statement", ""))
                        tags = list(rec.payload.get("tags", []))
                        if act == "created":
                            memory.belief_store.add(
                                statement=stmt,
                                tags=tags,
                                confidence=float(rec.payload.get("confidence", 0.5)),
                                tick=rec.tick,
                            )
                        elif act == "support":
                            memory.belief_store.support(stmt, tick=rec.tick, tags=tags)
                        elif act == "contradict":
                            memory.belief_store.contradict(stmt, tick=rec.tick, tags=tags)
                    idx += 1
                else:
                    idx += 1

        replayed_records = replay_log.read_all()
        replay_log.close()

    # Compare logged_records vs replayed_records and find the first divergence
    first_div: Divergence | None = None
    pair_count = min(len(logged_records), len(replayed_records))
    for i in range(pair_count):
        div = _compare_record_pair(i, logged_records[i], replayed_records[i])
        if div is not None:
            first_div = div
            break

    if first_div is None and len(logged_records) != len(replayed_records):
        if len(logged_records) > len(replayed_records):
            missing = logged_records[pair_count]
            first_div = Divergence(
                tick=missing.tick,
                index=pair_count,
                kind=missing.kind,
                reason=f"logged record {missing.kind!r} at tick {missing.tick} was not produced during replay",
                expected=missing.to_dict(),
                actual=None,
            )
        else:
            extra = replayed_records[pair_count]
            first_div = Divergence(
                tick=extra.tick,
                index=pair_count,
                kind=extra.kind,
                reason=f"replay produced unexpected extra record {extra.kind!r} at tick {extra.tick}",
                expected=None,
                actual=extra.to_dict(),
            )

    return ReplayResult(
        run_id=orig_log.run_id,
        run_dir=resolved_dir,
        ok=(first_div is None),
        diverged=(first_div is not None),
        first_divergence=first_div,
        ticks_replayed=world.tick,
        records_checked=pair_count,
        replayed_records=replayed_records,
    )


def _build_intent_event_from_payload(tick: int, payload: Mapping[str, Any]) -> Any:
    from arden.world.world import IntentSetEvent

    return IntentSetEvent(
        tick=tick,
        source="agent",
        intent=payload.get("intent"),
        goal=payload.get("goal"),
        thought=payload.get("thought"),
        expect=payload.get("expect"),
        confidence=payload.get("confidence"),
        steps=list(payload.get("steps", [])),
        wake_if=list(payload.get("wake_if", [])),
        reasons=list(payload.get("reasons", [])),
    )


replay = replay_run

__all__ = [
    "Divergence",
    "ReplayMockLLM",
    "ReplayResult",
    "replay",
    "replay_run",
]
