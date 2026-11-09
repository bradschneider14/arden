"""Compare agent beliefs against observer log ground-truth events (memory vs reality)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from arden.memory.beliefs import Belief, BeliefStore
from arden.observer.log import (
    LogRecord,
    ObserverLog,
    extract_object_ids,
    payload_mentions_object_id,
)

_NEGATIVE_MARKERS = (
    "cannot",
    "can't",
    "not ",
    "never",
    "impossible",
    "immovable",
    "unmovable",
    "too heavy",
    "too_heavy",
    "fixed",
    "stuck",
    "fails",
    "failed",
    "unreachable",
    "out of reach",
    "out_of_reach",
    "nothing happened",
    "nothing_happened",
    "no effect",
    "closed",
)

_POSITIVE_ACTION_MARKERS = (
    "can ",
    "movable",
    "grabbable",
    "light",
    "moves",
    "moved",
    "pushed",
    "grabbed",
    "lifted",
    "opens",
    "opened",
    "open",
    "succeeds",
    "easy",
)

_COLORS = ("red", "blue", "green", "yellow", "white", "black", "gray", "grey", "brown", "orange", "purple")
_WEATHER_WORDS = ("clear", "rain", "wind", "cold", "storm")
_VERB_WORDS = ("move", "grab", "drop", "push", "strike", "place", "stack", "bind", "insert", "separate", "look", "speak")


def _statement_is_negative(statement: str) -> bool:
    low = statement.lower()
    return any(marker in low for marker in _NEGATIVE_MARKERS)


def _statement_is_positive_action(statement: str) -> bool:
    low = statement.lower()
    if _statement_is_negative(low):
        return False
    return any(marker in low for marker in _POSITIVE_ACTION_MARKERS)


def _extract_claimed_colors(statement: str, tags: Sequence[str]) -> set[str]:
    text = f"{statement} {' '.join(tags)}".lower()
    found: set[str] = set()
    for c in _COLORS:
        if re.search(rf"\b{re.escape(c)}\b", text):
            found.add("gray" if c == "grey" else c)
    return found


def _extract_claimed_verbs(statement: str, tags: Sequence[str]) -> set[str]:
    text = f"{statement} {' '.join(tags)}".lower()
    found: set[str] = set()
    for v in _VERB_WORDS:
        if re.search(rf"\b{re.escape(v)}(?:s|ed|ing)?\b", text):
            found.add(v)
    return found


def _extract_claimed_weather(statement: str, tags: Sequence[str]) -> set[str]:
    text = f"{statement} {' '.join(tags)}".lower()
    found: set[str] = set()
    for w in _WEATHER_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text):
            found.add(w)
    return found


def _extract_claimed_coords(statement: str) -> tuple[float, float] | None:
    m = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", statement)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m_xy = re.search(r"x\s*=\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*y\s*=\s*(-?\d+(?:\.\d+)?)", statement, re.IGNORECASE)
    if m_xy:
        return (float(m_xy.group(1)), float(m_xy.group(2)))
    return None


def _norm_statement(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _classify_event_for_belief(
    belief: Belief,
    rec: LogRecord,
    *,
    obj_ids: set[str],
    claimed_verbs: set[str],
    claimed_colors: set[str],
    claimed_weather: set[str],
    claimed_coords: tuple[float, float] | None,
    support_ticks: set[int],
    contradict_ticks: set[int],
) -> Literal["formed", "contradicted", "none"]:
    """Classify whether a ground-truth ``LogRecord`` formed or contradicted ``belief``."""
    kind = rec.kind
    payload = rec.payload
    stmt_norm = _norm_statement(belief.statement)
    is_neg = _statement_is_negative(belief.statement)
    is_pos = _statement_is_positive_action(belief.statement)

    # 1. Direct belief_change record for this belief
    if kind == "belief_change":
        rec_bid = payload.get("belief_id")
        rec_stmt = _norm_statement(str(payload.get("statement", "")))
        same_belief = (
            (belief.id is not None and rec_bid is not None and int(rec_bid) == int(belief.id))
            or (rec_stmt and rec_stmt == stmt_norm)
        )
        if same_belief:
            act = str(payload.get("action", payload.get("outcome", "created"))).lower()
            if act in ("contradict", "contradicted", "evicted"):
                return "contradicted"
            return "formed"
        return "none"

    # 2. Reflection record mentioning this belief
    if kind == "reflection":
        for b_item in payload.get("beliefs", []) or []:
            if isinstance(b_item, Mapping):
                b_stmt = _norm_statement(str(b_item.get("statement", "")))
                b_id = b_item.get("id", b_item.get("belief_id"))
                if (belief.id is not None and b_id == belief.id) or (b_stmt and b_stmt == stmt_norm):
                    act = str(b_item.get("action", "propose")).lower()
                    if act in ("contradict", "contradicted"):
                        return "contradicted"
                    if rec.tick in contradict_ticks and rec.tick not in support_ticks:
                        return "contradicted"
                    return "formed"
        if obj_ids and any(payload_mentions_object_id(payload, oid) for oid in obj_ids):
            if rec.tick in contradict_ticks:
                return "contradicted"
            if rec.tick in support_ticks:
                return "formed"
        return "none"

    # Check if this record involves any of the belief's objects (or weather if weather belief)
    mentions_obj = bool(obj_ids) and any(payload_mentions_object_id(payload, oid) for oid in obj_ids)

    # 3. Creator action on an object the belief is about
    if kind == "creator_action" and mentions_obj:
        c_action = str(payload.get("action", "")).lower()
        if c_action == "despawn":
            # Object was removed from reality -> contradicts active beliefs about its presence/state
            return "contradicted"
        if c_action == "move":
            # If belief claims object is fixed/immovable/too heavy or at specific coordinates, creator moving it contradicts
            if is_neg or claimed_coords is not None or "at " in belief.statement.lower() or "position" in belief.statement.lower() or "near" in belief.statement.lower():
                return "contradicted"
            if rec.tick > belief.created_tick:
                return "contradicted"
            return "formed"
        if c_action == "update_object":
            changes = payload.get("changes") if isinstance(payload.get("changes"), Mapping) else {}
            new_color = changes.get("color")
            if isinstance(new_color, str) and claimed_colors:
                norm_c = "gray" if new_color.lower() == "grey" else new_color.lower()
                if norm_c not in claimed_colors:
                    return "contradicted"
                return "formed"
            if rec.tick > belief.created_tick:
                return "contradicted"
            return "formed"
        if c_action == "spawn":
            if rec.tick <= belief.updated_tick:
                return "formed"

    # 4. Action result on an object the belief is about
    if kind == "action_result" and mentions_obj:
        ok = bool(payload.get("ok", True))
        reason = str(payload.get("reason") or "").lower()
        verb = str(payload.get("verb") or "").lower()
        verb_matches = (not claimed_verbs) or (verb in claimed_verbs) or (verb == "move" and "push" in claimed_verbs)

        if verb_matches:
            # Negative belief (e.g., "object_3 is too heavy to grab" or "cannot move object_5")
            if is_neg:
                if not ok:
                    return "formed"
                # If belief specifically says too_heavy / cannot grab / immovable, and the action succeeded:
                if verb in claimed_verbs or (not claimed_verbs and verb in ("grab", "push", "move", "separate")):
                    return "contradicted"
            # Positive belief (e.g., "object_7 can be pushed" or "grabbing object_15 opens lid")
            if is_pos:
                if ok:
                    return "formed"
                return "contradicted"

        if rec.tick in contradict_ticks:
            return "contradicted"
        if rec.tick <= belief.updated_tick:
            return "formed" if ok != is_neg else "contradicted"
        return "formed" if ok else "contradicted"

    # 5. Perception involving the belief's object(s)
    if kind == "perception" and mentions_obj:
        objects_list = payload.get("objects", [])
        if not isinstance(objects_list, list) and isinstance(payload.get("percept"), Mapping):
            objects_list = payload["percept"].get("objects", [])
        for o in objects_list if isinstance(objects_list, list) else []:
            if not isinstance(o, Mapping) or o.get("id") not in obj_ids:
                continue
            # Check color mismatch
            o_color = o.get("color")
            if isinstance(o_color, str) and claimed_colors:
                norm_c = "gray" if o_color.lower() == "grey" else o_color.lower()
                if norm_c not in claimed_colors:
                    return "contradicted"
            # Check coordinate mismatch
            if claimed_coords is not None and isinstance(o.get("position"), Mapping):
                px = float(o["position"].get("x", 0.0))
                py = float(o["position"].get("y", 0.0))
                if abs(px - claimed_coords[0]) > 1.0 or abs(py - claimed_coords[1]) > 1.0:
                    return "contradicted"
            # Check open/closed lid mismatch
            state = o.get("state") if isinstance(o.get("state"), Mapping) else {}
            lid = state.get("lid")
            if isinstance(lid, str):
                low_stmt = belief.statement.lower()
                if "always closed" in low_stmt and lid == "open":
                    return "contradicted"
                if "always open" in low_stmt and lid == "closed":
                    return "contradicted"
        if rec.tick in contradict_ticks:
            return "contradicted"
        return "formed"

    # 6. Weather change vs weather belief
    if kind == "weather_change" and (claimed_weather or "weather" in belief.tags or "weather" in belief.statement.lower()):
        w_type = str(payload.get("weather_type", "")).lower()
        if claimed_weather:
            if w_type in claimed_weather:
                return "formed" if not is_neg else "contradicted"
            return "contradicted" if not is_neg else "formed"
        return "formed"

    # 7. Intent involving the belief's object(s)
    if kind == "intent" and mentions_obj:
        if rec.tick in contradict_ticks:
            return "contradicted"
        if rec.tick <= belief.updated_tick:
            return "formed"

    return "none"


class BeliefComparison(dict[str, Any]):
    """Single belief paired with the ground-truth events that formed or contradicted it."""

    def __init__(
        self,
        *,
        belief: Belief,
        formed_by: list[LogRecord],
        contradicted_by: list[LogRecord],
    ) -> None:
        all_events = [*formed_by, *contradicted_by]
        all_events.sort(key=lambda r: r.tick)
        has_contradiction = len(contradicted_by) > 0
        reality_status = (
            "contradicted"
            if has_contradiction
            else ("accurate" if len(formed_by) > 0 else "unverified")
        )
        if has_contradiction:
            gap = (
                f"Belief {belief.statement!r} (confidence={belief.confidence:.2f}) has "
                f"{len(contradicted_by)} contradicting ground-truth event(s) vs "
                f"{len(formed_by)} supporting event(s)."
            )
        else:
            gap = (
                f"Belief {belief.statement!r} (confidence={belief.confidence:.2f}) aligns with "
                f"{len(formed_by)} supporting ground-truth event(s) and 0 contradictions."
            )

        belief_dict = belief.to_dict()
        super().__init__(
            belief_id=belief.id,
            statement=belief.statement,
            confidence=belief.confidence,
            tags=list(belief.tags),
            evidence_count=belief.evidence_count,
            created_tick=belief.created_tick,
            updated_tick=belief.updated_tick,
            belief=belief_dict,
            formed_by=formed_by,
            forming_events=formed_by,
            supporting_events=formed_by,
            formed=formed_by,
            contradicted_by=contradicted_by,
            contradicting_events=contradicted_by,
            contradicted=contradicted_by,
            events=all_events,
            has_contradiction=has_contradiction,
            contradicted_in_reality=has_contradiction,
            reality_status=reality_status,
            reality_gap=gap,
            gap=gap,
        )
        self._belief_obj = belief

    @property
    def belief_id(self) -> int | None:
        return super().__getitem__("belief_id")

    @property
    def statement(self) -> str:
        return str(super().__getitem__("statement"))

    @property
    def confidence(self) -> float:
        return float(super().__getitem__("confidence"))

    @property
    def tags(self) -> list[str]:
        return list(super().__getitem__("tags"))

    @property
    def belief(self) -> dict[str, Any]:
        return super().__getitem__("belief")

    @property
    def formed_by(self) -> list[LogRecord]:
        return super().__getitem__("formed_by")

    @property
    def forming_events(self) -> list[LogRecord]:
        return super().__getitem__("formed_by")

    @property
    def supporting_events(self) -> list[LogRecord]:
        return super().__getitem__("formed_by")

    @property
    def formed(self) -> list[LogRecord]:
        return super().__getitem__("formed_by")

    @property
    def contradicted_by(self) -> list[LogRecord]:
        return super().__getitem__("contradicted_by")

    @property
    def contradicting_events(self) -> list[LogRecord]:
        return super().__getitem__("contradicted_by")

    @property
    def contradicted(self) -> list[LogRecord]:
        return super().__getitem__("contradicted_by")

    @property
    def events(self) -> list[LogRecord]:
        return super().__getitem__("events")

    @property
    def has_contradiction(self) -> bool:
        return bool(super().__getitem__("has_contradiction"))

    @property
    def reality_status(self) -> str:
        return str(super().__getitem__("reality_status"))

    @property
    def reality_gap(self) -> str:
        return str(super().__getitem__("gap"))

    @property
    def gap(self) -> str:
        return str(super().__getitem__("gap"))


class ComparisonResult(list[BeliefComparison]):
    """List of ``BeliefComparison`` entries that also supports dict/attribute access."""

    def __init__(self, comparisons: Iterable[BeliefComparison]) -> None:
        super().__init__(comparisons)

    @property
    def beliefs(self) -> list[BeliefComparison]:
        return list(self)

    @property
    def comparisons(self) -> list[BeliefComparison]:
        return list(self)

    @property
    def items(self) -> list[BeliefComparison]:
        return list(self)

    @property
    def count(self) -> int:
        return len(self)

    @property
    def contradictions(self) -> list[BeliefComparison]:
        return [c for c in self if c.has_contradiction]

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            if key in ("beliefs", "comparisons", "items", "results"):
                return list(self)
            if key == "count":
                return len(self)
            if key == "contradictions":
                return self.contradictions
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str) and item in ("beliefs", "comparisons", "items", "results", "count", "contradictions"):
            return True
        return super().__contains__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        items_list = [dict(c) for c in self]
        return {
            "beliefs": items_list,
            "comparisons": items_list,
            "items": items_list,
            "count": len(items_list),
        }


def _extract_records(source: Any) -> list[LogRecord]:
    if source is None:
        return []
    if isinstance(source, ObserverLog):
        return source.read_all()
    if isinstance(source, Path | str):
        p = Path(source)
        if not p.exists():
            alt = Path("data/runs") / str(source)
            if alt.exists():
                p = alt
        if p.is_dir():
            return ObserverLog.open(p).read_all()
        if p.is_file():
            records: list[LogRecord] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    import json

                    records.append(LogRecord(json.loads(line)))
            return records
        return []
    if hasattr(source, "observer_log") and getattr(source, "observer_log") is not None:
        return _extract_records(getattr(source, "observer_log"))
    if hasattr(source, "read_all") and callable(source.read_all):
        return list(source.read_all())
    if isinstance(source, Iterable) and not isinstance(source, Mapping | bytes | bytearray):
        out: list[LogRecord] = []
        for item in source:
            if isinstance(item, LogRecord):
                out.append(item)
            elif isinstance(item, Mapping) and "kind" in item and "payload" in item:
                out.append(LogRecord(item))
        return out
    return []


def _reconstruct_beliefs_from_records(records: Sequence[LogRecord]) -> list[Belief]:
    """Reconstruct the active set of ``Belief`` objects from logged ``belief_change`` and ``reflection`` records."""
    by_key: dict[str, Belief] = {}
    next_id = 1
    for rec in records:
        if rec.kind == "belief_change":
            p = rec.payload
            act = str(p.get("action", p.get("outcome", "created"))).lower()
            stmt = str(p.get("statement", "")).strip()
            if not stmt and isinstance(p.get("belief"), Mapping):
                stmt = str(p["belief"].get("statement", "")).strip()
            if not stmt:
                continue
            key = _norm_statement(stmt)
            if act == "evicted":
                by_key.pop(key, None)
                continue
            if isinstance(p.get("belief"), Mapping):
                try:
                    by_key[key] = Belief.model_validate(p["belief"])
                    continue
                except Exception:
                    pass
            existing = by_key.get(key)
            conf = float(p.get("confidence", 0.5))
            tags = list(p.get("tags", []))
            if existing is None:
                bid = p.get("belief_id")
                by_key[key] = Belief(
                    id=int(bid) if isinstance(bid, int) else next_id,
                    statement=stmt,
                    tags=tags,
                    confidence=conf,
                    evidence_count=int(p.get("evidence_count", 1)),
                    created_tick=rec.tick,
                    updated_tick=rec.tick,
                )
                next_id += 1
            else:
                if act in ("contradict", "contradicted"):
                    existing.contradict(tick=rec.tick, tags=tags)
                else:
                    existing.support(tick=rec.tick, tags=tags)
                existing.confidence = conf
        elif rec.kind == "reflection":
            for b_raw in rec.payload.get("beliefs", []) or []:
                if isinstance(b_raw, Mapping) and b_raw.get("statement"):
                    key = _norm_statement(str(b_raw["statement"]))
                    if key not in by_key:
                        try:
                            by_key[key] = Belief.model_validate(b_raw)
                        except Exception:
                            pass
    return list(by_key.values())


def _extract_beliefs(source: Any, records: Sequence[LogRecord]) -> list[Belief]:
    if source is None:
        return _reconstruct_beliefs_from_records(records)
    if isinstance(source, BeliefStore):
        return source.all()
    if isinstance(source, Belief):
        return [source]
    if hasattr(source, "belief_store") and isinstance(getattr(source, "belief_store"), BeliefStore):
        return getattr(source, "belief_store").all()
    if hasattr(source, "memory") and getattr(source, "memory") is not None:
        mem = getattr(source, "memory")
        if hasattr(mem, "belief_store") and isinstance(mem.belief_store, BeliefStore):
            return mem.belief_store.all()
    if hasattr(source, "beliefs"):
        b_attr = getattr(source, "beliefs")
        b_val = b_attr() if callable(b_attr) else b_attr
        if isinstance(b_val, Iterable):
            return _extract_beliefs(list(b_val), records)
    if isinstance(source, Iterable) and not isinstance(source, Mapping | str | bytes | bytearray | ObserverLog):
        out: list[Belief] = []
        for item in source:
            if isinstance(item, Belief):
                out.append(item)
            elif isinstance(item, Mapping) and "statement" in item:
                out.append(Belief.model_validate(dict(item)))
            elif isinstance(item, str) and item.strip():
                out.append(Belief(statement=item.strip()))
        if out:
            return out
    return _reconstruct_beliefs_from_records(records)


def _looks_like_belief_source(arg: Any) -> bool:
    if isinstance(arg, BeliefStore | Belief):
        return True
    if isinstance(arg, Sequence) and not isinstance(arg, str | bytes | bytearray):
        if arg and isinstance(arg[0], Belief):
            return True
        if arg and isinstance(arg[0], Mapping) and "statement" in arg[0] and "kind" not in arg[0]:
            return True
    return False


def compare(
    first: Any = None,
    second: Any = None,
    *,
    log: Any = None,
    beliefs: Any = None,
    world: Any = None,
) -> ComparisonResult:
    """List beliefs next to the ground-truth events that formed or contradicted them.

    Supports flexible argument ordering:
    - ``compare(observer_log, belief_store)``
    - ``compare(belief_store, observer_log)``
    - ``compare(observer_log)`` or ``compare(run_dir)``
    - ``compare(agent_loop)`` or ``compare(simulation_service)``
    """
    log_src = log
    belief_src = beliefs

    if log_src is None and belief_src is None:
        if _looks_like_belief_source(first):
            belief_src = first
            log_src = second
        else:
            log_src = first
            belief_src = second
    elif log_src is None:
        log_src = first if first is not None else second
    elif belief_src is None:
        belief_src = first if first is not None else second

    if log_src is None and world is not None:
        log_src = getattr(world, "observer_log", None)

    records = _extract_records(log_src)
    if not records and belief_src is not None:
        records = _extract_records(belief_src)
        if not records and hasattr(belief_src, "world") and getattr(belief_src, "world") is not None:
            records = _extract_records(getattr(belief_src.world, "observer_log", None))

    belief_list = _extract_beliefs(belief_src if belief_src is not None else log_src, records)

    comparisons: list[BeliefComparison] = []
    for b in belief_list:
        obj_ids = set(extract_object_ids([b.statement, b.tags]))
        claimed_verbs = _extract_claimed_verbs(b.statement, b.tags)
        claimed_colors = _extract_claimed_colors(b.statement, b.tags)
        claimed_weather = _extract_claimed_weather(b.statement, b.tags)
        claimed_coords = _extract_claimed_coords(b.statement)

        support_ticks = {b.created_tick}
        contradict_ticks: set[int] = set()
        for h in b.history:
            if h.outcome in ("created", "support"):
                support_ticks.add(h.tick)
            elif h.outcome == "contradict":
                contradict_ticks.add(h.tick)

        formed_by: list[LogRecord] = []
        contradicted_by: list[LogRecord] = []

        for rec in records:
            cls = _classify_event_for_belief(
                b,
                rec,
                obj_ids=obj_ids,
                claimed_verbs=claimed_verbs,
                claimed_colors=claimed_colors,
                claimed_weather=claimed_weather,
                claimed_coords=claimed_coords,
                support_ticks=support_ticks,
                contradict_ticks=contradict_ticks,
            )
            if cls == "formed":
                formed_by.append(rec)
            elif cls == "contradicted":
                contradicted_by.append(rec)

        comparisons.append(
            BeliefComparison(
                belief=b,
                formed_by=formed_by,
                contradicted_by=contradicted_by,
            )
        )

    return ComparisonResult(comparisons)


compare_beliefs = compare

__all__ = [
    "BeliefComparison",
    "ComparisonResult",
    "compare",
    "compare_beliefs",
]
