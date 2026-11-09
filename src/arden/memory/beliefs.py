"""Semantic beliefs, evolving self-model, and relevance retrieval.

Rules from docs/DESIGN.md (Sections 10.4, 11, 13, 15) and Ticket T14:
- Beliefs hold ``statement`` text, ``tags``, ``confidence`` in ``0..1``, ``evidence_count``,
  ``created_tick``, ``updated_tick``, and a ``history`` of confidence changes (for the belief browser).
- Reflection can propose new beliefs and support or contradict existing ones; confidence rises with
  supporting evidence and falls on contradiction.
- A hard cap on beliefs (config ``max_beliefs``) evicts the lowest ``confidence * recency`` belief,
  logging each eviction as a ``BeliefEvictedEvent``.
- Self-model: a short text (max ~80 words) rewritten during reflection, with every version stored.
- Retrieval: ``retrieve(context_tags, k)`` scores episodes and beliefs by recency, importance
  (salience / confidence), and relevance (tag overlap), with an optional embedding hook defined
  and unused by default.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from arden.config import (
    DEFAULT_BELIEF_CONTRADICT_DELTA,
    DEFAULT_BELIEF_SUPPORT_DELTA,
    DEFAULT_INITIAL_BELIEF_CONFIDENCE,
    DEFAULT_MAX_BELIEFS,
    DEFAULT_MAX_SELF_MODEL_WORDS,
    DEFAULT_RECENCY_DECAY_RATE,
    DEFAULT_RETRIEVAL_K,
    MemoryConfig,
    Settings,
)
from arden.world.world import BeliefEvictedEvent, EventBus, World

if TYPE_CHECKING:
    from arden.memory.store import Episode, EpisodeStore

_OBJECT_ID_RE = re.compile(r"\bobject_[0-9]+\b")

BeliefAction = Literal["propose", "support", "contradict"]
ConfidenceOutcome = Literal["created", "support", "contradict", "updated"]
EmbeddingHook = Callable[[str], Sequence[float]]
DEFAULT_EMBEDDING_HOOK: EmbeddingHook | None = None

DEFAULT_RECENCY_WEIGHT: float = 1.0
DEFAULT_IMPORTANCE_WEIGHT: float = 1.0
DEFAULT_RELEVANCE_WEIGHT: float = 2.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_tags(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, Sequence | set | frozenset):
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            if item is None or isinstance(item, bool):
                continue
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result
    raise ValueError(f"expected a list of tag strings, got {type(value).__name__}")


def _extract_object_ids_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _OBJECT_ID_RE.findall(text):
        if match not in seen:
            seen.add(match)
            out.append(match)
    return out


def _normalize_statement_key(statement: str) -> str:
    return " ".join(statement.strip().lower().split())


def normalize_belief_action(value: Any) -> BeliefAction:
    """Normalize a belief action string into ``'propose'``, ``'support'``, or ``'contradict'``."""
    if value is None:
        return "propose"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("propose", "new", "create", "add", "created", "proposed"):
            return "propose"
        if cleaned in ("support", "supports", "supporting", "confirm", "confirmed", "reinforce", "strengthen"):
            return "support"
        if cleaned in ("contradict", "contradicts", "contradicting", "refute", "refuted", "weaken", "disprove"):
            return "contradict"
        raise ValueError(f"belief action must be propose, support, or contradict, got {value!r}")
    raise ValueError(f"belief action must be a string, got {type(value).__name__}")


def compute_recency(
    current_tick: int,
    reference_tick: int,
    decay_rate: float = DEFAULT_RECENCY_DECAY_RATE,
) -> float:
    """Compute a recency score in ``(0.0, 1.0]`` from elapsed ticks since ``reference_tick``."""
    elapsed = max(0, int(current_tick) - int(reference_tick))
    rate = max(0.0, float(decay_rate))
    return round(1.0 / (1.0 + rate * elapsed), 6)


class ConfidenceChange(BaseModel):
    """A single entry in a belief's history of confidence changes."""

    model_config = ConfigDict(extra="forbid")

    tick: int = Field(default=0, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    previous_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    delta: float = 0.0
    outcome: ConfidenceOutcome = "created"
    evidence_count: int = Field(default=1, ge=0)
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        if "new_confidence" in raw and "confidence" not in raw:
            raw["confidence"] = raw.pop("new_confidence")
        if "old_confidence" in raw and "previous_confidence" not in raw:
            raw["previous_confidence"] = raw.pop("old_confidence")
        for outcome_alias in ("reason", "action", "kind", "event"):
            if outcome_alias in raw and "outcome" not in raw:
                val = str(raw.pop(outcome_alias)).strip().lower()
                if val in ("propose", "new", "create", "created", "initial"):
                    raw["outcome"] = "created"
                elif val in ("support", "supports", "supported", "confirm"):
                    raw["outcome"] = "support"
                elif val in ("contradict", "contradicts", "contradicted", "refute"):
                    raw["outcome"] = "contradict"
                else:
                    raw["outcome"] = "updated"
        conf = float(raw.get("confidence", 0.5))
        prev = raw.get("previous_confidence")
        if "delta" not in raw or raw["delta"] is None:
            raw["delta"] = round(conf - float(prev), 6) if prev is not None else 0.0
        return raw

    @property
    def old_confidence(self) -> float | None:
        """Alias for ``previous_confidence``."""
        return self.previous_confidence

    @property
    def new_confidence(self) -> float:
        """Alias for ``confidence``."""
        return self.confidence

    @property
    def reason(self) -> str:
        """Alias for ``outcome``."""
        return self.outcome

    @property
    def action(self) -> str:
        """Alias for ``outcome``."""
        return self.outcome

    @property
    def kind(self) -> str:
        """Alias for ``outcome``."""
        return self.outcome

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


BeliefHistoryEntry = ConfidenceChange
ConfidenceHistoryEntry = ConfidenceChange


class ReflectionBelief(BaseModel):
    """Schema for a belief proposal or update returned by the LLM ``reflect`` role."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    tags: list[str] = Field(default_factory=list)
    action: BeliefAction = "propose"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    belief_id: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, values: Any) -> Any:
        if isinstance(values, str):
            return {"statement": values}
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        for stmt_alias in ("text", "summary", "belief"):
            if stmt_alias in raw and "statement" not in raw:
                raw["statement"] = raw.pop(stmt_alias)
        if "id" in raw and "belief_id" not in raw:
            raw["belief_id"] = raw.pop("id")
        for act_alias in ("outcome", "effect", "relation", "stance", "verdict", "type"):
            if act_alias in raw and "action" not in raw:
                raw["action"] = raw.pop(act_alias)
        if "supports" in raw:
            sup = raw.pop("supports")
            if sup is True and "action" not in raw:
                raw["action"] = "support"
        if "contradicts" in raw:
            con = raw.pop("contradicts")
            if con is True and "action" not in raw:
                raw["action"] = "contradict"
        return raw

    @field_validator("statement")
    @classmethod
    def _statement_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("belief statement must be a non-empty string")
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        return _normalize_tags(value)

    @field_validator("action", mode="before")
    @classmethod
    def _validate_action(cls, value: Any) -> BeliefAction:
        return normalize_belief_action(value)


BeliefProposal = ReflectionBelief
BeliefUpdate = ReflectionBelief


class Belief(BaseModel):
    """A semantic belief with confidence in ``0..1``, evidence count, ticks, and confidence history."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    statement: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=DEFAULT_INITIAL_BELIEF_CONFIDENCE, ge=0.0, le=1.0)
    evidence_count: int = Field(default=1, ge=0)
    created_tick: int = Field(default=0, ge=0)
    updated_tick: int = Field(default=0, ge=0)
    history: list[ConfidenceChange] = Field(default_factory=list)

    _score: float | None = PrivateAttr(default=None)
    _recency: float | None = PrivateAttr(default=None)
    _importance: float | None = PrivateAttr(default=None)
    _relevance: float | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, values: Any) -> Any:
        if isinstance(values, ReflectionBelief):
            conf = values.confidence if values.confidence is not None else DEFAULT_INITIAL_BELIEF_CONFIDENCE
            return {
                "id": values.belief_id,
                "statement": values.statement,
                "tags": list(values.tags),
                "confidence": conf,
            }
        if isinstance(values, str):
            return {"statement": values}
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        for stmt_alias in ("text", "summary"):
            if stmt_alias in raw and "statement" not in raw:
                raw["statement"] = raw.pop(stmt_alias)
            elif stmt_alias in raw and raw[stmt_alias] == raw.get("statement"):
                raw.pop(stmt_alias)

        if "belief_id" in raw and "id" not in raw:
            raw["id"] = raw.pop("belief_id")
        if "evidence" in raw and "evidence_count" not in raw:
            raw["evidence_count"] = raw.pop("evidence")
        if "confidence_history" in raw and "history" not in raw:
            raw["history"] = raw.pop("confidence_history")

        for created_alias in ("tick", "created_at", "timestamp"):
            if created_alias in raw:
                val = raw.pop(created_alias)
                if "created_tick" not in raw and val is not None:
                    raw["created_tick"] = val

        for updated_alias in ("last_updated_tick", "updated_at", "last_recalled_tick"):
            if updated_alias in raw:
                val = raw.pop(updated_alias)
                if "updated_tick" not in raw and val is not None:
                    raw["updated_tick"] = val

        created_val = int(raw.get("created_tick", 0) or 0)
        raw["created_tick"] = created_val
        if "updated_tick" not in raw or raw["updated_tick"] is None:
            raw["updated_tick"] = created_val
        else:
            raw["updated_tick"] = max(created_val, int(raw["updated_tick"]))

        conf_val = float(raw.get("confidence", DEFAULT_INITIAL_BELIEF_CONFIDENCE))
        ev_count = int(raw.get("evidence_count", 1) if raw.get("evidence_count") is not None else 1)
        raw["evidence_count"] = ev_count

        stmt_str = str(raw.get("statement", ""))
        raw_tags = _normalize_tags(raw.get("tags", []))
        inferred_ids = _extract_object_ids_from_text(stmt_str)
        raw["tags"] = _normalize_tags([*raw_tags, *inferred_ids])

        if not raw.get("history"):
            raw["history"] = [
                ConfidenceChange(
                    tick=created_val,
                    confidence=_clamp(conf_val, 0.0, 1.0),
                    previous_confidence=None,
                    delta=0.0,
                    outcome="created",
                    evidence_count=max(1, ev_count),
                )
            ]
        return raw

    @field_validator("statement")
    @classmethod
    def _statement_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("belief statement must be a non-empty string")
        return stripped

    @field_validator("confidence")
    @classmethod
    def _confidence_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return round(_clamp(value, 0.0, 1.0), 6)

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        return _normalize_tags(value)

    @property
    def text(self) -> str:
        """Alias for ``statement``."""
        return self.statement

    @property
    def summary(self) -> str:
        """Alias for ``statement``."""
        return self.statement

    @property
    def belief_id(self) -> int | None:
        """Alias for ``id``."""
        return self.id

    @property
    def evidence(self) -> int:
        """Alias for ``evidence_count``."""
        return self.evidence_count

    @property
    def tick(self) -> int:
        """Most recent world tick for this belief (``updated_tick``)."""
        return self.updated_tick

    @property
    def last_updated_tick(self) -> int:
        """Alias for ``updated_tick``."""
        return self.updated_tick

    @property
    def last_recalled_tick(self) -> int:
        """Alias for ``updated_tick``."""
        return self.updated_tick

    @property
    def confidence_history(self) -> list[ConfidenceChange]:
        """Alias for ``history``."""
        return self.history

    @property
    def salience(self) -> float:
        """Alias for ``confidence`` (importance score in retrieval)."""
        return self.confidence

    @property
    def importance(self) -> float:
        """Importance score for retrieval (``confidence``)."""
        return self._importance if self._importance is not None else self.confidence

    @property
    def recency(self) -> float:
        """Recency score from the most recent retrieval or ``1.0`` at ``updated_tick``."""
        return self._recency if self._recency is not None else 1.0

    @property
    def relevance(self) -> float:
        """Relevance score from the most recent retrieval."""
        return self._relevance if self._relevance is not None else 0.0

    @property
    def score(self) -> float:
        """Combined score from the most recent retrieval (or ``confidence``)."""
        return self._score if self._score is not None else self.confidence

    @property
    def item(self) -> Belief:
        """Self-reference so retrieval callers expecting ``res.item`` work seamlessly."""
        return self

    @property
    def kind(self) -> Literal["belief"]:
        """Memory tier kind identifier."""
        return "belief"

    def eviction_score(
        self,
        current_tick: int | None = None,
        *,
        decay_rate: float = DEFAULT_RECENCY_DECAY_RATE,
    ) -> float:
        """Return ``confidence * recency`` used by the hard belief cap eviction policy."""
        ref_tick = current_tick if current_tick is not None else self.updated_tick
        rec = compute_recency(ref_tick, self.updated_tick, decay_rate=decay_rate)
        return round(self.confidence * rec, 6)

    def support(
        self,
        tick: int | None = None,
        *,
        delta: float = DEFAULT_BELIEF_SUPPORT_DELTA,
        tags: Sequence[str] | None = None,
        note: str | None = None,
    ) -> Belief:
        """Apply supporting evidence: increase confidence, increment ``evidence_count``, and record history."""
        old_conf = self.confidence
        step = abs(float(delta))
        new_conf = round(_clamp(old_conf + step, 0.0, 1.0), 6)
        new_tick = max(self.updated_tick, int(tick) if tick is not None else self.updated_tick)
        self.confidence = new_conf
        self.evidence_count += 1
        self.updated_tick = new_tick
        if tags:
            self.tags = _normalize_tags([*self.tags, *tags])
        self.history.append(
            ConfidenceChange(
                tick=new_tick,
                confidence=new_conf,
                previous_confidence=old_conf,
                delta=round(new_conf - old_conf, 6),
                outcome="support",
                evidence_count=self.evidence_count,
                note=note,
            )
        )
        return self

    def contradict(
        self,
        tick: int | None = None,
        *,
        delta: float = DEFAULT_BELIEF_CONTRADICT_DELTA,
        tags: Sequence[str] | None = None,
        note: str | None = None,
    ) -> Belief:
        """Apply contradicting evidence: lower confidence, increment ``evidence_count``, and record history."""
        old_conf = self.confidence
        step = abs(float(delta))
        new_conf = round(_clamp(old_conf - step, 0.0, 1.0), 6)
        new_tick = max(self.updated_tick, int(tick) if tick is not None else self.updated_tick)
        self.confidence = new_conf
        self.evidence_count += 1
        self.updated_tick = new_tick
        if tags:
            self.tags = _normalize_tags([*self.tags, *tags])
        self.history.append(
            ConfidenceChange(
                tick=new_tick,
                confidence=new_conf,
                previous_confidence=old_conf,
                delta=round(new_conf - old_conf, 6),
                outcome="contradict",
                evidence_count=self.evidence_count,
                note=note,
            )
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize belief and its confidence history to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_text(self) -> str:
        """Render a compact text representation suitable for prompt retrieval."""
        parts = [
            f"belief tick={self.updated_tick}",
            f"confidence={round(self.confidence, 4)}",
            f"evidence={self.evidence_count}: {self.statement}",
        ]
        if self.tags:
            parts.append(f"tags=[{','.join(self.tags)}]")
        return " ".join(parts)


def truncate_self_model(text: str, max_words: int = DEFAULT_MAX_SELF_MODEL_WORDS) -> str:
    """Normalize whitespace and cap ``text`` at ``max_words`` words (default 80)."""
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    words = text.strip().split()
    if not words:
        return ""
    return " ".join(words[:max_words])


class SelfModelVersion(BaseModel):
    """A single version of the agent's evolving self-model recorded during reflection."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    tick: int = Field(default=0, ge=0)
    text: str
    word_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _populate_word_count(cls, values: Any) -> Any:
        if isinstance(values, str):
            cleaned = truncate_self_model(values)
            return {"text": cleaned, "word_count": len(cleaned.split())}
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        for text_alias in ("self_model", "statement", "description", "content"):
            if text_alias in raw and "text" not in raw:
                raw["text"] = raw.pop(text_alias)
        text_str = str(raw.get("text", "")).strip()
        raw["text"] = text_str
        raw["word_count"] = len(text_str.split()) if text_str else 0
        return raw

    @property
    def self_model(self) -> str:
        """Alias for ``text``."""
        return self.text

    @property
    def content(self) -> str:
        """Alias for ``text``."""
        return self.text

    def __str__(self) -> str:
        return self.text


SelfModelEntry = SelfModelVersion


_CREATE_BELIEFS_AND_SELF_MODEL_SQL = """
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    tags TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    created_tick INTEGER NOT NULL,
    updated_tick INTEGER NOT NULL,
    history TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_beliefs_updated_tick ON beliefs(updated_tick);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);

CREATE TABLE IF NOT EXISTS self_model_history (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER NOT NULL,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_self_model_tick ON self_model_history(tick);
"""


# Track active stores in the current test/session for standalone retrieve(context_tags, k) calls.
_active_belief_store: BeliefStore | None = None
_active_belief_store_test: str | None = None
_active_episode_store: EpisodeStore | None = None
_active_episode_store_test: str | None = None


def _current_test_id() -> str | None:
    return os.environ.get("PYTEST_CURRENT_TEST")


def _set_active_belief_store(store: BeliefStore | None) -> None:
    global _active_belief_store, _active_belief_store_test
    _active_belief_store = store
    _active_belief_store_test = _current_test_id()


def _get_active_belief_store() -> BeliefStore | None:
    if _active_belief_store is None:
        return None
    if _active_belief_store._conn is None:
        return None
    curr = _current_test_id()
    if curr is not None and _active_belief_store_test is not None and curr != _active_belief_store_test:
        return None
    return _active_belief_store


def _set_active_episode_store(store: EpisodeStore | None) -> None:
    global _active_episode_store, _active_episode_store_test
    _active_episode_store = store
    _active_episode_store_test = _current_test_id()


def _get_active_episode_store() -> EpisodeStore | None:
    if _active_episode_store is None:
        return None
    if getattr(_active_episode_store, "_conn", None) is None:
        return None
    curr = _current_test_id()
    if curr is not None and _active_episode_store_test is not None and curr != _active_episode_store_test:
        return None
    return _active_episode_store


class SelfModel:
    """Evolving short self-description (max ~80 words) with every version stored."""

    def __init__(
        self,
        initial_text: str | None = None,
        *,
        max_words: int = DEFAULT_MAX_SELF_MODEL_WORDS,
        connection: sqlite3.Connection | None = None,
        db_path: Path | str | None = None,
        tick: int = 0,
    ) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be > 0")
        self.max_words: int = int(max_words)
        self._owns_conn: bool = connection is None
        if connection is not None:
            self._conn: sqlite3.Connection | None = connection
        elif db_path is not None:
            p_str = str(db_path)
            if p_str != ":memory:":
                Path(p_str).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(p_str, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        self._conn.executescript(_CREATE_BELIEFS_AND_SELF_MODEL_SQL)
        self._conn.commit()

        if initial_text is not None and initial_text.strip() and initial_text.strip().lower() != "none":
            self.update(initial_text, tick=tick)

    def _row_to_version(self, row: sqlite3.Row) -> SelfModelVersion:
        return SelfModelVersion(
            version=int(row["version"]),
            tick=int(row["tick"]),
            text=str(row["text"]),
            word_count=int(row["word_count"]),
        )

    def update(self, text: str, *, tick: int = 0) -> SelfModelVersion:
        """Rewrite the self-model (truncating to ``max_words``) and store the new version."""
        cleaned = truncate_self_model(str(text), max_words=self.max_words)
        if not cleaned:
            raise ValueError("self_model text must be a non-empty string")
        word_count = len(cleaned.split())
        assert self._conn is not None
        cur = self._conn.execute(
            "INSERT INTO self_model_history (tick, text, word_count) VALUES (?, ?, ?)",
            (max(0, int(tick)), cleaned, word_count),
        )
        self._conn.commit()
        ver_id = int(cur.lastrowid)  # type: ignore[arg-type]
        return SelfModelVersion(
            version=ver_id,
            tick=max(0, int(tick)),
            text=cleaned,
            word_count=word_count,
        )

    rewrite = update
    set = update
    add = update
    record = update

    def all(self) -> list[SelfModelVersion]:
        """Return every stored version of the self-model in chronological order."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM self_model_history ORDER BY version ASC"
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    @property
    def history(self) -> list[SelfModelVersion]:
        """All stored versions of the self-model."""
        return self.all()

    @property
    def versions(self) -> list[SelfModelVersion]:
        """Alias for ``history``."""
        return self.all()

    @property
    def current_version(self) -> SelfModelVersion | None:
        """Return the latest ``SelfModelVersion``, or ``None`` if never written."""
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT * FROM self_model_history ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return self._row_to_version(row) if row is not None else None

    @property
    def text(self) -> str:
        """Current self-model text, or ``''`` if empty."""
        ver = self.current_version
        return ver.text if ver is not None else ""

    @property
    def current(self) -> str:
        """Alias for ``text``."""
        return self.text

    @property
    def latest(self) -> str:
        """Alias for ``text``."""
        return self.text

    @property
    def word_count(self) -> int:
        """Word count of the current self-model."""
        ver = self.current_version
        return ver.word_count if ver is not None else 0

    def clear(self) -> None:
        """Delete all stored self-model versions."""
        if self._conn is not None:
            self._conn.execute("DELETE FROM self_model_history")
            self._conn.commit()

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __len__(self) -> int:
        return len(self.all())

    def __iter__(self) -> Iterator[SelfModelVersion]:
        return iter(self.all())

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return self.text


SelfModelStore = SelfModel


class BeliefStore:
    """SQLite-backed store of semantic beliefs with a hard cap and ``confidence * recency`` eviction."""

    def __init__(
        self,
        db_path: Path | str | MemoryConfig | Settings | int | None = None,
        *,
        path: Path | str | None = None,
        connection: sqlite3.Connection | None = None,
        config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
        settings: Settings | None = None,
        max_beliefs: int | None = None,
        cap: int | None = None,
        belief_cap: int | None = None,
        capacity: int | None = None,
        limit: int | None = None,
        support_delta: float | None = None,
        contradict_delta: float | None = None,
        recency_decay_rate: float | None = None,
        world: World | None = None,
        events: EventBus | None = None,
    ) -> None:
        from arden.memory.store import resolve_memory_config

        explicit_cap_arg: int | None = None
        if isinstance(db_path, int) and not isinstance(db_path, bool):
            explicit_cap_arg = int(db_path)
            base_cfg = resolve_memory_config(config, settings=settings)
            explicit_path = path
        elif isinstance(db_path, MemoryConfig) or (
            db_path is not None and hasattr(db_path, "memory")
        ):
            base_cfg = resolve_memory_config(db_path, settings=settings)
            explicit_path = path
        else:
            base_cfg = resolve_memory_config(config, settings=settings)
            explicit_path = db_path if db_path is not None else path

        resolved_cap = (
            explicit_cap_arg
            if explicit_cap_arg is not None
            else max_beliefs
            if max_beliefs is not None
            else cap
            if cap is not None
            else belief_cap
            if belief_cap is not None
            else capacity
            if capacity is not None
            else limit
            if limit is not None
            else base_cfg.max_beliefs
        )
        if isinstance(resolved_cap, bool) or not isinstance(resolved_cap, int) or resolved_cap <= 0:
            raise ValueError("max_beliefs (cap) must be a positive integer")

        overrides: dict[str, Any] = {"max_beliefs": resolved_cap}
        if explicit_path is not None:
            overrides["db_path"] = explicit_path
        if support_delta is not None:
            overrides["belief_support_delta"] = float(support_delta)
        if contradict_delta is not None:
            overrides["belief_contradict_delta"] = float(contradict_delta)
        if recency_decay_rate is not None:
            overrides["recency_decay_rate"] = float(recency_decay_rate)

        self.config: MemoryConfig = MemoryConfig.model_validate({**base_cfg.model_dump(), **overrides})
        self.world: World | None = world
        self.events: EventBus = events if events is not None else EventBus()

        self._owns_conn: bool = connection is None
        if connection is not None:
            self._is_memory_db = False
            self.db_path = self.config.db_path
            self._conn: sqlite3.Connection | None = connection
        else:
            # Default to :memory: when no explicit path/config/settings were passed,
            # so standalone unit tests do not share disk state.
            use_memory = (
                explicit_path is None
                and config is None
                and settings is None
                and not isinstance(db_path, MemoryConfig | Settings)
            ) or str(explicit_path) == ":memory:"
            self._is_memory_db = use_memory
            raw_path_str = ":memory:" if use_memory else str(explicit_path or self.config.db_path)
            self.db_path = Path(raw_path_str)
            if not self._is_memory_db:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                ":memory:" if self._is_memory_db else str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row

        self._conn.executescript(_CREATE_BELIEFS_AND_SELF_MODEL_SQL)
        self._conn.commit()

        self._current_tick: int = self._query_max_tick()
        self.eviction_events: list[BeliefEvictedEvent] = []
        self.evicted_beliefs: list[Belief] = []
        if self._owns_conn:
            _set_active_belief_store(self)

    @property
    def max_beliefs(self) -> int:
        """Hard cap on the number of beliefs stored."""
        return self.config.max_beliefs

    @max_beliefs.setter
    def max_beliefs(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_beliefs must be a positive integer")
        self.config = self.config.model_copy(update={"max_beliefs": value})
        self._enforce_cap(self._current_tick)

    @property
    def cap(self) -> int:
        """Alias for ``max_beliefs``."""
        return self.config.max_beliefs

    @cap.setter
    def cap(self, value: int) -> None:
        self.max_beliefs = value

    @property
    def belief_cap(self) -> int:
        """Alias for ``max_beliefs``."""
        return self.config.max_beliefs

    @property
    def capacity(self) -> int:
        """Alias for ``max_beliefs``."""
        return self.config.max_beliefs

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                ":memory:" if self._is_memory_db else str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_CREATE_BELIEFS_AND_SELF_MODEL_SQL)
            self._conn.commit()
        return self._conn

    def subscribe(
        self,
        handler_or_type: Callable[[Any], None] | str,
        handler: Callable[[Any], None] | None = None,
        *,
        event_type: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to belief store events (such as ``'belief_evicted'``)."""
        return self.events.subscribe(handler_or_type, handler, event_type=event_type)  # type: ignore[arg-type]

    def _query_max_tick(self) -> int:
        row = self.connection.execute(
            "SELECT MAX(MAX(created_tick), MAX(updated_tick)) AS m FROM beliefs"
        ).fetchone()
        if row is not None and row["m"] is not None:
            return int(row["m"])
        return 0

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        tags = json.loads(row["tags"]) if row["tags"] else []
        raw_history = json.loads(row["history"]) if row["history"] else []
        history = [ConfidenceChange.model_validate(item) for item in raw_history]
        return Belief(
            id=int(row["id"]),
            statement=str(row["statement"]),
            tags=tags,
            confidence=round(_clamp(float(row["confidence"]), 0.0, 1.0), 6),
            evidence_count=int(row["evidence_count"]),
            created_tick=int(row["created_tick"]),
            updated_tick=int(row["updated_tick"]),
            history=history,
        )

    def _save_belief_row(self, belief: Belief) -> Belief:
        history_json = json.dumps([h.model_dump(mode="json") for h in belief.history])
        tags_json = json.dumps(belief.tags)
        conn = self.connection
        if belief.id is None:
            cur = conn.execute(
                """
                INSERT INTO beliefs (
                    statement, tags, confidence, evidence_count,
                    created_tick, updated_tick, history
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    belief.statement,
                    tags_json,
                    belief.confidence,
                    belief.evidence_count,
                    belief.created_tick,
                    belief.updated_tick,
                    history_json,
                ),
            )
            conn.commit()
            row_id = int(cur.lastrowid)  # type: ignore[arg-type]
            return belief.model_copy(update={"id": row_id})
        else:
            conn.execute(
                """
                UPDATE beliefs
                SET statement = ?,
                    tags = ?,
                    confidence = ?,
                    evidence_count = ?,
                    created_tick = ?,
                    updated_tick = ?,
                    history = ?
                WHERE id = ?
                """,
                (
                    belief.statement,
                    tags_json,
                    belief.confidence,
                    belief.evidence_count,
                    belief.created_tick,
                    belief.updated_tick,
                    history_json,
                    belief.id,
                ),
            )
            conn.commit()
            return belief

    def _emit_eviction(
        self,
        belief: Belief,
        *,
        current_tick: int,
        recency: float,
        score: float,
    ) -> BeliefEvictedEvent:
        ev = BeliefEvictedEvent(
            tick=current_tick,
            source="memory",
            belief=belief,
            belief_id=belief.id,
            statement=belief.statement,
            tags=list(belief.tags),
            confidence=belief.confidence,
            recency=recency,
            score=score,
            cap=self.max_beliefs,
        )
        self.eviction_events.append(ev)
        self.evicted_beliefs.append(belief)
        self.events.emit(ev)
        if self.world is not None and self.world.events is not self.events:
            self.world.events.emit(ev)
        return ev

    def evict_lowest(self, current_tick: int | None = None) -> Belief | None:
        """Evict and delete the single belief with the lowest ``confidence * recency``."""
        all_beliefs = self.all()
        if not all_beliefs:
            return None
        eff_tick = (
            max(self._current_tick, int(current_tick))
            if current_tick is not None
            else max(self._current_tick, max(b.updated_tick for b in all_beliefs))
        )
        rate = self.config.recency_decay_rate

        def _key(b: Belief) -> tuple[float, float, int, int]:
            rec = compute_recency(eff_tick, b.updated_tick, decay_rate=rate)
            score = round(b.confidence * rec, 8)
            return (score, b.confidence, b.updated_tick, b.id or 0)

        victim = min(all_beliefs, key=_key)
        rec = compute_recency(eff_tick, victim.updated_tick, decay_rate=rate)
        score = round(victim.confidence * rec, 6)
        if victim.id is not None:
            self.connection.execute("DELETE FROM beliefs WHERE id = ?", (victim.id,))
            self.connection.commit()
        self._emit_eviction(victim, current_tick=eff_tick, recency=rec, score=score)
        return victim

    def _enforce_cap(self, current_tick: int, *, reserve: int = 0) -> list[Belief]:
        evicted: list[Belief] = []
        target_count = max(0, self.max_beliefs - reserve)
        while self.count() > target_count:
            victim = self.evict_lowest(current_tick=current_tick)
            if victim is None:
                break
            evicted.append(victim)
        return evicted

    def get(self, target: int | Belief | str) -> Belief | None:
        """Fetch a belief by integer ``id``, ``Belief``, or statement string."""
        if isinstance(target, Belief):
            if target.id is not None:
                return self.get(target.id)
            return self.find_by_statement(target.statement)
        if isinstance(target, int) and not isinstance(target, bool):
            row = self.connection.execute(
                "SELECT * FROM beliefs WHERE id = ?", (int(target),)
            ).fetchone()
            return self._row_to_belief(row) if row is not None else None
        if isinstance(target, str):
            return self.find_by_statement(target)
        return None

    def find_by_statement(self, statement: str) -> Belief | None:
        """Find an existing belief with matching normalized statement text."""
        norm = _normalize_statement_key(statement)
        if not norm:
            return None
        for b in self.all():
            if _normalize_statement_key(b.statement) == norm:
                return b
        return None

    def all(self) -> list[Belief]:
        """Return all stored beliefs ordered by ``id`` ascending."""
        rows = self.connection.execute("SELECT * FROM beliefs ORDER BY id ASC").fetchall()
        return [self._row_to_belief(r) for r in rows]

    @property
    def beliefs(self) -> list[Belief]:
        """All beliefs currently in the store."""
        return self.all()

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS c FROM beliefs").fetchone()
        return int(row["c"]) if row is not None else 0

    def add(
        self,
        belief: Belief | ReflectionBelief | Mapping[str, Any] | str | None = None,
        *,
        statement: str | None = None,
        text: str | None = None,
        tags: Sequence[str] | None = None,
        confidence: float | None = None,
        evidence_count: int | None = None,
        tick: int | None = None,
        created_tick: int | None = None,
        updated_tick: int | None = None,
        action: BeliefAction | str | None = None,
    ) -> Belief:
        """Add or propose a belief, enforcing the hard belief cap by evicting lowest ``confidence * recency``."""
        _set_active_belief_store(self)
        if isinstance(belief, ReflectionBelief):
            eff_action = normalize_belief_action(action if action is not None else belief.action)
            if eff_action == "support":
                supported = self.support(
                    belief.belief_id if belief.belief_id is not None else belief.statement,
                    tick=tick,
                    tags=tags if tags is not None else belief.tags,
                    create_if_missing=True,
                    initial_confidence=confidence if confidence is not None else belief.confidence,
                )
                assert supported is not None
                return supported
            if eff_action == "contradict":
                contradicted = self.contradict(
                    belief.belief_id if belief.belief_id is not None else belief.statement,
                    tick=tick,
                    tags=tags if tags is not None else belief.tags,
                )
                if contradicted is not None:
                    return contradicted

        raw: dict[str, Any] = {}
        if isinstance(belief, Belief):
            raw = belief.model_dump(exclude={"id"})
        elif isinstance(belief, ReflectionBelief):
            raw = {
                "statement": belief.statement,
                "tags": list(belief.tags),
                "confidence": (
                    belief.confidence
                    if belief.confidence is not None
                    else self.config.initial_belief_confidence
                ),
            }
        elif isinstance(belief, Mapping):
            raw = dict(belief)
            raw.pop("id", None)
            if "action" in raw:
                act_val = normalize_belief_action(raw.pop("action"))
                if action is None:
                    action = act_val
            raw.pop("belief_id", None)
        elif isinstance(belief, str):
            raw = {"statement": belief}
        elif belief is not None:
            raise TypeError(f"Unsupported belief type: {type(belief).__name__}")

        if statement is not None:
            raw["statement"] = statement
        elif text is not None and "statement" not in raw:
            raw["statement"] = text

        if tags is not None:
            raw["tags"] = list(tags)
        if confidence is not None:
            raw["confidence"] = float(confidence)
        elif "confidence" not in raw or raw["confidence"] is None:
            raw["confidence"] = self.config.initial_belief_confidence
        if evidence_count is not None:
            raw["evidence_count"] = int(evidence_count)

        eff_created = (
            created_tick
            if created_tick is not None
            else tick
            if tick is not None
            else raw.get("created_tick", raw.get("tick", self._current_tick))
        )
        eff_updated = (
            updated_tick
            if updated_tick is not None
            else tick
            if tick is not None
            else raw.get("updated_tick", eff_created)
        )
        raw["created_tick"] = int(eff_created or 0)
        raw["updated_tick"] = max(raw["created_tick"], int(eff_updated or 0))

        eff_action_norm = normalize_belief_action(action) if action is not None else "propose"
        stmt_val = str(raw.get("statement", "")).strip()
        existing = self.find_by_statement(stmt_val) if stmt_val else None
        if existing is not None:
            if eff_action_norm == "contradict":
                res = self.contradict(existing, tick=raw["updated_tick"], tags=raw.get("tags"))
                assert res is not None
                return res
            res = self.support(existing, tick=raw["updated_tick"], tags=raw.get("tags"))
            assert res is not None
            return res

        validated = Belief.model_validate(raw)
        self._current_tick = max(
            self._current_tick,
            validated.created_tick,
            validated.updated_tick,
        )

        # If at cap, evict the lowest confidence * recency belief before inserting
        if self.count() >= self.max_beliefs:
            self._enforce_cap(self._current_tick, reserve=1)

        return self._save_belief_row(validated)

    propose = add
    save = add
    create = add
    insert = add
    put = add

    def support(
        self,
        target: int | Belief | str,
        *,
        tick: int | None = None,
        delta: float | None = None,
        tags: Sequence[str] | None = None,
        note: str | None = None,
        create_if_missing: bool = True,
        initial_confidence: float | None = None,
    ) -> Belief | None:
        """Support an existing belief (raising its confidence) and persist the change."""
        _set_active_belief_store(self)
        found = self.get(target)
        eff_tick = (
            max(self._current_tick, int(tick))
            if tick is not None
            else (max(self._current_tick, found.updated_tick) if found is not None else self._current_tick)
        )
        self._current_tick = max(self._current_tick, eff_tick)

        if found is None:
            if create_if_missing and isinstance(target, str) and target.strip():
                return self.add(
                    statement=target.strip(),
                    tags=tags,
                    confidence=initial_confidence if initial_confidence is not None else self.config.initial_belief_confidence,
                    tick=eff_tick,
                )
            return None

        eff_delta = float(delta) if delta is not None else self.config.belief_support_delta
        found.support(tick=eff_tick, delta=eff_delta, tags=tags, note=note)
        return self._save_belief_row(found)

    def contradict(
        self,
        target: int | Belief | str,
        *,
        tick: int | None = None,
        delta: float | None = None,
        tags: Sequence[str] | None = None,
        note: str | None = None,
    ) -> Belief | None:
        """Contradict an existing belief (lowering its confidence) and persist the change."""
        _set_active_belief_store(self)
        found = self.get(target)
        if found is None and tags:
            # Fallback: if contradicted by tag overlap when statement is paraphrased
            wanted = {t.strip().lower() for t in tags if t.strip()}
            best: Belief | None = None
            best_overlap = 0
            for b in self.all():
                b_tags = {t.lower() for t in b.tags}
                ov = len(wanted & b_tags)
                if ov > best_overlap:
                    best_overlap = ov
                    best = b
            found = best

        if found is None:
            return None

        eff_tick = max(
            self._current_tick,
            found.updated_tick,
            int(tick) if tick is not None else self._current_tick,
        )
        self._current_tick = max(self._current_tick, eff_tick)
        eff_delta = float(delta) if delta is not None else self.config.belief_contradict_delta
        found.contradict(tick=eff_tick, delta=eff_delta, tags=tags, note=note)
        return self._save_belief_row(found)

    refute = contradict

    def apply_reflection_beliefs(
        self,
        beliefs: Sequence[ReflectionBelief | Belief | Mapping[str, Any] | str],
        *,
        tick: int | None = None,
    ) -> list[Belief]:
        """Apply a list of reflection belief proposals/supports/contradictions to this store."""
        _set_active_belief_store(self)
        eff_tick = int(tick) if tick is not None else self._current_tick
        self._current_tick = max(self._current_tick, eff_tick)
        updated: list[Belief] = []
        for item in beliefs:
            if isinstance(item, Belief):
                rb = ReflectionBelief(
                    statement=item.statement,
                    tags=list(item.tags),
                    action="propose",
                    confidence=item.confidence,
                    belief_id=item.id,
                )
            elif isinstance(item, ReflectionBelief):
                rb = item
            else:
                rb = ReflectionBelief.model_validate(item)

            if rb.action == "support":
                target: int | str = rb.belief_id if rb.belief_id is not None else rb.statement
                res = self.support(
                    target,
                    tick=eff_tick,
                    tags=rb.tags,
                    create_if_missing=True,
                    initial_confidence=rb.confidence,
                )
                if res is not None:
                    updated.append(res)
            elif rb.action == "contradict":
                target = rb.belief_id if rb.belief_id is not None else rb.statement
                res = self.contradict(target, tick=eff_tick, tags=rb.tags)
                if res is not None:
                    updated.append(res)
            else:
                res = self.add(
                    statement=rb.statement,
                    tags=rb.tags,
                    confidence=rb.confidence,
                    tick=eff_tick,
                )
                updated.append(res)
        return updated

    def query(
        self,
        *,
        tag: str | None = None,
        tags: Sequence[str] | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[Belief]:
        """Filter beliefs by tag(s) or minimum confidence."""
        wanted: set[str] = set()
        if tag is not None and tag.strip():
            wanted.add(tag.strip().lower())
        if tags is not None:
            for t in tags:
                if isinstance(t, str) and t.strip():
                    wanted.add(t.strip().lower())

        results: list[Belief] = []
        for b in self.all():
            if min_confidence is not None and b.confidence < float(min_confidence):
                continue
            if wanted:
                b_tags = {t.lower() for t in b.tags}
                if not (wanted & b_tags):
                    continue
            results.append(b)
            if limit is not None and len(results) >= limit:
                break
        return results

    def retrieve(
        self,
        context_tags: Sequence[str] | str | Any,
        k: int = DEFAULT_RETRIEVAL_K,
        *,
        current_tick: int | None = None,
        embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
    ) -> list[Belief]:
        """Retrieve the top-``k`` beliefs from this store scored by recency, confidence, and tag overlap."""
        items = retrieve(
            context_tags,
            k,
            beliefs=self.all(),
            episodes=[],
            current_tick=current_tick if current_tick is not None else self._current_tick,
            recency_decay_rate=self.config.recency_decay_rate,
            embedding_fn=embedding_fn,
        )
        return [item for item in items if isinstance(item, Belief)]

    def delete(self, belief_id: int | Belief) -> bool:
        resolved_id = belief_id.id if isinstance(belief_id, Belief) else int(belief_id)
        if resolved_id is None:
            return False
        cur = self.connection.execute("DELETE FROM beliefs WHERE id = ?", (resolved_id,))
        self.connection.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        self.connection.execute("DELETE FROM beliefs")
        self.connection.commit()
        self._current_tick = 0
        self.eviction_events = []
        self.evicted_beliefs = []

    def close(self) -> None:
        global _active_belief_store
        if _active_belief_store is self:
            _active_belief_store = None
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __enter__(self) -> BeliefStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[Belief]:
        return iter(self.all())

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Belief):
            return self.get(item) is not None
        if isinstance(item, int | str) and not isinstance(item, bool):
            return self.get(item) is not None
        return False


SemanticBeliefStore = BeliefStore


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in ``[0.0, 1.0]`` between two numeric embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    cos = dot / (norm_a * norm_b)
    return _clamp(cos, 0.0, 1.0)


def extract_context_tags(context: Any) -> list[str]:
    """Extract normalized context tags (object IDs, verbs, properties) from a tag list, Percept, or string."""
    if context is None:
        return []
    if isinstance(context, str):
        stripped = context.strip()
        if not stripped:
            return []
        ids = _extract_object_ids_from_text(stripped)
        if ids:
            return _normalize_tags([*ids, stripped] if " " not in stripped else ids)
        return [t for t in re.split(r"[\s,]+", stripped) if t]

    # Percept or PerceivedObject
    if hasattr(context, "objects") or hasattr(context, "affordances") or hasattr(context, "id"):
        tags: list[str] = []
        oid = getattr(context, "id", None)
        if isinstance(oid, str) and oid:
            tags.append(oid)
        objs = getattr(context, "objects", None)
        if isinstance(objs, Sequence):
            for obj in objs:
                obj_id = getattr(obj, "id", None) if not isinstance(obj, Mapping) else obj.get("id")
                if isinstance(obj_id, str) and obj_id:
                    tags.append(obj_id)
                shape = getattr(obj, "shape", None) if not isinstance(obj, Mapping) else obj.get("shape")
                if isinstance(shape, str) and shape:
                    tags.append(shape)
        body = getattr(context, "body", None)
        if body is not None:
            held = getattr(body, "holding", None) if not isinstance(body, Mapping) else body.get("holding")
            if isinstance(held, str) and held:
                tags.append(held)
        aff = getattr(context, "affordances", None)
        if isinstance(aff, Mapping):
            for obj_id in aff:
                if isinstance(obj_id, str) and obj_id:
                    tags.append(obj_id)
        return _normalize_tags(tags)

    if isinstance(context, Sequence | set | frozenset):
        return _normalize_tags(context)

    return []


def compute_tag_relevance(
    item_tags: Sequence[str],
    context_tags: Sequence[str],
    *,
    item_text: str = "",
    embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
) -> float:
    """Compute relevance score in ``[0.0, 1.0]`` from tag overlap and optional embedding hook."""
    norm_context = {t.strip().lower() for t in context_tags if isinstance(t, str) and t.strip()}
    norm_item = {t.strip().lower() for t in item_tags if isinstance(t, str) and t.strip()}
    if item_text:
        for oid in _extract_object_ids_from_text(item_text):
            norm_item.add(oid.lower())
        # Also match word tokens in statement/summary against context tags
        word_tokens = {tok.lower() for tok in re.findall(r"[a-zA-Z0-9_]+", item_text)}
    else:
        word_tokens = set()

    tag_score = 0.0
    if norm_context:
        direct_hits = norm_context & norm_item
        text_hits = (norm_context - direct_hits) & word_tokens
        effective_overlap = len(direct_hits) + 0.5 * len(text_hits)
        union_size = len(norm_context | norm_item) or 1
        coverage = effective_overlap / len(norm_context)
        jaccard = effective_overlap / union_size
        tag_score = _clamp(0.75 * coverage + 0.25 * jaccard, 0.0, 1.0)

    if embedding_fn is not None and (norm_context or item_text):
        query_str = " ".join(context_tags) if context_tags else ""
        target_str = item_text or " ".join(item_tags)
        if query_str and target_str:
            emb_q = embedding_fn(query_str)
            emb_t = embedding_fn(target_str)
            emb_sim = cosine_similarity(emb_q, emb_t)
            if norm_context and tag_score > 0.0:
                return round(_clamp(0.5 * tag_score + 0.5 * emb_sim, 0.0, 1.0), 6)
            return round(_clamp(max(tag_score, emb_sim), 0.0, 1.0), 6)

    return round(tag_score, 6)


def score_memory_item(
    item: Episode | Belief,
    context_tags: Sequence[str],
    *,
    current_tick: int = 0,
    recency_decay_rate: float = DEFAULT_RECENCY_DECAY_RATE,
    embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
    w_recency: float = DEFAULT_RECENCY_WEIGHT,
    w_importance: float = DEFAULT_IMPORTANCE_WEIGHT,
    w_relevance: float = DEFAULT_RELEVANCE_WEIGHT,
) -> tuple[float, float, float, float]:
    """Return ``(total_score, recency, importance, relevance)`` for an ``Episode`` or ``Belief``."""
    if isinstance(item, Belief):
        ref_tick = max(item.created_tick, item.updated_tick)
        importance = _clamp(item.confidence, 0.0, 1.0)
        item_tags = item.tags
        item_text = item.statement
    else:
        ref_tick = max(item.tick, getattr(item, "last_recalled_tick", item.tick))
        importance = _clamp(item.salience, 0.0, 1.0)
        item_tags = [*item.tags, *getattr(item, "made_by_agent", ())]
        item_text = item.summary

    eff_tick = max(int(current_tick), ref_tick)
    recency = compute_recency(eff_tick, ref_tick, decay_rate=recency_decay_rate)
    relevance = compute_tag_relevance(
        item_tags,
        context_tags,
        item_text=item_text,
        embedding_fn=embedding_fn,
    )
    weight_sum = max(1e-9, float(w_recency) + float(w_importance) + float(w_relevance))
    total = (
        float(w_recency) * recency
        + float(w_importance) * importance
        + float(w_relevance) * relevance
    ) / weight_sum
    return round(total, 6), round(recency, 6), round(importance, 6), round(relevance, 6)


def _is_memory_source(obj: Any) -> bool:
    if obj is None or isinstance(obj, str):
        return False
    if isinstance(obj, BeliefStore):
        return True
    cls_name = type(obj).__name__
    if cls_name in ("EpisodeStore", "MemoryManager", "MemorySystem", "MemoryRetriever"):
        return True
    if isinstance(obj, Sequence) and obj and not isinstance(obj, str | bytes):
        first = obj[0]
        if isinstance(first, Belief) or type(first).__name__ == "Episode":
            return True
    return False


def retrieve(
    context_tags_or_source: Any = None,
    k: int | Sequence[str] | str | None = None,
    k_override: int | None = None,
    *,
    context_tags: Sequence[str] | str | Any = None,
    episodes: Sequence[Episode] | EpisodeStore | None = None,
    beliefs: Sequence[Belief] | BeliefStore | None = None,
    memories: Sequence[Any] | None = None,
    store: EpisodeStore | BeliefStore | Any = None,
    belief_store: BeliefStore | None = None,
    episode_store: EpisodeStore | None = None,
    manager: Any = None,
    current_tick: int | None = None,
    tick: int | None = None,
    recency_decay_rate: float = DEFAULT_RECENCY_DECAY_RATE,
    embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
    embed_fn: EmbeddingHook | None = None,
    embedding_hook: EmbeddingHook | None = None,
    w_recency: float = DEFAULT_RECENCY_WEIGHT,
    w_importance: float = DEFAULT_IMPORTANCE_WEIGHT,
    w_relevance: float = DEFAULT_RELEVANCE_WEIGHT,
) -> list[Any]:
    """Retrieve the top-``k`` episodes and beliefs scored by recency, importance, and relevance.

    Can be called as:
    - ``retrieve(context_tags, k)`` (uses explicit ``episodes``/``beliefs``/``store`` if passed,
      or falls back to the active ``EpisodeStore`` and ``BeliefStore`` in the current session)
    - ``retrieve(store_or_manager_or_items, context_tags, k)``
    """
    from arden.memory.store import Episode, EpisodeStore

    # Disambiguate positional arguments: retrieve(context_tags, k) vs retrieve(source, context_tags, k)
    resolved_source: Any = None
    if _is_memory_source(context_tags_or_source):
        resolved_source = context_tags_or_source
        raw_tags = context_tags if context_tags is not None else k
        resolved_k = int(k_override) if k_override is not None else DEFAULT_RETRIEVAL_K
    else:
        raw_tags = context_tags if context_tags is not None else context_tags_or_source
        resolved_k = int(k) if isinstance(k, int) and not isinstance(k, bool) else (
            int(k_override) if k_override is not None else DEFAULT_RETRIEVAL_K
        )

    if resolved_k <= 0:
        return []

    tags_list = extract_context_tags(raw_tags)
    eff_embed = (
        embedding_fn
        if embedding_fn is not None
        else (embed_fn if embed_fn is not None else embedding_hook)
    )

    candidate_items: list[Episode | Belief] = []
    explicit_sources_given = any(
        x is not None
        for x in (resolved_source, episodes, beliefs, memories, store, belief_store, episode_store, manager)
    )

    def _collect_from(src: Any) -> None:
        if src is None:
            return
        if isinstance(src, BeliefStore):
            candidate_items.extend(src.all())
        elif isinstance(src, EpisodeStore):
            candidate_items.extend(src.all())
            bs = getattr(src, "_belief_store", None)
            if isinstance(bs, BeliefStore):
                candidate_items.extend(bs.all())
        elif hasattr(src, "store") and hasattr(src, "belief_store"):
            # MemoryManager
            if src.store is not None:
                candidate_items.extend(src.store.all())
            if src.belief_store is not None:
                candidate_items.extend(src.belief_store.all())
        elif isinstance(src, Sequence) and not isinstance(src, str | bytes):
            for elem in src:
                if isinstance(elem, Episode | Belief):
                    candidate_items.append(elem)

    if explicit_sources_given:
        for src in (resolved_source, episodes, beliefs, memories, store, belief_store, episode_store, manager):
            _collect_from(src)
    else:
        active_bs = _get_active_belief_store()
        active_es = _get_active_episode_store()
        if active_bs is not None:
            candidate_items.extend(active_bs.all())
        if active_es is not None:
            candidate_items.extend(active_es.all())
            es_bs = getattr(active_es, "_belief_store", None)
            if isinstance(es_bs, BeliefStore) and es_bs is not active_bs:
                candidate_items.extend(es_bs.all())

    if not candidate_items:
        return []

    # Deduplicate while preserving order
    deduped: list[Episode | Belief] = []
    seen_keys: set[tuple[str, int | None, str]] = set()
    for item in candidate_items:
        if isinstance(item, Belief):
            key = ("belief", item.id, _normalize_statement_key(item.statement))
        else:
            key = ("episode", item.id, _normalize_statement_key(item.summary))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(item)

    eff_tick = tick if tick is not None else current_tick
    if eff_tick is None:
        max_t = 0
        for item in deduped:
            t_val = item.updated_tick if isinstance(item, Belief) else max(item.tick, item.last_recalled_tick)
            if t_val > max_t:
                max_t = t_val
        eff_tick = max_t

    scored: list[tuple[tuple[float, float, float, float, int], Episode | Belief]] = []
    for idx, item in enumerate(deduped):
        total, rec, imp, rel = score_memory_item(
            item,
            tags_list,
            current_tick=int(eff_tick),
            recency_decay_rate=recency_decay_rate,
            embedding_fn=eff_embed,
            w_recency=w_recency,
            w_importance=w_importance,
            w_relevance=w_relevance,
        )
        copy_item = item.model_copy(deep=True)
        copy_item._score = total
        copy_item._recency = rec
        copy_item._importance = imp
        copy_item._relevance = rel
        scored.append(((total, rel, imp, rec, -idx), copy_item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [pair[1] for pair in scored[:resolved_k]]


class MemoryRetriever:
    """Callable helper binding an ``EpisodeStore``, ``BeliefStore``, and optional ``EmbeddingHook``."""

    def __init__(
        self,
        *,
        episode_store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
        recency_decay_rate: float = DEFAULT_RECENCY_DECAY_RATE,
    ) -> None:
        self.episode_store = episode_store
        self.belief_store = belief_store
        self.embedding_fn = embedding_fn
        self.recency_decay_rate = recency_decay_rate

    def retrieve(
        self,
        context_tags: Sequence[str] | str | Any,
        k: int = DEFAULT_RETRIEVAL_K,
        *,
        current_tick: int | None = None,
        embedding_fn: EmbeddingHook | None = None,
    ) -> list[Any]:
        return retrieve(
            context_tags,
            k,
            episode_store=self.episode_store,
            belief_store=self.belief_store,
            current_tick=current_tick,
            recency_decay_rate=self.recency_decay_rate,
            embedding_fn=embedding_fn if embedding_fn is not None else self.embedding_fn,
        )

    __call__ = retrieve


__all__ = [
    "DEFAULT_EMBEDDING_HOOK",
    "DEFAULT_IMPORTANCE_WEIGHT",
    "DEFAULT_RELEVANCE_WEIGHT",
    "DEFAULT_RECENCY_WEIGHT",
    "Belief",
    "BeliefAction",
    "BeliefEvictedEvent",
    "BeliefHistoryEntry",
    "BeliefProposal",
    "BeliefStore",
    "BeliefUpdate",
    "ConfidenceChange",
    "ConfidenceHistoryEntry",
    "ConfidenceOutcome",
    "EmbeddingHook",
    "MemoryRetriever",
    "ReflectionBelief",
    "SelfModel",
    "SelfModelEntry",
    "SelfModelStore",
    "SelfModelVersion",
    "SemanticBeliefStore",
    "_get_active_belief_store",
    "_get_active_episode_store",
    "_set_active_belief_store",
    "_set_active_episode_store",
    "compute_recency",
    "compute_tag_relevance",
    "cosine_similarity",
    "extract_context_tags",
    "normalize_belief_action",
    "retrieve",
    "score_memory_item",
    "truncate_self_model",
]
