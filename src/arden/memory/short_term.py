"""Short-term memory buffer and reflection into episodic memories.

Rules from docs/DESIGN.md (Sections 9, 9.1, 10.4, 11) and Ticket T13:
- Fixed-size buffer (config, default 40) of observations, actions, and outcomes; emits ``buffer_full``.
- Reflection is triggered by a full buffer or by rest (idle for M ticks).
- Reflection calls the LLM ``reflect`` role with the buffer and returns ``Episode`` objects
  (summary text, tags such as object ids and verbs, salience ``0..1``, ``surprise_kind`` if any,
  and ``made_by_agent`` references).
- Comfort and surprise raise salience (uncomfortable or surprising episodes are remembered more strongly).
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from arden.actions.verbs import ActionResult
from arden.llm.base import LLMBackend, LLMOutputError
from arden.llm.prompts import build_reflect_prompt, format_body_state
from arden.llm.schemas import VERBS, Intent, Step
from arden.memory.beliefs import (
    DEFAULT_EMBEDDING_HOOK,
    Belief,
    BeliefEvictedEvent,
    BeliefStore,
    EmbeddingHook,
    ReflectionBelief,
    SelfModel,
    SelfModelVersion,
    _get_active_belief_store,
    retrieve,
)
from arden.memory.store import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_REST_IDLE_TICKS,
    DEFAULT_RETRIEVAL_K,
    SURPRISE_KIND_WEIGHTS,
    Episode,
    EpisodeStore,
    MemoryConfig,
    ReflectionOutput,
    SurpriseKind,
    _normalize_str_list,
    compute_episode_salience,
    normalize_surprise_kind,
    resolve_memory_config,
)
from arden.world.objects import WorldObject
from arden.world.world import (
    AgentBody,
    BufferFullEvent,
    EventBus,
    IntentEndedEvent,
    ReflectionEvent,
    World,
    WorldEvent,
)

if TYPE_CHECKING:
    from arden.config import Settings
    from arden.perception.feed import Percept

_OBJECT_ID_RE = re.compile(r"\bobject_[0-9]+\b")
_VERB_SET = frozenset(VERBS)
_IDLE_ACTIONS = frozenset({"idle", "rest", "none", ""})


def _extract_object_ids(value: Any) -> list[str]:
    if value is None:
        return []
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    seen: set[str] = set()
    out: list[str] = []
    for match in _OBJECT_ID_RE.findall(text):
        if match not in seen:
            seen.add(match)
            out.append(match)
    return out


def _extract_verbs(action: Any) -> list[str]:
    if action is None:
        return []
    if isinstance(action, Step):
        return [action.verb]
    if isinstance(action, Intent):
        seen: set[str] = set()
        verbs: list[str] = []
        for s in action.steps:
            if s.verb not in seen:
                seen.add(s.verb)
                verbs.append(s.verb)
        return verbs
    if isinstance(action, Mapping):
        v = action.get("verb")
        if isinstance(v, str) and v.strip() in _VERB_SET:
            return [v.strip()]
        steps = action.get("steps")
        if isinstance(steps, Sequence):
            out: list[str] = []
            for st in steps:
                for vb in _extract_verbs(st):
                    if vb not in out:
                        out.append(vb)
            return out
    if isinstance(action, str):
        tokens = re.findall(r"[a-z_]+", action.lower())
        found: list[str] = []
        for tok in tokens:
            if tok in _VERB_SET and tok not in found:
                found.append(tok)
        return found
    return []


def _extract_made_by_agent(
    observation: Any = None,
    outcome: Any = None,
    world: World | None = None,
) -> list[str]:
    made: list[str] = []

    def _add(obj_id: str) -> None:
        if _OBJECT_ID_RE.fullmatch(obj_id) and obj_id not in made:
            made.append(obj_id)

    # Check PerceivedObject / Percept
    if observation is not None:
        objs = getattr(observation, "objects", None)
        if isinstance(objs, Sequence):
            for obj in objs:
                oid = getattr(obj, "id", None) if not isinstance(obj, Mapping) else obj.get("id")
                st = getattr(obj, "state", None) if not isinstance(obj, Mapping) else obj.get("state")
                mba = getattr(obj, "made_by_agent", False) if not isinstance(obj, Mapping) else obj.get("made_by_agent", False)
                if isinstance(oid, str) and (mba is True or (isinstance(st, Mapping) and st.get("made_by_agent") is True)):
                    _add(oid)
        elif isinstance(observation, Mapping):
            raw_objs = observation.get("objects")
            if isinstance(raw_objs, Sequence):
                for obj in raw_objs:
                    if isinstance(obj, Mapping):
                        oid = obj.get("id")
                        st = obj.get("state")
                        if isinstance(oid, str) and (
                            obj.get("made_by_agent") is True
                            or (isinstance(st, Mapping) and st.get("made_by_agent") is True)
                        ):
                            _add(oid)

    # Check ActionResult / outcome data and events
    if isinstance(outcome, ActionResult):
        for key in ("composite", "object", "object_id", "created_id"):
            val = outcome.data.get(key)
            if isinstance(val, WorldObject) and getattr(val, "made_by_agent", False):
                _add(val.id)
            elif isinstance(val, str) and outcome.data.get("made_by_agent") is True:
                _add(val)
        for ev in outcome.events:
            oid = getattr(ev, "object_id", None)
            ev_obj = getattr(ev, "object", None)
            if isinstance(oid, str) and isinstance(ev_obj, Mapping):
                st = ev_obj.get("state")
                if ev_obj.get("made_by_agent") is True or (
                    isinstance(st, Mapping) and st.get("made_by_agent") is True
                ):
                    _add(oid)
    elif isinstance(outcome, Mapping):
        if outcome.get("made_by_agent") is True and isinstance(outcome.get("object_id"), str):
            _add(outcome["object_id"])
        mba_list = outcome.get("made_by_agent")
        if isinstance(mba_list, Sequence) and not isinstance(mba_list, str):
            for item in mba_list:
                if isinstance(item, str):
                    _add(item)

    if world is not None:
        for obj in world.objects.values():
            if getattr(obj, "made_by_agent", False):
                _add(obj.id)

    return made


def _format_field(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        s = value.strip()
        return s if s else "none"
    if hasattr(value, "to_text") and callable(value.to_text):
        t = str(value.to_text()).strip()
        return " ".join(t.split()) if t else "none"
    if isinstance(value, ActionResult):
        parts = [f"ok={str(value.ok).lower()}"]
        if value.reason is not None:
            parts.append(f"reason={value.reason}")
        return " ".join(parts)
    if isinstance(value, Step):
        target_str = f" target={value.target}" if value.target else ""
        params_str = (
            f" params={json.dumps(value.params, sort_keys=True, separators=(',', ':'))}"
            if value.params
            else ""
        )
        return f"verb={value.verb}{target_str}{params_str}"
    if isinstance(value, Intent):
        verbs = ",".join(s.verb for s in value.steps)
        return f"goal={value.goal!r} steps=[{verbs}] expect={value.expect!r}"
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return str(value).strip() or "none"


def _is_idle_action(action: Any, resting: bool | None = None) -> bool:
    if resting is True:
        return True
    if resting is False:
        return False
    if action is None:
        return False
    if isinstance(action, str):
        return action.strip().lower() in _IDLE_ACTIONS
    if isinstance(action, Mapping):
        v = action.get("verb", action.get("action"))
        if isinstance(v, str):
            return v.strip().lower() in _IDLE_ACTIONS
    return False


class BufferEntry(BaseModel):
    """A single entry in the short-term memory buffer (observation, action, outcome)."""

    model_config = ConfigDict(extra="forbid")

    tick: int = Field(default=0, ge=0)
    observation: Any = None
    action: Any = None
    outcome: Any = None
    comfort: float | None = Field(default=None, ge=0.0, le=1.0)
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0)
    surprise_kind: SurpriseKind | None = None
    made_by_agent: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    resting: bool = False

    @field_validator("surprise_kind", mode="before")
    @classmethod
    def _validate_surprise_kind(cls, value: Any) -> SurpriseKind | None:
        return normalize_surprise_kind(value)

    @field_validator("made_by_agent", "tags", mode="before")
    @classmethod
    def _validate_str_list(cls, value: Any) -> list[str]:
        return _normalize_str_list(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this buffer entry."""
        return {
            "tick": self.tick,
            "observation": _format_field(self.observation),
            "action": _format_field(self.action),
            "outcome": _format_field(self.outcome),
            "comfort": self.comfort,
            "energy": self.energy,
            "surprise": self.surprise,
            "surprise_kind": self.surprise_kind,
            "made_by_agent": list(self.made_by_agent),
            "tags": list(self.tags),
            "resting": self.resting,
        }

    def to_text(self) -> str:
        """Render a compact single-line summary for reflection prompts (no evaluative labels)."""
        parts = [
            f"tick={self.tick}",
            f"obs={_format_field(self.observation)}",
            f"act={_format_field(self.action)}",
            f"out={_format_field(self.outcome)}",
        ]
        if self.comfort is not None:
            parts.append(f"comfort={round(self.comfort, 4)}")
        if self.energy is not None:
            parts.append(f"energy={round(self.energy, 4)}")
        if self.surprise > 0.0:
            parts.append(f"surprise={round(self.surprise, 4)}")
        if self.surprise_kind is not None:
            parts.append(f"surprise_kind={self.surprise_kind}")
        if self.made_by_agent:
            parts.append(f"made_by_agent=[{','.join(self.made_by_agent)}]")
        return " ".join(parts)


ShortTermEntry = BufferEntry


class ShortTermBuffer:
    """Fixed-size window of recent observations, actions, and outcomes.

    Emits a ``buffer_full`` event when ``len(buffer) == capacity`` (default 40).
    Supports triggering reflection when full or after resting (idle) for ``M``
    (``rest_idle_ticks``) consecutive ticks.
    """

    def __init__(
        self,
        capacity: int | MemoryConfig | Settings | None = None,
        *,
        max_size: int | None = None,
        size: int | None = None,
        buffer_size: int | None = None,
        rest_idle_ticks: int | None = None,
        idle_ticks_threshold: int | None = None,
        m_ticks: int | None = None,
        config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
        settings: Settings | None = None,
        world: World | None = None,
        events: EventBus | None = None,
        llm: LLMBackend | None = None,
        store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        self_model: SelfModel | str | None = None,
        auto_reflect: bool | None = None,
    ) -> None:
        if isinstance(capacity, MemoryConfig) or (
            capacity is not None and hasattr(capacity, "memory")
        ):
            base_cfg = resolve_memory_config(capacity, settings=settings)
            explicit_cap: int | None = None
        else:
            base_cfg = resolve_memory_config(
                config,
                settings=settings if settings is not None else getattr(world, "config", None),
            )
            explicit_cap = int(capacity) if capacity is not None else None

        resolved_cap = (
            explicit_cap
            if explicit_cap is not None
            else buffer_size
            if buffer_size is not None
            else max_size
            if max_size is not None
            else size
            if size is not None
            else base_cfg.buffer_size
        )
        if isinstance(resolved_cap, bool) or not isinstance(resolved_cap, int) or resolved_cap <= 0:
            raise ValueError("buffer capacity must be a positive integer")

        resolved_idle = (
            rest_idle_ticks
            if rest_idle_ticks is not None
            else idle_ticks_threshold
            if idle_ticks_threshold is not None
            else m_ticks
            if m_ticks is not None
            else base_cfg.rest_idle_ticks
        )
        if isinstance(resolved_idle, bool) or not isinstance(resolved_idle, int) or resolved_idle <= 0:
            raise ValueError("rest_idle_ticks must be a positive integer")

        self.config: MemoryConfig = base_cfg.model_copy(
            update={"buffer_size": resolved_cap, "rest_idle_ticks": resolved_idle}
        )
        self.capacity: int = resolved_cap
        self.rest_idle_ticks: int = resolved_idle
        self.world: World | None = world
        self.events: EventBus = events if events is not None else EventBus()
        self.llm: LLMBackend | None = llm
        self.store: EpisodeStore | None = store
        self.belief_store: BeliefStore | None = (
            belief_store
            if belief_store is not None
            else (store.belief_store if store is not None else None)
        )
        if isinstance(self_model, SelfModel):
            self.self_model_store: SelfModel | None = self_model
        elif isinstance(self_model, str):
            self.self_model_store = SelfModel(
                self_model, max_words=self.config.max_self_model_words
            )
        elif store is not None:
            self.self_model_store = store.self_model_store
        else:
            self.self_model_store = None
        self.auto_reflect: bool = (llm is not None) if auto_reflect is None else bool(auto_reflect)

        self._entries: deque[BufferEntry] = deque(maxlen=self.capacity)
        self.idle_ticks: int = 0
        self.buffer_full_events: list[BufferFullEvent] = []
        self.reflection_events: list[ReflectionEvent] = []
        self.last_reflected_episodes: list[Episode] = []
        self.last_reflected_beliefs: list[Belief] = []
        self.last_reflected_self_model: str | None = None
        self._buffer_full_callbacks: list[Callable[[BufferFullEvent], None]] = []

        if self.world is not None:
            self.world.events.subscribe("tick", self._on_world_tick)

    def _on_world_tick(self, event: WorldEvent) -> None:
        if self.world is None:
            return
        bs = getattr(self.world, "body_state", None)
        is_resting = bool(
            getattr(bs, "_last_step_resting", False) or getattr(bs, "resting", False)
        )
        if is_resting:
            self.record_idle(1, tick=event.tick)
        else:
            self.idle_ticks = 0

    @property
    def max_size(self) -> int:
        """Alias for ``capacity``."""
        return self.capacity

    @property
    def buffer_size(self) -> int:
        """Alias for ``capacity``."""
        return self.capacity

    @property
    def size(self) -> int:
        """Number of entries currently in the buffer."""
        return len(self._entries)

    @property
    def entries(self) -> list[BufferEntry]:
        """List of ``BufferEntry`` items currently in the buffer (oldest to newest)."""
        return list(self._entries)

    @property
    def items(self) -> list[BufferEntry]:
        """Alias for ``entries``."""
        return list(self._entries)

    @property
    def is_full(self) -> bool:
        """True when the buffer holds ``capacity`` entries."""
        return len(self._entries) >= self.capacity

    @property
    def full(self) -> bool:
        """Alias for ``is_full``."""
        return self.is_full

    @property
    def episodes(self) -> list[Episode]:
        """Episodes produced by the most recent reflection (or all stored episodes if ``store`` is set)."""
        if self.store is not None:
            return self.store.all()
        return list(self.last_reflected_episodes)

    @property
    def beliefs(self) -> list[Belief]:
        """Beliefs currently stored in ``belief_store`` or produced by the most recent reflection."""
        if self.belief_store is not None:
            return self.belief_store.all()
        if self.store is not None:
            return self.store.beliefs
        return list(self.last_reflected_beliefs)

    @property
    def self_model(self) -> str:
        """Current self-model text."""
        if self.self_model_store is not None:
            return self.self_model_store.text
        if self.store is not None:
            return self.store.self_model
        return self.last_reflected_self_model or ""

    @property
    def self_model_history(self) -> list[SelfModelVersion]:
        """All stored versions of the self-model."""
        if self.self_model_store is not None:
            return self.self_model_store.all()
        if self.store is not None:
            return self.store.self_model_history
        return []

    def subscribe(
        self,
        handler_or_type: Callable[[Any], None] | str,
        handler: Callable[[Any], None] | None = None,
        *,
        event_type: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to buffer events (such as ``"buffer_full"`` or ``"reflection"``)."""
        return self.events.subscribe(handler_or_type, handler, event_type=event_type)  # type: ignore[arg-type]

    def on_buffer_full(
        self, callback: Callable[[BufferFullEvent], None]
    ) -> Callable[[], None]:
        """Register a callback invoked whenever ``buffer_full`` is emitted."""
        self._buffer_full_callbacks.append(callback)

        def _unsub() -> None:
            if callback in self._buffer_full_callbacks:
                self._buffer_full_callbacks.remove(callback)

        return _unsub

    def _emit_buffer_full(self, tick: int) -> BufferFullEvent:
        ev = BufferFullEvent(
            tick=tick,
            source="memory",
            count=len(self._entries),
            capacity=self.capacity,
            entries=list(self._entries),
        )
        self.buffer_full_events.append(ev)
        self.events.emit(ev)
        if self.world is not None and self.world.events is not self.events:
            self.world.events.emit(ev)
        for cb in list(self._buffer_full_callbacks):
            cb(ev)
        return ev

    def _build_entry(
        self,
        entry: BufferEntry | Mapping[str, Any] | str | None = None,
        *,
        tick: int | None = None,
        observation: Any = None,
        action: Any = None,
        outcome: Any = None,
        comfort: float | None = None,
        energy: float | None = None,
        surprise: float | None = None,
        surprise_kind: SurpriseKind | str | None = None,
        made_by_agent: Sequence[str] | str | bool | None = None,
        tags: Sequence[str] | None = None,
        resting: bool | None = None,
    ) -> BufferEntry:
        if isinstance(entry, BufferEntry):
            base = entry.model_dump()
        elif isinstance(entry, Mapping):
            base = dict(entry)
            if "percept" in base and "observation" not in base:
                base["observation"] = base.pop("percept")
            if "obs" in base and "observation" not in base:
                base["observation"] = base.pop("obs")
            if "step" in base and "action" not in base:
                base["action"] = base.pop("step")
            if "act" in base and "action" not in base:
                base["action"] = base.pop("act")
            if "result" in base and "outcome" not in base:
                base["outcome"] = base.pop("result")
            if "out" in base and "outcome" not in base:
                base["outcome"] = base.pop("out")
        elif isinstance(entry, str):
            base = {"observation": entry}
        elif entry is not None:
            # Passed a Percept, Step, ActionResult, or IntentEndedEvent directly
            if isinstance(entry, ActionResult):
                base = {"outcome": entry}
            elif isinstance(entry, Step | Intent):
                base = {"action": entry}
            elif isinstance(entry, IntentEndedEvent):
                base = {
                    "tick": entry.tick,
                    "observation": entry.observation,
                    "action": entry.intent or entry.step,
                    "outcome": {"status": entry.status, "ok": entry.ok, "reason": entry.reason},
                }
            else:
                base = {"observation": entry}
        else:
            base = {}

        obs = observation if observation is not None else base.get("observation")
        act = action if action is not None else base.get("action")
        out = outcome if outcome is not None else base.get("outcome")

        resolved_tick = tick if tick is not None else base.get("tick")
        if resolved_tick is None:
            if hasattr(obs, "tick") and isinstance(getattr(obs, "tick"), int):
                resolved_tick = int(getattr(obs, "tick"))
            elif self.world is not None:
                resolved_tick = self.world.tick
            elif self._entries:
                resolved_tick = self._entries[-1].tick
            else:
                resolved_tick = 0

        # Resolve comfort & energy from explicit args, observation.body, or world.body
        resolved_comfort = comfort if comfort is not None else base.get("comfort")
        resolved_energy = energy if energy is not None else base.get("energy")
        if obs is not None and (resolved_comfort is None or resolved_energy is None):
            body_obj = getattr(obs, "body", None) if not isinstance(obs, Mapping) else obs.get("body")
            if body_obj is not None:
                if resolved_comfort is None:
                    c_attr = getattr(body_obj, "comfort", None) if not isinstance(body_obj, Mapping) else body_obj.get("comfort")
                    if c_attr is not None:
                        resolved_comfort = float(c_attr)
                if resolved_energy is None:
                    e_attr = getattr(body_obj, "energy", None) if not isinstance(body_obj, Mapping) else body_obj.get("energy")
                    if e_attr is not None:
                        resolved_energy = float(e_attr)
        if self.world is not None:
            if resolved_comfort is None:
                resolved_comfort = float(self.world.body.comfort)
            if resolved_energy is None:
                resolved_energy = float(self.world.body.energy)

        resolved_surprise = (
            float(surprise)
            if surprise is not None
            else float(base.get("surprise", 0.0) or 0.0)
        )
        resolved_kind = (
            surprise_kind
            if surprise_kind is not None
            else base.get("surprise_kind")
        )

        # Collect made_by_agent object IDs
        explicit_mba = (
            _normalize_str_list(made_by_agent)
            if made_by_agent is not None
            else _normalize_str_list(base.get("made_by_agent"))
        )
        inferred_mba = _extract_made_by_agent(obs, out, self.world)
        combined_mba = _normalize_str_list([*explicit_mba, *inferred_mba])

        # Collect tags (object IDs and verbs)
        explicit_tags = (
            _normalize_str_list(tags)
            if tags is not None
            else _normalize_str_list(base.get("tags"))
        )
        inferred_tags = [
            *_extract_object_ids(obs),
            *_extract_object_ids(act),
            *_extract_object_ids(out),
            *_extract_verbs(act),
            *combined_mba,
        ]
        combined_tags = _normalize_str_list([*explicit_tags, *inferred_tags])

        is_idle = _is_idle_action(act, resting if resting is not None else base.get("resting"))

        return BufferEntry(
            tick=int(resolved_tick),
            observation=obs,
            action=act,
            outcome=out,
            comfort=resolved_comfort,
            energy=resolved_energy,
            surprise=resolved_surprise,
            surprise_kind=resolved_kind,
            made_by_agent=combined_mba,
            tags=combined_tags,
            resting=is_idle,
        )

    def add(
        self,
        entry: BufferEntry | Mapping[str, Any] | str | Any = None,
        *,
        tick: int | None = None,
        observation: Any = None,
        action: Any = None,
        outcome: Any = None,
        comfort: float | None = None,
        energy: float | None = None,
        surprise: float | None = None,
        surprise_kind: SurpriseKind | str | None = None,
        made_by_agent: Sequence[str] | str | bool | None = None,
        tags: Sequence[str] | None = None,
        resting: bool | None = None,
    ) -> BufferEntry:
        """Append an entry to the buffer; emit ``buffer_full`` when reaching ``capacity``."""
        built = self._build_entry(
            entry,
            tick=tick,
            observation=observation,
            action=action,
            outcome=outcome,
            comfort=comfort,
            energy=energy,
            surprise=surprise,
            surprise_kind=surprise_kind,
            made_by_agent=made_by_agent,
            tags=tags,
            resting=resting,
        )
        self._entries.append(built)

        if built.resting:
            self.idle_ticks += 1
        elif built.action is not None:
            self.idle_ticks = 0

        if len(self._entries) >= self.capacity:
            self._emit_buffer_full(built.tick)
            if self.auto_reflect and self.llm is not None:
                self.reflect(trigger="buffer_full")
        elif (
            self.idle_ticks >= self.rest_idle_ticks
            and len(self._entries) > 0
            and self.auto_reflect
            and self.llm is not None
        ):
            self.reflect(trigger="rest")

        return built

    append = add
    record = add
    push = add
    add_entry = add

    def add_observation(self, observation: Any, *, tick: int | None = None, **kwargs: Any) -> BufferEntry:
        """Record an observation entry in the short-term buffer."""
        return self.add(observation=observation, tick=tick, **kwargs)

    def add_action(self, action: Any, *, tick: int | None = None, **kwargs: Any) -> BufferEntry:
        """Record an action entry in the short-term buffer."""
        return self.add(action=action, tick=tick, **kwargs)

    def add_outcome(self, outcome: Any, *, tick: int | None = None, **kwargs: Any) -> BufferEntry:
        """Record an outcome entry in the short-term buffer."""
        return self.add(outcome=outcome, tick=tick, **kwargs)

    def record_idle(
        self,
        ticks: int = 1,
        *,
        tick: int | None = None,
        llm: LLMBackend | None = None,
        store: EpisodeStore | None = None,
    ) -> list[Episode]:
        """Advance consecutive idle (rest) ticks by ``ticks`` and trigger reflection if >= M."""
        if ticks < 0:
            raise ValueError("ticks must be >= 0")
        self.idle_ticks += int(ticks)
        resolved_llm = llm if llm is not None else self.llm
        if (
            self.idle_ticks >= self.rest_idle_ticks
            and len(self._entries) > 0
            and resolved_llm is not None
            and (llm is not None or self.auto_reflect)
        ):
            return self.reflect(
                llm=resolved_llm,
                store=store if store is not None else self.store,
                tick=tick,
                trigger="rest",
            )
        return []

    step_idle = record_idle
    rest = record_idle
    idle = record_idle

    def should_reflect(
        self,
        *,
        idle_ticks: int | None = None,
        resting: bool = False,
    ) -> bool:
        """Return True if reflection should run (buffer is full or idle for M ticks)."""
        if self.is_full:
            return True
        eff_idle = int(idle_ticks) if idle_ticks is not None else self.idle_ticks
        if resting and eff_idle < self.rest_idle_ticks and idle_ticks is None:
            # If caller explicitly passed resting=True without an idle_ticks counter
            pass
        if eff_idle >= self.rest_idle_ticks and len(self._entries) > 0:
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from the short-term buffer and reset ``idle_ticks``."""
        self._entries.clear()
        self.idle_ticks = 0

    def to_prompt_entries(self) -> list[str]:
        """Format all buffered entries as compact strings for the ``reflect`` prompt."""
        return [entry.to_text() for entry in self._entries]

    def to_text(self) -> str:
        """Render the entire buffer as multi-line structured text."""
        if not self._entries:
            return "none"
        return "\n".join(self.to_prompt_entries())

    def reflect(
        self,
        llm: LLMBackend | None = None,
        *,
        store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        world: World | None = None,
        self_model: SelfModel | str | None = None,
        body_state: Any = None,
        comfort: float | None = None,
        surprise: float | bool | None = None,
        surprise_kind: SurpriseKind | str | None = None,
        tick: int | None = None,
        timeout_s: float = 30.0,
        clear_buffer: bool = True,
        trigger: str | None = None,
    ) -> ReflectedMemories:
        """Run reflection on this buffer using the LLM ``reflect`` role."""
        resolved_llm = llm if llm is not None else self.llm
        if resolved_llm is None:
            raise ValueError("An LLMBackend is required to run reflection")
        resolved_store = store if store is not None else self.store
        resolved_belief_store = belief_store if belief_store is not None else self.belief_store
        resolved_self_model = self_model if self_model is not None else self.self_model_store
        resolved_world = world if world is not None else self.world

        # If buffer was already auto-reflected and is currently empty, return the last reflected result
        if len(self._entries) == 0 and (self.last_reflected_episodes or self.last_reflected_beliefs):
            return ReflectedMemories(
                self.last_reflected_episodes,
                beliefs=self.last_reflected_beliefs,
                self_model=self.last_reflected_self_model,
            )

        return reflect(
            self,
            resolved_llm,
            store=resolved_store,
            belief_store=resolved_belief_store,
            world=resolved_world,
            self_model=resolved_self_model,
            body_state=body_state,
            comfort=comfort,
            surprise=surprise,
            surprise_kind=surprise_kind,
            tick=tick,
            config=self.config,
            timeout_s=timeout_s,
            clear_buffer=clear_buffer,
            trigger=trigger,
        )

    def maybe_reflect(
        self,
        llm: LLMBackend | None = None,
        *,
        store: EpisodeStore | None = None,
        idle_ticks: int | None = None,
        **kwargs: Any,
    ) -> list[Episode]:
        """Run reflection if ``should_reflect()`` is True; otherwise return ``[]``."""
        if self.should_reflect(idle_ticks=idle_ticks):
            trigger = "buffer_full" if self.is_full else "rest"
            return self.reflect(llm, store=store, trigger=trigger, **kwargs)
        return []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[BufferEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> BufferEntry:
        return list(self._entries)[index]

    def __bool__(self) -> bool:
        return bool(self._entries)


def _summarize_buffer_context(
    entries: Sequence[BufferEntry],
    *,
    world: World | None = None,
    body_state: Any = None,
    comfort: float | None = None,
    surprise: float | bool | None = None,
    surprise_kind: SurpriseKind | str | None = None,
) -> tuple[int, float, float, SurpriseKind | None, list[str], list[str]]:
    """Extract (tick, effective_comfort, effective_surprise, dominant_surprise_kind, made_by_agent, tags) from buffer."""
    latest_tick = world.tick if world is not None else 0
    for e in entries:
        if e.tick > latest_tick:
            latest_tick = e.tick

    # Comfort: use explicit arg, or minimum comfort experienced across buffer entries, or body_state/world
    if comfort is not None:
        eff_comfort = float(comfort)
    else:
        entry_comforts = [e.comfort for e in entries if e.comfort is not None]
        if entry_comforts:
            eff_comfort = min(entry_comforts)
        elif body_state is not None:
            if isinstance(body_state, Mapping) and "comfort" in body_state:
                eff_comfort = float(body_state["comfort"])
            elif hasattr(body_state, "comfort"):
                eff_comfort = float(body_state.comfort)
            else:
                eff_comfort = 1.0
        elif world is not None:
            eff_comfort = float(world.body.comfort)
        else:
            eff_comfort = 1.0

    # Surprise magnitude: use explicit arg, or max surprise across entries, or world.drives
    if surprise is True:
        eff_surprise = 1.0
    elif surprise is False:
        eff_surprise = 0.0
    elif surprise is not None:
        eff_surprise = max(0.0, float(surprise))
    else:
        entry_surprises = [e.surprise for e in entries if e.surprise > 0.0]
        if entry_surprises:
            eff_surprise = max(entry_surprises)
        elif world is not None and hasattr(world, "drives"):
            eff_surprise = float(world.drives.curiosity)
        else:
            eff_surprise = 0.0

    # Dominant surprise_kind across entries (belief/expectation > physical > novelty)
    if surprise_kind is not None:
        dom_kind = normalize_surprise_kind(surprise_kind)
    else:
        dom_kind = None
        best_w = -1.0
        for e in entries:
            if e.surprise_kind is not None:
                w = SURPRISE_KIND_WEIGHTS.get(e.surprise_kind, 0.0)
                if w > best_w:
                    best_w = w
                    dom_kind = e.surprise_kind

    made_by: list[str] = []
    tags: list[str] = []
    for e in entries:
        for m in e.made_by_agent:
            if m not in made_by:
                made_by.append(m)
        for t in e.tags:
            if t not in tags:
                tags.append(t)
    if world is not None:
        for obj in world.objects.values():
            if getattr(obj, "made_by_agent", False) and obj.id not in made_by:
                made_by.append(obj.id)

    return latest_tick, eff_comfort, eff_surprise, dom_kind, made_by, tags


class ReflectedMemories(list[Episode]):
    """List of reflected ``Episode`` objects that also carries updated ``beliefs`` and ``self_model``."""

    def __init__(
        self,
        episodes: Sequence[Episode] = (),
        *,
        beliefs: Sequence[Belief] = (),
        self_model: str | None = None,
        self_model_version: SelfModelVersion | None = None,
    ) -> None:
        super().__init__(episodes)
        self.episodes: list[Episode] = list(episodes)
        self.beliefs: list[Belief] = list(beliefs)
        self.self_model: str | None = self_model
        self.self_model_version: SelfModelVersion | None = self_model_version


def reflect(
    buffer: ShortTermBuffer | Sequence[BufferEntry | Mapping[str, Any] | str] | None = None,
    llm: LLMBackend | None = None,
    *,
    store: EpisodeStore | None = None,
    belief_store: BeliefStore | None = None,
    world: World | None = None,
    self_model: SelfModel | str | None = None,
    body_state: Any = None,
    comfort: float | None = None,
    surprise: float | bool | None = None,
    surprise_kind: SurpriseKind | str | None = None,
    tick: int | None = None,
    config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
    timeout_s: float = 30.0,
    clear_buffer: bool = True,
    trigger: str | None = None,
) -> ReflectedMemories:
    """Compress the short-term buffer into salient ``Episode`` objects, update ``BeliefStore``,
    and rewrite ``SelfModel`` via the LLM ``reflect`` role.

    - Calls ``llm.complete_json("reflect", system, user, ReflectionOutput.model_json_schema(), timeout_s)``.
    - Adjusts episode salience for comfort (discomfort raises salience) and surprise.
    - Applies belief proposals, supports, and contradictions to ``belief_store``.
    - Rewrites ``self_model`` (max ~80 words) and stores the new version.
    - Persists episodes to ``store`` when provided and clears ``buffer`` when ``clear_buffer=True``.
    """
    if isinstance(buffer, LLMBackend) and llm is None:
        llm = buffer
        buffer = ()
    if buffer is None:
        buffer = ()

    if isinstance(buffer, ShortTermBuffer):
        st_buffer: ShortTermBuffer | None = buffer
        entries = buffer.entries
        resolved_llm = llm if llm is not None else buffer.llm
        resolved_store = store if store is not None else buffer.store
        resolved_belief_store = (
            belief_store
            if belief_store is not None
            else (buffer.belief_store if buffer.belief_store is not None else (resolved_store.belief_store if resolved_store is not None else None))
        )
        resolved_self_model_obj: SelfModel | None = (
            self_model
            if isinstance(self_model, SelfModel)
            else (buffer.self_model_store if buffer.self_model_store is not None else (resolved_store.self_model_store if resolved_store is not None else None))
        )
        resolved_world = world if world is not None else buffer.world
        cfg = resolve_memory_config(config if config is not None else buffer.config)
        eff_trigger = trigger or ("buffer_full" if buffer.is_full else "rest")
    else:
        st_buffer = None
        temp_buf = ShortTermBuffer(world=world, config=config)
        for item in buffer:
            temp_buf._entries.append(temp_buf._build_entry(item))
        entries = list(temp_buf._entries)
        resolved_llm = llm
        resolved_store = store
        resolved_belief_store = (
            belief_store
            if belief_store is not None
            else (resolved_store.belief_store if resolved_store is not None else _get_active_belief_store())
        )
        resolved_self_model_obj = (
            self_model
            if isinstance(self_model, SelfModel)
            else (resolved_store.self_model_store if resolved_store is not None else None)
        )
        resolved_world = world
        cfg = resolve_memory_config(config)
        eff_trigger = trigger or "manual"

    if resolved_llm is None:
        raise ValueError("An LLMBackend is required to run reflection")

    (
        inferred_tick,
        eff_comfort,
        eff_surprise,
        dom_surprise_kind,
        buffer_made_by_agent,
        _buffer_tags,
    ) = _summarize_buffer_context(
        entries,
        world=resolved_world,
        body_state=body_state,
        comfort=comfort,
        surprise=surprise,
        surprise_kind=surprise_kind,
    )
    resolved_tick = int(tick) if tick is not None else inferred_tick

    resolved_body_state = body_state
    if resolved_body_state is None:
        if resolved_world is not None:
            resolved_body_state = resolved_world.body
        else:
            resolved_body_state = {
                "position": {"x": 0.0, "y": 0.0},
                "level": "ground",
                "energy": entries[-1].energy if entries and entries[-1].energy is not None else 1.0,
                "comfort": eff_comfort,
                "holding": None,
            }

    if isinstance(self_model, str):
        current_self_model_str: str | None = self_model
    elif resolved_self_model_obj is not None and resolved_self_model_obj.text:
        current_self_model_str = resolved_self_model_obj.text
    else:
        current_self_model_str = None

    prompt_memories = [e.to_text() for e in entries]
    if resolved_belief_store is not None:
        for b in resolved_belief_store.all():
            prompt_memories.append(b.to_text())

    prompt = build_reflect_prompt(
        percept=st_buffer.to_text() if st_buffer is not None else ("\n".join(e.to_text() for e in entries) or "none"),
        self_model=current_self_model_str,
        memories=prompt_memories,
        body_state=format_body_state(resolved_body_state),
    )

    raw_response = resolved_llm.complete_json(
        "reflect",
        prompt.system,
        prompt.user,
        ReflectionOutput.model_json_schema(),
        float(timeout_s),
    )

    if isinstance(raw_response, ReflectionOutput):
        parsed_output = raw_response
    else:
        try:
            parsed_output = ReflectionOutput.model_validate(raw_response)
        except ValidationError as exc:
            raw_str = (
                json.dumps(raw_response, sort_keys=True)
                if isinstance(raw_response, Mapping)
                else str(raw_response)
            )
            raise LLMOutputError(raw_str, str(exc)) from exc

    episodes: list[Episode] = []
    for idx, raw_ep in enumerate(parsed_output.episodes):
        ep_surprise_kind = raw_ep.surprise_kind
        if ep_surprise_kind is None and dom_surprise_kind is not None and idx == 0:
            ep_surprise_kind = dom_surprise_kind

        ep_made_by = list(raw_ep.made_by_agent)
        if buffer_made_by_agent:
            for obj_id in buffer_made_by_agent:
                if obj_id not in ep_made_by and (
                    obj_id in raw_ep.tags or obj_id in raw_ep.summary or idx == len(parsed_output.episodes) - 1
                ):
                    ep_made_by.append(obj_id)

        ep_tags = _normalize_str_list(
            [
                *raw_ep.tags,
                *_extract_object_ids(raw_ep.summary),
                *_extract_verbs(raw_ep.summary),
                *ep_made_by,
            ]
        )

        adjusted_salience = compute_episode_salience(
            raw_ep.salience,
            comfort=eff_comfort,
            surprise=eff_surprise,
            surprise_kind=ep_surprise_kind,
            config=cfg,
        )

        episodes.append(
            Episode(
                tick=resolved_tick,
                summary=raw_ep.summary,
                tags=ep_tags,
                salience=adjusted_salience,
                last_recalled_tick=resolved_tick,
                surprise_kind=ep_surprise_kind,
                made_by_agent=ep_made_by,
            )
        )

    if resolved_store is not None and episodes:
        episodes = resolved_store.save_episodes(episodes, tick=resolved_tick)

    # Apply belief updates (propose, support, contradict)
    updated_beliefs: list[Belief] = []
    if parsed_output.beliefs:
        if resolved_belief_store is not None:
            if resolved_world is not None and resolved_belief_store.world is None:
                resolved_belief_store.world = resolved_world
            updated_beliefs = resolved_belief_store.apply_reflection_beliefs(
                parsed_output.beliefs,
                tick=resolved_tick,
            )
        else:
            temp_bstore = BeliefStore(db_path=":memory:", config=cfg, world=resolved_world)
            updated_beliefs = temp_bstore.apply_reflection_beliefs(
                parsed_output.beliefs,
                tick=resolved_tick,
            )

    # Apply self-model rewrite (max ~80 words, storing every version)
    updated_self_model_ver: SelfModelVersion | None = None
    updated_self_model_text: str | None = None
    if parsed_output.self_model is not None and parsed_output.self_model.strip():
        if resolved_self_model_obj is None:
            resolved_self_model_obj = SelfModel(max_words=cfg.max_self_model_words)
            if st_buffer is not None:
                st_buffer.self_model_store = resolved_self_model_obj
        updated_self_model_ver = resolved_self_model_obj.update(
            parsed_output.self_model,
            tick=resolved_tick,
        )
        updated_self_model_text = updated_self_model_ver.text
    elif resolved_self_model_obj is not None and resolved_self_model_obj.text:
        updated_self_model_text = resolved_self_model_obj.text

    if st_buffer is not None:
        st_buffer.last_reflected_episodes = list(episodes)
        st_buffer.last_reflected_beliefs = list(updated_beliefs)
        st_buffer.last_reflected_self_model = updated_self_model_text
        if clear_buffer:
            st_buffer.clear()
        ref_event = ReflectionEvent(
            tick=resolved_tick,
            source="memory",
            trigger=eff_trigger,
            episodes=list(episodes),
            beliefs=list(updated_beliefs),
            self_model=updated_self_model_text,
        )
        st_buffer.reflection_events.append(ref_event)
        st_buffer.events.emit(ref_event)
        if resolved_world is not None and resolved_world.events is not st_buffer.events:
            resolved_world.events.emit(ref_event)
    elif resolved_world is not None:
        ref_event = ReflectionEvent(
            tick=resolved_tick,
            source="memory",
            trigger=eff_trigger,
            episodes=list(episodes),
            beliefs=list(updated_beliefs),
            self_model=updated_self_model_text,
        )
        resolved_world.events.emit(ref_event)

    return ReflectedMemories(
        episodes,
        beliefs=updated_beliefs,
        self_model=updated_self_model_text,
        self_model_version=updated_self_model_ver,
    )


def maybe_reflect(
    buffer: ShortTermBuffer,
    llm: LLMBackend | None = None,
    *,
    store: EpisodeStore | None = None,
    idle_ticks: int | None = None,
    **kwargs: Any,
) -> ReflectedMemories:
    """Run ``reflect`` if ``buffer`` is full or has been idle for ``M`` ticks."""
    res = buffer.maybe_reflect(llm=llm, store=store, idle_ticks=idle_ticks, **kwargs)
    if isinstance(res, ReflectedMemories):
        return res
    return ReflectedMemories(res)


class MemoryManager:
    """Coordinates ``ShortTermBuffer``, ``EpisodeStore``, ``BeliefStore``, ``SelfModel``, and retrieval."""

    def __init__(
        self,
        llm: LLMBackend | None = None,
        *,
        buffer: ShortTermBuffer | None = None,
        store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        self_model: SelfModel | str | None = None,
        world: World | None = None,
        config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
        settings: Settings | None = None,
        db_path: Any = None,
        buffer_size: int | None = None,
        rest_idle_ticks: int | None = None,
        max_beliefs: int | None = None,
    ) -> None:
        cfg = resolve_memory_config(config, settings=settings)
        if buffer_size is not None:
            cfg = cfg.model_copy(update={"buffer_size": int(buffer_size)})
        if rest_idle_ticks is not None:
            cfg = cfg.model_copy(update={"rest_idle_ticks": int(rest_idle_ticks)})
        if max_beliefs is not None:
            cfg = cfg.model_copy(update={"max_beliefs": int(max_beliefs)})
        if db_path is not None:
            cfg = cfg.model_copy(update={"db_path": db_path})

        self.config = cfg
        self.llm = llm
        self.world = world
        self.store = store if store is not None else EpisodeStore(config=cfg)
        self.belief_store: BeliefStore = (
            belief_store if belief_store is not None else self.store.belief_store
        )
        if self.world is not None and self.belief_store.world is None:
            self.belief_store.world = self.world

        if isinstance(self_model, SelfModel):
            self.self_model_store: SelfModel = self_model
        elif isinstance(self_model, str):
            self.self_model_store = self.store.self_model_store
            if self_model.strip() and self_model.strip().lower() != "none":
                self.self_model_store.update(self_model, tick=world.tick if world is not None else 0)
        else:
            self.self_model_store = self.store.self_model_store

        self.buffer = (
            buffer
            if buffer is not None
            else ShortTermBuffer(
                config=cfg,
                world=world,
                llm=llm,
                store=self.store,
                belief_store=self.belief_store,
                self_model=self.self_model_store,
                auto_reflect=(llm is not None),
            )
        )
        if self.buffer.llm is None and llm is not None:
            self.buffer.llm = llm
        if self.buffer.store is None:
            self.buffer.store = self.store
        if self.buffer.belief_store is None:
            self.buffer.belief_store = self.belief_store
        if self.buffer.self_model_store is None:
            self.buffer.self_model_store = self.self_model_store
        if llm is not None:
            self.buffer.auto_reflect = True

    @property
    def short_term(self) -> ShortTermBuffer:
        """Alias for ``buffer``."""
        return self.buffer

    @property
    def beliefs(self) -> list[Belief]:
        """All beliefs currently stored in ``belief_store``."""
        return self.belief_store.all()

    @property
    def self_model(self) -> str:
        """Current self-model text."""
        return self.self_model_store.text

    @property
    def self_model_history(self) -> list[SelfModelVersion]:
        """All stored versions of the self-model."""
        return self.self_model_store.all()

    def record(self, *args: Any, **kwargs: Any) -> BufferEntry:
        """Add an entry to the short-term buffer (auto-reflecting if full)."""
        return self.buffer.add(*args, **kwargs)

    add = record
    append = record

    def record_idle(self, ticks: int = 1, *, tick: int | None = None) -> list[Episode]:
        """Advance idle ticks and trigger reflection if idle for M ticks."""
        return self.buffer.record_idle(ticks, tick=tick, llm=self.llm, store=self.store)

    rest = record_idle
    step_idle = record_idle

    def reflect(self, **kwargs: Any) -> ReflectedMemories:
        """Force reflection on the current buffer and save episodes, beliefs, and self-model."""
        return self.buffer.reflect(
            self.llm,
            store=self.store,
            belief_store=self.belief_store,
            self_model=self.self_model_store,
            **kwargs,
        )

    def decay(self, *args: Any, **kwargs: Any) -> list[Episode]:
        """Decay episode saliences in ``store`` and delete forgotten episodes."""
        return self.store.decay(*args, **kwargs)

    def recall(self, *args: Any, **kwargs: Any) -> Episode | list[Episode] | None:
        """Recall an episode or episodes in ``store``."""
        return self.store.recall(*args, **kwargs)

    def retrieve(
        self,
        context_tags: Sequence[str] | str | Any,
        k: int = DEFAULT_RETRIEVAL_K,
        *,
        current_tick: int | None = None,
        embedding_fn: EmbeddingHook | None = DEFAULT_EMBEDDING_HOOK,
    ) -> list[Episode | Belief]:
        """Retrieve top-``k`` episodes and beliefs scored by recency, importance, and relevance."""
        eff_tick = (
            current_tick
            if current_tick is not None
            else (self.world.tick if self.world is not None else None)
        )
        return retrieve(
            context_tags,
            k,
            episode_store=self.store,
            belief_store=self.belief_store,
            current_tick=eff_tick,
            recency_decay_rate=self.config.recency_decay_rate,
            embedding_fn=embedding_fn,
        )


MemorySystem = MemoryManager


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_REST_IDLE_TICKS",
    "BufferEntry",
    "BufferFullEvent",
    "MemoryManager",
    "MemorySystem",
    "ReflectedMemories",
    "ReflectionEvent",
    "ShortTermBuffer",
    "ShortTermEntry",
    "maybe_reflect",
    "reflect",
]
