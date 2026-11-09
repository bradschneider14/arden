"""Attention gate interface, rule-based gate (T15), condition language, and HeartbeatGate."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from arden.mailbox import is_inside_closed_container
from arden.world.objects import Position
from arden.world.physics import STOP_SPEED_EPSILON
from arden.world.world import (
    EventBus,
    GateDecisionEvent,
    IntentEndedEvent,
    MailArrivedEvent,
    World,
    WorldEvent,
)

if TYPE_CHECKING:
    from arden.controller.controller import Controller, ControllerStatus
    from arden.environment.body import DriveState
    from arden.llm.schemas import Intent
    from arden.perception.feed import Percept, PerceivedObject

# Defaults chosen to over-wake (docs/DESIGN.md Section 10.1 & docs/decisions.md T15)
DEFAULT_HEARTBEAT_INTERVAL: int = 30
DEFAULT_SURPRISE_THRESHOLD: float = 0.25
DEFAULT_LOW_ENERGY_WAKE_THRESHOLD: float = 0.30
DEFAULT_LOW_COMFORT_WAKE_THRESHOLD: float = 0.30
DEFAULT_BOREDOM_WAKE_THRESHOLD: float = 0.50
DEFAULT_GATE_MOVE_THRESHOLD: float = 0.05

WakeConditionKind = Literal[
    "object_moves",
    "object_stops_moving",
    "object_disappears",
    "new_object_in_range",
    "energy_below",
    "comfort_below",
    "unparseable",
]

_RE_OBJECT_MOVES = re.compile(
    r"^(object_[0-9]+)\s+(?:moves?|moved|is\s+moving)$",
    re.IGNORECASE,
)
_RE_OBJECT_STOPS_MOVING = re.compile(
    r"^(object_[0-9]+)\s+(?:stops?\s+moving|stopped\s+moving|stops?)$",
    re.IGNORECASE,
)
_RE_OBJECT_DISAPPEARS = re.compile(
    r"^(object_[0-9]+)\s+(?:disappears?|disappeared|vanishes|vanished|despawned)$",
    re.IGNORECASE,
)
_RE_NEW_OBJECT_IN_RANGE = re.compile(
    r"^(?:new\s+object(?:\s+in\s+range|\s+appears?)?|object\s+appears?)$",
    re.IGNORECASE,
)
_RE_ENERGY_BELOW = re.compile(
    r"^energy\s+(below|under|<|<=)\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))$",
    re.IGNORECASE,
)
_RE_COMFORT_BELOW = re.compile(
    r"^comfort\s+(below|under|<|<=)\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))$",
    re.IGNORECASE,
)

_TRANSIENT_PROP_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "position",
        "distance",
        "direction",
        "level",
        "affordances",
    }
)


class GateConfig(BaseModel):
    """Configuration for the rule-based attention gate under ``[gate]``.

    Defaults are tuned to err toward waking the LLM (docs/DESIGN.md 10.1):
    - ``heartbeat_interval = 30`` ticks (3s at 10 Hz) so the agent never runs on autopilot for long.
    - ``surprise_threshold = 0.25`` so mild prediction errors wake the LLM once T16 supplies scores.
    - ``low_energy_threshold = 0.30`` and ``low_comfort_threshold = 0.30`` so body needs wake early.
    - ``boredom_threshold = 0.50`` so idle stretches wake the LLM to explore or experiment.
    - ``move_threshold = 0.05`` meters for ``<id> moves`` / ``<id> stops moving`` condition checks.
    """

    model_config = ConfigDict(extra="forbid")

    heartbeat_interval: int = Field(default=DEFAULT_HEARTBEAT_INTERVAL, gt=0)
    surprise_threshold: float = Field(default=DEFAULT_SURPRISE_THRESHOLD, ge=0.0, le=1.0)
    low_energy_threshold: float = Field(default=DEFAULT_LOW_ENERGY_WAKE_THRESHOLD, ge=0.0, le=1.0)
    low_comfort_threshold: float = Field(default=DEFAULT_LOW_COMFORT_WAKE_THRESHOLD, ge=0.0, le=1.0)
    boredom_threshold: float = Field(default=DEFAULT_BOREDOM_WAKE_THRESHOLD, ge=0.0, le=1.0)
    move_threshold: float = Field(default=DEFAULT_GATE_MOVE_THRESHOLD, ge=0.0)

    @property
    def interval(self) -> int:
        """Alias for ``heartbeat_interval``."""
        return self.heartbeat_interval

    @property
    def energy_threshold(self) -> float:
        """Alias for ``low_energy_threshold``."""
        return self.low_energy_threshold

    @property
    def comfort_threshold(self) -> float:
        """Alias for ``low_comfort_threshold``."""
        return self.low_comfort_threshold


GateSettings = GateConfig


class WakeCondition(BaseModel):
    """Parsed representation of a single LLM-set ``wake_if`` condition string.

    Supported condition grammar (case-insensitive):
    - ``<id> moves`` (e.g. ``object_7 moves``)
    - ``<id> stops moving`` (e.g. ``object_7 stops moving``)
    - ``<id> disappears`` (e.g. ``object_7 disappears``)
    - ``new object in range``
    - ``energy below <x>`` (with ``<x>`` in ``[0.0, 1.0]``)
    - ``comfort below <x>`` (with ``<x>`` in ``[0.0, 1.0]``)

    Any condition string that does not match this grammar is preserved with
    ``kind="unparseable"``, ``is_valid=False``, and ``raw=<original text>``,
    and causes a wake at the next heartbeat rather than being ignored.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str
    kind: WakeConditionKind
    target_id: str | None = None
    threshold: float | None = None
    inclusive: bool = False
    is_valid: bool = True

    @property
    def text(self) -> str:
        """Original condition string."""
        return self.raw

    @property
    def condition(self) -> str:
        """Alias for ``raw``."""
        return self.raw

    @property
    def object_id(self) -> str | None:
        """Alias for ``target_id``."""
        return self.target_id

    @property
    def target(self) -> str | None:
        """Alias for ``target_id``."""
        return self.target_id

    @property
    def value(self) -> float | None:
        """Alias for ``threshold``."""
        return self.threshold

    @property
    def parsed(self) -> bool:
        """True if the condition matched the supported grammar."""
        return self.is_valid

    @property
    def parseable(self) -> bool:
        """Alias for ``is_valid``."""
        return self.is_valid

    @property
    def is_parseable(self) -> bool:
        """Alias for ``is_valid``."""
        return self.is_valid

    @property
    def is_unparseable(self) -> bool:
        """True when the condition could not be parsed and is kept as raw text."""
        return not self.is_valid


ParsedWakeCondition = WakeCondition


def parse_wake_if(condition: str) -> WakeCondition:
    """Parse a single ``wake_if`` condition string into a ``WakeCondition``.

    Unparseable conditions never raise; they return a ``WakeCondition`` with
    ``kind="unparseable"`` and ``is_valid=False`` while keeping the original text.
    """
    if not isinstance(condition, str):
        raw_str = str(condition)
        return WakeCondition(raw=raw_str, kind="unparseable", is_valid=False)

    normalized = " ".join(condition.strip().split())
    if not normalized:
        return WakeCondition(raw=condition, kind="unparseable", is_valid=False)

    m = _RE_OBJECT_MOVES.fullmatch(normalized)
    if m is not None:
        return WakeCondition(
            raw=condition,
            kind="object_moves",
            target_id=m.group(1).lower(),
            is_valid=True,
        )

    m = _RE_OBJECT_STOPS_MOVING.fullmatch(normalized)
    if m is not None:
        return WakeCondition(
            raw=condition,
            kind="object_stops_moving",
            target_id=m.group(1).lower(),
            is_valid=True,
        )

    m = _RE_OBJECT_DISAPPEARS.fullmatch(normalized)
    if m is not None:
        return WakeCondition(
            raw=condition,
            kind="object_disappears",
            target_id=m.group(1).lower(),
            is_valid=True,
        )

    if _RE_NEW_OBJECT_IN_RANGE.fullmatch(normalized) is not None:
        return WakeCondition(
            raw=condition,
            kind="new_object_in_range",
            is_valid=True,
        )

    m = _RE_ENERGY_BELOW.fullmatch(normalized)
    if m is not None:
        op = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            val = float("nan")
        if math.isfinite(val) and 0.0 <= val <= 1.0:
            return WakeCondition(
                raw=condition,
                kind="energy_below",
                threshold=val,
                inclusive=(op == "<="),
                is_valid=True,
            )

    m = _RE_COMFORT_BELOW.fullmatch(normalized)
    if m is not None:
        op = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            val = float("nan")
        if math.isfinite(val) and 0.0 <= val <= 1.0:
            return WakeCondition(
                raw=condition,
                kind="comfort_below",
                threshold=val,
                inclusive=(op == "<="),
                is_valid=True,
            )

    return WakeCondition(raw=condition, kind="unparseable", is_valid=False)


def parse_wake_conditions(conditions: Iterable[str] | str | None) -> list[WakeCondition]:
    """Parse a sequence of ``wake_if`` condition strings."""
    if conditions is None:
        return []
    if isinstance(conditions, str):
        return [parse_wake_if(conditions)]
    return [parse_wake_if(c) for c in conditions]


class WakeDecision(BaseModel):
    """Decision returned by ``Gate.should_wake(context)``."""

    model_config = ConfigDict(extra="forbid")

    wake: bool = False
    reasons: list[str] = Field(default_factory=list)

    _matched_wake_if: list[str] = PrivateAttr(default_factory=list)
    _unparseable_wake_if: list[str] = PrivateAttr(default_factory=list)
    _emitted_event: GateDecisionEvent | None = PrivateAttr(default=None)
    _emitted_buses: list[Any] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        wake: bool = False,
        reasons: Sequence[str] | str | None = None,
        *,
        matched_wake_if: Sequence[str] | None = None,
        unparseable_wake_if: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if "wake" in kwargs:
            wake = kwargs.pop("wake")
        if "reasons" in kwargs:
            reasons = kwargs.pop("reasons")
        if isinstance(reasons, str):
            reason_list = [reasons]
        elif reasons is None:
            reason_list = []
        else:
            reason_list = [str(r) for r in reasons]
        super().__init__(wake=bool(wake), reasons=reason_list, **kwargs)
        if matched_wake_if is not None:
            self._matched_wake_if = [str(item) for item in matched_wake_if]
        if unparseable_wake_if is not None:
            self._unparseable_wake_if = [str(item) for item in unparseable_wake_if]

    def __bool__(self) -> bool:
        return self.wake

    @property
    def woke(self) -> bool:
        """Alias for ``wake``."""
        return self.wake

    @property
    def continued(self) -> bool:
        """True when ``wake`` is False."""
        return not self.wake

    @property
    def reason(self) -> str | None:
        """Comma-joined wake reasons, or ``None`` when empty."""
        return ", ".join(self.reasons) if self.reasons else None

    @property
    def matched_wake_if(self) -> list[str]:
        """``wake_if`` conditions that contributed to this decision."""
        return list(self._matched_wake_if)

    @property
    def unparseable_wake_if(self) -> list[str]:
        """Unparseable ``wake_if`` conditions kept as text on this decision."""
        return list(self._unparseable_wake_if)

    @property
    def event(self) -> GateDecisionEvent | None:
        """The ``GateDecisionEvent`` emitted for this decision, if any."""
        return self._emitted_event

    def to_dict(self) -> dict[str, Any]:
        return {"wake": self.wake, "reasons": list(self.reasons)}


def _coerce_wake_decision(
    value: WakeDecision | tuple[bool, Sequence[str]] | bool | Mapping[str, Any],
) -> WakeDecision:
    if isinstance(value, WakeDecision):
        return value
    if isinstance(value, bool):
        return WakeDecision(wake=value, reasons=["gate"] if value else [])
    if isinstance(value, tuple) and len(value) == 2:
        wake_flag, raw_reasons = value
        return WakeDecision(wake=bool(wake_flag), reasons=raw_reasons)
    if isinstance(value, Mapping):
        return WakeDecision(
            wake=bool(value.get("wake", False)),
            reasons=value.get("reasons", []),
        )
    raise TypeError(f"Gate.should_wake must return a WakeDecision, got {type(value).__name__}")


@dataclass
class GateContext:
    """Per-tick context passed into ``Gate.should_wake(context)``.

    Supports both attribute access (``context.tick``, ``context.intent_ended``)
    and mapping access (``context["tick"]``, ``context.get("tick")``).
    """

    tick: int = 0
    percept: Percept | None = None
    world: World | None = None
    controller: Controller | None = None
    controller_status: ControllerStatus | None = None
    active_intent: Intent | None = None
    is_start: bool = False
    intent_ended: bool = False
    intent_ended_event: IntentEndedEvent | None = None
    ticks_since_wake: int = 0
    events: list[WorldEvent] = field(default_factory=list)
    surprise: float = 0.0
    novelty: float | bool | None = None
    wake_if: list[str] | None = None
    energy: float | None = None
    comfort: float | None = None
    boredom: float | None = None
    drives: DriveState | Mapping[str, Any] | None = None
    mail_arrived: bool = False
    creator_events: list[WorldEvent] = field(default_factory=list)

    @property
    def start(self) -> bool:
        """Alias for ``is_start``."""
        return self.is_start

    @property
    def at_start(self) -> bool:
        """Alias for ``is_start``."""
        return self.is_start

    @property
    def intent(self) -> Intent | None:
        """Alias for ``active_intent``."""
        return self.active_intent

    @property
    def status(self) -> ControllerStatus | None:
        """Alias for ``controller_status``."""
        return self.controller_status

    @property
    def surprise_score(self) -> float:
        """Alias for ``surprise``."""
        return self.surprise

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "percept": self.percept,
            "world": self.world,
            "controller": self.controller,
            "controller_status": self.controller_status,
            "active_intent": self.active_intent,
            "is_start": self.is_start,
            "start": self.is_start,
            "intent_ended": self.intent_ended,
            "intent_ended_event": self.intent_ended_event,
            "ticks_since_wake": self.ticks_since_wake,
            "events": self.events,
            "surprise": self.surprise,
            "surprise_score": self.surprise,
            "novelty": self.novelty,
            "wake_if": self.wake_if,
            "energy": self.energy,
            "comfort": self.comfort,
            "boredom": self.boredom,
            "drives": self.drives,
            "mail_arrived": self.mail_arrived,
            "creator_events": self.creator_events,
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.to_dict().keys())


@runtime_checkable
class Gate(Protocol):
    """Attention gate interface deciding when the LLM should wake and think.

    The gate decides *when* the agent thinks, never *what* it thinks or does.
    """

    def should_wake(self, context: Any = None, **kwargs: Any) -> WakeDecision:
        """Return a ``WakeDecision(wake=bool, reasons=list[str])`` for the current tick."""
        ...


class HeartbeatGate:
    """Placeholder attention gate for M4 (T12).

    Wakes the LLM only:
    - at start (``"start"``),
    - when the active intent finishes or fails (``"intent_ended"``), and
    - on a periodic heartbeat every ``interval`` ticks (``"heartbeat"``).

    ``RuleGate`` replaces this by default in M6 (T15) behind the same ``Gate`` interface.
    """

    def __init__(
        self,
        interval: int = 50,
        *,
        every: int | None = None,
        every_n_ticks: int | None = None,
        heartbeat_interval: int | None = None,
        period: int | None = None,
        n: int | None = None,
    ) -> None:
        resolved = (
            every
            if every is not None
            else every_n_ticks
            if every_n_ticks is not None
            else heartbeat_interval
            if heartbeat_interval is not None
            else period
            if period is not None
            else n
            if n is not None
            else interval
        )
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
            raise ValueError("heartbeat interval must be a positive integer")
        self.interval: int = int(resolved)
        self._started: bool = False
        self._ticks_since_wake: int = 0

    @property
    def every_n_ticks(self) -> int:
        """Alias for ``interval``."""
        return self.interval

    @property
    def heartbeat_interval(self) -> int:
        """Alias for ``interval``."""
        return self.interval

    @property
    def ticks_since_wake(self) -> int:
        """Number of ticks elapsed since the last wake decision."""
        return self._ticks_since_wake

    def reset(self) -> None:
        """Reset internal start and heartbeat counters."""
        self._started = False
        self._ticks_since_wake = 0

    def should_wake(self, context: Any = None, **kwargs: Any) -> WakeDecision:
        """Evaluate whether to wake the LLM on this tick."""
        tick: int | None = kwargs.get("tick")
        explicit_start: bool | None = kwargs.get("is_start", kwargs.get("start", kwargs.get("at_start")))
        intent_ended_flag: bool = bool(kwargs.get("intent_ended", False))
        explicit_ticks_since_wake: int | None = kwargs.get("ticks_since_wake")

        if isinstance(context, GateContext):
            if tick is None:
                tick = context.tick
            if explicit_start is None:
                explicit_start = context.is_start
            if not intent_ended_flag:
                intent_ended_flag = bool(
                    context.intent_ended
                    or context.intent_ended_event is not None
                    or (context.controller_status is not None and context.controller_status.done)
                    or any(getattr(ev, "type", None) == "intent_ended" for ev in context.events)
                )
        elif isinstance(context, Mapping):
            if tick is None and "tick" in context:
                raw_t = context.get("tick")
                if isinstance(raw_t, int) and not isinstance(raw_t, bool):
                    tick = raw_t
            if explicit_start is None:
                for key in ("is_start", "start", "at_start"):
                    if key in context and context[key] is not None:
                        explicit_start = bool(context[key])
                        break
            if not intent_ended_flag:
                raw_ie = context.get("intent_ended")
                st = context.get("controller_status", context.get("status"))
                ev_list = context.get("events") or ()
                intent_ended_flag = bool(
                    raw_ie
                    or context.get("intent_ended_event") is not None
                    or (st is not None and getattr(st, "done", False))
                    or any(
                        (getattr(e, "type", None) if not isinstance(e, Mapping) else e.get("type"))
                        == "intent_ended"
                        for e in ev_list
                    )
                )
            if explicit_ticks_since_wake is None and "ticks_since_wake" in context:
                raw_tsw = context.get("ticks_since_wake")
                if isinstance(raw_tsw, int) and not isinstance(raw_tsw, bool):
                    explicit_ticks_since_wake = raw_tsw
        elif context is not None:
            if tick is None and hasattr(context, "tick"):
                raw_t = getattr(context, "tick")
                if isinstance(raw_t, int) and not isinstance(raw_t, bool):
                    tick = raw_t
            if explicit_start is None:
                for attr in ("is_start", "start", "at_start"):
                    if hasattr(context, attr):
                        val = getattr(context, attr)
                        if isinstance(val, bool):
                            explicit_start = val
                            break
            if not intent_ended_flag:
                if bool(getattr(context, "intent_ended", False)):
                    intent_ended_flag = True
                elif getattr(context, "intent_ended_event", None) is not None:
                    intent_ended_flag = True
                else:
                    st = getattr(context, "controller_status", getattr(context, "status", None))
                    if st is not None and getattr(st, "done", False):
                        intent_ended_flag = True
                    else:
                        ev_list = getattr(context, "events", None) or ()
                        if any(getattr(e, "type", None) == "intent_ended" for e in ev_list):
                            intent_ended_flag = True

        reasons: list[str] = []

        # 1. Wake at start
        if explicit_start is True:
            is_start = True
        elif explicit_start is False:
            is_start = False if (tick is not None and tick != 0) else (not self._started and tick == 0)
        else:
            is_start = (not self._started) and (tick is None or tick == 0)

        self._started = True
        if is_start:
            reasons.append("start")

        # 2. Wake on intent_ended
        if intent_ended_flag:
            reasons.append("intent_ended")

        # 3. Wake on heartbeat (every N ticks)
        next_ticks_since_wake = (
            explicit_ticks_since_wake
            if explicit_ticks_since_wake is not None
            else (0 if is_start else self._ticks_since_wake + 1)
        )
        is_heartbeat = False
        if not is_start:
            if tick is not None and tick > 0 and (tick % self.interval == 0):
                is_heartbeat = True
            elif next_ticks_since_wake >= self.interval:
                is_heartbeat = True

        if is_heartbeat:
            reasons.append("heartbeat")

        wake = bool(reasons)
        if wake:
            self._ticks_since_wake = 0
        else:
            self._ticks_since_wake = next_ticks_since_wake

        return WakeDecision(wake=wake, reasons=reasons)


def _property_combo_signature(obj: Any) -> str:
    """Compute a deterministic signature of an object's perceptible property combination.

    Excludes identity and spatial location (``id``, ``position``, ``distance``,
    ``direction``, ``level``, ``affordances``) so moving or seeing a second
    identical object at another location does not register as a novel property mix.
    """
    if isinstance(obj, Mapping):
        raw_props = dict(obj.get("properties", obj))
        shape = obj.get("shape", raw_props.get("shape"))
        height = obj.get("height", raw_props.get("height", 0.0))
        volume = obj.get("volume", raw_props.get("volume", 0.0))
        weight = obj.get("weight", raw_props.get("weight", 0.0))
        rigidity = obj.get("rigidity", raw_props.get("rigidity", 0.5))
        binding = obj.get("binding", raw_props.get("binding", False))
        hollow = obj.get("hollow", raw_props.get("hollow", False))
        fragile = obj.get("fragile", raw_props.get("fragile", False))
        fixed = obj.get("fixed", raw_props.get("fixed", False))
        color = obj.get("color", raw_props.get("color"))
        state = obj.get("state", raw_props.get("state", {}))
    else:
        raw_props = dict(getattr(obj, "properties", {}) or {})
        shape = getattr(obj, "shape", raw_props.get("shape"))
        height = getattr(obj, "height", raw_props.get("height", 0.0))
        volume = getattr(obj, "volume", raw_props.get("volume", 0.0))
        weight = getattr(obj, "weight", raw_props.get("weight", 0.0))
        rigidity = getattr(obj, "rigidity", raw_props.get("rigidity", 0.5))
        binding = getattr(obj, "binding", raw_props.get("binding", False))
        hollow = getattr(obj, "hollow", raw_props.get("hollow", False))
        fragile = getattr(obj, "fragile", raw_props.get("fragile", False))
        fixed = getattr(obj, "fixed", raw_props.get("fixed", False))
        color = getattr(obj, "color", raw_props.get("color"))
        state = getattr(obj, "state", raw_props.get("state", {}))

    extra: dict[str, Any] = {}
    for k, v in raw_props.items():
        if k not in _TRANSIENT_PROP_KEYS and k not in {
            "shape",
            "height",
            "volume",
            "weight",
            "rigidity",
            "binding",
            "hollow",
            "fragile",
            "fixed",
            "color",
            "state",
        }:
            extra[k] = v

    payload: dict[str, Any] = {
        "shape": shape,
        "height": round(float(height), 4) if height is not None else None,
        "volume": round(float(volume), 4) if volume is not None else None,
        "weight": round(float(weight), 4) if weight is not None else None,
        "rigidity": round(float(rigidity), 4) if rigidity is not None else None,
        "binding": bool(binding),
        "hollow": bool(hollow),
        "fragile": bool(fragile),
        "fixed": bool(fixed),
        "color": color,
        "state": dict(state) if isinstance(state, Mapping) else {},
    }
    if extra:
        payload["extra"] = extra
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _extract_event_type(ev: Any) -> str | None:
    if isinstance(ev, Mapping):
        t = ev.get("type")
        return str(t) if t is not None else None
    t = getattr(ev, "type", None)
    return str(t) if t is not None else None


def _extract_event_object_id(ev: Any) -> str | None:
    if isinstance(ev, Mapping):
        oid = ev.get("object_id", ev.get("id"))
        return str(oid) if oid is not None else None
    oid = getattr(ev, "object_id", None)
    return str(oid) if oid is not None else None


def _extract_event_source(ev: Any) -> str | None:
    if isinstance(ev, Mapping):
        src = ev.get("source")
        return str(src) if src is not None else None
    src = getattr(ev, "source", None)
    return str(src) if src is not None else None


def _extract_position(obj: Any) -> Position | None:
    if obj is None:
        return None
    if isinstance(obj, Position):
        return obj
    if isinstance(obj, Mapping):
        if "x" in obj and "y" in obj:
            return Position(x=float(obj["x"]), y=float(obj["y"]))
        pos = obj.get("position")
        if pos is not None:
            return _extract_position(pos)
        return None
    if hasattr(obj, "x") and hasattr(obj, "y"):
        return Position(x=float(obj.x), y=float(obj.y))
    if hasattr(obj, "position"):
        return _extract_position(getattr(obj, "position"))
    return None


class RuleGate:
    """Phase 1 rule-based attention gate over all wake triggers (T15).

    Decides only *when* the agent thinks, never *what* it does, and errs toward waking.

    Wake triggers:
    - ``"start"``: Initial simulation tick (``is_start=True``).
    - ``"surprise"``: Surprise score ``>= surprise_threshold`` (and ``> 0``; 0 until T16).
    - ``"novelty"``: Never-seen object ID, perceptible property combination, or affordance.
    - ``"intent_ended"``: Active controller intent succeeded or failed.
    - ``"wake_if"``: Agent-set ``wake_if`` condition matched (``<id> moves``,
      ``<id> stops moving``, ``<id> disappears``, ``new object in range``,
      ``energy below <x>``, ``comfort below <x>``), or an unparseable ``wake_if``
      condition causing a wake at the next heartbeat.
    - ``"low_energy"``, ``"low_comfort"``, ``"boredom"``: Drive thresholds crossed.
    - ``"creator_spawn"``, ``"mail_arrived"``: Creator spawned an object nearby or mail arrived.
    - ``"heartbeat"``: Periodic timer every ``heartbeat_interval`` ticks.

    Every decision (wake or continue) is emitted as a ``GateDecisionEvent``
    (``type="gate_decision"``) with its reasons.
    """

    def __init__(
        self,
        config: GateConfig | Mapping[str, Any] | None = None,
        *,
        settings: Any = None,
        world: World | None = None,
        events: EventBus | None = None,
        interval: int | None = None,
        heartbeat_interval: int | None = None,
        every: int | None = None,
        every_n_ticks: int | None = None,
        surprise_threshold: float | None = None,
        low_energy_threshold: float | None = None,
        energy_threshold: float | None = None,
        low_comfort_threshold: float | None = None,
        comfort_threshold: float | None = None,
        boredom_threshold: float | None = None,
        move_threshold: float | None = None,
        wake_if: Iterable[str] | None = None,
        seen_objects: Iterable[str] | None = None,
        seen_property_combos: Iterable[str] | None = None,
        seen_affordances: Iterable[str] | None = None,
        moving_objects: Iterable[str] | None = None,
        baseline: Percept | None = None,
    ) -> None:
        base_cfg: GateConfig
        if isinstance(config, GateConfig):
            base_cfg = config.model_copy(deep=True)
        elif isinstance(config, Mapping):
            if "gate" in config and isinstance(config["gate"], GateConfig | Mapping):
                sub = config["gate"]
                base_cfg = (
                    sub.model_copy(deep=True)
                    if isinstance(sub, GateConfig)
                    else GateConfig.model_validate(dict(sub))
                )
            else:
                base_cfg = GateConfig.model_validate(dict(config))
        elif config is not None and hasattr(config, "gate"):
            sub = getattr(config, "gate")
            base_cfg = (
                sub.model_copy(deep=True)
                if isinstance(sub, GateConfig)
                else GateConfig.model_validate(dict(sub))
            )
        elif settings is not None and hasattr(settings, "gate"):
            sub = getattr(settings, "gate")
            base_cfg = (
                sub.model_copy(deep=True)
                if isinstance(sub, GateConfig)
                else GateConfig.model_validate(dict(sub))
            )
        elif world is not None and getattr(world, "config", None) is not None and hasattr(world.config, "gate"):
            sub = getattr(world.config, "gate")
            base_cfg = (
                sub.model_copy(deep=True)
                if isinstance(sub, GateConfig)
                else GateConfig.model_validate(dict(sub))
            )
        else:
            base_cfg = GateConfig()

        resolved_hb = (
            heartbeat_interval
            if heartbeat_interval is not None
            else interval
            if interval is not None
            else every
            if every is not None
            else every_n_ticks
        )
        if resolved_hb is not None:
            if isinstance(resolved_hb, bool) or not isinstance(resolved_hb, int) or resolved_hb <= 0:
                raise ValueError("heartbeat interval must be a positive integer")
            base_cfg.heartbeat_interval = int(resolved_hb)

        if surprise_threshold is not None:
            base_cfg.surprise_threshold = float(surprise_threshold)
        resolved_energy_thresh = (
            low_energy_threshold if low_energy_threshold is not None else energy_threshold
        )
        if resolved_energy_thresh is not None:
            base_cfg.low_energy_threshold = float(resolved_energy_thresh)
        resolved_comfort_thresh = (
            low_comfort_threshold if low_comfort_threshold is not None else comfort_threshold
        )
        if resolved_comfort_thresh is not None:
            base_cfg.low_comfort_threshold = float(resolved_comfort_thresh)
        if boredom_threshold is not None:
            base_cfg.boredom_threshold = float(boredom_threshold)
        if move_threshold is not None:
            base_cfg.move_threshold = float(move_threshold)

        # Validate updated config
        self.config: GateConfig = GateConfig.model_validate(base_cfg.model_dump())
        self.world: World | None = world
        self.events: EventBus = events if events is not None else EventBus()
        self.decisions: list[GateDecisionEvent] = []

        self.seen_objects: set[str] = {str(o) for o in (seen_objects or ())}
        self.seen_property_combos: set[str] = {str(p) for p in (seen_property_combos or ())}
        self.seen_affordances: set[str] = {str(a) for a in (seen_affordances or ())}

        self._started: bool = False
        self._ticks_since_wake: int = 0
        self._last_visible_objects: set[str] = set(self.seen_objects)
        self._last_object_positions: dict[str, Position] = {}
        self._moving_objects: set[str] = {str(o) for o in (moving_objects or ())}
        self._seen_mail_count: int = 0

        self._active_wake_if_raw: list[str] = []
        self._parsed_wake_if: list[WakeCondition] = []
        self._pending_unparseable_wake_if: list[str] = []
        self.unparseable_wake_if: list[str] = []

        if wake_if is not None:
            self.set_wake_if(wake_if)
        if baseline is not None:
            self.seed_seen(baseline)

    @property
    def interval(self) -> int:
        """Heartbeat interval in ticks."""
        return self.config.heartbeat_interval

    @property
    def heartbeat_interval(self) -> int:
        """Heartbeat interval in ticks."""
        return self.config.heartbeat_interval

    @property
    def every_n_ticks(self) -> int:
        """Alias for ``heartbeat_interval``."""
        return self.config.heartbeat_interval

    @property
    def surprise_threshold(self) -> float:
        """Surprise score threshold."""
        return self.config.surprise_threshold

    @property
    def low_energy_threshold(self) -> float:
        """Low-energy drive wake threshold."""
        return self.config.low_energy_threshold

    @property
    def energy_threshold(self) -> float:
        """Alias for ``low_energy_threshold``."""
        return self.config.low_energy_threshold

    @property
    def low_comfort_threshold(self) -> float:
        """Low-comfort drive wake threshold."""
        return self.config.low_comfort_threshold

    @property
    def comfort_threshold(self) -> float:
        """Alias for ``low_comfort_threshold``."""
        return self.config.low_comfort_threshold

    @property
    def boredom_threshold(self) -> float:
        """Boredom drive wake threshold."""
        return self.config.boredom_threshold

    @property
    def move_threshold(self) -> float:
        """Minimum position delta for ``<id> moves`` condition checks."""
        return self.config.move_threshold

    @property
    def ticks_since_wake(self) -> int:
        """Number of ticks elapsed since the last wake decision."""
        return self._ticks_since_wake

    @property
    def wake_if_conditions(self) -> list[WakeCondition]:
        """Currently active parsed (and unparseable) ``WakeCondition`` items."""
        return list(self._parsed_wake_if)

    @property
    def active_wake_if(self) -> list[str]:
        """Currently active raw ``wake_if`` strings."""
        return list(self._active_wake_if_raw)

    @property
    def decision_log(self) -> list[GateDecisionEvent]:
        """Alias for ``decisions``."""
        return self.decisions

    def subscribe(
        self,
        handler_or_type: Callable[[Any], None] | str,
        handler: Callable[[Any], None] | None = None,
        *,
        event_type: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to ``gate_decision`` events on this gate's event bus."""
        return self.events.subscribe(handler_or_type, handler, event_type=event_type)  # type: ignore[arg-type]

    def set_wake_if(self, conditions: Iterable[str] | str | None) -> list[WakeCondition]:
        """Set or replace the active LLM ``wake_if`` conditions on this gate."""
        if conditions is None:
            raw_list: list[str] = []
        elif isinstance(conditions, str):
            raw_list = [conditions]
        else:
            raw_list = [str(c) for c in conditions]

        parsed = parse_wake_conditions(raw_list)
        self._active_wake_if_raw = raw_list
        self._parsed_wake_if = parsed
        unparseable = [c.raw for c in parsed if not c.is_valid]
        self.unparseable_wake_if = list(unparseable)
        if unparseable:
            for item in unparseable:
                if item not in self._pending_unparseable_wake_if:
                    self._pending_unparseable_wake_if.append(item)
        else:
            self._pending_unparseable_wake_if.clear()
        return parsed

    def seed_seen(self, percept: Percept | None = None, *, world: World | None = None) -> None:
        """Seed baseline objects, property combinations, affordances, and positions without waking."""
        if percept is not None:
            for obj in getattr(percept, "objects", ()) or ():
                oid = getattr(obj, "id", None)
                if isinstance(oid, str):
                    self.seen_objects.add(oid)
                    self._last_visible_objects.add(oid)
                    pos = _extract_position(obj)
                    if pos is not None:
                        self._last_object_positions[oid] = pos.model_copy()
                self.seen_property_combos.add(_property_combo_signature(obj))
                for aff in getattr(obj, "affordances", ()) or ():
                    self.seen_affordances.add(str(aff))
            aff_map = getattr(percept, "affordances", None)
            if isinstance(aff_map, Mapping):
                for actions in aff_map.values():
                    for aff in actions or ():
                        self.seen_affordances.add(str(aff))
        if world is not None:
            for oid, wobj in world.objects.items():
                if not is_inside_closed_container(world, wobj):
                    self.seen_objects.add(oid)
                    self._last_object_positions[oid] = wobj.position.model_copy()
            mailbox = getattr(world, "mailbox", None)
            if mailbox is not None:
                arrivals = getattr(mailbox, "arrivals", None)
                if isinstance(arrivals, Sequence):
                    self._seen_mail_count = len(arrivals)

    observe_baseline = seed_seen

    def reset(self) -> None:
        """Reset all gate state, counters, and seen sets."""
        self._started = False
        self._ticks_since_wake = 0
        self.seen_objects.clear()
        self.seen_property_combos.clear()
        self.seen_affordances.clear()
        self._last_visible_objects.clear()
        self._last_object_positions.clear()
        self._moving_objects.clear()
        self._active_wake_if_raw.clear()
        self._parsed_wake_if.clear()
        self._pending_unparseable_wake_if.clear()
        self.unparseable_wake_if.clear()
        self._seen_mail_count = 0

    def _check_and_record_novelty(
        self,
        percept: Any,
        *,
        seed_only: bool = False,
    ) -> bool:
        """Return True if ``percept`` contains a never-seen object, property mix, or affordance."""
        if percept is None:
            return False

        is_novel = False
        is_delta = bool(getattr(percept, "is_delta", False))
        objects = list(getattr(percept, "objects", ()) or ())
        appeared = list(getattr(percept, "appeared", ()) or ())
        changed = list(getattr(percept, "changed", ()) or ())

        # 1. Check object IDs and property combinations
        if is_delta and not self.seen_objects and not self.seen_property_combos and not seed_only:
            # For an unseeded delta percept, `appeared` holds newly seen objects and
            # `changed` holds modified property combinations, while `moved` holds already-known objects.
            candidate_new_id_objs = appeared
            candidate_new_combo_objs = appeared + changed
        else:
            candidate_new_id_objs = objects + [o for o in appeared if o not in objects]
            candidate_new_combo_objs = candidate_new_id_objs

        for obj in candidate_new_id_objs:
            oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
            if isinstance(oid, str):
                if oid not in self.seen_objects:
                    if not seed_only:
                        is_novel = True
                    self.seen_objects.add(oid)

        for obj in candidate_new_combo_objs:
            sig = _property_combo_signature(obj)
            if sig not in self.seen_property_combos:
                if not seed_only:
                    is_novel = True
                self.seen_property_combos.add(sig)

        # Ensure all objects in `objects` have their IDs and combos recorded
        for obj in objects:
            oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
            if isinstance(oid, str):
                self.seen_objects.add(oid)
            self.seen_property_combos.add(_property_combo_signature(obj))

        # 2. Check affordances
        observed_affordances: list[str] = []
        aff_map = getattr(percept, "affordances", None)
        if isinstance(aff_map, Mapping):
            for actions in aff_map.values():
                for action in actions or ():
                    observed_affordances.append(str(action))
        for obj in objects + appeared:
            obj_affs = (
                obj.get("affordances")
                if isinstance(obj, Mapping)
                else getattr(obj, "affordances", None)
            )
            for action in obj_affs or ():
                observed_affordances.append(str(action))

        for action in observed_affordances:
            if action not in self.seen_affordances:
                if not seed_only:
                    is_novel = True
                self.seen_affordances.add(action)

        return is_novel

    def _did_object_move(
        self,
        target_id: str,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
    ) -> bool:
        if percept is not None:
            for m_obj in getattr(percept, "moved", ()) or ():
                oid = m_obj.get("id") if isinstance(m_obj, Mapping) else getattr(m_obj, "id", None)
                if oid == target_id:
                    return True
            for ev in getattr(percept, "events", ()) or ():
                if _extract_event_object_id(ev) == target_id and _extract_event_type(ev) in (
                    "moved",
                    "object_moved",
                ):
                    return True
            for obj in getattr(percept, "objects", ()) or ():
                oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
                if oid == target_id and target_id in self._last_object_positions:
                    curr_pos = _extract_position(obj)
                    prev_pos = self._last_object_positions[target_id]
                    if curr_pos is not None:
                        dist = math.hypot(curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y)
                        thresh = self.move_threshold if self.move_threshold > 0.0 else 1e-6
                        if dist > thresh:
                            return True

        for ev in events:
            if _extract_event_object_id(ev) == target_id and _extract_event_type(ev) in (
                "moved",
                "object_moved",
            ):
                return True

        if world is not None and target_id in world.objects:
            wobj = world.objects[target_id]
            if target_id in self._last_object_positions:
                prev_pos = self._last_object_positions[target_id]
                dist = math.hypot(wobj.position.x - prev_pos.x, wobj.position.y - prev_pos.y)
                thresh = self.move_threshold if self.move_threshold > 0.0 else 1e-6
                if dist > thresh:
                    return True
            speed = math.hypot(wobj.velocity.x, wobj.velocity.y)
            if speed > STOP_SPEED_EPSILON:
                return True

        return False

    def _did_object_stop_moving(
        self,
        target_id: str,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
    ) -> bool:
        if self._did_object_move(target_id, percept=percept, world=world, events=events):
            return False

        if world is not None and target_id in world.objects:
            wobj = world.objects[target_id]
            if math.hypot(wobj.velocity.x, wobj.velocity.y) > STOP_SPEED_EPSILON:
                return False

        # If previously tracked as moving, and now not moving, it just stopped
        if target_id in self._moving_objects:
            return True

        # Check explicit stop events
        for ev in events:
            if _extract_event_object_id(ev) == target_id and _extract_event_type(ev) in (
                "stopped",
                "object_stopped",
            ):
                return True

        # Single-tick evaluation when the gate has no prior stationary record for target_id
        if target_id not in self._last_object_positions:
            if percept is not None:
                for obj in getattr(percept, "objects", ()) or ():
                    oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
                    if oid == target_id:
                        return True
            if world is not None and target_id in world.objects:
                return True

        return False

    def _did_object_disappear(
        self,
        target_id: str,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
    ) -> bool:
        if percept is not None:
            if target_id in (getattr(percept, "disappeared", ()) or ()):
                return True
            for ev in getattr(percept, "events", ()) or ():
                if _extract_event_object_id(ev) == target_id and _extract_event_type(ev) in (
                    "disappeared",
                    "object_despawned",
                ):
                    return True

        for ev in events:
            if _extract_event_object_id(ev) == target_id and _extract_event_type(ev) in (
                "disappeared",
                "object_despawned",
            ):
                return True

        if percept is not None and not getattr(percept, "is_delta", False):
            visible_ids = {
                (o.get("id") if isinstance(o, Mapping) else getattr(o, "id", None))
                for o in (getattr(percept, "objects", ()) or ())
            }
            if target_id in self._last_visible_objects and target_id not in visible_ids:
                return True
            if not self._last_visible_objects and target_id not in visible_ids:
                return True
        elif percept is None and world is not None:
            if target_id not in world.objects:
                return True

        return False

    def _did_new_object_appear_in_range(
        self,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
        previously_seen_objects: set[str],
    ) -> bool:
        if percept is not None:
            if list(getattr(percept, "appeared", ()) or ()):
                return True
            for ev in getattr(percept, "events", ()) or ():
                if _extract_event_type(ev) in ("appeared", "object_spawned"):
                    return True
            if not getattr(percept, "is_delta", False):
                baseline_ids = self._last_visible_objects or previously_seen_objects
                for obj in getattr(percept, "objects", ()) or ():
                    oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
                    if isinstance(oid, str) and oid not in baseline_ids:
                        return True

        for ev in events:
            if _extract_event_type(ev) in ("appeared", "object_spawned"):
                if self._is_event_nearby(ev, percept=percept, world=world):
                    return True

        return False

    def _evaluate_wake_condition(
        self,
        condition: WakeCondition,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
        energy: float | None,
        comfort: float | None,
        previously_seen_objects: set[str],
    ) -> bool:
        if not condition.is_valid:
            return False

        if condition.kind == "object_moves" and condition.target_id is not None:
            return self._did_object_move(
                condition.target_id,
                percept=percept,
                world=world,
                events=events,
            )

        if condition.kind == "object_stops_moving" and condition.target_id is not None:
            return self._did_object_stop_moving(
                condition.target_id,
                percept=percept,
                world=world,
                events=events,
            )

        if condition.kind == "object_disappears" and condition.target_id is not None:
            return self._did_object_disappear(
                condition.target_id,
                percept=percept,
                world=world,
                events=events,
            )

        if condition.kind == "new_object_in_range":
            return self._did_new_object_appear_in_range(
                percept=percept,
                world=world,
                events=events,
                previously_seen_objects=previously_seen_objects,
            )

        if condition.kind == "energy_below" and condition.threshold is not None:
            if energy is None:
                return False
            return (energy <= condition.threshold) if condition.inclusive else (energy < condition.threshold)

        if condition.kind == "comfort_below" and condition.threshold is not None:
            if comfort is None:
                return False
            return (
                (comfort <= condition.threshold)
                if condition.inclusive
                else (comfort < condition.threshold)
            )

        return False

    def _effective_sensing_radius(self, world: World | None) -> float:
        base_radius = 10.0
        if world is not None:
            w_sr = getattr(world, "sensing_radius", None)
            if isinstance(w_sr, int | float) and not isinstance(w_sr, bool):
                base_radius = float(w_sr)
            else:
                w_cfg = getattr(world, "config", None)
                if w_cfg is not None:
                    p_cfg = getattr(w_cfg, "perception", None)
                    if p_cfg is not None and hasattr(p_cfg, "sensing_radius"):
                        base_radius = float(p_cfg.sensing_radius)
                    w_sub = getattr(w_cfg, "world", None)
                    if w_sub is not None and getattr(w_sub, "sensing_radius", None) is not None:
                        base_radius = float(w_sub.sensing_radius)
            vf = float(getattr(world, "visibility_factor", 1.0))
            mods = getattr(world, "body_modifiers", None)
            scale = float(getattr(mods, "sensing_radius_scale", 1.0)) if mods is not None else 1.0
            return base_radius * vf * scale
        return base_radius

    def _is_event_nearby(
        self,
        ev: Any,
        *,
        percept: Any,
        world: World | None,
    ) -> bool:
        ev_pos = _extract_position(ev)
        if ev_pos is None and world is not None:
            oid = _extract_event_object_id(ev)
            if oid is not None and oid in world.objects:
                ev_pos = world.objects[oid].position

        body_pos: Position | None = None
        if percept is not None and getattr(percept, "body", None) is not None:
            body_pos = _extract_position(percept.body)
        if body_pos is None and world is not None:
            body_pos = world.body.position

        if ev_pos is None or body_pos is None:
            return True

        radius = self._effective_sensing_radius(world)
        dist = math.hypot(ev_pos.x - body_pos.x, ev_pos.y - body_pos.y)
        return dist <= radius + 1e-9

    def _check_creator_events(
        self,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
        explicit_mail_arrived: bool,
    ) -> tuple[bool, bool]:
        """Return ``(creator_spawn_nearby, mail_arrived)``."""
        spawn_nearby = False
        mail_arrived = bool(explicit_mail_arrived)

        for ev in events:
            ev_type = _extract_event_type(ev)
            ev_source = _extract_event_source(ev)

            if ev_type in (
                "mail_arrived",
                "mail",
                "letter_arrived",
                "mailbox_message",
                "creator_mail",
                "creator_message",
            ):
                mail_arrived = True
                continue

            if ev_type == "object_spawned" and (ev_source in (None, "creator")):
                obj_dict = (
                    ev.get("object")
                    if isinstance(ev, Mapping)
                    else getattr(ev, "object", None)
                )
                state_dict = (
                    obj_dict.get("state", {})
                    if isinstance(obj_dict, Mapping)
                    else {}
                )
                if isinstance(state_dict, Mapping) and any(
                    k in state_dict for k in ("mail", "letter", "message", "messages", "inbox", "unread")
                ):
                    mail_arrived = True
                if self._is_event_nearby(ev, percept=percept, world=world):
                    spawn_nearby = True

            elif ev_type == "object_changed" and ev_source in (None, "creator", "world"):
                changes = (
                    ev.get("changes")
                    if isinstance(ev, Mapping)
                    else getattr(ev, "changes", None)
                )
                if isinstance(changes, Mapping):
                    state_changes = changes.get("state")
                    if any(
                        k in changes for k in ("mail", "letter", "message", "messages", "inbox", "unread")
                    ) or (
                        isinstance(state_changes, Mapping)
                        and any(
                            k in state_changes
                            for k in ("mail", "letter", "message", "messages", "inbox", "unread")
                        )
                    ):
                        mail_arrived = True

        if world is not None:
            mailbox = getattr(world, "mailbox", None)
            if mailbox is not None:
                arrivals = getattr(
                    mailbox,
                    "arrivals",
                    getattr(mailbox, "inbox", getattr(mailbox, "messages", getattr(mailbox, "mail", None))),
                )
                if isinstance(arrivals, Sequence) and len(arrivals) > self._seen_mail_count:
                    mail_arrived = True
                    self._seen_mail_count = len(arrivals)

        return spawn_nearby, mail_arrived

    def _update_tracking_state(
        self,
        *,
        percept: Any,
        world: World | None,
        events: Sequence[Any],
    ) -> None:
        """Record object positions, movement status, and visible IDs for the next tick."""
        moved_this_tick: set[str] = set()

        if percept is not None:
            for m_obj in getattr(percept, "moved", ()) or ():
                oid = m_obj.get("id") if isinstance(m_obj, Mapping) else getattr(m_obj, "id", None)
                if isinstance(oid, str):
                    moved_this_tick.add(oid)
            for ev in getattr(percept, "events", ()) or ():
                oid = _extract_event_object_id(ev)
                if oid is not None and _extract_event_type(ev) in ("moved", "object_moved"):
                    moved_this_tick.add(oid)

        for ev in events:
            oid = _extract_event_object_id(ev)
            if oid is not None and _extract_event_type(ev) in ("moved", "object_moved"):
                moved_this_tick.add(oid)

        if percept is not None:
            is_delta = bool(getattr(percept, "is_delta", False))
            current_vis: set[str] = set() if not is_delta else set(self._last_visible_objects)
            for d_id in getattr(percept, "disappeared", ()) or ():
                current_vis.discard(str(d_id))
                self._moving_objects.discard(str(d_id))
                self._last_object_positions.pop(str(d_id), None)

            for obj in getattr(percept, "objects", ()) or ():
                oid = obj.get("id") if isinstance(obj, Mapping) else getattr(obj, "id", None)
                if not isinstance(oid, str):
                    continue
                current_vis.add(oid)
                pos = _extract_position(obj)
                if pos is not None:
                    if oid in self._last_object_positions:
                        prev = self._last_object_positions[oid]
                        dist = math.hypot(pos.x - prev.x, pos.y - prev.y)
                        thresh = self.move_threshold if self.move_threshold > 0.0 else 1e-6
                        if dist > thresh:
                            moved_this_tick.add(oid)
                    self._last_object_positions[oid] = pos.model_copy()

            self._last_visible_objects = current_vis

        if world is not None:
            for oid, wobj in world.objects.items():
                if is_inside_closed_container(world, wobj):
                    continue
                if oid in self._last_object_positions:
                    prev = self._last_object_positions[oid]
                    dist = math.hypot(wobj.position.x - prev.x, wobj.position.y - prev.y)
                    thresh = self.move_threshold if self.move_threshold > 0.0 else 1e-6
                    if dist > thresh:
                        moved_this_tick.add(oid)
                if math.hypot(wobj.velocity.x, wobj.velocity.y) > STOP_SPEED_EPSILON:
                    moved_this_tick.add(oid)
                self._last_object_positions[oid] = wobj.position.model_copy()

        # Any object that moved this tick is marked as moving; any previously moving object
        # that did not move this tick has now stopped.
        self._moving_objects = moved_this_tick

    def should_wake(self, context: Any = None, **kwargs: Any) -> WakeDecision:
        """Evaluate all rule-based wake triggers for the current tick and emit ``gate_decision``."""
        tick: int | None = kwargs.get("tick")
        percept: Any = kwargs.get("percept")
        world: World | None = kwargs.get("world")
        controller: Any = kwargs.get("controller")
        controller_status: Any = kwargs.get("controller_status", kwargs.get("status"))
        active_intent: Any = kwargs.get("active_intent", kwargs.get("intent"))
        explicit_start: bool | None = kwargs.get(
            "is_start",
            kwargs.get("start", kwargs.get("at_start")),
        )
        intent_ended_flag: bool = bool(kwargs.get("intent_ended", False))
        explicit_ticks_since_wake: int | None = kwargs.get("ticks_since_wake")
        raw_events: list[Any] = list(kwargs.get("events") or ())
        raw_creator_events: list[Any] = list(kwargs.get("creator_events") or ())
        explicit_surprise: float | None = kwargs.get("surprise", kwargs.get("surprise_score"))
        explicit_novelty: Any = kwargs.get("novelty")
        explicit_wake_if: Any = kwargs.get("wake_if")
        explicit_energy: float | None = kwargs.get("energy")
        explicit_comfort: float | None = kwargs.get("comfort")
        explicit_boredom: float | None = kwargs.get("boredom")
        explicit_drives: Any = kwargs.get("drives")
        explicit_mail_arrived: bool = bool(kwargs.get("mail_arrived", False))

        if isinstance(context, GateContext):
            if tick is None:
                tick = context.tick
            if percept is None:
                percept = context.percept
            if world is None:
                world = context.world
            if controller is None:
                controller = context.controller
            if controller_status is None:
                controller_status = context.controller_status
            if active_intent is None:
                active_intent = context.active_intent
            if explicit_start is None:
                explicit_start = context.is_start
            if not intent_ended_flag:
                intent_ended_flag = bool(
                    context.intent_ended
                    or context.intent_ended_event is not None
                    or (context.controller_status is not None and context.controller_status.done)
                    or any(_extract_event_type(ev) == "intent_ended" for ev in context.events)
                )
            if explicit_ticks_since_wake is None and context.ticks_since_wake > 0:
                explicit_ticks_since_wake = context.ticks_since_wake
            if context.events:
                raw_events.extend(context.events)
            if context.creator_events:
                raw_creator_events.extend(context.creator_events)
            if explicit_surprise is None:
                explicit_surprise = context.surprise
            if explicit_novelty is None:
                explicit_novelty = context.novelty
            if explicit_wake_if is None:
                explicit_wake_if = context.wake_if
            if explicit_energy is None:
                explicit_energy = context.energy
            if explicit_comfort is None:
                explicit_comfort = context.comfort
            if explicit_boredom is None:
                explicit_boredom = context.boredom
            if explicit_drives is None:
                explicit_drives = context.drives
            if not explicit_mail_arrived:
                explicit_mail_arrived = context.mail_arrived
        elif isinstance(context, Mapping):
            if tick is None and "tick" in context:
                raw_t = context.get("tick")
                if isinstance(raw_t, int) and not isinstance(raw_t, bool):
                    tick = raw_t
            if percept is None:
                percept = context.get("percept")
            if world is None:
                world = context.get("world")
            if controller is None:
                controller = context.get("controller")
            if controller_status is None:
                controller_status = context.get("controller_status", context.get("status"))
            if active_intent is None:
                active_intent = context.get("active_intent", context.get("intent"))
            if explicit_start is None:
                for key in ("is_start", "start", "at_start"):
                    if key in context and context[key] is not None:
                        explicit_start = bool(context[key])
                        break
            if not intent_ended_flag:
                ev_list = context.get("events") or ()
                intent_ended_flag = bool(
                    context.get("intent_ended")
                    or context.get("intent_ended_event") is not None
                    or (controller_status is not None and getattr(controller_status, "done", False))
                    or any(_extract_event_type(e) == "intent_ended" for e in ev_list)
                )
            if explicit_ticks_since_wake is None and "ticks_since_wake" in context:
                raw_tsw = context.get("ticks_since_wake")
                if isinstance(raw_tsw, int) and not isinstance(raw_tsw, bool):
                    explicit_ticks_since_wake = raw_tsw
            if context.get("events"):
                raw_events.extend(context["events"])
            if context.get("creator_events"):
                raw_creator_events.extend(context["creator_events"])
            if explicit_surprise is None:
                explicit_surprise = context.get("surprise", context.get("surprise_score"))
            if explicit_novelty is None:
                explicit_novelty = context.get("novelty")
            if explicit_wake_if is None:
                explicit_wake_if = context.get("wake_if")
            if explicit_energy is None:
                explicit_energy = context.get("energy")
            if explicit_comfort is None:
                explicit_comfort = context.get("comfort")
            if explicit_boredom is None:
                explicit_boredom = context.get("boredom")
            if explicit_drives is None:
                explicit_drives = context.get("drives")
            if not explicit_mail_arrived:
                explicit_mail_arrived = bool(context.get("mail_arrived", False))
        elif context is not None:
            if tick is None and hasattr(context, "tick"):
                raw_t = getattr(context, "tick")
                if isinstance(raw_t, int) and not isinstance(raw_t, bool):
                    tick = raw_t
            if percept is None and hasattr(context, "percept"):
                percept = getattr(context, "percept")
            if world is None and hasattr(context, "world"):
                world = getattr(context, "world")
            if controller is None and hasattr(context, "controller"):
                controller = getattr(context, "controller")
            if controller_status is None:
                controller_status = getattr(
                    context,
                    "controller_status",
                    getattr(context, "status", None),
                )
            if active_intent is None:
                active_intent = getattr(context, "active_intent", getattr(context, "intent", None))
            if explicit_start is None:
                for attr in ("is_start", "start", "at_start"):
                    if hasattr(context, attr):
                        val = getattr(context, attr)
                        if isinstance(val, bool):
                            explicit_start = val
                            break
            if not intent_ended_flag:
                ev_list = getattr(context, "events", None) or ()
                intent_ended_flag = bool(
                    getattr(context, "intent_ended", False)
                    or getattr(context, "intent_ended_event", None) is not None
                    or (controller_status is not None and getattr(controller_status, "done", False))
                    or any(_extract_event_type(e) == "intent_ended" for e in ev_list)
                )
            if hasattr(context, "events") and getattr(context, "events"):
                raw_events.extend(getattr(context, "events"))
            if hasattr(context, "creator_events") and getattr(context, "creator_events"):
                raw_creator_events.extend(getattr(context, "creator_events"))
            if explicit_surprise is None:
                explicit_surprise = getattr(
                    context,
                    "surprise",
                    getattr(context, "surprise_score", None),
                )
            if explicit_novelty is None and hasattr(context, "novelty"):
                explicit_novelty = getattr(context, "novelty")
            if explicit_wake_if is None and hasattr(context, "wake_if"):
                explicit_wake_if = getattr(context, "wake_if")
            if explicit_energy is None and hasattr(context, "energy"):
                explicit_energy = getattr(context, "energy")
            if explicit_comfort is None and hasattr(context, "comfort"):
                explicit_comfort = getattr(context, "comfort")
            if explicit_boredom is None and hasattr(context, "boredom"):
                explicit_boredom = getattr(context, "boredom")
            if explicit_drives is None and hasattr(context, "drives"):
                explicit_drives = getattr(context, "drives")
            if not explicit_mail_arrived and hasattr(context, "mail_arrived"):
                explicit_mail_arrived = bool(getattr(context, "mail_arrived"))

        if world is None:
            world = self.world

        if tick is None:
            if percept is not None and hasattr(percept, "tick"):
                tick = int(percept.tick)
            elif world is not None:
                tick = int(world.tick)
            else:
                tick = 0

        all_events: list[Any] = list(raw_events) + list(raw_creator_events)
        reasons: list[str] = []
        matched_wake_if: list[str] = []

        # 1. Start trigger
        if explicit_start is True:
            is_start = True
        elif explicit_start is False:
            is_start = False
        elif context is None and not kwargs:
            is_start = not self._started
        else:
            is_start = False

        self._started = True
        if is_start:
            reasons.append("start")

        # 2. Surprise score over threshold (supplied by T16; 0 until then)
        surprise_val = 0.0
        if explicit_surprise is not None:
            surprise_val = float(explicit_surprise)
        elif percept is not None and getattr(percept, "surprise", None) is not None:
            surprise_val = float(getattr(percept, "surprise"))
        elif world is not None and getattr(world, "surprise", None) is not None:
            surprise_val = float(getattr(world, "surprise"))

        if surprise_val > 0.0 and surprise_val >= self.surprise_threshold:
            reasons.append("surprise")

        # 3. Novelty (never-seen object, property combination, or affordance)
        previously_seen_objects = set(self.seen_objects)
        if is_start:
            self._check_and_record_novelty(percept, seed_only=True)
            if explicit_novelty is True or (
                isinstance(explicit_novelty, int | float)
                and not isinstance(explicit_novelty, bool)
                and float(explicit_novelty) > 0.0
            ):
                reasons.append("novelty")
        else:
            percept_novelty = self._check_and_record_novelty(percept, seed_only=False)
            if explicit_novelty is True or (
                isinstance(explicit_novelty, int | float)
                and not isinstance(explicit_novelty, bool)
                and float(explicit_novelty) > 0.0
            ):
                reasons.append("novelty")
            elif explicit_novelty is not False and percept_novelty:
                reasons.append("novelty")

        # 4. Intent ended
        if intent_ended_flag:
            reasons.append("intent_ended")

        # Resolve body state (energy, comfort) and drives (boredom)
        energy_val = explicit_energy
        comfort_val = explicit_comfort
        if percept is not None and getattr(percept, "body", None) is not None:
            b_obj = percept.body
            if energy_val is None:
                energy_val = (
                    float(b_obj["energy"])
                    if isinstance(b_obj, Mapping) and "energy" in b_obj
                    else float(getattr(b_obj, "energy", 1.0))
                )
            if comfort_val is None:
                comfort_val = (
                    float(b_obj["comfort"])
                    if isinstance(b_obj, Mapping) and "comfort" in b_obj
                    else float(getattr(b_obj, "comfort", 1.0))
                )
        if world is not None:
            if energy_val is None:
                energy_val = float(world.body.energy)
            if comfort_val is None:
                comfort_val = float(world.body.comfort)

        boredom_val = explicit_boredom
        if boredom_val is None and explicit_drives is not None:
            if isinstance(explicit_drives, Mapping) and "boredom" in explicit_drives:
                boredom_val = float(explicit_drives["boredom"])
            elif hasattr(explicit_drives, "boredom"):
                boredom_val = float(explicit_drives.boredom)
        if boredom_val is None and world is not None:
            boredom_val = float(world.drives.boredom)

        # 5. Agent-set wake_if conditions
        if active_intent is None and controller is not None:
            active_intent = getattr(controller, "intent", None)

        raw_wake_if_source: Sequence[str] | None = None
        if explicit_wake_if is not None:
            raw_wake_if_source = (
                [explicit_wake_if] if isinstance(explicit_wake_if, str) else list(explicit_wake_if)
            )
        elif active_intent is not None:
            if isinstance(active_intent, Mapping) and "wake_if" in active_intent:
                raw_wake_if_source = list(active_intent["wake_if"] or ())
            elif hasattr(active_intent, "wake_if"):
                raw_wake_if_source = list(getattr(active_intent, "wake_if") or ())

        if raw_wake_if_source is not None:
            self.set_wake_if(raw_wake_if_source)

        if not is_start and self._parsed_wake_if:
            for cond in self._parsed_wake_if:
                if self._evaluate_wake_condition(
                    cond,
                    percept=percept,
                    world=world,
                    events=all_events,
                    energy=energy_val,
                    comfort=comfort_val,
                    previously_seen_objects=previously_seen_objects,
                ):
                    matched_wake_if.append(cond.raw)
            if matched_wake_if:
                reasons.append("wake_if")

        # 6. Drive thresholds (energy, comfort, boredom)
        if not is_start:
            if energy_val is not None and energy_val <= self.low_energy_threshold:
                reasons.append("low_energy")
            if comfort_val is not None and comfort_val <= self.low_comfort_threshold:
                reasons.append("low_comfort")
            if boredom_val is not None and boredom_val >= self.boredom_threshold:
                reasons.append("boredom")

        # 7. Creator events (spawn nearby, mail arrives)
        if not is_start:
            spawn_nearby, mail_arrived = self._check_creator_events(
                percept=percept,
                world=world,
                events=all_events,
                explicit_mail_arrived=explicit_mail_arrived,
            )
            if spawn_nearby:
                reasons.append("creator_spawn")
            if mail_arrived:
                reasons.append("mail_arrived")

        # 8. Heartbeat (every N ticks) and unparseable wake_if fallback
        if is_start:
            next_ticks_since_wake = 0
        elif explicit_ticks_since_wake is not None:
            next_ticks_since_wake = max(self._ticks_since_wake + 1, explicit_ticks_since_wake)
        else:
            next_ticks_since_wake = self._ticks_since_wake + 1

        is_heartbeat = False
        if not is_start:
            if next_ticks_since_wake >= self.heartbeat_interval:
                is_heartbeat = True
            elif tick is not None and tick > 0 and (tick % self.heartbeat_interval == 0):
                is_heartbeat = True

        unparseable_for_event = list(self._pending_unparseable_wake_if or self.unparseable_wake_if)
        if is_heartbeat:
            if self._pending_unparseable_wake_if:
                for raw_u in self._pending_unparseable_wake_if:
                    if raw_u not in matched_wake_if:
                        matched_wake_if.append(raw_u)
                if "wake_if" not in reasons:
                    reasons.append("wake_if")
                self._pending_unparseable_wake_if.clear()
            reasons.append("heartbeat")

        # Update movement/position tracking for subsequent ticks
        self._update_tracking_state(percept=percept, world=world, events=all_events)

        wake = bool(reasons)
        if wake:
            self._ticks_since_wake = 0
        else:
            self._ticks_since_wake = next_ticks_since_wake

        decision = WakeDecision(
            wake=wake,
            reasons=reasons,
            matched_wake_if=matched_wake_if,
            unparseable_wake_if=unparseable_for_event,
        )

        # Emit gate_decision event for every decision (wake or continue)
        decision_event = GateDecisionEvent(
            tick=int(tick),
            source="gate",
            wake=wake,
            reasons=list(reasons),
            decision=decision,
            surprise=surprise_val,
            matched_wake_if=list(matched_wake_if),
            unparseable_wake_if=list(unparseable_for_event),
        )
        decision._emitted_event = decision_event
        self.decisions.append(decision_event)
        self.events.emit(decision_event)
        decision._emitted_buses.append(self.events)

        if world is not None and world.events is not self.events:
            world.events.emit(decision_event)
            decision._emitted_buses.append(world.events)

        return decision


__all__ = [
    "DEFAULT_BOREDOM_WAKE_THRESHOLD",
    "DEFAULT_GATE_MOVE_THRESHOLD",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_LOW_COMFORT_WAKE_THRESHOLD",
    "DEFAULT_LOW_ENERGY_WAKE_THRESHOLD",
    "DEFAULT_SURPRISE_THRESHOLD",
    "Gate",
    "GateConfig",
    "GateContext",
    "GateDecisionEvent",
    "GateSettings",
    "HeartbeatGate",
    "MailArrivedEvent",
    "ParsedWakeCondition",
    "RuleGate",
    "WakeCondition",
    "WakeConditionKind",
    "WakeDecision",
    "parse_wake_conditions",
    "parse_wake_if",
]
