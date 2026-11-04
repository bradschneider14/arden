"""Headless agent loop: perceive -> gate -> decide -> controller -> world step."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from arden.config import Settings, make_rng
from arden.controller.controller import Controller, ControllerStatus
from arden.llm.base import LLMBackend, LLMOutputError
from arden.llm.factory import make_backend
from arden.llm.prompts import build_decide_prompt
from arden.llm.schemas import Intent
from arden.perception.feed import Percept, PerceptionTracker, perceive
from arden.world.clock import WorldClock
from arden.world.objects import BehaviorRule, Position
from arden.world.world import (
    EventBus,
    IntentEndedEvent,
    IntentSetEvent,
    LLMErrorEvent,
    WakeEvent,
    World,
    WorldEvent,
)

DEFAULT_HEARTBEAT_INTERVAL: int = 50
DEFAULT_RUN_TICKS: int = 500


class WakeDecision(BaseModel):
    """Decision returned by ``Gate.should_wake(context)``."""

    model_config = ConfigDict(extra="forbid")

    wake: bool = False
    reasons: list[str] = Field(default_factory=list)

    def __init__(
        self,
        wake: bool = False,
        reasons: Sequence[str] | str | None = None,
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

    def __bool__(self) -> bool:
        return self.wake

    @property
    def reason(self) -> str | None:
        """Comma-joined wake reasons, or ``None`` when empty."""
        return ", ".join(self.reasons) if self.reasons else None

    def to_dict(self) -> dict[str, Any]:
        return {"wake": self.wake, "reasons": list(self.reasons)}


def _coerce_wake_decision(value: WakeDecision | tuple[bool, Sequence[str]] | bool | Mapping[str, Any]) -> WakeDecision:
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

    The full rule-based gate replaces this in T15 behind the same ``Gate`` interface.
    """

    def __init__(
        self,
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
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
                    or (isinstance(st, ControllerStatus) and st.done)
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
                    if isinstance(st, ControllerStatus) and st.done:
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


@dataclass
class TickResult:
    """Outcome of a single headless agent-loop tick."""

    tick: int
    percept: Percept
    decision: WakeDecision
    controller_status: ControllerStatus
    intent: Intent | None = None
    llm_error: Exception | None = None
    summary: str | None = None

    @property
    def woke(self) -> bool:
        """True if the gate woke the LLM on this tick."""
        return self.decision.wake

    @property
    def wake(self) -> bool:
        """Alias for ``woke``."""
        return self.decision.wake

    @property
    def reasons(self) -> list[str]:
        """Wake reasons for this tick (empty when ``woke`` is False)."""
        return self.decision.reasons


def format_wake_summary(
    tick: int,
    reasons: Sequence[str],
    goal: str | None,
    thought: str | None,
) -> str:
    """Format a compact one-line summary for a single LLM wake."""
    reasons_str = ",".join(reasons) if reasons else "none"
    clean_goal = " ".join((goal or "idle").split())
    clean_thought = " ".join((thought or "none").split())
    return f"tick={tick} reasons=[{reasons_str}] goal={clean_goal!r} thought={clean_thought!r}"


def build_test_world(
    rng: random.Random | None = None,
    settings: Settings | None = None,
) -> World:
    """Construct a tiny deterministic test world for headless simulation runs.

    Includes 7 perceptible objects around a small clearing so all built-in mock
    intents (``look`` at ``object_3``, ``move`` toward and ``push`` ``object_7``
    toward the tree ``object_1``, and ``speak``) interact with real objects.
    """
    cfg = settings if settings is not None else Settings()
    world_rng = rng if rng is not None else make_rng(cfg)

    initial_objects: list[dict[str, Any]] = [
        # object_1: fixed tree-like tall cylinder north-east of the agent
        {
            "id": "object_1",
            "position": {"x": 4.5, "y": 1.5},
            "level": "ground",
            "shape": "cylinder",
            "height": 2.5,
            "volume": 0.8,
            "weight": 50.0,
            "rigidity": 0.9,
            "color": "green",
            "fixed": True,
        },
        # object_2: flat heavy stone
        {
            "id": "object_2",
            "position": {"x": -1.5, "y": 1.0},
            "level": "ground",
            "shape": "flat",
            "height": 0.2,
            "volume": 0.08,
            "weight": 3.0,
            "rigidity": 0.95,
            "color": "gray",
        },
        # object_3: small round red sphere within reach of starting position (0, 0)
        {
            "id": "object_3",
            "position": {"x": 0.8, "y": 0.4},
            "level": "ground",
            "shape": "sphere",
            "height": 0.2,
            "volume": 0.004,
            "weight": 0.3,
            "rigidity": 0.5,
            "color": "red",
            "behavior": [BehaviorRule(verb="push", scale=1.0)],
        },
        # object_4: long light stick
        {
            "id": "object_4",
            "position": {"x": 1.0, "y": -1.0},
            "level": "ground",
            "shape": "long",
            "height": 0.8,
            "volume": 0.02,
            "weight": 0.4,
            "rigidity": 0.8,
            "color": "brown",
        },
        # object_5: flexible binding vine
        {
            "id": "object_5",
            "position": {"x": -1.0, "y": -1.0},
            "level": "ground",
            "shape": "long",
            "height": 0.6,
            "volume": 0.01,
            "weight": 0.2,
            "rigidity": 0.2,
            "binding": True,
            "color": "green",
        },
        # object_6: hollow log
        {
            "id": "object_6",
            "position": {"x": -2.5, "y": -0.5},
            "level": "ground",
            "shape": "cylinder",
            "height": 0.6,
            "volume": 0.3,
            "weight": 8.0,
            "rigidity": 0.85,
            "hollow": True,
            "color": "brown",
        },
        # object_7: light movable sphere between the agent and the tree (object_1)
        {
            "id": "object_7",
            "position": {"x": 2.2, "y": 0.8},
            "level": "ground",
            "shape": "sphere",
            "height": 0.25,
            "volume": 0.008,
            "weight": 0.25,
            "rigidity": 0.6,
            "color": "red",
            "behavior": [BehaviorRule(verb="push", scale=1.0)],
        },
    ]

    return World(
        rng=world_rng,
        config=cfg,
        body={"position": Position(x=0.0, y=0.0), "level": "ground", "energy": 1.0, "comfort": 1.0},
        objects=initial_objects,
    )


make_test_world = build_test_world
create_test_world = build_test_world


class AgentLoop:
    """Headless agent loop wiring perception, attention gate, LLM decide, controller, and world clock.

    Each tick executes:
    1. ``perceive`` -> structured ``Percept``
    2. ``gate.should_wake(context)`` -> ``WakeDecision``
    3. If ``wake``: pause world, call LLM ``decide``, set the new ``Intent`` on the controller
       (or on ``LLMOutputError`` / timeout, emit ``llm_error`` and keep the current intent or idle)
    4. ``controller.tick(world)``
    5. ``clock.step_once()`` (``world.step()``)
    """

    def __init__(
        self,
        world: World | None = None,
        llm: LLMBackend | None = None,
        *,
        backend: LLMBackend | None = None,
        gate: Gate | None = None,
        controller: Controller | None = None,
        clock: WorldClock | None = None,
        settings: Settings | None = None,
        rng: random.Random | None = None,
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        timeout_s: float | None = None,
        self_model: str | None = None,
        memories: Sequence[Any] | None = None,
        perception_tracker: PerceptionTracker | None = None,
        on_wake: Callable[[str], None] | None = None,
        print_wakes: bool = False,
    ) -> None:
        self.settings: Settings = settings if settings is not None else Settings()
        self.rng: random.Random = (
            rng
            if rng is not None
            else (world.rng if world is not None else make_rng(self.settings))
        )
        self.world: World = (
            world
            if world is not None
            else build_test_world(rng=self.rng, settings=self.settings)
        )
        resolved_llm = llm if llm is not None else backend
        self.llm: LLMBackend = (
            resolved_llm if resolved_llm is not None else make_backend(self.settings)
        )
        self.gate: Gate = (
            gate if gate is not None else HeartbeatGate(interval=heartbeat_interval)
        )
        self.controller: Controller = (
            controller if controller is not None else Controller(world=self.world)
        )
        self.clock: WorldClock = (
            clock
            if clock is not None
            else WorldClock(self.world, tick_rate=self.settings.world.tick_rate)
        )
        self.timeout_s: float = (
            float(timeout_s)
            if timeout_s is not None
            else float(self.settings.llm.timeout_s)
        )
        self.self_model: str = self_model if self_model is not None else "none"
        self.memories: list[Any] = list(memories or ())
        self.perception_tracker: PerceptionTracker | None = perception_tracker
        self.on_wake: Callable[[str], None] | None = on_wake
        self.print_wakes: bool = bool(print_wakes)

        self.paused: bool = False
        self.ticks_run: int = 0
        self.wake_events: list[WakeEvent] = []
        self.intent_set_events: list[IntentSetEvent] = []
        self.llm_error_events: list[LLMErrorEvent] = []
        self.wake_summaries: list[str] = []

        self._has_started: bool = False
        self._ticks_since_wake: int = 0
        self._pending_intent_ended: bool = False
        self._pending_intent_ended_event: IntentEndedEvent | None = None
        self._last_completed_intent_event: IntentEndedEvent | None = None
        self._last_controller_status: ControllerStatus = self.controller.status
        self._tick_events: list[WorldEvent] = []

        self.world.events.subscribe(self._on_world_event)

    def _on_world_event(self, event: WorldEvent) -> None:
        if event.type not in ("tick", "wake", "intent_set", "llm_error"):
            self._tick_events.append(event)
        if isinstance(event, IntentEndedEvent) or event.type == "intent_ended":
            self._pending_intent_ended = True
            if isinstance(event, IntentEndedEvent):
                self._pending_intent_ended_event = event
                self._last_completed_intent_event = event

    @property
    def backend(self) -> LLMBackend:
        """Alias for ``llm``."""
        return self.llm

    @property
    def events(self) -> EventBus:
        """World event bus carrying ``wake``, ``intent_set``, ``llm_error``, ``intent_ended``, etc."""
        return self.world.events

    @property
    def event_bus(self) -> EventBus:
        """Alias for ``events``."""
        return self.world.events

    @property
    def intent(self) -> Intent | None:
        """Currently active ``Intent`` on the controller, or ``None`` if idle."""
        return self.controller.intent

    @property
    def current_intent(self) -> Intent | None:
        """Alias for ``intent``."""
        return self.controller.intent

    def subscribe(
        self,
        handler_or_type: Callable[[Any], None] | str,
        handler: Callable[[Any], None] | None = None,
        *,
        event_type: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to events on the simulation event bus."""
        return self.world.events.subscribe(handler_or_type, handler, event_type=event_type)  # type: ignore[arg-type]

    def _previous_intent_result_payload(self) -> dict[str, Any] | None:
        ev = self._last_completed_intent_event
        if ev is None:
            return None
        payload: dict[str, Any] = {
            "status": ev.status,
            "ok": ev.ok,
            "goal": ev.goal,
        }
        if ev.reason is not None:
            payload["reason"] = ev.reason
        return payload

    def _pause_world(self) -> None:
        self.paused = True
        self.clock.pause()
        self.world.paused = True

    def _resume_world(self) -> None:
        self.clock.resume()
        self.world.paused = False
        self.paused = False

    def tick(self, *, print_wakes: bool | None = None) -> TickResult:
        """Execute a single tick of the headless loop:

        perceive -> gate -> (if wake: pause world, decide, set_intent) -> controller.tick -> world.step.
        """
        should_print = self.print_wakes if print_wakes is None else bool(print_wakes)
        current_tick = self.world.tick

        # 1. Perceive
        if self.perception_tracker is not None:
            percept = self.perception_tracker.perceive(self.world)
        else:
            percept = perceive(self.world)

        # 2. Gate decides wake or continue
        context = GateContext(
            tick=current_tick,
            percept=percept,
            world=self.world,
            controller=self.controller,
            controller_status=self._last_controller_status,
            active_intent=self.controller.intent,
            is_start=not self._has_started,
            intent_ended=self._pending_intent_ended,
            intent_ended_event=self._pending_intent_ended_event,
            ticks_since_wake=self._ticks_since_wake,
            events=list(self._tick_events),
        )
        self._has_started = True
        self._pending_intent_ended = False
        self._pending_intent_ended_event = None
        self._tick_events.clear()

        decision = _coerce_wake_decision(self.gate.should_wake(context))
        llm_error: Exception | None = None
        summary: str | None = None

        # 3. If wake: pause world, call LLM decide, set the new intent (or survive LLM failure)
        if decision.wake:
            self._ticks_since_wake = 0
            wake_event = WakeEvent(
                tick=current_tick,
                source="gate",
                reasons=list(decision.reasons),
                decision=decision,
                percept=percept,
            )
            self.wake_events.append(wake_event)
            self.world.events.emit(wake_event)

            self._pause_world()
            try:
                prompt = build_decide_prompt(
                    percept=percept,
                    self_model=self.self_model,
                    memories=self.memories,
                    body_state=percept.body,
                    previous_intent_result=self._previous_intent_result_payload(),
                )
                raw_intent = self.llm.complete_json(
                    "decide",
                    prompt.system,
                    prompt.user,
                    Intent.model_json_schema(),
                    self.timeout_s,
                )
                if isinstance(raw_intent, Intent):
                    new_intent = raw_intent
                else:
                    try:
                        new_intent = Intent.model_validate(raw_intent)
                    except ValidationError as val_exc:
                        raw_str = (
                            json.dumps(raw_intent, sort_keys=True)
                            if isinstance(raw_intent, Mapping)
                            else str(raw_intent)
                        )
                        raise LLMOutputError(raw_str, str(val_exc)) from val_exc
            except (LLMOutputError, TimeoutError, OSError) as exc:
                llm_error = exc
                kept_intent = self.controller.intent
                err_event = LLMErrorEvent(
                    tick=current_tick,
                    source="agent",
                    role="decide",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    raw=getattr(exc, "raw", None),
                    kept_intent=kept_intent,
                    reasons=list(decision.reasons),
                )
                self.llm_error_events.append(err_event)
                self.world.events.emit(err_event)
                kept_goal = kept_intent.goal if kept_intent is not None else "idle"
                summary = format_wake_summary(
                    current_tick,
                    decision.reasons,
                    kept_goal,
                    f"llm_error: {exc}",
                )
            else:
                self.controller.set_intent(new_intent)
                intent_event = IntentSetEvent(
                    tick=current_tick,
                    source="agent",
                    intent=new_intent,
                    goal=new_intent.goal,
                    thought=new_intent.thought,
                    expect=new_intent.expect,
                    confidence=new_intent.confidence,
                    steps=list(new_intent.steps),
                    wake_if=list(new_intent.wake_if),
                    reasons=list(decision.reasons),
                )
                self.intent_set_events.append(intent_event)
                self.world.events.emit(intent_event)
                summary = format_wake_summary(
                    current_tick,
                    decision.reasons,
                    new_intent.goal,
                    new_intent.thought,
                )
            finally:
                self._resume_world()

            if summary is not None:
                self.wake_summaries.append(summary)
                if should_print:
                    print(summary)
                if self.on_wake is not None:
                    self.on_wake(summary)
        else:
            self._ticks_since_wake += 1

        # 4. Controller ticks
        controller_status = self.controller.tick(self.world)
        self._last_controller_status = controller_status
        if controller_status.done:
            self._pending_intent_ended = True
            if controller_status.event is not None:
                self._pending_intent_ended_event = controller_status.event
                self._last_completed_intent_event = controller_status.event

        # 5. World steps
        self.clock.step_once()
        self.ticks_run += 1

        return TickResult(
            tick=current_tick,
            percept=percept,
            decision=decision,
            controller_status=controller_status,
            intent=self.controller.intent if not controller_status.done else controller_status.intent,
            llm_error=llm_error,
            summary=summary,
        )

    step = tick

    def run(
        self,
        ticks: int = DEFAULT_RUN_TICKS,
        *,
        print_wakes: bool | None = None,
    ) -> list[TickResult]:
        """Run the headless agent loop for ``ticks`` ticks synchronously."""
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
            raise ValueError("ticks must be a non-negative integer")
        results: list[TickResult] = []
        for _ in range(ticks):
            results.append(self.tick(print_wakes=print_wakes))
        return results

    run_ticks = run


Loop = AgentLoop
HeadlessLoop = AgentLoop


def run_headless(
    settings: Settings | None = None,
    rng: random.Random | None = None,
    *,
    ticks: int = DEFAULT_RUN_TICKS,
    llm: LLMBackend | str | None = None,
    world: World | None = None,
    gate: Gate | None = None,
    controller: Controller | None = None,
    heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
    print_wakes: bool = True,
) -> AgentLoop:
    """Run a headless simulation for ``ticks`` ticks and return the ``AgentLoop``."""
    cfg = settings.model_copy(deep=True) if settings is not None else Settings()
    backend_instance: LLMBackend | None = None
    if isinstance(llm, str):
        cfg.llm.backend = llm
    elif llm is not None:
        backend_instance = llm

    sim_rng = rng if rng is not None else make_rng(cfg)
    loop = AgentLoop(
        world=world,
        llm=backend_instance,
        gate=gate,
        controller=controller,
        settings=cfg,
        rng=sim_rng,
        heartbeat_interval=heartbeat_interval,
        print_wakes=print_wakes,
    )
    loop.run(ticks, print_wakes=print_wakes)
    return loop


run_loop = run_headless
run_agent_loop = run_headless

__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_RUN_TICKS",
    "AgentLoop",
    "Gate",
    "GateContext",
    "HeadlessLoop",
    "HeartbeatGate",
    "IntentEndedEvent",
    "IntentSetEvent",
    "LLMErrorEvent",
    "Loop",
    "TickResult",
    "WakeDecision",
    "WakeEvent",
    "build_test_world",
    "create_test_world",
    "format_wake_summary",
    "make_test_world",
    "run_agent_loop",
    "run_headless",
    "run_loop",
]
