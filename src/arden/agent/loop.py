"""Headless agent loop: perceive -> gate -> decide -> controller -> world step."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from arden.agent.gate import (
    DEFAULT_BOREDOM_WAKE_THRESHOLD,
    DEFAULT_GATE_MOVE_THRESHOLD,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_LOW_COMFORT_WAKE_THRESHOLD,
    DEFAULT_LOW_ENERGY_WAKE_THRESHOLD,
    DEFAULT_SURPRISE_THRESHOLD,
    Gate,
    GateConfig,
    GateContext,
    GateDecisionEvent,
    GateSettings,
    HeartbeatGate,
    MailArrivedEvent,
    ParsedWakeCondition,
    RuleGate,
    WakeCondition,
    WakeConditionKind,
    WakeDecision,
    _coerce_wake_decision,
    parse_wake_conditions,
    parse_wake_if,
)
from arden.config import Settings, make_rng
from arden.controller.controller import Controller, ControllerStatus
from arden.llm.base import LLMBackend, LLMOutputError
from arden.llm.factory import make_backend
from arden.llm.prompts import BuiltPrompt, build_decide_prompt
from arden.llm.schemas import Intent
from arden.memory.beliefs import (
    BeliefStore,
    SelfModel,
    extract_context_tags,
    retrieve,
)
from arden.memory.short_term import MemoryManager
from arden.memory.store import EpisodeStore
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

DEFAULT_RUN_TICKS: int = 500


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
    prompt: BuiltPrompt | None = None

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
    2. ``gate.should_wake(context)`` -> ``WakeDecision`` (emitting ``gate_decision``)
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
        heartbeat_interval: int | None = None,
        timeout_s: float | None = None,
        self_model: str | SelfModel | None = None,
        memories: Sequence[Any] | None = None,
        memory: MemoryManager | None = None,
        store: EpisodeStore | None = None,
        episode_store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        retrieval_k: int | None = None,
        k: int | None = None,
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
            gate
            if gate is not None
            else RuleGate(
                config=self.settings.gate,
                heartbeat_interval=heartbeat_interval,
                world=self.world,
            )
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
        self.memory: MemoryManager | None = memory
        self.episode_store: EpisodeStore | None = (
            episode_store
            if episode_store is not None
            else (store if store is not None else (memory.store if memory is not None else None))
        )
        self.belief_store: BeliefStore | None = (
            belief_store
            if belief_store is not None
            else (
                memory.belief_store
                if memory is not None
                else (self.episode_store.belief_store if self.episode_store is not None else None)
            )
        )
        self.retrieval_k: int = (
            int(retrieval_k)
            if retrieval_k is not None
            else (int(k) if k is not None else int(self.settings.memory.retrieval_k))
        )
        if isinstance(self_model, SelfModel):
            self.self_model_store: SelfModel | None = self_model
            self.self_model: str = self_model.text or "none"
        else:
            self.self_model_store = (
                memory.self_model_store
                if memory is not None
                else (self.episode_store.self_model_store if self.episode_store is not None else None)
            )
            self.self_model = self_model if self_model is not None else "none"
        self.memories: list[Any] = list(memories or ())
        self.perception_tracker: PerceptionTracker | None = perception_tracker
        self.on_wake: Callable[[str], None] | None = on_wake
        self.print_wakes: bool = bool(print_wakes)

        self.paused: bool = False
        self.ticks_run: int = 0
        self.gate_decision_events: list[GateDecisionEvent] = []
        self.wake_events: list[WakeEvent] = []
        self.intent_set_events: list[IntentSetEvent] = []
        self.llm_error_events: list[LLMErrorEvent] = []
        self.wake_summaries: list[str] = []
        self.last_decide_prompt: BuiltPrompt | None = None
        self.last_retrieved_memories: list[Any] = []

        self._has_started: bool = False
        self._ticks_since_wake: int = 0
        self._pending_intent_ended: bool = False
        self._pending_intent_ended_event: IntentEndedEvent | None = None
        self._last_completed_intent_event: IntentEndedEvent | None = None
        self._last_controller_status: ControllerStatus = self.controller.status
        self._tick_events: list[WorldEvent] = []

        self.world.events.subscribe(self._on_world_event)

    def _on_world_event(self, event: WorldEvent) -> None:
        if isinstance(event, GateDecisionEvent) or event.type == "gate_decision":
            if isinstance(event, GateDecisionEvent):
                self.gate_decision_events.append(event)
            return
        if event.type not in ("tick", "wake", "intent_set", "llm_error", "gate_decision"):
            self._tick_events.append(event)
        if isinstance(event, IntentEndedEvent) or event.type == "intent_ended":
            self._pending_intent_ended = True
            if isinstance(event, IntentEndedEvent):
                self._pending_intent_ended_event = event
                self._last_completed_intent_event = event

    @property
    def gate_decisions(self) -> list[GateDecisionEvent]:
        """Alias for ``gate_decision_events``."""
        return self.gate_decision_events

    @property
    def decision_events(self) -> list[GateDecisionEvent]:
        """Alias for ``gate_decision_events``."""
        return self.gate_decision_events

    @property
    def backend(self) -> LLMBackend:
        """Alias for ``llm``."""
        return self.llm

    @property
    def events(self) -> EventBus:
        """World event bus carrying ``gate_decision``, ``wake``, ``intent_set``, ``llm_error``, ``intent_ended``, etc."""
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
        surprise_val = float(
            getattr(self.world, "surprise", getattr(self.world, "surprise_score", 0.0)) or 0.0
        )
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
            surprise=surprise_val,
        )
        self._has_started = True
        self._pending_intent_ended = False
        self._pending_intent_ended_event = None
        self._tick_events.clear()

        decision = _coerce_wake_decision(self.gate.should_wake(context))
        if self.world.events not in getattr(decision, "_emitted_buses", ()):
            decision_event = GateDecisionEvent(
                tick=current_tick,
                source="gate",
                wake=decision.wake,
                reasons=list(decision.reasons),
                decision=decision,
                surprise=surprise_val,
                matched_wake_if=list(decision.matched_wake_if),
                unparseable_wake_if=list(decision.unparseable_wake_if),
            )
            decision._emitted_event = decision_event
            self.world.events.emit(decision_event)

        llm_error: Exception | None = None
        summary: str | None = None
        built_prompt: BuiltPrompt | None = None

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
                effective_self_model: str | SelfModel | None = self.self_model
                if (
                    effective_self_model in (None, "", "none")
                    and self.self_model_store is not None
                    and self.self_model_store.text
                ):
                    effective_self_model = self.self_model_store.text
                elif (
                    effective_self_model in (None, "", "none")
                    and self.memory is not None
                    and self.memory.self_model
                ):
                    effective_self_model = self.memory.self_model

                context_tags = extract_context_tags(percept)
                retrieved = retrieve(
                    context_tags,
                    self.retrieval_k,
                    episode_store=self.episode_store,
                    belief_store=self.belief_store,
                    manager=self.memory,
                    memories=self.memories if self.memories else None,
                    current_tick=current_tick,
                )
                self.last_retrieved_memories = list(retrieved)
                prompt_memories = retrieved if retrieved else (self.memories if self.memories else None)

                prompt = build_decide_prompt(
                    percept=percept,
                    self_model=effective_self_model,
                    memories=prompt_memories,
                    body_state=percept.body,
                    previous_intent_result=self._previous_intent_result_payload(),
                    context_tags=context_tags,
                    k=self.retrieval_k,
                )
                built_prompt = prompt
                self.last_decide_prompt = prompt
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
                if hasattr(self.gate, "set_wake_if"):
                    self.gate.set_wake_if(new_intent.wake_if)
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
            prompt=built_prompt,
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
    heartbeat_interval: int | None = None,
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
    "DEFAULT_BOREDOM_WAKE_THRESHOLD",
    "DEFAULT_GATE_MOVE_THRESHOLD",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_LOW_COMFORT_WAKE_THRESHOLD",
    "DEFAULT_LOW_ENERGY_WAKE_THRESHOLD",
    "DEFAULT_RUN_TICKS",
    "DEFAULT_SURPRISE_THRESHOLD",
    "AgentLoop",
    "Gate",
    "GateConfig",
    "GateContext",
    "GateDecisionEvent",
    "GateSettings",
    "HeadlessLoop",
    "HeartbeatGate",
    "IntentEndedEvent",
    "IntentSetEvent",
    "LLMErrorEvent",
    "Loop",
    "MailArrivedEvent",
    "ParsedWakeCondition",
    "RuleGate",
    "TickResult",
    "WakeCondition",
    "WakeConditionKind",
    "WakeDecision",
    "WakeEvent",
    "build_test_world",
    "create_test_world",
    "format_wake_summary",
    "make_test_world",
    "parse_wake_conditions",
    "parse_wake_if",
    "run_agent_loop",
    "run_headless",
    "run_loop",
]
