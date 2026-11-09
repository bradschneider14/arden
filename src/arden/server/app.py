"""FastAPI headless simulation service, creator API, WebSocket stream, and auth."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import random
import secrets
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from arden import __version__
from arden.agent.gate import Gate, GateDecisionEvent
from arden.agent.loop import (
    AgentLoop,
    GateContext,
    TickResult,
    _coerce_wake_decision,
    build_test_world,
    format_wake_summary,
)
from arden.config import (
    CREATOR_TOKEN_ENV_VAR,
    DEFAULT_DB_PATH,
    Settings,
    load_config,
    make_rng,
)
from arden.controller.controller import Controller
from arden.llm.base import LLMBackend, LLMOutputError
from arden.llm.factory import make_backend
from arden.llm.prompts import BuiltPrompt, build_decide_prompt
from arden.llm.schemas import Intent
from arden.mailbox import CreatorInbox
from arden.memory.beliefs import (
    BeliefEvictedEvent,
    BeliefStore,
    SelfModel,
    extract_context_tags,
    retrieve,
)
from arden.memory.short_term import MemoryManager, ReflectionEvent
from arden.memory.store import EpisodeStore
from arden.perception.feed import PerceptionTracker, perceive
from arden.surprise.scorer import SurpriseScorer, SurpriseSignal
from arden.world.clock import WorldClock
from arden.world.loader import DEFAULT_FIRST_RUN_WORLD_PATH, load_world
from arden.world.objects import WorldObject
from arden.world.world import (
    CycleFrozenEvent,
    IntentEndedEvent,
    IntentSetEvent,
    LLMErrorEvent,
    MailArrivedEvent,
    ObjectChangedEvent,
    ObjectDespawnedEvent,
    ObjectMovedEvent,
    ObjectSpawnedEvent,
    SpokeEvent,
    TickEvent,
    TimeOfDayChangedEvent,
    WakeEvent,
    WeatherChangedEvent,
    World,
    WorldEvent,
)

SCHEMA_VERSION: int = 1
WS_SCHEMA_VERSION: int = SCHEMA_VERSION

LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def is_local_host(host: str) -> bool:
    """Return True if ``host`` refers to a local loopback interface."""
    cleaned = host.strip().lower()
    if cleaned in LOCAL_HOSTS:
        return True
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return False
    return bool(addr.is_loopback)


def resolve_creator_token(
    explicit_token: str | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Resolve the creator authentication token from explicit arg, config, or ``ARDEN_CREATOR_TOKEN``."""
    if explicit_token is not None:
        stripped = str(explicit_token).strip()
        if stripped:
            return stripped
    if settings is not None and settings.server.creator_token:
        stripped = settings.server.creator_token.strip()
        if stripped:
            return stripped
    env_val = os.environ.get(CREATOR_TOKEN_ENV_VAR)
    if env_val is not None:
        stripped = env_val.strip()
        if stripped:
            return stripped
    return None


def validate_server_binding(host: str, creator_token: str | None) -> None:
    """Refuse to start on a non-local interface when no creator token is configured."""
    if not is_local_host(host) and not (creator_token and creator_token.strip()):
        raise ValueError(
            f"Refusing to bind non-local host {host!r} without a creator token "
            f"(set [server].creator_token in config or {CREATOR_TOKEN_ENV_VAR} in environment)."
        )


def _extract_request_token(request: Request) -> str | None:
    """Extract a creator token from headers or query parameters."""
    auth_header = request.headers.get("authorization")
    if auth_header:
        parts = auth_header.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
            tok = parts[1].strip()
            if tok:
                return tok
        elif len(parts) == 1 and parts[0]:
            return parts[0].strip()

    for header_name in ("x-creator-token", "x-api-token", "x-auth-token", "x-token"):
        val = request.headers.get(header_name)
        if val and val.strip():
            return val.strip()

    for query_name in ("token", "creator_token", "api_key"):
        val = request.query_params.get(query_name)
        if val and val.strip():
            return val.strip()

    return None


def _extract_websocket_token(websocket: WebSocket) -> str | None:
    """Extract a creator token from WebSocket headers or query parameters."""
    auth_header = websocket.headers.get("authorization")
    if auth_header:
        parts = auth_header.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
            tok = parts[1].strip()
            if tok:
                return tok
        elif len(parts) == 1 and parts[0]:
            return parts[0].strip()

    for header_name in ("x-creator-token", "x-api-token", "x-auth-token", "x-token"):
        val = websocket.headers.get(header_name)
        if val and val.strip():
            return val.strip()

    for query_name in ("token", "creator_token", "api_key"):
        val = websocket.query_params.get(query_name)
        if val and val.strip():
            return val.strip()

    return None


def _is_spectator_request(request: Request) -> bool:
    """Return True if the request explicitly declares read-only spectator mode."""
    for key in ("spectator", "read_only", "readonly"):
        q = request.query_params.get(key)
        if q is not None and q.strip().lower() in ("1", "true", "yes"):
            return True
    role = request.headers.get("x-role", request.query_params.get("role", ""))
    if role.strip().lower() == "spectator":
        return True
    spec_hdr = request.headers.get("x-spectator", "")
    if spec_hdr.strip().lower() in ("1", "true", "yes"):
        return True
    return False


def _ensure_spawn_defaults(props: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure minimal object spawn payloads have valid physical defaults."""
    data = dict(props)
    if "shape" not in data:
        data["shape"] = "sphere"
    if "level" not in data:
        data["level"] = "ground"
    if "height" not in data:
        data["height"] = 0.25
    if "volume" not in data:
        data["volume"] = 0.01
    if "weight" not in data:
        data["weight"] = 0.5
    if "rigidity" not in data:
        data["rigidity"] = 0.5
    if "position" not in data:
        if "x" in data and "y" in data:
            data["position"] = {"x": float(data.pop("x")), "y": float(data.pop("y"))}
        elif "pos" in data:
            pos = data.pop("pos")
            if isinstance(pos, Sequence) and len(pos) == 2:
                data["position"] = {"x": float(pos[0]), "y": float(pos[1])}
            else:
                data["position"] = pos
        else:
            data["position"] = {"x": 0.0, "y": 0.0}
    else:
        pos = data["position"]
        if isinstance(pos, Sequence) and not isinstance(pos, str | bytes) and len(pos) == 2:
            data["position"] = {"x": float(pos[0]), "y": float(pos[1])}
        data.pop("x", None)
        data.pop("y", None)
        data.pop("pos", None)
    return data


class SimulationService:
    """Single-process headless simulation service coordinating World, AgentLoop, Memory, and WebSocket stream."""

    def __init__(
        self,
        world: World | Path | str | Mapping[str, Any] | None = None,
        *,
        settings: Settings | Path | str | Mapping[str, Any] | None = None,
        config: Settings | Path | str | Mapping[str, Any] | None = None,
        llm: LLMBackend | str | None = None,
        backend: LLMBackend | str | None = None,
        loop: AgentLoop | None = None,
        gate: Gate | None = None,
        controller: Controller | None = None,
        clock: WorldClock | None = None,
        memory: MemoryManager | None = None,
        episode_store: EpisodeStore | None = None,
        belief_store: BeliefStore | None = None,
        self_model: SelfModel | str | None = None,
        inbox: CreatorInbox | None = None,
        rng: random.Random | int | None = None,
        host: str | None = None,
        port: int | None = None,
        creator_token: str | None = None,
        spectator_enabled: bool | None = None,
        spectator_mode: bool | None = None,
        auto_start: bool | None = None,
        paused: bool | None = None,
        db_path: Path | str | None = None,
    ) -> None:
        raw_cfg = settings if settings is not None else config
        if raw_cfg is None:
            self.settings = Settings()
            has_explicit_settings = False
        elif isinstance(raw_cfg, Settings):
            self.settings = raw_cfg.model_copy(deep=True)
            has_explicit_settings = True
        elif isinstance(raw_cfg, Path | str):
            self.settings = load_config(Path(raw_cfg))
            has_explicit_settings = True
        elif isinstance(raw_cfg, Mapping):
            self.settings = Settings.model_validate(dict(raw_cfg))
            has_explicit_settings = True
        else:
            raise TypeError("settings must be a Settings instance, Path, mapping, or None")

        if host is not None:
            self.settings.server.host = str(host).strip()
        if port is not None:
            self.settings.server.port = int(port)
        if creator_token is not None:
            self.settings.server.creator_token = (
                str(creator_token).strip() if str(creator_token).strip() else None
            )
        if spectator_enabled is not None:
            self.settings.server.spectator_enabled = bool(spectator_enabled)
        if spectator_mode is not None:
            self.settings.server.spectator_mode = bool(spectator_mode)
        if auto_start is not None:
            self.settings.server.auto_start = bool(auto_start)

        self._explicit_creator_token: str | None = (
            str(creator_token).strip()
            if creator_token is not None and str(creator_token).strip()
            else None
        )
        validate_server_binding(self.host, self.creator_token)

        resolved_llm_arg = llm if llm is not None else backend
        if isinstance(resolved_llm_arg, str):
            self.settings.llm.backend = resolved_llm_arg
            self.llm: LLMBackend = make_backend(self.settings)
        elif resolved_llm_arg is not None:
            self.llm = resolved_llm_arg
        elif loop is not None:
            self.llm = loop.llm
        else:
            self.llm = make_backend(self.settings)

        if isinstance(rng, random.Random):
            self.rng: random.Random = rng
        elif isinstance(rng, int) and not isinstance(rng, bool):
            self.rng = random.Random(rng)
        elif isinstance(world, World):
            self.rng = world.rng
        elif loop is not None:
            self.rng = loop.rng
        else:
            self.rng = make_rng(self.settings)

        eff_world_target = world if world is not None else self.settings.server.world
        if loop is not None and world is None:
            self.world: World = loop.world
        elif isinstance(eff_world_target, World):
            self.world = eff_world_target
        elif eff_world_target is not None:
            self.world = load_world(
                eff_world_target,
                rng=self.rng,
                settings=self.settings,
                inbox=inbox,
                db_path=db_path,
            )
        elif DEFAULT_FIRST_RUN_WORLD_PATH.is_file():
            self.world = load_world(
                DEFAULT_FIRST_RUN_WORLD_PATH,
                rng=self.rng,
                settings=self.settings,
                inbox=inbox,
                db_path=db_path,
            )
        else:
            self.world = build_test_world(rng=self.rng, settings=self.settings)

        if inbox is not None:
            self.world._mailbox_manager.inbox = inbox

        # Resolve memory system (defaulting to :memory: in standalone tests unless db_path or custom settings.memory.db_path is set)
        if memory is not None:
            self.memory: MemoryManager = memory
            if self.memory.world is None:
                self.memory.world = self.world
            if self.memory.llm is None:
                self.memory.llm = self.llm
        else:
            eff_mem_db: Path | str
            if db_path is not None:
                eff_mem_db = db_path
            elif has_explicit_settings and self.settings.memory.db_path != DEFAULT_DB_PATH:
                eff_mem_db = self.settings.memory.db_path
            else:
                eff_mem_db = ":memory:"
            self.memory = MemoryManager(
                llm=self.llm,
                store=episode_store,
                belief_store=belief_store,
                self_model=self_model,
                world=self.world,
                settings=self.settings,
                db_path=eff_mem_db,
            )

        if self.memory.belief_store.world is None:
            self.memory.belief_store.world = self.world

        if loop is not None:
            self.loop: AgentLoop = loop
            if self.loop.memory is None:
                self.loop.memory = self.memory
            if self.loop.episode_store is None:
                self.loop.episode_store = self.memory.store
            if self.loop.belief_store is None:
                self.loop.belief_store = self.memory.belief_store
            if self.loop.self_model_store is None:
                self.loop.self_model_store = self.memory.self_model_store
        else:
            self.loop = AgentLoop(
                world=self.world,
                llm=self.llm,
                gate=gate,
                controller=controller,
                clock=clock,
                settings=self.settings,
                rng=self.rng,
                memory=self.memory,
                episode_store=self.memory.store,
                belief_store=self.memory.belief_store,
                self_model=self.memory.self_model_store,
                perception_tracker=PerceptionTracker(),
                surprise_scorer=SurpriseScorer(config=self.settings.surprise, llm=self.llm),
            )

        self.clock: WorldClock = self.loop.clock
        self._lock = threading.RLock()
        self._paused: bool = bool(paused) if paused is not None else False
        if self._paused:
            self.clock.pause()
            self.world.paused = True

        self.last_intent: Intent | None = self.loop.controller.intent
        self.last_thought: str | None = (
            self.last_intent.thought if self.last_intent is not None else None
        )
        self._ws_clients: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = []
        self.event_history: list[dict[str, Any]] = []
        self._bg_task: asyncio.Task[None] | None = None
        self._running: bool = False

        self.world.events.subscribe(self._on_world_event)

    @property
    def host(self) -> str:
        """Configured bind host."""
        return self.settings.server.host

    @property
    def port(self) -> int:
        """Configured TCP port."""
        return self.settings.server.port

    @property
    def creator_token(self) -> str | None:
        """Resolved creator authentication token (from explicit arg, config, or ``ARDEN_CREATOR_TOKEN``)."""
        return resolve_creator_token(self._explicit_creator_token, self.settings)

    @property
    def spectator_enabled(self) -> bool:
        """True if unauthenticated read-only spectator access is enabled."""
        return self.settings.server.spectator_enabled

    @spectator_enabled.setter
    def spectator_enabled(self, value: bool) -> None:
        self.settings.server.spectator_enabled = bool(value)

    @property
    def spectator_mode(self) -> bool:
        """True if the server itself is locked in read-only spectator mode."""
        return self.settings.server.spectator_mode

    @spectator_mode.setter
    def spectator_mode(self, value: bool) -> None:
        self.settings.server.spectator_mode = bool(value)

    @property
    def paused(self) -> bool:
        """True when simulation stepping is paused by the creator."""
        return self._paused

    @property
    def speed(self) -> float:
        """Current simulation clock speed multiplier."""
        return self.clock.speed

    def register_ws_queue(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Register a connected WebSocket client's event queue."""
        with self._lock:
            self._ws_clients.append((loop, queue))

    def unregister_ws_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unregister a disconnected WebSocket client's event queue."""
        with self._lock:
            self._ws_clients = [(lp, q) for lp, q in self._ws_clients if q is not queue]

    def _broadcast(self, message: dict[str, Any]) -> dict[str, Any]:
        """Publish a versioned JSON message to all connected WebSocket clients."""
        payload = {"schema_version": SCHEMA_VERSION, **message}
        with self._lock:
            self.event_history.append(payload)
            if len(self.event_history) > 500:
                self.event_history = self.event_history[-500:]
            clients = list(self._ws_clients)

        for ev_loop, queue in clients:
            try:
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is ev_loop:
                    queue.put_nowait(payload)
                else:
                    ev_loop.call_soon_threadsafe(queue.put_nowait, payload)
            except RuntimeError:
                continue
        return payload

    def _on_world_event(self, event: WorldEvent) -> None:
        """Translate world, gate, agent, and memory events into WebSocket stream messages."""
        if isinstance(event, GateDecisionEvent) or event.type == "gate_decision":
            reasons = list(getattr(event, "reasons", ()))
            wake = bool(getattr(event, "wake", False))
            self._broadcast(
                {
                    "type": "gate_decision",
                    "event": "gate_decision",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "wake": wake,
                    "outcome": "wake" if wake else "continue",
                    "reasons": reasons,
                    "surprise": float(getattr(event, "surprise", 0.0)),
                    "matched_wake_if": list(getattr(event, "matched_wake_if", ())),
                    "unparseable_wake_if": list(getattr(event, "unparseable_wake_if", ())),
                }
            )
            return

        if isinstance(event, WakeEvent) or event.type == "wake":
            reasons = list(getattr(event, "reasons", ()))
            self._broadcast(
                {
                    "type": "wake",
                    "event": "wake",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "reasons": reasons,
                    "reason": ", ".join(reasons) if reasons else None,
                }
            )
            return

        if isinstance(event, IntentSetEvent) or event.type == "intent_set":
            intent_obj = getattr(event, "intent", None)
            if isinstance(intent_obj, Intent):
                intent_dict = intent_obj.model_dump(mode="json")
                self.last_intent = intent_obj
            elif isinstance(intent_obj, Mapping):
                intent_dict = dict(intent_obj)
            else:
                intent_dict = None
            thought_str = getattr(event, "thought", None)
            if thought_str is not None:
                self.last_thought = str(thought_str)
            goal_str = getattr(event, "goal", None)
            expect_str = getattr(event, "expect", None)
            conf_val = getattr(event, "confidence", None)
            steps_raw = getattr(event, "steps", ())
            steps_list = [
                s.model_dump(mode="json") if hasattr(s, "model_dump") else s
                for s in steps_raw
            ]
            wake_if_list = list(getattr(event, "wake_if", ()))
            reasons_list = list(getattr(event, "reasons", ()))

            self._broadcast(
                {
                    "type": "intent",
                    "event": "intent_set",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "intent": intent_dict,
                    "goal": goal_str,
                    "thought": thought_str,
                    "expect": expect_str,
                    "confidence": conf_val,
                    "steps": steps_list,
                    "wake_if": wake_if_list,
                    "reasons": reasons_list,
                }
            )
            self._broadcast(
                {
                    "type": "thought",
                    "event": "thought",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "thought": thought_str,
                    "text": thought_str,
                    "goal": goal_str,
                    "intent": intent_dict,
                }
            )
            return

        if isinstance(event, ReflectionEvent) or event.type == "reflection":
            eps = [
                ep.to_dict() if hasattr(ep, "to_dict") else (
                    ep.model_dump(mode="json") if hasattr(ep, "model_dump") else ep
                )
                for ep in getattr(event, "episodes", ())
            ]
            bels = [
                b.to_dict() if hasattr(b, "to_dict") else (
                    b.model_dump(mode="json") if hasattr(b, "model_dump") else b
                )
                for b in getattr(event, "beliefs", ())
            ]
            sm = getattr(event, "self_model", None)
            self._broadcast(
                {
                    "type": "reflection",
                    "event": "reflection",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "trigger": getattr(event, "trigger", "buffer_full"),
                    "episodes": eps,
                    "beliefs": bels,
                    "self_model": sm,
                }
            )
            return

        if isinstance(event, LLMErrorEvent) or event.type == "llm_error":
            self._broadcast(
                {
                    "type": "llm_error",
                    "event": "llm_error",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "role": getattr(event, "role", "decide"),
                    "error": getattr(event, "error", ""),
                    "error_type": getattr(event, "error_type", "LLMOutputError"),
                }
            )
            return

        if isinstance(event, IntentEndedEvent) or event.type == "intent_ended":
            self._broadcast(
                {
                    "type": "intent_ended",
                    "event": "intent_ended",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "status": getattr(event, "status", "succeeded"),
                    "ok": bool(getattr(event, "ok", True)),
                    "goal": getattr(event, "goal", None),
                    "reason": getattr(event, "reason", None),
                }
            )
            return

        if isinstance(event, BeliefEvictedEvent) or event.type == "belief_evicted":
            self._broadcast(
                {
                    "type": "belief_evicted",
                    "event": "belief_evicted",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "belief_id": getattr(event, "belief_id", None),
                    "statement": getattr(event, "statement", ""),
                    "confidence": getattr(event, "confidence", 0.0),
                }
            )
            return

        # World deltas: object_spawned, object_despawned, object_moved, object_changed, weather_changed, time_of_day_changed, cycle_frozen, mail_arrived, spoke, tick
        if isinstance(
            event,
            (
                ObjectSpawnedEvent,
                ObjectDespawnedEvent,
                ObjectMovedEvent,
                ObjectChangedEvent,
                TimeOfDayChangedEvent,
                CycleFrozenEvent,
                WeatherChangedEvent,
                MailArrivedEvent,
                SpokeEvent,
                TickEvent,
            ),
        ) or event.type in (
            "object_spawned",
            "object_despawned",
            "object_moved",
            "object_changed",
            "time_of_day_changed",
            "cycle_frozen",
            "weather_changed",
            "mail_arrived",
            "spoke",
            "tick",
        ):
            self._emit_world_delta_from_event(event)

    def _emit_world_delta_from_event(self, event: WorldEvent) -> dict[str, Any]:
        """Build and broadcast a ``world_delta`` WebSocket message from a ``WorldEvent``."""
        obj_id = event.object_id
        obj_dict = event.object
        if obj_dict is None and obj_id is not None and obj_id in self.world.objects:
            obj_dict = self.world.objects[obj_id].model_dump(mode="json")

        pos_dict = event.position.model_dump(mode="json") if event.position is not None else None
        prev_pos_dict = (
            event.previous_position.model_dump(mode="json")
            if event.previous_position is not None
            else None
        )

        objects_map: dict[str, Any] = {}
        if obj_id is not None and obj_dict is not None and event.type != "object_despawned":
            objects_map[obj_id] = obj_dict
        elif event.type == "tick":
            objects_map = {
                oid: self.world.objects[oid].model_dump(mode="json")
                for oid in sorted(self.world.objects)
            }

        delta_payload: dict[str, Any] = {
            "event": event.type,
            "tick": event.tick,
            "source": event.source,
        }
        if obj_id is not None:
            delta_payload["object_id"] = obj_id
        if obj_dict is not None:
            delta_payload["object"] = obj_dict
        if pos_dict is not None:
            delta_payload["position"] = pos_dict
        if prev_pos_dict is not None:
            delta_payload["previous_position"] = prev_pos_dict
        if event.changes is not None:
            delta_payload["changes"] = event.changes
        if event.text is not None:
            delta_payload["text"] = event.text
        if objects_map:
            delta_payload["objects"] = objects_map
        if event.type == "object_spawned" and obj_dict is not None:
            delta_payload["spawned"] = [obj_dict]
            delta_payload["action"] = "spawn"
        elif event.type == "object_despawned" and obj_id is not None:
            delta_payload["despawned"] = [obj_id]
            delta_payload["action"] = "despawn"
        elif event.type == "object_moved" and obj_id is not None:
            delta_payload["moved"] = [obj_id]
            delta_payload["action"] = "move"

        msg: dict[str, Any] = {
            "type": "world_delta",
            "event": event.type,
            "category": "world_delta",
            "tick": event.tick,
            "source": event.source,
            "body": self.world.body.model_dump(mode="json"),
            "time": {
                "time_of_day": self.world.time_of_day,
                "frozen": self.world.cycle_frozen,
                "light_level": self.world.light_level,
                "ambient_temperature": self.world.ambient_temperature,
            },
            "weather": {
                "type": self.world.weather_type,
                "intensity": self.world.weather_intensity,
                "paused": self.world.weather_automation_paused,
            },
            "delta": delta_payload,
        }
        if obj_id is not None:
            msg["object_id"] = obj_id
        if obj_dict is not None:
            msg["object"] = obj_dict
        if pos_dict is not None:
            msg["position"] = pos_dict
        if prev_pos_dict is not None:
            msg["previous_position"] = prev_pos_dict
        if event.changes is not None:
            msg["changes"] = event.changes
        if objects_map:
            msg["objects"] = objects_map
        if event.type == "object_spawned":
            msg["action"] = "spawn"
        elif event.type == "object_despawned":
            msg["action"] = "despawn"
        elif event.type == "object_moved":
            msg["action"] = "move"

        return self._broadcast(msg)

    def emit_creator_action(
        self,
        action: str,
        details: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Broadcast a ``creator_action`` event on the WebSocket stream."""
        details_dict = dict(details) if details is not None else {}
        msg: dict[str, Any] = {
            "type": "creator_action",
            "event": "creator_action",
            "category": "creator_action",
            "action": action,
            "source": "creator",
            "tick": self.world.tick,
            "details": details_dict,
            **extra,
        }
        return self._broadcast(msg)

    # ------------------------------------------------------------------
    # Simulation stepping (sync and non-blocking async)
    # ------------------------------------------------------------------

    def _record_tick_in_memory(self, result: TickResult) -> None:
        """Record active controller actions into the short-term memory buffer."""
        status = result.controller_status
        if not status.is_idle and status.step is not None:
            s_val = float(result.surprise_signal.score) if result.surprise_signal is not None else 0.0
            s_kind = result.surprise_signal.kind if result.surprise_signal is not None else None
            self.memory.buffer.add(
                tick=result.tick,
                observation=result.percept,
                action=status.step,
                outcome=getattr(status, "result", None),
                comfort=self.world.body.comfort,
                energy=self.world.body.energy,
                surprise=s_val,
                surprise_kind=s_kind,
                resting=False,
            )

    def step(self, ticks: int = 1) -> list[TickResult]:
        """Advance the simulation and agent loop by ``ticks`` ticks synchronously (even if paused)."""
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        was_paused = self._paused
        results: list[TickResult] = []
        with self._lock:
            try:
                for _ in range(ticks):
                    res = self.loop.tick()
                    self._record_tick_in_memory(res)
                    results.append(res)
            finally:
                if was_paused:
                    self._paused = True
                    self.clock.pause()
                    self.world.paused = True
                    self.loop.paused = True
        return results

    async def async_tick(self) -> TickResult:
        """Execute one tick of the simulation loop without blocking the event loop on LLM calls."""
        loop = self.loop
        with self._lock:
            current_tick = self.world.tick
            if loop.perception_tracker is not None:
                percept = loop.perception_tracker.perceive(self.world)
            else:
                percept = perceive(self.world)

            surprise_signal: SurpriseSignal | None = None
            raw_world_surprise = getattr(
                self.world, "surprise", getattr(self.world, "surprise_score", None)
            )
            if raw_world_surprise is not None:
                surprise_val = float(raw_world_surprise or 0.0)
            elif loop.surprise_scorer is not None:
                ended_ev = loop._pending_intent_ended_event
                if loop._pending_intent_ended and ended_ev is not None and ended_ev.intent is not None:
                    surprise_signal = loop.surprise_scorer.score_expectation(
                        ended_ev.intent,
                        ended_ev,
                        after_state=self.world,
                        world=self.world,
                    )
                elif loop.belief_store is not None:
                    surprise_signal = loop.surprise_scorer.score_belief(
                        percept.to_text(),
                        loop.belief_store,
                        world=self.world,
                    )
                else:
                    surprise_signal = loop.surprise_scorer.score_physical(
                        self.world,
                        world=self.world,
                    )
                surprise_val = float(surprise_signal.error_score)
            else:
                surprise_val = 0.0

            context = GateContext(
                tick=current_tick,
                percept=percept,
                world=self.world,
                controller=loop.controller,
                controller_status=loop._last_controller_status,
                active_intent=loop.controller.intent,
                is_start=not loop._has_started,
                intent_ended=loop._pending_intent_ended,
                intent_ended_event=loop._pending_intent_ended_event,
                ticks_since_wake=loop._ticks_since_wake,
                events=list(loop._tick_events),
                surprise=surprise_val,
            )
            loop._has_started = True
            loop._pending_intent_ended = False
            loop._pending_intent_ended_event = None
            loop._tick_events.clear()

            decision = _coerce_wake_decision(loop.gate.should_wake(context))
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

        if decision.wake:
            with self._lock:
                loop._ticks_since_wake = 0
                wake_event = WakeEvent(
                    tick=current_tick,
                    source="gate",
                    reasons=list(decision.reasons),
                    decision=decision,
                    percept=percept,
                )
                loop.wake_events.append(wake_event)
                self.world.events.emit(wake_event)
                loop._pause_world()

                effective_self_model: str | SelfModel | None = loop.self_model
                if (
                    effective_self_model in (None, "", "none")
                    and loop.self_model_store is not None
                    and loop.self_model_store.text
                ):
                    effective_self_model = loop.self_model_store.text
                elif (
                    effective_self_model in (None, "", "none")
                    and loop.memory is not None
                    and loop.memory.self_model
                ):
                    effective_self_model = loop.memory.self_model

                context_tags = extract_context_tags(percept)
                retrieved = retrieve(
                    context_tags,
                    loop.retrieval_k,
                    episode_store=loop.episode_store,
                    belief_store=loop.belief_store,
                    manager=loop.memory,
                    memories=loop.memories if loop.memories else None,
                    current_tick=current_tick,
                )
                loop.last_retrieved_memories = list(retrieved)
                prompt_memories = (
                    retrieved if retrieved else (loop.memories if loop.memories else None)
                )
                prompt = build_decide_prompt(
                    percept=percept,
                    self_model=effective_self_model,
                    memories=prompt_memories,
                    body_state=percept.body,
                    previous_intent_result=loop._previous_intent_result_payload(),
                    context_tags=context_tags,
                    k=loop.retrieval_k,
                )
                built_prompt = prompt
                loop.last_decide_prompt = prompt
                schema = Intent.model_json_schema()

            try:
                # Offload blocking LLM call to a worker thread so the event loop stays responsive
                raw_intent = await asyncio.to_thread(
                    loop.llm.complete_json,
                    "decide",
                    prompt.system,
                    prompt.user,
                    schema,
                    loop.timeout_s,
                )
                if isinstance(raw_intent, Intent):
                    new_intent = raw_intent
                else:
                    try:
                        new_intent = Intent.model_validate(raw_intent)
                    except Exception as val_exc:
                        raw_str = (
                            json.dumps(raw_intent, sort_keys=True)
                            if isinstance(raw_intent, Mapping)
                            else str(raw_intent)
                        )
                        raise LLMOutputError(raw_str, str(val_exc)) from val_exc
            except (LLMOutputError, TimeoutError, OSError) as exc:
                llm_error = exc
                with self._lock:
                    kept_intent = loop.controller.intent
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
                    loop.llm_error_events.append(err_event)
                    self.world.events.emit(err_event)
                    kept_goal = kept_intent.goal if kept_intent is not None else "idle"
                    summary = format_wake_summary(
                        current_tick,
                        decision.reasons,
                        kept_goal,
                        f"llm_error: {exc}",
                    )
            else:
                with self._lock:
                    loop.controller.set_intent(new_intent)
                    if hasattr(loop.gate, "set_wake_if"):
                        loop.gate.set_wake_if(new_intent.wake_if)
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
                    loop.intent_set_events.append(intent_event)
                    self.world.events.emit(intent_event)
                    summary = format_wake_summary(
                        current_tick,
                        decision.reasons,
                        new_intent.goal,
                        new_intent.thought,
                    )
            finally:
                with self._lock:
                    if not self._paused:
                        loop._resume_world()
                    else:
                        self.clock.pause()
                        self.world.paused = True
                        loop.paused = True

            if summary is not None:
                with self._lock:
                    loop.wake_summaries.append(summary)
                    if loop.on_wake is not None:
                        loop.on_wake(summary)
        else:
            with self._lock:
                loop._ticks_since_wake += 1

        with self._lock:
            prev_auto_reflect = self.memory.buffer.auto_reflect
            self.memory.buffer.auto_reflect = False
            try:
                controller_status = loop.controller.tick(self.world)
                loop._last_controller_status = controller_status
                if controller_status.done:
                    loop._pending_intent_ended = True
                    if controller_status.event is not None:
                        loop._pending_intent_ended_event = controller_status.event
                        loop._last_completed_intent_event = controller_status.event

                self.clock.step_once()
                loop.ticks_run += 1

                tick_res = TickResult(
                    tick=current_tick,
                    percept=percept,
                    decision=decision,
                    controller_status=controller_status,
                    intent=(
                        loop.controller.intent
                        if not controller_status.done
                        else controller_status.intent
                    ),
                    llm_error=llm_error,
                    summary=summary,
                    prompt=built_prompt,
                    surprise_signal=surprise_signal,
                )
                self._record_tick_in_memory(tick_res)
                should_run_reflect = (
                    prev_auto_reflect
                    and self.memory.llm is not None
                    and self.memory.buffer.should_reflect()
                )
            finally:
                self.memory.buffer.auto_reflect = prev_auto_reflect

        if should_run_reflect:
            try:
                await asyncio.to_thread(self.memory.reflect)
            except (LLMOutputError, TimeoutError, OSError):
                pass

        return tick_res

    async def _background_loop(self) -> None:
        """Background task that advances the simulation at ``tick_rate * speed`` while unpaused."""
        try:
            while self._running:
                interval = max(0.001, self.clock.tick_interval)
                await asyncio.sleep(interval)
                if not self._running:
                    break
                if self._paused:
                    continue
                await self.async_tick()
        except asyncio.CancelledError:
            return

    async def start_background(self) -> None:
        """Start the background simulation task if not already running."""
        if self._bg_task is not None and not self._bg_task.done():
            return
        self._running = True
        self._bg_task = asyncio.create_task(self._background_loop())

    async def stop_background(self) -> None:
        """Stop the background simulation task cleanly."""
        self._running = False
        if self._bg_task is not None:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except (asyncio.CancelledError, Exception):
                pass
            self._bg_task = None

    # ------------------------------------------------------------------
    # Read state payloads
    # ------------------------------------------------------------------

    def get_snapshot_payload(self) -> dict[str, Any]:
        """Return the full world snapshot as a JSON-serializable dictionary."""
        with self._lock:
            raw = json.loads(self.world.snapshot())
            raw["schema_version"] = SCHEMA_VERSION
            raw["paused"] = self._paused
            raw["speed"] = self.clock.speed
            raw["tick_rate"] = self.clock.tick_rate
            raw["effective_tick_rate"] = self.clock.effective_tick_rate
            raw["creator_labels"] = self.world.creator_labels
            raw["time_of_day"] = self.world.time_of_day
            raw["light_level"] = self.world.light_level
            raw["ambient_temperature"] = self.world.ambient_temperature
            return raw

    def get_agent_state_payload(self) -> dict[str, Any]:
        """Return the agent's body state, current intent, and last thought."""
        with self._lock:
            active_intent = self.loop.controller.intent
            effective_intent = active_intent if active_intent is not None else self.last_intent
            intent_dict = (
                effective_intent.model_dump(mode="json") if effective_intent is not None else None
            )
            thought = (
                active_intent.thought
                if active_intent is not None
                else self.last_thought
            )
            goal = (
                effective_intent.goal
                if effective_intent is not None
                else None
            )
            body_dict = self.world.body.model_dump(mode="json")
            return {
                "tick": self.world.tick,
                "body": body_dict,
                "current_intent": intent_dict,
                "intent": intent_dict,
                "last_thought": thought,
                "thought": thought,
                "goal": goal,
                "status": self.loop.controller.status.state,
                "controller_state": self.loop.controller.status.state,
                "drives": {
                    "boredom": self.world.boredom,
                    "curiosity": self.world.curiosity,
                },
                "paused": self._paused,
                "speed": self.clock.speed,
            }

    def get_inbox_payload(self) -> dict[str, Any]:
        """Return the creator inbox messages, preserved objects, and audit log."""
        with self._lock:
            entries = [entry.to_dict() for entry in self.world.inbox.all()]
            logs = [log_item.to_dict() for log_item in self.world.inbox.logs()]
            return {
                "tick": self.world.tick,
                "inbox": entries,
                "messages": entries,
                "items": entries,
                "entries": entries,
                "count": len(entries),
                "logs": logs,
                "creator_note": self.world.creator_note,
            }

    def get_beliefs_payload(self) -> dict[str, Any]:
        """Return all semantic beliefs and their confidence histories."""
        with self._lock:
            beliefs = [b.to_dict() for b in self.memory.belief_store.all()]
            return {
                "tick": self.world.tick,
                "beliefs": beliefs,
                "items": beliefs,
                "count": len(beliefs),
                "max_beliefs": self.memory.belief_store.max_beliefs,
            }

    def get_self_model_payload(self) -> dict[str, Any]:
        """Return the current self-model text and version history."""
        with self._lock:
            sm_store = self.memory.self_model_store
            current = sm_store.current_version
            history = [v.model_dump(mode="json") for v in sm_store.all()]
            text = sm_store.text
            return {
                "tick": self.world.tick,
                "self_model": text,
                "text": text,
                "version": current.version if current is not None else 0,
                "updated_tick": current.tick if current is not None else 0,
                "word_count": current.word_count if current is not None else 0,
                "max_words": sm_store.max_words,
                "history": history,
            }

    def get_weather_payload(self) -> dict[str, Any]:
        """Return the current weather state, automation status, and scheduled events."""
        with self._lock:
            w = self.world.weather
            w_dict = w.to_dict()
            paused = bool(w.paused)
            return {
                "tick": self.world.tick,
                "type": w.type,
                "weather_type": w.type,
                "intensity": w.intensity,
                "duration": w.duration,
                "remaining_ticks": w.remaining_ticks,
                "source": w.source,
                "paused": paused,
                "automation_paused": paused,
                "automation_enabled": not paused,
                "automated": not paused,
                "automation": not paused,
                "light_multiplier": w.light_multiplier,
                "temperature_drop": w.temperature_drop,
                "surface_slipperiness": w.surface_slipperiness,
                "wind_direction": w.wind_direction.model_dump(mode="json"),
                "scheduled": w_dict.get("scheduled", []),
            }

    def get_time_payload(self) -> dict[str, Any]:
        """Return the current world tick, time of day, cycle freeze state, and clock state."""
        with self._lock:
            dn = self.world.daynight
            return {
                "tick": self.world.tick,
                "time_of_day": self.world.time_of_day,
                "time": self.world.time_of_day,
                "fraction": self.world.time_of_day,
                "frozen": self.world.cycle_frozen,
                "cycle_frozen": self.world.cycle_frozen,
                "day_length": dn.config.day_length,
                "daylight_share": dn.config.daylight_share,
                "min_visibility": dn.config.min_visibility,
                "temperature_swing": dn.config.temperature_swing,
                "base_temperature": dn.config.base_temperature,
                "light_level": self.world.light_level,
                "light": self.world.light_level,
                "coarse_light_level": self.world.coarse_light_level,
                "visibility_factor": self.world.visibility_factor,
                "ambient_temperature": self.world.ambient_temperature,
                "temperature": self.world.ambient_temperature,
                "paused": self._paused,
                "speed": self.clock.speed,
                "tick_rate": self.clock.tick_rate,
                "effective_tick_rate": self.clock.effective_tick_rate,
            }

    # ------------------------------------------------------------------
    # Creator actions
    # ------------------------------------------------------------------

    def spawn_object(
        self,
        props: Mapping[str, Any] | WorldObject,
        pos: Any = None,
    ) -> WorldObject:
        """Spawn a new object in the world and broadcast the creator action."""
        with self._lock:
            if isinstance(props, WorldObject):
                obj = self.world.spawn(props, pos, source="creator")
            else:
                normalized = _ensure_spawn_defaults(props)
                obj = self.world.spawn(normalized, pos, source="creator")
            obj_dict = obj.model_dump(mode="json")
            self.emit_creator_action(
                "spawn",
                details={"object_id": obj.id, "object": obj_dict},
                object_id=obj.id,
                position=obj.position.model_dump(mode="json"),
                object=obj_dict,
                objects={obj.id: obj_dict},
                delta={
                    "event": "object_spawned",
                    "action": "spawn",
                    "object_id": obj.id,
                    "object": obj_dict,
                    "objects": {obj.id: obj_dict},
                    "spawned": [obj_dict],
                },
            )
            return obj

    def despawn_object(self, object_id: str) -> WorldObject:
        """Despawn an object by id and broadcast the creator action."""
        with self._lock:
            obj = self.world.despawn(object_id, source="creator")
            obj_dict = obj.model_dump(mode="json")
            self.emit_creator_action(
                "despawn",
                details={"object_id": obj.id, "object": obj_dict},
                object_id=obj.id,
                position=obj.position.model_dump(mode="json"),
                object=obj_dict,
                delta={
                    "event": "object_despawned",
                    "action": "despawn",
                    "object_id": obj.id,
                    "despawned": [obj.id],
                },
            )
            return obj

    def move_object(self, object_id: str, pos: Any) -> WorldObject:
        """Move an object to ``pos`` and broadcast the creator action."""
        with self._lock:
            obj = self.world.move(object_id, pos, source="creator")
            obj_dict = obj.model_dump(mode="json")
            pos_dict = obj.position.model_dump(mode="json")
            self.emit_creator_action(
                "move",
                details={"object_id": obj.id, "position": pos_dict, "object": obj_dict},
                object_id=obj.id,
                position=pos_dict,
                object=obj_dict,
                objects={obj.id: obj_dict},
                delta={
                    "event": "object_moved",
                    "action": "move",
                    "object_id": obj.id,
                    "position": pos_dict,
                    "object": obj_dict,
                    "objects": {obj.id: obj_dict},
                    "moved": [obj.id],
                },
            )
            return obj

    def reply_mail(
        self,
        text: str,
        *,
        mailbox_id: str | None = None,
        color: str | None = "white",
        note: str | None = None,
    ) -> WorldObject:
        """Place a reply letter inside the mailbox and broadcast the creator action."""
        with self._lock:
            letter = self.world.reply(
                text,
                mailbox_id=mailbox_id,
                color=color,
                note=note,
            )
            letter_dict = letter.model_dump(mode="json")
            self.emit_creator_action(
                "reply",
                details={"object_id": letter.id, "text": text, "object": letter_dict},
                object_id=letter.id,
                text=text,
                object=letter_dict,
            )
            return letter

    def spawn_in_response(
        self,
        props: Mapping[str, Any] | WorldObject | int | str,
        *,
        mailbox_id: str | None = None,
        note: str | None = None,
    ) -> WorldObject:
        """Place an object inside the mailbox in response to mail and broadcast the creator action."""
        with self._lock:
            if isinstance(props, int | str) and not isinstance(props, bool):
                spawned = self.world.return_object(props, mailbox_id=mailbox_id, note=note)
            elif isinstance(props, WorldObject):
                spawned = self.world.spawn_in_response(props, mailbox_id=mailbox_id, note=note)
            else:
                normalized = _ensure_spawn_defaults(props)
                spawned = self.world.spawn_in_response(normalized, mailbox_id=mailbox_id, note=note)
            spawned_dict = spawned.model_dump(mode="json")
            self.emit_creator_action(
                "spawn_in_response",
                details={"object_id": spawned.id, "object": spawned_dict},
                object_id=spawned.id,
                object=spawned_dict,
            )
            return spawned

    def set_weather(
        self,
        weather_type: str | None = None,
        *,
        intensity: float | None = None,
        duration: int | None = None,
        start_at: int | None = None,
        wind_direction: Any = None,
    ) -> dict[str, Any]:
        """Set or schedule weather and broadcast the creator action."""
        with self._lock:
            self.world.set_weather(
                weather_type,  # type: ignore[arg-type]
                intensity=intensity,
                duration=duration,
                start_at=start_at,
                wind_direction=wind_direction,
            )
            payload = self.get_weather_payload()
            self.emit_creator_action(
                "set_weather",
                details={
                    "weather_type": weather_type,
                    "intensity": intensity,
                    "duration": duration,
                    "start_at": start_at,
                    "weather": payload,
                },
                weather=payload,
            )
            return payload

    def set_weather_automation(self, enabled: bool | None = None) -> dict[str, Any]:
        """Enable, disable, or toggle weather automation and broadcast the creator action."""
        with self._lock:
            if enabled is None:
                enabled = bool(self.world.weather.paused)
            self.world.set_weather_automation(bool(enabled))
            payload = self.get_weather_payload()
            self.emit_creator_action(
                "weather_automation",
                details={"automation_enabled": not self.world.weather.paused, "weather": payload},
                weather=payload,
            )
            return payload

    def set_time_of_day(
        self,
        fraction: float | None = None,
        *,
        frozen: bool | None = None,
    ) -> dict[str, Any]:
        """Set time of day and/or freeze cycle, and broadcast the creator action."""
        with self._lock:
            if fraction is not None:
                self.world.set_time_of_day(float(fraction))
            if frozen is not None:
                self.world.freeze_cycle(bool(frozen))
            payload = self.get_time_payload()
            action_name = (
                "set_time_of_day"
                if fraction is not None
                else ("freeze_cycle" if frozen is not None else "set_time_of_day")
            )
            self.emit_creator_action(
                action_name,
                details={
                    "time_of_day": self.world.time_of_day,
                    "frozen": self.world.cycle_frozen,
                    "time": payload,
                },
                time=payload,
            )
            return payload

    def pause(self) -> dict[str, Any]:
        """Pause simulation stepping and broadcast the creator action."""
        with self._lock:
            self._paused = True
            self.clock.pause()
            self.world.paused = True
            self.loop.paused = True
            payload = self._control_payload()
            self.emit_creator_action("pause", details=payload, paused=True)
            return payload

    def resume(self) -> dict[str, Any]:
        """Resume simulation stepping and broadcast the creator action."""
        with self._lock:
            self._paused = False
            self.clock.resume()
            self.world.paused = False
            self.loop.paused = False
            payload = self._control_payload()
            self.emit_creator_action("resume", details=payload, paused=False)
            return payload

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Set simulation speed multiplier and broadcast the creator action."""
        with self._lock:
            self.clock.set_speed(float(speed))
            payload = self._control_payload()
            self.emit_creator_action("set_speed", details=payload, speed=self.clock.speed)
            return payload

    def _control_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "tick": self.world.tick,
            "paused": self._paused,
            "speed": self.clock.speed,
            "tick_rate": self.clock.tick_rate,
            "effective_tick_rate": self.clock.effective_tick_rate,
        }


def create_app(
    world: World | Path | str | Mapping[str, Any] | None = None,
    *,
    settings: Settings | Path | str | Mapping[str, Any] | None = None,
    config: Settings | Path | str | Mapping[str, Any] | None = None,
    llm: LLMBackend | str | None = None,
    backend: LLMBackend | str | None = None,
    service: SimulationService | None = None,
    loop: AgentLoop | None = None,
    gate: Gate | None = None,
    controller: Controller | None = None,
    clock: WorldClock | None = None,
    memory: MemoryManager | None = None,
    episode_store: EpisodeStore | None = None,
    belief_store: BeliefStore | None = None,
    self_model: SelfModel | str | None = None,
    inbox: CreatorInbox | None = None,
    rng: random.Random | int | None = None,
    host: str | None = None,
    port: int | None = None,
    creator_token: str | None = None,
    token: str | None = None,
    auth_token: str | None = None,
    api_token: str | None = None,
    spectator_enabled: bool | None = None,
    spectator: bool | None = None,
    allow_spectators: bool | None = None,
    spectator_access: bool | None = None,
    spectator_mode: bool | None = None,
    read_only: bool | None = None,
    spectator_only: bool | None = None,
    auto_start: bool | None = None,
    paused: bool | None = None,
    db_path: Path | str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for the Arden simulation server."""
    eff_token = (
        creator_token
        if creator_token is not None
        else (token if token is not None else (auth_token if auth_token is not None else api_token))
    )
    eff_spectator_enabled = (
        spectator_enabled
        if spectator_enabled is not None
        else (
            spectator
            if spectator is not None
            else (allow_spectators if allow_spectators is not None else spectator_access)
        )
    )
    eff_spectator_mode = (
        spectator_mode
        if spectator_mode is not None
        else (read_only if read_only is not None else spectator_only)
    )

    sim_service = (
        service
        if service is not None
        else SimulationService(
            world=world,
            settings=settings,
            config=config,
            llm=llm,
            backend=backend,
            loop=loop,
            gate=gate,
            controller=controller,
            clock=clock,
            memory=memory,
            episode_store=episode_store,
            belief_store=belief_store,
            self_model=self_model,
            inbox=inbox,
            rng=rng,
            host=host,
            port=port,
            creator_token=eff_token,
            spectator_enabled=eff_spectator_enabled,
            spectator_mode=eff_spectator_mode,
            auto_start=auto_start,
            paused=paused,
            db_path=db_path,
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if sim_service.settings.server.auto_start:
            await sim_service.start_background()
        try:
            yield
        finally:
            await sim_service.stop_background()

    app = FastAPI(
        title="Arden Simulation Server",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service = sim_service
    app.state.world = sim_service.world
    app.state.loop = sim_service.loop
    app.state.clock = sim_service.clock
    app.state.memory = sim_service.memory
    app.state.settings = sim_service.settings
    app.service = sim_service  # type: ignore[attr-defined]
    app.world = sim_service.world  # type: ignore[attr-defined]
    app.loop = sim_service.loop  # type: ignore[attr-defined]

    def _require_read_access(request: Request) -> None:
        """Enforce read access: allowed if spectator_enabled is True, else requires creator token."""
        if sim_service.spectator_enabled:
            return
        expected = sim_service.creator_token
        provided = _extract_request_token(request)
        if not provided:
            raise HTTPException(
                status_code=401,
                detail="Spectator access is disabled; creator token required.",
            )
        if not expected or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=403,
                detail="Invalid creator token.",
            )

    def _require_creator_access(request: Request) -> None:
        """Enforce creator authentication and reject spectator mutations."""
        if sim_service.spectator_mode or _is_spectator_request(request):
            raise HTTPException(
                status_code=403,
                detail="Spectator mode cannot mutate.",
            )
        expected = sim_service.creator_token
        provided = _extract_request_token(request)
        if not provided:
            raise HTTPException(
                status_code=401,
                detail="Creator token required.",
            )
        if not expected or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=403,
                detail="Invalid creator token.",
            )

    # ------------------------------------------------------------------
    # Read endpoints
    # ------------------------------------------------------------------

    @app.get("/", dependencies=[Depends(_require_read_access)])
    @app.get("/health", dependencies=[Depends(_require_read_access)])
    @app.get("/status", dependencies=[Depends(_require_read_access)])
    @app.get("/api/status", dependencies=[Depends(_require_read_access)])
    async def get_status() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "tick": sim_service.world.tick,
            "paused": sim_service.paused,
            "speed": sim_service.speed,
            "spectator_enabled": sim_service.spectator_enabled,
            "spectator_mode": sim_service.spectator_mode,
        }

    @app.get("/snapshot", dependencies=[Depends(_require_read_access)])
    @app.get("/world", dependencies=[Depends(_require_read_access)])
    @app.get("/world/snapshot", dependencies=[Depends(_require_read_access)])
    @app.get("/state", dependencies=[Depends(_require_read_access)])
    @app.get("/api/snapshot", dependencies=[Depends(_require_read_access)])
    @app.get("/api/world", dependencies=[Depends(_require_read_access)])
    @app.get("/api/world/snapshot", dependencies=[Depends(_require_read_access)])
    async def get_snapshot() -> dict[str, Any]:
        return sim_service.get_snapshot_payload()

    @app.get("/objects", dependencies=[Depends(_require_read_access)])
    @app.get("/world/objects", dependencies=[Depends(_require_read_access)])
    @app.get("/api/objects", dependencies=[Depends(_require_read_access)])
    async def get_objects() -> dict[str, Any]:
        with sim_service._lock:
            objs = {
                oid: sim_service.world.objects[oid].model_dump(mode="json")
                for oid in sorted(sim_service.world.objects)
            }
            return {
                "tick": sim_service.world.tick,
                "objects": objs,
                "creator_labels": sim_service.world.creator_labels,
                "count": len(objs),
            }

    @app.get("/objects/{object_id}", dependencies=[Depends(_require_read_access)])
    @app.get("/world/objects/{object_id}", dependencies=[Depends(_require_read_access)])
    @app.get("/api/objects/{object_id}", dependencies=[Depends(_require_read_access)])
    async def get_object(object_id: str) -> dict[str, Any]:
        with sim_service._lock:
            if object_id not in sim_service.world.objects:
                raise HTTPException(status_code=404, detail=f"Unknown object id: {object_id}")
            obj = sim_service.world.objects[object_id]
            return obj.model_dump(mode="json")

    @app.get("/agent", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/state", dependencies=[Depends(_require_read_access)])
    @app.get("/api/agent", dependencies=[Depends(_require_read_access)])
    @app.get("/api/agent/state", dependencies=[Depends(_require_read_access)])
    async def get_agent_state() -> dict[str, Any]:
        return sim_service.get_agent_state_payload()

    @app.get("/inbox", dependencies=[Depends(_require_read_access)])
    @app.get("/mailbox/inbox", dependencies=[Depends(_require_read_access)])
    @app.get("/mailbox", dependencies=[Depends(_require_read_access)])
    @app.get("/mail", dependencies=[Depends(_require_read_access)])
    @app.get("/api/inbox", dependencies=[Depends(_require_read_access)])
    @app.get("/api/mailbox/inbox", dependencies=[Depends(_require_read_access)])
    async def get_inbox() -> dict[str, Any]:
        return sim_service.get_inbox_payload()

    @app.get("/beliefs", dependencies=[Depends(_require_read_access)])
    @app.get("/memory/beliefs", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/beliefs", dependencies=[Depends(_require_read_access)])
    @app.get("/api/beliefs", dependencies=[Depends(_require_read_access)])
    async def get_beliefs() -> dict[str, Any]:
        return sim_service.get_beliefs_payload()

    @app.get("/episodes", dependencies=[Depends(_require_read_access)])
    @app.get("/memory/episodes", dependencies=[Depends(_require_read_access)])
    @app.get("/memory", dependencies=[Depends(_require_read_access)])
    @app.get("/api/episodes", dependencies=[Depends(_require_read_access)])
    async def get_episodes() -> dict[str, Any]:
        with sim_service._lock:
            episodes = [ep.to_dict() for ep in sim_service.memory.store.all()]
            return {
                "tick": sim_service.world.tick,
                "episodes": episodes,
                "items": episodes,
                "count": len(episodes),
            }

    @app.get("/self-model", dependencies=[Depends(_require_read_access)])
    @app.get("/self_model", dependencies=[Depends(_require_read_access)])
    @app.get("/memory/self-model", dependencies=[Depends(_require_read_access)])
    @app.get("/memory/self_model", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/self-model", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/self_model", dependencies=[Depends(_require_read_access)])
    @app.get("/api/self-model", dependencies=[Depends(_require_read_access)])
    @app.get("/api/self_model", dependencies=[Depends(_require_read_access)])
    async def get_self_model() -> dict[str, Any]:
        return sim_service.get_self_model_payload()

    @app.get("/weather", dependencies=[Depends(_require_read_access)])
    @app.get("/world/weather", dependencies=[Depends(_require_read_access)])
    @app.get("/environment/weather", dependencies=[Depends(_require_read_access)])
    @app.get("/api/weather", dependencies=[Depends(_require_read_access)])
    async def get_weather() -> dict[str, Any]:
        return sim_service.get_weather_payload()

    @app.get("/time", dependencies=[Depends(_require_read_access)])
    @app.get("/world/time", dependencies=[Depends(_require_read_access)])
    @app.get("/daynight", dependencies=[Depends(_require_read_access)])
    @app.get("/time-of-day", dependencies=[Depends(_require_read_access)])
    @app.get("/time_of_day", dependencies=[Depends(_require_read_access)])
    @app.get("/clock", dependencies=[Depends(_require_read_access)])
    @app.get("/api/time", dependencies=[Depends(_require_read_access)])
    async def get_time() -> dict[str, Any]:
        return sim_service.get_time_payload()

    # ------------------------------------------------------------------
    # Creator endpoints
    # ------------------------------------------------------------------

    @app.post("/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/objects/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/objects", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/objects/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/objects", dependencies=[Depends(_require_creator_access)])
    async def creator_spawn(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        raw = dict(body)
        pos = raw.pop("pos", None)
        if "props" in raw and isinstance(raw["props"], Mapping):
            props = dict(raw["props"])
            if "position" in raw and "position" not in props:
                props["position"] = raw["position"]
        elif "object" in raw and isinstance(raw["object"], Mapping):
            props = dict(raw["object"])
            if "position" in raw and "position" not in props:
                props["position"] = raw["position"]
        elif "properties" in raw and isinstance(raw["properties"], Mapping):
            props = dict(raw["properties"])
            if "position" in raw and "position" not in props:
                props["position"] = raw["position"]
        else:
            props = raw

        try:
            obj = sim_service.spawn_object(props, pos=pos)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        obj_dict = obj.model_dump(mode="json")
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "object_id": obj.id,
            "id": obj.id,
            "position": obj.position.model_dump(mode="json"),
            "object": obj_dict,
        }

    async def _handle_despawn(object_id: str | None, request: Request) -> dict[str, Any]:
        target_id = object_id or request.query_params.get("object_id") or request.query_params.get("id")
        if not target_id:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, Mapping):
                target_id = (
                    body.get("object_id")
                    or body.get("id")
                    or body.get("target")
                )
        if not target_id or not isinstance(target_id, str):
            raise HTTPException(status_code=400, detail="object_id is required")
        try:
            obj = sim_service.despawn_object(target_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "object_id": obj.id,
            "id": obj.id,
            "object": obj.model_dump(mode="json"),
        }

    @app.post("/despawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/objects/despawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/despawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/despawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/despawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/objects/despawn", dependencies=[Depends(_require_creator_access)])
    async def creator_despawn_body(request: Request) -> dict[str, Any]:
        return await _handle_despawn(None, request)

    @app.post("/objects/{object_id}/despawn", dependencies=[Depends(_require_creator_access)])
    @app.delete("/objects/{object_id}", dependencies=[Depends(_require_creator_access)])
    @app.delete("/despawn/{object_id}", dependencies=[Depends(_require_creator_access)])
    @app.delete("/world/objects/{object_id}", dependencies=[Depends(_require_creator_access)])
    @app.delete("/api/objects/{object_id}", dependencies=[Depends(_require_creator_access)])
    async def creator_despawn_path(object_id: str, request: Request) -> dict[str, Any]:
        return await _handle_despawn(object_id, request)

    async def _handle_move(object_id: str | None, request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        target_id = (
            object_id
            or body.get("object_id")
            or body.get("id")
            or body.get("target")
            or request.query_params.get("object_id")
        )
        if not target_id or not isinstance(target_id, str):
            raise HTTPException(status_code=400, detail="object_id is required")

        pos = body.get("position", body.get("pos", body.get("to")))
        if pos is None and "x" in body and "y" in body:
            pos = {"x": body["x"], "y": body["y"]}
        if pos is None:
            raise HTTPException(status_code=400, detail="position is required")

        try:
            obj = sim_service.move_object(target_id, pos)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "object_id": obj.id,
            "id": obj.id,
            "position": obj.position.model_dump(mode="json"),
            "object": obj.model_dump(mode="json"),
        }

    @app.post("/move", dependencies=[Depends(_require_creator_access)])
    @app.post("/objects/move", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/move", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/move", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/move", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/objects/move", dependencies=[Depends(_require_creator_access)])
    async def creator_move_body(request: Request) -> dict[str, Any]:
        return await _handle_move(None, request)

    @app.post("/objects/{object_id}/move", dependencies=[Depends(_require_creator_access)])
    @app.put("/objects/{object_id}/move", dependencies=[Depends(_require_creator_access)])
    @app.patch("/objects/{object_id}", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/objects/{object_id}/move", dependencies=[Depends(_require_creator_access)])
    async def creator_move_path(object_id: str, request: Request) -> dict[str, Any]:
        return await _handle_move(object_id, request)

    @app.post("/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/inbox/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/mailbox/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/mail/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/inbox/reply", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/mailbox/reply", dependencies=[Depends(_require_creator_access)])
    async def creator_reply(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, str):
            text = body
            mailbox_id = None
            color = "white"
            note = None
        elif isinstance(body, Mapping):
            text = body.get("text", body.get("message", body.get("reply")))
            mailbox_id = body.get("mailbox_id")
            color = body.get("color", "white")
            note = body.get("note", body.get("creator_note"))
        else:
            raise HTTPException(status_code=400, detail="Invalid reply payload")

        if text is None or not str(text).strip():
            raise HTTPException(status_code=400, detail="Reply text must be a non-empty string")

        try:
            letter = sim_service.reply_mail(
                str(text),
                mailbox_id=mailbox_id,
                color=color,
                note=note,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "object_id": letter.id,
            "id": letter.id,
            "text": str(text),
            "object": letter.model_dump(mode="json"),
        }

    @app.post("/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    @app.post("/spawn_in_response", dependencies=[Depends(_require_creator_access)])
    @app.post("/inbox/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/inbox/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    @app.post("/inbox/spawn_in_response", dependencies=[Depends(_require_creator_access)])
    @app.post("/mailbox/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/mailbox/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    @app.post("/mailbox/spawn_in_response", dependencies=[Depends(_require_creator_access)])
    @app.post("/mail/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/spawn_in_response", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/inbox/spawn", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/mailbox/spawn-in-response", dependencies=[Depends(_require_creator_access)])
    async def creator_spawn_in_response(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        raw = dict(body)
        mailbox_id = raw.pop("mailbox_id", None)
        note = raw.pop("note", raw.pop("creator_note", None))

        if "entry_id" in raw or "inbox_id" in raw:
            target_ref: Any = raw.get("entry_id", raw.get("inbox_id"))
        elif "props" in raw and isinstance(raw["props"], Mapping):
            target_ref = dict(raw["props"])
        elif "object" in raw and isinstance(raw["object"], Mapping):
            target_ref = dict(raw["object"])
        elif "properties" in raw and isinstance(raw["properties"], Mapping):
            target_ref = dict(raw["properties"])
        else:
            target_ref = raw

        try:
            spawned = sim_service.spawn_in_response(
                target_ref,
                mailbox_id=mailbox_id,
                note=note,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "object_id": spawned.id,
            "id": spawned.id,
            "object": spawned.model_dump(mode="json"),
        }

    @app.post("/weather", dependencies=[Depends(_require_creator_access)])
    @app.put("/weather", dependencies=[Depends(_require_creator_access)])
    @app.patch("/weather", dependencies=[Depends(_require_creator_access)])
    @app.post("/weather/set", dependencies=[Depends(_require_creator_access)])
    @app.post("/weather/schedule", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/weather", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/weather", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/weather", dependencies=[Depends(_require_creator_access)])
    async def creator_set_weather(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        w_type = body.get("type", body.get("weather_type", body.get("weather")))
        intensity = body.get("intensity")
        duration = body.get("duration")
        start_at = body.get(
            "start_at",
            body.get("schedule_at", body.get("at_tick", body.get("tick"))),
        )
        wind_direction = body.get("wind_direction")

        auto_val: bool | None = None
        for key in ("automation", "automation_enabled", "automated", "enabled"):
            if key in body and body[key] is not None:
                auto_val = bool(body[key])
                break
        if auto_val is None:
            for key in ("paused", "automation_paused"):
                if key in body and body[key] is not None:
                    auto_val = not bool(body[key])
                    break

        try:
            payload: dict[str, Any] | None = None
            if w_type is not None:
                payload = sim_service.set_weather(
                    str(w_type),
                    intensity=float(intensity) if intensity is not None else None,
                    duration=int(duration) if duration is not None else None,
                    start_at=int(start_at) if start_at is not None else None,
                    wind_direction=wind_direction,
                )
            if auto_val is not None:
                payload = sim_service.set_weather_automation(auto_val)
            if payload is None:
                raise HTTPException(
                    status_code=400,
                    detail="Specify weather type or automation state",
                )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/weather/automation", dependencies=[Depends(_require_creator_access)])
    @app.put("/weather/automation", dependencies=[Depends(_require_creator_access)])
    @app.patch("/weather/automation", dependencies=[Depends(_require_creator_access)])
    @app.post("/weather/toggle-automation", dependencies=[Depends(_require_creator_access)])
    @app.post("/weather/toggle_automation", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/weather/automation", dependencies=[Depends(_require_creator_access)])
    async def creator_toggle_weather_automation(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        enabled: bool | None = None
        if isinstance(body, Mapping):
            for key in ("enabled", "automation", "automation_enabled", "automated"):
                if key in body and body[key] is not None:
                    enabled = bool(body[key])
                    break
            if enabled is None:
                for key in ("paused", "automation_paused"):
                    if key in body and body[key] is not None:
                        enabled = not bool(body[key])
                        break

        payload = sim_service.set_weather_automation(enabled)
        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/weather/pause", dependencies=[Depends(_require_creator_access)])
    async def creator_pause_weather() -> dict[str, Any]:
        payload = sim_service.set_weather_automation(False)
        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/weather/resume", dependencies=[Depends(_require_creator_access)])
    async def creator_resume_weather() -> dict[str, Any]:
        payload = sim_service.set_weather_automation(True)
        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/time", dependencies=[Depends(_require_creator_access)])
    @app.put("/time", dependencies=[Depends(_require_creator_access)])
    @app.patch("/time", dependencies=[Depends(_require_creator_access)])
    @app.post("/time/set", dependencies=[Depends(_require_creator_access)])
    @app.post("/daynight", dependencies=[Depends(_require_creator_access)])
    @app.post("/time-of-day", dependencies=[Depends(_require_creator_access)])
    @app.post("/time_of_day", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/time", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/time", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/time", dependencies=[Depends(_require_creator_access)])
    async def creator_set_time(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        fraction = body.get("time_of_day", body.get("time", body.get("fraction", body.get("value"))))
        frozen = body.get("frozen", body.get("freeze", body.get("cycle_frozen")))
        if fraction is None and frozen is None:
            raise HTTPException(status_code=400, detail="Specify time_of_day or frozen")

        try:
            payload = sim_service.set_time_of_day(
                float(fraction) if fraction is not None else None,
                frozen=bool(frozen) if frozen is not None else None,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/time/freeze", dependencies=[Depends(_require_creator_access)])
    @app.put("/time/freeze", dependencies=[Depends(_require_creator_access)])
    @app.post("/daynight/freeze", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/time/freeze", dependencies=[Depends(_require_creator_access)])
    async def creator_freeze_time(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        frozen = True
        if isinstance(body, Mapping):
            raw = body.get("frozen", body.get("freeze", body.get("cycle_frozen")))
            if raw is not None:
                frozen = bool(raw)
        payload = sim_service.set_time_of_day(frozen=frozen)
        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/time/unfreeze", dependencies=[Depends(_require_creator_access)])
    @app.post("/daynight/unfreeze", dependencies=[Depends(_require_creator_access)])
    async def creator_unfreeze_time() -> dict[str, Any]:
        payload = sim_service.set_time_of_day(frozen=False)
        await asyncio.sleep(0)
        return {"ok": True, **payload}

    @app.post("/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/simulation/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/clock/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/control/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/pause", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/simulation/pause", dependencies=[Depends(_require_creator_access)])
    async def creator_pause() -> dict[str, Any]:
        payload = sim_service.pause()
        await asyncio.sleep(0)
        return payload

    @app.post("/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/simulation/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/clock/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/control/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/resume", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/simulation/resume", dependencies=[Depends(_require_creator_access)])
    async def creator_resume() -> dict[str, Any]:
        payload = sim_service.resume()
        await asyncio.sleep(0)
        return payload

    @app.post("/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/simulation/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/clock/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/control/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/step", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/simulation/step", dependencies=[Depends(_require_creator_access)])
    async def creator_step(request: Request) -> dict[str, Any]:
        ticks = 1
        q_ticks = request.query_params.get("ticks", request.query_params.get("steps"))
        if q_ticks is not None:
            try:
                ticks = int(q_ticks)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="ticks must be an integer") from exc
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, Mapping):
                raw_t = body.get("ticks", body.get("steps", body.get("count", body.get("n"))))
                if raw_t is not None:
                    try:
                        ticks = int(raw_t)
                    except (ValueError, TypeError) as exc:
                        raise HTTPException(
                            status_code=400, detail="ticks must be an integer"
                        ) from exc

        if ticks < 1:
            raise HTTPException(status_code=400, detail="ticks must be >= 1")

        was_paused = sim_service.paused
        try:
            for _ in range(ticks):
                await sim_service.async_tick()
        finally:
            if was_paused:
                sim_service._paused = True
                sim_service.clock.pause()
                sim_service.world.paused = True
                sim_service.loop.paused = True

        payload = sim_service._control_payload()
        payload["steps"] = ticks
        payload["ticks_stepped"] = ticks
        sim_service.emit_creator_action("step", details=payload, ticks=ticks)
        await asyncio.sleep(0)
        return payload

    @app.post("/speed", dependencies=[Depends(_require_creator_access)])
    @app.put("/speed", dependencies=[Depends(_require_creator_access)])
    @app.patch("/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/simulation/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/clock/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/world/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/control/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/speed", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/simulation/speed", dependencies=[Depends(_require_creator_access)])
    async def creator_set_speed(request: Request) -> dict[str, Any]:
        speed_val: Any = request.query_params.get("speed", request.query_params.get("multiplier"))
        if speed_val is None:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, int | float) and not isinstance(body, bool):
                speed_val = body
            elif isinstance(body, Mapping):
                speed_val = body.get(
                    "speed",
                    body.get("multiplier", body.get("rate", body.get("value"))),
                )
        if speed_val is None:
            raise HTTPException(status_code=400, detail="speed is required")
        try:
            payload = sim_service.set_speed(float(speed_val))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return payload

    @app.post("/reflect", dependencies=[Depends(_require_creator_access)])
    @app.post("/agent/reflect", dependencies=[Depends(_require_creator_access)])
    @app.post("/memory/reflect", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/reflect", dependencies=[Depends(_require_creator_access)])
    async def creator_reflect() -> dict[str, Any]:
        reflected = await asyncio.to_thread(sim_service.memory.reflect)
        await asyncio.sleep(0)
        return {
            "ok": True,
            "tick": sim_service.world.tick,
            "episodes": [ep.to_dict() for ep in reflected.episodes],
            "beliefs": [b.to_dict() for b in reflected.beliefs],
            "self_model": reflected.self_model,
        }

    # ------------------------------------------------------------------
    # WebSocket stream (/ws)
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_stream(websocket: WebSocket) -> None:
        if not sim_service.spectator_enabled:
            expected = sim_service.creator_token
            provided = _extract_websocket_token(websocket)
            if not provided or not expected or not secrets.compare_digest(provided, expected):
                await websocket.close(code=1008)
                return

        await websocket.accept()
        ev_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        sim_service.register_ws_queue(ev_loop, queue)

        send_initial = websocket.query_params.get("snapshot", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if send_initial:
            await websocket.send_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": "snapshot",
                    "event": "snapshot",
                    "tick": sim_service.world.tick,
                    "snapshot": sim_service.get_snapshot_payload(),
                }
            )

        receiver_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(websocket.receive())
        sender_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(queue.get())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    {receiver_task, sender_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if sender_task in done:
                    msg = sender_task.result()
                    await websocket.send_json(msg)
                    sender_task = asyncio.create_task(queue.get())

                if receiver_task in done:
                    ws_event = receiver_task.result()
                    if ws_event.get("type") == "websocket.disconnect":
                        break
                    # Spectators or unauthenticated WS clients cannot mutate via WebSocket
                    await websocket.send_json(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "type": "error",
                            "error": "forbidden",
                            "detail": "WebSocket stream is read-only; use authenticated HTTP creator endpoints to mutate.",
                        }
                    )
                    receiver_task = asyncio.create_task(websocket.receive())
        except WebSocketDisconnect:
            pass
        finally:
            receiver_task.cancel()
            sender_task.cancel()
            for t in (receiver_task, sender_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            sim_service.unregister_ws_queue(queue)

    return app


make_app = create_app
build_app = create_app
get_app = create_app
SimulationServer = SimulationService


def run_server(
    settings: Settings | None = None,
    rng: random.Random | None = None,
    *,
    world: World | Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    creator_token: str | None = None,
    spectator_enabled: bool | None = None,
    llm: LLMBackend | str | None = None,
) -> FastAPI:
    """Validate binding, construct the FastAPI app, and run uvicorn."""
    cfg = settings.model_copy(deep=True) if settings is not None else Settings()
    if host is not None:
        cfg.server.host = str(host).strip()
    if port is not None:
        cfg.server.port = int(port)
    if creator_token is not None:
        cfg.server.creator_token = str(creator_token).strip() or None
    if spectator_enabled is not None:
        cfg.server.spectator_enabled = bool(spectator_enabled)
    if isinstance(llm, str):
        cfg.llm.backend = llm

    resolved_token = resolve_creator_token(creator_token, cfg)
    validate_server_binding(cfg.server.host, resolved_token)

    app = create_app(
        world=world,
        settings=cfg,
        rng=rng,
        llm=llm if not isinstance(llm, str) else None,
        host=cfg.server.host,
        port=cfg.server.port,
        creator_token=resolved_token,
        spectator_enabled=cfg.server.spectator_enabled,
    )
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.logging.level.lower(),
    )
    return app


serve = run_server

__all__ = [
    "LOCAL_HOSTS",
    "SCHEMA_VERSION",
    "WS_SCHEMA_VERSION",
    "SimulationServer",
    "SimulationService",
    "build_app",
    "create_app",
    "get_app",
    "is_local_host",
    "make_app",
    "resolve_creator_token",
    "run_server",
    "serve",
    "validate_server_binding",
]
