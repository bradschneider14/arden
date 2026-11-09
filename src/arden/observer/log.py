"""Append-only observer log subscribing to the world event bus with size rotation and querying."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from arden.config import ObserverConfig, Settings, load_config
from arden.llm.mock import derive_llm_seed
from arden.world.world import EventBus, World, WorldEvent

ObserverKind = Literal[
    "perception",
    "gate_decision",
    "intent",
    "action_result",
    "reflection",
    "belief_change",
    "weather_change",
    "creator_action",
]

OBSERVER_KINDS: tuple[ObserverKind, ...] = (
    "perception",
    "gate_decision",
    "intent",
    "action_result",
    "reflection",
    "belief_change",
    "weather_change",
    "creator_action",
)

_KIND_ALIASES: dict[str, ObserverKind] = {
    "perception": "perception",
    "percept": "perception",
    "gate_decision": "gate_decision",
    "decision": "gate_decision",
    "gate": "gate_decision",
    "intent": "intent",
    "intent_set": "intent",
    "action_result": "action_result",
    "action": "action_result",
    "verb_result": "action_result",
    "reflection": "reflection",
    "reflected": "reflection",
    "belief_change": "belief_change",
    "belief_changed": "belief_change",
    "belief_evicted": "belief_change",
    "belief": "belief_change",
    "weather_change": "weather_change",
    "weather_changed": "weather_change",
    "weather": "weather_change",
    "creator_action": "creator_action",
    "creator": "creator_action",
}

_OBJECT_ID_RE = re.compile(r"\bobject_[0-9]+\b")


def normalize_observer_kind(kind: str) -> str:
    """Normalize an observer record kind or event type alias to its canonical kind."""
    cleaned = kind.strip().lower()
    return _KIND_ALIASES.get(cleaned, cleaned)


def extract_object_ids(value: Any) -> list[str]:
    """Recursively extract all ``object_<n>`` identifiers from ``value`` in deterministic order."""
    found: list[str] = []
    seen: set[str] = set()

    def _visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            for match in _OBJECT_ID_RE.findall(node):
                if match not in seen:
                    seen.add(match)
                    found.append(match)
            return
        if hasattr(node, "model_dump"):
            _visit(node.model_dump(mode="json"))
            return
        if isinstance(node, Mapping):
            for k, v in node.items():
                if isinstance(k, str):
                    _visit(k)
                _visit(v)
            return
        if isinstance(node, Iterable) and not isinstance(node, bytes | bytearray):
            for item in node:
                _visit(item)

    _visit(value)
    return found


def payload_mentions_object_id(payload: Any, object_id: str) -> bool:
    """Return True if ``payload`` references ``object_id`` anywhere in its structure."""
    target = object_id.strip()
    if not target:
        return True
    if isinstance(payload, Mapping):
        if payload.get("object_id") == target or payload.get("target") == target:
            return True
        obj_ids = payload.get("object_ids")
        if isinstance(obj_ids, list | tuple | set) and target in obj_ids:
            return True
    return target in extract_object_ids(payload)


def _to_jsonable(value: Any) -> Any:
    """Convert Pydantic models, sets, paths, and nested containers into JSON-serializable data."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_jsonable(value.to_dict())
        except TypeError:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_to_jsonable(item) for item in value]
    return str(value)


class LogRecord(dict[str, Any]):
    """Observer log record dictionary with attribute and alias access.

    Keys are strictly ``run_id``, ``tick``, ``timestamp``, ``kind``, and ``payload``.
    """

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
        tick: int | None = None,
        timestamp: str | None = None,
        kind: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        raw = dict(data) if data is not None else {}
        eff_run_id = str(run_id if run_id is not None else raw.get("run_id", "default"))
        eff_tick = int(tick if tick is not None else raw.get("tick", raw.get("world_tick", 0)))
        eff_ts = str(
            timestamp
            if timestamp is not None
            else raw.get("timestamp", datetime.now(timezone.utc).isoformat())
        )
        eff_kind = normalize_observer_kind(str(kind if kind is not None else raw.get("kind", "")))
        raw_payload = payload if payload is not None else raw.get("payload", {})
        eff_payload = dict(_to_jsonable(raw_payload)) if isinstance(raw_payload, Mapping) else {"value": _to_jsonable(raw_payload)}

        super().__init__(
            run_id=eff_run_id,
            tick=eff_tick,
            timestamp=eff_ts,
            kind=eff_kind,
            payload=eff_payload,
        )

    @property
    def run_id(self) -> str:
        return str(super().__getitem__("run_id"))

    @property
    def tick(self) -> int:
        return int(super().__getitem__("tick"))

    @property
    def world_tick(self) -> int:
        return self.tick

    @property
    def timestamp(self) -> str:
        return str(super().__getitem__("timestamp"))

    @property
    def kind(self) -> str:
        return str(super().__getitem__("kind"))

    @property
    def type(self) -> str:
        return self.kind

    @property
    def payload(self) -> dict[str, Any]:
        return super().__getitem__("payload")

    @property
    def object_id(self) -> str | None:
        p = self.payload
        for key in ("object_id", "target"):
            val = p.get(key)
            if isinstance(val, str) and _OBJECT_ID_RE.fullmatch(val):
                return val
        ids = self.object_ids
        return ids[0] if ids else None

    @property
    def object_ids(self) -> list[str]:
        p = self.payload
        if isinstance(p.get("object_ids"), list):
            return list(p["object_ids"])
        return extract_object_ids(p)

    def __getitem__(self, key: str) -> Any:
        if key == "world_tick":
            return self.tick
        if key == "type":
            return self.kind
        if key == "object_id":
            return self.object_id
        if key == "object_ids":
            return self.object_ids
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tick": self.tick,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


class QueryPage(dict[str, Any]):
    """Paged query result from ``ObserverLog.query`` that also supports iteration."""

    def __init__(
        self,
        *,
        run_id: str,
        items: list[LogRecord],
        total: int,
        limit: int,
        offset: int,
        page: int,
    ) -> None:
        has_more = (offset + len(items)) < total
        next_offset = (offset + len(items)) if has_more else None
        super().__init__(
            run_id=run_id,
            items=items,
            records=items,
            events=items,
            logs=items,
            total=total,
            count=len(items),
            limit=limit,
            offset=offset,
            page=page,
            page_size=limit,
            has_more=has_more,
            next_offset=next_offset,
            next_cursor=str(next_offset) if next_offset is not None else None,
        )

    @property
    def items_list(self) -> list[LogRecord]:
        return super().__getitem__("items")

    @property
    def records(self) -> list[LogRecord]:
        return super().__getitem__("items")

    @property
    def events(self) -> list[LogRecord]:
        return super().__getitem__("items")

    @property
    def total(self) -> int:
        return int(super().__getitem__("total"))

    @property
    def count(self) -> int:
        return int(super().__getitem__("count"))

    @property
    def has_more(self) -> bool:
        return bool(super().__getitem__("has_more"))

    @property
    def next_offset(self) -> int | None:
        return super().__getitem__("next_offset")

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, int | slice):
            return self.records[key]
        return super().__getitem__(key)


def _format_settings_toml(settings: Settings, seed: int) -> str:
    """Serialize key settings and seed to a valid TOML string loadable by ``load_config``."""
    lines = [
        "[world]",
        f"tick_rate = {float(settings.world.tick_rate)}",
        f"seed = {int(seed)}",
    ]
    if settings.world.sensing_radius is not None:
        lines.append(f"sensing_radius = {float(settings.world.sensing_radius)}")
    lines.extend(
        [
            "",
            "[llm]",
            f'backend = {json.dumps(settings.llm.backend)}',
            f'base_url = {json.dumps(settings.llm.base_url)}',
            f'model = {json.dumps(settings.llm.model)}',
            f"temperature = {float(settings.llm.temperature)}",
            f"timeout_s = {float(settings.llm.timeout_s)}",
            f"max_retries = {int(settings.llm.max_retries)}",
            "",
            "[observer]",
            f'runs_dir = {json.dumps(settings.observer.runs_dir.as_posix())}',
            f"max_bytes = {int(settings.observer.max_bytes)}",
            f"enabled = {'true' if settings.observer.enabled else 'false'}",
            "",
        ]
    )
    return "\n".join(lines)


class ObserverLog:
    """Subscribes to the world event bus and writes append-only JSONL files rotated by size."""

    def __init__(
        self,
        world_or_bus: World | EventBus | Path | str | None = None,
        *,
        run_id: str | None = None,
        runs_dir: Path | str | None = None,
        run_dir: Path | str | None = None,
        max_bytes: int | None = None,
        rotate_size_bytes: int | None = None,
        max_file_size: int | None = None,
        rotation_size_bytes: int | None = None,
        config: Settings | ObserverConfig | Mapping[str, Any] | None = None,
        settings: Settings | None = None,
        seed: int | None = None,
        world: World | None = None,
        bus: EventBus | None = None,
    ) -> None:
        resolved_world: World | None = world
        resolved_bus: EventBus | None = bus
        path_arg: Path | None = None

        if isinstance(world_or_bus, World):
            resolved_world = world_or_bus
            resolved_bus = world_or_bus.events
        elif isinstance(world_or_bus, EventBus):
            resolved_bus = world_or_bus
        elif isinstance(world_or_bus, Path):
            path_arg = world_or_bus
        elif isinstance(world_or_bus, str):
            candidate = Path(world_or_bus)
            if candidate.exists() or "/" in world_or_bus or "\\" in world_or_bus:
                path_arg = candidate
            elif run_id is None:
                run_id = world_or_bus

        if resolved_world is not None and resolved_bus is None:
            resolved_bus = resolved_world.events

        # Resolve Settings and ObserverConfig
        if isinstance(settings, Settings):
            resolved_settings = settings
        elif isinstance(config, Settings):
            resolved_settings = config
        elif resolved_world is not None and isinstance(getattr(resolved_world, "_settings", None), Settings):
            resolved_settings = getattr(resolved_world, "_settings")
        else:
            resolved_settings = Settings()

        obs_cfg = resolved_settings.observer
        if isinstance(config, ObserverConfig):
            obs_cfg = config
        elif isinstance(config, Mapping) and not isinstance(config, Settings):
            if "observer" in config and isinstance(config["observer"], Mapping):
                obs_cfg = ObserverConfig.model_validate(config["observer"])
            else:
                try:
                    obs_cfg = ObserverConfig.model_validate(config)
                except Exception:
                    pass

        eff_max_bytes = (
            max_bytes
            if max_bytes is not None
            else rotate_size_bytes
            if rotate_size_bytes is not None
            else max_file_size
            if max_file_size is not None
            else rotation_size_bytes
            if rotation_size_bytes is not None
            else obs_cfg.max_bytes
        )
        if isinstance(eff_max_bytes, bool) or not isinstance(eff_max_bytes, int) or eff_max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes: int = int(eff_max_bytes)

        eff_runs_dir = Path(runs_dir) if runs_dir is not None else obs_cfg.runs_dir
        if path_arg is not None and run_dir is None:
            if path_arg.is_dir() or path_arg.parent != Path("."):
                run_dir = path_arg
            elif run_id is None:
                run_id = path_arg.name

        if run_dir is not None:
            self.run_dir: Path = Path(run_dir)
            self.run_id: str = str(run_id) if run_id is not None else self.run_dir.name
            self.runs_dir: Path = self.run_dir.parent
        else:
            self.runs_dir = eff_runs_dir
            self.run_id = (
                str(run_id)
                if run_id is not None and str(run_id).strip()
                else f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
            self.run_dir = self.runs_dir / self.run_id

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.world: World | None = resolved_world
        self.bus: EventBus | None = resolved_bus
        self.settings: Settings = resolved_settings

        # Resolve seed
        if seed is not None:
            self.seed: int = int(seed)
        elif resolved_world is not None and getattr(resolved_world, "seed", 0) != 0:
            self.seed = int(getattr(resolved_world, "seed", 0))
        else:
            self.seed = int(resolved_settings.world.seed)

        if self.settings.world.seed != self.seed:
            self.settings = self.settings.model_copy(
                update={"world": self.settings.world.model_copy(update={"seed": self.seed})}
            )

        self._file_index: int = 1
        self._current_file_bytes: int = 0
        self._discover_existing_files()

        self._runner_meta: dict[str, Any] = {"runner": "world"}
        self._initial_world_snapshot: dict[str, Any] | None = None
        if self.world is not None:
            try:
                self._initial_world_snapshot = json.loads(self.world.snapshot())
            except Exception:
                self._initial_world_snapshot = None

        # Only write config/seed metadata if not already present (or if attaching a live world)
        config_json_path = self.run_dir / "config.json"
        if not config_json_path.exists() or self.world is not None:
            self.save_metadata()
        else:
            self._load_existing_metadata()

        self._unsubscribe: Callable[[], None] | None = None
        if self.bus is not None:
            self._unsubscribe = self.bus.subscribe(self._on_event)
        if self.world is not None:
            setattr(self.world, "observer_log", self)

    @classmethod
    def open(cls, run_dir: Path | str) -> ObserverLog:
        """Open an existing run directory for querying or replay inspection."""
        return cls(run_dir=Path(run_dir))

    def _discover_existing_files(self) -> None:
        existing = sorted(self.run_dir.glob("log_*.jsonl"))
        if not existing:
            self._file_index = 1
            self._current_file_bytes = 0
            return
        last_file = existing[-1]
        m = re.search(r"log_([0-9]+)\.jsonl$", last_file.name)
        if m:
            self._file_index = int(m.group(1))
        else:
            self._file_index = len(existing)
        self._current_file_bytes = last_file.stat().st_size if last_file.exists() else 0

    def _load_existing_metadata(self) -> None:
        config_json_path = self.run_dir / "config.json"
        if config_json_path.is_file():
            try:
                raw = json.loads(config_json_path.read_text(encoding="utf-8"))
                if isinstance(raw, Mapping):
                    if "run_id" in raw:
                        self.run_id = str(raw["run_id"])
                    if "seed" in raw and isinstance(raw["seed"], int):
                        self.seed = int(raw["seed"])
                    if isinstance(raw.get("config"), Mapping):
                        try:
                            self.settings = Settings.model_validate(raw["config"])
                        except Exception:
                            pass
                    if isinstance(raw.get("initial_world"), Mapping):
                        self._initial_world_snapshot = dict(raw["initial_world"])
                    if isinstance(raw.get("runner_meta"), Mapping):
                        self._runner_meta = dict(raw["runner_meta"])
            except Exception:
                pass
        elif (self.run_dir / "config.toml").is_file():
            try:
                self.settings = load_config(self.run_dir / "config.toml")
                self.seed = int(self.settings.world.seed)
            except Exception:
                pass

    def save_metadata(self, *, runner_meta: Mapping[str, Any] | None = None) -> None:
        """Save the run's config, seed, and initial world state alongside the JSONL files."""
        if runner_meta is not None:
            self._runner_meta.update(dict(runner_meta))
        if self.world is not None and self._initial_world_snapshot is None:
            try:
                self._initial_world_snapshot = json.loads(self.world.snapshot())
            except Exception:
                pass

        cfg_dump = self.settings.model_dump(mode="json")
        cfg_dump.setdefault("world", {})["seed"] = self.seed
        meta_payload: dict[str, Any] = {
            **cfg_dump,
            "run_id": self.run_id,
            "seed": self.seed,
            "llm_seed": derive_llm_seed(self.seed),
            "config": cfg_dump,
            "runner_meta": dict(self._runner_meta),
        }
        if self._initial_world_snapshot is not None:
            meta_payload["initial_world"] = self._initial_world_snapshot

        config_json_path = self.run_dir / "config.json"
        config_json_path.write_text(
            json.dumps(meta_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        config_toml_path = self.run_dir / "config.toml"
        config_toml_path.write_text(
            _format_settings_toml(self.settings, self.seed),
            encoding="utf-8",
        )

        seed_txt_path = self.run_dir / "seed.txt"
        seed_txt_path.write_text(f"{self.seed}\n", encoding="utf-8")

    def update_initial_world(self, world: World) -> None:
        """Update the saved initial world snapshot if no ticks have been logged yet."""
        self.world = world
        self.seed = int(world.seed)
        try:
            self._initial_world_snapshot = json.loads(world.snapshot())
        except Exception:
            pass
        self.save_metadata()

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def log_files(self) -> list[Path]:
        """Return all rotated JSONL log files in chronological order."""
        files = sorted(self.run_dir.glob("log_*.jsonl"))
        if files:
            return files
        return sorted(self.run_dir.glob("*.jsonl"))

    files = log_files
    jsonl_files = log_files

    def _current_file_path(self) -> Path:
        return self.run_dir / f"log_{self._file_index:05d}.jsonl"

    def close(self) -> None:
        """Unsubscribe from the event bus."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def _resolve_tick(self, tick: int | None = None) -> int:
        if tick is not None:
            return int(tick)
        if self.world is not None:
            return int(self.world.tick)
        return 0

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        tick: int | None = None,
        timestamp: str | None = None,
        **extra_payload: Any,
    ) -> LogRecord:
        """Append a single record to the current JSONL log file, rotating by size if needed."""
        norm_kind = normalize_observer_kind(kind)
        eff_tick = (
            int(tick)
            if tick is not None
            else (int(self.world.tick) if self.world is not None else 0)
        )
        eff_ts = (
            str(timestamp)
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        merged_payload: dict[str, Any] = {}
        if payload is not None:
            merged_payload.update(_to_jsonable(dict(payload)))
        if extra_payload:
            merged_payload.update(_to_jsonable(extra_payload))

        rec = LogRecord(
            run_id=self.run_id,
            tick=eff_tick,
            timestamp=eff_ts,
            kind=norm_kind,
            payload=merged_payload,
        )
        line = json.dumps(rec.to_dict(), sort_keys=True) + "\n"
        encoded = line.encode("utf-8")

        if self._current_file_bytes > 0 and (self._current_file_bytes + len(encoded)) > self.max_bytes:
            self._file_index += 1
            self._current_file_bytes = 0

        target_path = self._current_file_path()
        with target_path.open("ab") as fh:
            fh.write(encoded)
            fh.flush()
        self._current_file_bytes += len(encoded)
        return rec

    write = record
    append = record

    def _on_event(self, event: WorldEvent | Mapping[str, Any]) -> None:
        """Handle an event emitted on the world EventBus."""
        ev_type = (
            event.type
            if isinstance(event, WorldEvent)
            else str(event.get("type", event.get("kind", "")))
        )
        ev_tick = (
            int(event.tick)
            if isinstance(event, WorldEvent)
            else int(event.get("tick", self.world.tick if self.world is not None else 0))
        )
        ev_source = (
            str(event.source)
            if isinstance(event, WorldEvent)
            else str(event.get("source", "world"))
        )

        # 1. Perception
        if ev_type == "perception":
            raw_percept = getattr(event, "percept", None) if isinstance(event, WorldEvent) else event.get("percept")
            percept_dict = _to_jsonable(raw_percept) if isinstance(raw_percept, Mapping) or hasattr(raw_percept, "to_dict") else {}
            if not isinstance(percept_dict, dict):
                percept_dict = {}
            obj_ids = extract_object_ids(percept_dict)
            is_delta = bool(getattr(event, "is_delta", percept_dict.get("is_delta", False)))
            payload = {
                **percept_dict,
                "percept": percept_dict,
                "is_delta": is_delta,
                "object_ids": obj_ids,
            }
            if len(obj_ids) == 1:
                payload.setdefault("object_id", obj_ids[0])
            self.record("perception", payload, tick=ev_tick)
            return

        # 2. Gate decision
        if ev_type == "gate_decision":
            wake = bool(getattr(event, "wake", False) if isinstance(event, WorldEvent) else event.get("wake", False))
            reasons = list(getattr(event, "reasons", []) if isinstance(event, WorldEvent) else event.get("reasons", []))
            outcome = "wake" if wake else "continue"
            surprise = float(getattr(event, "surprise", 0.0) if isinstance(event, WorldEvent) else event.get("surprise", 0.0))
            matched_wake_if = list(
                getattr(event, "matched_wake_if", [])
                if isinstance(event, WorldEvent)
                else event.get("matched_wake_if", [])
            )
            unparseable_wake_if = list(
                getattr(event, "unparseable_wake_if", [])
                if isinstance(event, WorldEvent)
                else event.get("unparseable_wake_if", [])
            )
            obj_ids = extract_object_ids([reasons, matched_wake_if])
            payload = {
                "wake": wake,
                "reasons": reasons,
                "outcome": outcome,
                "surprise": surprise,
                "matched_wake_if": matched_wake_if,
                "unparseable_wake_if": unparseable_wake_if,
                "object_ids": obj_ids,
            }
            self.record("gate_decision", payload, tick=ev_tick)
            return

        # 3. Intent
        if ev_type in ("intent_set", "intent"):
            raw_intent = getattr(event, "intent", None) if isinstance(event, WorldEvent) else event.get("intent")
            intent_dict = _to_jsonable(raw_intent) if raw_intent is not None else {}
            if not isinstance(intent_dict, dict):
                intent_dict = {}
            goal = getattr(event, "goal", None) if isinstance(event, WorldEvent) else event.get("goal")
            thought = getattr(event, "thought", None) if isinstance(event, WorldEvent) else event.get("thought")
            expect = getattr(event, "expect", None) if isinstance(event, WorldEvent) else event.get("expect")
            confidence = getattr(event, "confidence", None) if isinstance(event, WorldEvent) else event.get("confidence")
            steps = _to_jsonable(getattr(event, "steps", []) if isinstance(event, WorldEvent) else event.get("steps", []))
            wake_if = list(getattr(event, "wake_if", []) if isinstance(event, WorldEvent) else event.get("wake_if", []))
            reasons = list(getattr(event, "reasons", []) if isinstance(event, WorldEvent) else event.get("reasons", []))
            if not intent_dict and goal is not None:
                intent_dict = {
                    "goal": goal,
                    "steps": steps,
                    "wake_if": wake_if,
                    "expect": expect or "",
                    "thought": thought or "",
                    "confidence": float(confidence if confidence is not None else 0.5),
                }
            obj_ids = extract_object_ids([intent_dict, steps, goal, expect, wake_if])
            payload = {
                "intent": intent_dict,
                "goal": goal if goal is not None else intent_dict.get("goal"),
                "thought": thought if thought is not None else intent_dict.get("thought"),
                "expect": expect if expect is not None else intent_dict.get("expect"),
                "confidence": confidence if confidence is not None else intent_dict.get("confidence"),
                "steps": steps if steps else intent_dict.get("steps", []),
                "wake_if": wake_if if wake_if else intent_dict.get("wake_if", []),
                "reasons": reasons,
                "object_ids": obj_ids,
            }
            if len(obj_ids) == 1:
                payload["object_id"] = obj_ids[0]
            self.record("intent", payload, tick=ev_tick)
            return

        # 4. Action result
        if ev_type == "action_result":
            verb = getattr(event, "verb", "") if isinstance(event, WorldEvent) else event.get("verb", "")
            target = getattr(event, "target", None) if isinstance(event, WorldEvent) else event.get("target")
            object_id = getattr(event, "object_id", None) if isinstance(event, WorldEvent) else event.get("object_id")
            params = _to_jsonable(getattr(event, "params", {}) if isinstance(event, WorldEvent) else event.get("params", {}))
            ok = bool(getattr(event, "ok", True) if isinstance(event, WorldEvent) else event.get("ok", True))
            reason = getattr(event, "reason", None) if isinstance(event, WorldEvent) else event.get("reason")
            events_list = _to_jsonable(getattr(event, "events", []) if isinstance(event, WorldEvent) else event.get("events", []))
            data_dict = _to_jsonable(getattr(event, "data", {}) if isinstance(event, WorldEvent) else event.get("data", {}))
            obj_ids = extract_object_ids([target, object_id, params, data_dict, events_list])
            primary_oid = object_id or (target if isinstance(target, str) and _OBJECT_ID_RE.fullmatch(target) else (obj_ids[0] if obj_ids else None))
            payload = {
                "verb": verb,
                "target": target,
                "object_id": primary_oid,
                "params": params,
                "ok": ok,
                "status": "ok" if ok else "failed",
                "reason": reason,
                "events": events_list,
                "data": data_dict,
                "object_ids": obj_ids,
            }
            self.record("action_result", payload, tick=ev_tick)
            return

        # 5. Reflection
        if ev_type == "reflection":
            trigger = getattr(event, "trigger", "buffer_full") if isinstance(event, WorldEvent) else event.get("trigger", "buffer_full")
            episodes = _to_jsonable(getattr(event, "episodes", []) if isinstance(event, WorldEvent) else event.get("episodes", []))
            beliefs = _to_jsonable(getattr(event, "beliefs", []) if isinstance(event, WorldEvent) else event.get("beliefs", []))
            self_model = getattr(event, "self_model", None) if isinstance(event, WorldEvent) else event.get("self_model")
            obj_ids = extract_object_ids([episodes, beliefs, self_model])
            payload = {
                "trigger": trigger,
                "episodes": episodes,
                "beliefs": beliefs,
                "self_model": self_model,
                "object_ids": obj_ids,
            }
            self.record("reflection", payload, tick=ev_tick)
            return

        # 6. Belief change
        if ev_type == "belief_change":
            action = getattr(event, "action", "created") if isinstance(event, WorldEvent) else event.get("action", "created")
            belief_id = getattr(event, "belief_id", None) if isinstance(event, WorldEvent) else event.get("belief_id")
            statement = getattr(event, "statement", "") if isinstance(event, WorldEvent) else event.get("statement", "")
            tags = list(getattr(event, "tags", []) if isinstance(event, WorldEvent) else event.get("tags", []))
            confidence = float(getattr(event, "confidence", 0.0) if isinstance(event, WorldEvent) else event.get("confidence", 0.0))
            prev_conf = getattr(event, "previous_confidence", None) if isinstance(event, WorldEvent) else event.get("previous_confidence")
            delta = float(getattr(event, "delta", 0.0) if isinstance(event, WorldEvent) else event.get("delta", 0.0))
            ev_count = int(getattr(event, "evidence_count", 1) if isinstance(event, WorldEvent) else event.get("evidence_count", 1))
            belief_obj = _to_jsonable(getattr(event, "belief", None) if isinstance(event, WorldEvent) else event.get("belief"))
            obj_ids = extract_object_ids([statement, tags, belief_obj])
            payload = {
                "action": action,
                "outcome": action,
                "belief_id": belief_id,
                "statement": statement,
                "tags": tags,
                "confidence": confidence,
                "previous_confidence": prev_conf,
                "delta": delta,
                "evidence_count": ev_count,
                "belief": belief_obj,
                "object_ids": obj_ids,
            }
            if len(obj_ids) == 1:
                payload["object_id"] = obj_ids[0]
            self.record("belief_change", payload, tick=ev_tick)
            return

        # 7. Weather change (manual or automated)
        if ev_type in ("weather_changed", "weather_change"):
            w_source = getattr(event, "source", "automated") if isinstance(event, WorldEvent) else event.get("source", "automated")
            w_type = getattr(event, "weather_type", None) if isinstance(event, WorldEvent) else event.get("weather_type")
            prev_w_type = getattr(event, "previous_weather_type", None) if isinstance(event, WorldEvent) else event.get("previous_weather_type")
            intensity = getattr(event, "intensity", 0.0) if isinstance(event, WorldEvent) else event.get("intensity", 0.0)
            prev_intensity = getattr(event, "previous_intensity", 0.0) if isinstance(event, WorldEvent) else event.get("previous_intensity", 0.0)
            duration = getattr(event, "duration", 1) if isinstance(event, WorldEvent) else event.get("duration", 1)
            light_level = getattr(event, "light_level", None) if isinstance(event, WorldEvent) else event.get("light_level")
            ambient_temp = getattr(event, "ambient_temperature", None) if isinstance(event, WorldEvent) else event.get("ambient_temperature")
            from_schedule = bool(getattr(event, "from_schedule", False) if isinstance(event, WorldEvent) else event.get("from_schedule", False))

            payload = {
                "source": w_source,
                "weather_type": w_type,
                "previous_weather_type": prev_w_type,
                "intensity": intensity,
                "previous_intensity": prev_intensity,
                "duration": duration,
                "light_level": light_level,
                "ambient_temperature": ambient_temp,
                "from_schedule": from_schedule,
            }
            if w_source == "manual" and not from_schedule:
                # Record the manual weather invocation as a creator_action first so replay
                # re-applies set_weather at this exact point and triggers the weather_change.
                self.record(
                    "creator_action",
                    {
                        "action": "set_weather",
                        "weather_type": w_type,
                        "intensity": intensity,
                        "duration": duration,
                        "source": "creator",
                    },
                    tick=ev_tick,
                )
            self.record("weather_change", payload, tick=ev_tick)
            return

        # 8. Explicit CreatorActionEvent or creator-sourced WorldEvents
        if ev_type == "creator_action":
            action = getattr(event, "action", "") if isinstance(event, WorldEvent) else event.get("action", "")
            details = _to_jsonable(getattr(event, "details", {}) if isinstance(event, WorldEvent) else event.get("details", {}))
            if not isinstance(details, dict):
                details = {"value": details}
            obj_id = getattr(event, "object_id", None) if isinstance(event, WorldEvent) else event.get("object_id")
            obj_ids = extract_object_ids([obj_id, details])
            payload = {
                **details,
                "action": action,
                "object_ids": obj_ids,
            }
            if obj_id is not None:
                payload["object_id"] = obj_id
            elif len(obj_ids) == 1:
                payload["object_id"] = obj_ids[0]
            self.record("creator_action", payload, tick=ev_tick)
            return

        if ev_source == "creator":
            if ev_type == "object_spawned":
                oid = getattr(event, "object_id", None)
                pos = _to_jsonable(getattr(event, "position", None))
                obj_data = _to_jsonable(getattr(event, "object", {}))
                self.record(
                    "creator_action",
                    {
                        "action": "spawn",
                        "object_id": oid,
                        "position": pos,
                        "object": obj_data,
                        "props": obj_data,
                        "object_ids": [oid] if oid else [],
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "object_moved":
                oid = getattr(event, "object_id", None)
                pos = _to_jsonable(getattr(event, "position", None))
                prev_pos = _to_jsonable(getattr(event, "previous_position", None))
                self.record(
                    "creator_action",
                    {
                        "action": "move",
                        "object_id": oid,
                        "position": pos,
                        "previous_position": prev_pos,
                        "object_ids": [oid] if oid else [],
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "object_despawned":
                oid = getattr(event, "object_id", None)
                pos = _to_jsonable(getattr(event, "position", None))
                obj_data = _to_jsonable(getattr(event, "object", {}))
                self.record(
                    "creator_action",
                    {
                        "action": "despawn",
                        "object_id": oid,
                        "position": pos,
                        "object": obj_data,
                        "object_ids": [oid] if oid else [],
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "object_changed":
                oid = getattr(event, "object_id", None)
                changes = _to_jsonable(getattr(event, "changes", {}))
                obj_data = _to_jsonable(getattr(event, "object", {}))
                self.record(
                    "creator_action",
                    {
                        "action": "update_object",
                        "object_id": oid,
                        "changes": changes,
                        "object": obj_data,
                        "object_ids": [oid] if oid else [],
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "time_of_day_changed":
                self.record(
                    "creator_action",
                    {
                        "action": "set_time_of_day",
                        "time_of_day": getattr(event, "time_of_day", None),
                        "previous_time_of_day": getattr(event, "previous_time_of_day", None),
                        "light_level": getattr(event, "light_level", None),
                        "ambient_temperature": getattr(event, "ambient_temperature", None),
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "cycle_frozen":
                self.record(
                    "creator_action",
                    {
                        "action": "freeze_cycle",
                        "frozen": getattr(event, "frozen", None),
                        "time_of_day": getattr(event, "time_of_day", None),
                    },
                    tick=ev_tick,
                )
                return
            if ev_type == "mail_arrived":
                oid = getattr(event, "object_id", None)
                mb_id = getattr(event, "mailbox_id", None)
                arrival_kind = getattr(event, "arrival_kind", None) or "reply"
                text = getattr(event, "text", None)
                obj_data = _to_jsonable(getattr(event, "object", None))
                obj_ids = [x for x in (oid, mb_id) if isinstance(x, str)]
                self.record(
                    "creator_action",
                    {
                        "action": arrival_kind,
                        "mailbox_id": mb_id,
                        "object_id": oid,
                        "text": text,
                        "object": obj_data,
                        "props": obj_data,
                        "object_ids": obj_ids,
                    },
                    tick=ev_tick,
                )
                return

    def read_all(self) -> list[LogRecord]:
        """Read all records across all rotated JSONL files in order."""
        records: list[LogRecord] = []
        for path in self.log_files:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    raw = json.loads(stripped)
                    if isinstance(raw, Mapping):
                        records.append(LogRecord(raw))
        return records

    @property
    def records(self) -> list[LogRecord]:
        """All logged records read from the JSONL files."""
        return self.read_all()

    def query(
        self,
        *,
        kind: str | Sequence[str] | None = None,
        from_tick: int | str | None = None,
        to_tick: int | str | None = None,
        object_id: str | None = None,
        limit: int | str | None = None,
        offset: int | str | None = None,
        page: int | str | None = None,
        page_size: int | str | None = None,
        cursor: int | str | None = None,
    ) -> QueryPage:
        """Query observer log records with optional filters and pagination."""
        allowed_kinds: set[str] | None = None
        if kind is not None:
            if isinstance(kind, str):
                parts = [p.strip() for p in kind.split(",") if p.strip()]
            else:
                parts = [
                    p.strip()
                    for item in kind
                    if isinstance(item, str)
                    for p in item.split(",")
                    if p.strip()
                ]
            if parts:
                allowed_kinds = {normalize_observer_kind(p) for p in parts}

        min_tick: int | None = None
        if from_tick is not None and str(from_tick).strip() != "":
            min_tick = int(from_tick)

        max_tick: int | None = None
        if to_tick is not None and str(to_tick).strip() != "":
            max_tick = int(to_tick)

        eff_object_id: str | None = None
        if object_id is not None and str(object_id).strip() != "":
            eff_object_id = str(object_id).strip()

        matched: list[LogRecord] = []
        for rec in self.read_all():
            if allowed_kinds is not None and rec.kind not in allowed_kinds:
                continue
            if min_tick is not None and rec.tick < min_tick:
                continue
            if max_tick is not None and rec.tick > max_tick:
                continue
            if eff_object_id is not None and not payload_mentions_object_id(rec.payload, eff_object_id):
                continue
            matched.append(rec)

        total = len(matched)

        eff_limit = 100
        if page_size is not None and str(page_size).strip() != "":
            eff_limit = max(1, int(page_size))
        if limit is not None and str(limit).strip() != "":
            eff_limit = max(1, int(limit))

        eff_offset = 0
        if cursor is not None and str(cursor).strip() != "":
            eff_offset = max(0, int(cursor))
        elif offset is not None and str(offset).strip() != "":
            eff_offset = max(0, int(offset))
        elif page is not None and str(page).strip() != "":
            page_num = max(1, int(page))
            eff_offset = (page_num - 1) * eff_limit

        eff_page = (eff_offset // eff_limit) + 1 if eff_limit > 0 else 1
        sliced = matched[eff_offset : eff_offset + eff_limit]

        return QueryPage(
            run_id=self.run_id,
            items=sliced,
            total=total,
            limit=eff_limit,
            offset=eff_offset,
            page=eff_page,
        )

    query_page = query

    def compare(self, beliefs: Any = None) -> Any:
        """Compare beliefs against the ground-truth events in this observer log."""
        from arden.observer.compare import compare

        return compare(self, beliefs)


ObserverLogger = ObserverLog
RunLogger = ObserverLog

__all__ = [
    "OBSERVER_KINDS",
    "LogRecord",
    "ObserverKind",
    "ObserverLog",
    "ObserverLogger",
    "QueryPage",
    "RunLogger",
    "extract_object_ids",
    "normalize_observer_kind",
    "payload_mentions_object_id",
]
