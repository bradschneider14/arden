"""Mailbox mechanics, persistent SQLite creator inbox, replies, and audit logging.

Rules from Ticket T21 and docs/DESIGN.md (Section 8):
- The mailbox is an ordinary object whose hidden behavior rules respond to
  existing body verbs (``grab``, ``separate``, ``speak``, ``insert``). It has
  no special or opaque action.
- Lid: perceptible ``state["lid"]`` (``"open"`` / ``"closed"``). ``separate``
  or ``grab`` on the mailbox toggles it. While closed, its contents are neither
  perceptible nor reachable.
- Speak: with the lid open and the agent within reach, ``speak`` text is
  captured as an outgoing message. It is delivered when the lid closes or after
  a configurable delay (default ``10.0`` s sim time), whichever comes first.
  ``speak`` anywhere else has no mailbox effect.
- Insert: ``insert`` of a held object into the open mailbox removes it from the
  world and delivers it with any speech captured in the same window. The object
  is preserved so the creator can return it.
- Inbox: delivered messages and objects go to a persistent creator inbox
  (SQLite) with world tick and real timestamp. Nothing is shown to the agent to
  acknowledge delivery.
- Replies: ``reply(text)`` places a flat letter object inside the mailbox whose
  ``text`` is a perceptible property revealed only via ``look`` with the lid
  open; ``spawn_in_response(props)`` places an object inside with no note;
  silence is the default.
- Optional nudge: config ``mailbox.lid_drift_chance`` (per in-game hour,
  default ``0.0``) lets the lid occasionally swing open on its own.
- Wake trigger: arrivals in the box emit a creator event (``MailArrivedEvent``)
  so the attention gate (``RuleGate``) can wake the agent.
- Creator/mail logging: every send, delivery, and creator action is logged with
  tick, timestamp, and an optional private creator note per experiment.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from arden.world.objects import Composite, Position, WorldObject
    from arden.world.world import MailArrivedEvent, World


def _validate_world_object(data: Mapping[str, Any] | WorldObject) -> WorldObject:
    from arden.world.objects import validate_world_object

    return validate_world_object(data)


def _is_world_object(obj: object) -> bool:
    from arden.world.objects import WorldObject

    return isinstance(obj, WorldObject)

DEFAULT_MAILBOX_DB_PATH: Path = Path("data/mailbox.db")
DEFAULT_MAILBOX_DELIVERY_DELAY_S: float = 10.0
DEFAULT_MAILBOX_LID_DRIFT_CHANCE: float = 0.0
DEFAULT_TICK_RATE: float = 10.0

DeliveryReason = Literal["lid_closed", "timeout", "insert", "manual"]
MailboxLogKind = Literal[
    "send",
    "delivery",
    "reply",
    "spawn_in_response",
    "creator_action",
    "lid_drift",
]


def _utc_now() -> tuple[str, float]:
    """Return ``(iso_timestamp, unix_timestamp)`` in UTC."""
    now_ts = time.time()
    iso_str = datetime.fromtimestamp(now_ts, tz=UTC).isoformat()
    return iso_str, now_ts


class MailboxConfig(BaseModel):
    """Configuration for the mailbox and creator inbox under ``[mailbox]``."""

    model_config = ConfigDict(extra="forbid")

    lid_drift_chance: float = Field(default=DEFAULT_MAILBOX_LID_DRIFT_CHANCE, ge=0.0, le=1.0)
    delivery_delay_s: float = Field(default=DEFAULT_MAILBOX_DELIVERY_DELAY_S, ge=0.0)
    delivery_delay_ticks: int | None = Field(default=None, ge=0)
    db_path: Path = Field(default=DEFAULT_MAILBOX_DB_PATH)
    creator_note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, values: object) -> object:
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        for alias in ("drift_chance", "lid_drift"):
            if alias in raw and "lid_drift_chance" not in raw:
                raw["lid_drift_chance"] = raw.pop(alias)
        for alias in ("delay_s", "delivery_delay", "delay", "timeout_s"):
            if alias in raw and "delivery_delay_s" not in raw:
                raw["delivery_delay_s"] = raw.pop(alias)
        for alias in ("delay_ticks", "timeout_ticks"):
            if alias in raw and "delivery_delay_ticks" not in raw:
                raw["delivery_delay_ticks"] = raw.pop(alias)
        for alias in ("note", "experiment_note"):
            if alias in raw and "creator_note" not in raw:
                raw["creator_note"] = raw.pop(alias)
        return raw

    @field_validator("lid_drift_chance", "delivery_delay_s")
    @classmethod
    def _must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("mailbox config numbers must be finite")
        return value

    @field_validator("db_path", mode="before")
    @classmethod
    def _validate_db_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            if not str(value).strip():
                raise ValueError("db_path must be a non-empty path")
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("db_path must be a non-empty string")
            return Path(stripped)
        raise TypeError("db_path must be a Path or str")

    @property
    def delay_s(self) -> float:
        """Alias for ``delivery_delay_s``."""
        return self.delivery_delay_s

    @property
    def delivery_delay(self) -> float:
        """Alias for ``delivery_delay_s``."""
        return self.delivery_delay_s

    @property
    def drift_chance(self) -> float:
        """Alias for ``lid_drift_chance``."""
        return self.lid_drift_chance

    @property
    def note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.creator_note

    @property
    def experiment_note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.creator_note

    def delay_in_ticks(self, tick_rate: float = DEFAULT_TICK_RATE) -> int:
        """Return the speech delivery delay converted to integer simulation ticks."""
        if self.delivery_delay_ticks is not None:
            return max(0, int(self.delivery_delay_ticks))
        rate = float(tick_rate) if tick_rate and tick_rate > 0.0 else DEFAULT_TICK_RATE
        return max(0, round(self.delivery_delay_s * rate))


MailboxSettings = MailboxConfig


class InboxEntry(BaseModel):
    """One message and/or preserved object delivered from the mailbox to the creator inbox."""

    model_config = ConfigDict(extra="forbid")

    id: int
    tick: int = Field(ge=0)
    timestamp: str
    unix_timestamp: float
    mailbox_id: str
    text: str | None = None
    texts: list[str] = Field(default_factory=list)
    object_id: str | None = None
    object_data: dict[str, Any] | None = None
    delivery_reason: DeliveryReason = "lid_closed"
    creator_note: str | None = None

    @property
    def object(self) -> WorldObject | None:
        """Reconstructed ``WorldObject`` (or ``Composite``) preserved at delivery time."""
        if self.object_data is None:
            return None
        return _validate_world_object(self.object_data)

    @property
    def preserved_object(self) -> WorldObject | None:
        """Alias for ``object``."""
        return self.object

    @property
    def message(self) -> str | None:
        """Alias for ``text``."""
        return self.text

    @property
    def speech(self) -> str | None:
        """Alias for ``text``."""
        return self.text

    @property
    def world_tick(self) -> int:
        """Alias for ``tick``."""
        return self.tick

    @property
    def real_timestamp(self) -> str:
        """Alias for ``timestamp``."""
        return self.timestamp

    @property
    def note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.creator_note

    @property
    def reason(self) -> DeliveryReason:
        """Alias for ``delivery_reason``."""
        return self.delivery_reason

    def to_dict(self) -> dict[str, Any]:
        """Serialize this inbox entry to a plain dictionary."""
        return self.model_dump(mode="json")

    def __getitem__(self, key: str) -> Any:
        if key in ("object", "preserved_object"):
            return self.object
        if key in ("message", "speech"):
            return self.text
        if key == "note":
            return self.creator_note
        if key == "world_tick":
            return self.tick
        if key == "real_timestamp":
            return self.timestamp
        if key == "reason":
            return self.delivery_reason
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            _ = self[key]
            return True
        except KeyError:
            return False


InboxItem = InboxEntry
MailMessage = InboxEntry


class MailboxLogEntry(BaseModel):
    """Audit log entry for a send, delivery, or creator action on the mailbox."""

    model_config = ConfigDict(extra="forbid")

    id: int
    kind: MailboxLogKind
    tick: int = Field(ge=0)
    timestamp: str
    unix_timestamp: float
    mailbox_id: str | None = None
    text: str | None = None
    object_id: str | None = None
    object_data: dict[str, Any] | None = None
    creator_note: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def type(self) -> MailboxLogKind:
        """Alias for ``kind``."""
        return self.kind

    @property
    def event_type(self) -> MailboxLogKind:
        """Alias for ``kind``."""
        return self.kind

    @property
    def action(self) -> MailboxLogKind:
        """Alias for ``kind``."""
        return self.kind

    @property
    def note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.creator_note

    @property
    def experiment_note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.creator_note

    @property
    def world_tick(self) -> int:
        """Alias for ``tick``."""
        return self.tick

    @property
    def real_timestamp(self) -> str:
        """Alias for ``timestamp``."""
        return self.timestamp

    @property
    def object(self) -> WorldObject | None:
        """Reconstructed ``WorldObject`` if ``object_data`` is present."""
        if self.object_data is None:
            return None
        return _validate_world_object(self.object_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this log entry to a plain dictionary."""
        return self.model_dump(mode="json")

    def __getitem__(self, key: str) -> Any:
        if key in ("type", "event_type", "action"):
            return self.kind
        if key in ("note", "experiment_note"):
            return self.creator_note
        if key == "world_tick":
            return self.tick
        if key == "real_timestamp":
            return self.timestamp
        if key == "object":
            return self.object
        d = self.to_dict()
        if key in d:
            return d[key]
        if key in self.details:
            return self.details[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            _ = self[key]
            return True
        except KeyError:
            return False


_CREATE_MAILBOX_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS creator_inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    unix_timestamp REAL NOT NULL,
    mailbox_id TEXT NOT NULL,
    text TEXT,
    texts_json TEXT NOT NULL DEFAULT '[]',
    object_id TEXT,
    object_json TEXT,
    delivery_reason TEXT NOT NULL,
    creator_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_creator_inbox_tick ON creator_inbox(tick);
CREATE INDEX IF NOT EXISTS idx_creator_inbox_object_id ON creator_inbox(object_id);

CREATE TABLE IF NOT EXISTS mailbox_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    tick INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    unix_timestamp REAL NOT NULL,
    mailbox_id TEXT,
    text TEXT,
    object_id TEXT,
    object_json TEXT,
    creator_note TEXT,
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mailbox_log_tick ON mailbox_log(tick);
CREATE INDEX IF NOT EXISTS idx_mailbox_log_kind ON mailbox_log(kind);
"""


class CreatorInbox:
    """Persistent SQLite creator inbox and creator/mail audit log."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        path: Path | str | None = None,
        config: MailboxConfig | Mapping[str, Any] | None = None,
        creator_note: str | None = None,
        note: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if isinstance(config, MailboxConfig):
            self.config = config.model_copy()
        elif isinstance(config, Mapping):
            self.config = MailboxConfig.model_validate(dict(config))
        else:
            self.config = MailboxConfig()

        resolved_note = creator_note if creator_note is not None else note
        if resolved_note is not None:
            self.config.creator_note = resolved_note

        explicit_path = db_path if db_path is not None else path
        if explicit_path is not None:
            raw_path_str = str(explicit_path).strip()
        elif config is not None:
            raw_path_str = str(self.config.db_path)
        else:
            raw_path_str = ":memory:"

        self._is_memory_db: bool = raw_path_str == ":memory:"
        self.db_path: Path = Path(raw_path_str)
        if not self._is_memory_db:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.db_path = self.db_path

        if connection is not None:
            self._conn: sqlite3.Connection | None = connection
        else:
            self._conn = sqlite3.connect(
                ":memory:" if self._is_memory_db else str(self.db_path),
                check_same_thread=False,
            )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_MAILBOX_TABLES_SQL)
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active SQLite connection, reopening if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                ":memory:" if self._is_memory_db else str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_CREATE_MAILBOX_TABLES_SQL)
            self._conn.commit()
        return self._conn

    @property
    def creator_note(self) -> str | None:
        """Private creator note attached to this experiment/inbox."""
        return self.config.creator_note

    @creator_note.setter
    def creator_note(self, value: str | None) -> None:
        self.config.creator_note = value

    @property
    def note(self) -> str | None:
        """Alias for ``creator_note``."""
        return self.config.creator_note

    @note.setter
    def note(self, value: str | None) -> None:
        self.config.creator_note = value

    def set_creator_note(self, note: str | None, *, tick: int = 0) -> None:
        """Set or update the private experiment creator note and record a log entry."""
        self.config.creator_note = note
        self.log_event(
            "creator_action",
            tick=tick,
            creator_note=note,
            details={"action": "set_creator_note"},
        )

    set_note = set_creator_note

    def log_event(
        self,
        kind: MailboxLogKind,
        *,
        tick: int,
        mailbox_id: str | None = None,
        text: str | None = None,
        object_id: str | None = None,
        object_data: Mapping[str, Any] | WorldObject | None = None,
        creator_note: str | None = None,
        details: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
        unix_timestamp: float | None = None,
    ) -> MailboxLogEntry:
        """Record a send, delivery, or creator action in ``mailbox_log``."""
        iso_ts, unix_ts = _utc_now()
        if timestamp is not None:
            iso_ts = timestamp
        if unix_timestamp is not None:
            unix_ts = float(unix_timestamp)

        effective_note = creator_note if creator_note is not None else self.config.creator_note
        obj_dict: dict[str, Any] | None = None
        if _is_world_object(object_data):
            obj_dict = object_data.model_dump(mode="json")  # type: ignore[union-attr]
        elif isinstance(object_data, Mapping):
            obj_dict = json.loads(json.dumps(dict(object_data)))

        details_dict = json.loads(json.dumps(dict(details))) if details else {}
        conn = self.connection
        cur = conn.execute(
            """
            INSERT INTO mailbox_log (
                kind, tick, timestamp, unix_timestamp, mailbox_id,
                text, object_id, object_json, creator_note, details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                int(tick),
                iso_ts,
                unix_ts,
                mailbox_id,
                text,
                object_id,
                json.dumps(obj_dict, sort_keys=True) if obj_dict is not None else None,
                effective_note,
                json.dumps(details_dict, sort_keys=True),
            ),
        )
        conn.commit()
        row_id = int(cur.lastrowid or 0)
        return MailboxLogEntry(
            id=row_id,
            kind=kind,
            tick=int(tick),
            timestamp=iso_ts,
            unix_timestamp=unix_ts,
            mailbox_id=mailbox_id,
            text=text,
            object_id=object_id,
            object_data=obj_dict,
            creator_note=effective_note,
            details=details_dict,
        )

    def deliver(
        self,
        *,
        tick: int,
        mailbox_id: str,
        text: str | None = None,
        texts: list[str] | None = None,
        obj: WorldObject | Mapping[str, Any] | None = None,
        delivery_reason: DeliveryReason = "lid_closed",
        creator_note: str | None = None,
    ) -> InboxEntry:
        """Persist a delivered message and/or object to the creator inbox and log the delivery."""
        iso_ts, unix_ts = _utc_now()
        effective_note = creator_note if creator_note is not None else self.config.creator_note

        text_list = list(texts) if texts is not None else ([text] if text is not None else [])
        combined_text = text if text is not None else ("\n".join(text_list) if text_list else None)

        obj_id: str | None = None
        obj_dict: dict[str, Any] | None = None
        if _is_world_object(obj):
            obj_id = obj.id  # type: ignore[union-attr]
            obj_dict = obj.model_dump(mode="json")  # type: ignore[union-attr]
        elif isinstance(obj, Mapping):
            obj_dict = json.loads(json.dumps(dict(obj)))
            raw_id = obj_dict.get("id")
            if isinstance(raw_id, str):
                obj_id = raw_id

        conn = self.connection
        cur = conn.execute(
            """
            INSERT INTO creator_inbox (
                tick, timestamp, unix_timestamp, mailbox_id,
                text, texts_json, object_id, object_json,
                delivery_reason, creator_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(tick),
                iso_ts,
                unix_ts,
                mailbox_id,
                combined_text,
                json.dumps(text_list),
                obj_id,
                json.dumps(obj_dict, sort_keys=True) if obj_dict is not None else None,
                delivery_reason,
                effective_note,
            ),
        )
        conn.commit()
        entry_id = int(cur.lastrowid or 0)

        entry = InboxEntry(
            id=entry_id,
            tick=int(tick),
            timestamp=iso_ts,
            unix_timestamp=unix_ts,
            mailbox_id=mailbox_id,
            text=combined_text,
            texts=text_list,
            object_id=obj_id,
            object_data=obj_dict,
            delivery_reason=delivery_reason,
            creator_note=effective_note,
        )

        self.log_event(
            "delivery",
            tick=int(tick),
            mailbox_id=mailbox_id,
            text=combined_text,
            object_id=obj_id,
            object_data=obj_dict,
            creator_note=effective_note,
            details={"inbox_id": entry_id, "delivery_reason": delivery_reason},
            timestamp=iso_ts,
            unix_timestamp=unix_ts,
        )
        return entry

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> InboxEntry:
        texts_raw = json.loads(row["texts_json"]) if row["texts_json"] else []
        obj_raw = json.loads(row["object_json"]) if row["object_json"] else None
        return InboxEntry(
            id=int(row["id"]),
            tick=int(row["tick"]),
            timestamp=str(row["timestamp"]),
            unix_timestamp=float(row["unix_timestamp"]),
            mailbox_id=str(row["mailbox_id"]),
            text=row["text"],
            texts=list(texts_raw),
            object_id=row["object_id"],
            object_data=obj_raw,
            delivery_reason=row["delivery_reason"],
            creator_note=row["creator_note"],
        )

    @staticmethod
    def _row_to_log(row: sqlite3.Row) -> MailboxLogEntry:
        obj_raw = json.loads(row["object_json"]) if row["object_json"] else None
        details_raw = json.loads(row["details_json"]) if row["details_json"] else {}
        return MailboxLogEntry(
            id=int(row["id"]),
            kind=row["kind"],
            tick=int(row["tick"]),
            timestamp=str(row["timestamp"]),
            unix_timestamp=float(row["unix_timestamp"]),
            mailbox_id=row["mailbox_id"],
            text=row["text"],
            object_id=row["object_id"],
            object_data=obj_raw,
            creator_note=row["creator_note"],
            details=details_raw,
        )

    def all(self) -> list[InboxEntry]:
        """Return all delivered entries in chronological order."""
        rows = self.connection.execute(
            "SELECT * FROM creator_inbox ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    entries = all
    list_entries = all
    messages = all
    items = all

    def latest(self) -> InboxEntry | None:
        """Return the most recently delivered ``InboxEntry``, or ``None`` if empty."""
        row = self.connection.execute(
            "SELECT * FROM creator_inbox ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def get(self, entry_id: int) -> InboxEntry | None:
        """Return the ``InboxEntry`` with primary key ``entry_id``, or ``None``."""
        row = self.connection.execute(
            "SELECT * FROM creator_inbox WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def preserved_objects(self) -> list[WorldObject]:
        """Return all preserved ``WorldObject`` instances delivered to the inbox."""
        result: list[WorldObject] = []
        for entry in self.all():
            obj = entry.object
            if obj is not None:
                result.append(obj)
        return result

    @property
    def objects(self) -> list[WorldObject]:
        """All preserved ``WorldObject`` instances delivered to the inbox."""
        return self.preserved_objects()

    def get_preserved_object(self, object_or_entry_id: str | int) -> WorldObject | None:
        """Look up a preserved ``WorldObject`` by its ``object_<n>`` id or inbox entry id."""
        if isinstance(object_or_entry_id, int) and not isinstance(object_or_entry_id, bool):
            entry = self.get(object_or_entry_id)
            return entry.object if entry is not None else None
        target_id = str(object_or_entry_id)
        row = self.connection.execute(
            "SELECT * FROM creator_inbox WHERE object_id = ? ORDER BY id DESC LIMIT 1",
            (target_id,),
        ).fetchone()
        if row is not None:
            return self._row_to_entry(row).object
        return None

    def logs(self, *, kind: MailboxLogKind | None = None) -> list[MailboxLogEntry]:
        """Return all audit log entries in chronological order (optionally filtered by ``kind``)."""
        if kind is not None:
            rows = self.connection.execute(
                "SELECT * FROM mailbox_log WHERE kind = ? ORDER BY id ASC",
                (kind,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM mailbox_log ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_log(r) for r in rows]

    log_entries = logs
    list_logs = logs

    def clear(self) -> None:
        """Delete all entries from ``creator_inbox`` and ``mailbox_log``."""
        conn = self.connection
        conn.execute("DELETE FROM creator_inbox")
        conn.execute("DELETE FROM mailbox_log")
        conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize inbox entries and logs for world snapshots."""
        return {
            "creator_note": self.config.creator_note,
            "entries": [e.to_dict() for e in self.all()],
            "logs": [l.to_dict() for l in self.logs()],
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Restore inbox entries and logs from a ``to_dict()`` snapshot."""
        self.clear()
        if "creator_note" in data:
            self.config.creator_note = data["creator_note"]
        conn = self.connection
        for raw_entry in data.get("entries", ()) or ():
            if not isinstance(raw_entry, Mapping):
                continue
            conn.execute(
                """
                INSERT INTO creator_inbox (
                    id, tick, timestamp, unix_timestamp, mailbox_id,
                    text, texts_json, object_id, object_json,
                    delivery_reason, creator_note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(raw_entry["id"]),
                    int(raw_entry["tick"]),
                    str(raw_entry["timestamp"]),
                    float(raw_entry["unix_timestamp"]),
                    str(raw_entry["mailbox_id"]),
                    raw_entry.get("text"),
                    json.dumps(list(raw_entry.get("texts") or [])),
                    raw_entry.get("object_id"),
                    (
                        json.dumps(raw_entry["object_data"], sort_keys=True)
                        if raw_entry.get("object_data") is not None
                        else None
                    ),
                    str(raw_entry.get("delivery_reason", "lid_closed")),
                    raw_entry.get("creator_note"),
                ),
            )
        for raw_log in data.get("logs", ()) or ():
            if not isinstance(raw_log, Mapping):
                continue
            conn.execute(
                """
                INSERT INTO mailbox_log (
                    id, kind, tick, timestamp, unix_timestamp, mailbox_id,
                    text, object_id, object_json, creator_note, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(raw_log["id"]),
                    str(raw_log["kind"]),
                    int(raw_log["tick"]),
                    str(raw_log["timestamp"]),
                    float(raw_log["unix_timestamp"]),
                    raw_log.get("mailbox_id"),
                    raw_log.get("text"),
                    raw_log.get("object_id"),
                    (
                        json.dumps(raw_log["object_data"], sort_keys=True)
                        if raw_log.get("object_data") is not None
                        else None
                    ),
                    raw_log.get("creator_note"),
                    json.dumps(dict(raw_log.get("details") or {}), sort_keys=True),
                ),
            )
        conn.commit()

    def __len__(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS cnt FROM creator_inbox").fetchone()
        return int(row["cnt"]) if row is not None else 0

    def __iter__(self) -> Iterator[InboxEntry]:
        return iter(self.all())

    def __getitem__(self, index: int | slice) -> InboxEntry | list[InboxEntry]:
        return self.all()[index]

    def __bool__(self) -> bool:
        return len(self) > 0


def has_lid(obj: WorldObject) -> bool:
    """Return True if ``obj`` has a perceptible ``lid`` state or ``toggle_lid`` rule."""
    if "lid" in obj.state and obj.state["lid"] in ("open", "closed"):
        return True
    for rule in obj.behavior:
        if rule.verb in ("grab", "separate"):
            extras = rule.__pydantic_extra__ or {}
            if str(extras.get("response", "")).lower() == "toggle_lid":
                return True
    return False


def is_mailbox_object(obj: WorldObject) -> bool:
    """Return True if ``obj`` is configured as a mailbox."""
    for rule in obj.behavior:
        extras = rule.__pydantic_extra__ or {}
        if str(extras.get("response", "")).lower() in (
            "capture_mail",
            "send_object",
            "toggle_lid",
            "mailbox",
        ):
            return True
    if obj.creator_label is not None and "mailbox" in obj.creator_label.lower():
        return True
    return has_lid(obj)


def is_inside_closed_container(world: World, obj: WorldObject) -> bool:
    """Return True if ``obj`` is inside a container whose ``state['lid']`` is ``'closed'``."""
    if not obj.level.startswith("inside:"):
        return False
    container_id = obj.level.removeprefix("inside:")
    container = world.objects.get(container_id)
    if container is None:
        return False
    return container.state.get("lid") == "closed"


class _PendingSpeechWindow(BaseModel):
    """In-flight speech captured while a mailbox lid is open."""

    model_config = ConfigDict(extra="forbid")

    mailbox_id: str
    texts: list[str] = Field(default_factory=list)
    first_captured_tick: int = 0
    last_captured_tick: int = 0
    creator_note: str | None = None


class MailboxManager:
    """Manages mailbox lid state, outgoing speech/object capture, timeout delivery, replies, and lid drift."""

    def __init__(
        self,
        world: World,
        *,
        config: MailboxConfig | Mapping[str, Any] | None = None,
        inbox: CreatorInbox | None = None,
        db_path: Path | str | None = None,
        creator_note: str | None = None,
        seed: int = 0,
    ) -> None:
        self.world: World = world
        self.config: MailboxConfig = self._resolve_config(config, world, creator_note=creator_note)

        explicit_db = db_path
        if explicit_db is None and config is not None:
            if isinstance(config, MailboxConfig):
                explicit_db = config.db_path
            elif isinstance(config, Mapping) and "db_path" in config:
                explicit_db = config["db_path"]
        elif explicit_db is None:
            world_cfg = getattr(world, "config", None)
            if world_cfg is not None and hasattr(world_cfg, "mailbox"):
                mb_sub = getattr(world_cfg, "mailbox")
                if isinstance(mb_sub, MailboxConfig) and mb_sub.db_path != DEFAULT_MAILBOX_DB_PATH:
                    explicit_db = mb_sub.db_path

        if inbox is not None:
            self.inbox: CreatorInbox = inbox
            if creator_note is not None:
                self.inbox.creator_note = creator_note
        else:
            self.inbox = CreatorInbox(
                db_path=explicit_db if explicit_db is not None else ":memory:",
                config=self.config,
                creator_note=self.config.creator_note,
            )

        self._pending_speech: dict[str, _PendingSpeechWindow] = {}
        self.arrivals: list[MailArrivedEvent] = []
        self._drift_rng: random.Random = random.Random(int(seed) ^ 0x5A17B0)

    @staticmethod
    def _resolve_config(
        config: MailboxConfig | Mapping[str, Any] | None,
        world: World,
        *,
        creator_note: str | None = None,
    ) -> MailboxConfig:
        base: MailboxConfig | None = None
        if isinstance(config, MailboxConfig):
            base = config.model_copy()
        elif isinstance(config, Mapping):
            base = MailboxConfig.model_validate(dict(config))
        else:
            world_cfg = getattr(world, "config", None)
            if world_cfg is not None:
                mb_cfg = getattr(world_cfg, "mailbox", None)
                if isinstance(mb_cfg, MailboxConfig):
                    base = mb_cfg.model_copy()
                elif isinstance(mb_cfg, Mapping):
                    base = MailboxConfig.model_validate(dict(mb_cfg))
        if base is None:
            base = MailboxConfig()
        if creator_note is not None:
            base.creator_note = creator_note
        return base

    @property
    def tick_rate(self) -> float:
        """World tick rate in Hz (default ``10.0``)."""
        world_cfg = getattr(self.world, "config", None)
        if world_cfg is not None:
            w_sub = getattr(world_cfg, "world", None)
            if w_sub is not None and hasattr(w_sub, "tick_rate"):
                return float(w_sub.tick_rate)
            if hasattr(world_cfg, "tick_rate"):
                return float(world_cfg.tick_rate)
        return DEFAULT_TICK_RATE

    @property
    def delivery_delay_ticks(self) -> int:
        """Speech delivery timeout in simulation ticks."""
        return self.config.delay_in_ticks(self.tick_rate)

    @property
    def creator_note(self) -> str | None:
        """Private creator note for this experiment."""
        return self.inbox.creator_note

    @creator_note.setter
    def creator_note(self, value: str | None) -> None:
        self.config.creator_note = value
        self.inbox.creator_note = value

    def set_creator_note(self, note: str | None) -> None:
        """Set or update the private experiment creator note and log the change."""
        self.config.creator_note = note
        self.inbox.set_creator_note(note, tick=self.world.tick)

    set_note = set_creator_note

    def mailbox_objects(self) -> list[WorldObject]:
        """Return all mailbox objects in ``world.objects`` sorted by id."""
        return [
            self.world.objects[oid]
            for oid in sorted(self.world.objects)
            if is_mailbox_object(self.world.objects[oid])
        ]

    def mailbox_object(self, mailbox_id: str | WorldObject | None = None) -> WorldObject | None:
        """Resolve a specific or primary mailbox object in ``world.objects``."""
        if _is_world_object(mailbox_id):
            return self.world.objects.get(mailbox_id.id, mailbox_id)  # type: ignore[union-attr, return-value]
        if isinstance(mailbox_id, str):
            return self.world.objects.get(mailbox_id)
        boxes = self.mailbox_objects()
        return boxes[0] if boxes else None

    def pending_speech(self, mailbox_id: str | WorldObject | None = None) -> list[str]:
        """Return currently buffered (undelivered) speech lines for the mailbox."""
        mb = self.mailbox_object(mailbox_id)
        target_id = mb.id if mb is not None else (mailbox_id if isinstance(mailbox_id, str) else None)
        if target_id is None:
            return []
        window = self._pending_speech.get(target_id)
        return list(window.texts) if window is not None else []

    @property
    def object_id(self) -> str | None:
        """ID of the primary mailbox object in the world, if any."""
        mb = self.mailbox_object()
        return mb.id if mb is not None else None

    def lid_state(self, mailbox_id: str | WorldObject | None = None) -> Literal["open", "closed"] | None:
        """Return ``'open'`` or ``'closed'`` for the mailbox lid, or ``None`` if no mailbox exists."""
        mb = self.mailbox_object(mailbox_id)
        if mb is None:
            return None
        raw = mb.state.get("lid", "closed")
        return "open" if raw == "open" else "closed"

    @property
    def lid(self) -> Literal["open", "closed"] | None:
        """Lid state of the primary mailbox (``'open'`` or ``'closed'``)."""
        return self.lid_state()

    def is_open(self, mailbox_id: str | WorldObject | None = None) -> bool:
        """Return True when the mailbox lid is ``'open'``."""
        return self.lid_state(mailbox_id) == "open"

    def is_closed(self, mailbox_id: str | WorldObject | None = None) -> bool:
        """Return True when the mailbox lid is ``'closed'``."""
        return self.lid_state(mailbox_id) == "closed"

    def contents(self, mailbox_id: str | WorldObject | None = None) -> list[WorldObject]:
        """Return all world objects currently inside the mailbox."""
        mb = self.mailbox_object(mailbox_id)
        if mb is None:
            return []
        return [
            self.world.objects[oid]
            for oid in sorted(self.world.objects)
            if self.world.objects[oid].level == f"inside:{mb.id}"
        ]

    def toggle_lid(
        self,
        mailbox_id: str | WorldObject | None = None,
        *,
        source: str = "agent",
    ) -> Literal["open", "closed"]:
        """Toggle the mailbox lid between ``'open'`` and ``'closed'``.

        When closing from ``'open'`` to ``'closed'``, any pending captured speech
        is immediately delivered to the creator inbox.
        """
        mb = self.mailbox_object(mailbox_id)
        if mb is None:
            raise KeyError("no mailbox object found in world")

        current_lid = "open" if mb.state.get("lid") == "open" else "closed"
        new_lid: Literal["open", "closed"] = "closed" if current_lid == "open" else "open"
        new_state = {**mb.state, "lid": new_lid}
        self.world.update_object(mb.id, {"state": new_state}, source=source)

        if current_lid == "open" and new_lid == "closed":
            self._flush_pending_speech(mb.id, delivery_reason="lid_closed")

        return new_lid

    def _flush_pending_speech(
        self,
        mailbox_id: str,
        *,
        delivery_reason: DeliveryReason,
        obj: WorldObject | None = None,
        note: str | None = None,
    ) -> InboxEntry | None:
        window = self._pending_speech.pop(mailbox_id, None)
        if window is None and obj is None:
            return None

        texts = list(window.texts) if window is not None else []
        combined_text = "\n".join(texts) if texts else None
        effective_note = (
            note
            if note is not None
            else (window.creator_note if window is not None else None)
        )
        return self.inbox.deliver(
            tick=self.world.tick,
            mailbox_id=mailbox_id,
            text=combined_text,
            texts=texts,
            obj=obj,
            delivery_reason=delivery_reason,
            creator_note=effective_note,
        )

    def capture_speech(
        self,
        text: str,
        *,
        reach: float = 1.5,
        modifiers: Any = None,
        note: str | None = None,
    ) -> bool:
        """Capture spoken ``text`` if an open mailbox is within reach of the agent.

        Returns True if captured by an open mailbox within reach, False otherwise.
        """
        from arden.actions.verbs import in_reach

        captured_any = False
        for mb in self.mailbox_objects():
            if mb.state.get("lid") != "open":
                continue
            if not in_reach(self.world, mb, reach=reach, modifiers=modifiers):
                continue

            window = self._pending_speech.get(mb.id)
            if window is None:
                window = _PendingSpeechWindow(
                    mailbox_id=mb.id,
                    texts=[text],
                    first_captured_tick=self.world.tick,
                    last_captured_tick=self.world.tick,
                    creator_note=note if note is not None else self.creator_note,
                )
                self._pending_speech[mb.id] = window
            else:
                window.texts.append(text)
                window.last_captured_tick = self.world.tick
                if note is not None:
                    window.creator_note = note

            self.inbox.log_event(
                "send",
                tick=self.world.tick,
                mailbox_id=mb.id,
                text=text,
                creator_note=note if note is not None else self.creator_note,
                details={"medium": "speak"},
            )
            if self.delivery_delay_ticks == 0:
                self._flush_pending_speech(mb.id, delivery_reason="timeout", note=note)
            captured_any = True

        return captured_any

    def deliver_inserted_object(
        self,
        held_obj: WorldObject,
        mailbox_obj: WorldObject,
        *,
        note: str | None = None,
    ) -> InboxEntry:
        """Remove ``held_obj`` from the world, preserve it, and deliver it with any pending speech."""
        preserved_copy = held_obj.model_copy(deep=True)
        if held_obj.id in self.world.objects:
            self.world.despawn(held_obj.id, source="agent")

        self.inbox.log_event(
            "send",
            tick=self.world.tick,
            mailbox_id=mailbox_obj.id,
            object_id=preserved_copy.id,
            object_data=preserved_copy,
            creator_note=note if note is not None else self.creator_note,
            details={"medium": "insert"},
        )

        entry = self._flush_pending_speech(
            mailbox_obj.id,
            delivery_reason="insert",
            obj=preserved_copy,
            note=note,
        )
        assert entry is not None
        return entry

    def step(self) -> list[InboxEntry]:
        """Advance mailbox timers and optional lid drift by one simulation tick."""
        delivered: list[InboxEntry] = []
        delay_ticks = self.delivery_delay_ticks

        # 1. Deliver any pending speech that has reached the timeout delay
        for mb_id in sorted(list(self._pending_speech.keys())):
            window = self._pending_speech[mb_id]
            # If the mailbox was removed or its lid is already closed, flush immediately
            mb = self.world.objects.get(mb_id)
            if mb is None or mb.state.get("lid") != "open":
                entry = self._flush_pending_speech(mb_id, delivery_reason="lid_closed")
                if entry is not None:
                    delivered.append(entry)
                continue

            if self.world.tick - window.first_captured_tick >= delay_ticks:
                entry = self._flush_pending_speech(mb_id, delivery_reason="timeout")
                if entry is not None:
                    delivered.append(entry)

        # 2. Optional lid drift nudge (per in-game hour, default 0.0)
        drift_chance = self.config.lid_drift_chance
        if drift_chance > 0.0:
            ticks_per_hour = 25
            weather = getattr(self.world, "weather", None)
            if weather is not None and hasattr(weather, "config"):
                tph = getattr(weather.config, "ticks_per_hour", 25)
                if isinstance(tph, int) and tph > 0:
                    ticks_per_hour = tph

            if drift_chance >= 1.0:
                p_tick = 1.0
            else:
                p_tick = 1.0 - (1.0 - drift_chance) ** (1.0 / ticks_per_hour)

            for mb in self.mailbox_objects():
                if mb.state.get("lid") == "closed":
                    if self._drift_rng.random() < p_tick:
                        new_state = {**mb.state, "lid": "open"}
                        self.world.update_object(mb.id, {"state": new_state}, source="world")
                        self.inbox.log_event(
                            "lid_drift",
                            tick=self.world.tick,
                            mailbox_id=mb.id,
                            creator_note=self.creator_note,
                            details={"lid": "open", "lid_drift_chance": drift_chance},
                        )

        return delivered

    def reply(
        self,
        text: str,
        *,
        mailbox_id: str | WorldObject | None = None,
        color: str | None = "white",
        note: str | None = None,
    ) -> WorldObject:
        """Place a flat letter object with perceptible ``text`` inside the mailbox and emit ``MailArrivedEvent``."""
        from arden.world.world import MailArrivedEvent

        if not isinstance(text, str):
            raise TypeError("reply text must be a string")

        mb = self.mailbox_object(mailbox_id)
        if mb is None:
            raise KeyError("no mailbox object found in world")

        letter_props: dict[str, Any] = {
            "shape": "flat",
            "height": 0.01,
            "volume": 0.0002,
            "weight": 0.02,
            "rigidity": 0.2,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": color,
            "fixed": False,
            "level": f"inside:{mb.id}",
            "position": {"x": mb.position.x, "y": mb.position.y},
            "creator_label": "creator letter",
            "state": {"text": text},
        }
        letter = self.world.spawn(letter_props, mb.position, source="mailbox")

        mail_event = MailArrivedEvent(
            tick=self.world.tick,
            source="creator",
            mailbox_id=mb.id,
            text=text,
            object_id=letter.id,
            arrival_kind="reply",
            position=mb.position.model_copy(),
        )
        self.arrivals.append(mail_event)
        self.world.events.emit(mail_event)

        self.inbox.log_event(
            "reply",
            tick=self.world.tick,
            mailbox_id=mb.id,
            text=text,
            object_id=letter.id,
            object_data=letter,
            creator_note=note if note is not None else self.creator_note,
        )
        return letter

    def spawn_in_response(
        self,
        props: Mapping[str, Any] | WorldObject | InboxEntry,
        *,
        mailbox_id: str | WorldObject | None = None,
        note: str | None = None,
    ) -> WorldObject:
        """Place an object inside the mailbox with no note and emit ``MailArrivedEvent``."""
        from arden.world.world import MailArrivedEvent

        mb = self.mailbox_object(mailbox_id)
        if mb is None:
            raise KeyError("no mailbox object found in world")

        if isinstance(props, InboxEntry):
            if props.object is None:
                raise ValueError("InboxEntry does not contain a preserved object")
            props = props.object

        if _is_world_object(props):
            raw_data = props.model_dump()  # type: ignore[union-attr]
            # Keep original ID if it is not currently in world.objects
            orig_id = raw_data.get("id")
            if isinstance(orig_id, str) and orig_id not in self.world.objects and orig_id != "object_0":
                raw_data["position"] = {"x": mb.position.x, "y": mb.position.y}
                raw_data["level"] = f"inside:{mb.id}"
                raw_data["velocity"] = {"x": 0.0, "y": 0.0}
                state_copy = dict(raw_data.get("state") or {})
                state_copy.pop("text", None)
                raw_data["state"] = state_copy
                restored = _validate_world_object(raw_data)
                self.world._factory._claim(restored.id)
                self.world.objects[restored.id] = restored
                from arden.world.world import ObjectSpawnedEvent

                self.world.events.emit(
                    ObjectSpawnedEvent(
                        tick=self.world.tick,
                        source="mailbox",
                        object_id=restored.id,
                        position=restored.position,
                        object=restored.model_dump(mode="json"),
                    )
                )
                spawned = restored
            else:
                raw_data.pop("id", None)
                raw_data["position"] = {"x": mb.position.x, "y": mb.position.y}
                raw_data["level"] = f"inside:{mb.id}"
                raw_data["velocity"] = {"x": 0.0, "y": 0.0}
                state_copy = dict(raw_data.get("state") or {})
                state_copy.pop("text", None)
                raw_data["state"] = state_copy
                spawned = self.world.spawn(raw_data, mb.position, source="mailbox")
        elif isinstance(props, Mapping):
            data: dict[str, Any] = {
                "shape": "cube",
                "height": 0.15,
                "volume": 0.002,
                "weight": 0.25,
                "rigidity": 0.8,
                "binding": False,
                "hollow": False,
                "fragile": False,
                "color": None,
                "fixed": False,
                **dict(props),
            }
            data["position"] = {"x": mb.position.x, "y": mb.position.y}
            data["level"] = f"inside:{mb.id}"
            state_copy = dict(data.get("state") or {})
            state_copy.pop("text", None)
            data["state"] = state_copy
            orig_id = data.get("id")
            if isinstance(orig_id, str) and orig_id not in self.world.objects and orig_id != "object_0":
                restored = _validate_world_object(data)
                self.world._factory._claim(restored.id)
                self.world.objects[restored.id] = restored
                from arden.world.world import ObjectSpawnedEvent

                self.world.events.emit(
                    ObjectSpawnedEvent(
                        tick=self.world.tick,
                        source="mailbox",
                        object_id=restored.id,
                        position=restored.position,
                        object=restored.model_dump(mode="json"),
                    )
                )
                spawned = restored
            else:
                data.pop("id", None)
                spawned = self.world.spawn(data, mb.position, source="mailbox")
        else:
            raise TypeError("props must be a mapping or WorldObject")

        mail_event = MailArrivedEvent(
            tick=self.world.tick,
            source="creator",
            mailbox_id=mb.id,
            text=None,
            object_id=spawned.id,
            arrival_kind="spawn_in_response",
            position=mb.position.model_copy(),
        )
        self.arrivals.append(mail_event)
        self.world.events.emit(mail_event)

        self.inbox.log_event(
            "spawn_in_response",
            tick=self.world.tick,
            mailbox_id=mb.id,
            object_id=spawned.id,
            object_data=spawned,
            creator_note=note if note is not None else self.creator_note,
        )
        return spawned

    def return_object(
        self,
        object_or_entry: str | int | InboxEntry | WorldObject,
        *,
        mailbox_id: str | WorldObject | None = None,
        note: str | None = None,
    ) -> WorldObject:
        """Return a previously delivered object from the creator inbox back into the mailbox."""
        if _is_world_object(object_or_entry):
            target_obj: WorldObject | None = object_or_entry  # type: ignore[assignment]
        elif isinstance(object_or_entry, InboxEntry):
            target_obj = object_or_entry.object
        else:
            target_obj = self.inbox.get_preserved_object(object_or_entry)  # type: ignore[arg-type]

        if target_obj is None:
            raise KeyError(f"no preserved object found for {object_or_entry!r}")
        return self.spawn_in_response(target_obj, mailbox_id=mailbox_id, note=note)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mailbox state, pending speech windows, and inbox to a dictionary."""
        version, internal_state, gauss_next = self._drift_rng.getstate()
        return {
            "config": self.config.model_dump(mode="json"),
            "pending_speech": {
                k: v.model_dump(mode="json") for k, v in self._pending_speech.items()
            },
            "inbox": self.inbox.to_dict(),
            "drift_rng_state": [version, list(internal_state), gauss_next],
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Restore mailbox state from a ``to_dict()`` snapshot."""
        if "config" in data and isinstance(data["config"], Mapping):
            self.config = MailboxConfig.model_validate(dict(data["config"]))
        self._pending_speech = {}
        raw_pending = data.get("pending_speech")
        if isinstance(raw_pending, Mapping):
            for k, v in raw_pending.items():
                if isinstance(v, Mapping):
                    self._pending_speech[str(k)] = _PendingSpeechWindow.model_validate(dict(v))
        raw_inbox = data.get("inbox")
        if isinstance(raw_inbox, Mapping):
            self.inbox.load_dict(raw_inbox)
        raw_rng = data.get("drift_rng_state")
        if isinstance(raw_rng, list | tuple) and len(raw_rng) == 3:
            version, internal_state, gauss_next = raw_rng
            if isinstance(internal_state, list | tuple):
                self._drift_rng.setstate(
                    (int(version), tuple(int(x) for x in internal_state), gauss_next)
                )


Mailbox = MailboxManager


def _resolve_mailbox_world(world: World | None = None) -> World:
    from arden.world.physics import _get_active_world, _set_active_world

    if world is not None:
        _set_active_world(world)
        return world
    active = _get_active_world()
    if active is None:
        raise RuntimeError("no active World found; pass world explicitly or instantiate a World first")
    return active


def reply(
    world_or_text: World | str,
    text: str | None = None,
    *,
    world: World | None = None,
    mailbox_id: str | WorldObject | None = None,
    color: str | None = "white",
    note: str | None = None,
) -> WorldObject:
    """Place a flat letter object with perceptible ``text`` inside the mailbox.

    Supports both ``reply(text)`` (on the active ``World``) and ``reply(world, text)``.
    """
    from arden.world.world import World as _World

    if isinstance(world_or_text, _World):
        resolved_world = _resolve_mailbox_world(world_or_text)
        if text is None:
            raise TypeError("reply(world, text) requires a text argument")
        actual_text = text
    else:
        resolved_world = _resolve_mailbox_world(world)
        actual_text = world_or_text

    return resolved_world.mailbox.reply(
        actual_text,
        mailbox_id=mailbox_id,
        color=color,
        note=note,
    )


def spawn_in_response(
    world_or_props: World | Mapping[str, Any] | WorldObject,
    props: Mapping[str, Any] | WorldObject | None = None,
    *,
    world: World | None = None,
    mailbox_id: str | WorldObject | None = None,
    note: str | None = None,
) -> WorldObject:
    """Place an object inside the mailbox with no note in response to the agent.

    Supports both ``spawn_in_response(props)`` (on the active ``World``) and
    ``spawn_in_response(world, props)``.
    """
    from arden.world.world import World as _World

    if isinstance(world_or_props, _World):
        resolved_world = _resolve_mailbox_world(world_or_props)
        if props is None:
            raise TypeError("spawn_in_response(world, props) requires a props argument")
        actual_props = props
    else:
        resolved_world = _resolve_mailbox_world(world)
        actual_props = world_or_props

    return resolved_world.mailbox.spawn_in_response(
        actual_props,
        mailbox_id=mailbox_id,
        note=note,
    )


__all__ = [
    "DEFAULT_MAILBOX_DB_PATH",
    "DEFAULT_MAILBOX_DELIVERY_DELAY_S",
    "DEFAULT_MAILBOX_LID_DRIFT_CHANCE",
    "CreatorInbox",
    "DeliveryReason",
    "InboxEntry",
    "InboxItem",
    "MailMessage",
    "Mailbox",
    "MailboxConfig",
    "MailboxLogEntry",
    "MailboxLogKind",
    "MailboxManager",
    "MailboxSettings",
    "has_lid",
    "is_inside_closed_container",
    "is_mailbox_object",
    "reply",
    "spawn_in_response",
]
