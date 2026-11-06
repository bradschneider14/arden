"""SQLite episodic memory store, episode schema, salience adjustment, and decay/forgetting.

Rules from docs/DESIGN.md (Sections 9.1, 10.4, 11) and Ticket T13:
- Reflection compresses the short-term buffer into short episodes with a salience score in ``[0.0, 1.0]``.
- Episodes record ``summary`` text, ``tags`` (such as object ids and verbs), ``salience`` (``0..1``),
  ``surprise_kind`` if any (``"physical"``, ``"expectation"``, ``"belief"``, ``"novelty"``),
  ``made_by_agent`` references, world ``tick`` timestamp, and ``last_recalled_tick``.
- Persisted via stdlib ``sqlite3``; default DB path from config under ``data/`` (``data/memory.db``).
- Comfort and surprise raise salience (uncomfortable or surprising episodes are remembered more strongly).
- Salience decays over world time unless recalled; episodes below a floor are deleted (real forgetting).
- Recalling an episode refreshes its ``last_recalled_tick`` and salience.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arden.config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_DECAY_RATE,
    DEFAULT_DISCOMFORT_SALIENCE_BOOST,
    DEFAULT_RECALL_BOOST,
    DEFAULT_REST_IDLE_TICKS,
    DEFAULT_SALIENCE_FLOOR,
    DEFAULT_SURPRISE_SALIENCE_BOOST,
    MemoryConfig,
    MemorySettings,
    Settings,
)

SURPRISE_KINDS: tuple[str, ...] = ("physical", "expectation", "belief", "novelty")
SurpriseKind = Literal["physical", "expectation", "belief", "novelty"]

# Expectation and belief surprises outrank physical ones (docs/DESIGN.md 10.4).
SURPRISE_KIND_WEIGHTS: dict[str, float] = {
    "novelty": 0.25,
    "physical": 0.30,
    "expectation": 0.50,
    "belief": 0.55,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def normalize_surprise_kind(value: Any) -> SurpriseKind | None:
    """Normalize a surprise kind value to one of ``SURPRISE_KINDS`` or ``None``."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned or cleaned in ("none", "null", "normal"):
            return None
        if cleaned in SURPRISE_KINDS:
            return cleaned  # type: ignore[return-value]
        raise ValueError(f"surprise_kind must be one of {SURPRISE_KINDS} or None, got {value!r}")
    raise ValueError(f"surprise_kind must be a string or None, got {type(value).__name__}")


def _normalize_str_list(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence):
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
    raise ValueError(f"expected a list of strings, got {type(value).__name__}")


DEFAULT_MEMORY_CONFIG = MemoryConfig()


def resolve_memory_config(
    config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
    *,
    settings: Settings | None = None,
) -> MemoryConfig:
    """Resolve a ``MemoryConfig`` from ``config``, ``settings``, or defaults."""
    if isinstance(config, MemoryConfig):
        return config
    if config is not None and hasattr(config, "memory") and isinstance(getattr(config, "memory"), MemoryConfig):
        return getattr(config, "memory")
    if settings is not None and hasattr(settings, "memory") and isinstance(getattr(settings, "memory"), MemoryConfig):
        return getattr(settings, "memory")
    if isinstance(config, Mapping):
        if "memory" in config and isinstance(config["memory"], Mapping | MemoryConfig):
            return resolve_memory_config(config["memory"])
        raw = dict(config)
        # Support common constructor aliases when passed as a mapping
        if "capacity" in raw and "buffer_size" not in raw:
            raw["buffer_size"] = raw.pop("capacity")
        if "short_term_size" in raw and "buffer_size" not in raw:
            raw["buffer_size"] = raw.pop("short_term_size")
        if "max_size" in raw and "buffer_size" not in raw:
            raw["buffer_size"] = raw.pop("max_size")
        if "idle_ticks" in raw and "rest_idle_ticks" not in raw:
            raw["rest_idle_ticks"] = raw.pop("idle_ticks")
        if "rest_ticks" in raw and "rest_idle_ticks" not in raw:
            raw["rest_idle_ticks"] = raw.pop("rest_ticks")
        if "floor" in raw and "salience_floor" not in raw:
            raw["salience_floor"] = raw.pop("floor")
        if "min_salience" in raw and "salience_floor" not in raw:
            raw["salience_floor"] = raw.pop("min_salience")
        if "path" in raw and "db_path" not in raw:
            raw["db_path"] = raw.pop("path")
        return MemoryConfig.model_validate(raw)
    return DEFAULT_MEMORY_CONFIG


class ReflectionEpisode(BaseModel):
    """Schema for a single episode returned by the LLM ``reflect`` role."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    tags: list[str] = Field(default_factory=list)
    salience: float = Field(ge=0.0, le=1.0)
    surprise_kind: SurpriseKind | None = None
    made_by_agent: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _summary_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("episode summary must be a non-empty string")
        return stripped

    @field_validator("surprise_kind", mode="before")
    @classmethod
    def _validate_surprise_kind(cls, value: Any) -> SurpriseKind | None:
        return normalize_surprise_kind(value)

    @field_validator("tags", "made_by_agent", mode="before")
    @classmethod
    def _validate_str_lists(cls, value: Any) -> list[str]:
        return _normalize_str_list(value)


class ReflectionOutput(BaseModel):
    """Structured JSON output schema for the LLM ``reflect`` role."""

    model_config = ConfigDict(extra="forbid")

    episodes: list[ReflectionEpisode]


ReflectionResult = ReflectionOutput
ReflectResponse = ReflectionOutput
ReflectSchema = ReflectionOutput


class Episode(BaseModel):
    """An episodic memory compressed from the short-term buffer and stored in SQLite."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    tick: int = Field(default=0, ge=0)
    summary: str
    tags: list[str] = Field(default_factory=list)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    last_recalled_tick: int = Field(default=0, ge=0)
    surprise_kind: SurpriseKind | None = None
    made_by_agent: list[str] = Field(default_factory=list)
    recall_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, values: Any) -> Any:
        if isinstance(values, ReflectionEpisode):
            return {
                "summary": values.summary,
                "tags": list(values.tags),
                "salience": values.salience,
                "surprise_kind": values.surprise_kind,
                "made_by_agent": list(values.made_by_agent),
            }
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        if "text" in raw and "summary" not in raw:
            raw["summary"] = raw.pop("text")
        elif "text" in raw and raw["text"] == raw.get("summary"):
            raw.pop("text")

        for tick_alias in ("timestamp", "created_tick", "created_at", "world_tick"):
            if tick_alias in raw:
                val = raw.pop(tick_alias)
                if "tick" not in raw and val is not None:
                    raw["tick"] = val

        for recall_alias in ("last_recalled", "recalled_tick", "last_recall_tick"):
            if recall_alias in raw:
                val = raw.pop(recall_alias)
                if "last_recalled_tick" not in raw and val is not None:
                    raw["last_recalled_tick"] = val

        tick_val = int(raw.get("tick", 0) or 0)
        raw["tick"] = tick_val
        if raw.get("last_recalled_tick") is None:
            raw["last_recalled_tick"] = tick_val
        return raw

    @field_validator("summary")
    @classmethod
    def _summary_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("episode summary must be a non-empty string")
        return stripped

    @field_validator("salience")
    @classmethod
    def _salience_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("salience must be finite")
        return float(value)

    @field_validator("surprise_kind", mode="before")
    @classmethod
    def _validate_surprise_kind(cls, value: Any) -> SurpriseKind | None:
        return normalize_surprise_kind(value)

    @field_validator("tags", "made_by_agent", mode="before")
    @classmethod
    def _validate_str_lists(cls, value: Any) -> list[str]:
        return _normalize_str_list(value)

    @property
    def text(self) -> str:
        """Alias for ``summary``."""
        return self.summary

    @property
    def timestamp(self) -> int:
        """Alias for ``tick`` (world tick when the episode occurred)."""
        return self.tick

    @property
    def created_tick(self) -> int:
        """Alias for ``tick``."""
        return self.tick

    @property
    def last_recalled(self) -> int:
        """Alias for ``last_recalled_tick``."""
        return self.last_recalled_tick

    @property
    def recalled_tick(self) -> int:
        """Alias for ``last_recalled_tick``."""
        return self.last_recalled_tick

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_text(self) -> str:
        """Render a compact text representation suitable for prompt retrieval."""
        parts = [f"tick={self.tick}", f"salience={round(self.salience, 4)}: {self.summary}"]
        if self.tags:
            parts.append(f"tags=[{','.join(self.tags)}]")
        if self.surprise_kind is not None:
            parts.append(f"surprise={self.surprise_kind}")
        if self.made_by_agent:
            parts.append(f"made_by_agent=[{','.join(self.made_by_agent)}]")
        return " ".join(parts)


def compute_episode_salience(
    base_salience: float = 0.5,
    *,
    comfort: float | None = None,
    surprise: float | bool | None = None,
    surprise_kind: SurpriseKind | str | None = None,
    body: Any = None,
    drives: Any = None,
    config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
) -> float:
    """Adjust episode salience in ``[0.0, 1.0]`` for body comfort and surprise.

    - Lower ``comfort`` (discomfort = ``1 - comfort``) increases salience so
      uncomfortable episodes are remembered more strongly (docs/DESIGN.md 9.1).
    - Higher ``surprise`` and ``surprise_kind`` (``"belief"``/``"expectation"`` >
      ``"physical"``/``"novelty"``) increase salience so surprising episodes are
      remembered more strongly (docs/DESIGN.md 10.4).
    """
    cfg = resolve_memory_config(config)
    base = _clamp(float(base_salience), 0.0, 1.0)

    c_val: float = 1.0
    if comfort is not None:
        c_val = float(comfort)
    elif body is not None:
        if isinstance(body, Mapping) and "comfort" in body:
            c_val = float(body["comfort"])
        elif hasattr(body, "comfort"):
            c_val = float(body.comfort)
    discomfort = _clamp(1.0 - _clamp(c_val, 0.0, 1.0), 0.0, 1.0)

    s_val: float = 0.0
    if surprise is True:
        s_val = 1.0
    elif surprise is False or surprise is None:
        if drives is not None:
            if isinstance(drives, Mapping):
                s_val = float(drives.get("curiosity", drives.get("surprise", 0.0)))
            elif hasattr(drives, "curiosity"):
                s_val = float(drives.curiosity)
    else:
        s_val = max(0.0, float(surprise))
    s_val = _clamp(s_val, 0.0, 1.0)

    norm_kind = normalize_surprise_kind(surprise_kind)
    kind_weight = SURPRISE_KIND_WEIGHTS.get(norm_kind, 0.0) if norm_kind is not None else 0.0

    surprise_signal = _clamp(
        max(s_val, kind_weight) + 0.25 * min(s_val, kind_weight),
        0.0,
        1.0,
    )

    boost = (
        cfg.discomfort_salience_boost * discomfort
        + cfg.surprise_salience_boost * surprise_signal
    )
    return round(_clamp(base + boost, 0.0, 1.0), 6)


adjust_salience = compute_episode_salience
episode_salience = compute_episode_salience


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER NOT NULL,
    summary TEXT NOT NULL,
    tags TEXT NOT NULL,
    salience REAL NOT NULL,
    initial_salience REAL NOT NULL,
    last_recalled_tick INTEGER NOT NULL,
    last_decayed_tick INTEGER NOT NULL,
    surprise_kind TEXT,
    made_by_agent TEXT NOT NULL,
    recall_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_tick ON episodes(tick);
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience);
CREATE INDEX IF NOT EXISTS idx_episodes_last_recalled ON episodes(last_recalled_tick);
"""


class EpisodeStore:
    """SQLite persistence for episodic memories with salience decay and recall refresh.

    Uses Python's standard library ``sqlite3``. By default, the database path is
    read from ``config.db_path`` (``data/memory.db``).
    """

    def __init__(
        self,
        db_path: Path | str | MemoryConfig | Settings | None = None,
        *,
        path: Path | str | None = None,
        config: MemoryConfig | Settings | Mapping[str, Any] | None = None,
        settings: Settings | None = None,
        decay_rate: float | None = None,
        salience_floor: float | None = None,
        floor: float | None = None,
        min_salience: float | None = None,
        recall_boost: float | None = None,
    ) -> None:
        if isinstance(db_path, MemoryConfig) or (
            db_path is not None and hasattr(db_path, "memory")
        ):
            base_cfg = resolve_memory_config(db_path, settings=settings)
            explicit_path = path
        else:
            base_cfg = resolve_memory_config(config, settings=settings)
            explicit_path = db_path if db_path is not None else path

        overrides: dict[str, Any] = {}
        if explicit_path is not None:
            overrides["db_path"] = explicit_path
        if decay_rate is not None:
            overrides["decay_rate"] = float(decay_rate)
        resolved_floor = (
            salience_floor
            if salience_floor is not None
            else (floor if floor is not None else min_salience)
        )
        if resolved_floor is not None:
            overrides["salience_floor"] = float(resolved_floor)
        if recall_boost is not None:
            overrides["recall_boost"] = float(recall_boost)

        if overrides:
            self.config = MemoryConfig.model_validate({**base_cfg.model_dump(), **overrides})
        else:
            self.config = base_cfg

        raw_path_str = str(explicit_path) if explicit_path is not None else str(self.config.db_path)
        self._is_memory_db: bool = raw_path_str == ":memory:"
        self.db_path: Path = Path(raw_path_str)
        if not self._is_memory_db:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = sqlite3.connect(
            ":memory:" if self._is_memory_db else str(self.db_path)
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE_SQL)
        self._conn.commit()

        self._current_tick: int = self._query_max_tick()
        self.last_deleted_episodes: list[Episode] = []

    @property
    def path(self) -> Path:
        """Alias for ``db_path``."""
        return self.db_path

    @property
    def decay_rate(self) -> float:
        """Per-tick salience decay rate."""
        return self.config.decay_rate

    @decay_rate.setter
    def decay_rate(self, value: float) -> None:
        self.config = self.config.model_copy(update={"decay_rate": float(value)})

    @property
    def salience_floor(self) -> float:
        """Minimum salience floor below which episodes are deleted."""
        return self.config.salience_floor

    @salience_floor.setter
    def salience_floor(self, value: float) -> None:
        self.config = self.config.model_copy(update={"salience_floor": float(value)})

    @property
    def floor(self) -> float:
        """Alias for ``salience_floor``."""
        return self.config.salience_floor

    @property
    def recall_boost(self) -> float:
        """Salience boost applied when an episode is recalled."""
        return self.config.recall_boost

    @property
    def forgotten_episodes(self) -> list[Episode]:
        """Episodes deleted during the most recent ``decay()`` call."""
        return list(self.last_deleted_episodes)

    @property
    def connection(self) -> sqlite3.Connection:
        """Return an open SQLite connection to the store."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                ":memory:" if self._is_memory_db else str(self.db_path)
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_CREATE_TABLE_SQL)
            self._conn.commit()
        return self._conn

    def _query_max_tick(self) -> int:
        row = self.connection.execute(
            "SELECT MAX(MAX(tick), MAX(last_recalled_tick), MAX(last_decayed_tick)) AS m FROM episodes"
        ).fetchone()
        if row is not None and row["m"] is not None:
            return int(row["m"])
        return 0

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        tags = json.loads(row["tags"]) if row["tags"] else []
        made_by = json.loads(row["made_by_agent"]) if row["made_by_agent"] else []
        return Episode(
            id=int(row["id"]),
            tick=int(row["tick"]),
            summary=str(row["summary"]),
            tags=tags,
            salience=round(_clamp(float(row["salience"]), 0.0, 1.0), 6),
            last_recalled_tick=int(row["last_recalled_tick"]),
            surprise_kind=row["surprise_kind"],
            made_by_agent=made_by,
            recall_count=int(row["recall_count"]),
        )

    def add(
        self,
        episode: Episode | ReflectionEpisode | Mapping[str, Any] | str | None = None,
        *,
        summary: str | None = None,
        text: str | None = None,
        tick: int | None = None,
        timestamp: int | None = None,
        salience: float | None = None,
        tags: Sequence[str] | None = None,
        surprise_kind: SurpriseKind | str | None = None,
        made_by_agent: Sequence[str] | str | bool | None = None,
        last_recalled_tick: int | None = None,
        comfort: float | None = None,
        surprise: float | bool | None = None,
        body: Any = None,
        drives: Any = None,
    ) -> Episode:
        """Insert one episode into SQLite and return the stored ``Episode`` with its ``id``."""
        raw: dict[str, Any] = {}
        if isinstance(episode, Episode):
            raw = episode.model_dump(exclude={"id"})
        elif isinstance(episode, ReflectionEpisode):
            raw = episode.model_dump()
        elif isinstance(episode, Mapping):
            raw = dict(episode)
            raw.pop("id", None)
        elif isinstance(episode, str):
            raw = {"summary": episode}
        elif episode is not None:
            raise TypeError(f"Unsupported episode type: {type(episode).__name__}")

        if summary is not None:
            raw["summary"] = summary
        elif text is not None and "summary" not in raw:
            raw["summary"] = text

        resolved_tick = (
            tick
            if tick is not None
            else (timestamp if timestamp is not None else raw.get("tick", raw.get("timestamp", self._current_tick)))
        )
        raw["tick"] = int(resolved_tick or 0)

        if tags is not None:
            raw["tags"] = list(tags)
        if surprise_kind is not None:
            raw["surprise_kind"] = surprise_kind
        if made_by_agent is not None:
            raw["made_by_agent"] = _normalize_str_list(made_by_agent)
        if salience is not None:
            raw["salience"] = float(salience)
        if last_recalled_tick is not None:
            raw["last_recalled_tick"] = int(last_recalled_tick)
        elif "last_recalled_tick" not in raw or raw["last_recalled_tick"] is None:
            raw["last_recalled_tick"] = raw["tick"]

        validated = Episode.model_validate(raw)

        # If comfort or surprise context is provided, adjust salience accordingly
        final_salience = validated.salience
        if (
            comfort is not None
            or surprise is not None
            or body is not None
            or drives is not None
        ):
            final_salience = compute_episode_salience(
                validated.salience,
                comfort=comfort,
                surprise=surprise,
                surprise_kind=validated.surprise_kind,
                body=body,
                drives=drives,
                config=self.config,
            )

        conn = self.connection
        cur = conn.execute(
            """
            INSERT INTO episodes (
                tick, summary, tags, salience, initial_salience,
                last_recalled_tick, last_decayed_tick, surprise_kind,
                made_by_agent, recall_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                validated.tick,
                validated.summary,
                json.dumps(validated.tags),
                final_salience,
                final_salience,
                validated.last_recalled_tick,
                max(validated.tick, validated.last_recalled_tick),
                validated.surprise_kind,
                json.dumps(validated.made_by_agent),
                validated.recall_count,
            ),
        )
        conn.commit()
        row_id = int(cur.lastrowid)  # type: ignore[arg-type]
        self._current_tick = max(
            self._current_tick,
            validated.tick,
            validated.last_recalled_tick,
        )
        return validated.model_copy(update={"id": row_id, "salience": final_salience})

    save = add
    add_episode = add
    save_episode = add
    insert = add
    put = add

    def save_episodes(
        self,
        episodes: Sequence[Episode | ReflectionEpisode | Mapping[str, Any] | str],
        *,
        tick: int | None = None,
        comfort: float | None = None,
        surprise: float | bool | None = None,
        body: Any = None,
        drives: Any = None,
    ) -> list[Episode]:
        """Save multiple episodes to SQLite and return the stored ``Episode`` list."""
        saved: list[Episode] = []
        for ep in episodes:
            saved.append(
                self.add(
                    ep,
                    tick=tick,
                    comfort=comfort,
                    surprise=surprise,
                    body=body,
                    drives=drives,
                )
            )
        return saved

    add_episodes = save_episodes
    save_many = save_episodes
    add_many = save_episodes

    def get(self, episode_id: int | Episode) -> Episode | None:
        """Fetch an episode by its integer ``id`` (or ``None`` if forgotten/absent)."""
        resolved_id = episode_id.id if isinstance(episode_id, Episode) else int(episode_id)
        if resolved_id is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (resolved_id,),
        ).fetchone()
        return self._row_to_episode(row) if row is not None else None

    get_episode = get
    fetch = get

    def all(self) -> list[Episode]:
        """Return all stored episodes ordered by ``id`` ascending."""
        rows = self.connection.execute(
            "SELECT * FROM episodes ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    list_episodes = all
    get_all = all

    @property
    def episodes(self) -> list[Episode]:
        """All episodes currently persisted in SQLite."""
        return self.all()

    def count(self) -> int:
        """Return the number of episodes currently in the store."""
        row = self.connection.execute("SELECT COUNT(*) AS c FROM episodes").fetchone()
        return int(row["c"]) if row is not None else 0

    def query(
        self,
        *,
        tag: str | None = None,
        tags: Sequence[str] | None = None,
        surprise_kind: SurpriseKind | str | None = None,
        made_by_agent: str | bool | None = None,
        min_salience: float | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        """Filter episodes by tag(s), ``surprise_kind``, ``made_by_agent``, or ``min_salience``."""
        wanted_tags: set[str] = set()
        if tag is not None and tag.strip():
            wanted_tags.add(tag.strip())
        if tags is not None:
            for t in tags:
                if isinstance(t, str) and t.strip():
                    wanted_tags.add(t.strip())

        norm_kind = normalize_surprise_kind(surprise_kind) if surprise_kind is not None else None

        results: list[Episode] = []
        for ep in self.all():
            if min_salience is not None and ep.salience < float(min_salience):
                continue
            if norm_kind is not None and ep.surprise_kind != norm_kind:
                continue
            if made_by_agent is True and not ep.made_by_agent:
                continue
            if made_by_agent is False and ep.made_by_agent:
                continue
            if isinstance(made_by_agent, str) and made_by_agent not in ep.made_by_agent:
                continue
            if wanted_tags and not (wanted_tags & set(ep.tags) or wanted_tags & set(ep.made_by_agent)):
                continue
            results.append(ep)
            if limit is not None and len(results) >= limit:
                break
        return results

    search = query
    find = query
    filter = query

    def _recall_row(
        self,
        row: sqlite3.Row,
        *,
        resolved_tick: int,
        boost: float,
    ) -> Episode:
        ep_id = int(row["id"])
        curr_salience = float(row["salience"])
        init_salience = float(row["initial_salience"])
        new_salience = round(
            _clamp(max(curr_salience, init_salience) + boost, 0.0, 1.0),
            6,
        )
        new_recalled_tick = max(int(row["last_recalled_tick"]), resolved_tick)
        new_decayed_tick = max(int(row["last_decayed_tick"]), new_recalled_tick)
        new_count = int(row["recall_count"]) + 1

        self.connection.execute(
            """
            UPDATE episodes
            SET salience = ?,
                last_recalled_tick = ?,
                last_decayed_tick = ?,
                recall_count = ?
            WHERE id = ?
            """,
            (new_salience, new_recalled_tick, new_decayed_tick, new_count, ep_id),
        )
        self.connection.commit()
        updated = self.connection.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (ep_id,),
        ).fetchone()
        assert updated is not None
        return self._row_to_episode(updated)

    def recall(
        self,
        target: int | Episode | str | Sequence[str] | None = None,
        tick: int | None = None,
        *,
        episode_id: int | None = None,
        tag: str | None = None,
        tags: Sequence[str] | None = None,
        current_tick: int | None = None,
        boost: float | None = None,
    ) -> Episode | list[Episode] | None:
        """Recall an episode (by ``id``/``Episode``) or matching episodes (by ``tag``/``tags``).

        Recalling refreshes the episode's ``last_recalled_tick`` to the current
        world tick and boosts/restores its ``salience`` so subsequent decay does
        not forget it.
        """
        explicit_tick = tick if tick is not None else current_tick
        if explicit_tick is not None:
            resolved_tick = max(0, int(explicit_tick))
            self._current_tick = max(self._current_tick, resolved_tick)
        else:
            resolved_tick = self._current_tick

        eff_boost = float(boost) if boost is not None else self.config.recall_boost

        # 1. Recall a specific episode by ID or Episode instance
        single_id: int | None = None
        if episode_id is not None:
            single_id = int(episode_id)
        elif isinstance(target, Episode):
            if target.id is None:
                raise ValueError("Cannot recall an Episode without an id")
            single_id = int(target.id)
        elif isinstance(target, int) and not isinstance(target, bool):
            single_id = int(target)

        if single_id is not None:
            row = self.connection.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (single_id,),
            ).fetchone()
            if row is None:
                return None
            ep_tick = max(resolved_tick, int(row["tick"]), int(row["last_recalled_tick"]))
            self._current_tick = max(self._current_tick, ep_tick)
            return self._recall_row(row, resolved_tick=ep_tick, boost=eff_boost)

        # 2. Recall by tag(s)
        wanted_tags: list[str] = []
        if isinstance(target, str) and target.strip():
            wanted_tags.append(target.strip())
        elif isinstance(target, Sequence):
            wanted_tags.extend(str(t).strip() for t in target if str(t).strip())
        if tag is not None and tag.strip():
            wanted_tags.append(tag.strip())
        if tags is not None:
            wanted_tags.extend(str(t).strip() for t in tags if str(t).strip())

        matching = self.query(tags=wanted_tags if wanted_tags else None)
        refreshed: list[Episode] = []
        for ep in matching:
            if ep.id is None:
                continue
            row = self.connection.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (ep.id,),
            ).fetchone()
            if row is not None:
                ep_tick = max(resolved_tick, int(row["tick"]), int(row["last_recalled_tick"]))
                self._current_tick = max(self._current_tick, ep_tick)
                refreshed.append(self._recall_row(row, resolved_tick=ep_tick, boost=eff_boost))
        return refreshed

    recall_episode = recall

    def recall_by_tag(
        self,
        tag: str | Sequence[str],
        tick: int | None = None,
        *,
        boost: float | None = None,
    ) -> list[Episode]:
        """Recall all episodes matching ``tag`` (or ``tags``) and return the refreshed episodes."""
        res = self.recall(
            tags=[tag] if isinstance(tag, str) else list(tag),
            tick=tick,
            boost=boost,
        )
        return res if isinstance(res, list) else ([res] if res is not None else [])

    def decay(
        self,
        current_tick_or_ticks: int | None = None,
        *,
        current_tick: int | None = None,
        tick: int | None = None,
        ticks: int | None = None,
        dt: int | None = None,
        elapsed_ticks: int | None = None,
        decay_rate: float | None = None,
        salience_floor: float | None = None,
        floor: float | None = None,
        min_salience: float | None = None,
    ) -> list[Episode]:
        """Decay episode salience over world time and delete episodes below the floor.

        Supports calling as:
        - ``store.decay(current_tick=100)`` or ``store.decay(tick=100)`` (world tick timestamp)
        - ``store.decay(ticks=50)`` or ``store.decay(dt=50)`` (elapsed ticks delta)
        - ``store.decay(50)`` (treats ``50`` as target tick if ``50 > current_tick``, else as delta ticks)
        - ``store.decay()`` (advances by 1 tick)

        Episodes whose decayed salience drops below ``salience_floor`` (or ``<= 0.0``)
        are permanently deleted from SQLite (real forgetting). Returns the surviving
        episodes.
        """
        eff_rate = float(decay_rate) if decay_rate is not None else self.config.decay_rate
        eff_floor = (
            float(salience_floor)
            if salience_floor is not None
            else float(floor)
            if floor is not None
            else float(min_salience)
            if min_salience is not None
            else self.config.salience_floor
        )

        explicit_delta = (
            ticks
            if ticks is not None
            else (dt if dt is not None else elapsed_ticks)
        )
        explicit_abs = current_tick if current_tick is not None else tick

        if explicit_delta is not None:
            step_dt = max(0, int(explicit_delta))
            target_tick = self._current_tick + step_dt
        elif explicit_abs is not None:
            target_tick = max(0, int(explicit_abs))
        elif current_tick_or_ticks is not None:
            val = max(0, int(current_tick_or_ticks))
            if val > self._current_tick:
                target_tick = val
            else:
                target_tick = self._current_tick + val
        else:
            target_tick = self._current_tick + 1

        self._current_tick = max(self._current_tick, target_tick)

        conn = self.connection
        rows = conn.execute("SELECT * FROM episodes ORDER BY id ASC").fetchall()
        deleted: list[Episode] = []

        for row in rows:
            ep_id = int(row["id"])
            ref_tick = max(
                int(row["tick"]),
                int(row["last_recalled_tick"]),
                int(row["last_decayed_tick"]),
            )
            elapsed = max(0, target_tick - ref_tick)
            curr_salience = float(row["salience"])
            new_salience = round(max(0.0, curr_salience - elapsed * eff_rate), 6)

            if new_salience < eff_floor or new_salience <= 0.0:
                deleted.append(
                    self._row_to_episode(row).model_copy(update={"salience": new_salience})
                )
                conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
            elif elapsed > 0:
                conn.execute(
                    """
                    UPDATE episodes
                    SET salience = ?, last_decayed_tick = ?
                    WHERE id = ?
                    """,
                    (new_salience, target_tick, ep_id),
                )

        conn.commit()
        self.last_deleted_episodes = deleted
        return self.all()

    step_decay = decay
    apply_decay = decay
    tick_decay = decay

    def delete(self, episode_id: int | Episode) -> bool:
        """Delete a single episode by ``id`` and return True if a row was removed."""
        resolved_id = episode_id.id if isinstance(episode_id, Episode) else int(episode_id)
        if resolved_id is None:
            return False
        cur = self.connection.execute("DELETE FROM episodes WHERE id = ?", (resolved_id,))
        self.connection.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        """Delete all episodes from the store."""
        self.connection.execute("DELETE FROM episodes")
        self.connection.commit()
        self._current_tick = 0
        self.last_deleted_episodes = []

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __enter__(self) -> EpisodeStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.all())

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Episode):
            return item.id is not None and self.get(item.id) is not None
        if isinstance(item, int) and not isinstance(item, bool):
            return self.get(item) is not None
        return False


MemoryStore = EpisodeStore
EpisodicStore = EpisodeStore
SQLiteEpisodeStore = EpisodeStore


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_DB_PATH",
    "DEFAULT_DECAY_RATE",
    "DEFAULT_DISCOMFORT_SALIENCE_BOOST",
    "DEFAULT_MEMORY_CONFIG",
    "DEFAULT_RECALL_BOOST",
    "DEFAULT_REST_IDLE_TICKS",
    "DEFAULT_SALIENCE_FLOOR",
    "DEFAULT_SURPRISE_SALIENCE_BOOST",
    "SURPRISE_KINDS",
    "SURPRISE_KIND_WEIGHTS",
    "Episode",
    "EpisodeStore",
    "EpisodicStore",
    "MemoryConfig",
    "MemorySettings",
    "MemoryStore",
    "ReflectResponse",
    "ReflectSchema",
    "ReflectionEpisode",
    "ReflectionOutput",
    "ReflectionResult",
    "SQLiteEpisodeStore",
    "SurpriseKind",
    "adjust_salience",
    "compute_episode_salience",
    "episode_salience",
    "normalize_surprise_kind",
    "resolve_memory_config",
]
