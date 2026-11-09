"""FastAPI headless simulation service, creator API, WebSocket stream, and auth."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import mimetypes
import os
import random
import re
import secrets
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

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

STATIC_DIR: Path = Path(__file__).resolve().parent.parent / "web" / "static"

_STATIC_MEDIA_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

for _ext, _mime in _STATIC_MEDIA_TYPES.items():
    mimetypes.add_type(_mime.split(";")[0], _ext)


def _serve_static_file(rel_path: str) -> FileResponse:
    """Resolve and serve a static asset from ``src/arden/web/static/``."""
    base = STATIC_DIR.resolve()
    cleaned = rel_path.lstrip("/\\") or "index.html"
    try:
        candidate = (base / cleaned).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Static asset not found") from exc

    if candidate != base and base not in candidate.parents:
        raise HTTPException(status_code=404, detail="Static asset not found")
    if candidate.is_dir():
        candidate = (candidate / "index.html").resolve()
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found")

    media_type = _STATIC_MEDIA_TYPES.get(candidate.suffix.lower())
    if media_type is None:
        guessed, _ = mimetypes.guess_type(str(candidate))
        media_type = guessed or "application/octet-stream"
    return FileResponse(candidate, media_type=media_type)


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


FIRST_RUN_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "key": "ball",
        "label": "Ball (red sphere)",
        "category": "curiosity",
        "props": {
            "creator_label": "ball",
            "shape": "sphere",
            "height": 0.25,
            "volume": 0.008,
            "weight": 0.25,
            "rigidity": 0.5,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": "red",
            "fixed": False,
            "behavior": [
                {"verb": "push", "scale": 4},
                {"verb": "strike", "response": "bounce"},
            ],
        },
    },
    {
        "key": "stick",
        "label": "Stick (long rigid)",
        "category": "material",
        "props": {
            "creator_label": "stick",
            "shape": "long",
            "height": 0.85,
            "volume": 0.006,
            "weight": 0.3,
            "rigidity": 0.88,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": "brown",
            "fixed": False,
        },
    },
    {
        "key": "flat_stone",
        "label": "Flat Stone (heavy stackable)",
        "category": "material",
        "props": {
            "creator_label": "flat stone",
            "shape": "flat",
            "height": 0.28,
            "volume": 0.045,
            "weight": 2.6,
            "rigidity": 0.95,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": "gray",
            "fixed": False,
        },
    },
    {
        "key": "vine",
        "label": "Vine (binding)",
        "category": "material",
        "props": {
            "creator_label": "vine",
            "shape": "long",
            "height": 1.2,
            "volume": 0.003,
            "weight": 0.15,
            "rigidity": 0.2,
            "binding": True,
            "hollow": False,
            "fragile": False,
            "color": "green",
            "fixed": False,
        },
    },
    {
        "key": "leaf",
        "label": "Leaf (very light)",
        "category": "material",
        "props": {
            "creator_label": "leaf",
            "shape": "flat",
            "height": 0.02,
            "volume": 0.0004,
            "weight": 0.05,
            "rigidity": 0.15,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": "green",
            "fixed": False,
        },
    },
    {
        "key": "berry",
        "label": "Berry (small fragile edible)",
        "category": "curiosity",
        "props": {
            "creator_label": "berry",
            "shape": "sphere",
            "height": 0.08,
            "volume": 0.0003,
            "weight": 0.02,
            "rigidity": 0.2,
            "binding": False,
            "hollow": False,
            "fragile": True,
            "color": "red",
            "fixed": False,
            "behavior": [
                {"verb": "edible", "energy": 0.15},
                {"verb": "strike", "response": "break", "above_force": 0.2},
            ],
        },
    },
    {
        "key": "berry_bush",
        "label": "Berry Bush (fixed bush)",
        "category": "curiosity",
        "props": {
            "creator_label": "berry bush",
            "shape": "sphere",
            "height": 0.9,
            "volume": 0.35,
            "weight": 8.0,
            "rigidity": 0.4,
            "binding": False,
            "hollow": False,
            "fragile": True,
            "color": "green",
            "fixed": True,
            "state": {"berries": 3},
            "behavior": [
                {"verb": "edible", "restore_energy": 0.15},
                {"verb": "strike", "response": "break", "above_force": 0.85},
            ],
        },
    },
    {
        "key": "hollow_log",
        "label": "Hollow Log (container/shelter)",
        "category": "curiosity",
        "props": {
            "creator_label": "hollow log",
            "shape": "cylinder",
            "height": 0.55,
            "volume": 0.18,
            "weight": 8.0,
            "rigidity": 0.9,
            "binding": False,
            "hollow": True,
            "fragile": False,
            "color": "brown",
            "fixed": False,
        },
    },
    {
        "key": "large_tree",
        "label": "Large Tree (tall fixed shade)",
        "category": "terrain",
        "props": {
            "creator_label": "large tree",
            "shape": "cylinder",
            "height": 3.5,
            "volume": 0.8,
            "weight": 200.0,
            "rigidity": 0.98,
            "binding": False,
            "hollow": False,
            "fragile": False,
            "color": "brown",
            "fixed": True,
        },
    },
    {
        "key": "mailbox",
        "label": "Mailbox (lidded container)",
        "category": "curiosity",
        "props": {
            "creator_label": "mailbox",
            "shape": "cube",
            "height": 1.1,
            "volume": 0.12,
            "weight": 15.0,
            "rigidity": 0.95,
            "binding": False,
            "hollow": True,
            "fragile": False,
            "color": "blue",
            "fixed": True,
            "state": {"lid": "closed"},
            "behavior": [
                {"verb": "grab", "response": "toggle_lid"},
                {"verb": "separate", "response": "toggle_lid"},
                {"verb": "speak", "response": "capture_mail"},
                {"verb": "insert", "response": "send_object"},
            ],
        },
    },
)

FIRST_RUN_PRESETS = tuple(
    {**item["props"], **item} for item in FIRST_RUN_PRESETS
)

_PRESET_LOOKUP: dict[str, dict[str, Any]] = {}
for _p in FIRST_RUN_PRESETS:
    _k = str(_p["key"]).lower()
    _PRESET_LOOKUP[_k] = _p["props"]
    _PRESET_LOOKUP[_k.replace("_", " ")] = _p["props"]
    _PRESET_LOOKUP[_k.replace("_", "-")] = _p["props"]
_PRESET_LOOKUP["stone"] = _PRESET_LOOKUP["flat_stone"]
_PRESET_LOOKUP["tree"] = _PRESET_LOOKUP["large_tree"]
_PRESET_LOOKUP["log"] = _PRESET_LOOKUP["hollow_log"]
_PRESET_LOOKUP["bush"] = _PRESET_LOOKUP["berry_bush"]


def get_spawn_presets() -> list[dict[str, Any]]:
    """Return the first-run object property presets list for creator spawning."""
    return json.loads(json.dumps(list(FIRST_RUN_PRESETS)))


def _ensure_spawn_defaults(props: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure minimal object spawn payloads (or preset references) have valid physical defaults."""
    raw = dict(props)
    preset_name = raw.pop("preset", None)
    data: dict[str, Any] = {}
    if isinstance(preset_name, str) and preset_name.strip():
        key = preset_name.strip().lower()
        if key not in _PRESET_LOOKUP:
            raise ValueError(f"Unknown spawn preset: {preset_name!r}")
        data.update(json.loads(json.dumps(_PRESET_LOOKUP[key])))
    data.update(raw)
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


_OBJ_ID_PATTERN = re.compile(r"\b(object_[0-9]+)\b", re.IGNORECASE)
_WORD_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,24}")

_EXPLICIT_NAME_PATTERNS: tuple[tuple[re.Pattern[str], int, int], ...] = (
    # "call object_7 the red ball" / "called object_7 'bouncer'"
    (
        re.compile(
            r"\b(?:call(?:ing|ed|s)?|name(?:d|s|ing)?|dub(?:bed|bing|s)?)\s+"
            r"(object_[0-9]+)\s+(?:the\s+|a\s+|an\s+|as\s+)?[\"']?"
            r"([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?)[\"']?",
            re.IGNORECASE,
        ),
        1,
        2,
    ),
    # "object_7 is called the red ball" / "object_7 is a red ball" / "object_7: red ball"
    (
        re.compile(
            r"\b(object_[0-9]+)\s+(?:is\s+called|is\s+named|called|named|is\s+a|is\s+an|is\s+the|=|:)\s+"
            r"(?:the\s+|a\s+|an\s+)?[\"']?"
            r"([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?)[\"']?",
            re.IGNORECASE,
        ),
        1,
        2,
    ),
    # "object_7 (red ball)"
    (
        re.compile(
            r"\b(object_[0-9]+)\s*\(\s*[\"']?"
            r"([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?)[\"']?\s*\)",
            re.IGNORECASE,
        ),
        1,
        2,
    ),
    # "the red ball (object_7)"
    (
        re.compile(
            r"\b(?:the\s+|a\s+|an\s+)?[\"']?"
            r"([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?)[\"']?\s*\(\s*(object_[0-9]+)\s*\)",
            re.IGNORECASE,
        ),
        2,
        1,
    ),
)

_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "onto", "upon", "over", "under",
        "near", "next", "beside", "toward", "towards", "away", "around", "through",
        "across", "along", "about", "between", "within", "without", "inside", "outside",
        "that", "this", "these", "those", "which", "what", "where", "when", "while",
        "after", "before", "because", "since", "until", "though", "although", "whether",
        "there", "here", "then", "than", "very", "much", "more", "most", "less", "least",
        "some", "many", "each", "every", "other", "another", "such", "same", "only",
        "just", "also", "even", "still", "already", "again", "back", "down",
        "will", "would", "could", "should", "might", "must", "shall", "may", "can",
        "cannot", "have", "has", "had", "having", "been", "being", "were", "was", "are",
        "is", "not", "nor", "but", "yet", "both", "either", "neither", "its", "their",
        "his", "her", "our", "your", "my", "me", "we", "us", "you", "they", "them",
        "it", "he", "she", "who", "whom", "whose",
        "move", "moved", "moving", "moves", "look", "looked", "looking", "looks",
        "push", "pushed", "pushing", "pushes", "strike", "struck", "striking", "strikes",
        "grab", "grabbed", "grabbing", "grabs", "drop", "dropped", "dropping", "drops",
        "place", "placed", "placing", "places", "stack", "stacked", "stacking", "stacks",
        "bind", "bound", "binding", "binds", "insert", "inserted", "inserting", "inserts",
        "separate", "separated", "separating", "separates", "speak", "spoke", "speaking",
        "speaks", "rest", "resting", "rested", "walk", "walked", "walking", "reach",
        "reached", "reaching", "reachable", "test", "testing", "tested", "try", "trying",
        "tried", "explore", "exploring", "explored", "inspect", "inspecting", "inspected",
        "examine", "examining", "examined", "approach", "approaching", "approached",
        "check", "checking", "checked", "observe", "observing", "observed", "see", "saw",
        "seen", "find", "found", "make", "made", "making", "take", "took", "taken",
        "give", "gave", "keep", "kept", "need", "needs", "needed", "want", "wants",
        "wanted", "seem", "seems", "seemed", "appear", "appears", "appeared",
        "object", "objects", "thing", "things", "item", "items", "target", "targets",
        "agent", "self", "world", "ground", "raised", "level", "position", "distance",
        "meter", "meters", "tick", "ticks", "force", "speed", "energy", "comfort",
        "north", "south", "east", "west", "left", "right", "top", "bottom", "side",
        "far", "close", "closer", "further", "farther", "high", "low", "new", "old",
        "first", "last", "next", "previous", "current", "active", "idle", "none", "null",
        "true", "false", "yes",
    }
)


def _clean_extracted_phrase(phrase: str) -> str | None:
    """Clean and filter a candidate extracted name or word phrase."""
    tokens = [
        t.lower()
        for t in _WORD_TOKEN_PATTERN.findall(phrase)
        if not _OBJ_ID_PATTERN.fullmatch(t)
    ]
    filtered = [t for t in tokens if t not in _NAME_STOPWORDS and len(t) >= 2]
    if not filtered:
        return None
    return " ".join(filtered[:3])


def extract_object_names(
    *,
    speeches: Sequence[Any] = (),
    thoughts: Sequence[Any] = (),
    beliefs: Sequence[Any] = (),
    episodes: Sequence[Any] = (),
    world: World | None = None,
) -> dict[str, Any]:
    """Extract words and names the agent uses for objects from ``speak``, thoughts, and beliefs."""
    # Map object_id -> list of {"word": str, "source": str, "tick": int, "text": str, "explicit": bool}
    records_by_obj: dict[str, list[dict[str, Any]]] = {}

    def _add_record(
        obj_id: str,
        word: str | None,
        source: str,
        tick: int,
        raw_text: str,
        *,
        explicit: bool = False,
    ) -> None:
        if not word:
            return
        cleaned = _clean_extracted_phrase(word)
        if not cleaned:
            return
        norm_id = obj_id.strip().lower()
        if not _OBJ_ID_PATTERN.fullmatch(norm_id):
            return
        bucket = records_by_obj.setdefault(norm_id, [])
        bucket.append(
            {
                "word": cleaned,
                "source": source,
                "tick": int(tick),
                "text": str(raw_text).strip(),
                "explicit": explicit,
            }
        )
        # If it is a multi-word phrase (e.g. "red ball"), also record individual salient words
        parts = cleaned.split()
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 3 and part not in _NAME_STOPWORDS:
                    bucket.append(
                        {
                            "word": part,
                            "source": source,
                            "tick": int(tick),
                            "text": str(raw_text).strip(),
                            "explicit": False,
                        }
                    )

    def _process_text(
        text: str,
        source: str,
        tick: int = 0,
        context_obj_ids: Sequence[str] = (),
    ) -> None:
        if not text or not isinstance(text, str):
            return
        stripped = text.strip()
        if not stripped:
            return

        # 1. Explicit naming patterns
        for pat, id_group, phrase_group in _EXPLICIT_NAME_PATTERNS:
            for m in pat.finditer(stripped):
                oid = m.group(id_group).lower()
                phrase = m.group(phrase_group)
                _add_record(oid, phrase, source, tick, stripped, explicit=True)

        # 2. Check all object_<n> mentions in the text
        mentions = list(_OBJ_ID_PATTERN.finditer(stripped))
        if mentions:
            # Partition text around each object_<n> mention so words associate with the nearest object_<n>
            for idx, match in enumerate(mentions):
                oid = match.group(1).lower()
                start_bound = mentions[idx - 1].end() if idx > 0 else 0
                end_bound = mentions[idx + 1].start() if idx + 1 < len(mentions) else len(stripped)

                before_segment = stripped[start_bound:match.start()]
                after_segment = stripped[match.end():end_bound]

                # Preceding modifier phrase right before object_<n>, e.g. "the red ball object_7"
                before_words = [
                    w.lower()
                    for w in _WORD_TOKEN_PATTERN.findall(before_segment)
                    if not _OBJ_ID_PATTERN.fullmatch(w) and w.lower() not in _NAME_STOPWORDS
                ]
                if before_words:
                    tail = before_words[-2:]
                    if len(tail) == 2:
                        _add_record(oid, " ".join(tail), source, tick, stripped, explicit=True)
                    for w in tail:
                        _add_record(oid, w, source, tick, stripped, explicit=False)

                # Following descriptive words in the same clause after object_<n>
                clause_after = re.split(r"[.;!?]|,\s*(?:and|but|so|then)\b", after_segment)[0]
                after_words = [
                    w.lower()
                    for w in _WORD_TOKEN_PATTERN.findall(clause_after)
                    if not _OBJ_ID_PATTERN.fullmatch(w) and w.lower() not in _NAME_STOPWORDS
                ]
                for w in after_words[:4]:
                    _add_record(oid, w, source, tick, stripped, explicit=False)
        elif context_obj_ids:
            # No literal object_<n> in the text (common in `speak`), use context object IDs
            words = [
                w.lower()
                for w in _WORD_TOKEN_PATTERN.findall(stripped)
                if not _OBJ_ID_PATTERN.fullmatch(w) and w.lower() not in _NAME_STOPWORDS
            ]
            for oid in context_obj_ids:
                if 1 <= len(words) <= 3:
                    _add_record(oid, " ".join(words), source, tick, stripped, explicit=True)
                for w in words[:4]:
                    _add_record(oid, w, source, tick, stripped, explicit=False)

    # 1. Process speeches (`speak`)
    for item in speeches:
        if isinstance(item, str):
            _process_text(item, "speak", 0)
        elif isinstance(item, Mapping):
            txt = item.get("text") or item.get("speech") or item.get("message")
            t = int(item.get("tick", 0) or 0)
            ctx: list[str] = []
            for key in ("object_id", "target", "holding"):
                val = item.get(key)
                if isinstance(val, str) and _OBJ_ID_PATTERN.fullmatch(val):
                    ctx.append(val.lower())
            for key in ("context_objects", "targets", "objects"):
                vals = item.get(key)
                if isinstance(vals, Sequence) and not isinstance(vals, str | bytes):
                    for v in vals:
                        if isinstance(v, str) and _OBJ_ID_PATTERN.fullmatch(v):
                            ctx.append(v.lower())
            if isinstance(txt, str):
                _process_text(txt, "speak", t, ctx)
        else:
            txt = getattr(item, "text", None)
            t = int(getattr(item, "tick", 0) or 0)
            ctx_ids: list[str] = []
            for attr in ("object_id", "target", "mailbox_id"):
                val = getattr(item, attr, None)
                if isinstance(val, str) and _OBJ_ID_PATTERN.fullmatch(val):
                    ctx_ids.append(val.lower())
            if isinstance(txt, str):
                _process_text(txt, "speak", t, ctx_ids)

    # 2. Process thoughts (and intent goals/expectations)
    for item in thoughts:
        if isinstance(item, str):
            _process_text(item, "thought", 0)
        elif isinstance(item, Mapping):
            t = int(item.get("tick", 0) or 0)
            step_targets: list[str] = []
            for s in item.get("steps") or ():
                st = s.get("target") if isinstance(s, Mapping) else getattr(s, "target", None)
                if isinstance(st, str) and _OBJ_ID_PATTERN.fullmatch(st):
                    step_targets.append(st.lower())
            for field_name in ("thought", "text", "goal", "expect"):
                val = item.get(field_name)
                if isinstance(val, str) and val.strip():
                    _process_text(val, "thought", t, step_targets)
        else:
            t = int(getattr(item, "tick", 0) or 0)
            step_targets = []
            for s in getattr(item, "steps", ()) or ():
                st = s.get("target") if isinstance(s, Mapping) else getattr(s, "target", None)
                if isinstance(st, str) and _OBJ_ID_PATTERN.fullmatch(st):
                    step_targets.append(st.lower())
            for attr_name in ("thought", "goal", "expect"):
                val = getattr(item, attr_name, None)
                if isinstance(val, str) and val.strip():
                    _process_text(val, "thought", t, step_targets)

    # 3. Process beliefs (and tags)
    for item in beliefs:
        if isinstance(item, str):
            _process_text(item, "belief", 0)
        elif isinstance(item, Mapping):
            stmt = item.get("statement") or item.get("text") or ""
            tags = list(item.get("tags") or ())
            t = int(item.get("updated_tick", item.get("tick", 0)) or 0)
            tag_objs = [
                str(tg).lower()
                for tg in tags
                if isinstance(tg, str) and _OBJ_ID_PATTERN.fullmatch(tg.strip())
            ]
            stmt_objs = [m.lower() for m in _OBJ_ID_PATTERN.findall(str(stmt))]
            all_objs = list(dict.fromkeys([*tag_objs, *stmt_objs]))
            if isinstance(stmt, str) and stmt.strip():
                _process_text(stmt, "belief", t, all_objs)
            for tg in tags:
                if isinstance(tg, str) and not _OBJ_ID_PATTERN.fullmatch(tg.strip()):
                    for oid in all_objs:
                        _add_record(oid, tg, "belief", t, str(stmt or tg), explicit=False)
        else:
            stmt = getattr(item, "statement", "") or ""
            tags = list(getattr(item, "tags", ()) or ())
            t = int(getattr(item, "updated_tick", getattr(item, "created_tick", 0)) or 0)
            tag_objs = [
                str(tg).lower()
                for tg in tags
                if isinstance(tg, str) and _OBJ_ID_PATTERN.fullmatch(tg.strip())
            ]
            stmt_objs = [m.lower() for m in _OBJ_ID_PATTERN.findall(str(stmt))]
            all_objs = list(dict.fromkeys([*tag_objs, *stmt_objs]))
            if isinstance(stmt, str) and stmt.strip():
                _process_text(stmt, "belief", t, all_objs)
            for tg in tags:
                if isinstance(tg, str) and not _OBJ_ID_PATTERN.fullmatch(tg.strip()):
                    for oid in all_objs:
                        _add_record(oid, tg, "belief", t, str(stmt or tg), explicit=False)

    # 4. Process episodes if provided
    for ep in episodes:
        summary = ep.get("summary") if isinstance(ep, Mapping) else getattr(ep, "summary", None)
        t = int((ep.get("tick") if isinstance(ep, Mapping) else getattr(ep, "tick", 0)) or 0)
        if isinstance(summary, str) and summary.strip():
            _process_text(summary, "thought", t)

    creator_labels = world.creator_labels if world is not None else {}

    def _obj_sort_key(oid: str) -> tuple[int, str]:
        try:
            return (int(oid.replace("object_", "")), oid)
        except ValueError:
            return (999999, oid)

    names_by_object: dict[str, dict[str, Any]] = {}
    entries_list: list[dict[str, Any]] = []
    word_to_objects: dict[str, list[str]] = {}

    for oid in sorted(records_by_obj.keys(), key=_obj_sort_key):
        raw_recs = records_by_obj[oid]
        # Score words by explicitness, frequency, and recency
        word_stats: dict[str, dict[str, Any]] = {}
        by_source: dict[str, list[str]] = {"speak": [], "thought": [], "belief": []}
        dedup_sources: list[dict[str, Any]] = []
        seen_src_keys: set[tuple[str, str, int]] = set()

        for r in raw_recs:
            w = r["word"]
            src = r["source"]
            tk = r["tick"]
            st = word_stats.setdefault(
                w,
                {"word": w, "count": 0, "explicit": False, "last_tick": tk, "sources": set()},
            )
            st["count"] += 1
            if r.get("explicit"):
                st["explicit"] = True
            if tk >= st["last_tick"]:
                st["last_tick"] = tk
            st["sources"].add(src)

            if src in by_source and w not in by_source[src]:
                by_source[src].append(w)

            src_key = (w, src, tk)
            if src_key not in seen_src_keys:
                seen_src_keys.add(src_key)
                dedup_sources.append(
                    {
                        "word": w,
                        "source": src,
                        "tick": tk,
                        "text": r["text"],
                    }
                )

        ranked_words = sorted(
            word_stats.values(),
            key=lambda item: (
                1 if item["explicit"] else 0,
                len(item["sources"]),
                item["count"],
                item["last_tick"],
                len(item["word"].split()),
            ),
            reverse=True,
        )
        words_list = [item["word"] for item in ranked_words]
        for w in words_list:
            objs_for_w = word_to_objects.setdefault(w, [])
            if oid not in objs_for_w:
                objs_for_w.append(oid)

        c_label = creator_labels.get(oid)
        if c_label is None and world is not None and oid in world.objects:
            c_label = world.objects[oid].creator_label

        source_kinds = [src for src in ("speak", "thought", "belief") if by_source.get(src)]
        entry_dict = {
            "object_id": oid,
            "id": oid,
            "primary_name": words_list[0] if words_list else None,
            "name": words_list[0] if words_list else None,
            "names": words_list,
            "words": words_list,
            "sources": source_kinds,
            "source_records": dedup_sources,
            "by_source": by_source,
            "creator_label": c_label,
            "mentions": len(dedup_sources),
            "count": len(dedup_sources),
        }
        names_by_object[oid] = entry_dict
        entries_list.append(entry_dict)

    return {
        "tick": world.tick if world is not None else 0,
        "names": names_by_object,
        "by_object": names_by_object,
        "objects": names_by_object,
        "mappings": entries_list,
        "items": entries_list,
        "entries": entries_list,
        "word_to_objects": word_to_objects,
        "count": len(entries_list),
    }


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
        self.timeline_entries: list[dict[str, Any]] = []
        self._timeline_seq: int = 0
        self.spoke_history: list[dict[str, Any]] = []
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

    def _append_timeline_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Append a structured observer timeline entry."""
        with self._lock:
            self._timeline_seq += 1
            raw_entry = dict(entry)
            details = dict(raw_entry.get("details") or {})
            for k, v in raw_entry.items():
                if k not in ("id", "details") and k not in details:
                    details[k] = v
            record = {
                "id": self._timeline_seq,
                **raw_entry,
                "details": details,
            }
            self.timeline_entries.append(record)
            if len(self.timeline_entries) > 1000:
                self.timeline_entries = self.timeline_entries[-1000:]
            return record

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
            reason_str = ", ".join(reasons) if reasons else None
            self._append_timeline_entry(
                {
                    "kind": "wake",
                    "type": "wake",
                    "event": "wake",
                    "tick": event.tick,
                    "source": event.source,
                    "reasons": reasons,
                    "reason": reason_str,
                    "summary": f"Wake ({reason_str or 'trigger'})",
                }
            )
            self._broadcast(
                {
                    "type": "wake",
                    "event": "wake",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "reasons": reasons,
                    "reason": reason_str,
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

            self._append_timeline_entry(
                {
                    "kind": "intent",
                    "type": "intent",
                    "event": "intent_set",
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
                    "summary": f"Intent: {goal_str or 'idle'}",
                }
            )
            self._append_timeline_entry(
                {
                    "kind": "thought",
                    "type": "thought",
                    "event": "thought",
                    "tick": event.tick,
                    "source": event.source,
                    "thought": thought_str,
                    "text": thought_str,
                    "goal": goal_str,
                    "summary": str(thought_str or ""),
                }
            )
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
            status_str = getattr(event, "status", "succeeded")
            ok_val = bool(getattr(event, "ok", True))
            goal_str = getattr(event, "goal", None)
            reason_str = getattr(event, "reason", None)
            self._append_timeline_entry(
                {
                    "kind": "intent",
                    "type": "intent_ended",
                    "event": "intent_ended",
                    "tick": event.tick,
                    "source": event.source,
                    "status": status_str,
                    "ok": ok_val,
                    "goal": goal_str,
                    "reason": reason_str,
                    "summary": (
                        f"Intent {status_str}: {goal_str or ''}"
                        f"{f' ({reason_str})' if reason_str else ''}"
                    ).strip(),
                }
            )
            self._broadcast(
                {
                    "type": "intent_ended",
                    "event": "intent_ended",
                    "category": "agent_event",
                    "tick": event.tick,
                    "source": event.source,
                    "status": status_str,
                    "ok": ok_val,
                    "goal": goal_str,
                    "reason": reason_str,
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

        if isinstance(event, SpokeEvent) or event.type == "spoke":
            spoke_text = str(event.text or "")
            context_objs: list[str] = []
            if self.world.body.holding:
                context_objs.append(self.world.body.holding)
            active_int = self.loop.controller.intent or self.last_intent
            if active_int is not None:
                for st in active_int.steps:
                    if st.target and st.target not in context_objs:
                        context_objs.append(st.target)
                for m in _OBJ_ID_PATTERN.findall(f"{active_int.goal} {active_int.thought}"):
                    ml = m.lower()
                    if ml not in context_objs:
                        context_objs.append(ml)
            self.spoke_history.append(
                {
                    "tick": event.tick,
                    "text": spoke_text,
                    "holding": self.world.body.holding,
                    "context_objects": context_objs,
                }
            )
            self._append_timeline_entry(
                {
                    "kind": "action",
                    "type": "spoke",
                    "event": "spoke",
                    "subkind": "speak",
                    "tick": event.tick,
                    "source": "agent",
                    "verb": "speak",
                    "action": "speak",
                    "text": spoke_text,
                    "params": {"text": spoke_text},
                    "summary": f'speak: "{spoke_text}"',
                }
            )
        elif isinstance(event, WeatherChangedEvent) or event.type == "weather_changed":
            w_type = getattr(event, "weather_type", self.world.weather_type)
            w_int = getattr(event, "intensity", self.world.weather_intensity)
            self._append_timeline_entry(
                {
                    "kind": "action",
                    "type": "weather_changed",
                    "event": "weather_changed",
                    "subkind": "weather",
                    "tick": event.tick,
                    "source": event.source,
                    "verb": "weather_changed",
                    "action": "weather_changed",
                    "weather_type": w_type,
                    "intensity": w_int,
                    "summary": f"Weather changed to {w_type} ({round(float(w_int or 0) * 100)}%)",
                }
            )

        agent_action_dict: dict[str, Any] | None = None
        if event.type == "tick":
            last_status = getattr(self.loop, "_last_controller_status", None)
            if last_status is not None and not last_status.is_idle and last_status.step is not None:
                step_obj = last_status.step
                step_d = (
                    step_obj.model_dump(mode="json")
                    if hasattr(step_obj, "model_dump")
                    else dict(step_obj)
                )
                res_obj = getattr(last_status, "result", None)
                agent_action_dict = {
                    "verb": step_d.get("verb"),
                    "target": step_d.get("target"),
                    "params": step_d.get("params") or {},
                    "status": last_status.state,
                    "ok": bool(res_obj.ok) if res_obj is not None else last_status.ok,
                    "reason": last_status.reason
                    or (res_obj.reason if res_obj is not None else None),
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
        if agent_action_dict is not None:
            delta_payload["agent_action"] = agent_action_dict
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
                "wind_direction": self.world.weather.wind_direction.model_dump(mode="json"),
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
        if agent_action_dict is not None:
            msg["agent_action"] = agent_action_dict
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
        self._append_timeline_entry(
            {
                "kind": "action",
                "type": "creator_action",
                "event": "creator_action",
                "subkind": "creator_action",
                "action": action,
                "verb": action,
                "source": "creator",
                "tick": self.world.tick,
                "object_id": extra.get("object_id") or details_dict.get("object_id"),
                "text": extra.get("text") or details_dict.get("text"),
                "details": details_dict,
                "summary": f"Creator: {action}",
            }
        )
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
        """Record active controller actions into the short-term memory buffer and timeline."""
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
            step_obj = status.step
            verb_str = str(getattr(step_obj, "verb", "") or "")
            target_str = getattr(step_obj, "target", None)
            params_dict = dict(getattr(step_obj, "params", {}) or {})
            res_obj = getattr(status, "result", None)
            ok_val = bool(res_obj.ok) if res_obj is not None else status.ok
            reason_val = status.reason or (res_obj.reason if res_obj is not None else None)
            # Avoid duplicating a speak entry if SpokeEvent already added one on this tick
            already_spoke = (
                verb_str == "speak"
                and self.timeline_entries
                and self.timeline_entries[-1].get("subkind") == "speak"
                and self.timeline_entries[-1].get("tick") == result.tick
            )
            if not already_spoke:
                self._append_timeline_entry(
                    {
                        "kind": "action",
                        "type": "agent_action",
                        "event": "agent_action",
                        "subkind": "agent_action",
                        "tick": result.tick,
                        "source": "agent",
                        "verb": verb_str,
                        "action": verb_str,
                        "target": target_str,
                        "object_id": target_str,
                        "params": params_dict,
                        "status": status.state,
                        "ok": ok_val,
                        "reason": reason_val,
                        "summary": (
                            f"{verb_str}{f' {target_str}' if target_str else ''}"
                            f"{f' ({reason_val})' if reason_val else ''}"
                        ),
                    }
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
            entries = []
            for entry in self.world.inbox.all():
                d = entry.to_dict()
                d.setdefault("message", d.get("text"))
                d.setdefault("note", d.get("creator_note"))
                d.setdefault("reason", d.get("delivery_reason"))
                entries.append(d)
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

    def get_presets_payload(self) -> dict[str, Any]:
        """Return the first-run object property presets."""
        items = get_spawn_presets()
        by_key = {item["key"]: item["props"] for item in items}
        return {
            "presets": items,
            "items": items,
            "by_key": by_key,
            "count": len(items),
        }

    def get_timeline_payload(
        self,
        *,
        kind: str | Sequence[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return observer timeline entries (wakes, thoughts, intents, actions), optionally filtered by kind."""
        with self._lock:
            all_entries = list(self.timeline_entries)

            allowed_kinds: set[str] = set()
            if isinstance(kind, str) and kind.strip():
                for part in kind.split(","):
                    cleaned = part.strip().lower()
                    if not cleaned or cleaned == "all":
                        continue
                    if cleaned in ("thoughts", "thought"):
                        allowed_kinds.add("thought")
                    elif cleaned in ("intents", "intent", "intent_set", "intent_ended"):
                        allowed_kinds.add("intent")
                    elif cleaned in ("wakes", "wake"):
                        allowed_kinds.add("wake")
                    elif cleaned in ("actions", "action", "agent_action", "creator_action"):
                        allowed_kinds.add("action")
                    else:
                        allowed_kinds.add(cleaned)
            elif isinstance(kind, Sequence) and not isinstance(kind, str | bytes):
                for item in kind:
                    cleaned = str(item).strip().lower()
                    if cleaned and cleaned != "all":
                        allowed_kinds.add(cleaned.rstrip("s") if cleaned.endswith("s") else cleaned)

            if allowed_kinds:
                filtered = [
                    e
                    for e in all_entries
                    if str(e.get("kind", "")).lower() in allowed_kinds
                    or str(e.get("type", "")).lower() in allowed_kinds
                    or str(e.get("subkind", "")).lower() in allowed_kinds
                ]
            else:
                filtered = all_entries

            if limit is not None and limit > 0:
                filtered = filtered[-int(limit):]

            return {
                "tick": self.world.tick,
                "kind": kind if isinstance(kind, str) else "all",
                "timeline": filtered,
                "items": filtered,
                "entries": filtered,
                "events": filtered,
                "count": len(filtered),
                "total": len(all_entries),
                "kinds": ["wake", "thought", "intent", "action"],
            }

    def get_names_payload(self) -> dict[str, Any]:
        """Extract and return words/names the agent uses for objects from speak, thoughts, and beliefs."""
        with self._lock:
            speeches: list[Any] = list(self.spoke_history)
            for entry in self.world.inbox.all():
                if entry.text:
                    speeches.append(
                        {
                            "tick": entry.tick,
                            "text": entry.text,
                            "object_id": entry.object_id,
                            "context_objects": [
                                oid
                                for oid in (entry.object_id, entry.mailbox_id)
                                if isinstance(oid, str)
                            ],
                        }
                    )
            for buf_entry in getattr(self.memory.buffer, "entries", ()):
                act = getattr(buf_entry, "action", None)
                verb = act.get("verb") if isinstance(act, Mapping) else getattr(act, "verb", None)
                if verb == "speak":
                    params = (
                        act.get("params")
                        if isinstance(act, Mapping)
                        else getattr(act, "params", {})
                    ) or {}
                    txt = params.get("text")
                    if isinstance(txt, str) and txt.strip():
                        speeches.append({"tick": getattr(buf_entry, "tick", 0), "text": txt})

            thoughts: list[Any] = list(self.loop.intent_set_events)
            if self.last_intent is not None and not any(
                getattr(ev, "intent", None) is self.last_intent
                for ev in self.loop.intent_set_events
            ):
                thoughts.append(self.last_intent)
            for tl_item in self.timeline_entries:
                if tl_item.get("kind") in ("thought", "intent"):
                    thoughts.append(tl_item)

            beliefs = list(self.memory.belief_store.all())
            episodes = list(self.memory.store.all())

            return extract_object_names(
                speeches=speeches,
                thoughts=thoughts,
                beliefs=beliefs,
                episodes=episodes,
                world=self.world,
            )

    def update_inbox_note(
        self,
        note: str | None,
        *,
        entry_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a private creator note on an inbox message or experiment without sending any acknowledgement."""
        with self._lock:
            updated_entry_dict: dict[str, Any] | None = None
            if entry_id is not None:
                updated_entry = self.world.mailbox.update_entry_note(int(entry_id), note)
                updated_entry_dict = updated_entry.to_dict()
                updated_entry_dict.setdefault("message", updated_entry_dict.get("text"))
                updated_entry_dict.setdefault("note", updated_entry_dict.get("creator_note"))
                updated_entry_dict.setdefault("reason", updated_entry_dict.get("delivery_reason"))
            else:
                self.world.mailbox.set_creator_note(note)

            inbox_payload = self.get_inbox_payload()
            return {
                "ok": True,
                "tick": self.world.tick,
                "creator_note": self.world.creator_note,
                "note": note,
                "entry_id": int(entry_id) if entry_id is not None else None,
                "entry": updated_entry_dict,
                **inbox_payload,
                "inbox": inbox_payload,
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
        if expected:
            if not provided:
                raise HTTPException(
                    status_code=401,
                    detail="Creator token required.",
                )
            if not secrets.compare_digest(provided, expected):
                raise HTTPException(
                    status_code=403,
                    detail="Invalid creator token.",
                )
        elif not is_local_host(sim_service.host):
            if not provided:
                raise HTTPException(
                    status_code=401,
                    detail="Creator token required.",
                )
            raise HTTPException(
                status_code=403,
                detail="Invalid creator token.",
            )

    # ------------------------------------------------------------------
    # Static client & Read endpoints
    # ------------------------------------------------------------------

    @app.get("/", dependencies=[Depends(_require_read_access)])
    @app.get("/index.html", dependencies=[Depends(_require_read_access)])
    async def get_root_index() -> FileResponse:
        return _serve_static_file("index.html")

    @app.get("/static/{file_path:path}", dependencies=[Depends(_require_read_access)])
    async def get_static_prefixed(file_path: str) -> FileResponse:
        return _serve_static_file(file_path)

    @app.get("/health", dependencies=[Depends(_require_read_access)])
    @app.get("/status", dependencies=[Depends(_require_read_access)])
    @app.get("/api/status", dependencies=[Depends(_require_read_access)])
    async def get_status(request: Request) -> dict[str, Any]:
        expected = sim_service.creator_token
        provided = _extract_request_token(request)
        is_auth = False
        if not sim_service.spectator_mode and not _is_spectator_request(request):
            if expected:
                is_auth = bool(provided and secrets.compare_digest(provided, expected))
            else:
                is_auth = is_local_host(sim_service.host)
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "tick": sim_service.world.tick,
            "paused": sim_service.paused,
            "speed": sim_service.speed,
            "spectator_enabled": sim_service.spectator_enabled,
            "spectator_mode": sim_service.spectator_mode,
            "auth_required": bool(expected),
            "authenticated": is_auth,
        }

    @app.post("/auth", response_model=None)
    @app.post("/login", response_model=None)
    @app.post("/api/auth", response_model=None)
    @app.post("/api/login", response_model=None)
    async def verify_creator_auth(request: Request) -> dict[str, Any] | JSONResponse:
        if sim_service.spectator_mode or _is_spectator_request(request):
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "authenticated": False,
                    "mode": "spectator",
                    "detail": "Spectator mode cannot mutate.",
                },
            )
        provided = _extract_request_token(request)
        if not provided:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, str) and body.strip():
                provided = body.strip()
            elif isinstance(body, Mapping):
                for k in ("token", "creator_token", "api_key", "password"):
                    val = body.get(k)
                    if isinstance(val, str) and val.strip():
                        provided = val.strip()
                        break
        expected = sim_service.creator_token
        if expected:
            if not provided:
                return JSONResponse(
                    status_code=401,
                    content={
                        "ok": False,
                        "authenticated": False,
                        "auth_required": True,
                        "mode": "spectator",
                        "detail": "Creator token required.",
                    },
                )
            if not secrets.compare_digest(provided, expected):
                return JSONResponse(
                    status_code=401,
                    content={
                        "ok": False,
                        "authenticated": False,
                        "auth_required": True,
                        "mode": "spectator",
                        "detail": "Invalid creator token.",
                    },
                )
        elif not is_local_host(sim_service.host):
            return JSONResponse(
                status_code=401,
                content={
                    "ok": False,
                    "authenticated": False,
                    "auth_required": True,
                    "mode": "spectator",
                    "detail": "Invalid creator token.",
                },
            )
        return {
            "ok": True,
            "authenticated": True,
            "auth_required": bool(expected),
            "mode": "creator",
            "role": "creator",
            "spectator_mode": False,
        }

    @app.get("/presets", dependencies=[Depends(_require_read_access)])
    @app.get("/objects/presets", dependencies=[Depends(_require_read_access)])
    @app.get("/world/presets", dependencies=[Depends(_require_read_access)])
    @app.get("/api/presets", dependencies=[Depends(_require_read_access)])
    async def get_presets() -> dict[str, Any]:
        return sim_service.get_presets_payload()

    @app.get("/timeline", dependencies=[Depends(_require_read_access)])
    @app.get("/events", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/timeline", dependencies=[Depends(_require_read_access)])
    @app.get("/observer/timeline", dependencies=[Depends(_require_read_access)])
    @app.get("/api/timeline", dependencies=[Depends(_require_read_access)])
    async def get_timeline(
        kind: str | None = None,
        filter: str | None = None,
        type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        eff_kind = kind if kind is not None else (filter if filter is not None else type)
        return sim_service.get_timeline_payload(kind=eff_kind, limit=limit)

    @app.get("/names", dependencies=[Depends(_require_read_access)])
    @app.get("/agent/names", dependencies=[Depends(_require_read_access)])
    @app.get("/observer/names", dependencies=[Depends(_require_read_access)])
    @app.get("/api/names", dependencies=[Depends(_require_read_access)])
    async def get_names() -> dict[str, Any]:
        return sim_service.get_names_payload()

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

        if "props" in raw and isinstance(raw["props"], Mapping) and not raw.get("return_preserved_object"):
            target_ref: Any = dict(raw["props"])
        elif "object" in raw and isinstance(raw["object"], Mapping) and not raw.get("return_preserved_object"):
            target_ref = dict(raw["object"])
        elif "properties" in raw and isinstance(raw["properties"], Mapping) and not raw.get("return_preserved_object"):
            target_ref = dict(raw["properties"])
        elif "entry_id" in raw or "inbox_id" in raw:
            target_ref = raw.get("entry_id", raw.get("inbox_id"))
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

    async def _handle_inbox_note(entry_id: int | None, request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_entry_id = entry_id
        note_val: str | None = None
        if isinstance(body, str):
            note_val = body
        elif isinstance(body, Mapping):
            raw_id = body.get("entry_id", body.get("inbox_id", body.get("id")))
            if target_entry_id is None and raw_id is not None and not isinstance(raw_id, bool):
                try:
                    target_entry_id = int(raw_id)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(status_code=400, detail="entry_id must be an integer") from exc
            raw_note = body.get(
                "note",
                body.get("creator_note", body.get("private_note", body.get("text"))),
            )
            note_val = str(raw_note) if raw_note is not None else None
        else:
            raise HTTPException(status_code=400, detail="Invalid note payload")

        try:
            payload = sim_service.update_inbox_note(note_val, entry_id=target_entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await asyncio.sleep(0)
        return payload

    @app.post("/inbox/note", dependencies=[Depends(_require_creator_access)])
    @app.put("/inbox/note", dependencies=[Depends(_require_creator_access)])
    @app.patch("/inbox/note", dependencies=[Depends(_require_creator_access)])
    @app.post("/mailbox/note", dependencies=[Depends(_require_creator_access)])
    @app.post("/creator/note", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/inbox/note", dependencies=[Depends(_require_creator_access)])
    async def creator_set_inbox_note(request: Request) -> dict[str, Any]:
        return await _handle_inbox_note(None, request)

    @app.post("/inbox/{entry_id}/note", dependencies=[Depends(_require_creator_access)])
    @app.put("/inbox/{entry_id}/note", dependencies=[Depends(_require_creator_access)])
    @app.patch("/inbox/{entry_id}/note", dependencies=[Depends(_require_creator_access)])
    @app.post("/api/inbox/{entry_id}/note", dependencies=[Depends(_require_creator_access)])
    async def creator_set_inbox_entry_note(entry_id: int, request: Request) -> dict[str, Any]:
        return await _handle_inbox_note(entry_id, request)

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

    @app.get("/{file_path:path}", dependencies=[Depends(_require_read_access)])
    async def get_static_root_asset(file_path: str) -> FileResponse:
        return _serve_static_file(file_path)

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
    "FIRST_RUN_PRESETS",
    "LOCAL_HOSTS",
    "SCHEMA_VERSION",
    "WS_SCHEMA_VERSION",
    "SimulationServer",
    "SimulationService",
    "build_app",
    "create_app",
    "extract_object_names",
    "get_app",
    "get_spawn_presets",
    "is_local_host",
    "make_app",
    "resolve_creator_token",
    "run_server",
    "serve",
    "validate_server_binding",
]
