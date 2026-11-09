"""Web interface package and static client asset directory."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from arden.server import (
    FIRST_RUN_PRESETS,
    LOCAL_HOSTS,
    SCHEMA_VERSION,
    WS_SCHEMA_VERSION,
    SimulationServer,
    SimulationService,
    build_app,
    create_app,
    extract_object_names,
    get_app,
    get_spawn_presets,
    is_local_host,
    make_app,
    resolve_creator_token,
    run_server,
    serve,
    validate_server_binding,
)

mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"


def get_static_dir() -> Path:
    """Return the path to ``src/arden/web/static``."""
    return STATIC_DIR


__all__ = [
    "FIRST_RUN_PRESETS",
    "LOCAL_HOSTS",
    "SCHEMA_VERSION",
    "STATIC_DIR",
    "WS_SCHEMA_VERSION",
    "SimulationServer",
    "SimulationService",
    "build_app",
    "create_app",
    "extract_object_names",
    "get_app",
    "get_spawn_presets",
    "get_static_dir",
    "is_local_host",
    "make_app",
    "resolve_creator_token",
    "run_server",
    "serve",
    "validate_server_binding",
]
