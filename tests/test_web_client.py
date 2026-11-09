"""Tests for T23: Browser client static assets and FastAPI static serving."""

from __future__ import annotations

from fastapi.testclient import TestClient

from arden.server import create_app
from arden.web import STATIC_DIR, get_static_dir


def test_static_directory_layout_and_es_modules_exist() -> None:
    """`src/arden/web/static/` contains `index.html`, `app.js`, and supporting ES modules."""
    static_dir = get_static_dir()
    assert static_dir == STATIC_DIR
    assert static_dir.is_dir()
    assert static_dir.parts[-3:] == ("arden", "web", "static")

    expected_files = (
        "index.html",
        "app.js",
        "state.js",
        "renderer.js",
        "ws.js",
        "pixi-shim.js",
        "styles.css",
    )
    for filename in expected_files:
        path = static_dir / filename
        assert path.is_file(), f"Missing static file: {path}"
        assert path.stat().st_size > 0


def test_fastapi_serves_root_index_and_static_assets_200() -> None:
    """FastAPI serves `/` (`index.html`) and all static JS/CSS modules with 200 status."""
    app = create_app(world="worlds/first_run.toml", auto_start=False)

    with TestClient(app) as client:
        # Root `/` serves index.html
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert "text/html" in root_resp.headers["content-type"]
        html = root_resp.text
        assert '<canvas id="world-canvas"' in html
        assert 'type="module" src="./app.js"' in html
        assert "pixi.js@7.4.2" in html
        assert 'id="conn-badge"' in html
        assert 'id="hover-tooltip"' in html
        assert 'id="object-inspector"' in html

        # `/index.html` and `/static/index.html` also return 200
        assert client.get("/index.html").status_code == 200
        assert client.get("/static/index.html").status_code == 200

        # Static JS modules and CSS return 200 at both `/<asset>` and `/static/<asset>`
        for asset in (
            "app.js",
            "state.js",
            "renderer.js",
            "ws.js",
            "pixi-shim.js",
            "styles.css",
        ):
            r_root = client.get(f"/{asset}")
            assert r_root.status_code == 200, f"Expected 200 for /{asset}"
            if asset.endswith(".js"):
                assert "javascript" in r_root.headers["content-type"]
            elif asset.endswith(".css"):
                assert "text/css" in r_root.headers["content-type"]

            r_prefixed = client.get(f"/static/{asset}")
            assert r_prefixed.status_code == 200, f"Expected 200 for /static/{asset}"
            assert r_prefixed.text == r_root.text

        # Non-existent asset returns 404
        assert client.get("/does-not-exist.js").status_code == 404
        assert client.get("/static/does-not-exist.js").status_code == 404

        # API endpoints continue to return JSON normally alongside static serving
        status_resp = client.get("/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "ok"

        snap_resp = client.get("/snapshot")
        assert snap_resp.status_code == 200
        assert "objects" in snap_resp.json()


def test_client_modules_cover_renderer_deltas_tooltips_weather_and_inspector() -> None:
    """Static ES modules implement terrain/object shapes, creator_label tooltips, deltas, weather/time, and pan/zoom."""
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    state_js = (STATIC_DIR / "state.js").read_text(encoding="utf-8")
    renderer_js = (STATIC_DIR / "renderer.js").read_text(encoding="utf-8")
    ws_js = (STATIC_DIR / "ws.js").read_text(encoding="utf-8")

    # Terrain & object shapes, height shade/outline, agent & held items
    for shape in ("sphere", "cube", "cylinder", "flat", "long"):
        assert f'"{shape}"' in renderer_js
    for tile_type in ("grass", "water", "ledge"):
        assert f'"{tile_type}"' in renderer_js
    assert "drawAgentAndHeldItem" in renderer_js
    assert "holding" in renderer_js

    # Hover tooltips use creator_label
    assert "creator_label" in state_js
    assert "getObjectLabel" in app_js

    # WebSocket incremental deltas, reconnect, and connection status
    assert "world_delta" in state_js
    assert "object_spawned" in state_js
    assert "object_despawned" in state_js
    assert "object_moved" in state_js
    assert "reconnecting" in ws_js
    assert "connected" in ws_js

    # Time of day night tint & weather overlay (rain/wind)
    assert "drawWeatherAndNightOverlay" in renderer_js
    assert "light_level" in renderer_js
    assert "rain" in renderer_js
    assert "wind" in renderer_js

    # Pan, zoom, and clicking an object shows properties and internal id
    assert "panBy" in renderer_js
    assert "zoomAt" in renderer_js
    assert "selected-object-id" in app_js
    assert "Internal ID" in app_js
