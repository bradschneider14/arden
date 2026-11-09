/**
 * Main browser client entry module for the Arden live world viewer.
 */

import { WorldRenderer } from "./renderer.js";
import {
  applySnapshot,
  applyWorldDelta,
  applyWsMessage,
  computeObjectElevation,
  createWorldStore,
  formatTimeOfDay,
  getDayPhase,
  getObjectLabel,
  hitTestWorld,
  resolveObjectPosition,
} from "./state.js";
import { WorldSocketClient } from "./ws.js";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNum(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "0.00";
}

export function initArdenApp() {
  const canvas = document.getElementById("world-canvas");
  if (!canvas) return null;

  const store = createWorldStore();
  const renderer = new WorldRenderer(canvas, store);
  const wsClient = new WorldSocketClient(store);

  // DOM elements
  const elConnBadge = document.getElementById("conn-badge");
  const elConnText = document.getElementById("conn-text");
  const elTick = document.getElementById("hud-tick");
  const elTime = document.getElementById("hud-time");
  const elLight = document.getElementById("hud-light");
  const elTemp = document.getElementById("hud-temp");
  const elWeather = document.getElementById("hud-weather");
  const elWind = document.getElementById("hud-wind");
  const elEnergy = document.getElementById("hud-energy");
  const elComfort = document.getElementById("hud-comfort");
  const elHolding = document.getElementById("hud-holding");
  const elAgentPos = document.getElementById("hud-agent-pos");
  const elAgentGoal = document.getElementById("hud-agent-goal");
  const elAgentThought = document.getElementById("hud-agent-thought");
  const elTooltip = document.getElementById("hover-tooltip");
  const elInspector = document.getElementById("object-inspector");
  const elInspectorBody = document.getElementById("inspector-body");
  const elObjectList = document.getElementById("object-list");
  const elObjectCount = document.getElementById("object-count");
  const elZoomLabel = document.getElementById("zoom-label");
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const btnResetView = document.getElementById("btn-reset-view");
  const btnCloseInspector = document.getElementById("btn-close-inspector");

  let lastHoverClientX = 0;
  let lastHoverClientY = 0;

  function updateConnectionIndicator(state) {
    if (!elConnBadge || !elConnText) return;
    const status = state.connectionStatus || "connecting";
    elConnBadge.dataset.status = status;
    if (status === "connected") {
      elConnText.textContent = "Connected";
    } else if (status === "reconnecting") {
      elConnText.textContent = `Reconnecting (${state.reconnectAttempt})…`;
    } else if (status === "connecting") {
      elConnText.textContent = "Connecting…";
    } else {
      elConnText.textContent = "Disconnected";
    }
  }

  function updateHud(state) {
    updateConnectionIndicator(state);

    if (elTick) {
      elTick.textContent = `${state.tick}${state.paused ? " (paused)" : ""}`;
    }
    if (elTime) {
      const timeStr = formatTimeOfDay(state.time?.time_of_day ?? 0.5);
      elTime.textContent = state.time?.frozen ? `${timeStr} [frozen]` : timeStr;
    }
    if (elLight) {
      const phase = getDayPhase(state.time?.time_of_day ?? 0.5);
      elLight.textContent = `${formatNum(state.time?.light_level ?? 1.0)} (${phase})`;
    }
    if (elTemp) {
      elTemp.textContent = formatNum(state.time?.ambient_temperature ?? 1.0);
    }
    if (elWeather) {
      const wType = state.weather?.type || "clear";
      const wInt = Number(state.weather?.intensity ?? 0);
      elWeather.textContent =
        wType === "clear" && wInt === 0
          ? "clear"
          : `${wType} (${Math.round(wInt * 100)}%)`;
      elWeather.dataset.weather = wType;
    }
    if (elWind) {
      const wd = state.weather?.wind_direction || { x: 1, y: 0 };
      const wType = state.weather?.type || "clear";
      const isWindy = wType === "wind" || wType === "storm";
      elWind.textContent = isWindy
        ? `(${formatNum(wd.x, 1)}, ${formatNum(wd.y, 1)})`
        : "calm";
    }

    const body = state.body || {};
    if (elEnergy) elEnergy.textContent = formatNum(body.energy ?? 1.0);
    if (elComfort) elComfort.textContent = formatNum(body.comfort ?? 1.0);
    if (elAgentPos) {
      const px = formatNum(body.position?.x ?? 0, 1);
      const py = formatNum(body.position?.y ?? 0, 1);
      elAgentPos.textContent = `(${px}, ${py}) [${body.level || "ground"}]`;
    }
    if (elHolding) {
      if (body.holding) {
        const heldLabel = getObjectLabel(state, body.holding);
        elHolding.textContent = heldLabel
          ? `${body.holding} (${heldLabel})`
          : body.holding;
      } else {
        elHolding.textContent = "none";
      }
    }
    if (elAgentGoal) {
      elAgentGoal.textContent = state.agent?.goal || "idle";
    }
    if (elAgentThought) {
      elAgentThought.textContent = state.agent?.thought || "—";
    }
    if (elZoomLabel) {
      elZoomLabel.textContent = `${Math.round(renderer.zoom * 100)}%`;
    }
  }

  function updateTooltip(state) {
    if (!elTooltip) return;
    const target = state.hoveredTarget;
    if (!target) {
      elTooltip.hidden = true;
      return;
    }

    if (target.kind === "object") {
      const obj = state.objects?.[target.id] || target.object;
      if (!obj) {
        elTooltip.hidden = true;
        return;
      }
      const label = getObjectLabel(state, obj) || obj.id;
      const pos = resolveObjectPosition(obj, state.objects);
      const elev = computeObjectElevation(obj, state.objects);
      elTooltip.innerHTML = `
        <div class="tooltip-title">${escapeHtml(label)}</div>
        <div class="tooltip-sub">id: <code>${escapeHtml(obj.id)}</code> &middot; ${escapeHtml(obj.shape)}</div>
        <div class="tooltip-meta">
          pos: (${formatNum(pos.x, 1)}, ${formatNum(pos.y, 1)}) &middot;
          h: ${formatNum(obj.height, 2)}m${elev > 0 ? ` (+${formatNum(elev, 1)}m)` : ""} &middot;
          level: ${escapeHtml(obj.level)}
        </div>
      `;
    } else if (target.kind === "agent") {
      const body = state.body || {};
      const heldId = body.holding;
      const heldLabel = heldId ? getObjectLabel(state, heldId) : null;
      elTooltip.innerHTML = `
        <div class="tooltip-title">Agent</div>
        <div class="tooltip-sub">energy: ${formatNum(body.energy)} &middot; comfort: ${formatNum(body.comfort)}</div>
        <div class="tooltip-meta">
          pos: (${formatNum(body.position?.x, 1)}, ${formatNum(body.position?.y, 1)}) &middot;
          holding: ${heldId ? `${escapeHtml(heldId)}${heldLabel ? ` (${escapeHtml(heldLabel)})` : ""}` : "none"}
        </div>
      `;
    }

    elTooltip.hidden = false;
    elTooltip.style.left = `${lastHoverClientX + 14}px`;
    elTooltip.style.top = `${lastHoverClientY + 14}px`;
  }

  function updateInspector(state) {
    if (!elInspector || !elInspectorBody) return;
    const selectedId = state.selectedObjectId;
    if (!selectedId) {
      elInspector.dataset.empty = "true";
      elInspectorBody.innerHTML = `
        <p class="inspector-placeholder">
          Click any object in the world (or in the object list below) to inspect its internal ID and physical properties.
        </p>
      `;
      return;
    }

    const obj = state.objects?.[selectedId];
    if (!obj) {
      elInspector.dataset.empty = "false";
      elInspectorBody.innerHTML = `
        <div class="inspector-header-row">
          <span class="inspector-id">${escapeHtml(selectedId)}</span>
          <span class="badge badge-warn">despawned</span>
        </div>
        <p class="inspector-placeholder">This object is no longer present in the world.</p>
      `;
      return;
    }

    elInspector.dataset.empty = "false";
    const label = getObjectLabel(state, obj);
    const pos = resolveObjectPosition(obj, state.objects);
    const elev = computeObjectElevation(obj, state.objects);

    // Find any objects inside or on_top_of this object
    const childrenInside = Object.values(state.objects || {}).filter(
      (o) => o && o.level === `inside:${obj.id}`
    );
    const childrenOnTop = Object.values(state.objects || {}).filter(
      (o) => o && o.level === `on_top_of:${obj.id}`
    );

    const flags = [];
    if (obj.fixed) flags.push("fixed");
    if (obj.hollow) flags.push("hollow");
    if (obj.fragile) flags.push("fragile");
    if (obj.binding) flags.push("binding");
    if (state.body?.holding === obj.id) flags.push("held by agent");

    const stateEntries = obj.state ? Object.entries(obj.state) : [];
    const behaviorRules = Array.isArray(obj.behavior) ? obj.behavior : [];

    elInspectorBody.innerHTML = `
      <div class="inspector-header-row">
        <div>
          <div class="inspector-id" data-testid="selected-object-id">${escapeHtml(obj.id)}</div>
          <div class="inspector-label" data-testid="selected-creator-label">
            ${label ? escapeHtml(label) : "<em>(no creator_label)</em>"}
          </div>
        </div>
        <span class="shape-pill">${escapeHtml(obj.shape)}</span>
      </div>

      ${
        flags.length > 0
          ? `<div class="flag-row">${flags.map((f) => `<span class="badge">${escapeHtml(f)}</span>`).join("")}</div>`
          : ""
      }

      <dl class="prop-grid">
        <dt>Internal ID</dt>
        <dd><code>${escapeHtml(obj.id)}</code></dd>
        <dt>Creator label</dt>
        <dd>${label ? escapeHtml(label) : "—"}</dd>
        <dt>Position</dt>
        <dd>(${formatNum(pos.x, 2)}, ${formatNum(pos.y, 2)})</dd>
        <dt>Level</dt>
        <dd><code>${escapeHtml(obj.level)}</code>${elev > 0 ? ` (elev ${formatNum(elev, 2)}m)` : ""}</dd>
        <dt>Shape</dt>
        <dd>${escapeHtml(obj.shape)}</dd>
        <dt>Color</dt>
        <dd>${escapeHtml(obj.color || "none")}</dd>
        <dt>Height</dt>
        <dd>${formatNum(obj.height, 3)} m</dd>
        <dt>Volume</dt>
        <dd>${formatNum(obj.volume, 4)} m&sup3;</dd>
        <dt>Weight</dt>
        <dd>${formatNum(obj.weight, 2)} kg</dd>
        <dt>Rigidity</dt>
        <dd>${formatNum(obj.rigidity, 2)}</dd>
      </dl>

      <div class="inspector-subhead">State</div>
      ${
        stateEntries.length > 0
          ? `<pre class="json-box">${escapeHtml(JSON.stringify(obj.state, null, 2))}</pre>`
          : `<div class="muted-note">Empty state <code>{}</code></div>`
      }

      <div class="inspector-subhead">Hidden Behavior Rules</div>
      ${
        behaviorRules.length > 0
          ? `<pre class="json-box">${escapeHtml(JSON.stringify(behaviorRules, null, 2))}</pre>`
          : `<div class="muted-note">None</div>`
      }

      ${
        childrenInside.length > 0
          ? `<div class="inspector-subhead">Contents (${childrenInside.length})</div>
             <div class="related-list">
               ${childrenInside
                 .map(
                   (c) =>
                     `<button type="button" class="related-item-btn" data-select-id="${escapeHtml(c.id)}">
                        <code>${escapeHtml(c.id)}</code> ${escapeHtml(getObjectLabel(state, c) || c.shape)}
                      </button>`
                 )
                 .join("")}
             </div>`
          : ""
      }

      ${
        childrenOnTop.length > 0
          ? `<div class="inspector-subhead">Stacked On Top (${childrenOnTop.length})</div>
             <div class="related-list">
               ${childrenOnTop
                 .map(
                   (c) =>
                     `<button type="button" class="related-item-btn" data-select-id="${escapeHtml(c.id)}">
                        <code>${escapeHtml(c.id)}</code> ${escapeHtml(getObjectLabel(state, c) || c.shape)}
                      </button>`
                 )
                 .join("")}
             </div>`
          : ""
      }
    `;

    for (const btn of elInspectorBody.querySelectorAll("[data-select-id]")) {
      btn.addEventListener("click", () => {
        store.setSelectedObjectId(btn.getAttribute("data-select-id"));
      });
    }
  }

  function updateObjectList(state) {
    if (!elObjectList) return;
    const objects = Object.values(state.objects || {}).sort((a, b) => {
      const na = parseInt(String(a.id).replace("object_", ""), 10) || 0;
      const nb = parseInt(String(b.id).replace("object_", ""), 10) || 0;
      return na - nb;
    });

    if (elObjectCount) {
      elObjectCount.textContent = String(objects.length);
    }

    const selectedId = state.selectedObjectId;
    elObjectList.innerHTML = objects
      .map((obj) => {
        const label = getObjectLabel(state, obj) || obj.shape;
        const isSelected = obj.id === selectedId;
        const pos = resolveObjectPosition(obj, state.objects);
        return `
          <button
            type="button"
            class="object-list-row${isSelected ? " is-selected" : ""}"
            data-object-id="${escapeHtml(obj.id)}"
            title="${escapeHtml(label)} (${escapeHtml(obj.id)})"
          >
            <span class="obj-row-id">${escapeHtml(obj.id)}</span>
            <span class="obj-row-label">${escapeHtml(label)}</span>
            <span class="obj-row-pos">(${formatNum(pos.x, 1)}, ${formatNum(pos.y, 1)})</span>
          </button>
        `;
      })
      .join("");

    for (const row of elObjectList.querySelectorAll("[data-object-id]")) {
      row.addEventListener("click", () => {
        const oid = row.getAttribute("data-object-id");
        store.setSelectedObjectId(oid);
      });
    }
  }

  // Subscribe UI and renderer to store updates
  store.subscribe((state, reason) => {
    renderer.renderWorld();
    updateHud(state);
    if (reason === "hover") {
      updateTooltip(state);
    } else if (reason === "selection") {
      updateInspector(state);
      updateObjectList(state);
    } else {
      updateTooltip(state);
      updateInspector(state);
      updateObjectList(state);
    }
  });

  // Pointer interactions: Pan (drag), Hover tooltip, Click object selection
  let isDragging = false;
  let dragMoved = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastPointerX = 0;
  let lastPointerY = 0;

  function getCanvasLocalCoords(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 1) return;
    isDragging = true;
    dragMoved = false;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;
    if (typeof canvas.setPointerCapture === "function") {
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch (_err) {
        // Ignore pointer capture errors
      }
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    lastHoverClientX = event.clientX;
    lastHoverClientY = event.clientY;

    if (isDragging) {
      const dx = event.clientX - lastPointerX;
      const dy = event.clientY - lastPointerY;
      lastPointerX = event.clientX;
      lastPointerY = event.clientY;
      if (Math.hypot(event.clientX - dragStartX, event.clientY - dragStartY) > 4) {
        dragMoved = true;
      }
      if (dragMoved) {
        renderer.panBy(dx, dy);
        return;
      }
    }

    const local = getCanvasLocalCoords(event);
    const worldPt = renderer.screenToWorld(local.x, local.y);
    const hit = hitTestWorld(store.getState(), worldPt.x, worldPt.y);
    const prev = store.getState().hoveredTarget;
    if (hit?.id !== prev?.id || hit?.kind !== prev?.kind) {
      store.setHoveredTarget(hit);
    } else if (hit) {
      updateTooltip(store.getState());
    }
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!isDragging) return;
    isDragging = false;
    if (!dragMoved) {
      const local = getCanvasLocalCoords(event);
      const worldPt = renderer.screenToWorld(local.x, local.y);
      const hit = hitTestWorld(store.getState(), worldPt.x, worldPt.y);
      if (hit && hit.kind === "object") {
        store.setSelectedObjectId(hit.id);
      } else if (!hit) {
        store.setSelectedObjectId(null);
      }
    }
  });

  canvas.addEventListener("pointerleave", () => {
    isDragging = false;
    store.setHoveredTarget(null);
  });

  // Wheel zoom centered at pointer
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const local = getCanvasLocalCoords(event);
      const factor = event.deltaY < 0 ? 1.14 : 1 / 1.14;
      renderer.zoomAt(factor, local.x, local.y);
      updateHud(store.getState());
    },
    { passive: false }
  );

  // Zoom / Reset / Close Inspector buttons
  if (btnZoomIn) {
    btnZoomIn.addEventListener("click", () => {
      renderer.zoomAt(1.22);
      updateHud(store.getState());
    });
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener("click", () => {
      renderer.zoomAt(1 / 1.22);
      updateHud(store.getState());
    });
  }
  if (btnResetView) {
    btnResetView.addEventListener("click", () => {
      renderer.resetView();
      updateHud(store.getState());
    });
  }
  if (btnCloseInspector) {
    btnCloseInspector.addEventListener("click", () => {
      store.setSelectedObjectId(null);
    });
  }

  // Keyboard pan/zoom shortcuts
  window.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
      return;
    }
    const step = 36;
    if (event.key === "ArrowLeft" || event.key === "a") {
      renderer.panBy(step, 0);
    } else if (event.key === "ArrowRight" || event.key === "d") {
      renderer.panBy(-step, 0);
    } else if (event.key === "ArrowUp" || event.key === "w") {
      renderer.panBy(0, step);
    } else if (event.key === "ArrowDown" || event.key === "s") {
      renderer.panBy(0, -step);
    } else if (event.key === "+" || event.key === "=") {
      renderer.zoomAt(1.18);
      updateHud(store.getState());
    } else if (event.key === "-" || event.key === "_") {
      renderer.zoomAt(1 / 1.18);
      updateHud(store.getState());
    } else if (event.key === "0") {
      renderer.resetView();
      updateHud(store.getState());
    } else if (event.key === "Escape") {
      store.setSelectedObjectId(null);
    }
  });

  // Resize observer
  window.addEventListener("resize", () => {
    const wrap = canvas.parentElement;
    if (wrap) {
      renderer.resize(wrap.clientWidth, wrap.clientHeight);
    }
  });

  const wrap = canvas.parentElement;
  if (wrap && wrap.clientWidth > 0 && wrap.clientHeight > 0) {
    renderer.resize(wrap.clientWidth, wrap.clientHeight);
    renderer.centerViewOnWorld();
  }

  updateHud(store.getState());
  updateInspector(store.getState());
  updateObjectList(store.getState());

  wsClient.start();

  const appInstance = { store, renderer, wsClient };
  if (typeof window !== "undefined") {
    window.__ardenApp = appInstance;
  }
  return appInstance;
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initArdenApp());
  } else {
    initArdenApp();
  }
}

export {
  WorldRenderer,
  WorldSocketClient,
  applySnapshot,
  applyWorldDelta,
  applyWsMessage,
  createWorldStore,
  getObjectLabel,
  hitTestWorld,
};
