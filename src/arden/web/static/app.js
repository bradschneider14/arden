/**
 * Main browser client entry module for the Arden live world viewer and Creator console.
 */

import { WorldRenderer } from "./renderer.js";
import {
  FIRST_RUN_PRESETS,
  applySnapshot,
  applyWorldDelta,
  applyWsMessage,
  computeObjectElevation,
  createWorldStore,
  extractNamesFromText,
  filterTimelineEntries,
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

/**
 * Render an inline SVG confidence history sparkline chart for a belief's `history` array.
 */
export function renderBeliefHistoryChart(history = []) {
  const pts = Array.isArray(history) && history.length > 0 ? history : [{ tick: 0, confidence: 0.5 }];
  const w = 160;
  const h = 38;
  const padX = 6;
  const padY = 5;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const ticks = pts.map((p) => Number(p.tick || 0));
  const minTick = Math.min(...ticks);
  const maxTick = Math.max(...ticks);
  const tickSpan = Math.max(1, maxTick - minTick);

  const coords = pts.map((p, idx) => {
    const conf = Math.max(0, Math.min(1, Number(p.confidence ?? 0.5)));
    const xRatio = pts.length === 1 ? 0.5 : tickSpan > 0 ? (Number(p.tick || 0) - minTick) / tickSpan : idx / (pts.length - 1);
    const x = padX + xRatio * innerW;
    const y = padY + (1 - conf) * innerH;
    return { x, y, tick: Number(p.tick || 0), confidence: conf };
  });

  const polylinePoints =
    coords.length === 1
      ? `${padX},${coords[0].y.toFixed(1)} ${w - padX},${coords[0].y.toFixed(1)}`
      : coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");

  const circles = coords
    .map(
      (c) =>
        `<circle cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="2.6" fill="#38bdf8"><title>tick ${c.tick}: ${Math.round(c.confidence * 100)}%</title></circle>`
    )
    .join("");

  return `
    <svg class="belief-history-chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Belief confidence history chart">
      <line x1="${padX}" y1="${padY}" x2="${w - padX}" y2="${padY}" stroke="#1e293b" stroke-width="1" />
      <line x1="${padX}" y1="${(padY + innerH * 0.5).toFixed(1)}" x2="${w - padX}" y2="${(padY + innerH * 0.5).toFixed(1)}" stroke="#1e293b" stroke-dasharray="2,2" stroke-width="1" />
      <line x1="${padX}" y1="${h - padY}" x2="${w - padX}" y2="${h - padY}" stroke="#1e293b" stroke-width="1" />
      <polyline fill="none" stroke="#38bdf8" stroke-width="1.8" points="${polylinePoints}" />
      ${circles}
    </svg>
  `;
}

export function initArdenApp() {
  const canvas = document.getElementById("world-canvas");
  if (!canvas) return null;

  const store = createWorldStore();
  const renderer = new WorldRenderer(canvas, store);
  const wsClient = new WorldSocketClient(store);

  // DOM elements
  const elAppShell = document.getElementById("app-shell");
  const elConnBadge = document.getElementById("conn-badge");
  const elConnText = document.getElementById("conn-text");
  const elAuthModeBadge = document.getElementById("auth-mode-badge");
  const elLoginForm = document.getElementById("creator-login-form");
  const elTokenInput = document.getElementById("creator-token-input");
  const btnSpectatorToggle = document.getElementById("btn-spectator-toggle");
  const elAuthStatusMsg = document.getElementById("auth-status-msg");
  const elConsoleFeedback = document.getElementById("console-feedback");

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

  // Controls panel elements
  const btnPauseResume = document.getElementById("btn-pause-resume");
  const elStepTicksInput = document.getElementById("step-ticks-input");
  const btnStep = document.getElementById("btn-step");
  const elSpeedSelect = document.getElementById("speed-select");
  const btnSetSpeed = document.getElementById("btn-set-speed");

  const elTimeSlider = document.getElementById("time-slider");
  const elTimeSliderLabel = document.getElementById("time-slider-label");
  const elFreezeTimeCheckbox = document.getElementById("freeze-time-checkbox");
  const btnApplyTime = document.getElementById("btn-apply-time");

  const elWeatherTypeSelect = document.getElementById("weather-type-select");
  const elWeatherIntensitySlider = document.getElementById("weather-intensity-slider");
  const elWeatherIntensityVal = document.getElementById("weather-intensity-val");
  const elWeatherDurationInput = document.getElementById("weather-duration-input");
  const elWeatherStartAtInput = document.getElementById("weather-start-at-input");
  const elWeatherAutoCheckbox = document.getElementById("weather-automation-checkbox");
  const btnSetWeather = document.getElementById("btn-set-weather");
  const elWeatherScheduleList = document.getElementById("weather-schedule-list");

  const elPresetPalette = document.getElementById("preset-palette");
  const elSpawnPresetSelect = document.getElementById("spawn-preset-select");
  const elSpawnCreatorLabel = document.getElementById("spawn-creator-label");
  const elSpawnShape = document.getElementById("spawn-shape");
  const elSpawnColor = document.getElementById("spawn-color");
  const elSpawnX = document.getElementById("spawn-x");
  const elSpawnY = document.getElementById("spawn-y");
  const elSpawnLevel = document.getElementById("spawn-level");
  const elSpawnHeight = document.getElementById("spawn-height");
  const elSpawnVolume = document.getElementById("spawn-volume");
  const elSpawnWeight = document.getElementById("spawn-weight");
  const elSpawnRigidity = document.getElementById("spawn-rigidity");
  const elSpawnHollow = document.getElementById("spawn-hollow");
  const elSpawnBinding = document.getElementById("spawn-binding");
  const elSpawnFragile = document.getElementById("spawn-fragile");
  const elSpawnFixed = document.getElementById("spawn-fixed");
  const btnSpawnObject = document.getElementById("btn-spawn-object");

  const elMoveX = document.getElementById("move-x");
  const elMoveY = document.getElementById("move-y");
  const btnMoveSelected = document.getElementById("btn-move-selected");
  const btnDespawnSelected = document.getElementById("btn-despawn-selected");

  // Inbox elements
  const elInboxBadgeCount = document.getElementById("inbox-badge-count");
  const elInboxCount = document.getElementById("inbox-count");
  const elInboxPrivateNote = document.getElementById("inbox-private-note");
  const btnSaveInboxNote = document.getElementById("btn-save-inbox-note");
  const elInboxReplyText = document.getElementById("inbox-reply-text");
  const btnInboxReply = document.getElementById("btn-inbox-reply");
  const elInboxSpawnPreset = document.getElementById("inbox-spawn-preset");
  const btnInboxSpawnResponse = document.getElementById("btn-inbox-spawn-response");
  const elInboxList = document.getElementById("inbox-list");

  // Timeline elements
  const elTimelineFilters = document.getElementById("timeline-filters");
  const elTimelineList = document.getElementById("timeline-list");

  // Beliefs & Self-Model elements
  const btnTriggerReflect = document.getElementById("btn-trigger-reflect");
  const elSelfModelCurrent = document.getElementById("self-model-current");
  const elSelfModelVersions = document.getElementById("self-model-versions");
  const elBeliefCount = document.getElementById("belief-count");
  const elBeliefList = document.getElementById("belief-list");

  // Names elements
  const elNamesCount = document.getElementById("names-count");
  const elNamesList = document.getElementById("names-list");

  let lastHoverClientX = 0;
  let lastHoverClientY = 0;
  let activeTab = "world";
  let isTimeSliderFocused = false;
  let feedbackTimer = null;

  if (elTokenInput && wsClient.token) {
    elTokenInput.value = wsClient.token;
  }

  function showFeedback(text, isError = false) {
    if (!elConsoleFeedback) return;
    elConsoleFeedback.hidden = false;
    elConsoleFeedback.textContent = text;
    elConsoleFeedback.dataset.kind = isError ? "error" : "ok";
    if (feedbackTimer) clearTimeout(feedbackTimer);
    feedbackTimer = window.setTimeout(() => {
      elConsoleFeedback.hidden = true;
    }, 3500);
  }

  function isSpectator(state) {
    return Boolean(state.auth?.spectatorMode);
  }

  function switchTab(targetTab) {
    const state = store.getState();
    if (targetTab === "controls" && isSpectator(state)) {
      targetTab = "world";
    }
    activeTab = targetTab;
    for (const btn of document.querySelectorAll("[data-tab-target]")) {
      const isMatch = btn.getAttribute("data-tab-target") === targetTab;
      btn.classList.toggle("is-active", isMatch);
      btn.setAttribute("aria-selected", isMatch ? "true" : "false");
    }
    for (const pane of document.querySelectorAll("[data-tab-pane]")) {
      const isMatch = pane.getAttribute("data-tab-pane") === targetTab;
      pane.classList.toggle("is-active", isMatch);
    }
  }

  for (const tabBtn of document.querySelectorAll("[data-tab-target]")) {
    tabBtn.addEventListener("click", () => {
      switchTab(tabBtn.getAttribute("data-tab-target"));
    });
  }

  function updateAuthAndModeUI(state) {
    const spectator = isSpectator(state);
    if (elAppShell) {
      elAppShell.dataset.spectator = spectator ? "true" : "false";
    }
    document.body.classList.toggle("spectator-mode", spectator);

    for (const node of document.querySelectorAll("[data-creator-only]")) {
      if (spectator) {
        node.setAttribute("hidden", "");
      } else {
        node.removeAttribute("hidden");
      }
    }

    if (elAuthModeBadge) {
      elAuthModeBadge.dataset.mode = spectator ? "spectator" : "creator";
      elAuthModeBadge.textContent = spectator ? "Spectator Mode" : "Creator Mode";
    }
    if (btnSpectatorToggle) {
      btnSpectatorToggle.textContent = spectator ? "Exit Spectator" : "Spectator Mode";
    }
    if (elAuthStatusMsg) {
      if (state.auth?.lastAuthError) {
        elAuthStatusMsg.textContent = state.auth.lastAuthError;
      } else if (spectator) {
        elAuthStatusMsg.textContent = "Controls hidden";
      } else if (state.auth?.token) {
        elAuthStatusMsg.textContent = "Token stored in sessionStorage";
      } else {
        elAuthStatusMsg.textContent = "";
      }
    }

    if (spectator && activeTab === "controls") {
      switchTab("world");
    }
  }

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
    updateAuthAndModeUI(state);

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

    // Sync controls state
    if (btnPauseResume) {
      btnPauseResume.textContent = state.paused ? "Resume" : "Pause";
    }
    if (elTimeSlider && !isTimeSliderFocused) {
      const tod = Number(state.time?.time_of_day ?? 0.5);
      elTimeSlider.value = tod.toFixed(2);
      if (elTimeSliderLabel) {
        elTimeSliderLabel.textContent = `${tod.toFixed(2)} (${formatTimeOfDay(tod)})`;
      }
    }
    if (elFreezeTimeCheckbox && !isTimeSliderFocused) {
      elFreezeTimeCheckbox.checked = Boolean(state.time?.frozen);
    }
    if (elWeatherAutoCheckbox) {
      elWeatherAutoCheckbox.checked = !Boolean(state.weather?.paused);
    }
    if (elWeatherScheduleList) {
      const sched = Array.isArray(state.weather?.scheduled) ? state.weather.scheduled : [];
      elWeatherScheduleList.textContent =
        sched.length > 0
          ? `Scheduled: ${sched.map((s) => `${s.type} @ tick ${s.start_tick} (${s.duration}t)`).join(", ")}`
          : "No scheduled weather events queued.";
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

    if (elMoveX && document.activeElement !== elMoveX) elMoveX.value = formatNum(pos.x, 1);
    if (elMoveY && document.activeElement !== elMoveY) elMoveY.value = formatNum(pos.y, 1);

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
    const spectator = isSpectator(state);

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

      ${
        !spectator
          ? `<div class="control-row creator-only" data-creator-only="true">
               <button type="button" class="ctrl-btn ctrl-btn-danger" data-despawn-id="${escapeHtml(obj.id)}">
                 Despawn ${escapeHtml(obj.id)}
               </button>
             </div>`
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
    for (const btn of elInspectorBody.querySelectorAll("[data-despawn-id]")) {
      btn.addEventListener("click", async () => {
        const oid = btn.getAttribute("data-despawn-id");
        if (!oid) return;
        const res = await wsClient.apiRequest("POST", "/despawn", { object_id: oid });
        if (res.ok) {
          showFeedback(`Despawned ${oid}`);
        } else {
          showFeedback(res.data?.detail || "Failed to despawn object", true);
        }
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

  // Preset palette rendering and form population
  function fillSpawnFormFromPreset(preset) {
    if (!preset) return;
    if (elSpawnPresetSelect) elSpawnPresetSelect.value = preset.key || "";
    if (elSpawnCreatorLabel) elSpawnCreatorLabel.value = preset.creator_label || preset.key || "";
    if (elSpawnShape) elSpawnShape.value = preset.shape || "sphere";
    if (elSpawnColor) elSpawnColor.value = preset.color || "red";
    if (elSpawnHeight) elSpawnHeight.value = String(preset.height ?? 0.25);
    if (elSpawnVolume) elSpawnVolume.value = String(preset.volume ?? 0.01);
    if (elSpawnWeight) elSpawnWeight.value = String(preset.weight ?? 0.4);
    if (elSpawnRigidity) elSpawnRigidity.value = String(preset.rigidity ?? 0.6);
    if (elSpawnHollow) elSpawnHollow.checked = Boolean(preset.hollow);
    if (elSpawnBinding) elSpawnBinding.checked = Boolean(preset.binding);
    if (elSpawnFragile) elSpawnFragile.checked = Boolean(preset.fragile);
    if (elSpawnFixed) elSpawnFixed.checked = Boolean(preset.fixed);
  }

  function renderPresetPalette(state) {
    if (!elPresetPalette) return;
    const presets = Array.isArray(state.presets) && state.presets.length > 0
      ? state.presets
      : FIRST_RUN_PRESETS;
    elPresetPalette.innerHTML = presets
      .map(
        (p) => `
          <button
            type="button"
            class="preset-chip"
            data-preset-key="${escapeHtml(p.key)}"
            title="${escapeHtml(p.title || p.key)} (${escapeHtml(p.shape)}, ${escapeHtml(p.color)})"
          >
            ${escapeHtml(p.key)}
          </button>
        `
      )
      .join("");

    for (const chip of elPresetPalette.querySelectorAll("[data-preset-key]")) {
      chip.addEventListener("click", () => {
        const key = chip.getAttribute("data-preset-key");
        const found = presets.find((p) => p.key === key);
        if (found) fillSpawnFormFromPreset(found);
      });
    }
  }

  // Inbox Panel rendering
  function updateInboxPanel(state) {
    const messages = state.inbox?.messages || [];
    if (elInboxBadgeCount) elInboxBadgeCount.textContent = String(messages.length);
    if (elInboxCount) elInboxCount.textContent = String(messages.length);
    if (
      elInboxPrivateNote &&
      document.activeElement !== elInboxPrivateNote &&
      state.inbox?.creator_note !== null &&
      state.inbox?.creator_note !== undefined
    ) {
      elInboxPrivateNote.value = state.inbox.creator_note;
    }
    if (!elInboxList) return;

    if (messages.length === 0) {
      elInboxList.innerHTML = `
        <div class="empty-card">
          No messages in the creator inbox yet. When the agent places an item in the mailbox or speaks at the closed mailbox, it will appear here.
        </div>
      `;
      return;
    }

    const spectator = isSpectator(state);
    elInboxList.innerHTML = [...messages]
      .reverse()
      .map((entry) => {
        const objInfo = entry.object_data
          ? `${entry.object_id || entry.object_data.id} (${entry.object_data.creator_label || entry.object_data.shape}, ${entry.object_data.color || "neutral"})`
          : entry.object_id || null;
        return `
          <article class="stack-card" data-inbox-entry-id="${escapeHtml(entry.id)}">
            <div class="stack-card-head">
              <span class="badge">#${escapeHtml(entry.id)} &middot; tick ${escapeHtml(entry.tick)}</span>
              <span class="badge">${escapeHtml(entry.reason || "delivery")}</span>
            </div>
            ${
              entry.message
                ? `<div class="inbox-speech">&ldquo;${escapeHtml(entry.message)}&rdquo;</div>`
                : ""
            }
            ${
              objInfo
                ? `<div class="inbox-object">Preserved object: <code>${escapeHtml(objInfo)}</code></div>`
                : ""
            }
            ${
              entry.note
                ? `<div class="inbox-note-display">Private note: ${escapeHtml(entry.note)}</div>`
                : ""
            }
            ${
              !spectator
                ? `<div class="inbox-entry-actions creator-only" data-creator-only="true">
                     <div class="control-row">
                       <input
                         type="text"
                         class="console-input"
                         data-entry-note-input="${escapeHtml(entry.id)}"
                         value="${escapeHtml(entry.note || "")}"
                         placeholder="Private note for #${escapeHtml(entry.id)} (silent)…"
                       />
                       <button
                         type="button"
                         class="ctrl-btn"
                         data-save-entry-note="${escapeHtml(entry.id)}"
                       >
                         Save Note
                       </button>
                       ${
                         entry.object_id
                           ? `<button
                                type="button"
                                class="ctrl-btn"
                                data-return-entry-object="${escapeHtml(entry.id)}"
                              >
                                Return ${escapeHtml(entry.object_id)}
                              </button>`
                           : ""
                       }
                     </div>
                   </div>`
                : ""
            }
          </article>
        `;
      })
      .join("");

    for (const btn of elInboxList.querySelectorAll("[data-save-entry-note]")) {
      btn.addEventListener("click", async () => {
        const entryId = Number(btn.getAttribute("data-save-entry-note"));
        const input = elInboxList.querySelector(`[data-entry-note-input="${entryId}"]`);
        const noteVal = input ? input.value : "";
        const res = await wsClient.apiRequest("POST", `/inbox/${entryId}/note`, {
          entry_id: entryId,
          note: noteVal,
        });
        if (res.ok && res.data?.inbox) {
          store.setInboxPayload(res.data.inbox);
          showFeedback(`Saved private note on #${entryId} (no agent notification)`);
        }
      });
    }

    for (const btn of elInboxList.querySelectorAll("[data-return-entry-object]")) {
      btn.addEventListener("click", async () => {
        const entryId = Number(btn.getAttribute("data-return-entry-object"));
        const res = await wsClient.apiRequest("POST", "/inbox/spawn-in-response", {
          entry_id: entryId,
          return_preserved_object: true,
        });
        if (res.ok) {
          if (res.data?.inbox) store.setInboxPayload(res.data.inbox);
          showFeedback(`Returned preserved object from #${entryId} into mailbox`);
        } else {
          showFeedback(res.data?.detail || "Failed to return object", true);
        }
      });
    }
  }

  // Timeline Panel rendering
  function updateTimelinePanel(state) {
    if (!elTimelineList) return;
    if (elTimelineFilters) {
      for (const chip of elTimelineFilters.querySelectorAll("[data-timeline-filter]")) {
        const isMatch = chip.getAttribute("data-timeline-filter") === (state.timelineFilter || "all");
        chip.classList.toggle("is-active", isMatch);
      }
    }

    const filtered = filterTimelineEntries(state.timeline || [], state.timelineFilter || "all");
    if (filtered.length === 0) {
      elTimelineList.innerHTML = `
        <div class="empty-card">
          No timeline entries matching filter <code>${escapeHtml(state.timelineFilter || "all")}</code>.
        </div>
      `;
      return;
    }

    elTimelineList.innerHTML = [...filtered]
      .reverse()
      .slice(0, 120)
      .map((item) => {
        const kind = String(item.kind || "action");
        const reasons = Array.isArray(item.details?.reasons)
          ? item.details.reasons
              .map((r) => (typeof r === "object" ? r.reason || r.kind : String(r)))
              .join(", ")
          : "";
        return `
          <div class="stack-card timeline-card" data-timeline-kind="${escapeHtml(kind)}">
            <div class="stack-card-head">
              <span class="badge badge-kind-${escapeHtml(kind)}">${escapeHtml(kind)}</span>
              <span class="mono-val">tick ${escapeHtml(item.tick)}</span>
            </div>
            <div class="timeline-summary">${escapeHtml(item.summary)}</div>
            ${
              reasons
                ? `<div class="muted-note">Wake reasons: ${escapeHtml(reasons)}</div>`
                : ""
            }
          </div>
        `;
      })
      .join("");
  }

  // Beliefs & Self-Model Panel rendering
  function updateBeliefsPanel(state) {
    const sm = state.selfModel || {};
    if (elSelfModelCurrent) {
      if (sm.text) {
        elSelfModelCurrent.innerHTML = `
          <div class="stack-card-head">
            <span class="badge">v${escapeHtml(sm.version)} &middot; tick ${escapeHtml(sm.updated_tick)}</span>
            <span class="mono-val">${escapeHtml(sm.word_count || 0)}/${escapeHtml(sm.max_words || 80)} words</span>
          </div>
          <p class="self-model-quote">${escapeHtml(sm.text)}</p>
        `;
      } else {
        elSelfModelCurrent.innerHTML = `<em>No self-model recorded yet.</em>`;
      }
    }

    if (elSelfModelVersions) {
      const history = Array.isArray(sm.history) ? sm.history : [];
      if (history.length > 0) {
        elSelfModelVersions.innerHTML = `
          <div class="inspector-subhead">Self-Model Version History (${history.length})</div>
          ${[...history]
            .reverse()
            .map(
              (ver) => `
                <div class="version-row">
                  <span class="badge">v${escapeHtml(ver.version)} (t${escapeHtml(ver.tick)})</span>
                  <span>${escapeHtml(ver.text)}</span>
                </div>
              `
            )
            .join("")}
        `;
      } else {
        elSelfModelVersions.innerHTML = "";
      }
    }

    const beliefs = state.beliefs?.items || [];
    if (elBeliefCount) elBeliefCount.textContent = String(beliefs.length);
    if (!elBeliefList) return;

    if (beliefs.length === 0) {
      elBeliefList.innerHTML = `
        <div class="empty-card">
          No beliefs stored yet. Beliefs are created and updated during reflection cycles.
        </div>
      `;
      return;
    }

    elBeliefList.innerHTML = beliefs
      .map((b) => {
        const confPct = Math.round(Math.max(0, Math.min(1, Number(b.confidence ?? 0.5))) * 100);
        const tags = Array.isArray(b.tags) ? b.tags : [];
        const chartSvg = renderBeliefHistoryChart(b.history);
        return `
          <article class="stack-card belief-card" data-belief-id="${escapeHtml(b.id)}">
            <div class="stack-card-head">
              <span class="badge">${escapeHtml(b.id)} &middot; ${confPct}%</span>
              <span class="mono-val">updated t${escapeHtml(b.updated_tick)}</span>
            </div>
            <div class="belief-statement">${escapeHtml(b.statement)}</div>
            <div class="belief-chart-row">
              <div class="confidence-bar-wrap" title="Confidence ${confPct}%">
                <div class="confidence-bar-fill" style="width: ${confPct}%"></div>
              </div>
              ${chartSvg}
            </div>
            ${
              tags.length > 0
                ? `<div class="flag-row">${tags.map((t) => `<span class="badge">${escapeHtml(t)}</span>`).join("")}</div>`
                : ""
            }
          </article>
        `;
      })
      .join("");
  }

  // Names Panel rendering
  function updateNamesPanel(state) {
    if (!elNamesList) return;
    const byObject = state.names?.by_object || {};
    const entries = Object.values(byObject);
    if (elNamesCount) elNamesCount.textContent = String(entries.length);

    if (entries.length === 0) {
      elNamesList.innerHTML = `
        <div class="empty-card">
          No object names extracted yet. As the agent references <code>object_&lt;n&gt;</code> in <code>speak</code>, thoughts, or beliefs, its words are mapped here.
        </div>
      `;
      return;
    }

    elNamesList.innerHTML = entries
      .map((entry) => {
        const oid = entry.object_id;
        const words = Array.isArray(entry.words) ? entry.words : [];
        const sources = Array.isArray(entry.sources) ? entry.sources : [];
        const creatorLabel = entry.creator_label || getObjectLabel(state, oid);
        return `
          <article class="stack-card name-card" data-name-object-id="${escapeHtml(oid)}">
            <div class="stack-card-head">
              <button type="button" class="related-item-btn" data-select-object="${escapeHtml(oid)}">
                <code>${escapeHtml(oid)}</code>
              </button>
              <span class="mono-val">
                creator_label: ${creatorLabel ? escapeHtml(creatorLabel) : "—"}
              </span>
            </div>
            <div class="name-words-row">
              ${words
                .map((w) => `<span class="name-word-pill">&ldquo;${escapeHtml(w)}&rdquo;</span>`)
                .join(" ")}
            </div>
            <div class="flag-row">
              ${sources.map((s) => `<span class="badge">${escapeHtml(s)}</span>`).join("")}
              <span class="muted-note">${escapeHtml(entry.mentions || 1)} mention(s)</span>
            </div>
          </article>
        `;
      })
      .join("");

    for (const btn of elNamesList.querySelectorAll("[data-select-object]")) {
      btn.addEventListener("click", () => {
        const oid = btn.getAttribute("data-select-object");
        store.setSelectedObjectId(oid);
        switchTab("world");
      });
    }
  }

  // Subscribe UI and renderer to store updates
  store.subscribe((state, reason) => {
    renderer.renderWorld();
    updateHud(state);
    if (reason === "hover") {
      updateTooltip(state);
      return;
    }
    if (reason === "selection") {
      updateInspector(state);
      updateObjectList(state);
      return;
    }
    if (reason === "presets") {
      renderPresetPalette(state);
      return;
    }
    updateTooltip(state);
    updateInspector(state);
    updateObjectList(state);
    updateInboxPanel(state);
    updateTimelinePanel(state);
    updateBeliefsPanel(state);
    updateNamesPanel(state);
  });

  // Wire up Login & Spectator Mode controls
  if (elLoginForm) {
    elLoginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const tokenVal = elTokenInput ? elTokenInput.value : "";
      const res = await wsClient.loginWithToken(tokenVal);
      if (res.ok) {
        showFeedback("Authenticated with creator token (stored in sessionStorage)");
      } else {
        showFeedback(res.error || "Invalid creator token", true);
      }
    });
  }

  if (btnSpectatorToggle) {
    btnSpectatorToggle.addEventListener("click", () => {
      const nextSpectator = !isSpectator(store.getState());
      wsClient.setSpectatorMode(nextSpectator);
      showFeedback(nextSpectator ? "Entered Spectator Mode (controls hidden)" : "Entered Creator Mode");
    });
  }

  // Wire up Clock Controls
  if (btnPauseResume) {
    btnPauseResume.addEventListener("click", async () => {
      const nextPaused = !store.getState().paused;
      const res = await wsClient.apiRequest("POST", "/pause", { paused: nextPaused });
      if (res.ok) {
        showFeedback(nextPaused ? "Simulation paused" : "Simulation resumed");
      }
    });
  }

  if (btnStep) {
    btnStep.addEventListener("click", async () => {
      const ticks = Math.max(1, parseInt(elStepTicksInput?.value || "1", 10) || 1);
      const res = await wsClient.apiRequest("POST", "/step", { ticks });
      if (res.ok) {
        showFeedback(`Stepped ${ticks} tick(s)`);
      }
    });
  }

  if (btnSetSpeed) {
    btnSetSpeed.addEventListener("click", async () => {
      const speed = Math.max(0.05, parseFloat(elSpeedSelect?.value || "1.0") || 1.0);
      const res = await wsClient.apiRequest("POST", "/speed", { speed });
      if (res.ok) {
        showFeedback(`Simulation speed set to ${speed}x`);
      }
    });
  }

  // Wire up Time-of-Day Controls
  if (elTimeSlider) {
    elTimeSlider.addEventListener("focus", () => {
      isTimeSliderFocused = true;
    });
    elTimeSlider.addEventListener("blur", () => {
      isTimeSliderFocused = false;
    });
    elTimeSlider.addEventListener("input", () => {
      const tod = parseFloat(elTimeSlider.value) || 0;
      if (elTimeSliderLabel) {
        elTimeSliderLabel.textContent = `${tod.toFixed(2)} (${formatTimeOfDay(tod)})`;
      }
    });
    elTimeSlider.addEventListener("change", async () => {
      const tod = parseFloat(elTimeSlider.value) || 0;
      const frozen = Boolean(elFreezeTimeCheckbox?.checked);
      await wsClient.apiRequest("POST", "/time", { time_of_day: tod, frozen });
    });
  }

  if (elFreezeTimeCheckbox) {
    elFreezeTimeCheckbox.addEventListener("change", async () => {
      const frozen = Boolean(elFreezeTimeCheckbox.checked);
      const tod = parseFloat(elTimeSlider?.value || String(store.getState().time?.time_of_day ?? 0.5));
      const res = await wsClient.apiRequest("POST", "/time", { time_of_day: tod, frozen });
      if (res.ok) {
        showFeedback(frozen ? "Day/night cycle frozen" : "Day/night cycle unfrozen");
      }
    });
  }

  if (btnApplyTime) {
    btnApplyTime.addEventListener("click", async () => {
      const tod = parseFloat(elTimeSlider?.value || "0.5");
      const frozen = Boolean(elFreezeTimeCheckbox?.checked);
      const res = await wsClient.apiRequest("POST", "/time", { time_of_day: tod, frozen });
      if (res.ok) {
        showFeedback(`Time set to ${formatTimeOfDay(tod)}${frozen ? " [frozen]" : ""}`);
      }
    });
  }

  // Wire up Weather Controls
  if (elWeatherIntensitySlider && elWeatherIntensityVal) {
    elWeatherIntensitySlider.addEventListener("input", () => {
      elWeatherIntensityVal.textContent = Number(elWeatherIntensitySlider.value).toFixed(2);
    });
  }

  if (elWeatherAutoCheckbox) {
    elWeatherAutoCheckbox.addEventListener("change", async () => {
      const enabled = Boolean(elWeatherAutoCheckbox.checked);
      const res = await wsClient.apiRequest("POST", "/weather", { paused: !enabled });
      if (res.ok && res.data?.weather) {
        store.setWeatherPayload(res.data.weather);
        showFeedback(enabled ? "Weather automation enabled" : "Weather automation paused");
      }
    });
  }

  if (btnSetWeather) {
    btnSetWeather.addEventListener("click", async () => {
      const type = elWeatherTypeSelect?.value || "clear";
      const intensity = parseFloat(elWeatherIntensitySlider?.value || "0.6");
      const duration = Math.max(1, parseInt(elWeatherDurationInput?.value || "120", 10) || 120);
      const startAtRaw = elWeatherStartAtInput?.value?.trim() || "";
      const payload = {
        type,
        intensity,
        duration,
        paused: !Boolean(elWeatherAutoCheckbox?.checked),
      };
      if (startAtRaw !== "") {
        payload.start_at = Math.max(0, parseInt(startAtRaw, 10) || 0);
      }
      const res = await wsClient.apiRequest("POST", "/weather", payload);
      if (res.ok) {
        if (res.data?.weather) store.setWeatherPayload(res.data.weather);
        showFeedback(
          payload.start_at !== undefined
            ? `Scheduled ${type} weather for tick ${payload.start_at}`
            : `Weather set to ${type} (${Math.round(intensity * 100)}%)`
        );
      }
    });
  }

  // Wire up Spawn Preset & Property Form
  if (elSpawnPresetSelect) {
    elSpawnPresetSelect.addEventListener("change", () => {
      const key = elSpawnPresetSelect.value;
      const presets = store.getState().presets || FIRST_RUN_PRESETS;
      const found = presets.find((p) => p.key === key);
      if (found) fillSpawnFormFromPreset(found);
    });
  }

  if (btnSpawnObject) {
    btnSpawnObject.addEventListener("click", async () => {
      const payload = {
        preset: elSpawnPresetSelect?.value || undefined,
        creator_label: elSpawnCreatorLabel?.value?.trim() || undefined,
        shape: elSpawnShape?.value || "sphere",
        color: elSpawnColor?.value?.trim() || "red",
        x: parseFloat(elSpawnX?.value || "7.0") || 7.0,
        y: parseFloat(elSpawnY?.value || "7.0") || 7.0,
        level: elSpawnLevel?.value?.trim() || "ground",
        height: parseFloat(elSpawnHeight?.value || "0.25") || 0.25,
        volume: parseFloat(elSpawnVolume?.value || "0.01") || 0.01,
        weight: parseFloat(elSpawnWeight?.value || "0.4") || 0.4,
        rigidity: parseFloat(elSpawnRigidity?.value || "0.6") || 0.6,
        hollow: Boolean(elSpawnHollow?.checked),
        binding: Boolean(elSpawnBinding?.checked),
        fragile: Boolean(elSpawnFragile?.checked),
        fixed: Boolean(elSpawnFixed?.checked),
      };
      const res = await wsClient.apiRequest("POST", "/spawn", payload);
      if (res.ok && res.data?.object) {
        store.setSelectedObjectId(res.data.object.id);
        showFeedback(`Spawned ${res.data.object.id}`);
      } else {
        showFeedback(res.data?.detail || "Failed to spawn object", true);
      }
    });
  }

  if (btnMoveSelected) {
    btnMoveSelected.addEventListener("click", async () => {
      const selectedId = store.getState().selectedObjectId;
      if (!selectedId) {
        showFeedback("Select an object first to move it", true);
        return;
      }
      const x = parseFloat(elMoveX?.value || "7.0") || 7.0;
      const y = parseFloat(elMoveY?.value || "7.0") || 7.0;
      const res = await wsClient.apiRequest("POST", "/move", {
        object_id: selectedId,
        x,
        y,
      });
      if (res.ok) {
        showFeedback(`Moved ${selectedId} to (${x.toFixed(1)}, ${y.toFixed(1)})`);
      } else {
        showFeedback(res.data?.detail || "Failed to move object", true);
      }
    });
  }

  if (btnDespawnSelected) {
    btnDespawnSelected.addEventListener("click", async () => {
      const selectedId = store.getState().selectedObjectId;
      if (!selectedId) {
        showFeedback("Select an object first to despawn it", true);
        return;
      }
      const res = await wsClient.apiRequest("POST", "/despawn", { object_id: selectedId });
      if (res.ok) {
        showFeedback(`Despawned ${selectedId}`);
      } else {
        showFeedback(res.data?.detail || "Failed to despawn object", true);
      }
    });
  }

  // Wire up Inbox Actions
  if (btnSaveInboxNote) {
    btnSaveInboxNote.addEventListener("click", async () => {
      const note = elInboxPrivateNote?.value || "";
      const res = await wsClient.apiRequest("POST", "/inbox/note", { note });
      if (res.ok && res.data?.inbox) {
        store.setInboxPayload(res.data.inbox);
        showFeedback("Saved private note (no acknowledgement sent to agent)");
      }
    });
  }

  if (btnInboxReply) {
    btnInboxReply.addEventListener("click", async () => {
      const text = elInboxReplyText?.value?.trim() || "";
      if (!text) {
        showFeedback("Enter reply text first", true);
        return;
      }
      const res = await wsClient.apiRequest("POST", "/inbox/reply", { text });
      if (res.ok) {
        if (elInboxReplyText) elInboxReplyText.value = "";
        if (res.data?.inbox) store.setInboxPayload(res.data.inbox);
        showFeedback(`Placed reply letter (${res.data?.object?.id || "letter"}) in mailbox`);
      } else {
        showFeedback(res.data?.detail || "Failed to send reply", true);
      }
    });
  }

  if (btnInboxSpawnResponse) {
    btnInboxSpawnResponse.addEventListener("click", async () => {
      const presetKey = elInboxSpawnPreset?.value || "berry";
      const res = await wsClient.apiRequest("POST", "/inbox/spawn-in-response", {
        object: { preset: presetKey },
      });
      if (res.ok) {
        if (res.data?.inbox) store.setInboxPayload(res.data.inbox);
        showFeedback(`Spawned ${presetKey} (${res.data?.object?.id || "item"}) inside mailbox`);
      } else {
        showFeedback(res.data?.detail || "Failed to spawn in mailbox", true);
      }
    });
  }

  // Wire up Timeline Filters
  if (elTimelineFilters) {
    for (const chip of elTimelineFilters.querySelectorAll("[data-timeline-filter]")) {
      chip.addEventListener("click", () => {
        store.setTimelineFilter(chip.getAttribute("data-timeline-filter") || "all");
      });
    }
  }

  // Wire up Reflection button
  if (btnTriggerReflect) {
    btnTriggerReflect.addEventListener("click", async () => {
      const res = await wsClient.apiRequest("POST", "/reflect", {});
      if (res.ok) {
        await wsClient.refreshObserverPanels();
        showFeedback("Reflection cycle completed");
      }
    });
  }

  // Pointer interactions:
  // - In Creator mode: dragging a hit-tested world object moves it across the world (`POST /move` on pointerup).
  // - Dragging on empty ground (or any drag in Spectator mode): pans the viewport.
  let isDragging = false;
  let dragMoved = false;
  let draggingObjectId = null;
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
    draggingObjectId = null;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;

    // Check if pointerdown is on a movable world object while in Creator mode
    const state = store.getState();
    if (event.button === 0 && !isSpectator(state)) {
      const local = getCanvasLocalCoords(event);
      const worldPt = renderer.screenToWorld(local.x, local.y);
      const hit = hitTestWorld(state, worldPt.x, worldPt.y);
      if (hit && hit.kind === "object") {
        draggingObjectId = hit.id;
        store.setSelectedObjectId(hit.id);
      }
    }

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
        if (draggingObjectId && !isSpectator(store.getState())) {
          const local = getCanvasLocalCoords(event);
          const worldPt = renderer.screenToWorld(local.x, local.y);
          const state = store.getState();
          const maxW = (state.terrain?.width || 16) * (state.terrain?.tile_size || 1);
          const maxH = (state.terrain?.height || 16) * (state.terrain?.tile_size || 1);
          const clampedX = Math.max(0.2, Math.min(maxW - 0.2, worldPt.x));
          const clampedY = Math.max(0.2, Math.min(maxH - 0.2, worldPt.y));
          if (state.objects[draggingObjectId]) {
            state.objects[draggingObjectId].position = { x: clampedX, y: clampedY };
            state.objects[draggingObjectId].level = "ground";
            renderer.renderWorld();
            updateInspector(state);
            updateObjectList(state);
          }
          return;
        }
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

  canvas.addEventListener("pointerup", async (event) => {
    if (!isDragging) return;
    isDragging = false;
    const movedObjId = draggingObjectId;
    draggingObjectId = null;

    if (dragMoved && movedObjId && !isSpectator(store.getState())) {
      const obj = store.getState().objects?.[movedObjId];
      if (obj && obj.position) {
        const x = Number(obj.position.x.toFixed(2));
        const y = Number(obj.position.y.toFixed(2));
        const res = await wsClient.apiRequest("POST", "/move", {
          object_id: movedObjId,
          x,
          y,
          level: "ground",
        });
        if (res.ok) {
          showFeedback(`Moved ${movedObjId} to (${x.toFixed(1)}, ${y.toFixed(1)})`);
        }
      }
      return;
    }

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
    draggingObjectId = null;
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

  renderPresetPalette(store.getState());
  updateHud(store.getState());
  updateInspector(store.getState());
  updateObjectList(store.getState());
  updateInboxPanel(store.getState());
  updateTimelinePanel(store.getState());
  updateBeliefsPanel(store.getState());
  updateNamesPanel(store.getState());

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
  extractNamesFromText,
  filterTimelineEntries,
  getObjectLabel,
  hitTestWorld,
};
