/**
 * World state store and incremental WebSocket delta reducer for the Arden browser client.
 */

export const DEFAULT_WORLD_WIDTH = 16;
export const DEFAULT_WORLD_HEIGHT = 16;
export const DEFAULT_TILE_SIZE = 1.0;

/**
 * Compute approximate light level from time_of_day fraction [0..1) when explicit
 * light_level is not supplied. Matches DayNightCycle curve (0=midnight, 0.5=noon).
 */
export function computeLightFromTime(timeOfDay = 0.5) {
  const t = ((Number(timeOfDay) % 1) + 1) % 1;
  return Math.max(0, Math.min(1, 0.5 * (1 - Math.cos(2 * Math.PI * t))));
}

/**
 * Return human-readable day phase ("Night", "Dawn", "Day", "Dusk") for time_of_day in [0..1).
 */
export function getDayPhase(timeOfDay = 0.5) {
  const t = ((Number(timeOfDay) % 1) + 1) % 1;
  if (t >= 0.20 && t < 0.32) return "Dawn";
  if (t >= 0.32 && t < 0.68) return "Day";
  if (t >= 0.68 && t < 0.80) return "Dusk";
  return "Night";
}

/**
 * Format time_of_day fraction [0..1) as 24-hour clock string (e.g. "12:00 (Day)").
 */
export function formatTimeOfDay(timeOfDay = 0.5) {
  const t = ((Number(timeOfDay) % 1) + 1) % 1;
  const totalMinutes = Math.round(t * 24 * 60) % (24 * 60);
  const hours = Math.floor(totalMinutes / 60);
  const mins = totalMinutes % 60;
  const hh = String(hours).padStart(2, "0");
  const mm = String(mins).padStart(2, "0");
  return `${hh}:${mm} (${getDayPhase(t)})`;
}

/**
 * Compute an object's physical collision/display radius in meters from shape, volume, and height,
 * matching `object_radius` in `arden.world.objects`.
 */
export function computeObjectRadiusMeters(obj) {
  if (!obj || typeof obj !== "object") return 0.2;
  if (obj.state && typeof obj.state.radius === "number" && obj.state.radius > 0) {
    return Math.max(0.08, Math.min(2.0, obj.state.radius));
  }
  const volume = Math.max(0, Number(obj.volume) || 0);
  const height = Math.max(0.01, Number(obj.height) || 0.1);
  const shape = String(obj.shape || "sphere");
  let raw = 0.18;
  if (shape === "sphere") {
    raw = Math.cbrt((3 * volume) / (4 * Math.PI));
  } else if (shape === "cylinder") {
    raw = Math.sqrt(volume / (Math.PI * height));
  } else if (shape === "cube" || shape === "flat") {
    raw = 0.5 * Math.sqrt(volume / height);
  } else if (shape === "long") {
    raw = 0.5 * Math.max(height * 0.5, Math.sqrt(volume / height));
  }
  return Math.max(0.12, Math.min(1.5, raw || 0.18));
}

/**
 * Compute base elevation (in meters) for an object or body given its level string.
 */
export function computeObjectElevation(entity, objectsMap = {}) {
  if (!entity || typeof entity !== "object") return 0.0;
  const level = String(entity.level || "ground");
  if (level === "ground") return 0.0;
  if (level === "raised") return 1.0;
  if (level.startsWith("on_top_of:")) {
    const supportId = level.slice("on_top_of:".length);
    const support = objectsMap[supportId];
    if (!support) return 0.0;
    return computeObjectElevation(support, objectsMap) + (Number(support.height) || 0);
  }
  if (level.startsWith("inside:")) {
    const containerId = level.slice("inside:".length);
    const container = objectsMap[containerId];
    if (!container) return 0.0;
    return computeObjectElevation(container, objectsMap);
  }
  return 0.0;
}

/**
 * Resolve the effective world (x, y) coordinates of an object, following `on_top_of:` or `inside:`
 * references if needed.
 */
export function resolveObjectPosition(obj, objectsMap = {}) {
  if (!obj || typeof obj !== "object") return { x: 0, y: 0 };
  const level = String(obj.level || "ground");
  if (level.startsWith("inside:")) {
    const containerId = level.slice("inside:".length);
    const container = objectsMap[containerId];
    if (container && container.position) {
      return {
        x: Number(container.position.x) || 0,
        y: Number(container.position.y) || 0,
      };
    }
  }
  const pos = obj.position || { x: 0, y: 0 };
  return {
    x: Number(pos.x) || 0,
    y: Number(pos.y) || 0,
  };
}

/**
 * Look up the human-only `creator_label` for an object id.
 */
export function getObjectLabel(state, objOrId) {
  if (!objOrId) return null;
  const obj = typeof objOrId === "string" ? state.objects?.[objOrId] : objOrId;
  const id = typeof objOrId === "string" ? objOrId : objOrId.id;
  if (obj && typeof obj.creator_label === "string" && obj.creator_label.trim()) {
    return obj.creator_label.trim();
  }
  const fromMap = id && state.creator_labels ? state.creator_labels[id] : null;
  if (typeof fromMap === "string" && fromMap.trim()) {
    return fromMap.trim();
  }
  return null;
}

/**
 * Return true if an object is inside a container whose lid is closed.
 */
export function isHiddenInsideClosedContainer(obj, objectsMap = {}) {
  if (!obj || typeof obj.level !== "string") return false;
  if (!obj.level.startsWith("inside:")) return false;
  const containerId = obj.level.slice("inside:".length);
  const container = objectsMap[containerId];
  if (!container) return false;
  return container.state?.lid === "closed";
}

/**
 * Create a new initial world state object.
 */
export function createInitialState() {
  return {
    schema_version: 1,
    tick: 0,
    paused: false,
    speed: 1.0,
    tick_rate: 10.0,
    connectionStatus: "connecting",
    reconnectAttempt: 0,
    deltaCount: 0,
    lastEvent: null,
    terrain: {
      width: DEFAULT_WORLD_WIDTH,
      height: DEFAULT_WORLD_HEIGHT,
      tile_size: DEFAULT_TILE_SIZE,
      default_type: "grass",
      tiles: {},
    },
    objects: {},
    creator_labels: {},
    body: {
      position: { x: 6.5, y: 6.5 },
      level: "ground",
      energy: 1.0,
      comfort: 1.0,
      holding: null,
      wet: false,
      wetness: 0.0,
      eye_height: 0.5,
    },
    time: {
      time_of_day: 0.5,
      frozen: false,
      light_level: 1.0,
      ambient_temperature: 1.0,
      day_length: 600,
    },
    weather: {
      type: "clear",
      intensity: 0.0,
      duration: 0,
      remaining_ticks: 0,
      paused: false,
      wind_direction: { x: 1.0, y: 0.0 },
    },
    agent: {
      goal: null,
      thought: null,
      intent: null,
      status: "idle",
      drives: { boredom: 0.0, curiosity: 0.0 },
      lastWake: null,
      lastSpoke: null,
    },
    selectedObjectId: null,
    hoveredTarget: null,
  };
}

/**
 * Apply a full world snapshot payload onto `state` in place and return `state`.
 */
export function applySnapshot(state, snapshot) {
  if (!snapshot || typeof snapshot !== "object") return state;

  if (typeof snapshot.tick === "number") state.tick = snapshot.tick;
  if (typeof snapshot.paused === "boolean") state.paused = snapshot.paused;
  if (typeof snapshot.speed === "number") state.speed = snapshot.speed;
  if (typeof snapshot.tick_rate === "number") state.tick_rate = snapshot.tick_rate;

  if (snapshot.terrain && typeof snapshot.terrain === "object") {
    const t = snapshot.terrain;
    state.terrain = {
      width: Number(t.width) || DEFAULT_WORLD_WIDTH,
      height: Number(t.height) || DEFAULT_WORLD_HEIGHT,
      tile_size: Number(t.tile_size) || DEFAULT_TILE_SIZE,
      default_type: t.default_type || "grass",
      tiles: t.tiles && typeof t.tiles === "object" ? { ...t.tiles } : {},
    };
  }

  const nextCreatorLabels = {
    ...(state.creator_labels || {}),
    ...(snapshot.creator_labels || {}),
  };

  if (snapshot.objects && typeof snapshot.objects === "object") {
    const nextObjects = {};
    for (const [oid, rawObj] of Object.entries(snapshot.objects)) {
      if (!rawObj || typeof rawObj !== "object") continue;
      const cloned = structuredClone ? structuredClone(rawObj) : JSON.parse(JSON.stringify(rawObj));
      if (cloned.creator_label) {
        nextCreatorLabels[oid] = cloned.creator_label;
      } else if (nextCreatorLabels[oid]) {
        cloned.creator_label = nextCreatorLabels[oid];
      }
      nextObjects[oid] = cloned;
    }
    state.objects = nextObjects;
  }
  state.creator_labels = nextCreatorLabels;

  if (snapshot.body && typeof snapshot.body === "object") {
    state.body = {
      ...state.body,
      ...snapshot.body,
      position: {
        x: Number(snapshot.body.position?.x ?? state.body.position.x),
        y: Number(snapshot.body.position?.y ?? state.body.position.y),
      },
    };
  }

  const dn = snapshot.daynight || {};
  const timeOfDay = Number(
    snapshot.time_of_day ?? dn.time_of_day ?? state.time.time_of_day ?? 0.5
  );
  const lightLevel = Number(
    snapshot.light_level ?? dn.light_level ?? computeLightFromTime(timeOfDay)
  );
  const ambientTemp = Number(
    snapshot.ambient_temperature ?? dn.ambient_temperature ?? state.time.ambient_temperature ?? 1.0
  );
  state.time = {
    ...state.time,
    time_of_day: timeOfDay,
    frozen: Boolean(dn.frozen ?? snapshot.frozen ?? state.time.frozen),
    light_level: lightLevel,
    ambient_temperature: ambientTemp,
    day_length: Number(dn.config?.day_length ?? state.time.day_length ?? 600),
  };

  if (snapshot.weather && typeof snapshot.weather === "object") {
    const w = snapshot.weather.current || snapshot.weather;
    state.weather = {
      ...state.weather,
      type: String(w.type ?? w.weather_type ?? state.weather.type ?? "clear"),
      intensity: Number(w.intensity ?? state.weather.intensity ?? 0.0),
      duration: Number(w.duration ?? state.weather.duration ?? 0),
      remaining_ticks: Number(w.remaining_ticks ?? state.weather.remaining_ticks ?? 0),
      paused: Boolean(snapshot.weather.paused ?? w.paused ?? state.weather.paused),
      wind_direction: {
        x: Number(w.wind_direction?.x ?? snapshot.weather.wind_direction?.x ?? 1.0),
        y: Number(w.wind_direction?.y ?? snapshot.weather.wind_direction?.y ?? 0.0),
      },
    };
  }

  return state;
}

/**
 * Incrementally apply a `world_delta` (or `creator_action` containing delta/state) onto `state`.
 */
export function applyWorldDelta(state, msg) {
  if (!msg || typeof msg !== "object") return state;

  if (typeof msg.tick === "number") {
    state.tick = msg.tick;
  }
  if (typeof msg.paused === "boolean") {
    state.paused = msg.paused;
  }
  if (typeof msg.speed === "number") {
    state.speed = msg.speed;
  }

  if (msg.body && typeof msg.body === "object") {
    state.body = {
      ...state.body,
      ...msg.body,
      position: {
        x: Number(msg.body.position?.x ?? state.body.position.x),
        y: Number(msg.body.position?.y ?? state.body.position.y),
      },
    };
  }

  if (msg.time && typeof msg.time === "object") {
    const t = msg.time;
    const tod = Number(t.time_of_day ?? t.time ?? state.time.time_of_day);
    state.time = {
      ...state.time,
      time_of_day: tod,
      frozen: Boolean(t.frozen ?? t.cycle_frozen ?? state.time.frozen),
      light_level: Number(t.light_level ?? t.light ?? computeLightFromTime(tod)),
      ambient_temperature: Number(
        t.ambient_temperature ?? t.temperature ?? state.time.ambient_temperature
      ),
    };
  }

  if (msg.weather && typeof msg.weather === "object") {
    const w = msg.weather;
    state.weather = {
      ...state.weather,
      type: String(w.type ?? w.weather_type ?? state.weather.type),
      intensity: Number(w.intensity ?? state.weather.intensity),
      paused: Boolean(w.paused ?? w.automation_paused ?? state.weather.paused),
      wind_direction: w.wind_direction
        ? {
            x: Number(w.wind_direction.x ?? state.weather.wind_direction.x),
            y: Number(w.wind_direction.y ?? state.weather.wind_direction.y),
          }
        : state.weather.wind_direction,
    };
  }

  const delta = msg.delta && typeof msg.delta === "object" ? msg.delta : msg;
  const eventType = String(delta.event || msg.event || msg.action || "");

  const upsertObject = (objData) => {
    if (!objData || typeof objData !== "object" || !objData.id) return;
    const oid = objData.id;
    const prev = state.objects[oid] || {};
    const merged = {
      ...prev,
      ...objData,
      position: objData.position
        ? { x: Number(objData.position.x), y: Number(objData.position.y) }
        : prev.position || { x: 0, y: 0 },
      state: objData.state ? { ...(prev.state || {}), ...objData.state } : prev.state || {},
    };
    if (merged.creator_label) {
      state.creator_labels[oid] = merged.creator_label;
    } else if (state.creator_labels[oid]) {
      merged.creator_label = state.creator_labels[oid];
    }
    state.objects[oid] = merged;
  };

  if (Array.isArray(delta.spawned)) {
    for (const item of delta.spawned) {
      upsertObject(item);
    }
  }

  if (Array.isArray(delta.despawned)) {
    for (const oid of delta.despawned) {
      delete state.objects[oid];
    }
  }

  if (eventType === "object_despawned" || eventType === "despawn") {
    const targetId = delta.object_id || msg.object_id;
    if (targetId) {
      delete state.objects[targetId];
    }
  } else if (
    eventType === "object_spawned" ||
    eventType === "spawn" ||
    eventType === "spawn_in_response" ||
    eventType === "reply"
  ) {
    if (delta.object) upsertObject(delta.object);
    else if (msg.object) upsertObject(msg.object);
  } else if (eventType === "object_moved" || eventType === "move") {
    const targetId = delta.object_id || msg.object_id;
    const newPos = delta.position || msg.position;
    if (delta.object) {
      upsertObject(delta.object);
    } else if (targetId && state.objects[targetId] && newPos) {
      state.objects[targetId] = {
        ...state.objects[targetId],
        position: { x: Number(newPos.x), y: Number(newPos.y) },
      };
    }
  } else if (eventType === "object_changed") {
    const targetId = delta.object_id || msg.object_id;
    if (delta.object) {
      upsertObject(delta.object);
    } else if (targetId && state.objects[targetId] && delta.changes) {
      const existing = state.objects[targetId];
      state.objects[targetId] = {
        ...existing,
        ...delta.changes,
        state: delta.changes.state
          ? { ...(existing.state || {}), ...delta.changes.state }
          : existing.state,
      };
    }
  }

  if (delta.objects && typeof delta.objects === "object") {
    if (eventType === "tick") {
      // Full tick object map syncs positions/states and removes despawned objects
      const incomingIds = new Set(Object.keys(delta.objects));
      for (const existingId of Object.keys(state.objects)) {
        if (!incomingIds.has(existingId)) {
          delete state.objects[existingId];
        }
      }
    }
    for (const objData of Object.values(delta.objects)) {
      upsertObject(objData);
    }
  }

  if (eventType === "spoke" && (delta.text || msg.text)) {
    state.agent.lastSpoke = {
      tick: state.tick,
      text: String(delta.text || msg.text),
    };
  }

  state.deltaCount += 1;
  state.lastEvent = {
    type: String(msg.type || "world_delta"),
    event: eventType || "update",
    tick: state.tick,
  };

  return state;
}

/**
 * Apply any incoming WebSocket message (snapshot, world_delta, intent, thought, wake, etc.)
 * to `state`.
 */
export function applyWsMessage(state, msg) {
  if (!msg || typeof msg !== "object") return state;
  const msgType = String(msg.type || "");

  if (msgType === "snapshot" && msg.snapshot) {
    return applySnapshot(state, msg.snapshot);
  }
  if (msgType === "world_delta" || msgType === "creator_action") {
    return applyWorldDelta(state, msg);
  }
  if (msgType === "intent") {
    if (typeof msg.tick === "number") state.tick = msg.tick;
    state.agent.intent = msg.intent || null;
    if (msg.goal !== undefined) state.agent.goal = msg.goal;
    if (msg.thought !== undefined) state.agent.thought = msg.thought;
    state.lastEvent = { type: "intent", event: "intent_set", tick: state.tick };
    return state;
  }
  if (msgType === "thought") {
    if (typeof msg.tick === "number") state.tick = msg.tick;
    if (msg.thought !== undefined || msg.text !== undefined) {
      state.agent.thought = msg.thought ?? msg.text;
    }
    if (msg.goal !== undefined) state.agent.goal = msg.goal;
    state.lastEvent = { type: "thought", event: "thought", tick: state.tick };
    return state;
  }
  if (msgType === "wake") {
    if (typeof msg.tick === "number") state.tick = msg.tick;
    state.agent.lastWake = {
      tick: state.tick,
      reasons: Array.isArray(msg.reasons) ? msg.reasons : [],
    };
    state.lastEvent = { type: "wake", event: "wake", tick: state.tick };
    return state;
  }
  if (msgType === "gate_decision") {
    if (typeof msg.tick === "number") state.tick = msg.tick;
    state.lastEvent = {
      type: "gate_decision",
      event: msg.outcome || "continue",
      tick: state.tick,
    };
    return state;
  }
  return state;
}

/**
 * Hit-test world coordinates `(wx, wy)` (in meters) against objects and the agent.
 * Returns `{ kind: "object", id, object }` or `{ kind: "agent", id: "agent", body }` or `null`.
 */
export function hitTestWorld(state, wx, wy, minHitRadiusMeters = 0.28) {
  if (!state || !Number.isFinite(wx) || !Number.isFinite(wy)) return null;

  // Check objects from highest/smallest on top to largest below so small items on top of
  // or near big items are easy to click and hover.
  const candidates = [];
  const holdingId = state.body?.holding || null;

  for (const obj of Object.values(state.objects || {})) {
    if (!obj || !obj.id) continue;
    if (obj.id === holdingId) continue;
    if (isHiddenInsideClosedContainer(obj, state.objects)) continue;

    const pos = resolveObjectPosition(obj, state.objects);
    const physRadius = computeObjectRadiusMeters(obj);
    const hitRadius = Math.max(minHitRadiusMeters, physRadius * 1.15);
    const dist = Math.hypot(wx - pos.x, wy - pos.y);
    if (dist <= hitRadius) {
      const elev = computeObjectElevation(obj, state.objects);
      candidates.push({
        obj,
        dist,
        physRadius,
        elev,
      });
    }
  }

  if (candidates.length > 0) {
    // Prefer higher elevation (e.g. stacked/inside open container), then smaller radius, then closer center
    candidates.sort((a, b) => {
      if (Math.abs(b.elev - a.elev) > 0.05) return b.elev - a.elev;
      if (Math.abs(a.physRadius - b.physRadius) > 0.05) return a.physRadius - b.physRadius;
      return a.dist - b.dist;
    });
    const best = candidates[0].obj;
    return { kind: "object", id: best.id, object: best };
  }

  // Check agent body
  if (state.body && state.body.position) {
    const ax = Number(state.body.position.x) || 0;
    const ay = Number(state.body.position.y) || 0;
    const distAgent = Math.hypot(wx - ax, wy - ay);
    if (distAgent <= Math.max(0.35, minHitRadiusMeters)) {
      return { kind: "agent", id: "agent", body: state.body };
    }
  }

  return null;
}

/**
 * Create a reactive world store with subscribe/notify support.
 */
export function createWorldStore() {
  const state = createInitialState();
  const listeners = new Set();

  const notify = (reason = "update") => {
    for (const fn of listeners) {
      try {
        fn(state, reason);
      } catch (_err) {
        // Ignore subscriber error
      }
    }
  };

  return {
    state,
    getState() {
      return state;
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    setSnapshot(snapshot) {
      applySnapshot(state, snapshot);
      notify("snapshot");
    },
    setAgentPayload(agentPayload) {
      if (!agentPayload || typeof agentPayload !== "object") return;
      if (typeof agentPayload.tick === "number") state.tick = agentPayload.tick;
      if (agentPayload.body) {
        state.body = { ...state.body, ...agentPayload.body };
      }
      if (agentPayload.current_intent !== undefined || agentPayload.intent !== undefined) {
        state.agent.intent = agentPayload.current_intent ?? agentPayload.intent;
      }
      if (agentPayload.last_thought !== undefined || agentPayload.thought !== undefined) {
        state.agent.thought = agentPayload.last_thought ?? agentPayload.thought;
      }
      if (agentPayload.goal !== undefined) {
        state.agent.goal = agentPayload.goal;
      }
      if (agentPayload.status !== undefined) {
        state.agent.status = agentPayload.status;
      }
      if (agentPayload.drives) {
        state.agent.drives = { ...state.agent.drives, ...agentPayload.drives };
      }
      notify("agent");
    },
    applyMessage(msg) {
      applyWsMessage(state, msg);
      notify(msg?.type || "delta");
    },
    setConnectionStatus(status, reconnectAttempt = 0) {
      state.connectionStatus = status;
      state.reconnectAttempt = reconnectAttempt;
      notify("connection");
    },
    setSelectedObjectId(objectId) {
      state.selectedObjectId = objectId || null;
      notify("selection");
    },
    setHoveredTarget(target) {
      state.hoveredTarget = target || null;
      notify("hover");
    },
  };
}
