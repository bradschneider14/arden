/**
 * Live WebSocket client with incremental delta dispatch, automatic exponential-backoff
 * reconnection, sessionStorage token authentication, and HTTP observer/creator API helpers.
 */

export const INITIAL_RECONNECT_DELAY_MS = 500;
export const MAX_RECONNECT_DELAY_MS = 8000;
export const TOKEN_STORAGE_KEY = "arden_creator_token";
export const SPECTATOR_STORAGE_KEY = "arden_spectator_mode";

/**
 * Extract an optional creator token from URL query params or sessionStorage (with localStorage fallback).
 */
export function resolveClientToken() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromQuery =
      params.get("token") || params.get("creator_token") || params.get("api_key");
    if (fromQuery && fromQuery.trim()) {
      const cleaned = fromQuery.trim();
      window.sessionStorage?.setItem(TOKEN_STORAGE_KEY, cleaned);
      return cleaned;
    }
    const fromSession = window.sessionStorage?.getItem(TOKEN_STORAGE_KEY);
    if (fromSession && fromSession.trim()) {
      return fromSession.trim();
    }
    const fromLocal = window.localStorage?.getItem(TOKEN_STORAGE_KEY);
    if (fromLocal && fromLocal.trim()) {
      const cleaned = fromLocal.trim();
      window.sessionStorage?.setItem(TOKEN_STORAGE_KEY, cleaned);
      return cleaned;
    }
  } catch (_err) {
    // Ignore storage/location access errors
  }
  return null;
}

/**
 * Save or clear the creator token in `sessionStorage`.
 */
export function setClientToken(token) {
  try {
    const cleaned = typeof token === "string" ? token.trim() : "";
    if (cleaned) {
      window.sessionStorage?.setItem(TOKEN_STORAGE_KEY, cleaned);
      window.sessionStorage?.setItem(SPECTATOR_STORAGE_KEY, "false");
      return cleaned;
    }
    window.sessionStorage?.removeItem(TOKEN_STORAGE_KEY);
    window.localStorage?.removeItem(TOKEN_STORAGE_KEY);
  } catch (_err) {
    // Ignore storage errors
  }
  return null;
}

/**
 * Remove the stored creator token from `sessionStorage` and `localStorage`.
 */
export function clearClientToken() {
  return setClientToken(null);
}

/**
 * Resolve whether the console is currently in spectator mode (hiding all creator controls).
 */
export function resolveSpectatorMode() {
  try {
    const params = new URLSearchParams(window.location.search);
    const mode = params.get("mode");
    const specParam = params.get("spectator");
    if (
      mode === "spectator" ||
      specParam === "1" ||
      specParam === "true"
    ) {
      window.sessionStorage?.setItem(SPECTATOR_STORAGE_KEY, "true");
      return true;
    }
    const stored = window.sessionStorage?.getItem(SPECTATOR_STORAGE_KEY);
    if (stored === "true") return true;
    if (stored === "false") return false;
  } catch (_err) {
    // Ignore storage errors
  }
  return false;
}

/**
 * Persist spectator mode flag in `sessionStorage`.
 */
export function setSpectatorModeStorage(spectator) {
  try {
    window.sessionStorage?.setItem(SPECTATOR_STORAGE_KEY, spectator ? "true" : "false");
  } catch (_err) {
    // Ignore storage errors
  }
}

/**
 * Build the WebSocket URL for `/ws?snapshot=1`.
 */
export function buildWebSocketUrl(token = null) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "127.0.0.1:8000";
  const url = new URL(`${proto}//${host}/ws`);
  url.searchParams.set("snapshot", "1");
  if (token) {
    url.searchParams.set("token", token);
  }
  return url.toString();
}

export class WorldSocketClient {
  constructor(store, options = {}) {
    this.store = store;
    this.token = options.token !== undefined ? options.token : resolveClientToken();
    this.spectatorMode =
      options.spectatorMode !== undefined ? Boolean(options.spectatorMode) : resolveSpectatorMode();
    this.ws = null;
    this.reconnectAttempt = 0;
    this._reconnectTimer = null;
    this._stopped = false;

    this.store.setAuthState({
      token: this.token,
      spectatorMode: this.spectatorMode,
    });
  }

  /**
   * Build HTTP request headers including Bearer and X-Creator-Token when a token is stored.
   */
  buildHeaders(extra = {}) {
    const headers = { ...extra };
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
      headers["X-Creator-Token"] = this.token;
    }
    return headers;
  }

  /**
   * Perform a JSON HTTP request to the simulation server with creator token headers.
   */
  async apiRequest(method, path, body = null) {
    const headers = this.buildHeaders();
    const init = { method, headers };
    if (body !== null && body !== undefined && method !== "GET") {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    try {
      const res = await fetch(path, init);
      let data = null;
      try {
        data = await res.json();
      } catch (_jsonErr) {
        data = null;
      }
      if (res.status === 401) {
        this.store.setAuthState({
          authenticated: false,
          authRequired: true,
          lastAuthError: data?.detail || "Invalid or missing creator token",
        });
      }
      return { ok: res.ok, status: res.status, data };
    } catch (err) {
      return { ok: false, status: 0, data: null, error: String(err) };
    }
  }

  /**
   * Authenticate with a creator token, store it in `sessionStorage`, exit spectator mode,
   * and refresh all console panels.
   */
  async loginWithToken(rawToken) {
    const candidate = typeof rawToken === "string" ? rawToken.trim() : "";
    const prevToken = this.token;
    this.token = candidate || null;

    const res = await this.apiRequest("POST", "/auth", { token: candidate });
    if (res.ok && res.data && res.data.authenticated) {
      const saved = setClientToken(candidate);
      this.token = saved;
      this.spectatorMode = false;
      setSpectatorModeStorage(false);
      this.store.setAuthState({
        token: this.token,
        spectatorMode: false,
        authenticated: true,
        authRequired: Boolean(res.data.auth_required),
        lastAuthError: null,
      });
      await this.fetchInitialSnapshot();
      this.connect();
      return { ok: true, data: res.data };
    }

    this.token = prevToken;
    this.store.setAuthState({
      authenticated: false,
      lastAuthError: res.data?.detail || "Creator token rejected by server",
    });
    return { ok: false, status: res.status, error: res.data?.detail || "Authentication failed" };
  }

  /**
   * Switch between spectator mode (all creator controls hidden) and creator mode.
   */
  setSpectatorMode(enabled) {
    this.spectatorMode = Boolean(enabled);
    setSpectatorModeStorage(this.spectatorMode);
    this.store.setAuthState({
      spectatorMode: this.spectatorMode,
    });
  }

  /**
   * Clear the creator token from `sessionStorage` and enter spectator mode.
   */
  logoutToSpectator() {
    clearClientToken();
    this.token = null;
    this.spectatorMode = true;
    setSpectatorModeStorage(true);
    this.store.setAuthState({
      token: null,
      authenticated: false,
      spectatorMode: true,
      lastAuthError: null,
    });
    this.connect();
  }

  /**
   * Fetch initial `/snapshot`, `/agent`, `/status`, `/presets`, `/inbox`, `/beliefs`,
   * `/self-model`, `/timeline`, and `/names` over HTTP and open the WebSocket stream.
   */
  async start() {
    this._stopped = false;
    this.store.setConnectionStatus("connecting", 0);
    await this.fetchInitialSnapshot();
    this.connect();
  }

  async fetchInitialSnapshot() {
    const headers = this.buildHeaders();
    try {
      const [
        statusRes,
        snapRes,
        agentRes,
        presetsRes,
        inboxRes,
        beliefsRes,
        selfModelRes,
        timelineRes,
        namesRes,
        weatherRes,
      ] = await Promise.all([
        fetch("/status", { headers }),
        fetch("/snapshot", { headers }),
        fetch("/agent", { headers }),
        fetch("/presets", { headers }),
        fetch("/inbox", { headers }),
        fetch("/beliefs", { headers }),
        fetch("/self-model", { headers }),
        fetch("/timeline", { headers }),
        fetch("/names", { headers }),
        fetch("/weather", { headers }),
      ]);

      if (statusRes.ok) {
        const statusData = await statusRes.json();
        const authRequired = Boolean(statusData.auth_required);
        const authenticated = Boolean(statusData.authenticated);
        // If server requires a token and user has not provided a valid one, default to spectator view until login
        const effectiveSpectator =
          this.spectatorMode || (authRequired && !authenticated && !this.token);
        this.store.setAuthState({
          authRequired,
          authenticated,
          spectatorMode: effectiveSpectator,
        });
      }
      if (snapRes.ok) {
        const snap = await snapRes.json();
        this.store.setSnapshot(snap);
      }
      if (agentRes.ok) {
        const agent = await agentRes.json();
        this.store.setAgentPayload(agent);
      }
      if (presetsRes.ok) {
        const presets = await presetsRes.json();
        this.store.setPresetsPayload(presets);
      }
      if (inboxRes.ok) {
        const inbox = await inboxRes.json();
        this.store.setInboxPayload(inbox);
      }
      if (beliefsRes.ok) {
        const beliefs = await beliefsRes.json();
        this.store.setBeliefsPayload(beliefs);
      }
      if (selfModelRes.ok) {
        const sm = await selfModelRes.json();
        this.store.setSelfModelPayload(sm);
      }
      if (timelineRes.ok) {
        const tl = await timelineRes.json();
        this.store.setTimelinePayload(tl);
      }
      if (namesRes.ok) {
        const names = await namesRes.json();
        this.store.setNamesPayload(names);
      }
      if (weatherRes.ok) {
        const weather = await weatherRes.json();
        this.store.setWeatherPayload(weather);
      }
    } catch (_err) {
      // Will retry via WebSocket ?snapshot=1
    }
  }

  /**
   * Refresh observer panels (`/inbox`, `/beliefs`, `/self-model`, `/names`, `/weather`) from HTTP.
   */
  async refreshObserverPanels() {
    const headers = this.buildHeaders();
    try {
      const [inboxRes, beliefsRes, selfModelRes, namesRes, weatherRes] = await Promise.all([
        fetch("/inbox", { headers }),
        fetch("/beliefs", { headers }),
        fetch("/self-model", { headers }),
        fetch("/names", { headers }),
        fetch("/weather", { headers }),
      ]);
      if (inboxRes.ok) this.store.setInboxPayload(await inboxRes.json());
      if (beliefsRes.ok) this.store.setBeliefsPayload(await beliefsRes.json());
      if (selfModelRes.ok) this.store.setSelfModelPayload(await selfModelRes.json());
      if (namesRes.ok) this.store.setNamesPayload(await namesRes.json());
      if (weatherRes.ok) this.store.setWeatherPayload(await weatherRes.json());
    } catch (_err) {
      // Ignore transient refresh failure
    }
  }

  connect() {
    if (this._stopped) return;
    this._clearReconnectTimer();

    if (this.ws) {
      try {
        this.ws.onopen = null;
        this.ws.onmessage = null;
        this.ws.onerror = null;
        this.ws.onclose = null;
        this.ws.close();
      } catch (_err) {
        // Ignore cleanup errors
      }
      this.ws = null;
    }

    const status = this.reconnectAttempt > 0 ? "reconnecting" : "connecting";
    this.store.setConnectionStatus(status, this.reconnectAttempt);

    let socket;
    try {
      socket = new WebSocket(buildWebSocketUrl(this.token));
    } catch (_err) {
      this._scheduleReconnect();
      return;
    }
    this.ws = socket;

    socket.onopen = () => {
      if (this._stopped) return;
      this.reconnectAttempt = 0;
      this.store.setConnectionStatus("connected", 0);
    };

    socket.onmessage = (event) => {
      if (this._stopped || !event?.data) return;
      try {
        const msg = JSON.parse(event.data);
        this.store.applyMessage(msg);
        const msgType = String(msg?.type || "");
        const deltaEvent = String(msg?.delta?.event || msg?.event || msg?.action || "");
        if (
          msgType === "reflection" ||
          msgType === "creator_action" ||
          deltaEvent === "mail_arrived" ||
          deltaEvent === "spoke"
        ) {
          this.refreshObserverPanels();
        }
      } catch (_err) {
        // Ignore malformed frame
      }
    };

    socket.onerror = () => {
      if (this._stopped) return;
      this.store.setConnectionStatus("reconnecting", this.reconnectAttempt);
    };

    socket.onclose = () => {
      if (this._stopped) return;
      this.ws = null;
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    if (this._stopped) return;
    this._clearReconnectTimer();
    this.reconnectAttempt += 1;
    this.store.setConnectionStatus(statusForReconnect(), this.reconnectAttempt);

    const delay = Math.min(
      MAX_RECONNECT_DELAY_MS,
      INITIAL_RECONNECT_DELAY_MS * Math.pow(1.6, this.reconnectAttempt - 1)
    );
    this._reconnectTimer = window.setTimeout(() => {
      this._reconnectTimer = null;
      this.connect();
    }, Math.round(delay));
  }

  _clearReconnectTimer() {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  stop() {
    this._stopped = true;
    this._clearReconnectTimer();
    if (this.ws) {
      try {
        this.ws.close();
      } catch (_err) {
        // Ignore close error
      }
      this.ws = null;
    }
    this.store.setConnectionStatus("disconnected", 0);
  }
}

function statusForReconnect() {
  return "reconnecting";
}
