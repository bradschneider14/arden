/**
 * Live WebSocket client with incremental delta dispatch, automatic exponential-backoff
 * reconnection, and initial HTTP snapshot hydration.
 */

export const INITIAL_RECONNECT_DELAY_MS = 500;
export const MAX_RECONNECT_DELAY_MS = 8000;

/**
 * Extract an optional creator/spectator token from URL query params or localStorage.
 */
export function resolveClientToken() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromQuery =
      params.get("token") || params.get("creator_token") || params.get("api_key");
    if (fromQuery && fromQuery.trim()) {
      return fromQuery.trim();
    }
    const fromStorage = window.localStorage?.getItem("arden_creator_token");
    if (fromStorage && fromStorage.trim()) {
      return fromStorage.trim();
    }
  } catch (_err) {
    // Ignore storage/location access errors
  }
  return null;
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
    this.token = options.token ?? resolveClientToken();
    this.ws = null;
    this.reconnectAttempt = 0;
    this._reconnectTimer = null;
    this._stopped = false;
  }

  /**
   * Fetch initial `/snapshot` and `/agent` state over HTTP and open the WebSocket stream.
   */
  async start() {
    this._stopped = false;
    this.store.setConnectionStatus("connecting", 0);
    await this.fetchInitialSnapshot();
    this.connect();
  }

  async fetchInitialSnapshot() {
    const headers = {};
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    try {
      const [snapRes, agentRes] = await Promise.all([
        fetch("/snapshot", { headers }),
        fetch("/agent", { headers }),
      ]);
      if (snapRes.ok) {
        const snap = await snapRes.json();
        this.store.setSnapshot(snap);
      }
      if (agentRes.ok) {
        const agent = await agentRes.json();
        this.store.setAgentPayload(agent);
      }
    } catch (_err) {
      // Will retry via WebSocket ?snapshot=1
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
    this.store.setConnectionStatus("reconnecting", this.reconnectAttempt);

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
