// Minimal reconnecting WebSocket wrapper. Designed for vite dev proxy:
//   /ws/* requests are proxied to ws://localhost:8000 by vite.config.ts.

import type { DownlinkMsg } from "./types";

export type WSStatus = "connecting" | "open" | "closed";

export interface ManagedWS {
  send(data: string | ArrayBufferView | Blob): void;
  close(): void;
  status: WSStatus;
}

export function openWS(
  path: string,
  onMessage: (msg: DownlinkMsg) => void,
  onStatus?: (s: WSStatus) => void,
  binaryType: BinaryType = "arraybuffer",
): ManagedWS {
  let ws: WebSocket | null = null;
  let closedByUser = false;
  let backoff = 250;
  const state = { status: "connecting" as WSStatus };

  const url = (() => {
    if (path.startsWith("ws://") || path.startsWith("wss://")) return path;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}${path}`;
  })();

  function setStatus(s: WSStatus) {
    state.status = s;
    onStatus?.(s);
  }

  function connect() {
    setStatus("connecting");
    ws = new WebSocket(url);
    ws.binaryType = binaryType;
    ws.onopen = () => {
      backoff = 250;
      setStatus("open");
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          onMessage(JSON.parse(ev.data));
        } catch {
          /* ignore non-JSON text */
        }
      }
      // Binary downlink isn't used in v1; if added later it lands here.
    };
    ws.onclose = () => {
      setStatus("closed");
      if (!closedByUser) {
        setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, 5000);
      }
    };
    ws.onerror = () => {
      try { ws?.close(); } catch { /* noop */ }
    };
  }

  connect();

  return {
    send(data) {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(data as any);
    },
    close() {
      closedByUser = true;
      try { ws?.close(); } catch { /* noop */ }
    },
    get status() { return state.status; },
  };
}
