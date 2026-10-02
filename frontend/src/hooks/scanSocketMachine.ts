/**
 * Pure state machine of the scan WebSocket client (`/ws/scans/{scan_id}`).
 *
 * Server protocol (backend `api/routes/websocket.py`): one `snapshot` event on
 * every connect, then `state` / `progress` / `point` / `profile` / `error`;
 * the server closes with 1000 after the terminal state and with 4404 for an
 * unknown scan.
 *
 * ```
 * idle ──connect──► connecting ──open──► open ──event*──► open
 *                       ▲                  │
 *                       │ retry            │ closed (not 4404, no terminal state seen)
 *                       └── reconnecting ◄─┘   (exponential back-off, unlimited)
 * closed + terminal state seen ──► finished      (no reconnect)
 * closed with 4404             ──► not_found     (no reconnect)
 * ```
 *
 * A reconnect always starts with a new snapshot, which replaces the progress;
 * consumers re-synchronise missed points from the REST API on every snapshot.
 */
import { isTerminalState } from "../api/types";
import type { ScanPoint } from "../api/types";
import type { LiveProfile, ScanEvent, ScanProgress } from "../api/events";

export const CLOSE_NORMAL = 1000;
export const CLOSE_UNKNOWN_SCAN = 4404;

export const RECONNECT_BASE_MS = 500;
export const RECONNECT_MAX_MS = 10_000;

export type SocketStatus =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "finished"
  | "not_found";

export interface ScanSocketState {
  scanId: string | null;
  status: SocketStatus;
  /** Consecutive connection attempts that did not deliver a snapshot. */
  failures: number;
  /** Delay before the next connection attempt (status `reconnecting`). */
  retryInMs: number | null;
  /** Number of snapshots received (one per successful (re)connect). */
  snapshots: number;
  /** True once a terminal scan state (complete / cancelled / error) was received. */
  terminal: boolean;
  progress: ScanProgress | null;
  /** Live I(Z) of the point being measured (latest PROFILE event). */
  profile: LiveProfile | null;
  /** Last finished point (POINT event, or the snapshot's last point). */
  lastPoint: ScanPoint | null;
  /** Message of the latest ERROR event. */
  lastError: string | null;
  /** Latest event message of any type (state reasons, warnings). */
  lastMessage: string | null;
  lastCloseCode: number | null;
}

export type ScanSocketAction =
  | { type: "connect"; scanId: string }
  | { type: "open" }
  | { type: "event"; event: ScanEvent }
  | { type: "closed"; code: number }
  | { type: "reset" };

export const initialSocketState: ScanSocketState = {
  scanId: null,
  status: "idle",
  failures: 0,
  retryInMs: null,
  snapshots: 0,
  terminal: false,
  progress: null,
  profile: null,
  lastPoint: null,
  lastError: null,
  lastMessage: null,
  lastCloseCode: null,
};

/** Back-off before reconnect attempt `failures` (1-based): 0.5 s, 1 s, 2 s, ... max 10 s. */
export function reconnectDelay(failures: number): number {
  const exponent = Math.max(0, failures - 1);
  return Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** Math.min(exponent, 16));
}

function applyEvent(state: ScanSocketState, event: ScanEvent): ScanSocketState {
  if (state.scanId !== null && event.scan_id !== state.scanId) return state;
  const terminal = state.terminal || isTerminalState(event.progress.state);
  const next: ScanSocketState = {
    ...state,
    progress: event.progress,
    terminal,
    lastMessage: event.message ?? state.lastMessage,
  };
  switch (event.type) {
    case "snapshot":
      return {
        ...next,
        status: "open",
        failures: 0,
        retryInMs: null,
        snapshots: state.snapshots + 1,
        lastPoint: event.point ?? state.lastPoint,
      };
    case "point":
      return { ...next, lastPoint: event.point ?? state.lastPoint };
    case "profile":
      return { ...next, profile: event.profile ?? state.profile };
    case "error":
      return { ...next, lastError: event.message ?? "scan error" };
    case "state":
    case "progress":
      return next;
  }
}

export function scanSocketReducer(
  state: ScanSocketState,
  action: ScanSocketAction,
): ScanSocketState {
  switch (action.type) {
    case "reset":
      return initialSocketState;
    case "connect":
      if (action.scanId !== state.scanId) {
        return { ...initialSocketState, scanId: action.scanId, status: "connecting" };
      }
      if (state.status === "finished" || state.status === "not_found") return state;
      return { ...state, status: "connecting", retryInMs: null };
    case "open":
      if (state.status !== "connecting") return state;
      return { ...state, status: "open" };
    case "event":
      if (state.status === "idle" || state.status === "not_found") return state;
      return applyEvent(state, action.event);
    case "closed": {
      if (state.status === "idle" || state.status === "finished" || state.status === "not_found") {
        return state;
      }
      if (action.code === CLOSE_UNKNOWN_SCAN) {
        return { ...state, status: "not_found", retryInMs: null, lastCloseCode: action.code };
      }
      if (state.terminal) {
        return { ...state, status: "finished", retryInMs: null, lastCloseCode: action.code };
      }
      const failures = state.failures + 1;
      return {
        ...state,
        status: "reconnecting",
        failures,
        retryInMs: reconnectDelay(failures),
        lastCloseCode: action.code,
      };
    }
  }
}
