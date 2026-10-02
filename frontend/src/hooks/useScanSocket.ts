/**
 * React hook around {@link scanSocketReducer}: connects to
 * `/ws/scans/{scanId}`, reconnects with back-off until the scan reaches a
 * terminal state, and hands every event to `onEvent` (used to accumulate
 * points and re-synchronise from REST on each snapshot).
 */
import { useEffect, useRef, useState } from "react";

import { scanSocketUrl } from "../api/client";
import { parseScanEvent, type ScanEvent } from "../api/events";
import {
  CLOSE_NORMAL,
  initialSocketState,
  scanSocketReducer,
  type ScanSocketAction,
  type ScanSocketState,
} from "./scanSocketMachine";

/** The subset of the browser WebSocket the hook uses (lets tests inject a fake). */
export interface SocketLike {
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  close(code?: number, reason?: string): void;
}

export type SocketFactory = (url: string) => SocketLike;

const browserSocket: SocketFactory = (url) => new WebSocket(url);

export interface UseScanSocketOptions {
  /** Called for every event of this scan, after the state update. */
  onEvent?: (event: ScanEvent) => void;
  /** Override the WebSocket constructor (tests). */
  socketFactory?: SocketFactory;
  /** Override the URL builder (tests). */
  url?: (scanId: string) => string;
}

export function useScanSocket(
  scanId: string | null | undefined,
  options: UseScanSocketOptions = {},
): ScanSocketState {
  const [state, setState] = useState<ScanSocketState>(initialSocketState);
  const onEventRef = useRef(options.onEvent);
  const factory = options.socketFactory ?? browserSocket;
  const buildUrl = options.url ?? scanSocketUrl;
  const factoryRef = useRef(factory);
  const urlRef = useRef(buildUrl);

  useEffect(() => {
    onEventRef.current = options.onEvent;
    factoryRef.current = factory;
    urlRef.current = buildUrl;
  });

  useEffect(() => {
    if (!scanId) return undefined;
    let machine: ScanSocketState = initialSocketState;
    let socket: SocketLike | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let disposed = false;

    const dispatch = (action: ScanSocketAction): void => {
      machine = scanSocketReducer(machine, action);
      setState(machine);
    };

    const connect = (): void => {
      timer = null;
      if (disposed) return;
      dispatch({ type: "connect", scanId });
      let current: SocketLike;
      try {
        current = factoryRef.current(urlRef.current(scanId));
      } catch {
        // Invalid URL or the browser refused: treat like an abnormal close.
        dispatch({ type: "closed", code: 1006 });
        schedule();
        return;
      }
      socket = current;
      current.onopen = () => {
        if (!disposed && socket === current) dispatch({ type: "open" });
      };
      current.onmessage = (message) => {
        if (disposed || socket !== current) return;
        const event = parseScanEvent(message.data);
        if (event === null || event.scan_id !== scanId) return;
        dispatch({ type: "event", event });
        onEventRef.current?.(event);
      };
      current.onerror = () => {
        // Always followed by a close event, which drives the state machine.
      };
      current.onclose = (event) => {
        if (disposed || socket !== current) return;
        socket = null;
        dispatch({ type: "closed", code: event.code });
        schedule();
      };
    };

    const schedule = (): void => {
      if (disposed || machine.status !== "reconnecting" || machine.retryInMs === null) return;
      timer = setTimeout(connect, machine.retryInMs);
    };

    connect();
    return () => {
      disposed = true;
      if (timer !== null) clearTimeout(timer);
      const open = socket;
      socket = null;
      if (open) {
        open.onopen = null;
        open.onmessage = null;
        open.onclose = null;
        open.onerror = null;
        try {
          open.close(CLOSE_NORMAL, "page left");
        } catch {
          // Already closed.
        }
      }
    };
  }, [scanId]);

  // A stale state of a previous scan id (or of no scan) is never returned.
  return scanId && state.scanId === scanId ? state : initialSocketState;
}
