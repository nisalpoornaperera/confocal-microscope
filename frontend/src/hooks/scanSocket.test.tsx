import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { event, point } from "../test/fixtures";
import {
  CLOSE_UNKNOWN_SCAN,
  initialSocketState,
  reconnectDelay,
  RECONNECT_MAX_MS,
  scanSocketReducer,
  type ScanSocketAction,
  type ScanSocketState,
} from "./scanSocketMachine";
import { useScanSocket, type SocketLike } from "./useScanSocket";

function run(actions: ScanSocketAction[], start: ScanSocketState = initialSocketState): ScanSocketState {
  return actions.reduce(scanSocketReducer, start);
}

describe("scanSocketReducer", () => {
  it("connects, opens and applies the snapshot", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "open" },
      { type: "event", event: event("snapshot") },
    ]);
    expect(state.status).toBe("open");
    expect(state.snapshots).toBe(1);
    expect(state.progress?.state).toBe("scanning");
    expect(state.terminal).toBe(false);
  });

  it("keeps the latest profile, point and error", () => {
    const profile = { point_id: 3, x_um: 0, y_um: 0, phase: [0], z_um: [1], intensity: [0.5] };
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot") },
      { type: "event", event: event("profile", "scanning", { profile }) },
      { type: "event", event: event("point", "scanning", { point: point({ point_id: 3 }) }) },
      { type: "event", event: event("error", "scanning", { message: "ADC failure" }) },
    ]);
    expect(state.profile).toEqual(profile);
    expect(state.lastPoint?.point_id).toBe(3);
    expect(state.lastError).toBe("ADC failure");
    expect(state.lastMessage).toBe("ADC failure");
  });

  it("ignores events of another scan", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: { ...event("snapshot"), scan_id: "other" } },
    ]);
    expect(state.snapshots).toBe(0);
    expect(state.progress).toBeNull();
  });

  it("reconnects with exponential back-off after an abnormal close", () => {
    let state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot") },
      { type: "closed", code: 1006 },
    ]);
    expect(state.status).toBe("reconnecting");
    expect(state.failures).toBe(1);
    expect(state.retryInMs).toBe(500);
    state = run([{ type: "connect", scanId: "scan-1" }, { type: "closed", code: 1006 }], state);
    expect(state.failures).toBe(2);
    expect(state.retryInMs).toBe(1000);
    // A new snapshot resets the failure count.
    state = run([{ type: "connect", scanId: "scan-1" }, { type: "event", event: event("snapshot") }], state);
    expect(state.failures).toBe(0);
    expect(state.snapshots).toBe(2);
    expect(state.status).toBe("open");
  });

  it("also reconnects after a 1000 close when no terminal state was seen (server restart)", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot") },
      { type: "closed", code: 1000 },
    ]);
    expect(state.status).toBe("reconnecting");
  });

  it("finishes after the terminal state and refuses to reconnect", () => {
    let state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot") },
      { type: "event", event: event("state", "complete") },
      { type: "closed", code: 1000 },
    ]);
    expect(state.terminal).toBe(true);
    expect(state.status).toBe("finished");
    state = run([{ type: "connect", scanId: "scan-1" }], state);
    expect(state.status).toBe("finished");
  });

  it("treats a terminal snapshot (already finished scan) as finished", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot", "cancelled") },
      { type: "closed", code: 1000 },
    ]);
    expect(state.status).toBe("finished");
  });

  it("stops on 4404 (unknown scan)", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "open" },
      { type: "closed", code: CLOSE_UNKNOWN_SCAN },
    ]);
    expect(state.status).toBe("not_found");
    expect(state.retryInMs).toBeNull();
  });

  it("starts fresh for a different scan", () => {
    const state = run([
      { type: "connect", scanId: "scan-1" },
      { type: "event", event: event("snapshot", "complete") },
      { type: "closed", code: 1000 },
      { type: "connect", scanId: "scan-2" },
    ]);
    expect(state.scanId).toBe("scan-2");
    expect(state.status).toBe("connecting");
    expect(state.terminal).toBe(false);
    expect(state.progress).toBeNull();
  });

  it("caps the back-off", () => {
    expect(reconnectDelay(1)).toBe(500);
    expect(reconnectDelay(3)).toBe(2000);
    expect(reconnectDelay(50)).toBe(RECONNECT_MAX_MS);
  });
});

class FakeSocket implements SocketLike {
  static instances: FakeSocket[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  closedWith: number | null = null;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  close(code?: number): void {
    this.closedWith = code ?? 1000;
  }

  open(): void {
    this.onopen?.(new Event("open"));
  }

  send(data: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(data) }));
  }

  serverClose(code: number): void {
    this.onclose?.(new CloseEvent("close", { code }));
  }
}

describe("useScanSocket", () => {
  beforeEach(() => {
    FakeSocket.instances = [];
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const options = (onEvent?: (e: unknown) => void) => ({
    socketFactory: (url: string) => new FakeSocket(url),
    url: (id: string) => `ws://test/ws/scans/${id}`,
    onEvent,
  });

  it("stays idle without a scan id", () => {
    const { result } = renderHook(() => useScanSocket(null, options()));
    expect(result.current.status).toBe("idle");
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it("delivers events, reconnects after a drop and stops after the terminal state", () => {
    const received: unknown[] = [];
    const { result } = renderHook(() => useScanSocket("scan-1", options((e) => received.push(e))));
    expect(FakeSocket.instances).toHaveLength(1);
    const first = FakeSocket.instances[0];
    if (!first) throw new Error("no socket");
    expect(first.url).toBe("ws://test/ws/scans/scan-1");
    act(() => {
      first.open();
      first.send(event("snapshot"));
      first.send(event("point", "scanning", { point: point({ point_id: 7 }) }));
      first.send("not json at all");
    });
    expect(result.current.status).toBe("open");
    expect(result.current.lastPoint?.point_id).toBe(7);
    expect(received).toHaveLength(2);

    act(() => first.serverClose(1006));
    expect(result.current.status).toBe("reconnecting");
    act(() => {
      vi.advanceTimersByTime(499);
    });
    expect(FakeSocket.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeSocket.instances).toHaveLength(2);
    const second = FakeSocket.instances[1];
    if (!second) throw new Error("no socket");
    act(() => {
      second.open();
      second.send(event("snapshot"));
      second.send(event("state", "complete"));
      second.serverClose(1000);
    });
    expect(result.current.status).toBe("finished");
    expect(result.current.snapshots).toBe(2);
    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(FakeSocket.instances).toHaveLength(2);
  });

  it("does not reconnect to an unknown scan", () => {
    const { result } = renderHook(() => useScanSocket("missing", options()));
    const socket = FakeSocket.instances[0];
    if (!socket) throw new Error("no socket");
    act(() => {
      socket.open();
      socket.serverClose(4404);
    });
    expect(result.current.status).toBe("not_found");
    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(FakeSocket.instances).toHaveLength(1);
  });

  it("closes the socket and cancels a pending reconnect on unmount", () => {
    const { unmount } = renderHook(() => useScanSocket("scan-1", options()));
    const socket = FakeSocket.instances[0];
    if (!socket) throw new Error("no socket");
    unmount();
    expect(socket.closedWith).toBe(1000);

    const second = renderHook(() => useScanSocket("scan-1", options()));
    const dropped = FakeSocket.instances[1];
    if (!dropped) throw new Error("no socket");
    act(() => dropped.serverClose(1006));
    second.unmount();
    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(FakeSocket.instances).toHaveLength(2);
  });

  it("reconnects to the new scan when the id changes", () => {
    const { rerender, result } = renderHook(({ id }) => useScanSocket(id, options()), {
      initialProps: { id: "scan-1" },
    });
    const first = FakeSocket.instances[0];
    rerender({ id: "scan-2" });
    expect(first?.closedWith).toBe(1000);
    expect(FakeSocket.instances[1]?.url).toBe("ws://test/ws/scans/scan-2");
    expect(result.current.scanId).toBe("scan-2");
  });
});
