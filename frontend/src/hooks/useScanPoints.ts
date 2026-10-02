/**
 * Every point of one scan, kept in a {@link ScanGrid}: loaded from
 * `GET /scans/{id}/points` (paged), then extended by POINT events. After each
 * WebSocket (re)connect snapshot the points missed while disconnected are
 * fetched with `since`. Re-renders are throttled so a fast scan does not
 * redraw the maps for every point.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, isAbortError, toApiError } from "../api/client";
import { fetchAllPoints } from "../api/endpoints";
import type { ScanEvent } from "../api/events";
import type { ScanConfig, ScanPoint } from "../api/types";
import { ScanGrid } from "../lib/grid";

/** Minimum interval between re-renders caused by new points. */
export const POINTS_RENDER_INTERVAL_MS = 500;

export interface ScanPointsState {
  grid: ScanGrid | null;
  /** Points by id (mutated in place; `version` tells when). */
  points: ReadonlyMap<number, ScanPoint>;
  /** Changes whenever points were added (use as a memo dependency). */
  version: number;
  loading: boolean;
  error: ApiError | null;
  /** Feed WebSocket events here (POINT events add points, SNAPSHOTs re-sync). */
  handleEvent: (event: ScanEvent) => void;
}

interface Store {
  grid: ScanGrid | null;
  points: Map<number, ScanPoint>;
  maxId: number;
}

type Geometry = Pick<ScanConfig, "x_start_um" | "x_stop_um" | "y_start_um" | "y_stop_um" | "xy_step_um">;

function parseGeometry(key: string | null): Geometry | null {
  if (key === null) return null;
  const [x0, x1, y0, y1, step] = key.split("|").map(Number);
  if ([x0, x1, y0, y1, step].some((v) => v === undefined || !Number.isFinite(v))) return null;
  return { x_start_um: x0 ?? 0, x_stop_um: x1 ?? 0, y_start_um: y0 ?? 0, y_stop_um: y1 ?? 0, xy_step_um: step ?? 1 };
}

export function useScanPoints(scanId: string | undefined, config: ScanConfig | undefined): ScanPointsState {
  const geometryKey = config
    ? [config.x_start_um, config.x_stop_um, config.y_start_um, config.y_stop_um, config.xy_step_um].join("|")
    : null;

  // One store per scan geometry; mutated in place as points arrive.
  const store = useMemo<Store>(() => {
    const geometry = scanId ? parseGeometry(geometryKey) : null;
    return { grid: geometry ? new ScanGrid(geometry) : null, points: new Map(), maxId: -1 };
  }, [scanId, geometryKey]);

  const storeRef = useRef(store);
  const syncRef = useRef<AbortController | null>(null);
  /** A snapshot arrived while a sync was running: sync again when it ends. */
  const resyncRef = useRef(false);
  const syncAgainRef = useRef<() => void>(() => undefined);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastRenderRef = useRef(0);
  const [version, setVersion] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const scheduleRender = useCallback(() => {
    if (timerRef.current !== null) return;
    const wait = Math.max(0, lastRenderRef.current + POINTS_RENDER_INTERVAL_MS - Date.now());
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      lastRenderRef.current = Date.now();
      setVersion((v) => v + 1);
    }, wait);
  }, []);

  const addPoints = useCallback(
    (target: Store, points: readonly ScanPoint[]) => {
      for (const point of points) {
        target.points.set(point.point_id, point);
        if (point.point_id > target.maxId) target.maxId = point.point_id;
        target.grid?.add(point);
      }
      if (points.length > 0 && target === storeRef.current) scheduleRender();
    },
    [scheduleRender],
  );

  /** Fetch every point after the newest one we have (initial load and re-sync). */
  const sync = useCallback(() => {
    const target = storeRef.current;
    if (!scanId || target.grid === null) return;
    syncRef.current?.abort();
    const controller = new AbortController();
    syncRef.current = controller;
    setLoading(true);
    const since = target.maxId >= 0 ? target.maxId : undefined;
    fetchAllPoints(scanId, since, (page) => addPoints(target, page), controller.signal)
      .then(() => {
        if (!controller.signal.aborted) setError(null);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted || isAbortError(err)) return;
        setError(toApiError(err));
      })
      .finally(() => {
        if (syncRef.current === controller) {
          syncRef.current = null;
          setLoading(false);
          if (resyncRef.current && storeRef.current === target) {
            resyncRef.current = false;
            syncAgainRef.current();
          }
        }
      });
  }, [scanId, addPoints]);

  useEffect(() => {
    syncAgainRef.current = sync;
  }, [sync]);

  useEffect(() => {
    storeRef.current = store;
    resyncRef.current = false;
    sync();
    return () => {
      syncRef.current?.abort();
      syncRef.current = null;
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      timerRef.current = null;
    };
  }, [store, sync]);

  const handleEvent = useCallback(
    (event: ScanEvent) => {
      const target = storeRef.current;
      if (event.type === "point" && event.point) {
        addPoints(target, [event.point]);
      } else if (event.type === "snapshot" && target.grid !== null) {
        // Events may have been missed while disconnected: fetch what is new.
        if (syncRef.current === null) sync();
        else resyncRef.current = true;
      }
    },
    [addPoints, sync],
  );

  return { grid: store.grid, points: store.points, version, loading, error, handleEvent };
}
