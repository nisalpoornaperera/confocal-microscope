/**
 * Messages of the WebSocket `/ws/scans/{scan_id}`.
 *
 * WebSocket payloads are not part of the OpenAPI schema, so these types are
 * written by hand and mirror `confocal/models/scan.py` (`ScanEvent`,
 * `ScanProgress`, `LiveProfile`, `ScanEventType`). Keep them in sync.
 */
import type { ScanPoint, ScanState } from "./types";

export type ScanEventType = "snapshot" | "state" | "progress" | "point" | "profile" | "error";

export interface ScanProgress {
  scan_id: string;
  state: ScanState;
  progress: number;
  completed_points: number;
  total_points: number;
  current_point_id: number | null;
  current_x_um: number | null;
  current_y_um: number | null;
  current_z_um: number | null;
  current_intensity: number | null;
  elapsed_s: number;
  estimated_remaining_s: number | null;
  message: string | null;
}

/** Live I(Z) data of the point being measured: cumulative (coarse + fine so far). */
export interface LiveProfile {
  point_id: number;
  x_um: number;
  y_um: number;
  /** ProfilePhase per Z: 0 coarse, 1 fine, 2 fixed. */
  phase: number[];
  z_um: number[];
  intensity: (number | null)[];
}

export interface ScanEvent {
  type: ScanEventType;
  scan_id: string;
  timestamp: string;
  progress: ScanProgress;
  point: ScanPoint | null;
  profile: LiveProfile | null;
  message: string | null;
}

const EVENT_TYPES: readonly string[] = ["snapshot", "state", "progress", "point", "profile", "error"];

/** Parse one WebSocket text frame; `null` when it is not a ScanEvent. */
export function parseScanEvent(data: unknown): ScanEvent | null {
  if (typeof data !== "string") return null;
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  if (typeof record.type !== "string" || !EVENT_TYPES.includes(record.type)) return null;
  if (typeof record.scan_id !== "string") return null;
  const progress = record.progress;
  if (typeof progress !== "object" || progress === null) return null;
  if (typeof (progress as Record<string, unknown>).state !== "string") return null;
  return {
    type: record.type as ScanEventType,
    scan_id: record.scan_id,
    timestamp: typeof record.timestamp === "string" ? record.timestamp : "",
    progress: progress as ScanProgress,
    point: (record.point ?? null) as ScanPoint | null,
    profile: (record.profile ?? null) as LiveProfile | null,
    message: typeof record.message === "string" ? record.message : null,
  };
}
