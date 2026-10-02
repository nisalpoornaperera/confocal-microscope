/**
 * The XY grid of a scan and the live maps filled from its points
 * (completion / status map, confidence map, fixed-Z intensity map).
 */
import type { PointStatus, ScanConfig, ScanPoint } from "../api/types";

const GRID_EPS = 1e-9;

/** Number of positions from start to stop inclusive (mirrors `axis_count` in the backend). */
export function axisCount(startUm: number, stopUm: number, stepUm: number): number {
  if (!(stepUm > 0) || !Number.isFinite(stepUm)) return 0;
  const span = stopUm - startUm;
  if (!(span >= 0) || !Number.isFinite(span)) return 0;
  return Math.floor(span / stepUm + GRID_EPS) + 1;
}

export function axisValues(startUm: number, stopUm: number, stepUm: number): number[] {
  const n = axisCount(startUm, stopUm, stepUm);
  return Array.from({ length: n }, (_, i) => startUm + i * stepUm);
}

/**
 * Status categories of the completion map (the heatmap's discrete values).
 * 0 = not measured yet.
 */
export const STATUS_CATEGORY = {
  pending: 0,
  valid: 1,
  low_confidence: 2,
  failed: 3,
  aborted: 4,
  measured: 5,
} as const;

export const STATUS_CATEGORY_LABELS = [
  "Pending",
  "Valid",
  "Low confidence",
  "No peak / edge / fit failed",
  "Aborted / error",
  "Measured (fixed Z)",
] as const;

/** Colours of the completion-map categories (index = category), legible in both themes. */
export const STATUS_CATEGORY_COLORS = [
  "#9aa3ad",
  "#2e9d5b",
  "#e0a526",
  "#d9622b",
  "#b3261e",
  "#3b7dd8",
] as const;

/** Confidence 0..1: red (unreliable) -> amber -> green (reliable). */
export const CONFIDENCE_COLORSCALE: [number, string][] = [
  [0, "#b3261e"],
  [0.5, "#e0a526"],
  [1, "#2e9d5b"],
];

export function statusCategory(status: PointStatus): number {
  switch (status) {
    case "valid":
      return STATUS_CATEGORY.valid;
    case "low_confidence":
      return STATUS_CATEGORY.low_confidence;
    case "no_peak":
    case "peak_at_edge":
    case "fit_failed":
      return STATUS_CATEGORY.failed;
    case "aborted":
    case "error":
      return STATUS_CATEGORY.aborted;
    case "measured":
      return STATUS_CATEGORY.measured;
  }
}

/** Live maps of one scan, updated in place as points arrive (O(1) per point). */
export class ScanGrid {
  readonly nx: number;
  readonly ny: number;
  readonly x: number[];
  readonly y: number[];
  /** Status category per cell, row-major [iy * nx + ix]. */
  readonly category: Uint8Array;
  /** Confidence per cell (NaN = none). */
  readonly confidence: Float64Array;
  /** Surface height per cell (NaN = none). */
  readonly height: Float64Array;
  /** Fixed-Z intensity per cell (NaN = none). */
  readonly intensity: Float64Array;
  private readonly seen = new Set<number>();
  private readonly counts = new Array<number>(STATUS_CATEGORY_LABELS.length).fill(0);

  constructor(config: Pick<ScanConfig, "x_start_um" | "x_stop_um" | "y_start_um" | "y_stop_um" | "xy_step_um">) {
    this.x = axisValues(config.x_start_um, config.x_stop_um, config.xy_step_um);
    this.y = axisValues(config.y_start_um, config.y_stop_um, config.xy_step_um);
    this.nx = this.x.length;
    this.ny = this.y.length;
    const n = this.nx * this.ny;
    this.category = new Uint8Array(n);
    this.confidence = new Float64Array(n).fill(Number.NaN);
    this.height = new Float64Array(n).fill(Number.NaN);
    this.intensity = new Float64Array(n).fill(Number.NaN);
    this.counts[STATUS_CATEGORY.pending] = n;
  }

  get size(): number {
    return this.nx * this.ny;
  }

  /** Number of distinct points added. */
  get measured(): number {
    return this.seen.size;
  }

  countOf(category: number): number {
    return this.counts[category] ?? 0;
  }

  /** Add (or replace) one point. Returns false when it lies outside the grid. */
  add(point: ScanPoint): boolean {
    if (point.ix < 0 || point.iy < 0 || point.ix >= this.nx || point.iy >= this.ny) return false;
    const index = point.iy * this.nx + point.ix;
    const previous = this.category[index] ?? 0;
    const next = statusCategory(point.status);
    this.counts[previous] = (this.counts[previous] ?? 0) - 1;
    this.counts[next] = (this.counts[next] ?? 0) + 1;
    this.category[index] = next;
    this.confidence[index] = next === STATUS_CATEGORY.aborted ? Number.NaN : point.confidence;
    this.height[index] =
      point.surface_z_um != null && point.status === "valid" ? point.surface_z_um : Number.NaN;
    this.intensity[index] = point.intensity ?? Number.NaN;
    this.seen.add(point.point_id);
    return true;
  }

  /** Row-major `[iy][ix]` matrix of a cell array, NaN -> null (Plotly gap). */
  matrix(values: Float64Array | Uint8Array, mask?: (index: number) => boolean): (number | null)[][] {
    const rows: (number | null)[][] = [];
    for (let iy = 0; iy < this.ny; iy += 1) {
      const row: (number | null)[] = new Array<number | null>(this.nx);
      for (let ix = 0; ix < this.nx; ix += 1) {
        const index = iy * this.nx + ix;
        const value = values[index];
        row[ix] =
          value === undefined || Number.isNaN(value) || (mask !== undefined && !mask(index))
            ? null
            : value;
      }
      rows.push(row);
    }
    return rows;
  }

  /** Completion map: category per cell (pending cells are 0, not gaps). */
  categoryMatrix(): (number | null)[][] {
    return this.matrix(this.category);
  }
}
