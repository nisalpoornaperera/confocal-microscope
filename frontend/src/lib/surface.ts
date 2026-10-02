/** Pure helpers of the Surface Viewer (cross-sections, scaling, ranges). */
import type { SurfacePoint, SurfaceResult } from "../api/types";

export interface Section {
  coordinate: number[];
  z: (number | null)[];
  /** Fixed coordinate of the other axis (µm). */
  position: number;
}

/**
 * Cross-section of the height map: `along = "x"` takes row `index` (fixed Y),
 * `along = "y"` takes column `index` (fixed X). Gaps stay `null`.
 */
export function crossSection(surface: Pick<SurfaceResult, "x_um" | "y_um" | "z_um">, along: "x" | "y", index: number): Section {
  if (along === "x") {
    const iy = Math.min(Math.max(0, Math.round(index)), Math.max(0, surface.y_um.length - 1));
    const row = surface.z_um[iy] ?? [];
    return { coordinate: [...surface.x_um], z: surface.x_um.map((_, ix) => row[ix] ?? null), position: surface.y_um[iy] ?? Number.NaN };
  }
  const ix = Math.min(Math.max(0, Math.round(index)), Math.max(0, surface.x_um.length - 1));
  return {
    coordinate: [...surface.y_um],
    z: surface.y_um.map((_, iy) => surface.z_um[iy]?.[ix] ?? null),
    position: surface.x_um[ix] ?? Number.NaN,
  };
}

/** Measured points lying on a cross-section line (within half a grid step). */
export function pointsNearSection(
  points: readonly SurfacePoint[],
  along: "x" | "y",
  position: number,
  tolerance: number,
): SurfacePoint[] {
  return points.filter((p) => {
    const other = along === "x" ? p.y_um : p.x_um;
    return p.z_um !== null && Math.abs(other - position) <= tolerance;
  });
}

/** Min / max of the finite values of a grid (null when there are none). */
export function gridRange(grid: readonly (readonly (number | null)[])[]): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const row of grid) {
    for (const value of row) {
      if (value === null || !Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return min <= max ? [min, max] : null;
}

/** Slider position 0..100 -> vertical exaggeration 1x..1000x (logarithmic). */
export function sliderToScale(position: number): number {
  const clamped = Math.min(100, Math.max(0, position));
  return Math.round(10 ** ((clamped / 100) * 3) * 10) / 10;
}

export function scaleToSlider(scale: number): number {
  if (!(scale > 1)) return 0;
  return Math.min(100, (Math.log10(scale) / 3) * 100);
}

/**
 * Plotly scene aspect ratio for true XY proportions and Z exaggerated
 * `scale` times. Spans are in the same unit.
 */
export function sceneAspect(xSpan: number, ySpan: number, zSpan: number, scale: number): { x: number; y: number; z: number } {
  const base = Math.max(xSpan, ySpan, Number.EPSILON);
  const x = Math.max(xSpan / base, 0.05);
  const y = Math.max(ySpan / base, 0.05);
  const z = Math.min(Math.max((zSpan / base) * scale, 0.01), 10);
  return { x, y, z };
}

/**
 * The vertical exaggeration that makes the height range a fraction `target`
 * of the larger XY span (initial slider value).
 */
export function autoScale(xSpan: number, ySpan: number, zSpan: number, target = 0.35): number {
  const base = Math.max(xSpan, ySpan);
  if (!(zSpan > 0) || !(base > 0)) return 1;
  return Math.min(1000, Math.max(1, Math.round((target * base) / zSpan)));
}
