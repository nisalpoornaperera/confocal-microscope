/**
 * Display-only data reduction. Stored data is never modified: these helpers
 * only limit what is handed to Plotly so a Pi 5 stays responsive with large
 * grids (up to 250 000 points per scan).
 */

export type Grid = (number | null)[][];

export interface DownsampledGrid {
  x: number[];
  y: number[];
  z: Grid;
  /** Block size along each axis (1 = unchanged). */
  factor: number;
}

/** Smallest integer block size f with ceil(nx/f) * ceil(ny/f) <= maxCells. */
export function blockFactor(nx: number, ny: number, maxCells: number): number {
  if (nx <= 0 || ny <= 0) return 1;
  const limit = Math.max(1, Math.floor(maxCells));
  let factor = Math.max(1, Math.floor(Math.sqrt((nx * ny) / limit)));
  while (Math.ceil(nx / factor) * Math.ceil(ny / factor) > limit) factor += 1;
  return factor;
}

function blockMean(values: readonly number[], start: number, end: number): number {
  let sum = 0;
  for (let i = start; i < end; i += 1) sum += values[i] ?? 0;
  return sum / (end - start);
}

/**
 * Reduce a row-major `[iy][ix]` grid to at most `maxCells` cells by blocks:
 * `mean` averages a block (heights, confidence), `max` keeps its largest
 * value (categorical maps ordered by severity). Missing cells (`null`) are
 * ignored; a block without any value stays `null`, so gaps are never filled
 * in. Coordinates become the block centres.
 */
export function downsampleGrid(
  x: readonly number[],
  y: readonly number[],
  z: Grid,
  maxCells: number,
  mode: "mean" | "max" = "mean",
): DownsampledGrid {
  const nx = x.length;
  const ny = y.length;
  const factor = blockFactor(nx, ny, maxCells);
  if (factor === 1) return { x: [...x], y: [...y], z: z.map((row) => [...row]), factor };
  const outX: number[] = [];
  for (let i = 0; i < nx; i += factor) outX.push(blockMean(x, i, Math.min(nx, i + factor)));
  const outY: number[] = [];
  for (let j = 0; j < ny; j += factor) outY.push(blockMean(y, j, Math.min(ny, j + factor)));
  const outZ: Grid = [];
  for (let j = 0; j < ny; j += factor) {
    const row: (number | null)[] = [];
    for (let i = 0; i < nx; i += factor) {
      let sum = 0;
      let max = -Infinity;
      let count = 0;
      for (let jj = j; jj < Math.min(ny, j + factor); jj += 1) {
        const source = z[jj];
        if (!source) continue;
        for (let ii = i; ii < Math.min(nx, i + factor); ii += 1) {
          const value = source[ii];
          if (typeof value === "number" && Number.isFinite(value)) {
            sum += value;
            if (value > max) max = value;
            count += 1;
          }
        }
      }
      row.push(count === 0 ? null : mode === "max" ? max : sum / count);
    }
    outZ.push(row);
  }
  return { x: outX, y: outY, z: outZ, factor };
}

/**
 * Down-sample a boolean mask with the same blocks as {@link downsampleGrid}:
 * a block is `true` when any of its cells is.
 */
export function downsampleMask(mask: readonly (readonly boolean[])[], factor: number): boolean[][] {
  if (factor <= 1) return mask.map((row) => [...row]);
  const ny = mask.length;
  const nx = mask[0]?.length ?? 0;
  const out: boolean[][] = [];
  for (let j = 0; j < ny; j += factor) {
    const row: boolean[] = [];
    for (let i = 0; i < nx; i += factor) {
      let any = false;
      for (let jj = j; jj < Math.min(ny, j + factor) && !any; jj += 1) {
        for (let ii = i; ii < Math.min(nx, i + factor); ii += 1) {
          if (mask[jj]?.[ii]) {
            any = true;
            break;
          }
        }
      }
      row.push(any);
    }
    out.push(row);
  }
  return out;
}

/** Keep at most `max` items by taking every k-th one (always keeps the first). */
export function strideSample<T>(items: readonly T[], max: number): T[] {
  if (max <= 0) return [];
  if (items.length <= max) return [...items];
  const step = Math.ceil(items.length / max);
  const out: T[] = [];
  for (let i = 0; i < items.length; i += step) {
    const item = items[i];
    if (item !== undefined) out.push(item);
  }
  return out;
}

/**
 * Reduce a curve to at most `maxPoints` points, keeping each bucket's minimum
 * and maximum (in original order), so a narrow focus peak survives. `null`
 * values are treated as missing; an all-missing bucket becomes one gap.
 */
export function minMaxDecimate(
  x: readonly number[],
  y: readonly (number | null)[],
  maxPoints: number,
): { x: number[]; y: (number | null)[] } {
  const n = Math.min(x.length, y.length);
  if (n <= maxPoints || maxPoints < 2) return { x: x.slice(0, n), y: y.slice(0, n) };
  const buckets = Math.floor(maxPoints / 2);
  const outX: number[] = [];
  const outY: (number | null)[] = [];
  for (let b = 0; b < buckets; b += 1) {
    const start = Math.floor((b * n) / buckets);
    const end = Math.floor(((b + 1) * n) / buckets);
    let minIndex = -1;
    let maxIndex = -1;
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (let i = start; i < end; i += 1) {
      const value = y[i];
      if (value === null || value === undefined || !Number.isFinite(value)) continue;
      if (value < minValue) {
        minValue = value;
        minIndex = i;
      }
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
    if (minIndex < 0) {
      outX.push(x[start] ?? 0);
      outY.push(null);
      continue;
    }
    const indices = minIndex === maxIndex ? [minIndex] : [Math.min(minIndex, maxIndex), Math.max(minIndex, maxIndex)];
    for (const index of indices) {
      outX.push(x[index] ?? 0);
      outY.push(y[index] ?? null);
    }
  }
  return { x: outX, y: outY };
}
