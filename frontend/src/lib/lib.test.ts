import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import type { ScanEstimate } from "../api/types";
import { point } from "../test/fixtures";
import { blockFactor, downsampleGrid, downsampleMask, minMaxDecimate, strideSample } from "./downsample";
import { cleanViolation, describeEstimate, describeEstimateError } from "./estimate";
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatInteger,
  formatLength,
  formatNumber,
  formatPercent,
  formatVoltage,
  humanize,
  MISSING,
} from "./format";
import { axisCount, axisValues, ScanGrid, STATUS_CATEGORY } from "./grid";

describe("format", () => {
  it("shows a dash for missing values", () => {
    for (const fn of [formatNumber, formatInteger, formatPercent, formatVoltage, formatDuration, formatBytes]) {
      expect(fn(null)).toBe(MISSING);
      expect(fn(undefined)).toBe(MISSING);
      expect(fn(Number.NaN)).toBe(MISSING);
      expect(fn(Number.POSITIVE_INFINITY)).toBe(MISSING);
    }
    expect(formatLength(null)).toBe(MISSING);
    expect(formatDateTime(null)).toBe(MISSING);
    expect(formatDateTime("not a date")).toBe(MISSING);
  });

  it("formats numbers and lengths", () => {
    expect(formatNumber(1234.5678, 2)).toBe("1,234.57");
    expect(formatInteger(250000)).toBe("250,000");
    expect(formatLength(12.345)).toBe("12.35 µm");
    expect(formatLength(1500, "mm")).toBe("1.5000 mm");
    expect(formatLength(-0.5, "um", 3)).toBe("-0.500 µm");
    expect(formatPercent(0.4567)).toBe("45.7 %");
  });

  it("formats voltages, switching to mV for small values", () => {
    expect(formatVoltage(1.95)).toBe("1.9500 V");
    expect(formatVoltage(0.0123)).toBe("12.30 mV");
    expect(formatVoltage(0)).toBe("0.0000 V");
  });

  it("formats durations", () => {
    expect(formatDuration(0.25)).toBe("250 ms");
    expect(formatDuration(4.25)).toBe("4.3 s");
    expect(formatDuration(42)).toBe("42 s");
    expect(formatDuration(59.7)).toBe("1 min 00 s");
    expect(formatDuration(185)).toBe("3 min 05 s");
    expect(formatDuration(7620)).toBe("2 h 07 min");
    expect(formatDuration(3 * 86400 + 4 * 3600 + 10)).toBe("3 d 4 h");
    expect(formatDuration(-1)).toBe(MISSING);
  });

  it("formats byte counts", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.50 KiB");
    expect(formatBytes(50 * 1024 * 1024)).toBe("50.0 MiB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.00 GiB");
  });

  it("humanizes enum values", () => {
    expect(humanize("surface_reconstruction")).toBe("Surface reconstruction");
    expect(humanize("")).toBe(MISSING);
  });
});

describe("downsampling", () => {
  it("chooses the smallest sufficient block factor", () => {
    expect(blockFactor(100, 100, 10_000)).toBe(1);
    expect(blockFactor(100, 100, 2500)).toBe(2);
    expect(blockFactor(101, 101, 2500)).toBe(3);
    expect(blockFactor(500, 500, 40_000)).toBe(3);
    expect(blockFactor(0, 10, 5)).toBe(1);
  });

  it("returns the grid unchanged when it is small enough", () => {
    const z = [
      [1, 2],
      [3, null],
    ];
    const out = downsampleGrid([0, 1], [0, 1], z, 4);
    expect(out.factor).toBe(1);
    expect(out.z).toEqual(z);
    expect(out.z).not.toBe(z);
  });

  it("block-averages, ignoring gaps, and keeps all-gap blocks as gaps", () => {
    const z = [
      [1, 3, null, null],
      [5, 7, null, null],
      [2, null, 4, 4],
      [null, null, 4, 4],
    ];
    const out = downsampleGrid([0, 1, 2, 3], [0, 10, 20, 30], z, 4);
    expect(out.factor).toBe(2);
    expect(out.x).toEqual([0.5, 2.5]);
    expect(out.y).toEqual([5, 25]);
    expect(out.z).toEqual([
      [4, null],
      [2, 4],
    ]);
  });

  it("keeps the largest value per block in max mode", () => {
    const z = [
      [1, 4, 0, null],
      [3, 1, null, null],
    ];
    expect(downsampleGrid([0, 1, 2, 3], [0, 1], z, 2, "max").z).toEqual([[4, 0]]);
  });

  it("handles ragged edge blocks", () => {
    const z = [[1, 2, 3]];
    const out = downsampleGrid([0, 1, 2], [0], z, 2);
    expect(out.factor).toBe(2);
    expect(out.x).toEqual([0.5, 2]);
    expect(out.z).toEqual([[1.5, 3]]);
  });

  it("downsamples masks with any-semantics", () => {
    const mask = [
      [false, true, false],
      [false, false, false],
    ];
    expect(downsampleMask(mask, 2)).toEqual([[true, false]]);
    expect(downsampleMask(mask, 1)).toEqual(mask);
  });

  it("stride-samples", () => {
    expect(strideSample([1, 2, 3, 4, 5], 10)).toEqual([1, 2, 3, 4, 5]);
    expect(strideSample([1, 2, 3, 4, 5], 2)).toEqual([1, 4]);
    expect(strideSample([1, 2, 3], 0)).toEqual([]);
  });

  it("keeps a narrow peak through min/max decimation", () => {
    const n = 1000;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = x.map((i) => (i === 517 ? 10 : 0.1));
    const out = minMaxDecimate(x, y, 100);
    expect(out.x.length).toBeLessThanOrEqual(100);
    expect(Math.max(...out.y.map((v) => v ?? 0))).toBe(10);
    expect(out.x).toContain(517);
    // Monotonic x order is preserved.
    for (let i = 1; i < out.x.length; i += 1) {
      expect(out.x[i] ?? 0).toBeGreaterThanOrEqual(out.x[i - 1] ?? 0);
    }
  });

  it("returns short curves unchanged and turns empty buckets into gaps", () => {
    expect(minMaxDecimate([1, 2], [3, 4], 10)).toEqual({ x: [1, 2], y: [3, 4] });
    const out = minMaxDecimate([0, 1, 2, 3, 4, 5, 6, 7], [null, null, null, null, 1, 2, 3, 4], 4);
    expect(out.y[0]).toBeNull();
    expect(out.y).toContain(4);
  });
});

describe("grid", () => {
  it("counts axis positions like the backend", () => {
    expect(axisCount(0, 200, 20)).toBe(11);
    expect(axisCount(0, 0, 5)).toBe(1);
    expect(axisCount(0, 0.3, 0.1)).toBe(4); // floating-point tolerance
    expect(axisCount(0, 19.99, 20)).toBe(1);
    expect(axisCount(10, 0, 1)).toBe(0);
    expect(axisCount(0, 10, 0)).toBe(0);
    expect(axisValues(-10, 10, 10)).toEqual([-10, 0, 10]);
  });

  it("fills the completion and confidence maps", () => {
    const grid = new ScanGrid({ x_start_um: 0, x_stop_um: 10, y_start_um: 0, y_stop_um: 10, xy_step_um: 10 });
    expect(grid.nx).toBe(2);
    expect(grid.ny).toBe(2);
    expect(grid.countOf(STATUS_CATEGORY.pending)).toBe(4);
    grid.add(point({ point_id: 0, ix: 0, iy: 0, status: "valid", confidence: 0.8, surface_z_um: 2 }));
    grid.add(point({ point_id: 1, ix: 1, iy: 0, status: "no_peak", confidence: 0.1, surface_z_um: null }));
    grid.add(point({ point_id: 2, ix: 1, iy: 1, status: "aborted", confidence: 0 }));
    expect(grid.add(point({ point_id: 9, ix: 5, iy: 0 }))).toBe(false);
    expect(grid.measured).toBe(3);
    expect(grid.countOf(STATUS_CATEGORY.pending)).toBe(1);
    expect(grid.countOf(STATUS_CATEGORY.valid)).toBe(1);
    expect(grid.categoryMatrix()).toEqual([
      [STATUS_CATEGORY.valid, STATUS_CATEGORY.failed],
      [STATUS_CATEGORY.pending, STATUS_CATEGORY.aborted],
    ]);
    expect(grid.matrix(grid.confidence)).toEqual([
      [0.8, 0.1],
      [null, null],
    ]);
    expect(grid.matrix(grid.height)).toEqual([
      [2, null],
      [null, null],
    ]);
    // Re-adding a point replaces it without double counting.
    grid.add(point({ point_id: 1, ix: 1, iy: 0, status: "valid", confidence: 0.7 }));
    expect(grid.countOf(STATUS_CATEGORY.failed)).toBe(0);
    expect(grid.countOf(STATUS_CATEGORY.valid)).toBe(2);
    expect(grid.measured).toBe(3);
  });
});

const baseEstimate: ScanEstimate = {
  n_x: 11,
  n_y: 11,
  total_points: 121,
  z_positions_per_point: 99,
  total_measurements: 11979,
  total_adc_samples: 47916,
  estimated_duration_s: 3725,
  estimated_data_bytes: 5 * 1024 * 1024,
  within_limits: true,
  limit_violations: [],
  warnings: [],
};

describe("estimate display", () => {
  it("lists the numbers and allows starting a clean estimate", () => {
    const view = describeEstimate(baseEstimate);
    expect(view.canStart).toBe(true);
    expect(view.severity).toBe("ok");
    const rows = Object.fromEntries(view.rows.map((r) => [r.label, r.value]));
    expect(rows["XY grid"]).toBe("11 × 11");
    expect(rows["Total XY points"]).toBe("121");
    expect(rows["Total measurements"]).toBe("11,979");
    expect(rows["Estimated time"]).toBe("1 h 02 min");
    expect(rows["Estimated data size"]).toBe("5.00 MiB");
    expect(view.headline).toBe("121 points, about 1 h 02 min");
  });

  it("shows warnings without blocking", () => {
    const view = describeEstimate({ ...baseEstimate, warnings: ["long scan"] });
    expect(view.canStart).toBe(true);
    expect(view.severity).toBe("warning");
    expect(view.warnings).toEqual(["long scan"]);
    expect(view.headline).toContain("1 warning");
  });

  it("blocks on limit violations", () => {
    const view = describeEstimate({
      ...baseEstimate,
      within_limits: false,
      limit_violations: ["x=6000 um outside [-5000, 5000] um", "z sweep leaves the limits"],
    });
    expect(view.canStart).toBe(false);
    expect(view.severity).toBe("blocked");
    expect(view.blockers).toHaveLength(2);
    expect(view.headline).toBe("Cannot start: 2 limit violations");
  });

  it("blocks when out of limits even without a listed violation", () => {
    const view = describeEstimate({ ...baseEstimate, within_limits: false });
    expect(view.canStart).toBe(false);
    expect(view.blockers).toHaveLength(1);
  });

  it("explains an invalid configuration (422)", () => {
    const error = new ApiError(422, "RequestValidationError", "request validation failed (2 error(s))", [
      "body: Value error, fine_z_step_um must be <= coarse_z_step_um",
      "body.xy_step_um: Input should be greater than 0",
    ]);
    const view = describeEstimateError(error);
    expect(view.canStart).toBe(false);
    expect(view.blockers).toEqual([
      "fine_z_step_um must be <= coarse_z_step_um",
      "xy_step_um: Input should be greater than 0",
    ]);
    expect(view.headline).toContain("invalid");
  });

  it("explains other estimate failures", () => {
    const view = describeEstimateError(new ApiError(0, "NetworkError", "cannot reach the backend"));
    expect(view.blockers).toEqual(["cannot reach the backend"]);
    expect(view.headline).toContain("cannot reach the backend");
  });

  it("cleans validation prefixes", () => {
    expect(cleanViolation("body.processing.filter_window: Value error, filter_window must be odd")).toBe(
      "processing.filter_window: filter_window must be odd",
    );
    expect(cleanViolation("plain message")).toBe("plain message");
  });
});
