import { describe, expect, it } from "vitest";

import { DEFAULT_PREFERENCES, parsePreferences } from "./prefs";
import { applyDetectionPreset, configToForm, DEFAULT_SCAN_FORM, formToConfig, parseNumber } from "./scanForm";
import { autoScale, crossSection, sceneAspect, scaleToSlider, sliderToScale } from "./surface";

describe("scan form", () => {
  it("turns the defaults into a valid scan request", () => {
    const { config, errors } = formToConfig(DEFAULT_SCAN_FORM);
    expect(errors).toEqual({});
    expect(config).toMatchObject({
      name: null,
      mode: "confocal",
      x_stop_um: 200,
      xy_step_um: 20,
      fine_z_step_um: 0.25,
      samples_per_z: 4,
      reconstruction: { method: "linear" },
    });
  });

  it("reports per-field problems", () => {
    const { config, errors } = formToConfig({
      ...DEFAULT_SCAN_FORM,
      xy_step_um: "0",
      x_stop_um: "-5",
      samples_per_z: "2.5",
      settle_time_ms: "",
    });
    expect(config).toBeNull();
    expect(errors.xy_step_um).toBe("must be > 0");
    expect(errors.x_stop_um).toBe("must be ≥ X start");
    expect(errors.samples_per_z).toBe("must be a whole number");
    expect(errors.settle_time_ms).toBe("required");
  });

  it("ignores confocal-only fields in fixed-Z mode and disables reconstruction", () => {
    const { config, errors } = formToConfig({ ...DEFAULT_SCAN_FORM, mode: "fixed_z", z_range_um: "" });
    expect(errors).toEqual({});
    expect(config?.reconstruct_on_complete).toBe(false);
    expect(config?.z_range_um).toBe(100);
  });

  it("parses numbers strictly", () => {
    expect(parseNumber(" 12.5 ", {})).toEqual({ value: 12.5 });
    expect(parseNumber("abc", {}).error).toBe("not a number");
    expect(parseNumber("300", { max: 256 }).error).toBe("must be ≤ 256");
  });

  it("round-trips a stored scan configuration", () => {
    const { config } = formToConfig({ ...DEFAULT_SCAN_FORM, name: "repeat me", order: "raster" });
    if (!config) throw new Error("invalid");
    const form = configToForm({ ...config, processing: undefined, reconstruction: undefined });
    expect(form.name).toBe("repeat me");
    expect(form.order).toBe("raster");
    expect(form.xy_step_um).toBe("20");
    expect(form.fine_scan).toBe(true);
  });

  it("turning the fine scan off ignores invalid fine fields", () => {
    const bad = { ...DEFAULT_SCAN_FORM, fine_z_step_um: "", fine_z_range_um: "abc" };
    expect(formToConfig(bad).config).toBeNull();
    const { config, errors } = formToConfig({ ...bad, fine_scan: false });
    expect(errors).toEqual({});
    expect(config?.fine_scan).toBe(false);
  });

  it("applies the low-contrast detection preset", () => {
    const form = applyDetectionPreset(DEFAULT_SCAN_FORM, "low_contrast");
    expect(form.detection_preset).toBe("low_contrast");
    const { config } = formToConfig(form);
    expect(config?.processing).toEqual({
      accept_weak_peaks: true,
      peak_selection: "highest",
      min_snr: 2,
      min_relative_prominence: 0.05,
      min_confidence: 0.2,
    });
    expect(config?.reconstruction?.min_confidence).toBe(0.1);
    const back = applyDetectionPreset(form, "standard");
    expect(formToConfig(back).config?.processing?.accept_weak_peaks).toBe(false);
    expect(formToConfig(back).config?.processing?.min_snr).toBe(5);
    expect(formToConfig(back).config?.processing?.peak_selection).toBe("most_prominent");
  });

  it("rejects out-of-range detection thresholds", () => {
    const { config, errors } = formToConfig({ ...DEFAULT_SCAN_FORM, min_relative_prominence: "2" });
    expect(config).toBeNull();
    expect(errors.min_relative_prominence).toBe("must be ≤ 1");
  });
});

describe("preferences", () => {
  it("falls back to defaults for missing or invalid storage", () => {
    expect(parsePreferences(null)).toEqual(DEFAULT_PREFERENCES);
    expect(parsePreferences("{not json")).toEqual(DEFAULT_PREFERENCES);
    const parsed = parsePreferences(JSON.stringify({ theme: "dark", colorMap: "Nope", maxDisplayCells: 10 }));
    expect(parsed.theme).toBe("dark");
    expect(parsed.colorMap).toBe(DEFAULT_PREFERENCES.colorMap);
    expect(parsed.maxDisplayCells).toBe(2500);
  });
});

describe("surface helpers", () => {
  const surface = {
    x_um: [0, 10, 20],
    y_um: [0, 5],
    z_um: [
      [1, null, 3],
      [4, 5, 6],
    ],
  };

  it("extracts cross-sections with gaps", () => {
    expect(crossSection(surface, "x", 0)).toEqual({ coordinate: [0, 10, 20], z: [1, null, 3], position: 0 });
    expect(crossSection(surface, "y", 1)).toEqual({ coordinate: [0, 5], z: [null, 5], position: 10 });
    expect(crossSection(surface, "x", 99).position).toBe(5);
  });

  it("maps the vertical-scale slider logarithmically", () => {
    expect(sliderToScale(0)).toBe(1);
    expect(sliderToScale(100)).toBe(1000);
    expect(sliderToScale(scaleToSlider(10))).toBeCloseTo(10, 5);
  });

  it("computes scene aspect ratios", () => {
    const aspect = sceneAspect(200, 100, 2, 10);
    expect(aspect.x).toBe(1);
    expect(aspect.y).toBe(0.5);
    expect(aspect.z).toBeCloseTo(0.1);
    expect(autoScale(200, 200, 2)).toBe(35);
    expect(autoScale(200, 200, 0)).toBe(1);
  });
});
