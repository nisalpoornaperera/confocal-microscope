/**
 * Scan Setup form model: the user's text inputs <-> a `ScanConfig` request.
 *
 * Inputs stay strings so half-typed values ("-", "1.") are never rewritten;
 * {@link formToConfig} parses them and reports per-field problems. The backend
 * remains the authority: the live estimate returns every remaining violation.
 */
import type {
  InterpolationMethod,
  ReconstructionRequest,
  SamplingMethod,
  ScanConfig,
  ScanMode,
  ScanOrder,
} from "../api/types";
import { readStorage, writeStorage } from "./prefs";

/** Body of `POST /scans` and `/scans/estimate` (nested defaults filled in by the backend). */
export type ScanConfigBody = Omit<ScanConfig, "processing" | "reconstruction"> & {
  reconstruction?: Partial<ReconstructionRequest>;
};

export interface ScanForm {
  name: string;
  mode: ScanMode;
  x_start_um: string;
  x_stop_um: string;
  y_start_um: string;
  y_stop_um: string;
  xy_step_um: string;
  z_center_um: string;
  z_range_um: string;
  coarse_z_step_um: string;
  fine_z_step_um: string;
  fine_z_range_um: string;
  order: ScanOrder;
  adaptive_z: boolean;
  adaptive_z_range_um: string;
  samples_per_z: string;
  sampling_method: SamplingMethod;
  settle_time_ms: string;
  home_before_scan: boolean;
  calibrate_dark_before_scan: boolean;
  reconstruct_on_complete: boolean;
  reconstruction_method: InterpolationMethod;
  ml_on_complete: boolean;
}

/** Defaults: the backend's ScanConfig defaults and the README's first-scan example. */
export const DEFAULT_SCAN_FORM: ScanForm = {
  name: "",
  mode: "confocal",
  x_start_um: "0",
  x_stop_um: "200",
  y_start_um: "0",
  y_stop_um: "200",
  xy_step_um: "20",
  z_center_um: "0",
  z_range_um: "100",
  coarse_z_step_um: "2",
  fine_z_step_um: "0.25",
  fine_z_range_um: "12",
  order: "serpentine",
  adaptive_z: true,
  adaptive_z_range_um: "30",
  samples_per_z: "4",
  sampling_method: "mean",
  settle_time_ms: "10",
  home_before_scan: false,
  calibrate_dark_before_scan: false,
  reconstruct_on_complete: true,
  reconstruction_method: "linear",
  ml_on_complete: false,
};

export const NUMERIC_FIELDS = [
  "x_start_um",
  "x_stop_um",
  "y_start_um",
  "y_stop_um",
  "xy_step_um",
  "z_center_um",
  "z_range_um",
  "coarse_z_step_um",
  "fine_z_step_um",
  "fine_z_range_um",
  "adaptive_z_range_um",
  "samples_per_z",
  "settle_time_ms",
] as const satisfies readonly (keyof ScanForm)[];

export type NumericField = (typeof NUMERIC_FIELDS)[number];

export type FormErrors = Partial<Record<NumericField | "name", string>>;

export interface FormResult {
  config: ScanConfigBody | null;
  errors: FormErrors;
}

interface NumberRule {
  min?: number;
  /** Exclusive minimum. */
  above?: number;
  max?: number;
  integer?: boolean;
}

const RULES: Record<NumericField, NumberRule> = {
  x_start_um: {},
  x_stop_um: {},
  y_start_um: {},
  y_stop_um: {},
  xy_step_um: { above: 0 },
  z_center_um: {},
  z_range_um: { above: 0 },
  coarse_z_step_um: { above: 0 },
  fine_z_step_um: { above: 0 },
  fine_z_range_um: { above: 0 },
  adaptive_z_range_um: { above: 0 },
  samples_per_z: { min: 1, max: 256, integer: true },
  settle_time_ms: { min: 0, max: 10_000 },
};

/** Fields only used in confocal mode (ignored, not validated, for fixed-Z scans). */
export const CONFOCAL_ONLY: readonly NumericField[] = [
  "z_range_um",
  "coarse_z_step_um",
  "fine_z_step_um",
  "fine_z_range_um",
  "adaptive_z_range_um",
];

export function parseNumber(text: string, rule: NumberRule): { value?: number; error?: string } {
  const trimmed = text.trim();
  if (trimmed === "") return { error: "required" };
  const value = Number(trimmed);
  if (!Number.isFinite(value)) return { error: "not a number" };
  if (rule.integer && !Number.isInteger(value)) return { error: "must be a whole number" };
  if (rule.above !== undefined && !(value > rule.above)) return { error: `must be > ${rule.above}` };
  if (rule.min !== undefined && value < rule.min) return { error: `must be ≥ ${rule.min}` };
  if (rule.max !== undefined && value > rule.max) return { error: `must be ≤ ${rule.max}` };
  return { value };
}

export function formToConfig(form: ScanForm): FormResult {
  const errors: FormErrors = {};
  const values: Partial<Record<NumericField, number>> = {};
  const confocal = form.mode === "confocal";
  for (const key of NUMERIC_FIELDS) {
    const rule = RULES[key];
    if (!confocal && CONFOCAL_ONLY.includes(key)) {
      const parsed = parseNumber(form[key], rule);
      values[key] = parsed.value ?? Number(DEFAULT_SCAN_FORM[key]);
      continue;
    }
    const parsed = parseNumber(form[key], rule);
    if (parsed.error !== undefined) errors[key] = parsed.error;
    else values[key] = parsed.value;
  }
  const v = (key: NumericField): number => values[key] ?? Number.NaN;
  if (errors.x_stop_um === undefined && errors.x_start_um === undefined && v("x_stop_um") < v("x_start_um")) {
    errors.x_stop_um = "must be ≥ X start";
  }
  if (errors.y_stop_um === undefined && errors.y_start_um === undefined && v("y_stop_um") < v("y_start_um")) {
    errors.y_stop_um = "must be ≥ Y start";
  }
  if (form.name.length > 200) errors.name = "at most 200 characters";
  if (Object.keys(errors).length > 0) return { config: null, errors };

  const name = form.name.trim();
  const config: ScanConfigBody = {
    name: name === "" ? null : name,
    mode: form.mode,
    x_start_um: v("x_start_um"),
    x_stop_um: v("x_stop_um"),
    y_start_um: v("y_start_um"),
    y_stop_um: v("y_stop_um"),
    xy_step_um: v("xy_step_um"),
    z_center_um: v("z_center_um"),
    z_range_um: v("z_range_um"),
    coarse_z_step_um: v("coarse_z_step_um"),
    fine_z_step_um: v("fine_z_step_um"),
    fine_z_range_um: v("fine_z_range_um"),
    order: form.order,
    adaptive_z: form.adaptive_z,
    adaptive_z_range_um: v("adaptive_z_range_um"),
    samples_per_z: v("samples_per_z"),
    sampling_method: form.sampling_method,
    settle_time_ms: v("settle_time_ms"),
    home_before_scan: form.home_before_scan,
    calibrate_dark_before_scan: form.calibrate_dark_before_scan,
    reconstruct_on_complete: confocal && form.reconstruct_on_complete,
    reconstruction: { method: form.reconstruction_method },
    ml_on_complete: confocal && form.ml_on_complete,
  };
  return { config, errors };
}

/** Pre-fill the form from an earlier scan's configuration ("repeat scan"). */
export function configToForm(config: ScanConfig): ScanForm {
  const s = (value: number): string => String(value);
  return {
    name: config.name ?? "",
    mode: config.mode,
    x_start_um: s(config.x_start_um),
    x_stop_um: s(config.x_stop_um),
    y_start_um: s(config.y_start_um),
    y_stop_um: s(config.y_stop_um),
    xy_step_um: s(config.xy_step_um),
    z_center_um: s(config.z_center_um),
    z_range_um: s(config.z_range_um),
    coarse_z_step_um: s(config.coarse_z_step_um),
    fine_z_step_um: s(config.fine_z_step_um),
    fine_z_range_um: s(config.fine_z_range_um),
    order: config.order,
    adaptive_z: config.adaptive_z,
    adaptive_z_range_um: s(config.adaptive_z_range_um),
    samples_per_z: s(config.samples_per_z),
    sampling_method: config.sampling_method,
    settle_time_ms: s(config.settle_time_ms),
    home_before_scan: config.home_before_scan,
    calibrate_dark_before_scan: config.calibrate_dark_before_scan,
    reconstruct_on_complete: config.reconstruct_on_complete,
    reconstruction_method: config.reconstruction?.method ?? "linear",
    ml_on_complete: config.ml_on_complete,
  };
}

const ENUM_FIELDS: Record<string, readonly string[] | undefined> = {
  mode: ["confocal", "fixed_z"],
  order: ["serpentine", "raster"],
  sampling_method: ["mean", "median"],
  reconstruction_method: ["nearest", "linear", "cubic", "rbf"],
};

export const FORM_STORAGE_KEY = "confocal.ui.scanForm.v1";

/** The last form used in this browser (validated against the current shape). */
export function loadStoredForm(): ScanForm {
  const raw = readStorage(FORM_STORAGE_KEY);
  if (!raw) return DEFAULT_SCAN_FORM;
  try {
    const stored = JSON.parse(raw) as unknown;
    if (typeof stored !== "object" || stored === null) return DEFAULT_SCAN_FORM;
    const record = stored as Record<string, unknown>;
    const merged: Record<string, unknown> = { ...DEFAULT_SCAN_FORM };
    for (const [key, fallback] of Object.entries(DEFAULT_SCAN_FORM)) {
      const value = record[key];
      if (typeof value !== typeof fallback) continue;
      const allowed = ENUM_FIELDS[key];
      if (allowed !== undefined && !allowed.includes(value as string)) continue;
      merged[key] = value;
    }
    return merged as unknown as ScanForm;
  } catch {
    return DEFAULT_SCAN_FORM;
  }
}

export function storeForm(form: ScanForm): void {
  writeStorage(FORM_STORAGE_KEY, JSON.stringify(form));
}
