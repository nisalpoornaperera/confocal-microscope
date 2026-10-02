/**
 * Turns a `POST /api/v1/scans/estimate` result (or its error) into what
 * Scan Setup shows before a scan is started: the numbers, every limit
 * violation and warning, and whether the Start button may be enabled.
 */
import type { ApiError } from "../api/client";
import type { ScanEstimate } from "../api/types";
import { formatBytes, formatDuration, formatInteger } from "./format";

export interface EstimateRow {
  label: string;
  value: string;
}

export type EstimateSeverity = "ok" | "warning" | "blocked";

export interface EstimateView {
  rows: EstimateRow[];
  /** Problems that prevent starting (limit violations, invalid configuration). */
  blockers: string[];
  warnings: string[];
  severity: EstimateSeverity;
  canStart: boolean;
  /** One-line summary for the start button area. */
  headline: string;
}

export function describeEstimate(estimate: ScanEstimate): EstimateView {
  const rows: EstimateRow[] = [
    { label: "XY grid", value: `${formatInteger(estimate.n_x)} × ${formatInteger(estimate.n_y)}` },
    { label: "Total XY points", value: formatInteger(estimate.total_points) },
    { label: "Z positions per point", value: formatInteger(estimate.z_positions_per_point) },
    { label: "Total measurements", value: formatInteger(estimate.total_measurements) },
    { label: "ADC samples", value: formatInteger(estimate.total_adc_samples) },
    { label: "Estimated time", value: formatDuration(estimate.estimated_duration_s) },
    { label: "Estimated data size", value: formatBytes(estimate.estimated_data_bytes) },
  ];
  const blockers = [...(estimate.limit_violations ?? [])];
  if (!estimate.within_limits && blockers.length === 0) {
    blockers.push("the scan envelope leaves the stage travel limits");
  }
  const warnings = [...(estimate.warnings ?? [])];
  const severity: EstimateSeverity =
    blockers.length > 0 ? "blocked" : warnings.length > 0 ? "warning" : "ok";
  const headline =
    severity === "blocked"
      ? `Cannot start: ${blockers.length} limit violation${blockers.length === 1 ? "" : "s"}`
      : `${formatInteger(estimate.total_points)} points, about ${formatDuration(estimate.estimated_duration_s)}` +
        (warnings.length > 0 ? ` (${warnings.length} warning${warnings.length === 1 ? "" : "s"})` : "");
  return { rows, blockers, warnings, severity, canStart: blockers.length === 0, headline };
}

/**
 * An estimate request that failed: a 422 means the configuration itself is
 * invalid (every reason is in `violations`); anything else is shown as is.
 */
export function describeEstimateError(error: ApiError): EstimateView {
  const blockers =
    error.violations.length > 0 ? error.violations.map(cleanViolation) : [error.detail];
  const headline =
    error.status === 422
      ? "Cannot start: the scan settings are invalid"
      : `Cannot estimate the scan: ${error.detail}`;
  return { rows: [], blockers, warnings: [], severity: "blocked", canStart: false, headline };
}

/**
 * Validation messages look like `body.x_stop_um: Value error, x_stop_um must be >= x_start_um`;
 * drop the transport prefix so they read naturally.
 */
export function cleanViolation(message: string): string {
  return message
    .replace(/^body(\.[\w.]+)?: /, (_match: string, loc: string | undefined) =>
      loc ? `${loc.slice(1)}: ` : "",
    )
    .replace(/^([\w.]+: )?Value error, /, "$1")
    .replace(/^: /, "");
}
