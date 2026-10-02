/** Test data builders shared by the unit tests. */
import type { ScanEvent, ScanEventType, ScanProgress } from "../api/events";
import type { ScanPoint, ScanState } from "../api/types";

export function progress(overrides: Partial<ScanProgress> = {}): ScanProgress {
  return {
    scan_id: "scan-1",
    state: "scanning",
    progress: 0,
    completed_points: 0,
    total_points: 4,
    current_point_id: null,
    current_x_um: null,
    current_y_um: null,
    current_z_um: null,
    current_intensity: null,
    elapsed_s: 0,
    estimated_remaining_s: null,
    message: null,
    ...overrides,
  };
}

export function point(overrides: Partial<ScanPoint> = {}): ScanPoint {
  return {
    point_id: 0,
    ix: 0,
    iy: 0,
    x_um: 0,
    y_um: 0,
    status: "valid",
    z_estimate_um: null,
    coarse_peak_z_um: null,
    surface_z_um: 1.5,
    parabolic_z_um: null,
    gaussian_z_um: null,
    peak_intensity: null,
    snr: null,
    peak_width_um: null,
    prominence: null,
    fit_residual: null,
    confidence: 0.9,
    intensity: null,
    secondary_peak_ratio: null,
    asymmetry: null,
    n_z_positions: 10,
    flags: [],
    acquired_at: "2026-10-02T10:00:00Z",
    duration_s: 1,
    ...overrides,
  };
}

export function event(
  type: ScanEventType,
  state: ScanState = "scanning",
  overrides: Partial<ScanEvent> = {},
): ScanEvent {
  return {
    type,
    scan_id: "scan-1",
    timestamp: "2026-10-02T10:00:00Z",
    progress: progress({ state }),
    point: null,
    profile: null,
    message: null,
    ...overrides,
  };
}
