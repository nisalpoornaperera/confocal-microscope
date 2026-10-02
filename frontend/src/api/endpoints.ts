/**
 * One typed function per backend endpoint (see the README's API table).
 * Request / response types come from the generated OpenAPI schema.
 */
import { get, post, type RequestOptions } from "./client";
import type {
  ADCCalibrateBody,
  ADCCalibrateResponse,
  ADCStatus,
  Axis,
  CalibrationState,
  DarkCalibrationBody,
  HardwareStatus,
  IntensityMeasurement,
  MLModelInfo,
  MLResult,
  Position,
  ProfileRecord,
  ReconstructionBody,
  ReferenceCalibrationBody,
  SamplingMethod,
  ScanEstimate,
  ScanPoint,
  ScanSummary,
  StageMoveRequest,
  SurfaceResult,
  SystemInfo,
  SystemStatus,
} from "./types";
import type { ScanConfigBody } from "../lib/scanForm";

type Opts = Pick<RequestOptions, "signal" | "timeoutMs">;

const id = (value: string): string => encodeURIComponent(value);

/** Reconstruction (RBF on large grids) and reference Z search can take minutes on a Pi. */
const LONG_TIMEOUT_MS = 10 * 60_000;

export const api = {
  // ---- system
  systemInfo: (o?: Opts) => get<SystemInfo>("/system", o),
  systemStatus: (o?: Opts) => get<SystemStatus>("/system/status", o),

  // ---- stage
  stagePosition: (o?: Opts) => get<Position>("/stage/position", o),
  stageMove: (body: Partial<StageMoveRequest>, o?: Opts) =>
    post<Position>("/stage/move", body, { timeoutMs: LONG_TIMEOUT_MS, ...o }),
  stageHome: (axes?: Axis[], o?: Opts) =>
    post<Position>("/stage/home", axes ? { axes } : undefined, { timeoutMs: LONG_TIMEOUT_MS, ...o }),
  /** EMERGENCY STOP: always accepted, latches until {@link api.stageReset}. */
  emergencyStop: (reason = "operator emergency stop (UI)", o?: Opts) =>
    post<HardwareStatus>("/stage/stop", { reason }, { timeoutMs: 10_000, ...o }),
  stageReset: (o?: Opts) => post<HardwareStatus>("/stage/reset", undefined, o),

  // ---- adc
  adcStatus: (o?: Opts) => get<ADCStatus>("/adc/status", o),
  adcRead: (nSamples: number, method: SamplingMethod, o?: Opts) =>
    get<IntensityMeasurement>("/adc/read", { query: { n_samples: nSamples, method }, ...o }),
  adcCalibrate: (body: ADCCalibrateBody, o?: Opts) =>
    post<ADCCalibrateResponse>("/adc/calibrate", body, o),

  // ---- calibration
  calibration: (o?: Opts) => get<CalibrationState>("/calibration", o),
  calibrationHistory: (limit = 50, o?: Opts) =>
    get<CalibrationState[]>("/calibration/history", { query: { limit }, ...o }),
  calibrateDark: (body: DarkCalibrationBody, o?: Opts) =>
    post<CalibrationState>("/calibration/dark", body, { timeoutMs: LONG_TIMEOUT_MS, ...o }),
  calibrateReference: (body: ReferenceCalibrationBody, o?: Opts) =>
    post<CalibrationState>("/calibration/reference", body, { timeoutMs: LONG_TIMEOUT_MS, ...o }),

  // ---- scans
  createScan: (config: ScanConfigBody, o?: Opts) => post<ScanSummary>("/scans", config, o),
  estimateScan: (config: ScanConfigBody, o?: Opts) =>
    post<ScanEstimate>("/scans/estimate", config, o),
  listScans: (limit = 100, offset = 0, o?: Opts) =>
    get<ScanSummary[]>("/scans", { query: { limit, offset }, ...o }),
  getScan: (scanId: string, o?: Opts) => get<ScanSummary>(`/scans/${id(scanId)}`, o),
  pauseScan: (scanId: string, o?: Opts) => post<ScanSummary>(`/scans/${id(scanId)}/pause`, undefined, o),
  resumeScan: (scanId: string, o?: Opts) =>
    post<ScanSummary>(`/scans/${id(scanId)}/resume`, undefined, o),
  cancelScan: (scanId: string, o?: Opts) =>
    post<ScanSummary>(`/scans/${id(scanId)}/cancel`, undefined, o),
  /** Points with `point_id > since` (all when `since` is omitted), ordered by point_id. */
  scanPoints: (scanId: string, since?: number, limit?: number, o?: Opts) =>
    get<ScanPoint[]>(`/scans/${id(scanId)}/points`, { query: { since, limit }, ...o }),
  pointProfile: (scanId: string, pointId: number, o?: Opts) =>
    get<ProfileRecord>(`/scans/${id(scanId)}/profile/${pointId}`, o),
  reconstruct: (scanId: string, body?: ReconstructionBody, o?: Opts) =>
    post<SurfaceResult>(`/scans/${id(scanId)}/reconstruct`, body, { timeoutMs: LONG_TIMEOUT_MS, ...o }),
  surface: (scanId: string, o?: Opts) =>
    get<SurfaceResult>(`/scans/${id(scanId)}/surface`, { timeoutMs: 120_000, ...o }),

  // ---- ml
  analyseMl: (scanId: string, body?: { model_name?: string | null; threshold?: number }, o?: Opts) =>
    post<MLResult>(`/scans/${id(scanId)}/ml/analyse`, body, { timeoutMs: LONG_TIMEOUT_MS, ...o }),
  mlResults: (scanId: string, o?: Opts) => get<MLResult>(`/scans/${id(scanId)}/ml/results`, o),
  mlModels: (o?: Opts) => get<MLModelInfo[]>("/ml/models", o),
};

/** Page size used when loading every point of a scan (max 250 000 points per scan). */
export const POINTS_PAGE_SIZE = 5000;

/**
 * Fetch every point with `point_id > since`, page by page.
 * `onPage` receives each page as it arrives (lets large scans render progressively).
 */
export async function fetchAllPoints(
  scanId: string,
  since: number | undefined,
  onPage: (points: ScanPoint[]) => void,
  signal?: AbortSignal,
): Promise<void> {
  let cursor = since;
  for (;;) {
    const page = await api.scanPoints(scanId, cursor, POINTS_PAGE_SIZE, { signal, timeoutMs: 60_000 });
    if (page.length > 0) onPage(page);
    const last = page[page.length - 1];
    if (page.length < POINTS_PAGE_SIZE || last === undefined) return;
    cursor = last.point_id;
  }
}
