/**
 * Short names for the backend's Pydantic models, taken from the generated
 * OpenAPI types (`schema.ts`, regenerate with `npm run gen:api`).
 */
import type { components } from "./schema";

type Schemas = components["schemas"];

export type ADCCalibrateRequest = Schemas["ADCCalibrateRequest"];
export type ADCCalibrateResponse = Schemas["ADCCalibrateResponse"];
export type ADCStatus = Schemas["ADCStatus"];
export type AdcGain = Schemas["AdcGain"];
export type Axis = Schemas["Axis"];
export type AxisLimits = Schemas["AxisLimits"];
export type CalibrationState = Schemas["CalibrationState"];
export type CrossSection = Schemas["CrossSection"];
export type DarkCalibrationRequest = Schemas["DarkCalibrationRequest"];
export type ErrorResponse = Schemas["ErrorResponse"];
export type GapRegion = Schemas["GapRegion"];
export type HardwareInfo = Schemas["HardwareInfo"];
export type HardwareStatus = Schemas["HardwareStatus"];
export type IntensityMeasurement = Schemas["IntensityMeasurement"];
export type InterpolationMethod = Schemas["InterpolationMethod"];
export type LaserStatus = Schemas["LaserStatus"];
export type MLModelInfo = Schemas["MLModelInfo"];
export type MLResult = Schemas["MLResult"];
export type OutlierMethod = Schemas["OutlierMethod"];
export type PeakFit = Schemas["PeakFit"];
export type PointClassification = Schemas["PointClassification"];
export type PointStatus = Schemas["PointStatus"];
export type Position = Schemas["Position"];
export type ProcessingConfig = Schemas["ProcessingConfig"];
export type ProfileAnalysis = Schemas["ProfileAnalysis"];
export type ProfileRecord = Schemas["ProfileRecord"];
export type ReconstructionRequest = Schemas["ReconstructionRequest"];
export type ReferenceCalibrationRequest = Schemas["ReferenceCalibrationRequest"];
export type SamplingMethod = Schemas["SamplingMethod"];
export type ScanConfig = Schemas["ScanConfig"];
export type ScanEstimate = Schemas["ScanEstimate"];
export type ScanMode = Schemas["ScanMode"];
export type ScanOrder = Schemas["ScanOrder"];
export type ScanPoint = Schemas["ScanPoint"];
export type ScanState = Schemas["ScanState"];
export type ScanSummary = Schemas["ScanSummary"];
export type StageLimits = Schemas["StageLimits"];
export type StageMoveRequest = Schemas["StageMoveRequest"];
export type StageState = Schemas["StageState"];
export type StageStatus = Schemas["StageStatus"];
export type SurfacePoint = Schemas["SurfacePoint"];
export type SurfaceResult = Schemas["SurfaceResult"];
export type SurfaceStatistics = Schemas["SurfaceStatistics"];
export type SystemInfo = Schemas["SystemInfo"];
export type SystemStatus = Schemas["SystemStatus"];

/** Request bodies whose fields all have server-side defaults: send only what changes. */
export type DarkCalibrationBody = Partial<DarkCalibrationRequest>;
export type ReferenceCalibrationBody = Partial<ReferenceCalibrationRequest>;
export type ReconstructionBody = Partial<ReconstructionRequest>;
export type ADCCalibrateBody = Partial<ADCCalibrateRequest>;

/** States in which a scan owns the instrument (mirrors ACTIVE_SCAN_STATES). */
export const ACTIVE_SCAN_STATES: readonly ScanState[] = [
  "preparing",
  "calibrating",
  "homing",
  "scanning",
  "paused",
  "processing",
  "surface_reconstruction",
  "ml_processing",
];

/** States a scan never leaves (mirrors TERMINAL_SCAN_STATES). */
export const TERMINAL_SCAN_STATES: readonly ScanState[] = ["cancelled", "error", "complete"];

export function isActiveState(state: ScanState | null | undefined): boolean {
  return state != null && ACTIVE_SCAN_STATES.includes(state);
}

export function isTerminalState(state: ScanState | null | undefined): boolean {
  return state != null && TERMINAL_SCAN_STATES.includes(state);
}

export const ADC_GAINS: readonly AdcGain[] = ["2/3", "1", "2", "4", "8", "16"];

/** ADS1115 full-scale range (+/- V) of each PGA gain (mirrors AdcGain.full_scale_v). */
export const ADC_FULL_SCALE_V: Readonly<Record<AdcGain, number>> = {
  "2/3": 6.144,
  "1": 4.096,
  "2": 2.048,
  "4": 1.024,
  "8": 0.512,
  "16": 0.256,
};

export const INTERPOLATION_METHODS: readonly InterpolationMethod[] = [
  "nearest",
  "linear",
  "cubic",
  "rbf",
];
