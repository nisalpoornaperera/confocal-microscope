/**
 * 2 Scan Setup: every scan parameter, a live (debounced) server estimate
 * with limit violations and warnings BEFORE starting, and Start -> Live Scan.
 */
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { api } from "../api/endpoints";
import { INTERPOLATION_METHODS, type ScanConfig, type ScanEstimate } from "../api/types";
import {
  Alert,
  Card,
  CheckField,
  ErrorBox,
  Field,
  NumberField,
  PageHeader,
  SelectField,
} from "../components/ui";
import { useAction, useApiData, useDebounced } from "../hooks/useApi";
import { describeEstimate, describeEstimateError, type EstimateView } from "../lib/estimate";
import { formatLength } from "../lib/format";
import {
  applyDetectionPreset,
  configToForm,
  DEFAULT_SCAN_FORM,
  formToConfig,
  loadStoredForm,
  storeForm,
  type DetectionPreset,
  type NumericField,
  type PeakSelection,
  type ScanConfigBody,
  type ScanForm,
} from "../lib/scanForm";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";

/** Delay after the last keystroke before the estimate is requested. */
const ESTIMATE_DEBOUNCE_MS = 400;

function EstimatePanel({
  view,
  loading,
  stale,
}: {
  view: EstimateView | null;
  loading: boolean;
  stale: boolean;
}) {
  if (!view) {
    return <p className="muted">{loading ? "Estimating…" : "Fix the highlighted fields to get an estimate."}</p>;
  }
  return (
    <div className="stack" aria-live="polite" style={{ opacity: stale ? 0.6 : 1 }}>
      {view.rows.length > 0 && (
        <dl className="kv">
          {view.rows.map((row) => (
            <div key={row.label} style={{ display: "contents" }}>
              <dt>{row.label}</dt>
              <dd>
                <strong>{row.value}</strong>
              </dd>
            </div>
          ))}
        </dl>
      )}
      {view.blockers.length > 0 && (
        <Alert kind="error" title={view.headline}>
          <ul>
            {view.blockers.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </Alert>
      )}
      {view.warnings.length > 0 && (
        <Alert kind="warning" title="Warnings">
          <ul>
            {view.warnings.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </Alert>
      )}
      {view.blockers.length === 0 && view.warnings.length === 0 && (
        <Alert kind="ok">Within the travel limits. {view.headline}.</Alert>
      )}
      {stale && <span className="small muted">Updating…</span>}
    </div>
  );
}

interface RepeatState {
  repeat?: ScanConfig;
}

export default function ScanSetup() {
  const navigate = useNavigate();
  const location = useLocation();
  const { prefs } = usePreferences();
  const { scanActive, activeScanId, estopEngaged, status, info } = useSystem();
  // A manual laser cannot be switched off by the scanner, so a pre-scan dark
  // calibration is impossible (the backend refuses it): never request it then.
  const laserControllable = status?.hardware.laser.controllable ?? true;
  const [form, setForm] = useState<ScanForm>(() => {
    const repeat = (location.state as RepeatState | null)?.repeat;
    return repeat ? configToForm(repeat) : loadStoredForm();
  });
  const start = useAction();

  useEffect(() => {
    storeForm(form);
  }, [form]);

  const set = <K extends keyof ScanForm>(key: K, value: ScanForm[K]): void => {
    setForm((current) => ({ ...current, [key]: value }));
  };
  const num = (key: NumericField) => ({
    value: form[key],
    onChange: (value: string) => {
      set(key, value);
    },
  });
  /** Detection thresholds: editing one by hand makes the preset "custom". */
  const detection = (key: NumericField) => ({
    value: form[key],
    onChange: (value: string) => {
      setForm((current) => ({ ...current, [key]: value, detection_preset: "custom" }));
    },
  });

  const { config, errors } = useMemo(
    () =>
      formToConfig({
        ...form,
        calibrate_dark_before_scan: form.calibrate_dark_before_scan && laserControllable,
      }),
    [form, laserControllable],
  );
  const configKey = config ? JSON.stringify(config) : null;
  const debouncedKey = useDebounced(configKey, ESTIMATE_DEBOUNCE_MS);

  const estimate = useApiData<{ key: string; estimate: ScanEstimate } | null>(
    async (signal) => {
      if (debouncedKey === null) return null;
      const result = await api.estimateScan(JSON.parse(debouncedKey) as ScanConfigBody, { signal });
      return { key: debouncedKey, estimate: result };
    },
    [debouncedKey],
  );

  const view: EstimateView | null = useMemo(() => {
    if (configKey === null) return null;
    if (estimate.error && !estimate.loading) return describeEstimateError(estimate.error);
    const data = estimate.data;
    return data ? describeEstimate(data.estimate) : null;
  }, [configKey, estimate.data, estimate.error, estimate.loading]);

  const upToDate =
    configKey !== null && estimate.data?.key === configKey && !estimate.loading && estimate.error === null;
  const stale = configKey !== null && !upToDate;
  const confocal = form.mode === "confocal";

  const blockedReason = scanActive
    ? "A scan is already running."
    : estopEngaged
      ? "The emergency stop is latched: reset it first."
      : config === null
        ? "Some settings are invalid."
        : !upToDate
          ? "Waiting for the estimate…"
          : view && !view.canStart
            ? "The estimate reports problems."
            : null;

  const onStart = async (): Promise<void> => {
    if (!config) return;
    const summary = await start.run(() => api.createScan(config));
    if (summary) void navigate(`/live/${summary.id}`);
  };

  const position = status?.hardware.stage.position ?? null;
  const centreOnPosition = (): void => {
    if (!position) return;
    setForm((current) => {
      const step = Number(current.xy_step_um) || 20;
      const halfX = (Number(current.x_stop_um) - Number(current.x_start_um)) / 2 || step * 5;
      const halfY = (Number(current.y_stop_um) - Number(current.y_start_um)) / 2 || step * 5;
      const round = (value: number): string => String(Math.round(value * 1000) / 1000);
      return {
        ...current,
        x_start_um: round(position.x_um - halfX),
        x_stop_um: round(position.x_um + halfX),
        y_start_um: round(position.y_um - halfY),
        y_stop_um: round(position.y_um + halfY),
        z_center_um: round(position.z_um),
      };
    });
  };

  const limits = info?.limits;
  const unit = "µm";

  return (
    <>
      <PageHeader title="Scan Setup">
        <button
          type="button"
          onClick={centreOnPosition}
          disabled={!position}
          title="Centre the XY area on the current stage position and use its Z as the Z centre"
        >
          Centre on current position
        </button>
        <button
          type="button"
          onClick={() => {
            setForm(DEFAULT_SCAN_FORM);
          }}
        >
          Reset to defaults
        </button>
      </PageHeader>
      <div className="grid grid-sidebar">
        <form
          className="card"
          onSubmit={(event) => {
            event.preventDefault();
            void onStart();
          }}
          aria-label="Scan settings"
        >
          <fieldset>
            <legend>Scan</legend>
            <div className="form-grid">
              <Field label="Name (optional)" error={errors.name}>
                {(id) => (
                  <input
                    id={id}
                    value={form.name}
                    maxLength={200}
                    onChange={(event) => {
                      set("name", event.target.value);
                    }}
                  />
                )}
              </Field>
              <SelectField
                label="Mode"
                hint={form.mode === "confocal" ? "Z sweep and peak detection per point" : "Intensity at one Z"}
                value={form.mode}
                options={[
                  { value: "confocal", label: "Confocal surface" },
                  { value: "fixed_z", label: "Fixed-Z intensity" },
                ]}
                onChange={(value) => {
                  set("mode", value);
                }}
              />
              <SelectField
                label="Scan order"
                hint={form.order === "serpentine" ? "Alternate X direction every row" : "Every row in the same direction"}
                value={form.order}
                options={[
                  { value: "serpentine", label: "Serpentine" },
                  { value: "raster", label: "Raster" },
                ]}
                onChange={(value) => {
                  set("order", value);
                }}
              />
            </div>
          </fieldset>

          <fieldset>
            <legend>XY area</legend>
            <div className="form-grid">
              <NumberField label="X start" unit={unit} error={errors.x_start_um} {...num("x_start_um")} />
              <NumberField label="X stop" unit={unit} error={errors.x_stop_um} {...num("x_stop_um")} />
              <NumberField label="Y start" unit={unit} error={errors.y_start_um} {...num("y_start_um")} />
              <NumberField label="Y stop" unit={unit} error={errors.y_stop_um} {...num("y_stop_um")} />
              <NumberField label="XY step" unit={unit} error={errors.xy_step_um} min={0} {...num("xy_step_um")} />
            </div>
            {limits && (
              <p className="small muted" style={{ marginTop: "0.5rem" }}>
                Travel limits: X {formatLength(limits.x.min_um, prefs.lengthUnit)} …{" "}
                {formatLength(limits.x.max_um, prefs.lengthUnit)}, Y {formatLength(limits.y.min_um, prefs.lengthUnit)} …{" "}
                {formatLength(limits.y.max_um, prefs.lengthUnit)}, Z {formatLength(limits.z.min_um, prefs.lengthUnit)} …{" "}
                {formatLength(limits.z.max_um, prefs.lengthUnit)}
              </p>
            )}
          </fieldset>

          <fieldset>
            <legend>Z</legend>
            <div className="form-grid">
              <NumberField
                label="Z centre"
                unit={unit}
                error={errors.z_center_um}
                hint={confocal ? "Centre of the first coarse sweep" : "Height of the intensity map"}
                {...num("z_center_um")}
              />
              {confocal && (
                <>
                  <NumberField
                    label="Z range"
                    unit={unit}
                    error={errors.z_range_um}
                    hint="Full width of the coarse sweep"
                    {...num("z_range_um")}
                  />
                  <NumberField label="Coarse Z step" unit={unit} error={errors.coarse_z_step_um} {...num("coarse_z_step_um")} />
                  <CheckField
                    label="Fine Z scan"
                    hint={
                      form.fine_scan
                        ? "Second, finer sweep around the coarse peak"
                        : "Off: the surface is found from the coarse sweep alone (faster)"
                    }
                    checked={form.fine_scan}
                    onChange={(value) => {
                      set("fine_scan", value);
                    }}
                  />
                  <NumberField
                    label="Fine Z step"
                    unit={unit}
                    error={form.fine_scan ? errors.fine_z_step_um : undefined}
                    disabled={!form.fine_scan}
                    {...num("fine_z_step_um")}
                  />
                  <NumberField
                    label="Fine range"
                    unit={unit}
                    error={form.fine_scan ? errors.fine_z_range_um : undefined}
                    disabled={!form.fine_scan}
                    hint="Full width of the fine sweep (≥ 2 × coarse step)"
                    {...num("fine_z_range_um")}
                  />
                </>
              )}
            </div>
            {confocal && (
              <div className="form-grid" style={{ marginTop: "0.75rem", alignItems: "end" }}>
                <CheckField
                  label="Adaptive scanning"
                  hint="Centre each coarse sweep on the previous valid point's surface"
                  checked={form.adaptive_z}
                  onChange={(value) => {
                    set("adaptive_z", value);
                  }}
                />
                <NumberField
                  label="Adaptive Z range"
                  unit={unit}
                  error={errors.adaptive_z_range_um}
                  disabled={!form.adaptive_z}
                  hint="Coarse width when an estimate exists"
                  {...num("adaptive_z_range_um")}
                />
              </div>
            )}
          </fieldset>

          <fieldset>
            <legend>Acquisition</legend>
            <div className="form-grid">
              <NumberField
                label="Samples per Z"
                error={errors.samples_per_z}
                step={1}
                min={1}
                max={256}
                {...num("samples_per_z")}
              />
              <SelectField
                label="Sampling method"
                value={form.sampling_method}
                options={["mean", "median"] as const}
                onChange={(value) => {
                  set("sampling_method", value);
                }}
              />
              <NumberField
                label="Settle time"
                unit="ms"
                error={errors.settle_time_ms}
                min={0}
                max={10000}
                {...num("settle_time_ms")}
              />
            </div>
            <div className="stack" style={{ marginTop: "0.75rem" }}>
              <CheckField
                label="Home before scanning"
                checked={form.home_before_scan}
                onChange={(value) => {
                  set("home_before_scan", value);
                }}
              />
              <CheckField
                label="Dark calibration before scanning"
                hint={
                  laserControllable
                    ? "Switches the laser off, measures the dark level, switches it back on"
                    : "Not available with the manual laser: run the dark calibration on the Calibration page (it is used automatically)"
                }
                disabled={!laserControllable}
                checked={form.calibrate_dark_before_scan && laserControllable}
                onChange={(value) => {
                  set("calibrate_dark_before_scan", value);
                }}
              />
            </div>
          </fieldset>

          {confocal && (
            <fieldset>
              <legend>Peak detection</legend>
              <div className="form-grid" style={{ alignItems: "end" }}>
                <SelectField
                  label="Preset"
                  hint={
                    form.detection_preset === "low_contrast"
                      ? "For weak I(Z) peaks: weak points are kept, marked low-confidence"
                      : form.detection_preset === "standard"
                        ? "Strict: only clear peaks become surface points"
                        : "Your own thresholds"
                  }
                  value={form.detection_preset}
                  options={[
                    { value: "standard", label: "Standard" },
                    { value: "low_contrast", label: "Low contrast (sensitive)" },
                    { value: "custom", label: "Custom" },
                  ]}
                  onChange={(value: DetectionPreset) => {
                    setForm((current) => applyDetectionPreset(current, value));
                  }}
                />
                <SelectField
                  label="Peak choice"
                  hint={
                    form.peak_selection === "highest"
                      ? "Several peaks: the highest intensity is the surface (no penalty for the others)"
                      : "Several peaks: the most prominent is the surface; a strong second peak lowers confidence"
                  }
                  value={form.peak_selection}
                  options={[
                    { value: "most_prominent", label: "Most prominent" },
                    { value: "highest", label: "Highest intensity" },
                  ]}
                  onChange={(value: PeakSelection) => {
                    setForm((current) => ({ ...current, peak_selection: value, detection_preset: "custom" }));
                  }}
                />
                <CheckField
                  label="Accept weak peaks"
                  hint="Keep peaks that are hard to tell from the noise (never marked valid)"
                  checked={form.accept_weak_peaks}
                  onChange={(value) => {
                    setForm((current) => ({ ...current, accept_weak_peaks: value, detection_preset: "custom" }));
                  }}
                />
                <NumberField label="Min SNR" error={errors.min_snr} min={0} {...detection("min_snr")} />
                <NumberField
                  label="Min relative prominence"
                  hint="0 … 1: how far the signal must fall on both sides of the peak"
                  error={errors.min_relative_prominence}
                  min={0}
                  max={1}
                  {...detection("min_relative_prominence")}
                />
                <NumberField
                  label="Min confidence (valid point)"
                  hint="Below this a point is low-confidence"
                  error={errors.min_confidence}
                  min={0}
                  max={1}
                  {...detection("min_confidence")}
                />
                <NumberField
                  label="Min confidence (surface)"
                  hint="Lowest confidence used to build the surface"
                  error={errors.reconstruction_min_confidence}
                  min={0}
                  max={1}
                  {...detection("reconstruction_min_confidence")}
                />
              </div>
              {form.accept_weak_peaks && (
                <p className="small muted" style={{ marginTop: "0.5rem" }}>
                  Weak peaks are flagged <code>weak_peak</code> and are at most low-confidence. They only appear in
                  the surface if their confidence is at least the surface minimum above.
                </p>
              )}
            </fieldset>
          )}

          {confocal && (
            <fieldset>
              <legend>After the scan</legend>
              <div className="form-grid" style={{ alignItems: "end" }}>
                <CheckField
                  label="Reconstruct on complete"
                  checked={form.reconstruct_on_complete}
                  onChange={(value) => {
                    set("reconstruct_on_complete", value);
                  }}
                />
                <SelectField
                  label="Interpolation"
                  value={form.reconstruction_method}
                  options={INTERPOLATION_METHODS}
                  disabled={!form.reconstruct_on_complete}
                  onChange={(value) => {
                    set("reconstruction_method", value);
                  }}
                />
                <CheckField
                  label="Advisory ML analysis"
                  hint="Only when a model is deployed"
                  checked={form.ml_on_complete}
                  onChange={(value) => {
                    set("ml_on_complete", value);
                  }}
                />
              </div>
            </fieldset>
          )}
        </form>

        <div className="stack">
          <Card title="Estimate">
            <EstimatePanel view={view} loading={estimate.loading} stale={stale} />
          </Card>
          <Card title="Start">
            <div className="stack">
              {blockedReason !== null && <p className="muted">{blockedReason}</p>}
              {scanActive && activeScanId !== null && (
                <button
                  type="button"
                  onClick={() => {
                    void navigate(`/live/${activeScanId}`);
                  }}
                >
                  Go to the running scan
                </button>
              )}
              <button
                type="button"
                className="btn-primary btn-large"
                disabled={blockedReason !== null || start.busy}
                onClick={() => {
                  void onStart();
                }}
              >
                {start.busy ? "Starting…" : "Start scan"}
              </button>
              <ErrorBox error={start.error} title="The scan was not started" onDismiss={start.clearError} />
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
