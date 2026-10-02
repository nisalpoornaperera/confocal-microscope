/** ADC status, gain selection / auto-gain, and live reading (Calibration and Hardware pages). */
import { useState } from "react";

import { api } from "../api/endpoints";
import { ADC_FULL_SCALE_V, ADC_GAINS, type ADCStatus, type AdcGain, type IntensityMeasurement, type SamplingMethod } from "../api/types";
import { useAction, useApiData } from "../hooks/useApi";
import { formatDateTime, formatNumber, formatVoltage } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";
import { Badge, Card, CheckField, ErrorBox, KeyValue, Loading, NumberField, SelectField } from "./ui";

export function AdcStatusView({ status }: { status: ADCStatus }) {
  return (
    <KeyValue
      items={[
        ["Backend", `${status.backend}${status.connected ? "" : " (disconnected)"}`],
        ["Gain", `${status.gain} (±${status.full_scale_v} V full scale)`],
        ["Data rate", `${status.data_rate_sps} samples/s`],
        ["Channel", status.channel ?? "—"],
        [
          "Last reading",
          <span key="v">
            {formatVoltage(status.last_voltage_v)} {status.saturated && <Badge kind="danger">Saturated</Badge>}
          </span>,
        ],
        ["Last error", status.last_error ?? "none"],
      ]}
    />
  );
}

/** ADC status plus gain controls. */
export function AdcGainCard() {
  const { scanActive } = useSystem();
  const status = useApiData((signal) => api.adcStatus({ signal }), [], { intervalMs: 3000 });
  // null = follow the ADC's current gain until the user picks another one.
  const [chosenGain, setGain] = useState<AdcGain | null>(null);
  const [target, setTarget] = useState("0.8");
  const action = useAction();
  const [message, setMessage] = useState<string | null>(null);
  const gain: AdcGain = chosenGain ?? status.data?.gain ?? "2";

  const targetValue = Number(target);
  const targetValid = targetValue > 0.1 && targetValue <= 0.95;

  const apply = (body: { gain: AdcGain } | { auto: true; target_fraction: number }): void => {
    setMessage(null);
    void action.run(() => api.adcCalibrate(body)).then((result) => {
      if (result) {
        setGain(null);
        setMessage(
          `${result.message} Gain ${result.gain} (±${result.full_scale_v} V)` +
            (result.measured_max_v != null ? `, measured max ${formatVoltage(result.measured_max_v)}.` : "."),
        );
      }
      status.reload();
    });
  };

  return (
    <Card title="ADC gain">
      <div className="stack">
        {status.data ? <AdcStatusView status={status.data} /> : status.error ? <ErrorBox error={status.error} /> : <Loading what="ADC status" />}
        <div className="form-grid" style={{ alignItems: "end" }}>
          <SelectField
            label="Gain"
            value={gain}
            options={ADC_GAINS.map((g) => ({ value: g, label: `${g} (±${ADC_FULL_SCALE_V[g]} V)` }))}
            onChange={setGain}
            disabled={scanActive}
          />
          <button type="button" disabled={scanActive || action.busy} onClick={() => apply({ gain })}>
            Set gain
          </button>
        </div>
        <div className="form-grid" style={{ alignItems: "end" }}>
          <NumberField
            label="Auto-gain target"
            hint="Max signal as a fraction of full scale (0.1–0.95)"
            value={target}
            onChange={setTarget}
            error={targetValid ? null : "between 0.1 and 0.95"}
            disabled={scanActive}
          />
          <button
            type="button"
            disabled={scanActive || action.busy || !targetValid}
            onClick={() => apply({ auto: true, target_fraction: targetValue })}
            title="Measures the current signal and picks the highest gain that keeps it below the target"
          >
            Auto-gain
          </button>
        </div>
        <p className="small muted">
          The OPT101 on 3.3 V clips near 2.0 V, so gain 2 (±2.048 V) is the normal setting. Auto-gain measures at the
          current position: put the beam in focus on the brightest area first.
        </p>
        {message && <div className="alert alert-ok">{message}</div>}
        <ErrorBox error={action.error} onDismiss={action.clearError} />
      </div>
    </Card>
  );
}

/** Single or continuous calibrated reading at the current position. */
export function AdcReadCard() {
  const { prefs } = usePreferences();
  const { scanActive } = useSystem();
  const [samples, setSamples] = useState(String(prefs.adcSamples));
  const [method, setMethod] = useState<SamplingMethod>("median");
  const [continuous, setContinuous] = useState(false);
  const [reading, setReading] = useState<IntensityMeasurement | null>(null);
  const action = useAction();
  const n = Number(samples);
  const valid = Number.isInteger(n) && n >= 1 && n <= 1024;
  const live = continuous && valid && !scanActive;

  const poll = useApiData((signal) => api.adcRead(n, method, { signal }), [n, method], {
    intervalMs: 1000,
    enabled: live,
  });
  const shown = live ? (poll.data ?? reading) : reading;
  const error = live ? poll.error : action.error;

  const counts = shown?.raw_counts ?? [];
  return (
    <Card title="ADC live read">
      <div className="stack">
        <div className="form-grid" style={{ alignItems: "end" }}>
          <NumberField
            label="Samples"
            value={samples}
            onChange={setSamples}
            step={1}
            min={1}
            max={1024}
            error={valid ? null : "1 to 1024"}
          />
          <SelectField label="Method" value={method} options={["median", "mean"] as const} onChange={setMethod} />
          <button
            type="button"
            className="btn-primary"
            disabled={!valid || scanActive || action.busy || live}
            onClick={() => {
              void action.run(() => api.adcRead(n, method)).then((result) => {
                if (result) setReading(result);
              });
            }}
          >
            Read once
          </button>
          <CheckField label="Continuous (1 s)" checked={continuous} onChange={setContinuous} disabled={scanActive} />
        </div>
        {scanActive && <p className="muted">Readings are refused while a scan is running.</p>}
        <ErrorBox error={error} />
        {shown && (
          <div className="grid grid-2">
            <div className="stat-tiles">
              <div className="stat">
                <div className="stat-label">Voltage ({shown.method})</div>
                <div className="stat-value">{formatVoltage(shown.voltage_v)}</div>
                <div className="small muted">± {formatVoltage(shown.voltage_std_v)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Dark-corrected</div>
                <div className="stat-value">{formatVoltage(shown.corrected_v)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Normalized</div>
                <div className="stat-value">{formatNumber(shown.normalized, 4)}</div>
                <div className="small muted">{shown.normalized == null ? "needs a reference calibration" : "(V − dark) / (ref − dark)"}</div>
              </div>
            </div>
            <KeyValue
              items={[
                ["Saturated", shown.saturated ? <Badge key="s" kind="danger">YES: lower the gain or the light</Badge> : "no"],
                ["Gain", shown.gain],
                ["Samples", shown.n_samples],
                ["Raw codes", counts.length > 0 ? `${Math.min(...counts)} … ${Math.max(...counts)}` : "—"],
                ["Dark / reference", `${formatVoltage(shown.dark_v)} / ${formatVoltage(shown.reference_v)}`],
                ["Calibration", `v${shown.calibration_version ?? "—"}`],
                ["Time", formatDateTime(shown.timestamp)],
              ]}
            />
          </div>
        )}
      </div>
    </Card>
  );
}
