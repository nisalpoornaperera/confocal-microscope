/**
 * 5 Calibration: current calibration and its history; dark calibration
 * (manual laser: the operator must confirm the beam is blocked); reference
 * calibration with optional Z search; ADC gain / auto-gain; live ADC read.
 */
import { useState } from "react";

import { api } from "../api/endpoints";
import type { CalibrationState, SamplingMethod } from "../api/types";
import { AdcGainCard, AdcReadCard } from "../components/AdcPanels";
import { Alert, Card, CheckField, Empty, ErrorBox, Field, KeyValue, Loading, NumberField, PageHeader, SelectField } from "../components/ui";
import { useAction, useApiData } from "../hooks/useApi";
import { formatDateTime, formatLength, formatVoltage, humanize } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";

function CurrentCalibration({ calibration }: { calibration: CalibrationState }) {
  const { prefs } = usePreferences();
  const ref = calibration.reference_position;
  return (
    <KeyValue
      items={[
        ["Version", calibration.version ?? "never calibrated"],
        ["Updated", `${formatDateTime(calibration.created_at)}${calibration.updated_field ? ` (${calibration.updated_field})` : ""}`],
        [
          "Dark level",
          calibration.dark_v != null
            ? `${formatVoltage(calibration.dark_v)} ± ${formatVoltage(calibration.dark_std_v)} · ${calibration.dark_n_samples} samples · gain ${calibration.dark_gain ?? "—"} · ${formatDateTime(calibration.dark_measured_at)}`
            : "not measured",
        ],
        [
          "Reference level",
          calibration.reference_v != null
            ? `${formatVoltage(calibration.reference_v)} ± ${formatVoltage(calibration.reference_std_v)} · ${calibration.reference_n_samples} samples · gain ${calibration.reference_gain ?? "—"} · ${formatDateTime(calibration.reference_measured_at)}`
            : "not measured (intensities stay in dark-corrected volts)",
        ],
        [
          "Reference position",
          ref
            ? `X ${formatLength(ref.x_um, prefs.lengthUnit)}, Y ${formatLength(ref.y_um, prefs.lengthUnit)}, Z ${formatLength(ref.z_um, prefs.lengthUnit, 3)}`
            : "—",
        ],
        ["Notes", calibration.notes ?? "—"],
      ]}
    />
  );
}

function intOrError(text: string, min: number, max: number): { value?: number; error?: string } {
  const value = Number(text);
  if (text.trim() === "" || !Number.isInteger(value)) return { error: "whole number required" };
  if (value < min || value > max) return { error: `${min} to ${max}` };
  return { value };
}

function DarkCard({ onDone }: { onDone: () => void }) {
  const { laserManual, scanActive, status } = useSystem();
  const [samples, setSamples] = useState("64");
  const [method, setMethod] = useState<SamplingMethod>("median");
  const [notes, setNotes] = useState("");
  const [blocked, setBlocked] = useState(false);
  const action = useAction();
  const [done, setDone] = useState<CalibrationState | null>(null);
  const n = intOrError(samples, 1, 4096);
  const laserKnown = status !== undefined;
  const needsConfirm = laserManual || !laserKnown;
  const canRun = n.value !== undefined && !scanActive && !action.busy && (!needsConfirm || blocked);

  return (
    <Card title="Dark calibration">
      <div className="stack">
        <p>
          Measures the detector signal with no laser light reaching it. Subtracted from every reading.
        </p>
        {laserManual ? (
          <Alert kind="warning" title="The laser is switched by hand">
            Block the beam (or switch the laser off) before measuring. The software cannot do it for you.
          </Alert>
        ) : (
          laserKnown && <p className="small muted">The laser is switched off automatically during the measurement.</p>
        )}
        <div className="form-grid">
          <NumberField label="Samples" value={samples} onChange={setSamples} error={n.error} step={1} min={1} max={4096} />
          <SelectField label="Method" value={method} options={["median", "mean"] as const} onChange={setMethod} />
        </div>
        <Field label="Notes (optional)">
          {(id) => <input id={id} value={notes} maxLength={1000} onChange={(e) => setNotes(e.target.value)} />}
        </Field>
        {needsConfirm && (
          <CheckField
            label={<strong>I have blocked the beam</strong>}
            hint="Required: no laser light may reach the detector during the dark measurement"
            checked={blocked}
            onChange={setBlocked}
          />
        )}
        <button
          type="button"
          className="btn-primary"
          disabled={!canRun}
          onClick={() => {
            if (n.value === undefined) return;
            setDone(null);
            void action
              .run(() =>
                api.calibrateDark({
                  n_samples: n.value,
                  method,
                  beam_blocked_confirmed: needsConfirm ? blocked : false,
                  notes: notes.trim() || null,
                }),
              )
              .then((result) => {
                if (result) {
                  setDone(result);
                  setBlocked(false);
                  onDone();
                }
              });
          }}
        >
          {action.busy ? "Measuring…" : "Measure dark level"}
        </button>
        {scanActive && <p className="muted">Calibration is refused while a scan is running.</p>}
        {done && (
          <Alert kind="ok">
            Dark level {formatVoltage(done.dark_v)} saved as calibration v{done.version}.
            {laserManual && " Unblock the beam before scanning."}
          </Alert>
        )}
        <ErrorBox error={action.error} title="Dark calibration failed" onDismiss={action.clearError} />
      </div>
    </Card>
  );
}

function ReferenceCard({ onDone }: { onDone: () => void }) {
  const { scanActive } = useSystem();
  const [samples, setSamples] = useState("64");
  const [method, setMethod] = useState<SamplingMethod>("median");
  const [zSearch, setZSearch] = useState(true);
  const [range, setRange] = useState("40");
  const [step, setStep] = useState("1");
  const [notes, setNotes] = useState("");
  const action = useAction();
  const [done, setDone] = useState<CalibrationState | null>(null);
  const n = intOrError(samples, 1, 4096);
  const rangeValue = Number(range);
  const stepValue = Number(step);
  const rangeError = zSearch && !(rangeValue > 0 && rangeValue <= 2000) ? "0 < range ≤ 2000" : null;
  const stepError = zSearch && !(stepValue > 0 && stepValue <= 100) ? "0 < step ≤ 100" : null;
  const canRun = n.value !== undefined && !rangeError && !stepError && !scanActive && !action.busy;

  return (
    <Card title="Reference calibration">
      <div className="stack">
        <p>
          Measures the in-focus signal of a reference reflector (e.g. a plane mirror) so intensities can be normalized.
          Place the reference under the beam, near focus, with the laser on.
        </p>
        <div className="form-grid">
          <NumberField label="Samples" value={samples} onChange={setSamples} error={n.error} step={1} min={1} max={4096} />
          <SelectField label="Method" value={method} options={["median", "mean"] as const} onChange={setMethod} />
        </div>
        <CheckField
          label="Search Z for the maximum"
          hint="Sweeps Z around the current position and uses the brightest point; otherwise you must already be in focus"
          checked={zSearch}
          onChange={setZSearch}
        />
        {zSearch && (
          <div className="form-grid">
            <NumberField label="Search range" unit="µm" value={range} onChange={setRange} error={rangeError} />
            <NumberField label="Search step" unit="µm" value={step} onChange={setStep} error={stepError} />
          </div>
        )}
        <Field label="Notes (optional)">
          {(id) => <input id={id} value={notes} maxLength={1000} onChange={(e) => setNotes(e.target.value)} />}
        </Field>
        <button
          type="button"
          className="btn-primary"
          disabled={!canRun}
          onClick={() => {
            if (n.value === undefined) return;
            setDone(null);
            void action
              .run(() =>
                api.calibrateReference({
                  n_samples: n.value,
                  method,
                  z_search: zSearch,
                  ...(zSearch ? { z_search_range_um: rangeValue, z_search_step_um: stepValue } : {}),
                  notes: notes.trim() || null,
                }),
              )
              .then((result) => {
                if (result) {
                  setDone(result);
                  onDone();
                }
              });
          }}
        >
          {action.busy ? (zSearch ? "Searching Z and measuring…" : "Measuring…") : "Measure reference level"}
        </button>
        {done && (
          <Alert kind="ok">
            Reference level {formatVoltage(done.reference_v)} saved as calibration v{done.version}.
          </Alert>
        )}
        <ErrorBox error={action.error} title="Reference calibration failed" onDismiss={action.clearError} />
      </div>
    </Card>
  );
}

function HistoryCard({ version }: { version: number }) {
  const history = useApiData((signal) => api.calibrationHistory(50, { signal }), [version]);
  return (
    <Card title="Calibration history">
      <ErrorBox error={history.error} />
      {history.data === undefined ? (
        !history.error && <Loading what="history" />
      ) : history.data.length === 0 ? (
        <Empty>No calibration has been measured yet.</Empty>
      ) : (
        <div className="table-wrap" style={{ maxHeight: "24rem" }}>
          <table>
            <thead>
              <tr>
                <th className="num">Version</th>
                <th>Created</th>
                <th>Changed</th>
                <th className="num">Dark</th>
                <th className="num">Reference</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {history.data.map((item) => (
                <tr key={item.version ?? item.created_at}>
                  <td className="num">{item.version ?? "—"}</td>
                  <td>{formatDateTime(item.created_at)}</td>
                  <td>{humanize(item.updated_field)}</td>
                  <td className="num">{formatVoltage(item.dark_v)}</td>
                  <td className="num">{formatVoltage(item.reference_v)}</td>
                  <td style={{ whiteSpace: "normal" }}>{item.notes ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default function Calibration() {
  const [generation, setGeneration] = useState(0);
  const current = useApiData((signal) => api.calibration({ signal }), [generation], { intervalMs: 15_000 });
  const { scanActive } = useSystem();
  const bump = (): void => {
    setGeneration((g) => g + 1);
  };
  return (
    <>
      <PageHeader title="Calibration" />
      {scanActive && (
        <div className="banners">
          <Alert kind="info" title="A scan is running">
            Calibration and ADC changes are refused until it finishes.
          </Alert>
        </div>
      )}
      <div className="grid grid-2">
        <Card title="Current calibration" className="span-all">
          <ErrorBox error={current.error} />
          {current.data ? <CurrentCalibration calibration={current.data} /> : !current.error && <Loading what="calibration" />}
        </Card>
        <DarkCard onDone={bump} />
        <ReferenceCard onDone={bump} />
        <AdcGainCard />
        <AdcReadCard />
        <div className="span-all">
          <HistoryCard version={current.data?.version ?? generation} />
        </div>
      </div>
    </>
  );
}
