/**
 * 7 Settings: read-only system information, travel limits and versions
 * (`GET /api/v1/system`), and the UI preferences kept in this browser.
 */
import { useState } from "react";

import { api } from "../api/endpoints";
import { Card, CheckField, ErrorBox, Field, KeyValue, Loading, PageHeader, Segmented, SelectField } from "../components/ui";
import { useApiData } from "../hooks/useApi";
import { formatInteger, formatLength } from "../lib/format";
import { COLOR_MAPS, DEFAULT_PREFERENCES, usePreferences, type ThemeChoice } from "../lib/prefs";
import { useSystem } from "../lib/system";

/** Numeric preference edited as text and committed on blur / Enter (then clamped). */
function PrefNumber({
  label,
  hint,
  value,
  onCommit,
}: {
  label: string;
  hint: string;
  value: number;
  onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  const [shown, setShown] = useState(value);
  if (shown !== value) {
    // The stored value changed (clamped or reset): show it.
    setShown(value);
    setText(String(value));
  }
  const commit = (): void => {
    const n = Number(text);
    if (text.trim() !== "" && Number.isFinite(n)) onCommit(n);
    else setText(String(value));
  };
  return (
    <Field label={label} hint={hint}>
      {(id, describedBy) => (
        <input
          id={id}
          type="number"
          value={text}
          aria-describedby={describedBy}
          onChange={(event) => {
            setText(event.target.value);
          }}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
          }}
        />
      )}
    </Field>
  );
}

export default function Settings() {
  const { info, infoError } = useSystem();
  const { prefs, update, reset } = usePreferences();
  const models = useApiData((signal) => api.mlModels({ signal }), []);
  const unit = prefs.lengthUnit;

  return (
    <>
      <PageHeader title="Settings" />
      <div className="grid grid-2">
        <Card title="System (read only)">
          {info ? (
            <KeyValue
              items={[
                ["Name", info.name],
                ["Software version", info.software_version],
                ["API version", info.api_version],
                ["Python", info.python_version],
                ["Platform", info.platform],
                ["Simulation", info.simulation ? "yes (no real hardware)" : "no"],
                ["Data directory", <code key="d">{info.data_dir}</code>],
                ["UI build", import.meta.env.MODE],
              ]}
            />
          ) : infoError ? (
            <ErrorBox error={infoError} />
          ) : (
            <Loading what="system information" />
          )}
        </Card>

        <Card title="Travel limits (read only)">
          {info ? (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Axis</th>
                      <th className="num">Min</th>
                      <th className="num">Max</th>
                      <th className="num">Span</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(["x", "y", "z"] as const).map((axis) => {
                      const l = info.limits[axis];
                      return (
                        <tr key={axis}>
                          <td>{axis.toUpperCase()}</td>
                          <td className="num">{formatLength(l.min_um, unit)}</td>
                          <td className="num">{formatLength(l.max_um, unit)}</td>
                          <td className="num">{formatLength(l.max_um - l.min_um, unit)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="small muted" style={{ marginTop: "0.5rem" }}>
                Set in the machine configuration (<code>config/confocal.pi.toml</code>, <code>[limits.*]</code>). The stage
                has no end-stops: limits are relative to the power-up origin.
              </p>
            </>
          ) : (
            <Loading what="limits" />
          )}
        </Card>

        <Card title="Hardware versions (read only)">
          {info ? (
            <KeyValue
              items={[
                ["Controller", info.hardware.controller],
                ["Stage", `${info.hardware.stage_backend} ${info.hardware.stage_version ?? ""}`],
                ["ADC", `${info.hardware.adc_backend} ${info.hardware.adc_version ?? ""}`],
                ["Laser", info.hardware.laser_backend],
                ["Camera", info.hardware.camera_backend],
                ...Object.entries(info.hardware.details ?? {}).map(([key, value]): [string, string] => [key, value]),
              ]}
            />
          ) : (
            <Loading what="hardware" />
          )}
        </Card>

        <Card title="ML models (read only)">
          <ErrorBox error={models.error} />
          {models.data === undefined ? (
            !models.error && <Loading what="models" />
          ) : models.data.length === 0 ? (
            <p className="muted">No ML model is deployed. ML is advisory and optional.</p>
          ) : (
            <KeyValue
              items={models.data.map((m): [string, string] => [
                m.name,
                `${m.version} · ${m.task} · ${m.algorithm}${m.trained_at ? ` · trained ${m.trained_at.slice(0, 10)}` : ""}`,
              ])}
            />
          )}
        </Card>

        <Card
          title="Display preferences (this browser)"
          className="span-all"
          actions={
            <button type="button" onClick={reset}>
              Restore defaults
            </button>
          }
        >
          <div className="form-grid">
            <div className="field">
              <span className="field-label">Theme</span>
              <Segmented<ThemeChoice>
                label="Theme"
                value={prefs.theme}
                options={[
                  { value: "light", label: "Light" },
                  { value: "dark", label: "Dark" },
                  { value: "system", label: "System" },
                ]}
                onChange={(theme) => {
                  update({ theme });
                }}
              />
            </div>
            <div className="field">
              <span className="field-label">Length unit</span>
              <Segmented
                label="Length unit"
                value={prefs.lengthUnit}
                options={[
                  { value: "um", label: "µm" },
                  { value: "mm", label: "mm" },
                ]}
                onChange={(lengthUnit) => {
                  update({ lengthUnit });
                }}
              />
            </div>
            <SelectField
              label="Colour map (heights, intensity)"
              value={prefs.colorMap}
              options={COLOR_MAPS.map((c) => ({ value: c, label: c }))}
              onChange={(colorMap) => {
                update({ colorMap });
              }}
            />
            <PrefNumber
              label="Display limit: grid cells"
              hint={`Larger grids are reduced for display only (default ${formatInteger(DEFAULT_PREFERENCES.maxDisplayCells)}, 2,500–250,000)`}
              value={prefs.maxDisplayCells}
              onCommit={(maxDisplayCells) => {
                update({ maxDisplayCells });
              }}
            />
            <PrefNumber
              label="Display limit: point cloud"
              hint={`Default ${formatInteger(DEFAULT_PREFERENCES.maxCloudPoints)} (1,000–250,000)`}
              value={prefs.maxCloudPoints}
              onCommit={(maxCloudPoints) => {
                update({ maxCloudPoints });
              }}
            />
            <PrefNumber
              label="ADC read: default samples"
              hint="1–1024"
              value={prefs.adcSamples}
              onCommit={(adcSamples) => {
                update({ adcSamples });
              }}
            />
          </div>
          <div style={{ marginTop: "0.75rem" }}>
            <CheckField
              label="Follow the operating system theme"
              checked={prefs.theme === "system"}
              onChange={(checked) => {
                update({ theme: checked ? "system" : "light" });
              }}
            />
          </div>
          <p className="small muted" style={{ marginTop: "0.5rem" }}>
            Preferences are stored in this browser only (localStorage) and never change the machine.
          </p>
        </Card>
      </div>
    </>
  );
}
