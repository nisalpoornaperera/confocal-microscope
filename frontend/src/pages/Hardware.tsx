/**
 * 6 Hardware: stage position, jog with selectable step sizes per axis,
 * absolute move, home, ADC and laser status, e-stop reset. Every control is
 * disabled while a scan is active (the backend refuses them with 409 anyway).
 */
import { useState } from "react";

import { api } from "../api/endpoints";
import type { Axis, Position } from "../api/types";
import { AdcStatusView } from "../components/AdcPanels";
import { LaserNote } from "../components/Layout";
import { Alert, Badge, Card, ErrorBox, KeyValue, Loading, NumberField, PageHeader, SelectField } from "../components/ui";
import { useAction, useApiData } from "../hooks/useApi";
import { formatLength, humanize } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";

const XY_STEPS = ["1", "5", "10", "50", "100", "500"] as const;
const Z_STEPS = ["0.25", "0.5", "1", "5", "10", "50"] as const;

type XYStep = (typeof XY_STEPS)[number];
type ZStep = (typeof Z_STEPS)[number];

export default function Hardware() {
  const { prefs } = usePreferences();
  const { status, info, scanActive, estopEngaged, refreshStatus } = useSystem();
  const position = useApiData((signal) => api.stagePosition({ signal }), [], { intervalMs: 1000 });
  const move = useAction();
  const [stepX, setStepX] = useState<XYStep>("10");
  const [stepY, setStepY] = useState<XYStep>("10");
  const [stepZ, setStepZ] = useState<ZStep>("1");
  const [target, setTarget] = useState({ x: "", y: "", z: "" });
  const [lastMove, setLastMove] = useState<Position | null>(null);

  const locked = scanActive || estopEngaged;
  const unit = prefs.lengthUnit;
  const pos = position.data ?? status?.hardware.stage.position ?? null;
  const stage = status?.hardware.stage;
  const limits = info?.limits ?? stage?.limits;

  const run = (task: () => Promise<Position>): void => {
    void move.run(task).then((result) => {
      if (result) setLastMove(result);
      position.reload();
      refreshStatus();
    });
  };

  const jog = (axis: Axis, sign: 1 | -1): void => {
    const size = Number(axis === "x" ? stepX : axis === "y" ? stepY : stepZ) * sign;
    run(() => api.stageMove({ [`${axis}_um`]: size, relative: true }));
  };

  const parsed = (["x", "y", "z"] as const).map((axis) => {
    const text = target[axis].trim();
    if (text === "") return { axis, value: undefined, error: null };
    const value = Number(text);
    if (!Number.isFinite(value)) return { axis, value: undefined, error: "not a number" };
    const axisLimits = limits?.[axis];
    if (axisLimits && (value < axisLimits.min_um || value > axisLimits.max_um)) {
      return { axis, value, error: `outside ${axisLimits.min_um} … ${axisLimits.max_um} µm` };
    }
    return { axis, value, error: null };
  });
  const anyTarget = parsed.some((p) => p.value !== undefined);
  const targetErrors = parsed.some((p) => p.error !== null);

  const jogButton = (axis: Axis, sign: 1 | -1, label: string, title: string) => (
    <button type="button" disabled={locked || move.busy} onClick={() => jog(axis, sign)} title={title} aria-label={title}>
      {label}
    </button>
  );

  return (
    <>
      <PageHeader title="Hardware" />
      <div className="banners">
        {scanActive && (
          <Alert kind="info" title="A scan is running">
            Manual control is disabled until it finishes. EMERGENCY STOP is always available.
          </Alert>
        )}
        {estopEngaged && !scanActive && (
          <Alert kind="error" title="Emergency stop latched">
            Motion is refused. Check the machine, then reset the e-stop (banner above).
          </Alert>
        )}
      </div>
      <div className="grid grid-2">
        <Card title="Stage position">
          {pos ? (
            <div className="stat-tiles">
              <div className="stat">
                <div className="stat-label">X</div>
                <div className="stat-value">{formatLength(pos.x_um, unit)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Y</div>
                <div className="stat-value">{formatLength(pos.y_um, unit)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Z</div>
                <div className="stat-value">{formatLength(pos.z_um, unit, 3)}</div>
              </div>
            </div>
          ) : position.error ? null : (
            <Loading what="position" />
          )}
          <ErrorBox error={position.error} title="Position unavailable" />
          {stage && (
            <div style={{ marginTop: "0.75rem" }}>
              <KeyValue
                items={[
                  ["Stage", `${stage.backend}${stage.firmware_version ? ` · firmware ${stage.firmware_version}` : ""}`],
                  ["State", <span key="s">{humanize(stage.state)} {!stage.connected && <Badge kind="danger">Disconnected</Badge>}</span>],
                  ["Homed", stage.homed ? "yes" : "no (positions are relative to the power-up origin)"],
                  [
                    "Travel limits",
                    limits
                      ? `X ${limits.x.min_um} … ${limits.x.max_um}, Y ${limits.y.min_um} … ${limits.y.max_um}, Z ${limits.z.min_um} … ${limits.z.max_um} µm`
                      : "—",
                  ],
                  ["Last error", stage.last_error ?? "none"],
                ]}
              />
            </div>
          )}
        </Card>

        <Card title="Jog">
          <div className="row" style={{ alignItems: "flex-start", gap: "2rem" }}>
            <div className="stack">
              <div className="jog-grid" role="group" aria-label="XY jog">
                <span />
                {jogButton("y", 1, "Y+", `Move Y +${stepY} µm`)}
                <span />
                {jogButton("x", -1, "X−", `Move X −${stepX} µm`)}
                <span className="muted small" style={{ display: "grid", placeItems: "center" }}>
                  XY
                </span>
                {jogButton("x", 1, "X+", `Move X +${stepX} µm`)}
                <span />
                {jogButton("y", -1, "Y−", `Move Y −${stepY} µm`)}
                <span />
              </div>
            </div>
            <div className="jog-z" role="group" aria-label="Z jog">
              {jogButton("z", 1, "Z+", `Move Z +${stepZ} µm`)}
              <span className="muted small" style={{ display: "grid", placeItems: "center" }}>
                Z
              </span>
              {jogButton("z", -1, "Z−", `Move Z −${stepZ} µm`)}
            </div>
            <div className="stack" style={{ minWidth: "11rem" }}>
              <SelectField label="X step (µm)" value={stepX} options={XY_STEPS.map((s) => ({ value: s, label: `${s} µm` }))} onChange={setStepX} />
              <SelectField label="Y step (µm)" value={stepY} options={XY_STEPS.map((s) => ({ value: s, label: `${s} µm` }))} onChange={setStepY} />
              <SelectField label="Z step (µm)" value={stepZ} options={Z_STEPS.map((s) => ({ value: s, label: `${s} µm` }))} onChange={setStepZ} />
            </div>
          </div>
          {move.busy && <p className="muted">Moving…</p>}
        </Card>

        <Card title="Absolute move">
          <form
            className="stack"
            onSubmit={(event) => {
              event.preventDefault();
              if (!anyTarget || targetErrors) return;
              const body: Partial<Record<"x_um" | "y_um" | "z_um", number>> = {};
              for (const p of parsed) if (p.value !== undefined) body[`${p.axis}_um`] = p.value;
              run(() => api.stageMove({ ...body, relative: false }));
            }}
          >
            <div className="form-grid">
              {parsed.map((p) => (
                <NumberField
                  key={p.axis}
                  label={p.axis.toUpperCase()}
                  unit="µm"
                  hint="Empty: keep"
                  value={target[p.axis]}
                  error={p.error}
                  onChange={(value) => {
                    setTarget((current) => ({ ...current, [p.axis]: value }));
                  }}
                />
              ))}
            </div>
            <div className="row">
              <button type="submit" className="btn-primary" disabled={locked || move.busy || !anyTarget || targetErrors}>
                Move
              </button>
              <button
                type="button"
                disabled={!pos}
                onClick={() => {
                  if (pos) setTarget({ x: String(pos.x_um), y: String(pos.y_um), z: String(pos.z_um) });
                }}
              >
                Copy current position
              </button>
            </div>
          </form>
        </Card>

        <Card title="Home">
          <div className="stack">
            <p className="small muted">
              Moves the axes to 0. The 28BYJ-48 stage has no end-stops: 0 is the power-up / zeroed origin.
            </p>
            <div className="row">
              <button type="button" className="btn-primary" disabled={locked || move.busy} onClick={() => run(() => api.stageHome())}>
                Home all axes
              </button>
              {(["x", "y", "z"] as const).map((axis) => (
                <button key={axis} type="button" disabled={locked || move.busy} onClick={() => run(() => api.stageHome([axis]))}>
                  Home {axis.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </Card>

        <div className="span-all">
          <ErrorBox error={move.error} title="Move refused" onDismiss={move.clearError} />
          {lastMove && !move.error && (
            <p className="small muted">
              Last move ended at X {formatLength(lastMove.x_um, unit)}, Y {formatLength(lastMove.y_um, unit)}, Z{" "}
              {formatLength(lastMove.z_um, unit, 3)} (verified by the stage).
            </p>
          )}
        </div>

        <Card title="Detector (ADC)">
          {status ? <AdcStatusView status={status.hardware.adc} /> : <Loading what="ADC status" />}
          <p className="small muted" style={{ marginTop: "0.5rem" }}>
            Gain and live readings: Calibration page.
          </p>
        </Card>

        <Card title="Laser">
          {status ? (
            <div className="stack">
              <KeyValue
                items={[
                  ["Backend", status.hardware.laser.backend],
                  ["Wavelength", `${status.hardware.laser.wavelength_nm} nm`],
                  ["Power", status.hardware.laser.power_mw != null ? `${status.hardware.laser.power_mw} mW` : "—"],
                  ["Software control", status.hardware.laser.controllable ? "yes" : "no (manual switch)"],
                  [
                    "State",
                    status.hardware.laser.enabled == null ? "unknown (manual)" : status.hardware.laser.enabled ? "on" : "off",
                  ],
                  ["Last error", status.hardware.laser.last_error ?? "none"],
                ]}
              />
              <LaserNote />
            </div>
          ) : (
            <Loading what="laser status" />
          )}
        </Card>

        <EstopResetCard />
      </div>
    </>
  );
}

function EstopResetCard() {
  const { status, estopEngaged, scanActive, refreshStatus } = useSystem();
  const reset = useAction();
  return (
    <Card title="Emergency stop" className="span-all">
      <div className="row">
        <span>
          State:{" "}
          {estopEngaged ? <Badge kind="danger">LATCHED</Badge> : <Badge kind="ok">Released</Badge>}
          {status?.hardware.estop_reason && estopEngaged ? ` (${status.hardware.estop_reason})` : ""}
        </span>
        <button
          type="button"
          disabled={!estopEngaged || scanActive || reset.busy}
          onClick={() => {
            void reset.run(() => api.stageReset()).then(refreshStatus);
          }}
        >
          Reset e-stop
        </button>
      </div>
      <ErrorBox error={reset.error} title="Reset refused" onDismiss={reset.clearError} />
    </Card>
  );
}
