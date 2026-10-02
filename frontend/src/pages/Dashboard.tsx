/** 1 Dashboard: system status, active scan, recent scans, calibration, hardware summary. */
import { Link } from "react-router-dom";

import { api } from "../api/endpoints";
import { ScanTable } from "../components/ScanTable";
import {
  Badge,
  Card,
  Empty,
  ErrorBox,
  KeyValue,
  Loading,
  PageHeader,
  ProgressBar,
  ScanStateBadge,
} from "../components/ui";
import { useApiData } from "../hooks/useApi";
import {
  formatDateTime,
  formatDuration,
  formatInteger,
  formatLength,
  formatVoltage,
  humanize,
  shortId,
} from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";

function ActiveScanCard() {
  const { activeScanId } = useSystem();
  const scan = useApiData(
    (signal) => (activeScanId ? api.getScan(activeScanId, { signal }) : Promise.resolve(undefined)),
    [activeScanId],
    { intervalMs: 2000, enabled: activeScanId !== null },
  );
  if (activeScanId === null) {
    return (
      <Card title="Active scan">
        <p className="muted">No scan is running.</p>
        <Link className="button btn-primary" to="/setup">
          Set up a scan
        </Link>
      </Card>
    );
  }
  const summary = scan.data;
  return (
    <Card
      title="Active scan"
      actions={
        <Link className="button btn-primary" to={`/live/${activeScanId}`}>
          Open live view
        </Link>
      }
    >
      <ErrorBox error={scan.error} />
      {summary ? (
        <div className="stack">
          <div className="row">
            <strong>{summary.name ?? shortId(summary.id)}</strong>
            <ScanStateBadge state={summary.state} interrupted={summary.interrupted} />
          </div>
          <ProgressBar fraction={summary.progress} label="Scan progress" />
          <KeyValue
            items={[
              ["Points", `${formatInteger(summary.completed_points)} / ${formatInteger(summary.total_points)}`],
              ["Mode", humanize(summary.mode)],
              ["Started", formatDateTime(summary.started_at)],
            ]}
          />
        </div>
      ) : (
        <Loading what="scan" />
      )}
    </Card>
  );
}

function SystemCard() {
  const { status } = useSystem();
  const { prefs } = usePreferences();
  if (!status) return <Card title="System status"><Loading what="status" /></Card>;
  const { stage, adc, laser, camera } = status.hardware;
  const p = stage.position;
  return (
    <Card title="System status">
      <KeyValue
        items={[
          [
            "Overall",
            <Badge
              key="s"
              kind={status.status === "ok" ? "ok" : status.status === "degraded" ? "warn" : "danger"}
            >
              {status.status.toUpperCase()}
            </Badge>,
          ],
          ["E-stop", status.estop_engaged ? <Badge kind="danger">LATCHED</Badge> : "Released"],
          [
            "Stage",
            `${stage.backend} · ${humanize(stage.state)}${stage.connected ? "" : " (disconnected)"}${stage.homed ? " · homed" : ""}`,
          ],
          [
            "Position",
            p
              ? `X ${formatLength(p.x_um, prefs.lengthUnit)}, Y ${formatLength(p.y_um, prefs.lengthUnit)}, Z ${formatLength(p.z_um, prefs.lengthUnit, 3)}`
              : "—",
          ],
          [
            "Detector (ADC)",
            `${adc.backend} · gain ${adc.gain} (±${adc.full_scale_v} V) · ${adc.data_rate_sps} SPS`,
          ],
          [
            "Last reading",
            <span key="v">
              {formatVoltage(adc.last_voltage_v)} {adc.saturated && <Badge kind="danger">Saturated</Badge>}
            </span>,
          ],
          [
            "Laser",
            `${laser.backend} · ${laser.wavelength_nm} nm · ${
              laser.controllable ? (laser.enabled ? "on" : "off") : "manual switch (state unknown)"
            }`,
          ],
          ["Camera", camera.available ? camera.backend : "none"],
          ["Uptime", formatDuration(status.uptime_s)],
        ]}
      />
      {[stage.last_error, adc.last_error, laser.last_error, camera.last_error]
        .filter((e): e is string => typeof e === "string" && e.length > 0)
        .map((message, index) => (
          <div key={index} className="alert alert-error" style={{ marginTop: "0.6rem" }}>
            {message}
          </div>
        ))}
    </Card>
  );
}

function CalibrationCard() {
  const calibration = useApiData((signal) => api.calibration({ signal }), [], { intervalMs: 10_000 });
  const c = calibration.data;
  return (
    <Card
      title="Calibration"
      actions={
        <Link className="button" to="/calibration">
          Calibrate
        </Link>
      }
    >
      <ErrorBox error={calibration.error} />
      {c ? (
        <KeyValue
          items={[
            ["Version", c.version ?? "never calibrated"],
            ["Dark level", c.dark_v != null ? `${formatVoltage(c.dark_v)} (${formatDateTime(c.dark_measured_at)})` : <Badge kind="warn">missing</Badge>],
            [
              "Reference level",
              c.reference_v != null ? (
                `${formatVoltage(c.reference_v)} (${formatDateTime(c.reference_measured_at)})`
              ) : (
                <Badge kind="warn">missing: intensities stay in volts</Badge>
              ),
            ],
          ]}
        />
      ) : (
        !calibration.error && <Loading what="calibration" />
      )}
    </Card>
  );
}

function HardwareCard() {
  const { info, infoError } = useSystem();
  if (!info) return <Card title="Hardware">{infoError ? <ErrorBox error={infoError} /> : <Loading what="hardware" />}</Card>;
  const h = info.hardware;
  return (
    <Card
      title="Hardware"
      actions={
        <Link className="button" to="/hardware">
          Control
        </Link>
      }
    >
      {info.simulation && (
        <div className="alert alert-info" style={{ marginBottom: "0.6rem" }}>
          Simulation: no real hardware is driven.
        </div>
      )}
      <KeyValue
        items={[
          ["Controller", h.controller],
          ["Stage", `${h.stage_backend}${h.stage_version ? ` (${h.stage_version})` : ""}`],
          ["ADC", `${h.adc_backend}${h.adc_version ? ` (${h.adc_version})` : ""}`],
          ["Laser", h.laser_backend],
          ["Camera", h.camera_backend],
          ["Software", `${info.software_version} (API ${info.api_version})`],
        ]}
      />
    </Card>
  );
}

export default function Dashboard() {
  const scans = useApiData((signal) => api.listScans(8, 0, { signal }), [], { intervalMs: 5000 });
  return (
    <>
      <PageHeader title="Dashboard">
        <Link className="button btn-primary btn-large" to="/setup">
          New scan
        </Link>
      </PageHeader>
      <div className="grid grid-2">
        <SystemCard />
        <div className="stack">
          <ActiveScanCard />
          <CalibrationCard />
        </div>
        <Card
          title="Recent scans"
          className="span-all"
          actions={
            <Link className="button" to="/history">
              All scans
            </Link>
          }
        >
          <ErrorBox error={scans.error} />
          {scans.data === undefined ? (
            !scans.error && <Loading what="scans" />
          ) : scans.data.length === 0 ? (
            <Empty>No scans yet.</Empty>
          ) : (
            <ScanTable scans={scans.data} compact />
          )}
        </Card>
        <div className="span-all">
          <HardwareCard />
        </div>
      </div>
    </>
  );
}
