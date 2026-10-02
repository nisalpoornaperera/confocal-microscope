/**
 * 3 Live Scan: progress, stage position, live I(Z) graph (PROFILE events),
 * current point, completion and confidence maps (POINT events plus the stored
 * points), ETA, pause / resume / cancel. `/live` without an id opens the
 * active scan, or lists recent scans.
 */
import { useEffect, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";

import { api } from "../api/endpoints";
import { isActiveState, isTerminalState, type ScanPoint } from "../api/types";
import { LiveProfileChart } from "../components/ProfileCharts";
import { ScanMaps, type MapSelection } from "../components/ScanMaps";
import { ScanTable } from "../components/ScanTable";
import {
  Alert,
  Badge,
  Card,
  ConfirmButton,
  Empty,
  ErrorBox,
  KeyValue,
  Loading,
  PageHeader,
  pointStatusKind,
  ProgressBar,
  ScanStateBadge,
  Stat,
} from "../components/ui";
import { useAction, useApiData } from "../hooks/useApi";
import { useScanPoints } from "../hooks/useScanPoints";
import { useScanSocket } from "../hooks/useScanSocket";
import type { SocketStatus } from "../hooks/scanSocketMachine";
import {
  formatDuration,
  formatInteger,
  formatLength,
  formatNumber,
  formatPercent,
  humanize,
  shortId,
} from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";

const SOCKET_LABEL: Record<SocketStatus, { text: string; kind: "ok" | "warn" | "danger" | "info" | "neutral" }> = {
  idle: { text: "Not connected", kind: "neutral" },
  connecting: { text: "Connecting…", kind: "info" },
  open: { text: "Live", kind: "ok" },
  reconnecting: { text: "Reconnecting…", kind: "warn" },
  finished: { text: "Finished", kind: "neutral" },
  not_found: { text: "Unknown scan", kind: "danger" },
};

/** Wall-clock time, refreshed every `intervalMs`. */
function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now());
    }, intervalMs);
    return () => {
      clearInterval(timer);
    };
  }, [intervalMs]);
  return now;
}

function FinishTime({ remainingS }: { remainingS: number }) {
  const now = useNow(5000);
  return <>done ≈ {new Date(now + remainingS * 1000).toLocaleTimeString()}</>;
}

export default function LiveScanRoute() {
  const { scanId } = useParams();
  const { activeScanId } = useSystem();
  if (scanId) return <LiveScan key={scanId} scanId={scanId} />;
  if (activeScanId) return <Navigate to={`/live/${activeScanId}`} replace />;
  return <NoActiveScan />;
}

function NoActiveScan() {
  const scans = useApiData((signal) => api.listScans(10, 0, { signal }), []);
  return (
    <>
      <PageHeader title="Live Scan">
        <Link className="button btn-primary" to="/setup">
          Set up a scan
        </Link>
      </PageHeader>
      <Card title="No scan is running">
        <p className="muted">Start a scan from Scan Setup, or open a recent one:</p>
        <ErrorBox error={scans.error} />
        {scans.data && scans.data.length > 0 ? (
          <ScanTable scans={scans.data} compact linkTo={(scan) => `/live/${scan.id}`} />
        ) : (
          scans.data && <Empty>No scans yet.</Empty>
        )}
      </Card>
    </>
  );
}

function LiveScan({ scanId }: { scanId: string }) {
  const { prefs } = usePreferences();
  const { refreshStatus } = useSystem();
  const summary = useApiData((signal) => api.getScan(scanId, { signal }), [scanId], { intervalMs: 3000 });
  const scan = summary.data;
  const points = useScanPoints(scanId, scan?.config);
  const socket = useScanSocket(scanId, { onEvent: points.handleEvent });
  const control = useAction();
  const [selected, setSelected] = useState<MapSelection | null>(null);

  const progress = socket.progress;
  const state = progress?.state ?? scan?.state;
  const active = isActiveState(state);
  const terminal = isTerminalState(state);
  const fixedZ = scan?.mode === "fixed_z";
  const unit = prefs.lengthUnit;

  const act = (task: () => Promise<unknown>): void => {
    void control.run(task).then(() => {
      summary.reload();
      refreshStatus();
    });
  };

  if (socket.status === "not_found" || summary.error?.status === 404) {
    return (
      <>
        <PageHeader title="Live Scan" />
        <Alert kind="error" title="Unknown scan">
          There is no scan with the id <code>{scanId}</code>. <Link to="/live">Back</Link>
        </Alert>
      </>
    );
  }

  const selectedPoint: ScanPoint | undefined =
    selected && points.grid
      ? [...points.points.values()].find((p) => p.ix === selected.ix && p.iy === selected.iy)
      : undefined;
  const socketLabel = SOCKET_LABEL[socket.status];

  return (
    <>
      <PageHeader title={`Live Scan · ${scan?.name ?? shortId(scanId)}`}>
        {state && <ScanStateBadge state={state} interrupted={scan?.interrupted} />}
        <Badge kind={socketLabel.kind} title="WebSocket connection">
          {socketLabel.text}
        </Badge>
      </PageHeader>

      <div className="stack">
        <Card
          title="Progress"
          actions={
            <>
              {state === "paused" ? (
                <button type="button" className="btn-primary" disabled={control.busy} onClick={() => act(() => api.resumeScan(scanId))}>
                  Resume
                </button>
              ) : (
                <button
                  type="button"
                  disabled={control.busy || state !== "scanning"}
                  onClick={() => act(() => api.pauseScan(scanId))}
                  title="Pauses after the point being measured"
                >
                  Pause
                </button>
              )}
              <ConfirmButton
                className="btn-danger"
                disabled={control.busy || !active}
                question="Cancel this scan? Measured data is kept."
                confirmLabel="Cancel scan"
                onConfirm={() => act(() => api.cancelScan(scanId))}
              >
                Cancel
              </ConfirmButton>
            </>
          }
        >
          <ErrorBox error={control.error} onDismiss={control.clearError} />
          <ProgressBar fraction={progress?.progress ?? scan?.progress ?? 0} label="Scan progress" />
          <div className="stat-tiles" style={{ marginTop: "0.75rem" }}>
            <Stat
              label="Points"
              value={`${formatInteger(progress?.completed_points ?? scan?.completed_points)} / ${formatInteger(progress?.total_points ?? scan?.total_points)}`}
              hint={formatPercent(progress?.progress ?? scan?.progress)}
            />
            <Stat label="Elapsed" value={formatDuration(progress?.elapsed_s)} />
            <Stat
              label="Remaining (ETA)"
              value={terminal ? "—" : formatDuration(progress?.estimated_remaining_s)}
              hint={
                !terminal && progress?.estimated_remaining_s != null ? (
                  <FinishTime remainingS={progress.estimated_remaining_s} />
                ) : undefined
              }
            />
            <Stat label="Current point" value={progress?.current_point_id ?? "—"} />
            <Stat
              label="Stage X / Y"
              value={`${formatNumber(progress?.current_x_um, 1)} / ${formatNumber(progress?.current_y_um, 1)}`}
              hint="µm"
            />
            <Stat label="Stage Z" value={formatNumber(progress?.current_z_um, 2)} hint="µm" />
            <Stat label="Intensity" value={formatNumber(progress?.current_intensity, 4)} />
          </div>
          {socket.lastMessage && <p className="small muted" style={{ marginTop: "0.6rem" }}>{socket.lastMessage}</p>}
          {socket.lastError && (
            <Alert kind="error" title="Scan error">
              {socket.lastError}
            </Alert>
          )}
          {scan?.error_message && terminal && (
            <Alert kind="error" title="The scan ended with an error">
              {scan.error_message}
            </Alert>
          )}
          {terminal && scan && (
            <div className="row" style={{ marginTop: "0.75rem" }}>
              <Alert kind={state === "complete" ? "ok" : "warning"} title={`Scan ${humanize(state)}`}>
                {scan.interrupted ? "Not all points were measured; the measured data is kept." : "All points measured."}
              </Alert>
              {scan.has_surface && (
                <Link className="button btn-primary" to={`/surface/${scanId}`}>
                  Open Surface Viewer
                </Link>
              )}
              <Link className="button" to={`/history/${scanId}`}>
                Scan details
              </Link>
            </div>
          )}
        </Card>

        {!fixedZ && (
          <Card title="Live I(Z)">
            <LiveProfileChart profile={socket.profile} />
          </Card>
        )}

        <Card title="Maps" actions={points.loading ? <span className="small muted">Loading points…</span> : undefined}>
          <ErrorBox error={points.error} title="Could not load the stored points" />
          {points.grid ? (
            <ScanMaps
              grid={points.grid}
              version={points.version}
              fixedZ={fixedZ}
              current={
                active && progress?.current_x_um != null && progress.current_y_um != null
                  ? { x_um: progress.current_x_um, y_um: progress.current_y_um }
                  : null
              }
              onSelect={setSelected}
            />
          ) : summary.error ? (
            <ErrorBox error={summary.error} />
          ) : (
            <Loading what="scan" />
          )}
        </Card>

        {selected && (
          <Card
            title={`Point at grid (${selected.ix}, ${selected.iy})`}
            actions={
              <button type="button" className="btn-small" onClick={() => setSelected(null)}>
                Close
              </button>
            }
          >
            {selectedPoint ? (
              <div className="row" style={{ alignItems: "flex-start", gap: "2rem" }}>
                <KeyValue
                  items={[
                    ["Point id", selectedPoint.point_id],
                    ["Status", <Badge key="s" kind={pointStatusKind(selectedPoint.status)}>{humanize(selectedPoint.status)}</Badge>],
                    ["X / Y", `${formatLength(selectedPoint.x_um, unit)} / ${formatLength(selectedPoint.y_um, unit)}`],
                    [fixedZ ? "Intensity" : "Surface Z", fixedZ ? formatNumber(selectedPoint.intensity, 4) : formatLength(selectedPoint.surface_z_um, unit, 3)],
                    ["Confidence", formatPercent(selectedPoint.confidence)],
                    ["SNR", formatNumber(selectedPoint.snr, 1)],
                    ["Flags", (selectedPoint.flags ?? []).join(", ") || "none"],
                  ]}
                />
                <Link className="button" to={`/history/${scanId}?point=${selectedPoint.point_id}`}>
                  Open stored I(Z) profile
                </Link>
              </div>
            ) : (
              <p className="muted">Not measured yet.</p>
            )}
          </Card>
        )}
      </div>
    </>
  );
}
