/**
 * 8 Scan History: every scan with state / interrupted badges (paged), and the
 * detail view of one scan: summary, points table, per-point I(Z) profile
 * viewer, links to the Surface Viewer and the (advisory) ML results.
 */
import { useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { api } from "../api/endpoints";
import { isActiveState, type PointStatus, type ScanConfig, type ScanSummary } from "../api/types";
import { ProfileRecordChart } from "../components/ProfileCharts";
import { ScanTable } from "../components/ScanTable";
import {
  Alert,
  Badge,
  Card,
  Empty,
  ErrorBox,
  KeyValue,
  Loading,
  PageHeader,
  pointStatusKind,
  ScanStateBadge,
  SelectField,
} from "../components/ui";
import { useAction, useApiData } from "../hooks/useApi";
import {
  formatDateTime,
  formatDuration,
  formatInteger,
  formatLength,
  formatNumber,
  formatPercent,
  humanize,
} from "../lib/format";
import { usePreferences } from "../lib/prefs";

const HISTORY_PAGE = 50;
const POINT_PAGE = 100;

export default function ScanHistoryRoute() {
  const { scanId } = useParams();
  if (scanId) return <ScanDetail key={scanId} scanId={scanId} />;
  return <ScanHistory />;
}

function ScanHistory() {
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState<"all" | "complete" | "interrupted" | "active">("all");
  const scans = useApiData((signal) => api.listScans(HISTORY_PAGE, page * HISTORY_PAGE, { signal }), [page], {
    intervalMs: 10_000,
  });
  const rows = (scans.data ?? []).filter((scan) => {
    switch (filter) {
      case "all":
        return true;
      case "complete":
        return scan.state === "complete" && !scan.interrupted;
      case "interrupted":
        return scan.interrupted || scan.state === "error" || scan.state === "cancelled";
      case "active":
        return isActiveState(scan.state);
    }
  });
  return (
    <>
      <PageHeader title="Scan History">
        <SelectField
          label="Show"
          value={filter}
          options={[
            { value: "all", label: "All scans" },
            { value: "complete", label: "Complete" },
            { value: "interrupted", label: "Interrupted / cancelled / error" },
            { value: "active", label: "Running" },
          ]}
          onChange={setFilter}
        />
      </PageHeader>
      <Card>
        <ErrorBox error={scans.error} />
        {scans.data === undefined ? (
          !scans.error && <Loading what="scans" />
        ) : rows.length === 0 ? (
          <Empty>{page === 0 && filter === "all" ? "No scans yet." : "No scans on this page match."}</Empty>
        ) : (
          <ScanTable scans={rows} />
        )}
        <div className="row row-end" style={{ marginTop: "0.75rem" }}>
          <span className="muted small">Page {page + 1}</span>
          <button type="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
            Newer
          </button>
          <button
            type="button"
            disabled={(scans.data?.length ?? 0) < HISTORY_PAGE}
            onClick={() => setPage((p) => p + 1)}
          >
            Older
          </button>
        </div>
      </Card>
    </>
  );
}

function configItems(config: ScanConfig, unit: "um" | "mm"): [string, string][] {
  const L = (v: number) => formatLength(v, unit);
  const items: [string, string][] = [
    ["Mode", humanize(config.mode)],
    ["X range", `${L(config.x_start_um)} … ${L(config.x_stop_um)}`],
    ["Y range", `${L(config.y_start_um)} … ${L(config.y_stop_um)}`],
    ["XY step", L(config.xy_step_um)],
    ["Z centre", L(config.z_center_um)],
  ];
  if (config.mode === "confocal") {
    items.push(
      ["Z range", L(config.z_range_um)],
      ["Coarse / fine Z step", `${L(config.coarse_z_step_um)} / ${L(config.fine_z_step_um)}`],
      ["Fine range", L(config.fine_z_range_um)],
      ["Adaptive Z", config.adaptive_z ? `yes (${L(config.adaptive_z_range_um)})` : "no"],
    );
  }
  items.push(
    ["Order", humanize(config.order)],
    ["Samples per Z", `${config.samples_per_z} (${config.sampling_method})`],
    ["Settle time", `${config.settle_time_ms} ms`],
    ["Saturation threshold", config.processing?.saturation_v != null ? `${config.processing.saturation_v} V` : "ADC limit only (0.98 × full scale)"],
    ["Reconstruct on complete", config.reconstruct_on_complete ? `yes (${config.reconstruction?.method ?? "linear"})` : "no"],
  );
  return items;
}

function SummaryCard({ scan }: { scan: ScanSummary }) {
  const duration =
    scan.started_at && scan.finished_at
      ? (new Date(scan.finished_at).getTime() - new Date(scan.started_at).getTime()) / 1000
      : null;
  return (
    <Card title="Summary">
      <KeyValue
        items={[
          ["Id", <code key="id">{scan.id}</code>],
          ["Name", scan.name ?? "—"],
          ["State", <ScanStateBadge key="s" state={scan.state} interrupted={scan.interrupted} />],
          ["Points", `${formatInteger(scan.completed_points)} / ${formatInteger(scan.total_points)} (${formatPercent(scan.progress)})`],
          ["Created", formatDateTime(scan.created_at)],
          ["Started / finished", `${formatDateTime(scan.started_at)} / ${formatDateTime(scan.finished_at)}`],
          ["Duration", formatDuration(duration)],
          ["Calibration", scan.calibration_version != null ? `v${scan.calibration_version}` : "none"],
          ["Software", scan.software_version],
          [
            "Hardware",
            scan.hardware ? `${scan.hardware.stage_backend} / ${scan.hardware.adc_backend} / laser ${scan.hardware.laser_backend}` : "—",
          ],
          ["Data file", scan.data_file ? <code key="f">{scan.data_file}</code> : "—"],
        ]}
      />
      {scan.error_message && (
        <div style={{ marginTop: "0.6rem" }}>
          <Alert kind="error" title="Error">
            {scan.error_message}
          </Alert>
        </div>
      )}
    </Card>
  );
}

function MlCard({ scan }: { scan: ScanSummary }) {
  const [generation, setGeneration] = useState(0);
  const result = useApiData((signal) => api.mlResults(scan.id, { signal }), [scan.id, generation, scan.has_ml_result]);
  const models = useApiData((signal) => api.mlModels({ signal }), []);
  const analyse = useAction();
  const none = result.error?.status === 404;
  const data = result.data;
  const flagged = (data?.predictions ?? []).filter((p) => p.flagged);
  return (
    <Card title="ML results (advisory)">
      <div className="stack">
        <p className="small muted">ML only flags suspicious points; it never changes the measured surface heights.</p>
        {data ? (
          <>
            <KeyValue
              items={[
                ["Model", `${data.model.name} ${data.model.version} (${data.model.algorithm})`],
                ["Created", formatDateTime(data.created_at)],
                ["Threshold", data.threshold],
                ["Flagged", `${formatInteger(data.n_flagged)} of ${formatInteger(data.n_points)} points`],
              ]}
            />
            {flagged.length > 0 && (
              <div className="small">
                Flagged point ids: {flagged.slice(0, 60).map((p) => p.point_id).join(", ")}
                {flagged.length > 60 && ` … (+${flagged.length - 60})`}
              </div>
            )}
          </>
        ) : none ? (
          <p className="muted">No ML analysis for this scan.</p>
        ) : result.error ? (
          <ErrorBox error={result.error} />
        ) : (
          <Loading what="ML results" />
        )}
        {models.data && models.data.length > 0 && scan.mode === "confocal" && !isActiveState(scan.state) && (
          <button
            type="button"
            disabled={analyse.busy}
            onClick={() => {
              void analyse.run(() => api.analyseMl(scan.id)).then((r) => {
                if (r) setGeneration((g) => g + 1);
              });
            }}
          >
            {analyse.busy ? "Analysing…" : data ? "Analyse again" : "Run ML analysis"}
          </button>
        )}
        {models.data?.length === 0 && <p className="small muted">No ML model is deployed.</p>}
        <ErrorBox error={analyse.error} onDismiss={analyse.clearError} />
      </div>
    </Card>
  );
}

function PointsCard({ scan, selected, onSelect }: { scan: ScanSummary; selected: number | null; onSelect: (id: number) => void }) {
  const { prefs } = usePreferences();
  const [page, setPage] = useState(() => (selected !== null ? Math.floor(selected / POINT_PAGE) : 0));
  const [statusFilter, setStatusFilter] = useState<PointStatus | "all">("all");
  const [jump, setJump] = useState("");
  const points = useApiData(
    (signal) => api.scanPoints(scan.id, page * POINT_PAGE - 1, POINT_PAGE, { signal }),
    [scan.id, page],
    { intervalMs: isActiveState(scan.state) ? 5000 : undefined },
  );
  const pageCount = Math.max(1, Math.ceil(Math.max(scan.completed_points, (points.data?.length ?? 0) + page * POINT_PAGE) / POINT_PAGE));
  const rows = (points.data ?? []).filter((p) => statusFilter === "all" || p.status === statusFilter);
  const fixedZ = scan.mode === "fixed_z";
  const unit = prefs.lengthUnit;

  return (
    <Card
      title="Points"
      actions={
        <form
          className="row"
          onSubmit={(event) => {
            event.preventDefault();
            const id = Number(jump);
            if (Number.isInteger(id) && id >= 0) {
              setPage(Math.floor(id / POINT_PAGE));
              onSelect(id);
            }
          }}
        >
          <label htmlFor="jump-point" className="small">
            Point id
          </label>
          <input id="jump-point" type="number" min={0} style={{ width: "7rem" }} value={jump} onChange={(e) => setJump(e.target.value)} />
          <button type="submit" className="btn-small">
            Open
          </button>
        </form>
      }
    >
      <div className="row" style={{ marginBottom: "0.5rem" }}>
        <SelectField
          label="Status filter (this page)"
          value={statusFilter}
          options={[
            { value: "all", label: "All" },
            ...(["valid", "low_confidence", "no_peak", "peak_at_edge", "fit_failed", "measured", "aborted", "error"] as const).map((s) => ({
              value: s,
              label: humanize(s),
            })),
          ]}
          onChange={setStatusFilter}
        />
      </div>
      <ErrorBox error={points.error} />
      {points.data === undefined ? (
        !points.error && <Loading what="points" />
      ) : rows.length === 0 ? (
        <Empty>No points{statusFilter !== "all" ? " with this status on this page" : ""}.</Empty>
      ) : (
        <div className="table-wrap" style={{ maxHeight: "28rem" }}>
          <table>
            <thead>
              <tr>
                <th className="num">Id</th>
                <th className="num">ix, iy</th>
                <th className="num">X</th>
                <th className="num">Y</th>
                <th>Status</th>
                <th className="num">{fixedZ ? "Intensity" : "Surface Z"}</th>
                <th className="num">Confidence</th>
                <th className="num">SNR</th>
                <th className="num">FWHM</th>
                <th className="num">Z pos.</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr
                  key={p.point_id}
                  className={`clickable ${selected === p.point_id ? "selected" : ""}`}
                  onClick={() => onSelect(p.point_id)}
                  aria-selected={selected === p.point_id}
                >
                  <td className="num">
                    <button type="button" className="btn-small" onClick={() => onSelect(p.point_id)}>
                      {p.point_id}
                    </button>
                  </td>
                  <td className="num">
                    {p.ix}, {p.iy}
                  </td>
                  <td className="num">{formatLength(p.x_um, unit)}</td>
                  <td className="num">{formatLength(p.y_um, unit)}</td>
                  <td>
                    <Badge kind={pointStatusKind(p.status)}>{humanize(p.status)}</Badge>
                  </td>
                  <td className="num">{fixedZ ? formatNumber(p.intensity, 4) : formatLength(p.surface_z_um, unit, 3)}</td>
                  <td className="num">{formatPercent(p.confidence)}</td>
                  <td className="num">{formatNumber(p.snr, 1)}</td>
                  <td className="num">{formatLength(p.peak_width_um, unit)}</td>
                  <td className="num">{p.n_z_positions}</td>
                  <td>{(p.flags ?? []).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="row row-end" style={{ marginTop: "0.75rem" }}>
        <span className="muted small">
          Page {page + 1} of {pageCount} · {POINT_PAGE} points per page
        </span>
        <button type="button" disabled={page === 0} onClick={() => setPage(0)}>
          First
        </button>
        <button type="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          Previous
        </button>
        <button type="button" disabled={(points.data?.length ?? 0) < POINT_PAGE} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </Card>
  );
}

function ProfileCard({ scanId, pointId }: { scanId: string; pointId: number }) {
  const profile = useApiData((signal) => api.pointProfile(scanId, pointId, { signal, timeoutMs: 60_000 }), [scanId, pointId]);
  return (
    <Card title={`I(Z) profile of point ${pointId}`}>
      <ErrorBox error={profile.error} />
      {profile.data && profile.data.point_id === pointId ? (
        <ProfileRecordChart record={profile.data} />
      ) : (
        !profile.error && <Loading what="profile" />
      )}
    </Card>
  );
}

function ScanDetail({ scanId }: { scanId: string }) {
  const navigate = useNavigate();
  const { prefs } = usePreferences();
  const [params, setParams] = useSearchParams();
  const scan = useApiData((signal) => api.getScan(scanId, { signal }), [scanId], { intervalMs: 5000 });
  const mlRef = useRef<HTMLDivElement>(null);
  const pointParam = params.get("point");
  const selected = pointParam !== null && /^\d+$/.test(pointParam) ? Number(pointParam) : null;
  const select = (id: number): void => {
    setParams({ point: String(id) }, { replace: true });
  };

  if (scan.error?.status === 404) {
    return (
      <>
        <PageHeader title="Scan" />
        <Alert kind="error" title="Unknown scan">
          There is no scan with the id <code>{scanId}</code>. <Link to="/history">Back to the history</Link>
        </Alert>
      </>
    );
  }
  const data = scan.data;
  return (
    <>
      <PageHeader title={`Scan ${data?.name ?? scanId.slice(0, 8)}`}>
        <Link className="button" to="/history">
          All scans
        </Link>
        {data && isActiveState(data.state) && (
          <Link className="button btn-primary" to={`/live/${scanId}`}>
            Live view
          </Link>
        )}
        {data?.mode === "confocal" && (
          <Link className={`button ${data.has_surface ? "btn-primary" : ""}`} to={`/surface/${scanId}`}>
            {data.has_surface ? "Surface Viewer" : "Reconstruct surface"}
          </Link>
        )}
        <button
          type="button"
          onClick={() => {
            mlRef.current?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          ML results
        </button>
        {data && (
          <button
            type="button"
            onClick={() => {
              void navigate("/setup", { state: { repeat: data.config } });
            }}
            title="Open Scan Setup with this scan's settings"
          >
            Repeat scan
          </button>
        )}
      </PageHeader>
      <ErrorBox error={scan.error} />
      {!data ? (
        !scan.error && <Loading what="scan" />
      ) : (
        <div className="stack">
          <div className="grid grid-2">
            <SummaryCard scan={data} />
            <Card title="Configuration">
              <KeyValue items={configItems(data.config, prefs.lengthUnit)} />
            </Card>
          </div>
          <PointsCard scan={data} selected={selected} onSelect={select} />
          {selected !== null ? (
            <ProfileCard scanId={scanId} pointId={selected} />
          ) : (
            <Card title="I(Z) profile">
              <Empty>Select a point in the table to see its raw samples and processed curves.</Empty>
            </Card>
          )}
          <div ref={mlRef}>
            <MlCard scan={data} />
          </div>
        </div>
      )}
    </>
  );
}
