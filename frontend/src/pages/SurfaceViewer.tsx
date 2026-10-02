/**
 * 4 Surface Viewer: interactive 3-D surface (zoom / pan / rotate), height
 * map, confidence map, point cloud coloured by classification, X / Y
 * cross-sections at a selectable position, vertical scaling, statistics and
 * (re)construction with nearest / linear / cubic / rbf. Large grids are
 * reduced for display only.
 */
import type { PlotData, PlotLayout } from "plotly.js-dist-min";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import type { ApiError } from "../api/client";
import { api } from "../api/endpoints";
import {
  INTERPOLATION_METHODS,
  type InterpolationMethod,
  type OutlierMethod,
  type PointClassification,
  type ReconstructionBody,
  type SurfaceResult,
} from "../api/types";
import { Plot } from "../components/Plot";
import { ScanTable } from "../components/ScanTable";
import {
  Alert,
  Card,
  CheckField,
  Empty,
  ErrorBox,
  KeyValue,
  Loading,
  NumberField,
  PageHeader,
  Segmented,
  SelectField,
} from "../components/ui";
import { useAction, useApiData } from "../hooks/useApi";
import { downsampleGrid, strideSample } from "../lib/downsample";
import {
  formatDateTime,
  formatInteger,
  formatLength,
  formatNumber,
  formatPercent,
  humanize,
  lengthUnitLabel,
  shortId,
  toDisplayLength,
} from "../lib/format";
import { CONFIDENCE_COLORSCALE } from "../lib/grid";
import { usePreferences } from "../lib/prefs";
import { autoScale, crossSection, gridRange, pointsNearSection, sceneAspect, scaleToSlider, sliderToScale } from "../lib/surface";

type View = "surface" | "height" | "confidence" | "cloud" | "sections";

const VIEWS: { value: View; label: string }[] = [
  { value: "surface", label: "3-D surface" },
  { value: "height", label: "Height map" },
  { value: "confidence", label: "Confidence map" },
  { value: "cloud", label: "Point cloud" },
  { value: "sections", label: "Cross-sections" },
];

const CLASS_COLORS: Record<PointClassification, string> = {
  used: "#2e9d5b",
  low_confidence: "#e0a526",
  outlier: "#c8102e",
  invalid: "#7d8a99",
};

export default function SurfaceViewerRoute() {
  const { scanId } = useParams();
  if (scanId) return <SurfaceViewer key={scanId} scanId={scanId} />;
  return <SurfacePicker />;
}

function SurfacePicker() {
  const scans = useApiData((signal) => api.listScans(200, 0, { signal }), []);
  const candidates = (scans.data ?? []).filter((scan) => scan.mode === "confocal" && (scan.has_surface || scan.state === "complete"));
  return (
    <>
      <PageHeader title="Surface Viewer" />
      <Card title="Choose a scan">
        <ErrorBox error={scans.error} />
        {scans.data === undefined ? (
          !scans.error && <Loading what="scans" />
        ) : candidates.length === 0 ? (
          <Empty>No finished confocal scans yet.</Empty>
        ) : (
          <ScanTable scans={candidates} linkTo={(scan) => `/surface/${scan.id}`} />
        )}
      </Card>
    </>
  );
}

interface ReconstructForm {
  method: InterpolationMethod;
  min_confidence: string;
  outlier_method: OutlierMethod;
  outlier_threshold: string;
  grid_step_um: string;
  fill_gaps: boolean;
  rbf_smoothing: string;
}

const DEFAULT_RECONSTRUCT: ReconstructForm = {
  method: "linear",
  min_confidence: "0.5",
  outlier_method: "local_mad",
  outlier_threshold: "3.5",
  grid_step_um: "",
  fill_gaps: false,
  rbf_smoothing: "0",
};

function formFromRequest(surface: SurfaceResult | undefined): ReconstructForm {
  if (!surface) return DEFAULT_RECONSTRUCT;
  const r = surface.request;
  return {
    method: r.method,
    min_confidence: String(r.min_confidence),
    outlier_method: r.outlier_method,
    outlier_threshold: String(r.outlier_threshold),
    grid_step_um: r.grid_step_um != null ? String(r.grid_step_um) : "",
    fill_gaps: r.fill_gaps,
    rbf_smoothing: String(r.rbf_smoothing),
  };
}

function toRequest(form: ReconstructForm): { body?: ReconstructionBody; error?: string } {
  const minConfidence = Number(form.min_confidence);
  if (form.min_confidence.trim() === "" || !(minConfidence >= 0 && minConfidence <= 1)) {
    return { error: "Minimum confidence must be between 0 and 1." };
  }
  const threshold = Number(form.outlier_threshold);
  if (!(threshold > 0)) return { error: "Outlier threshold must be > 0." };
  const body: ReconstructionBody = {
    method: form.method,
    min_confidence: minConfidence,
    outlier_method: form.outlier_method,
    outlier_threshold: threshold,
    fill_gaps: form.fill_gaps,
  };
  if (form.grid_step_um.trim() !== "") {
    const step = Number(form.grid_step_um);
    if (!(step > 0)) return { error: "Grid step must be > 0 (or empty for the scan step)." };
    body.grid_step_um = step;
  }
  if (form.method === "rbf") {
    const smoothing = Number(form.rbf_smoothing);
    if (!(smoothing >= 0)) return { error: "RBF smoothing must be ≥ 0." };
    body.rbf_smoothing = smoothing;
  }
  return { body };
}

function ReconstructCard({
  scanId,
  surface,
  onDone,
}: {
  scanId: string;
  surface: SurfaceResult | undefined;
  onDone: (surface: SurfaceResult) => void;
}) {
  const [form, setForm] = useState<ReconstructForm>(() => formFromRequest(surface));
  const action = useAction();
  const { body, error } = toRequest(form);
  const set = <K extends keyof ReconstructForm>(key: K, value: ReconstructForm[K]): void => {
    setForm((current) => ({ ...current, [key]: value }));
  };
  return (
    <Card title="Reconstruct">
      <div className="stack">
        <SelectField
          label="Interpolation method"
          value={form.method}
          options={INTERPOLATION_METHODS.map((m) => ({ value: m, label: m === "rbf" ? "RBF (thin-plate spline)" : humanize(m) }))}
          onChange={(value) => {
            set("method", value);
          }}
        />
        <NumberField
          label="Minimum confidence"
          value={form.min_confidence}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => {
            set("min_confidence", value);
          }}
        />
        <SelectField
          label="Outlier removal"
          value={form.outlier_method}
          options={[
            { value: "local_mad", label: "Local MAD (k nearest)" },
            { value: "global_mad", label: "Global MAD (plane)" },
            { value: "none", label: "None" },
          ]}
          onChange={(value) => {
            set("outlier_method", value);
          }}
        />
        <NumberField
          label="Outlier threshold (robust z)"
          value={form.outlier_threshold}
          disabled={form.outlier_method === "none"}
          onChange={(value) => {
            set("outlier_threshold", value);
          }}
        />
        <NumberField
          label="Grid step"
          unit="µm"
          hint="Empty: the scan's XY step"
          value={form.grid_step_um}
          onChange={(value) => {
            set("grid_step_um", value);
          }}
        />
        {form.method === "rbf" && (
          <NumberField
            label="RBF smoothing"
            value={form.rbf_smoothing}
            min={0}
            onChange={(value) => {
              set("rbf_smoothing", value);
            }}
          />
        )}
        <CheckField
          label="Fill gaps"
          hint="Off: gaps stay empty (recommended, nothing is invented)"
          checked={form.fill_gaps}
          onChange={(value) => {
            set("fill_gaps", value);
          }}
        />
        {error && <div className="field-error">{error}</div>}
        <button
          type="button"
          className="btn-primary"
          disabled={!body || action.busy}
          onClick={() => {
            if (!body) return;
            void action.run(() => api.reconstruct(scanId, body)).then((result) => {
              if (result) onDone(result);
            });
          }}
        >
          {action.busy ? "Reconstructing…" : surface ? "Reconstruct again" : "Reconstruct"}
        </button>
        {action.busy && form.method === "rbf" && <span className="small muted">RBF on a large grid can take a while on the Pi.</span>}
        <ErrorBox error={action.error} title="Reconstruction failed" onDismiss={action.clearError} />
      </div>
    </Card>
  );
}

function StatisticsCard({ surface }: { surface: SurfaceResult }) {
  const { prefs } = usePreferences();
  const s = surface.statistics;
  const L = (value: number | null | undefined, digits = 3) => formatLength(value, prefs.lengthUnit, digits);
  return (
    <Card title="Surface statistics">
      <KeyValue
        items={[
          ["Points", `${formatInteger(s.n_used)} used of ${formatInteger(s.n_input)}`],
          ["Rejected", `${formatInteger(s.n_invalid)} invalid · ${formatInteger(s.n_low_confidence)} low confidence · ${formatInteger(s.n_outliers)} outliers`],
          ["Coverage", `${formatPercent(s.coverage_fraction)} (gaps ${formatPercent(s.gap_fraction)}, ${s.n_gaps} regions)`],
          ["Z min / max", `${L(s.z_min_um)} / ${L(s.z_max_um)}`],
          ["Z mean ± std", `${L(s.z_mean_um)} ± ${L(s.z_std_um)}`],
          ["Sa (arithmetic mean height)", L(s.sa_um)],
          ["Sq (RMS height)", L(s.sq_um)],
          ["Sz (peak to valley)", L(s.sz_um)],
          ["Ssk / Sku", `${formatNumber(s.ssk, 3)} / ${formatNumber(s.sku, 3)}`],
          [
            "Plane z = a·x + b·y + c",
            s.plane_coefficients
              ? `a ${s.plane_coefficients[0].toExponential(3)}, b ${s.plane_coefficients[1].toExponential(3)}, c ${formatNumber(s.plane_coefficients[2], 3)}`
              : "—",
          ],
          ["Mean confidence", formatPercent(s.mean_confidence)],
          ["Method", `${humanize(surface.request.method)} · min confidence ${surface.request.min_confidence} · outliers ${humanize(surface.request.outlier_method)}`],
          ["Created", formatDateTime(surface.created_at)],
        ]}
      />
      <p className="small muted" style={{ marginTop: "0.5rem" }}>Roughness parameters after plane removal (ISO 25178 style).</p>
    </Card>
  );
}

function SurfacePlots({ surface }: { surface: SurfaceResult }) {
  const { prefs } = usePreferences();
  const unit = prefs.lengthUnit;
  const u = lengthUnitLabel(unit);
  const [view, setView] = useState<View>("surface");
  const [along, setAlong] = useState<"x" | "y">("x");
  const [sectionIndex, setSectionIndex] = useState(() => Math.floor(surface.y_um.length / 2));

  const display = useMemo(() => {
    const conv = (v: number) => toDisplayLength(v, unit);
    const x = surface.x_um.map(conv);
    const y = surface.y_um.map(conv);
    const z = surface.z_um.map((row) => row.map((v) => (v === null ? null : conv(v))));
    const height = downsampleGrid(x, y, z, prefs.maxDisplayCells, "mean");
    const confidence = downsampleGrid(x, y, surface.confidence, prefs.maxDisplayCells, "mean");
    const range = gridRange(height.z);
    const xSpan = (x.at(-1) ?? 0) - (x[0] ?? 0);
    const ySpan = (y.at(-1) ?? 0) - (y[0] ?? 0);
    const zSpan = range ? range[1] - range[0] : 0;
    return { height, confidence, range, xSpan, ySpan, zSpan };
  }, [surface, unit, prefs.maxDisplayCells]);

  const [scale, setScale] = useState(() => autoScale(display.xSpan, display.ySpan, display.zSpan));
  const factor = display.height.factor;

  const surfaceData = useMemo<PlotData[]>(
    () => [
      {
        type: "surface",
        x: display.height.x,
        y: display.height.y,
        z: display.height.z,
        colorscale: prefs.colorMap,
        colorbar: { title: { text: `Z (${u})` }, thickness: 14 },
        contours: { z: { show: false, usecolormap: true, project: { z: true } } },
        hovertemplate: `X %{x:.2f}<br>Y %{y:.2f}<br>Z %{z:.3f} ${u}<extra></extra>`,
      },
    ],
    [display.height, prefs.colorMap, u],
  );

  const sceneLayout = useMemo<PlotLayout>(
    () => ({
      uirevision: `${surface.scan_id}-${surface.surface_id ?? 0}`,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        aspectmode: "manual",
        aspectratio: sceneAspect(display.xSpan, display.ySpan, display.zSpan, scale),
        xaxis: { title: { text: `X (${u})` } },
        yaxis: { title: { text: `Y (${u})` } },
        zaxis: { title: { text: `Z (${u})` } },
        camera: { eye: { x: 1.3, y: -1.5, z: 1.0 } },
      },
    }),
    [surface.scan_id, surface.surface_id, display.xSpan, display.ySpan, display.zSpan, scale, u],
  );

  const mapLayout = useMemo<PlotLayout>(
    () => ({
      uirevision: `map-${surface.scan_id}`,
      margin: { l: 70, r: 10, t: 10, b: 50 },
      xaxis: { title: { text: `X (${u})` }, constrain: "domain" },
      yaxis: { title: { text: `Y (${u})` }, scaleanchor: "x", scaleratio: 1 },
    }),
    [surface.scan_id, u],
  );

  const heightData = useMemo<PlotData[]>(
    () => [
      {
        type: "heatmap",
        x: display.height.x,
        y: display.height.y,
        z: display.height.z,
        colorscale: prefs.colorMap,
        hoverongaps: false,
        colorbar: { title: { text: `Z (${u})` }, thickness: 14 },
        hovertemplate: `X %{x:.2f}<br>Y %{y:.2f}<br>Z %{z:.3f} ${u}<extra></extra>`,
      },
    ],
    [display.height, prefs.colorMap, u],
  );

  const confidenceData = useMemo<PlotData[]>(
    () => [
      {
        type: "heatmap",
        x: display.confidence.x,
        y: display.confidence.y,
        z: display.confidence.z,
        zmin: 0,
        zmax: 1,
        colorscale: CONFIDENCE_COLORSCALE,
        hoverongaps: false,
        colorbar: { title: { text: "Confidence" }, thickness: 14 },
        hovertemplate: `X %{x:.2f}<br>Y %{y:.2f}<br>Confidence %{z:.3f}<extra></extra>`,
      },
    ],
    [display.confidence],
  );

  const cloudData = useMemo<PlotData[]>(() => {
    const groups: PointClassification[] = ["used", "low_confidence", "outlier", "invalid"];
    const withZ = surface.points.filter((p) => p.z_um !== null);
    const sampled = strideSample(withZ, prefs.maxCloudPoints);
    const conv = (v: number) => toDisplayLength(v, unit);
    return groups.map((group) => {
      const items = sampled.filter((p) => p.classification === group);
      return {
        type: "scatter3d",
        mode: "markers",
        name: `${humanize(group)} (${surface.points.filter((p) => p.classification === group).length})`,
        x: items.map((p) => conv(p.x_um)),
        y: items.map((p) => conv(p.y_um)),
        z: items.map((p) => conv(p.z_um ?? 0)),
        customdata: items.map((p) => [p.point_id, p.confidence]),
        marker: { size: 2.5, color: CLASS_COLORS[group] },
        hovertemplate: `Point %{customdata[0]}<br>Z %{z:.3f} ${u}<br>Confidence %{customdata[1]:.2f}<extra>${humanize(group)}</extra>`,
      };
    });
  }, [surface.points, prefs.maxCloudPoints, unit, u]);

  const cloudLayout = useMemo<PlotLayout>(
    () => ({ ...sceneLayout, uirevision: `cloud-${surface.scan_id}`, legend: { orientation: "h", y: 1.05 } }),
    [sceneLayout, surface.scan_id],
  );

  const axisLength = along === "x" ? surface.y_um.length : surface.x_um.length;
  const index = Math.min(sectionIndex, Math.max(0, axisLength - 1));
  const section = useMemo(() => crossSection(surface, along, index), [surface, along, index]);
  const step = (surface.x_um[1] ?? 0) - (surface.x_um[0] ?? 0) || (surface.y_um[1] ?? 0) - (surface.y_um[0] ?? 0) || 1;
  const sectionData = useMemo<PlotData[]>(() => {
    const conv = (v: number) => toDisplayLength(v, unit);
    const near = pointsNearSection(surface.points, along, section.position, Math.abs(step) / 2);
    return [
      {
        type: "scatter",
        mode: "lines",
        name: "Reconstructed surface",
        x: section.coordinate.map(conv),
        y: section.z.map((v) => (v === null ? null : conv(v))),
        line: { color: "#1f77d0", width: 2.5 },
        connectgaps: false,
      },
      ...(["used", "low_confidence", "outlier"] as PointClassification[]).map((group) => {
        const items = near.filter((p) => p.classification === group);
        return {
          type: "scatter",
          mode: "markers",
          name: `Measured: ${humanize(group)}`,
          x: items.map((p) => conv(along === "x" ? p.x_um : p.y_um)),
          y: items.map((p) => conv(p.z_um ?? 0)),
          marker: { size: 7, color: CLASS_COLORS[group] },
        };
      }),
    ];
  }, [surface.points, along, section, step, unit]);

  const sectionLayout = useMemo<PlotLayout>(
    () => ({
      uirevision: `section-${along}`,
      margin: { l: 70, r: 10, t: 10, b: 50 },
      xaxis: { title: { text: `${along.toUpperCase()} (${u})` } },
      yaxis: { title: { text: `Z (${u})` } },
      legend: { orientation: "h", y: 1.12 },
    }),
    [along, u],
  );

  const lineMapLayout = useMemo<PlotLayout>(() => {
    const pos = toDisplayLength(section.position, unit);
    return {
      ...mapLayout,
      shapes: [
        along === "x"
          ? { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: pos, y1: pos, line: { color: "#ff2fb4", width: 3 } }
          : { type: "line", yref: "paper", y0: 0, y1: 1, xref: "x", x0: pos, x1: pos, line: { color: "#ff2fb4", width: 3 } },
      ],
    };
  }, [mapLayout, section.position, along, unit]);

  return (
    <Card
      title="Surface"
      actions={<Segmented label="View" value={view} options={VIEWS} onChange={setView} />}
    >
      {view === "surface" && (
        <>
          <Plot data={surfaceData} layout={sceneLayout} height="min(68vh, 760px)" label="3-D surface" />
          <div className="row" style={{ marginTop: "0.5rem" }}>
            <label htmlFor="vscale">Vertical scaling</label>
            <input
              id="vscale"
              type="range"
              min={0}
              max={100}
              step={1}
              style={{ maxWidth: "26rem" }}
              value={scaleToSlider(scale)}
              onChange={(event) => {
                setScale(Math.max(1, sliderToScale(Number(event.target.value))));
              }}
              aria-valuetext={`${scale} times`}
            />
            <strong className="nowrap">{formatNumber(scale, scale < 10 ? 1 : 0)}×</strong>
            <button type="button" className="btn-small" onClick={() => setScale(1)}>
              True scale
            </button>
            <button type="button" className="btn-small" onClick={() => setScale(autoScale(display.xSpan, display.ySpan, display.zSpan))}>
              Auto
            </button>
            <span className="small muted">Drag to rotate, scroll to zoom, right-drag to pan. Gaps are left empty.</span>
          </div>
        </>
      )}
      {view === "height" && <Plot data={heightData} layout={mapLayout} height="min(68vh, 760px)" label="Height map" />}
      {view === "confidence" && (
        <Plot data={confidenceData} layout={mapLayout} height="min(68vh, 760px)" label="Confidence map" />
      )}
      {view === "cloud" && (
        <>
          <Plot data={cloudData} layout={cloudLayout} height="min(68vh, 760px)" label="Point cloud by classification" />
          <p className="small muted">
            Points without a surface height (invalid) have no Z and are not drawn.
            {surface.points.length > prefs.maxCloudPoints && ` Showing about ${formatInteger(prefs.maxCloudPoints)} of ${formatInteger(surface.points.length)} points.`}
          </p>
        </>
      )}
      {view === "sections" && (
        <div className="stack">
          <div className="row">
            <Segmented
              label="Section direction"
              value={along}
              options={[
                { value: "x", label: "Along X (fixed Y)" },
                { value: "y", label: "Along Y (fixed X)" },
              ]}
              onChange={(value) => {
                setAlong(value);
                setSectionIndex(Math.floor((value === "x" ? surface.y_um.length : surface.x_um.length) / 2));
              }}
            />
            <label htmlFor="section-pos">{along === "x" ? "Y" : "X"} position</label>
            <input
              id="section-pos"
              type="range"
              min={0}
              max={Math.max(0, axisLength - 1)}
              step={1}
              value={index}
              style={{ maxWidth: "30rem" }}
              onChange={(event) => {
                setSectionIndex(Number(event.target.value));
              }}
            />
            <strong className="nowrap">{formatLength(section.position, unit)}</strong>
          </div>
          <div className="grid grid-2">
            <Plot data={sectionData} layout={sectionLayout} height={420} label="Cross-section" />
            <Plot data={heightData} layout={lineMapLayout} height={420} label="Height map with the section line" />
          </div>
        </div>
      )}
      {factor > 1 && view !== "cloud" && view !== "sections" && (
        <p className="small muted">
          Display reduced {factor}×{factor} from the {surface.x_um.length}×{surface.y_um.length} grid (Settings → display limit); the stored surface is unchanged.
        </p>
      )}
    </Card>
  );
}

function SurfaceViewer({ scanId }: { scanId: string }) {
  const scan = useApiData((signal) => api.getScan(scanId, { signal }), [scanId]);
  const loaded = useApiData((signal) => api.surface(scanId, { signal }), [scanId]);
  const [override, setOverride] = useState<SurfaceResult | null>(null);
  const surface = override ?? loaded.data;
  const notFound: ApiError | null = loaded.error?.status === 404 ? loaded.error : null;

  return (
    <>
      <PageHeader title={`Surface Viewer · ${scan.data?.name ?? shortId(scanId)}`}>
        <Link className="button" to={`/history/${scanId}`}>
          Scan details
        </Link>
        <Link className="button" to="/surface">
          Other scans
        </Link>
      </PageHeader>
      <div className="grid grid-sidebar">
        <div className="stack">
          {scan.error?.status === 404 ? (
            <Alert kind="error" title="Unknown scan">
              There is no scan with the id <code>{scanId}</code>.
            </Alert>
          ) : surface ? (
            <SurfacePlots key={`${surface.surface_id ?? 0}-${surface.created_at}`} surface={surface} />
          ) : loaded.loading ? (
            <Card title="Surface">
              <Loading what="surface" />
            </Card>
          ) : notFound ? (
            <Alert kind="info" title="No surface yet">
              {scan.data?.mode === "fixed_z"
                ? "Fixed-Z scans are intensity maps and have no surface."
                : "This scan has no reconstruction yet. Choose a method and reconstruct it."}
            </Alert>
          ) : (
            <ErrorBox error={loaded.error} />
          )}
          {surface?.gaps && surface.gaps.length > 0 && (
            <Card title={`Gaps (${surface.gaps.length})`}>
              <div className="table-wrap" style={{ maxHeight: "16rem" }}>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th className="num">Cells</th>
                      <th className="num">Area (µm²)</th>
                      <th className="num">Centroid X, Y (µm)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {surface.gaps.map((gap) => (
                      <tr key={gap.label}>
                        <td>{gap.label}</td>
                        <td className="num">{gap.n_cells}</td>
                        <td className="num">{formatNumber(gap.area_um2, 0)}</td>
                        <td className="num">
                          {formatNumber(gap.centroid_x_um, 1)}, {formatNumber(gap.centroid_y_um, 1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
        <div className="stack">
          {surface && <StatisticsCard surface={surface} />}
          {scan.data?.mode !== "fixed_z" && (
            <ReconstructCard
              key={surface?.surface_id ?? "none"}
              scanId={scanId}
              surface={surface}
              onDone={(result) => {
                setOverride(result);
              }}
            />
          )}
        </div>
      </div>
    </>
  );
}
