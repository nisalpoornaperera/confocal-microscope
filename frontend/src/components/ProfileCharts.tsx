/**
 * I(Z) charts: the live profile of the point being measured (PROFILE events)
 * and the complete stored profile of one point (`GET .../profile/{point_id}`)
 * with raw samples, aggregated, normalized and filtered curves.
 */
import type { PlotData, PlotLayout } from "plotly.js-dist-min";
import { useMemo, useState } from "react";

import type { LiveProfile } from "../api/events";
import type { ProfileRecord } from "../api/types";
import { minMaxDecimate } from "../lib/downsample";
import { formatLength, formatNumber, formatPercent, formatVoltage, humanize } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { Plot } from "./Plot";
import { Badge, CheckField, KeyValue, pointStatusKind } from "./ui";

const PHASE_NAMES = ["Coarse", "Fine", "Fixed"] as const;
const PHASE_COLORS = ["#7d8a99", "#1f77d0", "#2e9d5b"] as const;

/** Above this many raw samples the raw-sample cloud is thinned for display. */
const MAX_RAW_MARKERS = 20_000;

export function LiveProfileChart({ profile, height = 340 }: { profile: LiveProfile | null; height?: number }) {
  const data = useMemo<PlotData[]>(() => {
    if (!profile) return [];
    const traces: PlotData[] = [];
    for (const phase of [0, 1, 2]) {
      const z: number[] = [];
      const y: (number | null)[] = [];
      profile.phase.forEach((p, i) => {
        if (p !== phase) return;
        z.push(profile.z_um[i] ?? Number.NaN);
        y.push(profile.intensity[i] ?? null);
      });
      if (z.length === 0) continue;
      traces.push({
        type: "scatter",
        mode: phase === 0 ? "lines+markers" : "lines+markers",
        name: PHASE_NAMES[phase],
        x: z,
        y,
        marker: { size: phase === 0 ? 6 : 5, color: PHASE_COLORS[phase] },
        line: { width: phase === 0 ? 1.5 : 2.5, color: PHASE_COLORS[phase] },
        connectgaps: false,
      });
    }
    return traces;
  }, [profile]);

  const layout = useMemo<PlotLayout>(
    () => ({
      uirevision: profile?.point_id ?? "none",
      margin: { l: 70, r: 10, t: 10, b: 50 },
      xaxis: { title: { text: "Z (µm)" } },
      yaxis: { title: { text: "Intensity" } },
      legend: { orientation: "h", y: 1.12 },
    }),
    [profile?.point_id],
  );

  if (!profile) {
    return (
      <div className="empty" style={{ height }}>
        The live I(Z) curve appears when the first Z sweep starts.
      </div>
    );
  }
  return (
    <>
      <div className="small muted">
        Point {profile.point_id} at X {formatNumber(profile.x_um, 1)} µm, Y {formatNumber(profile.y_um, 1)} µm ·{" "}
        {profile.z_um.length} Z positions
      </div>
      <Plot data={data} layout={layout} height={height} label={`Live I(Z) of point ${profile.point_id}`} />
    </>
  );
}

export function ProfileRecordChart({ record, height = 420 }: { record: ProfileRecord; height?: number }) {
  const { prefs } = usePreferences();
  const [show, setShow] = useState({ raw: true, aggregated: true, normalized: true, filtered: true, coarse: true });
  const analysis = record.analysis;
  const filteredUnits = analysis?.signal_units ?? "volts";

  const data = useMemo<PlotData[]>(() => {
    const traces: PlotData[] = [];
    const z = record.z_reported_um;
    const keep = (i: number): boolean => show.coarse || record.phase[i] !== 0;
    if (show.raw) {
      const rx: number[] = [];
      const ry: number[] = [];
      record.voltage_v.forEach((samples, i) => {
        if (!keep(i)) return;
        for (const v of samples) {
          rx.push(z[i] ?? Number.NaN);
          ry.push(v);
        }
      });
      const step = Math.max(1, Math.ceil(rx.length / MAX_RAW_MARKERS));
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: step > 1 ? `Raw samples (1 in ${step})` : "Raw samples",
        x: step > 1 ? rx.filter((_, i) => i % step === 0) : rx,
        y: step > 1 ? ry.filter((_, i) => i % step === 0) : ry,
        marker: { size: 3, color: "rgba(125,138,153,0.55)" },
        yaxis: "y",
      });
    }
    // Coarse and fine sweeps overlap in Z: draw each sweep in Z order, with a
    // break between sweeps, so the line never jumps back across the plot.
    const order = z
      .map((_, i) => i)
      .filter(keep)
      .sort((a, b) => (record.phase[a] ?? 0) - (record.phase[b] ?? 0) || (z[a] ?? 0) - (z[b] ?? 0));
    const series = (values: readonly (number | null)[] | null | undefined): { x: number[]; y: (number | null)[] } => {
      const x: number[] = [];
      const y: (number | null)[] = [];
      let previousPhase: number | undefined;
      for (const i of order) {
        const phase = record.phase[i];
        if (previousPhase !== undefined && phase !== previousPhase) {
          x.push(z[i] ?? Number.NaN);
          y.push(null);
        }
        previousPhase = phase;
        x.push(z[i] ?? Number.NaN);
        y.push(values?.[i] ?? null);
      }
      return minMaxDecimate(x, y, 4000);
    };
    if (show.aggregated) {
      const s = series(record.voltage_agg_v);
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: `Aggregated (${record.sampling_method}) V`,
        ...s,
        marker: { size: 4 },
        line: { color: "#1f77d0", width: 2 },
        yaxis: "y",
      });
    }
    if (show.normalized && record.normalized) {
      const s = series(record.normalized);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Normalized",
        ...s,
        line: { color: "#2e9d5b", width: 2 },
        yaxis: "y2",
        connectgaps: false,
      });
    }
    if (show.filtered && record.filtered) {
      const s = series(record.filtered);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `Filtered (${filteredUnits})`,
        ...s,
        line: { color: "#d9622b", width: 2.5, dash: "dot" },
        yaxis: filteredUnits === "normalized" ? "y2" : "y",
        connectgaps: false,
      });
    }
    return traces;
  }, [record, show, filteredUnits]);

  const surfaceZ = analysis?.surface_z_um ?? null;
  const layout = useMemo<PlotLayout>(() => {
    const shapes: Record<string, unknown>[] = [];
    if (surfaceZ !== null) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: surfaceZ,
        x1: surfaceZ,
        y0: 0,
        y1: 1,
        line: { color: "#c8102e", width: 2, dash: "dash" },
      });
    }
    if (record.dark_v != null) {
      shapes.push({
        type: "line",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: record.dark_v,
        y1: record.dark_v,
        line: { color: "#555", width: 1, dash: "dot" },
      });
    }
    return {
      uirevision: `${record.scan_id}-${record.point_id}`,
      margin: { l: 70, r: 70, t: 10, b: 50 },
      xaxis: { title: { text: "Z reported (µm)" } },
      yaxis: { title: { text: "Voltage (V)" } },
      yaxis2: { title: { text: "Normalized" }, overlaying: "y", side: "right", showgrid: false },
      legend: { orientation: "h", y: 1.14 },
      shapes,
    };
  }, [surfaceZ, record.dark_v, record.scan_id, record.point_id]);

  const toggle = (key: keyof typeof show) => (checked: boolean) => {
    setShow((current) => ({ ...current, [key]: checked }));
  };

  return (
    <div className="stack">
      <div className="row" style={{ gap: "1.25rem" }}>
        <CheckField label="Raw samples" checked={show.raw} onChange={toggle("raw")} />
        <CheckField label="Aggregated" checked={show.aggregated} onChange={toggle("aggregated")} />
        <CheckField
          label="Normalized"
          checked={show.normalized}
          onChange={toggle("normalized")}
          disabled={!record.normalized}
        />
        <CheckField label="Filtered" checked={show.filtered} onChange={toggle("filtered")} disabled={!record.filtered} />
        <CheckField label="Include coarse sweep" checked={show.coarse} onChange={toggle("coarse")} />
      </div>
      <Plot data={data} layout={layout} height={height} label={`Stored I(Z) profile of point ${record.point_id}`} />
      <p className="small muted">
        Dashed red line: surface Z. Dotted line: dark level. {record.z_um.length} Z positions ×{" "}
        {record.raw_counts[0]?.length ?? 0} samples, gain {record.gain}, calibration v{record.calibration_version ?? "—"}.
      </p>
      {analysis && (
        <div className="grid grid-2">
          <KeyValue
            items={[
              ["Status", <Badge key="s" kind={pointStatusKind(analysis.status)}>{humanize(analysis.status)}</Badge>],
              ["Surface Z", `${formatLength(analysis.surface_z_um, prefs.lengthUnit, 3)} (${analysis.surface_method ?? "—"})`],
              ["Peak Z (filtered max)", formatLength(analysis.peak_z_um, prefs.lengthUnit, 3)],
              ["Parabolic centre", formatLength(analysis.parabolic?.center_um, prefs.lengthUnit, 3)],
              ["Gaussian centre", formatLength(analysis.gaussian?.center_um, prefs.lengthUnit, 3)],
              ["Confidence", formatPercent(analysis.confidence)],
            ]}
          />
          <KeyValue
            items={[
              ["SNR", formatNumber(analysis.snr, 1)],
              ["FWHM", formatLength(analysis.fwhm_um, prefs.lengthUnit, 2)],
              ["Relative prominence", formatNumber(analysis.relative_prominence, 3)],
              ["Fit residual", formatNumber(analysis.fit_residual, 4)],
              ["Peaks / secondary ratio", `${analysis.n_peaks} / ${formatNumber(analysis.secondary_peak_ratio, 3)}`],
              ["Saturated samples", formatPercent(analysis.saturated_fraction)],
              ["Dark / reference", `${formatVoltage(record.dark_v)} / ${formatVoltage(record.reference_v)}`],
              ["Flags", analysis.flags && analysis.flags.length > 0 ? analysis.flags.join(", ") : "none"],
            ]}
          />
        </div>
      )}
    </div>
  );
}
