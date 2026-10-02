/**
 * 2-D maps of a scan built from its points: the completion / status map and
 * the confidence map (fixed-Z scans: the intensity map). Large grids are
 * reduced for display only (status: most severe per block, values: mean).
 */
import type { PlotData, PlotLayout } from "plotly.js-dist-min";
import { useMemo } from "react";

import { downsampleGrid } from "../lib/downsample";
import { lengthUnitLabel, toDisplayLength } from "../lib/format";
import {
  CONFIDENCE_COLORSCALE,
  STATUS_CATEGORY,
  STATUS_CATEGORY_COLORS,
  STATUS_CATEGORY_LABELS,
  type ScanGrid,
} from "../lib/grid";
import { usePreferences } from "../lib/prefs";
import { Plot, type PlotClickPoint } from "./Plot";

/** Discrete colour scale: category i occupies [(i)/n, (i+1)/n]. */
export function categoryColorscale(colors: readonly string[]): [number, string][] {
  const n = colors.length;
  const scale: [number, string][] = [];
  colors.forEach((color, i) => {
    scale.push([i / n, color], [(i + 1) / n, color]);
  });
  return scale;
}

export interface MapSelection {
  ix: number;
  iy: number;
}

function nearestIndex(values: readonly number[], target: number): number {
  let best = 0;
  let bestDistance = Infinity;
  values.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}

export function ScanMaps({
  grid,
  version,
  fixedZ,
  current,
  onSelect,
  height = 380,
}: {
  grid: ScanGrid;
  /** Changes when points were added. */
  version: number;
  fixedZ: boolean;
  /** Point being measured (marker). */
  current?: { x_um: number; y_um: number } | null;
  /** Click on a cell: grid indices of the nearest measured position. */
  onSelect?: (selection: MapSelection) => void;
  height?: number;
}) {
  const { prefs } = usePreferences();
  const unit = prefs.lengthUnit;
  const unitLabel = lengthUnitLabel(unit);

  const maps = useMemo(() => {
    const x = grid.x.map((value) => toDisplayLength(value, unit));
    const y = grid.y.map((value) => toDisplayLength(value, unit));
    const status = downsampleGrid(x, y, grid.categoryMatrix(), prefs.maxDisplayCells, "max");
    const values = downsampleGrid(
      x,
      y,
      grid.matrix(fixedZ ? grid.intensity : grid.confidence),
      prefs.maxDisplayCells,
      "mean",
    );
    return { status, values };
    // `version` signals in-place grid updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid, version, fixedZ, unit, prefs.maxDisplayCells]);

  const markerX = current ? toDisplayLength(current.x_um, unit) : null;
  const markerY = current ? toDisplayLength(current.y_um, unit) : null;
  const marker = useMemo<PlotData | null>(
    () =>
      markerX === null || markerY === null
        ? null
        : {
            type: "scatter",
            mode: "markers",
            x: [markerX],
            y: [markerY],
            marker: { symbol: "x-thin-open", size: 16, line: { width: 3, color: "#ff2fb4" } },
            name: "Current point",
            hoverinfo: "skip",
            showlegend: false,
          },
    [markerX, markerY],
  );

  const statusData = useMemo<PlotData[]>(
    () => [
      {
        type: "heatmap",
        x: maps.status.x,
        y: maps.status.y,
        z: maps.status.z,
        zmin: -0.5,
        zmax: STATUS_CATEGORY_LABELS.length - 0.5,
        colorscale: categoryColorscale(STATUS_CATEGORY_COLORS),
        showscale: false,
        hovertemplate: `X %{x:.2f} ${unitLabel}<br>Y %{y:.2f} ${unitLabel}<extra></extra>`,
        xgap: maps.status.x.length <= 60 ? 1 : 0,
        ygap: maps.status.y.length <= 60 ? 1 : 0,
      },
      ...(marker ? [marker] : []),
    ],
    [maps.status, unitLabel, marker],
  );

  const valueData = useMemo<PlotData[]>(
    () => [
      {
        type: "heatmap",
        x: maps.values.x,
        y: maps.values.y,
        z: maps.values.z,
        colorscale: fixedZ ? prefs.colorMap : CONFIDENCE_COLORSCALE,
        zmin: fixedZ ? undefined : 0,
        zmax: fixedZ ? undefined : 1,
        colorbar: { title: { text: fixedZ ? "Intensity" : "Confidence" }, thickness: 14 },
        hovertemplate: `X %{x:.2f} ${unitLabel}<br>Y %{y:.2f} ${unitLabel}<br>${fixedZ ? "Intensity" : "Confidence"} %{z:.3f}<extra></extra>`,
        hoverongaps: false,
      },
      ...(marker ? [marker] : []),
    ],
    [maps.values, fixedZ, prefs.colorMap, unitLabel, marker],
  );

  const layout = useMemo<PlotLayout>(
    () => ({
      uirevision: "scan-maps",
      margin: { l: 60, r: 10, t: 10, b: 50 },
      xaxis: { title: { text: `X (${unitLabel})` }, constrain: "domain" },
      yaxis: { title: { text: `Y (${unitLabel})` }, scaleanchor: "x", scaleratio: 1 },
      showlegend: false,
    }),
    [unitLabel],
  );

  const handleClick = onSelect
    ? (point: PlotClickPoint) => {
        if (typeof point.x !== "number" || typeof point.y !== "number") return;
        const xs = grid.x.map((value) => toDisplayLength(value, unit));
        const ys = grid.y.map((value) => toDisplayLength(value, unit));
        onSelect({ ix: nearestIndex(xs, point.x), iy: nearestIndex(ys, point.y) });
      }
    : undefined;

  const factor = maps.status.factor;
  return (
    <div className="grid grid-2">
      <div>
        <h3>Completion map</h3>
        <Plot data={statusData} layout={layout} height={height} label="Scan completion map" onClick={handleClick} />
        <div className="legend-row" aria-label="Completion map legend">
          {STATUS_CATEGORY_LABELS.map((label, index) =>
            fixedZ && (index === STATUS_CATEGORY.low_confidence || index === STATUS_CATEGORY.failed) ? null : (
              <span key={label}>
                <span className="legend-swatch" style={{ background: STATUS_CATEGORY_COLORS[index] }} />
                {label} ({grid.countOf(index).toLocaleString("en-US")})
              </span>
            ),
          )}
        </div>
      </div>
      <div>
        <h3>{fixedZ ? "Intensity map" : "Confidence map"}</h3>
        <Plot
          data={valueData}
          layout={layout}
          height={height}
          label={fixedZ ? "Intensity map" : "Confidence map"}
          onClick={handleClick}
        />
        {factor > 1 && (
          <p className="small muted">
            Display reduced {factor}×{factor} ({grid.nx}×{grid.ny} grid); the data is not changed.
          </p>
        )}
      </div>
    </div>
  );
}
