/**
 * Small typed React wrapper around Plotly.
 *
 * `plotly.js-dist-min` (~4.6 MB) is loaded with a dynamic import the first
 * time a plot is shown, so it lives in its own chunk and pages without plots
 * never download it. Updates use `Plotly.react` (diffing, keeps zoom when the
 * layout's `uirevision` stays the same); the plot follows its container size.
 */
import type { PlotConfig, PlotData, PlotLayout, PlotlyStatic } from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";

import { usePreferences } from "../lib/prefs";

let plotlyPromise: Promise<PlotlyStatic> | null = null;

/** Load (once) and return the Plotly module. */
export function loadPlotly(): Promise<PlotlyStatic> {
  plotlyPromise ??= import("plotly.js-dist-min")
    .then((module) => module.default)
    .catch((error: unknown) => {
      plotlyPromise = null; // allow a retry
      throw error;
    });
  return plotlyPromise;
}

export interface PlotClickPoint {
  x: unknown;
  y: unknown;
  z?: unknown;
  pointIndex?: number | number[];
  curveNumber: number;
  customdata?: unknown;
}

export interface PlotProps {
  data: PlotData[];
  layout?: PlotLayout;
  config?: PlotConfig;
  /** CSS height of the plot area (default 360px). */
  height?: number | string;
  /** Accessible description of what the plot shows. */
  label: string;
  onClick?: (point: PlotClickPoint) => void;
  className?: string;
}

const THEME = {
  light: { text: "#1d232b", grid: "#d5dae0", zero: "#9aa3ad", bg: "#ffffff" },
  dark: { text: "#e3e8ee", grid: "#36404b", zero: "#6b7682", bg: "#161b21" },
};

function axisStyle(colors: (typeof THEME)["light"]): Record<string, unknown> {
  return { gridcolor: colors.grid, zerolinecolor: colors.zero, linecolor: colors.grid, color: colors.text };
}

function mergeRecord(base: Record<string, unknown>, extra: unknown): Record<string, unknown> {
  return typeof extra === "object" && extra !== null ? { ...base, ...(extra as Record<string, unknown>) } : base;
}

/** Theme defaults merged under the caller's layout (axes and 3-D scene axes included). */
export function themedLayout(theme: "light" | "dark", layout: PlotLayout = {}): PlotLayout {
  const colors = THEME[theme];
  const axis = axisStyle(colors);
  const merged: PlotLayout = {
    autosize: true,
    margin: { l: 60, r: 20, t: 30, b: 50 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: colors.bg,
    hoverlabel: { font: { size: 14 } },
    legend: { orientation: "h", y: -0.2 },
    ...layout,
    font: mergeRecord({ family: "system-ui, 'Segoe UI', Roboto, sans-serif", size: 14, color: colors.text }, layout.font),
  };
  for (const key of Object.keys(merged)) {
    if (/^[xy]axis\d*$/.test(key)) merged[key] = mergeRecord(axis, merged[key]);
  }
  if (!("xaxis" in merged)) merged.xaxis = axis;
  if (!("yaxis" in merged)) merged.yaxis = axis;
  if ("scene" in merged) {
    const scene = mergeRecord({}, merged.scene);
    for (const key of ["xaxis", "yaxis", "zaxis"]) {
      scene[key] = mergeRecord({ ...axis, backgroundcolor: colors.bg, showbackground: true }, scene[key]);
    }
    merged.scene = scene;
  }
  return merged;
}

const DEFAULT_CONFIG: PlotConfig = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
  // Downloads are blocked in kiosk browsers anyway; keep the toolbar short.
  modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "lasso2d", "select2d"],
};

export function Plot({ data, layout, config, height = 360, label, onClick, className }: PlotProps) {
  const { resolvedTheme } = usePreferences();
  const containerRef = useRef<HTMLDivElement>(null);
  const plotlyRef = useRef<PlotlyStatic | null>(null);
  const onClickRef = useRef(onClick);
  const listenerAttached = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    onClickRef.current = onClick;
  });

  useEffect(() => {
    let cancelled = false;
    loadPlotly()
      .then((plotly) => {
        if (cancelled) return;
        plotlyRef.current = plotly;
        setReady(true);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const plotly = plotlyRef.current;
    const element = containerRef.current;
    if (!ready || !plotly || !element) return;
    let cancelled = false;
    plotly
      .react(element, data, themedLayout(resolvedTheme, layout), { ...DEFAULT_CONFIG, ...config })
      .then((plotElement) => {
        if (cancelled || listenerAttached.current) return;
        listenerAttached.current = true;
        plotElement.on("plotly_click", (event: unknown) => {
          const points = (event as { points?: PlotClickPoint[] } | null)?.points;
          const first = points?.[0];
          if (first) onClickRef.current?.(first);
        });
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [ready, data, layout, config, resolvedTheme]);

  useEffect(() => {
    const element = containerRef.current;
    if (!ready || !element) return undefined;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const plotly = plotlyRef.current;
        if (plotly && element.isConnected) plotly.Plots.resize(element);
      });
    });
    observer.observe(element);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [ready]);

  useEffect(() => {
    const element = containerRef.current;
    return () => {
      if (element && plotlyRef.current) plotlyRef.current.purge(element);
      listenerAttached.current = false;
    };
  }, []);

  return (
    <div className={`plot ${className ?? ""}`} style={{ height }} role="figure" aria-label={label}>
      {error !== null ? (
        <div className="plot-message error-text">Plot unavailable: {error}</div>
      ) : !ready ? (
        <div className="plot-message">Loading plot…</div>
      ) : null}
      <div ref={containerRef} className="plot-canvas" hidden={error !== null} />
    </div>
  );
}
