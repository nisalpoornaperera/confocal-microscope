/**
 * Minimal typings for the parts of `plotly.js-dist-min` this UI uses.
 * Traces and layouts are deliberately loose records (Plotly's schema is huge);
 * the typed surface is the module API used by `components/Plot.tsx`.
 */
declare module "plotly.js-dist-min" {
  export type PlotData = Record<string, unknown>;
  export type PlotLayout = Record<string, unknown>;
  export type PlotConfig = Record<string, unknown>;

  export interface PlotlyHTMLElement extends HTMLDivElement {
    on(event: string, handler: (event: unknown) => void): void;
    removeAllListeners?(event: string): void;
  }

  export interface PlotlyStatic {
    react(
      root: HTMLElement,
      data: PlotData[],
      layout?: PlotLayout,
      config?: PlotConfig,
    ): Promise<PlotlyHTMLElement>;
    purge(root: HTMLElement): void;
    Plots: { resize(root: HTMLElement): void };
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
