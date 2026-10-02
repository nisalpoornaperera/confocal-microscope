/**
 * UI preferences, stored per browser in localStorage (never on the server).
 * Every storage access is guarded: a private window or blocked storage just
 * falls back to the defaults.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import type { LengthUnit } from "./format";

export type ThemeChoice = "light" | "dark" | "system";

/** Plotly.js built-in colour scales offered for heights / intensities. */
export const COLOR_MAPS = ["Viridis", "Cividis", "Earth", "Portland", "YlGnBu", "Hot", "Greys", "RdBu"] as const;
export type ColorMap = (typeof COLOR_MAPS)[number];

export interface Preferences {
  theme: ThemeChoice;
  lengthUnit: LengthUnit;
  colorMap: ColorMap;
  /** Max grid cells handed to Plotly for surfaces / heatmaps (display only). */
  maxDisplayCells: number;
  /** Max points of the 3-D point cloud (display only). */
  maxCloudPoints: number;
  /** Default number of samples of the live ADC read on Calibration / Hardware. */
  adcSamples: number;
}

export const DEFAULT_PREFERENCES: Preferences = {
  theme: "system",
  lengthUnit: "um",
  colorMap: "Viridis",
  maxDisplayCells: 40_000,
  maxCloudPoints: 20_000,
  adcSamples: 16,
};

export const STORAGE_KEY = "confocal.ui.preferences.v1";

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

/** Validate stored preferences field by field (unknown / invalid values -> default). */
export function parsePreferences(raw: string | null): Preferences {
  if (!raw) return DEFAULT_PREFERENCES;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return DEFAULT_PREFERENCES;
  }
  if (typeof value !== "object" || value === null) return DEFAULT_PREFERENCES;
  const r = value as Record<string, unknown>;
  const d = DEFAULT_PREFERENCES;
  return {
    theme: r.theme === "light" || r.theme === "dark" || r.theme === "system" ? r.theme : d.theme,
    lengthUnit: r.lengthUnit === "mm" || r.lengthUnit === "um" ? r.lengthUnit : d.lengthUnit,
    colorMap: COLOR_MAPS.includes(r.colorMap as ColorMap) ? (r.colorMap as ColorMap) : d.colorMap,
    maxDisplayCells: clampInt(r.maxDisplayCells, 2500, 250_000, d.maxDisplayCells),
    maxCloudPoints: clampInt(r.maxCloudPoints, 1000, 250_000, d.maxCloudPoints),
    adcSamples: clampInt(r.adcSamples, 1, 1024, d.adcSamples),
  };
}

export function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage unavailable: the preference lasts for this page only.
  }
}

interface PreferencesContextValue {
  prefs: Preferences;
  update: (changes: Partial<Preferences>) => void;
  reset: () => void;
  /** The theme actually shown ("system" resolved). */
  resolvedTheme: "light" | "dark";
}

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

function systemPrefersDark(): boolean {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [prefs, setPrefs] = useState<Preferences>(() => parsePreferences(readStorage(STORAGE_KEY)));
  const [systemDark, setSystemDark] = useState<boolean>(systemPrefersDark);

  useEffect(() => {
    let query: MediaQueryList;
    try {
      query = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return undefined;
    }
    const listener = (event: MediaQueryListEvent): void => {
      setSystemDark(event.matches);
    };
    query.addEventListener("change", listener);
    return () => {
      query.removeEventListener("change", listener);
    };
  }, []);

  const resolvedTheme: "light" | "dark" =
    prefs.theme === "system" ? (systemDark ? "dark" : "light") : prefs.theme;

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.style.colorScheme = resolvedTheme;
  }, [resolvedTheme]);

  const update = useCallback((changes: Partial<Preferences>) => {
    setPrefs((current) => {
      const next = parsePreferences(JSON.stringify({ ...current, ...changes }));
      writeStorage(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    writeStorage(STORAGE_KEY, JSON.stringify(DEFAULT_PREFERENCES));
    setPrefs(DEFAULT_PREFERENCES);
  }, []);

  const value = useMemo(
    () => ({ prefs, update, reset, resolvedTheme }),
    [prefs, update, reset, resolvedTheme],
  );
  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences(): PreferencesContextValue {
  const value = useContext(PreferencesContext);
  if (!value) throw new Error("usePreferences must be used inside <PreferencesProvider>");
  return value;
}
