/** Display formatting. Every function accepts null / undefined / non-finite and shows "—". */

export const MISSING = "—";

export type LengthUnit = "um" | "mm";

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (!finite(value)) return MISSING;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Integer with thousands separators: 12345 -> "12,345". */
export function formatInteger(value: number | null | undefined): string {
  if (!finite(value)) return MISSING;
  return Math.round(value).toLocaleString("en-US");
}

/** A length given in micrometres, shown in the preferred unit. */
export function formatLength(
  valueUm: number | null | undefined,
  unit: LengthUnit = "um",
  digits?: number,
): string {
  if (!finite(valueUm)) return MISSING;
  if (unit === "mm") return `${formatNumber(valueUm / 1000, digits ?? 4)} mm`;
  return `${formatNumber(valueUm, digits ?? 2)} µm`;
}

export function lengthUnitLabel(unit: LengthUnit): string {
  return unit === "mm" ? "mm" : "µm";
}

/** Convert micrometres to the display unit (for plot axes). */
export function toDisplayLength(valueUm: number, unit: LengthUnit): number {
  return unit === "mm" ? valueUm / 1000 : valueUm;
}

export function formatVoltage(value: number | null | undefined, digits = 4): string {
  if (!finite(value)) return MISSING;
  if (Math.abs(value) < 0.1 && value !== 0) return `${formatNumber(value * 1000, 2)} mV`;
  return `${formatNumber(value, digits)} V`;
}

/** Fraction 0..1 as a percentage: 0.4567 -> "45.7 %". */
export function formatPercent(fraction: number | null | undefined, digits = 1): string {
  if (!finite(fraction)) return MISSING;
  return `${formatNumber(fraction * 100, digits)} %`;
}

/** Duration in seconds: "850 ms", "42 s", "3 min 05 s", "2 h 07 min", "3 d 4 h". */
export function formatDuration(seconds: number | null | undefined): string {
  if (!finite(seconds) || seconds < 0) return MISSING;
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 59.5) {
    return seconds < 10 ? `${formatNumber(seconds, 1)} s` : `${Math.round(seconds)} s`;
  }
  const total = Math.round(seconds);
  const days = Math.floor(total / 86_400);
  const hours = Math.floor((total % 86_400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number): string => String(n).padStart(2, "0");
  if (days > 0) return `${days} d ${hours} h`;
  if (hours > 0) return `${hours} h ${pad(minutes)} min`;
  return `${minutes} min ${pad(secs)} s`;
}

/** Byte count with binary prefixes: 1536 -> "1.50 KiB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (!finite(bytes) || bytes < 0) return MISSING;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  if (index === 0) return `${Math.round(value)} B`;
  return `${formatNumber(value, value < 10 ? 2 : 1)} ${units[index] ?? ""}`;
}

/** ISO timestamp in local time: "2026-10-02 14:03:07". */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return MISSING;
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  );
}

/** "scanning" -> "Scanning", "surface_reconstruction" -> "Surface reconstruction". */
export function humanize(value: string | null | undefined): string {
  if (!value) return MISSING;
  const text = value.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Short form of a scan id for tables. */
export function shortId(id: string): string {
  return id.length > 10 ? id.slice(0, 8) : id;
}
