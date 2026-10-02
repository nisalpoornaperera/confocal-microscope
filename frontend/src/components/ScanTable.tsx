/** Table of scans (Dashboard "recent scans", Scan History, Surface Viewer picker). */
import { Link } from "react-router-dom";

import type { ScanSummary } from "../api/types";
import { formatDateTime, formatInteger, formatLength, formatPercent, humanize, shortId } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { Badge, ScanStateBadge } from "./ui";

export function scanArea(scan: ScanSummary): string {
  const c = scan.config;
  return `${formatInteger(c.x_stop_um - c.x_start_um)} × ${formatInteger(c.y_stop_um - c.y_start_um)}`;
}

export function ScanTable({
  scans,
  compact = false,
  linkTo = (scan) => `/history/${scan.id}`,
}: {
  scans: ScanSummary[];
  compact?: boolean;
  linkTo?: (scan: ScanSummary) => string;
}) {
  const { prefs } = usePreferences();
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Scan</th>
            <th>State</th>
            <th>Mode</th>
            <th>Created</th>
            <th className="num">Points</th>
            {!compact && <th className="num">Area (µm)</th>}
            {!compact && <th className="num">XY step</th>}
            {!compact && <th>Results</th>}
          </tr>
        </thead>
        <tbody>
          {scans.map((scan) => (
            <tr key={scan.id}>
              <td>
                <Link to={linkTo(scan)} title={scan.id}>
                  {scan.name ? `${scan.name} (${shortId(scan.id)})` : shortId(scan.id)}
                </Link>
              </td>
              <td>
                <ScanStateBadge state={scan.state} interrupted={scan.interrupted} />
              </td>
              <td>{humanize(scan.mode)}</td>
              <td>{formatDateTime(scan.created_at)}</td>
              <td className="num">
                {formatInteger(scan.completed_points)} / {formatInteger(scan.total_points)}
                {scan.state !== "complete" && scan.total_points > 0 && (
                  <span className="muted small"> ({formatPercent(scan.progress, 0)})</span>
                )}
              </td>
              {!compact && <td className="num">{scanArea(scan)}</td>}
              {!compact && <td className="num">{formatLength(scan.config.xy_step_um, prefs.lengthUnit)}</td>}
              {!compact && (
                <td>
                  <span className="row" style={{ gap: "0.3rem" }}>
                    {scan.has_surface && <Badge kind="ok">Surface</Badge>}
                    {scan.has_ml_result && <Badge kind="info">ML</Badge>}
                  </span>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
