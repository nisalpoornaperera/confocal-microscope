/**
 * Application frame: header (system status, theme toggle, EMERGENCY STOP),
 * safety banners (e-stop latched with Reset, manual laser note, backend
 * unreachable), left navigation and the routed page.
 */
import { NavLink, Outlet, useNavigate } from "react-router-dom";

import { api } from "../api/endpoints";
import { useAction } from "../hooks/useApi";
import { formatDuration, humanize, shortId } from "../lib/format";
import { usePreferences } from "../lib/prefs";
import { useSystem } from "../lib/system";
import { Badge, ErrorBox } from "./ui";

export const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/setup", label: "Scan Setup" },
  { to: "/live", label: "Live Scan" },
  { to: "/surface", label: "Surface Viewer" },
  { to: "/calibration", label: "Calibration" },
  { to: "/hardware", label: "Hardware" },
  { to: "/settings", label: "Settings" },
  { to: "/history", label: "Scan History" },
] as const;

function StatusSummary() {
  const { status, statusError, activeScanId } = useSystem();
  const navigate = useNavigate();
  if (statusError?.isNetworkError || (statusError && !status)) {
    return <Badge kind="danger">Backend unreachable</Badge>;
  }
  if (!status) return <Badge>Connecting…</Badge>;
  const kind =
    status.status === "ok" ? "ok" : status.status === "degraded" ? "warn" : "danger";
  const position = status.hardware.stage.position;
  return (
    <>
      <Badge kind={kind} title="Overall system status">
        <span className="dot" aria-hidden="true" /> System {status.status.toUpperCase()}
      </Badge>
      {activeScanId !== null ? (
        <button
          type="button"
          className="btn-small"
          onClick={() => {
            void navigate(`/live/${activeScanId}`);
          }}
          title="Open the live view of the active scan"
        >
          Scan {shortId(activeScanId)}: {humanize(status.active_scan_state)}
        </button>
      ) : (
        <Badge>No active scan</Badge>
      )}
      <span className="small muted nowrap" title="Stage position (µm)">
        {position
          ? `X ${position.x_um.toFixed(1)}  Y ${position.y_um.toFixed(1)}  Z ${position.z_um.toFixed(2)} µm`
          : "Position unknown"}
      </span>
      <span className="small muted nowrap">
        Calibration v{status.calibration_version ?? "—"} · up {formatDuration(status.uptime_s)}
      </span>
    </>
  );
}

function EmergencyStopButton() {
  const { refreshStatus } = useSystem();
  const action = useAction();
  return (
    <div className="stack" style={{ gap: "0.25rem", alignItems: "flex-end" }}>
      <button
        type="button"
        className="estop"
        // Never disabled: the request is always worth sending, even while another one runs.
        onClick={() => {
          void action.run(() => api.emergencyStop()).then(refreshStatus);
        }}
        aria-label="Emergency stop: halt the stage now"
        title="Halts the stage immediately and aborts the active scan. Does NOT switch a manual laser off."
      >
        EMERGENCY STOP
      </button>
      {action.error && (
        <span className="small error-text" role="alert">
          Stop request failed: {action.error.detail}
        </span>
      )}
    </div>
  );
}

function ThemeToggle() {
  const { resolvedTheme, update } = usePreferences();
  const next = resolvedTheme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => {
        update({ theme: next });
      }}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      {resolvedTheme === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}

function EstopBanner() {
  const { status, estopEngaged, laserManual, refreshStatus } = useSystem();
  const reset = useAction();
  if (!estopEngaged) return null;
  const reason = status?.hardware.estop_reason;
  return (
    <div className="stack" style={{ gap: "0.4rem" }}>
      <div className="estop-banner" role="alert">
        <div>
          <strong>EMERGENCY STOP LATCHED.</strong> Motion is refused until you reset.
          {reason ? <div>Reason: {reason}</div> : null}
          {laserManual && <div>The laser is still on unless you switched it off by hand.</div>}
        </div>
        <button
          type="button"
          className="btn-large"
          disabled={reset.busy}
          onClick={() => {
            void reset.run(() => api.stageReset()).then(refreshStatus);
          }}
          title="Clear the latch after checking the machine"
        >
          {reset.busy ? "Resetting…" : "Reset e-stop"}
        </button>
      </div>
      <ErrorBox error={reset.error} title="Reset refused" onDismiss={reset.clearError} />
    </div>
  );
}

/** Persistent while the laser cannot be switched by software (laser = "manual"). */
export function LaserNote() {
  const { laserManual, status } = useSystem();
  if (!laserManual) return null;
  const laser = status?.hardware.laser;
  return (
    <div className="laser-note" role="note">
      <span className="laser-icon" aria-hidden="true">
        !
      </span>
      <div>
        <strong>The laser is switched by hand.</strong> The software cannot switch it on or off
        {laser ? ` (${laser.wavelength_nm} nm${laser.power_mw != null ? `, ${laser.power_mw} mW` : ""})` : ""}:
        EMERGENCY STOP halts the stage only. Use the laser&apos;s own switch, and never look into the beam
        or its reflection.
      </div>
    </div>
  );
}

function ConnectionBanner() {
  const { statusError, status } = useSystem();
  if (!statusError) return null;
  return (
    <ErrorBox
      error={statusError}
      title={
        status
          ? "Lost contact with the backend: the values shown may be out of date"
          : "Cannot reach the backend"
      }
    />
  );
}

export function Layout() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          Confocal scanner
          <small>Surface profilometer</small>
        </div>
        <div className="header-status" aria-live="polite">
          <StatusSummary />
        </div>
        <div className="header-actions">
          <ThemeToggle />
          <EmergencyStopButton />
        </div>
      </header>
      <nav className="app-nav" aria-label="Main">
        <ul>
          {NAV_ITEMS.map((item, index) => (
            <li key={item.to}>
              <NavLink to={item.to} end={"end" in item ? item.end : false}>
                <span className="nav-index" aria-hidden="true">
                  {index + 1}
                </span>
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main className="app-main">
        <div className="banners">
          <EstopBanner />
          <ConnectionBanner />
          <LaserNote />
        </div>
        <Outlet />
      </main>
    </div>
  );
}
