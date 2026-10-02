/**
 * System-wide state shared by every page: `GET /api/v1/system/status`
 * (polled), `GET /api/v1/system` (identity, limits; loaded once and on
 * reconnect) and the emergency-stop actions.
 */
import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";

import type { ApiError } from "../api/client";
import { api } from "../api/endpoints";
import { isActiveState, type SystemInfo, type SystemStatus } from "../api/types";
import { useApiData } from "../hooks/useApi";

/** Status poll period. The header, e-stop banner and "scan active" locks follow it. */
export const STATUS_POLL_MS = 1500;
/** System identity poll period. */
export const INFO_POLL_MS = 10_000;

export interface SystemContextValue {
  status: SystemStatus | undefined;
  /** Error of the latest status poll (backend unreachable, ...). */
  statusError: ApiError | null;
  info: SystemInfo | undefined;
  infoError: ApiError | null;
  /** True while a scan owns the instrument (manual control is refused by the backend). */
  scanActive: boolean;
  activeScanId: string | null;
  estopEngaged: boolean;
  /** True when the laser cannot be switched by software (manual laser). */
  laserManual: boolean;
  refreshStatus: () => void;
  refreshInfo: () => void;
}

const SystemContext = createContext<SystemContextValue | null>(null);

export function SystemProvider({ children }: { children: ReactNode }) {
  const statusData = useApiData((signal) => api.systemStatus({ signal, timeoutMs: 8000 }), [], {
    intervalMs: STATUS_POLL_MS,
  });
  // Identity and limits rarely change; re-reading them now and then also picks up
  // a backend that was restarted (possibly with another configuration).
  const infoData = useApiData((signal) => api.systemInfo({ signal }), [], {
    intervalMs: INFO_POLL_MS,
  });
  const status = statusData.data;
  const { reload: reloadInfo } = infoData;

  const { reload: refreshStatus } = statusData;
  const refreshInfo = useCallback(() => {
    reloadInfo();
  }, [reloadInfo]);

  const value = useMemo<SystemContextValue>(() => {
    const activeScanId =
      status?.active_scan_id != null && isActiveState(status.active_scan_state ?? null)
        ? status.active_scan_id
        : null;
    return {
      status,
      statusError: statusData.error,
      info: infoData.data,
      infoError: infoData.error,
      scanActive: activeScanId !== null,
      activeScanId,
      estopEngaged: status?.estop_engaged ?? false,
      laserManual: status !== undefined && !status.hardware.laser.controllable,
      refreshStatus,
      refreshInfo,
    };
  }, [status, statusData.error, infoData.data, infoData.error, refreshStatus, refreshInfo]);

  return <SystemContext.Provider value={value}>{children}</SystemContext.Provider>;
}

export function useSystem(): SystemContextValue {
  const value = useContext(SystemContext);
  if (!value) throw new Error("useSystem must be used inside <SystemProvider>");
  return value;
}
