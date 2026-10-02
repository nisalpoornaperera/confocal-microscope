/**
 * Data-loading helpers on top of the API client.
 *
 * - `useApiData(load, deps, {intervalMs})`: load on mount / when deps change,
 *   optionally poll; aborts the request in flight when deps change or the
 *   component unmounts; keeps the last good data while reloading.
 * - `useAction()`: run a user-triggered request (button), tracking busy state
 *   and the last error for display.
 */
import { useCallback, useEffect, useRef, useState, type DependencyList } from "react";

import { ApiError, isAbortError, toApiError } from "../api/client";

export interface ApiData<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean;
  /** Reload now (also restarts the polling interval). */
  reload: () => void;
}

export interface ApiDataOptions {
  /** Poll every `intervalMs` (after the previous request finished). */
  intervalMs?: number;
  /** Do not load while false. */
  enabled?: boolean;
}

export function useApiData<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: DependencyList,
  options: ApiDataOptions = {},
): ApiData<T> {
  const { intervalMs, enabled = true } = options;
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [generation, setGeneration] = useState(0);
  const loadRef = useRef(load);

  useEffect(() => {
    loadRef.current = load;
  });

  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;

    const run = async (): Promise<void> => {
      setLoading(true);
      try {
        const result = await loadRef.current(controller.signal);
        if (stopped) return;
        setData(result);
        setError(null);
      } catch (err) {
        if (stopped || isAbortError(err)) return;
        setError(toApiError(err));
      } finally {
        if (!stopped) {
          setLoading(false);
          if (intervalMs !== undefined) {
            timer = setTimeout(() => {
              void run();
            }, intervalMs);
          }
        }
      }
    };
    void run();
    return () => {
      stopped = true;
      controller.abort();
      if (timer !== null) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps are supplied by the caller
  }, [...deps, generation, enabled, intervalMs]);

  const reload = useCallback(() => {
    setGeneration((g) => g + 1);
  }, []);

  return { data, error, loading, reload };
}

export interface Action {
  busy: boolean;
  error: ApiError | null;
  clearError: () => void;
  /** Run `task`; resolves to its result, or `undefined` when it failed (error is stored). */
  run: <T>(task: () => Promise<T>) => Promise<T | undefined>;
}

export function useAction(): Action {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async <T,>(task: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true);
    setError(null);
    try {
      return await task();
    } catch (err) {
      if (mounted.current && !isAbortError(err)) setError(toApiError(err));
      return undefined;
    } finally {
      if (mounted.current) setBusy(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { busy, error, clearError, run };
}

/** `value`, updated only after it stopped changing for `delayMs`. */
export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebounced(value);
    }, delayMs);
    return () => {
      clearTimeout(timer);
    };
  }, [value, delayMs]);
  return debounced;
}
