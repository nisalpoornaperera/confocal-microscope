/**
 * Minimal typed fetch client for the backend REST API (`/api/v1`).
 *
 * Every failure is turned into an {@link ApiError} carrying the backend's
 * `ErrorResponse {error, detail, violations}` body, so pages can show the
 * exact reason (e.g. each travel-limit violation). Network failures and
 * timeouts become `ApiError`s with status 0; a caller-initiated abort is
 * re-thrown unchanged (`isAbortError`) so it can be ignored silently.
 */

export const API_PREFIX = "/api/v1";

/** Default request timeout. Long operations (reconstruction, Z search) pass their own. */
export const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  /** HTTP status, or 0 when the server could not be reached / timed out. */
  readonly status: number;
  /** Machine-readable error name (`ErrorResponse.error`, e.g. `LimitViolationError`). */
  readonly error: string;
  /** Human-readable explanation (`ErrorResponse.detail`). */
  readonly detail: string;
  /** Every individual problem (limit violations, validation errors); may be empty. */
  readonly violations: string[];

  constructor(status: number, error: string, detail: string, violations: string[] = []) {
    super(`${error}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.error = error;
    this.detail = detail;
    this.violations = violations;
  }

  get isNetworkError(): boolean {
    return this.status === 0;
  }

  /** 409: a scan is active, the e-stop is latched, invalid scan state, calibration. */
  get isConflict(): boolean {
    return this.status === 409;
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "AbortError"
  );
}

/** Any thrown value as an `ApiError` (for display). */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (error instanceof Error) return new ApiError(0, error.name || "Error", error.message);
  return new ApiError(0, "Error", String(error));
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => (typeof item === "string" ? item : JSON.stringify(item)));
}

/** FastAPI's own `{"detail": [{loc, msg}]}` validation format (used if a handler is bypassed). */
function describeValidationItems(items: unknown[]): string[] {
  return items.map((item) => {
    if (typeof item === "object" && item !== null) {
      const record = item as Record<string, unknown>;
      const loc = Array.isArray(record.loc) ? record.loc.map(String).join(".") : "";
      const msg = typeof record.msg === "string" ? record.msg : JSON.stringify(item);
      return loc ? `${loc}: ${msg}` : msg;
    }
    return String(item);
  });
}

function reasonPhrase(status: number, statusText: string): string {
  if (statusText) return statusText;
  return `HTTP ${status}`;
}

/**
 * Build an `ApiError` from a non-2xx response body (already parsed as JSON when
 * possible, else the raw text).
 */
export function errorFromBody(status: number, statusText: string, body: unknown): ApiError {
  if (typeof body === "object" && body !== null) {
    const record = body as Record<string, unknown>;
    if (typeof record.error === "string" && typeof record.detail === "string") {
      return new ApiError(status, record.error, record.detail, asStringList(record.violations));
    }
    if (typeof record.detail === "string") {
      return new ApiError(status, reasonPhrase(status, statusText), record.detail);
    }
    if (Array.isArray(record.detail)) {
      const violations = describeValidationItems(record.detail);
      return new ApiError(
        status,
        "RequestValidationError",
        `request validation failed (${violations.length} error(s))`,
        violations,
      );
    }
  }
  const text = typeof body === "string" ? body.trim() : "";
  return new ApiError(
    status,
    reasonPhrase(status, statusText),
    text ? text.slice(0, 500) : `the server answered ${status} without an explanation`,
  );
}

export type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  query?: Record<string, QueryValue>;
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const url = `${API_PREFIX}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue;
    params.append(key, String(value));
  }
  const search = params.toString();
  return search ? `${url}?${search}` : url;
}

async function readBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

/** The fetch implementation (replaced in unit tests). */
let fetchImpl: typeof fetch = (...args) => globalThis.fetch(...args);

export function setFetchImplementation(impl: typeof fetch): void {
  fetchImpl = impl;
}

/**
 * Perform one API request and return the parsed JSON body.
 *
 * @throws ApiError on any HTTP error, network failure, timeout or unparseable body.
 * @throws the original `AbortError` when `options.signal` aborted the request.
 */
export async function request<T>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeout = { fired: false };
  const timer = setTimeout(() => {
    timeout.fired = true;
    controller.abort();
  }, timeoutMs);
  const onCallerAbort = (): void => {
    controller.abort();
  };
  if (options.signal) {
    if (options.signal.aborted) controller.abort();
    else options.signal.addEventListener("abort", onCallerAbort, { once: true });
  }

  const init: RequestInit = {
    method,
    headers: { Accept: "application/json" },
    signal: controller.signal,
  };
  if (options.body !== undefined) {
    init.headers = { Accept: "application/json", "Content-Type": "application/json" };
    init.body = JSON.stringify(options.body);
  }

  try {
    let response: Response;
    try {
      response = await fetchImpl(buildUrl(path, options.query), init);
    } catch (error) {
      if (timeout.fired) {
        throw new ApiError(0, "Timeout", `no answer from the server within ${timeoutMs / 1000} s`);
      }
      if (isAbortError(error)) throw error;
      const message = error instanceof Error ? error.message : String(error);
      throw new ApiError(0, "NetworkError", `cannot reach the backend (${message})`);
    }

    let body: unknown;
    try {
      body = await readBody(response);
    } catch (error) {
      if (timeout.fired) {
        throw new ApiError(0, "Timeout", `the response did not arrive within ${timeoutMs / 1000} s`);
      }
      if (isAbortError(error)) throw error;
      throw new ApiError(response.status, "InvalidResponse", "the response could not be read");
    }

    if (!response.ok) throw errorFromBody(response.status, response.statusText, body);
    if (typeof body === "string") {
      throw new ApiError(response.status, "InvalidResponse", "the server did not return JSON");
    }
    return body as T;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", onCallerAbort);
  }
}

export function get<T>(path: string, options?: Omit<RequestOptions, "body">): Promise<T> {
  return request<T>("GET", path, options);
}

export function post<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
  return request<T>("POST", path, { ...options, body });
}

/** URL of the scan WebSocket on the page's own origin (the Vite dev server proxies /ws). */
export function scanSocketUrl(scanId: string, location: Location = window.location): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/scans/${encodeURIComponent(scanId)}`;
}
