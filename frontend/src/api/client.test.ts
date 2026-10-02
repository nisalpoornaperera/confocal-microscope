import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  buildUrl,
  errorFromBody,
  get,
  isAbortError,
  post,
  scanSocketUrl,
  setFetchImplementation,
  toApiError,
} from "./client";

function jsonResponse(status: number, body: unknown, statusText = ""): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText,
    headers: { "Content-Type": "application/json" },
  });
}

function mockFetch(handler: (url: string, init: RequestInit) => Promise<Response>) {
  const fn = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
    handler(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, init ?? {}),
  );
  setFetchImplementation(fn);
  return fn;
}

afterEach(() => {
  setFetchImplementation((...args) => globalThis.fetch(...args));
  vi.useRealTimers();
});

describe("buildUrl", () => {
  it("prefixes /api/v1 and skips null / undefined query values", () => {
    expect(buildUrl("/scans")).toBe("/api/v1/scans");
    expect(buildUrl("/scans/x/points", { since: undefined, limit: 50, a: null })).toBe(
      "/api/v1/scans/x/points?limit=50",
    );
    expect(buildUrl("/adc/read", { n_samples: 16, method: "median" })).toBe(
      "/api/v1/adc/read?n_samples=16&method=median",
    );
  });
});

describe("request", () => {
  it("returns the parsed JSON body of a 2xx response", async () => {
    mockFetch(() => Promise.resolve(jsonResponse(200, { x_um: 1, y_um: 2, z_um: 3 })));
    await expect(get("/stage/position")).resolves.toEqual({ x_um: 1, y_um: 2, z_um: 3 });
  });

  it("sends JSON bodies with a content type", async () => {
    const fetchFn = mockFetch(() => Promise.resolve(jsonResponse(201, { id: "abc" })));
    await post("/scans", { x_start_um: 0 });
    const [url, init] = fetchFn.mock.calls[0] ?? [];
    expect(url).toBe("/api/v1/scans");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ x_start_um: 0 }));
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("omits the body when none is given", async () => {
    const fetchFn = mockFetch(() => Promise.resolve(jsonResponse(200, {})));
    await post("/stage/reset");
    expect(fetchFn.mock.calls[0]?.[1]?.body).toBeUndefined();
  });

  it("maps an ErrorResponse body with violations to ApiError", async () => {
    mockFetch(() =>
      Promise.resolve(
        jsonResponse(422, {
          error: "LimitViolationError",
          detail: "target outside the travel limits",
          violations: ["x=6000.000 um outside [-5000.000, 5000.000] um"],
        }),
      ),
    );
    const error = await post("/stage/move", { x_um: 6000 }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(422);
    expect(apiError.error).toBe("LimitViolationError");
    expect(apiError.detail).toBe("target outside the travel limits");
    expect(apiError.violations).toEqual(["x=6000.000 um outside [-5000.000, 5000.000] um"]);
    expect(apiError.isNetworkError).toBe(false);
  });

  it("maps a 409 conflict without violations", async () => {
    mockFetch(() =>
      Promise.resolve(
        jsonResponse(409, { error: "ScanConflictError", detail: "scan abc is active", violations: null }),
      ),
    );
    const error = (await get("/adc/read").catch((e: unknown) => e)) as ApiError;
    expect(error.isConflict).toBe(true);
    expect(error.violations).toEqual([]);
  });

  it("understands FastAPI's default detail formats", () => {
    const text = errorFromBody(404, "Not Found", { detail: "Not Found" });
    expect(text.error).toBe("Not Found");
    expect(text.detail).toBe("Not Found");
    const list = errorFromBody(422, "", {
      detail: [{ loc: ["body", "xy_step_um"], msg: "Input should be greater than 0" }],
    });
    expect(list.error).toBe("RequestValidationError");
    expect(list.violations).toEqual(["body.xy_step_um: Input should be greater than 0"]);
  });

  it("uses the text of a non-JSON error body", async () => {
    mockFetch(() => Promise.resolve(new Response("Bad gateway from proxy", { status: 502, statusText: "Bad Gateway" })));
    const error = (await get("/system").catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(502);
    expect(error.error).toBe("Bad Gateway");
    expect(error.detail).toBe("Bad gateway from proxy");
  });

  it("explains an empty error body", async () => {
    mockFetch(() => Promise.resolve(new Response(null, { status: 500 })));
    const error = (await get("/system").catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(500);
    expect(error.detail).toContain("500");
  });

  it("rejects a 2xx response that is not JSON", async () => {
    mockFetch(() => Promise.resolve(new Response("<html></html>", { status: 200 })));
    const error = (await get("/system").catch((e: unknown) => e)) as ApiError;
    expect(error.error).toBe("InvalidResponse");
  });

  it("turns a network failure into a status-0 ApiError", async () => {
    mockFetch(() => Promise.reject(new TypeError("Failed to fetch")));
    const error = (await get("/system").catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.isNetworkError).toBe(true);
    expect(error.error).toBe("NetworkError");
    expect(error.detail).toContain("Failed to fetch");
  });

  it("times out with a Timeout ApiError", async () => {
    vi.useFakeTimers();
    mockFetch(
      (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    );
    const pending = get("/system", { timeoutMs: 1000 }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1001);
    const error = (await pending) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.error).toBe("Timeout");
    expect(error.status).toBe(0);
  });

  it("re-throws a caller abort unchanged", async () => {
    mockFetch(
      (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    );
    const controller = new AbortController();
    const pending = get("/system", { signal: controller.signal }).catch((e: unknown) => e);
    controller.abort();
    const error = await pending;
    expect(error).not.toBeInstanceOf(ApiError);
    expect(isAbortError(error)).toBe(true);
  });
});

describe("toApiError", () => {
  it("wraps arbitrary errors", () => {
    expect(toApiError(new Error("boom")).detail).toBe("boom");
    expect(toApiError("text").detail).toBe("text");
    const original = new ApiError(409, "X", "y");
    expect(toApiError(original)).toBe(original);
  });
});

describe("scanSocketUrl", () => {
  it("uses the page origin and the ws scheme", () => {
    expect(scanSocketUrl("a b", { protocol: "http:", host: "127.0.0.1:8000" } as Location)).toBe(
      "ws://127.0.0.1:8000/ws/scans/a%20b",
    );
    expect(scanSocketUrl("id", { protocol: "https:", host: "pi.local" } as Location)).toBe(
      "wss://pi.local/ws/scans/id",
    );
  });
});
