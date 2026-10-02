# Frontend (Phase 9)

React 18 + TypeScript (strict) + Vite + Plotly.js web UI of the confocal
surface scanner. It talks to the backend REST API (`/api/v1/...`) and the scan
WebSocket (`/ws/scans/{scan_id}`) on its own origin, and is designed for the
Pi's HDMI monitor with a mouse (1920x1080 and 1280x720), with a light theme and
a dark mode for a darkened lab.

## Pages

| # | Route | Page |
| --- | --- | --- |
| 1 | `#/` | **Dashboard**: system status, active scan, recent scans, calibration, hardware summary |
| 2 | `#/setup` | **Scan Setup**: every scan parameter; live debounced `POST /scans/estimate` (points, measurements, time, data size, limit violations, warnings) before Start |
| 3 | `#/live/:id` | **Live Scan**: progress, ETA, stage position, live I(Z) (PROFILE events), completion and confidence maps (POINT events + stored points), pause / resume / cancel |
| 4 | `#/surface/:id` | **Surface Viewer**: 3-D surface, height / confidence maps, point cloud by classification, X / Y cross-sections, vertical scaling, statistics, reconstruct (nearest / linear / cubic / rbf) |
| 5 | `#/calibration` | **Calibration**: current + history, dark (manual laser: "I have blocked the beam"), reference with Z search, ADC gain / auto-gain, live ADC read |
| 6 | `#/hardware` | **Hardware**: position, jog with step sizes, absolute move, home, ADC / laser status, e-stop reset |
| 7 | `#/settings` | **Settings**: read-only system info, limits, versions, ML models; UI preferences (localStorage) |
| 8 | `#/history/:id` | **Scan History**: list, scan summary, points table, per-point I(Z) (raw samples, aggregated, normalized, filtered) |

Always visible: the **EMERGENCY STOP** button (`POST /api/v1/stage/stop`, never
disabled), the e-stop latched banner with **Reset**, and - when
`/system/status` reports a laser that is not software-controllable - a note
that the laser is manual and the e-stop cannot switch it off. Every API error
is shown with its `ErrorResponse {error, detail, violations}`. Manual controls
are disabled while a scan is active (the backend refuses them with 409 anyway).

The router uses URL hashes (`/#/history/...`) because the backend serves the
build as static files without an SPA fallback; reloads and deep links always
load `/index.html`.

## Development

Requires Node.js 20.19+ or 22.12+ and, for `gen:api`, [uv](https://docs.astral.sh/uv/).

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 ; proxies /api and /ws to http://127.0.0.1:8000
```

Start the backend in another terminal (`cd backend && uv run confocal-server`,
full simulation by default). Another backend address: `CONFOCAL_BACKEND=http://host:port npm run dev`.

| Script | What |
| --- | --- |
| `npm run dev` | Vite dev server with the API / WebSocket proxy |
| `npm run build` | Type-check, then build to `frontend/dist` (relative asset paths) |
| `npm run preview` | Serve the build on :4173 (same proxy) |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | ESLint (typescript-eslint strict type-checked + react-hooks), no warnings allowed |
| `npm test` | Vitest unit tests (API client errors, WebSocket state machine and hook, downsampling, formatting, estimate display, forms, surface helpers) |
| `npm run gen:api` | Regenerate `src/api/schema.ts` from the backend's OpenAPI schema |

### API types

`src/api/schema.ts` is generated (and committed) from the backend's Pydantic
models; never edit it. After changing backend models run `npm run gen:api`: it
runs `uv run --directory ../backend python -c "...create_app().openapi()..."`
(no server needed; hardware only starts in the lifespan) and
`openapi-typescript`. `src/api/types.ts` gives the schemas short names. The
WebSocket messages are not in OpenAPI: `src/api/events.ts` mirrors
`ScanEvent` / `ScanProgress` / `LiveProfile` from `backend/confocal/models/scan.py`
by hand - keep it in sync.

### Layout of `src/`

| Path | Contents |
| --- | --- |
| `api/` | generated schema, typed fetch client (`client.ts`: `ApiError`, timeouts), one function per endpoint (`endpoints.ts`), WebSocket event types |
| `hooks/` | `useScanSocket` + its pure state machine (snapshot, reconnect with back-off, stop on terminal state / 4404), `useScanPoints` (REST + POINT events, re-sync after reconnect), `useApiData` / `useAction` / `useDebounced` |
| `lib/` | formatting, display-only downsampling, estimate display, scan grid maps, scan form model, surface helpers, preferences, shared system status |
| `components/` | layout (header, e-stop, banners, navigation), Plotly wrapper, maps, I(Z) charts, ADC panels, UI primitives |
| `pages/` | the eight pages, each lazy-loaded |

Plotly (`plotly.js-dist-min`, about 4.8 MB / 1.5 MB gzip) is loaded with a
dynamic import the first time a plot is shown, in its own chunk; the rest of
the UI is about 300 kB. Grids larger than the display limit (Settings, default
40 000 cells) are block-reduced for display only - status maps keep the most
severe status per block, gaps are never filled.

## How the Pi serves it

The backend mounts `frontend/dist` at `/` when that directory exists
(`backend/confocal/api/app.py`, `DEFAULT_FRONTEND_DIR = <repo>/frontend/dist`),
after all API routes, so UI, API and WebSocket share one origin
(`http://127.0.0.1:8000` with `config/confocal.pi.toml`) and no CORS or proxy
is needed. Build once (on the Pi, or on a PC and copy `frontend/dist/` into the
same place in the checkout on the Pi):

```bash
cd frontend && npm ci && npm run build
```

then restart the server and open `http://127.0.0.1:8000/` in the kiosk
browser. Without a build, `/` returns a JSON pointer to `/docs`.
