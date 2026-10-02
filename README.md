# Confocal surface scanner

A laser-scanning confocal **surface profilometer** built on an OpenFlexure
Delta Stage. At every XY grid point the stage sweeps the sample through the
focus of a 650 nm laser; a pinhole-filtered OPT101 photodiode records the
reflected intensity I(Z), and the peak of that axial response is the local
surface height. The software turns those sweeps into a calibrated height map,
a 3-D surface with confidence and gap information, and (optionally) advisory
ML quality flags. Every raw ADC code is kept.

This repository holds the Python backend (hardware abstraction, safety model,
scanning, signal processing, surface reconstruction, storage, REST/WebSocket
API), the Arduino Uno firmware, the web UI, and deployment files for the Pi.

## Target machine

| Part | What |
| --- | --- |
| Computer | Raspberry Pi 5 (Raspberry Pi OS 64-bit), standalone with an HDMI monitor: backend + UI in a local browser |
| Stage | OpenFlexure **Delta Stage** (legs / motors `a`, `b`, `c`) |
| Motion | Arduino **Uno** + 3 x ULN2003 + 3 x 28BYJ-48 over USB serial ([protocol](docs/serial-protocol.md)) |
| Detector | OPT101 photodiode amplifier on 3.3 V -> ADS1115 16-bit ADC on I2C |
| Light | 650 nm, 5 mW laser module, **switched manually** (`laser = "manual"`) |

Wiring: [docs/wiring.md](docs/wiring.md). Machine configuration:
[config/confocal.pi.toml](config/confocal.pi.toml) (all keys with defaults:
[config/confocal.example.toml](config/confocal.example.toml)). Deployment:
[deploy/README.md](deploy/README.md).

## Architecture

```
api (FastAPI) ──► services (container) ──► scanning (ScanManager) ──► processing / surface / ml (injected)
                                              │
                                              ▼
                               microscope (StandardMicroscopeController: limits, verification, e-stop, calibration)
                                              │
                                              ▼
                               hardware (Stage / ADC / Laser / Camera: simulation, Arduino, ADS1115, OpenFlexure)
storage (SQLite metadata + one HDF5 file per scan) ◄── via protocols
```

* All application coordinates are micrometres; motor steps exist only in the
  hardware layer (Delta Stage kinematics, `hardware/arduino/steps.py`).
* Hardware I/O is `async`; blocking SQLite / HDF5 / CPU work runs in worker
  threads so HTTP stays responsive during a scan.
* One scan at a time, run as a background `asyncio` task with the state machine
  `PREPARING -> [CALIBRATING] -> [HOMING] -> SCANNING <-> PAUSED -> PROCESSING ->
  [SURFACE_RECONSTRUCTION] -> [ML_PROCESSING] -> COMPLETE` (or `CANCELLED` / `ERROR`).
* A complete simulation (stage, detector physics, synthetic sample with ground
  truth) runs the whole stack without hardware; the automated tests use it.

The binding module contracts are in [docs/architecture.md](docs/architecture.md).

| Directory | Contents |
| --- | --- |
| `backend/` | Python package `confocal` and its tests ([developer notes](backend/README.md)) |
| `config/` | Example and Pi machine configuration (TOML) |
| `docs/` | Architecture, serial protocol, wiring |
| `firmware/` | Arduino Uno firmware (`confocal_stage/`), `build.py` (build / flash), native tests, bench check |
| `frontend/` | Web UI (Phase 9, [plan](frontend/README.md)) |
| `deploy/` | systemd unit and Raspberry Pi setup |

## Quick start (simulation, any PC)

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11+ itself).

```bash
cd backend
uv sync                   # create .venv with all dependencies
uv run pytest             # full test suite (unit + integration), under a minute
uv run confocal-server    # default config = full simulation, http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/docs> for the interactive API. A first scan:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/calibration/dark
curl -X POST http://127.0.0.1:8000/api/v1/calibration/reference -H "Content-Type: application/json" -d '{"z_search": true}'
curl -X POST http://127.0.0.1:8000/api/v1/scans -H "Content-Type: application/json" \
     -d '{"x_start_um": 0, "x_stop_um": 200, "y_start_um": 0, "y_stop_um": 200, "xy_step_um": 20}'
```

Environment variables: `CONFOCAL_CONFIG` (TOML file, e.g. `config/confocal.pi.toml`),
`CONFOCAL_DATA_DIR` (data directory, default `./data`). `confocal-server --help`
lists `--config`, `--host`, `--port`; `python -m confocal` is equivalent.

## API

All endpoints are under `/api/v1` (OpenAPI schema: `GET /openapi.json`). Error
responses have the body `{"error", "detail", "violations"?}`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/system` | Software, platform and hardware identity, travel limits |
| GET | `/system/status` | `ok` / `degraded` / `estop` / `error`, hardware status, active scan, calibration version, uptime |
| GET | `/stage/position` | Current position (µm) |
| POST | `/stage/move` | Absolute or relative (`relative: true`) move; limit-checked and verified |
| POST | `/stage/home` | Home the given axes (default all) |
| POST | `/stage/stop` | **Emergency stop**: always allowed, latches until reset |
| POST | `/stage/reset` | Clear the e-stop latch |
| GET | `/adc/status` | Gain, data rate, last reading |
| GET | `/adc/read?n_samples=1..1024&method=mean\|median` | Calibrated intensity reading |
| POST | `/adc/calibrate` | Set the ADC gain (`gain`) or auto-range it (`auto: true`) |
| GET | `/calibration` | Current calibration snapshot |
| GET | `/calibration/history` | Calibration versions, newest first |
| POST | `/calibration/dark` | Measure the dark level (manual laser: `beam_blocked_confirmed: true`) |
| POST | `/calibration/reference` | Measure the in-focus reference level (optional `z_search`) |
| POST | `/scans` | Validate, persist and start a scan (201, returns immediately) |
| POST | `/scans/estimate` | Points, measurements, duration, data size, limit check |
| GET | `/scans?limit&offset` | Scans, newest first |
| GET | `/scans/{id}` | One scan (live state while active) |
| POST | `/scans/{id}/pause` · `/resume` · `/cancel` | Scan control |
| GET | `/scans/{id}/points?since&limit` | Per-point results |
| GET | `/scans/{id}/profile/{point_id}` | Every stored value of one point: Z, raw ADC codes, voltages, analysis |
| POST | `/scans/{id}/reconstruct` | (Re)build the surface with optional `ReconstructionRequest` |
| GET | `/scans/{id}/surface` | Latest surface (height map, confidence, gaps, mesh, statistics) |
| POST | `/scans/{id}/ml/analyse` | Advisory ML analysis |
| GET | `/scans/{id}/ml/results` | Latest ML result |
| GET | `/ml/models` | Deployed ML models |
| WS | `/ws/scans/{id}` | Live events: `snapshot`, then `state` / `progress` / `point` / `profile` / `error`; closes 1000 after the end, 4404 for an unknown scan |

| HTTP | Meaning |
| --- | --- |
| 404 | Unknown scan / point / result / ML model |
| 409 | A scan is active (manual control refused), invalid scan state, e-stop latched, calibration problem |
| 422 | Request validation or travel-limit violation (`violations` lists every problem), reconstruction impossible |
| 503 | Hardware failure |
| 500 | Configuration or internal error |

## Safety

* **Travel limits** are enforced before anything moves, twice (controller and
  stage), the whole scan envelope is validated before a scan starts, and the
  Cartesian limit box is checked against each Delta-Stage motor's travel at
  startup; the firmware enforces motor limits as the last line of defence.
  The 28BYJ-48 stage has **no end-stops**: limits are relative to the power-up
  origin, so set them conservatively for your machine.
* **Every move is verified** against the position the device reports.
* **Emergency stop** (`POST /api/v1/stage/stop`) is always accepted, halts the
  stage immediately, ends the active scan in `ERROR` and refuses motion until
  `POST /api/v1/stage/reset`. A hardware failure during a scan latches it too.
* The real laser is **switched by hand**: software cannot switch it off. The
  e-stop response says so; switch the laser off yourself. Dark calibration
  requires you to block the beam and confirm (`beam_blocked_confirmed`).
* While a scan runs, manual move / home / ADC / calibration requests are
  refused (409); only stop is always allowed.
* Class 3R laser: never look into the beam or its specular reflection.
* The server **never falls back to simulation**: a configuration naming a
  driver that does not exist yet refuses to start and names the missing phase.
* Nothing is ever deleted; interrupted scans keep their data and are marked
  `interrupted`.

## Status

Done (all exercised against the simulator; ruff, mypy `--strict` and the full
test suite pass):

* Phase 1-2: models, configuration, storage (SQLite + HDF5), complete simulation
* Hardware-independent parts of Phases 3, 4, 6, 7, 8, 10: serial protocol codec,
  Delta-Stage kinematics and step mapping, firmware emulator, ADS1115 code
  conversion, microscope controller and safety model, peak detection and
  fitting, scan engine, surface reconstruction, ML inference
* Integration layer: service container, REST + WebSocket API, `confocal-server`

Also built (not yet tested on the real machine):

* **Arduino Uno firmware** ([firmware/README.md](firmware/README.md)): coordinated
  non-blocking stepping of motors a/b/c, byte-identical to the Python emulator
* **`ArduinoStage`** and **`ADS1115ADC`** drivers, and `confocal-hwcheck`, the
  first-run hardware check
* **Web UI** ([frontend/README.md](frontend/README.md)), served by the backend at `/`

Next: set up the Pi and run the first-run checks - [docs/pi-setup.md](docs/pi-setup.md).
Then: real I(Z) profiles on the machine (Phase 5), calibration of the
kinematics and travel limits, ML (Phase 10), OpenFlexure integration (Phase 11).
