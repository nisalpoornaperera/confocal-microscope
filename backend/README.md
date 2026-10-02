# confocal (backend)

Python backend of the OpenFlexure Delta Stage confocal surface scanner. Project
overview, API table and safety notes: [../README.md](../README.md). Module
contracts: [../docs/architecture.md](../docs/architecture.md).

## Developer commands

Run everything from this directory (`backend/`) with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                  # create .venv (Python 3.11+) with dev dependencies
uv run pytest -q                         # whole suite (unit + integration)
uv run pytest tests/unit -q              # unit tests only
uv run pytest -m integration -q          # API -> ScanManager -> simulation -> HDF5 tests
uv run ruff check confocal tests         # lint
uv run ruff format confocal tests        # format (use --check in CI)
uv run mypy                              # strict type check of the whole package
uv run confocal-server                   # serve the API (default: full simulation)
uv run python -m confocal --port 8001    # same entry point, other port
```

Server configuration:

| Variable / flag | Effect |
| --- | --- |
| `CONFOCAL_CONFIG=path.toml` / `--config` | Machine configuration (see `../config/`) |
| `CONFOCAL_DATA_DIR=dir` | Data directory (SQLite `confocal.db`, `scans/*.h5`, `models/`) |
| `--host`, `--port` | Override `[server] host` / `port` |

The default simulation runs at `simulation.time_scale = 0.05` (5 % of real
motion / conversion time). The API docs are at `http://127.0.0.1:8000/docs`; a
built UI in `../frontend/dist` is served at `/` when present.

## Package layout

| Package | Role |
| --- | --- |
| `confocal.api` | FastAPI app (`create_app`), error mapping, dependencies, one router per resource, WebSocket |
| `confocal.services` | `ServiceContainer` (wires everything; start / stop), system info and status |
| `confocal.scanning` | `ScanManager`, executor, plan, state machine, event broker |
| `confocal.processing` | I(Z) peak detection, fits, metrics (pure functions) |
| `confocal.surface` | Surface reconstruction |
| `confocal.ml` | Advisory inference (`MLService`); `confocal.ml.training` is offline only |
| `confocal.microscope` | `StandardMicroscopeController`: limits, verification, e-stop, calibration |
| `confocal.hardware` | Stage / ADC / laser / camera backends, simulation, kinematics, serial protocol |
| `confocal.storage` | `SQLiteHDF5Repository` |
| `confocal.models` | Shared Pydantic models |

## Tests

* Shared fixtures: `tests/conftest.py` (`sim_settings`: instantaneous,
  deterministic simulation in a temporary data directory).
* `tests/unit/api/`: every endpoint and error mapping through `TestClient`
  (used as a context manager so the lifespan starts the services);
  `tests/unit/api/helpers.py` holds polling helpers shared with the integration tests.
* `tests/integration/` (`-m integration`): calibration, confocal and fixed-Z
  scans checked against the simulated ground truth, HDF5 contents, WebSocket
  stream, cancel, e-stop, hardware fault, responsiveness during a scan,
  restart recovery.

On Windows every non-zero `asyncio.sleep` lasts at least one ~15 ms timer tick,
so a simulation with `time_scale > 0` runs much slower than the factor
suggests; tests that only need a scan to *stay active* use `time_scale = 0.05`
and cancel it, everything else uses `time_scale = 0`.
