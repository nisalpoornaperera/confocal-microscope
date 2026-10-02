# Architecture and module contracts

Target machine: Raspberry Pi 5 (Raspberry Pi OS 64-bit) with an HDMI monitor,
OpenFlexure **Delta Stage** driven by an **Arduino Uno** + 3 × ULN2003 +
3 × 28BYJ-48 (motors / legs a, b, c), ADS1115 + OPT101 on the Pi's 3.3 V,
manually switched 650 nm laser (`laser = "manual"`). Wiring: `docs/wiring.md`;
machine config: `config/confocal.pi.toml`. The Pi runs everything standalone
(backend + UI in a local browser); simulation exists for automated tests.

This document is the binding contract between the backend subsystems. The
shared *types* are code (`backend/confocal/models`, `confocal/hardware/base.py`,
`confocal/storage/interfaces.py`, `confocal/scanning/protocols.py`,
`confocal/processing/normalization.py`, `confocal/errors.py`,
`confocal/config.py`); this file specifies the *modules, classes and functions*
each subsystem must provide, and the behaviour the others rely on.

## 1. Layering

```
api ──► services ──► scanning ──► (injected) processing / surface / ml
                        │
                        ▼
                    microscope (MicroscopeController) ──► hardware (Stage/ADC/Laser/Camera)
                        │
storage ◄───────────────┘ (via Protocols in storage/interfaces.py)
models  ◄── everybody
```

Rules

* A layer imports only from layers below it. `scanning` never imports
  `processing`, `surface`, `ml` or `storage` implementations; it receives them
  by dependency injection (protocols in `scanning/protocols.py` and
  `storage/interfaces.py`). Only `services/` wires concrete classes together.
* `scanning`, `processing`, `surface` and `ml` never touch hardware or GPIO.
  Scan logic only talks to the `MicroscopeController` ABC.
* All application-level coordinates are **micrometres** (`Position.x_um` …).
  Motor steps exist only inside `confocal.hardware.arduino` (and the
  OpenFlexure adapter, which is also hardware layer).
* Hardware I/O is `async`. Blocking I/O (pyserial, smbus, h5py, SQLite) runs in
  `asyncio.to_thread` so HTTP is never blocked during a scan.
* API models never contain NaN/inf (`nan_to_none`); missing values are `None`.
* Nothing is ever deleted: raw ADC codes and voltages of every Z position are
  persisted; interrupted scans are marked `interrupted=True`.
* No Celery / Redis. Background work = `asyncio` tasks owned by `ScanManager`.
* ML is advisory: it never overwrites `surface_z_um` or any physics result.

## 2. Safety model

1. **Travel limits** (`Settings.limits`, `StageLimits`) are checked twice:
   by `MicroscopeController.move_to` *before* commanding motion, and again by
   every `Stage.move_to` implementation (`Stage.check_limits`). Violations raise
   `LimitViolationError`; nothing moves. `ScanPlan` validates the whole scan
   envelope before a scan starts, and every Z sweep is clipped to the limits.
   On the Delta Stage every Cartesian move drives all three legs, so at startup
   the Cartesian limit box is checked against each motor's travel
   (`MotorConfig.min_steps/max_steps`); the firmware enforces the same motor
   limits as the last line of defence.
2. **Move verification**: after every move the controller compares the
   device-reported position with the target; a per-axis error above
   `MotionConfig.position_tolerance_um` raises `MotionVerificationError`.
   A move is never assumed to have succeeded.
3. **Emergency stop** is latching. `MicroscopeController.emergency_stop(reason)`
   calls `Stage.stop()` immediately (without waiting for the hardware lock),
   switches a controllable laser off, records the reason, notifies listeners
   (the ScanManager aborts the active scan → `ERROR`, interrupted) and refuses
   motion with `EmergencyStopActiveError` until `reset_emergency_stop()`.
4. **Hardware failure** (`HardwareError`) during a scan: stop the stage, mark
   the point being measured `ABORTED` (its partial raw data is still stored),
   transition to `ERROR`, set `interrupted=True`, record an audit event.
5. **Manual control during a scan** (move, home, ADC read, calibration) is
   refused with `ScanConflictError` (HTTP 409). Only `stop` is always allowed.

## 3. Scan state machine

```
IDLE ─► PREPARING ─► [CALIBRATING] ─► [HOMING] ─► SCANNING ◄─► PAUSED
                                                     │
                                                     ▼
                     PROCESSING ─► [SURFACE_RECONSTRUCTION] ─► [ML_PROCESSING] ─► COMPLETE

any non-terminal state ─► CANCELLED   (user cancel)
any non-terminal state ─► ERROR       (hardware failure, e-stop, unexpected exception)
```

* CALIBRATING runs when `config.calibrate_dark_before_scan`, or when no dark
  calibration exists and the laser is controllable. Reference calibration is
  never automatic (it needs a reference reflector).
* HOMING runs when `config.home_before_scan`.
* SURFACE_RECONSTRUCTION runs when `reconstruct_on_complete` and mode is
  CONFOCAL; a `ReconstructionError` there is recorded as a warning, not a scan
  failure (the measured data is still COMPLETE).
* ML_PROCESSING runs when `ml_on_complete` and a model is deployed.
* CANCELLED / ERROR / COMPLETE are terminal. A scan that ends before all points
  are measured has `interrupted=True`.
* Pause takes effect after the point being measured completes; cancel and
  e-stop abort at the next Z step.

## 4. Confocal per-point procedure (executor)

For each XY grid point in plan order (serpentine by default):

1. Move to (x, y) at the current Z.
2. Estimate expected Z (`AdaptiveZEstimator`): previous valid point's surface Z
   (then nearest valid neighbour in the previous row, else `z_center_um`). With
   an estimate the coarse sweep width is `adaptive_z_range_um`, else `z_range_um`.
3. Coarse sweep (monotonically increasing Z, step `coarse_z_step_um`), clipped
   to limits. At every Z: move (verified), `controller.wait_settle(settle_time_ms / 1000)`, acquire
   `samples_per_z` raw ADC samples; aggregate with `sampling_method`.
4. `find_coarse_peak`. If not found or at the sweep edge while an adaptive
   (narrow) range was used, repeat the coarse sweep once with the full range.
5. Fine sweep centred on the coarse peak, width `fine_z_range_um`, step
   `fine_z_step_um`, clipped to limits.
6. Collect the profile (coarse + fine samples, phase-labelled, all raw data).
7–17. `analyse_profile` on the fine sweep (dark subtraction, normalization,
   filtering, peak detection, parabolic + Gaussian fit, surface Z, SNR, FWHM,
   prominence, fit residual, confidence). Uses the stage-*reported* Z values.
18. `repository.append_point(scan_id, ScanPoint, ProfileData, ProfileAnalysis)`
   (in a thread), publish a POINT event, update the Z estimator.

Fixed-Z mode: move to (x, y, `z_center_um`), acquire, store a one-position
profile (phase FIXED); `ScanPoint.intensity` = normalized (or corrected volts),
status `MEASURED`. No reconstruction.

## 5. Module contracts

### 5.1 hardware (simulation, protocol, conversion, adapters)

| Module | Provides |
|---|---|
| `hardware/simulation/surface.py` | `SimulatedConfocalSurface.from_config(SimulationConfig)`; `height_um(x, y)`, `reflectivity(x, y)`, `signal_v(x, y, z)` (noise-free detector voltage, laser on), ground truth for tests and offline ML training. Gaussian axial response (FWHM `axial_fwhm_um`), optional sinc² model, low-reflectivity patches, spurious secondary peaks. Vectorised over NumPy arrays. |
| `hardware/simulation/stage.py` | `SimulationStage(Stage)`: `__init__(limits, *, motion: MotionConfig, time_scale, kinematics=None, fault_after_moves=None)` (`kinematics`: optional `StageKinematics`); positions are quantised through `kinematics.quantize` (the delta's non-axis-aligned step grid) when given; sync property `current_position` (no I/O, used by SimulationADC); motion takes `distance / speed * time_scale` (asyncio.sleep, interruptible by `stop()` → `MotionAbortedError` with the partial position); enforces limits; fault injection raises `MotionError`. |
| `hardware/simulation/adc.py` | `SimulationADC(ADC)`: `__init__(surface, *, position_source: Callable[[], Position], laser_source: Callable[[], bool], config: SimulationConfig, gain, data_rate_sps, seed)`. Voltage = dark + (laser ? background + signal_v : 0) + read noise + shot noise, clipped to the OPT101 rail, quantised to int16 codes with the PGA gain (clipping at full scale). Conversion time `n / data_rate * time_scale`. Fault injection → `ADCError`. |
| `hardware/simulation/laser.py`, `camera.py` | `SimulationLaser(Laser)` (controllable), `ManualLaser(Laser)` (not controllable, `enabled=None`), `SimulationCamera(Camera)`, `NullCamera(Camera)` (`available=False`). |
| `hardware/kinematics.py` | (foundation, done) `StageKinematics`: linear map Cartesian µm ↔ motor steps of motors `a, b, c`; `openflexure_delta(...)` (OpenFlexure Delta Stage equations), `cartesian(...)`, measured `matrix`; `from_config(KinematicsConfig)`, `to_steps`, `to_position`, `quantize`, `motor_ranges`, `limit_violations` / `config_limit_violations` (Cartesian limit box vs motor travel, checked at the 8 corners), `max_speed_um_s`. |
| `hardware/arduino/steps.py` | `StepMapper(kinematics: StageKinematics, motors: Mapping[Motor, MotorConfig])` (+ `from_settings(settings)`): `to_motor_steps(Position) -> MotorSteps` (kinematics, then per-motor `invert`), `to_position(MotorSteps) -> Position`, `check_travel(MotorSteps)` (→ `LimitViolationError`, host-side defence), `resolution_um()`; and `plan_moves(current, target, *, limits)` / `plan_backlash_moves(...)`: every motor finishes its move approaching in its *logical* + direction (independent of `invert`; overshoot below the target by its `backlash_steps`, then come back up), so +Z sweeps never need compensation; overshoot waypoints are checked against the Cartesian limits and shrunk or skipped (with a warning) when they would leave the box. This is the ONLY place micrometres become motor steps. |
| `hardware/arduino/protocol.py` | Line protocol codec (see `docs/serial-protocol.md`): `Command` enum (MOVE, GETPOS, HOME, STOP, STATUS, plus PING, ZERO, RELEASE), `encode_command(seq, command, args) -> bytes`, `parse_line(line: bytes)` returning a `Response` or an `Event`, CRC-8 framing, `ProtocolError` on any malformed/corrupt line, sequence-number matching. `MOVE <a> <b> <c>` carries absolute motor steps. |
| `hardware/arduino/emulator.py` | `FirmwareEmulator`: in-memory reference implementation of the Arduino Uno firmware (motor steps only): **coordinated** moves - the three motors are interpolated (Bresenham-style) so they start and finish together and the platform follows a straight line; STOP mid-move leaves the interpolated partial position; firmware travel limits from `MotorConfig`. The executable spec of the firmware in `firmware/confocal_stage/` (which is tested byte-for-byte against it). |
| `hardware/ads1115/conversion.py` | `counts_to_volts(counts, gain)`, `volts_to_counts(v, gain)` (clipped to int16), `VALID_DATA_RATES`, `conversion_time_s(data_rate)`, `select_gain(max_abs_v, target_fraction) -> AdcGain`. |
| `hardware/openflexure/` | `OpenFlexureClient` Protocol (async: `get_position_steps`, `move_steps`, `stop`, `server_version`), `OpenFlexureStage(Stage)` mapping µm ↔ server steps via `StepConverter`; tested against a fake client. No real HTTP in this phase. |
| `hardware/factory.py` | `build_hardware(settings) -> HardwareSet(stage, adc, laser, camera, surface)`; `surface` is the `SimulatedConfocalSurface` or `None`. Builds `StageKinematics.from_config(settings.kinematics)` and refuses to start (`HardwareConfigError`) when `config_limit_violations(settings.limits, settings.arduino)` is non-empty. Backends not implemented yet (GPIO laser, OpenFlexure without a client) raise `HardwareConfigError`. |

`ArduinoStage` (`hardware/arduino/stage.py`, over `transport.py`) and
`ADS1115ADC` (`hardware/ads1115/adc.py`, smbus2) were added after the
simulation was complete and tested; they are tested against the firmware
emulator and a fake I2C bus, and `tests/hil/` runs them against real hardware
when `CONFOCAL_HIL_PORT` / `CONFOCAL_HIL_I2C` are set.

### 5.2 microscope

| Module | Provides |
|---|---|
| `microscope/controller.py` | `StandardMicroscopeController(MicroscopeController)`: `__init__(stage, adc, laser, camera, *, limits, motion: MotionConfig, calibration_store: CalibrationStore, laser_warmup_s=0.0, time_scale=1.0)`. Implements every abstract method in `hardware/base.py` with the safety model of §2; an `asyncio.Lock` serialises hardware operations (e-stop bypasses it). `connect()` loads the latest calibration from the store. `time_scale` scales `wait_settle` and laser warm-up (simulation passes `SimulationConfig.time_scale`). |
| `microscope/calibration.py` | Dark / reference calibration procedures and versioned persistence through `CalibrationStore` (in `asyncio.to_thread`). Dark: laser off (or operator confirmation if not controllable → `CalibrationError` without it), measure, laser back to its previous state. Reference: optional Z search for the maximum. |
| `microscope/measurement.py` | Aggregation and calibration of single readings → `IntensityMeasurement` using `processing/normalization.py`. |

### 5.3 processing (pure functions, NumPy/SciPy)

| Module | Provides |
|---|---|
| `processing/normalization.py` | (foundation) `aggregate_samples`, `subtract_dark`, `normalize`. |
| `processing/signal.py` | `filter_signal(signal, config)`, `estimate_baseline_and_noise(signal) -> (baseline, noise_std)` (robust, e.g. percentile + MAD of the off-peak region). |
| `processing/peaks.py` | Peak detection (`scipy.signal.find_peaks`), prominence, FWHM, secondary peak ratio, asymmetry, edge detection. |
| `processing/fitting.py` | `fit_parabolic(z, y, ...) -> PeakFit` (vertex of a quadratic over the top samples), `fit_gaussian(z, y, ...) -> PeakFit` (`curve_fit`, bounded, offset + amplitude + centre + sigma; FWHM = 2.3548·sigma). Never raise on bad data: return `success=False` with a message. |
| `processing/metrics.py` | `compute_snr`, `confidence_score(...) -> float` in [0, 1] combining SNR, relative prominence, fit quality (R², residual/amplitude), parabolic–Gaussian agreement, width plausibility, edge and saturation penalties, secondary peaks. |
| `processing/profile.py` | `find_coarse_peak` and `analyse_profile` exactly matching `scanning/protocols.py`. Status: NO_PEAK / PEAK_AT_EDGE / FIT_FAILED / LOW_CONFIDENCE / VALID. Surface Z = Gaussian centre when the fit succeeds and agrees with the parabolic vertex, else parabolic, else none. Never assume a peak is valid. |

### 5.4 surface

`surface/reconstruction.py`: `reconstruct_surface(scan_id, points, request, *,
xy_step_um) -> SurfaceResult` (matches `SurfaceReconstructor`). Pipeline:
remove invalid (no finite surface Z / status not VALID or LOW_CONFIDENCE) →
remove `confidence < min_confidence` → outliers (`LOCAL_MAD` k-NN robust
z-score, `GLOBAL_MAD` residual from a robust plane) → interpolation on a
regular grid (`nearest`, `linear`, `cubic` via `scipy.interpolate.griddata`;
`rbf` via `RBFInterpolator`) → gap detection (cells whose nearest measured position is a rejected or missing grid point, and cells farther than
`max_gap_distance_um` from any used point; connected components via
`skimage.measure.label`) → Delaunay mesh → confidence map, cross-sections,
statistics (ISO 25178 style Sa, Sq, Sz, Ssk, Sku after plane removal).
Helper modules: `filtering.py`, `interpolation.py`, `gaps.py`, `mesh.py`,
`statistics.py`.

### 5.5 ml (advisory)

* `ml/features.py`: `FEATURE_NAMES: tuple[str, ...]`, `build_feature_matrix(points) -> NDArray`
  (X, Y, coarse peak Z, peak intensity, width, prominence, SNR, fit residual,
  |parabolic − Gaussian|, neighbour-Z statistics from the grid, profile-derived
  features such as secondary peak ratio and asymmetry; NaN-safe).
* `ml/registry.py`: model artifact layout `<models_dir>/<name>/model.joblib` +
  `metadata.json` (`MLModelInfo`).
* `ml/inference.py`: `MLService(models_dir, default_model=None)` implementing
  `MLAnalyser` (+ `list_models()`). Raises `ModelNotAvailableError`.
* `ml/training.py`: **offline only** (PC). RandomForest / ExtraTrees /
  GradientBoosting; can generate labelled data from the simulator's ground
  truth. CLI `python -m confocal.ml.training`. Never imported by the server.

### 5.6 storage

* `storage/tables.py` (SQLModel): `ScanRow`, `ScanPointRow`, `CalibrationRow`,
  `SurfaceRow`, `MLResultRow`, `EventRow`.
* `storage/database.py`: `Database(path)` – SQLite (WAL, foreign keys,
  `check_same_thread=False`), `create_all()`, `session()`.
* `storage/hdf5.py`: `HDF5ScanStore(scans_dir)` – one file per scan
  (`<scans_dir>/<scan_id>.h5`), see layout below, thread-safe (lock), flush
  after every point, readable while the scan is running.
* `storage/repository.py`: `SQLiteHDF5Repository(database_path, scans_dir)`
  implementing **both** `ScanRepository` and `CalibrationStore`.

HDF5 layout (per scan):

```
/                     attrs: scan_id, format_version, created_at, software_version,
                             hardware_json, config_json, calibration_json, mode
/points               structured, resizable table of every ScanPoint scalar
/profiles/index       (N, 3) int64: point_id, start, count  (into the arrays below)
/profiles/z_um, z_reported_um, phase, voltage_agg_v, timestamps, normalized, filtered
                      (total_positions,) resizable, chunked
/profiles/raw_counts  (total_positions, samples_per_z) int32
/profiles/voltage_v   (total_positions, samples_per_z) float64
/profiles/meta        per-point JSON (dark_v, reference_v, gain, method, calibration
                      version, ProfileAnalysis)
/surfaces/<id>/       x_um, y_um, z_um, confidence, gap_mask datasets + attrs JSON
/ml/<id>/             predictions + attrs JSON
```

### 5.7 scanning

| Module | Provides |
|---|---|
| `scanning/plan.py` | `GridPoint(point_id, ix, iy, x_um, y_um)`; `grid_axes(config) -> (xs, ys)`; `order_points(xs, ys, order) -> list[GridPoint]` (serpentine reverses every odd row); `z_sweep(center, width, step, limits) -> NDArray` (increasing, clipped); `ScanPlan.from_config(config, limits, motion, data_rate_sps)` with `.points`, `.validate_limits()` (`LimitViolationError`), `.estimate() -> ScanEstimate`. |
| `scanning/state.py` | `ScanStateMachine`: `ALLOWED_TRANSITIONS`, `transition(to, reason=None)` (raises `ScanStateError`), `can_transition`, `is_active`, `is_terminal`, `history`. |
| `scanning/z_estimation.py` | `AdaptiveZEstimator` (§4 step 2). |
| `scanning/events.py` | `ScanEventBroker(queue_size)`: `publish(event)` never blocks (bounded per-subscriber queues, drop-oldest for PROGRESS/PROFILE, never drop STATE/POINT if avoidable), `subscribe(scan_id)` async-iterable subscription with `close()`, `last_event(scan_id)`. |
| `scanning/executor.py` | `ScanExecutor`: the per-point procedure of §4, pause/cancel checkpoints, ETA (moving average), throttled PROGRESS/PROFILE events. |
| `scanning/manager.py` | `ScanManager(controller, repository, *, broker, find_coarse_peak, analyse_profile, reconstruct_surface, ml_analyser=None, motion, software_version, default_saturation_v=None)`: `create_scan(config) -> ScanSummary` (reserves the instrument before its first await, 409 if active, builds and validates the plan in a worker thread, persists, starts the background task; scans without `processing.saturation_v` get `default_saturation_v` = `Settings.processing.saturation_v`, and analysis is always capped at 0.98 x the ADC full scale), `estimate(config)`, `pause/resume/cancel(scan_id)`, `get_scan`, `list_scans`, `get_progress(scan_id)`, `subscribe(scan_id)`, `reconstruct(scan_id, request)` and `analyse_ml(scan_id, request)` for finished scans (CPU work in `asyncio.to_thread`), `active_scan_id`, `is_active`, `wait_until_finished(scan_id, timeout)`, `handle_emergency_stop(reason)` (registered as e-stop listener), `shutdown()` (cancel active scan, mark interrupted). |

### 5.8 services and api

* `services/container.py`: `ServiceContainer.create(settings)` builds hardware
  (factory) → repository → controller → broker → MLService → ScanManager;
  `start()` connects hardware and recovers interrupted scans; `stop()` shuts
  down the scan manager, stops the stage, switches the laser off, closes all.
* `api/app.py`: `create_app(settings: Settings | None = None) -> FastAPI` with a
  lifespan that owns the container; exception handlers map errors (below).
* Routes: exactly the endpoints in the README plus `POST /api/v1/scans/estimate`,
  `POST /api/v1/stage/reset` (clear e-stop latch), `GET /api/v1/calibration/history`,
  `GET /api/v1/ml/models`.
* WebSocket `/ws/scans/{scan_id}`: SNAPSHOT on connect, then broker events;
  closes with 1000 after a terminal state, 4404 for an unknown scan.

| Exception | HTTP |
|---|---|
| `ScanNotFoundError`, `PointNotFoundError`, `ModelNotAvailableError` | 404 |
| `ScanConflictError`, `ScanStateError`, `EmergencyStopActiveError`, `CalibrationError` | 409 |
| `LimitViolationError`, `ReconstructionError`, Pydantic validation | 422 |
| `HardwareError` (all subclasses) | 503 |
| `HardwareConfigError`, other `ConfocalError` | 500 |

Body: `ErrorResponse {error, detail, violations?}`.

## 6. Development phases

1 repository & interfaces · 2 simulation · 3 Arduino control · 4 ADS1115/OPT101 ·
5 real I(Z) · 6 confocal peak detection · 7 XY scanning · 8 surface
reconstruction · 9 UI · 10 ML · 11 OpenFlexure integration.

The first deliverable covers phases 1–2 completely and the hardware-independent
parts of 3, 4, 6, 7, 8 and 10 (all exercised against the simulator).
