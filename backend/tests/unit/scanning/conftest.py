"""Fakes for the scan-engine tests.

* :class:`FakeController` implements the complete ``MicroscopeController`` ABC
  over a synthetic surface with a Gaussian axial response I(Z), with fault
  injection (hardware error / e-stop at the Nth move, slow moves, per-move hooks)
  and a controllable or manual laser.
* :class:`InMemoryRepository` implements ``ScanRepository``.
* Simple, call-recording stand-ins for the injected physics callables.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from confocal.config import MotionConfig
from confocal.errors import (
    CalibrationError,
    EmergencyStopActiveError,
    LaserError,
    LimitViolationError,
    MotionAbortedError,
    MotionError,
    PointNotFoundError,
    ReconstructionError,
    ScanNotFoundError,
)
from confocal.hardware.base import EstopListener, MicroscopeController
from confocal.models import (
    ADCCalibrateRequest,
    ADCCalibrateResponse,
    AdcGain,
    AdcSamples,
    ADCStatus,
    Axis,
    AxisLimits,
    CalibrationState,
    CameraStatus,
    CoarsePeak,
    DarkCalibrationRequest,
    FitMethod,
    HardwareInfo,
    HardwareStatus,
    IntensityMeasurement,
    LaserStatus,
    MLAnalysisRequest,
    MLModelInfo,
    MLPointPrediction,
    MLResult,
    MLTask,
    PeakFit,
    PointClassification,
    PointStatus,
    Position,
    ProcessingConfig,
    ProfileAnalysis,
    ProfileData,
    ProfileProcessingResult,
    ProfileRecord,
    ReconstructionRequest,
    ReferenceCalibrationRequest,
    SamplingMethod,
    ScanConfig,
    ScanPoint,
    ScanState,
    ScanSummary,
    StageLimits,
    StageState,
    StageStatus,
    SurfacePoint,
    SurfaceResult,
    SurfaceStatistics,
    utc_now,
)
from confocal.models.scan import ACTIVE_SCAN_STATES
from confocal.processing.normalization import aggregate_samples, normalize, subtract_dark
from confocal.scanning.events import ScanEventBroker
from confocal.scanning.manager import ScanManager
from confocal.scanning.progress import LiveEventConfig

FAKE_LIMITS = StageLimits(
    x=AxisLimits(min_um=-1000.0, max_um=1000.0),
    y=AxisLimits(min_um=-1000.0, max_um=1000.0),
    z=AxisLimits(min_um=-100.0, max_um=100.0),
)

MoveHook = Callable[[], Awaitable[object]]


# --------------------------------------------------------------------------- surface
@dataclass
class FakeSurface:
    """Tilted plane (optionally with a step) seen through a Gaussian axial response."""

    base_z_um: float = 2.0
    tilt_x: float = 0.05
    tilt_y: float = -0.05
    step_x_um: float | None = None  # x at/after which the surface is raised by step_height
    step_height_um: float = 0.0
    fwhm_um: float = 6.0
    peak_v: float = 1.5
    background_v: float = 0.05
    dark_v: float = 0.01

    def height_um(self, x_um: float, y_um: float) -> float:
        z = self.base_z_um + self.tilt_x * x_um + self.tilt_y * y_um
        if self.step_x_um is not None and x_um >= self.step_x_um:
            z += self.step_height_um
        return z

    def signal_v(self, x_um: float, y_um: float, z_um: float, laser_on: bool) -> float:
        if not laser_on:
            return self.dark_v
        sigma = self.fwhm_um / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        dz = z_um - self.height_um(x_um, y_um)
        return self.dark_v + self.background_v + self.peak_v * math.exp(-0.5 * (dz / sigma) ** 2)


# --------------------------------------------------------------------------- controller
class FakeController(MicroscopeController):
    """In-memory instrument. Moves are instantaneous unless ``move_delay_s`` is set."""

    def __init__(
        self,
        *,
        limits: StageLimits = FAKE_LIMITS,
        surface: FakeSurface | None = None,
        laser_controllable: bool = False,
        laser_on: bool = True,
        calibration: CalibrationState | None = None,
        gain: AdcGain = AdcGain.G2,
        data_rate_sps: int = 860,
        noise_v: float = 0.0005,
        seed: int = 7,
        move_delay_s: float = 0.0,
        fail_at_move: int | None = None,
        estop_at_move: int | None = None,
        z_report_offset_um: float = 0.0,
    ) -> None:
        self._limits = limits
        self.surface = surface or FakeSurface()
        self.laser_controllable = laser_controllable
        self.laser_on = laser_on
        self._calibration = calibration or CalibrationState()
        self.gain = gain
        self.data_rate_sps = data_rate_sps
        self.noise_v = noise_v
        self._rng = np.random.default_rng(seed)
        self.move_delay_s = move_delay_s
        self.fail_at_move = fail_at_move
        self.estop_at_move = estop_at_move
        self.z_report_offset_um = z_report_offset_um
        self.position = Position(x_um=0.0, y_um=0.0, z_um=0.0)
        self.moves = 0
        self.move_log: list[Position] = []
        self.move_hooks: dict[int, MoveHook] = {}
        self.emergency_stops: list[str] = []
        self.laser_switches: list[bool] = []
        self.dark_calibrations = 0
        self.homes = 0
        self.settle_waits: list[float] = []
        self._listeners: list[EstopListener] = []
        self._estop_reason: str | None = None

    # ----------------------------------------------------------------- lifecycle / identity
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @property
    def limits(self) -> StageLimits:
        return self._limits

    @property
    def estop_engaged(self) -> bool:
        return self._estop_reason is not None

    @property
    def calibration(self) -> CalibrationState:
        return self._calibration

    def hardware_info(self) -> HardwareInfo:
        return HardwareInfo(
            controller="fake",
            stage_backend="fake",
            adc_backend="fake",
            laser_backend="fake" if self.laser_controllable else "manual",
            camera_backend="none",
        )

    async def status(self) -> HardwareStatus:
        return HardwareStatus(
            stage=StageStatus(
                backend="fake",
                connected=True,
                state=StageState.STOPPED if self.estop_engaged else StageState.IDLE,
                position=self.position,
                estop_engaged=self.estop_engaged,
                limits=self._limits,
            ),
            adc=ADCStatus(
                backend="fake",
                connected=True,
                gain=self.gain,
                full_scale_v=self.gain.full_scale_v,
                data_rate_sps=self.data_rate_sps,
            ),
            laser=LaserStatus(
                backend="fake",
                connected=True,
                controllable=self.laser_controllable,
                enabled=self.laser_on if self.laser_controllable else None,
                wavelength_nm=650.0,
            ),
            camera=CameraStatus(backend="none", connected=False, available=False),
            estop_engaged=self.estop_engaged,
            estop_reason=self._estop_reason,
        )

    # ----------------------------------------------------------------- motion
    async def get_position(self) -> Position:
        return self.position

    async def move_to(
        self,
        *,
        x_um: float | None = None,
        y_um: float | None = None,
        z_um: float | None = None,
    ) -> Position:
        if self.estop_engaged:
            raise EmergencyStopActiveError("emergency stop latched")
        target = self.position.with_updates(x_um=x_um, y_um=y_um, z_um=z_um)
        violations = self._limits.violations(target)
        if violations:
            raise LimitViolationError("target outside limits", violations=violations)
        self.moves += 1
        number = self.moves
        hook = self.move_hooks.pop(number, None)
        if hook is not None:
            await hook()
        if self.fail_at_move == number:
            raise MotionError(f"simulated failure at move {number}")
        if self.estop_at_move == number:
            await self.emergency_stop("test e-stop")
            raise MotionAbortedError("stopped by emergency stop")
        await asyncio.sleep(self.move_delay_s)
        self.position = target
        self.move_log.append(target)
        return target.offset(dz_um=self.z_report_offset_um)

    async def move_relative(
        self, *, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0
    ) -> Position:
        target = self.position.offset(dx_um, dy_um, dz_um)
        return await self.move_to(x_um=target.x_um, y_um=target.y_um, z_um=target.z_um)

    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        if self.estop_engaged:
            raise EmergencyStopActiveError("emergency stop latched")
        self.homes += 1
        for axis in axes or list(Axis):
            self.position = self.position.with_axis(axis, 0.0)
        return self.position

    async def emergency_stop(self, reason: str) -> None:
        self._estop_reason = reason
        self.emergency_stops.append(reason)
        if self.laser_controllable:
            self.laser_on = False
        for listener in list(self._listeners):
            await listener(reason)

    async def reset_emergency_stop(self) -> Position:
        self._estop_reason = None
        return self.position

    def add_estop_listener(self, listener: EstopListener) -> None:
        self._listeners.append(listener)

    async def wait_settle(self, seconds: float) -> None:
        self.settle_waits.append(seconds)
        await asyncio.sleep(0)

    # ----------------------------------------------------------------- illumination / detection
    async def set_laser(self, enabled: bool) -> None:
        if not self.laser_controllable:
            raise LaserError("laser is switched manually")
        self.laser_on = enabled
        self.laser_switches.append(enabled)

    def true_voltage(self, position: Position | None = None) -> float:
        p = position or self.position
        return self.surface.signal_v(p.x_um, p.y_um, p.z_um, self.laser_on)

    async def acquire(self, n_samples: int) -> AdcSamples:
        await asyncio.sleep(0)
        volts = self.true_voltage() + self._rng.normal(0.0, self.noise_v, n_samples)
        lsb = self.gain.lsb_v
        counts = np.clip(np.round(volts / lsb), -32768, 32767).astype(np.int32)
        timestamps = time.time() + np.arange(n_samples, dtype=np.float64) / self.data_rate_sps
        return AdcSamples(
            counts=counts,
            volts=counts.astype(np.float64) * lsb,
            timestamps=timestamps,
            gain=self.gain,
            data_rate_sps=self.data_rate_sps,
        )

    async def measure_intensity(
        self, n_samples: int, method: SamplingMethod = SamplingMethod.MEDIAN
    ) -> IntensityMeasurement:
        samples = await self.acquire(n_samples)
        voltage = float(aggregate_samples(samples.volts, method))
        dark = self._calibration.dark_v
        return IntensityMeasurement(
            position=self.position,
            n_samples=samples.n,
            method=method,
            gain=samples.gain,
            raw_counts=[int(c) for c in samples.counts],
            voltages_v=[float(v) for v in samples.volts],
            voltage_v=voltage,
            voltage_std_v=float(np.std(samples.volts)),
            dark_v=dark,
            corrected_v=voltage - (dark or 0.0),
        )

    async def configure_adc(self, request: ADCCalibrateRequest) -> ADCCalibrateResponse:
        self.gain = request.gain or self.gain
        return ADCCalibrateResponse(
            gain=self.gain, full_scale_v=self.gain.full_scale_v, message="configured"
        )

    # ----------------------------------------------------------------- calibration
    def _next_calibration(self, **values: object) -> CalibrationState:
        version = (self._calibration.version or 0) + 1
        update: dict[str, object] = {"version": version, "created_at": utc_now(), **values}
        self._calibration = self._calibration.model_copy(update=update)
        return self._calibration

    async def calibrate_dark(self, request: DarkCalibrationRequest) -> CalibrationState:
        if not self.laser_controllable and not request.beam_blocked_confirmed:
            raise CalibrationError("manual laser: confirm that the beam is blocked")
        self.dark_calibrations += 1
        return self._next_calibration(
            updated_field="dark", dark_v=self.surface.dark_v, dark_n_samples=request.n_samples
        )

    async def calibrate_reference(self, request: ReferenceCalibrationRequest) -> CalibrationState:
        return self._next_calibration(
            updated_field="reference",
            reference_v=self.true_voltage(),
            reference_n_samples=request.n_samples,
        )


# --------------------------------------------------------------------------- repository
@dataclass
class StoredPoint:
    point: ScanPoint
    profile: ProfileData | None
    analysis: ProfileAnalysis | None


@dataclass
class InMemoryRepository:
    """``ScanRepository`` in memory, with a log of every persisted state."""

    scans: dict[str, ScanSummary] = field(default_factory=dict)
    points: dict[str, list[StoredPoint]] = field(default_factory=dict)
    surfaces: list[SurfaceResult] = field(default_factory=list)
    ml_results: list[MLResult] = field(default_factory=list)
    events: list[tuple[str, str, str | None]] = field(default_factory=list)
    state_log: dict[str, list[ScanState]] = field(default_factory=dict)

    def _get(self, scan_id: str) -> ScanSummary:
        try:
            return self.scans[scan_id]
        except KeyError:
            raise ScanNotFoundError(scan_id) from None

    def create_scan(
        self,
        *,
        scan_id: str,
        config: ScanConfig,
        total_points: int,
        calibration: CalibrationState,
        software_version: str,
        hardware: HardwareInfo,
    ) -> ScanSummary:
        summary = ScanSummary(
            id=scan_id,
            name=config.name,
            mode=config.mode,
            state=ScanState.IDLE,
            created_at=utc_now(),
            total_points=total_points,
            config=config,
            calibration_version=calibration.version,
            calibration=calibration,
            software_version=software_version,
            hardware=hardware,
        )
        self.scans[scan_id] = summary
        self.points[scan_id] = []
        self.state_log[scan_id] = [ScanState.IDLE]
        return summary

    def update_scan(
        self,
        scan_id: str,
        *,
        state: ScanState | None = None,
        completed_points: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_message: str | None = None,
        interrupted: bool | None = None,
        calibration: CalibrationState | None = None,
    ) -> ScanSummary:
        summary = self._get(scan_id)
        update: dict[str, object] = {}
        if state is not None:
            update["state"] = state
            self.state_log[scan_id].append(state)
        if completed_points is not None:
            update["completed_points"] = completed_points
            update["progress"] = min(1.0, completed_points / summary.total_points)
        for name, value in (
            ("started_at", started_at),
            ("finished_at", finished_at),
            ("error_message", error_message),
            ("interrupted", interrupted),
        ):
            if value is not None:
                update[name] = value
        if calibration is not None:
            update["calibration"] = calibration
            update["calibration_version"] = calibration.version
        self.scans[scan_id] = summary.model_copy(update=update)
        return self.scans[scan_id]

    def get_scan(self, scan_id: str) -> ScanSummary:
        return self._get(scan_id)

    def list_scans(self, *, limit: int = 100, offset: int = 0) -> list[ScanSummary]:
        ordered = sorted(self.scans.values(), key=lambda s: s.created_at, reverse=True)
        return ordered[offset : offset + limit]

    def append_point(
        self,
        scan_id: str,
        point: ScanPoint,
        profile: ProfileData | None,
        analysis: ProfileAnalysis | None,
    ) -> None:
        self._get(scan_id)
        self.points[scan_id].append(StoredPoint(point, profile, analysis))

    def get_points(
        self, scan_id: str, *, since_point_id: int | None = None, limit: int | None = None
    ) -> list[ScanPoint]:
        self._get(scan_id)
        points = sorted((s.point for s in self.points[scan_id]), key=lambda p: p.point_id)
        if since_point_id is not None:
            points = [p for p in points if p.point_id > since_point_id]
        return points if limit is None else points[:limit]

    def get_profile(self, scan_id: str, point_id: int) -> ProfileRecord:
        self._get(scan_id)
        for stored in self.points[scan_id]:
            if stored.point.point_id == point_id and stored.profile is not None:
                profile = stored.profile
                return ProfileRecord(
                    scan_id=scan_id,
                    point_id=point_id,
                    x_um=stored.point.x_um,
                    y_um=stored.point.y_um,
                    z_um=profile.z_um.tolist(),
                    z_reported_um=profile.z_reported_um.tolist(),
                    phase=[int(p) for p in profile.phase],
                    raw_counts=profile.raw_counts.tolist(),
                    voltage_v=profile.voltage_v.tolist(),
                    voltage_agg_v=profile.voltage_agg_v.tolist(),
                    timestamps=profile.timestamps.tolist(),
                    gain=profile.gain,
                    sampling_method=profile.sampling_method,
                    analysis=stored.analysis,
                )
        raise PointNotFoundError(f"{scan_id}/{point_id}")

    def save_surface(self, surface: SurfaceResult) -> SurfaceResult:
        saved = surface.model_copy(update={"surface_id": len(self.surfaces) + 1})
        self.surfaces.append(saved)
        self.scans[surface.scan_id] = self._get(surface.scan_id).model_copy(
            update={"has_surface": True}
        )
        return saved

    def get_latest_surface(self, scan_id: str) -> SurfaceResult | None:
        matching = [s for s in self.surfaces if s.scan_id == scan_id]
        return matching[-1] if matching else None

    def save_ml_result(self, result: MLResult) -> MLResult:
        saved = result.model_copy(update={"result_id": len(self.ml_results) + 1})
        self.ml_results.append(saved)
        self.scans[result.scan_id] = self._get(result.scan_id).model_copy(
            update={"has_ml_result": True}
        )
        return saved

    def get_latest_ml_result(self, scan_id: str) -> MLResult | None:
        matching = [r for r in self.ml_results if r.scan_id == scan_id]
        return matching[-1] if matching else None

    def record_event(self, kind: str, message: str, *, scan_id: str | None = None) -> None:
        self.events.append((kind, message, scan_id))

    def recover_interrupted_scans(self) -> list[str]:
        recovered = [s.id for s in self.scans.values() if s.state in ACTIVE_SCAN_STATES]
        for scan_id in recovered:
            self.update_scan(scan_id, state=ScanState.ERROR, interrupted=True)
        return recovered

    def close(self) -> None:
        return None

    def event_kinds(self, scan_id: str) -> list[str]:
        return [kind for kind, _, sid in self.events if sid == scan_id]

    def stored(self, scan_id: str) -> list[StoredPoint]:
        return self.points[scan_id]


# --------------------------------------------------------------------------- physics fakes
class FakeCoarsePeakFinder:
    """Arg-max of the coarse sweep; 'found' when it clearly exceeds the median."""

    def __init__(self, min_height_v: float = 0.1) -> None:
        self.min_height_v = min_height_v
        self.calls: list[NDArray[np.float64]] = []
        self.configs: list[ProcessingConfig] = []

    def __call__(
        self,
        z_um: NDArray[np.float64],
        voltage_v: NDArray[np.float64],
        *,
        dark_v: float | None,
        config: ProcessingConfig,
    ) -> CoarsePeak:
        self.calls.append(np.array(z_um))
        self.configs.append(config)
        signal = subtract_dark(voltage_v, dark_v)
        index = int(np.argmax(signal))
        height = float(signal[index] - np.median(signal))
        if signal.size < 3 or height < self.min_height_v:
            return CoarsePeak(found=False, message="no peak")
        at_edge = index < 1 or index > signal.size - 2
        return CoarsePeak(found=True, z_um=float(z_um[index]), index=index, at_edge=at_edge)


class FakeProfileAnalyser:
    """Three-point parabolic vertex of the arg-max; optionally fails on call N."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[NDArray[np.float64]] = []
        self.configs: list[ProcessingConfig] = []

    def __call__(
        self,
        z_um: NDArray[np.float64],
        voltage_v: NDArray[np.float64],
        *,
        dark_v: float | None,
        reference_v: float | None,
        config: ProcessingConfig,
    ) -> ProfileProcessingResult:
        self.calls.append(np.array(z_um))
        self.configs.append(config)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated analysis bug")
        corrected = subtract_dark(voltage_v, dark_v)
        normalized = normalize(voltage_v, dark_v, reference_v)
        signal = normalized if normalized is not None else corrected
        units = "normalized" if normalized is not None else "volts"
        i = int(np.argmax(signal))
        if i == 0 or i == signal.size - 1:
            analysis = ProfileAnalysis(status=PointStatus.PEAK_AT_EDGE, signal_units=units)
        else:
            y0, y1, y2 = (float(v) for v in signal[i - 1 : i + 2])
            denom = y0 - 2.0 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            vertex = float(z_um[i]) + shift * float(z_um[i + 1] - z_um[i - 1]) / 2.0
            analysis = ProfileAnalysis(
                status=PointStatus.VALID,
                signal_units=units,
                peak_found=True,
                peak_index=i,
                peak_z_um=float(z_um[i]),
                peak_intensity=y1,
                parabolic=PeakFit(method=FitMethod.PARABOLIC, success=True, center_um=vertex),
                gaussian=PeakFit(method=FitMethod.GAUSSIAN, success=False, center_um=math.nan),
                surface_z_um=vertex,
                surface_method=FitMethod.PARABOLIC,
                snr=50.0,
                fwhm_um=6.0,
                prominence=y1,
                fit_residual=math.nan,
                confidence=0.9,
                flags=["fake"],
            )
        return ProfileProcessingResult(
            analysis=analysis, corrected_v=corrected, normalized=normalized, filtered=signal.copy()
        )


class FakeReconstructor:
    """Minimal SurfaceResult from the valid points; ReconstructionError below 3 of them."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int, float]] = []

    def __call__(
        self,
        scan_id: str,
        points: Sequence[ScanPoint],
        request: ReconstructionRequest,
        *,
        xy_step_um: float,
    ) -> SurfaceResult:
        self.calls.append((scan_id, len(points), xy_step_um))
        valid = [p for p in points if p.surface_z_um is not None]
        if self.fail or len(valid) < 3:
            raise ReconstructionError("too few valid points")
        xs = sorted({p.x_um for p in points})
        ys = sorted({p.y_um for p in points})
        lookup = {(p.x_um, p.y_um): p.surface_z_um for p in valid}
        z = [[lookup.get((x, y)) for x in xs] for y in ys]
        return SurfaceResult(
            scan_id=scan_id,
            request=request,
            x_um=xs,
            y_um=ys,
            z_um=z,
            confidence=[[0.9 if v is not None else None for v in row] for row in z],
            gap_mask=[[v is None for v in row] for row in z],
            points=[
                SurfacePoint(
                    point_id=p.point_id,
                    x_um=p.x_um,
                    y_um=p.y_um,
                    z_um=p.surface_z_um,
                    confidence=p.confidence,
                    classification=PointClassification.USED
                    if p.surface_z_um is not None
                    else PointClassification.INVALID,
                )
                for p in points
            ],
            statistics=SurfaceStatistics(
                n_input=len(points),
                n_invalid=len(points) - len(valid),
                n_low_confidence=0,
                n_outliers=0,
                n_used=len(valid),
                coverage_fraction=1.0,
                gap_fraction=0.0,
                n_gaps=0,
            ),
        )


class FakeMLAnalyser:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls = 0

    def has_model(self, name: str | None = None) -> bool:
        return self.available

    def analyse(
        self, scan_id: str, points: Sequence[ScanPoint], request: MLAnalysisRequest
    ) -> MLResult:
        self.calls += 1
        return MLResult(
            scan_id=scan_id,
            model=MLModelInfo(
                name="fake", version="1", task=MLTask.BAD_POINT, algorithm="Fake", feature_names=[]
            ),
            threshold=request.threshold,
            n_points=len(points),
            n_flagged=0,
            predictions=[
                MLPointPrediction(point_id=p.point_id, label="ok", flagged=False) for p in points
            ],
        )


# --------------------------------------------------------------------------- configs
def small_config(**overrides: object) -> ScanConfig:
    """3 x 2 confocal grid: 21 coarse + 17 fine positions on the first point."""
    values: dict[str, object] = {
        "x_start_um": 0.0,
        "x_stop_um": 20.0,
        "y_start_um": 0.0,
        "y_stop_um": 10.0,
        "xy_step_um": 10.0,
        "z_center_um": 0.0,
        "z_range_um": 40.0,
        "coarse_z_step_um": 2.0,
        "fine_z_range_um": 8.0,
        "fine_z_step_um": 0.5,
        "adaptive_z_range_um": 16.0,
        "samples_per_z": 4,
        "settle_time_ms": 0.0,
    }
    values.update(overrides)
    return ScanConfig.model_validate(values)


# --------------------------------------------------------------------------- manager bench
class MoveGate:
    """Move hook that holds the scan inside a move until :meth:`release` (a slow move)."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self._released = asyncio.Event()

    async def __call__(self) -> None:
        self.entered.set()
        await self._released.wait()

    def release(self) -> None:
        self._released.set()


@dataclass
class Bench:
    manager: ScanManager
    controller: FakeController
    repository: InMemoryRepository
    broker: ScanEventBroker
    reconstructor: FakeReconstructor
    ml: FakeMLAnalyser | None


def make_bench(
    controller: FakeController | None = None,
    *,
    repository: InMemoryRepository | None = None,
    reconstructor: FakeReconstructor | None = None,
    analyser: FakeProfileAnalyser | None = None,
    ml: FakeMLAnalyser | None = None,
    default_saturation_v: float | None = None,
) -> Bench:
    controller = controller or FakeController()
    repository = repository or InMemoryRepository()
    reconstructor = reconstructor or FakeReconstructor()
    broker = ScanEventBroker(queue_size=100_000)
    manager = ScanManager(
        controller,
        repository,
        broker=broker,
        find_coarse_peak=FakeCoarsePeakFinder(),
        analyse_profile=analyser or FakeProfileAnalyser(),
        reconstruct_surface=reconstructor,
        ml_analyser=ml,
        motion=MotionConfig(),
        software_version="0.0.0-test",
        live_events=LiveEventConfig(progress_interval_s=0.0, profile_interval_s=0.0),
        default_saturation_v=default_saturation_v,
    )
    return Bench(manager, controller, repository, broker, reconstructor, ml)


async def wait_for(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    """Poll ``condition`` while letting the scan task run."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)
