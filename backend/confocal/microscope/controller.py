"""The standard :class:`MicroscopeController`: safety model, calibration and measurement.

Safety model (docs/architecture.md §2), in the order it is applied to a move:

1. A latched emergency stop refuses motion (:class:`EmergencyStopActiveError`).
2. The full target is built, for the axes the caller did not specify, from the
   last *commanded* target whose move was verified. The device read-back is
   step-quantised (a target on a limit face may read back a few nanometres
   outside it), so it is used for verification and reporting only. When no
   verified target is known (connect, failed move, e-stop) the position is
   re-read from the device and a read-back outside a limit by no more than
   ``position_tolerance_um`` is pulled onto that limit; anything further out
   is refused by the limit check. A commanded target is therefore never
   outside the limits.
3. The target is checked against the travel limits *before* anything is sent
   to the stage (:class:`LimitViolationError`; the stage is never called).
   The stage checks again (defence in depth) and the firmware enforces motor
   travel as the last line of defence.
4. One ``asyncio.Lock`` serialises hardware operations, so a scan and a
   manual request can never interleave commands.
5. The move is bounded by ``MotionConfig.move_timeout_for`` (distance / speed
   times a safety factor, plus the fixed ``move_timeout_s`` margin); on
   timeout the stage is stopped (:class:`MotionTimeoutError`).
6. The device-reported position is compared per axis with the target; an error
   above ``position_tolerance_um`` stops the stage
   (:class:`MotionVerificationError`). A move is never assumed to succeed.
7. Any :class:`HardwareError` during motion stops the stage (best effort) and
   is re-raised; the position is then treated as unknown.

The emergency stop does not take the lock: it latches first, then calls
``Stage.stop()`` directly so it pre-empts an in-flight move, switches a
controllable laser off and notifies the listeners. A manually switched laser
(the real machine) cannot be switched off by software; this is reported in
``estop_reason`` and the laser status so the UI can tell the operator.

An e-stop can never be lost to a move that is just starting: the latch is
checked again synchronously immediately before the stage command, and the
command is awaited directly (no task, no ``await`` in between), so the stage
has registered the move as in flight before any e-stop can run - and stages
honour a ``stop()`` from that point on. Each e-stop also bumps a stop
generation; a move that returns although an e-stop happened meanwhile (a
stage that ignored the stop) is stopped again and fails with
:class:`MotionAbortedError`.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from confocal.config import MotionConfig
from confocal.errors import (
    CameraError,
    EmergencyStopActiveError,
    HardwareError,
    LimitViolationError,
    MotionAbortedError,
    MotionTimeoutError,
    MotionVerificationError,
)
from confocal.hardware.base import (
    ADC,
    Camera,
    EstopListener,
    Laser,
    MicroscopeController,
    Stage,
)
from confocal.microscope import calibration as cal
from confocal.microscope.measurement import WIDEST_GAIN, build_intensity_measurement
from confocal.microscope.measurement import configure_adc as configure_adc_gain
from confocal.models.calibration import (
    ADCCalibrateRequest,
    ADCCalibrateResponse,
    CalibrationState,
    DarkCalibrationRequest,
    ReferenceCalibrationRequest,
)
from confocal.models.common import Axis, Position, StageLimits
from confocal.models.hardware import (
    ADCStatus,
    CameraStatus,
    HardwareInfo,
    HardwareStatus,
    LaserStatus,
    SamplingMethod,
    StageState,
    StageStatus,
)
from confocal.models.measurement import AdcSamples, IntensityMeasurement
from confocal.storage.interfaces import CalibrationStore

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Identifier stored in :class:`HardwareInfo` (scan provenance).
CONTROLLER_NAME = "standard"
#: Laser status fallback when the laser cannot report one (the machine's 650 nm diode).
NOMINAL_WAVELENGTH_NM = 650.0
#: Appended to the e-stop reason when the laser could not be switched off.
MANUAL_LASER_WARNING = (
    "the laser is switched manually and was NOT switched off: switch it off by hand"
)


def _describe(position: Position) -> str:
    return f"({position.x_um:.3f}, {position.y_um:.3f}, {position.z_um:.3f}) um"


def _safe_get(getter: Callable[[], T], default: T) -> T:
    """Read a component property for a status fallback without ever raising."""
    try:
        return getter()
    except Exception:
        return default


class StandardMicroscopeController(MicroscopeController):
    """Controller over one Stage / ADC / Laser / Camera set (simulated or real)."""

    def __init__(
        self,
        stage: Stage,
        adc: ADC,
        laser: Laser,
        camera: Camera,
        *,
        limits: StageLimits,
        motion: MotionConfig,
        calibration_store: CalibrationStore,
        laser_warmup_s: float = 0.0,
        time_scale: float = 1.0,
    ) -> None:
        if not (math.isfinite(laser_warmup_s) and laser_warmup_s >= 0.0):
            raise ValueError("laser_warmup_s must be finite and >= 0")
        if not (math.isfinite(time_scale) and time_scale >= 0.0):
            raise ValueError("time_scale must be finite and >= 0")
        self._stage = stage
        self._adc = adc
        self._laser = laser
        self._camera = camera
        self._limits = limits
        self._motion = motion
        self._store = calibration_store
        self._laser_warmup_s = float(laser_warmup_s)
        self._time_scale = float(time_scale)
        self._lock = asyncio.Lock()
        self._listeners: list[EstopListener] = []
        self._calibration = CalibrationState()
        self._position: Position | None = None  # last verified (device-reported) position
        self._target: Position | None = None  # commanded target of that verified move
        self._estop = False
        self._stop_generation = 0  # bumped by every emergency stop
        self._estop_reason: str | None = None
        self._laser_warning: str | None = None
        self._last_error: str | None = None

    # ----------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        """Connect the components, load the latest calibration, read the position."""
        async with self._lock:
            await self._stage.connect()
            await self._adc.connect()
            await self._laser.connect()
            try:
                await self._camera.connect()
            except CameraError:
                log.exception("camera unavailable; continuing without it")
            self._calibration = await cal.load_latest(self._store)
            self._position = await self._stage.get_position()

    async def close(self) -> None:
        """Leave the hardware safe: stop, laser off (if possible), motors released, closed."""
        await self._best_effort("stop the stage", self._stage.stop())
        async with self._lock:
            if self._laser.controllable and self._laser.connected:
                await self._best_effort("switch the laser off", self._laser.set_enabled(False))
            if self._stage.connected:
                await self._best_effort("release the motors", self._stage.release())
            for name, component in (
                ("stage", self._stage),
                ("ADC", self._adc),
                ("laser", self._laser),
                ("camera", self._camera),
            ):
                await self._best_effort(f"close the {name}", component.close())
            self._forget_position()

    async def release_motors(self) -> None:
        """De-energise the stepper coils (28BYJ-48 heat up when held)."""
        async with self._lock:
            await self._stage.release()

    # ----------------------------------------------------------------- identity / state
    @property
    def limits(self) -> StageLimits:
        return self._limits

    @property
    def estop_engaged(self) -> bool:
        return self._estop

    @property
    def estop_reason(self) -> str | None:
        return self._estop_reason

    @property
    def calibration(self) -> CalibrationState:
        return self._calibration

    @property
    def last_position(self) -> Position | None:
        """Last verified position (None when unknown, e.g. after a failed move)."""
        return self._position

    def hardware_info(self) -> HardwareInfo:
        return HardwareInfo(
            controller=CONTROLLER_NAME,
            stage_backend=self._stage.backend_name,
            stage_version=self._stage.version(),
            adc_backend=self._adc.backend_name,
            adc_version=self._adc.version(),
            laser_backend=self._laser.backend_name,
            camera_backend=self._camera.backend_name,
            details={
                "laser_controllable": str(self._laser.controllable).lower(),
                "adc_data_rate_sps": str(self._adc.data_rate_sps),
                "position_tolerance_um": str(self._motion.position_tolerance_um),
                "time_scale": str(self._time_scale),
            },
        )

    async def status(self) -> HardwareStatus:
        """Snapshot of every component. Never raises: failures become ``last_error``.

        It does not take the hardware lock, so the UI stays responsive during
        long moves; component ``status()`` implementations are cheap and safe to
        call concurrently.
        """
        stage = await self._stage_status()
        adc = await self._adc_status()
        laser = await self._laser_status()
        camera = await self._camera_status()
        return HardwareStatus(
            stage=stage,
            adc=adc,
            laser=laser,
            camera=camera,
            estop_engaged=self._estop,
            estop_reason=self._estop_reason,
        )

    # ----------------------------------------------------------------- motion
    async def get_position(self) -> Position:
        async with self._lock:
            self._position = await self._stage.get_position()
            target = self._target
            if (
                target is not None
                and self._position.max_axis_error(target) > self._motion.position_tolerance_um
            ):
                self._target = None  # the stage is no longer where it was commanded to be
            return self._position

    async def move_to(
        self,
        *,
        x_um: float | None = None,
        y_um: float | None = None,
        z_um: float | None = None,
    ) -> Position:
        for name, value in (("x_um", x_um), ("y_um", y_um), ("z_um", z_um)):
            if value is not None and not math.isfinite(value):
                raise LimitViolationError(
                    f"{name}={value} is not a finite coordinate", violations=[name]
                )
        self._require_estop_clear()
        async with self._lock:
            self._require_estop_clear()
            base = await self._move_base()
            return await self._move(base.with_updates(x_um=x_um, y_um=y_um, z_um=z_um))

    async def move_relative(
        self, *, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0
    ) -> Position:
        for name, value in (("dx_um", dx_um), ("dy_um", dy_um), ("dz_um", dz_um)):
            if not math.isfinite(value):
                raise LimitViolationError(f"{name}={value} is not finite", violations=[name])
        self._require_estop_clear()
        async with self._lock:
            self._require_estop_clear()
            base = await self._move_base()
            return await self._move(base.offset(dx_um, dy_um, dz_um))

    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        """Home the axes (default: all) and verify each homed axis reports the origin."""
        selected = list(Axis) if axes is None else list(dict.fromkeys(axes))
        self._require_estop_clear()
        async with self._lock:
            self._require_estop_clear()
            target = await self._move_base()
            for axis in selected:
                target = target.with_axis(axis, 0.0)
            self._check_limits(target)
            reported = await self._run_motion(
                lambda: self._stage.home(selected), f"homing {selected}", target
            )
            await self._verify(reported, target, selected)
            return reported

    async def emergency_stop(self, reason: str) -> None:
        """Latch, halt the stage without the lock, laser off where possible, notify."""
        first = not self._estop
        # Latch and bump the generation together, before any await: a move that has
        # passed its latch check sees the new generation when its stage call returns.
        self._estop = True
        self._stop_generation += 1
        if first:
            self._estop_reason = reason
        log.warning("EMERGENCY STOP: %s", reason)
        try:
            await self._stage.stop()
        except Exception:
            log.exception("emergency stop: Stage.stop() failed")
        self._forget_position()
        await self._estop_laser_off()
        if first and self._laser_warning is not None:
            self._estop_reason = f"{reason}; {self._laser_warning}"
        for listener in list(self._listeners):
            try:
                await listener(reason)
            except Exception:
                log.exception("emergency stop listener %r failed", listener)

    async def reset_emergency_stop(self) -> Position:
        """Re-read the stage position (it may have been moved by hand), then clear the latch."""
        async with self._lock:
            position = await self._stage.get_position()
            self._position = position
            self._estop = False
            self._estop_reason = None
            self._laser_warning = None
            log.info("emergency stop reset at %s", _describe(position))
            return position

    def add_estop_listener(self, listener: EstopListener) -> None:
        self._listeners.append(listener)

    # ----------------------------------------------------------------- timing
    async def wait_settle(self, seconds: float) -> None:
        """Sleep ``seconds * time_scale`` (always yields to the event loop)."""
        if not (math.isfinite(seconds) and seconds >= 0.0):
            raise ValueError(f"settle time must be finite and >= 0, got {seconds}")
        await asyncio.sleep(seconds * self._time_scale)

    # ----------------------------------------------------------------- illumination
    async def set_laser(self, enabled: bool) -> None:
        """Switch a controllable laser; switching on waits the (scaled) warm-up.

        Raises:
            EmergencyStopActiveError: switching on while the e-stop is latched.
            LaserError: the laser is switched manually.
        """
        if enabled:
            self._require_estop_clear()
        async with self._lock:
            if enabled:
                self._require_estop_clear()
            await self._laser.set_enabled(enabled)
            if enabled:
                await self.wait_settle(self._laser_warmup_s)

    # ----------------------------------------------------------------- detection
    async def acquire(self, n_samples: int) -> AdcSamples:
        async with self._lock:
            return await self._adc.read_samples(n_samples)

    async def measure_intensity(
        self, n_samples: int, method: SamplingMethod = SamplingMethod.MEDIAN
    ) -> IntensityMeasurement:
        """Aggregated, dark-corrected, normalized reading at the last verified position."""
        async with self._lock:
            samples = await self._adc.read_samples(n_samples)
            return build_intensity_measurement(
                samples, method, self._calibration, position=self._position
            )

    async def configure_adc(self, request: ADCCalibrateRequest) -> ADCCalibrateResponse:
        async with self._lock:
            return await configure_adc_gain(self._adc, request)

    # ----------------------------------------------------------------- calibration
    async def calibrate_dark(self, request: DarkCalibrationRequest) -> CalibrationState:
        """Measure the dark level (laser off or beam-blocked confirmation) and persist it."""
        self._require_estop_clear()
        async with self._lock:
            self._require_estop_clear()
            reading = await cal.measure_dark(
                self._adc,
                self._laser,
                request,
                laser_settle_s=self._laser_warmup_s,
                sleep=self.wait_settle,
                laser_on_allowed=lambda: not self._estop,
            )
            state = cal.with_dark(self._calibration, reading, request.notes)
            self._calibration = await cal.persist(self._store, state)
            return self._calibration

    async def calibrate_reference(self, request: ReferenceCalibrationRequest) -> CalibrationState:
        """Measure the in-focus reference level (optionally searching Z) and persist it."""
        self._require_estop_clear()
        async with self._lock:
            self._require_estop_clear()
            if self._laser.controllable and await self._laser.is_enabled() is not True:
                await self._laser.set_enabled(True)
                await self.wait_settle(self._laser_warmup_s)
            start = await self._move_base()
            position = await self._known_position()
            if request.z_search:

                async def move_z(z_um: float) -> Position:
                    self._require_estop_clear()
                    return await self._move(start.with_updates(z_um=z_um))

                position = await cal.find_reference_focus(
                    self._adc,
                    request,
                    center_um=start.z_um,
                    z_limits=self._limits.z,
                    move_z=move_z,
                    sleep=self.wait_settle,
                )
            reading = await cal.read_level(self._adc, request.n_samples, request.method)
            cal.check_reference(reading, self._calibration)
            state = cal.with_reference(self._calibration, reading, position, request.notes)
            self._calibration = await cal.persist(self._store, state)
            return self._calibration

    # ----------------------------------------------------------------- internals
    def _require_estop_clear(self) -> None:
        if self._estop:
            raise EmergencyStopActiveError(
                f"emergency stop is latched ({self._estop_reason}): reset it before moving"
            )

    def _check_limits(self, target: Position) -> None:
        violations = self._limits.violations(target)
        if violations:
            raise LimitViolationError(
                "target outside travel limits: " + "; ".join(violations), violations=violations
            )

    def _forget_position(self) -> None:
        """The stage may have moved (failure, e-stop, close): re-read before the next move."""
        self._position = None
        self._target = None

    async def _known_position(self) -> Position:
        """Last verified position, re-read from the device when unknown. Lock held."""
        if self._position is None:
            self._position = await self._stage.get_position()
        return self._position

    async def _move_base(self) -> Position:
        """Where the axes a request leaves unspecified are taken from. Lock held.

        The commanded target of the last verified move, never its step-quantised
        read-back (which may lie just outside a limit the target is on). Without
        one, the device position is re-read and each axis outside a limit by no
        more than ``position_tolerance_um`` (quantisation, never a real
        excursion) is pulled onto that limit; further out stays as read, so the
        limit check refuses it.
        """
        if self._target is not None:
            return self._target
        position = await self._known_position()
        tolerance = self._motion.position_tolerance_um
        for axis in Axis:
            limits = self._limits.for_axis(axis)
            value = position.get(axis)
            if limits.min_um - tolerance <= value < limits.min_um:
                position = position.with_axis(axis, limits.min_um)
            elif limits.max_um < value <= limits.max_um + tolerance:
                position = position.with_axis(axis, limits.max_um)
        return position

    async def _move(self, target: Position) -> Position:
        """Limit-checked, time-bounded, verified absolute move. Lock held."""
        self._check_limits(target)
        reported = await self._run_motion(
            lambda: self._stage.move_to(target), f"move to {_describe(target)}", target
        )
        await self._verify(reported, target, list(Axis))
        return reported

    def _timeout_s(self, target: Position) -> float:
        """Distance-aware timeout of a move to ``target`` (worst case if the start is unknown)."""
        start = self._target if self._target is not None else self._position
        if start is None:  # unknown: allow for a move from the farthest corner of the limits
            limits = self._limits
            corners = [
                Position(x_um=x, y_um=y, z_um=z)
                for x in (limits.x.min_um, limits.x.max_um)
                for y in (limits.y.min_um, limits.y.max_um)
                for z in (limits.z.min_um, limits.z.max_um)
            ]
            return max(self._motion.move_timeout_for(corner, target) for corner in corners)
        return self._motion.move_timeout_for(start, target)

    async def _run_motion(
        self, command: Callable[[], Awaitable[Position]], description: str, target: Position
    ) -> Position:
        """Run a stage motion, time-bounded and e-stop safe; stop the stage on failure.

        Lock held. ``command`` is called only after the final, synchronous latch
        check and awaited directly, so there is no window in which an e-stop
        could find nothing to stop and the motion then start anyway.
        """
        timeout_s = self._timeout_s(target)
        self._forget_position()
        generation = self._stop_generation
        self._require_estop_clear()  # no await between this check and the stage command
        try:
            async with asyncio.timeout(timeout_s):
                reported = await command()
        except TimeoutError:
            await self._safe_stop()
            message = f"{description} did not finish within {timeout_s:.3g} s"
            self._last_error = message
            raise MotionTimeoutError(message) from None
        except HardwareError as exc:
            await self._safe_stop()
            self._last_error = f"{description} failed: {exc}"
            raise
        except asyncio.CancelledError:
            # The caller gave up (scan task cancelled): never leave motors running.
            await self._safe_stop()
            raise
        if self._stop_generation != generation:
            # The stage finished although an e-stop happened meanwhile: never report success.
            await self._safe_stop()
            message = (
                f"{description} was interrupted by an emergency stop ({self._estop_reason}); "
                f"the stage reported {_describe(reported)}"
            )
            self._last_error = message
            raise MotionAbortedError(message)
        self._last_error = None
        return reported

    async def _verify(self, reported: Position, target: Position, axes: Sequence[Axis]) -> None:
        """Compare the device-reported position with the target on ``axes``. Lock held.

        A stage that reports the wrong position may still be moving or may have
        lost steps, so it is stopped and the position stays unknown.
        """
        tolerance = self._motion.position_tolerance_um
        bad = [
            f"{axis.value} off by {abs(reported.get(axis) - target.get(axis)):.3f} um"
            for axis in axes
            if abs(reported.get(axis) - target.get(axis)) > tolerance
        ]
        if bad:
            await self._safe_stop()
            message = (
                f"stage reports {_describe(reported)} after commanding {_describe(target)}: "
                + "; ".join(bad)
                + f" (tolerance {tolerance} um)"
            )
            self._last_error = message
            raise MotionVerificationError(message)
        self._position = reported
        self._target = target

    async def _safe_stop(self) -> None:
        await self._best_effort("stop the stage", self._stage.stop())

    @staticmethod
    async def _best_effort(action: str, operation: Awaitable[None]) -> None:
        """Await a safety action, logging (never raising) its failure."""
        try:
            await operation
        except Exception:
            log.exception("could not %s", action)

    async def _estop_laser_off(self) -> None:
        """Switch a controllable laser off; otherwise record why it is still on."""
        if not _safe_get(lambda: self._laser.controllable, False):
            self._laser_warning = MANUAL_LASER_WARNING
            return
        try:
            await self._laser.set_enabled(False)
        except Exception as exc:
            log.exception("emergency stop: could not switch the laser off")
            self._laser_warning = (
                f"the laser could not be switched off ({exc}): switch it off by hand"
            )

    # ----------------------------------------------------------------- status helpers
    async def _stage_status(self) -> StageStatus:
        try:
            status = await self._stage.status()
        except Exception as exc:
            return StageStatus(
                backend=self._stage.backend_name,
                connected=_safe_get(lambda: self._stage.connected, False),
                state=StageState.ERROR,
                position=self._position,
                estop_engaged=self._estop,
                limits=self._limits,
                last_error=f"stage status unavailable: {exc}",
            )
        return status.model_copy(
            update={
                "estop_engaged": self._estop,
                "last_error": status.last_error or self._last_error,
            }
        )

    async def _adc_status(self) -> ADCStatus:
        try:
            return await self._adc.status()
        except Exception as exc:
            gain = _safe_get(lambda: self._adc.gain, WIDEST_GAIN)
            return ADCStatus(
                backend=self._adc.backend_name,
                connected=_safe_get(lambda: self._adc.connected, False),
                gain=gain,
                full_scale_v=gain.full_scale_v,
                data_rate_sps=_safe_get(lambda: self._adc.data_rate_sps, 0),
                last_error=f"ADC status unavailable: {exc}",
            )

    async def _laser_status(self) -> LaserStatus:
        try:
            status = await self._laser.status()
        except Exception as exc:
            return LaserStatus(
                backend=self._laser.backend_name,
                connected=_safe_get(lambda: self._laser.connected, False),
                controllable=_safe_get(lambda: self._laser.controllable, False),
                enabled=None,
                wavelength_nm=NOMINAL_WAVELENGTH_NM,
                last_error=self._laser_warning or f"laser status unavailable: {exc}",
            )
        if self._laser_warning is not None:
            return status.model_copy(update={"last_error": self._laser_warning})
        return status

    async def _camera_status(self) -> CameraStatus:
        try:
            return await self._camera.status()
        except Exception as exc:
            return CameraStatus(
                backend=self._camera.backend_name,
                connected=_safe_get(lambda: self._camera.connected, False),
                available=False,
                last_error=f"camera status unavailable: {exc}",
            )
