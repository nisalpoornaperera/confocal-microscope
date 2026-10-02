"""ScanManager: the owner of every scan's lifecycle (docs/architecture.md §3, §5.7).

At most one scan is active at a time; while it is, the scan owns the
instrument. ``create_scan`` validates the whole envelope and persists the scan
before anything moves, then starts one background ``asyncio`` task and returns
at once (HTTP never waits for a scan). The task walks the state machine::

    PREPARING -> [CALIBRATING] -> [HOMING] -> SCANNING <-> PAUSED
              -> PROCESSING -> [SURFACE_RECONSTRUCTION] -> [ML_PROCESSING] -> COMPLETE

persisting every transition (``update_scan`` + audit event) before publishing
it as a STATE event. However the task ends - completion, cancel, emergency
stop, hardware failure, a bug, or cancellation at server shutdown - the scan is
left in a terminal state in the repository. ``interrupted`` is True whenever it
ended before all points were measured.

Safety: a hardware failure (or an unexpected exception while the stage may be
moving) stops the stage through ``MicroscopeController.emergency_stop``, the
only halt the controller interface offers; that latches the e-stop, so the
operator must inspect the machine and reset it before moving again. Cancel
and pause never touch the hardware: the executor stops issuing commands at its
next checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from confocal.config import MotionConfig
from confocal.errors import (
    CalibrationError,
    ConfocalError,
    EmergencyStopActiveError,
    HardwareError,
    ModelNotAvailableError,
    ReconstructionError,
    ScanConflictError,
    ScanStateError,
)
from confocal.hardware.base import MicroscopeController
from confocal.models.calibration import CalibrationState, DarkCalibrationRequest
from confocal.models.common import utc_now
from confocal.models.ml import MLAnalysisRequest, MLResult
from confocal.models.scan import (
    TERMINAL_SCAN_STATES,
    ScanConfig,
    ScanEstimate,
    ScanEvent,
    ScanEventType,
    ScanMode,
    ScanProgress,
    ScanState,
    ScanSummary,
)
from confocal.models.surface import ReconstructionRequest, SurfaceResult
from confocal.scanning.control import ScanControl, ScanStopRequested, StopKind
from confocal.scanning.events import ScanEventBroker, Subscription
from confocal.scanning.executor import ScanExecutor
from confocal.scanning.plan import ScanPlan
from confocal.scanning.postprocess import analyse_and_save, reconstruct_and_save
from confocal.scanning.profile_buffer import CalibrationSnapshot
from confocal.scanning.progress import LiveEventConfig, ProgressTracker
from confocal.scanning.protocols import (
    CoarsePeakFinder,
    MLAnalyser,
    ProfileAnalyser,
    SurfaceReconstructor,
)
from confocal.scanning.state import ScanStateMachine, StateTransition
from confocal.storage.interfaces import ScanRepository

log = logging.getLogger(__name__)

SHUTDOWN_MESSAGE = "server shutdown"

#: States in which the stage may be in motion: a failure there must stop it.
_MOTION_STATES = frozenset(
    {
        ScanState.PREPARING,
        ScanState.CALIBRATING,
        ScanState.HOMING,
        ScanState.SCANNING,
        ScanState.PAUSED,
    }
)


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _transition_text(record: StateTransition) -> str:
    previous = record.previous.value if record.previous is not None else "-"
    text = f"{previous} -> {record.state.value}"
    return f"{text}: {record.reason}" if record.reason else text


@dataclass(eq=False)
class _ActiveScan:
    """Runtime state of the scan that currently owns the instrument."""

    scan_id: str
    plan: ScanPlan
    machine: ScanStateMachine
    control: ScanControl
    tracker: ProgressTracker
    calibration_version: int | None = None  # recorded in the scan row
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    measurement_complete: bool = False

    @property
    def config(self) -> ScanConfig:
        return self.plan.config


@dataclass(frozen=True, slots=True)
class _Ending:
    """How a scan ends: terminal state, message, audit kind, whether to halt the stage."""

    state: ScanState
    message: str
    audit_kind: str
    stop_motion: bool = False


class ScanManager:
    """Creates scans, runs them in the background and answers queries about them."""

    def __init__(
        self,
        controller: MicroscopeController,
        repository: ScanRepository,
        *,
        broker: ScanEventBroker,
        find_coarse_peak: CoarsePeakFinder,
        analyse_profile: ProfileAnalyser,
        reconstruct_surface: SurfaceReconstructor,
        ml_analyser: MLAnalyser | None = None,
        motion: MotionConfig,
        software_version: str,
        live_events: LiveEventConfig | None = None,
        default_saturation_v: float | None = None,
    ) -> None:
        """``default_saturation_v`` is the detector rail (``Settings.processing``):
        it becomes ``processing.saturation_v`` of every scan that sets none, and
        is persisted with the scan's config.
        """
        if default_saturation_v is not None and not default_saturation_v > 0.0:
            raise ValueError("default_saturation_v must be > 0")
        self._default_saturation_v = default_saturation_v
        self._controller = controller
        self._repository = repository
        self._broker = broker
        self._find_coarse_peak = find_coarse_peak
        self._analyse_profile = analyse_profile
        self._reconstruct_surface = reconstruct_surface
        self._ml_analyser = ml_analyser
        self._motion = motion
        self._software_version = software_version
        self._live_events = live_events or LiveEventConfig()
        self._active: _ActiveScan | None = None
        # Id of the scan create_scan is building: it owns the instrument from the
        # first await on, so no manual command slips in before it is registered.
        self._reserved_id: str | None = None
        self._create_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        controller.add_estop_listener(self.handle_emergency_stop)

    # ------------------------------------------------------------------ status
    @property
    def active_scan_id(self) -> str | None:
        if self._active is not None:
            return self._active.scan_id
        return self._reserved_id

    @property
    def active_scan_state(self) -> ScanState | None:
        if self._active is not None:
            return self._active.machine.state
        return ScanState.PREPARING if self._reserved_id is not None else None

    @property
    def is_active(self) -> bool:
        """True while a scan owns the instrument (manual hardware control is refused).

        Already True while ``create_scan`` validates and persists a new scan.
        """
        return self._active is not None or self._reserved_id is not None

    # ------------------------------------------------------------------ creation
    async def estimate(self, config: ScanConfig) -> ScanEstimate:
        """Duration / data estimate and limit check, without creating anything.

        The plan (up to 250 000 grid points) is built in a worker thread.
        """
        config = self._with_defaults(config)
        status = await self._controller.status()
        limits = self._controller.limits
        rate = status.adc.data_rate_sps

        def build() -> ScanEstimate:
            return ScanPlan.from_config(config, limits, self._motion, rate).estimate()

        return await asyncio.to_thread(build)

    async def create_scan(self, config: ScanConfig) -> ScanSummary:
        """Validate, persist and start a scan; returns as soon as the task is started.

        Raises:
            ScanConflictError: another scan is active (or the manager is shutting down).
            EmergencyStopActiveError: the e-stop is latched.
            LimitViolationError: the scan envelope leaves the travel limits.
            CalibrationError: an automatic dark calibration was requested but the
                laser is switched manually (it needs the operator to block the beam).
        Nothing is persisted or moved when any of these is raised.

        The instrument is reserved for the new scan before the first ``await``
        (``is_active`` is True), so no manual command can run between these
        checks and the start of the scan; a failed creation releases it. The
        plan is built and checked in a worker thread.
        """
        async with self._create_lock:
            self._ensure_available()
            scan_id = uuid.uuid4().hex
            self._reserved_id = scan_id
            try:
                return await self._create_reserved(scan_id, self._with_defaults(config))
            finally:
                self._reserved_id = None

    async def _create_reserved(self, scan_id: str, config: ScanConfig) -> ScanSummary:
        status = await self._controller.status()
        limits = self._controller.limits
        rate = status.adc.data_rate_sps

        def build_plan() -> ScanPlan:
            plan = ScanPlan.from_config(config, limits, self._motion, rate)
            plan.validate_limits()
            return plan

        plan = await asyncio.to_thread(build_plan)
        if config.calibrate_dark_before_scan and not status.laser.controllable:
            raise CalibrationError(
                "calibrate_dark_before_scan needs a software-switchable laser; with a "
                "manual laser run the dark calibration (beam blocked) before the scan"
            )
        if self._controller.estop_engaged:
            raise EmergencyStopActiveError(
                "emergency stop is latched: reset it before starting a scan"
            )
        summary = await asyncio.to_thread(
            self._repository.create_scan,
            scan_id=scan_id,
            config=config,
            total_points=plan.total_points,
            calibration=self._controller.calibration,
            software_version=self._software_version,
            hardware=self._controller.hardware_info(),
        )
        scan = _ActiveScan(
            scan_id=scan_id,
            plan=plan,
            machine=ScanStateMachine(),
            control=ScanControl(),
            tracker=ProgressTracker(
                scan_id,
                plan.total_points,
                eta_window_points=self._live_events.eta_window_points,
            ),
            calibration_version=summary.calibration_version,
        )
        self._active = scan
        task = asyncio.create_task(self._run(scan), name=f"scan-{scan_id}")
        scan.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        log.info("scan %s created (%d points)", scan_id, plan.total_points)
        return summary

    def _with_defaults(self, config: ScanConfig) -> ScanConfig:
        """Fill in the settings-level detector rail when the scan sets no saturation_v."""
        rail_v = self._default_saturation_v
        if rail_v is None or config.processing.saturation_v is not None:
            return config
        processing = config.processing.model_copy(update={"saturation_v": rail_v})
        return config.model_copy(update={"processing": processing})

    def _ensure_available(self) -> None:
        if self._closing:
            raise ScanConflictError("the scan manager is shutting down")
        scan = self._active
        if scan is not None:
            raise ScanConflictError(
                f"scan {scan.scan_id} is active ({scan.machine.state.value}); "
                "wait for it to finish or cancel it"
            )
        if self._reserved_id is not None:
            raise ScanConflictError(f"scan {self._reserved_id} is being created")

    # ------------------------------------------------------------------ control
    async def pause(self, scan_id: str) -> ScanSummary:
        """Pause after the point being measured completes."""
        scan = await self._require_active(scan_id)
        state = scan.machine.state
        if state not in (ScanState.SCANNING, ScanState.PAUSED):
            raise ScanStateError(f"cannot pause a scan in state {state.value}")
        scan.control.request_pause()
        return await self.get_scan(scan_id)

    async def resume(self, scan_id: str) -> ScanSummary:
        scan = await self._require_active(scan_id)
        if not scan.control.pause_requested:
            raise ScanStateError(f"scan {scan_id} is not paused")
        scan.control.request_resume()
        return await self.get_scan(scan_id)

    async def cancel(self, scan_id: str) -> ScanSummary:
        """Request a cancel; the scan becomes CANCELLED at its next checkpoint."""
        scan = await self._require_active(scan_id)
        scan.control.request_cancel()
        return await self.get_scan(scan_id)

    async def handle_emergency_stop(self, reason: str) -> None:
        """E-stop listener: the active scan ends in ERROR (a paused scan is woken up)."""
        scan = self._active
        if scan is None or scan.machine.is_terminal:
            return
        log.warning("scan %s: emergency stop: %s", scan.scan_id, reason)
        scan.control.request_abort(f"emergency stop: {reason}")

    async def _require_active(self, scan_id: str) -> _ActiveScan:
        scan = self._active
        if scan is not None and scan.scan_id == scan_id and not scan.machine.is_terminal:
            return scan
        summary = await asyncio.to_thread(self._repository.get_scan, scan_id)
        raise ScanStateError(f"scan {scan_id} is not active (state {summary.state.value})")

    # ------------------------------------------------------------------ queries
    async def get_scan(self, scan_id: str) -> ScanSummary:
        """Stored summary, overlaid with the live state and progress of an active scan."""
        summary = await asyncio.to_thread(self._repository.get_scan, scan_id)
        return self._overlay(summary)

    async def list_scans(self, *, limit: int = 100, offset: int = 0) -> list[ScanSummary]:
        summaries = await asyncio.to_thread(self._repository.list_scans, limit=limit, offset=offset)
        return [self._overlay(summary) for summary in summaries]

    async def get_progress(self, scan_id: str) -> ScanProgress:
        scan = self._active
        if scan is not None and scan.scan_id == scan_id:
            return scan.tracker.snapshot()
        summary = await asyncio.to_thread(self._repository.get_scan, scan_id)
        total = summary.total_points
        return ScanProgress(
            scan_id=summary.id,
            state=summary.state,
            progress=min(1.0, summary.completed_points / total) if total > 0 else 0.0,
            completed_points=summary.completed_points,
            total_points=total,
        )

    async def subscribe(self, scan_id: str) -> Subscription:
        """Live events of a scan; the subscription of a finished scan ends immediately.

        Raises:
            ScanNotFoundError: unknown scan.
        """
        scan = self._active
        if scan is not None and scan.scan_id == scan_id:
            return self._broker.subscribe(scan_id)
        await asyncio.to_thread(self._repository.get_scan, scan_id)
        subscription = self._broker.subscribe(scan_id)
        subscription.close()
        return subscription

    async def wait_until_finished(
        self,
        scan_id: str,
        timeout: float | None = None,  # noqa: ASYNC109 - parameter name fixed by §5.7
    ) -> ScanSummary:
        """Wait for the scan's task to end; raises ``TimeoutError`` after ``timeout`` s."""
        scan = self._active
        if scan is not None and scan.scan_id == scan_id:
            await asyncio.wait_for(scan.done.wait(), timeout)
        return await self.get_scan(scan_id)

    def _overlay(self, summary: ScanSummary) -> ScanSummary:
        scan = self._active
        if scan is None or scan.scan_id != summary.id:
            return summary
        return summary.model_copy(
            update={
                "state": scan.machine.state,
                "completed_points": scan.tracker.completed_points,
                "progress": scan.tracker.progress,
            }
        )

    # ------------------------------------------------------------------ on-demand analysis
    async def reconstruct(
        self, scan_id: str, request: ReconstructionRequest | None = None
    ) -> SurfaceResult:
        """(Re)build and store the surface of a finished confocal scan.

        Raises:
            ScanConflictError: the scan is still active.
            ScanNotFoundError: unknown scan.
            ReconstructionError: fixed-Z scan, or too few usable points.
        """
        self._ensure_not_active(scan_id)
        summary = await asyncio.to_thread(self._repository.get_scan, scan_id)
        if summary.mode is not ScanMode.CONFOCAL:
            raise ReconstructionError("fixed-Z scans have no surface heights to reconstruct")
        return await asyncio.to_thread(
            reconstruct_and_save,
            self._repository,
            self._reconstruct_surface,
            scan_id,
            request or summary.config.reconstruction,
            xy_step_um=summary.config.xy_step_um,
        )

    async def analyse_ml(self, scan_id: str, request: MLAnalysisRequest) -> MLResult:
        """Advisory ML analysis of a finished scan (never changes physics results).

        Raises:
            ScanConflictError: the scan is still active.
            ScanNotFoundError: unknown scan.
            ModelNotAvailableError: no analyser or no deployed model.
        """
        self._ensure_not_active(scan_id)
        analyser = self._ml_analyser
        if analyser is None:
            raise ModelNotAvailableError("no ML analyser is configured")
        await asyncio.to_thread(self._repository.get_scan, scan_id)
        return await asyncio.to_thread(
            analyse_and_save, self._repository, analyser, scan_id, request
        )

    def _ensure_not_active(self, scan_id: str) -> None:
        if self.active_scan_id == scan_id:
            raise ScanConflictError(f"scan {scan_id} is still active")

    # ------------------------------------------------------------------ shutdown
    async def shutdown(self, timeout_s: float = 10.0) -> None:
        """Cancel the active scan's task and make sure it is stored as interrupted."""
        self._closing = True
        scan = self._active
        if scan is None:
            return
        task = scan.task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=timeout_s)
        # A task cancelled before its first step never ran its own cleanup.
        self._release(scan)
        try:
            summary = await asyncio.to_thread(self._repository.get_scan, scan.scan_id)
            if summary.state not in TERMINAL_SCAN_STATES:
                await asyncio.to_thread(
                    self._persist_end,
                    scan,
                    ScanState.ERROR,
                    SHUTDOWN_MESSAGE,
                    "scan_interrupted",
                    utc_now(),
                )
        except Exception:
            log.exception("scan %s: could not record the shutdown", scan.scan_id)

    # ------------------------------------------------------------------ the scan task
    async def _run(self, scan: _ActiveScan) -> None:
        try:
            await self._run_phases(scan)
        except asyncio.CancelledError:
            await self._end(scan, _Ending(ScanState.ERROR, SHUTDOWN_MESSAGE, "scan_interrupted"))
            raise
        except Exception as exc:
            await self._fail(scan, exc)
        finally:
            self._release(scan)

    async def _run_phases(self, scan: _ActiveScan) -> None:
        cfg = scan.config
        scan.tracker.start()
        await self._enter(scan, ScanState.PREPARING, started_at=utc_now())
        calibration = await self._prepare(scan)
        if cfg.home_before_scan:
            scan.control.checkpoint()
            await self._enter(scan, ScanState.HOMING)
            await self._controller.home()
        scan.control.checkpoint()
        await self._enter(scan, ScanState.SCANNING)
        await self._executor(scan, calibration).run()
        scan.measurement_complete = True
        await self._enter(scan, ScanState.PROCESSING)
        await self._post_process(scan)
        await self._end(scan, _Ending(ScanState.COMPLETE, "scan complete", "scan_state"))

    async def _prepare(self, scan: _ActiveScan) -> CalibrationState:
        """Laser on (when switchable) and the automatic dark calibration of §3."""
        status = await self._controller.status()
        laser = status.laser
        if laser.controllable and laser.enabled is not True:
            await self._controller.set_laser(True)
        calibration = self._controller.calibration
        if scan.config.calibrate_dark_before_scan or (
            not calibration.has_dark and laser.controllable
        ):
            scan.control.checkpoint()
            await self._enter(scan, ScanState.CALIBRATING, "measuring the dark level")
            calibration = await self._controller.calibrate_dark(DarkCalibrationRequest())
        elif not calibration.has_dark:
            await self._warn(
                scan,
                "no dark calibration: the dark level is treated as 0 V "
                "(block the beam and run a dark calibration for corrected intensities)",
            )
        if calibration.version != scan.calibration_version:
            # The scan row must name the calibration actually applied to its points:
            # the automatic dark calibration, or one that completed during creation.
            await asyncio.to_thread(
                self._repository.update_scan, scan.scan_id, calibration=calibration
            )
            scan.calibration_version = calibration.version
        return calibration

    def _executor(self, scan: _ActiveScan, calibration: CalibrationState) -> ScanExecutor:
        return ScanExecutor(
            scan_id=scan.scan_id,
            plan=scan.plan,
            controller=self._controller,
            repository=self._repository,
            broker=self._broker,
            find_coarse_peak=self._find_coarse_peak,
            analyse_profile=self._analyse_profile,
            calibration=CalibrationSnapshot.from_state(calibration),
            control=scan.control,
            progress=scan.tracker,
            on_pause=lambda: self._enter(scan, ScanState.PAUSED, "paused by user"),
            on_resume=lambda: self._enter(scan, ScanState.SCANNING, "resumed"),
            live_events=self._live_events,
        )

    async def _post_process(self, scan: _ActiveScan) -> None:
        cfg = scan.config
        scan.control.checkpoint()
        if cfg.mode is ScanMode.CONFOCAL and cfg.reconstruct_on_complete:
            await self._enter(scan, ScanState.SURFACE_RECONSTRUCTION)
            try:
                await asyncio.to_thread(
                    reconstruct_and_save,
                    self._repository,
                    self._reconstruct_surface,
                    scan.scan_id,
                    cfg.reconstruction,
                    xy_step_um=cfg.xy_step_um,
                )
            except ReconstructionError as exc:
                await self._warn(scan, f"surface reconstruction failed (data kept): {exc}")
            scan.control.checkpoint()
        analyser = self._ml_analyser
        if cfg.ml_on_complete and analyser is not None and await self._model_available(analyser):
            await self._enter(scan, ScanState.ML_PROCESSING)
            try:
                await asyncio.to_thread(
                    analyse_and_save, self._repository, analyser, scan.scan_id, MLAnalysisRequest()
                )
            except Exception as exc:  # ML is advisory: it never fails a measured scan
                await self._warn(scan, f"ML analysis failed (advisory only): {_describe(exc)}")
            scan.control.checkpoint()

    async def _model_available(self, analyser: MLAnalyser) -> bool:
        try:
            return await asyncio.to_thread(analyser.has_model)
        except Exception:
            log.exception("ML model lookup failed")
            return False

    # ------------------------------------------------------------------ failures
    async def _fail(self, scan: _ActiveScan, exc: Exception) -> None:
        ending = self._classify(scan, exc)
        if ending.state is ScanState.ERROR and ending.audit_kind != "emergency_stop":
            log.error("scan %s failed: %s", scan.scan_id, ending.message, exc_info=exc)
        if ending.stop_motion and scan.machine.state in _MOTION_STATES:
            try:
                await self._controller.emergency_stop(f"scan {scan.scan_id}: {ending.message}")
            except Exception:
                log.exception("scan %s: could not stop the stage", scan.scan_id)
        await self._end(scan, ending)

    def _classify(self, scan: _ActiveScan, exc: Exception) -> _Ending:
        stop = scan.control.stop_request
        if stop is not None and stop.kind is StopKind.ABORT:
            return _Ending(ScanState.ERROR, stop.message, "emergency_stop")
        if self._controller.estop_engaged:
            # The e-stop latched but its listener has not run yet.
            return _Ending(
                ScanState.ERROR, "emergency stop: engaged during the scan", "emergency_stop"
            )
        if isinstance(exc, ScanStopRequested):
            return _Ending(ScanState.CANCELLED, exc.request.message, "scan_cancelled")
        if isinstance(exc, HardwareError):
            return _Ending(
                ScanState.ERROR, f"hardware failure: {_describe(exc)}", "hardware_failure", True
            )
        # Domain / safety errors are raised before anything moves; anything else
        # (a bug, possibly inside a driver mid-move) must stop the stage.
        return _Ending(
            ScanState.ERROR,
            f"unexpected error: {_describe(exc)}",
            "scan_error",
            not isinstance(exc, ConfocalError),
        )

    # ------------------------------------------------------------------ transitions
    async def _enter(
        self,
        scan: _ActiveScan,
        state: ScanState,
        reason: str | None = None,
        *,
        started_at: datetime | None = None,
    ) -> None:
        """Validate, persist, then publish a non-terminal transition."""
        record = scan.machine.transition(state, reason)
        scan.tracker.state = state
        await asyncio.to_thread(self._persist_transition, scan.scan_id, record, started_at)
        self._publish_state(scan, reason)

    async def _end(self, scan: _ActiveScan, ending: _Ending) -> None:
        """Terminal transition. Never raises: persistence failures are logged."""
        if not scan.machine.is_terminal:
            scan.machine.transition(ending.state, ending.message)
        scan.tracker.state = scan.machine.state
        try:
            await asyncio.to_thread(
                self._persist_end,
                scan,
                scan.machine.state,
                ending.message,
                ending.audit_kind,
                scan.machine.entered_at,
            )
        except Exception:
            log.exception("scan %s: could not persist the final state", scan.scan_id)
        self._publish_state(scan, ending.message)
        log.info("scan %s ended: %s (%s)", scan.scan_id, scan.machine.state.value, ending.message)

    def _persist_transition(
        self, scan_id: str, record: StateTransition, started_at: datetime | None
    ) -> None:
        self._repository.update_scan(scan_id, state=record.state, started_at=started_at)
        self._repository.record_event("scan_state", _transition_text(record), scan_id=scan_id)

    def _persist_end(
        self,
        scan: _ActiveScan,
        state: ScanState,
        message: str,
        audit_kind: str,
        finished_at: datetime,
    ) -> None:
        self._repository.update_scan(
            scan.scan_id,
            state=state,
            completed_points=scan.tracker.completed_points,
            finished_at=finished_at,
            error_message=message if state is ScanState.ERROR else None,
            interrupted=not scan.measurement_complete,
        )
        self._repository.record_event(audit_kind, f"{state.value}: {message}", scan_id=scan.scan_id)

    def _publish_state(self, scan: _ActiveScan, message: str | None) -> None:
        self._broker.publish(
            ScanEvent(
                type=ScanEventType.STATE,
                scan_id=scan.scan_id,
                progress=scan.tracker.snapshot(message),
                message=message,
            )
        )

    async def _warn(self, scan: _ActiveScan, message: str) -> None:
        """Non-fatal problem: audit log + an ERROR event; the scan state is unaffected."""
        log.warning("scan %s: %s", scan.scan_id, message)
        try:
            await asyncio.to_thread(
                self._repository.record_event, "warning", message, scan_id=scan.scan_id
            )
        except Exception:
            log.exception("scan %s: could not record a warning", scan.scan_id)
        self._broker.publish(
            ScanEvent(
                type=ScanEventType.ERROR,
                scan_id=scan.scan_id,
                progress=scan.tracker.snapshot(message),
                message=f"warning: {message}",
            )
        )

    def _release(self, scan: _ActiveScan) -> None:
        """Free the instrument and end live subscriptions. Idempotent."""
        if self._active is scan:
            self._active = None
        self._broker.close_scan(scan.scan_id)
        scan.done.set()
