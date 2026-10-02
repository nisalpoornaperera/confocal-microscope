"""Per-point acquisition: the confocal procedure of docs/architecture.md §4.

For every XY point in plan order the executor moves to (x, y) at the current
Z, centres a coarse Z sweep on the adaptive estimate, locates the coarse peak
(retrying once over the full range when a narrow adaptive sweep missed it),
runs a fine sweep around it and hands the fine profile, with the Z positions
*reported* by the stage, to the injected physics pipeline. The complete raw
profile, the analysis and the scalar result are persisted before the next
point starts, so a crash loses at most the point being measured.

Every sweep is clipped to the validated scan envelope (``ScanPlan.sweep_limits``),
and every move goes through ``MicroscopeController.move_to``, which checks the
travel limits and verifies the reported position. The executor never knows
how the stage realises a Cartesian move.

Interruption: a cancel or emergency stop is honoured before the next Z step;
the point being measured is then stored as ABORTED with all raw data acquired
so far (raw data is never discarded) and the exception propagates to the
ScanManager, which decides the final scan state. A pause is honoured between
points. A failure of the injected analysis code is not a hardware problem:
that point is stored with status ERROR and the scan continues.

Saturation: a point any of whose raw ADC codes sits on a converter rail is
flagged ``saturated``. The analysis needs a voltage threshold for the clipped
top: :data:`ADC_SATURATION_FRACTION` x the full scale of the ADC gain in
effect, or the scan's ``processing.saturation_v`` when that is lower (the
ScanManager fills in the detector rail from the settings), so clipping at the
converter or at the detector is never invisible to the analysis.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from confocal.hardware.base import MicroscopeController
from confocal.models.common import AxisLimits
from confocal.models.hardware import AdcGain
from confocal.models.measurement import ProfileData, ProfilePhase
from confocal.models.processing import (
    CoarsePeak,
    PointStatus,
    ProcessingConfig,
    ProfileAnalysis,
)
from confocal.models.scan import LiveProfile, ScanEvent, ScanEventType, ScanPoint
from confocal.scanning.control import ScanControl
from confocal.scanning.events import ScanEventBroker
from confocal.scanning.plan import GridPoint, ScanPlan, sweep_is_clipped, z_sweep
from confocal.scanning.profile_buffer import CalibrationSnapshot, ProfileBuffer, hits_adc_rails
from confocal.scanning.progress import Clock, EventThrottle, LiveEventConfig, ProgressTracker
from confocal.scanning.protocols import CoarsePeakFinder, ProfileAnalyser
from confocal.scanning.results import (
    FLAG_ABORTED,
    FLAG_ANALYSIS_FAILED,
    FLAG_COARSE_CLIPPED,
    FLAG_COARSE_RETRY,
    FLAG_FINE_CLIPPED,
    FLAG_SATURATED,
    PointContext,
    confocal_point,
    failed_point,
    fixed_z_point,
    no_peak_analysis,
)
from confocal.scanning.z_estimation import AdaptiveZEstimator
from confocal.storage.interfaces import ScanRepository

log = logging.getLogger(__name__)

#: Coroutine run by the executor when it enters / leaves the paused state.
StateHook = Callable[[], Awaitable[None]]

_T = TypeVar("_T")

#: Without a configured ``saturation_v`` the analysis treats samples at or above
#: this fraction of the ADC full scale (of the gain in effect) as saturated.
ADC_SATURATION_FRACTION = 0.98


class _AnalysisFailedError(Exception):
    """An injected analysis function raised or returned inconsistent data."""


@dataclass(frozen=True, slots=True)
class _PointResult:
    point: ScanPoint
    profile: ProfileData | None
    analysis: ProfileAnalysis | None
    surface_z_um: float | None  # fed to the Z estimator: VALID points only


class ScanExecutor:
    """Runs the acquisition phase (state SCANNING / PAUSED) of one scan."""

    def __init__(
        self,
        *,
        scan_id: str,
        plan: ScanPlan,
        controller: MicroscopeController,
        repository: ScanRepository,
        broker: ScanEventBroker,
        find_coarse_peak: CoarsePeakFinder,
        analyse_profile: ProfileAnalyser,
        calibration: CalibrationSnapshot,
        control: ScanControl,
        progress: ProgressTracker,
        on_pause: StateHook | None = None,
        on_resume: StateHook | None = None,
        live_events: LiveEventConfig | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        events = live_events or LiveEventConfig()
        self._scan_id = scan_id
        self._plan = plan
        self._config = plan.config
        self._controller = controller
        self._repository = repository
        self._broker = broker
        self._find_coarse_peak = find_coarse_peak
        self._analyse_profile = analyse_profile
        self._calibration = calibration
        self._control = control
        self._progress = progress
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._clock = clock
        self._settle_s = self._config.settle_time_ms / 1000.0
        self._z_limits: AxisLimits | None = plan.sweep_limits() if plan.is_confocal else None
        self._estimator = AdaptiveZEstimator(self._config, n_x=plan.n_x, n_y=plan.n_y)
        self._progress_throttle = EventThrottle(events.progress_interval_s, clock)
        self._profile_throttle = EventThrottle(events.profile_interval_s, clock)
        self._processing_by_gain: dict[AdcGain, ProcessingConfig] = {}

    async def run(self) -> None:
        """Measure every point of the plan.

        Raises:
            ScanStopRequested: cancel / emergency stop requested (current point stored ABORTED).
            HardwareError, EmergencyStopActiveError, ...: from the controller (same).
            StorageError: a point could not be persisted.
        """
        for grid_point in self._plan.points:
            await self._between_points()
            await self._measure(grid_point)

    async def _between_points(self) -> None:
        self._control.checkpoint()
        if not self._control.pause_requested:
            return
        if self._on_pause is not None:
            await self._on_pause()
        await self._control.wait_while_paused()
        if self._on_resume is not None:
            await self._on_resume()

    # ------------------------------------------------------------------ one point
    async def _measure(self, grid_point: GridPoint) -> None:
        cfg = self._config
        ctx = PointContext(
            grid_point, ProfileBuffer(cfg.samples_per_z, cfg.sampling_method), self._clock()
        )
        self._progress.begin_point(grid_point)
        try:
            if self._plan.is_confocal:
                result = await self._measure_confocal(ctx)
            else:
                result = await self._measure_fixed_z(ctx)
        except _AnalysisFailedError as failure:
            log.error(
                "scan %s point %d: analysis failed, raw data kept: %s",
                self._scan_id,
                grid_point.point_id,
                failure,
            )
            result = self._failed_result(ctx, PointStatus.ERROR, FLAG_ANALYSIS_FAILED)
        except BaseException:
            await self._store_aborted(ctx)
            raise
        await self._store(ctx, result)

    async def _measure_confocal(self, ctx: PointContext) -> _PointResult:
        cfg = self._config
        gp = ctx.grid_point
        position = await self._controller.move_to(x_um=gp.x_um, y_um=gp.y_um)
        self._progress.update(position, None)
        self._publish_progress()

        estimate = self._estimator.estimate(gp)
        ctx.z_estimate_um = estimate.center_um
        peak = await self._coarse_sweep(ctx, estimate.center_um, estimate.width_um)
        if estimate.source != "default" and (
            not peak.found or peak.at_edge or peak.global_max_not_selected
        ):
            # The narrow adaptive range missed the surface (a step, a steep slope)
            # or caught only part of it (the brightest signal is not the selected
            # peak): repeat once over the full configured range before giving up.
            ctx.flag(FLAG_COARSE_RETRY)
            peak = await self._coarse_sweep(ctx, cfg.z_center_um, cfg.z_range_um)
        if not peak.found or peak.z_um is None or not math.isfinite(peak.z_um):
            return self._analysed(ctx, no_peak_analysis(self._calibration.signal_units), None)

        fine_z = self._sweep_positions(
            ctx, peak.z_um, cfg.fine_z_range_um, cfg.fine_z_step_um, FLAG_FINE_CLIPPED
        )
        fine = await self._sweep(ctx, fine_z, ProfilePhase.FINE)
        analysis, filtered = await self._analyse_fine(ctx, fine)
        return self._analysed(ctx, analysis, (fine, filtered))

    async def _measure_fixed_z(self, ctx: PointContext) -> _PointResult:
        gp = ctx.grid_point
        voltage = await self._acquire(
            ctx, ProfilePhase.FIXED, z_um=self._config.z_center_um, x_um=gp.x_um, y_um=gp.y_um
        )
        point = fixed_z_point(
            ctx,
            self._calibration.intensity_of(voltage),
            duration_s=self._clock() - ctx.started_s,
        )
        return _PointResult(point, ctx.buffer.build(self._calibration), None, None)

    # ------------------------------------------------------------------ sweeps
    async def _coarse_sweep(
        self, ctx: PointContext, center_um: float, width_um: float
    ) -> CoarsePeak:
        cfg = self._config
        positions = self._sweep_positions(
            ctx, center_um, width_um, cfg.coarse_z_step_um, FLAG_COARSE_CLIPPED
        )
        part = await self._sweep(ctx, positions, ProfilePhase.COARSE)
        peak = await self._run_analysis(
            functools.partial(
                self._find_coarse_peak,
                ctx.buffer.reported_z(part),
                ctx.buffer.aggregated(part),
                dark_v=self._calibration.dark_v,
                config=self._processing(ctx),
            )
        )
        ctx.coarse_peak = peak
        return peak

    def _sweep_positions(
        self, ctx: PointContext, center_um: float, width_um: float, step_um: float, flag: str
    ) -> NDArray[np.float64]:
        """Increasing sweep positions clipped to the scan envelope (never empty)."""
        limits = self._z_limits
        if limits is None:
            raise RuntimeError("Z sweeps are only made in confocal mode")
        center = limits.clamp(center_um)
        if sweep_is_clipped(center, width_um, limits):
            ctx.flag(flag)
        return z_sweep(center, width_um, step_um, limits)

    async def _sweep(
        self, ctx: PointContext, positions: NDArray[np.float64], phase: ProfilePhase
    ) -> slice:
        """Acquire at every position; returns the buffer slice the sweep occupies."""
        start = ctx.buffer.n
        for z in positions:
            await self._acquire(ctx, phase, z_um=float(z))
        return slice(start, ctx.buffer.n)

    async def _acquire(
        self,
        ctx: PointContext,
        phase: ProfilePhase,
        *,
        z_um: float,
        x_um: float | None = None,
        y_um: float | None = None,
    ) -> float:
        """One Z step: checkpoint, verified move, settle, raw burst. Returns the aggregate."""
        self._control.checkpoint()
        position = await self._controller.move_to(x_um=x_um, y_um=y_um, z_um=z_um)
        await self._controller.wait_settle(self._settle_s)
        samples = await self._controller.acquire(self._config.samples_per_z)
        voltage = ctx.buffer.add(
            z_um=z_um, z_reported_um=position.z_um, phase=phase, samples=samples
        )
        if hits_adc_rails(samples.counts):
            ctx.flag(FLAG_SATURATED)
        self._progress.update(position, self._calibration.intensity_of(voltage))
        self._publish_progress()
        if self._plan.is_confocal and self._profile_throttle.ready():
            self._publish_profile(ctx)
        return voltage

    # ------------------------------------------------------------------ analysis
    def _processing(self, ctx: PointContext) -> ProcessingConfig:
        """The scan's processing config with an effective saturation threshold.

        The converter rail always bounds the threshold: samples at or above
        :data:`ADC_SATURATION_FRACTION` of the full scale of the gain the point
        was measured with count as saturated. A configured ``saturation_v``
        (the detector rail) applies when it is lower, so neither the detector
        clipping below the ADC full scale nor the ADC clipping below the
        detector rail (a high PGA gain) is invisible to the analysis.
        """
        config = self._config.processing
        gain = ctx.buffer.gain
        if gain is None:
            return config
        effective = self._processing_by_gain.get(gain)
        if effective is None:
            rail_v = ADC_SATURATION_FRACTION * gain.full_scale_v
            configured = config.saturation_v
            if configured is not None and configured <= rail_v:
                effective = config
            else:
                effective = config.model_copy(update={"saturation_v": rail_v})
            self._processing_by_gain[gain] = effective
        return effective

    async def _run_analysis(self, call: Callable[[], _T]) -> _T:
        """Run injected CPU work off the event loop; failures become _AnalysisFailedError."""
        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            raise _AnalysisFailedError(f"{type(exc).__name__}: {exc}") from exc

    async def _analyse_fine(
        self, ctx: PointContext, fine: slice
    ) -> tuple[ProfileAnalysis, NDArray[np.float64]]:
        z_reported = ctx.buffer.reported_z(fine)
        processed = await self._run_analysis(
            functools.partial(
                self._analyse_profile,
                z_reported,
                ctx.buffer.aggregated(fine),
                dark_v=self._calibration.dark_v,
                reference_v=self._calibration.reference_v,
                config=self._processing(ctx),
            )
        )
        filtered = np.asarray(processed.filtered, dtype=np.float64)
        if filtered.shape != z_reported.shape:
            raise _AnalysisFailedError(
                f"analyser returned {filtered.shape} filtered values "
                f"for {z_reported.size} positions"
            )
        return processed.analysis, filtered

    def _analysed(
        self,
        ctx: PointContext,
        analysis: ProfileAnalysis,
        filtered: tuple[slice, NDArray[np.float64]] | None,
    ) -> _PointResult:
        self._publish_profile(ctx)  # the complete I(Z) of this point, unthrottled
        point = confocal_point(ctx, analysis, duration_s=self._clock() - ctx.started_s)
        profile = ctx.buffer.build(self._calibration, filtered=filtered)
        surface = point.surface_z_um if point.status is PointStatus.VALID else None
        return _PointResult(point, profile, analysis, surface)

    def _failed_result(self, ctx: PointContext, status: PointStatus, flag: str) -> _PointResult:
        point = failed_point(ctx, status, flag, duration_s=self._clock() - ctx.started_s)
        profile = ctx.buffer.build(self._calibration) if ctx.buffer.n else None
        return _PointResult(point, profile, None, None)

    # ------------------------------------------------------------------ persistence
    async def _store(self, ctx: PointContext, result: _PointResult) -> None:
        completed = self._progress.completed_points + 1
        await asyncio.to_thread(
            self._persist, result.point, result.profile, result.analysis, completed
        )
        self._progress.finish_point(result.point.duration_s)
        self._estimator.record(ctx.grid_point, result.surface_z_um)
        self._publish(ScanEventType.POINT, point=result.point)

    async def _store_aborted(self, ctx: PointContext) -> None:
        """Persist the interrupted point with whatever raw data it has (best effort)."""
        result = self._failed_result(ctx, PointStatus.ABORTED, FLAG_ABORTED)
        try:
            await asyncio.to_thread(self._persist, result.point, result.profile, None, None)
        except Exception:
            log.exception(
                "scan %s: could not store aborted point %d", self._scan_id, result.point.point_id
            )
            return
        self._publish(ScanEventType.POINT, point=result.point)

    def _persist(
        self,
        point: ScanPoint,
        profile: ProfileData | None,
        analysis: ProfileAnalysis | None,
        completed_points: int | None,
    ) -> None:
        self._repository.append_point(self._scan_id, point, profile, analysis)
        if completed_points is not None:
            self._repository.update_scan(self._scan_id, completed_points=completed_points)

    # ------------------------------------------------------------------ events
    def _publish(
        self,
        event_type: ScanEventType,
        *,
        point: ScanPoint | None = None,
        profile: LiveProfile | None = None,
    ) -> None:
        self._broker.publish(
            ScanEvent(
                type=event_type,
                scan_id=self._scan_id,
                progress=self._progress.snapshot(),
                point=point,
                profile=profile,
            )
        )

    def _publish_progress(self) -> None:
        if self._progress_throttle.ready():
            self._publish(ScanEventType.PROGRESS)

    def _publish_profile(self, ctx: PointContext) -> None:
        if ctx.buffer.n:
            self._publish(
                ScanEventType.PROFILE,
                profile=ctx.buffer.live(ctx.grid_point, self._calibration),
            )
