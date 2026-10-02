"""ScanExecutor: the per-point confocal procedure, fixed-Z mode and interruption."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pytest
from tests.unit.scanning.conftest import (
    FakeCoarsePeakFinder,
    FakeController,
    FakeProfileAnalyser,
    FakeSurface,
    InMemoryRepository,
    StoredPoint,
    small_config,
)

from confocal.config import MotionConfig
from confocal.errors import ADCError, MotionError
from confocal.models import (
    AdcGain,
    AdcSamples,
    CalibrationState,
    CoarsePeak,
    PointStatus,
    ProcessingConfig,
    ProfilePhase,
    ScanConfig,
    ScanEventType,
    ScanMode,
)
from confocal.processing import analyse_profile, find_coarse_peak
from confocal.scanning.control import ScanControl, ScanStopRequested
from confocal.scanning.events import ScanEventBroker, Subscription
from confocal.scanning.executor import ADC_SATURATION_FRACTION, ScanExecutor, StateHook
from confocal.scanning.plan import ScanPlan
from confocal.scanning.profile_buffer import CalibrationSnapshot
from confocal.scanning.progress import LiveEventConfig, ProgressTracker
from confocal.scanning.results import (
    FLAG_ABORTED,
    FLAG_ANALYSIS_FAILED,
    FLAG_COARSE_CLIPPED,
    FLAG_COARSE_RETRY,
    FLAG_FINE_CLIPPED,
    FLAG_NO_COARSE_PEAK,
    FLAG_SATURATED,
)

SCAN_ID = "scan-under-test"
FIRST_POINT_POSITIONS = 21 + 17  # full coarse range + fine sweep
LATER_POINT_POSITIONS = 9 + 17  # adaptive coarse range + fine sweep


@dataclass
class Rig:
    executor: ScanExecutor
    controller: FakeController
    repository: InMemoryRepository
    control: ScanControl
    progress: ProgressTracker
    events: Subscription
    analyser: FakeProfileAnalyser
    finder: FakeCoarsePeakFinder

    @property
    def stored(self) -> list[StoredPoint]:
        return self.repository.stored(SCAN_ID)


def make_rig(
    config: ScanConfig,
    *,
    controller: FakeController | None = None,
    calibration: CalibrationState | None = None,
    analyser: FakeProfileAnalyser | None = None,
    finder: FakeCoarsePeakFinder | None = None,
    on_pause: StateHook | None = None,
    on_resume: StateHook | None = None,
    real_physics: bool = False,
) -> Rig:
    """Executor over the fakes; ``real_physics`` injects the production analysis."""
    controller = controller or FakeController()
    repository = InMemoryRepository()
    calibration = calibration or CalibrationState()
    plan = ScanPlan.from_config(config, controller.limits, MotionConfig(), 860)
    plan.validate_limits()
    repository.create_scan(
        scan_id=SCAN_ID,
        config=config,
        total_points=plan.total_points,
        calibration=calibration,
        software_version="test",
        hardware=controller.hardware_info(),
    )
    broker = ScanEventBroker(queue_size=100_000)
    control = ScanControl()
    progress = ProgressTracker(SCAN_ID, plan.total_points)
    finder = finder or FakeCoarsePeakFinder()
    analyser = analyser or FakeProfileAnalyser()
    executor = ScanExecutor(
        scan_id=SCAN_ID,
        plan=plan,
        controller=controller,
        repository=repository,
        broker=broker,
        find_coarse_peak=find_coarse_peak if real_physics else finder,
        analyse_profile=analyse_profile if real_physics else analyser,
        calibration=CalibrationSnapshot.from_state(calibration),
        control=control,
        progress=progress,
        on_pause=on_pause,
        on_resume=on_resume,
        live_events=LiveEventConfig(progress_interval_s=0.0, profile_interval_s=0.0),
    )
    return Rig(
        executor,
        controller,
        repository,
        control,
        progress,
        broker.subscribe(SCAN_ID),
        analyser,
        finder,
    )


def _assert_no_nan(value: object) -> None:
    """API models must never carry NaN/inf (pydantic's JSON mode would hide it as null)."""
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_no_nan(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_no_nan(item)


def _phases(stored: StoredPoint) -> list[int]:
    assert stored.profile is not None
    return [int(p) for p in stored.profile.phase]


# --------------------------------------------------------------------------- confocal
async def test_confocal_scan_measures_the_surface() -> None:
    rig = make_rig(small_config())
    await rig.executor.run()
    surface = rig.controller.surface
    assert len(rig.stored) == 6
    for stored in rig.stored:
        point = stored.point
        assert point.status is PointStatus.VALID
        assert point.surface_z_um == pytest.approx(
            surface.height_um(point.x_um, point.y_um), abs=0.2
        )
        assert point.parabolic_z_um == point.surface_z_um
        assert point.gaussian_z_um is None  # fit not successful
        assert point.fit_residual is None  # NaN from the analyser never reaches the API
        assert point.confidence == 0.9
        assert point.coarse_peak_z_um is not None
        assert "fake" in point.flags
        assert stored.profile is not None
        assert point.n_z_positions == stored.profile.n_positions
        _assert_no_nan(point.model_dump())
    assert [s.point.point_id for s in rig.stored] == list(range(6))
    assert rig.stored[0].point.n_z_positions == FIRST_POINT_POSITIONS
    assert all(s.point.n_z_positions == LATER_POINT_POSITIONS for s in rig.stored[1:])
    assert rig.progress.completed_points == 6
    assert rig.repository.get_scan(SCAN_ID).completed_points == 6


async def test_adaptive_estimates_follow_the_previous_point() -> None:
    rig = make_rig(small_config())
    await rig.executor.run()
    points = [s.point for s in rig.stored]
    assert points[0].z_estimate_um == 0.0  # default: z_center
    for previous, current in itertools.pairwise(points):
        assert current.z_estimate_um == previous.surface_z_um


async def test_profile_layout_phases_and_reported_z() -> None:
    controller = FakeController(z_report_offset_um=0.02)
    rig = make_rig(small_config(settle_time_ms=5.0), controller=controller)
    await rig.executor.run()
    profile = rig.stored[0].profile
    assert profile is not None
    n = FIRST_POINT_POSITIONS
    assert _phases(rig.stored[0]) == [ProfilePhase.COARSE] * 21 + [ProfilePhase.FINE] * 17
    np.testing.assert_allclose(profile.z_um[:21], np.arange(-20.0, 21.0, 2.0))
    np.testing.assert_allclose(profile.z_reported_um - profile.z_um, 0.02)
    assert profile.raw_counts.shape == (n, 4)
    assert profile.raw_counts.dtype == np.int32
    assert profile.voltage_v.shape == (n, 4)
    np.testing.assert_allclose(profile.voltage_agg_v, profile.voltage_v.mean(axis=1))
    assert profile.timestamps.shape == (n,)
    assert profile.normalized is None  # no reference calibration
    assert profile.dark_v is None
    assert profile.calibration_version is None
    assert profile.filtered is not None
    assert np.all(np.isnan(profile.filtered[:21]))
    assert np.all(np.isfinite(profile.filtered[21:]))
    # the analyser received the stage-reported Z of the fine sweep only
    np.testing.assert_array_equal(rig.analyser.calls[0], profile.z_reported_um[21:])
    np.testing.assert_array_equal(rig.finder.calls[0], profile.z_reported_um[:21])
    # fine sweep centred on the coarse peak, increasing, step 0.5
    fine = profile.z_um[21:]
    assert fine[8] == pytest.approx(rig.stored[0].point.coarse_peak_z_um)
    np.testing.assert_allclose(np.diff(fine), 0.5)
    assert controller.settle_waits
    assert set(controller.settle_waits) == {0.005}
    assert rig.stored[0].analysis is not None


async def test_reference_calibration_normalizes_every_position() -> None:
    calibration = CalibrationState(version=3, dark_v=0.01, reference_v=1.01)
    rig = make_rig(small_config(), calibration=calibration)
    await rig.executor.run()
    stored = rig.stored[0]
    profile = stored.profile
    assert profile is not None
    assert profile.normalized is not None
    np.testing.assert_allclose(profile.normalized, (profile.voltage_agg_v - 0.01) / 1.0)
    assert (profile.dark_v, profile.reference_v, profile.calibration_version) == (0.01, 1.01, 3)
    assert stored.analysis is not None
    assert stored.analysis.signal_units == "normalized"


async def test_reference_below_dark_falls_back_to_volts() -> None:
    calibration = CalibrationState(version=1, dark_v=0.5, reference_v=0.2)
    rig = make_rig(small_config(), calibration=calibration)
    await rig.executor.run()
    profile = rig.stored[0].profile
    assert profile is not None
    assert profile.normalized is None
    assert profile.reference_v is None


async def test_narrow_adaptive_miss_is_retried_over_the_full_range() -> None:
    surface = FakeSurface(step_x_um=15.0, step_height_um=20.0)  # x = 20 sits ~20 um higher
    rig = make_rig(small_config(z_range_um=60.0), controller=FakeController(surface=surface))
    await rig.executor.run()
    stepped = rig.stored[2]  # ix 2, iy 0
    assert stepped.point.ix == 2
    assert FLAG_COARSE_RETRY in stepped.point.flags
    assert stepped.point.status is PointStatus.VALID
    assert stepped.point.surface_z_um == pytest.approx(surface.height_um(20.0, 0.0), abs=0.2)
    assert _phases(stepped) == [0] * 9 + [0] * 31 + [1] * 17  # narrow, retry, fine
    assert FLAG_COARSE_RETRY not in rig.stored[1].point.flags


class AmbiguousNarrowFinder(FakeCoarsePeakFinder):
    """Reports every narrow (adaptive) sweep's peak as not the brightest signal."""

    def __call__(
        self,
        z_um: np.ndarray,
        voltage_v: np.ndarray,
        *,
        dark_v: float | None,
        config: ProcessingConfig,
    ) -> CoarsePeak:
        peak = super().__call__(z_um, voltage_v, dark_v=dark_v, config=config)
        narrow = float(z_um[-1] - z_um[0]) < 30.0
        return peak.model_copy(update={"global_max_not_selected": peak.found and narrow})


async def test_ambiguous_adaptive_coarse_peak_is_retried_over_the_full_range() -> None:
    rig = make_rig(small_config(), finder=AmbiguousNarrowFinder())
    await rig.executor.run()
    assert FLAG_COARSE_RETRY not in rig.stored[0].point.flags  # full range: nothing to retry
    for stored in rig.stored[1:]:
        assert FLAG_COARSE_RETRY in stored.point.flags
        assert _phases(stored) == [0] * 9 + [0] * 21 + [1] * 17  # narrow, retry, fine


async def test_sweeps_never_leave_the_validated_envelope() -> None:
    surface = FakeSurface(base_z_um=18.0)  # close to the +20 um envelope edge
    controller = FakeController(surface=surface)
    rig = make_rig(small_config(), controller=controller)
    await rig.executor.run()
    assert max(p.z_um for p in controller.move_log) <= 20.0
    assert min(p.z_um for p in controller.move_log) >= -20.0
    assert FLAG_COARSE_CLIPPED in rig.stored[1].point.flags  # adaptive 16 um around ~18.5
    assert any(FLAG_FINE_CLIPPED in s.point.flags for s in rig.stored)


async def test_no_coarse_peak_skips_the_fine_sweep() -> None:
    controller = FakeController(surface=FakeSurface(peak_v=0.0))
    rig = make_rig(small_config(), controller=controller)
    await rig.executor.run()
    for stored in rig.stored:
        assert stored.point.status is PointStatus.NO_PEAK
        assert FLAG_NO_COARSE_PEAK in stored.point.flags
        assert stored.point.surface_z_um is None
        assert _phases(stored) == [ProfilePhase.COARSE] * 21  # default range, no retry
        assert stored.analysis is not None
        assert stored.analysis.status is PointStatus.NO_PEAK
        assert stored.profile is not None
        assert stored.profile.filtered is None
    assert rig.analyser.calls == []


# --------------------------------------------------------------------------- fixed Z
async def test_fixed_z_measures_one_position_per_point() -> None:
    config = small_config(mode=ScanMode.FIXED_Z, z_center_um=2.0)
    calibration = CalibrationState(version=2, dark_v=0.01)
    rig = make_rig(config, calibration=calibration)
    await rig.executor.run()
    assert rig.controller.moves == 6
    assert all(p.z_um == 2.0 for p in rig.controller.move_log)
    for stored in rig.stored:
        assert stored.point.status is PointStatus.MEASURED
        assert stored.point.n_z_positions == 1
        assert stored.point.surface_z_um is None
        assert stored.analysis is None
        assert _phases(stored) == [ProfilePhase.FIXED]
        profile = stored.profile
        assert profile is not None
        assert stored.point.intensity == pytest.approx(float(profile.voltage_agg_v[0]) - 0.01)
    assert rig.analyser.calls == []
    assert rig.finder.calls == []


# --------------------------------------------------------------------------- saturation
@pytest.mark.parametrize("gain", [AdcGain.G2, AdcGain.G4])
async def test_unset_saturation_analyses_at_the_adc_full_scale(gain: AdcGain) -> None:
    rig = make_rig(small_config(), controller=FakeController(gain=gain))
    await rig.executor.run()
    expected = ADC_SATURATION_FRACTION * gain.full_scale_v
    assert rig.finder.configs
    assert rig.analyser.configs
    for config in [*rig.finder.configs, *rig.analyser.configs]:
        assert config.saturation_v == pytest.approx(expected)
    assert rig.executor._config.processing.saturation_v is None  # the scan config is untouched


async def test_configured_saturation_is_used_unchanged() -> None:
    config = small_config(processing={"saturation_v": 1.2})
    rig = make_rig(config)
    await rig.executor.run()
    assert {c.saturation_v for c in [*rig.finder.configs, *rig.analyser.configs]} == {1.2}


async def test_configured_saturation_above_the_adc_rail_is_capped() -> None:
    """A 1.95 V detector rail must not hide clipping at gain 4 (full scale 1.024 V)."""
    config = small_config(processing={"saturation_v": 1.95})
    rig = make_rig(config, controller=FakeController(gain=AdcGain.G4))
    await rig.executor.run()
    expected = ADC_SATURATION_FRACTION * AdcGain.G4.full_scale_v
    configs = [*rig.finder.configs, *rig.analyser.configs]
    assert configs
    assert all(c.saturation_v == pytest.approx(expected) for c in configs)
    assert rig.executor._config.processing.saturation_v == 1.95


async def test_rail_codes_flag_the_point_saturated() -> None:
    controller = FakeController(surface=FakeSurface(peak_v=3.0))  # > 2.048 V full scale
    rig = make_rig(small_config(), controller=controller)
    await rig.executor.run()
    for stored in rig.stored:
        assert stored.profile is not None
        assert int(stored.profile.raw_counts.max()) == 32767
        assert FLAG_SATURATED in stored.point.flags
        assert stored.point.flags.count(FLAG_SATURATED) == 1
    unsaturated = make_rig(small_config())
    await unsaturated.executor.run()
    assert all(FLAG_SATURATED not in s.point.flags for s in unsaturated.stored)


async def test_clipped_peak_is_recognised_by_the_real_analysis() -> None:
    """The ADC clips the focus peak; no saturation_v is configured anywhere."""
    surface = FakeSurface(peak_v=3.0, fwhm_um=4.0)  # clips at 2.048 V (gain 2)
    controller = FakeController(surface=surface, noise_v=0.002)
    rig = make_rig(small_config(fine_z_range_um=16.0), controller=controller, real_physics=True)
    await rig.executor.run()
    for stored in rig.stored:
        point, analysis = stored.point, stored.analysis
        assert analysis is not None
        assert analysis.saturated_fraction > 0.0
        assert "saturated" in analysis.flags
        assert FLAG_SATURATED in point.flags
        assert point.surface_z_um is not None
        assert point.surface_z_um == pytest.approx(
            surface.height_um(point.x_um, point.y_um), abs=0.3
        )


# --------------------------------------------------------------------------- failures
async def test_analysis_failure_stores_an_error_point_and_continues() -> None:
    rig = make_rig(small_config(), analyser=FakeProfileAnalyser(fail_on_call=2))
    await rig.executor.run()
    failed = rig.stored[1]
    assert failed.point.status is PointStatus.ERROR
    assert FLAG_ANALYSIS_FAILED in failed.point.flags
    assert failed.profile is not None
    assert failed.profile.n_positions == LATER_POINT_POSITIONS  # raw data kept
    assert failed.analysis is None
    assert all(s.point.status is PointStatus.VALID for i, s in enumerate(rig.stored) if i != 1)
    assert rig.progress.completed_points == 6


async def test_cancel_aborts_at_the_next_z_step_and_keeps_partial_data() -> None:
    rig = make_rig(small_config())

    async def cancel() -> None:
        rig.control.request_cancel()

    rig.controller.move_hooks[30] = cancel  # move 1 = XY, 2..22 coarse, 23..39 fine
    with pytest.raises(ScanStopRequested):
        await rig.executor.run()
    assert len(rig.stored) == 1
    aborted = rig.stored[0]
    assert aborted.point.status is PointStatus.ABORTED
    assert FLAG_ABORTED in aborted.point.flags
    assert aborted.point.n_z_positions == 29
    assert aborted.profile is not None
    assert aborted.profile.n_positions == 29
    assert _phases(aborted) == [0] * 21 + [1] * 8
    assert aborted.analysis is None
    assert rig.controller.moves == 30  # nothing commanded after the checkpoint
    assert rig.progress.completed_points == 0


async def test_hardware_error_stores_the_partial_point_and_propagates() -> None:
    rig = make_rig(small_config(), controller=FakeController(fail_at_move=5))
    with pytest.raises(MotionError):
        await rig.executor.run()
    assert [s.point.status for s in rig.stored] == [PointStatus.ABORTED]
    assert rig.stored[0].point.n_z_positions == 3


async def test_wrong_sample_count_is_an_adc_failure() -> None:
    controller = FakeController()
    original = controller.acquire

    async def short_burst(n_samples: int) -> AdcSamples:
        return await original(n_samples - 1)

    controller.acquire = short_burst  # type: ignore[method-assign]
    rig = make_rig(small_config(), controller=controller)
    with pytest.raises(ADCError):
        await rig.executor.run()
    assert rig.stored[0].point.status is PointStatus.ABORTED
    assert rig.stored[0].profile is None  # no complete position was acquired


async def test_pause_takes_effect_between_points() -> None:
    log: list[str] = []
    rig_holder: list[Rig] = []

    async def on_pause() -> None:
        log.append(f"pause after {len(rig_holder[0].stored)} points")
        rig_holder[0].control.request_resume()

    async def on_resume() -> None:
        log.append("resume")

    rig = make_rig(small_config(), on_pause=on_pause, on_resume=on_resume)
    rig_holder.append(rig)

    async def pause() -> None:
        rig.control.request_pause()

    rig.controller.move_hooks[3] = pause
    await rig.executor.run()
    assert log == ["pause after 1 points", "resume"]
    assert len(rig.stored) == 6


# --------------------------------------------------------------------------- events
async def test_point_profile_and_progress_events_are_published() -> None:
    rig = make_rig(small_config())
    await rig.executor.run()
    events = []
    while (event := rig.events.get_nowait()) is not None:
        events.append(event)
    points = [e for e in events if e.type is ScanEventType.POINT]
    profiles = [e for e in events if e.type is ScanEventType.PROFILE]
    assert [e.point.point_id for e in points if e.point is not None] == list(range(6))
    assert points[-1].progress.completed_points == 6
    assert points[-1].progress.progress == 1.0
    assert any(e.type is ScanEventType.PROGRESS for e in events)
    final_first = [e.profile for e in profiles if e.profile is not None and e.profile.point_id == 0]
    assert len(final_first[-1].z_um) == FIRST_POINT_POSITIONS
    assert final_first[-1].phase[-1] == ProfilePhase.FINE
    assert all(v is not None for v in final_first[-1].intensity)
    for event in events:
        _assert_no_nan(event.model_dump())
    assert points[1].progress.estimated_remaining_s is not None
