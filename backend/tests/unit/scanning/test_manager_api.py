"""ScanManager queries, control-request validation and on-demand reconstruction / ML."""

from __future__ import annotations

import asyncio

import pytest
from tests.unit.scanning.conftest import (
    FAKE_LIMITS,
    Bench,
    FakeCoarsePeakFinder,
    FakeMLAnalyser,
    MoveGate,
    make_bench,
    small_config,
)

from confocal.config import MotionConfig
from confocal.errors import (
    ModelNotAvailableError,
    ReconstructionError,
    ScanConflictError,
    ScanNotFoundError,
    ScanStateError,
)
from confocal.models import (
    InterpolationMethod,
    MLAnalysisRequest,
    ReconstructionRequest,
    ScanMode,
    ScanState,
)
from confocal.scanning.plan import ScanPlan

S = ScanState


async def _finished_scan(bench: Bench, **overrides: object) -> str:
    created = await bench.manager.create_scan(small_config(**overrides))
    await bench.manager.wait_until_finished(created.id, timeout=5.0)
    return created.id


async def test_estimate_uses_the_adc_data_rate_and_creates_nothing() -> None:
    bench = make_bench()
    config = small_config()
    estimate = await bench.manager.estimate(config)
    expected = ScanPlan.from_config(config, FAKE_LIMITS, MotionConfig(), 860).estimate()
    assert estimate == expected
    assert bench.repository.scans == {}


async def test_settings_detector_rail_is_the_default_saturation_of_a_scan() -> None:
    bench = make_bench(default_saturation_v=1.95)
    finder = bench.manager._find_coarse_peak
    assert isinstance(finder, FakeCoarsePeakFinder)
    scan_id = await _finished_scan(bench)
    stored = bench.repository.scans[scan_id].config.processing
    assert stored.saturation_v == 1.95  # persisted: the scan is reproducible
    assert finder.configs
    assert {c.saturation_v for c in finder.configs} == {1.95}


async def test_scan_saturation_overrides_the_settings_default() -> None:
    bench = make_bench(default_saturation_v=1.95)
    scan_id = await _finished_scan(bench, processing={"saturation_v": 1.5})
    assert bench.repository.scans[scan_id].config.processing.saturation_v == 1.5
    plain = make_bench()
    scan_id = await _finished_scan(plain)
    assert plain.repository.scans[scan_id].config.processing.saturation_v is None


def test_invalid_default_saturation_is_refused() -> None:
    with pytest.raises(ValueError, match="default_saturation_v"):
        make_bench(default_saturation_v=0.0)


async def test_queries_of_a_finished_scan() -> None:
    bench = make_bench()
    scan_id = await _finished_scan(bench)
    progress = await bench.manager.get_progress(scan_id)
    assert progress.state is S.COMPLETE
    assert progress.progress == 1.0
    assert progress.completed_points == progress.total_points == 6
    listed = await bench.manager.list_scans()
    assert [s.id for s in listed] == [scan_id]
    with pytest.raises(ScanNotFoundError):
        await bench.manager.get_scan("missing")
    with pytest.raises(ScanNotFoundError):
        await bench.manager.get_progress("missing")


async def test_wait_until_finished_times_out_while_the_scan_runs() -> None:
    bench = make_bench()
    gate = MoveGate()
    bench.controller.move_hooks[2] = gate
    created = await bench.manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    with pytest.raises(TimeoutError):
        await bench.manager.wait_until_finished(created.id, timeout=0.01)
    gate.release()
    assert (await bench.manager.wait_until_finished(created.id, timeout=5.0)).state is S.COMPLETE


async def test_control_requests_are_validated() -> None:
    bench = make_bench()
    manager = bench.manager
    for request in (manager.pause, manager.resume, manager.cancel):
        with pytest.raises(ScanNotFoundError):
            await request("missing")
    scan_id = await _finished_scan(bench)
    for request in (manager.pause, manager.resume, manager.cancel):
        with pytest.raises(ScanStateError, match="not active"):
            await request(scan_id)

    gate = MoveGate()
    bench.controller.move_hooks[bench.controller.moves + 2] = gate
    created = await manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    with pytest.raises(ScanStateError, match="not paused"):
        await manager.resume(created.id)
    await manager.cancel(created.id)
    gate.release()
    assert (await manager.wait_until_finished(created.id, timeout=5.0)).state is S.CANCELLED


async def test_subscribing_to_a_finished_scan_ends_immediately() -> None:
    bench = make_bench()
    scan_id = await _finished_scan(bench)
    subscription = await bench.manager.subscribe(scan_id)
    assert subscription.finished
    assert [e async for e in subscription] == []
    with pytest.raises(ScanNotFoundError):
        await bench.manager.subscribe("missing")


async def test_reconstruct_on_demand_saves_a_new_surface() -> None:
    bench = make_bench()
    scan_id = await _finished_scan(bench)
    request = ReconstructionRequest(method=InterpolationMethod.NEAREST)
    surface = await bench.manager.reconstruct(scan_id, request)
    assert surface.surface_id == 2  # the scan's automatic reconstruction was the first
    assert surface.request.method is InterpolationMethod.NEAREST
    assert bench.reconstructor.calls[-1] == (scan_id, 6, 10.0)
    default = await bench.manager.reconstruct(scan_id)
    assert default.request == small_config().reconstruction


async def test_reconstruct_refuses_active_fixed_z_and_unknown_scans() -> None:
    bench = make_bench()
    fixed = await _finished_scan(bench, mode=ScanMode.FIXED_Z)
    with pytest.raises(ReconstructionError, match="fixed-Z"):
        await bench.manager.reconstruct(fixed)
    with pytest.raises(ScanNotFoundError):
        await bench.manager.reconstruct("missing")
    gate = MoveGate()
    bench.controller.move_hooks[bench.controller.moves + 2] = gate
    created = await bench.manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    with pytest.raises(ScanConflictError):
        await bench.manager.reconstruct(created.id)
    with pytest.raises(ScanConflictError):
        await bench.manager.analyse_ml(created.id, MLAnalysisRequest())
    gate.release()
    await bench.manager.wait_until_finished(created.id, timeout=5.0)


async def test_analyse_ml_on_demand() -> None:
    without = make_bench()
    scan_id = await _finished_scan(without)
    with pytest.raises(ModelNotAvailableError):
        await without.manager.analyse_ml(scan_id, MLAnalysisRequest())

    undeployed = make_bench(ml=FakeMLAnalyser(available=False))
    scan_id = await _finished_scan(undeployed)
    with pytest.raises(ModelNotAvailableError):
        await undeployed.manager.analyse_ml(scan_id, MLAnalysisRequest(model_name="rf"))

    deployed = make_bench(ml=FakeMLAnalyser(available=True))
    scan_id = await _finished_scan(deployed)
    result = await deployed.manager.analyse_ml(scan_id, MLAnalysisRequest(threshold=0.7))
    assert result.result_id == 1
    assert result.n_points == 6
    assert result.threshold == 0.7
    assert (await deployed.manager.get_scan(scan_id)).has_ml_result
    with pytest.raises(ScanNotFoundError):
        await deployed.manager.analyse_ml("missing", MLAnalysisRequest())
