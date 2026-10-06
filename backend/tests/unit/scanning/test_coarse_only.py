"""Coarse-only confocal scanning (``ScanConfig.fine_scan = False``)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from tests.unit.scanning.conftest import FakeController, small_config
from tests.unit.scanning.test_executor import LATER_POINT_POSITIONS, make_rig

from confocal.config import MotionConfig
from confocal.models import PointStatus, ProfilePhase
from confocal.scanning.plan import ScanPlan

COARSE_FIRST = 21  # full coarse range of small_config
COARSE_LATER = 9  # adaptive coarse range of small_config


async def test_coarse_only_scan_skips_the_fine_sweep_and_analyses_the_coarse_one() -> None:
    rig = make_rig(small_config(fine_scan=False))
    await rig.executor.run()
    assert len(rig.stored) == 6
    first = rig.stored[0]
    assert first.profile is not None
    assert [int(p) for p in first.profile.phase] == [ProfilePhase.COARSE] * COARSE_FIRST
    assert first.point.n_z_positions == COARSE_FIRST
    assert all(s.point.n_z_positions == COARSE_LATER for s in rig.stored[1:])
    # the analyser received the stage-reported Z of the coarse sweep
    np.testing.assert_array_equal(rig.analyser.calls[0], first.profile.z_reported_um)
    assert first.profile.filtered is not None
    assert np.all(np.isfinite(first.profile.filtered))
    assert all(s.point.status is PointStatus.VALID for s in rig.stored)


async def test_coarse_only_with_real_physics_finds_the_surface() -> None:
    rig = make_rig(small_config(fine_scan=False, coarse_z_step_um=1.0), real_physics=True)
    await rig.executor.run()
    surface = rig.controller.surface
    found = [s.point for s in rig.stored if s.point.surface_z_um is not None]
    assert len(found) >= 5
    for point in found:
        assert point.surface_z_um == pytest.approx(
            surface.height_um(point.x_um, point.y_um), abs=1.0
        )


def test_fine_validators_only_apply_with_the_fine_scan() -> None:
    # fine range smaller than 2 coarse steps and fine step larger than the coarse step
    bad_fine = {"fine_z_range_um": 1.0, "fine_z_step_um": 5.0}
    with pytest.raises(ValidationError):
        small_config(**bad_fine)
    assert small_config(fine_scan=False, **bad_fine).fine_scan is False


def test_estimate_counts_only_coarse_positions_without_the_fine_scan() -> None:
    limits = FakeController().limits
    with_fine = ScanPlan.from_config(small_config(), limits, MotionConfig(), 860).estimate()
    coarse = ScanPlan.from_config(small_config(fine_scan=False), limits, MotionConfig(), 860)
    estimate = coarse.estimate()
    # typical (adaptive) point: coarse only vs coarse + fine
    assert estimate.z_positions_per_point == COARSE_LATER
    assert with_fine.z_positions_per_point == LATER_POINT_POSITIONS
    assert estimate.total_measurements < with_fine.total_measurements
    assert estimate.estimated_duration_s < with_fine.estimated_duration_s
