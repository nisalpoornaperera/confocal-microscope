"""ProcessingConfig.peak_selection: most prominent vs highest peak."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.models.processing import PointStatus, ProcessingConfig
from confocal.processing import analyse_profile, find_coarse_peak

Z = np.linspace(-30.0, 30.0, 121)
HIGHEST = ProcessingConfig(peak_selection="highest")


def _gauss(center: float, height: float, fwhm: float) -> np.ndarray:
    return np.asarray(height * np.exp(-4 * np.log(2) * (Z - center) ** 2 / fwhm**2))


def _two_peaks(seed: int = 3) -> np.ndarray:
    """Main peak at -10 um and a second, 80 % high peak at +12 um (e.g. a second reflection)."""
    rng = np.random.default_rng(seed)
    v = 0.05 + _gauss(-10.0, 1.0, 5.0) + _gauss(12.0, 0.8, 5.0) + rng.normal(0.0, 0.003, Z.size)
    return np.asarray(v, dtype=np.float64)


def _analyse(v: np.ndarray, config: ProcessingConfig):
    return analyse_profile(Z, v, dark_v=0.0, reference_v=None, config=config).analysis


def test_highest_selection_takes_the_global_maximum_without_a_second_peak_penalty() -> None:
    v = _two_peaks()
    analysis = _analyse(v, HIGHEST)
    assert analysis.surface_z_um == pytest.approx(-10.0, abs=0.3)
    assert analysis.status is PointStatus.VALID
    assert analysis.confidence > 0.9
    assert "multiple_peaks" in analysis.flags  # still reported, just not penalised


def test_default_selection_still_penalises_a_strong_second_peak() -> None:
    analysis = _analyse(_two_peaks(), ProcessingConfig())
    assert analysis.surface_z_um == pytest.approx(-10.0, abs=0.3)
    assert analysis.status is PointStatus.LOW_CONFIDENCE
    assert analysis.confidence < 0.5


def test_highest_follows_the_brighter_peak_when_it_moves() -> None:
    rng = np.random.default_rng(5)
    v = 0.05 + _gauss(-10.0, 0.7, 5.0) + _gauss(12.0, 1.0, 5.0) + rng.normal(0.0, 0.003, Z.size)
    assert _analyse(v, HIGHEST).surface_z_um == pytest.approx(12.0, abs=0.3)


def test_single_peak_gives_the_same_answer_either_way() -> None:
    rng = np.random.default_rng(1)
    v = 0.05 + _gauss(2.0, 1.0, 6.0) + rng.normal(0.0, 0.003, Z.size)
    a = _analyse(v, ProcessingConfig())
    b = _analyse(v, HIGHEST)
    assert a.status is b.status is PointStatus.VALID
    assert a.surface_z_um == pytest.approx(b.surface_z_um)
    assert a.confidence == pytest.approx(b.confidence)


def test_coarse_finder_uses_the_selection() -> None:
    peak = find_coarse_peak(Z, _two_peaks(), dark_v=0.0, config=HIGHEST)
    assert peak.found
    assert peak.z_um == pytest.approx(-10.0, abs=1.0)
