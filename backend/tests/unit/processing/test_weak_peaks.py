"""Low-contrast mode (``ProcessingConfig.accept_weak_peaks``)."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.models.processing import PointStatus, ProcessingConfig
from confocal.processing import analyse_profile, find_coarse_peak

Z = np.linspace(-15.0, 15.0, 61)
TRUE_Z = 1.0
NOISE = 0.01
STANDARD = ProcessingConfig()
SENSITIVE = ProcessingConfig(
    accept_weak_peaks=True, min_snr=2.0, min_relative_prominence=0.05, min_confidence=0.2
)


def _weak_profile(seed: int, amplitude: float = 3 * NOISE) -> np.ndarray:
    """A peak only ~3 noise standard deviations high (below the 4-sigma rule)."""
    rng = np.random.default_rng(seed)
    peak = amplitude * np.exp(-4 * np.log(2) * (Z - TRUE_Z) ** 2 / 6.0**2)
    return np.asarray(0.1 + peak + rng.normal(0.0, NOISE, Z.size), dtype=np.float64)


def _analyse(v: np.ndarray, config: ProcessingConfig):
    return analyse_profile(Z, v, dark_v=0.0, reference_v=None, config=config).analysis


def test_weak_peaks_are_rejected_by_default() -> None:
    statuses = [_analyse(_weak_profile(seed), STANDARD).status for seed in range(20)]
    assert statuses.count(PointStatus.NO_PEAK) >= 15


def test_sensitive_mode_recovers_weak_peaks_as_low_confidence() -> None:
    recovered = []
    for seed in range(20):
        analysis = _analyse(_weak_profile(seed), SENSITIVE)
        assert analysis.status is not PointStatus.VALID  # weak peaks never become VALID
        if "weak_peak" in analysis.flags:
            assert analysis.status is not PointStatus.VALID
        if analysis.status is PointStatus.LOW_CONFIDENCE:
            assert analysis.surface_z_um is not None
            recovered.append(abs(analysis.surface_z_um - TRUE_Z))
    assert len(recovered) >= 10
    weak = sum("weak_peak" in _analyse(_weak_profile(s), SENSITIVE).flags for s in range(20))
    assert weak >= 5  # the relaxed rule is what recovers these points
    assert float(np.median(recovered)) < 1.0


def test_sensitive_mode_still_rejects_pure_noise() -> None:
    for seed in range(10):
        flat = 0.1 + np.random.default_rng(seed).normal(0.0, NOISE, Z.size)
        analysis = _analyse(flat, SENSITIVE)
        assert analysis.status in {
            PointStatus.NO_PEAK,
            PointStatus.FIT_FAILED,
            PointStatus.LOW_CONFIDENCE,
            PointStatus.PEAK_AT_EDGE,
        }
        assert analysis.status is not PointStatus.VALID


def test_sensitive_mode_leaves_strong_peaks_valid() -> None:
    analysis = _analyse(_weak_profile(0, amplitude=0.5), SENSITIVE)
    assert analysis.status is PointStatus.VALID
    assert "weak_peak" not in analysis.flags
    assert analysis.surface_z_um == pytest.approx(TRUE_Z, abs=0.2)


def test_coarse_peak_finder_accepts_weak_peaks_only_in_sensitive_mode() -> None:
    found_standard = found_sensitive = 0
    for seed in range(20):
        v = _weak_profile(seed)
        found_standard += find_coarse_peak(Z, v, dark_v=0.0, config=STANDARD).found
        peak = find_coarse_peak(Z, v, dark_v=0.0, config=SENSITIVE)
        if peak.found:
            found_sensitive += 1
            assert peak.z_um is not None
            assert abs(peak.z_um - TRUE_Z) < 5.0
    assert found_sensitive > found_standard
    assert found_sensitive >= 12
