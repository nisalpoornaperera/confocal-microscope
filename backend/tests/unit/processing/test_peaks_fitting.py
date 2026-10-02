"""Peak detection, width / asymmetry and the parabolic and Gaussian fits."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.processing import (
    GAUSSIAN_FWHM_PER_SIGMA,
    compute_snr,
    detect_peaks,
    fit_gaussian,
    fit_parabolic,
    relative_prominence,
)


def gaussian(z: np.ndarray, centre: float, fwhm: float, amplitude: float = 1.0) -> np.ndarray:
    sigma = fwhm / GAUSSIAN_FWHM_PER_SIGMA
    return np.asarray(amplitude * np.exp(-0.5 * ((z - centre) / sigma) ** 2), dtype=np.float64)


Z = np.arange(-6.0, 6.0001, 0.25)


def test_detect_single_peak_width_and_symmetry() -> None:
    y = gaussian(Z, 0.4, 3.0)
    result = detect_peaks(Z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=2)
    assert result.significant
    assert result.n_peaks == 1
    assert result.secondary_peak_ratio == 0.0
    assert result.main is not None
    assert result.main.z_refined_um == pytest.approx(0.4, abs=0.02)
    assert result.main.fwhm_um == pytest.approx(3.0, abs=0.05)
    assert result.main.asymmetry == pytest.approx(0.0, abs=0.02)
    assert not result.main.at_edge


def test_detect_secondary_peak_ratio() -> None:
    y = gaussian(Z, -2.5, 1.5) + gaussian(Z, 2.5, 1.5, amplitude=0.5)
    result = detect_peaks(Z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=2)
    assert result.n_peaks == 2
    assert result.main is not None
    assert result.main.z_um == pytest.approx(-2.5)
    assert result.secondary_peak_ratio == pytest.approx(0.5, abs=0.05)


def test_detect_peak_rising_towards_edge() -> None:
    y = gaussian(Z, 7.0, 3.0)
    result = detect_peaks(Z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=2)
    assert result.main is not None
    assert result.main.at_edge
    assert result.main.index == Z.size - 1


def test_flank_cut_by_the_sweep_does_not_reduce_prominence() -> None:
    """The left flank of the highest peak runs into the sweep start at 0.6 (still falling)."""
    z = np.arange(0.0, 20.01, 0.5)
    y = gaussian(z, 1.5, 3.5) + gaussian(z, 14.0, 3.5, amplitude=0.5)
    result = detect_peaks(z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=2)
    assert result.main is not None
    assert result.main.z_um == pytest.approx(1.5)
    assert result.main.prominence == pytest.approx(1.0, abs=0.01)
    assert result.n_peaks == 2
    assert result.secondary_peak_ratio == pytest.approx(0.5, abs=0.02)
    assert not result.global_max_not_selected


def test_flank_reaching_the_baseline_still_bounds_prominence() -> None:
    """A complete flank keeps the conservative 'falls on both sides' rule."""
    y = gaussian(Z, 0.0, 3.0) + 0.5 * (Z > 0.0) * np.exp(-0.5 * ((Z - 6.0) / 0.5) ** 2)
    y = np.where(Z < 0.0, np.maximum(y, 0.4), y)  # left side stops falling at 0.4 (shoulder)
    y[0] = 0.45  # ... and rises again at the sweep start: not a truncated flank
    result = detect_peaks(Z, y, baseline=0.0, min_prominence=0.01, edge_margin_samples=2)
    assert result.main is not None
    assert result.main.z_um == pytest.approx(0.0)
    assert result.main.prominence == pytest.approx(0.6, abs=0.01)


def test_global_maximum_not_selected_is_reported() -> None:
    """Both flanks of the highest peak are cut above the baseline; a lower peak is complete."""
    y = np.array([0.8, 0.9, 1.0, 0.9, 0.6, 0.3, 0.6, 0.9, 0.6, 0.3, 0.2, 0.1])
    z = np.arange(y.size, dtype=np.float64)
    result = detect_peaks(z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=0)
    assert result.main is not None
    assert result.main.index == 7
    assert result.global_max_not_selected


def test_detect_constant_and_short_profiles_have_no_peak() -> None:
    for z, y in ((Z, np.ones_like(Z)), (Z[:2], np.array([0.0, 1.0]))):
        result = detect_peaks(z, y, baseline=0.0, min_prominence=0.1, edge_margin_samples=2)
        assert result.main is None
        assert not result.significant


def test_asymmetry_sign_follows_the_tail() -> None:
    sigma = np.where(Z > 0.0, 2.0, 1.0)
    y = np.exp(-0.5 * (Z / sigma) ** 2)
    result = detect_peaks(Z, y, baseline=0.0, min_prominence=0.05, edge_margin_samples=2)
    assert result.main is not None
    assert result.main.asymmetry is not None
    assert result.main.asymmetry > 0.2


@pytest.mark.parametrize(
    ("centre", "fwhm", "noise"),
    [(0.0, 3.0, 0.0), (0.37, 3.0, 0.01), (-1.3, 2.0, 0.02), (1.9, 4.0, 0.03), (0.8, 2.5, 0.05)],
)
def test_gaussian_fit_accuracy(
    centre: float, fwhm: float, noise: float, rng: np.random.Generator
) -> None:
    y = 0.1 + gaussian(Z, centre, fwhm) + rng.normal(0.0, noise, Z.size)
    fit = fit_gaussian(Z, y)
    assert fit.success, fit.message
    assert fit.center_um == pytest.approx(centre, abs=0.02 + 2.0 * noise)
    assert fit.fwhm_um == pytest.approx(fwhm, rel=0.02 + 2.0 * noise)
    assert fit.amplitude == pytest.approx(1.0, abs=0.02 + 3.0 * noise)
    assert fit.offset == pytest.approx(0.1, abs=0.02 + 3.0 * noise)
    assert fit.r_squared is not None
    assert fit.r_squared > 0.9


def test_gaussian_fit_excludes_saturated_samples() -> None:
    y = gaussian(Z, 0.6, 3.0, amplitude=3.0)
    clipped = np.minimum(y, 1.5)
    fit = fit_gaussian(Z, clipped, exclude=y >= 1.5)
    assert fit.success
    assert fit.center_um == pytest.approx(0.6, abs=1e-3)
    assert fit.amplitude == pytest.approx(3.0, rel=1e-3)
    biased = fit_gaussian(Z, clipped)
    assert biased.amplitude is not None
    assert biased.amplitude < 2.0


@pytest.mark.parametrize(
    ("z", "y"),
    [
        (np.array([]), np.array([])),
        (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
        (Z, np.ones_like(Z)),
        (np.zeros(10), np.arange(10.0)),
        (Z, np.full(Z.size, np.nan)),
        (Z, np.linspace(0.0, 1.0, Z.size)),
    ],
)
def test_fits_never_raise_on_bad_data(z: np.ndarray, y: np.ndarray) -> None:
    for fit in (fit_gaussian(z, y), fit_parabolic(z, y)):
        assert not fit.success
        assert fit.message


def test_parabolic_vertex_of_quadratic_and_gaussian() -> None:
    quad = 2.0 - 0.5 * (Z - 0.3) ** 2
    fit = fit_parabolic(Z, quad, baseline=float(quad.min()))
    assert fit.success
    assert fit.center_um == pytest.approx(0.3, abs=1e-9)
    gfit = fit_parabolic(Z, gaussian(Z, -0.45, 3.0), baseline=0.0)
    assert gfit.success
    assert gfit.center_um == pytest.approx(-0.45, abs=0.05)


def test_snr_and_relative_prominence() -> None:
    assert compute_snr(1.1, 0.1, 0.05) == pytest.approx(20.0)
    assert compute_snr(1.0, 0.0, 0.0) is None
    assert compute_snr(None, 0.0, 1.0) is None
    assert relative_prominence(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert relative_prominence(2.0, 1.0, 0.0) == 1.0
    assert relative_prominence(0.5, 0.0, 0.0) is None
