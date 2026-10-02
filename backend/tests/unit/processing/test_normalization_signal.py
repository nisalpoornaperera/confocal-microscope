"""Dark subtraction, normalization, aggregation, filtering and baseline / noise."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.errors import CalibrationError
from confocal.models.hardware import SamplingMethod
from confocal.models.processing import FilterMethod, ProcessingConfig
from confocal.processing import (
    aggregate_samples,
    estimate_baseline_and_noise,
    filter_signal,
    normalize,
    subtract_dark,
)


def test_subtract_dark_with_and_without_calibration() -> None:
    v = np.array([0.1, 0.5, 1.0])
    np.testing.assert_allclose(subtract_dark(v, 0.1), [0.0, 0.4, 0.9])
    np.testing.assert_allclose(subtract_dark(v, None), v)


def test_normalize_maps_dark_to_zero_and_reference_to_one() -> None:
    out = normalize(np.array([0.1, 0.6, 1.1]), 0.1, 1.1)
    assert out is not None
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_normalize_without_reference_is_none() -> None:
    assert normalize(np.array([0.1, 0.2]), 0.0, None) is None


@pytest.mark.parametrize(("dark", "reference"), [(0.5, 0.5), (0.5, 0.2), (None, 0.0)])
def test_normalize_rejects_reference_not_above_dark(dark: float | None, reference: float) -> None:
    with pytest.raises(CalibrationError):
        normalize(np.array([0.1]), dark, reference)


def test_aggregate_mean_versus_median() -> None:
    samples = np.array([[1.0, 1.0, 1.0, 9.0], [2.0, 2.0, 2.0, 2.0]])
    np.testing.assert_allclose(aggregate_samples(samples, SamplingMethod.MEAN), [3.0, 2.0])
    np.testing.assert_allclose(aggregate_samples(samples, SamplingMethod.MEDIAN), [1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        aggregate_samples(np.array([]), SamplingMethod.MEAN)


@pytest.mark.parametrize("method", list(FilterMethod))
def test_filters_preserve_length_and_reduce_noise(
    method: FilterMethod, rng: np.random.Generator
) -> None:
    z = np.linspace(-6.0, 6.0, 49)
    clean = np.exp(-0.5 * (z / 1.5) ** 2)
    noisy = clean + rng.normal(0.0, 0.05, z.size)
    out = filter_signal(noisy, ProcessingConfig(filter_method=method))
    assert out.shape == noisy.shape
    err_in = np.sqrt(np.mean((noisy - clean) ** 2))
    err_out = np.sqrt(np.mean((out - clean) ** 2))
    if method is FilterMethod.NONE:
        np.testing.assert_array_equal(out, noisy)
    else:
        assert err_out < err_in


def test_filter_handles_short_and_non_finite_input() -> None:
    config = ProcessingConfig(filter_window=11)
    np.testing.assert_array_equal(filter_signal(np.array([1.0, 2.0]), config), [1.0, 2.0])
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    out = filter_signal(y, config)
    assert np.isnan(out[2])
    assert np.isfinite(np.delete(out, 2)).all()


def test_baseline_and_noise_are_robust() -> None:
    # The MAD of ~120 second differences has a ~12 % relative spread.
    rng = np.random.default_rng(5)
    z = np.linspace(-10.0, 10.0, 201)
    sigma = 0.02
    y = 0.3 + 2.0 * np.exp(-0.5 * (z / 1.0) ** 2) + rng.normal(0.0, sigma, z.size)
    baseline, noise = estimate_baseline_and_noise(y, exclude=np.abs(z) < 4.0)
    assert baseline == pytest.approx(0.3, abs=0.01)
    assert noise == pytest.approx(sigma, rel=0.3)


def test_baseline_noise_floor_and_errors() -> None:
    baseline, noise = estimate_baseline_and_noise(np.full(20, 0.5))
    assert baseline == 0.5
    assert noise > 0.0
    with pytest.raises(ValueError, match="finite"):
        estimate_baseline_and_noise(np.array([np.nan, np.nan]))
