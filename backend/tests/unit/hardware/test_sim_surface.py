"""SimulatedConfocalSurface: sample model, axial response and deterministic defects."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.config import SimulatedSurfaceConfig, SimulationConfig
from confocal.hardware.simulation import (
    LOW_REFLECTIVITY,
    SPURIOUS_AMPLITUDE,
    PsfModel,
    SimulatedConfocalSurface,
)


def _surface(**overrides: object) -> SimulatedConfocalSurface:
    cfg = SimulatedSurfaceConfig.model_validate(
        {"low_reflectivity_fraction": 0.0, "spurious_peak_probability": 0.0, **overrides}
    )
    return SimulatedConfocalSurface(cfg, axial_fwhm_um=6.0, peak_voltage_v=2.0, seed=7)


def test_from_config_copies_optics() -> None:
    config = SimulationConfig(psf_model="sinc2", axial_fwhm_um=4.0, peak_voltage_v=1.5, seed=3)
    surface = SimulatedConfocalSurface.from_config(config)
    assert surface.psf_model == "sinc2"
    assert surface.axial_fwhm_um == 4.0
    assert surface.peak_voltage_v == 1.5
    assert surface.seed == 3
    assert surface.config == config.surface


def test_plane_height_is_the_tilted_plane() -> None:
    surface = _surface(kind="plane", base_z_um=2.0, tilt_x=0.01, tilt_y=-0.02)
    assert float(surface.height_um(100.0, 50.0)) == pytest.approx(2.0 + 1.0 - 1.0)


@pytest.mark.parametrize("kind", ["plane", "sinusoid", "steps", "sphere", "composite"])
def test_height_is_vectorised_and_finite(kind: str) -> None:
    surface = _surface(kind=kind)
    x = np.linspace(-500.0, 500.0, 11)
    heights = surface.height_um(x[:, None], x[None, :])
    assert heights.shape == (11, 11)
    assert np.all(np.isfinite(heights))
    assert float(surface.height_um(x[3], x[4])) == pytest.approx(heights[3, 4])


def test_steps_and_sphere_features() -> None:
    steps = _surface(kind="steps", tilt_x=0.0, tilt_y=0.0, feature_period_um=100.0)
    assert float(steps.height_um(150.0, 0.0)) == pytest.approx(3.0)
    sphere = _surface(kind="sphere", tilt_x=0.0, tilt_y=0.0, sphere_height_um=8.0)
    assert float(sphere.height_um(0.0, 0.0)) == pytest.approx(8.0)
    assert float(sphere.height_um(1000.0, 0.0)) == 0.0


@pytest.mark.parametrize("model", ["gaussian", "sinc2"])
def test_axial_response_has_the_configured_fwhm(model: PsfModel) -> None:
    surface = SimulatedConfocalSurface(_surface().config, psf_model=model, axial_fwhm_um=6.0)
    assert float(surface.axial_response(0.0)) == pytest.approx(1.0)
    np.testing.assert_allclose(surface.axial_response([-3.0, 3.0]), [0.5, 0.5], rtol=1e-9)
    assert float(surface.axial_response(10.0)) < 0.5


def test_signal_peaks_at_the_true_height() -> None:
    surface = _surface(kind="composite")
    height = float(surface.height_um(37.0, -81.0))
    z = np.linspace(height - 20.0, height + 20.0, 4001)
    signal = surface.signal_v(37.0, -81.0, z)
    assert z[int(np.argmax(signal))] == pytest.approx(height, abs=0.01)
    assert float(np.max(signal)) == pytest.approx(0.9 * 2.0, rel=1e-6)


def test_ground_truth_matches_the_model() -> None:
    surface = _surface(kind="composite", low_reflectivity_fraction=0.3)
    x = np.arange(-300.0, 300.0, 25.0)
    truth = surface.ground_truth(x, 10.0)
    np.testing.assert_array_equal(truth.height_um, surface.height_um(x, 10.0))
    np.testing.assert_array_equal(truth.reflectivity, surface.reflectivity(x, 10.0))
    assert truth.x_um.shape == truth.y_um.shape == x.shape
    np.testing.assert_array_equal(truth.low_reflectivity, truth.reflectivity == LOW_REFLECTIVITY)
    assert not truth.spurious_peak.any()
    assert np.isnan(truth.spurious_offset_um).all()


def test_low_reflectivity_area_fraction() -> None:
    surface = _surface(low_reflectivity_fraction=0.25)
    grid = np.arange(-5000.0, 5000.0, 47.0)
    dark = surface.ground_truth(grid[:, None], grid[None, :]).low_reflectivity
    assert dark.mean() == pytest.approx(0.25, abs=0.03)


def test_spurious_peaks_add_a_weaker_secondary_peak() -> None:
    surface = _surface(kind="plane", tilt_x=0.0, tilt_y=0.0, spurious_peak_probability=1.0)
    truth = surface.ground_truth(5.0, 5.0)
    assert bool(truth.spurious_peak)
    offset = float(truth.spurious_offset_um)
    assert 2.5 * 6.0 <= abs(offset) <= 4.0 * 6.0
    secondary = float(surface.signal_v(5.0, 5.0, offset))
    main = float(surface.signal_v(5.0, 5.0, 0.0))
    assert secondary == pytest.approx(SPURIOUS_AMPLITUDE * main, rel=0.01)


def test_defects_are_deterministic_per_seed() -> None:
    cfg = SimulatedSurfaceConfig(low_reflectivity_fraction=0.3, spurious_peak_probability=0.3)
    grid = np.arange(-1000.0, 1000.0, 13.0)
    first = SimulatedConfocalSurface(cfg, seed=1).ground_truth(grid[:, None], grid[None, :])
    again = SimulatedConfocalSurface(cfg, seed=1).ground_truth(grid[:, None], grid[None, :])
    other = SimulatedConfocalSurface(cfg, seed=2).ground_truth(grid[:, None], grid[None, :])
    np.testing.assert_array_equal(first.low_reflectivity, again.low_reflectivity)
    np.testing.assert_array_equal(first.spurious_peak, again.spurious_peak)
    assert not np.array_equal(first.low_reflectivity, other.low_reflectivity)


def test_invalid_optics_are_rejected() -> None:
    with pytest.raises(ValueError, match="fwhm"):
        SimulatedConfocalSurface(axial_fwhm_um=0.0)
    with pytest.raises(ValueError, match="peak_voltage"):
        SimulatedConfocalSurface(peak_voltage_v=float("inf"))
    with pytest.raises(ValueError, match="psf_model"):
        SimulatedConfocalSurface(psf_model="airy")  # type: ignore[arg-type]
