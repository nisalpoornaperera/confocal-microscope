"""analyse_profile / find_coarse_peak: status precedence, surface Z choice, robustness."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from confocal.errors import CalibrationError
from confocal.models.processing import FitMethod, PeakFit, PointStatus, ProcessingConfig
from confocal.processing import (
    GAUSSIAN_FWHM_PER_SIGMA,
    PeakDetection,
    PeakQuality,
    ProfileFlag,
    analyse_profile,
    confidence_score,
    find_coarse_peak,
)
from confocal.processing import profile as profile_module

Z = np.arange(-6.0, 6.0001, 0.25)
CONFIG = ProcessingConfig()


def signal_v(
    z: np.ndarray, centre: float, *, amplitude: float = 1.0, fwhm: float = 2.8
) -> np.ndarray:
    sigma = fwhm / GAUSSIAN_FWHM_PER_SIGMA
    return np.asarray(0.05 + amplitude * np.exp(-0.5 * ((z - centre) / sigma) ** 2))


def analyse(z: np.ndarray, v: np.ndarray, config: ProcessingConfig = CONFIG, **kw: float | None):
    return analyse_profile(
        z, v, dark_v=kw.get("dark_v"), reference_v=kw.get("reference_v"), config=config
    )


# --------------------------------------------------------------------------- happy path


@pytest.mark.parametrize("centre", [-2.0, -0.3, 0.0, 0.77, 2.1])
def test_valid_profile_uses_gaussian_centre(centre: float, rng: np.random.Generator) -> None:
    v = signal_v(Z, centre) + rng.normal(0.0, 0.01, Z.size)
    result = analyse(Z, v, dark_v=0.02, reference_v=1.2)
    a = result.analysis
    assert a.status is PointStatus.VALID
    assert a.signal_units == "normalized"
    assert a.surface_method is FitMethod.GAUSSIAN
    assert a.surface_z_um == pytest.approx(centre, abs=0.05)
    assert a.fwhm_um == pytest.approx(2.8, rel=0.1)
    assert a.snr is not None
    assert a.snr > 20.0
    assert 0.8 < a.confidence <= 1.0
    assert a.n_peaks == 1
    assert result.normalized is not None
    np.testing.assert_allclose(result.corrected_v, v - 0.02)
    np.testing.assert_allclose(result.normalized, (v - 0.02) / 1.18)


def test_volts_without_reference() -> None:
    result = analyse(Z, signal_v(Z, 0.0))
    assert result.analysis.signal_units == "volts"
    assert result.normalized is None
    assert result.analysis.status is PointStatus.VALID


def test_invalid_reference_raises() -> None:
    with pytest.raises(CalibrationError):
        analyse(Z, signal_v(Z, 0.0), dark_v=0.5, reference_v=0.4)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="differ"):
        analyse(Z, signal_v(Z, 0.0)[:-1])


# --------------------------------------------------------------------------- status precedence


def test_noise_only_is_no_peak(rng: np.random.Generator) -> None:
    a = analyse(Z, 0.05 + rng.normal(0.0, 0.01, Z.size)).analysis
    assert a.status is PointStatus.NO_PEAK
    assert a.surface_z_um is None
    assert a.confidence == 0.0


def test_low_snr_is_no_peak(rng: np.random.Generator) -> None:
    a = analyse(Z, signal_v(Z, 0.0, amplitude=0.02) + rng.normal(0.0, 0.01, Z.size)).analysis
    assert a.status is PointStatus.NO_PEAK


def test_peak_outside_sweep_is_at_edge() -> None:
    a = analyse(Z, signal_v(Z, 6.5)).analysis
    assert a.status is PointStatus.PEAK_AT_EDGE
    assert ProfileFlag.PEAK_AT_EDGE.value in a.flags


def _failed(method: FitMethod) -> PeakFit:
    return PeakFit(method=method, success=False, message="forced failure")


def test_both_fits_failing_is_fit_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_module, "fit_gaussian", lambda *a, **k: _failed(FitMethod.GAUSSIAN))
    monkeypatch.setattr(
        profile_module, "fit_parabolic", lambda *a, **k: _failed(FitMethod.PARABOLIC)
    )
    a = analyse(Z, signal_v(Z, 0.0)).analysis
    assert a.status is PointStatus.FIT_FAILED
    assert a.surface_z_um is None
    assert a.peak_found


def test_edge_takes_precedence_over_fit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_module, "fit_gaussian", lambda *a, **k: _failed(FitMethod.GAUSSIAN))
    monkeypatch.setattr(
        profile_module, "fit_parabolic", lambda *a, **k: _failed(FitMethod.PARABOLIC)
    )
    assert analyse(Z, signal_v(Z, 6.5)).analysis.status is PointStatus.PEAK_AT_EDGE


def test_confidence_threshold_gives_low_confidence(rng: np.random.Generator) -> None:
    v = signal_v(Z, 0.0) + rng.normal(0.0, 0.01, Z.size)
    a = analyse(Z, v, ProcessingConfig(min_confidence=1.0)).analysis
    assert a.status is PointStatus.LOW_CONFIDENCE
    assert a.surface_z_um is not None


# --------------------------------------------------------------------------- surface Z choice


def test_parabolic_used_when_gaussian_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_module, "fit_gaussian", lambda *a, **k: _failed(FitMethod.GAUSSIAN))
    a = analyse(Z, signal_v(Z, 0.4)).analysis
    assert a.surface_method is FitMethod.PARABOLIC
    assert a.surface_z_um == pytest.approx(0.4, abs=0.1)
    assert ProfileFlag.GAUSSIAN_FAILED.value in a.flags


def test_parabolic_used_when_fits_disagree(monkeypatch: pytest.MonkeyPatch) -> None:
    real = profile_module.fit_gaussian

    def shifted(*args: object, **kwargs: object) -> PeakFit:
        fit = real(*args, **kwargs)  # type: ignore[arg-type]
        assert fit.center_um is not None
        return fit.model_copy(update={"center_um": fit.center_um + 2.0})

    monkeypatch.setattr(profile_module, "fit_gaussian", shifted)
    a = analyse(Z, signal_v(Z, 0.4)).analysis
    assert a.surface_method is FitMethod.PARABOLIC
    assert ProfileFlag.FIT_DISAGREEMENT.value in a.flags


def test_saturated_top_is_excluded_from_fits(rng: np.random.Generator) -> None:
    raw = signal_v(Z, 0.7, amplitude=3.0)
    v = np.minimum(raw, 2.0) + rng.normal(0.0, 0.003, Z.size)
    config = ProcessingConfig(saturation_v=1.99)
    a = analyse(Z, v, config).analysis
    assert a.saturated_fraction > 0.0
    assert ProfileFlag.SATURATED.value in a.flags
    assert a.surface_z_um == pytest.approx(0.7, abs=0.05)
    unsaturated = analyse(Z, signal_v(Z, 0.7)).analysis
    assert a.confidence < unsaturated.confidence


# --------------------------------------------------------------------------- confidence


BASE_QUALITY = PeakQuality(
    snr=30.0,
    relative_prominence=0.95,
    r_squared=0.99,
    residual_ratio=0.02,
    fit_disagreement_um=0.05,
    secondary_peak_ratio=0.0,
)


def test_confidence_in_unit_interval_and_monotonic_in_snr() -> None:
    scores = [confidence_score(replace(BASE_QUALITY, snr=s)) for s in (0.0, 1, 3, 5, 10, 30, 1e6)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores)
    assert scores[0] == 0.0
    assert len(set(scores)) == len(scores)


@pytest.mark.parametrize(
    "change",
    [
        {"at_edge": True},
        {"saturated_fraction": 0.1},
        {"secondary_peak_ratio": 0.6},
        {"fit_disagreement_um": 0.9},
        {"fit_disagreement_um": None},
        {"relative_prominence": 0.3},
        {"residual_ratio": 0.3},
        {"fwhm_um": 6.0, "expected_fwhm_um": 2.8},
    ],
)
def test_confidence_penalties(change: dict[str, object]) -> None:
    assert confidence_score(replace(BASE_QUALITY, **change)) < confidence_score(BASE_QUALITY)  # type: ignore[arg-type]


def test_confidence_decreases_with_noise() -> None:
    rng = np.random.default_rng(7)
    means = []
    for noise in (0.005, 0.03, 0.08):
        conf = [
            analyse(Z, signal_v(Z, 0.0) + rng.normal(0.0, noise, Z.size)).analysis.confidence
            for _ in range(20)
        ]
        means.append(float(np.mean(conf)))
    assert means[0] > means[1] > means[2]


# --------------------------------------------------------------------------- robustness


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_tiny_profiles_never_raise(n: int) -> None:
    a = analyse(Z[:n], signal_v(Z, 0.0)[:n]).analysis
    assert a.status in {PointStatus.NO_PEAK, PointStatus.PEAK_AT_EDGE}
    assert a.surface_z_um is None or a.status is not PointStatus.VALID


@pytest.mark.parametrize("value", [0.3, np.nan, np.inf])
def test_constant_and_non_finite_profiles(value: float) -> None:
    result = analyse(Z, np.full(Z.size, value))
    assert result.analysis.status is PointStatus.NO_PEAK
    assert result.filtered.shape == Z.shape


def test_unsorted_duplicate_and_nan_samples(rng: np.random.Generator) -> None:
    z = np.concatenate((Z, Z[10:20]))
    v = signal_v(z, 0.3) + rng.normal(0.0, 0.005, z.size)
    v[5] = np.nan
    order = rng.permutation(z.size)
    result = analyse(z[order], v[order])
    a = result.analysis
    assert a.status is PointStatus.VALID
    assert a.surface_z_um == pytest.approx(0.3, abs=0.05)
    assert ProfileFlag.DUPLICATE_Z.value in a.flags
    assert ProfileFlag.NON_FINITE_SAMPLES.value in a.flags
    np.testing.assert_allclose(result.corrected_v, v[order])
    nan_at = int(np.flatnonzero(order == 5)[0])
    assert np.isnan(result.filtered[nan_at])
    assert a.peak_index is not None
    assert abs(z[order][a.peak_index] - 0.3) < 0.3


def test_analysis_is_json_safe() -> None:
    for v in (signal_v(Z, 0.0), np.full(Z.size, np.nan), signal_v(Z, 9.0)):
        analysis = analyse(Z, v).analysis
        dumped = analysis.model_dump_json()
        assert "NaN" not in dumped
        assert "Infinity" not in dumped


#: Per-profile analysis budget. A confocal point visits >= 2 sweeps of tens of Z
#: positions with >= 10 ms settling each, so 10 ms of analysis is a few percent
#: of the point time even on a Raspberry Pi 5 (~2-3x slower than a desktop, where
#: one 49-sample profile takes ~3 ms). A regression to O(n^2) per-candidate work
#: or an unbounded fit loop exceeds it by far.
ANALYSIS_BUDGET_S = 0.010


def test_analysis_is_fast(rng: np.random.Generator) -> None:
    """Median over several batches, so a burst of machine load cannot fail it."""
    profiles = [signal_v(Z, c) + rng.normal(0.0, 0.01, Z.size) for c in rng.uniform(-2, 2, 20)]
    for v in profiles[:3]:
        analyse(Z, v)  # warm-up (imports, SciPy caches)
    per_profile: list[float] = []
    for _ in range(9):
        start = time.perf_counter()
        for v in profiles:
            analyse(Z, v)
        per_profile.append((time.perf_counter() - start) / len(profiles))
    assert float(np.median(per_profile)) < ANALYSIS_BUDGET_S, per_profile


# --------------------------------------------------------------------------- coarse peak


def test_coarse_peak_found_with_sub_step_accuracy(rng: np.random.Generator) -> None:
    z = np.arange(-50.0, 50.001, 2.0)
    v = signal_v(z, 13.3) + rng.normal(0.0, 0.005, z.size)
    peak = find_coarse_peak(z, v, dark_v=0.0, config=CONFIG)
    assert peak.found
    assert not peak.at_edge
    assert peak.z_um == pytest.approx(13.3, abs=1.0)
    assert peak.index is not None
    assert z[peak.index] == pytest.approx(14.0)


def test_coarse_peak_at_edge_and_not_found(rng: np.random.Generator) -> None:
    z = np.arange(-50.0, 50.001, 2.0)
    edge = find_coarse_peak(z, signal_v(z, 51.0), dark_v=None, config=CONFIG)
    assert edge.found
    assert edge.at_edge
    noise = find_coarse_peak(z, 0.05 + rng.normal(0, 0.01, z.size), dark_v=None, config=CONFIG)
    assert not noise.found
    assert noise.message
    empty = find_coarse_peak(z[:0], z[:0], dark_v=None, config=CONFIG)
    assert not empty.found


def test_coarse_peak_reports_caller_index_for_unsorted_input() -> None:
    z = np.arange(-50.0, 50.001, 2.0)
    order = np.random.default_rng(3).permutation(z.size)
    peak = find_coarse_peak(z[order], signal_v(z, -20.0)[order], dark_v=None, config=CONFIG)
    assert peak.found
    assert peak.index is not None
    assert z[order][peak.index] == pytest.approx(-20.0)


# --------------------------------------------------------------------------- truncated flanks


def _focus_and_ghost(z: np.ndarray, focus_z: float = 1.5) -> np.ndarray:
    """True focus peak (height 1.0) near the sweep start plus a weaker ghost reflection."""
    return np.asarray(
        0.05
        + 1.0 * np.exp(-0.5 * ((z - focus_z) / 1.5) ** 2)
        + 0.5 * np.exp(-0.5 * ((z - 14.0) / 1.5) ** 2)
    )


GHOST_CONFIG = ProcessingConfig(expected_fwhm_um=3.5)


def test_coarse_peak_prefers_truncated_focus_over_weaker_ghost() -> None:
    """The sweep cuts the left flank of the real peak: it must still beat the ghost."""
    rng = np.random.default_rng(1)
    z = np.arange(0.0, 20.01, 0.5)
    peak = find_coarse_peak(
        z, _focus_and_ghost(z) + rng.normal(0.0, 0.002, z.size), dark_v=0.0, config=GHOST_CONFIG
    )
    assert peak.found
    assert peak.z_um == pytest.approx(1.5, abs=0.3)
    assert not peak.global_max_not_selected


@pytest.mark.parametrize("focus_z", [1.5, 2.0])
def test_fine_analysis_never_reports_the_ghost(focus_z: float) -> None:
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 20.0, 41)
    a = analyse(z, _focus_and_ghost(z, focus_z) + rng.normal(0.0, 0.002, z.size), GHOST_CONFIG)
    a = a.analysis
    assert a.surface_z_um is not None
    assert a.surface_z_um == pytest.approx(focus_z, abs=0.3)
    assert a.peak_intensity is not None
    assert a.peak_intensity > 1.0
    assert ProfileFlag.MULTIPLE_PEAKS.value in a.flags


def test_coarse_then_fine_sweep_finds_the_focus_not_the_ghost() -> None:
    """repro_ghost2: the fine sweep is centred on the coarse peak, so that must be right."""
    rng = np.random.default_rng(1)
    zc = np.arange(0.0, 20.01, 0.5)
    vc = _focus_and_ghost(zc) + rng.normal(0.0, 0.002, zc.size)
    coarse = find_coarse_peak(zc, vc, dark_v=0.0, config=GHOST_CONFIG)
    assert coarse.z_um is not None
    zf = np.linspace(coarse.z_um - 5.0, coarse.z_um + 5.0, 51)
    a = analyse(zf, _focus_and_ghost(zf) + rng.normal(0.0, 0.002, zf.size), GHOST_CONFIG)
    assert a.analysis.surface_z_um == pytest.approx(1.5, abs=0.1)


def test_selected_peak_below_global_maximum_is_never_valid(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator
) -> None:
    """When a brighter signal than the chosen peak exists, the point is ambiguous."""
    real = profile_module.detect_peaks

    def ambiguous(*args: object, **kwargs: object) -> PeakDetection:
        return replace(real(*args, **kwargs), global_max_not_selected=True)  # type: ignore[arg-type]

    v = signal_v(Z, 0.0) + rng.normal(0.0, 0.01, Z.size)
    assert analyse(Z, v).analysis.status is PointStatus.VALID
    monkeypatch.setattr(profile_module, "detect_peaks", ambiguous)
    a = analyse(Z, v, ProcessingConfig(min_confidence=0.0)).analysis
    assert a.status is PointStatus.LOW_CONFIDENCE
    assert ProfileFlag.GLOBAL_MAX_NOT_SELECTED.value in a.flags
    assert a.confidence < CONFIG.min_confidence
    coarse = find_coarse_peak(Z, v, dark_v=None, config=CONFIG)
    assert coarse.found
    assert coarse.global_max_not_selected
    assert coarse.message
