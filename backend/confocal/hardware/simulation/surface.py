"""Synthetic confocal sample: surface height, reflectivity and the axial response.

Confocal principle: light reflected by the sample passes the detection pinhole
only when the reflecting surface lies in the focal plane. Sweeping Z at a fixed
(x, y) therefore gives an axial response I(Z) that peaks where the focal plane
meets the surface; locating that peak is how the scanner measures height.
This module models the sample and that response *noise-free*; the simulated
ADC adds the dark level, stray light, noise, the OPT101 rail and quantisation::

    signal_v(x, y, z) = R(x, y) * peak_voltage_v * A(z - h(x, y))   (+ secondary peak)
    detector volts    = dark_voltage_v + [laser on] * (background_v + signal_v)

Height models (``SimulatedSurfaceConfig.kind``), each on top of the mounting
plane ``base_z_um + tilt_x * x + tilt_y * y`` (no real sample is perfectly
level; set both tilts to 0 for the bare feature). ``A`` = feature amplitude,
``P`` = feature period:

* ``plane``     - the tilted plane alone.
* ``sinusoid``  - egg-crate ``A sin(2 pi x / P) sin(2 pi y / P)``.
* ``steps``     - staircase along X, ``step_height_um`` per period ``P``.
* ``sphere``    - spherical cap (sphere radius ``sphere_radius_um``, cap height
  ``sphere_height_um``) centred on the origin.
* ``composite`` - sinusoid + one step edge (``x >= 0`` raised by
  ``step_height_um``) + one spherical bump centred at ``(-P / 2, P / 2)``.

Axial response ``A`` (1 at focus):

* ``gaussian`` - ``exp(-4 ln 2 dz^2 / FWHM^2)``;
* ``sinc2``    - ``sinc^2`` (the paraxial confocal response to a plane mirror),
  scaled so that its FWHM is also ``axial_fwhm_um``; it has weak side lobes.

Deterministic defects - pure functions of ``(x, y, seed)``, so the same seed
always gives the same sample (needed for reproducible tests and ML labels):

* **low-reflectivity patches**: the plane is divided into 47 um cells and a
  cell is "dark" (reflectivity :data:`LOW_REFLECTIVITY`, a peak of a few mV,
  lost in the noise) with probability ``low_reflectivity_fraction``, which is
  therefore the expected dark area fraction;
* **spurious peaks**: in 1 um cells selected with probability
  ``spurious_peak_probability`` a weaker secondary peak
  (:data:`SPURIOUS_AMPLITUDE` of the main one) appears 2.5-4 FWHM above or
  below the surface - like a reflection from a second interface or dust.

Cells are centred on multiples of their size, so their boundaries lie on
half-integer micrometres: a scan on an integer-micrometre grid never sits on a
boundary, and the stage's step quantisation (well below 0.1 um) cannot flip a
point between cells during its Z sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confocal.config import SimulatedSurfaceConfig, SimulationConfig

PsfModel = Literal["gaussian", "sinc2"]

#: Reflectivity inside a low-reflectivity patch: a ~3 mV peak, below the noise floor.
LOW_REFLECTIVITY = 0.001

#: Edge length of the cells that are independently low-reflective (um).
PATCH_CELL_UM = 47.0

#: Edge length of the cells of the spurious-peak lottery (um).
SPURIOUS_CELL_UM = 1.0

#: Amplitude of a spurious secondary peak relative to the main peak.
SPURIOUS_AMPLITUDE = 0.35

#: Distance of a spurious peak from the surface, in units of the axial FWHM.
SPURIOUS_OFFSET_FWHM: tuple[float, float] = (2.5, 4.0)

#: ``sinc(t)^2 = 1/2`` at ``t = 0.44295`` (numpy's normalised sinc).
_SINC2_HALF_WIDTH = 0.4429464706890664
_GAUSSIAN_K = 4.0 * math.log(2.0)

_SALT_PATCH = 0x51A7_0001
_SALT_SPURIOUS = 0x51A7_0002
_SALT_OFFSET = 0x51A7_0003
_UINT64_MOD = 1 << 64


def _cell_uniform(
    x: NDArray[np.float64], y: NDArray[np.float64], cell_um: float, seed: int, salt: int
) -> NDArray[np.float64]:
    """Deterministic pseudo-random number in [0, 1) per square cell.

    A splitmix64-style integer hash of (cell index x, cell index y, seed, salt):
    stateless, vectorised and independent of the evaluation order, unlike a
    random generator.
    """
    shape = np.broadcast_shapes(x.shape, y.shape)
    ix = np.floor(np.atleast_1d(x) / cell_um + 0.5).astype(np.int64)
    iy = np.floor(np.atleast_1d(y) / cell_um + 0.5).astype(np.int64)
    ix, iy = np.broadcast_arrays(ix, iy)
    key = np.uint64((seed * 0x9E37_79B9_7F4A_7C15 + salt) % _UINT64_MOD)
    with np.errstate(over="ignore"):
        h: NDArray[np.uint64] = ix.astype(np.uint64) * np.uint64(0x9E37_79B9_7F4A_7C15)
        h = h ^ (iy.astype(np.uint64) * np.uint64(0xC2B2_AE3D_27D4_EB4F))
        h = h ^ key
        h = h ^ (h >> np.uint64(30))
        h = h * np.uint64(0xBF58_476D_1CE4_E5B9)
        h = h ^ (h >> np.uint64(27))
        h = h * np.uint64(0x94D0_49BB_1331_11EB)
        h = h ^ (h >> np.uint64(31))
    uniform = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.asarray(uniform.reshape(shape), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SurfaceGroundTruth:
    """The true sample at a set of (x, y) positions (arrays of a common shape)."""

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    height_um: NDArray[np.float64]
    reflectivity: NDArray[np.float64]
    low_reflectivity: NDArray[np.bool_]
    spurious_peak: NDArray[np.bool_]
    #: Signed distance of the secondary peak from the surface; NaN where there is none.
    spurious_offset_um: NDArray[np.float64]


class SimulatedConfocalSurface:
    """Noise-free model of the sample and of the microscope's axial response.

    Every method is vectorised: arguments broadcast against each other and the
    result has the broadcast shape (0-d for scalar arguments; use ``float()``).
    """

    def __init__(
        self,
        surface: SimulatedSurfaceConfig | None = None,
        *,
        psf_model: PsfModel = "gaussian",
        axial_fwhm_um: float = 6.0,
        peak_voltage_v: float = 2.8,
        seed: int = 1234,
    ) -> None:
        if not (axial_fwhm_um > 0 and math.isfinite(axial_fwhm_um)):
            raise ValueError("axial_fwhm_um must be positive and finite")
        if not (peak_voltage_v > 0 and math.isfinite(peak_voltage_v)):
            raise ValueError("peak_voltage_v must be positive and finite")
        if psf_model not in ("gaussian", "sinc2"):
            raise ValueError(f"unknown psf_model {psf_model!r}")
        self._config = surface if surface is not None else SimulatedSurfaceConfig()
        self._psf_model: PsfModel = psf_model
        self._fwhm = float(axial_fwhm_um)
        self._peak_v = float(peak_voltage_v)
        self._seed = int(seed)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> SimulatedConfocalSurface:
        return cls(
            config.surface,
            psf_model=config.psf_model,
            axial_fwhm_um=config.axial_fwhm_um,
            peak_voltage_v=config.peak_voltage_v,
            seed=config.seed,
        )

    # ------------------------------------------------------------------ properties
    @property
    def config(self) -> SimulatedSurfaceConfig:
        return self._config

    @property
    def psf_model(self) -> PsfModel:
        return self._psf_model

    @property
    def axial_fwhm_um(self) -> float:
        return self._fwhm

    @property
    def peak_voltage_v(self) -> float:
        return self._peak_v

    @property
    def seed(self) -> int:
        return self._seed

    # ------------------------------------------------------------------ sample
    def height_um(self, x_um: ArrayLike, y_um: ArrayLike) -> NDArray[np.float64]:
        """True surface height at (x, y)."""
        x, y = _xy(x_um, y_um)
        cfg = self._config
        z = cfg.base_z_um + cfg.tilt_x * x + cfg.tilt_y * y
        period = cfg.feature_period_um
        if cfg.kind == "sinusoid":
            z = z + self._sinusoid(x, y)
        elif cfg.kind == "steps":
            z = z + cfg.step_height_um * np.floor(x / period)
        elif cfg.kind == "sphere":
            z = z + self._spherical_cap(x, y)
        elif cfg.kind == "composite":
            z = (
                z
                + self._sinusoid(x, y)
                + np.where(x >= 0.0, cfg.step_height_um, 0.0)
                + self._spherical_cap(x + 0.5 * period, y - 0.5 * period)
            )
        return np.asarray(z, dtype=np.float64)

    def reflectivity(self, x_um: ArrayLike, y_um: ArrayLike) -> NDArray[np.float64]:
        """Fraction of the in-focus peak reaching the detector (0..1)."""
        x, y = _xy(x_um, y_um)
        base = self._config.reflectivity
        return np.asarray(
            np.where(self._low_reflectivity(x, y), min(LOW_REFLECTIVITY, base), base),
            dtype=np.float64,
        )

    def ground_truth(self, x_um: ArrayLike, y_um: ArrayLike) -> SurfaceGroundTruth:
        """Heights, reflectivity and defect flags, e.g. as labels for offline ML training."""
        x, y = _xy(x_um, y_um)
        spurious = self._spurious(x, y)
        offset = np.where(spurious, self._spurious_offset_um(x, y), np.nan)
        return SurfaceGroundTruth(
            x_um=np.array(x, dtype=np.float64),
            y_um=np.array(y, dtype=np.float64),
            height_um=self.height_um(x, y),
            reflectivity=self.reflectivity(x, y),
            low_reflectivity=np.asarray(self._low_reflectivity(x, y), dtype=np.bool_),
            spurious_peak=np.asarray(spurious, dtype=np.bool_),
            spurious_offset_um=np.asarray(offset, dtype=np.float64),
        )

    # ------------------------------------------------------------------ optics
    def axial_response(self, dz_um: ArrayLike) -> NDArray[np.float64]:
        """Normalised axial response A(dz): 1 at focus, 1/2 at dz = +/- FWHM / 2."""
        dz = np.asarray(dz_um, dtype=np.float64)
        if self._psf_model == "sinc2":
            response = np.sinc(dz * (2.0 * _SINC2_HALF_WIDTH / self._fwhm)) ** 2
        else:
            response = np.exp(-_GAUSSIAN_K * dz**2 / self._fwhm**2)
        return np.asarray(response, dtype=np.float64)

    def signal_v(self, x_um: ArrayLike, y_um: ArrayLike, z_um: ArrayLike) -> NDArray[np.float64]:
        """Noise-free confocal signal above the stray-light background, laser on (volts)."""
        x, y = _xy(x_um, y_um)
        z = np.asarray(z_um, dtype=np.float64)
        height = self.height_um(x, y)
        amplitude = self.reflectivity(x, y) * self._peak_v
        main = amplitude * self.axial_response(z - height)
        spurious = self._spurious(x, y)
        if not np.any(spurious):
            return np.asarray(main, dtype=np.float64)
        offset = np.where(spurious, self._spurious_offset_um(x, y), 0.0)
        secondary = SPURIOUS_AMPLITUDE * amplitude * self.axial_response(z - height - offset)
        return np.asarray(main + np.where(spurious, secondary, 0.0), dtype=np.float64)

    # ------------------------------------------------------------------ internals
    def _sinusoid(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        cfg = self._config
        k = 2.0 * math.pi / cfg.feature_period_um
        return np.asarray(cfg.feature_amplitude_um * np.sin(k * x) * np.sin(k * y))

    def _spherical_cap(
        self, dx: NDArray[np.float64], dy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        radius = self._config.sphere_radius_um
        cap = min(self._config.sphere_height_um, radius)
        if cap <= 0.0:
            return np.zeros(np.broadcast_shapes(dx.shape, dy.shape))
        r2 = dx**2 + dy**2
        inside = r2 < 2.0 * radius * cap - cap**2  # within the cap's base circle
        z = np.sqrt(np.maximum(radius**2 - r2, 0.0)) - (radius - cap)
        return np.asarray(np.where(inside, z, 0.0), dtype=np.float64)

    def _low_reflectivity(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        fraction = self._config.low_reflectivity_fraction
        u = _cell_uniform(x, y, PATCH_CELL_UM, self._seed, _SALT_PATCH)
        return np.asarray(u < fraction, dtype=np.bool_)

    def _spurious(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.bool_]:
        probability = self._config.spurious_peak_probability
        u = _cell_uniform(x, y, SPURIOUS_CELL_UM, self._seed, _SALT_SPURIOUS)
        return np.asarray(u < probability, dtype=np.bool_)

    def _spurious_offset_um(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        u = _cell_uniform(x, y, SPURIOUS_CELL_UM, self._seed, _SALT_OFFSET)
        sign = np.where(u < 0.5, 1.0, -1.0)
        low, high = SPURIOUS_OFFSET_FWHM
        magnitude = low + (high - low) * ((2.0 * u) % 1.0)
        return np.asarray(sign * magnitude * self._fwhm, dtype=np.float64)


def _xy(x_um: ArrayLike, y_um: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    shape = np.broadcast_shapes(x.shape, y.shape)
    return np.broadcast_to(x, shape), np.broadcast_to(y, shape)
