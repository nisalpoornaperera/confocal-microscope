"""Stage kinematics: Cartesian micrometres <-> motor steps.

The application works exclusively in Cartesian micrometres (X, Y, Z). The
three stepper motors (a, b, c) are related to Cartesian space by a constant
linear transform::

    position_um = M @ steps            steps = inv(M) @ position_um

``M[i, j]`` is the displacement in micrometres along Cartesian axis ``i``
(x, y, z) produced by one step of motor ``j`` (a, b, c). This module is part
of the hardware layer: nothing above it ever sees motor steps.

Geometries
----------
cartesian
    Classic OpenFlexure microscope stage: motor a drives X, b drives Y and
    c drives Z, so ``M = diag(1 / steps_per_um)``.

delta
    OpenFlexure Delta Stage. Three actuators 120 degrees apart each raise one
    leg of the sample platform: moving all three legs together translates the
    platform in Z, moving them differentially tilts the legs and translates it
    in X/Y. This follows the linearised transform used by the OpenFlexure
    server (``SangaDeltaStage``), in "stage step" units::

        x = (2 / sqrt(3)) (b / h) (A - B)
        y = (b / h) (C - (A + B) / 2)
        z = (1 / 3) (b / a) (A + B + C)

    followed by an optional rotation of the X/Y frame about Z (see
    ``xy_rotation_deg``) and a per-axis scale in micrometres per stage step.
    With no rotation, +X is produced by stepping motor A forward and B backward,
    and +Y by stepping C forward and A and B backward (the delta's natural
    frame); which physical direction that is must be checked on the machine.
    Consequences for motion: a pure Z move drives
    all three motors by the same amount; an X or Y move drives at least two
    motors in opposite directions. The firmware must therefore interpolate
    the three motors so they start and finish together, otherwise the platform
    wanders sideways during every Z step.

matrix
    A fully measured 3 x 3 matrix, recommended once the stage is calibrated:
    move one motor by N steps, measure the X/Y displacement (e.g. a fixed-Z
    confocal image of a calibration grating before and after) and the Z
    displacement (shift of the confocal peak on a mirror against an external
    reference), divide by N - that is one column of ``M``. This also absorbs
    manufacturing tolerances that the nominal delta geometry ignores.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confocal.config import ArduinoConfig, KinematicsConfig
from confocal.models.common import Axis, Position, StageLimits


class Motor(StrEnum):
    A = "a"
    B = "b"
    C = "c"


MOTORS: tuple[Motor, Motor, Motor] = (Motor.A, Motor.B, Motor.C)

MotorSteps = tuple[int, int, int]

#: Matrices worse conditioned than this are rejected as (numerically) singular.
_MAX_CONDITION_NUMBER = 1e8


def _round_half_away(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sign(values) * np.floor(np.abs(values) + 0.5), dtype=np.float64)


class StageKinematics:
    """Linear map between Cartesian micrometres and integer motor steps."""

    def __init__(self, um_per_step: ArrayLike, *, name: str = "matrix") -> None:
        matrix = np.array(um_per_step, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("um_per_step must be a finite 3 x 3 matrix")
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition) or condition > _MAX_CONDITION_NUMBER:
            raise ValueError(
                f"kinematics matrix is singular or ill-conditioned (cond = {condition:.3g})"
            )
        inverse = np.asarray(np.linalg.inv(matrix), dtype=np.float64)
        matrix.flags.writeable = False
        inverse.flags.writeable = False
        self._um_per_step: NDArray[np.float64] = matrix
        self._steps_per_um: NDArray[np.float64] = inverse
        self.name = name

    # ------------------------------------------------------------------ constructors
    @classmethod
    def cartesian(cls, steps_per_um: Sequence[float]) -> StageKinematics:
        """One motor per axis: a -> X, b -> Y, c -> Z."""
        if len(steps_per_um) != 3 or any(not (s > 0 and math.isfinite(s)) for s in steps_per_um):
            raise ValueError("steps_per_um needs three positive, finite values (x, y, z)")
        return cls(np.diag([1.0 / float(s) for s in steps_per_um]), name="cartesian")

    @classmethod
    def openflexure_delta(
        cls,
        *,
        um_per_step: Sequence[float],
        flex_h: float = 80.0,
        flex_a: float = 50.0,
        flex_b: float = 50.0,
        xy_rotation_deg: float = 0.0,
    ) -> StageKinematics:
        """OpenFlexure Delta Stage geometry (defaults match the OpenFlexure software).

        Args:
            um_per_step: micrometres per "stage step" along x, y, z (calibrate).
            flex_h, flex_a, flex_b: flexure lever dimensions; only ratios matter.
            xy_rotation_deg: rotation of the application's X/Y frame relative to
                the delta's natural frame. OpenFlexure calls this the camera
                angle because it aligns stage moves with the image of the Pi
                camera. This scanner images with the photodiode, which has no
                orientation, so leave it at 0 unless the scan rows should follow
                a sample feature or a camera is added later. Z is unaffected.
        """
        if len(um_per_step) != 3 or any(not (s > 0 and math.isfinite(s)) for s in um_per_step):
            raise ValueError("um_per_step needs three positive, finite values (x, y, z)")
        if min(flex_h, flex_a, flex_b) <= 0:
            raise ValueError("flexure dimensions must be positive")
        x_fac = -(2.0 / math.sqrt(3.0)) * (flex_b / flex_h)
        y_fac = -(flex_b / flex_h)
        z_fac = (1.0 / 3.0) * (flex_b / flex_a)
        # Motor steps -> Cartesian stage steps (OpenFlexure's Tvd).
        t_vd = np.array(
            [
                [-x_fac, x_fac, 0.0],
                [0.5 * y_fac, 0.5 * y_fac, -y_fac],
                [z_fac, z_fac, z_fac],
            ]
        )
        theta = math.radians(xy_rotation_deg)
        # OpenFlexure reports position = inv(R_camera) @ Tvd @ steps.
        inv_rotation = np.array(
            [
                [math.cos(theta), math.sin(theta), 0.0],
                [-math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = np.diag([float(s) for s in um_per_step])
        return cls(scale @ inv_rotation @ t_vd, name="delta")

    @classmethod
    def from_config(cls, config: KinematicsConfig) -> StageKinematics:
        if config.geometry == "cartesian":
            return cls.cartesian(config.steps_per_um)
        if config.geometry == "matrix":
            if config.um_per_step_matrix is None:  # guarded by KinematicsConfig validation
                raise ValueError("geometry 'matrix' requires um_per_step_matrix")
            return cls(config.um_per_step_matrix, name="matrix")
        return cls.openflexure_delta(
            um_per_step=config.um_per_step,
            flex_h=config.delta_flex_h,
            flex_a=config.delta_flex_a,
            flex_b=config.delta_flex_b,
            xy_rotation_deg=config.xy_rotation_deg,
        )

    # ------------------------------------------------------------------ matrices
    @property
    def um_per_step(self) -> NDArray[np.float64]:
        """``M``: column j is the Cartesian displacement of one step of motor j."""
        return self._um_per_step.copy()

    @property
    def steps_per_um(self) -> NDArray[np.float64]:
        """``inv(M)``."""
        return self._steps_per_um.copy()

    # ------------------------------------------------------------------ conversions
    def exact_steps(self, position: Position) -> NDArray[np.float64]:
        """Fractional motor steps that would reach ``position`` exactly."""
        return np.asarray(self._steps_per_um @ np.array(position.as_tuple()), dtype=np.float64)

    def to_steps(self, position: Position) -> MotorSteps:
        """Nearest integer motor steps (round half away from zero)."""
        a, b, c = (int(v) for v in _round_half_away(self.exact_steps(position)))
        return (a, b, c)

    def to_position(self, steps: Sequence[float]) -> Position:
        """Cartesian position of the given motor step counters."""
        if len(steps) != 3:
            raise ValueError("expected three motor step values (a, b, c)")
        x, y, z = (float(v) for v in self._um_per_step @ np.asarray(steps, dtype=np.float64))
        return Position(x_um=x, y_um=y, z_um=z)

    def quantize(self, position: Position) -> Position:
        """The position the stage actually reaches when commanded to ``position``."""
        return self.to_position(self.to_steps(position))

    def steps_for_displacement(
        self, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0
    ) -> NDArray[np.float64]:
        """Fractional motor steps for a relative Cartesian displacement."""
        return np.asarray(self._steps_per_um @ np.array([dx_um, dy_um, dz_um]), dtype=np.float64)

    def max_quantization_error_um(self) -> float:
        """Upper bound of the per-axis error introduced by rounding to whole steps."""
        return float(0.5 * np.max(np.sum(np.abs(self._um_per_step), axis=1)))

    # ------------------------------------------------------------------ limits / speed
    def motor_ranges(self, limits: StageLimits) -> dict[Motor, tuple[float, float]]:
        """Motor step range needed to reach every point of the Cartesian limit box.

        The map is linear and the box convex, so the extremes occur at the
        eight corners of the box.
        """
        corners = np.array(
            [
                self.exact_steps(Position(x_um=x, y_um=y, z_um=z))
                for x, y, z in itertools.product(
                    (limits.x.min_um, limits.x.max_um),
                    (limits.y.min_um, limits.y.max_um),
                    (limits.z.min_um, limits.z.max_um),
                )
            ]
        )
        return {
            motor: (float(corners[:, j].min()), float(corners[:, j].max()))
            for j, motor in enumerate(MOTORS)
        }

    def limit_violations(
        self, limits: StageLimits, motor_limits: Mapping[Motor, tuple[int, int]]
    ) -> list[str]:
        """Motors whose travel cannot cover the Cartesian limit box (empty if safe).

        When this is empty every target accepted by the Cartesian travel limits
        is also reachable by every actuator, so a single Cartesian limit check
        before each move is sufficient.
        """
        problems: list[str] = []
        for motor, (needed_min, needed_max) in self.motor_ranges(limits).items():
            low, high = motor_limits[motor]
            if needed_min < low or needed_max > high:
                problems.append(
                    f"motor {motor.value} needs steps [{needed_min:.0f}, {needed_max:.0f}] "
                    f"but its travel is [{low}, {high}]"
                )
        return problems

    def max_speed_um_s(
        self,
        direction: tuple[float, float, float] | Axis,
        max_steps_per_s: float | Mapping[Motor, float],
    ) -> float:
        """Fastest Cartesian speed along ``direction`` for a coordinated move.

        All motors start and finish together, so the speed is limited by the
        motor that has to turn fastest relative to its own maximum rate.
        """
        if isinstance(direction, Axis):
            vector = np.array([1.0 if axis is direction else 0.0 for axis in Axis])
        else:
            vector = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not norm > 0:
            raise ValueError("direction must be non-zero")
        rates = np.abs(self._steps_per_um @ (vector / norm))  # steps per um travelled
        if isinstance(max_steps_per_s, Mapping):
            limits = np.array([float(max_steps_per_s[m]) for m in MOTORS])
        else:
            limits = np.full(3, float(max_steps_per_s))
        if np.any(limits <= 0):
            raise ValueError("motor speeds must be positive")
        return float(1.0 / np.max(rates / limits))

    def config_limit_violations(self, limits: StageLimits, arduino: ArduinoConfig) -> list[str]:
        """:meth:`limit_violations` against the motor travel configured for the Arduino."""
        travel = {
            Motor.A: (arduino.a.min_steps, arduino.a.max_steps),
            Motor.B: (arduino.b.min_steps, arduino.b.max_steps),
            Motor.C: (arduino.c.min_steps, arduino.c.max_steps),
        }
        return self.limit_violations(limits, travel)

    def __repr__(self) -> str:
        return f"StageKinematics(name={self.name!r}, um_per_step={self._um_per_step.tolist()!r})"
