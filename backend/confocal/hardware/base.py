"""Hardware abstraction interfaces.

The scanning and processing layers depend only on :class:`MicroscopeController`
(and, through it, on these abstract components). Concrete backends:

* ``hardware.simulation``  - SimulationStage / SimulationADC / SimulationLaser /
  SimulationCamera driven by a SimulatedConfocalSurface (Phase 2)
* ``hardware.arduino``     - ArduinoStage over the serial protocol (Phase 3)
* ``hardware.ads1115``     - ADS1115ADC reading the OPT101 (Phase 4)
* ``hardware.openflexure`` - stage adapter for an OpenFlexure server (Phase 11)

Contract for every implementation:

* All coordinates are micrometres. Never expose motor steps.
* All methods are ``async``. Blocking I/O (pyserial, smbus) must be run via
  ``asyncio.to_thread`` so the event loop (and HTTP) never blocks.
* Never report success that was not confirmed by the device. ``Stage.move_to``
  returns the position *reported by the device* after the move.
* ``Stage.stop`` must work at any time, concurrently with a running
  ``move_to`` from another task, and must never raise because motion is idle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from confocal.errors import LimitViolationError
from confocal.models.calibration import (
    ADCCalibrateRequest,
    ADCCalibrateResponse,
    CalibrationState,
    DarkCalibrationRequest,
    ReferenceCalibrationRequest,
)
from confocal.models.common import Axis, Position, StageLimits
from confocal.models.hardware import (
    AdcGain,
    ADCStatus,
    CameraStatus,
    HardwareInfo,
    HardwareStatus,
    LaserStatus,
    SamplingMethod,
    StageStatus,
)
from confocal.models.measurement import AdcSamples, IntensityMeasurement


class HardwareComponent(ABC):
    """Lifecycle shared by every hardware component."""

    #: Short backend identifier, e.g. "simulation", "arduino", "ads1115".
    backend_name: str = "unknown"

    @abstractmethod
    async def connect(self) -> None:
        """Open the device. Idempotent."""

    @abstractmethod
    async def close(self) -> None:
        """Release the device. Idempotent; must leave hardware in a safe state."""

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    def version(self) -> str | None:
        """Firmware / driver version string, if known."""
        return None


class Stage(HardwareComponent):
    """Three-axis positioning stage, coordinates in micrometres."""

    @property
    @abstractmethod
    def limits(self) -> StageLimits:
        """Travel limits this stage enforces (defence in depth below the controller)."""

    @abstractmethod
    async def get_position(self) -> Position:
        """Query the device for its current position."""

    @abstractmethod
    async def move_to(self, target: Position) -> Position:
        """Absolute move; returns the device-reported position after motion completes.

        Raises:
            LimitViolationError: target outside :attr:`limits` (nothing moves).
            MotionAbortedError: :meth:`stop` was called during the move.
            MotionTimeoutError / CommunicationError / MotionError: device failure.
        """

    @abstractmethod
    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        """Return the given axes (default: all) to the reference origin."""

    @abstractmethod
    async def stop(self) -> None:
        """Immediately halt all motion. Safe to call at any time from any task."""

    @abstractmethod
    async def status(self) -> StageStatus: ...

    async def release(self) -> None:
        """De-energise the motors (28BYJ-48 coils heat up when held). Optional."""

    def check_limits(self, target: Position) -> None:
        """Raise :class:`LimitViolationError` if ``target`` is outside :attr:`limits`."""
        violations = self.limits.violations(target)
        if violations:
            raise LimitViolationError(
                "target outside travel limits: " + "; ".join(violations), violations=violations
            )


class ADC(HardwareComponent):
    """Analogue-to-digital converter reading the photodiode amplifier."""

    @property
    @abstractmethod
    def gain(self) -> AdcGain: ...

    @abstractmethod
    async def set_gain(self, gain: AdcGain) -> None: ...

    @property
    @abstractmethod
    def data_rate_sps(self) -> int: ...

    @abstractmethod
    async def read_samples(self, n: int) -> AdcSamples:
        """Take ``n`` consecutive raw conversions (n >= 1). Never averages internally."""

    @abstractmethod
    async def status(self) -> ADCStatus: ...

    def counts_to_volts(self, counts: NDArray[np.int32]) -> NDArray[np.float64]:
        """Convert raw signed 16-bit codes to volts with the current gain."""
        return np.asarray(counts, dtype=np.float64) * self.gain.lsb_v


class Laser(HardwareComponent):
    """The illumination laser (650 nm, 5 mW)."""

    @property
    @abstractmethod
    def controllable(self) -> bool:
        """False for a manually switched laser; dark calibration then needs confirmation."""

    @abstractmethod
    async def set_enabled(self, enabled: bool) -> None:
        """Switch the laser. Must raise LaserError when not controllable."""

    @abstractmethod
    async def is_enabled(self) -> bool | None:
        """Current state, or None when it cannot be known."""

    @abstractmethod
    async def status(self) -> LaserStatus: ...


class Camera(HardwareComponent):
    """Optional imaging camera (OpenFlexure Pi camera in later phases)."""

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def capture(self) -> NDArray[np.uint8]:
        """Capture one frame (H x W or H x W x 3). Raise CameraError if unavailable."""

    @abstractmethod
    async def status(self) -> CameraStatus: ...


EstopListener = Callable[[str], Awaitable[None]]


class MicroscopeController(ABC):
    """The single entry point the scanning layer and the API use to touch hardware.

    Responsibilities of every implementation:

    * enforce travel limits *before* commanding motion (LimitViolationError);
    * verify every move against the device-reported position
      (MotionVerificationError when off by more than the configured tolerance);
    * serialise hardware access (one operation at a time);
    * implement a latching emergency stop: ``emergency_stop`` halts the stage,
      switches the laser off where possible, notifies listeners and refuses
      further motion (EmergencyStopActiveError) until ``reset_emergency_stop``;
    * own calibration (dark / reference) and apply it in ``measure_intensity``.

    A future ``OpenFlexureMicroscopeController`` can implement this interface on
    top of an OpenFlexure server without changing scanning or processing code.
    """

    # ----------------------------------------------------------------- lifecycle
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    # ----------------------------------------------------------------- identity / state
    @property
    @abstractmethod
    def limits(self) -> StageLimits: ...

    @property
    @abstractmethod
    def estop_engaged(self) -> bool: ...

    @property
    @abstractmethod
    def calibration(self) -> CalibrationState:
        """Current calibration snapshot (may be uncalibrated: dark_v/reference_v None)."""

    @abstractmethod
    def hardware_info(self) -> HardwareInfo: ...

    @abstractmethod
    async def status(self) -> HardwareStatus: ...

    # ----------------------------------------------------------------- motion
    @abstractmethod
    async def get_position(self) -> Position: ...

    @abstractmethod
    async def move_to(
        self,
        *,
        x_um: float | None = None,
        y_um: float | None = None,
        z_um: float | None = None,
    ) -> Position:
        """Validated, verified absolute move. Unspecified axes keep their position."""

    @abstractmethod
    async def move_relative(
        self, *, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0
    ) -> Position: ...

    @abstractmethod
    async def home(self, axes: Sequence[Axis] | None = None) -> Position: ...

    @abstractmethod
    async def emergency_stop(self, reason: str) -> None:
        """Latch the e-stop, halt motion immediately, laser off, notify listeners.

        Must not wait for the hardware lock held by an in-flight operation.
        Never raises (failures are logged; the latch is engaged regardless).
        """

    @abstractmethod
    async def reset_emergency_stop(self) -> Position:
        """Clear the latch after re-reading the (possibly changed) stage position."""

    @abstractmethod
    def add_estop_listener(self, listener: EstopListener) -> None:
        """Register a coroutine called with the reason whenever the e-stop is triggered."""

    # ----------------------------------------------------------------- timing
    @abstractmethod
    async def wait_settle(self, seconds: float) -> None:
        """Wait for mechanical / optical settling after a move.

        Hardware controllers sleep for ``seconds``; simulation controllers scale
        the wait by the simulation time scale (0 = no wait, but still yield).
        Scan code must use this instead of ``asyncio.sleep`` so timing stays a
        hardware concern.
        """

    # ----------------------------------------------------------------- illumination
    @abstractmethod
    async def set_laser(self, enabled: bool) -> None: ...

    # ----------------------------------------------------------------- detection
    @abstractmethod
    async def acquire(self, n_samples: int) -> AdcSamples:
        """Raw ADC burst at the current position (no calibration applied)."""

    @abstractmethod
    async def measure_intensity(
        self, n_samples: int, method: SamplingMethod = SamplingMethod.MEDIAN
    ) -> IntensityMeasurement:
        """Aggregated, dark-corrected and (when possible) normalized intensity."""

    @abstractmethod
    async def configure_adc(self, request: ADCCalibrateRequest) -> ADCCalibrateResponse:
        """Set the ADC gain explicitly or auto-range it on the current signal."""

    # ----------------------------------------------------------------- calibration
    @abstractmethod
    async def calibrate_dark(self, request: DarkCalibrationRequest) -> CalibrationState:
        """Measure and persist a new dark level; returns the new calibration version."""

    @abstractmethod
    async def calibrate_reference(self, request: ReferenceCalibrationRequest) -> CalibrationState:
        """Measure and persist a new reference level; returns the new calibration version."""
