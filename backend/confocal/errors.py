"""Exception hierarchy shared by every layer.

Rules:
* Anything derived from :class:`HardwareError` means the physical (or
  simulated) instrument misbehaved. A running scan must stop immediately,
  the stage must be stopped and the scan recorded as interrupted.
* :class:`SafetyError` subclasses are raised *before* anything moves
  (travel-limit violations, emergency stop latched).
* Domain errors (scan state, calibration, reconstruction, ML) never imply
  that the hardware is unsafe.

The API layer maps these classes to HTTP status codes; lower layers must
never import FastAPI.
"""

from __future__ import annotations


class ConfocalError(Exception):
    """Base class for all application errors."""


# --------------------------------------------------------------------------- hardware


class HardwareError(ConfocalError):
    """The instrument failed. Scans must stop immediately on this error."""


class HardwareNotConnectedError(HardwareError):
    """A hardware component was used before ``connect()`` or after ``close()``."""


class CommunicationError(HardwareError):
    """Transport-level failure: timeout, disconnected port, I2C NACK, ..."""


class ProtocolError(CommunicationError):
    """A message was malformed, failed its checksum, or was out of sequence."""


class MotionError(HardwareError):
    """A stage movement failed."""


class MotionTimeoutError(MotionError):
    """The stage did not report completion within the allowed time."""


class MotionAbortedError(MotionError):
    """The movement was interrupted (stop / emergency stop) before completion."""


class MotionVerificationError(MotionError):
    """The position reported after a move does not match the commanded target."""


class ADCError(HardwareError):
    """The analogue-to-digital converter failed or returned invalid data."""


class LaserError(HardwareError):
    """The laser could not be switched or its state could not be determined."""


class CameraError(HardwareError):
    """The camera failed."""


class HardwareConfigError(ConfocalError):
    """The configured hardware combination is invalid or not yet implemented."""


# --------------------------------------------------------------------------- safety


class SafetyError(ConfocalError):
    """A request was refused for safety reasons before any hardware action."""


class LimitViolationError(SafetyError):
    """A target lies outside the configured travel limits. Nothing was moved."""

    def __init__(self, message: str, *, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations: list[str] = list(violations or [])


class EmergencyStopActiveError(SafetyError):
    """The emergency stop is latched; motion is refused until it is reset."""


# --------------------------------------------------------------------------- domain


class CalibrationError(ConfocalError):
    """Calibration is missing, invalid or could not be measured."""


class ScanError(ConfocalError):
    """Base class for scan-management errors."""


class ScanNotFoundError(ScanError):
    """No scan exists with the requested id."""


class PointNotFoundError(ScanError):
    """No point exists with the requested id in the scan."""


class ScanStateError(ScanError):
    """The requested action or transition is invalid in the scan's current state."""


class ScanConflictError(ScanError):
    """The instrument is busy (another scan is active) or the request conflicts with it."""


class ReconstructionError(ConfocalError):
    """Surface reconstruction is impossible with the available data."""


class ModelNotAvailableError(ConfocalError):
    """No deployed ML model is available for inference."""


class StorageError(ConfocalError):
    """Persisting or loading data failed."""
