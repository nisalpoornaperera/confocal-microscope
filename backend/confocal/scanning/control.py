"""Cooperative pause / cancel / abort signalling between the ScanManager and the executor.

The manager never interrupts an in-flight hardware operation to cancel or
pause: it only sets a flag. The executor polls :meth:`ScanControl.checkpoint`
before every Z step (cancel and emergency stop take effect at the next step,
§3) and honours a pause request between points, so the point being measured
is always completed or stored as ABORTED, never left half-written.

Physically stopping the stage on an emergency stop is the controller's job
(``MicroscopeController.emergency_stop`` halts it immediately); the abort
signal here only makes the scan logic stop issuing commands.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum


class StopKind(StrEnum):
    CANCEL = "cancel"  # operator cancel -> CANCELLED
    ABORT = "abort"  # emergency stop -> ERROR


@dataclass(frozen=True, slots=True)
class StopRequest:
    kind: StopKind
    message: str


class ScanStopRequested(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised at a checkpoint after a cancel or abort was requested."""

    def __init__(self, request: StopRequest) -> None:
        super().__init__(request.message)
        self.request = request


class ScanControl:
    """Pending pause / stop requests of one scan. Use from the event loop thread only."""

    def __init__(self) -> None:
        self._pause_requested = False
        self._stop: StopRequest | None = None
        self._wake = asyncio.Event()

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    @property
    def stop_request(self) -> StopRequest | None:
        return self._stop

    def request_pause(self) -> None:
        """Pause after the point being measured (ignored once a stop was requested)."""
        if self._stop is None:
            self._pause_requested = True

    def request_resume(self) -> None:
        self._pause_requested = False
        self._wake.set()

    def request_cancel(self, message: str = "cancelled by user") -> None:
        """Stop at the next checkpoint and end the scan as CANCELLED."""
        if self._stop is None:
            self._stop = StopRequest(StopKind.CANCEL, message)
        self._wake.set()

    def request_abort(self, message: str) -> None:
        """Emergency stop: end the scan as ERROR. Overrides a pending cancel."""
        if self._stop is None or self._stop.kind is StopKind.CANCEL:
            self._stop = StopRequest(StopKind.ABORT, message)
        self._wake.set()

    def checkpoint(self) -> None:
        """Raise :class:`ScanStopRequested` if a cancel or abort is pending."""
        if self._stop is not None:
            raise ScanStopRequested(self._stop)

    async def wait_while_paused(self) -> None:
        """Block while a pause is requested; raises if the scan is stopped meanwhile."""
        while self._pause_requested and self._stop is None:
            self._wake.clear()
            await self._wake.wait()
        self.checkpoint()
