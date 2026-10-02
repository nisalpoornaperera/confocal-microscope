"""The slice of an OpenFlexure server that the stage adapter needs.

The OpenFlexure server (``openflexure-microscope-server``) exposes its stage
as integer *step* coordinates ``x``, ``y``, ``z``. For a Delta Stage the server
applies the delta transform itself (``SangaDeltaStage``): its x/y/z are
Cartesian "stage steps", and moving +z raises all three legs together. The
adapter therefore needs only a per-axis micrometre scale, not the delta
geometry (compare ``StageKinematics.openflexure_delta``, whose
``um_per_step`` is the same per-axis scale).

This module only defines the interface. A concrete HTTP client (Phase 11)
will implement it with the server's REST API (stage position / move / stop
actions) and must turn transport failures into
:class:`~confocal.errors.CommunicationError`. Tests use an in-memory fake.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from confocal.models.common import Axis


@runtime_checkable
class OpenFlexureClient(Protocol):
    """Asynchronous access to an OpenFlexure server's stage, in server steps."""

    async def get_position_steps(self) -> dict[Axis, int]:
        """Current stage position as reported by the server (all three axes)."""
        ...

    async def move_steps(self, steps: dict[Axis, int], absolute: bool = True) -> None:
        """Move to (``absolute``) or by the given step coordinates; returns when motion ends.

        Axes missing from ``steps`` are not moved. A concurrent :meth:`stop`
        makes this return early, wherever the stage is.
        """
        ...

    async def stop(self) -> None:
        """Halt any motion as soon as possible. Must not fail when idle."""
        ...

    async def server_version(self) -> str:
        """Server software version; also serves as a connectivity check."""
        ...
