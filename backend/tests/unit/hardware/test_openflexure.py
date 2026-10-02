"""OpenFlexureStage against an in-memory fake OpenFlexure server."""

from __future__ import annotations

import asyncio

import pytest

from confocal.config import MotionConfig
from confocal.errors import (
    CommunicationError,
    HardwareNotConnectedError,
    LimitViolationError,
    MotionAbortedError,
    MotionError,
    MotionTimeoutError,
    MotionVerificationError,
)
from confocal.hardware.openflexure import OpenFlexureClient, OpenFlexureStage, StepConverter
from confocal.models.common import Axis, AxisLimits, Position, StageLimits
from confocal.models.hardware import StageState

LIMITS = StageLimits(
    x=AxisLimits(min_um=-1000.0, max_um=1000.0),
    y=AxisLimits(min_um=-1000.0, max_um=1000.0),
    z=AxisLimits(min_um=-200.0, max_um=200.0),
)
SCALES = {Axis.X: 0.1, Axis.Y: 0.2, Axis.Z: 0.05}


class FakeServer:
    """Implements OpenFlexureClient; moves take ``move_time_s`` and honour stop()."""

    def __init__(self, *, move_time_s: float = 0.0) -> None:
        self.steps: dict[Axis, int] = dict.fromkeys(Axis, 0)
        self.move_time_s = move_time_s
        self.moves: list[tuple[dict[Axis, int], bool]] = []
        self.stops = 0
        self.drift_steps = 0  # added to X after every move (a lost-step fault)
        self.fail_with: Exception | None = None
        self._stop = asyncio.Event()

    async def get_position_steps(self) -> dict[Axis, int]:
        self._maybe_fail()
        return dict(self.steps)

    async def move_steps(self, steps: dict[Axis, int], absolute: bool = True) -> None:
        self._maybe_fail()
        self.moves.append((dict(steps), absolute))
        self._stop.clear()
        start = dict(self.steps)
        goal = {axis: steps.get(axis, start[axis]) for axis in Axis}
        if not absolute:
            goal = {axis: start[axis] + steps.get(axis, 0) for axis in Axis}
        if self.move_time_s > 0:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.move_time_s)
            except TimeoutError:
                pass
            else:  # stopped half way
                self.steps = {axis: (start[axis] + goal[axis]) // 2 for axis in Axis}
                return
        self.steps = goal
        self.steps[Axis.X] += self.drift_steps

    async def stop(self) -> None:
        self.stops += 1
        self._stop.set()

    async def server_version(self) -> str:
        self._maybe_fail()
        return "v2.11.0"

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with


async def _stage(
    server: FakeServer,
    *,
    timeout_s: float = 60.0,
    speed_um_s: float = 40.0,
    connect: bool = True,
) -> OpenFlexureStage:
    stage = OpenFlexureStage(
        server,
        LIMITS,
        converter=StepConverter(SCALES),
        motion=MotionConfig(
            position_tolerance_um=0.5,
            move_timeout_s=timeout_s,
            xy_speed_um_s=speed_um_s,
            z_speed_um_s=speed_um_s,
        ),
    )
    if connect:
        await stage.connect()
    return stage


def test_fake_satisfies_the_client_protocol() -> None:
    assert isinstance(FakeServer(), OpenFlexureClient)


def test_step_converter() -> None:
    converter = StepConverter((0.1, 0.2, 0.05))
    assert converter.um_per_step == SCALES
    steps = converter.to_steps(Position(x_um=10.0, y_um=-3.05, z_um=0.025))
    assert steps == {Axis.X: 100, Axis.Y: -15, Axis.Z: 1}  # -15.25 -> -15, 0.5 -> 1
    assert converter.to_position(steps) == Position(x_um=10.0, y_um=-3.0, z_um=0.05)
    assert converter.resolution_um() == pytest.approx(0.1)
    with pytest.raises(CommunicationError):
        converter.to_position({Axis.X: 1})
    with pytest.raises(ValueError, match="positive"):
        StepConverter((0.1, 0.0, 0.1))
    with pytest.raises(ValueError, match="missing"):
        StepConverter({Axis.X: 0.1})
    with pytest.raises(ValueError, match="three"):
        StepConverter((0.1, 0.1))


async def test_connect_reads_version_and_position() -> None:
    server = FakeServer()
    server.steps = {Axis.X: 10, Axis.Y: 5, Axis.Z: -4}
    stage = await _stage(server)
    assert stage.connected
    assert stage.version() == "v2.11.0"
    assert await stage.get_position() == Position(x_um=1.0, y_um=1.0, z_um=-0.2)
    status = await stage.status()
    assert status.state is StageState.IDLE
    assert status.firmware_version == "v2.11.0"
    await stage.close()
    assert not stage.connected


async def test_requires_connection() -> None:
    stage = await _stage(FakeServer(), connect=False)
    with pytest.raises(HardwareNotConnectedError):
        await stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=0.0))


async def test_move_sends_absolute_steps_and_returns_reported_position() -> None:
    server = FakeServer()
    stage = await _stage(server)
    reached = await stage.move_to(Position(x_um=12.34, y_um=-8.0, z_um=1.01))
    assert server.moves == [({Axis.X: 123, Axis.Y: -40, Axis.Z: 20}, True)]
    assert reached == Position(x_um=12.3, y_um=-8.0, z_um=1.0)
    assert stage.converter.quantize(Position(x_um=12.34, y_um=-8.0, z_um=1.01)) == reached


async def test_limits_are_enforced_before_anything_is_sent() -> None:
    server = FakeServer()
    stage = await _stage(server)
    with pytest.raises(LimitViolationError):
        await stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=500.0))
    assert server.moves == []


async def test_position_mismatch_is_a_verification_error() -> None:
    server = FakeServer()
    server.drift_steps = 10  # 1 um off in X, tolerance 0.5 um
    stage = await _stage(server)
    with pytest.raises(MotionVerificationError, match=r"error 1.000"):
        await stage.move_to(Position(x_um=5.0, y_um=0.0, z_um=0.0))
    status = await stage.status()
    assert status.state is StageState.ERROR
    assert status.last_error is not None


async def test_stop_aborts_an_in_flight_move() -> None:
    server = FakeServer(move_time_s=5.0)
    stage = await _stage(server)
    task = asyncio.create_task(stage.move_to(Position(x_um=100.0, y_um=0.0, z_um=0.0)))
    await asyncio.sleep(0.01)
    await stage.stop()
    with pytest.raises(MotionAbortedError, match="stopped at"):
        await task
    assert server.stops == 1
    assert await stage.get_position() == Position(x_um=50.0, y_um=0.0, z_um=0.0)
    assert (await stage.status()).state is StageState.STOPPED


async def test_stop_when_idle_is_forwarded_and_harmless() -> None:
    server = FakeServer()
    stage = await _stage(server)
    await stage.stop()
    assert server.stops == 1
    assert await stage.move_to(Position(x_um=1.0, y_um=0.0, z_um=0.0)) is not None


async def test_timeout_stops_the_server() -> None:
    server = FakeServer(move_time_s=5.0)
    # 100 um at 10 mm/s: expected 0.01 s, timeout 2 x 0.01 + 0.02 s, the server hangs for 5 s.
    stage = await _stage(server, timeout_s=0.02, speed_um_s=10_000.0)
    with pytest.raises(MotionTimeoutError, match="did not finish"):
        await stage.move_to(Position(x_um=100.0, y_um=0.0, z_um=0.0))
    assert server.stops == 1


async def test_long_move_is_not_mistaken_for_a_stall() -> None:
    server = FakeServer(move_time_s=0.3)
    # 300 um at 1 mm/s: 0.3 s, far longer than the 0.05 s fixed margin.
    stage = await _stage(server, timeout_s=0.05, speed_um_s=1000.0)
    reached = await stage.move_to(Position(x_um=300.0, y_um=0.0, z_um=0.0))
    assert reached == Position(x_um=300.0, y_um=0.0, z_um=0.0)
    assert server.stops == 0


class SlowReadServer(FakeServer):
    """Position reads take a while once ``slow_reads`` is set (a busy server)."""

    slow_reads = False

    async def get_position_steps(self) -> dict[Axis, int]:
        if self.slow_reads:
            await asyncio.sleep(0.05)
        return await super().get_position_steps()


async def test_stop_while_home_reads_the_position_aborts_before_anything_moves() -> None:
    server = SlowReadServer()
    stage = await _stage(server)
    await stage.move_to(Position(x_um=10.0, y_um=20.0, z_um=5.0))
    server.slow_reads = True
    task = asyncio.create_task(stage.home())
    await asyncio.sleep(0.01)  # home() is reading the position, nothing sent yet
    await stage.stop()
    with pytest.raises(MotionAbortedError):
        await task
    assert len(server.moves) == 1  # only the first move: the homing move was never sent
    assert (await stage.status()).state is StageState.STOPPED


async def test_concurrent_move_is_refused() -> None:
    server = FakeServer(move_time_s=5.0)
    stage = await _stage(server)
    task = asyncio.create_task(stage.move_to(Position(x_um=100.0, y_um=0.0, z_um=0.0)))
    await asyncio.sleep(0.01)
    with pytest.raises(MotionError, match="in progress"):
        await stage.move_to(Position(x_um=0.0, y_um=1.0, z_um=0.0))
    await stage.stop()
    with pytest.raises(MotionAbortedError):
        await task


async def test_client_failures_become_communication_errors() -> None:
    server = FakeServer()
    stage = await _stage(server)
    server.fail_with = OSError("connection refused")
    with pytest.raises(CommunicationError, match="connection refused"):
        await stage.move_to(Position(x_um=1.0, y_um=0.0, z_um=0.0))
    with pytest.raises(CommunicationError):
        await stage.get_position()
    server.fail_with = CommunicationError("already typed")
    with pytest.raises(CommunicationError, match="already typed"):
        await stage.get_position()


async def test_home_returns_selected_axes_to_the_origin() -> None:
    server = FakeServer()
    stage = await _stage(server)
    await stage.move_to(Position(x_um=10.0, y_um=20.0, z_um=5.0))
    assert await stage.home([Axis.Z]) == Position(x_um=10.0, y_um=20.0, z_um=0.0)
    assert not (await stage.status()).homed
    assert await stage.home() == Position(x_um=0.0, y_um=0.0, z_um=0.0)
    assert (await stage.status()).homed
