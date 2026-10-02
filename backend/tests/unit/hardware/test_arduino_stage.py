"""ArduinoStage against the FirmwareEmulator through an EmulatorTransport."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from confocal.config import ArduinoConfig, MotionConfig, MotorConfig, Settings
from confocal.errors import (
    CommunicationError,
    HardwareNotConnectedError,
    LimitViolationError,
    MotionAbortedError,
    MotionError,
    MotionTimeoutError,
    ProtocolError,
)
from confocal.hardware.arduino.emulator import FIRMWARE_VERSION, FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    Command,
    Event,
    Response,
    encode_command,
    parse_request,
)
from confocal.hardware.arduino.stage import ArduinoStage
from confocal.hardware.arduino.steps import StepMapper
from confocal.hardware.arduino.transport import (
    EmulatorClock,
    EmulatorTransport,
    ManualClock,
    ScaledClock,
)
from confocal.hardware.kinematics import MotorSteps
from confocal.models.common import Axis, Position
from confocal.models.hardware import StageState

ORIGIN = Position(x_um=0.0, y_um=0.0, z_um=0.0)
SPEED = 600.0


def make_settings(*, backlash: int = 0) -> Settings:
    motor = MotorConfig(max_speed_steps_s=SPEED, backlash_steps=backlash)
    return Settings(
        arduino=ArduinoConfig(
            port="emulator", timeout_s=0.2, handshake_timeout_s=0.3, a=motor, b=motor, c=motor
        ),
        motion=MotionConfig(move_timeout_s=0.5, move_timeout_factor=1.5),
    )


class Rig:
    """A stage wired to an emulator through one EmulatorTransport."""

    def __init__(
        self,
        settings: Settings,
        clock: EmulatorClock,
        *,
        emulator: FirmwareEmulator | None = None,
        boot_banner: bool = True,
    ) -> None:
        self.settings = settings
        self.emulator = emulator or FirmwareEmulator.from_config(settings.arduino)
        self.transport = EmulatorTransport(self.emulator, clock, boot_banner=boot_banner)
        self.clock = clock
        self.stage = ArduinoStage.from_settings(settings, transport_factory=lambda: self.transport)

    @property
    def mapper(self) -> StepMapper:
        return self.stage.mapper

    def commands(self) -> list[Command]:
        return [parse_request(line).command for line in self.transport.sent]

    def moves(self) -> list[MotorSteps]:
        requests = [parse_request(line) for line in self.transport.sent]
        return [(r.args[0], r.args[1], r.args[2]) for r in requests if r.command is Command.MOVE]

    def manual(self) -> ManualClock:
        assert isinstance(self.clock, ManualClock)
        return self.clock


async def wait_for(predicate: Callable[[], bool], timeout_s: float = 2.0) -> None:
    """Poll ``predicate`` (state owned by the emulator, which has no event to wait on)."""
    async with asyncio.timeout(timeout_s):
        while True:
            if predicate():
                return
            await asyncio.sleep(0.001)


@pytest.fixture
def fast() -> Rig:
    """Emulated time runs 10 000 x faster than real time: moves finish at once."""
    return Rig(make_settings(), ScaledClock(speed=10_000.0))


@pytest.fixture
def manual() -> Rig:
    """Emulated time passes only when the test advances the clock."""
    return Rig(make_settings(), ManualClock())


# --------------------------------------------------------------------------- handshake


async def test_handshake_boot_ping_status_getpos(fast: Rig) -> None:
    await fast.stage.connect()
    assert fast.stage.connected
    assert fast.stage.version() == FIRMWARE_VERSION
    assert fast.commands() == [Command.PING, Command.STATUS, Command.GETPOS]
    status = await fast.stage.status()
    assert status.connected
    assert status.state is StageState.IDLE
    assert status.position == ORIGIN
    assert status.firmware_version == FIRMWARE_VERSION
    assert not status.homed
    await fast.stage.connect()  # idempotent
    assert fast.transport.opens == 1
    await fast.stage.close()
    assert not fast.stage.connected
    assert not fast.transport.is_open


async def test_missing_boot_banner_fails_the_handshake() -> None:
    rig = Rig(make_settings(), ManualClock(), boot_banner=False)
    with pytest.raises(CommunicationError, match="no BOOT banner"):
        await rig.stage.connect()
    assert not rig.stage.connected
    assert not rig.transport.is_open
    assert rig.transport.sent == []  # nothing was sent to a device that never booted
    status = await rig.stage.status()
    assert status.state is StageState.ERROR
    assert status.last_error is not None


async def test_bootloader_noise_before_the_banner_is_ignored() -> None:
    rig = Rig(make_settings(), ScaledClock(speed=10_000.0), boot_banner=False)
    connecting = asyncio.create_task(rig.stage.connect())
    await wait_for(lambda: rig.transport.is_open)
    rig.transport.inject(b"\x00\xfe garbage\n")
    rig.transport.inject(b"1 OK*00\n")  # a stale, corrupt reply from before the reset
    rig.transport.inject(Event.boot(FIRMWARE_VERSION).encode())
    await connecting
    assert rig.stage.connected


async def test_open_failure_is_a_communication_error(fast: Rig) -> None:
    fast.transport.fail_open = CommunicationError("cannot open serial port /dev/ttyACM0")
    with pytest.raises(CommunicationError, match="cannot open"):
        await fast.stage.connect()
    assert not fast.stage.connected


async def test_operations_before_connect_raise(fast: Rig) -> None:
    with pytest.raises(HardwareNotConnectedError):
        await fast.stage.get_position()
    with pytest.raises(HardwareNotConnectedError):
        await fast.stage.move_to(ORIGIN)
    await fast.stage.stop()  # never raises: there is nothing to stop
    status = await fast.stage.status()
    assert status.state is StageState.DISCONNECTED


# --------------------------------------------------------------------------- moves


async def test_move_round_trip_on_the_delta_kinematics(fast: Rig) -> None:
    await fast.stage.connect()
    target = Position(x_um=12.0, y_um=-7.5, z_um=4.0)
    reported = await fast.stage.move_to(target)
    expected_steps = fast.mapper.to_motor_steps(target)
    assert fast.emulator.position == expected_steps
    assert fast.moves()[-1] == expected_steps
    # The delta drives all three legs, and X/Y moves need opposite directions.
    a, b, c = expected_steps
    assert a != b
    assert len({a, b, c}) == 3
    assert reported == fast.mapper.to_position(expected_steps)
    assert reported.max_axis_error(target) <= fast.mapper.resolution_um()
    assert await fast.stage.get_position() == reported
    status = await fast.stage.status()
    assert status.state is StageState.IDLE
    assert status.position == reported


async def test_pure_z_move_drives_all_motors_equally(fast: Rig) -> None:
    await fast.stage.connect()
    await fast.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=6.0))
    a, b, c = fast.emulator.position
    assert a == b == c != 0


async def test_backlash_waypoints_are_sent() -> None:
    rig = Rig(make_settings(backlash=40), ScaledClock(speed=10_000.0))
    await rig.stage.connect()
    start = Position(x_um=0.0, y_um=0.0, z_um=10.0)
    await rig.stage.move_to(start)
    before = rig.emulator.position
    target = Position(x_um=8.0, y_um=0.0, z_um=10.0)  # +X: motor a up, motor b down
    goal = rig.mapper.to_motor_steps(target)
    expected = rig.mapper.plan_moves(before, goal, limits=rig.settings.limits)
    assert len(expected) == 2
    sent_before = len(rig.moves())
    reported = await rig.stage.move_to(target)
    assert rig.moves()[sent_before:] == expected
    assert rig.emulator.position == goal
    assert reported == rig.mapper.to_position(goal)


async def test_increasing_z_needs_no_backlash_waypoint() -> None:
    rig = Rig(make_settings(backlash=40), ScaledClock(speed=10_000.0))
    await rig.stage.connect()
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=1.0))
    count = len(rig.moves())
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=2.0))
    assert len(rig.moves()) == count + 1


async def test_zero_length_move_completes(fast: Rig) -> None:
    await fast.stage.connect()
    assert await fast.stage.move_to(ORIGIN) == ORIGIN


async def test_target_outside_the_limits_sends_nothing(fast: Rig) -> None:
    await fast.stage.connect()
    sent = len(fast.transport.sent)
    with pytest.raises(LimitViolationError):
        await fast.stage.move_to(Position(x_um=1e6, y_um=0.0, z_um=0.0))
    assert len(fast.transport.sent) == sent


async def test_concurrent_move_is_refused(manual: Rig) -> None:
    await manual.stage.connect()
    first = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=20.0)))
    await wait_for(lambda: manual.emulator.moving)
    with pytest.raises(MotionError, match="already in progress"):
        await manual.stage.move_to(ORIGIN)
    manual.manual().advance(60.0)
    await first


async def test_status_and_position_during_a_move(manual: Rig) -> None:
    await manual.stage.connect()
    target = Position(x_um=0.0, y_um=0.0, z_um=30.0)
    task = asyncio.create_task(manual.stage.move_to(target))
    await wait_for(lambda: manual.emulator.moving)
    assert manual.stage.moving
    duration = manual.emulator.move_duration_s(manual.mapper.to_motor_steps(target))
    manual.manual().advance(duration / 2)
    status = await manual.stage.status()
    assert status.state is StageState.MOVING
    halfway = await manual.stage.get_position()
    assert 0.0 < halfway.z_um < target.z_um
    manual.manual().advance(duration)
    reported = await task
    assert reported.max_axis_error(target) <= manual.mapper.resolution_um()
    assert not manual.stage.moving


# --------------------------------------------------------------------------- stop


async def test_stop_mid_move_aborts_at_the_partial_position(manual: Rig) -> None:
    await manual.stage.connect()
    target = Position(x_um=20.0, y_um=10.0, z_um=40.0)
    goal = manual.mapper.to_motor_steps(target)
    task = asyncio.create_task(manual.stage.move_to(target))
    await wait_for(lambda: manual.emulator.moving)
    manual.manual().advance(0.4 * manual.emulator.move_duration_s(goal))
    await manual.stage.stop()
    with pytest.raises(MotionAbortedError, match="stopped at"):
        await task
    partial = manual.emulator.position
    assert partial not in ((0, 0, 0), goal)
    assert manual.stage.last_steps == partial
    assert not manual.emulator.moving
    assert (await manual.stage.status()).state is StageState.STOPPED
    assert await manual.stage.get_position() == manual.mapper.to_position(partial)
    # The stage keeps working after a stop.
    follow_up = asyncio.create_task(manual.stage.move_to(ORIGIN))
    await wait_for(lambda: manual.emulator.moving)
    manual.manual().advance(60.0)
    assert await follow_up == ORIGIN


async def test_stop_wins_over_a_move_waiting_to_be_sent(manual: Rig) -> None:
    await manual.stage.connect()
    lock = manual.stage._writer_lock  # an exchange of another task is in progress
    await lock.acquire()
    move = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=5.0)))
    await asyncio.sleep(0.01)
    assert manual.stage.moving  # registered synchronously, now queued on the lock
    stop = asyncio.create_task(manual.stage.stop())
    await asyncio.sleep(0.01)
    lock.release()
    with pytest.raises(MotionAbortedError, match="before it was sent"):
        await move
    await stop
    assert Command.MOVE not in manual.commands()
    assert manual.commands()[-1] is Command.STOP
    assert manual.emulator.position == (0, 0, 0)


async def test_stop_when_idle_never_raises(fast: Rig) -> None:
    await fast.stage.connect()
    await fast.stage.stop()
    assert fast.commands()[-1] is Command.STOP
    await fast.stage.move_to(Position(x_um=1.0, y_um=0.0, z_um=0.0))  # stop does not stick


async def test_cancelled_move_stops_the_motors(manual: Rig) -> None:
    await manual.stage.connect()
    task = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=50.0)))
    await wait_for(lambda: manual.emulator.moving)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_for(lambda: not manual.emulator.moving)
    assert manual.commands()[-1] is Command.STOP


# --------------------------------------------------------------------------- firmware errors


async def test_e_range_from_the_firmware_is_a_limit_violation() -> None:
    settings = make_settings()
    narrow = MotorConfig(max_speed_steps_s=SPEED, min_steps=-100, max_steps=100)
    firmware = FirmwareEmulator.from_config(ArduinoConfig(a=narrow, b=narrow, c=narrow))
    rig = Rig(settings, ScaledClock(speed=10_000.0), emulator=firmware)
    await rig.stage.connect()
    with pytest.raises(LimitViolationError, match="E_RANGE"):
        await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=100.0))
    assert rig.emulator.position == (0, 0, 0)
    assert (await rig.stage.status()).state is StageState.IDLE
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=1.0))  # still usable


async def test_e_busy_from_the_firmware_is_a_motion_error(manual: Rig) -> None:
    await manual.stage.connect()
    # A move the host does not know about (e.g. left running after a crash).
    manual.emulator.handle_line(encode_command(999, Command.MOVE, (500, 500, 500)))
    with pytest.raises(MotionError, match="E_BUSY"):
        await manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=1.0))
    with pytest.raises(MotionError, match="E_BUSY"):
        await manual.stage.release()


async def test_host_side_travel_check_before_the_firmware() -> None:
    settings = make_settings()
    tight = MotorConfig(max_speed_steps_s=SPEED, min_steps=-50, max_steps=50)
    settings = settings.model_copy(
        update={"arduino": settings.arduino.model_copy(update={"a": tight, "b": tight, "c": tight})}
    )
    rig = Rig(settings, ScaledClock(speed=10_000.0))
    await rig.stage.connect()
    sent = len(rig.transport.sent)
    with pytest.raises(LimitViolationError, match="motor travel"):
        await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=100.0))
    assert len(rig.transport.sent) == sent


# --------------------------------------------------------------------------- link failures


async def test_corrupt_reply_is_a_protocol_error_then_resynchronises(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.corrupt_next()
    with pytest.raises(ProtocolError, match="corrupt"):
        await fast.stage.get_position()
    status = await fast.stage.status()
    assert status.state is StageState.ERROR
    assert status.last_error is not None
    sent = len(fast.transport.sent)
    assert await fast.stage.get_position() == ORIGIN
    assert fast.commands()[sent:] == [Command.PING, Command.GETPOS, Command.GETPOS]
    await fast.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=1.0))
    assert (await fast.stage.status()).state is StageState.IDLE


async def test_corrupt_done_event_fails_the_move(manual: Rig) -> None:
    await manual.stage.connect()
    task = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=10.0)))
    await wait_for(lambda: manual.emulator.moving)
    manual.transport.corrupt_next()
    manual.manual().advance(60.0)  # the DONE that now arrives is corrupted
    with pytest.raises(ProtocolError):
        await task
    assert (await manual.stage.status()).state is StageState.ERROR


async def test_reply_timeout_then_recovery(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.mute = True
    with pytest.raises(CommunicationError, match="no reply to GETPOS"):
        await fast.stage.get_position()
    assert fast.stage.connected  # out of step, not gone
    fast.transport.mute = False
    fast.transport.inject(Response.success(1, 7, 7, 7).encode())  # a late, stale reply
    assert await fast.stage.get_position() == ORIGIN


async def test_failed_resynchronisation_disconnects(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.mute = True
    with pytest.raises(CommunicationError):
        await fast.stage.get_position()
    with pytest.raises(CommunicationError, match="resynchronise"):
        await fast.stage.get_position()
    assert not fast.stage.connected
    with pytest.raises(HardwareNotConnectedError):
        await fast.stage.get_position()


async def test_out_of_sequence_reply_is_a_protocol_error(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.mute = True
    task = asyncio.create_task(fast.stage.get_position())
    await wait_for(lambda: len(fast.transport.sent) == 4)
    fast.transport.inject(Response.success(4242, 1, 2, 3).encode())
    with pytest.raises(ProtocolError, match="4242"):
        await task


async def test_missing_done_is_a_motion_timeout() -> None:
    settings = make_settings()
    settings = settings.model_copy(
        update={"motion": MotionConfig(move_timeout_s=0.05, move_timeout_factor=1.0)}
    )
    rig = Rig(settings, ManualClock())  # time never advances: DONE never comes
    await rig.stage.connect()
    with pytest.raises(MotionTimeoutError, match="no DONE"):
        await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=0.05))
    assert rig.commands()[-1] is Command.STOP
    assert not rig.emulator.moving
    assert (await rig.stage.status()).state is StageState.ERROR


async def test_unexpected_boot_means_position_lost(manual: Rig) -> None:
    await manual.stage.connect()
    task = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=10.0)))
    await wait_for(lambda: manual.emulator.moving)
    manual.transport.reset_device()
    with pytest.raises(MotionError, match="reset unexpectedly"):
        await task
    await wait_for(lambda: not manual.transport.is_open)
    assert not manual.stage.connected
    with pytest.raises(HardwareNotConnectedError, match="position is lost"):
        await manual.stage.get_position()
    status = await manual.stage.status()
    assert not status.connected
    assert status.state is StageState.ERROR
    assert status.last_error is not None
    assert "reset" in status.last_error


async def test_link_loss_disconnects(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.fail_link("USB unplugged")
    await wait_for(lambda: not fast.stage.connected)
    with pytest.raises(HardwareNotConnectedError, match="USB unplugged"):
        await fast.stage.move_to(Position(x_um=1.0, y_um=0.0, z_um=0.0))


async def test_reconnect_re_references_the_origin(fast: Rig) -> None:
    await fast.stage.connect()
    await fast.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=5.0))
    assert fast.emulator.position != (0, 0, 0)
    await fast.stage.close()
    await fast.stage.connect()  # reopening resets the Uno: its counters restart at 0
    assert fast.transport.opens == 2
    assert await fast.stage.get_position() == ORIGIN
    assert await fast.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=2.0)) != ORIGIN


async def test_reconnect_after_an_unexpected_reset(fast: Rig) -> None:
    await fast.stage.connect()
    fast.transport.reset_device()
    await wait_for(lambda: not fast.stage.connected)
    await wait_for(lambda: not fast.transport.is_open)
    await fast.stage.connect()
    assert fast.stage.connected
    assert (await fast.stage.status()).state is StageState.IDLE


# --------------------------------------------------------------------------- home / release


async def test_home_all_axes_uses_the_firmware_home(fast: Rig) -> None:
    await fast.stage.connect()
    await fast.stage.move_to(Position(x_um=5.0, y_um=3.0, z_um=8.0))
    assert await fast.stage.home() == ORIGIN
    assert fast.commands()[-1] is Command.HOME
    assert fast.emulator.position == (0, 0, 0)
    assert fast.stage.homed
    assert (await fast.stage.status()).homed


async def test_home_one_axis_moves_only_that_axis(fast: Rig) -> None:
    await fast.stage.connect()
    await fast.stage.move_to(Position(x_um=5.0, y_um=3.0, z_um=8.0))
    homed = await fast.stage.home([Axis.Z])
    assert homed.z_um == pytest.approx(0.0, abs=fast.mapper.resolution_um())
    assert homed.x_um == pytest.approx(5.0, abs=fast.mapper.resolution_um())
    assert fast.commands()[-1] is Command.MOVE
    assert not fast.stage.homed


async def test_home_with_backlash_finishes_with_home() -> None:
    rig = Rig(make_settings(backlash=30), ScaledClock(speed=10_000.0))
    await rig.stage.connect()
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=5.0))
    count = len(rig.moves())
    assert await rig.stage.home() == ORIGIN
    assert len(rig.moves()) == count + 1  # the overshoot below the origin
    assert rig.commands()[-1] is Command.HOME


async def test_release(manual: Rig) -> None:
    await manual.stage.connect()
    task = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=2.0)))
    await wait_for(lambda: manual.emulator.moving)
    with pytest.raises(MotionError, match="during a move"):
        await manual.stage.release()
    manual.manual().advance(60.0)
    await task
    assert manual.emulator.coils_enabled
    await manual.stage.release()
    assert not manual.emulator.coils_enabled


async def test_ping_and_firmware_status(fast: Rig) -> None:
    await fast.stage.connect()
    assert await fast.stage.ping() >= 0.0
    firmware = await fast.stage.firmware_status()
    assert firmware.version == FIRMWARE_VERSION
    assert await fast.stage.get_steps() == (0, 0, 0)


async def test_close_during_a_move_stops_it(manual: Rig) -> None:
    await manual.stage.connect()
    task = asyncio.create_task(manual.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=20.0)))
    await wait_for(lambda: manual.emulator.moving)
    await manual.stage.close()
    with pytest.raises(MotionError):
        await task
    assert not manual.emulator.moving
    assert (await manual.stage.status()).state is StageState.DISCONNECTED
