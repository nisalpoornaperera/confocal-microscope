"""FirmwareEmulator: the executable specification of the Arduino Uno firmware."""

from __future__ import annotations

import pytest

from confocal.config import ArduinoConfig, MotorConfig
from confocal.hardware.arduino.emulator import FIRMWARE_VERSION, FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    Command,
    ErrorCode,
    Event,
    EventKind,
    FirmwareState,
    FirmwareStatus,
    Response,
    crc8,
    encode_command,
    parse_line,
)
from confocal.hardware.kinematics import Motor, MotorSteps

SPEED = 500.0  # steps / s on every motor


@pytest.fixture
def emulator() -> FirmwareEmulator:
    motor = MotorConfig(max_speed_steps_s=SPEED, min_steps=-1000, max_steps=1000)
    return FirmwareEmulator(dict.fromkeys(Motor, motor))


class Link:
    """Host side helper: numbered requests, parsed replies and events."""

    def __init__(self, emulator: FirmwareEmulator) -> None:
        self.emulator = emulator
        self.seq = 0

    def send(self, command: Command, *args: int) -> tuple[Response, list[Event]]:
        self.seq += 1
        messages = [
            parse_line(line)
            for line in self.emulator.handle_line(encode_command(self.seq, command, args))
        ]
        replies = [m for m in messages if isinstance(m, Response)]
        events = [m for m in messages if isinstance(m, Event)]
        assert len(replies) == 1
        assert replies[0].seq == self.seq
        return replies[0], events

    def advance(self, dt_s: float) -> list[Event]:
        events = [parse_line(line) for line in self.emulator.advance(dt_s)]
        assert all(isinstance(e, Event) for e in events)
        return [e for e in events if isinstance(e, Event)]

    # Fresh reads of the emulator state (methods, so type checkers do not narrow them).
    def state(self) -> FirmwareState:
        return self.emulator.state

    def moving(self) -> bool:
        return self.emulator.moving

    def coils(self) -> bool:
        return self.emulator.coils_enabled


@pytest.fixture
def link(emulator: FirmwareEmulator) -> Link:
    return Link(emulator)


def test_reset_sends_boot_banner_and_zeroes(link: Link) -> None:
    link.send(Command.MOVE, 10, 10, 10)
    link.advance(1.0)
    (banner,) = [parse_line(line) for line in link.emulator.reset()]
    assert banner == Event.boot(FIRMWARE_VERSION)
    assert link.emulator.position == (0, 0, 0)
    assert not link.coils()
    assert link.state() is FirmwareState.IDLE


def test_move_is_acknowledged_then_completed_by_event(link: Link) -> None:
    reply, events = link.send(Command.MOVE, 100, -50, 25)
    assert reply.ok
    assert reply.payload == ()
    assert events == []
    assert link.moving()
    assert link.state() is FirmwareState.MOVING
    assert link.emulator.move_duration_s((100, -50, 25)) == pytest.approx(100 / SPEED)

    assert link.advance(0.1) == []
    assert link.advance(0.1) == [Event.done(link.seq, (100, -50, 25))]
    assert link.emulator.position == (100, -50, 25)
    assert not link.moving()
    assert link.coils()


def test_zero_length_move_completes_immediately(link: Link) -> None:
    reply, events = link.send(Command.MOVE, 0, 0, 0)
    assert reply.ok
    assert events == [Event.done(link.seq, (0, 0, 0))]


def _expected_on_line(start: MotorSteps, delta: MotorSteps, fraction: float) -> list[float]:
    return [s + fraction * d for s, d in zip(start, delta, strict=True)]


def test_coordinated_move_follows_the_straight_line(link: Link) -> None:
    delta: MotorSteps = (300, -120, 45)
    link.send(Command.MOVE, *delta)
    total = 300
    previous: MotorSteps = (0, 0, 0)
    for tick in range(1, total + 1):
        link.advance(1.0 / SPEED)
        position = link.emulator.position
        ideal = _expected_on_line((0, 0, 0), delta, tick / total)
        assert all(abs(p - i) <= 0.5 + 1e-9 for p, i in zip(position, ideal, strict=True))
        assert all(abs(p - q) <= 1 for p, q in zip(position, previous, strict=True))
        previous = position
    assert link.emulator.position == delta
    assert not link.moving()


def test_all_motors_start_and_finish_together(link: Link) -> None:
    link.send(Command.MOVE, 400, 40, -400)
    link.advance(0.5 * 400 / SPEED)
    a, b, c = link.emulator.position
    assert (a, b, c) == (200, 20, -200)
    events = link.advance(0.5 * 400 / SPEED)
    assert events == [Event.done(link.seq, (400, 40, -400))]


def test_slowest_motor_limits_move_duration() -> None:
    motors = {
        Motor.A: MotorConfig(max_speed_steps_s=1000.0),
        Motor.B: MotorConfig(max_speed_steps_s=100.0),
        Motor.C: MotorConfig(max_speed_steps_s=1000.0),
    }
    emulator = FirmwareEmulator(motors)
    assert emulator.move_duration_s((500, 100, 0)) == pytest.approx(1.0)


def test_stop_aborts_at_the_interpolated_partial_position(link: Link) -> None:
    link.send(Command.MOVE, 200, -100, 50)
    move_seq = link.seq
    link.advance(0.5 * 200 / SPEED)
    reply, events = link.send(Command.STOP)
    assert events == [Event.aborted(move_seq, (100, -50, 25))]
    assert reply.ok
    assert reply.steps() == (100, -50, 25)
    assert link.state() is FirmwareState.STOPPED
    assert not link.moving()
    assert link.advance(10.0) == []
    assert link.emulator.position == (100, -50, 25)


def test_stop_while_idle_just_reports_position(link: Link) -> None:
    reply, events = link.send(Command.STOP)
    assert events == []
    assert reply.steps() == (0, 0, 0)
    assert link.state() is FirmwareState.IDLE


@pytest.mark.parametrize(
    ("command", "args"),
    [(Command.MOVE, (1, 1, 1)), (Command.HOME, ()), (Command.ZERO, ()), (Command.RELEASE, ())],
)
def test_busy_while_moving(link: Link, command: Command, args: tuple[int, ...]) -> None:
    link.send(Command.MOVE, 500, 500, 500)
    reply, _ = link.send(command, *args)
    assert not reply.ok
    assert reply.error is ErrorCode.E_BUSY
    assert link.moving()  # the running move is unaffected


@pytest.mark.parametrize("command", [Command.GETPOS, Command.STATUS, Command.PING])
def test_queries_are_answered_while_moving(link: Link, command: Command) -> None:
    link.send(Command.MOVE, 500, 500, 500)
    reply, _ = link.send(command)
    assert reply.ok


@pytest.mark.parametrize("target", [(1001, 0, 0), (0, -1001, 0), (0, 0, 5000)])
def test_out_of_travel_move_is_e_range_and_nothing_moves(
    link: Link, target: tuple[int, int, int]
) -> None:
    reply, events = link.send(Command.MOVE, *target)
    assert reply.error is ErrorCode.E_RANGE
    assert events == []
    assert not link.moving()
    assert link.emulator.position == (0, 0, 0)


def test_home_returns_to_origin_with_a_coordinated_move(link: Link) -> None:
    link.send(Command.MOVE, 100, 200, -300)
    link.advance(10.0)
    reply, _ = link.send(Command.HOME)
    assert reply.ok
    assert link.state() is FirmwareState.HOMING
    link.advance(150 / SPEED)
    assert link.emulator.position == (50, 100, -150)
    assert link.advance(10.0) == [Event.done(link.seq, (0, 0, 0))]
    assert link.state() is FirmwareState.IDLE


def test_zero_redefines_origin(link: Link) -> None:
    link.send(Command.MOVE, 10, 20, 30)
    link.advance(1.0)
    reply, _ = link.send(Command.ZERO)
    assert reply.steps() == (0, 0, 0)
    assert link.emulator.position == (0, 0, 0)
    reply, _ = link.send(Command.MOVE, 1000, 0, 0)  # travel is relative to the new origin
    assert reply.ok


def test_getpos_reports_interpolated_position(link: Link) -> None:
    link.send(Command.MOVE, 100, 0, -100)
    link.advance(0.25 * 100 / SPEED)
    reply, _ = link.send(Command.GETPOS)
    assert reply.steps() == (25, 0, -25)


def test_status_reports_state_flags_and_version(link: Link) -> None:
    reply, _ = link.send(Command.STATUS)
    status = FirmwareStatus.from_response(reply)
    assert status == FirmwareStatus(
        state=FirmwareState.IDLE,
        steps=(0, 0, 0),
        moving=False,
        coils_enabled=False,
        version=FIRMWARE_VERSION,
    )
    link.send(Command.MOVE, 100, 0, 0)
    status = FirmwareStatus.from_response(link.send(Command.STATUS)[0])
    assert status.state is FirmwareState.MOVING
    assert status.moving
    assert status.coils_enabled


def test_release_de_energises_coils(link: Link) -> None:
    link.send(Command.MOVE, 5, 5, 5)
    link.advance(1.0)
    assert link.coils()
    reply, _ = link.send(Command.RELEASE)
    assert reply.ok
    assert not link.coils()
    assert link.emulator.position == (5, 5, 5)


def test_ping(link: Link) -> None:
    reply, events = link.send(Command.PING)
    assert reply.ok
    assert events == []


def test_corrupt_line_gets_unattributed_e_crc(emulator: FirmwareEmulator) -> None:
    line = bytearray(encode_command(3, Command.MOVE, (1, 2, 3)))
    line[4] ^= 0x01
    (reply,) = [parse_line(raw) for raw in emulator.handle_line(bytes(line))]
    assert isinstance(reply, Response)
    assert reply.error is ErrorCode.E_CRC
    assert reply.seq == 0
    assert not emulator.moving


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("4 JUMP", ErrorCode.E_UNKNOWN),
        ("4 MOVE 1 2", ErrorCode.E_ARGS),
        ("4", ErrorCode.E_SYNTAX),
    ],
)
def test_bad_requests_get_error_replies(
    emulator: FirmwareEmulator, body: str, code: ErrorCode
) -> None:
    data = body.encode()
    (reply,) = [parse_line(raw) for raw in emulator.handle_line(data + b"*%02X\n" % crc8(data))]
    assert isinstance(reply, Response)
    assert reply.error is code
    assert reply.seq == 4


def test_oversized_line_gets_e_length(emulator: FirmwareEmulator) -> None:
    (reply,) = [parse_line(raw) for raw in emulator.handle_line(b"1 PING " + b"x" * 200 + b"\n")]
    assert isinstance(reply, Response)
    assert reply.error is ErrorCode.E_LENGTH


def test_from_config_and_validation() -> None:
    emulator = FirmwareEmulator.from_config(ArduinoConfig())
    assert emulator.version == FIRMWARE_VERSION
    with pytest.raises(ValueError, match="missing"):
        FirmwareEmulator({Motor.A: MotorConfig()})
    with pytest.raises(ValueError, match="protocol range"):
        FirmwareEmulator({m: MotorConfig(max_steps=10_000_001) for m in Motor})
    with pytest.raises(ValueError, match="dt_s"):
        emulator.advance(-1.0)


def test_boot_event_kind() -> None:
    assert Event.boot("2.0").kind is EventKind.BOOT
