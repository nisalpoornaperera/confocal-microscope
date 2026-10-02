"""Arduino Uno motor controller of the Delta Stage (3 x ULN2003 + 28BYJ-48).

* :mod:`.steps`     - the only micrometre <-> motor-step conversion, travel
  checks and backlash planning;
* :mod:`.protocol`  - the CRC-framed line protocol codec (``docs/serial-protocol.md``);
* :mod:`.emulator`  - the reference firmware model (executable specification);
* :mod:`.transport` - the USB-serial link (pyserial, reader thread) and an
  in-memory link to the emulator;
* :mod:`.stage`     - ``ArduinoStage``, the :class:`~confocal.hardware.base.Stage`
  built on all of the above.
"""

from confocal.hardware.arduino.emulator import FIRMWARE_VERSION, FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    SEQ_MAX,
    SEQ_MIN,
    STEP_LIMIT,
    UNATTRIBUTED_SEQ,
    Command,
    ErrorCode,
    Event,
    EventKind,
    FirmwareState,
    FirmwareStatus,
    MalformedMessageError,
    Request,
    Response,
    SequenceCounter,
    crc8,
    encode_command,
    frame,
    match_reply,
    next_seq,
    parse_line,
    parse_request,
    unframe,
)
from confocal.hardware.arduino.stage import ArduinoStage, serial_transport_factory
from confocal.hardware.arduino.steps import StepMapper, motor_configs, plan_backlash_moves
from confocal.hardware.arduino.transport import (
    EmulatorTransport,
    ManualClock,
    ScaledClock,
    SerialTransport,
    Transport,
    TransportClosedError,
    TransportTimeoutError,
)

__all__ = [
    "FIRMWARE_VERSION",
    "MAX_LINE_BYTES",
    "SEQ_MAX",
    "SEQ_MIN",
    "STEP_LIMIT",
    "UNATTRIBUTED_SEQ",
    "ArduinoStage",
    "Command",
    "EmulatorTransport",
    "ErrorCode",
    "Event",
    "EventKind",
    "FirmwareEmulator",
    "FirmwareState",
    "FirmwareStatus",
    "MalformedMessageError",
    "ManualClock",
    "Request",
    "Response",
    "ScaledClock",
    "SequenceCounter",
    "SerialTransport",
    "StepMapper",
    "Transport",
    "TransportClosedError",
    "TransportTimeoutError",
    "crc8",
    "encode_command",
    "frame",
    "match_reply",
    "motor_configs",
    "next_seq",
    "parse_line",
    "parse_request",
    "plan_backlash_moves",
    "serial_transport_factory",
    "unframe",
]
