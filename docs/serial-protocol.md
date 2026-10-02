# Serial protocol: Raspberry Pi ⇄ Arduino Uno motor controller

The Arduino Uno drives the three legs (motors **a**, **b**, **c**) of the
OpenFlexure Delta Stage through three ULN2003 boards and 28BYJ-48 steppers
(pin mapping: `docs/wiring.md` §3, firmware: `firmware/README.md`).
The firmware knows **only motor steps**. Micrometres, the delta kinematics
and backlash planning live on the Pi (`confocal.hardware.arduino.steps`).

Reference implementations, which this document describes and which win if
the two ever disagree:

| Side | Code |
|---|---|
| Codec (both directions) | `backend/confocal/hardware/arduino/protocol.py` |
| Device behaviour (executable spec of the firmware) | `backend/confocal/hardware/arduino/emulator.py` (`FirmwareEmulator`) |
| Tests | `backend/tests/unit/hardware/test_protocol.py`, `test_emulator.py` |
| Firmware (implements this document) | `firmware/confocal_stage/` - replayed natively against vectors recorded from the two files above (`firmware/test/`) |

## 1. Link

* USB CDC serial, **115200 baud, 8N1**, no flow control.
* Opening the port toggles DTR, which **resets the Uno** (auto-reset). The
  bootloader runs for ~1-2 s, then the firmware prints the BOOT banner
  (§5). The host waits up to `arduino.handshake_timeout_s` (default 5 s) for
  it, then sends `PING` and `STATUS`.
* One ASCII message per line, terminated by `\n` (a preceding `\r` is
  tolerated). At most **96 bytes per line**, terminator included.
* The host has **one request outstanding** at a time and waits for its reply
  (`arduino.timeout_s`, default 2 s). Events (§5) may arrive at any time,
  including between a request and its reply.

## 2. Framing and checksum

```
host → device    <seq> <CMD> [<arg> ...]*<CRC>\n
device → host    <seq> OK [<payload> ...]*<CRC>\n
                 <seq> ERR <CODE> [<message>]*<CRC>\n
                 ! <EVENT> [<payload> ...]*<CRC>\n
```

* Tokens are separated by **exactly one space**; no leading or trailing
  spaces. Only printable ASCII (0x20-0x7E) is allowed, and `*` only as the
  checksum separator.
* `<CRC>` is **CRC-8, polynomial 0x07, initial value 0x00, no reflection, no
  final XOR** over every byte before the `*`, written as two hex digits
  (upper case when sending; either case accepted). Check value:
  `crc8("123456789") = 0xF4`. It detects every single corrupted byte and every
  error burst up to 8 bits.
* The checksum is verified on the raw bytes *before* anything is parsed, so
  any corruption produces `E_CRC`.

```c
uint8_t crc8(const uint8_t *data, size_t n) {
  uint8_t crc = 0x00;
  while (n--) {
    crc ^= *data++;
    for (uint8_t i = 0; i < 8; i++)
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}
```

### Sequence numbers

* `<seq>` is a decimal integer **1..65535**, chosen by the host, rolling over
  from 65535 to 1. Every reply carries the `seq` of its request.
* The device replies with **seq 0** when it cannot read the request's own
  number: bad checksum, line too long, a forbidden character or wrong
  spacing anywhere in the line (checked before any token is read), or an
  unreadable or out-of-range sequence number. Because
  only one request is outstanding, the host attributes such an `ERR` to it.
* A reply with any other `seq` than the outstanding one means the link is out
  of step (e.g. a late reply after a timeout): the host raises `ProtocolError`
  and resynchronises (`PING`) before continuing.

### Numbers

Step values are decimal integers with an optional `-`, at most 7 digits
(±9 999 999; they fit the Uno's 32-bit `long`). Leading zeros and `-0` are
accepted (`0012` is 12); so are leading zeros in `<seq>` (`007 PING` is
answered `7 OK`), which is 1-5 digits. Replies never contain leading zeros.
Motor travel configured in
`[arduino.a|b|c]` must lie within that range.

## 3. Commands

All positions are **absolute firmware step counters** of motors a, b, c,
counted from the power-up origin (the stage has no end-stops).

| Command | Reply | Notes |
|---|---|---|
| `MOVE <a> <b> <c>` | `OK`, later event `! DONE <seq> <a> <b> <c>` | Coordinated move (§4). Acknowledged at once; completion is the event. `E_RANGE` if any target is outside that motor's travel (nothing moves); `E_BUSY` while a move runs. A zero-length move replies `OK` and `DONE` immediately. Energises the coils. |
| `GETPOS` | `OK <a> <b> <c>` | Current counters; during a move, the interpolated position reached so far. |
| `HOME` | `OK`, later `! DONE <seq> 0 0 0` | Coordinated move back to step 0 on every motor. `E_BUSY` while moving. |
| `STOP` | `! ABORTED <seq_of_move> <a> <b> <c>` (only if a move was running), then `OK <a> <b> <c>` | Halts within one step period, leaves the motors where the interpolation had got to, state `stopped`. Never fails when idle. Coils stay energised. |
| `STATUS` | `OK state=<s> a=<a> b=<b> c=<c> moving=<0\|1> enabled=<0\|1> version=<v>` | `state` ∈ `idle`, `moving`, `homing`, `stopped` (the last move was pre-empted by STOP); `enabled` = coils energised. Allowed while moving. |
| `PING` | `OK` | Liveness / resynchronisation. Allowed while moving. |
| `ZERO` | `OK 0 0 0` | Defines the current position as the origin (all counters 0). `E_BUSY` while moving. |
| `RELEASE` | `OK` | De-energises all coils (§6). `E_BUSY` while moving. |

Only `GETPOS`, `STATUS`, `PING` and `STOP` are accepted while a move runs.

## 4. Coordinated, interpolated, non-blocking moves

On the Delta Stage a pure Z move turns all three motors by the same amount,
but every X/Y move turns at least two motors in **opposite** directions. If the
motors ran independently at full speed, the platform would leave the straight
line and wander sideways (and the Z of a scan point would be wrong while the
legs are out of step). The firmware therefore interpolates the three motors
so they **start and finish together**:

* For a move by `d = (da, db, dc)` steps, `N = max(|da|, |db|, |dc|)` ticks are
  executed (multi-axis Bresenham). Every motor has an error accumulator
  initialised to `N / 2` (integer division); at each tick it adds `|d_i|`, and
  whenever the accumulator reaches `N` the motor takes one step towards its
  target and `N` is subtracted. Closed form: after `k` ticks motor `i` has taken
  `floor((k·|d_i| + N div 2) / N)` steps. No motor ever steps more than once
  per tick, all arrive at tick `N`, and every intermediate position lies within
  half a step of the straight line from start to target.
* Duration `T = max_i(|d_i| / max_speed_steps_s_i)`; ticks are spaced `T / N`
  apart. The motor that is furthest from its target relative to its own speed
  limit runs at its maximum rate, the others proportionally slower. (With
  equal limits on all motors - the normal setup - no motor ever exceeds its
  limit. If the limits differ, every motor stays within its own limit on
  *average* over the move, but a motor other than the one setting the pace
  may step on two consecutive ticks, i.e. briefly at the tick rate, which is
  at most the pacing motor's limit.)
* **Non-blocking:** stepping is driven from `loop()` by comparing `micros()`
  with the next tick time - never `delay()`. Serial input is read and handled
  between ticks, so `STOP`, `GETPOS`, `STATUS` and `PING` are answered during a
  move and `STOP` pre-empts it within one tick period.
* **Firmware travel limits** (`min_steps`/`max_steps` per motor, compiled in
  from `firmware/confocal_stage/config.h`, which must match `[arduino.a|b|c]`)
  are checked on the *target* before a
  move starts. The path is a straight line between two in-range points, so it
  never leaves the range. This is the last line of defence below the Pi's own
  Cartesian and per-motor checks (`docs/architecture.md` §2).

Backlash is *not* handled by the firmware: the Pi splits a move into
waypoints so every motor finishes travelling in the + direction
(`plan_backlash_moves`) and sends one `MOVE` per waypoint.

## 5. Events

| Event | Meaning |
|---|---|
| `! DONE <seq> <a> <b> <c>` | The move started by request `<seq>` (`MOVE` or `HOME`) reached its target; final counters. |
| `! ABORTED <seq> <a> <b> <c>` | `STOP` pre-empted the move started by `<seq>`; counters where it stopped. Sent just before the `STOP` reply. |
| `! BOOT <version>` | The firmware (re)started: counters are 0, coils off, state `idle`. `<version>` is 1-8 characters of `[0-9A-Za-z._+-]`. |

A `BOOT` that arrives other than right after opening the port means the Uno
reset (brown-out, USB glitch, watchdog): **the position is lost**. The host
must treat it as a hardware failure (`MotionError`), abort any scan and require
the operator to re-reference the stage.

## 6. Motors and coils

* 28BYJ-48 driven in **half-step** mode (8 phases), coil order IN1..IN4 of the
  ULN2003 board:

  | phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | IN1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
  | IN2 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
  | IN3 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
  | IN4 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

  Each motor has a phase index (0..7, starting at 0 at power-up): a + step
  advances it, a − step goes back. It is kept separately from the step
  counter, so `ZERO` (which resets the counters) never makes a motor jump
  phase.
  About 4076 half-steps per output-shaft revolution (gear ratio ≈ 63.7 : 1).
  Speeds up to ~600-1000 half-steps/s are reliable at 5 V
  (`max_speed_steps_s`, default 600). Mirrored mounting is handled on the Pi
  (`invert`), not in the firmware.
* `RELEASE` drives all four inputs of every board low. Held coils draw roughly
  0.2-0.25 A per motor and warm the stage, which causes thermal drift; the Pi
  releases them when idle (`motion.release_motors_when_idle`). The gearbox is
  not back-drivable in practice, so the stage keeps its position.
* The phase index is kept while released; the next move first re-energises the
  current phase, so no step is lost or gained.

## 7. Error codes

| Code | Cause | `seq` in the reply |
|---|---|---|
| `E_CRC` | checksum missing (including a blank line), not two hex digits, or wrong | 0 |
| `E_LENGTH` | line longer than 96 bytes | 0 |
| `E_SYNTAX` | forbidden character (non-ASCII, control, `*` in the body), leading / trailing / double space, empty message (`*00`), bad or out-of-range sequence number, missing command | 0, except for a missing command (request's) |
| `E_UNKNOWN` | unknown command word | request's |
| `E_ARGS` | wrong number or format of arguments, step value beyond ±9 999 999 | request's |
| `E_BUSY` | `MOVE`, `HOME`, `ZERO` or `RELEASE` while a move runs | request's |
| `E_RANGE` | `MOVE` target outside a motor's travel; nothing moved | request's |

The firmware's error message is a fixed short text (it never echoes received
bytes). On the Pi, `E_RANGE` maps to `LimitViolationError`, `E_BUSY` to
`MotionError` and every other code to `ProtocolError`.

## 8. Example session

Generated by `FirmwareEmulator` (motor speed 600 steps/s):

```
<- ! BOOT 1.0.0*D2
-> 1 PING*F7
<- 1 OK*92
-> 2 STATUS*C9
<- 2 OK state=idle a=0 b=0 c=0 moving=0 enabled=0 version=1.0.0*4D
-> 3 MOVE 1200 -600 300*EB
<- 3 OK*BE
   (1 s later)
-> 4 GETPOS*9E
<- 4 OK 600 -300 150*7A
-> 5 STOP*2A
<- ! ABORTED 3 600 -300 150*66
<- 5 OK 600 -300 150*74
-> 6 MOVE 0 0 0*1B
<- 6 OK*F0
-> 7 MOVE 1 1 1*48
<- 7 ERR E_BUSY moving*E1
<- ! DONE 6 0 0 0*9D
-> 8 MOVE 300000 0 0*70
<- 8 ERR E_RANGE outside travel*A9
-> 9 RELEASE*03
<- 9 OK*22
-> 10 MOVE 1 2 3*00
<- 0 ERR E_CRC bad checksum*2F
```
