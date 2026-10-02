# Arduino Uno motor firmware

The Uno drives the three legs (motors **a**, **b**, **c**) of the OpenFlexure
Delta Stage through three ULN2003 boards and 28BYJ-48 steppers, and talks to
the Raspberry Pi over USB serial (115200 8N1) with the CRC-framed line
protocol in [`docs/serial-protocol.md`](../docs/serial-protocol.md).

The firmware works in motor steps only. Micrometres, the delta kinematics and
backlash handling are done on the Pi (`confocal.hardware.arduino.steps`).

**Specification:** `backend/confocal/hardware/arduino/protocol.py` (codec) and
`backend/confocal/hardware/arduino/emulator.py` (`FirmwareEmulator`, the
device behaviour). For any sequence of input lines the firmware sends the same
bytes as the emulator. This is checked natively by replaying vectors recorded
from the Python code (see [Tests](#tests)), so the Pi-side `ArduinoStage`
driver that is tested against the emulator works with the real board unchanged.

Version: **1.0.0** (sent in the `! BOOT` banner and the `STATUS` reply).

## Files

| Path | Contents |
|---|---|
| `confocal_stage/confocal_stage.ino` | `setup()` / `loop()`: the non-blocking main loop |
| `confocal_stage/config.h` | **Machine constants**: travel limits, speeds, pins, version, watchdog |
| `confocal_stage/protocol.h/.cpp` | CRC-8, line assembly, request parsing, reply framing (hardware independent) |
| `confocal_stage/motion.h/.cpp` | Multi-axis Bresenham interpolation and wrap-safe `micros()` tick scheduler (hardware independent) |
| `confocal_stage/controller.h/.cpp` | Command dispatch and motion state machine (hardware independent) |
| `confocal_stage/coils.h/.cpp` | ULN2003 coil outputs |
| `confocal_stage/serial_link.h/.cpp` | Non-blocking transmit queue |
| `confocal_stage/pgm_compat.h` | Flash-string shim so the core also compiles on a PC |
| `build.py` | Build (and optionally flash) without the Arduino IDE |
| `test/` | Native tests and the bench check (see [Tests](#tests)) |

## Pin mapping (must match `docs/wiring.md` §3)

| ULN2003 board | IN1 | IN2 | IN3 | IN4 |
|---|---|---|---|---|
| Motor A (leg A) | D2 | D3 | D4 | D5 |
| Motor B (leg B) | D6 | D7 | D8 | D9 |
| Motor C (leg C) | A0 | A1 | A2 | A3 (used as digital outputs) |

* **D0 / D1 are reserved** for the USB serial link to the Pi. Never use them.
* **D13 is not used.** The bootloader blinks the on-board LED on it at every
  reset, which would twitch a motor connected there.
* `setup()` drives every coil output LOW before anything else, so the motors
  are de-energised from the first instant after a reset.

> **Power:** the Uno is powered **only** by the Pi's USB cable. The ULN2003
> boards get 5 V from the **fused** motor rail through the motor switch (your
> hardware emergency stop), and one ground wire joins the motor supply to the
> Uno. Never feed 5 V into the Uno's 5V pin, and never run the motors from the
> Uno's 5V pin. Fusing, grounding and the mains safety notes are in
> [`docs/wiring.md`](../docs/wiring.md) §1-2. Read them before you power anything.

## Configuration: `confocal_stage/config.h`

The travel limits and speeds compiled into the firmware are its last line of
defence. They **must equal** the `[arduino.a]`, `[arduino.b]` and `[arduino.c]`
sections of the Pi's configuration (`config/confocal.pi.toml` or
`/etc/confocal/confocal.toml`). The values shipped are the backend defaults:

| Constant | Default | Pi setting |
|---|---|---|
| `MOTOR_x_MIN_STEPS` | -200000 | `[arduino.x] min_steps` |
| `MOTOR_x_MAX_STEPS` | 200000 | `[arduino.x] max_steps` |
| `MOTOR_x_MAX_SPEED` | 600.0 half-steps/s | `[arduino.x] max_speed_steps_s` |

`config/confocal.pi.toml` uses these defaults, so nothing needs changing for
it. If you change the Pi values, change `config.h` the same way and re-flash.
`test/bench_check.py` tells you if they disagree. Compile-time checks reject
travel that doesn't include 0, travel beyond ±9 999 999 steps, speeds outside
1-2000 steps/s, and version strings longer than 8 characters.

`CONFOCAL_USE_WATCHDOG` (default 1) enables a 1 s hardware watchdog. If the
loop ever hangs, the Uno resets and the Pi sees an unexpected `! BOOT`, which
it treats as lost position. Set it to 0 only for a clone with an old,
non-Optiboot bootloader that loops after a watchdog reset.

## Build and flash

Close anything that has the Uno's serial port open before uploading: the
confocal server (`sudo systemctl stop confocal`), the Arduino IDE serial
monitor, or a running `bench_check.py`.

### Arduino IDE 2 (Windows / macOS / Linux PC)

1. Install **Arduino AVR Boards** in the Boards Manager (it is usually there
   already).
2. Open `firmware/confocal_stage/confocal_stage.ino`. The IDE opens the whole
   folder, including all `.h`/`.cpp` tabs.
3. **Tools → Board → Arduino AVR Boards → Arduino Uno**, then
   **Tools → Port** → the Uno's port (`COMx` on Windows).
4. **Upload**. The IDE reports about 8.2 KB of flash (25 %) and 714 bytes of
   RAM (34 %).
5. Optional: **Serial Monitor** at 115200 baud, line ending "Newline". After
   each reset it shows `! BOOT 1.0.0*D2`. Type `1 PING*F7` and the reply is
   `1 OK*92`. Close the monitor afterwards.

### `build.py` (PC or Pi, no IDE needed)

`build.py` uses only the Python standard library. It finds avr-gcc and the
Arduino AVR core automatically: the Arduino IDE 2 / arduino-cli installation
(`%LOCALAPPDATA%\Arduino15`, `~/.arduino15`, `~/Library/Arduino15`), `PATH`,
or `/usr/share/arduino`. You can also pass `--toolchain` / `--core`.

```bash
python firmware/build.py                          # -> firmware/build/confocal_stage.hex
python firmware/build.py --flash --port COM5      # build + upload (Windows)
uv run --directory backend python ../firmware/build.py   # same, from the backend env
```

It compiles the Arduino core and the sketch for the ATmega328P at 16 MHz,
links with `--gc-sections`, and writes `firmware/build/confocal_stage.hex`.
It then prints flash and RAM use and fails if either is over the Uno's limit.
`--strict` treats warnings in the sketch as errors, and `-v` prints every
command. `--flash` runs
`avrdude -c arduino -p atmega328p -b 115200 -D -U flash:w:<hex>:i` on `--port`.

### On the Raspberry Pi

The Arduino IDE 2 has no official Raspberry Pi build. Use one of these:

* **apt toolchain + build.py**

  ```bash
  sudo apt install gcc-avr avr-libc avrdude arduino-core-avr
  python3 firmware/build.py --flash --port /dev/serial/by-id/usb-Arduino*   # the Uno's by-id path
  ```

* **arduino-cli** (official ARM64 build), either with build.py, which finds
  `~/.arduino15`, or on its own:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=$HOME/.local/bin sh
  arduino-cli core update-index && arduino-cli core install arduino:avr
  arduino-cli compile --fqbn arduino:avr:uno firmware/confocal_stage
  arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyACM0 firmware/confocal_stage
  ```

* **Flash a `.hex` built on the PC** (needs only `sudo apt install avrdude`):

  ```bash
  python3 firmware/build.py --flash --port /dev/ttyACM0 --hex confocal_stage.hex
  ```

Your user must be in the `dialout` group to open the port
(`sudo usermod -aG dialout $USER`, then log in again). Genuine Unos appear as
`/dev/ttyACM*` and CH340 clones as `/dev/ttyUSB*`. `ls /dev/serial/by-id/`
shows the stable name.

## Behaviour (summary of `docs/serial-protocol.md`)

* **Commands:** `MOVE a b c` (acknowledged at once, then `! DONE`), `GETPOS`,
  `HOME`, `STOP` (`! ABORTED` then `OK` at the partial position), `STATUS`,
  `PING`, `ZERO`, `RELEASE`. During a move only `GETPOS`, `STATUS`, `PING` and
  `STOP` are accepted; `MOVE`, `HOME`, `ZERO` and `RELEASE` get `E_BUSY`.
  A target outside a motor's travel gets `E_RANGE`, and nothing moves.
* **Coordinated moves:** a multi-axis Bresenham interpolation over
  `N = max |d_i|` ticks spaced `T / N`, where `T = max |d_i| / speed_i`. The
  three motors start and finish together and stay on the straight line. The
  tick interval is kept to 1/256 µs, rounded up, so no motor exceeds its limit.
* **Non-blocking:** `loop()` runs one tick whenever `micros()` reaches its due
  time, then reads serial input, at most one complete line per pass. `STOP`
  therefore takes effect before the next step. Replies go through a 192-byte
  queue that is handed to the UART only as fast as it has room, so even the
  92-byte `STATUS` reply never stalls stepping. If the loop is ever more than a
  whole interval late, the schedule restarts from "now" instead of bursting
  steps.
* **Coils:** half-step sequence (8 phases, IN1..IN4) on all three motors. All
  motors are energised from the start of a move (including a zero-length one)
  and stay energised after `DONE` and `STOP`. `RELEASE` and every reset drive
  all twelve outputs LOW. Each motor's phase index is kept separately from its
  step counter, so `ZERO` never makes a motor jump. The next move re-applies
  the current phase after `RELEASE`, so no step is lost.
* **No automatic coil release:** the Pi releases the coils when idle
  (`motion.release_motors_when_idle`). An automatic timeout in the firmware
  would make its `STATUS` differ from the emulator's.
* **Robustness:** lines are assembled into a fixed 95-byte buffer, and a longer
  line is discarded up to its `\n` and answered `0 ERR E_LENGTH`. Garbage,
  NUL bytes, bad checksums and wrong spacing each get one error reply and
  never desynchronise the link. Opening the port resets the Uno (DTR), and the
  firmware then sends `! BOOT 1.0.0` with all counters 0 and the coils off.
* **Resources:** about 9.3 KB of flash (8.2 KB with the IDE's LTO) and 714
  bytes of static RAM, leaving 1.3 KB for the stack. No `String`, no heap; the
  constant strings are kept in flash.

## Tests

**Native tests (PC or Pi, no board).** These compile `protocol.cpp`,
`motion.cpp` and `controller.cpp` for the host and replay
`test/vectors.txt` against them:

```bash
uv run --directory backend python ../firmware/test/run_tests.py   # regenerates vectors first
python3 firmware/test/run_tests.py                                # uses the committed vectors
```

`gen_vectors.py` records the vectors from the Python reference. They cover
CRC values, Bresenham positions after k ticks, tick intervals, and 17
complete serial sessions. Those sessions include the example in
`serial-protocol.md` §8, every case of `test_emulator.py`, per-motor speed
limits, the full ±9 999 999 range, sequence number rollover, every framing and
syntax error, and 12 randomised sessions.

Every emitted byte must match. The harness also checks that each step moves
exactly the stepping motor's coils to the adjacent half-step phase, that
coils are on during moves and off after `RELEASE` and boot, and that no motor
steps faster than the tick rate. The runner uses `$CXX`, `c++`, `g++` or
`clang++`, or falls back to Zig's clang via `uvx --from ziglang`.

**Bench check (real board, after flashing).** Turn the motor switch **off**
for the first run:

```bash
uv run --directory backend python ../firmware/test/bench_check.py --port /dev/ttyACM0
uv run --directory backend python ../firmware/test/bench_check.py --port /dev/ttyACM0 --moves
```

The check sends 34 requests, valid and malformed, and compares each reply with
the emulator byte for byte. That includes `E_RANGE` just beyond each motor's
configured travel, so a `config.h` that doesn't match the Pi's settings shows
up here. `--moves` adds small real moves: `DONE`, `STOP` → `ABORTED`, `HOME`
and `RELEASE`. `--self-test` runs the check against an in-process emulator.

## First power-on checks

1. Check the wiring against `docs/wiring.md` §6 with the PSU unplugged: no
   shorts, correct polarity, fuses fitted.
2. Flash the firmware with the motor switch **off**. Run `bench_check.py`
   without `--moves`; every check must pass.
3. Switch the motor rail on, with the stage's legs free to move or the motors
   off the stage. Run `bench_check.py --moves`: each motor turns briefly, a
   longer move is stopped after 1 s, then everything returns home. Each
   28BYJ-48 should turn smoothly and quietly. If a motor only buzzes or
   jerks back and forth, its IN1..IN4 wires are out of order.
4. Check each axis's direction on the Pi (deployment guide). A motor that turns
   the wrong way is fixed with `invert = true` in `[arduino.x]` of the Pi
   configuration, never by rewiring or changing the firmware.
5. On the stage, start with small moves and tight travel limits, and keep
   the motor switch within reach.

## Known limitations

* The stage has no end-stops. Positions count from the power-up origin, and
  any reset (USB glitch, watchdog, brown-out) loses the position. The Pi
  detects this from the unexpected `! BOOT`.
* Travel limits and speeds are compile-time constants, so changing them needs a
  re-flash. There is no runtime configuration command because the protocol has
  none.
* With *unequal* per-motor speed limits, a motor that is not setting the pace
  stays within its limit on average, but may briefly step at the tick rate
  (see `serial-protocol.md` §4). With equal limits, the normal setup, this
  cannot happen.
* Extreme moves (millions of steps over hours) can drift from the emulator's
  ideal timing by about 1 part in 10^5, because of the interval rounding and
  float arithmetic. Positions and step counts are exact.
