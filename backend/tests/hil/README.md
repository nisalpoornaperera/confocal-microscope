# Hardware-in-the-loop tests

These tests talk to the real Arduino Uno and ADS1115. They are part of the
normal `pytest` run but **skip themselves** unless the environment variables
below are set, so they never touch hardware on a development PC or in CI.

| Variable | Meaning |
|---|---|
| `CONFOCAL_HIL_PORT` | Serial port of a Uno flashed with the confocal firmware, e.g. `/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_...-if00`. Enables `test_hil_arduino_protocol.py` and `test_hil_arduino_stage.py`. |
| `CONFOCAL_HIL_I2C` | I2C bus number of the ADS1115 (`1` on the Pi). Enables `test_hil_ads1115.py`. |
| `CONFOCAL_HIL_I2C_ADDRESS` | ADS1115 address, default `0x48`. |
| `CONFOCAL_HIL_ADC_CHANNEL` | Input channel, default `0` (OPT101 on A0). |
| `CONFOCAL_CONFIG` | Optional machine configuration (motor travel and speeds, kinematics, limits). Without it the built-in defaults are used. |

## What they check

* `test_hil_arduino_protocol.py` - protocol conformance at the line level:
  the BOOT banner after the DTR reset, every command (`PING`, `STATUS`,
  `GETPOS`, `ZERO`, `RELEASE`, `MOVE` + `DONE`, `HOME`, `STOP` + `ABORTED`),
  `E_BUSY` for every command refused during a move, and every error code
  (`E_CRC`, `E_LENGTH`, `E_SYNTAX`, `E_UNKNOWN`, `E_ARGS`, `E_RANGE`). Error
  replies and idle commands are compared **byte for byte** with
  `FirmwareEmulator`, the executable specification of the firmware - error
  message texts included.
* `test_hil_arduino_stage.py` - `ArduinoStage` end to end: handshake, a small
  coordinated X/Y/Z move and back verified against the reported position,
  `STOP` mid-move (`MotionAbortedError` at the partial position), `home`.
* `test_hil_ads1115.py` - bursts of conversions in single-shot and continuous
  mode, timestamps paced by the data rate, a gain change. Reads only.

## Safety

The stage tests move the motors at most a few hundred steps (about 20 um)
from where the stage is when each test starts; the Uno resets when the port
opens, so that place becomes the origin. Put the stage roughly in the middle
of its travel, keep fingers and loose cables away, and keep the laser
switched off. Every test ends with `STOP` and `RELEASE`.

## Running on the Pi

The confocal service holds the serial port and the I2C bus; stop it first.

```bash
sudo systemctl stop confocal
cd /opt/confocal/backend
sudo -u confocal uv sync                      # the dev group provides pytest
ls /dev/serial/by-id/                         # find the Uno
sudo -u confocal env \
    CONFOCAL_CONFIG=/etc/confocal/confocal.toml \
    CONFOCAL_HIL_PORT=/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_XXXX-if00 \
    CONFOCAL_HIL_I2C=1 \
    uv run pytest tests/hil -v
sudo systemctl start confocal
```

Run only one part by leaving the other variable unset (for example only
`CONFOCAL_HIL_I2C=1` to check the ADC before the firmware is flashed).
