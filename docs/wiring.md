# Wiring and power

Target machine: Raspberry Pi 5 + HDMI monitor, Arduino Uno + 3 × ULN2003 +
3 × 28BYJ-48 on an OpenFlexure Delta Stage, ADS1115 + OPT101, 650 nm 5 mW
laser module (manually switched), one 5 V 20 A switching power supply.

> **Safety first.** A 5 V 20 A supply can push 20 A into a short circuit -
> enough to melt thin wire and start a fire. Every branch it feeds MUST have
> its own fuse. Its mains side (L / N / earth) is lethal: fit the terminal
> cover, connect protective earth, use a fused and switched inlet or plug,
> and never work on it while it is plugged in. If you are not confident with
> mains wiring, have a qualified person do it - or replace it with an enclosed
> 5 V 3 A plug-in adapter, which is plenty for this machine (motors + laser
> draw under 1 A).

## 1. Which supply powers what

| Load | Supplied from | Why |
|---|---|---|
| Raspberry Pi 5 | Official 27 W USB-C supply | The Pi 5 needs USB-PD negotiation for full USB current; feeding it from the 20 A supply bypasses its protection. |
| Arduino Uno | USB cable from the Pi | Same cable carries power and serial. Never also feed 5 V into the Uno's 5V pin. |
| 3 × ULN2003 + motors | 20 A supply via fuse F1 (2 A) and the motor switch | ~0.2-0.25 A per energised motor, < 0.8 A total. |
| Laser module | 20 A supply via fuse F2 (0.5 A) and the laser switch | Typical 5 V modules draw 20-40 mA. Use a *module* with a built-in driver rated for 5 V, never a bare laser diode. |
| ADS1115 + OPT101 | Pi 3.3 V pin | Quiet, low current (< 1 mA), and keeps every signal inside the Pi's 3.3 V I2C limits and the ADC's input range. |

The sensor must **not** share the motor supply: motor switching noise would
appear directly in the photodiode signal.

## 2. Power distribution

```text
 Mains ──► 5 V 20 A PSU   (set V.ADJ to 5.0-5.1 V with a multimeter, no load)
             +5V ─┬─ F1 2 A (slow) ── MOTOR SWITCH (e-stop) ──┬─► ULN2003 A  (+)
                  │                                           ├─► ULN2003 B  (+)
                  │                                           └─► ULN2003 C  (+)
                  └─ F2 0.5 A ─────── LASER SWITCH ───────────────► laser (+)
             GND ── star point ──┬─► ULN2003 A/B/C (−)
                                 ├─► laser (−)
                                 └─► Arduino Uno GND   (one wire only)

 27 W USB-C PSU ──► Raspberry Pi 5 ──USB──► Arduino Uno (power + serial)
                       │
                       ├─ pin 1  (3V3) ──► ADS1115 VDD, OPT101 V+
                       └─ pin 9  (GND) ──► ADS1115 GND, OPT101 GND
```

* Put the fuses as close to the PSU terminals as possible; the wire between
  the terminal and the fuse is unprotected, so keep it short and at least
  1 mm² (18 AWG).
* The **motor switch** is your hardware emergency stop: it cuts the motors
  only, so the Pi keeps running and the scan data is saved. Use a large,
  easy-to-reach switch. (The software e-stop also exists, but a hardware cut
  works even if the Pi freezes.)
* **Grounding:** motor current returns to the PSU through the star point, not
  through the USB cable. The single PSU-GND → Uno-GND wire gives the motor
  drivers a common logic reference. Sensor grounds go only to the Pi.
* Add a 470-1000 µF electrolytic capacitor across +5 V / GND near the ULN2003
  boards.

## 3. Motors: Arduino Uno → ULN2003

| ULN2003 board | IN1 | IN2 | IN3 | IN4 |
|---|---|---|---|---|
| Motor A (leg A) | D2 | D3 | D4 | D5 |
| Motor B (leg B) | D6 | D7 | D8 | D9 |
| Motor C (leg C) | A0 | A1 | A2 | A3 |

* D0/D1 stay free (USB serial to the Pi). D13 is avoided: the bootloader
  flashes it at every reset, which would twitch a motor.
* Keep the power jumper on each ULN2003 board fitted; power goes to its
  + / − pins from the fused motor rail.
* The 28BYJ-48 plugs into the white socket of its ULN2003 board.

## 4. Sensor: OPT101 → ADS1115 → Pi

| From | To |
|---|---|
| Pi pin 1 (3V3) | ADS1115 VDD, OPT101 V+ |
| Pi pin 9 (GND) | ADS1115 GND, OPT101 GND |
| Pi pin 3 (GPIO2, SDA) | ADS1115 SDA |
| Pi pin 5 (GPIO3, SCL) | ADS1115 SCL |
| ADS1115 ADDR | GND → I2C address 0x48 |
| OPT101 output | ADS1115 A0 |

* Bare OPT101 chip (8-pin DIP): follow the datasheet's basic single-supply
  circuit - pin 1 V+, pins 3 and 8 to GND, pin 4 joined to pin 5 for the
  internal 1 MΩ feedback, output on pin 5, and a 0.1 µF capacitor from V+ to
  GND right at the chip. Check this against the datasheet before powering up;
  OPT101 breakout modules simply expose V+, GND and OUT.
* Keep the OPT101 → A0 wire short and away from the motor cables. An optional
  1 kΩ + 100 nF RC filter at A0 reduces noise.
* With a 3.3 V supply the OPT101 output tops out at roughly 2 V, so the
  ADS1115 gain is set to "2" (±2.048 V full scale, 62.5 µV per count). The
  software's auto-gain (`POST /api/v1/adc/calibrate`) can change it.
* **Saturation:** with the internal 1 MΩ feedback the OPT101 gives about
  0.45 V per µW at 650 nm, so it saturates at a few microwatts. A 5 mW laser
  reflected from a mirror through the pinhole can easily exceed that. If the
  software reports saturation on the reference mirror, add an ND filter or use
  a smaller external feedback resistor (see the OPT101 datasheet). If the
  signal is noisy, a dedicated low-noise 3.3 V regulator for the OPT101 and
  ADS1115 is the next upgrade.

## 5. Laser

The laser is switched manually, so the software cannot turn it off:

* dark calibration asks you to block the beam and confirm;
* the software emergency stop halts the motors but **cannot** switch the
  laser off - use the laser switch.

A 5 mW 650 nm laser is Class 3R: never look into the beam or its mirror
reflections.

## 6. Checks before first power-on

1. PSU unplugged: with a multimeter, check there is no short between +5 V and
   GND on every branch, and that the polarity at each ULN2003 and the laser is
   correct.
2. PSU on, motors unplugged, switches off: set and verify 5.0-5.1 V.
3. Switch the motor rail on: verify ~5 V at each ULN2003 board.
4. Plug in the motors; connect the Uno to the Pi by USB.
5. On the Pi: `ls /dev/serial/by-id/` should list the Uno;
   `sudo apt install -y i2c-tools && i2cdetect -y 1` should show `48`.
