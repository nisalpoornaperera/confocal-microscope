# Raspberry Pi 5 setup and first run

This takes the scanner from a fresh Raspberry Pi OS (64-bit, desktop) to a Pi
that boots straight into the scanner UI on its HDMI monitor. Do the steps in
order; each one has a check before you move on.

Hardware assumed: Raspberry Pi 5 + HDMI monitor + mouse, Arduino Uno + 3 x
ULN2003 + 3 x 28BYJ-48 on the OpenFlexure Delta Stage, ADS1115 + OPT101 on the
Pi's 3.3 V, manually switched laser. Wiring and power: [wiring.md](wiring.md).

## Quick setup: one click

1. Flash the Uno (step 4 below) and plug it into the Pi.
2. Get the project onto the Pi (any folder, e.g. your home folder), either with
   `git clone https://github.com/nisalpoornaperera/confocal-microscope.git`
   or by copying the folder from a USB stick.
3. In the File Manager, double-click **`setup.sh`** in the project folder and
   choose **Execute in Terminal** (or run `bash setup.sh` in a terminal).
   Enter your password when asked.

`setup.sh` (`deploy/install.sh`) does steps 1-3 and 6 for you: system
packages, I2C, uv, the `confocal` user, installation into `/opt/confocal`, the
Python libraries (using the Pi's own Python), `/etc/confocal/confocal.toml`
from `config/confocal.pi.toml` with the Uno's port detected, a hardware check,
the background service, a **Confocal Scanner** desktop icon, the full-screen UI
at login, and it opens the UI. If I2C was just enabled it asks for one reboot.

Running it again updates the installation and keeps your settings and scans.
Options: `bash setup.sh --network` (also open the UI from a phone or PC on the
same network), `--no-autostart`, `--no-hwcheck`, `--help`. If double-clicking
does not offer "Execute", the executable bit was lost while copying: use
`bash setup.sh`.

The manual steps below are what the script does, for reference and repairs.

## 1. Prepare the Pi

```bash
uname -m                     # must print aarch64
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git i2c-tools avrdude
sudo raspi-config nonint do_i2c 0    # enable I2C (/dev/i2c-1)
sudo raspi-config nonint do_ssh 0    # optional: SSH from another computer
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo reboot
```

## 2. Get the code onto the Pi

Either clone it (after you have pushed the repository from the PC to GitHub or
similar):

```bash
sudo git clone <repository-url> /opt/confocal
```

or copy the project folder from the PC with a USB stick or
`scp -r "confocal microscope" pi@<pi-name>.local:/tmp/confocal` and
`sudo mv /tmp/confocal /opt/confocal`. Then:

```bash
sudo useradd --system --create-home --groups dialout,i2c,gpio,video confocal
sudo chown -R confocal:confocal /opt/confocal
cd /opt/confocal/backend
sudo -u confocal ~/.local/bin/uv sync --no-dev   # or: sudo -u confocal uv sync --no-dev, if uv is on root's PATH
```

The built UI is in `frontend/dist/` (committed), so the Pi needs no Node.js.
If you change the UI, rebuild it on the PC (`cd frontend && npm run build`) and
copy `frontend/dist/` across.

**Check:** `ls /opt/confocal/frontend/dist/index.html` exists.

## 3. Configure this machine

```bash
sudo mkdir -p /etc/confocal /var/lib/confocal
sudo cp /opt/confocal/config/confocal.pi.toml /etc/confocal/confocal.toml
sudo chown confocal:confocal /var/lib/confocal
ls /dev/serial/by-id/        # note the Uno's path
sudoedit /etc/confocal/confocal.toml
```

In the file set `[arduino] port` to the `/dev/serial/by-id/...` path. Leave
the small travel limits as they are for now; you widen them in step 7.

## 4. Flash the Uno firmware

Easiest is from the PC, with the Uno plugged into the PC by USB:

* **Arduino IDE 2:** open `firmware/confocal_stage/confocal_stage.ino`, select
  board *Arduino Uno* and the port, click Upload; or
* **build.py:** `python firmware/build.py --flash --port COM5` (your COM port).

Or on the Pi, using a `.hex` built on the PC (`firmware/build/confocal_stage.hex`):

```bash
sudo systemctl stop confocal 2>/dev/null   # the service must not hold the port
cd /opt/confocal
sudo -u confocal python3 firmware/build.py --flash --hex firmware/build/confocal_stage.hex --port /dev/ttyACM0
```

Details and other options: [firmware/README.md](../firmware/README.md).

## 5. Check the hardware (motor switch OFF first)

Plug the Uno into the Pi. With the **motor power switch off**:

```bash
cd /opt/confocal/backend
export CONFOCAL_CONFIG=/etc/confocal/confocal.toml
sudo -u confocal -E .venv/bin/python ../firmware/test/bench_check.py --port /dev/ttyACM0
sudo -u confocal -E .venv/bin/confocal-hwcheck
```

`bench_check.py` compares the board's replies with the emulator;
`confocal-hwcheck` checks the serial link, the I2C bus (ADS1115 at `0x48`) and
the sensor reading, and ends with a PASS / FAIL summary.

Then switch the **motor power on**, keep a hand on the motor switch, and jog:

```bash
sudo -u confocal -E .venv/bin/confocal-hwcheck --jog 20
```

It moves +20 um and back on X, then Y, then Z. Watch the stage:

* **Z** must move all three legs together and change focus. If it moves the
  wrong way, the motor directions are inverted.
* **X** and **Y** should translate the platform without tilting. If an axis
  goes the wrong way or the stage tilts, a motor is wired to the wrong leg or
  is inverted: fix it with `invert = true` under `[arduino.a]` / `.b` / `.c`,
  or by swapping which ULN2003 board is plugged into which Uno pins.

Re-run until all three axes move the right way.

## 6. Start the service and the full-screen UI

```bash
sudo cp /opt/confocal/deploy/systemd/confocal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now confocal
curl -s http://127.0.0.1:8000/api/v1/system/status    # "status":"ok"
```

Open the UI full-screen at login with a standard autostart entry (this is what
`setup.sh` installs):

```bash
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/confocal-scanner.desktop <<'EOF'
[Desktop Entry]
Type=Application
Name=Confocal Scanner
Exec=chromium-browser --kiosk --noerrdialogs --disable-infobars --app=http://127.0.0.1:8000/
EOF
```

If the command is called `chromium` on your image, use that name. Reboot: the
Pi should come up in the scanner UI. Alt+F4 leaves kiosk mode.

Do **not** create `~/.config/labwc/autostart`: a personal labwc autostart file
replaces the system one, and the taskbar and desktop then no longer start. If
you added the browser there following an older version of this guide, remove
that line (and the file, if nothing else is in it), or just run `setup.sh`,
which does this for you.

## 7. First scans

1. **Travel limits.** Jog carefully (Hardware page) towards each end of the
   stage's real travel and note the positions, then set `[limits.x/y/z]` in
   `/etc/confocal/confocal.toml` with a safety margin. The server refuses to
   start if the limits exceed any leg's travel (`[arduino.a/b/c] min_steps /
   max_steps`). `sudo systemctl restart confocal` after editing.
2. **Dark calibration.** Calibration page: block the laser beam, tick "I have
   blocked the beam", run dark calibration.
3. **Gain and reference.** Put a plane mirror at the focus, use ADC auto-gain,
   then reference calibration with Z search. If it reports saturation, the
   OPT101 is getting too much light: add an ND filter (see wiring.md).
4. **A small test scan.** Scan Setup: e.g. 5 x 5 points, 20 um step, Z range
   60 um. Check the estimate, start, watch the live I(Z) curve, then open the
   result in the Surface Viewer.
5. **Calibrate the kinematics.** The default micrometres-per-step values are
   nominal. Measure them against a known reference (a stage micrometer for
   X/Y, a known step height or dial gauge for Z) and set
   `[kinematics] um_per_step`, or the full measured `um_per_step_matrix`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Permission denied` on `/dev/ttyACM0` | the `confocal` user must be in `dialout`; the service must be stopped while flashing or running hwcheck |
| No `48` in `i2cdetect -y 1` | I2C enabled? SDA/SCL swapped? ADS1115 on 3.3 V and ADDR to GND? |
| Server will not start | `journalctl -u confocal -e` names the problem (port, limits vs motor travel, config error) |
| Motors hot | the Pi releases the coils when idle; a stuck move can be stopped with the e-stop button and the motor power switch |
| Noisy signal | keep the OPT101 wire short and away from motor cables; see wiring.md |

Safety reminders: the software e-stop stops the motors but cannot switch the
manual laser off; the motor power switch is the hardware e-stop. Never look
into the laser beam or its reflections.

## Low-contrast samples

If the I(Z) curves show only a small bump and most points end up `no_peak`:

1. **Fix the optics first** - it is the real cure: keep the OPT101 fully dark
   apart from the light through the pinhole, align the pinhole with the focus,
   and check the signal is not saturated (add an ND filter if it is) nor far
   too weak (laser power, alignment).
2. **Scan Setup -> Peak detection -> Preset "Low contrast (sensitive)"**: keeps
   peaks that are hard to tell from the noise (minimum SNR 2, relative
   prominence 0.05). Such points are flagged `weak_peak` and are at most
   *low-confidence*, never *valid*. The preset also lowers the surface's
   minimum confidence to 0.1 so they are used in the 3-D surface; check the
   confidence map to see how much to trust each area.
3. **Fine Z scan off** (Scan Setup -> Z): the surface is found from the coarse
   sweep alone. Use a smaller coarse step (e.g. 0.5-1 um) instead; this is
   faster and helps when the peak is broad compared with the fine range.
