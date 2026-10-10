#!/usr/bin/env bash
# One-step installer / updater for the confocal surface scanner on a
# Raspberry Pi 5 (Raspberry Pi OS 64-bit, desktop).
#
#   bash setup.sh                  install or update, start the service, open the UI
#   bash setup.sh --network        also allow other devices (phone, PC) to open the UI
#   bash setup.sh --no-autostart   do not open the UI full-screen at login
#   bash setup.sh --no-hwcheck     skip the hardware check
#
# Safe to run again: it updates the code and the Python packages, and keeps
# /etc/confocal/confocal.toml (your machine settings) and /var/lib/confocal
# (your scans). See docs/pi-setup.md for what every step does.

set -Eeuo pipefail

INSTALL_DIR=/opt/confocal
CONFIG_DIR=/etc/confocal
CONFIG_FILE=$CONFIG_DIR/confocal.toml
DATA_DIR=/var/lib/confocal
SERVICE_USER=confocal
SERVICE_NAME=confocal
URL=http://127.0.0.1:8000/

NETWORK=0
AUTOSTART=1
HWCHECK=1
for arg in "$@"; do
    case "$arg" in
        --network) NETWORK=1 ;;
        --no-autostart) AUTOSTART=0 ;;
        --no-hwcheck) HWCHECK=0 ;;
        -h | --help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (see --help)"; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------------- output
step_no=0
step() { step_no=$((step_no + 1)); printf '\n\033[1;34m[%d] %s\033[0m\n' "$step_no" "$*"; }
ok()   { printf '    \033[32mOK\033[0m %s\n' "$*"; }
warn() { printf '    \033[33mWARNING\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mSETUP FAILED:\033[0m %s\n' "$*"; pause_if_terminal; exit 1; }
pause_if_terminal() {
    # Keep a double-clicked terminal window open so the result can be read.
    if [[ -t 0 ]]; then read -r -p $'\nPress Enter to close this window...' _ || true; fi
}
trap 'die "command failed on line $LINENO: $BASH_COMMAND"' ERR

# ----------------------------------------------------------------------------- checks
step "Checking the system"
[[ $EUID -ne 0 ]] || die "run this as your normal desktop user (not with sudo); it asks for your password itself"
DESKTOP_USER=$(id -un)
DESKTOP_HOME=$HOME
[[ $(uname -m) == aarch64 ]] || die "this needs the 64-bit Raspberry Pi OS (uname -m says $(uname -m))"
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f $SOURCE_DIR/backend/pyproject.toml && -d $SOURCE_DIR/frontend/dist ]] \
    || die "run setup.sh from the top folder of the confocal project (backend/ and frontend/dist/ missing in $SOURCE_DIR)"
command -v sudo >/dev/null || die "sudo is not installed"
echo "    You may be asked for your password once."
sudo -v || die "sudo password needed"
# Keep sudo alive while the script runs.
( trap - ERR; while true; do sudo -n true || true; sleep 50; kill -0 "$$" 2>/dev/null || exit 0; done ) >/dev/null 2>&1 &
ok "user $DESKTOP_USER, project in $SOURCE_DIR"

# ----------------------------------------------------------------------------- packages
step "Installing system packages (git, I2C tools, avrdude, curl, rsync)"
sudo apt-get update -qq || warn "apt update failed (no internet?) - continuing with the installed package lists"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git i2c-tools avrdude curl rsync python3 >/dev/null \
    || die "apt-get install failed (check the internet connection)"
ok "system packages installed"

step "Enabling I2C (for the ADS1115)"
REBOOT_NEEDED=0
if command -v raspi-config >/dev/null; then
    if [[ $(sudo raspi-config nonint get_i2c) != 0 ]]; then
        sudo raspi-config nonint do_i2c 0
        REBOOT_NEEDED=1
        ok "I2C enabled (takes effect after a reboot)"
    else
        ok "I2C already enabled"
    fi
else
    warn "raspi-config not found - enable I2C yourself"
fi
[[ -e /dev/i2c-1 ]] || REBOOT_NEEDED=1

step "Installing uv (Python package manager)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh >/dev/null \
        || die "could not install uv (check the internet connection)"
fi
UV=$(command -v uv)
ok "$($UV --version)"

# ----------------------------------------------------------------------------- user + code
step "Creating the service user '$SERVICE_USER'"
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    sudo useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi
for group in dialout i2c gpio; do
    if getent group "$group" >/dev/null; then
        sudo usermod -aG "$group" "$SERVICE_USER"
        sudo usermod -aG "$group" "$DESKTOP_USER"
    fi
done
ok "user $SERVICE_USER (serial, I2C, GPIO access)"

step "Installing the program into $INSTALL_DIR"
if sudo systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    sudo systemctl stop "$SERVICE_NAME"
    ok "stopped the running service"
fi
if [[ $SOURCE_DIR != "$INSTALL_DIR" ]]; then
    sudo mkdir -p "$INSTALL_DIR"
    sudo rsync -a --delete \
        --exclude '.git/' --exclude '.venv/' --exclude 'node_modules/' --exclude 'backend/data/' \
        --exclude '__pycache__/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' --exclude '.pytest_cache/' \
        "$SOURCE_DIR/" "$INSTALL_DIR/"
    ok "copied from $SOURCE_DIR"
else
    ok "already in $INSTALL_DIR"
fi
# You own the code (so 'git pull' works); the service user owns only its venv.
sudo chown -R "$DESKTOP_USER:$DESKTOP_USER" "$INSTALL_DIR"
sudo chmod -R a+rX "$INSTALL_DIR"
if [[ -d $INSTALL_DIR/.git ]]; then
    git config --global --add safe.directory "$INSTALL_DIR" 2>/dev/null || true
fi

# ----------------------------------------------------------------------------- python
step "Installing the Python libraries (this can take a few minutes the first time)"
VENV=$INSTALL_DIR/backend/.venv
PY_OK=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 11)))' 2>/dev/null || echo 0)
if [[ $PY_OK == 1 ]]; then
    PYTHON_ARG=(--python /usr/bin/python3)
else
    PYTHON_ARG=(--python 3.11)  # uv downloads it
fi
# A venv copied from Windows (Scripts/) or built on another interpreter is replaced.
if [[ -d $VENV ]] && { [[ ! -x $VENV/bin/python ]] || [[ $PY_OK == 1 && $(readlink -f "$VENV/bin/python") != /usr/* ]]; }; then
    sudo rm -rf "$VENV"
    ok "removed an incompatible old environment"
fi
sudo mkdir -p "$VENV"
sudo chown "$SERVICE_USER:$SERVICE_USER" "$VENV"
cd "$INSTALL_DIR/backend"
if ! sudo -u "$SERVICE_USER" -H "$UV" sync --no-dev --frozen "${PYTHON_ARG[@]}"; then
    warn "locked install failed - retrying with a fresh dependency resolution"
    sudo -u "$SERVICE_USER" -H "$UV" sync --no-dev "${PYTHON_ARG[@]}" \
        || die "installing the Python libraries failed (see the messages above)"
fi
cd - >/dev/null
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$VENV"
[[ -x $VENV/bin/confocal-server ]] || die "confocal-server was not installed"
ok "Python libraries installed"

# ----------------------------------------------------------------------------- config
step "Machine configuration ($CONFIG_FILE)"
sudo mkdir -p "$CONFIG_DIR" "$DATA_DIR"
sudo chown "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
UNO_PORT=$(ls /dev/serial/by-id/*Arduino* 2>/dev/null | head -n 1 || true)
[[ -n $UNO_PORT ]] || UNO_PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -n 1 || true)
if [[ ! -f $CONFIG_FILE ]]; then
    sudo cp "$INSTALL_DIR/config/confocal.pi.toml" "$CONFIG_FILE"
    ok "created from config/confocal.pi.toml"
    if [[ -n $UNO_PORT ]]; then
        # Replace the port only inside the [arduino] section.
        sudo awk -v port="$UNO_PORT" '
            /^\[/ { section = $0 }
            section == "[arduino]" && /^port[[:space:]]*=/ { print "port = \"" port "\"   # detected by setup.sh"; next }
            { print }' "$CONFIG_FILE" | sudo tee "$CONFIG_FILE.new" >/dev/null
        sudo mv "$CONFIG_FILE.new" "$CONFIG_FILE"
        ok "Arduino found at $UNO_PORT"
    else
        warn "no Arduino found - plug in the Uno and run setup again, or edit [arduino] port in $CONFIG_FILE"
    fi
else
    sudo cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
    ok "kept your existing settings (backup: $CONFIG_FILE.bak)"
fi
if [[ $NETWORK == 1 ]]; then
    sudo sed -i 's|^host = "127.0.0.1".*|host = "0.0.0.0"     # reachable from other devices (setup.sh --network)|' "$CONFIG_FILE"
    ok "UI reachable from other devices on this network"
fi
sudo chmod 644 "$CONFIG_FILE"
if ! sudo -u "$SERVICE_USER" "$VENV/bin/python" -c "from confocal.config import load_settings; load_settings('$CONFIG_FILE')" 2>/tmp/confocal-config-error; then
    cat /tmp/confocal-config-error
    die "$CONFIG_FILE is not valid (see the error above); fix it or delete it to start from the defaults"
fi
ok "configuration is valid"

# ----------------------------------------------------------------------------- hwcheck
if [[ $HWCHECK == 1 && $REBOOT_NEEDED == 0 ]]; then
    step "Hardware check (Arduino, ADS1115, sensor)"
    if sudo -u "$SERVICE_USER" env CONFOCAL_CONFIG="$CONFIG_FILE" "$VENV/bin/confocal-hwcheck"; then
        ok "hardware check passed"
    else
        warn "the hardware check reported problems (see above); the software is installed anyway"
    fi
fi

# ----------------------------------------------------------------------------- service
step "Installing the background service (starts at every boot)"
sudo cp "$INSTALL_DIR/deploy/systemd/confocal.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME" >/dev/null 2>&1
sudo systemctl restart "$SERVICE_NAME"
for _ in $(seq 1 30); do
    curl -fsS "${URL}api/v1/system/status" >/dev/null 2>&1 && break
    sleep 1
done
if curl -fsS "${URL}api/v1/system/status" >/dev/null 2>&1; then
    ok "the scanner software is running at $URL"
else
    warn "the service did not answer yet - last log lines:"
    sudo journalctl -u "$SERVICE_NAME" -n 15 --no-pager || true
fi

# ----------------------------------------------------------------------------- UI
step "UI launcher, desktop icon and full-screen start at login"
BROWSER=$(command -v chromium-browser || command -v chromium || true)
[[ -n $BROWSER ]] || { sudo apt-get install -y -qq chromium >/dev/null 2>&1 || true; BROWSER=$(command -v chromium || command -v chromium-browser || true); }
[[ -n $BROWSER ]] || warn "Chromium not found - open $URL in any browser"
sudo tee /usr/local/bin/confocal-ui >/dev/null <<EOF
#!/usr/bin/env bash
# Open the confocal scanner UI once the background service answers.
for _ in \$(seq 1 60); do curl -fsS ${URL}api/v1/system/status >/dev/null 2>&1 && break; sleep 1; done
exec ${BROWSER:-xdg-open} --kiosk --noerrdialogs --disable-infobars --no-first-run --app=$URL "\$@"
EOF
sudo chmod 755 /usr/local/bin/confocal-ui
for dir in "$DESKTOP_HOME/Desktop" "$DESKTOP_HOME/.local/share/applications"; do
    mkdir -p "$dir"
    cat >"$dir/confocal-scanner.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Confocal Scanner
Comment=Open the confocal surface scanner UI (Alt+F4 to close)
Exec=/usr/local/bin/confocal-ui
Icon=applications-science
Terminal=false
Categories=Science;Education;
EOF
    chmod 755 "$dir/confocal-scanner.desktop"
done
ok "desktop icon 'Confocal Scanner' created"
# Earlier guides added the browser to ~/.config/labwc/autostart. A personal
# labwc autostart file REPLACES the system one (taskbar, desktop), so remove
# our line again and delete the file if nothing else is left in it.
LABWC=$DESKTOP_HOME/.config/labwc/autostart
if [[ -f $LABWC ]] && grep -qE '127\.0\.0\.1:8000|confocal-ui' "$LABWC"; then
    sed -i -E '/127\.0\.0\.1:8000|confocal-ui/d' "$LABWC"
    if ! grep -q '[^[:space:]]' "$LABWC"; then
        rm -f "$LABWC"
        ok "removed ~/.config/labwc/autostart (it hid the taskbar); the standard autostart is used now"
    fi
fi
if [[ $AUTOSTART == 1 ]]; then
    # Standard XDG autostart, run by the Raspberry Pi desktop (labwc, wayfire, X11).
    mkdir -p "$DESKTOP_HOME/.config/autostart"
    cp "$DESKTOP_HOME/.local/share/applications/confocal-scanner.desktop" "$DESKTOP_HOME/.config/autostart/"
    ok "the UI opens full-screen at every login (Alt+F4 closes it)"
else
    rm -f "$DESKTOP_HOME/.config/autostart/confocal-scanner.desktop"
fi

# ----------------------------------------------------------------------------- done
printf '\n\033[1;32mSETUP COMPLETE\033[0m\n'
echo "    UI:      $URL   (or double-click 'Confocal Scanner' on the desktop)"
[[ $NETWORK == 1 ]] && echo "    Network: http://$(hostname -I | awk '{print $1}'):8000/"
echo "    Logs:    journalctl -u $SERVICE_NAME -f"
echo "    Update:  run setup.sh again (your settings and scans are kept)"
if [[ $REBOOT_NEEDED == 1 ]]; then
    printf '\n\033[1;33m    REBOOT NEEDED\033[0m to activate I2C: sudo reboot\n'
elif [[ -n ${WAYLAND_DISPLAY:-}${DISPLAY:-} && -n $BROWSER ]]; then
    echo "    Opening the UI now..."
    nohup /usr/local/bin/confocal-ui >/dev/null 2>&1 &
fi
trap - ERR
pause_if_terminal
