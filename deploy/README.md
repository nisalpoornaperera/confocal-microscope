# Deployment on a Raspberry Pi 5 (Raspberry Pi OS 64-bit)

The backend runs as a systemd service under a dedicated `confocal` user, with
dependencies managed by [uv](https://docs.astral.sh/uv/).

## 1. System preparation

```bash
sudo apt update && sudo apt install -y git libhdf5-dev i2c-tools
sudo raspi-config nonint do_i2c 0          # enable I2C (/dev/i2c-1) for the ADS1115; reboot once
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo useradd --system --create-home --groups dialout,i2c,gpio confocal
```

## 2. Install the application

```bash
sudo git clone <repository-url> /opt/confocal
sudo chown -R confocal:confocal /opt/confocal
cd /opt/confocal/backend
sudo -u confocal uv sync --no-dev            # includes pyserial (Arduino) and smbus2 (ADS1115)
```

## 3. Configure

```bash
sudo mkdir -p /etc/confocal
sudo cp /opt/confocal/config/confocal.example.toml /etc/confocal/confocal.toml
sudoedit /etc/confocal/confocal.toml         # travel limits, steps/um, backends, server.host
```

Keep `hardware.*` on `simulation` until the corresponding drivers are
implemented and tested (Phase 3: Arduino stage, Phase 4: ADS1115).

## 4. Install and start the service

```bash
sudo cp /opt/confocal/deploy/systemd/confocal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now confocal
journalctl -u confocal -f
curl http://localhost:8000/api/v1/system/status
```

Data lives in `/var/lib/confocal` (`confocal.db` metadata, `scans/*.h5`
measurements, `models/` deployed ML models). It is never deleted by the
application; back it up like any experimental dataset.

## 5. ML models

Models are trained **offline on a PC** (`python -m confocal.ml.training ...`)
and copied to `/var/lib/confocal/models/<name>/` (`model.joblib` +
`metadata.json`). The Pi only runs inference. Model files are pickles: only
deploy models you trained yourself.

## 6. Updating

```bash
cd /opt/confocal && sudo -u confocal git pull
cd backend && sudo -u confocal uv sync --no-dev
sudo systemctl restart confocal
```

Restarting during a scan records that scan as interrupted (its data is kept).
