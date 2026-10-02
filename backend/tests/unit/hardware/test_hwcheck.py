"""confocal-hwcheck against the firmware emulator and a fake ADS1115 (no real hardware)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.hardware.fakes import FakeADS1115, RealTime

from confocal.config import load_settings
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.protocol import Command, parse_request
from confocal.hardware.arduino.transport import EmulatorTransport, ScaledClock
from confocal.tools import hwcheck
from confocal.tools.hwcheck import Dependencies, SerialPortInfo, main

REPO = Path(__file__).resolve().parents[4]
PI_CONFIG = REPO / "config" / "confocal.pi.toml"


class Machine:
    """The Pi's view of an emulated Uno and a fake ADS1115."""

    def __init__(self, tmp_path: Path, *, boot_banner: bool = True, volts: float = 0.8) -> None:
        settings = load_settings(PI_CONFIG)
        self.emulator = FirmwareEmulator.from_config(settings.arduino)
        self.transport = EmulatorTransport(
            self.emulator, ScaledClock(speed=10_000.0), boot_banner=boot_banner
        )
        self.chip = FakeADS1115(RealTime(), {"AIN0": volts}, tag_conversions=False)
        self.chip.present = {0x48, 0x3C}
        by_id = tmp_path / "by-id"
        by_id.mkdir()
        (by_id / "usb-Arduino__www.arduino.cc__0043_7563-if00").write_text("")
        self.deps = Dependencies(
            transport_factory=lambda: self.transport,
            bus_factory=lambda bus: self.chip.reopen(),
            probe_bus_factory=lambda bus: self.chip.reopen(),
            list_serial_ports=lambda: [
                SerialPortInfo("/dev/ttyACM0", "Arduino Uno", 0x2341, 0x0043),
                SerialPortInfo("/dev/ttyAMA10", "ttyAMA10"),
            ],
            serial_by_id_dir=by_id,
        )

    def run(self, *args: str, config: Path = PI_CONFIG) -> int:
        return main(["--config", str(config), *args], deps=self.deps)

    def commands(self) -> list[Command]:
        return [parse_request(line).command for line in self.transport.sent]


def test_full_check_with_jog_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path)
    assert machine.run("--jog", "20", "--samples", "32") == 0
    out = capsys.readouterr().out
    for expected in (
        "[PASS] travel limits",
        "[INFO] serial port: /dev/ttyACM0: Arduino Uno [2341:0043] <- likely the Arduino board",
        "[PASS] configured port: /dev/ttyACM0 is present",
        "[PASS] open + BOOT banner: firmware 1.0.0",
        "[PASS] PING",
        "[PASS] STATUS: state=idle",
        "[PASS] GETPOS",
        "[PASS] jog X +20 um",
        "[PASS] jog Y +20 um",
        "[PASS] jog Z +20 um",
        "[PASS] RELEASE",
        "devices at 0x3C, 0x48",
        "[PASS] ADS1115 configure: AIN0-GND, gain 2 (+/-2.048 V), 860 SPS, single-shot",
        "[PASS] ADS1115 read: 32 samples: mean 0.8000 V",
        "[PASS] saturation",
        "RESULT: PASS",
    ):
        assert expected in out, expected
    assert "stable port name" in out
    # Each axis went out and back: 3 x 2 MOVEs (no backlash configured), then RELEASE.
    commands = machine.commands()
    assert commands[:3] == [Command.PING, Command.STATUS, Command.GETPOS]
    assert commands.count(Command.MOVE) == 6
    assert commands[-1] is Command.RELEASE
    assert machine.emulator.position == (0, 0, 0)
    assert not machine.emulator.coils_enabled
    assert not machine.transport.is_open
    assert machine.chip.closed


def test_without_jog_nothing_moves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path)
    assert machine.run() == 0
    assert Command.MOVE not in machine.commands()
    assert "[PASS] jog" not in capsys.readouterr().out


def test_missing_boot_banner_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path, boot_banner=False)
    machine.deps = Dependencies(
        transport_factory=machine.deps.transport_factory,
        bus_factory=machine.deps.bus_factory,
        probe_bus_factory=machine.deps.probe_bus_factory,
        list_serial_ports=list,
        serial_by_id_dir=tmp_path / "missing",
    )
    config = tmp_path / "quick.toml"
    text = PI_CONFIG.read_text(encoding="utf-8")
    quick = text.replace("[arduino]\n", "[arduino]\nhandshake_timeout_s = 0.2\n")
    config.write_text(quick, encoding="utf-8")
    assert machine.run("--jog", "5", "--samples", "4", config=config) == 1
    out = capsys.readouterr().out
    assert "[FAIL] open + BOOT + PING + STATUS + GETPOS: no BOOT banner" in out
    assert "[SKIP] jog" in out
    assert "[WARN] configured port: /dev/ttyACM0 is not present" in out
    assert "RESULT: FAIL" in out
    assert "[PASS] ADS1115 read" in out  # the ADC is still checked


def test_jog_outside_the_limits_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "small.toml"
    text = PI_CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        "[limits.y]\nmin_um = -1000.0\nmax_um = 1000.0",
        "[limits.y]\nmin_um = -1000.0\nmax_um = 10.0",
    )
    config.write_text(text, encoding="utf-8")
    machine = Machine(tmp_path)
    assert machine.run("--jog", "20", config=config) == 1
    out = capsys.readouterr().out
    assert "[PASS] jog X +20 um" in out
    assert "[FAIL] jog Y +20 um: refused, outside the travel limits" in out
    assert "[PASS] jog Z +20 um" in out
    assert machine.commands().count(Command.MOVE) == 4


def test_unsafe_configuration_skips_the_jog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "unsafe.toml"
    config.write_text(
        PI_CONFIG.read_text(encoding="utf-8")
        + "\n[arduino.a]\nmin_steps = -100\nmax_steps = 100\n",
        encoding="utf-8",
    )
    machine = Machine(tmp_path)
    assert machine.run("--jog", "20", config=config) == 1
    out = capsys.readouterr().out
    assert "[FAIL] travel limits: motor a" in out
    assert "[SKIP] Arduino stage" in out
    assert machine.transport.opens == 0


def test_missing_adc_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path)
    machine.chip.present = {0x3C}
    assert machine.run("--no-arduino") == 1
    out = capsys.readouterr().out
    assert "[SKIP] Arduino stage: --no-arduino" in out
    assert "[FAIL] ADS1115 address: nothing answers at 0x48" in out
    assert "[SKIP] ADS1115 read" in out


@pytest.mark.parametrize(
    ("volts", "message"),
    [
        (1.97, "signal reaches 1.970 V >= saturation_v 1.95 V"),
        (2.5, "codes at the ADC full scale (+/-2.048 V)"),
    ],
)
def test_saturation_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], volts: float, message: str
) -> None:
    machine = Machine(tmp_path, volts=volts)
    assert machine.run("--no-arduino", "--samples", "8") == 0  # a warning, not a failure
    out = capsys.readouterr().out
    assert f"[WARN] saturation: {message}" in out


def test_skip_everything(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path)
    assert machine.run("--no-arduino", "--no-adc") == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS (2 passed, 0 warnings, 0 failed, 2 skipped)" in out


def test_port_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine = Machine(tmp_path)
    assert machine.run("--no-adc", "--port", "/dev/ttyUSB7") == 0
    assert "[WARN] configured port: /dev/ttyUSB7 is not present" in capsys.readouterr().out


def test_config_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CONFOCAL_CONFIG", str(PI_CONFIG))
    machine = Machine(tmp_path)
    assert main(["--no-arduino", "--no-adc"], deps=machine.deps) == 0
    assert f"[INFO] configuration: {PI_CONFIG}" in capsys.readouterr().out


def test_unreadable_config_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    machine = Machine(tmp_path)
    assert machine.run(config=tmp_path / "nope.toml") == 2
    assert "cannot load the configuration" in capsys.readouterr().out


@pytest.mark.parametrize("jog", ["0", "-5", "500", "abc"])
def test_jog_bounds(jog: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--jog", jog])
    assert excinfo.value.code == 2
    assert "--jog" in capsys.readouterr().err


def test_default_dependencies_are_the_real_ones() -> None:
    deps = Dependencies()
    assert deps.transport_factory is None
    assert deps.bus_factory is None
    assert deps.probe_bus_factory is hwcheck.open_probe_bus
    assert deps.list_serial_ports is hwcheck.list_serial_ports
    assert isinstance(hwcheck.list_serial_ports(), list)  # pyserial listing works anywhere
