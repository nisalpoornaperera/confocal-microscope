"""``confocal-hwcheck``: safe, non-interactive first-run check of the machine's hardware.

Run it on the Raspberry Pi with the confocal service stopped (it needs the
serial port and the I2C bus for itself)::

    sudo systemctl stop confocal
    cd /opt/confocal/backend
    sudo -u confocal uv run confocal-hwcheck --config /etc/confocal/confocal.toml
    sudo -u confocal uv run confocal-hwcheck --config /etc/confocal/confocal.toml --jog 20

What it does, in order (each step prints PASS / WARN / FAIL / SKIP):

1. loads the configuration (``--config``, default ``$CONFOCAL_CONFIG``, else
   the built-in defaults) and checks that the Cartesian travel limits fit
   inside every motor's travel (the same check the server runs at startup);
2. lists the serial ports and ``/dev/serial/by-id`` (the stable names to put
   in ``[arduino] port``);
3. opens the configured Arduino - which resets it - waits for the ``BOOT``
   banner, then ``PING``, ``STATUS`` and ``GETPOS``;
4. only with ``--jog N``: moves +N um and back on X, then Y, then Z, one axis
   at a time, each move verified against the position the firmware reports,
   refusing any target outside the configured limits. Watch the stage while
   it runs and confirm the *physical* direction of each axis. Because the
   Uno resets when the port opens, the jogs start from wherever the stage is
   (that place becomes the origin). The motors are released at the end;
5. scans the I2C bus, opens the ADS1115 at the configured address and takes
   ``--samples`` conversions at the configured gain: mean / std / min / max
   in volts, with a warning when the signal reaches the ADC full scale or the
   configured ``[processing] saturation_v`` (the OPT101 clips near 2.0 V on 3.3 V).

Exit code: 0 when nothing failed (warnings allowed), 1 when a check failed,
2 for a usage or configuration-file error. The laser is switched by hand; the
tool never needs it, but with the laser on and a sample under the objective
the ADC reading shows the detector working.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO, cast

import numpy as np

from confocal.config import CONFIG_ENV_VAR, Settings, load_settings
from confocal.errors import ConfocalError
from confocal.hardware.ads1115.adc import ADS1115ADC, BusFactory, open_smbus
from confocal.hardware.ads1115.conversion import is_saturated
from confocal.hardware.arduino.stage import ArduinoStage, TransportFactory
from confocal.hardware.factory import limit_problems
from confocal.hardware.kinematics import StageKinematics
from confocal.models.common import Axis, Position

#: Largest jog the check performs (micrometres): it is a direction check, not a range test.
MAX_JOG_UM = 100.0
#: Conversions taken from the ADS1115 by default.
DEFAULT_SAMPLES = 64
#: I2C addresses probed by the bus scan (the 7-bit range i2cdetect scans).
I2C_SCAN_RANGE = range(0x08, 0x78)
#: USB vendor ids of boards that are usually the Arduino.
_ARDUINO_VIDS = {0x2341: "Arduino", 0x2A03: "Arduino", 0x1A86: "CH340", 0x0403: "FTDI"}


class Outcome(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    outcome: Outcome
    detail: str


@dataclass(slots=True)
class Report:
    """Collects results and prints each one as it happens."""

    out: TextIO
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, outcome: Outcome, detail: str = "") -> CheckResult:
        result = CheckResult(name, outcome, detail)
        self.results.append(result)
        text = f"[{outcome.value}] {name}"
        if detail:
            text += f": {detail}"
        print(text, file=self.out, flush=True)
        return result

    def section(self, title: str) -> None:
        print(f"\n== {title} ==", file=self.out, flush=True)

    def count(self, outcome: Outcome) -> int:
        return sum(1 for result in self.results if result.outcome is outcome)

    @property
    def failed(self) -> bool:
        return self.count(Outcome.FAIL) > 0

    def summary(self) -> int:
        verdict = "FAIL" if self.failed else "PASS"
        print(
            f"\nRESULT: {verdict} ({self.count(Outcome.PASS)} passed, "
            f"{self.count(Outcome.WARN)} warnings, {self.count(Outcome.FAIL)} failed, "
            f"{self.count(Outcome.SKIP)} skipped)",
            file=self.out,
            flush=True,
        )
        if self.failed:
            for result in self.results:
                if result.outcome is Outcome.FAIL:
                    print(f"  FAILED: {result.name}: {result.detail}", file=self.out)
        return 1 if self.failed else 0


@dataclass(frozen=True, slots=True)
class SerialPortInfo:
    device: str
    description: str
    vid: int | None = None
    pid: int | None = None


class I2CProbe(Protocol):
    """The bus-scan part of ``smbus2.SMBus``."""

    def write_quick(self, i2c_addr: int) -> None: ...

    def read_byte(self, i2c_addr: int) -> int: ...

    def close(self) -> None: ...


def list_serial_ports() -> list[SerialPortInfo]:
    from serial.tools import list_ports

    return [
        SerialPortInfo(port.device, port.description or "", port.vid, port.pid)
        for port in sorted(list_ports.comports(), key=lambda p: p.device)
    ]


def open_probe_bus(bus: int) -> I2CProbe:
    return cast(I2CProbe, open_smbus(bus))


@dataclass(frozen=True, slots=True)
class Dependencies:
    """What the check talks to; tests replace the hardware with the emulator and fakes."""

    transport_factory: TransportFactory | None = None  # None: the serial port of the config
    bus_factory: BusFactory | None = None  # None: smbus2
    probe_bus_factory: Callable[[int], I2CProbe] = open_probe_bus
    list_serial_ports: Callable[[], list[SerialPortInfo]] = list_serial_ports
    serial_by_id_dir: Path = Path("/dev/serial/by-id")


# --------------------------------------------------------------------------- arguments


def _jog_um(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < value <= MAX_JOG_UM:
        raise argparse.ArgumentTypeError(f"jog must be in (0, {MAX_JOG_UM:g}] um, got {value:g}")
    return value


def _samples(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not 1 <= value <= 10_000:
        raise argparse.ArgumentTypeError("samples must be in 1..10000")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confocal-hwcheck",
        description=(
            "Safe first-run check of the confocal scanner's Arduino stage and ADS1115 "
            "(stop the confocal service first: it needs the serial port and the I2C bus)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"configuration file (default: ${CONFIG_ENV_VAR}, else the built-in defaults)",
    )
    parser.add_argument("--port", help="override [arduino] port for this check")
    parser.add_argument(
        "--jog",
        type=_jog_um,
        metavar="UM",
        default=None,
        help=(
            f"also move +UM and back on X, Y and Z, one at a time (0 < UM <= {MAX_JOG_UM:g}); "
            "watch the stage to confirm each axis' direction"
        ),
    )
    parser.add_argument(
        "--samples",
        type=_samples,
        default=DEFAULT_SAMPLES,
        help=f"ADS1115 conversions to take (default {DEFAULT_SAMPLES})",
    )
    parser.add_argument("--no-arduino", action="store_true", help="skip the stage checks")
    parser.add_argument("--no-adc", action="store_true", help="skip the I2C / ADS1115 checks")
    return parser


# --------------------------------------------------------------------------- checks


def _fmt_position(position: Position) -> str:
    return f"(x {position.x_um:+.3f}, y {position.y_um:+.3f}, z {position.z_um:+.3f}) um"


def check_configuration(settings: Settings, source: str, report: Report) -> bool:
    report.section("Configuration")
    report.add("configuration", Outcome.INFO, source)
    selection = settings.hardware
    report.add(
        "backends",
        Outcome.INFO,
        f"stage={selection.stage}, adc={selection.adc}, laser={selection.laser}, "
        f"camera={selection.camera}",
    )
    if selection.stage != "arduino" or selection.adc != "ads1115":
        report.add(
            "machine backends",
            Outcome.WARN,
            "the configuration does not select stage 'arduino' and adc 'ads1115'; the check "
            "still tests the devices of its [arduino] and [ads1115] sections",
        )
    try:
        kinematics = StageKinematics.from_config(settings.kinematics)
    except ValueError as exc:
        report.add("kinematics", Outcome.FAIL, str(exc))
        return False
    report.add("kinematics", Outcome.PASS, f"geometry '{kinematics.name}'")
    problems = limit_problems(settings, kinematics)
    if problems:
        report.add("travel limits", Outcome.FAIL, "; ".join(problems))
        return False
    limits = settings.limits
    report.add(
        "travel limits",
        Outcome.PASS,
        f"x [{limits.x.min_um:g}, {limits.x.max_um:g}], y [{limits.y.min_um:g}, "
        f"{limits.y.max_um:g}], z [{limits.z.min_um:g}, {limits.z.max_um:g}] um fit the motors",
    )
    return True


def check_serial_ports(settings: Settings, deps: Dependencies, report: Report) -> None:
    report.section("Serial ports")
    configured = settings.arduino.port
    try:
        ports = deps.list_serial_ports()
    except Exception as exc:  # listing is diagnostic only
        report.add("list serial ports", Outcome.WARN, f"cannot list ports: {exc}")
        ports = []
    for port in ports:
        kind = _ARDUINO_VIDS.get(port.vid or -1)
        ids = f" [{port.vid:04X}:{port.pid or 0:04X}]" if port.vid is not None else ""
        tag = f" <- likely the {kind} board" if kind else ""
        report.add("serial port", Outcome.INFO, f"{port.device}: {port.description}{ids}{tag}")
    if not ports:
        report.add("serial port", Outcome.INFO, "no serial ports found")
    by_id: list[tuple[str, str]] = []
    if deps.serial_by_id_dir.is_dir():
        for entry in sorted(deps.serial_by_id_dir.iterdir()):
            by_id.append((str(entry), str(entry.resolve())))
            report.add("by-id", Outcome.INFO, f"{entry} -> {entry.resolve()}")
    devices = {port.device for port in ports} | {path for path, _ in by_id}
    devices |= {target for _, target in by_id}
    if configured in devices or Path(configured).exists():
        report.add("configured port", Outcome.PASS, f"{configured} is present")
    else:
        report.add(
            "configured port",
            Outcome.WARN,
            f"{configured} is not present (is the Uno plugged in? use a path listed above)",
        )
    if not configured.startswith("/dev/serial/by-id/") and by_id:
        report.add(
            "stable port name",
            Outcome.INFO,
            f'consider [arduino] port = "{by_id[0][0]}" (survives re-plugging)',
        )


async def check_arduino(
    settings: Settings, deps: Dependencies, report: Report, jog_um: float | None
) -> None:
    report.section("Arduino stage")
    stage = ArduinoStage.from_settings(settings, transport_factory=deps.transport_factory)
    try:
        await stage.connect()
    except ConfocalError as exc:
        report.add("open + BOOT + PING + STATUS + GETPOS", Outcome.FAIL, str(exc))
        if jog_um is not None:
            report.add("jog", Outcome.SKIP, "the stage is not connected")
        return
    try:
        transport = stage.transport
        where = transport.description if transport is not None else settings.arduino.port
        report.add(
            "open + BOOT banner",
            Outcome.PASS,
            f"firmware {stage.version()} on {where} (the board reset when the port opened)",
        )
        await _arduino_queries(stage, report)
        if jog_um is not None:
            await _jog_axes(stage, settings, report, jog_um)
    except ConfocalError as exc:
        report.add("Arduino communication", Outcome.FAIL, str(exc))
    finally:
        if stage.connected:
            try:
                await stage.release()
                report.add("RELEASE", Outcome.PASS, "motor coils de-energised")
            except ConfocalError as exc:
                report.add("RELEASE", Outcome.WARN, str(exc))
        await stage.close()


async def _arduino_queries(stage: ArduinoStage, report: Report) -> None:
    round_trip = await stage.ping()
    report.add("PING", Outcome.PASS, f"round trip {round_trip * 1000:.1f} ms")
    firmware = await stage.firmware_status()
    detail = (
        f"state={firmware.state.value}, steps a/b/c={firmware.steps}, "
        f"coils {'on' if firmware.coils_enabled else 'off'}, version {firmware.version}"
    )
    report.add("STATUS", Outcome.PASS if not firmware.moving else Outcome.WARN, detail)
    position = await stage.get_position()
    report.add(
        "GETPOS", Outcome.PASS, f"steps {stage.last_steps} = {_fmt_position(position)} (origin)"
    )


async def _jog_axes(stage: ArduinoStage, settings: Settings, report: Report, jog_um: float) -> None:
    tolerance = settings.motion.position_tolerance_um
    print(
        f"\nJogging +{jog_um:g} um and back on X, Y, Z. Watch the stage and note which way "
        "the platform moves for each axis.",
        file=report.out,
        flush=True,
    )
    for axis in (Axis.X, Axis.Y, Axis.Z):
        name = f"jog {axis.value.upper()} +{jog_um:g} um"
        start = await stage.get_position()
        target = start.with_axis(axis, start.get(axis) + jog_um)
        violations = settings.limits.violations(target)
        if violations:
            report.add(
                name, Outcome.FAIL, "refused, outside the travel limits: " + "; ".join(violations)
            )
            continue
        steps_before = stage.last_steps
        try:
            reported = await stage.move_to(target)
            steps_out = stage.last_steps
            back = await stage.move_to(start)
        except ConfocalError as exc:
            report.add(name, Outcome.FAIL, str(exc))
            await _stop_quietly(stage, report)
            return
        moved = reported.get(axis) - start.get(axis)
        cross = max(
            abs(reported.get(other) - start.get(other)) for other in Axis if other is not axis
        )
        returned = back.max_axis_error(start)
        motor_delta = (
            tuple(after - before for before, after in zip(steps_before, steps_out, strict=True))
            if steps_before is not None and steps_out is not None
            else None
        )
        detail = (
            f"reported {_fmt_position(reported)} (moved {moved:+.3f} um, other axes "
            f"{cross:.3f} um), motor steps a/b/c changed by {motor_delta}; back at "
            f"{_fmt_position(back)}"
        )
        ok = abs(moved - jog_um) <= tolerance and cross <= tolerance and returned <= tolerance
        report.add(name, Outcome.PASS if ok else Outcome.FAIL, detail)
        if ok:
            print(
                f"       -> confirm the platform moved in the machine's +{axis.value.upper()} "
                "direction and came back",
                file=report.out,
                flush=True,
            )


async def _stop_quietly(stage: ArduinoStage, report: Report) -> None:
    try:
        await stage.stop()
    except ConfocalError as exc:
        report.add("STOP", Outcome.WARN, f"could not stop the stage: {exc}")


def check_i2c_bus(settings: Settings, deps: Dependencies, report: Report) -> bool:
    report.section("I2C bus")
    cfg = settings.ads1115
    try:
        bus = deps.probe_bus_factory(cfg.i2c_bus)
    except ConfocalError as exc:
        report.add(f"open I2C bus {cfg.i2c_bus}", Outcome.FAIL, str(exc))
        return False
    found: list[int] = []
    try:
        for address in I2C_SCAN_RANGE:
            try:
                # i2cdetect's choice: read for EEPROM-like ranges, quick write elsewhere.
                if 0x30 <= address <= 0x37 or 0x50 <= address <= 0x5F:
                    bus.read_byte(address)
                else:
                    bus.write_quick(address)
            except OSError:
                continue
            found.append(address)
    finally:
        bus.close()
    listing = ", ".join(f"0x{address:02X}" for address in found) or "none"
    report.add(f"scan bus {cfg.i2c_bus}", Outcome.INFO, f"devices at {listing}")
    if cfg.address in found:
        report.add("ADS1115 address", Outcome.PASS, f"a device answers at 0x{cfg.address:02X}")
        return True
    report.add(
        "ADS1115 address",
        Outcome.FAIL,
        f"nothing answers at 0x{cfg.address:02X} (check VDD 3.3 V, GND, SDA GPIO2, SCL GPIO3, "
        "and ADDR to GND for 0x48)",
    )
    return False


async def check_adc(settings: Settings, deps: Dependencies, report: Report, samples: int) -> None:
    report.section("ADS1115 / OPT101")
    adc = ADS1115ADC.from_config(settings.ads1115, bus_factory=deps.bus_factory)
    try:
        await adc.connect()
    except ConfocalError as exc:
        report.add("ADS1115 configure", Outcome.FAIL, str(exc))
        return
    try:
        gain = adc.gain
        report.add(
            "ADS1115 configure",
            Outcome.PASS,
            f"{adc.input_description}, gain {gain.value} (+/-{gain.full_scale_v:g} V), "
            f"{adc.data_rate_sps} SPS, {adc.mode}",
        )
        burst = await adc.read_samples(samples)
    except ConfocalError as exc:
        report.add("ADS1115 read", Outcome.FAIL, str(exc))
        return
    finally:
        await adc.close()
    volts = burst.volts
    duration = float(burst.timestamps[-1] - burst.timestamps[0]) if burst.n > 1 else 0.0
    rate = (burst.n - 1) / duration if duration > 0 else float("nan")
    report.add(
        "ADS1115 read",
        Outcome.PASS,
        f"{burst.n} samples: mean {float(np.mean(volts)):.4f} V, std {float(np.std(volts)):.4f} V, "
        f"min {float(np.min(volts)):.4f} V, max {float(np.max(volts)):.4f} V "
        f"(~{rate:.0f} samples/s)",
    )
    limit = settings.processing.saturation_v
    if is_saturated(burst.counts):
        report.add(
            "saturation",
            Outcome.WARN,
            f"codes at the ADC full scale (+/-{gain.full_scale_v:g} V): the input exceeds the "
            "range of this gain",
        )
    elif limit is not None and float(np.max(volts)) >= limit:
        report.add(
            "saturation",
            Outcome.WARN,
            f"signal reaches {float(np.max(volts)):.3f} V >= saturation_v {limit:g} V: the "
            "OPT101 clips there (defocus, attenuate the beam or use a less reflective sample)",
        )
    else:
        report.add("saturation", Outcome.PASS, "below full scale and saturation_v")


# --------------------------------------------------------------------------- entry points


async def run_checks(args: argparse.Namespace, deps: Dependencies, out: TextIO) -> int:
    report = Report(out)
    source_path = args.config if args.config is not None else os.environ.get(CONFIG_ENV_VAR)
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as exc:
        print(f"cannot load the configuration {source_path}: {exc}", file=out)
        return 2
    if args.port:
        settings = settings.model_copy(
            update={"arduino": settings.arduino.model_copy(update={"port": args.port})}
        )
    source = (
        str(source_path) if source_path else "built-in defaults (no --config, no $CONFOCAL_CONFIG)"
    )
    config_ok = check_configuration(settings, source, report)

    if args.no_arduino:
        report.add("Arduino stage", Outcome.SKIP, "--no-arduino")
    elif not config_ok and args.jog is not None:
        report.add("Arduino stage", Outcome.SKIP, "the configuration is unsafe; fix it first")
    else:
        check_serial_ports(settings, deps, report)
        await check_arduino(settings, deps, report, args.jog)

    if args.no_adc:
        report.add("ADS1115", Outcome.SKIP, "--no-adc")
    elif check_i2c_bus(settings, deps, report):
        await check_adc(settings, deps, report, args.samples)
    else:
        report.add("ADS1115 read", Outcome.SKIP, "no ADS1115 on the bus")
    return report.summary()


def main(argv: Sequence[str] | None = None, *, deps: Dependencies | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_checks(args, deps or Dependencies(), sys.stdout))


if __name__ == "__main__":
    sys.exit(main())
