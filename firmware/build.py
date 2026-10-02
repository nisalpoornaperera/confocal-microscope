#!/usr/bin/env python3
"""Build (and optionally flash) the Arduino Uno motor firmware without the Arduino IDE.

Compiles ``firmware/confocal_stage/`` together with the Arduino AVR core for
the Uno (ATmega328P, 16 MHz), links it, writes ``firmware/build/confocal_stage.hex``
and reports flash and RAM use. Standard library only; runs on Windows, Linux
(Raspberry Pi OS) and macOS::

    python firmware/build.py                         # build
    python firmware/build.py --flash --port COM5     # build, then upload (Windows)
    python3 firmware/build.py --flash --port /dev/serial/by-id/usb-Arduino...  # Pi
    python3 firmware/build.py --flash --port /dev/ttyACM0 --hex confocal_stage.hex
                                                     # upload a .hex built elsewhere

From the backend's environment: ``uv run --directory backend python ../firmware/build.py``.

Toolchain (avr-gcc, avr-libc) and Arduino AVR core are found automatically in

* the Arduino IDE 2 / arduino-cli package directory (``Arduino15/packages/arduino``
  under ``%LOCALAPPDATA%`` on Windows, ``~/.arduino15`` on Linux,
  ``~/Library/Arduino15`` on macOS) - install with the IDE's Boards Manager
  ("Arduino AVR Boards") or ``arduino-cli core install arduino:avr``;
* ``PATH`` (``sudo apt install gcc-avr avr-libc avrdude``) for the compiler, and
  ``/usr/share/arduino/hardware/arduino/avr`` (``sudo apt install arduino-core-avr``)
  for the core.

or given explicitly with ``--toolchain`` / ``--core`` (or the environment
variables ``AVR_TOOLCHAIN`` / ``ARDUINO_AVR_CORE``).

Uploading uses avrdude with the Uno's bootloader protocol
(``-c arduino -p atmega328p -b 115200 -D``). Stop the confocal server first:
the serial port can only be open in one program.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parent
SKETCH_DIR = FIRMWARE_DIR / "confocal_stage"
SKETCH_NAME = "confocal_stage"
DEFAULT_BUILD_DIR = FIRMWARE_DIR / "build"

MCU = "atmega328p"
F_CPU = "16000000L"
ARDUINO_VERSION = "10819"  # what Arduino IDE 2 passes as -DARDUINO
BOARD_DEFINES = ("-DARDUINO_AVR_UNO", "-DARDUINO_ARCH_AVR")
VARIANT = "standard"

#: Uno: 32 KB flash minus the 512-byte Optiboot bootloader; 2 KB SRAM.
FLASH_LIMIT = 32256
RAM_LIMIT = 2048
#: Static RAM above this leaves too little for the stack: warn.
RAM_WARN = 1536

UPLOAD_BAUD = 115200
IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""


class BuildError(RuntimeError):
    """A missing tool or a failed step; the message says what to do."""


# --------------------------------------------------------------------------- discovery


def _version_key(path: Path) -> tuple[int, ...]:
    """Sort key for version directories like ``7.3.0-atmel3.6.1-arduino7`` or ``1.8.6``."""
    return tuple(int(n) for n in re.findall(r"\d+", path.name)) or (0,)


def _arduino15_dirs() -> list[Path]:
    homes: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        homes.append(Path(local) / "Arduino15")
    home = Path.home()
    homes += [home / ".arduino15", home / "Library" / "Arduino15", home / "AppData/Local/Arduino15"]
    seen: set[Path] = set()
    result: list[Path] = []
    for path in homes:
        packages = path / "packages" / "arduino"
        if packages.is_dir() and packages not in seen:
            seen.add(packages)
            result.append(packages)
    return result


def _newest(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir()), key=_version_key, reverse=True)


def find_toolchain(explicit: str | None) -> Path:
    """Directory containing avr-gcc, avr-g++, avr-objcopy and avr-size."""
    candidates: list[Path] = []
    given = explicit or os.environ.get("AVR_TOOLCHAIN")
    if given:
        candidates.append(Path(given))
    else:
        on_path = shutil.which("avr-g++")
        if on_path:
            candidates.append(Path(on_path).resolve().parent)
        for packages in _arduino15_dirs():
            candidates += [v / "bin" for v in _newest(packages / "tools" / "avr-gcc")]
    for directory in candidates:
        if all(
            (directory / f"{tool}{EXE}").is_file()
            for tool in ("avr-gcc", "avr-g++", "avr-objcopy", "avr-size")
        ):
            return directory
    if given:
        raise BuildError(f"no avr-gcc/avr-g++/avr-objcopy/avr-size in {given}")
    raise BuildError(
        "AVR toolchain not found. Install it with the Arduino IDE 2 Boards Manager "
        "('Arduino AVR Boards'), `arduino-cli core install arduino:avr`, or on the Pi "
        "`sudo apt install gcc-avr avr-libc avrdude`; or pass --toolchain <bin dir>."
    )


def _is_core(path: Path) -> bool:
    return (path / "cores" / "arduino" / "Arduino.h").is_file() and (
        path / "variants" / VARIANT / "pins_arduino.h"
    ).is_file()


def find_core(explicit: str | None) -> Path:
    """Root of the Arduino AVR core (contains ``cores/arduino`` and ``variants/standard``)."""
    given = explicit or os.environ.get("ARDUINO_AVR_CORE")
    if given:
        if _is_core(Path(given)):
            return Path(given)
        raise BuildError(
            f"{given} is not an Arduino AVR core (needs cores/arduino and variants/{VARIANT})"
        )
    candidates: list[Path] = []
    for packages in _arduino15_dirs():
        candidates += _newest(packages / "hardware" / "avr")
    candidates += [
        Path("/usr/share/arduino/hardware/arduino/avr"),
        Path("/usr/share/arduino/hardware/arduino"),
    ]
    for path in candidates:
        if _is_core(path):
            return path
    raise BuildError(
        "Arduino AVR core not found. Install 'Arduino AVR Boards' in the Arduino IDE 2 "
        "Boards Manager, run `arduino-cli core install arduino:avr`, or on the Pi "
        "`sudo apt install arduino-core-avr`; or pass --core <dir>."
    )


@dataclass(frozen=True)
class Avrdude:
    executable: Path
    config: Path | None


def find_avrdude(explicit: str | None) -> Avrdude:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise BuildError(f"avrdude not found at {explicit}")
        conf = path.parent.parent / "etc" / "avrdude.conf"
        return Avrdude(path, conf if conf.is_file() else None)
    on_path = shutil.which("avrdude")
    if on_path:
        return Avrdude(Path(on_path), None)  # system avrdude knows its own config
    for packages in _arduino15_dirs():
        for version in _newest(packages / "tools" / "avrdude"):
            exe = version / "bin" / f"avrdude{EXE}"
            conf = version / "etc" / "avrdude.conf"
            if exe.is_file():
                return Avrdude(exe, conf if conf.is_file() else None)
    raise BuildError(
        "avrdude not found. Install it with the Arduino IDE 2 (Arduino AVR Boards) or "
        "`sudo apt install avrdude`, or pass --avrdude <path>."
    )


# --------------------------------------------------------------------------- build


@dataclass(frozen=True)
class Toolchain:
    bin: Path
    core: Path

    def tool(self, name: str) -> str:
        return str(self.bin / f"{name}{EXE}")

    @property
    def core_src(self) -> Path:
        return self.core / "cores" / "arduino"

    @property
    def variant(self) -> Path:
        return self.core / "variants" / VARIANT


def _run(command: Sequence[str], *, verbose: bool) -> str:
    if verbose:
        print(" ".join(f'"{c}"' if " " in c else c for c in command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise BuildError(f"command failed ({result.returncode}): {Path(command[0]).name}\n{output}")
    return output


def _common_flags(tc: Toolchain) -> list[str]:
    return [
        "-c",
        "-g",
        "-Os",
        f"-mmcu={MCU}",
        f"-DF_CPU={F_CPU}",
        f"-DARDUINO={ARDUINO_VERSION}",
        *BOARD_DEFINES,
        "-ffunction-sections",
        "-fdata-sections",
        f"-I{tc.core_src}",
        f"-I{tc.variant}",
    ]


def _compile_command(
    tc: Toolchain, source: Path, obj: Path, *, sketch: bool, strict: bool
) -> list[str]:
    flags = _common_flags(tc)
    # Our sources get full warnings; the Arduino core is built as the IDE builds it.
    warnings = ["-Wall", "-Wextra", *(["-Werror"] if strict else [])] if sketch else ["-w"]
    suffix = source.suffix.lower()
    if suffix == ".c":
        return [tc.tool("avr-gcc"), *flags, *warnings, "-std=gnu11", str(source), "-o", str(obj)]
    if suffix == ".s":
        return [tc.tool("avr-gcc"), *flags, "-x", "assembler-with-cpp", str(source), "-o", str(obj)]
    cxx = ["-std=gnu++11", "-fno-exceptions", "-fno-threadsafe-statics", "-fno-rtti"]
    if not sketch:
        cxx += ["-fpermissive", "-Wno-error=narrowing"]  # as the Arduino IDE builds the core
    else:
        cxx += [f"-I{SKETCH_DIR}"]
    return [tc.tool("avr-g++"), *flags, *warnings, *cxx, str(source), "-o", str(obj)]


def _preprocess_ino(ino: Path, out: Path) -> Path:
    """What the Arduino IDE does to a .ino that needs no generated prototypes."""
    text = ino.read_text(encoding="utf-8")
    ino_path = str(ino).replace("\\", "/")
    out.write_text(f'#include <Arduino.h>\n#line 1 "{ino_path}"\n{text}', encoding="utf-8")
    return out


def _compile_all(
    tc: Toolchain,
    jobs: Iterable[tuple[Path, Path, bool]],
    *,
    strict: bool,
    verbose: bool,
    workers: int,
) -> list[str]:
    commands = [
        _compile_command(tc, src, obj, sketch=sketch, strict=strict) for src, obj, sketch in jobs
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda c: _run(c, verbose=verbose), commands))
    return [o for o in outputs if o]


@dataclass(frozen=True)
class SizeReport:
    flash: int
    ram: int

    def describe(self) -> str:
        return (
            f"Flash: {self.flash} bytes ({100 * self.flash / FLASH_LIMIT:.1f}% of {FLASH_LIMIT})\n"
            f"RAM:   {self.ram} bytes static ({100 * self.ram / RAM_LIMIT:.1f}% of {RAM_LIMIT}), "
            f"{RAM_LIMIT - self.ram} bytes left for the stack"
        )


def measure(tc: Toolchain, elf: Path, *, verbose: bool) -> SizeReport:
    """Flash = .text + .data, RAM = .data + .bss + .noinit (from ``avr-size -A``)."""
    output = _run([tc.tool("avr-size"), "-A", str(elf)], verbose=verbose)
    sections: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
            sections[parts[0]] = int(parts[1])
    data = sections.get(".data", 0)
    return SizeReport(
        flash=sections.get(".text", 0) + data,
        ram=data + sections.get(".bss", 0) + sections.get(".noinit", 0),
    )


def build(
    tc: Toolchain, build_dir: Path, *, strict: bool, verbose: bool, workers: int
) -> tuple[Path, SizeReport]:
    if not (SKETCH_DIR / f"{SKETCH_NAME}.ino").is_file():
        raise BuildError(f"sketch not found: {SKETCH_DIR / (SKETCH_NAME + '.ino')}")
    core_dir = build_dir / "core"
    sketch_dir = build_dir / "sketch"
    for directory in (core_dir, sketch_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    # Arduino core -> core.a (everything; --gc-sections drops what is unused).
    core_sources = sorted(
        p for p in tc.core_src.iterdir() if p.is_file() and p.suffix.lower() in (".c", ".cpp", ".s")
    )
    core_jobs = [(src, core_dir / f"{src.name}.o", False) for src in core_sources]

    # Sketch: the .ino (as .ino.cpp) and every .cpp beside it.
    ino_cpp = _preprocess_ino(
        SKETCH_DIR / f"{SKETCH_NAME}.ino", sketch_dir / f"{SKETCH_NAME}.ino.cpp"
    )
    sketch_sources = [ino_cpp, *sorted(SKETCH_DIR.glob("*.cpp"))]
    sketch_jobs = [(src, sketch_dir / f"{src.name}.o", True) for src in sketch_sources]

    started = time.monotonic()
    messages = _compile_all(
        tc, core_jobs + sketch_jobs, strict=strict, verbose=verbose, workers=workers
    )
    for message in messages:
        print(message)

    archive = build_dir / "core.a"
    if archive.exists():
        archive.unlink()
    _run(
        [tc.tool("avr-ar"), "rcs", str(archive), *(str(obj) for _, obj, _ in core_jobs)],
        verbose=verbose,
    )

    elf = build_dir / f"{SKETCH_NAME}.elf"
    hex_file = build_dir / f"{SKETCH_NAME}.hex"
    _run(
        [
            tc.tool("avr-gcc"),
            "-Os",
            "-g",
            "-Wl,--gc-sections",
            f"-mmcu={MCU}",
            "-o",
            str(elf),
            *(str(obj) for _, obj, _ in sketch_jobs),
            str(archive),
            f"-L{build_dir}",
            "-lm",
        ],
        verbose=verbose,
    )
    _run(
        [tc.tool("avr-objcopy"), "-O", "ihex", "-R", ".eeprom", str(elf), str(hex_file)],
        verbose=verbose,
    )
    report = measure(tc, elf, verbose=verbose)
    if verbose:
        print(f"compiled and linked in {time.monotonic() - started:.1f} s")
    if report.flash > FLASH_LIMIT:
        raise BuildError(f"program too large: {report.flash} bytes > {FLASH_LIMIT}")
    if report.ram > RAM_LIMIT:
        raise BuildError(f"static RAM too large: {report.ram} bytes > {RAM_LIMIT}")
    return hex_file, report


def flash(avrdude: Avrdude, hex_file: Path, port: str, *, verbose: bool) -> None:
    command = [str(avrdude.executable)]
    if avrdude.config is not None:
        command += ["-C", str(avrdude.config)]
    command += [
        "-p",
        MCU,
        "-c",
        "arduino",
        "-P",
        port,
        "-b",
        str(UPLOAD_BAUD),
        "-D",
        "-U",
        f"flash:w:{hex_file}:i",
    ]
    if verbose:
        command.insert(1, "-v")
    print("Uploading:", " ".join(f'"{c}"' if " " in c else c for c in command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise BuildError(
            f"avrdude failed ({result.returncode}). Check the port, that no other program "
            "(the confocal server, the Arduino IDE serial monitor) has it open, and the USB cable."
        )


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--toolchain", help="directory with avr-gcc / avr-g++ (default: auto-detect)"
    )
    parser.add_argument("--core", help="Arduino AVR core directory (default: auto-detect)")
    parser.add_argument(
        "--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="output directory"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings in the sketch as errors"
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count() or 2, help="parallel compiles"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print every command")
    parser.add_argument("--flash", action="store_true", help="upload to the Uno after building")
    parser.add_argument("--port", help="serial port for --flash, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument(
        "--hex", type=Path, help="with --flash: upload this .hex instead of building"
    )
    parser.add_argument("--avrdude", help="avrdude executable (default: auto-detect)")
    args = parser.parse_args(argv)

    if args.flash and not args.port:
        parser.error("--flash needs --port")
    if args.hex and not args.flash:
        parser.error("--hex is only used with --flash")

    try:
        if args.hex:
            if not args.hex.is_file():
                raise BuildError(f"{args.hex} does not exist")
            hex_file = args.hex
        else:
            tc = Toolchain(bin=find_toolchain(args.toolchain), core=find_core(args.core))
            print(f"Toolchain: {tc.bin}")
            print(f"AVR core:  {tc.core}")
            hex_file, report = build(
                tc,
                args.build_dir.resolve(),
                strict=args.strict,
                verbose=args.verbose,
                workers=max(1, args.jobs),
            )
            print(f"Built {hex_file}")
            print(report.describe())
            if report.ram > RAM_WARN:
                print(
                    f"warning: static RAM above {RAM_WARN} bytes leaves little room for the stack"
                )
        if args.flash:
            flash(find_avrdude(args.avrdude), hex_file, args.port, verbose=args.verbose)
            print("Upload complete. The Uno restarts and prints '! BOOT <version>'.")
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
