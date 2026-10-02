#!/usr/bin/env python3
"""Compile and run the firmware's native tests on the host PC (no Arduino needed).

1. If the backend package ``confocal`` is importable, regenerate
   ``vectors.txt`` from the Python reference (``gen_vectors.py``); otherwise use
   the committed copy.
2. Compile ``test_firmware.cpp`` with the firmware's hardware-independent
   sources (protocol, motion, controller) using a host C++ compiler.
3. Run it against ``vectors.txt``.

Usage (from the repository root)::

    uv run --directory backend python ../firmware/test/run_tests.py
    python3 firmware/test/run_tests.py            # uses the committed vectors

Compiler: ``$CXX`` if set, else the first of ``c++``, ``g++``, ``clang++`` on
PATH, else Zig's bundled clang (``python -m ziglang c++`` if the ``ziglang``
package is installed, or ``uvx --from ziglang python -m ziglang c++``, which
downloads it once). On the Raspberry Pi ``sudo apt install g++`` is enough.
Standard library only.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SKETCH_DIR = TEST_DIR.parent / "confocal_stage"
BUILD_DIR = TEST_DIR.parent / "build" / "native-test"
SOURCES = [
    TEST_DIR / "test_firmware.cpp",
    SKETCH_DIR / "protocol.cpp",
    SKETCH_DIR / "motion.cpp",
    SKETCH_DIR / "controller.cpp",
]
VECTORS = TEST_DIR / "vectors.txt"


def find_compiler() -> list[str]:
    if os.environ.get("CXX"):
        return shlex.split(os.environ["CXX"])
    for name in ("c++", "g++", "clang++"):
        path = shutil.which(name)
        if path:
            return [path]
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang", "c++"]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "ziglang", "python", "-m", "ziglang", "c++"]
    raise SystemExit(
        "error: no host C++ compiler found. Install one (Pi: `sudo apt install g++`), "
        "set CXX, or install uv so `uvx --from ziglang` can provide clang."
    )


def regenerate_vectors() -> None:
    if importlib.util.find_spec("confocal") is None:
        print(
            f"backend package 'confocal' not importable: using committed {VECTORS.name}", flush=True
        )
        return
    print("regenerating test vectors from the Python reference ...", flush=True)
    subprocess.run([sys.executable, str(TEST_DIR / "gen_vectors.py")], check=True)


def main() -> int:
    regenerate_vectors()
    if not VECTORS.is_file():
        print(f"error: {VECTORS} is missing", file=sys.stderr, flush=True)
        return 1
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = BUILD_DIR / ("test_firmware.exe" if os.name == "nt" else "test_firmware")
    command = [
        *find_compiler(),
        "-std=c++11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{SKETCH_DIR}",
        *map(str, SOURCES),
        "-o",
        str(exe),
    ]
    print("compiling:", " ".join(shlex.quote(c) for c in command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    # Zig prints "N warnings generated" for its own libc++ build; show real diagnostics only.
    noise = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "warnings generated" not in line
    ]
    if noise:
        print("\n".join(noise), flush=True)
    if result.returncode != 0:
        print("error: compilation failed", file=sys.stderr, flush=True)
        return result.returncode
    print("running native firmware tests ...", flush=True)
    return subprocess.run([str(exe), str(VECTORS)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
