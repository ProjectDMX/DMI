"""
setup.py — custom build_ext that compiles the DMI native backend.

Source-distribution / compile-on-install model.  torch must already be
importable when the build runs so the Makefile can query its include/lib
paths; always install with --no-build-isolation:

    # one-time: install a CUDA-capable torch first
    pip install torch

    # then install DMI (compiles the native backend)
    pip install -e . --no-build-isolation

The build chain (delegated from build_ext.run):
  1. git submodule update --init libs/clickhouse-cpp   (if not present)
  2. cmake -S libs/clickhouse-cpp -B …/build           (configure)
  3. cmake --build …/build                             (build static lib)
  4. make -C monitoring                                (build .so via nvcc/g++)

Artifacts land at:
  monitoring/monitoring_native_backend.<EXT_SUFFIX>.so
  monitoring_native_backend.<EXT_SUFFIX>.so            (project root copy)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _BuildExt


ROOT = Path(__file__).parent.resolve()
MONITORING_DIR = ROOT / "monitoring"
CLICKHOUSE_SRC = ROOT / "libs" / "clickhouse-cpp"
CLICKHOUSE_BUILD = CLICKHOUSE_SRC / "build"
# Stamp file: cmake build writes this when clickhouse-cpp is ready.
_CLICKHOUSE_STAMP = CLICKHOUSE_BUILD / "clickhouse" / "libclickhouse-cpp-lib.a"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, *, cwd: Path, env: dict | None = None) -> None:
    print(f"[DMI build] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd), env=env)


def _init_clickhouse_submodule() -> None:
    """Initialise libs/clickhouse-cpp if the working tree doesn't have it."""
    if not (CLICKHOUSE_SRC / "CMakeLists.txt").exists():
        _run(
            [
                "git", "submodule", "update", "--init", "--recursive",
                "libs/clickhouse-cpp",
            ],
            cwd=ROOT,
        )


def _build_clickhouse() -> None:
    """Configure + build the clickhouse-cpp static library via cmake."""
    if _CLICKHOUSE_STAMP.exists():
        print("[DMI build] clickhouse-cpp already built, skipping.", flush=True)
        return
    CLICKHOUSE_BUILD.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "cmake",
            "-S", CLICKHOUSE_SRC,
            "-B", CLICKHOUSE_BUILD,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARK=OFF",
        ],
        cwd=ROOT,
    )
    _run(
        ["cmake", "--build", CLICKHOUSE_BUILD, f"-j{os.cpu_count() or 4}"],
        cwd=ROOT,
    )


def _build_native_backend() -> None:
    """Compile monitoring_native_backend.so via the existing Makefile."""
    _run(
        ["make", "-C", MONITORING_DIR, f"-j{os.cpu_count() or 4}"],
        cwd=ROOT,
        # PYTHON must point at the active interpreter so the Makefile can
        # query torch include/lib paths and the correct EXT_SUFFIX.
        env={**os.environ, "PYTHON": sys.executable},
    )


# ---------------------------------------------------------------------------
# Custom build_ext
# ---------------------------------------------------------------------------

class NativeBuildExt(_BuildExt):
    """Delegates the build entirely to the existing Makefile.

    A placeholder Extension in ext_modules causes pip to invoke build_ext;
    we override every method that would touch the placeholder so that
    setuptools never tries to compile it — all real work happens in run().
    """

    def run(self) -> None:
        if os.environ.get("SKIP_NATIVE_BUILD"):
            print("[DMI build] SKIP_NATIVE_BUILD set — skipping native backend.", flush=True)
            return
        _init_clickhouse_submodule()
        _build_clickhouse()
        _build_native_backend()
        # Intentionally skip super().run(): no distutils-managed C sources.

    def build_extension(self, ext) -> None:  # noqa: ARG002
        # Placeholder; real artifacts come from the Makefile.
        pass

    def copy_extensions_to_source(self) -> None:
        # For editable installs: the .so is already at the project root and
        # inside monitoring/ — no copying needed.
        pass

    def get_outputs(self) -> list[str]:
        # Don't advertise the placeholder as an output file.
        return []


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    cmdclass={"build_ext": NativeBuildExt},
    # The placeholder extension below exists only to make pip invoke
    # build_ext.  The real artifact is produced by the Makefile; sources=[]
    # is intentional and safe because build_extension() is a no-op.
    ext_modules=[
        Extension("monitoring._dmi_native_sentinel", sources=[]),
    ],
)
