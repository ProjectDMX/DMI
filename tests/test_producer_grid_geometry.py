"""Host-side checks of the D2D producer grid geometry (producer.cuh).

Compiles tests/native/ring/test_producer_grid_geometry.cpp with the host C++
compiler. The helpers are plain integer arithmetic shared with the CUDA
launchers, so no CUDA toolkit or device is needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_producer_grid_geometry(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[1]
    source = root / "tests" / "native" / "ring" / "test_producer_grid_geometry.cpp"
    executable = tmp_path / "test_producer_grid_geometry"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
         f"-I{root / 'native' / 'csrc'}", str(source), "-o", str(executable)],
        capture_output=True, text=True, check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True,
                         check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "producer grid geometry: ok" in run.stdout
