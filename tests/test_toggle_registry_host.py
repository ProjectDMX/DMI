"""Host-only tests of ToggleRegistry exec rebinding (no CUDA toolkit or GPU).

The host half of native/csrc/ring/toggle_registry.cu (namespace ring_py) is
compiled unchanged against a CPU stand-in for the CUDA runtime API
(tests/native/ring/cpu_cuda_stub/), together with
tests/native/ring/test_toggle_registry_host.cpp.  The device-kernel half
(namespace ring) is left out.  tests/native/ring/test_toggle_registry.cu
covers the same registry on a GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_toggle_registry_rebind_resets_per_exec_state(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[1]
    ring_dir = root / "native" / "csrc" / "ring"
    tests_dir = root / "tests" / "native" / "ring"
    source = (ring_dir / "toggle_registry.cu").read_text()
    host, separator, _ = source.partition("\nnamespace ring {")
    assert separator, "toggle_registry.cu no longer splits at namespace ring"
    assert host.rstrip().endswith("}  // namespace ring_py")
    generated = tmp_path / "toggle_registry_host_test.cpp"
    generated.write_text(
        host + "\n" + (tests_dir / "test_toggle_registry_host.cpp").read_text())
    executable = tmp_path / "toggle_registry_host_test"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-DDMX_NO_DEVICE_GRAPH_UPDATE",
         f"-I{tests_dir / 'cpu_cuda_stub'}", f"-I{ring_dir}",
         str(generated), "-o", str(executable)],
        capture_output=True, text=True, check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True,
                         check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "toggle registry host: ok" in run.stdout
