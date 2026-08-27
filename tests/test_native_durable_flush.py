from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.cpu
def test_native_nonterminal_flush_deadline_and_async_failure(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")

    root = Path(__file__).resolve().parents[1]
    source = root / "tests" / "native" / "test_durable_flush.cpp"
    executable = tmp_path / "test_durable_flush"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-pthread",
            f"-I{root / 'native' / 'csrc'}",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, (
        compile_result.stdout + compile_result.stderr
    )

    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
