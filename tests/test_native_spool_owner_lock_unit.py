"""The spool owner lock in process: tests/native/test_spool_owner_lock.cpp.

Compiles the C++ test against the real spool.cpp and runs it. It pins what
the cross-process driver tests (test_native_spool_owner_lock.py) cannot
reach: two Spool objects in ONE process that both take a directory refuse
each other -- the regression the engine avoids by holding one lock and
opening its sink's and service's Spools with held_by_caller -- the node-local
check through its statfs test seam, adoption's try-lock, and the directory
layout. One case forks, so the refusal across processes is covered here too.

Needs a C++17 compiler and libcrypto (the same as conformance_spool).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_spool_owner_lock_unit(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")

    root = Path(__file__).resolve().parents[1]
    csrc = root / "native" / "csrc"
    source = root / "tests" / "native" / "test_spool_owner_lock.cpp"
    executable = tmp_path / "test_spool_owner_lock"
    extra: list[str] = []
    # Homebrew OpenSSL is not on the default search path on macOS; Linux
    # hosts (CI) find libcrypto without help.
    for candidate in ("/opt/homebrew/opt/openssl@3", "/usr/local/opt/openssl@3"):
        if Path(candidate, "include", "openssl", "sha.h").exists():
            extra += [f"-I{candidate}/include", f"-L{candidate}/lib"]
            break
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-pthread",
            f"-I{csrc}",
            *extra,
            str(source),
            str(csrc / "store" / "spool.cpp"),
            "-o",
            str(executable),
            "-lcrypto",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_result.returncode != 0 and "openssl" in (
        compile_result.stdout + compile_result.stderr
    ):
        pytest.skip("libcrypto headers are unavailable: "
                    + compile_result.stderr[-400:])
    assert compile_result.returncode == 0, (
        compile_result.stdout + compile_result.stderr
    )

    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SPOOL_TEST_ROOT": str(tmp_path)},
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
