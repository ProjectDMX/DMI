"""Spool capacity accounting: committed files and in-flight reservations.

Compiles and runs tests/native/test_spool_reservations.cpp against the
real spool.cpp. Two regressions are pinned together because a fix for one
broke the other:

* the SERIAL case -- an uploader through a second Spool object removes a
  ready file, and the writer's next stage must reconcile from the directory
  rather than refuse on a stale counter;
* the CONCURRENT case -- a stager that holds a reservation but has written
  nothing yet must keep that reservation across another stager's
  reconciliation scan (the scan cannot see it), so the second stager is
  refused and only 1000 bytes land against a 1500 limit, not 2000.

The C++ test pauses the first stager through Spool::SetStageHookForTesting,
a seam placed exactly between the reservation and the temp-file write.

Needs a C++17 compiler and libcrypto (the same as conformance_spool).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_spool_reservations_survive_reconciliation(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")

    root = Path(__file__).resolve().parents[1]
    csrc = root / "native" / "csrc"
    source = root / "tests" / "native" / "test_spool_reservations.cpp"
    executable = tmp_path / "test_spool_reservations"
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
