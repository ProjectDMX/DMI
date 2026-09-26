"""The kBlock admission timeout in the native sink, pinned deterministically.

Compiles and runs tests/native/test_pack_sink_timeout.cpp against the real
pack_sink.cpp. The C++ test wedges the whole pipeline bottom-up — the
stager parked inside Spool::Stage through Spool::SetStageHookForTesting,
the stage queue full behind it, the packer blocked in SealForStager, the
admission queue full behind the packer — so a further kBlock submit waits
on space that provably cannot free, times out, and must return kTimedOut
with timed_out_records == 1. No admission along the way ever depends on
where the packer thread happens to be.

This is the native-tier coverage of the deadline wait loop in
PackSink::Submit; the conformance-driver suite (test_native_pack_sink.py)
covers the admission bounds around it.

Needs a C++17 compiler and libcrypto (the same as conformance_sink).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_blocked_pipeline_times_out_and_counts_it(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")

    root = Path(__file__).resolve().parents[1]
    csrc = root / "native" / "csrc"
    source = root / "tests" / "native" / "test_pack_sink_timeout.cpp"
    executable = tmp_path / "test_pack_sink_timeout"
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
            str(csrc / "sink" / "pack_sink.cpp"),
            str(csrc / "sink" / "object_key.cpp"),
            str(csrc / "pack" / "pack_builder.cpp"),
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
