"""Unescape's \\uXXXX failure path: the decoder must stay on the string.

Compiles and runs tests/native/test_json_unescape.cpp against the real
common/json.cpp. hex4 validates the four digits, but the advance (`q += 5`)
ran BEFORE its verdict, so a malformed short escape whose "digits" include
the field's real closing quote (`"x\\u12", ...`) stepped past that quote and
the value silently swallowed following JSON text -- every FindString caller
with no `ok` flag (pack footer parsing, the reader's cursor) received a
value containing structural JSON -- and a `\\u12` at the end of the buffer
left q strictly past text.size(), breaking json.h's "returns with `q` on
the closing quote" contract.

Needs a C++17 compiler (no libcrypto, unlike conformance_spool).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_unescape_failure_keeps_the_decoder_on_the_string(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")

    root = Path(__file__).resolve().parents[1]
    csrc = root / "native" / "csrc"
    source = root / "tests" / "native" / "test_json_unescape.cpp"
    executable = tmp_path / "test_json_unescape"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            f"-I{csrc}",
            str(source),
            str(csrc / "common" / "json.cpp"),
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
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
