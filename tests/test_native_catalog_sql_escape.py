"""The native SQL string escaper against clickhouse-driver's, character
for character.

clickhouse-driver's ``escape_chars_map`` is the oracle: every statement the
native catalog renders inline reproduces what the driver would have sent,
so any character the driver escapes and the port does not is a textual
divergence on a live server -- silent when the server happens to accept
both spellings, and an unterminated string literal when it does not.

The live suite gates this end to end off ``system.query_log``; this file is
the cheap companion, so the gate does not depend on a reachable server.
Both text validators admit these bytes: ``model.py``'s ``_validate_text``
only checks non-empty UTF-8 inside a byte limit, and the native side only
bounds the lease holder's length, so a CR- or NUL-bearing store_id is
legal on both sides and must render the same on both.

Build: make -C native build/conformance_catalog
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "native" / "build" / "conformance_catalog"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]


def _escape(values):
    """Round-trip each value through the driver's server-free escape op."""
    proc = subprocess.Popen(
        [str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        out = []
        for value in values:
            proc.stdin.write(json.dumps({"op": "escape", "value": value})
                             + "\n")
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            assert reply["ok"], reply
            out.append(reply["escaped"])
        return out
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


def _oracle(value):
    from clickhouse_driver.util.escape import escape_param

    return escape_param(value, None)


def test_every_character_the_driver_escapes_is_escaped_the_same_way():
    """One case per escape_chars_map entry, plus the whole map at once.

    Four of the ten used to be covered -- backslash, quote, newline, tab --
    and the other six (\\b \\f \\r \\0 \\a \\v) passed through raw.
    """
    from clickhouse_driver.util.escape import escape_chars_map

    cases = [f"before{c}after" for c in escape_chars_map]
    cases.append("".join(escape_chars_map))
    cases.append("it's\\odd\r\n\t\b\f\v\a\0")
    for value, native in zip(cases, _escape(cases)):
        assert native == _oracle(value), (
            f"escaping diverged for {value!r}: native {native!r} vs "
            f"driver {_oracle(value)!r}")


def test_the_ordinary_characters_are_left_alone():
    """The escaper must not invent escapes the driver does not emit."""
    plain = ["", "plain", "dmi_native_store", "a-b_c.d/e", "\x7f", "héllo",
             "tab\tand quote'"]
    for value, native in zip(plain, _escape(plain)):
        assert native == _oracle(value), value
