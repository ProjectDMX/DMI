"""The catalog driver refuses an integer literal it cannot represent.

The wire protocol's integer fields are all 64 bits wide, and the shared JSON
scan reported anything wider as ``-1``. The catalog driver read that -1 as a
legal value everywhere: the unsigned config fields
(``lease_ttl_ns``/``publish_timeout_ns``/``clock_skew_ns``/``insert_quorum``,
the reader's read bounds, the index versions and row counts) cast it to
18446744073709551615, and a ``uint``/``int`` query parameter rendered
18446744073709551615 or -1 straight into a statement -- valid SQL carrying a
value nobody asked for, which is the same failure mode
``test_native_catalog_sql_substitute.py`` pins for the scanner.

The ops used here are the driver's session-less ones (``substitute``) plus
``open``, whose ClickHouse client is constructed but never dialled, so this
whole file runs in the CPU gate with no server.

The other half of the contract is pinned here too: [-2**63, 2**64 - 1] is
LEGAL and must round-trip exactly. ``step_number`` and friends are UInt64 in
the catalog, so a migration that refused above INT64_MAX would break parity
in the other direction.

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

# Anything wider than the 64-bit union, in each direction and each spelling.
OUT_OF_RANGE = [
    2**64 + 1,
    2**64,
    int("9" * 40),
    -(2**63) - 1,
    -int("9" * 40),
]

# The union's own edges, which must all still be accepted.
IN_RANGE = [0, 1, 2**63 - 1, 2**63, 2**64 - 1]


def _call(**fields) -> dict:
    proc = subprocess.run(
        [str(DRIVER)],
        input=json.dumps(fields) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"driver produced no output: {proc.stderr}"
    return json.loads(lines[0])


def _open(**overrides) -> dict:
    request = {
        "op": "open",
        "database": "bounds_db",
        "table_prefix": "bounds",
        "lease_ttl_ns": 30_000_000_000,
        "publish_timeout_ns": 1_000_000_000,
        "clock_skew_ns": 1_000_000,
        "allocation_attempts": 3,
    }
    request.update(overrides)
    return _call(**request)


# --- query parameters ----------------------------------------------------------


@pytest.mark.parametrize("value", OUT_OF_RANGE)
@pytest.mark.parametrize("arm", ("uint", "int"))
def test_a_query_parameter_wider_than_64_bits_is_refused(arm, value):
    """A rendered statement must never carry a value nobody asked for."""
    response = _call(op="substitute", query="SELECT %(v)s",
                     params=[{"name": "v", arm: value}])
    assert not response["ok"], response
    assert response["error"] == "ValueError", response
    # The refusal names the variant arm, which is the key the scan read --
    # the parameter's own name never reaches the integer decode. Pinned by
    # equality, not `arm in message`: "int" is a substring of "uint", so the
    # `int` arm's containment check passed against the `uint` message too,
    # and a driver that read the wrong arm went unnoticed for half the
    # parametrization.
    assert response["message"] == f"{arm} does not fit a 64-bit integer", (
        response)


@pytest.mark.parametrize("value", IN_RANGE)
def test_an_unsigned_query_parameter_keeps_the_whole_union(value):
    """UInt64 params are what the catalog binds, so 2**64 - 1 must render."""
    response = _call(op="substitute", query="SELECT %(v)s",
                     params=[{"name": "v", "uint": value}])
    assert response["ok"], response
    assert response["statement"] == f"SELECT {value}", response


@pytest.mark.parametrize("value", [0, 1, -1, 2**63 - 1, -(2**63)])
def test_a_signed_query_parameter_keeps_the_int64_limits(value):
    response = _call(op="substitute", query="SELECT %(v)s",
                     params=[{"name": "v", "int": value}])
    assert response["ok"], response
    assert response["statement"] == f"SELECT {value}", response


def test_minus_zero_is_still_zero():
    """JSON allows -0 and Python reads it as 0; the scan must not refuse it."""
    proc = subprocess.run(
        [str(DRIVER)],
        input='{"op": "substitute", "query": "SELECT %(v)s", '
              '"params": [{"name": "v", "int": -0}]}\n',
        capture_output=True, text=True, timeout=60,
    )
    response = json.loads(proc.stdout.splitlines()[0])
    assert response["ok"], response
    assert response["statement"] == "SELECT 0", response


# --- the writer config ---------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ("lease_ttl_ns", "publish_timeout_ns", "clock_skew_ns",
     "allocation_attempts", "query_pack_limit", "insert_quorum"),
)
def test_an_open_config_field_wider_than_64_bits_is_refused(field):
    """A TTL of 18446744073709551615 is not the TTL the caller asked for."""
    response = _open(**{field: 2**64 + 1})
    assert not response["ok"], response
    assert response["error"] == "ValueError", response
    assert "does not fit" in response["message"], response
    assert field in response["message"], response


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_an_open_config_field_keeps_the_whole_union(value):
    """The config counters are UInt64, so the union is legal input.

    `lease_ttl_ns` is the one config field with no upper bound of its own
    (the others are cross-checked against it), which makes it the field that
    can carry the union's top end all the way through.
    """
    assert _open(lease_ttl_ns=value)["ok"], value
    assert _open(lease_ttl_ns=2**64 - 1, insert_quorum=value)["ok"], value


# --- the reader's read bounds --------------------------------------------------


def _dead_server_session():
    """A driver pointed at a closed port.

    The read bounds ride on a statement, so reaching them needs an op that
    would talk to a server. Pointing the client at a port nothing listens on
    keeps this in the CPU gate and still tells the two outcomes apart: a
    ValueError is the refusal, a ClickHouseError is the connection the old
    code went on to make with UINT64_MAX bounds attached.
    """
    import os

    env = dict(os.environ)
    env["DMI_CLICKHOUSE_HOST"] = "127.0.0.1"
    env["DMI_CLICKHOUSE_HTTP_PORT"] = "1"
    return subprocess.Popen(
        [str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )


READ_BOUNDS = ("max_rows_to_read", "max_bytes_to_read", "max_execution_time_s")


def _watermark_with(field, value) -> dict:
    """One `current_watermark` against the dead server, carrying field=value."""
    session = _dead_server_session()
    try:
        session.stdin.write(json.dumps({
            "op": "open", "database": "bounds_db", "table_prefix": "bounds",
            "lease_ttl_ns": 30_000_000_000,
            "publish_timeout_ns": 1_000_000_000, "clock_skew_ns": 1_000_000,
            "allocation_attempts": 3,
        }) + "\n")
        session.stdin.flush()
        assert json.loads(session.stdout.readline())["ok"]
        session.stdin.write(json.dumps({
            "op": "current_watermark", field: value,
        }) + "\n")
        session.stdin.flush()
        return json.loads(session.stdout.readline())
    finally:
        session.stdin.close()
        session.wait(timeout=30)


@pytest.mark.parametrize("field", READ_BOUNDS)
def test_a_reader_bound_wider_than_64_bits_is_refused(field):
    """-1 lifted a read bound to UINT64_MAX -- the opposite of a bound."""
    response = _watermark_with(field, 2**64 + 1)
    assert not response["ok"], response
    assert response["error"] == "ValueError", response
    assert "does not fit" in response["message"], response
    assert field in response["message"], response


def _write_descriptors_with(**overrides) -> dict:
    """One `write_descriptors` against the dead server, one descriptor."""
    descriptor = {
        "capture_id": "cap-0", "tenant_id": "t", "experiment_id": "e",
        "run_id": "r", "session_id": "s", "request_id": "q",
        "sequence_id": "n", "model_id": "m", "model_revision": "mr",
        "adapter_revision": None, "capture_policy_version": "v",
        "hook_name": "h", "layer_number": 0, "producer_rank": 0,
        "step_number": 0, "token_start": 0, "token_end": 1,
        "batch_position": 0, "dtype": "uint8", "shape": [4],
        "captured_at_ns": 1,
        "pack_id": "018f0000-0000-7000-8000-00000000dead",
        "store_id": "store", "object_key": "k", "object_bytes": 1,
        "pack_checksum": "c", "pack_record_count": 1, "payload_offset": 0,
        "stored_length": 4, "decoded_length": 4, "codec": "none",
        "payload_checksum": "p",
    }
    descriptor.update(overrides)
    session = _dead_server_session()
    try:
        session.stdin.write(json.dumps({
            "op": "open", "database": "bounds_db", "table_prefix": "bounds",
            "lease_ttl_ns": 30_000_000_000,
            "publish_timeout_ns": 1_000_000_000, "clock_skew_ns": 1_000_000,
            "allocation_attempts": 3,
        }) + "\n")
        session.stdin.flush()
        assert json.loads(session.stdout.readline())["ok"]
        session.stdin.write(json.dumps({
            "op": "write_descriptors", "descriptors": [descriptor],
            "index_version": 1,
        }) + "\n")
        session.stdin.flush()
        return json.loads(session.stdout.readline())
    finally:
        session.stdin.close()
        session.wait(timeout=30)


# --- the descriptor row's signed column ----------------------------------------
#
# `layer_number` is the one capture column the catalog types as SIGNED
# (Int32). The 64-bit union the scan accepts hands back a two's-complement
# bit pattern, so 18446744073709551615 arrives as -1 -- which is that
# column's legal "no layer" sentinel -- and renders straight into the Int32
# column. Live, that wrote a row and `SELECT` read it back as -1, where
# CaptureMetadata raises "layer_number must be an integer in [-1, 2^31 - 1]".
# Here the dead server tells the two outcomes apart exactly as it does for
# the read bounds above: a ValueError is the refusal landing BEFORE the
# insert, a ClickHouseError is the row having been rendered and dispatched.


@pytest.mark.parametrize("value", [2**64 - 1, 2**63, 2**64 - 2])
def test_a_signed_descriptor_column_refuses_the_unsigned_half(value):
    response = _write_descriptors_with(layer_number=value)
    assert not response["ok"], response
    assert response["error"] == "ValueError", response
    assert "layer_number" in response["message"], response
    assert "does not fit" in response["message"], response


@pytest.mark.parametrize("value", [-1, 0, 1, 2**31 - 1, -(2**63), 2**63 - 1])
def test_a_signed_descriptor_column_keeps_the_signed_half(value):
    """The refusal must be the unsigned half, not "this column at all".

    -1 is the legal "no layer" sentinel and 2**31 - 1 the column's top; the
    int64 limits are outside the COLUMN's range but inside the parse's, and
    the driver renders them for the server to refuse, as it always has.
    """
    response = _write_descriptors_with(layer_number=value)
    assert not response["ok"], response
    assert response["error"] == "ClickHouseError", response


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
@pytest.mark.parametrize("field", READ_BOUNDS)
def test_a_reader_bound_keeps_the_whole_union(field, value):
    """The accepting half this file's docstring promises and never asserted.

    Without the mirror, the refusal above cannot tell "refuses the literal
    it cannot represent" from "refuses this key at all", and a driver
    narrowed to INT64_MAX would pass the whole file. These three settings
    are UInt64 on the server, so 2**63 and 2**64 - 1 have to travel.

    `_dead_server_session` supplies the discriminator its own docstring
    names: a ValueError is the refusal, a ClickHouseError is the connection
    the driver went on to attempt with the bound accepted and attached.
    Only `max_rows_to_read` had a positive test anywhere before this, and
    only at 4242424 (`test_the_watermark_read_carries_the_configured_bounds`).
    """
    response = _watermark_with(field, value)
    assert not response["ok"], response
    assert response["error"] == "ClickHouseError", response
