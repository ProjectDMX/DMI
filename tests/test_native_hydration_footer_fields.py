"""The footer-binding decoder: rendered footer rows against decoded catalog text.

hydrate binds every catalog descriptor to the pack footer by comparing the
fields of the footer's rendered VALUES row -- one per CAPTURE_COLUMNS entry
bar index_version (pack_index's renderer, SQL
escaping and all) -- against the catalog row the reader returns (TSV escapes
already undone, a NULL arriving as the empty string). The two sides only
compare correctly in ONE representation, so the decoder is pinned here on
the CPU gate through the driver's session-less `footer_row_fields` op:

* a quoted string is decoded through sql_quote's map -- backslash, quote
  and tab-bearing hook names compared equal on the live catalog only once
  this held (they were refused as "does not match the pack footer");
* the unquoted NULL token is a kind, not a spelling -- the string 'NULL'
  stays the four-letter string and does NOT match a catalog NULL;
* toUUID('...') unwraps to the bare UUID the catalog carries;
* arrays split on top-level commas only, spaces dropped.

Build: make -C native build/conformance_catalog
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dmi.storage.capture.clickhouse_schema import CAPTURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "native" / "build" / "conformance_catalog"

# The footer's VALUES row is every capture column but the trailing
# index_version, which the indexer supplies rather than the pack.
FOOTER_COLUMNS = CAPTURE_COLUMNS[:-1]

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]


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


def _quote(value: str) -> str:
    """The renderer's own sql_quote, through the driver's `escape` op."""
    response = _call(op="escape", value=value)
    assert response["ok"], response
    return response["escaped"]


def _fields(row: str, catalog: list[str] | None = None) -> list[dict]:
    fields = {"op": "footer_row_fields", "row": row}
    if catalog is not None:
        fields["catalog"] = catalog
    response = _call(**fields)
    assert response["ok"], response
    return response["fields"]


@pytest.mark.parametrize("hook_name", [
    "block\\resid",
    "block'quoted",
    "block\tresid",
    "block\nresid",
    "plain",
])
def test_a_quoted_string_is_decoded_before_comparison(hook_name):
    rendered = _quote(hook_name)
    if hook_name != "plain":
        assert "\\" in rendered, rendered  # the escaper did rewrite it
    (field,) = _fields(rendered, [hook_name])
    assert field == {"null": False, "text": hook_name, "matches": True}


def test_sql_null_and_the_string_null_stay_apart():
    # Footer NULL (absent adapter_revision) matches the catalog's empty
    # field and nothing else.
    (null_field,) = _fields("NULL", [""])
    assert null_field == {"null": True, "text": "", "matches": True}
    (null_vs_text,) = _fields("NULL", ["NULL"])
    assert null_vs_text["matches"] is False

    # Footer 'NULL' (the four-letter STRING) matches the string and not a
    # catalog NULL: a pack whose adapter_revision is literally "NULL" must
    # not bind to a catalog row where that field was made SQL NULL.
    (text_field,) = _fields(_quote("NULL"), ["NULL"])
    assert text_field == {"null": False, "text": "NULL", "matches": True}
    (text_vs_null,) = _fields(_quote("NULL"), [""])
    assert text_vs_null["matches"] is False


def test_uuid_literals_arrays_and_numbers_split_at_top_level():
    pack_id = "0190e8d0-4b2a-7c3e-9f00-0123456789ab"
    row = ",".join([
        _quote("capture-0"),
        f"toUUID('{pack_id}')",
        "[2, 8]",
        "-1",
        "18446744073709551615",
        "NULL",
        _quote("a,b"),
    ])
    fields = _fields(row, ["capture-0", pack_id, "[2,8]", "-1",
                           "18446744073709551615", "", "a,b"])
    assert [f["text"] for f in fields] == [
        "capture-0", pack_id, "[2,8]", "-1", "18446744073709551615", "",
        "a,b",
    ]
    assert [f["null"] for f in fields] == [
        False, False, False, False, False, True, False]
    assert all(f["matches"] for f in fields), fields


def test_a_full_footer_row_decodes_to_one_field_per_footer_column():
    """A rank-2 shape's inner comma must not split the row.

    Width and positions come from CAPTURE_COLUMNS, the oracle this decoder
    is a port of, not from a hand-counted 32. The footer carries every
    capture column except the trailing index_version, and
    hydration.cpp:708 guards on that same width -- so with the count
    hard-coded, adding a column left this row at 32 tokens, the test
    passing, and the native guard silently out of date. The row is
    assembled per column name for the same reason: a new column has no
    rendering here and says so by name.
    """
    pack_id = "0190e8d0-4b2a-7c3e-9f00-0123456789ab"
    rendered = {
        "capture_id": _quote("capture-0"),
        "tenant_id": _quote("t"),
        "experiment_id": _quote("e"),
        "run_id": _quote("r"),
        "session_id": _quote("s"),
        "request_id": _quote("q"),
        "sequence_id": _quote("n"),
        "model_id": _quote("m"),
        "model_revision": _quote("mr"),
        "adapter_revision": "NULL",
        "capture_policy_version": _quote("v"),
        "hook_name": _quote("block\\resid"),
        "layer_number": "3",
        "producer_rank": "0",
        "step_number": "0",
        "token_start": "0",
        "token_end": "1",
        "batch_position": "0",
        "dtype": _quote("float32"),
        "shape": "[2,8]",
        "captured_at_ns": "1700000000000000000",
        "pack_id": f"toUUID('{pack_id}')",
        "store_id": _quote("local"),
        "object_key": _quote("packs/a\\b.dmi-pack"),
        "object_bytes": "4096",
        "pack_checksum": _quote("c" * 64),
        "pack_record_count": "1",
        "payload_offset": "64",
        "stored_length": "64",
        "decoded_length": "64",
        "codec": _quote("none"),
        "payload_checksum": _quote("deadbeef"),
    }
    assert sorted(rendered) == sorted(FOOTER_COLUMNS), (
        sorted(set(FOOTER_COLUMNS) - set(rendered)),
        sorted(set(rendered) - set(FOOTER_COLUMNS)),
    )

    row = ",".join(rendered[column] for column in FOOTER_COLUMNS)
    fields = _fields(row)
    assert len(fields) == len(FOOTER_COLUMNS), fields
    by_column = dict(zip(FOOTER_COLUMNS, fields))
    assert by_column["hook_name"]["text"] == "block\\resid"
    assert by_column["shape"]["text"] == "[2,8]"
    assert by_column["pack_id"]["text"] == pack_id
    assert by_column["object_key"]["text"] == "packs/a\\b.dmi-pack"
    assert by_column["adapter_revision"]["null"] is True


def test_a_mismatching_decoded_value_is_still_a_mismatch():
    (field,) = _fields(_quote("block\\resid"), ["block\\\\resid"])
    assert field["matches"] is False
    (field,) = _fields(_quote("h"), ["evil"])
    assert field["matches"] is False


# --- the catalog half of the same comparison ---------------------------------
#
# hydrate compares the footer fields above against the catalog descriptor
# fields, which come out of the reader's resolved-tuple parser. That parser
# is the only place that can still see whether a value was quoted, so it is
# the only place that can tell a SQL NULL from the four-character string
# "NULL" -- and it decides what "field 10" (adapter_revision, the schema's
# one Nullable column) means on the catalog side.
#
# This was live-only, and it is exactly where the first attempt at the NULL
# fix broke: making the footer side null-aware while the catalog side still
# reported the text "NULL" refused every capture with adapter_revision=None.


def _tuple_fields(rendered: str) -> list[str]:
    response = _call(op="tuple_fields", tuple=rendered)
    assert response["ok"], response
    return response["fields"]


def test_a_sql_null_in_the_resolved_tuple_is_not_the_string_null():
    # ClickHouse renders a SQL NULL inside the tuple as a bare token and the
    # string as a quoted literal. Unquoted -> "" (the sentinel; no legal
    # adapter_revision is empty), quoted -> the four characters.
    assert _tuple_fields("('a',NULL,'b')") == ["a", "", "b"]
    assert _tuple_fields("('a','NULL','b')") == ["a", "NULL", "b"]
    # Alone in the tuple, and unwrapped, both ways.
    assert _tuple_fields("(NULL)") == [""]
    assert _tuple_fields("('NULL')") == ["NULL"]


def test_the_two_null_forms_bind_to_the_matching_footer_field_only():
    """The catalog half and the footer half, compared as hydrate compares them."""
    catalog_null, catalog_text = _tuple_fields("(NULL)")[0], _tuple_fields("('NULL')")[0]

    # A footer NULL binds to a catalog NULL and not to the string.
    assert _fields("NULL", [catalog_null])[0]["matches"] is True
    assert _fields("NULL", [catalog_text])[0]["matches"] is False
    # A footer 'NULL' binds to the string and not to a catalog NULL. This is
    # the pair the reviewer's repro turns on.
    assert _fields(_quote("NULL"), [catalog_text])[0]["matches"] is True
    assert _fields(_quote("NULL"), [catalog_null])[0]["matches"] is False


def test_the_resolved_tuple_keeps_values_the_footer_binding_compares():
    """Escapes, arrays and numbers survive unchanged -- NULL is the only remap."""
    assert _tuple_fields("('block\\\\resid')") == ["block\\resid"]
    assert _tuple_fields("('block\\'quoted')") == ["block'quoted"]
    assert _tuple_fields("('block\\tresid')") == ["block\tresid"]
    # A rank>=2 shape's inner comma must not split the tuple, and a number
    # is never mistaken for a null.
    assert _tuple_fields("('h',[2,8],0,'none')") == ["h", "[2,8]", "0", "none"]
    # A value that merely CONTAINS the token is untouched.
    assert _tuple_fields("('NULLABLE','a NULL b')") == ["NULLABLE", "a NULL b"]


# --- the two decoders have to agree, byte for byte ---------------------------
#
# hydrate compares the tuple decoder's output (above) against the footer
# decoder's (unquote_sql, sql_quote's map entry for entry). Where the two
# disagree on ONE byte the binding refuses a legal capture with "catalog
# descriptor does not match the pack footer: field N", and nothing but a
# live hydrate of a capture carrying that byte could see it.
#
# The rendering below is MEASURED, not assumed: `SELECT tuple(concat('block',
# char(N), 'resid')) FORMAT TSV` was driven against the live server for the
# whole 1..127 range (plus char(0)). ClickHouse escapes exactly eight bytes
# inside a tuple -- \0 \b \t \n \f \r \' \\ -- and leaves every other one
# RAW, including 0x07 (BEL) and 0x0B (VT), which sql_quote does rewrite (as
# \a and \v). So the two escape SETS are deliberately different, and it is
# the decoded bytes, not the spellings, that have to match.
TUPLE_RENDERING = {
    0x00: "\\0", 0x07: "\x07", 0x08: "\\b", 0x09: "\\t", 0x0A: "\\n",
    0x0B: "\x0b", 0x0C: "\\f", 0x0D: "\\r", 0x27: "\\'", 0x5C: "\\\\",
}


@pytest.mark.parametrize("byte", sorted(TUPLE_RENDERING))
def test_the_tuple_and_footer_decoders_agree_on_every_byte_clickhouse_rewrites(byte):
    """Both halves of the footer binding decode to the same bytes.

    \\b and \\f decoded to the letters "b" and "f" on the catalog side while
    the footer side decoded them to 0x08 and 0x0C, so a hook_name carrying
    either (legal: _validate_text only asks for non-empty UTF-8 under 512
    bytes) staged, uploaded and indexed, and then failed to hydrate.
    """
    value = "block" + chr(byte) + "resid"
    (catalog,) = _tuple_fields(f"('block{TUPLE_RENDERING[byte]}resid')")
    assert catalog == value
    assert _fields(_quote(value), [catalog])[0]["matches"] is True
