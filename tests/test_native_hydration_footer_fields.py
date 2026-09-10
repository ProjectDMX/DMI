"""The footer-binding decoder: rendered footer rows against decoded catalog text.

hydrate binds every catalog descriptor to the pack footer by comparing the
32 fields of the footer's rendered VALUES row (pack_index's renderer, SQL
escaping and all) against the catalog row the reader returns (TSV escapes
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


def test_a_full_32_field_row_decodes_to_32_fields():
    """A rank-2 shape's inner comma must not split the row."""
    strings = ["capture-0", "t", "e", "r", "s", "q", "n", "m", "mr"]
    row = ",".join(
        [_quote(s) for s in strings]
        + ["NULL", _quote("v"), _quote("block\\resid"), "3", "0", "0", "0",
           "1", "0", _quote("float32"), "[2,8]", "1700000000000000000",
           "toUUID('0190e8d0-4b2a-7c3e-9f00-0123456789ab')", _quote("local"),
           _quote("packs/a\\b.dmi-pack"), "4096", _quote("c" * 64), "1", "64",
           "64", "64", _quote("none"), _quote("deadbeef")]
    )
    fields = _fields(row)
    assert len(fields) == 32, fields
    assert fields[11]["text"] == "block\\resid"
    assert fields[19]["text"] == "[2,8]"
    assert fields[21]["text"] == "0190e8d0-4b2a-7c3e-9f00-0123456789ab"
    assert fields[23]["text"] == "packs/a\\b.dmi-pack"
    assert fields[9]["null"] is True


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
