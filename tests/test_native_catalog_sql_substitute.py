"""The native ``%(name)s`` scanner against clickhouse-driver's, statement
for statement.

``test_native_catalog_sql_escape.py`` is the escaper's half of this gate:
it pins how ONE value is rendered. This is the other half -- how the
statement is WALKED -- and it was the half without a gate. The escaper had
a session-less ``escape`` op and the scanner had nothing, so the port could
render every value exactly as the driver does and still send the server
different SQL.

The oracle is ``clickhouse_driver/client.py``'s
``substitute_params``, whose body is ``query % escape_params(params)``.
``%`` is a SINGLE left-to-right pass: what it writes out it never reads
again. Substituting one parameter at a time across the whole statement is
not the same thing -- the next parameter's pass starts at the top and
reaches into text an earlier parameter already inserted. The lease claim
binds ``holder``, ``lease_id``, ``term`` and ``ttl_ns`` together and
``holder`` sorts first, so a holder that literally reads ``%(ttl_ns)s``
(a legal 11-byte string; only its length is checked, on either side) was
stored as the TTL: valid SQL carrying the wrong value.

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

# The lease claim, as `lease_coordinator.cpp`'s `insert` renders it: four
# parameters in one statement, which is what makes the cross-parameter
# reach reachable at all.
CLAIM = (
    "INSERT INTO `db`.`p_publisher_lease` "
    "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
    "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
    "now_ns, now_ns + toUInt64(%(ttl_ns)s) "
    "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)"
)
LEASE_ID = "015f2ebf-4c1a-4b7f-9f4e-2c6f8a1d0e33"


def _wire(params):
    """A Params map as the driver's `substitute` op reads it.

    One key per element names the variant arm, because a str is quoted
    and the integers are not.
    """
    out = []
    for name, value in params.items():
        arm = "uint" if isinstance(value, int) else "str"
        out.append({"name": name, arm: value})
    return out


def _substitute(cases):
    """Render each (query, params) through the server-free substitute op."""
    proc = subprocess.Popen(
        [str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        out = []
        for query, params in cases:
            proc.stdin.write(json.dumps({
                "op": "substitute", "query": query,
                "params": _wire(params),
            }) + "\n")
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            assert reply["ok"], reply
            out.append(reply["statement"])
        return out
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


def _oracle(query, params):
    """clickhouse-driver's own substitution, verbatim."""
    from clickhouse_driver.util.escape import escape_params

    return query % escape_params(params, None)


def _claim(holder):
    return CLAIM, {
        "term": 2, "lease_id": LEASE_ID, "holder": holder,
        "ttl_ns": 30_000_000_000,
    }


def test_a_holder_that_looks_like_another_placeholder_is_not_one():
    """The defect, at the CPU level: a value is never rescanned.

    ``holder`` sorts before ``lease_id``, ``term`` and ``ttl_ns``, so
    under per-parameter passes all three later parameters reached into
    the rendered holder. The numeric two produced valid SQL carrying the
    wrong holder; the UUID one doubled the quotes and was refused by the
    server outright.
    """
    cases = [_claim(h) for h in ("%(ttl_ns)s", "%(term)s", "%(lease_id)s",
                                 "%(holder)s", "%(nothing)s",
                                 "ttl=%(ttl_ns)s and term=%(term)s")]
    # The control: an ordinary holder must keep working, so a regression
    # that breaks every holder is distinguishable from this one.
    cases.append(_claim("plain-holder"))
    for (query, params), native in zip(cases, _substitute(cases)):
        assert native == _oracle(query, params), (
            f"the native scanner rendered {params['holder']!r} as "
            f"{native!r}; the driver renders it as "
            f"{_oracle(query, params)!r}")


def test_the_escaped_value_is_the_drivers_and_lands_where_the_driver_puts_it():
    """The two halves together: escape_chars_map through the scanner.

    ``test_native_catalog_sql_escape.py`` gates the rendering of the
    value alone. This gates the same value once it is inside a statement,
    which is what the server receives.
    """
    from clickhouse_driver.util.escape import escape_chars_map

    holders = [f"before{c}after" for c in escape_chars_map]
    holders.append("".join(escape_chars_map))
    holders.append("it's\\odd\r\n\t\b\f\v\a\0")
    cases = [_claim(h) for h in holders]
    for (query, params), native in zip(cases, _substitute(cases)):
        assert native == _oracle(query, params), params["holder"]


def test_a_parameter_used_more_than_once_is_rendered_at_every_use():
    """Two of the ported statements bind one parameter twice.

    ``%`` fills every occurrence, and so must the scanner -- a scan that
    stopped at the first would leave a live ``%(name)s`` in the SQL.
    """
    cases = [
        ("SELECT %(term)s, %(term)s, %(term)s", {"term": 7}),
        ("SELECT %(a)s, %(b)s, %(a)s, %(b)s", {"a": "x", "b": 3}),
        # Adjacent, with no text between them to resynchronise on.
        ("SELECT %(a)s%(a)s%(a)s", {"a": "-"}),
    ]
    for (query, params), native in zip(cases, _substitute(cases)):
        assert native == _oracle(query, params), query


def test_a_value_that_is_itself_a_placeholder_is_left_as_written():
    """Self-reference, the case the removed cursor bookkeeping guarded.

    Advancing past each replacement stopped the scan from eating its own
    output, which is why this one case already agreed with the driver.
    It has to keep agreeing.
    """
    cases = [
        ("SELECT %(holder)s", {"holder": "%(holder)s"}),
        ("SELECT %(a)s", {"a": "%(a)s%(a)s%(a)s"}),
    ]
    for (query, params), native in zip(cases, _substitute(cases)):
        assert native == _oracle(query, params), query


def test_text_that_is_not_a_placeholder_is_passed_through_untouched():
    """Pinned as-is, NOT as parity: here the port and the driver differ.

    ``%`` raises on all three of these -- KeyError for an unknown key,
    ``"incomplete format key"`` for a ``%(`` with no ``)s``, and
    ``"unsupported format character"`` for a bare ``%``. The port instead
    passes the text through, and every statement it renders is a literal
    in this repository with its parameters supplied beside it, so no
    caller reaches the divergence. It is recorded rather than repaired
    because changing it is a different change from this one; the point of
    the assertions is that rewriting the scanner did not move it.
    """
    cases = [
        # An unknown name: the placeholder stays, a known one beside it
        # is still filled.
        ("SELECT %(nope)s, %(term)s", {"term": 7}),
        # `%(` with no `)s` at all.
        ("SELECT %(term", {"term": 7}),
        # `%(` whose `)s` belongs to a LATER placeholder: the scan
        # resumes inside the malformed text and finds the good one.
        ("SELECT %(term)x and %(other)s", {"term": 7, "other": 8}),
        # A bare `%`, including the driver's own `%s` and a percentage.
        ("SELECT 'a%b', '%s', '100%', %(term)s", {"term": 7}),
        # No parameters supplied at all.
        ("SELECT %(term)s", {}),
    ]
    assert _substitute(cases) == [
        "SELECT %(nope)s, 7",
        "SELECT %(term",
        "SELECT %(term)x and 8",
        "SELECT 'a%b', '%s', '100%', 7",
        "SELECT %(term)s",
    ]
