"""C1: the native reader, at parity with the Python CaptureReader.

Both implementations run the same queries against the same catalog and
must return the same descriptors in the same order — items compared
field-for-field in the 32-column projection layout, cursors issued by
either side accepted and walked by the other. The Python reader is the
oracle; the SQL derivations (one aggregate so a row cannot be mixed, the
total resolution order, the membership pair) live in clickhouse_reader.py.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from os import environ
from pathlib import Path

import pytest

# Module-level so the fake-S3 fixture re-exports into this module's
# namespace (function-local imports do not register fixtures).
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, fake_s3,
)

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "native" / "build" / "conformance_catalog"

pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]


class CatalogDriver:
    def __init__(self):
        self.proc = subprocess.Popen(
            [str(DRIVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise AssertionError(
                f"driver returned unparseable JSON for op="
                f"{fields.get('op')!r}: {line[:400]}")

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=30)


import subprocess  # noqa: E402


def _client():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    return clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )


DEFAULTS = {
    "lease_ttl_ns": 30_000_000_000,
    "publish_timeout_ns": 5_000_000_000,
    "clock_skew_ns": 0,
    "allocation_attempts": 16,
}


@contextmanager
def _catalog(**overrides):
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig,
        ClickHouseCatalogWriter,
    )

    client = _client()
    prefix = f"dmi_native_c1_{uuid.uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix, **overrides)
    writer = ClickHouseCatalogWriter(client, config)
    created = False
    try:
        created = True
        writer.ensure_schema()
        yield client, config, prefix
    finally:
        if created:
            writer.drop_schema()


def _descriptor_dicts(count, *, pack_id=None, capture_prefix="capture",
                      tenant="t", hook="resid_pre"):
    from benchmarks.bench_capture_catalog import synthetic_descriptors

    out = []
    for item in synthetic_descriptors(count):
        meta = item.metadata
        loc = item.locator
        entry = {
            "capture_id": f"{capture_prefix}-{len(out)}",
            "tenant_id": tenant,
            "experiment_id": meta.experiment_id,
            "run_id": meta.run_id,
            "session_id": meta.session_id,
            "request_id": meta.request_id,
            "sequence_id": meta.sequence_id,
            "model_id": meta.model_id,
            "model_revision": meta.model_revision,
            "adapter_revision": meta.adapter_revision,
            "capture_policy_version": meta.capture_policy_version,
            "hook_name": hook if len(out) % 2 == 0 else "other_hook",
            "layer_number": meta.layer_number,
            "producer_rank": meta.producer_rank,
            "step_number": meta.step_number,
            "token_start": meta.token_start,
            "token_end": meta.token_end,
            "batch_position": meta.batch_position,
            "dtype": meta.dtype,
            "shape": list(meta.shape),
            "captured_at_ns": meta.captured_at_ns + len(out) * 1_000,
            "pack_id": pack_id or loc.pack_id,
            "store_id": loc.store_id,
            "object_key": loc.object_key,
            "object_bytes": loc.object_bytes,
            "pack_checksum": loc.pack_checksum,
            "pack_record_count": loc.pack_record_count,
            "payload_offset": loc.offset,
            "stored_length": loc.stored_length,
            "decoded_length": loc.decoded_length,
            "codec": loc.codec,
            "payload_checksum": loc.checksum,
        }
        out.append(entry)
    return out


def _publish_native(driver, prefix, descriptors, index_version):
    # Only the lease holder can make a snapshot visible; the publish renews
    # first, so the session must hold the lease before this runs. Every
    # response is asserted — a refused write here is what a silent empty
    # catalog downstream looks like.
    held = driver.call(op="lease")["lease"]
    if held is None:
        response = driver.call(op="acquire", holder="writer")
        assert response["ok"], response
    refs = [{"store_id": descriptors[0]["store_id"],
             "pack_id": descriptors[0]["pack_id"]}]
    written = driver.call(op="write_descriptors", descriptors=descriptors,
                          index_version=index_version)
    assert written["ok"], written
    published = driver.call(
        op="publish_snapshot", index_version=index_version, refs=refs,
        published_at_ns=index_version, indexed_rows=len(descriptors),
        indexed_packs=1)
    assert published["ok"], published
    return refs


def _python_reader(client, config):
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog,
        ClickHouseReaderConfig,
    )

    return ClickHouseCaptureCatalog(
        client, ClickHouseReaderConfig.from_catalog(config))


def _python_page_items(reader, **query_fields):
    from dmi.storage.capture.model import CaptureQuery

    page = reader.search(CaptureQuery(**query_fields))
    return page


def _descriptor_fields(item):
    """One descriptor — native row list or Python CaptureDescriptor — as
    the 32-field string layout both readers return (the projection order;
    adapter_revision None flattens to an empty field, matching the TSV)."""
    if not isinstance(item, (list, tuple)):
        # The native layout: the sort-key columns first (tenant, experiment,
        # run, captured_at_ns, capture_id), then the resolved columns in
        # projection-minus-sort-key order.
        from dmi.storage.capture.clickhouse_reader import _RESOLVED
        meta, loc = item.metadata, item.locator
        values = {
            "capture_id": meta.capture_id, "tenant_id": meta.tenant_id,
            "experiment_id": meta.experiment_id, "run_id": meta.run_id,
            "session_id": meta.session_id, "request_id": meta.request_id,
            "sequence_id": meta.sequence_id, "model_id": meta.model_id,
            "model_revision": meta.model_revision,
            "adapter_revision": meta.adapter_revision,
            "capture_policy_version": meta.capture_policy_version,
            "hook_name": meta.hook_name, "layer_number": meta.layer_number,
            "producer_rank": meta.producer_rank, "step_number": meta.step_number,
            "token_start": meta.token_start, "token_end": meta.token_end,
            "batch_position": meta.batch_position, "dtype": meta.dtype,
            # The wire form, not Python's repr: TSV renders the array
            # without the spaces `str(list(...))` inserts.
            "shape": "[" + ",".join(str(dim) for dim in meta.shape) + "]",
            "captured_at_ns": meta.captured_at_ns, "pack_id": loc.pack_id,
            "store_id": loc.store_id, "object_key": loc.object_key,
            "object_bytes": loc.object_bytes,
            "pack_checksum": loc.pack_checksum,
            "pack_record_count": loc.pack_record_count,
            "payload_offset": loc.offset, "stored_length": loc.stored_length,
            "decoded_length": loc.decoded_length, "codec": loc.codec,
            "payload_checksum": loc.checksum,
        }
        sort_key = ("tenant_id", "experiment_id", "run_id",
                    "captured_at_ns", "capture_id")
        ordered = list(sort_key) + list(_RESOLVED)
        return [str(values[name]) if values[name] is not None else ""
                for name in ordered]
    fields = [str(f) for f in item]
    # Python flattens a NULL column to None → "". The native reader now
    # hands back "" for a SQL NULL directly (parse_tsv_tuple keeps nullness
    # apart from the string "NULL"), so "NULL" is deliberately NOT tolerated
    # here: seeing it would mean the two representations had drifted back
    # together, which is what let a catalog NULL bind to a footer 'NULL'.
    return ["" if f == "None" else f for f in fields]


def _normalize(items):
    """Both readers' rows onto one comparable 32-field string layout."""
    return sorted(tuple(_descriptor_fields(item)) for item in items)


def test_search_parity_no_filters():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)
            watermark = driver.call(op="current_watermark")["watermark"]
            assert watermark == "7"

            native = driver.call(op="search", limit=100)
            page = _python_page_items(_python_reader(client, config))
            assert native["watermark"] == page.watermark == "7"
            assert native["next_cursor"] is None and page.next_cursor is None
            assert len(native["items"]) == 4 == len(page.items)
            assert _normalize(native["items"]) == _normalize(page.items)
        finally:
            driver.close()


def test_search_parity_on_a_multi_dimensional_shape():
    """A rank-2 shape is the ordinary case, and it must survive the tuple.

    `shape` is Array(UInt32), and the resolved columns travel back as one
    TSV-rendered argMax tuple: `('a','b',[1,128,4096],...)`. A splitter
    that tracks quoted strings but not brackets sees the array's own
    commas as tuple separators, over-splits the row, and search fails
    outright -- for every real activation, since the only shapes that
    survive are rank 1. The synthetic corpus is rank 1, which is why
    nothing noticed.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(2)
            for entry in descriptors:
                entry["shape"] = [1, 128, 4096]
            _publish_native(driver, prefix, descriptors, 7)

            native = driver.call(op="search", limit=100)
            assert native["ok"], native
            assert len(native["items"]) == 2, native
            page = _python_page_items(_python_reader(client, config))
            assert _normalize(native["items"]) == _normalize(page.items)
        finally:
            driver.close()


def test_search_parity_on_text_holding_quotes_and_backslashes():
    """A quote or backslash must survive both halves of the row.

    The grouped columns and the aggregate tuple arrive with different
    escaping, so this is really two bugs in one shape: the grouped
    columns needed their TSV escapes undone (`alan's run` came back
    `alan\\'s run`), while the tuple's contents must NOT be unescaped a
    second time (`packs/a\\b.dmi-pack` turned into a backspace when they
    were). Both halves are asserted here because fixing either one alone
    is what broke the other.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(2)
            for entry in descriptors:
                entry["run_id"] = "alan's run"          # grouped column
                entry["object_key"] = "packs/a\\b.dmi-pack"  # in the tuple
            _publish_native(driver, prefix, descriptors, 7)

            native = driver.call(op="search", limit=100)
            assert native["ok"], native
            page = _python_page_items(_python_reader(client, config))
            assert {item.metadata.run_id for item in page.items} == {
                "alan's run"}
            assert {item.locator.object_key for item in page.items} == {
                "packs/a\\b.dmi-pack"}
            assert _normalize(native["items"]) == _normalize(page.items)
        finally:
            driver.close()


def test_a_cursor_over_quote_bearing_values_crosses_implementations():
    """The cursor's JSON must be JSON even when the data holds quotes.

    Both the filter hash and the cursor key render values into JSON by
    concatenation. Python's json.dumps escapes a quote or backslash; a
    renderer that does not produces a different filter digest (so the
    other side refuses the cursor as belonging to different filters) and,
    in the key, a cursor that is not valid JSON at all.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(3)
            for entry in descriptors:
                entry["run_id"] = 'alan "quoted\\" run'
            _publish_native(driver, prefix, descriptors, 7)

            # Native issues a cursor under a quote-bearing filter...
            first = driver.call(op="search", limit=1,
                                run_id='alan "quoted\\" run')
            assert first["ok"] and first["next_cursor"], first

            # ...and the PYTHON reader accepts it and serves page two.
            reader = _python_reader(client, config)
            page = _python_page_items(
                reader, limit=1, run_id='alan "quoted\\" run',
                cursor=first["next_cursor"])
            assert len(page.items) == 1, page
            assert page.items[0].metadata.capture_id == "capture-1"

            # And the reverse: a Python cursor resumes natively.
            third = driver.call(op="search", limit=1,
                                run_id='alan "quoted\\" run',
                                cursor=page.next_cursor)
            assert third["ok"], third
            assert third["items"][0][4] == "capture-2", third
        finally:
            driver.close()


def _open_helper(driver, prefix, **overrides):
    from tests.test_native_catalog_lease_live import DEFAULTS

    fields = {"op": "open", "table_prefix": prefix, **DEFAULTS, **overrides}
    if "database" not in fields:
        fields["database"] = environ.get("DMI_CLICKHOUSE_DATABASE", "default")
    response = driver.call(**fields)
    assert response["ok"], response
    return response


def test_search_parity_with_filters():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)  # alternating hook_name
            _publish_native(driver, prefix, descriptors, 7)
            from dmi.storage.capture.model import CaptureQuery

            reader = _python_reader(client, config)
            for query_fields, native_fields in (
                ({"hook_names": ("resid_pre",)},
                 {"hook_names": ["resid_pre"]}),
                ({"captured_after_ns": descriptors[1]["captured_at_ns"]},
                 {"captured_after_ns": descriptors[1]["captured_at_ns"]}),
                ({"tenant_id": "t"}, {"tenant_id": "t"}),
            ):
                native = driver.call(op="search", limit=100, **native_fields)
                page = _python_page_items(reader, **query_fields)
                assert len(native["items"]) == len(page.items), (
                    query_fields, native["items"], page.items)
                assert _normalize(native["items"]) == _normalize(page.items), (
                    query_fields)
        finally:
            driver.close()


def test_an_empty_filter_list_is_absent_rather_than_impossible():
    """`hook_names: []` means "no hook filter", not "hook_name IN ('')".

    An empty JSON array splits into ONE empty element, so the driver built
    a filter matching the empty string -- nothing -- and an empty
    layer_numbers filtered to layer 0 through atoll(""). Python treats an
    empty filter tuple as absent, so a caller clearing a filter got every
    row from one reader and none from the other.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)

            reader = _python_reader(client, config)
            unfiltered = _python_page_items(reader)
            assert len(unfiltered.items) == 4, unfiltered

            for native_fields in ({"hook_names": []},
                                  {"layer_numbers": []},
                                  {"hook_names": [], "layer_numbers": []}):
                native = driver.call(op="search", limit=100, **native_fields)
                assert native["ok"], native
                assert len(native["items"]) == 4, (native_fields, native)
                assert _normalize(native["items"]) == _normalize(
                    unfiltered.items), native_fields
        finally:
            driver.close()


def test_the_watermark_read_carries_the_configured_bounds():
    """"Bounds on every catalog read" has to include the watermark read.

    Every other read sends max_rows_to_read / max_bytes_to_read /
    max_execution_time; the head read sent either the deciding setting
    alone or nothing at all, so the one read taken on every search ran
    unbounded. Python builds it the other way round -- it starts from the
    config's settings and merely ADDS the deciding one.

    Read off the SERVER rather than the client: a low row bound cannot be
    made to bite here, because `max(index_version)` over a table ordered by
    index_version is answered from the primary index without reading a row.
    What the server records for the statement is the honest evidence, and
    it is the same check the replicated-quorum verifier makes.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(2), 7)

            marker = 4242424
            fine = driver.call(op="current_watermark",
                               max_rows_to_read=marker)
            assert fine["ok"] and fine["watermark"] == "7", fine

            client.execute("SYSTEM FLUSH LOGS")
            recorded = client.execute(
                "SELECT Settings['max_rows_to_read'] FROM system.query_log "
                f"WHERE query LIKE '%{prefix}_index_watermark%' "
                "AND type = 'QueryFinish' "
                "ORDER BY event_time_microseconds DESC LIMIT 1")
            assert recorded and recorded[0][0] == str(marker), recorded
        finally:
            driver.close()


def test_a_cursor_envelope_is_validated_the_way_python_validates_it():
    """The envelope checks are the cursor's whole contract.

    decode_cursor requires version 1, exactly the fields {v,w,fh,k}, and a
    bounded size. A reader that reads `k` and `w` out of whatever JSON
    arrives will happily page a v2 cursor with v1 semantics, or one
    carrying fields it does not understand -- which is how a forward-
    compatible format silently becomes an incompatible one.
    """
    import base64

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)

            # A real, accepted cursor first, so the rejections below are
            # about the envelope rather than about the page.
            first = driver.call(op="search", limit=1)
            assert first["ok"] and first["next_cursor"], first
            good = first["next_cursor"]
            payload = json.loads(base64.urlsafe_b64decode(
                good + "=" * (-len(good) % 4)))
            assert set(payload) == {"v", "w", "fh", "k"}, payload

            def encode(obj):
                raw = json.dumps(obj, sort_keys=True,
                                 separators=(",", ":")).encode()
                return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

            from dmi.storage.capture.cursor import InvalidCursorError
            from dmi.storage.capture.model import CaptureQuery
            reader = _python_reader(client, config)

            future = dict(payload, v=2)
            extra = dict(payload, surprise=1)
            for broken in (future, extra):
                cursor = encode(broken)
                refused = driver.call(op="search", limit=1, cursor=cursor)
                assert not refused["ok"], (broken, refused)
                # The oracle refuses the same cursor.
                with pytest.raises(InvalidCursorError):
                    reader.search(CaptureQuery(limit=1, cursor=cursor))
        finally:
            driver.close()


def test_a_cursor_key_with_an_empty_component_is_refused():
    """An empty key string is a malformed cursor, not a position.

    Python's decoder requires every key string to be non-empty. The native
    fallback returned the RAW JSON token whenever the decoded value was
    empty, so a crafted cursor with `""` in its key paged after the
    two-character string `\"\"` -- an accepted cursor naming a position no
    encoder can ever have issued.
    """
    import base64

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(3), 7)

            first = driver.call(op="search", limit=1)
            assert first["ok"] and first["next_cursor"], first
            good = first["next_cursor"]
            payload = json.loads(base64.urlsafe_b64decode(
                good + "=" * (-len(good) % 4)))
            payload["k"][0] = ""
            raw = json.dumps(payload, sort_keys=True,
                             separators=(",", ":")).encode()
            crafted = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

            refused = driver.call(op="search", limit=1, cursor=crafted)
            assert not refused["ok"], refused
            assert refused["error"] == "ValueError", refused

            # The oracle refuses the same cursor.
            from dmi.storage.capture.cursor import InvalidCursorError
            from dmi.storage.capture.model import CaptureQuery
            with pytest.raises(InvalidCursorError):
                _python_reader(client, config).search(
                    CaptureQuery(limit=1, cursor=crafted))
        finally:
            driver.close()


def test_search_refusal_parity_on_filter_bounds():
    """CaptureQuery bounds SIX things; search() ported only the limit.

    The oracle refuses a query whose filter lists exceed their bounded
    cardinality, whose layer numbers fall below -1, or whose time window
    runs backwards. Native rendered all three straight into SQL, so it
    ACCEPTED queries the oracle refuses outright -- 1025 layer numbers came
    back as a successful four-item page.
    """
    from dmi.storage.capture.model import CaptureQuery

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(4), 7)

            cases = (
                # Bounded cardinality: 128 hook names, 1024 layer numbers.
                {"hook_names": [f"hook-{i}" for i in range(10_000)]},
                {"hook_names": [f"hook-{i}" for i in range(129)]},
                {"layer_numbers": list(range(1025))},
                # A window that runs backwards selects nothing, and asking
                # for it is a caller error rather than an empty answer.
                {"captured_after_ns": 2000, "captured_before_ns": 1000},
                # -1 is the "no layer" sentinel; below it means nothing.
                {"layer_numbers": [-5]},
            )
            for fields in cases:
                refused = driver.call(op="search", limit=100, **fields)
                assert not refused["ok"], (fields.keys(), refused)
                assert refused["error"] == "ValueError", (
                    fields.keys(), refused)

                # The oracle refuses the same filters.
                python_fields = {
                    name: tuple(value) if isinstance(value, list) else value
                    for name, value in fields.items()
                }
                with pytest.raises(ValueError):
                    CaptureQuery(limit=100, **python_fields)

            # The bounds bite only at the edge: the largest ACCEPTED lists
            # and a forward window still serve the same page both sides.
            accepted = {"hook_names": ["resid_pre"] + [
                            f"hook-{i}" for i in range(127)],
                        "layer_numbers": list(range(1024))}
            native = driver.call(op="search", limit=100, **accepted)
            assert native["ok"], native
            page = _python_page_items(
                _python_reader(client, config),
                hook_names=tuple(accepted["hook_names"]),
                layer_numbers=tuple(accepted["layer_numbers"]), limit=100)
            assert _normalize(native["items"]) == _normalize(page.items)
        finally:
            driver.close()


def test_search_refusal_parity_on_filter_text():
    """CaptureQuery validates its text filters, and search() did not.

    __post_init__ runs _validate_text over tenant_id/experiment_id/run_id/
    session_id/model_id and over every hook_name: non-empty and at most 512
    bytes. Native ported the cardinality and window bounds but not these, so
    an empty tenant_id rendered ``AND `tenant_id` = ''`` and came back as a
    successful empty page -- a caller error answered as data -- and a 10 KB
    run_id was shipped to the server rather than refused.

    The staged rows carry the 512-byte run_id and, for half of them, the
    512-byte hook_name, so the accepted-edge control at the bottom selects
    real rows. It used to filter on 512-byte values nothing was staged with,
    which made its parity assertion ``[] == []``.
    """
    from dmi.storage.capture.model import CaptureQuery

    long_run = "x" * 512
    long_hook = "y" * 512

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            # hook= lands on the even rows; the odd ones stay "other_hook".
            descriptors = _descriptor_dicts(4, hook=long_hook)
            for entry in descriptors:
                entry["run_id"] = long_run
            _publish_native(driver, prefix, descriptors, 7)

            oversized = "x" * 513
            cases = (
                {"tenant_id": ""},
                {"experiment_id": ""},
                {"run_id": ""},
                {"session_id": ""},
                {"model_id": ""},
                {"hook_names": [""]},
                {"hook_names": ["resid_pre", ""]},
                {"run_id": oversized},
                {"tenant_id": "x" * 10_000},
                {"hook_names": [oversized]},
            )
            for fields in cases:
                refused = driver.call(op="search", limit=100, **fields)
                assert not refused["ok"], (fields.keys(), refused)
                assert refused["error"] == "ValueError", (
                    fields.keys(), refused)

                # The oracle refuses the same filters.
                python_fields = {
                    name: tuple(value) if isinstance(value, list) else value
                    for name, value in fields.items()
                }
                with pytest.raises(ValueError):
                    CaptureQuery(limit=100, **python_fields)

            # The bound bites only past the edge: 512 bytes is ACCEPTED, and
            # both readers serve the same page for it. The page is non-empty
            # and is a strict subset of the four staged rows, so the parity
            # assertion compares selected rows rather than two empty lists.
            edge = {"run_id": long_run, "hook_names": [long_hook]}
            native = driver.call(op="search", limit=100, **edge)
            assert native["ok"], native
            page = _python_page_items(
                _python_reader(client, config), limit=100,
                run_id=edge["run_id"], hook_names=tuple(edge["hook_names"]))
            assert len(page.items) == 2, page.items
            assert len(native["items"]) == 2, native
            assert _normalize(native["items"]) == _normalize(page.items)
        finally:
            driver.close()


def test_a_negative_captured_bound_is_refused_rather_than_wrapped():
    """A negative time bound is a caller error, not the far future.

    The harness decoded the two captured bounds with
    static_cast<uint64_t>(jc::FindInt(...)), so -1 arrived as
    18446744073709551615 and `captured_after_ns: -1` -- which the oracle
    refuses as "must be a non-negative integer" -- came back as a
    successful EMPTY page. A non-numeric spelling was the same story by a
    different route: HasKey passes, FindInt returns its -1 sentinel, and
    the bound is UINT64_MAX again.
    """
    from dmi.storage.capture.model import CaptureQuery

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(4), 7)

            for name in ("captured_after_ns", "captured_before_ns"):
                # -1 wraps; the rest are non-integers CaptureQuery rejects
                # on `type(value) is not int` (True included: its type is
                # bool, not int).
                for value in (-1, -1700000000000000000, "abc", 1.5, True):
                    fields = {name: value}
                    refused = driver.call(op="search", limit=100, **fields)
                    assert not refused["ok"], (fields, refused)
                    assert refused["error"] == "ValueError", (fields, refused)

                    # The oracle refuses the same bound.
                    with pytest.raises(ValueError):
                        CaptureQuery(limit=100, **fields)

            # 0 is a legal bound and still selects the whole table, so the
            # refusal above is about the sign rather than about the field.
            native = driver.call(op="search", limit=100, captured_after_ns=0)
            assert native["ok"], native
            page = _python_page_items(
                _python_reader(client, config), limit=100, captured_after_ns=0)
            assert _normalize(native["items"]) == _normalize(page.items)
            assert len(native["items"]) == 4, native
        finally:
            driver.close()


def test_a_cursor_captured_at_ns_outside_uint64_is_refused():
    """The key's timestamp is a UInt64, and nothing checked it.

    `parts[3]` travelled out of the cursor as a raw JSON token and into the
    statement as a quoted STRING literal, so the server coerced it with
    String -> UInt64 -- which wraps modulo 2**64. A crafted key holding
    `2**64 + honest` therefore named a DIFFERENT position than the one it
    spells, and search answered `ok` with a page that silently skips rows.
    Python's decoder requires `type(key[3]) is int` and `0 <= v <= 2**64-1`.
    """
    import base64

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)

            first = driver.call(op="search", limit=1)
            assert first["ok"] and first["next_cursor"], first
            good = first["next_cursor"]
            payload = json.loads(base64.urlsafe_b64decode(
                good + "=" * (-len(good) % 4)))
            honest = payload["k"][3]

            def encode(key_value):
                crafted = dict(payload, k=list(payload["k"]))
                crafted["k"][3] = key_value
                raw = json.dumps(crafted, sort_keys=True,
                                 separators=(",", ":")).encode()
                return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

            from dmi.storage.capture.cursor import InvalidCursorError
            from dmi.storage.capture.model import CaptureQuery
            reader = _python_reader(client, config)

            # The decisive case: wrapping lands 1500ns past the honest
            # position, i.e. after capture-1 -- so the accepted page used to
            # come back missing a row, with ok=True and no error.
            oversized = 2**64 + honest + 1500
            for key_value in (oversized, "abc", -5, 1.5, True, None):
                cursor = encode(key_value)
                refused = driver.call(op="search", limit=100, cursor=cursor)
                assert not refused["ok"], (key_value, refused)
                assert refused["error"] == "ValueError", (key_value, refused)
                # The oracle refuses the same cursor.
                with pytest.raises(InvalidCursorError):
                    reader.search(CaptureQuery(limit=100, cursor=cursor))

            # The honest cursor still pages, so the check above is about the
            # crafted timestamp rather than about the cursor as a whole.
            walked = driver.call(op="search", limit=100, cursor=good)
            assert walked["ok"], walked
            assert [item[4] for item in walked["items"]] == [
                "capture-1", "capture-2", "capture-3"], walked
        finally:
            driver.close()


def test_a_cursor_envelope_integer_outside_uint64_is_refused():
    """`v` and `w` are UInt64s too, and nothing checked either.

    The captured_at_ns fix bounded only the key's timestamp. The envelope's
    version and watermark still went through jc::FindInt, whose accumulator
    multiplies into an int64_t with no bound check, so `2**64 + 1` on either
    field is a wrapping parse rather than a refusal. What the wrapped value
    IS is signed-overflow UB and therefore build-dependent; what matters is
    that the oracle refuses both spellings outright -- `w` on
    _uint64(..., "cursor watermark") and `v` on the version equality -- so
    native must refuse them too rather than serve any page at all.
    """
    import base64

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(4), 7)

            first = driver.call(op="search", limit=1)
            assert first["ok"] and first["next_cursor"], first
            good = first["next_cursor"]
            payload = json.loads(base64.urlsafe_b64decode(
                good + "=" * (-len(good) % 4)))

            def encode(field, value):
                crafted = dict(payload)
                crafted[field] = value
                raw = json.dumps(crafted, sort_keys=True,
                                 separators=(",", ":")).encode()
                return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

            from dmi.storage.capture.cursor import InvalidCursorError
            from dmi.storage.capture.model import CaptureQuery
            reader = _python_reader(client, config)

            # 2**64 + 1 on each envelope integer. Neither fits UInt64, so
            # neither names a snapshot or a version this reader can serve.
            for field in ("w", "v"):
                cursor = encode(field, 2**64 + 1)
                refused = driver.call(op="search", limit=100, cursor=cursor)
                assert not refused["ok"], (field, refused)
                assert refused["error"] == "ValueError", (field, refused)
                # The oracle refuses the same cursor bytes.
                with pytest.raises(InvalidCursorError):
                    reader.search(CaptureQuery(limit=100, cursor=cursor))

            # The honest cursor still pages, so the refusals above are about
            # the crafted envelope rather than about the cursor as a whole.
            walked = driver.call(op="search", limit=100, cursor=good)
            assert walked["ok"], walked
            assert [item[4] for item in walked["items"]] == [
                "capture-1", "capture-2", "capture-3"], walked
        finally:
            driver.close()


def test_a_cursor_integer_with_leading_zeros_is_refused():
    """`007` is not a JSON number, so it is not a cursor position.

    The native digit scan asked only whether every character was a digit,
    which `007` satisfies -- so a crafted key paged from position 7 (i.e.
    from the very beginning, capture-0 included) while spelling the honest
    1.7e18 position's neighbourhood. Python never gets that far: json.loads
    rejects a leading-zero number outright and decode_cursor reports
    "cursor does not contain valid JSON".
    """
    import base64

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            _publish_native(driver, prefix, _descriptor_dicts(4), 7)

            first = driver.call(op="search", limit=1)
            assert first["ok"] and first["next_cursor"], first
            good = first["next_cursor"]
            payload = json.loads(base64.urlsafe_b64decode(
                good + "=" * (-len(good) % 4)))
            canonical = json.dumps(payload, sort_keys=True,
                                   separators=(",", ":"))

            def encode(raw_json):
                return base64.urlsafe_b64encode(
                    raw_json.encode()).rstrip(b"=").decode()

            # Textual crafting: json.dumps cannot emit a leading-zero
            # number, which is the whole point of the spelling.
            honest_ns = str(payload["k"][3])
            crafted = {
                # The decisive one: position 7 rather than 1.7e18, so the
                # accepted page used to START at capture-0 -- a cursor that
                # pages BACKWARDS over rows the caller already saw.
                "k[3]": canonical.replace(f",{honest_ns},", ",007,", 1),
                "w": canonical.replace('"w":7}', '"w":007}', 1),
                "v": canonical.replace('"v":1,', '"v":01,', 1),
            }
            from dmi.storage.capture.cursor import InvalidCursorError
            from dmi.storage.capture.model import CaptureQuery
            reader = _python_reader(client, config)

            for field, raw_json in crafted.items():
                assert raw_json != canonical, field  # the craft landed
                cursor = encode(raw_json)
                refused = driver.call(op="search", limit=100, cursor=cursor)
                assert not refused["ok"], (field, refused)
                assert refused["error"] == "ValueError", (field, refused)
                # The oracle refuses the same cursor bytes.
                with pytest.raises(InvalidCursorError):
                    reader.search(CaptureQuery(limit=100, cursor=cursor))

            # A single `0` is a legal JSON number and must stay legal: it is
            # the position before every row, so the whole table pages.
            zero = encode(canonical.replace(f",{honest_ns},", ",0,", 1))
            walked = driver.call(op="search", limit=100, cursor=zero)
            assert walked["ok"], walked
            assert [item[4] for item in walked["items"]] == [
                "capture-0", "capture-1", "capture-2", "capture-3"], walked
        finally:
            driver.close()


def test_cursor_parity_across_implementations():
    """A cursor either side issues, the other side accepts and walks."""
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)
            from dmi.storage.capture.model import CaptureQuery

            reader = _python_reader(client, config)
            seen_native, seen_python = [], []
            cursor, source = None, "native"
            for _ in range(8):
                if source == "native":
                    fields = {"op": "search", "limit": 1}
                    if cursor:
                        fields["cursor"] = cursor
                    page = driver.call(**fields)
                    # The native item's capture_id sits at sort-key
                    # position 4 (tenant, experiment, run, captured, id).
                    seen_native.extend(
                        item[4] for item in page["items"])
                    cursor = page["next_cursor"]
                else:
                    page = _python_page_items(
                        reader, cursor=cursor, limit=1)
                    seen_python.extend(
                        item.metadata.capture_id for item in page.items)
                    cursor = page.next_cursor
                if not cursor:
                    break
                source = "python" if source == "native" else "native"
            # Alternating pages: each side serves every OTHER item, each
            # side accepted the other's cursor, and the union walked the
            # whole corpus exactly once, in order.
            assert sorted(seen_native + seen_python) == [
                "capture-0", "capture-1", "capture-2", "capture-3"
            ], (seen_native, seen_python)
            assert seen_native == ["capture-0", "capture-2"]
            assert seen_python == ["capture-1", "capture-3"]
        finally:
            driver.close()


def test_get_by_ids_parity_and_watermark_validation():
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(4)
            _publish_native(driver, prefix, descriptors, 7)
            ids = [d["capture_id"] for d in descriptors]
            native = driver.call(
                op="get_by_ids", capture_ids=ids[:2], tenant_id="t",
                watermark="7")
            assert len(native["items"]) == 2, native
            from dmi.storage.capture.model import CaptureQuery  # noqa

            reader = _python_reader(client, config)
            python_rows = reader.get_by_ids(ids[:2], tenant_id="t",
                                            watermark="7")
            assert _normalize(native["items"]) == _normalize(python_rows)

            # Unknown ids resolve to nothing on both sides.
            assert driver.call(
                op="get_by_ids", capture_ids=["nope"], tenant_id="t",
                watermark="7")["items"] == []
            assert reader.get_by_ids(["nope"], tenant_id="t",
                                     watermark="7") == ()

            # A watermark above the published head is refused by both.
            refused = driver.call(
                op="get_by_ids", capture_ids=ids[:1], tenant_id="t",
                watermark="9")
            assert not refused["ok"]
            assert "exceeds the published watermark" in refused["message"]
            with pytest.raises(ValueError, match="exceeds the published"):
                reader.get_by_ids(ids[:1], tenant_id="t", watermark="9")

            # An EMPTY watermark is refused, not read as snapshot 0. The
            # digit check passes vacuously on an empty string (all_of over
            # an empty range is true) and the parse then answers 0, so the
            # call silently resolved against a snapshot that admits
            # nothing and returned no rows -- a caller passing a watermark
            # it failed to populate gets "no such captures" instead of an
            # error. Python raises on both empty and non-numeric.
            for bad in ("", "  ", "seven", "-1"):
                refused = driver.call(
                    op="get_by_ids", capture_ids=ids[:1], tenant_id="t",
                    watermark=bad)
                assert not refused["ok"], (bad, refused)
                assert refused["error"] == "ValueError", (bad, refused)
                with pytest.raises(ValueError):
                    reader.get_by_ids(ids[:1], tenant_id="t", watermark=bad)

            # A watermark WIDER than UInt64 is caller data too, and the digit
            # check above passes on it: `parse_u64_field` accumulated it into
            # a uint64_t with no bound, so 2**64 wrapped to 0 and 2**64 + 7
            # to 7 -- the published head. Both then slipped past the
            # published-head guard and resolved against a snapshot the caller
            # never named: 2**64 returned NO rows (snapshot 0) and 2**64 + 7
            # returned the FULL result of snapshot 7. `_parse_watermark`
            # raises "watermark must fit UInt64" on both, so the wrap value is
            # chosen here to land at or below the head on purpose -- a wrap
            # ABOVE it is masked by the head guard's own refusal.
            for wide in (2**64, 2**64 + 7, int("9" * 40)):
                refused = driver.call(
                    op="get_by_ids", capture_ids=ids[:1], tenant_id="t",
                    watermark=str(wide))
                assert not refused["ok"], (wide, refused)
                assert refused["error"] == "ValueError", (wide, refused)
                assert "must fit UInt64" in refused["message"], (wide, refused)
                with pytest.raises(ValueError, match="must fit UInt64"):
                    reader.get_by_ids(ids[:1], tenant_id="t",
                                      watermark=str(wide))

            # 2**64 - 1 is the widest watermark UInt64 holds, so it must reach
            # the published-head guard rather than be refused as unwidth --
            # and leading zeros are insignificant to `int()`, so a 24-digit
            # rendering of the same value must be judged the same way and not
            # refused on its LENGTH.
            for widest in (str(2**64 - 1), "0000" + str(2**64 - 1)):
                refused = driver.call(
                    op="get_by_ids", capture_ids=ids[:1], tenant_id="t",
                    watermark=widest)
                assert not refused["ok"], (widest, refused)
                assert "exceeds the published watermark" in refused[
                    "message"], (widest, refused)
                with pytest.raises(ValueError, match="exceeds the published"):
                    reader.get_by_ids(ids[:1], tenant_id="t",
                                      watermark=widest)

            # A leading-zero watermark AT the head still resolves, on both
            # sides: the width check must not refuse what the oracle admits.
            padded = driver.call(
                op="get_by_ids", capture_ids=ids[:2], tenant_id="t",
                watermark="0007")
            assert padded["ok"], padded
            assert len(padded["items"]) == 2, padded
            assert _normalize(padded["items"]) == _normalize(
                reader.get_by_ids(ids[:2], tenant_id="t", watermark="0007"))
        finally:
            driver.close()


def test_supersession_resolves_the_newest_pack():
    """The same capture re-described by a later pack: newest wins, both sides.

    The resolution order is (index_version, store_id, pack_id) — a later
    version supersedes; within one version the highest (store_id, pack_id)
    wins, a fixed choice so a selection resolved twice resolves to the
    same bytes.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            old_pack = str(uuid.uuid4())
            new_pack = str(uuid.uuid4())
            first = _descriptor_dicts(2, pack_id=old_pack)
            # The same captures, a different pack, a later version: the
            # locator differs, the metadata does not.
            second = _descriptor_dicts(2, pack_id=new_pack,
                                       capture_prefix="capture")
            _publish_native(driver, prefix, first, 7)
            _publish_native(driver, prefix, second, 8)

            from dmi.storage.capture.model import CaptureQuery

            reader = _python_reader(client, config)
            native = driver.call(op="search", limit=100)
            page = _python_page_items(reader)
            assert _normalize(native["items"]) == _normalize(page.items)
            assert len(native["items"]) == 2, native["items"]
            # Both resolve each capture to the NEW pack's locator.
            for item in native["items"]:
                assert item[21] == new_pack, item  # pack_id column
            for item in page.items:
                assert item.locator.pack_id == new_pack
        finally:
            driver.close()


# --- C2: hydration and core summary at parity --------------------------------

def _e2e_setup(fake_s3, prefix, record_count=3, **stage_kwargs):
    """sink → uploader → native index; returns (drivers, refs, descriptor ids).

    `stage_kwargs` are forwarded verbatim to `_stage` (payload, dtype,
    shape), so a test that needs a real element type or a rank>=2 shape
    can stage one without duplicating the pipeline.
    """
    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )

    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    driver = CatalogDriver()
    try:
        import tempfile
        spool_root = Path(tempfile.mkdtemp()) / "spool"
        staged = [_stage(sink, spool_root, 60 + i, **stage_kwargs)
                  for i in range(record_count)]
        uploaded = store.call(
            op="upload_pending", **_store_base(fake_s3),
            root=str(spool_root), spool_max_bytes=1 << 40, limit=-1)
        assert uploaded["ok"], uploaded
        refs = uploaded["refs"]
        _open_helper(driver, prefix)
        driver.call(op="acquire", holder="indexer")
        result = driver.call(
            op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
            region=REGION, access=ACCESS, secret=SECRET, insecure=True)
        assert result["ok"], result
        assert result["result"]["indexed_packs"] == record_count, result
        return sink, store, driver, refs
    except Exception:
        sink.close()
        store.close()
        driver.close()
        raise


def test_hydrate_parity_identical_payload_bytes(fake_s3):
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )

    with _catalog() as (client, config, prefix):
        sink, store, driver, refs = None, None, None, None
        try:
            sink, store, driver, refs = _e2e_setup(fake_s3, prefix)
            query_fields = {"tenant_id": "t", "limit": 10}

            from dmi.storage.capture.model import CaptureQuery
            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(CaptureQuery(**query_fields))

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            native_selection = native_select["selection"]
            # The selection identity is byte-compatible: same id.
            assert native_selection["selection_id"] == selection.selection_id

            native = driver.call(
                op="hydrate", selection_id=native_selection["selection_id"],
                capture_ids=native_selection["capture_ids"],
                catalog_watermark=native_selection["catalog_watermark"],
                filter_hash=native_selection["filter_hash"],
                tenant_id=native_selection["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native
            import base64
            native_payloads = [
                base64.b64decode(payload) for payload in native["payloads"]]

            hydrated = python_reader.hydrate(
                selection, byte_limit=1 << 30)
            assert len(native_payloads) == len(hydrated)
            by_id = {item.capture_id: item.payload for item in hydrated}
            for index, capture_id in enumerate(
                    native_selection["capture_ids"]):
                assert native_payloads[index] == by_id[capture_id], capture_id
        finally:
            for closer in (sink, store, driver):
                if closer is not None:
                    closer.close()


def test_hydrate_parity_on_a_multi_dimensional_shape(fake_s3):
    """A rank>=2 shape must survive hydration's footer binding.

    The sibling of test_search_parity_on_a_multi_dimensional_shape, one
    layer down. The pack footer row renders `shape` verbatim as a compact
    JSON array, so a rank-2 tensor arrives as `[4,3]` inside the row's
    own comma-separated field list. A footer splitter that tracks quoted
    strings but not bracket depth splits `[4,3]` into `[4` and `3]`,
    shifts every field from `shape` onward, and refuses the descriptor
    with "does not match the pack footer: field 20" -- which is EVERY
    real activation tensor, since only rank 1 survives.
    """
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )

    # float32, 4x3: twelve elements of 1.0f, so the payload length agrees
    # with dtype x prod(shape) and the pack index accepts the record.
    payload = b"\x00\x00\x80?" * 12

    with _catalog() as (client, config, prefix):
        sink, store, driver, refs = None, None, None, None
        try:
            sink, store, driver, refs = _e2e_setup(
                fake_s3, prefix, payload=payload, dtype="float32",
                shape=(4, 3))

            from dmi.storage.capture.model import CaptureQuery
            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(
                CaptureQuery(tenant_id="t", limit=10))

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            assert sel["selection_id"] == selection.selection_id

            # THE ORACLE: Python hydrates the rank-2 capture.
            hydrated = python_reader.hydrate(selection, byte_limit=1 << 30)
            by_id = {item.capture_id: item.payload for item in hydrated}
            assert set(by_id.values()) == {payload}

            native = driver.call(
                op="hydrate", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"],
                tenant_id=sel["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native
            import base64
            native_payloads = [
                base64.b64decode(item) for item in native["payloads"]]
            assert len(native_payloads) == len(hydrated)
            for index, capture_id in enumerate(sel["capture_ids"]):
                assert native_payloads[index] == by_id[capture_id], capture_id
        finally:
            for closer in (sink, store, driver):
                if closer is not None:
                    closer.close()


def _footer_binding_case(fake_s3, client, config, driver, mutate):
    """Stage one pack, publish a catalog row with `mutate` applied, hydrate.

    The pack footer is written once and is the authority for what each
    payload IS; the catalog row is the thing under test. `mutate` receives
    the row dict built from the footer's own descriptor and edits one
    field, so the row and the footer disagree by exactly that field.

    Returns `(oracle_error, native_response)`: the PackFormatError message
    the Python reader raises (or None if it accepted the row) and the raw
    native `hydrate` response.
    """
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import (
        CaptureMetadata, CaptureQuery, CaptureRecord,
    )
    from dmi.storage.capture.pack import PackReader as PR, PackWriter
    from dmi.storage.capture.reader import PackFormatError

    payload = b"\x00\x00\x80?" * 12  # float32, twelve 1.0f elements
    meta = CaptureMetadata(
        capture_id="bind-0", tenant_id="t", experiment_id="e", run_id="r",
        session_id="s", request_id="q", sequence_id="n0", model_id="m",
        model_revision="mr", adapter_revision=None,
        capture_policy_version="v", hook_name="h", layer_number=3,
        producer_rank=0, step_number=0, token_start=0, token_end=1,
        batch_position=0, dtype="float32", shape=(12,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    pack_id = str(uuid.uuid4())
    object_key = f"packs/{pack_id}.dmi-pack"
    writer = PackWriter(
        pack_id=pack_id, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024)
    writer.append(CaptureRecord(metadata=meta, payload=payload))
    sealed = writer.seal()

    import boto3 as _boto3
    from botocore.config import Config as _BotoConfig
    _boto3.client(
        "s3", endpoint_url=fake_s3, region_name=REGION,
        aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
        config=_BotoConfig(s3={"addressing_style": "path"}),
        verify=False,
    ).put_object(Bucket=BUCKET, Key=object_key, Body=sealed.data)

    _open_helper(driver, prefix=config.table_prefix)
    rows = []
    for d in PR.from_bytes(sealed.data).descriptors(
            store_id="native-test", object_key=object_key):
        m, l = d.metadata, d.locator
        rows.append({
            "capture_id": m.capture_id, "tenant_id": m.tenant_id,
            "experiment_id": m.experiment_id, "run_id": m.run_id,
            "session_id": m.session_id, "request_id": m.request_id,
            "sequence_id": m.sequence_id, "model_id": m.model_id,
            "model_revision": m.model_revision,
            "adapter_revision": m.adapter_revision,
            "capture_policy_version": m.capture_policy_version,
            "hook_name": m.hook_name, "layer_number": m.layer_number,
            "producer_rank": m.producer_rank, "step_number": m.step_number,
            "token_start": m.token_start, "token_end": m.token_end,
            "batch_position": m.batch_position, "dtype": m.dtype,
            "shape": list(m.shape), "captured_at_ns": m.captured_at_ns,
            "pack_id": pack_id, "store_id": "native-test",
            "object_key": object_key, "object_bytes": len(sealed.data),
            "pack_checksum": sealed.checksum, "pack_record_count": 1,
            "payload_offset": l.offset, "stored_length": l.stored_length,
            "decoded_length": l.decoded_length, "codec": l.codec,
            "payload_checksum": l.checksum,
        })
    assert len(rows) == 1, rows
    mutate(rows[0])
    driver.call(op="acquire", holder="writer")
    assert driver.call(
        op="write_descriptors", descriptors=rows, index_version=7)["ok"]
    assert driver.call(
        op="publish_snapshot", index_version=7,
        refs=[{"store_id": "native-test", "pack_id": pack_id}],
        published_at_ns=7, indexed_rows=1, indexed_packs=1)["ok"]

    python_reader = CaptureReader(
        ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)),
        {"native-test": S3PackStore.from_config(
            S3StoreConfig(
                endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                access_key_id=ACCESS, secret_access_key=SECRET,
                store_id="native-test", allow_insecure_http=True))})
    selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
    oracle_error = None
    try:
        python_reader.hydrate(selection, byte_limit=1 << 30)
    except PackFormatError as error:
        oracle_error = str(error)

    native_select = driver.call(
        op="select", tenant_id="t", limit=10, endpoint=fake_s3,
        bucket=BUCKET, access=ACCESS, secret=SECRET, insecure=True)
    assert native_select["ok"], native_select
    sel = native_select["selection"]
    native = driver.call(
        op="hydrate", selection_id=sel["selection_id"],
        capture_ids=sel["capture_ids"],
        catalog_watermark=sel["catalog_watermark"],
        filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
        byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
        secret=SECRET, insecure=True)
    return oracle_error, native


@pytest.mark.parametrize("field, value", [
    ("layer_number", 99),
    ("hook_name", "evil"),
    ("step_number", 41),
    ("request_id", "spoofed"),
    ("captured_at_ns", 1_700_000_000_000_000_999),
])
def test_hydrate_refuses_a_metadata_field_the_footer_contradicts(
        fake_s3, field, value):
    """The whole metadata record is the footer's authority, not nine fields.

    Python compares the entire CaptureMetadata dataclass plus the
    five-field record locator (reader.py `_require_footer_match` /
    `_record_locator`). A native binding that compares only the placement
    fields and dtype/shape lets a rewritten `layer_number` or `hook_name`
    through: the payload is handed back attributed to a capture the pack
    itself says it is not. Each field here is one the narrow list missed.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            oracle_error, native = _footer_binding_case(
                fake_s3, client, config, driver,
                lambda row: row.__setitem__(field, value))
            assert oracle_error is not None, (
                f"the oracle must refuse a contradicted {field}")
            assert "does not match the pack footer" in oracle_error
            assert not native["ok"], native
            assert "does not match the pack footer" in native["message"], (
                native)
        finally:
            driver.close()


@pytest.mark.parametrize("field, value", [
    ("dtype", "float16"),
    ("payload_offset", None),   # resolved below: the real offset + 4
    ("payload_checksum", "deadbeef"),
])
def test_hydrate_refuses_a_locator_field_the_footer_contradicts(
        fake_s3, field, value):
    """REGRESSION GUARD, not a bug proof: these three already refused.

    dtype, payload_offset and payload_checksum were inside the narrow
    nine-field comparison, so this test passed on first write. It is the
    native mirror of test_capture_storage.py's footer-mismatch cases and
    exists so a future narrowing of the comparison cannot pass unnoticed;
    it deliberately does NOT stand in for the metadata coverage above,
    since a suite holding only these is exactly what let the wider gap
    through.
    """
    def mutate(row):
        row[field] = row["payload_offset"] + 4 if value is None else value

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            oracle_error, native = _footer_binding_case(
                fake_s3, client, config, driver, mutate)
            assert oracle_error is not None, (
                f"the oracle must refuse a contradicted {field}")
            assert "does not match the pack footer" in oracle_error
            assert not native["ok"], native
            assert "does not match the pack footer" in native["message"], (
                native)
        finally:
            driver.close()


def test_summary_core_stats_parity(fake_s3):
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )

    with _catalog() as (client, config, prefix):
        try:
            sink, store, driver, refs = _e2e_setup(fake_s3, prefix)
            from dmi.storage.capture.model import CaptureQuery
            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(
                CaptureQuery(tenant_id="t", limit=10))

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            native = driver.call(
                op="summarize_core", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native
            python_summaries = python_reader.summarize(
                selection, byte_limit=1 << 30)

            by_id = {s.capture_id: s for s in python_summaries}
            for summary in native["summaries"]:
                expected = by_id[summary["capture_id"]]
                assert summary["summary_version"] == expected.core.summary_version
                assert summary["element_count"] == expected.core.element_count
                assert summary["finite_count"] == expected.core.finite_count
                assert summary["zero_fraction"] == expected.core.zero_fraction
                assert summary["mean"] == expected.core.mean, summary["capture_id"]
                assert summary["l2_norm"] == expected.core.l2_norm, (
                    summary["capture_id"])
                assert summary["minimum_int"] == expected.core.minimum, (
                    summary["capture_id"])
                assert summary["maximum_int"] == expected.core.maximum, (
                    summary["capture_id"])
                # abs_max was silently unasserted here, which is how a
                # NEGATIVE abs_max_int survived: Python takes it in
                # unbounded int space, so it is never negative.
                assert summary["abs_max_int"] == expected.core.abs_max, (
                    summary["capture_id"])
        finally:
            sink.close()
            store.close()
            driver.close()


def test_summary_parity_on_float16_including_subnormals(fake_s3):
    """Every float16 bit pattern must decode to the number Python decodes.

    The subnormal branch built the float32 bits by copying the low four
    bytes of a DOUBLE into the bit field and discarded the sign, so
    0x0001 (2^-24) and 0x8001 (-2^-24) summarized as garbage while the
    corpus every other test stages -- uint8 -- never touched the branch.
    Normals, zeros, and both subnormal signs ride together here and the
    Python summary is the oracle for all of them.
    """
    import struct
    import tempfile

    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )
    from dmi.storage.capture import CaptureReader, S3PackStore, S3StoreConfig
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureQuery

    # 0x0001 = 2^-24 (smallest subnormal), 0x8001 = -2^-24, 0x3C00 = 1.0,
    # 0xC000 = -2.0, 0x0000 = 0.0, 0x03FF = largest subnormal.
    payload = struct.pack("<6H", 0x0001, 0x8001, 0x3C00, 0xC000, 0x0000,
                          0x03FF)

    with _catalog() as (client, config, prefix):
        sink = DriverSession(SINK_DRIVER)
        store = DriverSession(STORE_DRIVER)
        driver = CatalogDriver()
        try:
            spool_root = Path(tempfile.mkdtemp()) / "spool"
            _stage(sink, spool_root, 90, payload, dtype="float16", shape=(6,))
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1)
            assert uploaded["ok"], uploaded

            _open_helper(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            indexed = driver.call(
                op="index", refs=uploaded["refs"], endpoint=fake_s3,
                bucket=BUCKET, region=REGION, access=ACCESS, secret=SECRET,
                insecure=True)
            assert indexed["ok"], indexed

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            native = driver.call(
                op="summarize_core", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native

            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(
                CaptureQuery(tenant_id="t", limit=10))
            (expected,) = python_reader.summarize(selection,
                                                  byte_limit=1 << 30)

            (summary,) = native["summaries"]
            assert summary["element_count"] == expected.core.element_count
            assert summary["finite_count"] == expected.core.finite_count
            assert summary["zero_fraction"] == expected.core.zero_fraction
            assert summary["mean"] == expected.core.mean, summary
            assert summary["l2_norm"] == expected.core.l2_norm, summary
        finally:
            sink.close()
            store.close()
            driver.close()


def test_summary_parity_on_int64_beyond_the_double_mantissa(fake_s3):
    """Integer order statistics must never travel through a double.

    Every integer extreme was derived by casting the float64-decoded value
    back to int64_t, which is wrong twice over: `[2**63-2, 2**63-1]` -- a
    strictly POSITIVE tensor -- rounded up to 2**63, and
    `static_cast<int64_t>(9.22e18)` is undefined-turned-INT64_MIN on
    x86-64, so both minimum_int and maximum_int came back maximally
    NEGATIVE. `[2**53+1, 2**53+3]` merely rounded to the even neighbours.
    The magnitude comparison had the same disease: for
    `[INT64_MIN, INT64_MAX]` abs_max_int came out negative.

    summary.py is the oracle -- int(flat.min()), int(flat.max()) and
    max(abs(...)) in Python's unbounded int space, pinned as the
    CORE_SUMMARY_VERSION 2 contract. The one value the oracle can name and
    the wire cannot is abs_max == 2**63: `abs_max_int` is Int64 on the
    wire, so it saturates at INT64_MAX there while `abs_max` (double)
    carries the exact magnitude.
    """
    import struct
    import tempfile

    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )
    from dmi.storage.capture import CaptureReader, S3PackStore, S3StoreConfig
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureQuery

    INT64_MAX = 2**63 - 1
    payloads = {
        # the widest possible span, whose abs_max is the unrepresentable 2**63
        91: struct.pack("<2q", -(2**63), INT64_MAX),
        # all-negative: the magnitude lives at the minimum end
        92: struct.pack("<2q", -(2**63), -1),
        # strictly positive, above the mantissa: nothing may turn negative
        93: struct.pack("<2q", 2**63 - 2, INT64_MAX),
        # just past 2**53, where a double starts skipping odd integers
        94: struct.pack("<2q", 2**53 + 1, 2**53 + 3),
    }

    with _catalog() as (client, config, prefix):
        sink = DriverSession(SINK_DRIVER)
        store = DriverSession(STORE_DRIVER)
        driver = CatalogDriver()
        try:
            spool_root = Path(tempfile.mkdtemp()) / "spool"
            for index, payload in payloads.items():
                _stage(sink, spool_root, index, payload, dtype="int64",
                       shape=(2,))
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1)
            assert uploaded["ok"], uploaded

            _open_helper(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            indexed = driver.call(
                op="index", refs=uploaded["refs"], endpoint=fake_s3,
                bucket=BUCKET, region=REGION, access=ACCESS, secret=SECRET,
                insecure=True)
            assert indexed["ok"], indexed

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            native = driver.call(
                op="summarize_core", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native

            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(
                CaptureQuery(tenant_id="t", limit=10))
            expected = {s.capture_id: s.core for s in python_reader.summarize(
                selection, byte_limit=1 << 30)}
            assert len(expected) == len(payloads), expected

            for summary in native["summaries"]:
                core = expected[summary["capture_id"]]
                assert isinstance(core.minimum, int), core
                assert summary["minimum_int"] == core.minimum, (
                    summary["capture_id"], summary)
                assert summary["maximum_int"] == core.maximum, (
                    summary["capture_id"], summary)
                # The exact magnitude rides the double, always.
                assert summary["abs_max"] == float(core.abs_max), (
                    summary["capture_id"], summary)
                # ...and the Int64 wire field carries it whenever Int64 can,
                # saturating only for the single value that has no Int64
                # representation.
                assert summary["abs_max_int"] == min(core.abs_max, INT64_MAX), (
                    summary["capture_id"], summary)
        finally:
            sink.close()
            store.close()
            driver.close()


def test_the_summary_element_budget_counts_each_element_once(fake_s3):
    """The element budget is prod(shape), not prod(shape) squared.

    The budget multiplied prod(shape) by decoded_length/dtype_bytes -- but
    the pack index enforces decoded_length == prod(shape) * dtype_bytes, so
    the second factor IS prod(shape) and the budget counted N**2 elements
    for an N-element capture. With the shared 64_000_000 default and a
    strict `>`, a single capture of 8001 elements or more was refused
    natively while Python summarised it fine. Python (reader.py, which sums
    math.prod(shape) alone) is the oracle.
    """
    import struct
    import tempfile

    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )
    from dmi.storage.capture import CaptureReader, S3PackStore, S3StoreConfig
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureQuery

    # 16384 elements: 16384 <= 64_000_000 but 16384**2 is not. Rank 1 on
    # purpose -- a rank-2 shape trips the footer-row parser first, which is
    # a different defect and not what this test pins.
    elements = 16384
    payload = struct.pack(f"<{elements}f", *([0.5] * elements))

    with _catalog() as (client, config, prefix):
        sink = DriverSession(SINK_DRIVER)
        store = DriverSession(STORE_DRIVER)
        driver = CatalogDriver()
        try:
            spool_root = Path(tempfile.mkdtemp()) / "spool"
            _stage(sink, spool_root, 91, payload, dtype="float32",
                   shape=(elements,))
            uploaded = store.call(
                op="upload_pending", **_store_base(fake_s3),
                root=str(spool_root), spool_max_bytes=1 << 40, limit=-1)
            assert uploaded["ok"], uploaded

            _open_helper(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            indexed = driver.call(
                op="index", refs=uploaded["refs"], endpoint=fake_s3,
                bucket=BUCKET, region=REGION, access=ACCESS, secret=SECRET,
                insecure=True)
            assert indexed["ok"], indexed

            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            native = driver.call(
                op="summarize_core", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
                byte_limit=1 << 30, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert native["ok"], native

            python_catalog = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            python_store = S3PackStore.from_config(
                S3StoreConfig(
                    endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                    access_key_id=ACCESS, secret_access_key=SECRET,
                    store_id="native-test", allow_insecure_http=True))
            python_reader = CaptureReader(
                python_catalog, {"native-test": python_store})
            selection = python_reader.select(
                CaptureQuery(tenant_id="t", limit=10))
            (expected,) = python_reader.summarize(selection,
                                                  byte_limit=1 << 30)

            (summary,) = native["summaries"]
            assert expected.core.element_count == elements, expected.core
            assert summary["element_count"] == expected.core.element_count
            assert summary["mean"] == expected.core.mean, summary
        finally:
            sink.close()
            store.close()
            driver.close()


def test_hydration_budget_refused_before_any_fetch(fake_s3):
    with _catalog() as (client, config, prefix):
        try:
            sink, store, driver, refs = _e2e_setup(fake_s3, prefix)
            native_select = driver.call(
                op="select", tenant_id="t", limit=10,
                endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                secret=SECRET, insecure=True)
            assert native_select["ok"], native_select
            sel = native_select["selection"]
            refused = driver.call(
                op="hydrate", selection_id=sel["selection_id"],
                capture_ids=sel["capture_ids"],
                catalog_watermark=sel["catalog_watermark"],
                filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
                byte_limit=0, endpoint=fake_s3, bucket=BUCKET,
                access=ACCESS, secret=SECRET, insecure=True)
            assert not refused["ok"]
            assert "hydration byte limit exceeded" in refused["message"], (
                refused)
        finally:
            sink.close()
            store.close()
            driver.close()


def test_filter_hash_escapes_like_python_json(fake_s3):
    """The filter hash must hash the SAME bytes Python's json.dumps does.

    Values are hashed raw today: a quote, backslash or non-ASCII character
    in a human-named id produces a different byte sequence on each side,
    and the cursor the native reader issues is refused by the Python
    reader's decode_cursor (filter_hash mismatch) — and vice versa.
    """
    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open_helper(driver, prefix)
            descriptors = _descriptor_dicts(2)
            for entry in descriptors:
                entry["run_id"] = 'a"b\\c'
                entry["tenant_id"] = "tény"
            _publish_native(driver, prefix, descriptors, 7)

            from dmi.storage.capture.model import CaptureQuery

            reader = _python_reader(client, config)
            native = driver.call(
                op="search", limit=1, tenant_id="tény",
                run_id='a"b\\c')
            assert native["ok"], native
            assert native["next_cursor"], "the page must owe a cursor"
            # THE BINDING: the Python reader accepts the native cursor —
            # its decode_cursor compares the embedded filter_hash against
            # Python's own json.dumps-derived hash.
            page = reader.search(
                CaptureQuery(tenant_id="tény", run_id='a"b\\c',
                             cursor=native["next_cursor"], limit=1))
            assert len(page.items) == 1, page
            assert page.items[0].metadata.run_id == 'a"b\\c'

            # And the reverse: a Python-issued cursor accepted natively.
            python_page = reader.search(
                CaptureQuery(tenant_id="tény", run_id='a"b\\c', limit=1))
            walked = driver.call(
                op="search", limit=1, tenant_id="tény", run_id='a"b\\c',
                cursor=python_page.next_cursor)
            assert walked["ok"], walked
            assert walked["items"], walked
        finally:
            driver.close()


def test_fp8_and_wide_int_summaries_parity(fake_s3):
    """fp8 and uint16/uint32 are first-class dtypes on both sides.

    fp8 carries NaN and Inf (e4m3fn has no infinities — exponent all-ones
    is NaN; e5m2 has both), so the NaN/Inf counting paths are part of what
    parity means. The core summaries must agree exactly, and the payloads
    must hydrate to the same bytes.
    """
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureMetadata, CaptureRecord

    def meta_and_payload(dtype, payload, index, elements):
        meta = CaptureMetadata(
            capture_id=f"fp8-{index}", tenant_id="t", experiment_id="e",
            run_id="r", session_id="s", request_id="q",
            sequence_id=f"n{index}", model_id="m", model_revision="mr",
            adapter_revision=None, capture_policy_version="v",
            hook_name="h", layer_number=0, producer_rank=0, step_number=index,
            token_start=0, token_end=1, batch_position=0, dtype=dtype,
            shape=(elements,), captured_at_ns=1_700_000_000_000_000_000 + index,
        )
        return CaptureRecord(metadata=meta, payload=payload)

    with _catalog() as (client, config, prefix):
        try:
            sink, store, driver, refs = None, None, None, None
            # The fp8 payloads: e4m3fn (+0, -0, 1, -1, NaN, subnormal, NaN)
            # and e5m2 (1, +inf, NaN, subnormal, -2, 0, 4).
            e4m3_payload = bytes([0x00, 0x80, 0x38, 0xB8, 0x7F, 0x01, 0x7E])
            e5m2_payload = bytes([0x3C, 0x7C, 0x7E, 0x01, 0xFC, 0x00, 0x40])
            u32_payload = (1024).to_bytes(4, "little") + (0).to_bytes(
                4, "little") + (2**32 - 1).to_bytes(4, "little")
            u16_payload = (65535).to_bytes(2, "little") + (1).to_bytes(
                2, "little")

            # The native write path takes descriptor dicts; the pack round
            # trip goes through the reference PackWriter (the bytes must be
            # identical either way — the format is dtype-agnostic).
            from benchmarks.bench_capture_catalog import synthetic_descriptors
            base = synthetic_descriptors(1)[0]
            loc = base.locator
            corpus = [
                meta_and_payload("float8_e4m3fn", e4m3_payload, 0, 7),
                meta_and_payload("float8_e5m2", e5m2_payload, 1, 7),
                meta_and_payload("uint32", u32_payload, 2, 3),
                meta_and_payload("uint16", u16_payload, 3, 2),
            ]

            import tempfile
            from dmi.storage.capture import PackWriter, PackReader
            pack_id = str(uuid.uuid4())
            writer = PackWriter(pack_id=pack_id,
                                created_at_ns=1_700_000_000_000_000_000,
                                max_pack_bytes=8 * 1024 * 1024)
            for record in corpus:
                writer.append(record)
            sealed = writer.seal()

            # The catalog + store: publish the pack, then read it back
            # through BOTH readers.
            from dmi.storage.capture.clickhouse_catalog import (
                ClickHouseCatalogWriter,
            )
            from dmi.storage.capture.clickhouse_catalog import (
                ClickHouseCatalogWriter,
            )
            catalog_writer = ClickHouseCatalogWriter(client, config)
            driver = CatalogDriver()
            import base64 as b64mod
            try:
                catalog_writer.ensure_schema()
                _open_helper(driver, prefix)
                # The locator fields (offset, checksum) exist only after
                # packing — read the sealed pack back and take the real
                # descriptors rather than inventing them.
                from dmi.storage.capture.pack import PackReader
                packed = PackReader.from_bytes(sealed.data)
                packed_descriptors = packed.descriptors(
                    store_id="native-test",
                    object_key=f"packs/{pack_id}.dmi-pack")
                rows = []
                for d in packed_descriptors:
                    m, l = d.metadata, d.locator
                    rows.append({
                        "capture_id": m.capture_id, "tenant_id": m.tenant_id,
                        "experiment_id": m.experiment_id, "run_id": m.run_id,
                        "session_id": m.session_id,
                        "request_id": m.request_id,
                        "sequence_id": m.sequence_id, "model_id": m.model_id,
                        "model_revision": m.model_revision,
                        "adapter_revision": m.adapter_revision,
                        "capture_policy_version": m.capture_policy_version,
                        "hook_name": m.hook_name,
                        "layer_number": m.layer_number,
                        "producer_rank": m.producer_rank,
                        "step_number": m.step_number,
                        "token_start": m.token_start, "token_end": m.token_end,
                        "batch_position": m.batch_position, "dtype": m.dtype,
                        "shape": list(m.shape),
                        "captured_at_ns": m.captured_at_ns,
                        "pack_id": pack_id, "store_id": "native-test",
                        "object_key": f"packs/{pack_id}.dmi-pack",
                        "object_bytes": len(sealed.data),
                        "pack_checksum": sealed.checksum,
                        "pack_record_count": len(corpus),
                        "payload_offset": l.offset,
                        "stored_length": l.stored_length,
                        "decoded_length": l.decoded_length,
                        "codec": l.codec, "payload_checksum": l.checksum,
                    })
                version = 7
                driver.call(op="acquire", holder="writer")
                driver.call(op="write_descriptors", descriptors=rows,
                            index_version=version)
                driver.call(
                    op="publish_snapshot", index_version=version,
                    refs=[{"store_id": "native-test", "pack_id": pack_id}],
                    published_at_ns=version, indexed_rows=len(corpus),
                    indexed_packs=1)

                # The object bytes: upload the sealed pack via the fake S3.
                import boto3
                from botocore.config import Config as BotoConfig
                s3 = boto3.client(
                    "s3", endpoint_url=fake_s3, region_name=REGION,
                    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
                    config=BotoConfig(s3={"addressing_style": "path"}),
                    verify=False)
                s3.put_object(
                    Bucket=BUCKET, Key=f"packs/{pack_id}.dmi-pack",
                    Body=sealed.data)

                from dmi.storage.capture.model import CaptureQuery
                python_catalog = ClickHouseCaptureCatalog(
                    client, ClickHouseReaderConfig.from_catalog(config))
                python_store = S3PackStore.from_config(
                    S3StoreConfig(
                        endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                        access_key_id=ACCESS, secret_access_key=SECRET,
                        store_id="native-test", allow_insecure_http=True))
                python_reader = CaptureReader(
                    python_catalog, {"native-test": python_store})
                selection = python_reader.select(
                    CaptureQuery(tenant_id="t", limit=10))
                assert len(selection.capture_ids) == 4, selection

                hydrated = python_reader.hydrate(
                    selection, byte_limit=1 << 30)
                by_id = {i.capture_id: i.payload for i in hydrated}
                assert by_id["fp8-0"] == e4m3_payload
                assert by_id["fp8-1"] == e5m2_payload
                assert by_id["fp8-2"] == u32_payload
                assert by_id["fp8-3"] == u16_payload

                # The native side: same bytes.
                native_select = driver.call(
                    op="select", tenant_id="t", limit=10,
                    endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                    secret=SECRET, insecure=True)
                assert native_select["ok"], native_select
                sel = native_select["selection"]
                native_hydrated = driver.call(
                    op="hydrate", selection_id=sel["selection_id"],
                    capture_ids=sel["capture_ids"],
                    catalog_watermark=sel["catalog_watermark"],
                    filter_hash=sel["filter_hash"],
                    tenant_id=sel["tenant_id"], byte_limit=1 << 30,
                    endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                    secret=SECRET, insecure=True)
                assert native_hydrated["ok"], native_hydrated
                for i, payload in enumerate(native_hydrated["payloads"]):
                    assert b64mod.b64decode(payload) == \
                        corpus[i].payload, corpus[i].capture_id

                # Core summaries: exact parity, NaN/Inf counting included.
                python_summaries = python_reader.summarize(
                    selection, byte_limit=1 << 30)
                by_id = {s.capture_id: s.core for s in python_summaries}
                native_summaries = driver.call(
                    op="summarize_core",
                    selection_id=sel["selection_id"],
                    capture_ids=sel["capture_ids"],
                    catalog_watermark=sel["catalog_watermark"],
                    filter_hash=sel["filter_hash"],
                    tenant_id=sel["tenant_id"], byte_limit=1 << 30,
                    endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                    secret=SECRET, insecure=True)
                assert native_summaries["ok"], native_summaries
                for summary in native_summaries["summaries"]:
                    expected = by_id[summary["capture_id"]]
                    assert summary["element_count"] == expected.element_count
                    assert summary["nan_count"] == expected.nan_count, (
                        summary["capture_id"])
                    assert summary["inf_count"] == expected.inf_count, (
                        summary["capture_id"])
                    assert summary["finite_count"] == expected.finite_count
                    assert summary["zero_fraction"] == \
                        expected.zero_fraction
                    assert summary["mean"] == expected.mean, (
                        summary["capture_id"])
                    assert summary["minimum"] == expected.minimum
                    assert summary["maximum"] == expected.maximum
                    assert summary["l2_norm"] == expected.l2_norm, (
                        summary["capture_id"])
            finally:
                driver.close()
        finally:
            pass


def test_zero_element_tensor_hydrates_to_empty_bytes(fake_s3):
    """A zero-dimension tensor's payload is genuinely zero bytes — legal.

    The sentinel for 'unresolved' must be tracked separately from the
    payload length: an empty payload is a VALID result, not evidence that
    a slot was never filled. Python hydration handles this; the native
    side must too.
    """
    from dmi.storage.capture import (
        CaptureReader, S3PackStore, S3StoreConfig,
    )
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureMetadata, CaptureRecord
    from dmi.storage.capture.pack import PackWriter, PackReader as PR

    with _catalog() as (client, config, prefix):
        try:
            sink, store, driver = None, None, None
            empty_payload = b""
            meta = CaptureMetadata(
                capture_id="scalar-0", tenant_id="t", experiment_id="e",
                run_id="r", session_id="s", request_id="q",
                sequence_id="n0", model_id="m", model_revision="mr",
                adapter_revision=None, capture_policy_version="v",
                hook_name="h", layer_number=0, producer_rank=0,
                step_number=0, token_start=0, token_end=0,
                batch_position=0, dtype="float32", shape=(0,),
                captured_at_ns=1_700_000_000_000_000_000,
            )
            record = CaptureRecord(metadata=meta, payload=empty_payload)
            assert record.payload == b"", "the record validates"
            assert meta.logical_bytes == 0, "scalar: zero elements"

            pack_id = str(uuid.uuid4())
            writer = PackWriter(
                pack_id=pack_id, created_at_ns=1_700_000_000_000_000_000,
                max_pack_bytes=8 * 1024 * 1024)
            writer.append(record)
            sealed = writer.seal()

            from dmi.storage.capture.clickhouse_catalog import (
                ClickHouseCatalogWriter,
            )
            catalog_writer = ClickHouseCatalogWriter(client, config)
            catalog_writer.ensure_schema()
            driver = CatalogDriver()
            try:
                _open_helper(driver, prefix)
                packed = PR.from_bytes(sealed.data)
                packed_descriptors = packed.descriptors(
                    store_id="native-test",
                    object_key=f"packs/{pack_id}.dmi-pack")
                rows = []
                for d in packed_descriptors:
                    m, l = d.metadata, d.locator
                    rows.append({
                        "capture_id": m.capture_id,
                        "tenant_id": m.tenant_id,
                        "experiment_id": m.experiment_id,
                        "run_id": m.run_id,
                        "session_id": m.session_id,
                        "request_id": m.request_id,
                        "sequence_id": m.sequence_id,
                        "model_id": m.model_id,
                        "model_revision": m.model_revision,
                        "adapter_revision": m.adapter_revision,
                        "capture_policy_version": m.capture_policy_version,
                        "hook_name": m.hook_name,
                        "layer_number": m.layer_number,
                        "producer_rank": m.producer_rank,
                        "step_number": m.step_number,
                        "token_start": m.token_start,
                        "token_end": m.token_end,
                        "batch_position": m.batch_position,
                        "dtype": m.dtype,
                        "shape": list(m.shape),
                        "captured_at_ns": m.captured_at_ns,
                        "pack_id": pack_id, "store_id": "native-test",
                        "object_key": f"packs/{pack_id}.dmi-pack",
                        "object_bytes": len(sealed.data),
                        "pack_checksum": sealed.checksum,
                        "pack_record_count": 1,
                        "payload_offset": l.offset,
                        "stored_length": l.stored_length,
                        "decoded_length": l.decoded_length,
                        "codec": l.codec, "payload_checksum": l.checksum,
                    })
                driver.call(op="acquire", holder="writer")
                driver.call(op="write_descriptors", descriptors=rows,
                            index_version=7)
                driver.call(
                    op="publish_snapshot", index_version=7,
                    refs=[{"store_id": "native-test", "pack_id": pack_id}],
                    published_at_ns=7, indexed_rows=1, indexed_packs=1)

                import boto3 as _boto3
                from botocore.config import Config as _BotoConfig
                s3_client = _boto3.client(
                    "s3", endpoint_url=fake_s3, region_name=REGION,
                    aws_access_key_id=ACCESS,
                    aws_secret_access_key=SECRET,
                    config=_BotoConfig(s3={"addressing_style": "path"}),
                    verify=False)
                s3_client.put_object(
                    Bucket=BUCKET, Key=f"packs/{pack_id}.dmi-pack",
                    Body=sealed.data)

                # THE ORACLE: Python hydrates the empty tensor correctly.
                python_catalog = ClickHouseCaptureCatalog(
                    client, ClickHouseReaderConfig.from_catalog(config))
                python_store = S3PackStore.from_config(
                    S3StoreConfig(
                        endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                        access_key_id=ACCESS, secret_access_key=SECRET,
                        store_id="native-test",
                        allow_insecure_http=True))
                python_reader = CaptureReader(
                    python_catalog, {"native-test": python_store})
                from dmi.storage.capture.model import CaptureQuery
                selection = python_reader.select(
                    CaptureQuery(tenant_id="t", limit=10))
                hydrated = python_reader.hydrate(
                    selection, byte_limit=1 << 30)
                assert len(hydrated) == 1
                assert hydrated[0].payload == b"", (
                    "python hydrates the zero-element tensor to empty bytes")

                # THE NATIVE SIDE: must also return empty bytes, not refuse.
                native_select = driver.call(
                    op="select", tenant_id="t", limit=10,
                    endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                    secret=SECRET, insecure=True)
                assert native_select["ok"], native_select
                sel = native_select["selection"]
                native_hydrated = driver.call(
                    op="hydrate", selection_id=sel["selection_id"],
                    capture_ids=sel["capture_ids"],
                    catalog_watermark=sel["catalog_watermark"],
                    filter_hash=sel["filter_hash"],
                    tenant_id=sel["tenant_id"], byte_limit=1 << 30,
                    endpoint=fake_s3, bucket=BUCKET, access=ACCESS,
                    secret=SECRET, insecure=True)
                assert native_hydrated["ok"], native_hydrated
                assert len(native_hydrated["payloads"]) == 1
                import base64
                decoded = base64.b64decode(native_hydrated["payloads"][0])
                assert decoded == b"", (
                    "the zero-element tensor hydrates to empty bytes, "
                    "not a refusal")
            finally:
                driver.close()
        finally:
            pass


# --- review findings on the hydrate/summary path (PR #127) --------------------
#
# Each of these reproduces one reviewer repro against the live catalog:
# metadata the SQL escaper rewrites, SQL NULL versus the string "NULL", the
# footer byte budget, a rank-0 capture, and a non-BMP identifier. Python is
# the oracle in every case.


def _stage_meta(sink, root, meta, payload):
    """`_stage` with the caller's CaptureMetadata, for values `_stage` fixes."""
    import base64 as _base64
    import subprocess as _subprocess

    from tests.test_native_uploader import STORE_DRIVER

    before = set(root.rglob("*.dmi-pack.ready")) if root.exists() else set()
    assert sink.call(
        op="open", root=str(root), max_bytes=1 << 40,
        max_queue_records=256, max_queue_bytes=1 << 20,
        max_pack_bytes=8 * 1024 * 1024, max_pack_records=10_000,
        max_linger_ns=1_000_000_000, overload="drop_newest",
        admission_timeout=-1,
    )["ok"]
    response = sink.call(
        op="submit", metadata=meta.to_mapping(),
        payload_b64=_base64.b64encode(payload).decode(),
    )
    assert response["admission"] == "accepted", response
    assert sink.call(op="flush", timeout=30)["ok"]
    snapshot = sink.call(op="close", timeout=30)["snapshot"]
    assert snapshot["persisted_records"] == 1, snapshot
    new = set(root.rglob("*.dmi-pack.ready")) - before
    assert len(new) == 1, new
    recover = _subprocess.run(
        [str(STORE_DRIVER.parent / "conformance_spool")],
        input=json.dumps({"op": "recover", "root": str(root),
                          "max_bytes": 1 << 40}) + "\n",
        capture_output=True, text=True, timeout=30,
    )
    entries = json.loads(recover.stdout.strip())["staged"]
    match = [e for e in entries if Path(e["path"]) in new]
    assert len(match) == 1, entries
    return match[0]


def _review_meta(index, **overrides):
    from dmi.storage.capture.model import CaptureMetadata

    fields = dict(
        capture_id=f"upload-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v", hook_name="h",
        layer_number=0, producer_rank=0, step_number=index,
        token_start=index, token_end=index + 1, batch_position=0,
        dtype="uint8", shape=(16,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )
    fields.update(overrides)
    return CaptureMetadata(**fields)


def _review_setup(fake_s3, prefix, records):
    """sink -> uploader -> native index for `records` = [(meta, payload)]."""
    import tempfile

    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _store_base,
    )

    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    driver = CatalogDriver()
    try:
        spool_root = Path(tempfile.mkdtemp()) / "spool"
        for meta, payload in records:
            _stage_meta(sink, spool_root, meta, payload)
        uploaded = store.call(
            op="upload_pending", **_store_base(fake_s3),
            root=str(spool_root), spool_max_bytes=1 << 40, limit=-1)
        assert uploaded["ok"], uploaded
        _open_helper(driver, prefix)
        driver.call(op="acquire", holder="indexer")
        result = driver.call(
            op="index", refs=uploaded["refs"], endpoint=fake_s3,
            bucket=BUCKET, region=REGION, access=ACCESS, secret=SECRET,
            insecure=True)
        assert result["ok"], result
        assert result["result"]["indexed_packs"] == len(records), result
        return sink, store, driver
    except Exception:
        sink.close()
        store.close()
        driver.close()
        raise


def _python_reader_for(client, config, fake_s3):
    from dmi.storage.capture import CaptureReader, S3PackStore, S3StoreConfig
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )

    return CaptureReader(
        ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)),
        {"native-test": S3PackStore.from_config(
            S3StoreConfig(
                endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
                access_key_id=ACCESS, secret_access_key=SECRET,
                store_id="native-test", allow_insecure_http=True))})


def _native_selection(driver, fake_s3):
    native_select = driver.call(
        op="select", tenant_id="t", limit=10, endpoint=fake_s3,
        bucket=BUCKET, access=ACCESS, secret=SECRET, insecure=True)
    assert native_select["ok"], native_select
    return native_select["selection"]


def _native_read(driver, op, sel, fake_s3, **limits):
    limits.setdefault("byte_limit", 1 << 30)
    return driver.call(
        op=op, selection_id=sel["selection_id"],
        capture_ids=sel["capture_ids"],
        catalog_watermark=sel["catalog_watermark"],
        filter_hash=sel["filter_hash"], tenant_id=sel["tenant_id"],
        endpoint=fake_s3, bucket=BUCKET, access=ACCESS, secret=SECRET,
        insecure=True, **limits)


@pytest.mark.parametrize("hook_name", [
    "block\\resid", "block'quoted", "block\tresid",
])
def test_hydrate_accepts_metadata_the_sql_escaper_rewrites(fake_s3, hook_name):
    """The footer binding compares DECODED values, not escaped footer text.

    The footer row renders hook_name through sql_quote (backslash, quote and
    tab all rewritten); the catalog row is already decoded. Comparing the
    two representations refused these three valid captures with "catalog
    descriptor does not match the pack footer: field 12". Python hydrates
    them; native must too, and hand back the same 16 bytes.
    """
    import base64 as _base64

    from dmi.storage.capture.model import CaptureQuery

    payload = bytes(range(16))
    with _catalog() as (client, config, prefix):
        sink, store, driver = _review_setup(
            fake_s3, prefix, [(_review_meta(1, hook_name=hook_name), payload)])
        try:
            python_reader = _python_reader_for(client, config, fake_s3)
            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            (hydrated,) = python_reader.hydrate(selection, byte_limit=1 << 30)
            assert hydrated.payload == payload
            assert hydrated.descriptor.metadata.hook_name == hook_name

            sel = _native_selection(driver, fake_s3)
            native = _native_read(driver, "hydrate", sel, fake_s3)
            assert native["ok"], native
            assert [_base64.b64decode(p) for p in native["payloads"]] == [payload]
        finally:
            sink.close()
            store.close()
            driver.close()


def test_hydrate_keeps_sql_null_apart_from_the_string_null(fake_s3):
    """A footer adapter_revision of "NULL" (the string) must not bind to a
    catalog row whose adapter_revision is SQL NULL.

    The pack is written and indexed with the four-letter string; the
    catalog row alone is then altered to NULL. Python refuses the
    descriptor/footer mismatch; native used to normalise both spellings to
    the same value and return the payload.
    """
    from dmi.storage.capture.model import CaptureQuery
    from dmi.storage.capture.reader import PackFormatError

    payload = bytes(range(16))
    with _catalog() as (client, config, prefix):
        sink, store, driver = _review_setup(
            fake_s3, prefix,
            [(_review_meta(1, adapter_revision="NULL"), payload)])
        try:
            python_reader = _python_reader_for(client, config, fake_s3)
            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            # Control: with the catalog agreeing, both sides hydrate.
            (hydrated,) = python_reader.hydrate(selection, byte_limit=1 << 30)
            assert hydrated.descriptor.metadata.adapter_revision == "NULL"
            sel = _native_selection(driver, fake_s3)
            assert _native_read(driver, "hydrate", sel, fake_s3)["ok"]

            database = config.database
            client.execute(
                f"ALTER TABLE `{database}`.`{prefix}_capture_raw` "
                "UPDATE adapter_revision = NULL WHERE 1 "
                "SETTINGS mutations_sync = 2")

            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            with pytest.raises(PackFormatError, match="does not match the pack footer"):
                python_reader.hydrate(selection, byte_limit=1 << 30)

            sel = _native_selection(driver, fake_s3)
            native = _native_read(driver, "hydrate", sel, fake_s3)
            assert not native["ok"], native
            assert "does not match the pack footer" in native["message"], native
        finally:
            sink.close()
            store.close()
            driver.close()


def test_the_footer_byte_budget_is_charged_before_the_footer_is_fetched(fake_s3):
    """byte_limit covers the footer reads too, checked BEFORE each fetch.

    One 16-byte capture, byte_limit=16, request_limit=3: the payload alone
    fits both limits, the trailer and footer do not fit the bytes. Python
    refuses (HydrationLimitError). Native checked the running footer total
    before charging the current pack, so for a one-pack selection there was
    no later iteration to refuse and it made all three GETs.
    """
    from dmi.storage.capture.model import CaptureQuery
    from dmi.storage.capture.reader import HydrationLimitError

    payload = bytes(range(16))
    with _catalog() as (client, config, prefix):
        sink, store, driver = _review_setup(
            fake_s3, prefix, [(_review_meta(1), payload)])
        try:
            python_reader = _python_reader_for(client, config, fake_s3)
            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            with pytest.raises(HydrationLimitError, match="byte limit"):
                python_reader.hydrate(selection, byte_limit=16, request_limit=3)

            sel = _native_selection(driver, fake_s3)
            refused = _native_read(driver, "hydrate", sel, fake_s3,
                                   byte_limit=16, request_limit=3)
            assert not refused["ok"], refused
            assert "hydration byte limit exceeded" in refused["message"], refused

            # Control: the same three requests with room for the bytes.
            granted = _native_read(driver, "hydrate", sel, fake_s3,
                                   byte_limit=1 << 30, request_limit=3)
            assert granted["ok"], granted
            # And the request half is still enforced with room for the bytes.
            refused = _native_read(driver, "hydrate", sel, fake_s3,
                                   byte_limit=1 << 30, request_limit=2)
            assert not refused["ok"], refused
            assert "hydration request limit exceeded" in refused["message"], refused
        finally:
            sink.close()
            store.close()
            driver.close()


def test_a_scalar_capture_hydrates_and_summarises(fake_s3):
    """shape=() is one element: admitted by the sink, summarised natively."""
    import base64 as _base64
    import struct

    from dmi.storage.capture.model import CaptureQuery

    payload = struct.pack("<f", 3.5)
    with _catalog() as (client, config, prefix):
        sink, store, driver = _review_setup(
            fake_s3, prefix,
            [(_review_meta(1, dtype="float32", shape=()), payload)])
        try:
            python_reader = _python_reader_for(client, config, fake_s3)
            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            (expected,) = python_reader.summarize(selection, byte_limit=1 << 30)
            assert expected.core.element_count == 1

            sel = _native_selection(driver, fake_s3)
            hydrated = _native_read(driver, "hydrate", sel, fake_s3)
            assert hydrated["ok"], hydrated
            assert [_base64.b64decode(p) for p in hydrated["payloads"]] == [payload]
            native = _native_read(driver, "summarize_core", sel, fake_s3)
            assert native["ok"], native
            (summary,) = native["summaries"]
            assert summary["element_count"] == 1
            assert summary["mean"] == expected.core.mean == 3.5
        finally:
            sink.close()
            store.close()
            driver.close()


def test_a_non_bmp_hook_name_indexes_intact(fake_s3):
    """A surrogate pair in the footer JSON is one code point in the catalog.

    The canonical footer spells U+1F600 as \\ud83d\\ude00; the indexer used to
    encode each half separately, so the catalog held six bytes that are not
    UTF-8 and the Python reader's search raised UnicodeDecodeError.
    """
    from dmi.storage.capture.model import CaptureQuery

    hook_name = "block\U0001F600"
    payload = bytes(range(16))
    with _catalog() as (client, config, prefix):
        sink, store, driver = _review_setup(
            fake_s3, prefix, [(_review_meta(1, hook_name=hook_name), payload)])
        try:
            (stored,) = client.execute(
                f"SELECT hex(hook_name) FROM `{config.database}`."
                f"`{prefix}_capture_raw`")
            assert stored[0].lower() == hook_name.encode("utf-8").hex()

            python_reader = _python_reader_for(client, config, fake_s3)
            selection = python_reader.select(CaptureQuery(tenant_id="t", limit=10))
            (hydrated,) = python_reader.hydrate(selection, byte_limit=1 << 30)
            assert hydrated.descriptor.metadata.hook_name == hook_name

            native = driver.call(op="search", limit=10, tenant_id="t",
                                 hook_names=[hook_name])
            assert native["ok"], native
            assert len(native["items"]) == 1, native
        finally:
            sink.close()
            store.close()
            driver.close()
