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
        return json.loads(self.proc.stdout.readline())

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
    # The native TSV renders an unquoted NULL inside the resolved tuple;
    # Python flattens the same column to None → "".
    return ["" if f in ("None", "NULL") else f for f in fields]


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

def _e2e_setup(fake_s3, prefix, record_count=3):
    """sink → uploader → native index; returns (drivers, refs, descriptor ids)."""
    from tests.test_native_uploader import (
        SINK_DRIVER, STORE_DRIVER, DriverSession, _stage, _store_base,
    )

    sink = DriverSession(SINK_DRIVER)
    store = DriverSession(STORE_DRIVER)
    driver = CatalogDriver()
    try:
        import tempfile
        spool_root = Path(tempfile.mkdtemp()) / "spool"
        staged = [_stage(sink, spool_root, 60 + i)
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
