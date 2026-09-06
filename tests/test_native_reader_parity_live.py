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
            "shape": str(list(meta.shape)),
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
