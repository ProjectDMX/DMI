"""Corner cases the first round of snapshot tests did not reach.

Two defects got through review because the earlier live tests confirmed the
mechanism rather than challenging the engine underneath it:

* ``ReplacingMergeTree`` deduplicates "at an unknown time". A test that writes
  two versions and reads immediately only ever exercises the pre-merge state,
  so it cannot see that a merge deletes the versions a pinned watermark needs.
* ``max_rows_per_insert`` defaults to 10,000, so a small corpus never splits a
  batch into multiple INSERTs and the non-atomic window is unreachable.

Both are forced here: merges are triggered with ``OPTIMIZE ... FINAL``, and
batches are written with a deliberately small insert size.

Run against a reachable ClickHouse:

    DMI_CLICKHOUSE_HOST=127.0.0.1 python -m pytest \
        tests/test_clickhouse_snapshot_live.py -m "manual and clickhouse" -q
"""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from uuid import uuid4

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureQuery,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
)


pytestmark = [pytest.mark.manual, pytest.mark.clickhouse]


def _publish(writer, index_version: int, *, rows: int = 0, packs: int = 0) -> None:
    """Publishing is a separate step; CatalogIndexer does it, direct writes must."""
    writer.publish_watermark(
        index_version=index_version,
        published_at_ns=index_version,
        indexed_rows=rows,
        indexed_packs=packs,
    )


@contextmanager
def _catalog():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    prefix = f"dmi_snapshot_test_{uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
    )
    writer = ClickHouseCatalogWriter(client, config)
    reader = ClickHouseCaptureCatalog(
        client, ClickHouseReaderConfig.from_catalog(config)
    )
    created = False
    try:
        writer.ensure_schema()
        created = True
        yield writer, reader, client, config
    finally:
        if created:
            database = config.database
            for kind, suffix in (
                ("VIEW", "capture"),
                ("VIEW", "pack_inventory"),
                ("TABLE", "capture_raw"),
                ("TABLE", "pack_inventory_raw"),
            ):
                client.execute(
                    f"DROP {kind} IF EXISTS `{database}`.`{prefix}_{suffix}`"
                )


def _merge(client, config):
    """Force the deduplication a background merge would eventually do."""
    client.execute(
        f"OPTIMIZE TABLE `{config.database}`.`{config.table_prefix}_capture_raw` FINAL"
    )


def _raw_rows(client, config) -> int:
    return client.execute(
        f"SELECT count() FROM `{config.database}`.`{config.table_prefix}_capture_raw`"
    )[0][0]


# --- merges versus pinned watermarks -----------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open design problem: argMax(...) WHERE index_version <= W needs rows "
        "ReplacingMergeTree is defined to delete on merge, so a pinned snapshot "
        "expires at a time nobody controls. Needs a snapshot mechanism that does "
        "not depend on superseded versions surviving."
    ),
)
def test_a_merge_does_not_destroy_a_pinned_snapshot():
    """A watermark must keep resolving after ClickHouse deduplicates.

    ReplacingMergeTree keeps only the highest version per sorting key, so any
    snapshot that reads superseded versions has an expiry no caller controls.
    """
    original = synthetic_descriptors(1)
    revised = tuple(
        d.__class__(metadata=d.metadata, locator=d.locator.__class__(
            **{**{f: getattr(d.locator, f) for f in d.locator.__slots__},
               "offset": 999_999}))
        for d in original
    )
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(original, index_version=1)
        _publish(writer, 1)
        writer.write_descriptors(revised, index_version=2)
        _publish(writer, 2)

        before = reader.get_by_ids([original[0].capture_id], watermark="1")
        assert len(before) == 1, "precondition: the pinned read works pre-merge"
        assert before[0].locator.offset == original[0].locator.offset

        _merge(client, config)

        after = reader.get_by_ids([original[0].capture_id], watermark="1")
        assert len(after) == 1, (
            "a pinned watermark stopped resolving after a merge: "
            f"raw rows went to {_raw_rows(client, config)}"
        )
        assert after[0].locator.offset == original[0].locator.offset


def test_a_walk_still_completes_after_a_merge_mid_pagination():
    """Merges run concurrently with reads; a walk must not lose rows to one."""
    corpus = synthetic_descriptors(60)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(corpus, index_version=1)
        _publish(writer, 1)
        # Re-index everything, so every row has a superseded version to lose.
        writer.write_descriptors(corpus, index_version=2)
        _publish(writer, 2)

        first = reader.search(CaptureQuery(limit=20))
        _merge(client, config)

        items = list(first.items)
        cursor = first.next_cursor
        while cursor is not None:
            page = reader.search(CaptureQuery(limit=20, cursor=cursor))
            items.extend(page.items)
            cursor = page.next_cursor

        assert len(items) == len(corpus), "a merge mid-walk dropped rows"


# --- batch atomicity ---------------------------------------------------------


def test_a_watermark_is_not_published_before_its_batch_completes():
    """One logical batch must become visible all at once.

    CatalogIndexer assigns one index_version and then writes descriptors across
    several INSERTs. A reader that samples between them pins a half-written
    batch, and the same watermark then returns more rows later.
    """
    corpus = synthetic_descriptors(4)
    with _catalog() as (writer, reader, client, config):
        # Two INSERTs at one version, as max_rows_per_insert would produce for
        # any batch larger than the insert size.
        writer.write_descriptors(corpus[:2], index_version=123)
        mid_batch = reader.current_watermark()
        rows_visible = _raw_rows(client, config)
        writer.write_descriptors(corpus[2:], index_version=123)
        _publish(writer, 123, rows=len(corpus))

        # Mid-batch the version must not be readable at all: two of four rows
        # were durable, so publishing 123 there would pin a snapshot that keeps
        # growing under the caller.
        assert int(mid_batch) < 123, (
            f"watermark {mid_batch} was readable with only {rows_visible} of "
            f"{len(corpus)} rows durable"
        )

        resolved = reader.search(
            CaptureQuery(limit=10, tenant_id=corpus[0].metadata.tenant_id)
        )
        assert len(resolved.items) == len(corpus)


def test_an_indexed_batch_is_visible_all_at_once():
    """A watermark taken after indexing must see the whole batch."""
    corpus = synthetic_descriptors(30)
    with _catalog() as (writer, reader, client, config):
        for start in range(0, len(corpus), 7):
            writer.write_descriptors(corpus[start : start + 7], index_version=9)
        _publish(writer, 9, rows=len(corpus))

        page = reader.search(CaptureQuery(limit=100))

        assert len(page.items) == len(corpus)
        assert page.watermark == "9"


# --- ordering and boundary conditions ----------------------------------------


def test_a_watermark_below_every_row_returns_nothing():
    corpus = synthetic_descriptors(5)
    with _catalog() as (writer, reader, _, _):
        writer.write_descriptors(corpus, index_version=10)
        _publish(writer, 10)

        assert reader.get_by_ids([corpus[0].capture_id], watermark="9") == ()


def test_an_empty_catalog_reports_a_zero_watermark():
    with _catalog() as (_, reader, _, _):
        assert reader.current_watermark() == "0"
        assert reader.search(CaptureQuery(limit=10)).items == ()
