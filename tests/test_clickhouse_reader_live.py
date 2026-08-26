"""Live snapshot and pagination tests for the ClickHouse catalog reader.

Run against a reachable ClickHouse:

    DMI_CLICKHOUSE_HOST=127.0.0.1 python -m pytest tests/test_clickhouse_reader_live.py \
        -m "manual and clickhouse" -q
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
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


def _commit(writer, descriptors, index_version: int) -> None:
    """Snapshots are bounded by committed packs, so a direct write must commit."""
    seen, refs = set(), []
    for item in descriptors:
        ref = item.locator.pack_ref
        if (ref.store_id, ref.pack_id) not in seen:
            seen.add((ref.store_id, ref.pack_id))
            refs.append(ref)
    writer.commit_packs(refs, index_version=index_version)


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
    prefix = f"dmi_reader_test_{uuid4().hex}"
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
        yield writer, reader
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


def _walk(reader, query: CaptureQuery):
    """Page through a query, returning every descriptor and the page count."""
    items, pages, cursor = [], 0, query.cursor
    while True:
        page = reader.search(replace(query, cursor=cursor))
        items.extend(page.items)
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            return tuple(items), pages, page.watermark
        assert pages < 1000, "pagination failed to terminate"


def test_a_full_walk_returns_the_corpus_exactly():
    corpus = synthetic_descriptors(250)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _commit(writer, corpus, 1)
        _publish(writer, 1)

        walked, pages, _ = _walk(reader, CaptureQuery(limit=40))

        assert pages == 7
        assert walked == corpus
        assert len({item.capture_id for item in walked}) == len(corpus)


def test_watermark_isolates_rows_indexed_after_the_first_page():
    corpus = synthetic_descriptors(100)
    # Its own pack: a pack is sealed before it is committed, so captures never
    # appear inside one that the catalog already knows about.
    late_pack = str(uuid4())
    later = tuple(
        replace(
            item,
            metadata=replace(item.metadata, capture_id=f"late-{index}"),
            locator=replace(item.locator, pack_id=late_pack),
        )
        for index, item in enumerate(synthetic_descriptors(50))
    )
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _commit(writer, corpus, 1)
        _publish(writer, 1)

        first = reader.search(CaptureQuery(limit=40))
        writer.write_descriptors(later, index_version=2)
        _commit(writer, later, 2)
        _publish(writer, 2)

        rest, _, _ = _walk(reader, CaptureQuery(limit=40, cursor=first.next_cursor))

        walked = first.items + rest
        assert walked == corpus
        assert not any(item.capture_id.startswith("late-") for item in walked)


def test_replay_is_invisible_because_it_rewrites_identical_descriptors():
    """Why the snapshot needs no version selection.

    Descriptors are derived from an immutable pack footer, so re-indexing a
    pack writes byte-identical rows. That invariant is what lets the snapshot
    be "packs committed at or before W" and lets a merge collapse duplicate
    descriptor rows freely -- there is no content to choose between.

    If a future change ever makes a re-indexed descriptor differ from the
    original, this test fails, and the snapshot design has to be revisited.
    """
    corpus = synthetic_descriptors(5)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _commit(writer, corpus, 1)
        _publish(writer, 1)
        at_first = reader.get_by_ids([corpus[0].capture_id], watermark="1")

        writer.write_descriptors(corpus, index_version=2)
        _commit(writer, corpus, 2)
        _publish(writer, 2)
        at_second = reader.get_by_ids(
            [corpus[0].capture_id], watermark=reader.current_watermark()
        )

        assert at_first == at_second, "a replay changed a descriptor"


def test_replayed_indexing_yields_one_logical_row():
    corpus = synthetic_descriptors(10)
    with _catalog() as (writer, reader):
        for version in (1, 2, 3):
            writer.write_descriptors(corpus, index_version=version)
            _commit(writer, corpus, version)
            _publish(writer, version)

        walked, _, _ = _walk(reader, CaptureQuery(limit=10))

        assert walked == corpus


def test_get_by_ids_resolves_a_selection_at_its_watermark():
    corpus = synthetic_descriptors(20)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _commit(writer, corpus, 1)
        _publish(writer, 1)

        page = reader.search(CaptureQuery(limit=20))
        wanted = [item.capture_id for item in page.items[:5]]

        resolved = reader.get_by_ids(wanted, watermark=page.watermark)

        assert {item.capture_id for item in resolved} == set(wanted)


def test_filters_narrow_the_walk():
    corpus = synthetic_descriptors(30)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _commit(writer, corpus, 1)
        _publish(writer, 1)

        walked, _, _ = _walk(
            reader,
            CaptureQuery(
                limit=10,
                hook_names=("resid_pre",),
                captured_after_ns=corpus[10].metadata.captured_at_ns,
                captured_before_ns=corpus[19].metadata.captured_at_ns,
            ),
        )

        assert walked == corpus[10:20]
