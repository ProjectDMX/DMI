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


def _refs(descriptors):
    """The distinct packs a set of descriptors lives in."""
    seen, refs = set(), []
    for item in descriptors:
        ref = item.locator.pack_ref
        if (ref.store_id, ref.pack_id) not in seen:
            seen.add((ref.store_id, ref.pack_id))
            refs.append(ref)
    return refs


def _commit(writer, descriptors, index_version: int) -> None:
    """Record the replay guard; visibility comes from _publish, not from here."""
    writer.commit_packs(_refs(descriptors), index_version=index_version)


def _publish(writer, index_version: int, *, refs=(), rows: int = 0) -> None:
    """Publishing is a separate step; CatalogIndexer does it, direct writes must.

    Membership rides on the publish now, so `refs` is what makes those packs
    visible; commit_packs only records the replay guard.
    """
    writer.publish_snapshot(
        index_version=index_version,
        refs=list(refs),
        published_at_ns=index_version,
        indexed_rows=rows,
        indexed_packs=len(refs),
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
        # Armed BEFORE ensure_schema: it issues many statements, and one that
        # fails partway leaves every table created ahead of it behind. Arming
        # afterwards skips teardown for exactly that case and the tables leak
        # onto the shared server. Every drop below is IF EXISTS, so tearing
        # down a partial or empty schema is safe.
        created = True
        writer.ensure_schema()
        yield writer, reader
    finally:
        if created:
            database = config.database
            for kind, suffix in (
                ("VIEW", "capture"),
                ("VIEW", "pack_inventory"),
                ("TABLE", "capture_raw"),
                ("TABLE", "pack_inventory_raw"),
                ("TABLE", "capture_version_claims"),
                ("TABLE", "index_watermark"),
                ("TABLE", "snapshot_manifest"),
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
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)

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
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)

        first = reader.search(CaptureQuery(limit=40))
        writer.write_descriptors(later, index_version=2)
        _publish(writer, 2, refs=_refs(later))
        _commit(writer, later, 2)

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
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)
        at_first = reader.get_by_ids(
            [corpus[0].capture_id],
            tenant_id=corpus[0].metadata.tenant_id,
            watermark="1",
        )

        writer.write_descriptors(corpus, index_version=2)
        _publish(writer, 2, refs=_refs(corpus))
        _commit(writer, corpus, 2)
        at_second = reader.get_by_ids(
            [corpus[0].capture_id],
            tenant_id=corpus[0].metadata.tenant_id,
            watermark=reader.current_watermark(),
        )

        assert at_first == at_second, "a replay changed a descriptor"


def test_replayed_indexing_yields_one_logical_row():
    corpus = synthetic_descriptors(10)
    with _catalog() as (writer, reader):
        for version in (1, 2, 3):
            writer.write_descriptors(corpus, index_version=version)
            _publish(writer, version, refs=_refs(corpus))
            _commit(writer, corpus, version)

        walked, _, _ = _walk(reader, CaptureQuery(limit=10))

        assert walked == corpus


def test_get_by_ids_resolves_a_selection_at_its_watermark():
    corpus = synthetic_descriptors(20)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)

        page = reader.search(CaptureQuery(limit=20))
        wanted = [item.capture_id for item in page.items[:5]]

        resolved = reader.get_by_ids(
            wanted,
            tenant_id=corpus[0].metadata.tenant_id,
            watermark=page.watermark,
        )

        assert {item.capture_id for item in resolved} == set(wanted)


def test_filters_narrow_the_walk():
    corpus = synthetic_descriptors(30)
    with _catalog() as (writer, reader):
        writer.write_descriptors(corpus, index_version=1)
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)

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
