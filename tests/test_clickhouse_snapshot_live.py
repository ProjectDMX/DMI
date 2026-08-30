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
from dataclasses import replace
from uuid import uuid4

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureQuery,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
    SnapshotPublishRaceError,
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
        # Armed BEFORE ensure_schema, not after: it issues many statements, so
        # one that fails partway leaves every table created ahead of it behind.
        # Arming afterwards skips teardown for exactly that case and the tables
        # leak onto the shared server -- one mutation run left 39 orphaned
        # `*_capture_raw` / `*_pack_inventory_raw` tables that way. Every drop
        # below is IF EXISTS, so tearing down a partial or empty schema is safe.
        created = True
        writer.ensure_schema()
        yield writer, reader, client, config
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


def test_a_merge_does_not_destroy_a_pinned_snapshot():
    """A watermark must keep resolving after ClickHouse deduplicates.

    The snapshot is the set of packs committed at or before the watermark, so
    it survives a merge collapsing duplicate descriptor rows. Replay writes
    byte-identical descriptors, which is why it does not matter which copy the
    merge keeps.
    """
    corpus = synthetic_descriptors(3)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(corpus, index_version=1)
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)

        # Replay the identical batch, as an ambiguous commit would.
        writer.write_descriptors(corpus, index_version=2)
        _publish(writer, 2, refs=_refs(corpus))
        _commit(writer, corpus, 2)

        tenant = corpus[0].metadata.tenant_id
        before = reader.get_by_ids(
            [corpus[0].capture_id], tenant_id=tenant, watermark="1"
        )
        assert len(before) == 1, "precondition: the pinned read works pre-merge"

        _merge(client, config)

        after = reader.get_by_ids(
            [corpus[0].capture_id], tenant_id=tenant, watermark="1"
        )
        assert len(after) == 1, (
            "a pinned watermark stopped resolving after a merge: "
            f"raw rows went to {_raw_rows(client, config)}"
        )
        assert after[0] == before[0], "the resolved descriptor changed under a merge"


def test_a_pinned_snapshot_does_not_gain_a_slower_indexers_packs():
    """The mid-batch gap, over a live server.

    Indexer A allocates a version and spends time writing descriptors. Indexer
    B allocates a higher one and publishes. A reader pins B. A then publishes.

    Under the old design A's membership rows were written before its watermark
    and admitted by ``index_version <= W``, so they appeared inside the already
    pinned snapshot. Now A's publish loses the barrier, never writes a
    watermark row, and its manifest rows stay inert -- membership requires
    both. A retries above B and becomes visible only at a fresh watermark.
    """
    early = synthetic_descriptors(2)
    late_pack = str(uuid4())
    late = tuple(
        replace(
            item,
            metadata=replace(item.metadata, capture_id=f"late-{index}"),
            locator=replace(item.locator, pack_id=late_pack),
        )
        for index, item in enumerate(synthetic_descriptors(2))
    )
    tenant = early[0].metadata.tenant_id
    late_ids = [item.capture_id for item in late]

    with _catalog() as (writer, reader, _, _):
        # A allocates first and is still writing when B publishes.
        version_a = writer.allocate_version()
        writer.write_descriptors(late, index_version=version_a)

        version_b = writer.allocate_version()
        assert version_b > version_a
        writer.write_descriptors(early, index_version=version_b)
        _publish(writer, version_b, refs=_refs(early), rows=len(early))

        pinned = reader.current_watermark()
        assert reader.get_by_ids(late_ids, tenant_id=tenant, watermark=pinned) == ()

        # A publishes late, and loses.
        with pytest.raises(SnapshotPublishRaceError):
            _publish(writer, version_a, refs=_refs(late), rows=len(late))

        assert reader.get_by_ids(late_ids, tenant_id=tenant, watermark=pinned) == (), (
            "the snapshot pinned at the published watermark grew after the fact"
        )
        assert reader.current_watermark() == pinned, (
            "a losing publish moved the watermark"
        )

        # Retrying above the winner is what makes A's packs visible, and only
        # at a watermark taken after that.
        version_retry = writer.allocate_version()
        assert version_retry > version_b
        _publish(writer, version_retry, refs=_refs(late), rows=len(late))

        fresh = reader.current_watermark()
        assert int(fresh) > int(pinned)
        assert {
            item.capture_id
            for item in reader.get_by_ids(late_ids, tenant_id=tenant, watermark=fresh)
        } == set(late_ids)
        assert reader.get_by_ids(late_ids, tenant_id=tenant, watermark=pinned) == ()


def _copied_to_another_store(descriptors):
    """An operator copies the pack object into a second bucket, then reconciles.

    Replay dedup keys on (store_id, pack_id), so a new store is not a replay:
    the indexer writes a fresh descriptor row for every capture in the pack.
    Same pack id, same captures, different store and object key. Reachable
    today with shipped components.
    """
    return tuple(
        replace(
            item,
            locator=replace(
                item.locator,
                store_id="second-store",
                object_key="mirror/" + item.locator.object_key,
            ),
        )
        for item in descriptors
    )


def _retried_into_a_new_pack(descriptors):
    """A producer retries a capture_id after the first pack was sealed.

    Within one open pack a repeat raises DuplicateCaptureError, but a sealed
    pack is forgotten, so retry-after-ambiguity lands the same capture in a
    second pack with a different offset.
    """
    pack_id = str(uuid4())
    return tuple(
        replace(
            item,
            locator=replace(
                item.locator,
                pack_id=pack_id,
                object_key=f"packs/{pack_id}.dmi-pack",
                offset=item.locator.offset + 1_048_576,
            ),
        )
        for item in descriptors
    )


@pytest.mark.parametrize(
    "second_description", (_copied_to_another_store, _retried_into_a_new_pack)
)
def test_two_packs_describing_one_capture_both_survive_a_merge(second_description):
    """The reason pack identity is in the sort key.

    ReplacingMergeTree deletes rows sharing a sort key. On capture identity
    alone these two rows share one, so a merge would keep the higher version
    and silently delete the other -- and a snapshot pinned to the deleted row's
    pack then fails with "selection no longer resolves". A forced OPTIMIZE is
    the honest test: a background merge does the same at a time nobody
    controls, so a read-immediately test could never see it.
    """
    original = synthetic_descriptors(3)
    copied = second_description(original)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(original, index_version=1)
        _publish(writer, 1, refs=_refs(original))
        _commit(writer, original, 1)
        pinned = reader.current_watermark()

        writer.write_descriptors(copied, index_version=2)
        _publish(writer, 2, refs=_refs(copied))
        _commit(writer, copied, 2)

        _merge(client, config)

        assert _raw_rows(client, config) == len(original) + len(copied), (
            "a merge collapsed two packs' descriptions of one capture; the "
            "pinned snapshot over the losing pack can no longer resolve"
        )

        # And the pin still resolves to the pack it was taken over, while a
        # fresh watermark sees the newer pack win.
        tenant = original[0].metadata.tenant_id
        capture_id = original[0].capture_id
        at_pin = reader.get_by_ids([capture_id], tenant_id=tenant, watermark=pinned)
        assert len(at_pin) == 1
        assert at_pin[0].locator == original[0].locator

        at_fresh = reader.get_by_ids(
            [capture_id], tenant_id=tenant, watermark=reader.current_watermark()
        )
        assert len(at_fresh) == 1, "supersession must still yield one row"
        assert at_fresh[0].locator == copied[0].locator


def test_a_pin_ignores_a_second_store_holding_the_same_pack_id():
    """Pack identity is the PAIR, proven by behaviour rather than by SQL text.

    The unit tests assert the clause reads ``(store_id, pack_id) IN (SELECT
    store_id, pack_id ...)``, which a rewrite could satisfy in text while
    matching on ``pack_id`` alone. Only a server evaluating the predicate can
    tell the difference, and it matters exactly here: the mirror carries the
    SAME pack id as the original, committed at a later version. Matching the
    id alone would admit the mirror's rows into a snapshot pinned before it
    existed, and argMax would then resolve every capture to the mirror's store.

    Both membership sites are covered: get_by_ids pins the watermark directly,
    and search pins it through a cursor.
    """
    original = synthetic_descriptors(3)
    mirror = _copied_to_another_store(original)
    assert mirror[0].locator.pack_id == original[0].locator.pack_id
    assert mirror[0].locator.store_id != original[0].locator.store_id

    tenant = original[0].metadata.tenant_id
    with _catalog() as (writer, reader, _, _):
        writer.write_descriptors(original, index_version=1)
        _publish(writer, 1, refs=_refs(original))
        _commit(writer, original, 1)
        pinned = reader.current_watermark()
        # A cursor carries the same pin through the paginated path.
        first = reader.search(CaptureQuery(limit=2, tenant_id=tenant))
        assert first.next_cursor is not None

        writer.write_descriptors(mirror, index_version=2)
        _publish(writer, 2, refs=_refs(mirror))
        _commit(writer, mirror, 2)

        # (a) get_by_ids at the pin resolves to the store that was committed.
        at_pin = reader.get_by_ids(
            [original[0].capture_id], tenant_id=tenant, watermark=pinned
        )
        assert len(at_pin) == 1
        assert at_pin[0].locator == original[0].locator

        # (b) the pinned walk continues over the original store only.
        rest = reader.search(
            CaptureQuery(limit=2, tenant_id=tenant, cursor=first.next_cursor)
        )
        assert first.items + rest.items == original

        # And a fresh watermark does see the mirror supersede it.
        at_fresh = reader.get_by_ids(
            [original[0].capture_id],
            tenant_id=tenant,
            watermark=reader.current_watermark(),
        )
        assert at_fresh[0].locator == mirror[0].locator


def test_re_indexing_one_pack_still_collapses_to_a_single_row():
    """The case ReplacingMergeTree is actually here for.

    Rows for one capture in the SAME pack share the whole sort key, pack
    identity included, and are byte identical -- so a merge is free to keep
    either. Losing that would make every replay grow the table forever.
    """
    corpus = synthetic_descriptors(4)
    with _catalog() as (writer, reader, client, config):
        for version in (1, 2, 3):
            writer.write_descriptors(corpus, index_version=version)
            _publish(writer, version, refs=_refs(corpus))
            _commit(writer, corpus, version)
        assert _raw_rows(client, config) == 3 * len(corpus)

        _merge(client, config)

        assert _raw_rows(client, config) == len(corpus), (
            "replayed descriptor rows stopped collapsing"
        )
        page = reader.search(
            CaptureQuery(limit=10, tenant_id=corpus[0].metadata.tenant_id)
        )
        assert page.items == corpus


def test_a_pack_committed_after_the_watermark_is_not_visible():
    """The snapshot boundary has to actually exclude later work."""
    early = synthetic_descriptors(2)
    # A genuinely later pack: its own pack id, and its own captures. A capture
    # id belongs to exactly one pack, so reusing ids across packs would be an
    # invalid corpus rather than a harder test.
    late_pack = str(uuid4())
    later = tuple(
        replace(
            item,
            metadata=replace(item.metadata, capture_id=f"late-{index}"),
            locator=replace(item.locator, pack_id=late_pack),
        )
        for index, item in enumerate(synthetic_descriptors(2))
    )
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(early, index_version=1)
        _publish(writer, 1, refs=_refs(early))
        _commit(writer, early, 1)
        pinned = reader.current_watermark()

        writer.write_descriptors(later, index_version=2)
        _publish(writer, 2, refs=_refs(later))
        _commit(writer, later, 2)
        _merge(client, config)

        # The pinned snapshot sees the early pack and nothing after it.
        tenant = early[0].metadata.tenant_id
        assert (
            len(
                reader.get_by_ids(
                    [early[0].capture_id], tenant_id=tenant, watermark=pinned
                )
            )
            == 1
        )
        assert (
            reader.get_by_ids(
                [later[0].capture_id], tenant_id=tenant, watermark=pinned
            )
            == ()
        )

        # And the later watermark sees both.
        current = reader.current_watermark()
        assert (
            len(
                reader.get_by_ids(
                    [later[0].capture_id], tenant_id=tenant, watermark=current
                )
            )
            == 1
        )


def test_a_walk_still_completes_after_a_merge_mid_pagination():
    """Merges run concurrently with reads; a walk must not lose rows to one."""
    corpus = synthetic_descriptors(60)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(corpus, index_version=1)
        _publish(writer, 1, refs=_refs(corpus))
        _commit(writer, corpus, 1)
        # Re-index everything, so every row has a superseded version to lose.
        writer.write_descriptors(corpus, index_version=2)
        _publish(writer, 2, refs=_refs(corpus))
        _commit(writer, corpus, 2)

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
        _publish(writer, 123, refs=_refs(corpus), rows=len(corpus))
        _commit(writer, corpus, 123)

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
        _publish(writer, 9, refs=_refs(corpus), rows=len(corpus))
        _commit(writer, corpus, 9)

        page = reader.search(CaptureQuery(limit=100))

        assert len(page.items) == len(corpus)
        assert page.watermark == "9"


# --- ordering and boundary conditions ----------------------------------------


def test_a_watermark_below_every_row_returns_nothing():
    corpus = synthetic_descriptors(5)
    with _catalog() as (writer, reader, _, _):
        writer.write_descriptors(corpus, index_version=10)
        _publish(writer, 10, refs=_refs(corpus))
        _commit(writer, corpus, 10)

        assert (
            reader.get_by_ids(
                [corpus[0].capture_id],
                tenant_id=corpus[0].metadata.tenant_id,
                watermark="9",
            )
            == ()
        )


def test_an_empty_catalog_reports_a_zero_watermark():
    with _catalog() as (_, reader, _, _):
        assert reader.current_watermark() == "0"
        assert reader.search(CaptureQuery(limit=10)).items == ()
