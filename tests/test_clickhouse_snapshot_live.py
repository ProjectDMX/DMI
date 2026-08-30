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
from dmi.storage.capture.clickhouse_catalog import _CAPTURE_COLUMNS


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
                ("TABLE", "schema_version"),
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


def test_a_foreign_row_at_this_version_is_not_reported_as_a_publish():
    """"Does a row for V exist?" is not "is V mine?", over a live server.

    A row for the version arrives from somewhere this publisher does not
    control -- an operator's INSERT, a second build, a publisher whose
    statement overlapped this one. The conditional INSERT then writes nothing,
    because V is no longer strictly above the head, and a ``count()`` check sees
    the foreign row and calls that success: the caller is told it published a
    snapshot it did not publish, and `CatalogIndexer` records the packs in the
    replay inventory so no later pass ever indexes them again.

    Reading ``publish_id`` back turns that into the lost race it is. The
    manifest rows this attempt left behind are inert twice over -- the publish
    that wrote them never reached the watermark, and membership pairs the two.
    """
    corpus = synthetic_descriptors(3)
    tenant = corpus[0].metadata.tenant_id
    ids = [item.capture_id for item in corpus]
    foreign = str(uuid4())

    with _catalog() as (writer, reader, client, config):
        version = writer.allocate_version()
        writer.write_descriptors(corpus, index_version=version)

        # Somebody else's row lands at this publisher's version.
        client.execute(
            f"INSERT INTO `{config.database}`.`{config.table_prefix}_index_watermark` "
            "(index_version, publish_id, published_at_ns, indexed_rows, "
            "indexed_packs) VALUES",
            [(version, foreign, 1, 0, 0)],
        )

        with pytest.raises(SnapshotPublishRaceError):
            _publish(writer, version, refs=_refs(corpus), rows=len(corpus))

        # The foreign row is still the only one at V: nothing this publisher
        # wrote is visible under it.
        assert client.execute(
            "SELECT toString(publish_id) FROM "
            f"`{config.database}`.`{config.table_prefix}_index_watermark` "
            "WHERE index_version = %(version)s",
            {"version": version},
        ) == [(foreign,)]
        current = reader.current_watermark()
        assert int(current) == version
        assert reader.get_by_ids(ids, tenant_id=tenant, watermark=current) == (), (
            "manifest rows entered a snapshot under a watermark row written by "
            "someone else"
        )

        # And retrying above it publishes normally.
        retry = writer.allocate_version()
        _publish(writer, retry, refs=_refs(corpus), rows=len(corpus))
        fresh = reader.current_watermark()
        assert {
            item.capture_id
            for item in reader.get_by_ids(ids, tenant_id=tenant, watermark=fresh)
        } == set(ids)


def test_the_deciding_reads_carry_sequential_consistency_to_the_server():
    """The setting is applied, and the server accepts it on these tables.

    Asserting the kwarg alone would pass against a server that rejects the
    setting; asserting only that the suite runs would pass against code that
    never sets it. This does both at once: a proxy records what rides on each
    statement while a real ClickHouse executes it.
    """
    corpus = synthetic_descriptors(2)

    with _catalog() as (writer, reader, client, config):
        recorded: list[tuple[str, object]] = []

        class _Recording:
            def execute(self, query, params=None, **kwargs):
                recorded.append((query, kwargs.get("settings")))
                return client.execute(query, params, **kwargs)

        taps = ClickHouseCatalogWriter(_Recording(), config)
        version = taps.allocate_version()
        taps.write_descriptors(corpus, index_version=version)
        _publish(taps, version, refs=_refs(corpus), rows=len(corpus))
        taps.commit_packs(_refs(corpus), index_version=version)

        def _settings(fragment: str, kind: str) -> list:
            return [
                settings
                for query, settings in recorded
                if fragment in query and query.lstrip().startswith(kind)
            ]

        consistent = {"select_sequential_consistency": 1}
        assert _settings("toString(claim_id)", "SELECT") == [consistent]
        assert _settings("max(version)", "SELECT") == [consistent]
        assert _settings("max(index_version)", "SELECT") == [consistent]
        assert _settings("index_watermark", "INSERT") == [consistent]
        assert _settings("toString(publish_id)", "SELECT") == [consistent]
        # Bulk writes decide nothing and pay nothing.
        assert _settings("capture_raw", "INSERT") == [None]
        assert _settings("pack_inventory_raw", "INSERT") == [None]
        # And the batch really is readable, so the settings did not merely fail
        # quietly on the way.
        assert len(reader.search(CaptureQuery(limit=10)).items) == len(corpus)


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


# --- the public capture view -------------------------------------------------
#
# `{prefix}_capture` is what anything querying the catalog by hand reads --
# dashboards, ad-hoc SQL, the operator debugging an indexing pass. It is not on
# the reader's path, so nothing else in this suite would notice it disagreeing
# with the reader about what exists.


_PUBLIC_COLUMNS = ", ".join(_CAPTURE_COLUMNS[:-1])


def _view_rows(client, config):
    return client.execute(
        f"SELECT {_PUBLIC_COLUMNS} FROM "
        f"`{config.database}`.`{config.table_prefix}_capture` "
        "ORDER BY capture_id, store_id, pack_id"
    )


def _identities(client, config, table: str):
    """(capture_id, store_id, pack_id) as `table` reports it."""
    return sorted(
        client.execute(
            f"SELECT capture_id, store_id, toString(pack_id) FROM "
            f"`{config.database}`.`{config.table_prefix}_{table}`"
        )
    )


def test_the_public_view_hides_a_batch_until_it_is_published():
    """Written is not published, and the view has to tell the difference.

    Descriptor rows become durable several INSERTs before the publish that
    admits them, and a pass that crashes in between leaves them durable
    forever. `FINAL` alone applies no membership, so the view showed both --
    rows a reader correctly reports as nonexistent, offered to anyone querying
    the catalog by hand as though they were catalog contents.
    """
    corpus = synthetic_descriptors(3)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(corpus, index_version=1)

        # Durable in the raw table, and that is the whole point: the view is
        # not hiding missing rows, it is declining to publish unpublished ones.
        raw = _identities(client, config, "capture_raw")
        assert len(raw) == len(corpus)
        assert _view_rows(client, config) == []
        assert reader.current_watermark() == "0"

        _publish(writer, 1, refs=_refs(corpus), rows=len(corpus))

        assert _identities(client, config, "capture") == raw


def test_the_public_view_keeps_both_packs_describing_one_capture():
    """One row per (capture, store, pack) -- deliberately not one per capture.

    The view's meaning is "published descriptor rows, engine-deduplicated". A
    capture described by two packs is two published rows and appears twice;
    deciding which one wins is supersession, and that lives in the reader
    (argMax over index_version, grouped on capture identity). Grouping here
    too would be a second copy of those semantics in SQL, free to drift. This
    pins the meaning so a later change to it has to be deliberate.
    """
    original = synthetic_descriptors(3)
    retried = _retried_into_a_new_pack(original)
    with _catalog() as (writer, reader, client, config):
        writer.write_descriptors(original, index_version=1)
        _publish(writer, 1, refs=_refs(original), rows=len(original))
        writer.write_descriptors(retried, index_version=2)
        _publish(writer, 2, refs=_refs(retried), rows=len(retried))

        rows = _identities(client, config, "capture")

        assert len(rows) == len(original) + len(retried)
        for item in original:
            assert sum(1 for row in rows if row[0] == item.capture_id) == 2
        # And the reader, over the same data, still resolves one row per
        # capture: the two answers differ because they answer different
        # questions.
        page = reader.search(
            CaptureQuery(limit=10, tenant_id=original[0].metadata.tenant_id)
        )
        assert len(page.items) == len(original)


def test_the_public_view_of_an_empty_catalog_is_empty():
    """max() over an empty UInt64 watermark is 0, and versions start at 1.

    So the bound admits nothing rather than everything -- the failure mode a
    NULL or a missing row would produce.
    """
    with _catalog() as (_, _, client, config):
        assert client.execute(
            "SELECT max(index_version) FROM "
            f"`{config.database}`.`{config.table_prefix}_index_watermark`"
        ) == [(0,)]
        assert _view_rows(client, config) == []


def test_the_public_view_is_the_raw_rows_of_published_packs():
    """The bound is exactly membership: no row more, no row fewer."""
    published = synthetic_descriptors(4)
    unpublished = _retried_into_a_new_pack(published)
    with _catalog() as (writer, _, client, config):
        writer.write_descriptors(published, index_version=1)
        _publish(writer, 1, refs=_refs(published), rows=len(published))
        _commit(writer, published, 1)
        # Written at a version that never publishes: durable, never a member.
        writer.write_descriptors(unpublished, index_version=2)
        _merge(client, config)

        identities = [
            (ref.store_id, ref.pack_id) for ref in _refs(published)
        ]
        expected = client.execute(
            f"SELECT {_PUBLIC_COLUMNS} FROM "
            f"`{config.database}`.`{config.table_prefix}_capture_raw` FINAL "
            "WHERE (store_id, pack_id) IN %(identities)s "
            "ORDER BY capture_id, store_id, pack_id",
            {"identities": identities},
        )

        assert len(expected) == len(published)
        assert _view_rows(client, config) == expected
        assert _raw_rows(client, config) == len(published) + len(unpublished)


# --- resolving two packs written at one version ------------------------------


def _mirrored_into_a_second_store(descriptors):
    """The same captures, in a second pack, with a wholly different locator.

    Every locator field moves, so a projection that resolved its columns from
    different rows would show up in any of them rather than in one lucky column.
    """
    pack_id = str(uuid4())
    return tuple(
        replace(
            item,
            locator=replace(
                item.locator,
                store_id="second-store",
                pack_id=pack_id,
                object_key=f"mirror/{pack_id}.dmi-pack",
                object_bytes=item.locator.object_bytes + 4_096,
                pack_checksum="b" * 64,
                pack_record_count=item.locator.pack_record_count + 1,
                offset=item.locator.offset + 1_048_576,
                stored_length=item.locator.stored_length + 32,
                decoded_length=item.locator.decoded_length + 32,
                codec="zstd",
                checksum="ffffffff",
            ),
        )
        for item in descriptors
    )


def test_two_packs_written_at_one_version_resolve_to_one_pack_stably():
    """The tie ``index_version`` alone cannot break, over a live server.

    ``CatalogIndexer.index`` writes every pack of a batch at ONE version, so two
    packs describing the same capture in one call produce rows whose
    ``index_version`` is EQUAL -- reproduced here by writing both at version 1,
    which is the same physical state without needing two real pack objects. The
    projection resolves each column with its own ``argMax``, and ClickHouse does
    not say which row an ``argMax`` picks out of a tie, so two things can go
    wrong: the columns can come from different rows, and the row they come from
    can change between reads.

    Both are asserted, because only the second is reachable on this engine.
    ClickHouse 25.12 does break every tie in a query the same way -- the columns
    agree -- but *which* row wins moves with the physical layout: with the two
    packs in separate parts the later-inserted one won, and after a merge put
    both rows in one part the lower sort key did. A background merge does that
    at a time nobody controls, so pre-fix this is a pinned selection that
    resolves to one pack now and the other pack later. The coherence assertion
    is kept as a guard: it pins behaviour the engine does not promise, and an
    upgrade that starts resolving argMax per column is free to break it.
    """
    original = synthetic_descriptors(40)
    mirror = _mirrored_into_a_second_store(original)
    locators = {
        item.capture_id: {original[index].locator, mirror[index].locator}
        for index, item in enumerate(original)
    }
    tenant = original[0].metadata.tenant_id
    ids = [item.capture_id for item in original]

    with _catalog() as (writer, reader, client, config):
        # One version for both packs, one publish: exactly what index() emits.
        writer.write_descriptors(original, index_version=1)
        writer.write_descriptors(mirror, index_version=1)
        _publish(
            writer,
            1,
            refs=_refs(original) + _refs(mirror),
            rows=len(original) + len(mirror),
        )
        _commit(writer, original + mirror, 1)
        assert _raw_rows(client, config) == len(original) + len(mirror)

        def resolved():
            by_id = reader.get_by_ids(ids, tenant_id=tenant, watermark="1")
            assert len(by_id) == len(original), "a capture split across two rows"
            page = reader.search(CaptureQuery(limit=100, tenant_id=tenant))
            assert [item.capture_id for item in page.items] == ids
            return {item.capture_id: item.locator for item in by_id}, {
                item.capture_id: item.locator for item in page.items
            }

        def coherent(mapping):
            for capture_id, locator in mapping.items():
                assert locator in locators[capture_id], (
                    f"{capture_id} resolved to a locator belonging to neither "
                    f"pack -- columns came from different rows: {locator}"
                )

        first_lookup, first_page = resolved()
        coherent(first_lookup)
        coherent(first_page)
        assert first_lookup == first_page, (
            "a lookup and a page resolved the same capture to different packs"
        )
        for _ in range(3):
            again_lookup, again_page = resolved()
            assert (again_lookup, again_page) == (first_lookup, first_page), (
                "repeating the query at one watermark moved the answer"
            )

        # A merge does not delete either row -- pack identity is in the sort key
        # -- but it does put both in one part, which is enough to change which
        # one an untied argMax picks.
        _merge(client, config)
        assert _raw_rows(client, config) == len(original) + len(mirror)

        merged_lookup, merged_page = resolved()
        coherent(merged_lookup)
        coherent(merged_page)
        assert merged_lookup == first_lookup, (
            "a merge changed which pack a pinned selection resolves to"
        )
        assert merged_page == first_page, (
            "a merge changed which pack a pinned page resolves to"
        )
