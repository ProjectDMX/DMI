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
from time import sleep
from uuid import uuid4

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureQuery,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
    PublisherLeaseError,
    PublisherLeaseHeldError,
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


def _client():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    return clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )


@contextmanager
def _catalog(**overrides):
    client = _client()
    prefix = f"dmi_snapshot_test_{uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
        **overrides,
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
        # Only the lease holder can make a snapshot visible, and the check
        # rides inside the publish statement, so a fixture that skipped this
        # would write nothing and every assertion here would be about an empty
        # catalog.
        writer.acquire_publisher_lease("snapshot-suite")
        yield writer, reader, client, config
    finally:
        if created:
            # The writer owns the object list, in drop order. Hand-copied
            # lists in these fixtures drifted from it and leaked tables.
            writer.drop_schema()


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


# --- the publisher lease, over a live server ---------------------------------
#
# The in-memory fake in tests/test_capture_version_allocation.py serialises two
# publishers in the interpreter, so it can drive the PROTOCOL but never the
# window the fence exists for. These do, against a real ClickHouse: the
# takeover lands inside the window between one publisher renewing its lease and
# the server executing its write, which is the only place a takeover can slip
# past the client-side check.


def _lease_rows(client, config):
    return client.execute(
        "SELECT term, toString(lease_id), holder FROM "
        f"`{config.database}`.`{config.table_prefix}_publisher_lease` "
        "ORDER BY term, lease_id"
    )


def _catalog_state(client, config):
    """Everything a publish would change, as one comparable value."""
    return {
        table: client.execute(
            f"SELECT * FROM `{config.database}`.`{config.table_prefix}_{table}` "
            "ORDER BY ALL"
        )
        for table in ("index_watermark", "snapshot_manifest")
    }


def test_a_taken_over_publisher_writes_nothing_at_all():
    """The load-bearing one: the fence rides inside the write.

    Publisher A holds the lease and renews it, so its client-side check passes.
    B then takes it over -- legitimately, once A's short lease has expired --
    in the window between that renewal and the statements A is about to issue.
    A's manifest INSERT and its watermark INSERT both carry the fence, so the
    server evaluates it and A writes NOTHING: not an inert manifest row, not a
    watermark row, nothing. That is what separates this from the two designs
    this branch has already rejected, where the loser wrote first and
    discovered afterwards.

    "Nothing" is exact for THIS window -- the takeover lands before A's first
    statement, so the fence rejects both. It is not a property of a publish in
    general: the two statements are fenced separately, and
    `test_a_takeover_between_the_two_publish_statements_leaves_orphan_rows`
    drives the gap between them, where the first has already landed.

    Asserted directly against the tables rather than through the exception,
    because "it raised" is exactly what the rejected designs also did.
    """
    corpus = synthetic_descriptors(3)
    tenant = corpus[0].metadata.tenant_id
    later = _retried_into_a_new_pack(corpus)

    # A short lease so the takeover is a real expiry rather than a forced one,
    # and a publish cap comfortably above a one-row INSERT -- the shortest the
    # whole-second publish cap admits. The one wall clock left in this test
    # runs the safe way: the wedge sleeps 2.5 s against a 2 s TTL, and load
    # only makes the expiry more certain, never less.
    with _catalog(
        lease_ttl_ns=2_000_000_000, publish_timeout_ns=1_000_000_000
    ) as (writer, reader, client, config):
        version = writer.allocate_version()
        writer.write_descriptors(corpus, index_version=version)
        _publish(writer, version, refs=_refs(corpus), rows=len(corpus))
        pinned = reader.current_watermark()
        before_state = _catalog_state(client, config)
        before_page = reader.search(CaptureQuery(limit=100, tenant_id=tenant))
        assert len(before_page.items) == len(corpus)

        writer.release_publisher_lease()
        # The successor runs on the DEFAULT knobs. Only the stalled publisher's
        # lease has to lapse, and its config is what decides that; giving the
        # successor a 200 ms lease and a 100 ms publish cap would have put the
        # closing publish below on a deadline for no reason.
        default = ClickHouseCatalogConfig()
        successor = ClickHouseCatalogWriter(
            _client(),
            replace(
                config,
                lease_ttl_ns=default.lease_ttl_ns,
                publish_timeout_ns=default.publish_timeout_ns,
            ),
        )
        armed = []

        class _TakeOverAfterTheRenewal:
            """A's client, with B's takeover wedged into the one open window."""

            def execute(self, query, params=None, **kwargs):
                rows = client.execute(query, params, **kwargs)
                if armed and "acquired_at_ns, expires_at_ns" in query:
                    armed.clear()
                    # A's renewal has just succeeded by its own reckoning. Wait
                    # out the short lease and let B take it, exactly as a
                    # successor would.
                    sleep(2.5)
                    successor.acquire_publisher_lease("successor")
                return rows

        stalled = ClickHouseCatalogWriter(_TakeOverAfterTheRenewal(), config)
        stalled.acquire_publisher_lease("stalled")
        lost = stalled.allocate_version()
        stalled.write_descriptors(later, index_version=lost)

        armed.append(True)
        with pytest.raises(PublisherLeaseError) as raised:
            _publish(stalled, lost, refs=_refs(later), rows=len(later))

        assert not armed, "the takeover never ran; the test proved nothing"
        assert "fenced out and made no snapshot visible" in str(raised.value)
        assert "'successor'" in str(raised.value)

        # Nothing landed. Not a watermark row, and not one inert manifest row.
        assert _catalog_state(client, config) == before_state
        assert not any(
            row[0] == lost
            for row in client.execute(
                "SELECT index_version FROM "
                f"`{config.database}`.`{config.table_prefix}_snapshot_manifest`"
            )
        )
        # The descriptors it wrote ARE durable, which is the point: they are
        # invisible because nothing admitted them, not because they are gone.
        assert client.execute(
            f"SELECT count() FROM `{config.database}`.`{config.table_prefix}"
            "_capture_raw`"
        ) == [(len(corpus) + len(later),)]

        # And the reader pinned before any of this sees exactly what it saw.
        assert reader.current_watermark() == pinned
        assert reader.search(
            CaptureQuery(limit=100, tenant_id=tenant, cursor=None)
        ).items == before_page.items
        assert reader.get_by_ids(
            [item.capture_id for item in corpus], tenant_id=tenant, watermark=pinned
        ) == reader.get_by_ids(
            [item.capture_id for item in corpus], tenant_id=tenant, watermark=pinned
        )

        # The successor publishes normally over the same catalog.
        successor_version = successor.allocate_version()
        _publish(
            successor, successor_version, refs=_refs(later), rows=len(later)
        )
        assert int(reader.current_watermark()) == successor_version
        successor.release_publisher_lease()


def test_a_takeover_between_the_two_publish_statements_leaves_orphan_rows():
    """The guarantee is per STATEMENT, not per publish, and this is the gap.

    `publish_snapshot` issues two separately fenced statements: the manifest
    rows, then the watermark row that admits them. Both carry the fence, so
    neither can land after a takeover -- but the gap between them is a full
    client round trip, which `max_execution_time` does not bound because it
    caps each statement rather than the pair.

    A takeover wedged there leaves the manifest rows of the first statement
    behind while the second is refused. Those rows are INERT -- membership
    pairs them with a watermark row that will never exist, so no snapshot can
    admit them and no reader can see them -- but they are durable, and they
    are what the "a fenced-out publisher writes nothing" claim gets wrong.
    That is the difference between the safety property, which holds, and the
    stronger property the documents used to claim.
    """
    corpus = synthetic_descriptors(3)
    tenant = corpus[0].metadata.tenant_id
    later = _retried_into_a_new_pack(corpus)

    with _catalog(
        lease_ttl_ns=2_000_000_000, publish_timeout_ns=1_000_000_000
    ) as (writer, reader, client, config):
        version = writer.allocate_version()
        writer.write_descriptors(corpus, index_version=version)
        _publish(writer, version, refs=_refs(corpus), rows=len(corpus))
        pinned = reader.current_watermark()
        before_page = reader.search(CaptureQuery(limit=100, tenant_id=tenant))

        writer.release_publisher_lease()
        default = ClickHouseCatalogConfig()
        successor = ClickHouseCatalogWriter(
            _client(),
            replace(
                config,
                lease_ttl_ns=default.lease_ttl_ns,
                publish_timeout_ns=default.publish_timeout_ns,
            ),
        )
        armed = []

        class _TakeOverBetweenTheTwoStatements:
            """A's client, with B's takeover wedged AFTER the manifest INSERT."""

            def execute(self, query, params=None, **kwargs):
                rows = client.execute(query, params, **kwargs)
                if armed and query.lstrip().startswith(
                    "INSERT INTO "
                    f"`{config.database}`.`{config.table_prefix}_snapshot_manifest`"
                ):
                    armed.clear()
                    # A's manifest rows are durable. Its lease lapses here and
                    # B takes over before A issues its watermark INSERT.
                    sleep(2.5)
                    successor.acquire_publisher_lease("successor")
                return rows

        stalled = ClickHouseCatalogWriter(_TakeOverBetweenTheTwoStatements(), config)
        stalled.acquire_publisher_lease("stalled")
        lost = stalled.allocate_version()
        stalled.write_descriptors(later, index_version=lost)

        armed.append(True)
        with pytest.raises(PublisherLeaseError) as raised:
            _publish(stalled, lost, refs=_refs(later), rows=len(later))

        assert not armed, "the takeover never ran; the test proved nothing"
        assert "fenced out" in str(raised.value)

        manifest = f"`{config.database}`.`{config.table_prefix}_snapshot_manifest`"
        orphans = client.execute(
            f"SELECT count() FROM {manifest} WHERE index_version = %(v)s",
            {"v": lost},
        )
        # The rows the first statement wrote ARE there. This is the assertion
        # that separates "wrote nothing" from what actually happens.
        assert orphans == [(len(_refs(later)),)], (
            "precondition: the manifest INSERT must have landed before the "
            "takeover, or this test is exercising the other window"
        )
        # And they are inert: no watermark row carries that version, so no
        # snapshot admits them and every reader is unmoved.
        assert client.execute(
            "SELECT count() FROM "
            f"`{config.database}`.`{config.table_prefix}_index_watermark` "
            "WHERE index_version = %(v)s",
            {"v": lost},
        ) == [(0,)]
        assert reader.current_watermark() == pinned
        assert reader.search(
            CaptureQuery(limit=100, tenant_id=tenant, cursor=None)
        ).items == before_page.items
        # The public view agrees with the reader: unpaired membership admits
        # nothing.
        assert client.execute(
            "SELECT count() FROM "
            f"`{config.database}`.`{config.table_prefix}_capture`"
        ) == [(len(corpus),)]
        successor.release_publisher_lease()


def test_a_contested_head_term_is_made_safe_by_the_read_back_not_the_fence():
    """Where the safety of a contested term actually comes from.

    Both documents said "a contested head term satisfies neither condition" of
    the fence. It is not true and the difference matters to anyone reasoning
    from it: the fence resolves ONE row with `ORDER BY term DESC, lease_id
    DESC`, so a contested term's higher-ordering claimant satisfies it exactly
    as an uncontested holder would.

    What makes the term safe is `_claim_lease`: both claimants see two rows in
    their read-back, both abandon the term, and neither is ever handed a
    `PublisherLease`. Nobody is holding the `lease_id` the fence would accept.
    Asserted on the server, because the ordering here is ClickHouse's UUID
    collation -- the low half first -- and not text order.
    """
    with _catalog() as (writer, reader, client, config):
        writer.release_publisher_lease()
        lease_table = f"`{config.database}`.`{config.table_prefix}_publisher_lease`"
        client.execute(f"TRUNCATE TABLE {lease_table}")
        # Two live claimants at one term, and nothing above it: the shape a
        # contested head has before anybody claims higher. Chosen so that
        # ClickHouse's UUID order and Python's text order disagree, which is
        # the trap a fake resolving this in Python would fall into.
        low_text_high_uuid = "00000000-0000-0000-ffff-ffffffffffff"
        high_text_low_uuid = "ffffffff-ffff-ffff-0000-000000000000"
        for lease_id in (low_text_high_uuid, high_text_low_uuid):
            client.execute(
                f"INSERT INTO {lease_table} (term, lease_id, holder, "
                "acquired_at_ns, expires_at_ns) SELECT toUInt64(7), "
                "toUUID(%(lease_id)s), 'claimant', now_ns, "
                "now_ns + toUInt64(600000000000) FROM "
                "(SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
                {"lease_id": lease_id},
            )

        fence = writer._lease_fence()
        # The fence admits one of them, which is the claim both documents got
        # wrong. It is the UUID-order winner, not the text-order winner.
        assert client.execute(
            f"SELECT {fence}", {"lease_id": low_text_high_uuid}
        ) == [(1,)]
        assert client.execute(
            f"SELECT {fence}", {"lease_id": high_text_low_uuid}
        ) == [(0,)]

        # And it is safe anyway, because no protocol run ever returns that
        # lease: a claimant that sees the contest abandons the term and claims
        # above it, so the id the fence would accept is one nobody holds.
        successor = ClickHouseCatalogWriter(_client(), config)
        lease = successor.acquire_publisher_lease("successor")
        assert lease.term > 7 and lease.lease_id not in (
            low_text_high_uuid, high_text_low_uuid
        )
        version = successor.allocate_version()
        _publish(successor, version, refs=(), rows=0)
        assert int(reader.current_watermark()) == version
        successor.release_publisher_lease()


def test_a_contested_lease_term_is_held_by_nobody():
    """Sole-claimant, over a live server, with a real competing row.

    A competing claim lands between the claimant's INSERT and its read-back.
    Resolving the tie either way would let both sides keep the term when each
    sees only its own insert first, so both walk away and claim above it.

    What this does NOT show is a publish under the contested term: by the time
    the claimant publishes it holds a HIGHER term, so the fence is resolving
    that row and not the contested one. The docstring used to claim otherwise.
    `test_a_contested_head_term_is_made_safe_by_the_read_back_not_the_fence`
    covers the contested head itself.
    """
    competitor = str(uuid4())
    with _catalog() as (writer, reader, client, config):
        writer.release_publisher_lease()
        contested: list[int] = []

        class _ContestTheFirstClaim:
            def execute(self, query, params=None, **kwargs):
                if not contested and "acquired_at_ns, expires_at_ns" in query:
                    contested.append(params["term"])
                    client.execute(
                        "INSERT INTO "
                        f"`{config.database}`.`{config.table_prefix}"
                        "_publisher_lease` (term, lease_id, holder, "
                        "acquired_at_ns, expires_at_ns) SELECT "
                        "toUInt64(%(term)s), toUUID(%(lease_id)s), 'rival', "
                        "now_ns, now_ns + toUInt64(600000000000) FROM "
                        "(SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
                        {"term": params["term"], "lease_id": competitor},
                    )
                return client.execute(query, params, **kwargs)

        claimant = ClickHouseCatalogWriter(_ContestTheFirstClaim(), config)
        lease = claimant.acquire_publisher_lease("claimant")

        assert contested, "the first claim should have been contested"
        assert lease.term > contested[0], "a contested term must never be held"
        # The contested term really does hold two rows and neither is a lease.
        terms = [row[0] for row in _lease_rows(client, config)]
        assert terms.count(contested[0]) == 2
        # And the claimant, holding the term above it, publishes normally.
        version = claimant.allocate_version()
        _publish(claimant, version, refs=(), rows=0)
        assert int(reader.current_watermark()) == version


def test_the_fence_refuses_on_an_empty_lease_table_rather_than_throwing():
    """An empty lease table must make the fence FALSE, not raise.

    The predicate used to select a tuple, and a scalar subquery over no rows is
    not NULL on 25.12 -- it is `Code: 125, Scalar subquery returned empty result
    of type Tuple(UUID, UInt8) which cannot be Nullable`, a raw ServerException
    that is not a `CaptureStorageError` and that nothing in this package
    catches.

    No public call reaches it today, because `publish_snapshot` renews first
    and a renewal leaves a row behind. The GC obligation this branch created
    makes it reachable: a retention job that collects lease rows -- the
    document says any of these tables may be swept -- hands the next publish an
    empty table. Asserted at the statement, which is where the branch lives.
    """
    with _catalog() as (writer, reader, client, config):
        version = writer.allocate_version()
        lease = writer.publisher_lease
        table = f"`{config.database}`.`{config.table_prefix}_publisher_lease`"
        watermark = f"`{config.database}`.`{config.table_prefix}_index_watermark`"
        client.execute(f"TRUNCATE TABLE {table}")
        assert client.execute(f"SELECT count() FROM {table}") == [(0,)]

        fence = writer._lease_fence()
        # Evaluating it is not an error, and the answer is "no".
        assert client.execute(
            f"SELECT {fence}", {"lease_id": lease.lease_id}
        ) == [(0,)]
        # And the statement it guards writes nothing rather than throwing.
        client.execute(
            f"INSERT INTO {watermark} (index_version, publish_id, "
            "published_at_ns, indexed_rows, indexed_packs) SELECT "
            "%(index_version)s, toUUID(%(publish_id)s), 1, 0, 0 FROM system.one "
            f"WHERE {fence}",
            {
                "index_version": version,
                "publish_id": str(uuid4()),
                "lease_id": lease.lease_id,
            },
        )
        assert client.execute(f"SELECT count() FROM {watermark}") == [(0,)]
        assert int(reader.current_watermark()) == 0


def test_a_lease_is_taken_over_on_expiry_and_given_back_on_release():
    """Expiry, takeover, and a re-acquiring publisher that republishes.

    A live lease cannot be stolen. An expired one can be taken, which is what
    makes a crashed indexer recoverable rather than terminal. A release hands
    it back at once, so an orderly restart does not cost a whole TTL.

    Every clock here now runs in the direction load makes MORE true, which the
    first version of this did not. It gave the fixture a 200 ms TTL and then
    asserted, about five round trips later, that the lease was still live
    enough to refuse a successor -- so a busy server made it fail with DID NOT
    RAISE, and a correctness test that fails at random gets deleted by whoever
    is unlucky enough to hit it. The leases that must be LIVE now carry the
    default 30 s TTL; the lease that must be EXPIRED carries the shortest TTL
    the whole-second publish cap admits and is slept PAST, so load only makes
    its expiry more certain. Neither assertion is racing a deadline.
    """
    corpus = synthetic_descriptors(2)
    tenant = corpus[0].metadata.tenant_id
    with _catalog() as (writer, reader, client, config):
        # A live lease is not takeable, and the refusal names who holds it.
        successor = ClickHouseCatalogWriter(_client(), config)
        with pytest.raises(PublisherLeaseHeldError, match="snapshot-suite"):
            successor.acquire_publisher_lease("successor")

        # A crashed holder leaves a lease that lapses. Reproduced with the
        # shortest TTL the whole-second publish cap admits, slept past so the
        # DURABLE state -- a head row whose expires_at_ns is behind the
        # server's own clock -- is certain by the time it is read, and load
        # only makes it more so.
        writer.release_publisher_lease()
        crashed = ClickHouseCatalogWriter(
            _client(),
            replace(
                config,
                lease_ttl_ns=1_100_000_000,
                publish_timeout_ns=1_000_000_000,
            ),
        )
        held = crashed.acquire_publisher_lease("crashed")
        sleep(1.2)
        expires_at_ns, now_ns = client.execute(
            "SELECT expires_at_ns, toUnixTimestamp64Nano(now64(9)) FROM "
            f"`{config.database}`.`{config.table_prefix}_publisher_lease` "
            "ORDER BY term DESC, lease_id DESC LIMIT 1"
        )[0]
        assert expires_at_ns <= now_ns, (
            "precondition: the crashed holder's lease has to have lapsed"
        )

        taken = successor.acquire_publisher_lease("successor")
        assert taken.term > held.term

        # The previous holder is refused. Here it is the renewal at the head of
        # every publish that catches it, one round trip before the fence;
        # test_a_taken_over_publisher_writes_nothing_at_all drives the case the
        # renewal cannot see, where the takeover lands behind it.
        with pytest.raises(PublisherLeaseHeldError, match="'successor'"):
            _publish(crashed, crashed.allocate_version(), refs=(), rows=0)
        assert reader.current_watermark() == "0"
        assert crashed.publisher_lease is None

        # A release hands the lease straight back rather than making the next
        # publisher wait out the TTL.
        successor.release_publisher_lease()
        writer.acquire_publisher_lease("snapshot-suite")
        version = writer.allocate_version()
        writer.write_descriptors(corpus, index_version=version)
        _publish(writer, version, refs=_refs(corpus), rows=len(corpus))

        assert int(reader.current_watermark()) == version
        assert len(reader.search(CaptureQuery(limit=10, tenant_id=tenant)).items) == (
            len(corpus)
        )


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

        # One publisher at a time, so the fixture's writer hands the lease over.
        writer.release_publisher_lease()
        taps = ClickHouseCatalogWriter(_Recording(), config)
        taps.acquire_publisher_lease("recording-proxy")
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
        fenced = {
            "select_sequential_consistency": 1,
            "max_execution_time": 5.0,
            "timeout_overflow_mode": "throw",
        }
        assert _settings("toString(claim_id)", "SELECT") == [consistent]
        assert _settings("max(version)", "SELECT") == [consistent]
        assert _settings("max(index_version)", "SELECT") == [consistent]
        assert _settings("toString(publish_id)", "SELECT") == [consistent]
        assert _settings("ORDER BY term DESC", "SELECT") == [consistent] * 3
        assert _settings("acquired_at_ns, expires_at_ns", "SELECT") == [consistent] * 3
        assert _settings("snapshot_manifest", "SELECT") == [consistent]
        # The fenced writes carry the consistency AND the statement cap that
        # keeps the fence from being evaluated long before the row lands.
        assert _settings("snapshot_manifest", "INSERT") == [fenced]
        assert _settings("index_watermark", "INSERT") == [fenced]
        # Bulk writes decide nothing and pay nothing.
        assert _settings("capture_raw", "INSERT") == [None]
        assert _settings("pack_inventory_raw", "INSERT") == [None]
        assert _settings("publisher_lease` (term", "INSERT") == [None] * 3
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


def test_an_empty_membership_bound_admits_nothing_rather_than_everything():
    """The degenerate case of the bound, with rows behind it to lose.

    This was `test_the_public_view_of_an_empty_catalog_is_empty`, and it was
    vacuous: over an empty catalog the view is empty with its entire WHERE
    clause deleted, because the raw table is empty too. It also described a
    mechanism the view no longer has -- there is no `max(index_version)` in it,
    the bound pairs `(index_version, publish_id)` across two tables.

    The question the degenerate case actually asks is what an EMPTY bound
    admits. If the server read `IN (empty)` as unconditionally true, or if the
    bound dropped out of the statement, "published rows only" would silently
    become "every row in the raw table" -- so the catalog here holds rows, and
    the bound is emptied in each of the two ways a half-finished publish
    leaves it. The last step publishes for real, so "empty" is the bound
    working rather than the view being broken.
    """
    corpus = synthetic_descriptors(3)
    with _catalog() as (writer, _, client, config):
        writer.write_descriptors(corpus, index_version=1)

        # (a) Nothing published: both sides of the pair are empty, and the raw
        # table is not.
        assert _raw_rows(client, config) == len(corpus)
        assert _catalog_state(client, config) == {
            "index_watermark": [], "snapshot_manifest": []
        }
        assert _view_rows(client, config) == []

        # (b) Membership written, watermark never reached -- what a publish
        # that lost the race leaves behind. The manifest now names exactly the
        # packs these descriptors are in, so a bound that asked only "is this
        # pack in the manifest?" would admit every one of them.
        client.execute(
            f"INSERT INTO `{config.database}`.`{config.table_prefix}"
            "_snapshot_manifest` (index_version, publish_id, store_id, pack_id) "
            "VALUES",
            [(1, uuid4(), ref.store_id, ref.pack_id) for ref in _refs(corpus)],
        )
        assert _catalog_state(client, config)["snapshot_manifest"] != []
        assert _view_rows(client, config) == []

        # And the view is not simply always empty.
        _publish(writer, 1, refs=_refs(corpus), rows=len(corpus))
        assert len(_view_rows(client, config)) == len(corpus)


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
