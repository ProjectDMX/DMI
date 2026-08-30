from __future__ import annotations

from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    CatalogSchemaVersionError,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    PackIndex,
    PackRef,
    PackWriter,
    SnapshotPublishRaceError,
)
from dmi.storage.capture.clickhouse_catalog import _SCHEMA_VERSION
from tests._catalog_fakes import FakeLeaseTable


pytestmark = pytest.mark.cpu


# Every object `ensure_schema` owns, spelled out rather than derived from the
# writer: the compatibility check exists to notice one of them missing, so a
# test that asked the writer which objects it expects could not see the writer
# forget one.
_CURRENT_OBJECTS = (
    "dmi_capture",
    "dmi_pack_inventory",
    "dmi_capture_raw",
    "dmi_pack_inventory_raw",
    "dmi_capture_version_claims",
    "dmi_publisher_lease",
    "dmi_index_watermark",
    "dmi_snapshot_manifest",
    "dmi_schema_version",
)


class _Client:
    """A ClickHouse stand-in whose catalog state the test declares.

    ``tables``, ``schema_version``, ``inventory_rows`` and ``manifest_rows``
    describe a server the writer is about to meet: an empty default is a fresh
    install, and the other combinations are the upgrade states `ensure_schema`
    has to tell apart.
    """

    def __init__(
        self,
        *,
        tables=(),
        schema_version=None,
        inventory_rows=0,
        manifest_rows=0,
    ):
        self.calls = []
        self.committed = []
        self.claims = []
        self.watermarks = []
        self.publishes = []
        # Rows at the version under test that this writer did not write: an
        # operator's INSERT, a second build, a publisher whose statement
        # overlapped. The point of reading publish_id back is that these are
        # not success.
        self.foreign_publishes = []
        self.lease = FakeLeaseTable()
        self.tables = tuple(tables)
        self.schema_version = schema_version
        self.inventory_rows = inventory_rows
        self.manifest_rows = manifest_rows

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, params, kwargs))
        leased = self.lease.execute(query, params)
        if leased is not None:
            return leased
        if "system.tables" in query:
            return [(name,) for name in self.tables]
        if "dmi_schema_version` ORDER BY version DESC" in query:
            return [] if self.schema_version is None else [(self.schema_version,)]
        if "dmi_pack_inventory_raw`), (SELECT count()" in query:
            return [(self.inventory_rows, self.manifest_rows)]
        if query.lstrip().startswith("INSERT"):
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            elif "index_watermark" in query:
                # The barrier and the fence are server-side conditions, so the
                # fake has to enforce them or every publish test would pass
                # vacuously.
                version = params["index_version"]
                if version > max(self.watermarks, default=0) and (
                    self.lease.fence_passes(params["lease_id"])
                ):
                    self.watermarks.append(version)
                    self.publishes.append((version, params["publish_id"]))
            return []
        if "publish_id" in query and "index_watermark" in query:
            return [
                (publish_id,)
                for version, publish_id in self.publishes
                if version == params["version"]
            ] + [
                (publish_id,) for publish_id in self.foreign_publishes
            ]
        # The version allocator's queries are answered from real claim state;
        # every other SELECT returns the canned rows.
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            return [(cid,) for v, cid in self.claims if v == params["version"]]
        if query.lstrip().startswith("SELECT"):
            return self.committed
        return []


def _leased(client) -> ClickHouseCatalogWriter:
    """A writer holding the publisher lease, which every publish requires."""
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
    writer.acquire_publisher_lease("indexer-a")
    return writer


def _descriptor():
    metadata = CaptureMetadata(
        capture_id="capture-a",
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    pack = PackWriter(
        pack_id=UUID("018f0000-0000-7000-8000-000000000001"),
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=1024 * 1024,
    )
    pack.append(CaptureRecord(metadata, b"abcdefgh"))
    sealed = pack.seal()
    ref = PackRef(
        sealed.pack_id, "garage", "packs/a.dmi-pack", len(sealed.data),
        sealed.checksum, sealed.record_count,
    )

    class Store:
        store_id = "garage"
        def read_range(self, ref, offset, length):
            return sealed.data[offset : offset + length]

    return ref, PackIndex.from_store(Store(), ref).descriptors()[0]


def test_clickhouse_catalog_creates_replay_safe_raw_tables_and_final_views():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    ddl = "\n".join(call[0] for call in client.calls)
    assert "ReplacingMergeTree(index_version)" in ddl
    assert "FROM `default`.`dmi_capture_raw` FINAL" in ddl
    assert "FROM `default`.`dmi_pack_inventory_raw` FINAL" in ddl
    # Pack identity closes the sort key. Without it ReplacingMergeTree would be
    # free to delete one of two rows describing one capture in two packs, and a
    # snapshot pinned to the deleted row's pack would stop resolving. ORDER BY
    # cannot be altered in place, so this pin is the thing that keeps a
    # deployment from needing a copy migration.
    assert (
        "ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, "
        "capture_id, store_id, pack_id)"
    ) in ddl
    # The allocator's claim ledger is append-only and ordered by the claim
    # itself -- claimed_at_ns is diagnostic, never part of the ordering.
    assert "`dmi_capture_version_claims`" in ddl
    assert "ORDER BY (version, claim_id)" in ddl


def test_the_capture_view_is_bounded_to_the_published_snapshot():
    """`FINAL` deduplicates; it does not decide what exists.

    Unbounded, the view showed rows from batches that were written and never
    published, and rows orphaned by a crashed indexing pass -- data the reader
    correctly reports as nonexistent, offered to anyone querying the catalog by
    hand as though it were catalog contents.
    """
    from dmi.storage.capture.clickhouse_catalog import _CAPTURE_COLUMNS

    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    view = next(
        call[0] for call in client.calls if "`default`.`dmi_capture` AS" in call[0]
    )
    # IF NOT EXISTS would leave an earlier build's unbounded view in place,
    # serving unpublished rows for the life of the deployment.
    assert view.startswith(
        "CREATE OR REPLACE VIEW `default`.`dmi_capture` AS SELECT "
        + ", ".join(_CAPTURE_COLUMNS[:-1])
        + " FROM `default`.`dmi_capture_raw` FINAL "
    )
    # The bound is the publish PAIR, exactly as the reader's membership clause
    # spells it: owning a version and owning its membership are separate claims.
    assert (
        "WHERE (store_id, pack_id) IN ("
        "SELECT store_id, pack_id FROM `default`.`dmi_snapshot_manifest` "
        "WHERE (index_version, publish_id) IN "
        "(SELECT index_version, publish_id FROM `default`.`dmi_index_watermark`))"
    ) in view
    # No grouping, deliberately. The view's meaning is "published descriptor
    # rows, engine-deduplicated" -- one row per (capture, store, pack).
    # Supersession lives in the reader; a second copy of it here could drift.
    assert "argMax" not in view
    assert "GROUP BY" not in view


def test_the_capture_view_is_created_after_the_tables_it_reads():
    """The view names the manifest and the watermark, so both must precede it.

    Reordered, ensure_schema breaks on a fresh server only -- a rerun finds the
    tables already there and passes -- so the order is pinned rather than left
    to whoever edits the method next.
    """
    client = _Client()
    ClickHouseCatalogWriter(client, ClickHouseCatalogConfig()).ensure_schema()

    statements = [call[0] for call in client.calls]

    def _position(fragment: str) -> int:
        return next(i for i, item in enumerate(statements) if fragment in item)

    view = _position("`default`.`dmi_capture` AS")
    assert _position("CREATE TABLE IF NOT EXISTS `default`.`dmi_snapshot_manifest`") < view
    assert _position("CREATE TABLE IF NOT EXISTS `default`.`dmi_index_watermark`") < view


def test_publish_writes_membership_before_the_watermark_that_admits_it():
    """Order is the whole mechanism, so it is pinned rather than assumed.

    A reader counts a manifest row only once its version also appears in the
    watermark log. Writing the watermark first would open a window where the
    version is readable but its packs are not yet members, so a snapshot pinned
    there would gain them a moment later.
    """
    ref, descriptor = _descriptor()
    client = _Client()
    writer = _leased(client)

    writer.write_descriptors([descriptor], index_version=42)
    writer.publish_snapshot(
        index_version=42, refs=[ref], published_at_ns=7,
        indexed_rows=1, indexed_packs=1,
    )
    writer.commit_packs([ref], index_version=42)

    inserts = [
        call for call in client.calls
        if call[0].startswith("INSERT") and "publisher_lease` (term" not in call[0]
    ]
    assert "dmi_capture_raw" in inserts[0][0]
    assert "dmi_snapshot_manifest" in inserts[1][0]
    assert "dmi_index_watermark" in inserts[2][0]
    # The inventory is only the replay guard now, and CatalogIndexer writes it
    # after a successful publish: a crash in between costs redundant work, not
    # a pack that is skipped forever and never visible.
    assert "dmi_pack_inventory_raw" in inserts[3][0]
    assert inserts[0][1][0][0] == "capture-a"
    # One publish identity on both rows, so membership pairs with the watermark.
    assert inserts[1][1]["index_version"] == 42
    assert inserts[1][1]["members"] == [(ref.store_id, ref.pack_id)]
    publish_id = inserts[1][1]["publish_id"]
    assert inserts[2][1]["publish_id"] == publish_id
    assert client.publishes == [(42, publish_id)]


def test_publish_is_a_single_statement_barrier_over_the_watermark():
    """The check and the visibility write cannot be separated by a round trip.

    A SELECT-then-INSERT leaves the whole client round trip -- network, driver,
    a GC pause -- between "am I the highest?" and "I am now visible". As one
    conditional INSERT the server evaluates both together.
    """
    client = _Client()
    writer = _leased(client)

    writer.publish_snapshot(
        index_version=42, refs=(), published_at_ns=7,
        indexed_rows=0, indexed_packs=0,
    )

    watermark_insert = next(
        call[0] for call in client.calls
        if call[0].startswith("INSERT") and "index_watermark" in call[0]
    )
    assert "SELECT" in watermark_insert
    assert (
        "WHERE (SELECT max(index_version) FROM `default`.`dmi_index_watermark`) "
        "< %(index_version)s"
    ) in watermark_insert


def test_publish_below_the_published_head_raises_the_race_error():
    """The barrier fires, and says nothing was made visible."""
    client = _Client()
    writer = _leased(client)
    writer.publish_snapshot(
        index_version=9, refs=(), published_at_ns=1,
        indexed_rows=0, indexed_packs=0,
    )

    with pytest.raises(SnapshotPublishRaceError, match="lost the publish race"):
        writer.publish_snapshot(
            index_version=8, refs=(), published_at_ns=2,
            indexed_rows=0, indexed_packs=0,
        )

    # The loser's watermark row never landed, which is exactly what makes the
    # manifest rows it already wrote inert.
    assert client.watermarks == [9]


def test_publish_verifies_the_row_at_its_version_is_its_own():
    """Ownership, not occupancy.

    ``count() > 0`` answers "does a row for V exist?", so a row written by
    anything else reads as success and the publisher reports a snapshot it did
    not make. The check reads ``publish_id`` back and compares it to the one
    this attempt minted.
    """
    client = _Client()
    writer = _leased(client)

    writer.publish_snapshot(
        index_version=42, refs=(), published_at_ns=7,
        indexed_rows=0, indexed_packs=0,
    )

    check = next(
        call for call in client.calls
        if call[0].startswith("SELECT") and "index_watermark" in call[0]
        and "publish_id" in call[0]
    )
    assert check[0] == (
        "SELECT toString(publish_id) FROM `default`.`dmi_index_watermark` "
        "WHERE index_version = %(version)s"
    )
    assert check[1] == {"version": 42}
    # And nothing counts rows any more: a count cannot tell whose row it is.
    assert not any(
        "count()" in call[0] and "dmi_index_watermark" in call[0]
        for call in client.calls
    )


def test_a_foreign_row_at_the_published_version_is_not_success():
    """The failure a count could not see.

    The sole-claimant allocator makes a foreign row at V unlikely, not
    impossible -- a stray operator INSERT, a second build, a publisher whose
    statement overlapped this one. A publisher that finds anything but its own
    identity standing at V has not published V and must say so.
    """
    client = _Client()
    client.foreign_publishes = ["ffffffff-ffff-ffff-ffff-ffffffffffff"]
    writer = _leased(client)

    with pytest.raises(SnapshotPublishRaceError, match="lost the publish race"):
        writer.publish_snapshot(
            index_version=42, refs=(), published_at_ns=7,
            indexed_rows=0, indexed_packs=0,
        )


def test_each_publish_attempt_mints_its_own_identity():
    """A retry at one version is a different write, and must not read as its own.

    Reusing the allocator's claim id would make a second attempt at V read the
    FIRST attempt's row back as its own and report success for a statement that
    inserted nothing.
    """
    client = _Client()
    writer = _leased(client)

    writer.publish_snapshot(
        index_version=42, refs=(), published_at_ns=7,
        indexed_rows=0, indexed_packs=0,
    )
    with pytest.raises(SnapshotPublishRaceError):
        writer.publish_snapshot(
            index_version=42, refs=(), published_at_ns=8,
            indexed_rows=0, indexed_packs=0,
        )

    identities = [
        call[1]["publish_id"] for call in client.calls
        if call[0].startswith("INSERT") and "dmi_index_watermark" in call[0]
    ]
    assert len(set(identities)) == 2


def test_the_deciding_reads_carry_sequential_consistency():
    """The read-backs the sole-claimant protocols rest on, and only those.

    A replica answers from the log entries it has fetched, so a read-back can
    miss a row another claimant has already committed and two claimants can both
    see themselves alone. Descriptor and inventory inserts carry nothing: they
    decide nothing.
    """
    from dmi.storage.capture.clickhouse_catalog import _DECIDING_READ

    ref, descriptor = _descriptor()
    client = _Client()
    writer = _leased(client)

    version = writer.allocate_version()
    writer.write_descriptors([descriptor], index_version=version)
    writer.publish_snapshot(
        index_version=version, refs=[ref], published_at_ns=7,
        indexed_rows=1, indexed_packs=1,
    )
    writer.commit_packs([ref], index_version=version)

    def _settings(fragment: str, kind: str) -> list:
        return [
            call[2].get("settings")
            for call in client.calls
            if fragment in call[0] and call[0].lstrip().startswith(kind)
        ]

    assert _DECIDING_READ == {"select_sequential_consistency": 1}
    # The claim and lease read-backs, the published-head reads and the publish
    # verification.
    assert _settings("toString(claim_id)", "SELECT") == [_DECIDING_READ]
    assert _settings("max(version)", "SELECT") == [_DECIDING_READ]
    assert _settings("max(index_version)", "SELECT") == [_DECIDING_READ]
    assert _settings("toString(publish_id)", "SELECT") == [_DECIDING_READ]
    assert _settings("ORDER BY term DESC", "SELECT") == [_DECIDING_READ] * 2
    assert _settings("acquired_at_ns, expires_at_ns", "SELECT") == [_DECIDING_READ] * 2
    # The two fenced writes carry the same consistency AND the statement cap
    # that keeps the fence from being evaluated long before the row lands.
    fenced = {
        "select_sequential_consistency": 1,
        "max_execution_time": 5.0,
        "timeout_overflow_mode": "throw",
    }
    assert _settings("dmi_snapshot_manifest", "INSERT") == [fenced]
    assert _settings("dmi_index_watermark", "INSERT") == [fenced]
    # Bulk writes decide nothing and pay nothing.
    assert _settings("dmi_capture_raw", "INSERT") == [None]
    assert _settings("dmi_pack_inventory_raw", "INSERT") == [None]
    assert _settings("publisher_lease` (term", "INSERT") == [None] * 2


def test_commit_packs_writes_only_the_replay_inventory():
    ref, _ = _descriptor()
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.commit_packs([ref], index_version=42)

    inserts = [call for call in client.calls if call[0].startswith("INSERT")]
    assert len(inserts) == 1
    assert "dmi_pack_inventory_raw" in inserts[0][0]


def test_clickhouse_catalog_queries_committed_pack_ids_in_one_batch():
    client = _Client()
    client.committed = [("garage", "018f0000-0000-7000-8000-000000000001")]
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    found = writer.committed_pack_ids(
        [("garage", "018f0000-0000-7000-8000-000000000001")]
    )

    assert found == set(client.committed)
    assert "IN %(identities)s" in client.calls[0][0]


@pytest.mark.parametrize("name", ["bad-name", "x; DROP TABLE y", "`quoted`"])
def test_clickhouse_catalog_rejects_unsafe_identifiers(name: str):
    with pytest.raises(ValueError, match="identifier"):
        ClickHouseCatalogConfig(database=name)


# --- schema version ----------------------------------------------------------
#
# `CREATE TABLE IF NOT EXISTS` cannot alter an existing table, so this build's
# descriptor sort key and its membership table are unreachable by any statement
# `ensure_schema` could issue against a catalog an older build created. Running
# anyway is silent: the DDL succeeds, the old sort key survives, and the
# populated pack inventory then makes every pre-existing pack look already
# indexed, so none of them ever reaches the new manifest and every capture in
# them becomes invisible. The catalog is derived, so the answer is to refuse
# and rebuild -- these pin the refusal.


def _writer(client) -> ClickHouseCatalogWriter:
    return ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())


def test_compatibility_is_checked_before_any_ddl_is_issued():
    """A refusal after `CREATE DATABASE` would already have changed the server."""
    client = _Client()

    _writer(client).ensure_schema()

    assert "system.tables" in client.calls[0][0]
    # Version 1's membership table is looked for too: after a half-finished
    # manual cleanup it is often the only object left standing, and a catalog
    # with it still there is not one to create tables beside.
    assert client.calls[0][1] == {
        "database": "default",
        "names": list(_CURRENT_OBJECTS) + ["dmi_pack_commit_log"],
    }


def test_a_fresh_install_stamps_the_schema_version_last():
    """The stamp means "every object exists", so it cannot be written earlier.

    Written first, an install that died partway would leave a catalog claiming
    to be complete, and the next start would trust it.
    """
    client = _Client()

    _writer(client).ensure_schema()

    statements = [call[0] for call in client.calls]
    stamp = next(
        index
        for index, item in enumerate(statements)
        if item.startswith("INSERT") and "dmi_schema_version" in item
    )
    assert stamp == len(statements) - 1
    assert client.calls[stamp][1]["version"] == _SCHEMA_VERSION
    # Conditional server-side, so a rerun against a stamped catalog inserts
    # nothing and no read-then-write window exists between the two.
    assert (
        "WHERE (SELECT count() FROM `default`.`dmi_schema_version`) = 0"
    ) in statements[stamp]
    # And the table itself is created first, so an install interrupted below
    # leaves a catalog that reads as "this build, unfinished" rather than one
    # indistinguishable from version 1 and refused forever.
    assert statements[2] == (
        "CREATE TABLE IF NOT EXISTS `default`.`dmi_schema_version` (\n"
        "version UInt32, applied_at_ns UInt64\n"
        ") ENGINE = MergeTree ORDER BY version"
    )
    assert statements[1].startswith("CREATE DATABASE")


def test_a_version_one_catalog_is_refused_with_the_rebuild_procedure():
    """The upgrade this branch cannot perform, refused instead of half-done."""
    client = _Client(
        tables=("dmi_capture_raw", "dmi_pack_inventory_raw", "dmi_pack_commit_log")
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "schema version 1" in message
    assert f"requires version {_SCHEMA_VERSION}" in message
    # Both incompatible changes are named, because an operator reading this has
    # to know the catalog cannot simply be altered into shape.
    assert "ORDER BY" in message and "(store_id, pack_id)" in message
    assert "dmi_pack_commit_log" in message and "dmi_snapshot_manifest" in message
    # And the exact recovery, including the part that is easy to skip.
    assert "CatalogReconciler.rebuild()" in message
    assert "Dropping the pack inventory is mandatory" in message
    assert "`default`.`dmi_pack_inventory_raw`" in message
    # Nothing was created, altered or written on the way to the refusal.
    assert [call[0] for call in client.calls[1:]] == []


def test_an_unstamped_install_of_this_build_is_completed_not_refused():
    """A crash between creating the objects and stamping them is recoverable.

    The version table exists and holds no row only when THIS build got that
    far, so re-running the (idempotent) DDL is the repair, not a refusal that
    would wedge a half-created catalog forever.
    """
    client = _Client(tables=_CURRENT_OBJECTS, schema_version=None)

    _writer(client).ensure_schema()

    assert any(call[0].startswith("CREATE TABLE") for call in client.calls)
    assert client.calls[-1][1]["version"] == _SCHEMA_VERSION


def test_a_catalog_written_by_a_newer_build_is_refused():
    newer = _SCHEMA_VERSION + 1
    client = _Client(tables=_CURRENT_OBJECTS, schema_version=newer)

    with pytest.raises(
        CatalogSchemaVersionError, match=f"schema version {newer}"
    ):
        _writer(client).ensure_schema()


def test_a_stamped_catalog_with_a_dropped_table_is_refused():
    """Half a drop is worse than none, so it is not quietly completed.

    Recreating the manifest empty beside a populated inventory is exactly the
    state that publishes nothing and hides everything.
    """
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != "dmi_snapshot_manifest"],
        schema_version=_SCHEMA_VERSION,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    assert "missing `dmi_snapshot_manifest`" in str(raised.value)
    assert "CatalogReconciler.rebuild()" in str(raised.value)


def test_a_populated_inventory_with_an_empty_manifest_is_refused():
    """The defect this whole check exists for, in its purest form.

    `committed_pack_ids` reads the inventory and reports every pack as already
    indexed; readers bound their snapshot by the manifest, which names none of
    them. Every capture in those packs is durable and invisible, and an
    indexing pass over them reports success.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION, inventory_rows=4, manifest_rows=0
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "dmi_pack_inventory_raw" in message and "dmi_snapshot_manifest" in message
    assert "empty but reports success" in message


def test_a_catalog_at_this_builds_version_is_accepted():
    client = _Client(
        tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION, inventory_rows=4, manifest_rows=2
    )

    _writer(client).ensure_schema()

    assert any("CREATE OR REPLACE VIEW" in call[0] for call in client.calls)


def test_an_empty_catalog_at_this_builds_version_is_accepted():
    """Nothing indexed yet is not the same state as membership gone missing."""
    client = _Client(tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION)

    _writer(client).ensure_schema()

    assert any("CREATE OR REPLACE VIEW" in call[0] for call in client.calls)



# --- catalog facets ---------------------------------------------------------


def test_ensure_schema_declares_every_facet_as_a_materialized_column():
    from dmi.storage.capture.clickhouse_catalog import _FACET_COLUMNS

    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    create = next(
        call[0]
        for call in client.calls
        if call[0].startswith("CREATE TABLE") and "dmi_capture_raw" in call[0]
    )
    for name, kind, expression in _FACET_COLUMNS:
        assert f"{name} {kind} MATERIALIZED {expression}" in create


def test_ensure_schema_upgrades_pre_facet_tables_idempotently():
    from dmi.storage.capture.clickhouse_catalog import _FACET_COLUMNS

    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    alters = [call[0] for call in client.calls if call[0].startswith("ALTER TABLE")]
    column_alters = [item for item in alters if "ADD COLUMN" in item]
    assert len(column_alters) == len(_FACET_COLUMNS)
    for statement in column_alters:
        # Without IF NOT EXISTS a second start would fail on an upgraded table.
        assert "ADD COLUMN IF NOT EXISTS" in statement
        assert "`default`.`dmi_capture_raw`" in statement


def test_ensure_schema_adds_a_bloom_filter_index_on_capture_id():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    alters = [call[0] for call in client.calls if call[0].startswith("ALTER TABLE")]
    # Point lookups arrive with tenant + capture_id; the primary key prunes to
    # the tenant range and the bloom filter prunes granules within it.
    added = [item for item in alters if "ADD INDEX" in item]
    assert added == [
        "ALTER TABLE `default`.`dmi_capture_raw` ADD INDEX IF NOT EXISTS "
        "capture_id_bloom capture_id TYPE bloom_filter(0.01) GRANULARITY 4"
    ]
    # Existing parts only get the index through MATERIALIZE; new parts are
    # indexed at insert.
    materialized = [item for item in alters if "MATERIALIZE INDEX" in item]
    assert materialized == [
        "ALTER TABLE `default`.`dmi_capture_raw` MATERIALIZE INDEX capture_id_bloom"
    ]
    # And the index lands after the ADD, never before it.
    assert alters.index(added[0]) < alters.index(materialized[0])


def test_facets_never_collide_with_an_inserted_column():
    from dmi.storage.capture.clickhouse_catalog import (
        _CAPTURE_COLUMNS,
        _FACET_COLUMNS,
    )

    # A MATERIALIZED column cannot be written to, so a collision with the
    # writer's column list would break every insert.
    assert {name for name, _, _ in _FACET_COLUMNS}.isdisjoint(_CAPTURE_COLUMNS)


def test_clickhouse_catalog_config_rejects_non_positive_query_pack_limit():
    with pytest.raises(ValueError, match="query_pack_limit"):
        ClickHouseCatalogConfig(query_pack_limit=0)


def test_committed_pack_ids_short_circuits_on_empty_input():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    assert writer.committed_pack_ids([]) == set()
    assert client.calls == []


def test_committed_pack_ids_bounds_the_identity_batch():
    client = _Client()
    writer = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(query_pack_limit=1)
    )

    with pytest.raises(ValueError, match="query_pack_limit"):
        writer.committed_pack_ids([("garage", "a"), ("garage", "b")])
    assert client.calls == []


def test_committed_pack_ids_decodes_bytes_identities():
    client = _Client()
    client.committed = [(b"garage", b"018f0000-0000-7000-8000-000000000001")]
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    found = writer.committed_pack_ids(
        [("garage", "018f0000-0000-7000-8000-000000000001")]
    )

    assert found == {("garage", "018f0000-0000-7000-8000-000000000001")}


def test_committed_pack_ids_rejects_non_text_identities():
    client = _Client()
    client.committed = [(5, "018f0000-0000-7000-8000-000000000001")]
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    with pytest.raises(ValueError, match="non-text identifier"):
        writer.committed_pack_ids([("garage", "a")])


def test_write_descriptors_short_circuits_on_empty_input():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.write_descriptors([], index_version=42)

    assert client.calls == []


def test_commit_packs_short_circuits_on_empty_input():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.commit_packs([], index_version=42)

    assert client.calls == []


@pytest.mark.parametrize("version", [-1, 2**64, "42"])
def test_index_versions_must_fit_uint64(version):
    _, descriptor = _descriptor()
    writer = ClickHouseCatalogWriter(_Client(), ClickHouseCatalogConfig())

    with pytest.raises(ValueError, match="UInt64"):
        writer.write_descriptors([descriptor], index_version=version)


def test_last_published_version_returns_the_reported_maximum():
    client = _Client()
    client.committed = [(5,)]
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    assert writer.last_published_version() == 5


def test_last_published_version_rejects_a_non_integer_watermark():
    client = _Client()
    client.committed = [("5",)]
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    with pytest.raises(ValueError, match="invalid version"):
        writer.last_published_version()
