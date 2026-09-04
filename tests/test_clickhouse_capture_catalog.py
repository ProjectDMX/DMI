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
    SnapshotPublishConflictError,
    SnapshotPublishRaceError,
)
from dmi.storage.capture.clickhouse_schema import SCHEMA_VERSION as _SCHEMA_VERSION
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

# The kinds those objects have, so the fake can answer `system.tables` the way
# the server does. The compatibility check reads the engine to tell a view from
# a table, and a missing view is recoverable where a missing table is not.
_VIEWS = ("dmi_capture", "dmi_pack_inventory")

# This build's descriptor sort key, spelled out rather than imported, so a
# change to the constant has to be restated here on purpose.
_CURRENT_SORT_KEY = (
    "tenant_id, experiment_id, run_id, captured_at_ns, capture_id, "
    "store_id, pack_id"
)
_VERSION_ONE_SORT_KEY = (
    "tenant_id, experiment_id, run_id, captured_at_ns, capture_id"
)

# Every object version 1 created, which is not the same set: it had a commit
# log and no manifest, no publisher lease and no version stamp. Kept beside
# `_CURRENT_OBJECTS` so the two schemas the refusal has to tell apart are both
# written down rather than one being the other's negation.
_VERSION_ONE_OBJECTS = (
    "dmi_capture",
    "dmi_pack_inventory",
    "dmi_capture_raw",
    "dmi_pack_inventory_raw",
    "dmi_capture_version_claims",
    "dmi_index_watermark",
    "dmi_pack_commit_log",
)


class _Client:
    """A ClickHouse stand-in whose catalog state the test declares.

    ``tables``, ``schema_version`` and ``inventory_without_membership``
    describe a server the writer is about to meet: an empty default is a fresh
    install, and the other combinations are the upgrade states `ensure_schema`
    has to tell apart.

    ``sort_key``, ``engines`` and ``publish_id_tables`` are the rest of what
    `system.tables` and `system.columns` report, because the refusals now
    describe the catalog they actually found instead of reciting one version's
    differences. Their defaults are this build's own shape, so a test that says
    nothing about them is describing a catalog that differs only in the ways it
    named.
    """

    def __init__(
        self,
        *,
        tables=(),
        schema_version=None,
        inventory_without_membership=0,
        # Whether any surviving DATA table holds a row. The missing-table
        # refusal turns on it: with every data table empty there is nothing a
        # recreated-empty neighbour could hide, and re-running the DDL is the
        # documented completion of an interrupted install.
        data_rows=0,
        sort_key=_CURRENT_SORT_KEY,
        engines=None,
        engine_full=None,
        publish_id_tables=("dmi_index_watermark", "dmi_snapshot_manifest"),
    ):
        self.calls = []
        self.committed = []
        self.claims = []
        self.watermarks = []
        self.publishes = []
        self.manifest = []
        # Rows at the version under test that this writer did not write: an
        # operator's INSERT, a second build, a publisher whose statement
        # overlapped. The point of reading publish_id back is that these are
        # not success.
        self.foreign_publishes = []
        self.lease = FakeLeaseTable()
        self.tables = tuple(tables)
        self.schema_version = schema_version
        self.inventory_without_membership = inventory_without_membership
        self.data_rows = data_rows
        self.sort_key = sort_key
        self.engine_full = engine_full or {}
        self.engines = dict(engines or {})
        self.publish_id_tables = tuple(publish_id_tables)

    def _engine(self, name: str) -> str:
        if name in self.engines:
            return self.engines[name]
        if name in _VIEWS:
            return "View"
        return "ReplacingMergeTree" if name.endswith("_raw") else "MergeTree"

    def _engine_full(self, name: str) -> str:
        """What `system.tables.engine_full` reports: engine AND its arguments.

        `engine` alone cannot distinguish `ReplacingMergeTree(index_version)`
        from `ReplacingMergeTree()`, and only the first collapses duplicate
        descriptor rows by version.
        """
        if name in self.engine_full:
            return self.engine_full[name]
        engine = self._engine(name)
        if engine == "ReplacingMergeTree":
            # The shape the SERVER renders, not just the engine call: engine_full
            # carries the whole engine clause, so ORDER BY and SETTINGS trail the
            # arguments. A fake that answered `ReplacingMergeTree(index_version)`
            # bare let a parser that read to the LAST `)` pass here and refuse
            # every healthy catalog against a real server.
            return (
                "ReplacingMergeTree(index_version) "
                f"ORDER BY ({self.sort_key or 'store_id, pack_id'}) "
                "SETTINGS index_granularity = 8192"
            )
        if engine == "MergeTree":
            return "MergeTree ORDER BY version SETTINGS index_granularity = 8192"
        return engine

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, params, kwargs))
        leased = self.lease.execute(query, params)
        if leased is not None:
            return leased
        if "system.columns" in query:
            return [
                (name,)
                for name in params["tables"]
                if name in self.publish_id_tables
            ]
        if query.startswith("CHECK GRANT SHOW TABLES"):
            return [(1,)]
        if query.startswith(("CREATE TABLE", "CREATE VIEW", "CREATE OR REPLACE VIEW")):
            # Modelled, because ensure_schema now re-reads `system.tables`
            # after a fresh install to prove the objects it created are
            # visible -- a check a static table list would fail outright.
            created = query.split("`")[3]
            if created not in self.tables:
                self.tables = self.tables + (created,)
            return []
        if "system.tables" in query:
            # Only the descriptor table's sort key is ever read, so the others
            # answer with a placeholder rather than a fiction that looks
            # meaningful. `engine_full` is appended only when the statement
            # asks for it, so this fake answers the read either shape.
            rows = [
                (
                    name,
                    self._engine(name),
                    self.sort_key if name == "dmi_capture_raw" else "",
                )
                for name in self.tables
            ]
            if "engine_full" in query:
                return [row + (self._engine_full(row[0]),) for row in rows]
            return rows
        if "dmi_schema_version` ORDER BY version DESC" in query:
            return [] if self.schema_version is None else [(self.schema_version,)]
        if "dmi_pack_inventory_raw` FINAL" in query and "NOT IN" in query:
            return [(self.inventory_without_membership,)]
        if query.startswith("SELECT count() FROM `") and " WHERE " in query:
            # The retention pass counts what a predicate matches before it
            # deletes exactly those rows. Answered as "none", so the fake never
            # issues the mutation and the test can assert on the predicates.
            return [(0,)]
        if query.startswith("SELECT count() FROM (SELECT 1 FROM"):
            # The surviving-data probe behind the missing-table refusal. Keyed
            # on the shape rather than on the UNION ALL, which a catalog with
            # one surviving data table does not have.
            return [(
                1 if (self.data_rows or self.inventory_without_membership) else 0,
            )]
        if query.lstrip().startswith("INSERT"):
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            elif "snapshot_manifest" in query:
                if self.lease.fence_admits(query, params):
                    self.manifest.extend(
                        (
                            params["index_version"],
                            params["publish_id"],
                            store_id,
                            pack_id,
                        )
                        for store_id, pack_id in params["members"]
                    )
            elif "index_watermark" in query:
                # The barrier and the fence are server-side conditions, so the
                # fake has to enforce them or every publish test would pass
                # vacuously. The fence check runs UNCONDITIONALLY --
                # short-circuited behind the barrier, a statement missing the
                # fence would slip by whenever the barrier refused it.
                version = params["index_version"]
                fenced = self.lease.fence_admits(query, params)
                if fenced and version > max(self.watermarks, default=0):
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
        if "snapshot_manifest" in query and "SELECT count()" in query:
            wanted = set(params["members"])
            found = {
                (store_id, pack_id)
                for version, publish_id, store_id, pack_id in self.manifest
                if version == params["index_version"]
                and publish_id == params["publish_id"]
                and (store_id, pack_id) in wanted
            }
            return [(len(found),)]
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
    from dmi.storage.capture.clickhouse_schema import CAPTURE_COLUMNS as _CAPTURE_COLUMNS

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
    # ifNull, not a bare scalar: on a profile where max() over an empty table
    # answers NULL, a bare comparison is NULL and the FIRST publish into a
    # fresh catalog would be refused forever as a phantom lost race.
    assert (
        "WHERE ifNull((SELECT max(index_version) FROM "
        "`default`.`dmi_index_watermark`), 0) "
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
    # And nothing READS the watermark by counting any more: a count cannot tell
    # whose row it is. (The lease fence counts, but over the lease table, and
    # only inside the statements that write.)
    assert not any(
        call[0].lstrip().startswith("SELECT")
        and "count()" in call[0]
        and "dmi_index_watermark" in call[0]
        for call in client.calls
    )


def test_a_foreign_row_where_this_publish_is_absent_is_a_lost_race():
    """The failure a count could not see.

    The sole-claimant allocator makes a foreign row at V unlikely, not
    impossible -- a stray operator INSERT, a second build, a publisher whose
    statement overlapped this one. Here this publish's own conditional INSERT
    was refused (the barrier: 42 is below a published head of 100), so the only
    row standing at V belongs to somebody else. Nothing this attempt wrote can
    enter a snapshot, and re-allocating above the head is the recovery.
    """
    client = _Client()
    client.watermarks = [100]
    client.foreign_publishes = ["ffffffff-ffff-ffff-ffff-ffffffffffff"]
    writer = _leased(client)

    with pytest.raises(SnapshotPublishRaceError) as raised:
        writer.publish_snapshot(
            index_version=42, refs=(), published_at_ns=7,
            indexed_rows=0, indexed_packs=0,
        )

    assert "lost the publish race" in str(raised.value)
    assert "Allocate a higher version" in str(raised.value)


def test_a_foreign_row_BESIDE_this_publish_is_not_a_lost_race():
    """The other half of the same read-back, and the opposite recovery.

    A foreign row arriving at V *after* this publish's landed does not undo it:
    the watermark row carrying this ``publish_id`` is standing, its manifest
    rows are paired with it, and its packs are in the snapshot. Calling that a
    lost race told the caller "nothing it wrote is visible" -- false -- and had
    `CatalogIndexer` republish the same batch underneath a snapshot that
    already contained it. It is an anomaly, so it is raised; it is not
    retryable, so it is not a ``SnapshotPublishRaceError``.
    """
    client = _Client()
    client.foreign_publishes = ["ffffffff-ffff-ffff-ffff-ffffffffffff"]
    writer = _leased(client)

    with pytest.raises(SnapshotPublishConflictError) as raised:
        writer.publish_snapshot(
            index_version=42, refs=(), published_at_ns=7,
            indexed_rows=0, indexed_packs=0,
        )

    assert not isinstance(raised.value, SnapshotPublishRaceError), (
        "the indexer's publish retry absorbs SnapshotPublishRaceError, and "
        "retrying this one republishes a batch that is already visible"
    )
    message = str(raised.value)
    assert "ffffffff-ffff-ffff-ffff-ffffffffffff" in message
    assert "must not be retried" in message
    # The publish really did land: this is not a loss dressed up as an anomaly.
    assert client.watermarks == [42]


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
    from dmi.storage.capture.clickhouse_sql import DECIDING_READ as _DECIDING_READ

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
    assert _settings("ORDER BY term DESC", "SELECT") == [_DECIDING_READ] * 3
    assert _settings("acquired_at_ns, expires_at_ns", "SELECT") == [_DECIDING_READ] * 3
    assert _settings("dmi_snapshot_manifest", "SELECT") == [_DECIDING_READ]
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
    assert _settings("publisher_lease` (term", "INSERT") == [None] * 3


def test_the_fenced_release_resolves_the_head_with_sequential_consistency():
    """The tombstone's in-statement head read decides who gets revoked.

    The release resolves the lease head inside the INSERT ... SELECT and
    fences on it. Answered from a replica that has not fetched a successor's
    takeover row, that subquery resolves this writer's lapsed lease as the
    head, passes the ``lease_id`` fence, and tombstones the successor's live
    term -- so the read carries the same consistency every other deciding
    read does.
    """
    from dmi.storage.capture.clickhouse_sql import DECIDING_READ as _DECIDING_READ

    client = _Client()
    writer = _leased(client)

    writer.release_publisher_lease()

    release = next(
        call for call in client.calls
        if call[0].startswith("INSERT") and "SELECT term + 1" in call[0]
    )
    assert release[2].get("settings") == _DECIDING_READ


def test_descriptor_writes_use_the_configured_quorum():
    _, descriptor = _descriptor()
    client = _Client()
    writer = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(insert_quorum=2)
    )

    writer.write_descriptors([descriptor], index_version=42)

    quorum = {
        "insert_quorum": 2,
        "insert_quorum_parallel": 0,
        "insert_quorum_timeout": 5000,
    }
    descriptor_insert = next(
        call for call in client.calls
        if call[0].startswith("INSERT") and "dmi_capture_raw" in call[0]
    )
    assert descriptor_insert[2].get("settings") == quorum


def test_fenced_release_uses_the_configured_quorum():
    client = _Client()
    writer = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(insert_quorum=2)
    )
    writer.acquire_publisher_lease("indexer-a")

    writer.release_publisher_lease()

    release = next(
        call for call in client.calls
        if call[0].startswith("INSERT") and "SELECT term + 1" in call[0]
    )
    assert release[2].get("settings") == {
        "select_sequential_consistency": 1,
        "insert_quorum": 2,
        "insert_quorum_parallel": 0,
        "insert_quorum_timeout": 5000,
    }


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


def _rendered_bytes(name: str, value) -> int:
    from clickhouse_driver.util.escape import escape_params

    rendered = escape_params({name: value}, {"strings_as_bytes": False})[name]
    return len(rendered.encode("utf-8"))


def test_committed_pack_queries_are_chunked_by_rendered_size():
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
    store_id = "s" * 255
    identities = [(store_id, str(UUID(int=value))) for value in range(1, 1001)]

    writer.committed_pack_ids(identities)

    chunks = [
        params["identities"]
        for query, params, _ in client.calls
        if "IN %(identities)s" in query
    ]
    assert len(chunks) > 1
    assert max(_rendered_bytes("identities", chunk) for chunk in chunks) <= 192 * 1024


def test_manifest_inserts_are_chunked_by_rendered_size():
    client = _Client()
    writer = _leased(client)
    store_id = "s" * 255
    refs = [
        PackRef(
            str(UUID(int=value)),
            store_id,
            f"packs/{value}.dmi-pack",
            1,
            "checksum",
            1,
        )
        for value in range(1, 1001)
    ]

    writer.publish_snapshot(
        index_version=1,
        refs=refs,
        published_at_ns=1,
        indexed_rows=1000,
        indexed_packs=1000,
    )

    chunks = [
        params["members"]
        for query, params, _ in client.calls
        if query.startswith("INSERT") and "dmi_snapshot_manifest" in query
    ]
    assert len(chunks) > 1
    assert max(_rendered_bytes("members", chunk) for chunk in chunks) <= 192 * 1024


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


def test_a_fresh_install_refuses_without_complete_catalog_visibility():
    class _NoCatalogVisibility(_Client):
        def execute(self, query, params=None, **kwargs):
            if query.startswith("CHECK GRANT SHOW TABLES"):
                self.calls.append((query, params, kwargs))
                return [(0,)]
            return super().execute(query, params, **kwargs)

    client = _NoCatalogVisibility()

    with pytest.raises(CatalogSchemaVersionError, match="SHOW TABLES"):
        _writer(client).ensure_schema()

    assert not any(
        call[0].startswith(("CREATE TABLE", "CREATE VIEW", "ALTER", "INSERT"))
        for call in client.calls
    )


def test_catalog_visibility_needs_no_database_wide_grant():
    class _ObjectOnlyVisibility(_Client):
        def execute(self, query, params=None, **kwargs):
            if query.startswith("CHECK GRANT SHOW TABLES") and query.endswith(".*"):
                self.calls.append((query, params, kwargs))
                return [(0,)]
            return super().execute(query, params, **kwargs)

    _writer(_ObjectOnlyVisibility()).ensure_schema()


def test_an_existing_catalog_hidden_by_one_grant_names_the_grant():
    """A missing grant is not a missing table, and must not read as one.

    `system.tables` is grant-filtered per role, so an object this role holds no
    privilege on is absent from the answer every compatibility verdict is read
    off -- indistinguishable there from an object that was dropped. Verified
    first, that read refused a healthy stamped catalog as "missing
    `dmi_snapshot_manifest` ... drop ALL of its objects", so the visibility
    check has to be reached before any verdict is drawn.
    """

    class _OneObjectHidden(_Client):
        def execute(self, query, params=None, **kwargs):
            if query.startswith("CHECK GRANT SHOW TABLES") and (
                "dmi_snapshot_manifest" in query
            ):
                self.calls.append((query, params, kwargs))
                return [(0,)]
            rows = super().execute(query, params, **kwargs)
            if "system.tables" in query:
                return [row for row in rows if row[0] != "dmi_snapshot_manifest"]
            return rows

    # Stamped at this version and holding rows: a live catalog, not an
    # unfinished install, so nothing else excuses the missing-table verdict.
    client = _OneObjectHidden(
        tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION, data_rows=1
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "lacks SHOW TABLES" in message
    assert "dmi_snapshot_manifest" in message
    assert "missing" not in message
    assert "drop ALL of its objects" not in message


def test_a_fresh_install_requires_every_created_object_to_be_visible():
    class _PartialCatalogVisibility(_Client):
        def execute(self, query, params=None, **kwargs):
            rows = super().execute(query, params, **kwargs)
            if "system.tables" in query:
                return [row for row in rows if row[0] == "dmi_schema_version"]
            return rows

    client = _PartialCatalogVisibility()

    with pytest.raises(CatalogSchemaVersionError, match="not visible"):
        _writer(client).ensure_schema()

    assert not any(
        call[0].startswith("INSERT") and "dmi_schema_version" in call[0]
        for call in client.calls
    ), "an incomplete catalog must not be stamped"


def test_an_existing_catalog_runs_the_inventory_check_once():
    client = _Client(tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION)

    _writer(client).ensure_schema()

    checks = [
        call for call in client.calls
        if "dmi_pack_inventory_raw` FINAL" in call[0] and "NOT IN" in call[0]
    ]
    assert len(checks) == 1


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
    # The last statement that writes anything. The complete object set is
    # validated immediately before this stamp.
    assert not any(
        item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
        for item in statements[stamp + 1 :]
    ), statements[stamp + 1 :]
    assert client.calls[stamp][1]["version"] == _SCHEMA_VERSION
    # Conditional server-side, so a rerun against a stamped catalog inserts
    # nothing and no read-then-write window exists between the two.
    assert (
        "WHERE (SELECT count() FROM `default`.`dmi_schema_version`) = 0"
    ) in statements[stamp]
    # And the table itself is created first, so an install interrupted below
    # leaves a catalog that reads as "this build, unfinished" rather than one
    # indistinguishable from version 1 and refused forever.
    schema_create = (
        "CREATE TABLE IF NOT EXISTS `default`.`dmi_schema_version` (\n"
        "version UInt32, applied_at_ns UInt64\n"
        ") ENGINE = MergeTree ORDER BY version"
    )
    assert statements.index(schema_create) == next(
        index for index, item in enumerate(statements)
        if item.startswith("CREATE TABLE")
    )
    # And the database is the first statement that changes the server at all:
    # everything before it -- the state read and the grant probes -- is a read.
    assert next(
        item
        for item in statements
        if item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
    ).startswith("CREATE DATABASE")
    assert _past_the_visibility_checks(client)[0].startswith("CREATE DATABASE")


def _created_objects(client) -> set[str]:
    """Every `{prefix}_` object the DDL just created, read off the statements.

    `CREATE ... IF NOT EXISTS` is a no-op against a live object, so the writer
    issues the whole DDL every time; what this asserts is that the whole DDL is
    what it issues.
    """
    created = set()
    for query, _, _ in client.calls:
        if not query.startswith("CREATE"):
            continue
        for name in _CURRENT_OBJECTS:
            if f"`{name}`" in query.split(" AS ")[0]:
                created.add(name)
    return created


def _findings(message: str) -> list[str]:
    """The bulleted differences a refusal claims it found."""
    return [
        line.strip()[2:] for line in message.splitlines() if line.startswith("  - ")
    ]


def _past_the_visibility_checks(client) -> list[str]:
    """Every statement past the state read, minus the grant probes.

    `ensure_schema` asks `CHECK GRANT SHOW TABLES` for every object it owns,
    current and superseded, before it draws any conclusion from the state it
    read: `system.tables` is grant-filtered per role, so an object this role
    holds no privilege on reads there exactly like one that was dropped, and a
    verdict formed first refuses a healthy catalog and names a rebuild. The
    probes are reads and are always issued, so the tests below that pin
    "nothing else happened on the way to this refusal" state them once here --
    exactly, in the order `ensure_schema` owns its objects -- rather than
    repeating ten statements each.
    """
    statements = [call[0] for call in client.calls[1:]]
    assert [item for item in statements if item.startswith("CHECK GRANT")] == [
        f"CHECK GRANT SHOW TABLES ON `default`.`{name}`"
        for name in _CURRENT_OBJECTS + ("dmi_pack_commit_log",)
    ]
    return [item for item in statements if not item.startswith("CHECK GRANT")]


def test_a_version_one_catalog_is_refused_with_the_rebuild_procedure():
    """The upgrade this branch cannot perform, refused instead of half-done."""
    client = _Client(
        tables=_VERSION_ONE_OBJECTS,
        sort_key=_VERSION_ONE_SORT_KEY,
        publish_id_tables=(),
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "carries no schema stamp" in message
    assert f"requires version {_SCHEMA_VERSION}" in message
    findings = _findings(message)
    # Both incompatible changes are named, because an operator reading this has
    # to know the catalog cannot simply be altered into shape -- and both are
    # named as this catalog's own state, not as version 1's by definition.
    assert any(
        "ORDER BY" in item and _VERSION_ONE_SORT_KEY in item for item in findings
    )
    assert any(
        "membership is in `dmi_pack_commit_log`" in item
        and "`dmi_snapshot_manifest`, which is absent" in item
        for item in findings
    )
    # And the exact recovery, including the part that is easy to skip.
    assert "CatalogReconciler.rebuild()" in message
    assert "Dropping the pack inventory is mandatory" in message
    assert "`default`.`dmi_pack_inventory_raw`" in message
    # Nothing was created, altered or written on the way to the refusal. The
    # only extra statement is the `system.columns` probe the diagnosis reads.
    assert _past_the_visibility_checks(client) == [
        "SELECT table FROM system.columns WHERE database = %(database)s "
        "AND table IN %(tables)s AND name = 'publish_id'"
    ]


def test_a_stamped_catalog_with_the_wrong_descriptor_sort_key_is_refused():
    """The stamp is not evidence about the table it stands next to.

    Every check on the descriptor sort key sat inside `_unstamped_diagnosis`,
    i.e. on the path taken only when the stamp table is ABSENT. Reach
    `ensure_schema()` with the stamp present and saying this version, and the
    sort key was never looked at -- though `_catalog_state()` already reads
    `sorting_key` for exactly this comparison.

    A partial restore is enough to get there: `{prefix}_capture_raw` recovered
    from a pre-`(store_id, pack_id)` backup beside a surviving stamp. Nothing
    refuses it, `CREATE TABLE IF NOT EXISTS` cannot alter it, and the
    descriptor table this branch exists to protect goes back to a key on which
    ReplacingMergeTree collapses two packs' rows for one capture into one --
    silently, on the next merge.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        sort_key=_VERSION_ONE_SORT_KEY,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    # Named as this catalog's own state and as what this build requires, the
    # way every other refusal here names both sides.
    assert _VERSION_ONE_SORT_KEY in message
    assert _CURRENT_SORT_KEY in message
    assert "CatalogReconciler.rebuild()" in message
    # And refused before the DDL, like every other incompatibility: a
    # `CREATE ... IF NOT EXISTS` pass over this catalog would report success
    # while leaving the wrong key in place.
    assert not [
        item
        for item, *_ in client.calls
        if item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
    ]


def test_a_stamped_catalog_whose_descriptor_engine_drops_the_version_is_refused():
    """A correct sort key on the wrong engine collapses rows just the same.

    The compatibility read captured `engine` and `sorting_key` only, and
    `engine` is `ReplacingMergeTree` for both `ReplacingMergeTree(index_version)`
    and a bare `ReplacingMergeTree()`. Only the first keeps the row with the
    highest `index_version` when a merge collapses a duplicate key; the second
    keeps an arbitrary one. So a stamped catalog whose descriptor table had lost
    the version argument passed every check this branch added -- the sort key
    was right, which is all anything looked at -- and then resolved a capture to
    whichever pack a merge happened to keep.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        engine_full={
            "dmi_capture_raw": (
                "ReplacingMergeTree() ORDER BY (tenant_id, experiment_id, "
                "run_id, captured_at_ns, capture_id, store_id, pack_id) "
                "SETTINGS index_granularity = 8192"
            )
        },
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    # Both sides named, as every other refusal here names them.
    assert "ReplacingMergeTree()" in message
    assert "ReplacingMergeTree(index_version)" in message
    assert "dmi_capture_raw" in message
    assert "CatalogReconciler.rebuild()" in message
    assert not [
        item
        for item, *_ in client.calls
        if item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
    ]

def test_an_unstamped_catalog_is_not_told_its_sort_key_is_wrong_when_it_is_not():
    """Defect A: the refusal that named two facts neither of which was true.

    This branch's own immediate predecessor (`bea3ed8`) already creates the
    descriptor table with `(store_id, pack_id)` on the sort key, writes
    membership to `{prefix}_snapshot_manifest`, and creates no commit log --
    and it stamps nothing, because the stamp arrived one commit later. The
    first refusal read the absent stamp as "version 1" and recited version 1's
    two differences at it. An operator who checks those two facts finds both
    false, concludes the guard is broken, and goes around it -- onto the path
    the guard exists to prevent.

    So the diagnosis is read off the live schema, and where a difference is
    NOT there it says so.
    """
    client = _Client(
        tables=(
            "dmi_capture",
            "dmi_pack_inventory",
            "dmi_capture_raw",
            "dmi_pack_inventory_raw",
            "dmi_capture_version_claims",
            "dmi_index_watermark",
            "dmi_snapshot_manifest",
        ),
        publish_id_tables=(),
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    findings = _findings(message)
    assert "schema version 1" not in message
    # The sort key it really has, described as already correct.
    assert any(
        _CURRENT_SORT_KEY in item and "NOT what is wrong" in item
        for item in findings
    ), findings
    # No commit-log migration is described, because there is no commit log.
    assert any(
        "membership is already in `dmi_snapshot_manifest`" in item
        and "NO commit-log migration owed" in item
        for item in findings
    ), findings
    assert not any("membership is in `dmi_pack_commit_log`" in x for x in findings)
    # What IS different: the publish identity columns and two absent tables.
    assert any(
        "no `publish_id` column" in item
        and "dmi_index_watermark" in item
        and "dmi_snapshot_manifest" in item
        for item in findings
    ), findings
    assert any(
        "dmi_publisher_lease" in item and "dmi_schema_version" in item
        for item in findings
    ), findings
    # Refused all the same, and the reason for refusing anyway is stated.
    assert "refused whichever version it is" in message
    assert "cannot know that the differences listed are all of them" in message


def test_an_unstamped_catalog_whose_shape_already_matches_is_still_refused():
    """Everything this build can probe agrees, and it is refused anyway.

    An unstamped catalog is one this build did not create. The probes compare
    object names, one sort key and one column; column types, view definitions,
    codecs and skip indices are not compared at all, so "the differences listed
    are all of them" is not something this build is in a position to claim.
    """
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != "dmi_schema_version"]
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    findings = _findings(str(raised.value))
    assert any("already carry `publish_id`" in item for item in findings), findings
    # `absent` holds what is NOT here, so the sentence has to say so. The
    # singular branch read ", which is present" -- and this assertion used to
    # spell that out as the first half of an `or` whose second half matched any
    # sentence naming the table, so it codified the defect AND could not have
    # failed on it. No escape hatch now.
    assert (
        "this build also creates `dmi_schema_version`, which is not present"
        in findings
    ), findings
    assert "refused whichever version it is" in str(raised.value)


def test_an_unstamped_catalog_with_no_descriptor_table_says_so():
    """A membership table with no descriptors behind it, named as it is."""
    client = _Client(tables=("dmi_snapshot_manifest", "dmi_pack_commit_log"))

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    findings = _findings(str(raised.value))
    assert any(
        "`dmi_capture_raw` is absent" in item for item in findings
    ), findings
    # Both membership tables are there, so neither is described as the one.
    assert any(
        "both membership tables are present" in item for item in findings
    ), findings


def test_an_unstamped_catalog_with_no_membership_table_at_all_says_so():
    """Neither membership table exists, so no migration is described."""
    client = _Client(tables=("dmi_capture_raw",), publish_id_tables=())

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    findings = _findings(str(raised.value))
    assert any(
        "neither `dmi_pack_commit_log` nor `dmi_snapshot_manifest` is present"
        in item
        for item in findings
    ), findings
    # And with no watermark and no manifest there is nothing to say about
    # publish identity, so nothing is said.
    assert not any("publish_id" in item for item in findings), findings


def test_a_lone_version_one_commit_log_is_named_rather_than_misdiagnosed():
    """Defect B: one inert leftover wedged every start after a clean rebuild.

    `{prefix}_pack_commit_log` is probed but is not one of this build's
    objects, so it could never be reported missing and never be named. Left
    behind by a drop that missed it, it made the catalog non-empty (so not a
    fresh install) and unstamped (so an old schema), and the refusal then
    prescribed the rebuild the operator had just finished, over tables that no
    longer exist. Nothing could be created and no indexer could start until
    somebody noticed one table nothing reads.
    """
    client = _Client(tables=("dmi_pack_commit_log",))

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "holds none of the objects this build creates" in message
    assert "the only object present is `default`.`dmi_pack_commit_log`" in message
    assert "Drop it and run ensure_schema() again" in message
    assert "nothing to rebuild first" in message
    # Emphatically NOT the version-1 recital, and not the rebuild procedure:
    # there is nothing left to rebuild.
    assert "CatalogReconciler.rebuild()" not in message
    assert "carries no schema stamp" not in message
    # Nothing was created or written on the way to the refusal.
    assert _past_the_visibility_checks(client) == []


def test_an_object_of_the_wrong_kind_is_named_rather_than_left_to_the_server():
    """A table standing where a view belongs fails the DDL halfway through.

    `CREATE OR REPLACE VIEW` cannot replace a table, so without this the run
    gets several statements in before ClickHouse objects, and the error names
    SQL rather than the catalog.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        engines={"dmi_capture": "MergeTree"},
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert (
        "`dmi_capture` is a TABLE (engine MergeTree) where this build creates "
        "a VIEW"
    ) in message
    assert _past_the_visibility_checks(client) == []


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


def test_an_unstamped_install_missing_a_table_is_still_refused():
    """An empty stamp narrows the version; it does not excuse a missing table.

    Returning as soon as the stamp table held no row skipped both remaining
    checks -- and clearing the stamp is the obvious operator workaround for a
    refusal, which makes this the most likely route into the state the guard
    exists to prevent. Measured against 25.12: truncating the stamp on a
    version 3 catalog made `ensure_schema` accept a catalog with no
    `{prefix}_publisher_lease` and re-stamp it as 4, performing the in-place
    upgrade this design says is never performed; on a version 2 catalog the DDL
    then died mid-way with `Code: 47` over `publish_id` and left the catalog
    half written.
    """
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != "dmi_publisher_lease"],
        schema_version=None,
        # A catalog that KEPT its rows, which is what makes recreating a table
        # empty beside them dangerous rather than merely unfinished.
        data_rows=1,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "missing `dmi_publisher_lease`" in message
    # And it says which state it is in, rather than claiming a stamp it does
    # not have -- a refusal an operator can check and find false gets worked
    # around, onto the path it exists to prevent.
    assert "holds `dmi_schema_version` with no row in it" in message
    assert f"stamped schema version {_SCHEMA_VERSION}" not in message
    assert "CatalogReconciler.rebuild()" in message
    assert not any(
        call[0].startswith(("CREATE TABLE", "CREATE OR REPLACE", "ALTER"))
        for call in client.calls
    ), "the DDL ran against a catalog this build cannot complete"


def test_an_unstamped_install_with_an_empty_manifest_is_still_refused():
    """The catalog-hiding state, reached by clearing the stamp.

    Same early return, other check: an inventory with rows beside an empty
    manifest reports every pack as indexed and admits none to a snapshot.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=None,
        inventory_without_membership=4,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    assert "remain invisible" in str(raised.value)


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
        data_rows=1,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    assert "missing `dmi_snapshot_manifest`" in str(raised.value)
    assert "CatalogReconciler.rebuild()" in str(raised.value)


def test_a_stamped_catalog_with_a_dropped_view_recreates_it():
    """Defect C: a view is derived, so losing one is not a data emergency.

    Every object under this prefix is derived from the packs, but a view is
    derived from the TABLES -- `ensure_schema` recreates it outright and the
    result cannot disagree with rows that survived. Demanding a full rebuild,
    which re-reads every pack footer in the store and leaves readers on an
    empty and then partial catalog while it runs, is a cost with no risk
    behind it. A missing TABLE stays refused, because recreating it empty
    beside a surviving inventory is the state that hides every capture.
    """
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != "dmi_capture"],
        schema_version=_SCHEMA_VERSION,
        inventory_without_membership=0,
    )

    _writer(client).ensure_schema()

    # The DROPPED view is recreated, which "some CREATE OR REPLACE VIEW ran"
    # could not say: `ensure_schema` issues one on every call, so that
    # assertion held for any input that did not raise.
    assert _created_objects(client) == set(_CURRENT_OBJECTS)
    assert client.calls[-1][0].startswith("INSERT INTO `default`.`dmi_schema_version`")


@pytest.mark.parametrize("dropped", _VIEWS)
def test_either_missing_view_is_recreated_rather_than_refused(dropped: str):
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != dropped],
        schema_version=_SCHEMA_VERSION,
    )

    _writer(client).ensure_schema()

    created = [
        call[0]
        for call in client.calls
        if call[0].startswith("CREATE") and f"`default`.`{dropped}` AS" in call[0]
    ]
    assert len(created) == 1 and "VIEW" in created[0]


def test_a_populated_inventory_with_an_empty_manifest_is_refused():
    """The defect this whole check exists for, in its purest form.

    `committed_pack_ids` reads the inventory and reports every pack as already
    indexed; readers bound their snapshot by the manifest, which names none of
    them. Every capture in those packs is durable and invisible, and an
    indexing pass over them reports success.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        inventory_without_membership=4,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "dmi_pack_inventory_raw" in message and "dmi_snapshot_manifest" in message
    assert "remain invisible" in message


def test_partial_membership_loss_is_refused():
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        inventory_without_membership=1,
    )

    with pytest.raises(CatalogSchemaVersionError, match="report success"):
        _writer(client).ensure_schema()


def test_a_catalog_at_this_builds_version_is_accepted():
    """Nothing indexed yet is not the same state as membership gone missing.

    This used to assert only that SOME `CREATE OR REPLACE VIEW` was issued,
    which `ensure_schema` does on every call. What it is for is that the full
    DDL runs and the stamp is rewritten, idempotently, over a catalog already
    at this version.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        inventory_without_membership=0,
    )

    _writer(client).ensure_schema()

    assert _created_objects(client) == set(_CURRENT_OBJECTS)
    # The stamp is written last and is conditional server-side, so a rerun
    # against a stamped catalog inserts nothing without a read-then-write
    # window.
    stamp = client.calls[-1]
    assert stamp[0].startswith("INSERT INTO `default`.`dmi_schema_version`")
    assert "WHERE (SELECT count() FROM `default`.`dmi_schema_version`) = 0" in stamp[0]
    assert stamp[1]["version"] == _SCHEMA_VERSION



# --- catalog facets ---------------------------------------------------------


def test_ensure_schema_declares_every_facet_as_a_materialized_column():
    from dmi.storage.capture.clickhouse_schema import FACET_COLUMNS as _FACET_COLUMNS

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
    from dmi.storage.capture.clickhouse_schema import FACET_COLUMNS as _FACET_COLUMNS

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
    from dmi.storage.capture.clickhouse_schema import (
        CAPTURE_COLUMNS as _CAPTURE_COLUMNS,
        FACET_COLUMNS as _FACET_COLUMNS,
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


# --- the rebuild drop list ---------------------------------------------------
#
# The rebuild is the only supported recovery, and it is only a recovery if it
# names EVERY object. An object left behind is not inert: a surviving pack
# inventory makes the next pass skip every pack it lists, and a surviving
# superseded table makes the next start refuse a catalog that is otherwise
# empty. This branch has leaked tables twice from exactly that omission, once
# in the code and once in the prose, so both lists are checked against
# `self._objects` mechanically rather than read over by eye.


def _writer_objects() -> tuple[str, ...]:
    """Every object the writer owns, in drop order, superseded ones last."""
    writer = _writer(_Client())
    return tuple(
        name for _, name in writer._objects + writer._legacy_objects
    )


def _documented_drop_list() -> tuple[str, ...]:
    """Step 2 of the documented rebuild, as an ordered list of object names."""
    from pathlib import Path
    import re

    document = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "capture-storage-design.md"
    ).read_text()
    step = re.search(
        r"^2\. Drop all of its objects.*?(?=^\d+\. )",
        document,
        re.MULTILINE | re.DOTALL,
    )
    assert step is not None, "the documented rebuild lost its drop step"
    return tuple(
        f"dmi_{suffix}" for suffix in re.findall(r"`\{prefix\}_(\w+)`", step.group(0))
    )


def test_the_rebuild_instruction_names_every_object_the_writer_owns():
    instruction = _writer(_Client())._rebuild_instruction()

    named = [name for name in _writer_objects() if f"`default`.`{name}`" in instruction]

    assert named == list(_writer_objects())


def test_the_documented_rebuild_drops_exactly_what_the_writer_owns():
    """The prose and the code have to name the same objects, in the same order.

    `docs/capture-storage-design.md` is the procedure an operator follows by
    hand; `_rebuild_instruction()` is the one the refusal prints. A table added
    to the writer and forgotten in either is a table that survives the rebuild,
    and the states that survive a rebuild are the ones this whole check exists
    to refuse.
    """
    assert _documented_drop_list() == _writer_objects()


def test_an_install_interrupted_between_two_creates_is_completed_not_refused():
    """The reason the stamp table is created first, made true.

    `ensure` writes `{prefix}_schema_version` before anything else so that an
    install interrupted partway leaves a catalog that says "this build,
    unfinished" rather than one refused forever. The missing-table check ran
    regardless of whether anything survived, so it refused that catalog anyway
    -- reciting a surviving pack inventory that was itself one of the MISSING
    tables -- and the only recovery it offered was a manual drop of every
    object. With no data anywhere there is nothing to hide and nothing to
    lose: re-running the DDL is the completion.
    """
    client = _Client(
        tables=("dmi_schema_version", "dmi_capture_raw"),
        schema_version=None,
    )

    _writer(client).ensure_schema()

    assert any(
        call[0].startswith("CREATE TABLE") for call in client.calls
    ), "the interrupted install was never completed"


def test_a_missing_table_beside_surviving_rows_is_still_refused():
    """The other side of the same gate: rows change the answer.

    One row anywhere in the data tables makes recreating a missing one empty
    the dangerous state again -- a surviving inventory makes the next pass skip
    every pack it lists, and nothing refills what was dropped.
    """
    client = _Client(
        tables=[name for name in _CURRENT_OBJECTS if name != "dmi_pack_inventory_raw"],
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    assert "missing `dmi_pack_inventory_raw`" in str(raised.value)


def test_a_legacy_object_beside_this_builds_catalog_is_refused():
    """Two builds sharing one prefix, caught at the only moment it is cheap.

    The pre-PR build's `ensure_schema` is all CREATE ... IF NOT EXISTS, so it
    no-ops over a version 4 catalog and recreates its own
    `{prefix}_pack_commit_log` beside it. Everything after that is silent: its
    publish writes the pack inventory and an unconditional watermark row and
    never a manifest row, so every pack it touches is recorded as indexed and
    admitted by no snapshot -- invisible to every reader and skipped by every
    later pass, rebuilds included.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS + ("dmi_pack_commit_log",),
        schema_version=_SCHEMA_VERSION,
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "dmi_pack_commit_log" in message
    assert "an earlier build" in message
    assert "CatalogReconciler.rebuild()" in message
    assert not any(
        call[0].startswith(("CREATE TABLE", "CREATE OR REPLACE", "ALTER"))
        for call in client.calls
    ), "the DDL ran beside a writer this build cannot coexist with"


def test_drop_schema_drops_every_object_whatever_kind_it_is():
    """The teardown has to work on the catalog that most needs it.

    `_reject_wrong_kinds` refuses a prefix where a view stands as a table and
    then prescribes `rebuild_instruction`, whose only implementation is this
    method. `DROP VIEW` against a table is refused by ClickHouse even with IF
    EXISTS (Code 80 on 25.12), so a kind-specific drop aborted on its first
    statement against exactly that catalog. `DROP TABLE` removes both kinds.
    """
    client = _Client()

    _writer(client).drop_schema()

    drops = [call[0] for call in client.calls if call[0].startswith("DROP")]
    assert drops, "nothing was dropped"
    assert all(drop.startswith("DROP TABLE IF EXISTS") for drop in drops)
    # Views still go first: a table dropped out from under a surviving view
    # leaves a broken projection for as long as the teardown takes.
    assert "dmi_capture`" in drops[0]
    assert len(drops) == len(_CURRENT_OBJECTS) + 1  # + the legacy commit log


def test_collect_garbage_deletes_only_within_its_stated_bounds():
    """The predicates, pinned. Widening one by a comparison is the whole risk.

    Each bound below is what makes a deletion safe, and none of them is
    observable in CI any other way: `ALTER TABLE ... DELETE` reports nothing,
    and the live test that exercises the behaviour is not collected here. So
    the statements themselves are the assertion.
    """
    client = _Client()
    writer = _leased(client)

    writer.collect_garbage()

    deletes = {
        query.split("`")[3]: query
        for query, _, _ in client.calls
        if query.startswith("ALTER TABLE") or (
            query.startswith("SELECT count()") and " WHERE " in query
        )
    }
    counted = [
        (query, params)
        for query, params, _ in client.calls
        if query.startswith("SELECT count() FROM `") and " WHERE " in query
    ]
    by_table = {query.split("`")[3]: (query, params) for query, params in counted}

    manifest_query, manifest_params = by_table["dmi_snapshot_manifest"]
    # Strictly BELOW the published head: an in-flight publish always sits above
    # it, so this cannot delete membership out from under one.
    assert "index_version < %(published)s" in manifest_query
    # And only rows no watermark row pairs with.
    assert "NOT IN (SELECT index_version, publish_id FROM" in manifest_query
    assert manifest_params["published"] == writer.last_published_version()

    lease_query, _ = by_table["dmi_publisher_lease"]
    # Strictly below the head TERM, so the row the fence resolves survives
    # whether or not it has expired.
    assert "term < %(term)s" in lease_query

    claims_query, claims_params = by_table["dmi_capture_version_claims"]
    # At or below the published head: the watermark keeps the allocator's floor
    # once these are gone, and a claim ABOVE the head may be a version a pass
    # has allocated and not yet published.
    assert "version <= %(published)s" in claims_query
    assert claims_params["published"] == writer.last_published_version()
    # The watermark table is never collected -- it IS the floor.
    assert "dmi_index_watermark" not in deletes
