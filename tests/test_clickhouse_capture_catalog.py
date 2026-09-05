from __future__ import annotations

import os
import threading
import time
import warnings
from uuid import UUID, uuid4

import pytest

from dmi.storage.capture import (
    PublisherLeaseHeldError,
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
_CURRENT_PACK_SORT_KEY = "store_id, pack_id"

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
        pack_sort_key=_CURRENT_PACK_SORT_KEY,
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
        # `(index_version, publish_id)` pairs the retention pass's initiator-
        # side read reports as orphaned: below the head with no watermark row.
        # Each is modelled as one manifest row.
        self.orphan_publishes: list[tuple[int, str]] = []
        self.lease = FakeLeaseTable()
        self.tables = tuple(tables)
        self.schema_version = schema_version
        self.inventory_without_membership = inventory_without_membership
        self.data_rows = data_rows
        self.sort_key = sort_key
        self.pack_sort_key = pack_sort_key
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
            key = self.sort_key if name == "dmi_capture_raw" else self.pack_sort_key
            return (
                "ReplacingMergeTree(index_version) "
                f"ORDER BY ({key or 'store_id, pack_id'}) "
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
            # Both ReplacingMergeTree tables answer a real sort key: a merge
            # collapses rows on it, so the inventory's key is load-bearing in
            # the same way the descriptors' is, and a fake that answered a
            # placeholder there left the check on it untestable. Everything
            # else answers a placeholder rather than a fiction that looks
            # meaningful. `engine_full` is appended only when the statement
            # asks for it, so this fake answers the read either shape.
            keys = {
                "dmi_capture_raw": self.sort_key,
                "dmi_pack_inventory_raw": self.pack_sort_key,
            }
            rows = [
                (name, self._engine(name), keys.get(name, ""))
                for name in self.tables
            ]
            if "engine_full" in query:
                return [row + (self._engine_full(row[0]),) for row in rows]
            return rows
        if "dmi_schema_version` ORDER BY version DESC" in query:
            return [] if self.schema_version is None else [(self.schema_version,)]
        if "dmi_pack_inventory_raw` FINAL" in query and "NOT IN" in query:
            return [(self.inventory_without_membership,)]
        if "snapshot_manifest" in query and (
            "NOT IN (SELECT index_version, publish_id FROM" in query
        ):
            # The retention pass's initiator-side read of orphaned membership.
            # The anti-join lives HERE, in a plain SELECT, and never in the
            # mutation that follows it.
            return list(self.orphan_publishes)
        if query.startswith("SELECT count() FROM `") and " WHERE " in query:
            # The retention pass counts what a predicate matches before it
            # deletes exactly those rows. Literal pair deletions match one
            # modelled row per pair; the other predicates match nothing, so
            # the fake never issues those mutations and the test can assert
            # on the predicates.
            if "IN %(pairs)s" in query:
                return [(len(params["pairs"]),)]
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
            # The chunk read-back names its members; the whole-publish check
            # after the watermark lands does not, and counts every distinct
            # pack the publish still has membership for.
            wanted = set(params["members"]) if "members" in params else None
            found = {
                (store_id, pack_id)
                for version, publish_id, store_id, pack_id in self.manifest
                if version == params["index_version"]
                and publish_id == params["publish_id"]
                and (wanted is None or (store_id, pack_id) in wanted)
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
    assert _settings("GROUP BY term, lease_id", "SELECT") == [_DECIDING_READ] * 3
    assert _settings("acquired_at_ns, expires_at_ns", "SELECT") == [_DECIDING_READ] * 3
    # The chunk read-back, and the whole-publish confirmation after the
    # watermark row stands.
    assert _settings("dmi_snapshot_manifest", "SELECT") == [_DECIDING_READ] * 2
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


def test_the_release_tombstone_is_written_at_the_writers_own_term_and_reads_nothing():
    """A release cannot race a successor for a term it never reads.

    The earlier release resolved the head inside an ``INSERT ... SELECT`` and
    fenced on it, which was not a compare-and-set against a concurrent claim:
    a stale release could land its expired row at a successor's freshly
    granted term. The tombstone now goes to the term THIS writer was granted,
    server-stamped and already expired, with no head read to be stale about.
    """
    client = _Client()
    writer = _leased(client)
    granted = writer.publisher_lease.term

    writer.release_publisher_lease()

    release = next(
        call for call in client.calls
        if call[0].startswith("INSERT") and "now_ns, now_ns FROM" in call[0]
    )
    statement, params, kwargs = release
    assert params["term"] == granted
    assert "publisher_lease" not in statement.split("FROM", 1)[1], (
        "the release must not read the lease table"
    )
    assert "%(ttl_ns)s" not in statement, "a tombstone buys no lease life"
    # Nothing to read, so nothing to read consistently: the only settings a
    # release carries are the deployment's write quorum, none by default.
    assert kwargs.get("settings") is None


def _publish(writer: ClickHouseCatalogWriter, version: int) -> None:
    writer.publish_snapshot(
        index_version=version,
        refs=(),
        published_at_ns=version,
        indexed_rows=0,
        indexed_packs=0,
    )


def test_two_concurrent_publishes_on_one_writer_are_serialised():
    """One publish in flight per writer, and so per lease.

    The fence identifies a HOLDER, not an operation. Two publishes issued under
    one ``lease_id`` both renew it and both pass the fence, and the version
    barrier inside each statement is evaluated when that statement is admitted
    rather than against the other, so on a real server both watermark rows
    land in either order and a reader pinned at the higher one watches the
    lower version's membership arrive underneath it (300 of 300 rounds on
    ClickHouse 25.12; 57 exposed the higher watermark first). ClickHouse has
    nothing that serialises the two statements, so the writer does.

    The fake cannot model the server-side overlap -- its statements run under
    the interpreter lock -- so the test asserts the thing the writer is
    responsible for: while publisher A is inside a fenced statement, publisher
    B on the same writer issues NOTHING, not even the lease renewal that
    precedes its first write.
    """
    client = _Client()
    writer = _leased(client)
    by_thread: dict[str, list[str]] = {}
    real_execute = client.execute

    def recording_execute(query, params=None, **kwargs):
        by_thread.setdefault(threading.current_thread().name, []).append(query)
        return real_execute(query, params, **kwargs)

    client.execute = recording_execute

    a_is_inside_a_fenced_statement = threading.Event()
    let_a_finish = threading.Event()

    def hold_a_at_the_fence(_lease_id):
        if threading.current_thread().name == "publisher-a":
            a_is_inside_a_fenced_statement.set()
            assert let_a_finish.wait(5), "the test never released publisher A"

    client.lease.on_fence = hold_a_at_the_fence
    failures: list[BaseException] = []

    def run(version: int) -> None:
        try:
            _publish(writer, version)
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    a = threading.Thread(target=run, args=(1,), name="publisher-a")
    b = threading.Thread(target=run, args=(2,), name="publisher-b")
    a.start()
    assert a_is_inside_a_fenced_statement.wait(5)
    b.start()
    # A negative assertion needs a window in which B COULD have run: on the
    # unserialised writer B issues its lease renewal within microseconds of
    # starting, so a quarter second is generous rather than tight.
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline and "publisher-b" not in by_thread:
        time.sleep(0.005)
    issued_by_b_while_a_was_publishing = list(by_thread.get("publisher-b", ()))
    let_a_finish.set()
    a.join(5)
    b.join(5)

    assert not a.is_alive() and not b.is_alive()
    # The load-bearing assertion first, so an unserialised writer fails on the
    # statement B issued rather than on whatever the fake's barrier made of
    # the resulting order.
    assert issued_by_b_while_a_was_publishing == [], (
        "publisher B issued statements while publisher A's publish was in "
        "flight on the same writer; two publishes under one lease both pass "
        "the fence and are not ordered by the server"
    )
    assert failures == []
    # Serialised, both publish -- in the order they took the lock, so the
    # barrier admits each: A at 1, then B at 2 above it.
    assert client.watermarks == [1, 2]
    assert writer.publisher_lease is not None


def test_every_client_operation_waits_behind_a_publish_in_flight():
    """The lock covers the CONNECTION, not only the lease.

    The writer may be shared between the threads of its process, and what its
    methods share is one driver client that is not thread-safe: its only guard
    is a flag check that raises when it happens to notice an overlap. So the
    indexer's unlocked neighbours of a publish -- the inventory read that
    decides what to skip, the descriptor write, the inventory commit, the head
    read -- must wait for a publish in flight on another thread rather than
    interleave their packets with it. Asserted the same way as for a second
    publish: while A is inside a fenced statement, B issues NOTHING.
    """
    ref, descriptor = _descriptor()
    client = _Client()
    writer = _leased(client)
    by_thread: dict[str, list[str]] = {}
    real_execute = client.execute

    def recording_execute(query, params=None, **kwargs):
        by_thread.setdefault(threading.current_thread().name, []).append(query)
        return real_execute(query, params, **kwargs)

    client.execute = recording_execute
    a_is_inside_a_fenced_statement = threading.Event()
    let_a_finish = threading.Event()

    def hold_a_at_the_fence(_lease_id):
        if threading.current_thread().name == "publisher-a":
            a_is_inside_a_fenced_statement.set()
            assert let_a_finish.wait(5), "the test never released publisher A"

    client.lease.on_fence = hold_a_at_the_fence
    failures: list[BaseException] = []
    results: dict[str, object] = {}

    def neighbours() -> None:
        try:
            results["skip"] = writer.committed_pack_ids([(ref.store_id, ref.pack_id)])
            results["head"] = writer.last_published_version()
            writer.write_descriptors([descriptor], index_version=2)
            writer.commit_packs([ref], index_version=2)
            results["gc"] = writer.collect_garbage(sleep=lambda _s: None)
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    a = threading.Thread(target=_publish, args=(writer, 1), name="publisher-a")
    b = threading.Thread(target=neighbours, name="neighbour-b")
    a.start()
    assert a_is_inside_a_fenced_statement.wait(5)
    b.start()
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline and "neighbour-b" not in by_thread:
        time.sleep(0.005)
    issued_by_b_while_a_was_publishing = list(by_thread.get("neighbour-b", ()))
    let_a_finish.set()
    a.join(5)
    b.join(5)

    assert not a.is_alive() and not b.is_alive()
    assert issued_by_b_while_a_was_publishing == [], (
        "a second thread reached the shared client while a publish was in "
        "flight on it; the driver client is not thread-safe"
    )
    assert failures == []
    # Once A is done, every one of B's calls ran to completion against the
    # server: waiting is not refusing.
    assert client.watermarks == [1]
    issued_by_b = by_thread["neighbour-b"]
    assert any("pack_inventory`" in q and q.startswith("SELECT") for q in issued_by_b)
    assert any("max(index_version)" in q for q in issued_by_b)
    assert any(q.startswith("INSERT INTO") and "capture_raw" in q for q in issued_by_b)
    assert any(q.startswith("INSERT INTO") and "pack_inventory_raw" in q for q in issued_by_b)
    # Collection's orphan scan; the fake counts nothing to delete, so no ALTER.
    assert any("snapshot_manifest" in q and "DISTINCT index_version" in q for q in issued_by_b)
    assert results["skip"] == set()
    assert isinstance(results["gc"], dict)


def test_a_publish_that_fails_releases_the_writer_for_the_next_one():
    """The exclusion is scoped to one call, including its failure path."""
    client = _Client()
    writer = _leased(client)
    client.watermarks.append(5)

    with pytest.raises(SnapshotPublishRaceError):
        _publish(writer, 3)  # refused by the barrier: 3 is not above 5

    done = threading.Event()

    def next_publish() -> None:
        _publish(writer, 6)
        done.set()

    publisher_b = threading.Thread(target=next_publish, name="publisher-b")
    publisher_b.start()
    assert done.wait(5), "the failed publish left the writer locked"
    publisher_b.join(5)
    assert client.watermarks == [5, 6]


def test_a_writer_used_from_another_process_refuses_to_publish(monkeypatch):
    """A writer and its lease belong to the process that created them.

    A forked child inherits the parent's ``PublisherLease`` byte for byte, so
    its publishes would carry the parent's ``lease_id`` from a second address
    space that no in-process lock can reach -- the same double-publish, with
    no way to observe it from either side. The writer records its owning PID
    and refuses every lease-bearing operation from any other.
    """
    import os

    from dmi.storage.capture import PublisherLeaseError

    client = _Client()
    writer = _leased(client)
    issued_before_the_fork = len(client.calls)
    owner = os.getpid()
    # Patched on ``os`` itself rather than through the writer's module, so a
    # writer that never consults the PID fails this test with DID NOT RAISE
    # instead of tripping over an attribute the module does not import.
    monkeypatch.setattr(os, "getpid", lambda: owner + 1)

    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        _publish(writer, 1)
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.renew_publisher_lease()
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.acquire_publisher_lease("child")
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.release_publisher_lease()
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.ensure_schema()
    # Allocation too: a version claimed from a forked child would be published
    # under the parent's lease, which is the same double-publish by another
    # route, and the contract names allocate_version explicitly.
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.allocate_version()
    # And every other operation that reaches the server: a forked child shares
    # the parent's SOCKET as well as its lease, and a descriptor write or an
    # inventory read from the child would interleave its packets with whatever
    # the parent has in flight on that connection.
    ref, descriptor = _descriptor()
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.write_descriptors([descriptor], index_version=1)
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.commit_packs([ref], index_version=1)
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.committed_pack_ids([(ref.store_id, ref.pack_id)])
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.last_published_version()
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.collect_garbage(sleep=lambda _s: None)
    with pytest.raises(PublisherLeaseError, match="belong to one process"):
        writer.drop_schema()
    # Nothing reached the server from the wrong process.
    assert client.watermarks == []
    assert client.claims == []
    assert client.committed == []
    assert not any(
        query.startswith(("INSERT", "ALTER", "DROP"))
        for query, _, _ in client.calls[issued_before_the_fork:]
    )
    assert writer.publisher_lease is not None, (
        "the refusal must not drop the parent's lease from under it"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_is_refused_even_when_the_parent_holds_the_lock():
    """The ownership check runs BEFORE the lock, or a fork can hang the child.

    A fork copies the writer's lock in whatever state it was in. Taken while
    another thread of the parent is mid-publish, the child's copy is held by a
    thread that does not exist in the child, so a check placed UNDER the lock
    never runs and the child blocks on the inherited lock forever -- a hang
    where the contract promises a refusal. This holds the lock in a parent
    thread, forks, and requires the child to be refused promptly.
    """
    from dmi.storage.capture import PublisherLeaseError

    client = _Client()
    writer = _leased(client)
    parent_is_inside_a_fenced_statement = threading.Event()
    let_parent_finish = threading.Event()

    def hold_the_lock(_lease_id):
        if threading.current_thread().name == "holder":
            parent_is_inside_a_fenced_statement.set()
            let_parent_finish.wait(10)

    client.lease.on_fence = hold_the_lock
    holder = threading.Thread(
        target=_publish, args=(writer, 1), name="holder", daemon=True
    )
    holder.start()
    assert parent_is_inside_a_fenced_statement.wait(5)
    try:
        read_end, write_end = os.pipe()
        with warnings.catch_warnings():
            # Python 3.12+ warns that forking a multi-threaded process may
            # deadlock the child. A second live thread is the scenario under
            # test, not an accident, and a deadlocked child is exactly what the
            # bounded wait below detects and reports.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # pragma: no cover - the child reports through the pipe
            os.close(read_end)
            try:
                _publish(writer, 2)
                os.write(write_end, b"published")
            except PublisherLeaseError:
                os.write(write_end, b"refused")
            except BaseException as exc:
                os.write(write_end, f"raised {type(exc).__name__}".encode())
            finally:
                os._exit(0)
        os.close(write_end)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            finished, _ = os.waitpid(pid, os.WNOHANG)
            if finished:
                break
            time.sleep(0.01)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail(
                "the forked child hung on the writer lock it inherited held, "
                "instead of being refused for using a foreign process's writer"
            )
        outcome = os.read(read_end, 64).decode()
        os.close(read_end)
    finally:
        let_parent_finish.set()
        holder.join(5)
    assert outcome == "refused"
    # The parent's own publish, held open across the fork, completes untouched.
    assert client.watermarks == [1]
    assert writer.publisher_lease is not None


def test_descriptor_writes_use_the_configured_quorum():
    _, descriptor = _descriptor()
    client = _Client()
    writer = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(insert_quorum=2, clock_skew_ns=1)
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


def test_the_release_tombstone_uses_the_configured_quorum():
    """A deciding WRITE: the successor's head read is what the row is for."""
    client = _Client()
    writer = ClickHouseCatalogWriter(
        client, ClickHouseCatalogConfig(insert_quorum=2, clock_skew_ns=1)
    )
    writer.acquire_publisher_lease("indexer-a")

    writer.release_publisher_lease()

    release = next(
        call for call in client.calls
        if call[0].startswith("INSERT") and "now_ns, now_ns FROM" in call[0]
    )
    assert release[2].get("settings") == {
        "insert_quorum": 2,
        "insert_quorum_parallel": 0,
        "insert_quorum_timeout": 5000,
    }


def test_a_replicated_deployment_must_declare_its_clock_skew_bound():
    """`insert_quorum` says "several hosts"; a zero skew bound says "one clock"."""
    with pytest.raises(ValueError, match="clock_skew_ns must be set alongside"):
        ClickHouseCatalogConfig(insert_quorum=2)


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
    # The last statement that writes to the LAYOUT. The complete object set is
    # validated immediately before this stamp. What may follow it is the
    # initialiser giving back the lease it held around the whole install --
    # a row in the lease table, not an object.
    assert not any(
        item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
        and "publisher_lease" not in item
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


def _never_sleeps(_seconds: float) -> None:
    """A sleep for tests: the retry budget is counted in attempts, not seconds."""


def _lease_claims(client) -> list[str]:
    return [
        query for query, params, _ in client.calls
        if query.startswith("INSERT") and "publisher_lease` (term" in query
        and params is not None and "ttl_ns" in params
    ]


def _lease_releases(client) -> list[str]:
    return [
        query for query, params, _ in client.calls
        if query.startswith("INSERT") and "publisher_lease` (term" in query
        and params is not None and "ttl_ns" not in params
    ]


def test_a_fresh_install_runs_under_the_publisher_lease_from_ddl_to_stamp():
    """Initialisation of a prefix is serialised, not merely re-checked.

    Two initialisers reading an empty prefix both proceeded to their DDL. The
    layout re-check before the stamp turns the ordering the reviewer injected
    into a refusal, but it does not stop two of THIS build's initialisers
    from interleaving their DDL and their stamps. So a fresh install creates
    the stamp table and the lease table -- both idempotent, both this build's
    own -- and then takes the publisher lease before any other DDL, holding it
    until the stamp is written. The lease is the catalog's one mutual-exclusion
    primitive already; nobody can be publishing into a catalog that does not
    exist yet, so taking it costs a fresh install nothing.
    """
    client = _Client()

    _writer(client).ensure_schema()

    statements = [call[0] for call in client.calls]
    claims = _lease_claims(client)
    releases = _lease_releases(client)
    assert len(claims) == 1 and len(releases) == 1
    claim = statements.index(claims[0])
    release = statements.index(releases[0])
    stamp = next(
        index for index, item in enumerate(statements)
        if item.startswith("INSERT") and "dmi_schema_version" in item
    )
    layout = [
        index for index, item in enumerate(statements)
        if item.startswith(("CREATE TABLE", "ALTER", "CREATE OR REPLACE", "CREATE VIEW"))
        and "dmi_schema_version" not in item
        and "dmi_publisher_lease" not in item
    ]
    assert claim < min(layout), "the lease is taken before the layout DDL"
    assert max(layout) < stamp < release, "and held until the stamp is written"
    # Given back: the next publisher does not wait out a TTL for an installer.
    head = client.lease.claimants()
    assert head and head[0][3] <= client.lease.now_ns


def test_a_second_initialiser_waits_for_the_first_rather_than_crashing():
    """A cold fleet start must not lose every process but one.

    Refusing outright made all but one initialiser raise
    `PublisherLeaseHeldError` out of a call every caller treats as infallible
    setup -- and the taxonomy's advice for a lease error, re-acquire and
    re-index, is not the recovery here. So the second waits; when the first
    finishes, the prefix reads complete and the second is simply done.
    """
    client = _Client()
    first = _writer(client)
    second = _writer(client)
    execute = client.execute
    outcome = []

    def start_the_second_initialiser_mid_install(query, params=None, **kwargs):
        result = execute(query, params, **kwargs)
        if query.startswith("CREATE TABLE") and "dmi_capture_raw" in query and not outcome:
            outcome.append("started")
            client.execute = execute
            waits = []

            def the_holder_finishes_while_the_second_waits(_seconds):
                # What a real fleet start looks like from the second process:
                # the holder completes its DDL and its stamp, so the prefix the
                # second re-reads is a complete catalog.
                waits.append(_seconds)
                if len(waits) == 2:
                    client.tables = tuple(_CURRENT_OBJECTS)
                    client.schema_version = _SCHEMA_VERSION

            second.ensure_schema(sleep=the_holder_finishes_while_the_second_waits)
            outcome.append(("waited", len(waits)))
            client.execute = start_the_second_initialiser_mid_install
        return result

    client.execute = start_the_second_initialiser_mid_install

    first.ensure_schema(sleep=_never_sleeps)

    assert outcome[1][0] == "waited" and outcome[1][1] >= 2, (
        "the second initialiser should have waited, not raised"
    )
    # Every stamp is the conditional INSERT, so however many initialisers ran
    # the row is written once by the server.
    stamps = [
        call[0] for call in client.calls
        if call[0].startswith("INSERT") and "dmi_schema_version" in call[0]
    ]
    assert stamps and all(
        "WHERE (SELECT count() FROM `default`.`dmi_schema_version`) = 0" in item
        for item in stamps
    )
    assert len(_lease_claims(client)) == 1, "only the first took the install lease"


def test_an_initialiser_that_waits_out_the_whole_budget_raises():
    """A lease nobody gives back is not an install, and is worth raising about.

    The budget spans one lease TTL plus a margin, which is when a dead holder's
    row is takeable at the latest. Past that, something is renewing it.
    """
    client = _Client()
    holder = _writer(client)
    holder.acquire_publisher_lease("indexer-a")
    waits = []

    with pytest.raises(PublisherLeaseHeldError, match="indexer-a"):
        _writer(client).ensure_schema(sleep=waits.append)

    ttl = ClickHouseCatalogConfig().lease_ttl_ns / 1e9
    assert len(waits) >= ttl / max(waits), "the budget must span a whole TTL"
    assert not [
        call for call in client.calls
        if call[0].startswith("INSERT") and "dmi_schema_version" in call[0]
    ], "an initialiser that never held the lease must not stamp"


def test_a_complete_catalog_is_ensured_without_taking_the_lease():
    """Routine startup must not fail because an indexer is publishing.

    Serialisation is for installs. Against a complete, stamped catalog the DDL
    is idempotent and the stamp is a server-side no-op, so there is nothing to
    serialise -- and taking the lease there would refuse every second process
    for as long as the indexer holds it.
    """
    client = _Client(tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION)

    _writer(client).ensure_schema()

    assert _lease_claims(client) == []
    assert _lease_releases(client) == []


def test_completing_an_interrupted_install_runs_under_the_lease():
    client = _Client(tables=("dmi_schema_version", "dmi_capture_raw"), schema_version=None)

    _writer(client).ensure_schema()

    assert len(_lease_claims(client)) == 1
    assert len(_lease_releases(client)) == 1


def test_ensure_schema_renews_a_lease_the_writer_already_holds_and_keeps_it():
    """A rebuilding writer that acquired first is not locked out of its own install."""
    client = _Client()
    writer = _writer(client)
    held = writer.acquire_publisher_lease("rebuilder")

    writer.ensure_schema()

    assert writer.publisher_lease is not None
    assert writer.publisher_lease.lease_id == held.lease_id
    assert _lease_releases(client) == [], "the writer's own lease is not given back"


def test_a_replicated_descriptor_table_with_the_version_argument_is_accepted():
    """`ReplicatedReplacingMergeTree(path, replica, index_version)` is healthy.

    The engine check compared the parsed name against exactly
    `ReplacingMergeTree`, so the Replicated and Shared members of the family
    -- which keep the version argument and collapse duplicates identically --
    were refused with a rebuild instruction that reproduces the refusal.
    docs/capture-storage-design.md tells operators they may convert to
    `Replicated`; and under a `Replicated` database engine or ClickHouse Cloud
    the server converts this build's OWN `CREATE`, so the pre-stamp recheck
    refused the catalog it had just created and never stamped it.

    What the check exists for is the version argument, so that is what it
    compares: the family name may carry a `Replicated`/`Shared` prefix and the
    replication arguments precede the version column.
    """
    replicated = (
        "ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/x', "
        "'{replica}', index_version) ORDER BY (tenant_id, experiment_id, "
        "run_id, captured_at_ns, capture_id, store_id, pack_id) "
        "SETTINGS index_granularity = 8192"
    )
    shared = (
        "SharedReplacingMergeTree('/x', '{replica}', index_version) "
        "ORDER BY (store_id, pack_id) SETTINGS index_granularity = 8192"
    )
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        engine_full={
            "dmi_capture_raw": replicated,
            "dmi_pack_inventory_raw": shared,
        },
    )

    _writer(client).ensure_schema()  # must not raise

    # And a Replicated table that DROPPED the version argument is still refused.
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        engine_full={
            "dmi_capture_raw": (
                "ReplicatedReplacingMergeTree('/x', '{replica}') ORDER BY "
                "(tenant_id, experiment_id, run_id, captured_at_ns, "
                "capture_id, store_id, pack_id) SETTINGS index_granularity = 8192"
            )
        },
    )
    with pytest.raises(CatalogSchemaVersionError, match="index_version"):
        _writer(client).ensure_schema()

def test_a_fresh_install_rechecks_the_layout_before_it_stamps():
    """The sort-key and engine checks ran only on the STAMPED path.

    A fresh install reads an empty catalog, issues `CREATE TABLE IF NOT
    EXISTS`, confirms the object names and kinds, and stamps. Nothing in that
    sequence looks at the sort key or the engine arguments -- so if another
    initializer created a pre-v4 `capture_raw` between the empty read and this
    one's CREATE, the CREATE is a no-op, the names and kinds all check out, and
    the stamp goes on over a layout it does not describe.

    This does not close that race -- only serialising initialisation per prefix
    does, and the reviewer who found it said so. What it does is stop the stamp
    from being written over a layout this build can see is wrong, which turns a
    silent disagreement into the same refusal any other incompatible catalog
    gets.
    """
    client = _Client(sort_key=_VERSION_ONE_SORT_KEY)
    # The fake grows `tables` as CREATE statements run, so this begins empty --
    # a fresh install -- and the descriptor table it "creates" answers with the
    # version 1 sort key, standing in for the concurrent initializer's table.

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert _VERSION_ONE_SORT_KEY in message
    # And it refused BEFORE stamping: no row claims this build owns that layout.
    assert not [
        item
        for item, *_ in client.calls
        if item.startswith("INSERT") and "dmi_schema_version" in item
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

def test_a_replicated_engine_without_the_version_argument_is_still_refused():
    """The property, not the prefix: Replicated does not excuse a lost version."""
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        engines={"dmi_capture_raw": "ReplicatedReplacingMergeTree"},
        engine_full={
            "dmi_capture_raw": (
                "ReplicatedReplacingMergeTree('/clickhouse/tables/x', 'r1') "
                f"ORDER BY ({_CURRENT_SORT_KEY}) SETTINGS index_granularity = 8192"
            )
        },
    )

    with pytest.raises(CatalogSchemaVersionError, match="index_version"):
        _writer(client).ensure_schema()


def test_a_stamped_catalog_with_the_wrong_inventory_sort_key_is_refused():
    """The inventory's key is load-bearing for the same reason the descriptors' is.

    Keyed on `pack_id` alone, a merge collapses the rows of one pack held in
    two stores -- a supported state -- so `committed_pack_ids` stops reporting
    the losing store's copy and that copy is re-indexed on every pass forever,
    while `_inventory_without_membership` reads the same FINAL and cannot see
    it. Only the descriptor table's key was ever compared.
    """
    client = _Client(
        tables=_CURRENT_OBJECTS,
        schema_version=_SCHEMA_VERSION,
        data_rows=1,
        pack_sort_key="pack_id",
    )

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "dmi_pack_inventory_raw" in message
    assert "ORDER BY (pack_id)" in message
    assert f"ORDER BY ({_CURRENT_PACK_SORT_KEY})" in message
    assert "CatalogReconciler.rebuild()" in message
    assert not [
        item for item, *_ in client.calls
        if item.startswith(("CREATE", "ALTER", "INSERT", "DROP"))
    ]


def test_a_server_without_check_grant_is_told_which_version_it_needs():
    """An old server must not surface as a raw SQL syntax error.

    `CHECK GRANT` arrived in ClickHouse 24.11 and is the first statement that
    can fail on an older one, before any verdict or DDL -- so an operator on
    24.8 met a driver exception outside this module's taxonomy, for a catalog
    whose layout may be perfectly compatible. There is nothing to fall back to,
    since every verdict is read off a grant-filtered `system.tables`, so the
    requirement is named instead.
    """
    class _PreCheckGrantServer(_Client):
        def execute(self, query, params=None, **kwargs):
            if query.startswith("CHECK GRANT"):
                raise RuntimeError("Code: 62. DB::Exception: Syntax error")
            return super().execute(query, params, **kwargs)

    client = _PreCheckGrantServer(tables=_CURRENT_OBJECTS, schema_version=_SCHEMA_VERSION)

    with pytest.raises(CatalogSchemaVersionError) as raised:
        _writer(client).ensure_schema()

    message = str(raised.value)
    assert "24.11" in message
    assert "Syntax error" in message, "the server's own words are kept"
    assert not [
        item for item, *_ in client.calls
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


def test_a_materialized_view_is_not_mistaken_for_the_view_this_build_creates():
    """`kind` was "does the engine name end in View", which is three engines.

    `MaterializedView`, `LiveView` and `WindowView` all end in "View", so a
    materialized view standing at `{prefix}_capture` passed `_reject_wrong_kinds`
    and reached the DDL -- where `CREATE OR REPLACE VIEW` cannot replace it, so
    every startup died on the driver's own error with no rebuild instruction.
    That is precisely the outcome the kind check exists to prevent, and the one
    the sibling test above pins for a table.
    """
    for engine in ("MaterializedView", "LiveView", "WindowView"):
        client = _Client(
            tables=_CURRENT_OBJECTS,
            schema_version=_SCHEMA_VERSION,
            engines={"dmi_capture": engine},
        )

        with pytest.raises(CatalogSchemaVersionError) as raised:
            _writer(client).ensure_schema()

        message = str(raised.value)
        assert f"`dmi_capture` is a TABLE (engine {engine})" in message, engine
        assert "where this build creates a VIEW" in message, engine
        # And nothing was issued on the way to the refusal.
        assert _past_the_visibility_checks(client) == [], engine


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


def test_collect_garbage_resolves_orphans_on_the_initiator_and_deletes_literal_pairs():
    """The mutation carries no subquery; the anti-join is a plain read first.

    `ALTER TABLE ... DELETE WHERE ... NOT IN (SELECT ... FROM other_table)`
    reads an independently replicated table inside the mutation predicate,
    and on `ReplicatedMergeTree` ClickHouse refuses that form under the
    default settings (`allow_nondeterministic_mutations` and
    `mutations_execute_subqueries_on_initiator` both 0 on 25.12) -- so the
    retention pass worked on a single node and failed on exactly the
    deployment the rest of the module takes care over. The orphan set is now
    resolved by an ordinary SELECT on the initiator, with the same two bounds,
    and the mutation deletes those literal `(index_version, publish_id)` pairs
    and nothing else.

    The bounds are pinned here too, because widening one by a comparison is
    the whole risk and `ALTER TABLE ... DELETE` reports nothing.
    """
    client = _Client()
    orphans = [(3, str(uuid4())), (5, str(uuid4()))]
    client.orphan_publishes = list(orphans)
    writer = _leased(client)

    removed = writer.collect_garbage(sleep=_never_sleeps)

    orphan_read = next(
        (query, params)
        for query, params, _ in client.calls
        if query.startswith("SELECT") and "snapshot_manifest" in query
        and "NOT IN (SELECT index_version, publish_id FROM" in query
    )
    # Strictly BELOW the published head: an in-flight publish always sits above
    # it, so this cannot select membership out from under one.
    assert "index_version < %(published)s" in orphan_read[0]
    assert orphan_read[1]["published"] == writer.last_published_version()
    # And only pairs no watermark row admits.
    assert "dmi_index_watermark" in orphan_read[0]

    deletes = [
        (query, params, kwargs)
        for query, params, kwargs in client.calls
        if query.startswith("ALTER TABLE") and "dmi_snapshot_manifest" in query
    ]
    assert len(deletes) == 1
    statement, params, kwargs = deletes[0]
    assert "(index_version, publish_id) IN %(pairs)s" in statement
    assert params["pairs"] == orphans
    assert "SELECT" not in statement, (
        "a replicated mutation must not read another table in its predicate"
    )
    assert kwargs.get("settings") == {"mutations_sync": 1}
    assert removed["dmi_snapshot_manifest"] == len(orphans)

    counted = {
        query.split("`")[3]: (query, params)
        for query, params, _ in client.calls
        if query.startswith("SELECT count() FROM `") and " WHERE " in query
    }
    lease_query, _ = counted["dmi_publisher_lease"]
    # Strictly below the head TERM, so the row the fence resolves survives
    # whether or not it has expired.
    assert "term < %(term)s" in lease_query
    claims_query, claims_params = counted["dmi_capture_version_claims"]
    # At or below the published head: the watermark keeps the allocator's floor
    # once these are gone, and a claim ABOVE the head may be a version a pass
    # has allocated and not yet published.
    assert "version <= %(published)s" in claims_query
    assert claims_params["published"] == writer.last_published_version()
    # The watermark table is never collected -- it IS the floor.
    assert not [
        query for query, _, _ in client.calls
        if query.startswith("ALTER TABLE") and "dmi_index_watermark" in query
    ]


def test_collect_garbage_deletes_only_membership_orphaned_in_both_reads():
    """A publish whose watermark lands mid-sweep keeps its membership.

    "Below the head" means no watermark statement can be ADMITTED for that
    version any more; one admitted above the head and then stalled can still
    LAND below it. The publisher confirms its membership after its watermark
    row stands, but that confirmation can read rows this sweep is about to
    delete -- and then the publish commits packs whose membership is gone,
    which is the inventory-without-membership state ensure_schema refuses.

    So the orphan set is intersected across two reads a publish timeout apart:
    a statement admitted before the first read has landed or been aborted by
    the second, and one landing puts a watermark row beside its pair, which
    drops it from the second read.
    """
    client = _Client()
    landed = (5, str(uuid4()))
    still_orphaned = (3, str(uuid4()))
    client.orphan_publishes = [still_orphaned, landed]
    writer = _leased(client)

    def the_stalled_watermark_lands(_seconds):
        # Its pair now has a watermark row, so the second read omits it.
        client.orphan_publishes = [still_orphaned]

    removed = writer.collect_garbage(sleep=the_stalled_watermark_lands)

    deleted = [
        params["pairs"]
        for query, params, _ in client.calls
        if query.startswith("ALTER TABLE") and "dmi_snapshot_manifest" in query
    ]
    assert deleted == [[still_orphaned]], (
        "membership whose watermark landed during the sweep must survive"
    )
    assert removed["dmi_snapshot_manifest"] == 1
    # And the wait is the publish cap, which is what bounds an admitted
    # statement's execution.
    waits = []
    _leased(_Client()).collect_garbage(sleep=waits.append)
    assert waits == [], "an empty orphan set waits for nothing"
    client.orphan_publishes = [still_orphaned]
    writer.collect_garbage(sleep=waits.append)
    assert waits == [ClickHouseCatalogConfig().publish_timeout_ns / 1e9]


def test_collect_garbage_issues_no_manifest_mutation_when_nothing_is_orphaned():
    client = _Client()
    writer = _leased(client)

    removed = writer.collect_garbage(sleep=_never_sleeps)

    assert removed["dmi_snapshot_manifest"] == 0
    assert not [
        query for query, _, _ in client.calls
        if query.startswith("ALTER TABLE") and "dmi_snapshot_manifest" in query
    ]


def test_collect_garbage_chunks_the_orphan_deletions_by_rendered_size():
    """The pairs ride in the statement TEXT, so a big orphan set is bounded."""
    from dmi.storage.capture.clickhouse_sql import MAX_INLINE_PARAMETER_BYTES

    client = _Client()
    orphans = [(version, str(uuid4())) for version in range(1, 6001)]
    client.orphan_publishes = list(orphans)
    writer = _leased(client)

    removed = writer.collect_garbage(sleep=_never_sleeps)

    deletes = [
        params["pairs"]
        for query, params, _ in client.calls
        if query.startswith("ALTER TABLE") and "dmi_snapshot_manifest" in query
    ]
    assert len(deletes) > 1, "6000 pairs should not fit one statement"
    assert [pair for chunk in deletes for pair in chunk] == orphans
    for chunk in deletes:
        rendered = len(repr(chunk).encode())
        assert rendered <= MAX_INLINE_PARAMETER_BYTES
    assert removed["dmi_snapshot_manifest"] == len(orphans)


def test_a_publish_whose_membership_was_collected_before_its_watermark_landed_is_refused():
    """A late lower commit must not leave inventory with no membership.

    The watermark barrier and fence are evaluated when the statement is
    ADMITTED, and the row can land later. A lower publisher can therefore have
    its watermark statement admitted, stall, and land below a head that a
    higher publisher established in the meantime -- and `collect_garbage()`,
    run in that gap, sees the lower manifest rows below the head with no
    watermark row and removes them. When the delayed watermark lands, the
    publish has a watermark row that admits nothing. Passing the ownership
    read-back there and letting the caller `commit_packs` produced exactly the
    inventory-without-membership state `ensure_schema` refuses (reproduced on
    25.12 in that ordering).

    So a publish confirms its membership AFTER its watermark row stands, and
    refuses -- as a lost race, which is what re-allocating a higher version
    repairs -- when the manifest it wrote is no longer whole.
    """
    ref, descriptor = _descriptor()
    client = _Client()
    writer = _leased(client)
    execute = client.execute

    def collect_between_the_watermark_and_its_confirmation(query, params=None, **kwargs):
        result = execute(query, params, **kwargs)
        if query.lstrip().startswith("INSERT") and "index_watermark" in query:
            # The retention pass, having run while this watermark statement
            # was in flight below a newer head.
            client.manifest = [
                row for row in client.manifest if row[1] != params["publish_id"]
            ]
        return result

    client.execute = collect_between_the_watermark_and_its_confirmation

    with pytest.raises(SnapshotPublishRaceError) as raised:
        writer.publish_snapshot(
            index_version=7, refs=[ref], published_at_ns=7,
            indexed_rows=1, indexed_packs=1,
        )

    message = str(raised.value)
    assert "membership" in message and "collected" in message
    assert "higher version" in message
    # The watermark row stands, and stands harmless: it admits nothing.
    assert client.watermarks == [7]
    assert client.manifest == []
    # And the lease is intact -- this was not a takeover.
    assert writer.publisher_lease is not None


@pytest.mark.parametrize(
    "engine_full, expected",
    [
        (
            "ReplacingMergeTree(index_version) ORDER BY (tenant_id, experiment_id) "
            "SETTINGS index_granularity = 8192",
            ("ReplacingMergeTree", ("index_version",)),
        ),
        # A bare engine call is rendered WITHOUT its parentheses: an embedded
        # ClickHouse 26.7 reported `ReplacingMergeTree ORDER BY ...` for a
        # table created as `ReplacingMergeTree()`. Read naively, the first `(`
        # in that string is ORDER BY's, and the engine "name" swallows the
        # clause up to it.
        (
            "ReplacingMergeTree ORDER BY (tenant_id, experiment_id) "
            "SETTINGS index_granularity = 8192",
            ("ReplacingMergeTree", ()),
        ),
        ("ReplacingMergeTree() ORDER BY (a, b)", ("ReplacingMergeTree", ())),
        ("MergeTree ORDER BY version SETTINGS index_granularity = 8192", ("MergeTree", ())),
        (
            "ReplicatedReplacingMergeTree('/clickhouse/tables/x', 'r1', index_version) "
            "ORDER BY (a, b)",
            ("ReplicatedReplacingMergeTree", ("'/clickhouse/tables/x'", "'r1'", "index_version")),
        ),
        ("View", ("View", ())),
    ],
)
def test_engine_arguments_are_read_off_the_clause_the_server_renders(engine_full, expected):
    from dmi.storage.capture.clickhouse_schema import _engine_arguments

    assert _engine_arguments(engine_full) == expected
