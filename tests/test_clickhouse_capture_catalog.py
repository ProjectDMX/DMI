from __future__ import annotations

from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    PackIndex,
    PackRef,
    PackWriter,
    SnapshotPublishRaceError,
)


pytestmark = pytest.mark.cpu


class _Client:
    def __init__(self):
        self.calls = []
        self.committed = []
        self.claims = []
        self.watermarks = []

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, params, kwargs))
        if query.lstrip().startswith("INSERT"):
            if "version_claims" in query:
                self.claims.extend((row[0], str(row[1])) for row in params)
            elif "index_watermark" in query:
                # The publish barrier is a server-side condition, so the fake
                # has to enforce it or every publish test would pass vacuously.
                version = params["index_version"]
                if version > max(self.watermarks, default=0):
                    self.watermarks.append(version)
            return []
        if "count()" in query and "index_watermark" in query:
            return [(self.watermarks.count(params["version"]),)]
        # The version allocator's queries are answered from real claim state;
        # every other SELECT returns the canned rows.
        if "version_claims" in query:
            if "max(version)" in query:
                return [(max((v for v, _ in self.claims), default=None),)]
            return [(cid,) for v, cid in self.claims if v == params["version"]]
        if query.lstrip().startswith("SELECT"):
            return self.committed
        return []


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
    assert (
        "WHERE (store_id, pack_id) IN ("
        "SELECT store_id, pack_id FROM `default`.`dmi_snapshot_manifest` "
        "WHERE index_version <= "
        "(SELECT max(index_version) FROM `default`.`dmi_index_watermark`))"
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
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.write_descriptors([descriptor], index_version=42)
    writer.publish_snapshot(
        index_version=42, refs=[ref], published_at_ns=7,
        indexed_rows=1, indexed_packs=1,
    )
    writer.commit_packs([ref], index_version=42)

    inserts = [call for call in client.calls if call[0].startswith("INSERT")]
    assert "dmi_capture_raw" in inserts[0][0]
    assert "dmi_snapshot_manifest" in inserts[1][0]
    assert "dmi_index_watermark" in inserts[2][0]
    # The inventory is only the replay guard now, and CatalogIndexer writes it
    # after a successful publish: a crash in between costs redundant work, not
    # a pack that is skipped forever and never visible.
    assert "dmi_pack_inventory_raw" in inserts[3][0]
    assert inserts[0][1][0][0] == "capture-a"
    assert inserts[1][1][0] == (42, ref.store_id, ref.pack_id)


def test_publish_is_a_single_statement_barrier_over_the_watermark():
    """The check and the visibility write cannot be separated by a round trip.

    A SELECT-then-INSERT leaves the whole client round trip -- network, driver,
    a GC pause -- between "am I the highest?" and "I am now visible". As one
    conditional INSERT the server evaluates both together.
    """
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

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
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())
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



# --- catalog facets ---------------------------------------------------------


def test_ensure_schema_declares_every_facet_as_a_materialized_column():
    from dmi.storage.capture.clickhouse_catalog import _FACET_COLUMNS

    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.ensure_schema()

    create = [
        call[0] for call in client.calls if call[0].startswith("CREATE TABLE")
    ][0]
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

    with pytest.raises(ValueError, match="pack identity"):
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
