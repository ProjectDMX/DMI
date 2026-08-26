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
)


pytestmark = pytest.mark.cpu


class _Client:
    def __init__(self):
        self.calls = []
        self.committed = []

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, params, kwargs))
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


def test_clickhouse_catalog_inserts_descriptors_before_pack_commit():
    ref, descriptor = _descriptor()
    client = _Client()
    writer = ClickHouseCatalogWriter(client, ClickHouseCatalogConfig())

    writer.write_descriptors([descriptor], index_version=42)
    writer.commit_packs([ref], index_version=42)

    inserts = [call for call in client.calls if call[0].startswith("INSERT")]
    assert "dmi_capture_raw" in inserts[0][0]
    assert "dmi_pack_inventory_raw" in inserts[1][0]
    assert inserts[0][1][0][0] == "capture-a"
    assert inserts[1][1][0][0] == ref.pack_id


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
    assert len(alters) == len(_FACET_COLUMNS)
    for statement in alters:
        # Without IF NOT EXISTS a second start would fail on an upgraded table.
        assert "ADD COLUMN IF NOT EXISTS" in statement
        assert "`default`.`dmi_capture_raw`" in statement


def test_facets_never_collide_with_an_inserted_column():
    from dmi.storage.capture.clickhouse_catalog import (
        _CAPTURE_COLUMNS,
        _FACET_COLUMNS,
    )

    # A MATERIALIZED column cannot be written to, so a collision with the
    # writer's column list would break every insert.
    assert {name for name, _, _ in _FACET_COLUMNS}.isdisjoint(_CAPTURE_COLUMNS)
