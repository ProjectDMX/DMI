"""Live tests for catalog facets and the idempotent schema upgrade.

Run against a reachable ClickHouse:

    DMI_CLICKHOUSE_HOST=127.0.0.1 python -m pytest tests/test_clickhouse_facets_live.py \
        -m "manual and clickhouse" -q
"""

from __future__ import annotations

from contextlib import contextmanager
from math import prod
from os import environ
from uuid import uuid4

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import ClickHouseCatalogConfig, ClickHouseCatalogWriter
from dmi.storage.capture.clickhouse_catalog import _CAPTURE_COLUMNS, _FACET_COLUMNS


pytestmark = [pytest.mark.manual, pytest.mark.clickhouse]


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
def _writer():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    prefix = f"dmi_facet_test_{uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
    )
    try:
        yield ClickHouseCatalogWriter(client, config), client, config
    finally:
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
            client.execute(f"DROP {kind} IF EXISTS `{database}`.`{prefix}_{suffix}`")


def _facet_rows(client, config, prefix_table: str):
    names = ", ".join(name for name, _, _ in _FACET_COLUMNS)
    return client.execute(
        f"SELECT capture_id, {names} FROM `{config.database}`.`{prefix_table}` "
        "ORDER BY capture_id"
    )


def test_facet_columns_are_created_with_their_declared_types():
    with _writer() as (writer, client, config):
        writer.ensure_schema()

        columns = dict(
            client.execute(
                "SELECT name, type FROM system.columns "
                "WHERE database = %(db)s AND table = %(table)s",
                {"db": config.database, "table": f"{config.table_prefix}_capture_raw"},
            )
        )

        for name, kind, _ in _FACET_COLUMNS:
            assert columns[name] == kind, f"{name} has type {columns[name]}, want {kind}"


def test_ensure_schema_is_idempotent():
    with _writer() as (writer, client, config):
        writer.ensure_schema()
        writer.write_descriptors(synthetic_descriptors(5), index_version=1)

        # Running against a populated table must not raise or lose rows.
        writer.ensure_schema()
        writer.ensure_schema()

        table = f"{config.table_prefix}_capture_raw"
        assert client.execute(
            f"SELECT count() FROM `{config.database}`.`{table}`"
        ) == [(5,)]


def test_facets_are_computed_from_the_descriptor():
    descriptors = synthetic_descriptors(3)
    with _writer() as (writer, client, config):
        writer.ensure_schema()
        writer.write_descriptors(descriptors, index_version=1)

        rows = _facet_rows(client, config, f"{config.table_prefix}_capture_raw")

        by_id = {item.capture_id: item for item in descriptors}
        for capture_id, version, element_count, rank, span, ratio in rows:
            descriptor = by_id[capture_id]
            metadata, locator = descriptor.metadata, descriptor.locator
            assert version == 1
            assert element_count == prod(metadata.shape)
            assert rank == len(metadata.shape)
            assert span == metadata.token_end - metadata.token_start
            assert ratio == pytest.approx(
                locator.decoded_length / locator.stored_length
            )


def test_a_descriptor_table_missing_its_facets_is_repaired_in_place():
    """Facet columns are added back to a populated table, rows intact.

    This used to hand-build the pre-facet Phase 4 table -- old sort key, no
    facet columns -- and prove ensure_schema upgraded it. That table is a
    version 1 catalog, and this build now refuses those outright rather than
    altering them, because the sort key it also carries cannot be altered at
    all (see tests/test_clickhouse_schema_migration_live.py). So the missing
    columns are produced by dropping them from a catalog this build created,
    which reaches the same `ADD COLUMN IF NOT EXISTS` with the same question:
    do rows written before the ALTER still resolve correct facet values?
    """
    descriptors = synthetic_descriptors(4)
    with _writer() as (writer, client, config):
        table = f"`{config.database}`.`{config.table_prefix}_capture_raw`"
        writer.ensure_schema()
        writer.write_descriptors(descriptors, index_version=1)
        for name, _, _ in _FACET_COLUMNS:
            client.execute(f"ALTER TABLE {table} DROP COLUMN {name}")
        columns = {
            row[0]
            for row in client.execute(
                "SELECT name FROM system.columns "
                "WHERE database = %(db)s AND table = %(table)s",
                {"db": config.database, "table": f"{config.table_prefix}_capture_raw"},
            )
        }
        assert columns.isdisjoint({name for name, _, _ in _FACET_COLUMNS})

        # Rows exist and predate the facet columns entirely.
        writer.ensure_schema()

        rows = _facet_rows(client, config, f"{config.table_prefix}_capture_raw")
        assert len(rows) == len(descriptors)
        by_id = {item.capture_id: item for item in descriptors}
        for capture_id, _, element_count, rank, span, _ in rows:
            metadata = by_id[capture_id].metadata
            # Rows written before the ALTER still resolve correct facet values.
            assert element_count == prod(metadata.shape)
            assert rank == len(metadata.shape)
            assert span == metadata.token_end - metadata.token_start


def test_facets_do_not_disturb_the_written_column_set():
    with _writer() as (writer, client, config):
        writer.ensure_schema()

        # A facet must never collide with a column the writer inserts into.
        facet_names = {name for name, _, _ in _FACET_COLUMNS}
        assert facet_names.isdisjoint(_CAPTURE_COLUMNS)

        writer.write_descriptors(synthetic_descriptors(2), index_version=1)
        table = f"{config.table_prefix}_capture_raw"
        assert client.execute(
            f"SELECT count() FROM `{config.database}`.`{table}`"
        ) == [(2,)]


def test_facets_support_server_side_filtering():
    descriptors = synthetic_descriptors(6)
    with _writer() as (writer, client, config):
        writer.ensure_schema()
        writer.write_descriptors(descriptors, index_version=1)

        table = f"`{config.database}`.`{config.table_prefix}_capture_raw`"
        expected = prod(descriptors[0].metadata.shape)
        rows = client.execute(
            f"SELECT count() FROM {table} WHERE element_count = %(n)s",
            {"n": expected},
        )

        assert rows == [(len(descriptors),)]
