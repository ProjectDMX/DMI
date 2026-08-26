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


def test_upgrade_adds_facets_to_a_table_created_without_them():
    """The pre-facet Phase 4 schema must upgrade in place, rows intact."""
    descriptors = synthetic_descriptors(4)
    with _writer() as (writer, client, config):
        table = f"`{config.database}`.`{config.table_prefix}_capture_raw`"
        client.execute(f"CREATE DATABASE IF NOT EXISTS `{config.database}`")
        # Exactly the Phase 4 table: every written column, no facets.
        client.execute(
            f"""CREATE TABLE {table} (
capture_id String, tenant_id String, experiment_id String, run_id String,
session_id String, request_id String, sequence_id String, model_id String,
model_revision String, adapter_revision Nullable(String),
capture_policy_version String, hook_name LowCardinality(String), layer_number Int32,
producer_rank UInt32, step_number UInt64, token_start UInt64, token_end UInt64,
batch_position UInt32, dtype LowCardinality(String), shape Array(UInt32),
captured_at_ns UInt64, pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), pack_record_count UInt32,
payload_offset UInt64, stored_length UInt64, decoded_length UInt64,
codec LowCardinality(String), payload_checksum FixedString(8), index_version UInt64
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, capture_id)"""
        )
        writer.write_descriptors(descriptors, index_version=1)

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
