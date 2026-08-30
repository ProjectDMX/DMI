from __future__ import annotations

from os import environ
from uuid import uuid4

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import ClickHouseCatalogConfig, ClickHouseCatalogWriter


pytestmark = [pytest.mark.manual, pytest.mark.clickhouse]


def test_duplicate_catalog_replay_is_logically_deduplicated():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    prefix = f"dmi_catalog_test_{uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
    )
    writer = ClickHouseCatalogWriter(client, config)
    database = config.database
    descriptor = synthetic_descriptors(1)[0]
    ref = descriptor.locator.pack_ref
    schema_created = False
    try:
        # Armed BEFORE ensure_schema: it issues many statements, and one that
        # fails partway leaves every table created ahead of it behind. Arming
        # afterwards skips teardown for exactly that case and the tables leak
        # onto the shared server. Every drop below is IF EXISTS, so tearing
        # down a partial or empty schema is safe.
        schema_created = True
        writer.ensure_schema()
        for version in (1, 2):
            writer.write_descriptors([descriptor], index_version=version)
            # The publish is what admits the pack to the public view, which is
            # now bounded by snapshot membership rather than showing every row
            # the raw table holds. commit_packs is only the replay guard.
            writer.publish_snapshot(
                index_version=version,
                refs=[ref],
                published_at_ns=version,
                indexed_rows=1,
                indexed_packs=1,
            )
            writer.commit_packs([ref], index_version=version)

        assert client.execute(
            f"SELECT count() FROM `{database}`.`{prefix}_capture_raw`"
        ) == [(2,)]
        assert client.execute(
            f"SELECT count() FROM `{database}`.`{prefix}_capture`"
        ) == [(1,)]
        assert client.execute(
            f"SELECT count() FROM `{database}`.`{prefix}_pack_inventory`"
        ) == [(1,)]
        assert writer.committed_pack_ids([(ref.store_id, ref.pack_id)]) == {
            (ref.store_id, ref.pack_id)
        }
    finally:
        if schema_created:
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
