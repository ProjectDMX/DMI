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
        writer.ensure_schema()
        schema_created = True
        for version in (1, 2):
            writer.write_descriptors([descriptor], index_version=version)
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
                ("TABLE", "pack_commit_log"),
            ):
                client.execute(
                    f"DROP {kind} IF EXISTS `{database}`.`{prefix}_{suffix}`"
                )
