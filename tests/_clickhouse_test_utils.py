"""Narrow ClickHouse cleanup helpers for E2E test captures."""

from __future__ import annotations

import json
from pathlib import Path
import re


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def delete_capture(
    host: str,
    port: int,
    model_id: str,
    *,
    database: str = "default",
    table: str = "offload",
) -> None:
    """Synchronously delete one capture without touching unrelated rows."""
    if not _IDENTIFIER.fullmatch(database) or not _IDENTIFIER.fullmatch(table):
        raise ValueError("ClickHouse database and table must be simple identifiers")

    import clickhouse_driver

    client = clickhouse_driver.Client(host, port=port)
    client.execute(
        f"ALTER TABLE {database}.{table} DELETE WHERE model_id = %(model_id)s",
        {"model_id": model_id},
        settings={"mutations_sync": 1},
    )


def delete_capture_from_meta(metadata_dir: str | Path) -> None:
    """Synchronously delete only the capture named by a runner's metadata."""
    metadata_path = Path(metadata_dir) / "meta.json"
    if not metadata_path.is_file():
        return

    metadata = json.loads(metadata_path.read_text())
    model_id = metadata["model_id"]
    delete_capture(
        metadata["db_host"],
        int(metadata["db_port"]),
        model_id,
        database=metadata.get("db_database", "default"),
        table=metadata.get("db_table", "offload"),
    )
