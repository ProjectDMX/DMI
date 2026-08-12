"""Narrow ClickHouse cleanup helpers for E2E test captures."""

from __future__ import annotations

import json
from pathlib import Path


def delete_capture_from_meta(metadata_dir: str | Path) -> None:
    """Synchronously delete only the capture named by a runner's metadata."""
    metadata_path = Path(metadata_dir) / "meta.json"
    if not metadata_path.is_file():
        return

    metadata = json.loads(metadata_path.read_text())
    model_id = metadata["model_id"]
    import clickhouse_driver

    client = clickhouse_driver.Client(
        metadata["db_host"], port=int(metadata["db_port"])
    )
    client.execute(
        "ALTER TABLE default.offload DELETE WHERE model_id = %(model_id)s",
        {"model_id": model_id},
        settings={"mutations_sync": 1},
    )
