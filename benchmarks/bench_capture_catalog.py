"""Measure batched capture-metadata inserts into ClickHouse."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from statistics import median
from time import perf_counter_ns, time_ns
from uuid import UUID, uuid4

from dmi.storage.capture import (
    CaptureDescriptor,
    CaptureMetadata,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    PayloadLocator,
)


def synthetic_descriptors(rows: int) -> tuple[CaptureDescriptor, ...]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    pack_id = str(UUID("018f0000-0000-7000-8000-000000000001"))
    base = CaptureMetadata(
        capture_id="capture-0", tenant_id="tenant-a", experiment_id="exp-a",
        run_id="run-a", session_id="session-a", request_id="request-a",
        sequence_id="sequence-a", model_id="model-a", model_revision="revision-a",
        adapter_revision=None, capture_policy_version="policy-v1",
        hook_name="resid_pre", layer_number=3, producer_rank=0, step_number=0,
        token_start=0, token_end=1, batch_position=0, dtype="float32",
        shape=(4096,), captured_at_ns=1_700_000_000_000_000_000,
    )
    locator = PayloadLocator(
        pack_id=pack_id, store_id="garage", object_key="packs/synthetic.dmi-pack",
        object_bytes=rows * 16_384, pack_checksum="0" * 64,
        pack_record_count=rows, offset=64, stored_length=16_384,
        decoded_length=16_384, codec="none", checksum="00000000",
    )
    return tuple(
        CaptureDescriptor(
            replace(
                base, capture_id=f"capture-{index}", step_number=index,
                token_start=index, token_end=index + 1,
                captured_at_ns=base.captured_at_ns + index,
            ),
            replace(locator, offset=64 + index * 16_384),
        )
        for index in range(rows)
    )


def measure_inserts(writer, descriptors, *, batch_rows: int, trials: int) -> dict:
    if batch_rows <= 0 or trials <= 0:
        raise ValueError("batch_rows and trials must be positive")
    samples = []
    inserts = (len(descriptors) + batch_rows - 1) // batch_rows
    for trial in range(trials):
        start = perf_counter_ns()
        for offset in range(0, len(descriptors), batch_rows):
            writer.write_descriptors(
                descriptors[offset : offset + batch_rows],
                index_version=time_ns() + trial,
            )
        elapsed = (perf_counter_ns() - start) / 1e9
        samples.append(len(descriptors) / elapsed)
    return {
        "rows": len(descriptors),
        "batch_rows": batch_rows,
        "inserts_per_trial": inserts,
        "trials": trials,
        "rows_per_second_median": median(samples),
        "rows_per_second_samples": samples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--database", default="default")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-rows", type=int, default=10_000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args(argv)

    from clickhouse_driver import Client

    prefix = f"dmi_catalog_bench_{uuid4().hex}"
    client = Client(args.host, port=args.port)
    writer = ClickHouseCatalogWriter(
        client,
        ClickHouseCatalogConfig(database=args.database, table_prefix=prefix),
    )
    writer.ensure_schema()
    try:
        result = measure_inserts(
            writer,
            synthetic_descriptors(args.rows),
            batch_rows=args.batch_rows,
            trials=args.trials,
        )
        print(json.dumps(result, sort_keys=True))
    finally:
        for kind, suffix in (
            ("VIEW", "capture"), ("VIEW", "pack_inventory"),
            ("TABLE", "capture_raw"), ("TABLE", "pack_inventory_raw"),
        ):
            client.execute(
                f"DROP {kind} IF EXISTS `{args.database}`.`{prefix}_{suffix}`"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
