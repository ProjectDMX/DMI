"""Measure bounded catalog search: snapshot cost, page latency, summary throughput.

The headline number is the argMax snapshot read against a plain FINAL read.
Phase 6 has to decide whether the public views keep FINAL, and that decision
needs a measured cost rather than an assumption.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from statistics import median
from time import perf_counter_ns
from uuid import uuid4

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureQuery,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
)


def _timed(call, *, trials: int) -> dict:
    samples = []
    for _ in range(trials):
        start = perf_counter_ns()
        result = call()
        samples.append((perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "rows": len(result) if hasattr(result, "__len__") else None,
    }


def measure_snapshot_shapes(client, database: str, prefix: str, *, trials: int) -> dict:
    """argMax at a watermark against FINAL, on identical data.

    These are not interchangeable -- FINAL silently drops captures re-indexed
    above the watermark -- so this measures what correctness costs, not which
    query to prefer.
    """
    table = f"`{database}`.`{prefix}_capture_raw`"
    watermark = client.execute(f"SELECT max(index_version) FROM {table}")[0][0]
    projection = (
        "capture_id, argMax(payload_offset, index_version), "
        "argMax(stored_length, index_version)"
    )
    return {
        "argmax_at_watermark": _timed(
            lambda: client.execute(
                f"SELECT {projection} FROM {table} "
                f"WHERE index_version <= {watermark} "
                "GROUP BY tenant_id, experiment_id, run_id, captured_at_ns, capture_id"
            ),
            trials=trials,
        ),
        "final_no_watermark": _timed(
            lambda: client.execute(
                f"SELECT capture_id, payload_offset, stored_length FROM {table} FINAL"
            ),
            trials=trials,
        ),
        "watermark_aggregate": _timed(
            lambda: client.execute(f"SELECT max(index_version) FROM {table}"),
            trials=trials,
        ),
    }


def measure_page_latency(reader, *, page_sizes, trials: int) -> dict:
    """Page cost against page size, and against depth at a fixed size.

    Keyset pagination should make depth irrelevant; an offset scheme would show
    the last page costing more than the first.
    """
    by_size = {}
    for size in page_sizes:
        by_size[str(size)] = _timed(
            lambda size=size: reader.search(CaptureQuery(limit=size)).items,
            trials=trials,
        )

    depth_probe = []
    query = CaptureQuery(limit=page_sizes[0])
    cursor = None
    while True:
        sample = _timed(
            lambda cursor=cursor: reader.search(replace(query, cursor=cursor)).items,
            trials=1,
        )
        page = reader.search(replace(query, cursor=cursor))
        depth_probe.append(sample["median_ms"])
        cursor = page.next_cursor
        if cursor is None or len(depth_probe) >= 25:
            break

    return {
        "by_page_size": by_size,
        "by_depth_ms": depth_probe,
        "depth_first_ms": depth_probe[0],
        "depth_last_ms": depth_probe[-1],
        "depth_pages": len(depth_probe),
    }


def measure_selectivity(reader, descriptors, *, trials: int) -> dict:
    metadata = descriptors[0].metadata
    cases = {
        "unfiltered": CaptureQuery(limit=1000),
        "tenant": CaptureQuery(tenant_id=metadata.tenant_id, limit=1000),
        "tenant_run": CaptureQuery(
            tenant_id=metadata.tenant_id, run_id=metadata.run_id, limit=1000
        ),
        "hook": CaptureQuery(hook_names=(metadata.hook_name,), limit=1000),
        "time_window": CaptureQuery(
            captured_after_ns=descriptors[len(descriptors) // 2].metadata.captured_at_ns,
            limit=1000,
        ),
    }
    return {
        name: _timed(lambda q=query: reader.search(q).items, trials=trials)
        for name, query in cases.items()
    }


def measure_summary_throughput(*, elements: int, trials: int) -> dict:
    """Core summary cost per dtype, in elements per second."""
    import numpy as np

    from dmi.storage.capture import summarize_tensor
    from dmi.storage.capture.model import CaptureDescriptor, CaptureMetadata

    base = synthetic_descriptors(1)[0]
    results = {}
    for dtype, numpy_dtype in (
        ("float32", np.float32),
        ("float64", np.float64),
        ("float16", np.float16),
        ("int64", np.int64),
        ("bfloat16", np.uint16),
    ):
        array = (np.random.default_rng(seed=7).random(elements) * 100).astype(numpy_dtype)
        descriptor = CaptureDescriptor(
            metadata=replace(base.metadata, dtype=dtype, shape=(elements,)),
            locator=replace(
                base.locator,
                stored_length=array.nbytes,
                decoded_length=array.nbytes,
            ),
        )
        payload = array.tobytes()
        sample = _timed(lambda: summarize_tensor(descriptor, payload), trials=trials)
        seconds = sample["median_ms"] / 1e3
        results[dtype] = {
            **sample,
            "elements_per_second": elements / seconds if seconds else None,
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--database", default="default")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--replays", type=int, default=2,
                        help="times to re-index the corpus, creating duplicate versions")
    parser.add_argument("--page-sizes", default="100,1000,5000")
    parser.add_argument("--summary-elements", type=int, default=1_000_000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args(argv)

    from clickhouse_driver import Client

    page_sizes = [int(value) for value in args.page_sizes.split(",")]
    prefix = f"dmi_search_bench_{uuid4().hex}"
    client = Client(args.host, port=args.port)
    config = ClickHouseCatalogConfig(database=args.database, table_prefix=prefix)
    writer = ClickHouseCatalogWriter(client, config)
    reader = ClickHouseCaptureCatalog(client, ClickHouseReaderConfig.from_catalog(config))

    writer.ensure_schema()
    try:
        descriptors = synthetic_descriptors(args.rows)
        # Descriptors alone are not readable. A snapshot is bounded by committed
        # packs and the watermark comes from the published log, so a benchmark
        # that only writes descriptors measures empty result sets.
        refs, seen = [], set()
        for item in descriptors:
            ref = item.locator.pack_ref
            if (ref.store_id, ref.pack_id) not in seen:
                seen.add((ref.store_id, ref.pack_id))
                refs.append(ref)
        for version in range(1, args.replays + 1):
            writer.write_descriptors(descriptors, index_version=version)
            writer.commit_packs(refs, index_version=version)
            writer.publish_watermark(
                index_version=version,
                published_at_ns=version,
                indexed_rows=len(descriptors),
                indexed_packs=len(refs),
            )

        result = {
            "rows": args.rows,
            "replays": args.replays,
            "snapshot": measure_snapshot_shapes(
                client, args.database, prefix, trials=args.trials
            ),
            "pages": measure_page_latency(
                reader, page_sizes=page_sizes, trials=args.trials
            ),
            "selectivity": measure_selectivity(reader, descriptors, trials=args.trials),
            "summary": measure_summary_throughput(
                elements=args.summary_elements, trials=args.trials
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        for kind, suffix in (
            ("VIEW", "capture"), ("VIEW", "pack_inventory"),
            ("TABLE", "capture_raw"), ("TABLE", "pack_inventory_raw"),
            ("TABLE", "index_watermark"), ("TABLE", "pack_commit_log"),
        ):
            client.execute(
                f"DROP {kind} IF EXISTS `{args.database}`.`{prefix}_{suffix}`"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
