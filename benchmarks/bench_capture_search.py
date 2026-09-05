"""Measure bounded catalog search: snapshot cost, page latency, summary throughput.

The headline number is the PRODUCTION pinned read -- the statement
`ClickHouseCaptureCatalog.search` actually issues, with the real watermark and
manifest membership -- against the public `{prefix}_capture` view, which is a
plain `FINAL` over the same rows and is not a snapshot. Phase 6 has to decide
whether the public views keep FINAL, and that decision needs a measured cost
rather than an assumption.

An earlier version of this file measured something else and reported it under
this name: `max(index_version)` over the DESCRIPTOR table rather than the
watermark log, separate per-column `argMax` expressions rather than the
reader's single tuple ordering, no manifest membership at all, and a pin at the
raw maximum so the historical case never arose. Its numbers did not describe
the reader. Everything here now goes through the reader's own statement.
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


class _RecordingClient:
    """Pass every statement through, remembering the last descriptor read.

    The reader builds its statement inline, so the only way to time exactly
    that statement -- projection, membership bound, ordering argument and
    settings -- with a different pin or limit is to capture it as issued.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.last_search = None

    def execute(self, query, params=None, **kwargs):
        if query.startswith("SELECT") and "argMax(tuple(" in query:
            self.last_search = (query, dict(params or {}), dict(kwargs))
        return self._client.execute(query, params, **kwargs)


def measure_snapshot_shapes(
    client, reader, database: str, prefix: str, *, historical_pin: int, trials: int
) -> dict:
    """The production pinned read against a plain FINAL read, on identical rows.

    Three statements are timed, all over the same table:

    * ``production_at_head``: the reader's own statement, pinned at the
      published head, with the LIMIT raised to cover the whole corpus so it
      returns as many rows as the FINAL read does.
    * ``production_at_historical_pin``: the same statement pinned at
      ``historical_pin``, a version BELOW a later publish that re-indexed a
      capture into a new pack. The pinned read must exclude that pack; that is
      the case the design document describes and the one FINAL cannot answer.
    * ``final_view``: ``SELECT <the same columns> FROM {prefix}_capture``, the
      public view -- FINAL plus unbounded membership. Not a snapshot: it shows
      the re-indexed capture twice, once per pack.

    ``watermark_read`` is the reader's `current_watermark()`, i.e. the real
    watermark log, not an aggregate over the descriptor table.
    """
    from dmi.storage.capture.clickhouse_schema import CAPTURE_COLUMNS

    recording = _RecordingClient(client)
    probe = ClickHouseCaptureCatalog(recording, reader.config)
    head = int(probe.current_watermark())
    probe.search(CaptureQuery(limit=1))
    if recording.last_search is None:
        raise RuntimeError("the reader issued no descriptor read to record")
    sql, params, kwargs = recording.last_search
    settings = kwargs.get("settings")
    total = client.execute(f"SELECT count() FROM `{database}`.`{prefix}_capture_raw`")[0][0]
    whole_corpus = {**params, "row_limit": total + 1}

    def production(pin: int):
        return client.execute(sql, {**whole_corpus, "watermark": pin}, settings=settings)

    columns = ", ".join(f"`{name}`" for name in CAPTURE_COLUMNS[:-1])
    view = f"`{database}`.`{prefix}_capture`"
    at_head = _timed(lambda: production(head), trials=trials)
    at_pin = _timed(lambda: production(historical_pin), trials=trials)
    final = _timed(lambda: client.execute(f"SELECT {columns} FROM {view}"), trials=trials)
    return {
        "head": head,
        "historical_pin": historical_pin,
        "production_at_head": at_head,
        "production_at_historical_pin": at_pin,
        "final_view": final,
        "watermark_read": _timed(reader.current_watermark, trials=trials),
        # The row counts are the proof the shapes differ: the pinned reads
        # resolve one row per capture, the FINAL view one per (capture, pack).
        "rows": {
            "production_at_head": at_head["rows"],
            "production_at_historical_pin": at_pin["rows"],
            "final_view": final["rows"],
        },
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

    try:
        # Inside the try, not above it: ensure_schema issues many statements,
        # and one that fails partway leaves every table created ahead of it
        # behind. Run outside, that failure skips the drops entirely and the
        # tables leak onto the shared server. Every drop is IF EXISTS, so
        # tearing down a partial or empty schema is safe.
        writer.ensure_schema()
        # Only the lease holder can publish, and the fence rides inside the
        # publish statement -- without one the corpus below would never become
        # visible and every page measured would be empty.
        writer.acquire_publisher_lease("search-bench")
        descriptors = synthetic_descriptors(args.rows)
        # Descriptors alone are not readable. A snapshot is bounded by the
        # packs a publish made members and the watermark comes from the
        # published log, so a benchmark that only writes descriptors measures
        # empty result sets.
        refs, seen = [], set()
        for item in descriptors:
            ref = item.locator.pack_ref
            if (ref.store_id, ref.pack_id) not in seen:
                seen.add((ref.store_id, ref.pack_id))
                refs.append(ref)
        for version in range(1, args.replays + 1):
            writer.write_descriptors(descriptors, index_version=version)
            writer.publish_snapshot(
                index_version=version,
                refs=refs,
                published_at_ns=version,
                indexed_rows=len(descriptors),
                indexed_packs=len(refs),
            )
            writer.commit_packs(refs, index_version=version)
        # The historical case: one capture re-indexed into a NEW pack at a
        # later version. A read pinned at `historical_pin` must resolve it to
        # the old pack; the head resolves it to the new one; FINAL shows both.
        historical_pin = args.replays
        new_pack = str(uuid4())
        moved = replace(
            descriptors[0],
            locator=replace(
                descriptors[0].locator,
                pack_id=new_pack,
                object_key=f"packs/{new_pack}.dmi-pack",
            ),
        )
        later = historical_pin + 1
        writer.write_descriptors([moved], index_version=later)
        writer.publish_snapshot(
            index_version=later,
            refs=[moved.locator.pack_ref],
            published_at_ns=later,
            indexed_rows=1,
            indexed_packs=1,
        )
        writer.commit_packs([moved.locator.pack_ref], index_version=later)

        result = {
            "rows": args.rows,
            "replays": args.replays,
            "snapshot": measure_snapshot_shapes(
                client, reader, args.database, prefix,
                historical_pin=historical_pin, trials=args.trials,
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
        # The writer owns the object list, in drop order; a hand-copied list
        # here has drifted and leaked tables twice.
        writer.drop_schema()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
