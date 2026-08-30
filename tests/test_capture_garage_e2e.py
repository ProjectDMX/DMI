"""The whole capture loop over a real Garage bucket and a real ClickHouse.

`test_capture_end_to_end_live.py` proves the chain over a filesystem store; its
fixtures put a pack on local disk and hand the reader that store. This suite
proves the same chain over the production storage substrate, and adds the two
steps a filesystem store cannot exercise: discovery by bucket listing, and
`S3PackStore` acting as the `PackInventory` a `CatalogReconciler` reads.

    records -> HostCapturePipeline -> DurablePackSpool -> ParallelSpoolUploader
            -> Garage -> CatalogReconciler (list + inspect + footer read)
            -> ClickHouse -> pinned search / get_by_ids -> range hydration
            -> original payload bytes

Both halves are required, so the suite carries both markers. Run it under the
ephemeral Garage harness against a reachable ClickHouse:

    DMI_GARAGE_BINARY=/path/to/garage python tests/tools/run_garage_live.py \
        --tests tests/test_capture_garage_e2e.py \
        --marker "garage and clickhouse and manual"
"""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from dmi.storage.capture import (
    AdmissionResult,
    CaptureMetadata,
    CaptureQuery,
    CaptureReader,
    CaptureRecord,
    CatalogIndexer,
    CatalogReconciler,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
    DurablePackSink,
    DurablePackSpool,
    HostCapturePipeline,
    OverloadPolicy,
    ParallelSpoolUploader,
    ParallelUploadConfig,
    PipelineConfig,
    S3PackStore,
    S3StoreConfig,
    decode_tensor,
)


pytestmark = [pytest.mark.manual, pytest.mark.garage, pytest.mark.clickhouse]


# Two captures per session, so the assembler's session scope seals a pack every
# two records and the bucket listing the reconciler walks holds several objects.
_DTYPE_CASES = (
    ("float32", np.float32, (4, 8)),
    ("float64", np.float64, (16,)),
    ("int32", np.int32, (2, 6)),
    ("int16", np.int16, (12,)),
    ("uint8", np.uint8, (8, 4)),
    ("bool", np.bool_, (10,)),
)

# Every table ClickHouseCatalogWriter.ensure_schema creates, so teardown leaves
# nothing behind even when a test fails part way through.
_SCHEMA_OBJECTS = (
    ("VIEW", "capture"),
    ("VIEW", "pack_inventory"),
    ("TABLE", "capture_raw"),
    ("TABLE", "pack_inventory_raw"),
    ("TABLE", "capture_version_claims"),
    ("TABLE", "index_watermark"),
    ("TABLE", "pack_commit_log"),
)


def _store(store_id: str) -> S3PackStore:
    names = (
        "DMI_S3_ENDPOINT",
        "DMI_S3_BUCKET",
        "DMI_S3_ACCESS_KEY_ID",
        "DMI_S3_SECRET_ACCESS_KEY",
    )
    values = {name: environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip("missing Garage environment: " + ", ".join(missing))
    return S3PackStore.from_config(
        S3StoreConfig(
            endpoint_url=values["DMI_S3_ENDPOINT"],
            bucket=values["DMI_S3_BUCKET"],
            region=environ.get("DMI_S3_REGION", "garage"),
            access_key_id=values["DMI_S3_ACCESS_KEY_ID"],
            secret_access_key=values["DMI_S3_SECRET_ACCESS_KEY"],
            store_id=store_id,
            allow_insecure_http=environ.get("DMI_S3_ALLOW_HTTP") == "1",
            multipart_threshold_bytes=8 * 1024**2,
            multipart_chunk_bytes=5 * 1024**2,
            multipart_concurrency=2,
        )
    )


def _metadata(tenant: str, index: int, dtype: str, shape) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"{tenant}-capture-{index:02d}",
        tenant_id=tenant,
        experiment_id="experiment-garage-e2e",
        run_id="run-garage-e2e",
        session_id=f"session-{index // 2}",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="model-garage-e2e",
        model_revision="revision-e2e",
        adapter_revision=None if index % 2 else f"adapter-{index}",
        capture_policy_version="policy-v1",
        hook_name="resid_pre" if index % 2 else "attn_out",
        # Disjoint ranges per field, so a swapped projection column cannot hide.
        layer_number=3 + index,
        producer_rank=0,
        batch_position=900 + index,
        step_number=100_000 + index,
        token_start=200_000 + index,
        token_end=300_000 + index,
        dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _corpus(tenant: str):
    """Real tensors, one per dtype, with the records that carry them."""

    rng = np.random.default_rng(seed=91)
    tensors, records = {}, []
    for index, (dtype, numpy_dtype, shape) in enumerate(_DTYPE_CASES):
        array = (
            (rng.random(int(np.prod(shape))) * 60)
            .astype(numpy_dtype)
            .reshape(shape)
        )
        metadata = _metadata(tenant, index, dtype, shape)
        tensors[metadata.capture_id] = array
        records.append(CaptureRecord(metadata=metadata, payload=array.tobytes()))
    return tensors, tuple(records)


class _RecordingSpoolSink:
    """A DurablePackSink that keeps the sealed bytes it staged."""

    def __init__(self, spool: DurablePackSpool) -> None:
        self._inner = DurablePackSink(spool)
        self.packs: dict[str, object] = {}

    def persist(self, ready):
        staged = self._inner.persist(ready)
        self.packs[staged.pack_id] = ready.pack
        return staged


def _object_keys(store: S3PackStore, prefix: str) -> list[str]:
    keys: list[str] = []
    cursor = None
    while True:
        page = store.list_objects(prefix=prefix, cursor=cursor, limit=1000)
        keys.extend(item.object_key for item in page.items)
        cursor = page.next_cursor
        if cursor is None:
            return sorted(keys)


@contextmanager
def _stack(tmp_path: Path):
    """Packs in Garage, reconciled into a live ClickHouse catalog."""

    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    tenant = f"garage-e2e-{uuid4().hex}"
    prefix = f"v1/tenant={tenant}/"
    store = _store(store_id=f"garage-{uuid4().hex[:8]}")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=f"dmi_garage_e2e_{uuid4().hex}",
    )
    writer = ClickHouseCatalogWriter(client, config)
    created = False
    try:
        writer.ensure_schema()
        created = True

        tensors, records = _corpus(tenant)

        # (1) the real host path: pipeline -> durable spool -> uploader.
        spool = DurablePackSpool(tmp_path / "spool", max_bytes=64 * 1024**2)
        sink = _RecordingSpoolSink(spool)
        pipeline = HostCapturePipeline(
            PipelineConfig(
                max_queue_records=64,
                max_queue_bytes=8 * 1024**2,
                max_pack_bytes=1024**2,
                max_pack_records=8,
                max_linger_ns=60 * 1_000_000_000,
                overload_policy=OverloadPolicy.BLOCK,
            ),
            sink,
        )
        pipeline.start()
        for record in records:
            assert pipeline.submit(record) is AdmissionResult.ACCEPTED
        pipeline_snapshot = pipeline.close(timeout=30)
        assert pipeline_snapshot.failures == 0
        assert pipeline_snapshot.persisted_records == len(records)

        upload = ParallelSpoolUploader(
            spool,
            store,
            ParallelUploadConfig(max_workers=4, max_in_flight_bytes=16 * 1024**2),
        ).upload_pending()
        assert upload.failures == ()
        assert upload.snapshot.uploaded_packs == len(sink.packs)
        assert spool.snapshot().entries == 0
        assert _object_keys(store, prefix) == sorted(
            ref.object_key for ref in upload.refs
        )

        # (2) discovery by listing, not by the refs the uploader happens to
        # hold: S3PackStore is used as the PackInventory here, so inspect()
        # has to rebuild every PackRef from Garage object metadata alone.
        indexer = CatalogIndexer(store, writer)
        reconciler = CatalogReconciler(store, indexer)
        index = reconciler.rebuild(prefix=prefix, page_size=16)
        assert index.failures == (), index.failures
        assert index.failed_packs == 0
        assert index.indexed_packs == len(sink.packs)
        assert index.indexed_rows == len(records)

        catalog = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)
        )
        reader = CaptureReader(
            catalog, {store.store_id: store}, max_coalesce_gap_bytes=0
        )
        yield {
            "tenant": tenant,
            "prefix": prefix,
            "tensors": tensors,
            "records": records,
            "packs": sink.packs,
            "refs": upload.refs,
            "store": store,
            "client": client,
            "config": config,
            "writer": writer,
            "indexer": indexer,
            "reconciler": reconciler,
            "catalog": catalog,
            "reader": reader,
        }
    finally:
        if created:
            for kind, suffix in _SCHEMA_OBJECTS:
                client.execute(
                    f"DROP {kind} IF EXISTS "
                    f"`{config.database}`.`{config.table_prefix}_{suffix}`"
                )
        for key in _object_keys(store, prefix):
            store._client.delete_object(Bucket=store._bucket, Key=key)


def test_reconciled_garage_bucket_hydrates_the_original_bytes(tmp_path: Path):
    """The phase gate: bucket listing to original payload bytes, live."""

    with _stack(tmp_path) as env:
        catalog, reader = env["catalog"], env["reader"]
        tenant = env["tenant"]

        # A pinned watermark is the snapshot every later read is taken at.
        pinned = catalog.current_watermark()
        assert int(pinned) >= 1

        page = catalog.search(CaptureQuery(tenant_id=tenant, limit=100))
        assert {item.capture_id for item in page.items} == {
            record.metadata.capture_id for record in env["records"]
        }
        assert page.watermark == pinned

        by_ids = catalog.get_by_ids(
            [record.metadata.capture_id for record in env["records"]],
            tenant_id=tenant,
            watermark=pinned,
        )
        assert {item.capture_id for item in by_ids} == {
            item.capture_id for item in page.items
        }
        # Locators point at objects that really exist in the bucket.
        assert {item.locator.object_key for item in by_ids} == set(
            _object_keys(env["store"], env["prefix"])
        )
        for item in by_ids:
            assert item.locator.store_id == env["store"].store_id

        # Bounded range hydration straight out of Garage.
        selection = reader.select(CaptureQuery(tenant_id=tenant, limit=100))
        estimate = reader.estimate(selection)
        assert estimate.capture_count == len(env["records"])
        assert estimate.object_count == len(env["packs"])
        hydrated = reader.hydrate(selection, byte_limit=8 << 20)

        payloads = {item.capture_id: item.payload for item in hydrated}
        assert len(payloads) == len(env["records"])
        for record in env["records"]:
            assert payloads[record.metadata.capture_id] == record.payload

        # ...and the tensors decode back to what was captured.
        for item in hydrated:
            source = env["tensors"][item.capture_id]
            decoded = decode_tensor(item.descriptor, item.payload)
            assert decoded.dtype == source.dtype
            assert decoded.shape == source.shape
            assert np.array_equal(decoded, source)


def test_reconciling_the_same_bucket_twice_indexes_nothing_new(tmp_path: Path):
    """Committed packs are skipped, and a pinned read is unchanged by the pass."""

    with _stack(tmp_path) as env:
        catalog, reader = env["catalog"], env["reader"]
        tenant = env["tenant"]
        pinned = catalog.current_watermark()
        before = reader.select(CaptureQuery(tenant_id=tenant, limit=100))

        again = env["reconciler"].rebuild(prefix=env["prefix"], page_size=16)
        assert again.failures == ()
        assert again.indexed_packs == 0
        assert again.indexed_rows == 0
        assert again.skipped_packs == len(env["packs"])

        after = reader.select(CaptureQuery(tenant_id=tenant, limit=100))
        assert after.capture_ids == before.capture_ids
        # The replay published a new (empty) version, so the catalog moved on
        # -- the selection is taken at a later watermark -- without changing
        # which captures it contains or what the pinned snapshot holds.
        assert int(after.catalog_watermark) > int(before.catalog_watermark)
        assert int(catalog.current_watermark()) > int(pinned)
        assert {
            item.capture_id
            for item in catalog.get_by_ids(
                list(before.capture_ids), tenant_id=tenant, watermark=pinned
            )
        } == set(before.capture_ids)


def test_reconciler_reports_a_foreign_bucket_object_without_losing_packs(
    tmp_path: Path,
):
    """A non-pack object in the prefix is one failure, not a failed rebuild."""

    with _stack(tmp_path) as env:
        store, prefix = env["store"], env["prefix"]
        foreign = f"{prefix}session-x/rank-0/not-a-pack.dmi-pack"
        store._client.put_object(
            Bucket=store._bucket, Key=foreign, Body=b"not a DMI pack"
        )

        result = env["reconciler"].rebuild(prefix=prefix, page_size=16)

        assert result.failed_packs == 1
        assert [failure.object_key for failure in result.failures] == [foreign]
        assert result.skipped_packs == len(env["packs"])
        # The healthy packs are still readable end to end.
        selection = env["reader"].select(CaptureQuery(tenant_id=env["tenant"], limit=100))
        hydrated = env["reader"].hydrate(selection, byte_limit=8 << 20)
        payloads = {item.capture_id: item.payload for item in hydrated}
        for record in env["records"]:
            assert payloads[record.metadata.capture_id] == record.payload
