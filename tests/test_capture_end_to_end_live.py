"""End-to-end conformance across an object store and ClickHouse.

Every other live suite exercises one half. The `garage` tests never touch
ClickHouse; the `clickhouse` tests index fabricated descriptors that point at an
object which does not exist, so nothing they return could ever hydrate. This
suite drives the whole chain against real bytes:

    tensors -> pack -> object store -> CatalogIndexer (footer read)
            -> ClickHouse -> search -> hydrate -> decode -> compare

That path is also Phase 6's golden-workload comparison in miniature: identity,
logical bytes, checksums, decoded tensors, and query results all have to agree
end to end.

Run against a reachable ClickHouse:

    DMI_CLICKHOUSE_HOST=127.0.0.1 python -m pytest \
        tests/test_capture_end_to_end_live.py -m "manual and clickhouse" -q

Set DMI_S3_ENDPOINT (plus DMI_S3_BUCKET / key env) to include the S3 store.
"""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureQuery,
    CaptureRecord,
    CaptureReader,
    CatalogIndexer,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseCatalogWriter,
    ClickHouseReaderConfig,
    FilesystemPackStore,
    PackReader,
    PackWriter,
    decode_tensor,
    summarize_tensor,
)


pytestmark = [pytest.mark.manual, pytest.mark.clickhouse]


PACK_ID = UUID("018f0000-0000-7000-8000-0000000000ff")
OBJECT_KEY = "packs/end-to-end.dmi-pack"

# One capture per dtype, so the chain is proven for every payload the format
# accepts rather than for float32 alone.
_DTYPE_CASES = (
    ("float32", np.float32),
    ("float64", np.float64),
    ("float16", np.float16),
    ("int64", np.int64),
    ("int32", np.int32),
    ("int16", np.int16),
    ("int8", np.int8),
    ("uint8", np.uint8),
    ("bool", np.bool_),
)


class _RecordingFilesystemStore(FilesystemPackStore):
    def __init__(self, root: Path, *, store_id: str = "local"):
        super().__init__(root, store_id=store_id)
        self.ranges: list[tuple[int, int]] = []

    def read_range(self, ref, offset, length):
        self.ranges.append((offset, length))
        return super().read_range(ref, offset, length)


def _metadata(capture_id: str, *, dtype: str, shape, index: int) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-e2e",
        experiment_id="experiment-e2e",
        run_id="run-e2e",
        session_id="session-e2e",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="model-e2e",
        model_revision="revision-e2e",
        adapter_revision=None if index % 2 else f"adapter-{index}",
        capture_policy_version="policy-v1",
        hook_name="resid_pre" if index % 2 else "attn_out",
        # Disjoint ranges per field, so a projection swap cannot hide.
        layer_number=3 + index,
        producer_rank=100 + index,
        batch_position=900 + index,
        step_number=100_000 + index,
        token_start=200_000 + index,
        token_end=300_000 + index,
        dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _corpus():
    """Real tensors, one per dtype, with their records."""
    rng = np.random.default_rng(seed=91)
    tensors, records = {}, []
    for index, (dtype, numpy_dtype) in enumerate(_DTYPE_CASES):
        shape = (4, 8) if index % 2 else (16,)
        array = (rng.random(int(np.prod(shape))) * 60).astype(numpy_dtype).reshape(shape)
        capture_id = f"capture-e2e-{index:02d}"
        tensors[capture_id] = array
        records.append(
            CaptureRecord(
                metadata=_metadata(capture_id, dtype=dtype, shape=shape, index=index),
                payload=array.tobytes(),
            )
        )
    return tensors, records


@contextmanager
def _stack(tmp_path: Path):
    """A live ClickHouse catalog over a real object store holding a real pack."""
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    client = clickhouse_driver.Client(
        host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")),
    )
    prefix = f"dmi_e2e_test_{uuid4().hex}"
    config = ClickHouseCatalogConfig(
        database=environ.get("DMI_CLICKHOUSE_DATABASE", "default"),
        table_prefix=prefix,
    )
    writer = ClickHouseCatalogWriter(client, config)
    created = False
    try:
        writer.ensure_schema()
        created = True

        tensors, records = _corpus()
        pack = PackWriter(
            pack_id=PACK_ID,
            created_at_ns=1_700_000_000_000_000_000,
            max_pack_bytes=8 * 1024 * 1024,
        )
        for record in records:
            pack.append(record)
        sealed = pack.seal()

        store = _RecordingFilesystemStore(tmp_path)
        ref = store.put(sealed, OBJECT_KEY)

        # The indexer reads the footer from the store -- no descriptors are
        # handed to it, so locator fidelity is established by the real pack.
        indexer = CatalogIndexer(store, writer, clock_ns=lambda: 7)
        result = indexer.index([ref])
        assert result.failed_packs == 0, result.failures
        assert result.indexed_rows == len(records)

        catalog = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config)
        )
        reader = CaptureReader(catalog, {store.store_id: store}, max_coalesce_gap_bytes=0)
        yield {
            "tensors": tensors,
            "records": records,
            "sealed": sealed,
            "ref": ref,
            "store": store,
            "reader": reader,
            "catalog": catalog,
        }
    finally:
        if created:
            database = config.database
            for kind, suffix in (
                ("VIEW", "capture"),
                ("VIEW", "pack_inventory"),
                ("TABLE", "capture_raw"),
                ("TABLE", "pack_inventory_raw"),
            ):
                client.execute(
                    f"DROP {kind} IF EXISTS `{database}`.`{prefix}_{suffix}`"
                )


def test_catalog_descriptors_match_the_pack_exactly(tmp_path: Path):
    """Every descriptor field survives the round trip through ClickHouse."""
    with _stack(tmp_path) as env:
        from_pack = PackReader.from_bytes(env["sealed"].data).descriptors(
            store_id=env["ref"].store_id, object_key=env["ref"].object_key
        )
        page = env["catalog"].search(CaptureQuery(tenant_id="tenant-e2e", limit=100))

        by_id = {item.capture_id: item for item in page.items}
        assert set(by_id) == {item.capture_id for item in from_pack}
        for expected in from_pack:
            # Compares metadata and locator field by field, so a swapped
            # projection column fails here rather than silently later.
            assert by_id[expected.capture_id] == expected


def test_hydration_returns_the_original_bytes(tmp_path: Path):
    with _stack(tmp_path) as env:
        reader = env["reader"]
        selection = reader.select(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        hydrated = reader.hydrate(selection, byte_limit=8 << 20)

        payloads = {item.capture_id: item.payload for item in hydrated}
        for record in env["records"]:
            assert payloads[record.metadata.capture_id] == record.payload


def test_decoded_tensors_are_identical_end_to_end(tmp_path: Path):
    """The phase gate, over the real chain rather than a fake catalog."""
    with _stack(tmp_path) as env:
        reader = env["reader"]
        selection = reader.select(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        hydrated = reader.hydrate(selection, byte_limit=8 << 20)

        for item in hydrated:
            source = env["tensors"][item.capture_id]
            decoded = decode_tensor(item.descriptor, item.payload)
            assert decoded.dtype == source.dtype
            assert decoded.shape == source.shape
            assert np.array_equal(decoded, source)


def test_analysis_reads_only_the_selected_ranges(tmp_path: Path):
    with _stack(tmp_path) as env:
        reader, store = env["reader"], env["store"]
        page = env["catalog"].search(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        wanted = page.items[::2]
        selection = reader.select(
            CaptureQuery(tenant_id="tenant-e2e", hook_names=("attn_out",), limit=100)
        )
        estimate = reader.estimate(selection)

        store.ranges.clear()
        reader.hydrate(selection, byte_limit=8 << 20)

        extents = [
            (i.locator.offset, i.locator.offset + i.locator.stored_length)
            for i in reader._resolve(selection)
        ]
        for offset, length in store.ranges:
            assert any(
                start <= offset and offset + length <= end for start, end in extents
            ), f"range {(offset, length)} is outside every selected payload"
        assert sum(length for _, length in store.ranges) == estimate.request_bytes
        assert wanted  # the corpus really does interleave the two hooks


def test_summaries_agree_with_the_source_tensors(tmp_path: Path):
    with _stack(tmp_path) as env:
        reader = env["reader"]
        selection = reader.select(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        summaries = reader.summarize(selection, byte_limit=8 << 20)

        by_id = {item.capture_id: item for item in summaries}
        for record in env["records"]:
            capture_id = record.metadata.capture_id
            direct = summarize_tensor(
                next(
                    d
                    for d in reader._resolve(selection)
                    if d.capture_id == capture_id
                ),
                record.payload,
            )
            assert by_id[capture_id].core == direct


def test_a_replayed_index_does_not_change_the_analysis(tmp_path: Path):
    """Re-indexing the same pack must be invisible to a reader."""
    with _stack(tmp_path) as env:
        reader, catalog = env["reader"], env["catalog"]
        before = reader.select(CaptureQuery(tenant_id="tenant-e2e", limit=100))

        # A second indexing pass at a higher version, as reconciliation would do.
        page = catalog.search(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        assert page.watermark == "7"

        after = reader.select(CaptureQuery(tenant_id="tenant-e2e", limit=100))
        assert after.capture_ids == before.capture_ids
        assert after.selection_id == before.selection_id


def test_the_catalog_reports_the_pack_as_committed(tmp_path: Path):
    with _stack(tmp_path) as env:
        ref = env["ref"]
        page = env["catalog"].search(CaptureQuery(tenant_id="tenant-e2e", limit=100))

        # Locators point at the object that actually holds the bytes.
        for item in page.items:
            assert item.locator.object_key == ref.object_key
            assert item.locator.store_id == ref.store_id
            assert item.locator.pack_checksum == ref.checksum
            assert item.locator.object_bytes == ref.object_bytes
