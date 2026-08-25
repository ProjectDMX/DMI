from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureCatalog,
    CaptureMetadata,
    CapturePage,
    CaptureQuery,
    CaptureReader,
    CaptureRecord,
    DuplicateCaptureError,
    FilesystemPackStore,
    HydrationLimitError,
    PackConflictError,
    PackFormatError,
    PackIntegrityError,
    PackIndex,
    PackReader,
    PackWriter,
)


pytestmark = pytest.mark.cpu


PACK_ID = UUID("018f0000-0000-7000-8000-000000000001")


def _metadata(capture_id: str, *, step: int = 0) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=step,
        token_start=step,
        token_end=step + 1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _record(capture_id: str, payload: bytes, *, step: int = 0) -> CaptureRecord:
    return CaptureRecord(metadata=_metadata(capture_id, step=step), payload=payload)


def _sealed_pack(*records: CaptureRecord):
    writer = PackWriter(
        pack_id=PACK_ID,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=1024 * 1024,
    )
    for record in records:
        writer.append(record)
    return writer.seal()


def test_pack_round_trip_preserves_analysis_coordinates_and_payloads():
    first = _record("capture-a", b"\x00\x00\x80?\x00\x00\x00@")
    second = _record("capture-b", b"\x00\x00@@\x00\x00\x80@", step=1)

    sealed = _sealed_pack(first, second)
    reader = PackReader.from_bytes(sealed.data)
    descriptors = reader.descriptors(store_id="local", object_key="packs/a.dmi-pack")

    assert reader.pack_id == str(PACK_ID)
    assert [item.capture_id for item in descriptors] == ["capture-a", "capture-b"]
    assert descriptors[0].metadata.model_revision == "revision-a"
    assert descriptors[1].metadata.step_number == 1
    assert reader.read_payload(descriptors[0]) == first.payload
    assert reader.read_payload(descriptors[1]) == second.payload


def test_pack_rejects_truncation_and_unknown_major_version():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(PackFormatError, match="truncated"):
        PackReader.from_bytes(sealed.data[:-1])

    changed = bytearray(sealed.data)
    changed[8:10] = (99).to_bytes(2, "little")
    with pytest.raises(PackFormatError, match="major version"):
        PackReader.from_bytes(changed)


def test_pack_rejects_a_record_when_its_footer_would_exceed_the_limit():
    record = _record("capture-a", b"\x00" * 8)
    sealed = _sealed_pack(record)
    writer = PackWriter(
        pack_id=PACK_ID,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=len(sealed.data) - 1,
    )

    with pytest.raises(ValueError, match="max_pack_bytes"):
        writer.append(record)

    assert writer.record_count == 0


def test_pack_writer_rejects_duplicate_capture_ids():
    writer = PackWriter(
        pack_id=PACK_ID,
        created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=1024 * 1024,
    )
    writer.append(_record("capture-a", b"\x00" * 8))

    with pytest.raises(DuplicateCaptureError, match="duplicate capture ID"):
        writer.append(_record("capture-a", b"\x01" * 8, step=1))

    assert writer.record_count == 1


def test_pack_detects_payload_and_footer_corruption():
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    reader = PackReader.from_bytes(sealed.data)
    descriptor = reader.descriptors(store_id="local", object_key="a")[0]

    payload_corruption = bytearray(sealed.data)
    payload_corruption[descriptor.locator.offset] ^= 0xFF
    with pytest.raises(PackIntegrityError, match="pack checksum"):
        PackReader.from_bytes(payload_corruption)

    footer_corruption = bytearray(sealed.data)
    footer_corruption[sealed.footer_offset] ^= 0x01
    with pytest.raises(PackIntegrityError, match="pack checksum"):
        PackReader.from_bytes(footer_corruption)


def test_capture_record_rejects_shape_payload_mismatch():
    with pytest.raises(ValueError, match="payload length"):
        CaptureRecord(metadata=_metadata("capture-a"), payload=b"\x00" * 4)


def test_capture_metadata_rejects_boolean_numeric_fields():
    with pytest.raises(ValueError, match="step_number"):
        replace(_metadata("capture-a"), step_number=True)

    with pytest.raises(ValueError, match="shape dimensions"):
        replace(_metadata("capture-a"), shape=(True,))


def test_filesystem_store_is_idempotent_and_rejects_conflicts(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    first = store.put(sealed, "tenant=a/date=2026-08-25/a.dmi-pack")
    second = store.put(sealed, "tenant=a/date=2026-08-25/a.dmi-pack")

    assert first == second
    assert store.stat(first).size == len(sealed.data)
    assert store.read_range(first, 0, 8) == sealed.data[:8]

    conflicting = PackWriter(
        pack_id=UUID("018f0000-0000-7000-8000-000000000002"),
        created_at_ns=1,
        max_pack_bytes=1024 * 1024,
    )
    conflicting.append(_record("capture-b", b"\x00" * 8))
    with pytest.raises(PackConflictError, match="different content"):
        store.put(conflicting.seal(), first.object_key)


def test_filesystem_store_rejects_a_symlinked_object(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    object_path = store.root / ref.object_key
    outside = tmp_path / "outside.dmi-pack"
    outside.write_bytes(sealed.data)
    object_path.unlink()
    object_path.symlink_to(outside)

    with pytest.raises(PackFormatError, match="regular file"):
        store.read_range(ref, 0, 8)


def test_pack_index_reads_only_the_trailer_and_footer(tmp_path: Path):
    store = _RecordingStore(tmp_path, store_id="local")
    sealed = _sealed_pack(
        _record("capture-a", b"\x00" * 8),
        _record("capture-b", b"\x01" * 8, step=1),
    )
    ref = store.put(sealed, "packs/a.dmi-pack")

    index = PackIndex.from_store(store, ref)
    descriptors = index.descriptors()

    assert [item.capture_id for item in descriptors] == ["capture-a", "capture-b"]
    assert store.ranges == [
        (len(sealed.data) - PackIndex.trailer_size(), PackIndex.trailer_size()),
        (sealed.footer_offset, len(sealed.data) - PackIndex.trailer_size() - sealed.footer_offset),
    ]
    assert sum(length for _, length in store.ranges) < len(sealed.data) - 16


@pytest.mark.parametrize("key", ("../escape", "/absolute", "a/../../escape", "a\\escape"))
def test_filesystem_store_rejects_unsafe_object_keys(tmp_path: Path, key: str):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(ValueError, match="object key"):
        store.put(sealed, key)


class _Catalog(CaptureCatalog):
    def __init__(self, descriptors):
        self._descriptors = tuple(descriptors)

    def search(self, query: CaptureQuery) -> CapturePage:
        items = tuple(
            item
            for item in self._descriptors
            if query.model_id is None or item.metadata.model_id == query.model_id
        )[: query.limit]
        return CapturePage(items=items, next_cursor=None, watermark="catalog-7")

    def get_by_ids(self, capture_ids, *, watermark):
        assert watermark == "catalog-7"
        wanted = set(capture_ids)
        return tuple(item for item in self._descriptors if item.capture_id in wanted)


class _RecordingStore(FilesystemPackStore):
    def __init__(self, root: Path, *, store_id: str):
        super().__init__(root, store_id=store_id)
        self.ranges = []

    def read_range(self, ref, offset, length):
        self.ranges.append((offset, length))
        return super().read_range(ref, offset, length)


class _ShortReadStore(_RecordingStore):
    def read_range(self, ref, offset, length):
        return super().read_range(ref, offset, length)[:-1]


def test_selection_is_stable_and_rejects_duplicate_logical_captures(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(
        _record("capture-a", b"\x00" * 8),
        _record("capture-b", b"\x01" * 8, step=1),
    )
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(_Catalog(descriptors), {"local": store})

    first = reader.select(CaptureQuery(model_id="model-a", limit=10))
    second = reader.select(CaptureQuery(model_id="model-a", limit=10))

    assert first == second
    assert first.capture_ids == ("capture-a", "capture-b")
    assert first.catalog_watermark == "catalog-7"

    duplicate = replace(descriptors[1], metadata=descriptors[0].metadata)
    with pytest.raises(DuplicateCaptureError, match="capture-a"):
        CaptureReader(_Catalog((descriptors[0], duplicate)), {"local": store}).select(
            CaptureQuery(limit=10)
        )


def test_reader_coalesces_ranges_and_enforces_fetch_budget(tmp_path: Path):
    store = _RecordingStore(tmp_path, store_id="local")
    first_payload = b"\x00\x00\x80?\x00\x00\x00@"
    second_payload = b"\x00\x00@@\x00\x00\x80@"
    sealed = _sealed_pack(
        _record("capture-a", first_payload),
        _record("capture-b", second_payload, step=1),
    )
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(
        _Catalog(descriptors),
        {"local": store},
        max_coalesce_gap_bytes=64,
    )
    selection = reader.select(CaptureQuery(limit=10))
    estimate = reader.estimate(selection)

    assert estimate.capture_count == 2
    assert estimate.object_count == 1
    assert estimate.request_count == 1
    assert estimate.request_bytes >= estimate.stored_bytes

    with pytest.raises(HydrationLimitError, match="byte limit"):
        reader.hydrate(selection, byte_limit=estimate.request_bytes - 1)

    hydrated = reader.hydrate(selection, byte_limit=estimate.request_bytes)

    assert [item.capture_id for item in hydrated] == ["capture-a", "capture-b"]
    assert [item.payload for item in hydrated] == [first_payload, second_payload]
    assert store.ranges == [
        (descriptors[0].locator.offset, estimate.request_bytes),
    ]


def test_reader_enforces_request_budget_before_fetching(tmp_path: Path):
    store = _RecordingStore(tmp_path, store_id="local")
    sealed = _sealed_pack(
        _record("capture-a", b"\x00" * 8),
        _record("capture-b", b"\x01" * 8, step=1),
    )
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(
        _Catalog(descriptors), {"local": store}, max_coalesce_gap_bytes=0
    )
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(HydrationLimitError, match="request limit"):
        reader.hydrate(selection, byte_limit=16, request_limit=1)

    assert store.ranges == []


def test_reader_rejects_catalog_drift_after_selection(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )[0]
    catalog = _Catalog((descriptor,))
    reader = CaptureReader(catalog, {"local": store})
    selection = reader.select(CaptureQuery(limit=10))
    catalog._descriptors = ()

    with pytest.raises(PackFormatError, match="selection no longer resolves"):
        reader.estimate(selection)


def test_reader_rejects_a_short_object_store_range(tmp_path: Path):
    store = _ShortReadStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(_Catalog(descriptors), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(PackIntegrityError, match="short range"):
        reader.hydrate(selection, byte_limit=8)


def test_reader_detects_corruption_in_a_partial_range(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )[0]
    reader = CaptureReader(_Catalog((descriptor,)), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    path = tmp_path / ref.object_key
    with path.open("r+b") as handle:
        handle.seek(descriptor.locator.offset)
        handle.write(b"\xff")

    with pytest.raises(PackIntegrityError, match="record checksum"):
        reader.hydrate(selection, byte_limit=descriptor.locator.stored_length)
