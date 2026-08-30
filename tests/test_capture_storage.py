from __future__ import annotations

from dataclasses import replace
from io import BytesIO
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


def test_capture_metadata_rejects_values_beyond_their_storage_columns():
    # producer_rank/batch_position land in UInt32 catalog columns, the
    # counters and captured_at_ns in UInt64 (and the pack header packs
    # created_at_ns as a uint64 struct field), layer_number in Int32. A value
    # beyond those bounds must fail here, at metadata construction, not on
    # the persistence thread or in the catalog INSERT.
    with pytest.raises(ValueError, match="producer_rank"):
        replace(_metadata("capture-a"), producer_rank=2**32)
    with pytest.raises(ValueError, match="batch_position"):
        replace(_metadata("capture-a"), batch_position=2**32)
    with pytest.raises(ValueError, match="captured_at_ns"):
        replace(_metadata("capture-a"), captured_at_ns=2**64)
    with pytest.raises(ValueError, match="step_number"):
        replace(_metadata("capture-a"), step_number=2**64)
    with pytest.raises(ValueError, match="token_end"):
        replace(_metadata("capture-a"), token_end=2**64)
    with pytest.raises(ValueError, match="layer_number"):
        replace(_metadata("capture-a"), layer_number=2**31)

    # The maxima themselves remain valid.
    replace(
        _metadata("capture-a"),
        producer_rank=2**32 - 1,
        batch_position=2**32 - 1,
        layer_number=2**31 - 1,
    )


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

    def get_by_ids(self, capture_ids, *, tenant_id, watermark):
        assert watermark == "catalog-7"
        wanted = set(capture_ids)
        return tuple(
            item
            for item in self._descriptors
            if item.capture_id in wanted and item.metadata.tenant_id == tenant_id
        )


class _RecordingStore(FilesystemPackStore):
    def __init__(self, root: Path, *, store_id: str):
        super().__init__(root, store_id=store_id)
        self.ranges = []

    def read_range(self, ref, offset, length):
        self.ranges.append((offset, length))
        return super().read_range(ref, offset, length)


class _ShortReadStore(_RecordingStore):
    def read_range(self, ref, offset, length):
        data = super().read_range(ref, offset, length)
        # The footer-binding reads (trailer + footer) come first and must be
        # whole, so hydration reaches the payload read this store shortens.
        if len(self.ranges) <= 2:
            return data
        return data[:-1]


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
    # Hydration first binds the catalog descriptors to the pack footer (two
    # small range reads: trailer, then footer), then fetches the one
    # coalesced payload range the estimate priced.
    trailer_size = PackIndex.trailer_size()
    assert store.ranges == [
        (len(sealed.data) - trailer_size, trailer_size),
        (sealed.footer_offset, len(sealed.data) - trailer_size - sealed.footer_offset),
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


def test_hydration_rejects_a_catalog_row_that_re_describes_the_tensor(
    tmp_path: Path,
):
    """A re-described catalog row must not decode garbage.

    verify_payload checks length and CRC32 of the bytes only, so a catalog
    row claiming int32 (1, 2) for a payload the footer says is float32 (2,)
    -- same bytes, same CRC, same logical_bytes -- passed every check and
    decoded garbage. Hydration must bind the row to the authoritative footer
    before any payload is read.
    """
    store = _RecordingStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00\x00\x80?\x00\x00\x00@"))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )[0]
    lying = replace(
        descriptor,
        metadata=replace(descriptor.metadata, dtype="int32", shape=(1, 2)),
    )
    reader = CaptureReader(_Catalog((lying,)), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(
        PackFormatError,
        match="does not match the pack footer: metadata for capture-a",
    ):
        reader.hydrate(selection, byte_limit=1 << 20)

    # The mismatch is caught pre-payload: only the trailer and footer were
    # read, never the record bytes.
    trailer_size = PackIndex.trailer_size()
    assert store.ranges == [
        (len(sealed.data) - trailer_size, trailer_size),
        (sealed.footer_offset, len(sealed.data) - trailer_size - sealed.footer_offset),
    ]


def test_hydration_rejects_a_catalog_locator_contradicting_the_footer(
    tmp_path: Path,
):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )[0]
    lying = replace(
        descriptor, locator=replace(descriptor.locator, checksum="00000000")
    )
    reader = CaptureReader(_Catalog((lying,)), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(
        PackFormatError,
        match="does not match the pack footer: locator for capture-a",
    ):
        reader.hydrate(selection, byte_limit=1 << 20)


def test_hydration_rejects_a_catalog_capture_absent_from_the_footer(
    tmp_path: Path,
):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptor = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )[0]
    renamed = replace(
        descriptor, metadata=replace(descriptor.metadata, capture_id="capture-x")
    )
    reader = CaptureReader(_Catalog((renamed,)), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    with pytest.raises(PackFormatError, match="capture-x is not in pack"):
        reader.hydrate(selection, byte_limit=1 << 20)


def test_hydration_fetches_the_footer_once_per_pack(tmp_path: Path):
    store = _RecordingStore(tmp_path, store_id="local")
    sealed = _sealed_pack(
        _record("capture-a", b"\x00" * 8),
        _record("capture-b", b"\x01" * 8, step=1),
    )
    ref = store.put(sealed, "packs/a.dmi-pack")
    descriptors = PackReader.from_bytes(sealed.data).descriptors(
        store_id=ref.store_id, object_key=ref.object_key
    )
    reader = CaptureReader(_Catalog(descriptors), {"local": store})
    selection = reader.select(CaptureQuery(limit=10))

    reader.hydrate(selection, byte_limit=1 << 20)
    reader.hydrate(selection, byte_limit=1 << 20)

    trailer_offset = len(sealed.data) - PackIndex.trailer_size()
    trailer_reads = [item for item in store.ranges if item[0] == trailer_offset]
    assert len(trailer_reads) == 1, "the second hydration must hit the cache"
    assert list(reader._footer_cache) == [(ref.store_id, ref.pack_id, ref.checksum)]


def test_the_footer_cache_evicts_its_least_recent_pack(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    descriptors = []
    for index, capture_id in enumerate(("capture-a", "capture-b"), start=1):
        writer = PackWriter(
            pack_id=UUID(int=index),
            created_at_ns=1_700_000_000_000_000_000,
            max_pack_bytes=1024 * 1024,
        )
        writer.append(_record(capture_id, b"\x00" * 8, step=index))
        sealed = writer.seal()
        ref = store.put(sealed, f"packs/{index}.dmi-pack")
        descriptors.extend(
            PackReader.from_bytes(sealed.data).descriptors(
                store_id=ref.store_id, object_key=ref.object_key
            )
        )
    reader = CaptureReader(_Catalog(tuple(descriptors)), {"local": store})
    reader._footer_cache_limit = 1
    selection = reader.select(CaptureQuery(limit=10))

    reader.hydrate(selection, byte_limit=1 << 20)

    assert len(reader._footer_cache) == 1


def test_verify_pack_source_checks_the_stream_against_its_own_checksum():
    """The standalone utility: a full read-and-hash of a pack source.

    S3 uploads now hash their single upload stream instead of calling this,
    but the utility remains public for stores that need an out-of-band check.
    """
    from io import BytesIO

    from dmi.storage.capture.filesystem import verify_pack_source

    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    verify_pack_source(sealed)  # a faithful source passes

    class _Lying:
        pack_id = sealed.pack_id
        created_at_ns = sealed.created_at_ns
        record_count = sealed.record_count
        checksum = "0" * 64
        object_bytes = len(sealed.data)

        def open(self):
            return BytesIO(sealed.data)

    with pytest.raises(PackIntegrityError, match="checksum"):
        verify_pack_source(_Lying())


def test_put_fsyncs_the_chain_before_acknowledging_an_existing_object(
    tmp_path: Path, monkeypatch
):
    """The exists() fast path must not acknowledge a dirent nobody synced.

    The writer that linked the object may not have reached its
    fsync_path_to_root yet (or crashed before it), so a second put() that
    finds the object and returns without syncing acknowledges a pack a power
    loss can still drop.
    """
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    synced: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        "dmi.storage.capture.filesystem.fsync_path_to_root",
        lambda leaf, root: synced.append((leaf, root)),
    )
    expected = (store.root / "packs", store.root)

    store.put(sealed, "packs/a.dmi-pack")
    assert synced == [expected], "the writing path itself must sync the chain"

    ref = store.put(sealed, "packs/a.dmi-pack")

    assert ref.checksum == sealed.checksum
    assert synced == [expected, expected]


def test_the_link_race_loser_fsyncs_the_winners_dirent_before_acknowledging(
    tmp_path: Path, monkeypatch
):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    synced: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        "dmi.storage.capture.filesystem.fsync_path_to_root",
        lambda leaf, root: synced.append((leaf, root)),
    )

    def concurrent_link(src, dst, *args, **kwargs):
        # The winner has linked its dirent but not yet fsynced anything.
        Path(dst).write_bytes(sealed.data)
        raise FileExistsError(dst)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.link", concurrent_link)

    ref = store.put(sealed, "packs/a.dmi-pack")

    assert ref.checksum == sealed.checksum
    assert synced == [(store.root / "packs", store.root)]


def test_a_new_store_root_is_fsynced_at_construction(tmp_path: Path, monkeypatch):
    import os

    synced: list[Path] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.fsync", record)
    root = (tmp_path / "store").resolve()

    FilesystemPackStore(root, store_id="local")

    # mkdir(parents=True) returned before the new root or the parent entry
    # naming it hit disk; both must be synced before the store reports ready.
    assert synced == [root, root.parent]


def test_a_new_root_survives_a_parent_that_cannot_be_opened(
    tmp_path: Path, monkeypatch
):
    import errno
    import os

    from dmi.storage.capture.filesystem import fsync_new_root

    root = (tmp_path / "store").resolve()
    root.mkdir()
    real_open = os.open

    def deny_parent(path, flags, *args, **kwargs):
        if Path(path) == root.parent:
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.open", deny_parent)

    # Best effort on the parent only: a system temp directory may deny the
    # open, and that must not refuse construction -- the root still syncs.
    fsync_new_root(root)


def test_fsync_path_to_root_covers_every_new_directory(tmp_path: Path, monkeypatch):
    import os

    from dmi.storage.capture.filesystem import fsync_path_to_root

    root = (tmp_path / "store").resolve()
    leaf = root / "tenant=a" / "date=2026-08-28" / "session=s" / "rank=0"
    leaf.mkdir(parents=True)
    synced: list[Path] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.fsync", record)

    fsync_path_to_root(leaf, root)

    # mkdir(parents=True) may have created any level of the chain, and each
    # new directory is durable only once its parent's entry is fsynced too.
    expected = [leaf, leaf.parent, leaf.parent.parent, leaf.parent.parent.parent, root]
    assert synced == expected

    with pytest.raises(ValueError, match="not under root"):
        fsync_path_to_root(tmp_path.resolve(), root)


# --- metadata and query validation ---------------------------------------------


def test_capture_metadata_requires_its_text_fields():
    with pytest.raises(ValueError, match="tenant_id is required"):
        replace(_metadata("capture-a"), tenant_id=None)


def test_capture_metadata_rejects_an_unsupported_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        replace(_metadata("capture-a"), dtype="complex64")


def test_capture_metadata_normalises_list_shapes():
    assert replace(_metadata("capture-a"), shape=[2]).shape == (2,)


def test_capture_metadata_bounds_shape_rank():
    with pytest.raises(ValueError, match="rank"):
        replace(_metadata("capture-a"), shape=(1,) * 33)


def test_capture_metadata_rejects_a_reversed_token_span():
    with pytest.raises(ValueError, match="token_end must be >= token_start"):
        replace(_metadata("capture-a"), token_start=2)


def test_capture_metadata_from_mapping_validates_its_input():
    mapping = _metadata("capture-a").to_mapping()

    incomplete = dict(mapping)
    del incomplete["run_id"]
    with pytest.raises(PackFormatError, match="missing: run_id"):
        CaptureMetadata.from_mapping(incomplete)

    with pytest.raises(PackFormatError, match="integer list"):
        CaptureMetadata.from_mapping(dict(mapping, shape=(2,)))
    with pytest.raises(PackFormatError, match="integer list"):
        CaptureMetadata.from_mapping(dict(mapping, shape=[1.5]))

    with pytest.raises(PackFormatError, match="invalid capture metadata"):
        CaptureMetadata.from_mapping(dict(mapping, tenant_id=""))


def test_capture_record_coerces_buffer_payloads_to_bytes():
    coerced = CaptureRecord(
        metadata=_metadata("capture-a"), payload=bytearray(b"\x00" * 8)
    )
    assert type(coerced.payload) is bytes

    view = CaptureRecord(
        metadata=_metadata("capture-b"), payload=memoryview(b"\x01" * 8)
    )
    assert view.payload == b"\x01" * 8


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"limit": 0}, "limit"),
        ({"limit": 10_001}, "limit"),
        ({"hook_names": tuple(f"hook-{i}" for i in range(129))}, "cardinality"),
        ({"layer_numbers": tuple(range(1025))}, "cardinality"),
        ({"layer_numbers": (-2,)}, "layer numbers"),
        ({"captured_after_ns": -1}, "captured_after_ns"),
        ({"captured_before_ns": True}, "captured_before_ns"),
        ({"captured_after_ns": 5, "captured_before_ns": 4}, "captured_before_ns"),
    ),
)
def test_capture_query_validates_its_bounds(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CaptureQuery(**kwargs)


# --- filesystem store validation --------------------------------------------------


class _ScriptedStream:
    """A byte stream that returns exactly the scripted chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, size=-1):
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _LyingPack:
    """Metadata copied from a real pack, a stream that disagrees with it."""

    def __init__(self, sealed, chunks):
        self.pack_id = sealed.pack_id
        self.created_at_ns = sealed.created_at_ns
        self.record_count = sealed.record_count
        self.checksum = sealed.checksum
        self._object_bytes = sealed.object_bytes
        self._chunks = tuple(chunks)

    @property
    def object_bytes(self):
        return self._object_bytes

    def open(self):
        return _ScriptedStream(self._chunks)


def test_filesystem_store_validates_its_store_id(tmp_path: Path):
    with pytest.raises(ValueError, match="store_id"):
        FilesystemPackStore(tmp_path, store_id="")
    with pytest.raises(ValueError, match="store_id"):
        FilesystemPackStore(tmp_path, store_id="x" * 129)


def test_put_rejects_a_non_pack_source(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")

    with pytest.raises(TypeError, match="PackSource"):
        store.put(object(), "packs/a.dmi-pack")


def test_put_validates_the_pack_source_metadata(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(ValueError, match="invalid pack ID"):
        store.put(replace(sealed, pack_id="not-a-uuid"), "packs/a.dmi-pack")
    with pytest.raises(ValueError, match="canonical pack ID"):
        store.put(replace(sealed, pack_id=str(PACK_ID).upper()), "packs/a.dmi-pack")
    with pytest.raises(ValueError, match="numeric metadata"):
        store.put(replace(sealed, created_at_ns=-1), "packs/a.dmi-pack")
    with pytest.raises(ValueError, match="records and bytes"):
        store.put(replace(sealed, record_count=0), "packs/a.dmi-pack")
    with pytest.raises(ValueError, match="invalid checksum"):
        store.put(replace(sealed, checksum="nope"), "packs/a.dmi-pack")


def test_put_rejects_a_pack_source_whose_stream_lies(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    # Ends early, returns text, returns more than asked for.
    with pytest.raises(PackIntegrityError, match="invalid byte stream"):
        store.put(_LyingPack(sealed, []), "packs/a.dmi-pack")
    with pytest.raises(PackIntegrityError, match="invalid byte stream"):
        store.put(_LyingPack(sealed, ["text"]), "packs/b.dmi-pack")
    with pytest.raises(PackIntegrityError, match="invalid byte stream"):
        store.put(_LyingPack(sealed, [sealed.data + b"!"]), "packs/c.dmi-pack")
    # Declared size reached with bytes still in the stream.
    with pytest.raises(PackIntegrityError, match="declared size"):
        store.put(_LyingPack(sealed, [sealed.data, b"!"]), "packs/d.dmi-pack")


class _VerifiedPack:
    """A source that offers ``verified_bytes``, counting the streamed reads.

    ``offer`` is exactly what ``verified_bytes()`` returns, so a test can hand
    over honest bytes, ``None``, or something that breaks the contract.
    ``opened`` records how often the store fell back to streaming.
    """

    def __init__(self, sealed, offer):
        self.pack_id = sealed.pack_id
        self.created_at_ns = sealed.created_at_ns
        self.record_count = sealed.record_count
        self.checksum = sealed.checksum
        self._data = sealed.data
        self._offer = offer
        self.opened = 0

    @property
    def object_bytes(self) -> int:
        return len(self._data)

    def open(self):
        self.opened += 1
        return BytesIO(self._data)

    def verified_bytes(self):
        return self._offer


def test_a_verified_source_is_written_without_being_hashed_again(tmp_path: Path):
    """The pack seal() just hashed is not read back and hashed a second time.

    PackWriter.seal() computes the object checksum over the very bytes
    SealedPack carries, and ``bytes`` cannot change afterwards, so a second
    SHA-256 pass here would re-derive a digest of an object that never left
    the process. The store writes them straight through -- it never opens the
    stream at all -- and what lands on disk still matches the declaration.
    """

    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    pack = _VerifiedPack(sealed, sealed.data)

    ref = store.put(pack, "packs/a.dmi-pack")

    assert pack.opened == 0
    assert (tmp_path / "packs" / "a.dmi-pack").read_bytes() == sealed.data
    assert store.stat(ref).checksum == sealed.checksum


def test_a_source_offering_no_verified_bytes_is_streamed_and_hashed(tmp_path: Path):
    """``None`` means "stream and verify me" -- the contract every other
    source gets, and the one a file- or network-backed source must keep."""

    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    pack = _VerifiedPack(sealed, None)

    ref = store.put(pack, "packs/a.dmi-pack")

    assert pack.opened == 1
    assert ref.checksum == sealed.checksum


@pytest.mark.parametrize(
    "corrupt",
    (
        pytest.param(lambda data: data[:-1], id="short"),
        pytest.param(lambda data: data + b"!", id="long"),
        pytest.param(bytearray, id="mutable"),
    ),
)
def test_put_rejects_verified_bytes_that_break_their_contract(tmp_path: Path, corrupt):
    """The fast path trusts the hash but never the length or the type.

    A store sizes an object from ``object_bytes`` independently of the
    checksum, so bytes that contradict the declaration must not reach storage.
    A mutable buffer is refused outright: the promise is that nothing can
    change the bytes between the assertion and the write, and only ``bytes``
    can make it.
    """

    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(PackIntegrityError, match="invalid verified bytes"):
        store.put(_VerifiedPack(sealed, corrupt(sealed.data)), "packs/a.dmi-pack")

    assert not (tmp_path / "packs" / "a.dmi-pack").exists()
    assert not tuple(tmp_path.rglob("*.open"))


def test_put_rejects_a_key_whose_parent_escapes_the_root(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "store", store_id="local")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "link").symlink_to(outside, target_is_directory=True)
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    with pytest.raises(ValueError, match="escapes the store root"):
        store.put(sealed, "link/a.dmi-pack")


def test_open_regular_refuses_symlinks_and_propagates_other_errors(
    tmp_path: Path, monkeypatch
):
    import errno

    target = tmp_path / "target"
    target.write_bytes(b"data")
    link = tmp_path / "link.dmi-pack"
    link.symlink_to(target)

    with pytest.raises(PackFormatError, match="regular file"):
        FilesystemPackStore._open_regular(link)

    def denied(path, flags):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.open", denied)
    with pytest.raises(OSError, match="denied"):
        FilesystemPackStore._open_regular(target)


def test_open_regular_refuses_a_non_regular_file(tmp_path: Path):
    with pytest.raises(PackFormatError, match="regular file"):
        FilesystemPackStore._open_regular(tmp_path)


def test_put_survives_losing_the_link_race_to_an_identical_writer(
    tmp_path: Path, monkeypatch
):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))

    def concurrent_link(src, dst, *args, **kwargs):
        # The other writer wins the race with the same content.
        Path(dst).write_bytes(sealed.data)
        raise FileExistsError(dst)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.link", concurrent_link)

    ref = store.put(sealed, "packs/a.dmi-pack")

    assert ref.object_bytes == len(sealed.data)
    assert ref.checksum == sealed.checksum


def test_read_range_validates_its_arguments(tmp_path: Path):
    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")

    with pytest.raises(ValueError, match="non-negative integers"):
        store.read_range(ref, -1, 4)
    with pytest.raises(ValueError, match="non-negative integers"):
        store.read_range(ref, 0, 4.0)
    with pytest.raises(PackFormatError, match="exceeds object size"):
        store.read_range(ref, 0, len(sealed.data) + 1)
    with pytest.raises(ValueError, match="pack store mismatch"):
        store.read_range(replace(ref, store_id="other"), 0, 4)


def test_read_range_detects_a_short_filesystem_read(tmp_path: Path, monkeypatch):
    from io import BytesIO

    store = FilesystemPackStore(tmp_path, store_id="local")
    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    ref = store.put(sealed, "packs/a.dmi-pack")
    monkeypatch.setattr(
        FilesystemPackStore,
        "_open_regular",
        staticmethod(lambda path: BytesIO(sealed.data[: len(sealed.data) // 2])),
    )

    with pytest.raises(PackIntegrityError, match="short range"):
        store.read_range(ref, len(sealed.data) - 8, 8)


def test_verify_pack_source_still_hashes_a_verified_source():
    # The write path trusts verified_bytes, but a caller asking to *verify*
    # gets the digest recomputed: a sealed pack whose checksum was tampered
    # with is caught here even though the write path would have trusted it.
    from dmi.storage.capture.filesystem import verify_pack_source

    sealed = _sealed_pack(_record("capture-a", b"\x00" * 8))
    tampered = replace(sealed, checksum="0" * 64)

    verify_pack_source(sealed)  # a faithful source still passes

    with pytest.raises(PackIntegrityError, match="checksum"):
        verify_pack_source(tampered)
