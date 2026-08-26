from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CatalogIndexer,
    CatalogIndexerConfig,
    CatalogReconciler,
    CaptureMetadata,
    CaptureRecord,
    FilesystemPackStore,
    ObjectPage,
    PackIndex,
    PackRef,
    PackWriter,
    StoredObject,
)


pytestmark = pytest.mark.cpu


def _metadata(capture_id: str, step: int) -> CaptureMetadata:
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


class _Inventory:
    def __init__(self, store: FilesystemPackStore, refs: list[PackRef]):
        self.store_id = store.store_id
        self._store = store
        self._refs = {ref.object_key: ref for ref in refs}
        self.ranges: list[tuple[str, int, int]] = []

    def inspect(self, object_key: str) -> PackRef:
        return self._refs[object_key]

    def list_objects(self, *, prefix="", cursor=None, limit=1000) -> ObjectPage:
        keys = sorted(key for key in self._refs if key.startswith(prefix))
        start = int(cursor or 0)
        selected = keys[start : start + limit]
        next_index = start + len(selected)
        return ObjectPage(
            items=tuple(
                StoredObject(key, self._refs[key].object_bytes) for key in selected
            ),
            next_cursor=str(next_index) if next_index < len(keys) else None,
        )

    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes:
        self.ranges.append((ref.pack_id, offset, length))
        return self._store.read_range(ref, offset, length)

    def stat(self, ref):
        return self._store.stat(ref)

    def put(self, pack, object_key):
        return self._store.put(pack, object_key)


class _CatalogWriter:
    def __init__(self):
        self.committed: set[tuple[str, str]] = set()
        self.descriptor_batches: list[tuple] = []
        self.pack_batches: list[tuple[PackRef, ...]] = []
        self.fail_commit_once = False

    def committed_pack_ids(self, identities):
        return self.committed.intersection(identities)

    def write_descriptors(self, descriptors, *, index_version):
        self.descriptor_batches.append(tuple(descriptors))

    def commit_packs(self, refs, *, index_version):
        self.pack_batches.append(tuple(refs))
        if self.fail_commit_once:
            self.fail_commit_once = False
            raise RuntimeError("ambiguous commit")
        self.committed.update((ref.store_id, ref.pack_id) for ref in refs)


def _packs(tmp_path: Path, counts: tuple[int, ...] = (2, 1)):
    store = FilesystemPackStore(tmp_path, store_id="local")
    refs = []
    step = 0
    for pack_number, count in enumerate(counts, 1):
        writer = PackWriter(
            pack_id=UUID(f"018f0000-0000-7000-8000-{pack_number:012d}"),
            created_at_ns=1_700_000_000_000_000_000 + pack_number,
            max_pack_bytes=1024 * 1024,
        )
        for _ in range(count):
            metadata = _metadata(f"capture-{step}", step)
            writer.append(CaptureRecord(metadata, b"abcdefgh"))
            step += 1
        sealed = writer.seal()
        refs.append(store.put(sealed, f"packs/{sealed.pack_id}.dmi-pack"))
    return _Inventory(store, refs), refs


def test_indexer_reads_only_footers_and_batches_rows(tmp_path: Path):
    inventory, refs = _packs(tmp_path)
    writer = _CatalogWriter()
    indexer = CatalogIndexer(
        inventory,
        writer,
        config=CatalogIndexerConfig(max_packs=8, max_rows_per_insert=2),
        clock_ns=lambda: 42,
        timer_ns=iter((100, 125)).__next__,
    )

    result = indexer.index(refs)

    assert result.indexed_packs == 2
    assert result.indexed_rows == 3
    assert result.descriptor_inserts == 2
    assert result.estimated_bytes > 0
    assert result.elapsed_ns == 25
    assert [len(batch) for batch in writer.descriptor_batches] == [2, 1]
    assert [len(batch) for batch in writer.pack_batches] == [2]
    assert len(inventory.ranges) == 4
    assert sum(length for _, _, length in inventory.ranges) < sum(
        ref.object_bytes for ref in refs
    )


def test_duplicate_event_and_missed_event_converge_on_rebuild(tmp_path: Path):
    inventory, refs = _packs(tmp_path)
    writer = _CatalogWriter()
    reconciler = CatalogReconciler(
        inventory,
        CatalogIndexer(inventory, writer, clock_ns=lambda: 42),
    )

    duplicate = reconciler.index_object_keys([refs[0].object_key, refs[0].object_key])
    rebuilt = reconciler.rebuild(prefix="packs/", page_size=1, max_pages=8)

    assert duplicate.indexed_packs == 1
    assert rebuilt.indexed_packs == 1
    assert rebuilt.skipped_packs == 1
    assert writer.committed == {(ref.store_id, ref.pack_id) for ref in refs}
    assert sum(len(batch) for batch in writer.descriptor_batches) == 3


def test_ambiguous_pack_commit_replays_descriptors_but_converges(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (2,))
    writer = _CatalogWriter()
    writer.fail_commit_once = True
    indexer = CatalogIndexer(inventory, writer, clock_ns=lambda: 42)

    with pytest.raises(RuntimeError, match="ambiguous commit"):
        indexer.index(refs)
    result = indexer.index(refs)

    assert result.indexed_packs == 1
    assert sum(len(batch) for batch in writer.descriptor_batches) == 4
    assert writer.committed == {(refs[0].store_id, refs[0].pack_id)}


def test_indexer_rejects_conflicting_pack_identity(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))
    indexer = CatalogIndexer(inventory, _CatalogWriter())
    conflict = replace(refs[0], checksum="f" * 64)

    with pytest.raises(ValueError, match="conflicting pack identity"):
        indexer.index([refs[0], conflict])


def test_indexer_bounds_notification_batch(tmp_path: Path):
    inventory, refs = _packs(tmp_path)
    indexer = CatalogIndexer(
        inventory,
        _CatalogWriter(),
        config=CatalogIndexerConfig(max_packs=1),
    )

    with pytest.raises(ValueError, match="max_packs"):
        indexer.index(refs)

    with pytest.raises(ValueError, match="max_packs"):
        indexer.index([refs[0], refs[0]])


def test_reconciler_bounds_raw_duplicate_notifications(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))
    reconciler = CatalogReconciler(
        inventory,
        CatalogIndexer(
            inventory,
            _CatalogWriter(),
            config=CatalogIndexerConfig(max_packs=1),
        ),
    )

    with pytest.raises(ValueError, match="max_packs"):
        reconciler.index_object_keys([refs[0].object_key, refs[0].object_key])


def test_corrupt_footer_does_not_commit_pack(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))
    original = inventory.read_range

    def corrupt(ref, offset, length):
        data = bytearray(original(ref, offset, length))
        if length != PackIndex.trailer_size():
            data[0] ^= 1
        return bytes(data)

    inventory.read_range = corrupt
    writer = _CatalogWriter()

    result = CatalogIndexer(inventory, writer).index(refs)

    assert result.failed_packs == 1
    assert result.indexed_packs == 0
    assert not writer.pack_batches


def test_rebuild_caps_failure_details_without_losing_failure_count(tmp_path: Path):
    inventory, _ = _packs(tmp_path, (1, 1))
    original = inventory.read_range

    def corrupt(ref, offset, length):
        data = bytearray(original(ref, offset, length))
        if length != PackIndex.trailer_size():
            data[0] ^= 1
        return bytes(data)

    inventory.read_range = corrupt
    reconciler = CatalogReconciler(
        inventory,
        CatalogIndexer(
            inventory,
            _CatalogWriter(),
            config=CatalogIndexerConfig(max_failure_details=1),
        ),
    )

    result = reconciler.rebuild(prefix="packs/", page_size=1)

    assert result.failed_packs == 2
    assert len(result.failures) == 1


def test_an_oversized_batch_raises_instead_of_blaming_a_pack(tmp_path: Path):
    inventory, refs = _packs(tmp_path)
    writer = _CatalogWriter()
    indexer = CatalogIndexer(
        inventory,
        writer,
        # Small enough that the second pack pushes the batch over.
        config=CatalogIndexerConfig(max_packs=8, max_estimated_bytes=1),
        clock_ns=lambda: 42,
    )

    # A batch-level bound is a caller error. Reporting it as a per-pack failure
    # would blame an innocent pack and silently skip the rest of the batch while
    # index() still returned normally.
    with pytest.raises(ValueError, match="max_estimated_bytes"):
        indexer.index(refs)

    assert writer.descriptor_batches == []
    assert writer.pack_batches == []
