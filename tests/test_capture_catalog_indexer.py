from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CatalogIndexer,
    SnapshotPublishConflictError,
    SnapshotPublishExhaustedError,
    SnapshotPublishRaceError,
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
        # The version `commit_packs` was called with. Recorded because the
        # replay guard has to carry the version that WON: discarded here, the
        # claim that it is "recorded at the published version" was untestable
        # and `commit_packs(index_version=0)` passed every test in this file.
        self.pack_versions: list[int] = []
        self.watermarks: list[tuple[int, int, int, int]] = []
        self.manifests: list[tuple[int, tuple]] = []
        self.descriptor_versions: list[int] = []
        self.fail_commit_once = False
        self.fail_publish_once = False
        self.fail_publish_always = False
        self.conflict_once = False
        self.allocations = 0

    def committed_pack_ids(self, identities):
        return self.committed.intersection(identities)

    def write_descriptors(self, descriptors, *, index_version):
        self.descriptor_batches.append(tuple(descriptors))
        self.descriptor_versions.append(index_version)

    def commit_packs(self, refs, *, index_version):
        self.pack_batches.append(tuple(refs))
        self.pack_versions.append(index_version)
        if self.fail_commit_once:
            self.fail_commit_once = False
            raise RuntimeError("ambiguous commit")
        self.committed.update((ref.store_id, ref.pack_id) for ref in refs)

    def publish_snapshot(
        self, *, index_version, refs, published_at_ns, indexed_rows, indexed_packs
    ):
        if index_version <= self.last_published_version():
            # The server-side barrier, modelled: a publish is only visible if
            # it is strictly above the published head.
            raise SnapshotPublishRaceError(f"{index_version} lost the race")
        if self.fail_publish_once:
            self.fail_publish_once = False
            raise SnapshotPublishRaceError("scripted lost race")
        if self.fail_publish_always:
            raise SnapshotPublishRaceError("scripted lost race")
        self.manifests.append((index_version, tuple(refs)))
        self.watermarks.append(
            (index_version, published_at_ns, indexed_rows, indexed_packs)
        )
        if self.conflict_once:
            # The conflict's contract: this publish IS visible (the rows above
            # landed), and so is somebody else's at the same version.
            self.conflict_once = False
            raise SnapshotPublishConflictError(f"{index_version} was shared")

    def last_published_version(self):
        return self.watermarks[-1][0] if self.watermarks else 0

    def allocate_version(self):
        # Deterministic and monotonic: strictly above everything published,
        # counting prior allocations so an unpublished claim is never reused.
        self.allocations += 1
        return self.last_published_version() + self.allocations


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
    # Publication is not optional: without it the rows above are durably
    # stored but permanently invisible to readers pinned to the watermark log.
    # The version is allocator-owned (first allocation on a fresh catalog is
    # 1); the clock stamps only published_at_ns.
    assert writer.watermarks == [(1, 42, 3, 2)]


def test_a_lost_publish_is_retried_at_a_higher_version(tmp_path: Path):
    """Losing the barrier costs a fresh version, not the batch.

    A losing publish made nothing visible, so the recovery is another attempt
    above the winner -- and the DESCRIPTORS ride along: they are rewritten at
    the version that publishes. index_version leads the reader's supersession
    order between two packs describing one capture, so rows left at the lost
    version would rank below another pack's rows written between the lost and
    the winning version, and the reader would resolve a capture to the OLDER
    publish's pack -- and to its locator, the one field that may differ.
    """
    inventory, refs = _packs(tmp_path, (2,))
    writer = _CatalogWriter()
    writer.fail_publish_once = True
    indexer = CatalogIndexer(inventory, writer, clock_ns=lambda: 42)

    result = indexer.index(refs)

    assert result.indexed_packs == 1
    assert writer.allocations == 2, "the retry must take a fresh version"
    published = writer.watermarks[-1][0]
    assert len(writer.watermarks) == 1, "only the winning publish is visible"
    assert writer.manifests == [(published, tuple(refs))]

    # Written at the lost version, rewritten at the winning one -- and the
    # extra insert is reported, so descriptor_inserts stays an honest count.
    assert len(writer.descriptor_batches) == 2
    assert writer.descriptor_versions == [published - 1, published]
    assert writer.descriptor_batches[0] == writer.descriptor_batches[1], (
        "the rewrite must be byte-identical rows at the new version"
    )
    assert result.descriptor_inserts == 2

    # The replay guard is recorded at the published version, after the publish
    # -- at the version that WON, not the one the descriptors were first
    # written with.
    assert writer.pack_batches == [tuple(refs)]
    assert writer.pack_versions == [published]


def test_publishing_gives_up_after_its_bounded_retries(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))
    writer = _CatalogWriter()
    writer.fail_publish_always = True
    indexer = CatalogIndexer(
        inventory,
        writer,
        config=CatalogIndexerConfig(max_publish_attempts=3),
        clock_ns=lambda: 42,
    )

    with pytest.raises(SnapshotPublishExhaustedError, match="after 3 attempts") as raised:
        indexer.index(refs)

    # Inside the module's own taxonomy (a supervisor catching
    # CaptureStorageError must see retry exhaustion), and chained to the last
    # race so the traceback still names the terminal cause.
    assert isinstance(raised.value.__cause__, SnapshotPublishRaceError)

    # Nothing was made visible, and the replay guard was never written -- so
    # the next pass re-indexes rather than skipping a pack nobody can see.
    assert writer.watermarks == []
    assert writer.pack_batches == []


def test_a_conflicted_publish_is_committed_before_it_propagates(tmp_path: Path):
    """A conflict IS visible, so its packs must be skippable before it raises.

    SnapshotPublishConflictError's contract is that this publish landed --
    its watermark row stands and its packs are in the snapshot -- and that it
    must NOT be retried. Propagating it before commit_packs would leave the
    packs out of the replay inventory, so the next scheduled pass would
    re-index and re-publish the very batch the error forbids retrying, burying
    the operator-required anomaly under a later success.
    """
    inventory, refs = _packs(tmp_path, (2,))
    writer = _CatalogWriter()
    writer.conflict_once = True
    indexer = CatalogIndexer(inventory, writer, clock_ns=lambda: 42)

    with pytest.raises(SnapshotPublishConflictError):
        indexer.index(refs)

    published = writer.watermarks[-1][0]
    assert writer.pack_batches == [tuple(refs)], (
        "a visible publish's packs must enter the replay inventory"
    )
    assert writer.pack_versions == [published]
    # And the anomaly still surfaced: the error propagated, it was not
    # absorbed as a retry.
    assert writer.allocations == 1


def test_a_batch_with_nothing_to_do_publishes_nothing(tmp_path: Path):
    """A complete no-op performs no catalog writes at all.

    Publishing requires the exclusive publisher lease, so a sweep that merely
    CONFIRMS a catalog -- a periodic CatalogReconciler.rebuild over packs the
    live indexer already committed -- must not contend for it, burn a version,
    or advance the watermark on every page it has nothing to say about.
    """
    inventory, refs = _packs(tmp_path, (2,))
    writer = _CatalogWriter()
    indexer = CatalogIndexer(inventory, writer, clock_ns=lambda: 42)

    first = indexer.index(refs)
    watermarks = list(writer.watermarks)
    second = indexer.index(refs)

    assert first.indexed_packs == 1
    assert second.indexed_packs == 0
    assert second.skipped_packs == 1
    assert second.descriptor_inserts == 0
    assert writer.watermarks == watermarks, "a no-op pass must not publish"
    assert writer.allocations == 1, "a no-op pass must not burn a version"


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


def test_indexer_contains_a_conflicting_pack_identity_as_failures(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1, 1))
    writer = _CatalogWriter()
    indexer = CatalogIndexer(inventory, writer)
    conflict = replace(refs[0], checksum="f" * 64)

    result = indexer.index([refs[0], conflict, refs[1]])

    # Neither claimant of the shared identity is trustworthy, so both fail
    # and nothing is committed for it -- but one duplicated object must not
    # abort the batch: the healthy pack still indexes.
    assert result.indexed_packs == 1
    assert result.failed_packs == 2
    assert {failure.error_type for failure in result.failures} == {
        "PackConflictError"
    }
    assert {failure.object_key for failure in result.failures} == {
        refs[0].object_key,
        conflict.object_key,
    }
    committed = {identity for batch in writer.pack_batches for identity in batch}
    assert committed == {refs[1]}


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


def test_a_restarted_indexer_cannot_publish_under_the_durable_watermark(
    tmp_path: Path,
):
    inventory, refs = _packs(tmp_path, (1, 1))
    writer = _CatalogWriter()

    CatalogIndexer(inventory, writer, clock_ns=lambda: 2_000).index([refs[0]])
    # A new process starts with no in-memory guard state and a wall clock
    # stepped back below the previous indexer's. Versions come from the
    # catalog's own allocator, so the clock rollback cannot land this batch
    # inside snapshots readers have already pinned -- the clock stamps only
    # published_at_ns.
    CatalogIndexer(inventory, writer, clock_ns=lambda: 1_000).index([refs[1]])

    published = [watermark[0] for watermark in writer.watermarks]
    assert published == sorted(published), f"versions went backwards: {published}"
    assert len(set(published)) == len(published), "a version was reused"
    assert published[1] > published[0]


@pytest.mark.parametrize(
    "field",
    ["max_packs", "max_rows_per_insert", "max_estimated_bytes", "max_failure_details"],
)
@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_indexer_config_rejects_non_positive_fields(field: str, value):
    with pytest.raises(ValueError, match="must be positive"):
        CatalogIndexerConfig(**{field: value})


@pytest.mark.parametrize("bad_clock", [lambda: -1, lambda: "42"])
def test_indexer_rejects_an_invalid_clock_value(tmp_path: Path, bad_clock):
    inventory, refs = _packs(tmp_path, (1,))
    indexer = CatalogIndexer(inventory, _CatalogWriter(), clock_ns=bad_clock)

    with pytest.raises(ValueError, match="clock_ns"):
        indexer.index(refs)


def test_indexer_emits_a_completion_event_per_index_call(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))
    events = []
    indexer = CatalogIndexer(
        inventory, _CatalogWriter(), clock_ns=lambda: 42, on_event=events.append
    )

    result = indexer.index(refs)

    assert len(events) == 1
    event = events[0]
    assert event.event == "catalog_index_completed"
    assert event.indexed_packs == result.indexed_packs == 1
    assert event.indexed_rows == result.indexed_rows
    assert event.estimated_bytes == result.estimated_bytes
    assert indexer.callback_failures == 0


def test_indexer_contains_a_raising_event_callback(tmp_path: Path):
    inventory, refs = _packs(tmp_path, (1,))

    def broken_observer(event):
        raise RuntimeError("observer down")

    indexer = CatalogIndexer(
        inventory, _CatalogWriter(), clock_ns=lambda: 42, on_event=broken_observer
    )

    result = indexer.index(refs)

    assert result.indexed_packs == 1
    assert indexer.callback_failures == 1


def test_reconciler_rejects_mismatched_store_ids(tmp_path: Path):
    inventory, _ = _packs(tmp_path, (1,))
    (tmp_path / "other").mkdir()
    other = FilesystemPackStore(tmp_path / "other", store_id="other")
    indexer = CatalogIndexer(other, _CatalogWriter())

    with pytest.raises(ValueError, match="store IDs differ"):
        CatalogReconciler(inventory, indexer)


def test_reconcile_page_bounds_the_listing_limit(tmp_path: Path):
    inventory, _ = _packs(tmp_path, (1,))
    reconciler = CatalogReconciler(
        inventory,
        CatalogIndexer(
            inventory, _CatalogWriter(), config=CatalogIndexerConfig(max_packs=1)
        ),
    )

    with pytest.raises(ValueError, match="max_packs"):
        reconciler.reconcile_page(limit=2)


def test_rebuild_rejects_a_non_positive_max_pages(tmp_path: Path):
    inventory, _ = _packs(tmp_path, (1,))
    reconciler = CatalogReconciler(
        inventory, CatalogIndexer(inventory, _CatalogWriter())
    )

    with pytest.raises(ValueError, match="max_pages"):
        reconciler.rebuild(max_pages=0)


def test_rebuild_raises_when_the_listing_never_terminates(tmp_path: Path):
    inventory, _ = _packs(tmp_path, (1,))
    inventory.list_objects = lambda *, prefix="", cursor=None, limit=1000: ObjectPage(
        items=(), next_cursor="0"
    )
    reconciler = CatalogReconciler(
        inventory, CatalogIndexer(inventory, _CatalogWriter(), clock_ns=lambda: 42)
    )

    with pytest.raises(RuntimeError, match="max_pages"):
        reconciler.rebuild(max_pages=3)
