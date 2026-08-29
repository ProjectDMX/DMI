from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from uuid import uuid4

import pytest

from tests._faults import FaultInjected, FaultyPackStore, fail_on
from dmi.storage.capture import (
    AdmissionResult,
    CaptureMetadata,
    CaptureRecord,
    DurablePackSink,
    DurablePackSpool,
    HostCapturePipeline,
    OverloadPolicy,
    PackIndex,
    PackWriter,
    ParallelSpoolUploader,
    ParallelUploadConfig,
    PipelineConfig,
    S3PackStore,
    S3StoreConfig,
)


pytestmark = [pytest.mark.manual, pytest.mark.garage]


def _store() -> S3PackStore:
    names = (
        "DMI_S3_ENDPOINT",
        "DMI_S3_BUCKET",
        "DMI_S3_ACCESS_KEY_ID",
        "DMI_S3_SECRET_ACCESS_KEY",
    )
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip("missing Garage environment: " + ", ".join(missing))
    config = S3StoreConfig(
        endpoint_url=values["DMI_S3_ENDPOINT"],
        bucket=values["DMI_S3_BUCKET"],
        region=os.environ.get("DMI_S3_REGION", "garage"),
        access_key_id=values["DMI_S3_ACCESS_KEY_ID"],
        secret_access_key=values["DMI_S3_SECRET_ACCESS_KEY"],
        store_id="garage-live",
        allow_insecure_http=os.environ.get("DMI_S3_ALLOW_HTTP") == "1",
        multipart_threshold_bytes=5 * 1024**2,
        multipart_chunk_bytes=5 * 1024**2,
        multipart_concurrency=2,
    )
    return S3PackStore.from_config(config)


def test_garage_multipart_retry_listing_and_two_range_footer_read():
    store = _store()
    pack_id = uuid4()
    metadata = CaptureMetadata(
        capture_id=f"garage-live-{pack_id}",
        tenant_id="test",
        experiment_id="garage-live",
        run_id=str(pack_id),
        session_id="session-0",
        request_id="request-0",
        sequence_id="sequence-0",
        model_id="synthetic",
        model_revision="test-v1",
        adapter_revision=None,
        capture_policy_version="all-v1",
        hook_name="resid_pre",
        layer_number=0,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="uint8",
        shape=(6 * 1024**2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    writer = PackWriter(
        pack_id=pack_id,
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=8 * 1024**2,
    )
    writer.append(
        CaptureRecord(metadata=metadata, payload=b"x" * (6 * 1024**2))
    )
    pack = writer.seal()
    prefix = f"tests/dmi/{pack_id}/"
    key = f"{prefix}{pack.pack_id}.dmi-pack"

    first = store.put(pack, key)
    second = store.put(pack, key)
    info = store.stat(first)
    index = PackIndex.from_store(store, first)
    page = store.list_objects(prefix=prefix, limit=10)

    assert first == second
    assert info.size == len(pack.data)
    assert info.checksum == pack.checksum
    assert index.descriptors()[0].capture_id == metadata.capture_id
    assert [item.object_key for item in page.items] == [key]


# --- the production path: pipeline -> spool -> uploader -> Garage -------------
#
# The contract test above drives S3PackStore directly. These drive the path a
# capture host actually runs: HostCapturePipeline assembles packs, DurablePackSink
# stages them on local disk, and ParallelSpoolUploader moves them to Garage and
# only then drops the local copy.


PAYLOAD_BYTES = 32 * 1024


def _pipeline_metadata(tenant: str, index: int) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"{tenant}-capture-{index:03d}",
        tenant_id=tenant,
        experiment_id="garage-pipeline",
        run_id="run-0",
        # One session per pair of records, so the assembler's session scope
        # closes a pack every two records and the spool holds several packs.
        session_id=f"session-{index // 2}",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="synthetic",
        model_revision="test-v1",
        adapter_revision=None,
        capture_policy_version="all-v1",
        hook_name="resid_pre" if index % 2 else "attn_out",
        layer_number=index,
        producer_rank=0,
        step_number=index,
        token_start=index,
        token_end=index + 1,
        batch_position=index,
        dtype="uint8",
        shape=(PAYLOAD_BYTES,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _records(tenant: str, count: int) -> tuple[CaptureRecord, ...]:
    return tuple(
        CaptureRecord(
            metadata=_pipeline_metadata(tenant, index),
            payload=bytes((index + 7) % 251 for _ in range(1)) * PAYLOAD_BYTES,
        )
        for index in range(count)
    )


class _RecordingSpoolSink:
    """A DurablePackSink that also keeps the sealed bytes it staged.

    The uploader hands back PackRefs only, so the test needs the pack the
    pipeline actually sealed to prove the object in Garage is byte-identical.
    """

    def __init__(self, spool: DurablePackSpool) -> None:
        self._inner = DurablePackSink(spool)
        self.packs: dict[str, object] = {}
        self.object_keys: dict[str, str] = {}

    def persist(self, ready):
        staged = self._inner.persist(ready)
        self.packs[staged.pack_id] = ready.pack
        self.object_keys[staged.pack_id] = staged.object_key
        return staged


def _spool_records(root: Path, tenant: str, count: int) -> _RecordingSpoolSink:
    """Run `count` records through the real pipeline into a real spool."""

    spool = DurablePackSpool(root, max_bytes=256 * 1024**2)
    sink = _RecordingSpoolSink(spool)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=64,
            max_queue_bytes=64 * 1024**2,
            max_pack_bytes=1024**2,
            max_pack_records=8,
            max_linger_ns=60 * 1_000_000_000,
            overload_policy=OverloadPolicy.BLOCK,
        ),
        sink,
    )
    pipeline.start()
    for record in _records(tenant, count):
        assert pipeline.submit(record) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=30)
    assert snapshot.failures == 0
    assert snapshot.persisted_records == count
    assert snapshot.packs_persisted == len(sink.packs)
    return sink


def _uploader(spool: DurablePackSpool, store, events: list, **overrides):
    config = ParallelUploadConfig(
        max_workers=overrides.pop("max_workers", 4),
        max_in_flight_bytes=64 * 1024**2,
        **overrides,
    )
    return ParallelSpoolUploader(spool, store, config, event_callback=events.append)


def _object_keys(store: S3PackStore, prefix: str) -> list[str]:
    keys: list[str] = []
    cursor = None
    while True:
        page = store.list_objects(prefix=prefix, cursor=cursor, limit=1000)
        keys.extend(item.object_key for item in page.items)
        cursor = page.next_cursor
        if cursor is None:
            return sorted(keys)


def _delete_prefix(store: S3PackStore, prefix: str) -> None:
    for key in _object_keys(store, prefix):
        store._client.delete_object(Bucket=store._bucket, Key=key)


def test_garage_pipeline_spool_upload_commits_every_pack(tmp_path: Path):
    """CapturePipeline -> spool -> uploader -> Garage, with nothing lost."""

    store = _store()
    tenant = f"garage-pipeline-{uuid4().hex}"
    prefix = f"v1/tenant={tenant}/"
    try:
        sink = _spool_records(tmp_path / "spool", tenant, 6)
        spool = DurablePackSpool(tmp_path / "spool", max_bytes=256 * 1024**2)
        assert spool.snapshot().entries == len(sink.packs)

        events: list = []
        result = _uploader(spool, store, events).upload_pending()

        assert result.failures == ()
        assert result.snapshot.failed_packs == 0
        assert result.snapshot.retries == 0
        assert result.snapshot.attempted_packs == len(sink.packs)
        assert result.snapshot.uploaded_packs == len(sink.packs)
        assert [event.event for event in events] == [
            "pack_upload_committed"
        ] * len(sink.packs)
        assert {event.fields["attempt"] for event in events} == {1}

        # Every staged pack became exactly one object, and no others appeared.
        assert _object_keys(store, prefix) == sorted(sink.object_keys.values())
        assert {ref.object_key for ref in result.refs} == set(
            sink.object_keys.values()
        )

        for ref in result.refs:
            sealed = sink.packs[ref.pack_id]
            assert ref.object_key == sink.object_keys[ref.pack_id]
            assert ref.object_bytes == len(sealed.data)
            assert ref.checksum == sealed.checksum
            assert ref.record_count == sealed.record_count
            # Head metadata agrees with the ref...
            info = store.stat(ref)
            assert info.size == ref.object_bytes
            assert info.checksum == ref.checksum
            # ...and so do the bytes Garage actually stored.
            stored = store.read_range(ref, 0, ref.object_bytes)
            assert stored == sealed.data
            assert sha256(stored).hexdigest() == ref.checksum
            # The footer is readable straight out of the object.
            assert PackIndex.from_store(store, ref).descriptors()

        # Packs are removed only after a successful upload, so the spool is
        # now empty -- and stays empty across a fresh scan of the directory.
        assert spool.snapshot().entries == 0
        assert spool.snapshot().bytes == 0
        assert DurablePackSpool(
            tmp_path / "spool", max_bytes=256 * 1024**2
        ).recover() == ()
    finally:
        _delete_prefix(store, prefix)


def test_garage_restart_recovers_exactly_the_un_uploaded_packs(tmp_path: Path):
    """A crash between uploads loses nothing and duplicates nothing."""

    store = _store()
    tenant = f"garage-restart-{uuid4().hex}"
    prefix = f"v1/tenant={tenant}/"
    root = tmp_path / "spool"
    try:
        sink = _spool_records(root, tenant, 6)
        total = len(sink.packs)
        assert total >= 3, "the corpus must span several packs"

        first_spool = DurablePackSpool(root, max_bytes=256 * 1024**2)
        first_events: list = []
        # max_workers=1 so `limit` selects a deterministic prefix of the
        # recovered order rather than whichever packs a race finished first.
        first = _uploader(
            first_spool, store, first_events, max_workers=1
        ).upload_pending(limit=total - 2)
        assert first.snapshot.failed_packs == 0
        assert len(first.refs) == total - 2
        uploaded = {ref.pack_id for ref in first.refs}

        # The process dies here: brand new spool and uploader over the same
        # directory, with no in-memory state carried across.
        second_spool = DurablePackSpool(root, max_bytes=256 * 1024**2)
        pending = second_spool.recover()
        assert {staged.pack_id for staged in pending} == set(sink.packs) - uploaded
        assert len(pending) == 2

        second_events: list = []
        second = _uploader(second_spool, store, second_events).upload_pending()
        assert second.failures == ()
        assert second.snapshot.retries == 0
        assert {ref.pack_id for ref in second.refs} == {
            staged.pack_id for staged in pending
        }

        # Exactly one object per pack -- no duplicates, no losses.
        assert _object_keys(store, prefix) == sorted(sink.object_keys.values())
        assert len(sink.object_keys) == total
        for ref in (*first.refs, *second.refs):
            sealed = sink.packs[ref.pack_id]
            assert store.read_range(ref, 0, ref.object_bytes) == sealed.data
        assert second_spool.snapshot().entries == 0
    finally:
        _delete_prefix(store, prefix)


def test_garage_failed_upload_keeps_the_local_copy(tmp_path: Path):
    """A pack whose upload fails stays staged and lands on the next pass."""

    store = _store()
    tenant = f"garage-failure-{uuid4().hex}"
    prefix = f"v1/tenant={tenant}/"
    root = tmp_path / "spool"
    try:
        sink = _spool_records(root, tenant, 6)
        total = len(sink.packs)

        # Fail the first put of the pass. FaultInjected carries no HTTP
        # response, so the uploader classifies it as permanent and stops after
        # one attempt -- the case that must not delete the spooled bytes.
        faulty = FaultyPackStore(store, put=fail_on(1))
        spool = DurablePackSpool(root, max_bytes=256 * 1024**2)
        events: list = []
        first = _uploader(spool, faulty, events, max_workers=1).upload_pending()

        assert first.snapshot.failed_packs == 1
        assert first.snapshot.uploaded_packs == total - 1
        assert first.snapshot.retries == 0
        [failure] = first.failures
        assert failure.error_type == FaultInjected.__name__
        assert failure.attempts == 1
        assert [
            event.event for event in events if event.event != "pack_upload_committed"
        ] == ["pack_upload_failed"]

        # The failed pack is still on disk, and its object was never created.
        staged = spool.recover()
        assert [item.pack_id for item in staged] == [failure.pack_id]
        assert spool.snapshot().entries == 1
        assert _object_keys(store, prefix) == sorted(
            key
            for pack_id, key in sink.object_keys.items()
            if pack_id != failure.pack_id
        )

        # Next pass, no fault scheduled: the pack completes.
        second = _uploader(spool, faulty, events).upload_pending()
        assert second.failures == ()
        assert [ref.pack_id for ref in second.refs] == [failure.pack_id]
        assert spool.snapshot().entries == 0
        assert _object_keys(store, prefix) == sorted(sink.object_keys.values())
        ref = second.refs[0]
        assert store.read_range(ref, 0, ref.object_bytes) == sink.packs[
            ref.pack_id
        ].data
    finally:
        _delete_prefix(store, prefix)
