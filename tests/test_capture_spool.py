from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    DurablePackSink,
    DurablePackSpool,
    FilesystemPackStore,
    FlushReason,
    PackIntegrityError,
    PackWriter,
    ReadyPack,
    SpoolFullError,
    SpoolUploader,
    HostCapturePipeline,
    PipelineConfig,
    AdmissionResult,
)


pytestmark = pytest.mark.cpu


def _sealed(pack_id: str, capture_id: str):
    metadata = CaptureMetadata(
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
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    writer = PackWriter(
        pack_id=UUID(pack_id),
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=1024 * 1024,
    )
    writer.append(CaptureRecord(metadata=metadata, payload=b"\x00" * 8))
    return writer.seal(), metadata


class _AmbiguousStore(FilesystemPackStore):
    def __init__(self, root: Path):
        super().__init__(root, store_id="remote")
        self.fail_once = True

    def put(self, pack, object_key):
        ref = super().put(pack, object_key)
        if self.fail_once:
            self.fail_once = False
            raise OSError("connection lost after commit")
        return ref


def test_spool_stage_is_atomic_bounded_and_restart_recoverable(tmp_path: Path):
    first, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    second, _ = _sealed(
        "018f0000-0000-7000-8000-000000000002", "capture-b"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(first.data))
    sink = DurablePackSink(spool)

    staged = sink.persist(ReadyPack(first, metadata, FlushReason.SHUTDOWN))

    assert staged.path.suffix == ".ready"
    assert staged.created_at_ns == first.created_at_ns
    assert staged.path.exists()
    assert not tuple((tmp_path / "spool").rglob("*.open"))
    assert spool.snapshot().bytes == len(first.data)
    with pytest.raises(SpoolFullError, match="spool byte limit"):
        sink.persist(ReadyPack(second, metadata, FlushReason.SHUTDOWN))

    recovered = DurablePackSpool(
        tmp_path / "spool", max_bytes=len(first.data)
    ).recover()
    assert recovered == (staged,)
    assert recovered[0].created_at_ns == first.created_at_ns


def test_spool_rejects_an_object_key_it_cannot_recover_exactly(tmp_path: Path):
    pack, _ = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)

    with pytest.raises(ValueError, match="pack ID"):
        spool.stage(pack, "v1/custom-name.dmi-pack")


def test_spool_upload_retry_resolves_an_ambiguous_remote_commit(tmp_path: Path):
    pack, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = DurablePackSink(spool).persist(
        ReadyPack(pack, metadata, FlushReason.SHUTDOWN)
    )
    remote = _AmbiguousStore(tmp_path / "objects")
    uploader = SpoolUploader(spool, remote)

    with pytest.raises(OSError, match="after commit"):
        uploader.upload(staged)
    assert staged.path.exists()

    ref = uploader.upload(staged)

    assert not staged.path.exists()
    assert remote.stat(ref).checksum == pack.checksum
    assert spool.snapshot().bytes == 0


def test_spool_recovery_rejects_corrupt_ready_pack(tmp_path: Path):
    pack, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    root = tmp_path / "spool"
    spool = DurablePackSpool(root, max_bytes=len(pack.data) * 2)
    staged = DurablePackSink(spool).persist(
        ReadyPack(pack, metadata, FlushReason.SHUTDOWN)
    )
    with staged.path.open("r+b") as handle:
        handle.seek(64)
        handle.write(b"\xff")

    with pytest.raises(PackIntegrityError, match="checksum"):
        DurablePackSpool(root, max_bytes=len(pack.data) * 2).recover()

    assert staged.path.exists()


def test_spool_recovery_removes_incomplete_open_files(tmp_path: Path):
    root = tmp_path / "spool"
    root.mkdir()
    incomplete = root / ".interrupted.open"
    incomplete.write_bytes(b"partial")

    spool = DurablePackSpool(root, max_bytes=1024)
    assert spool.snapshot().bytes == len(b"partial")

    recovered = spool.recover()

    assert recovered == ()
    assert not incomplete.exists()
    assert spool.snapshot().bytes == 0


def test_durable_pipeline_commits_to_spool_without_remote_storage(tmp_path: Path):
    pack, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    record = CaptureRecord(metadata=metadata, payload=b"\x00" * 8)
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=1_000_000_000,
        ),
        DurablePackSink(spool),
        pack_id_factory=lambda: UUID(
            "018f0000-0000-7000-8000-000000000003"
        ),
    )

    pipeline.start()
    assert pipeline.submit(record) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    assert snapshot.persisted_records == 1
    assert len(spool.recover()) == 1
