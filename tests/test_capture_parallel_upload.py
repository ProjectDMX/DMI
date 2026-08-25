from __future__ import annotations

from pathlib import Path
import threading
import time
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    DurablePackSink,
    DurablePackSpool,
    FilesystemPackStore,
    FlushReason,
    ParallelSpoolUploader,
    ParallelUploadConfig,
    PackWriter,
    ReadyPack,
)


pytestmark = pytest.mark.cpu


def _stage(spool: DurablePackSpool, index: int):
    metadata = CaptureMetadata(
        capture_id=f"capture-{index}",
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=index,
        producer_rank=0,
        step_number=index,
        token_start=index,
        token_end=index + 1,
        batch_position=0,
        dtype="uint8",
        shape=(256,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )
    writer = PackWriter(
        pack_id=UUID(f"018f0000-0000-7000-8000-{index:012d}"),
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=1024 * 1024,
    )
    writer.append(CaptureRecord(metadata=metadata, payload=bytes([index]) * 256))
    pack = writer.seal()
    return DurablePackSink(spool).persist(
        ReadyPack(pack, metadata, FlushReason.SHUTDOWN)
    )


class _MeasuredStore(FilesystemPackStore):
    def __init__(self, root: Path, *, fail_once: bool = False):
        super().__init__(root, store_id="remote")
        self.fail_once = fail_once
        self._failed: set[str] = set()
        self._lock = threading.Lock()
        self.active = 0
        self.peak_active = 0

    def put(self, pack, object_key):
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(0.01)
            if self.fail_once and object_key not in self._failed:
                self._failed.add(object_key)
                raise OSError("transient upload failure")
            return super().put(pack, object_key)
        finally:
            with self._lock:
                self.active -= 1


def test_parallel_uploader_bounds_workers_and_in_flight_bytes(tmp_path: Path):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    staged = [_stage(spool, index) for index in range(4)]
    budget = staged[0].object_bytes * 2
    remote = _MeasuredStore(tmp_path / "remote")
    uploader = ParallelSpoolUploader(
        spool,
        remote,
        ParallelUploadConfig(max_workers=4, max_in_flight_bytes=budget),
    )

    result = uploader.upload_pending()

    assert len(result.refs) == 4
    assert not result.failures
    assert result.snapshot.peak_active_uploads == 2
    assert result.snapshot.peak_in_flight_bytes <= budget
    assert remote.peak_active == 2
    assert spool.snapshot().entries == 0


def test_parallel_uploader_retries_transient_errors_without_losing_spool_files(
    tmp_path: Path,
):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    staged = [_stage(spool, index) for index in range(2)]
    remote = _MeasuredStore(tmp_path / "remote", fail_once=True)
    events = []
    uploader = ParallelSpoolUploader(
        spool,
        remote,
        ParallelUploadConfig(
            max_workers=2,
            max_in_flight_bytes=sum(item.object_bytes for item in staged),
            max_attempts=2,
            base_backoff_seconds=0,
        ),
        event_callback=events.append,
        sleep=lambda _: None,
    )

    result = uploader.upload_pending()

    assert len(result.refs) == 2
    assert result.snapshot.retries == 2
    assert result.snapshot.uploaded_packs == 2
    assert [event.event for event in events].count("pack_upload_retry") == 2
    assert spool.snapshot().entries == 0


def test_parallel_uploader_serializes_event_callbacks(tmp_path: Path):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    staged = [_stage(spool, index) for index in range(4)]
    remote = _MeasuredStore(tmp_path / "remote")
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def callback(_event):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.005)
        with lock:
            active -= 1

    result = ParallelSpoolUploader(
        spool,
        remote,
        ParallelUploadConfig(
            max_workers=4,
            max_in_flight_bytes=sum(item.object_bytes for item in staged),
        ),
        event_callback=callback,
    ).upload_pending()

    assert not result.failures
    assert peak_active == 1


def test_parallel_uploader_reports_permanent_failure_and_keeps_ready_pack(
    tmp_path: Path,
):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    staged = _stage(spool, 1)
    remote = _MeasuredStore(tmp_path / "remote", fail_once=True)
    uploader = ParallelSpoolUploader(
        spool,
        remote,
        ParallelUploadConfig(
            max_workers=1,
            max_in_flight_bytes=staged.object_bytes,
            max_attempts=1,
        ),
        sleep=lambda _: None,
    )

    result = uploader.upload_pending()

    assert not result.refs
    assert len(result.failures) == 1
    assert result.failures[0].pack_id == staged.pack_id
    assert result.snapshot.failed_packs == 1
    assert staged.path.exists()


def test_parallel_uploader_rejects_a_pack_larger_than_its_byte_budget(
    tmp_path: Path,
):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    staged = _stage(spool, 1)
    uploader = ParallelSpoolUploader(
        spool,
        FilesystemPackStore(tmp_path / "remote"),
        ParallelUploadConfig(
            max_workers=1, max_in_flight_bytes=staged.object_bytes - 1
        ),
    )

    with pytest.raises(ValueError, match="in-flight byte limit"):
        uploader.upload_pending()
