from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path
import threading
from uuid import UUID

import pytest

from dmi.storage.capture import (
    AdmissionResult,
    BoundedRecordQueue,
    CaptureMetadata,
    CaptureRecord,
    DirectPackSink,
    DuplicateCaptureError,
    FilesystemPackStore,
    FlushReason,
    HostCapturePipeline,
    OverloadPolicy,
    PackAssembler,
    PipelineConfig,
    PipelineFailedError,
    object_key_for,
)


pytestmark = pytest.mark.cpu


def _metadata(capture_id: str, *, session_id: str = "session-a") -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant/a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id=session_id,
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


def _record(capture_id: str, *, session_id: str = "session-a") -> CaptureRecord:
    return CaptureRecord(
        metadata=_metadata(capture_id, session_id=session_id),
        payload=b"\x00\x00\x80?\x00\x00\x00@",
    )


def _ids():
    values = iter(
        (
            UUID("018f0000-0000-7000-8000-000000000001"),
            UUID("018f0000-0000-7000-8000-000000000002"),
            UUID("018f0000-0000-7000-8000-000000000003"),
            UUID("018f0000-0000-7000-8000-000000000004"),
            UUID("018f0000-0000-7000-8000-000000000005"),
        )
    )
    return lambda: next(values)


def test_bounded_queue_enforces_record_and_byte_limits():
    queue = BoundedRecordQueue(max_records=1, max_bytes=8)

    assert queue.put(_record("capture-a"), policy=OverloadPolicy.DROP_NEWEST) \
        is AdmissionResult.ACCEPTED
    assert queue.put(_record("capture-b"), policy=OverloadPolicy.DROP_NEWEST) \
        is AdmissionResult.DROPPED
    snapshot = queue.snapshot()

    assert snapshot.records == 1
    assert snapshot.bytes == 8
    assert snapshot.peak_records == 1
    assert snapshot.peak_bytes == 8
    assert queue.get(timeout=0) == _record("capture-a")


def test_bounded_queue_reports_oversized_timeout_and_close():
    queue = BoundedRecordQueue(max_records=1, max_bytes=8)

    assert queue.put(
        replace(_record("capture-a"), payload=b"\x00" * 8),
        policy=OverloadPolicy.BLOCK,
    ) is AdmissionResult.ACCEPTED
    assert queue.put(
        _record("capture-b"), policy=OverloadPolicy.BLOCK, timeout=0
    ) is AdmissionResult.TIMED_OUT
    oversized = CaptureRecord(
        metadata=replace(_metadata("capture-c"), shape=(3,)),
        payload=b"\x00" * 12,
    )
    assert queue.put(oversized, policy=OverloadPolicy.DROP_NEWEST) \
        is AdmissionResult.TOO_LARGE

    queue.close()
    assert queue.put(_record("capture-d"), policy=OverloadPolicy.DROP_NEWEST) \
        is AdmissionResult.CLOSED


def test_bounded_queue_stays_within_limits_under_sustained_overload():
    queue = BoundedRecordQueue(max_records=3, max_bytes=24)

    results = [
        queue.put(_record(f"capture-{index}"), policy=OverloadPolicy.DROP_NEWEST)
        for index in range(100)
    ]

    assert results.count(AdmissionResult.ACCEPTED) == 3
    assert results.count(AdmissionResult.DROPPED) == 97
    assert queue.snapshot().peak_records == 3
    assert queue.snapshot().peak_bytes == 24


def test_bounded_queue_rejects_an_unknown_overload_policy():
    queue = BoundedRecordQueue(max_records=1, max_bytes=8)
    queue.put(_record("capture-a"), policy=OverloadPolicy.DROP_NEWEST)

    with pytest.raises(ValueError, match="overload policy"):
        queue.put(_record("capture-b"), policy="unknown", timeout=0)  # type: ignore[arg-type]


def test_pipeline_config_rejects_an_unknown_overload_policy():
    with pytest.raises(ValueError, match="overload policy"):
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=100,
            overload_policy="drop",  # type: ignore[arg-type]
        )


def test_pack_assembler_seals_on_size_linger_session_and_shutdown():
    first = _record("capture-a")
    probe = PackAssembler(
        max_pack_bytes=1024 * 1024,
        max_records=10,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    probe.append(first, now_ns=0)
    one_pack_bytes = len(probe.flush(FlushReason.SHUTDOWN)[0].pack.data)

    assembler = PackAssembler(
        max_pack_bytes=one_pack_bytes + 16,
        max_records=10,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    assert assembler.append(first, now_ns=0) == ()
    size_flush = assembler.append(_record("capture-b"), now_ns=10)
    assert size_flush[0].reason is FlushReason.SIZE
    assert size_flush[0].pack.record_count == 1

    linger_flush = assembler.flush_expired(now_ns=110)
    assert linger_flush[0].reason is FlushReason.LINGER
    assert linger_flush[0].pack.record_count == 1

    assert assembler.append(_record("capture-c"), now_ns=200) == ()
    session_flush = assembler.append(
        _record("capture-d", session_id="session-b"), now_ns=210
    )
    assert session_flush[0].reason is FlushReason.SESSION
    shutdown_flush = assembler.flush(FlushReason.SHUTDOWN)
    assert shutdown_flush[0].reason is FlushReason.SHUTDOWN


def test_pack_assembler_rejects_duplicates_without_losing_the_open_pack():
    assembler = PackAssembler(
        max_pack_bytes=1024 * 1024,
        max_records=10,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    assembler.append(_record("capture-a"), now_ns=0)

    with pytest.raises(DuplicateCaptureError, match="capture-a"):
        assembler.append(_record("capture-a"), now_ns=1)

    flushed = assembler.flush(FlushReason.SHUTDOWN)
    assert flushed[0].pack.record_count == 1


def test_pack_assembler_rejects_oversized_input_without_losing_prior_records():
    first = _record("capture-a")
    probe = PackAssembler(
        max_pack_bytes=1024 * 1024,
        max_records=10,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    probe.append(first, now_ns=0)
    one_pack_bytes = len(probe.flush(FlushReason.SHUTDOWN)[0].pack.data)
    assembler = PackAssembler(
        max_pack_bytes=one_pack_bytes,
        max_records=10,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    assembler.append(first, now_ns=0)
    oversized = CaptureRecord(
        metadata=replace(_metadata("capture-b"), shape=(1024,)),
        payload=b"\x00" * 4096,
    )

    with pytest.raises(ValueError, match="does not fit an empty pack"):
        assembler.append(oversized, now_ns=1)

    flushed = assembler.flush(FlushReason.SHUTDOWN)
    assert flushed[0].pack.record_count == 1


def test_object_key_bounds_long_metadata_components(tmp_path: Path):
    record = replace(
        _record("capture-a"),
        metadata=replace(
            _metadata("capture-a"),
            tenant_id="a" * 512,
            session_id="b" * 512,
        ),
    )
    assembler = PackAssembler(
        max_pack_bytes=1024 * 1024,
        max_records=2,
        max_linger_ns=100,
        pack_id_factory=_ids(),
    )
    assembler.append(record, now_ns=0)
    ready = assembler.flush(FlushReason.SHUTDOWN)[0]

    key = object_key_for(ready)
    ref = FilesystemPackStore(tmp_path, store_id="local").put(ready.pack, key)

    assert max(len(part.encode()) for part in key.split("/")) <= 200
    assert ref.object_key == key


def test_direct_pipeline_persists_on_close_and_reports_bounded_metrics(
    tmp_path: Path,
):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    sink = DirectPackSink(store)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=32,
            max_pack_bytes=1024 * 1024,
            max_pack_records=4,
            max_linger_ns=1_000_000_000,
        ),
        sink,
        pack_id_factory=_ids(),
    )

    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    assert snapshot.admitted_records == 2
    assert snapshot.persisted_records == 2
    assert snapshot.packs_persisted == 1
    assert snapshot.queue_peak_bytes <= 32
    assert snapshot.queue_peak_records <= 4
    assert snapshot.flush_shutdown == 1
    assert snapshot.failures == 0
    assert snapshot.admission_duration.count == 2
    assert snapshot.persist_duration.count == 1
    assert sink.last_ref is not None
    assert "%2F" in sink.last_ref.object_key
    assert store.stat(sink.last_ref).size == snapshot.packed_bytes


class _FailingSink:
    def persist(self, ready):
        raise OSError("sink unavailable")


def test_pipeline_surfaces_sink_failure_and_closes_admission():
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=1_000_000_000,
        ),
        _FailingSink(),
        pack_id_factory=_ids(),
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))

    with pytest.raises(PipelineFailedError, match="pipeline failed"):
        pipeline.close(timeout=2)

    assert pipeline.snapshot().failures == 1


def test_pipeline_is_not_failed_by_an_observability_callback(tmp_path: Path):
    def broken_callback(event):
        raise RuntimeError("telemetry unavailable")

    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=100,
        ),
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local")),
        pack_id_factory=_ids(),
        event_callback=broken_callback,
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))

    snapshot = pipeline.close(timeout=2)

    assert snapshot.failures == 0
    assert snapshot.event_callback_failures == 1


def test_pipeline_flushes_an_idle_pack_at_its_linger_deadline(tmp_path: Path):
    persisted = threading.Event()
    ticks = itertools.count(start=0, step=10)

    def capture_event(event):
        if event.event == "pack_persisted":
            persisted.set()

    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=1,
        ),
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local")),
        pack_id_factory=_ids(),
        clock_ns=lambda: next(ticks),
        event_callback=capture_event,
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))

    assert persisted.wait(timeout=1)
    snapshot = pipeline.close(timeout=2)

    assert snapshot.flush_linger == 1
    assert snapshot.flush_shutdown == 0


def test_pipeline_rejects_a_capture_no_pack_could_hold(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            # The queue would happily take it; max_pack_bytes is what it cannot
            # satisfy, and the two bounds are unrelated.
            max_queue_bytes=1 << 20,
            max_pack_bytes=4096,
            max_pack_records=4,
            max_linger_ns=1_000_000_000,
        ),
        DirectPackSink(store),
        pack_id_factory=_ids(),
    )
    pipeline.start()

    oversized = CaptureRecord(
        metadata=replace(_metadata("capture-big"), shape=(2048,)),
        payload=b"\x00" * 8192,
    )
    assert pipeline.submit(oversized) is AdmissionResult.TOO_LARGE
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    # Rejected at admission, so the caller knows, and the pipeline lives.
    assert snapshot.oversized_records == 1
    assert snapshot.failures == 0
    assert snapshot.persisted_records == 1


def test_pipeline_survives_a_capture_only_framing_pushes_over(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    payload = b"\x00" * 4096
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=1 << 20,
            # The payload itself fits, so admission passes; header, footer and
            # trailer are what take it past the limit inside the assembler.
            max_pack_bytes=len(payload) + 64,
            max_pack_records=4,
            max_linger_ns=1_000_000_000,
        ),
        DirectPackSink(store),
        pack_id_factory=_ids(),
    )
    pipeline.start()

    big = CaptureRecord(
        metadata=replace(_metadata("capture-big"), shape=(1024,)), payload=payload
    )
    assert pipeline.submit(big) is AdmissionResult.ACCEPTED
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    # The one record is dropped and counted; the pipeline is not failed and the
    # following capture still persists.
    assert snapshot.oversized_records == 1
    assert snapshot.failures == 0
    assert snapshot.persisted_records == 1
