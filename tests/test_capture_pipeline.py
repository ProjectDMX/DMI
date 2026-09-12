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


def test_bounded_queue_refuses_a_flush_barrier_once_closed():
    """A closed queue must refuse control items, not just records.

    Nothing drains a closed queue, so a barrier accepted onto one would never
    complete and its waiter would sleep for its whole timeout -- forever, for
    the ``timeout=None`` callers ``flush`` still allows.
    """
    from dmi.storage.capture import pipeline as pipeline_module

    queue = BoundedRecordQueue(max_records=1, max_bytes=8)
    assert queue.put_barrier(
        pipeline_module._FlushBarrier(target_admitted=0)
    ) is AdmissionResult.ACCEPTED

    queue.close()

    assert queue.put_barrier(
        pipeline_module._FlushBarrier(target_admitted=0)
    ) is AdmissionResult.CLOSED


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


def test_pipeline_survives_a_duplicate_capture_id(tmp_path: Path):
    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=1 << 20,
            max_pack_bytes=1 << 20,
            max_pack_records=4,
            max_linger_ns=60_000_000_000,
        ),
        DirectPackSink(store),
        pack_id_factory=_ids(),
    )
    pipeline.start()

    # A producer retry after an ambiguous admission lands the same capture_id
    # twice in one open pack. That is a fault of the one record: the duplicate
    # is dropped and counted, and the pipeline keeps running.
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    assert snapshot.duplicate_records == 1
    assert snapshot.failures == 0
    assert snapshot.persisted_records == 2


def test_pack_assembler_flushes_an_expired_pack_under_continuous_traffic():
    assembler = PackAssembler(
        max_pack_bytes=1024 * 1024,
        max_records=100,
        max_linger_ns=10,
        pack_id_factory=_ids(),
    )

    assert assembler.append(_record("capture-a"), now_ns=0) == ()
    # The queue never goes idle, so expiry must be enforced on the append
    # path itself: the expired pack flushes as LINGER before the new record
    # opens a fresh one.
    emitted = assembler.append(_record("capture-b"), now_ns=25)

    assert [ready.reason for ready in emitted] == [FlushReason.LINGER]
    assert emitted[0].pack.record_count == 1
    flushed = assembler.flush(FlushReason.SHUTDOWN)
    assert flushed[0].pack.record_count == 1


# --- construction and admission validation --------------------------------------


def _config(**overrides) -> PipelineConfig:
    settings = dict(
        max_queue_records=4,
        max_queue_bytes=64,
        max_pack_bytes=1024 * 1024,
        max_pack_records=4,
        max_linger_ns=60_000_000_000,
    )
    settings.update(overrides)
    return PipelineConfig(**settings)


def test_bounded_queue_validates_construction_and_timeouts():
    with pytest.raises(ValueError, match="queue limits"):
        BoundedRecordQueue(max_records=0, max_bytes=8)
    with pytest.raises(ValueError, match="queue limits"):
        BoundedRecordQueue(max_records=1, max_bytes="8")  # type: ignore[arg-type]

    queue = BoundedRecordQueue(max_records=1, max_bytes=8)
    with pytest.raises(ValueError, match="timeout"):
        queue.put(_record("capture-a"), policy=OverloadPolicy.BLOCK, timeout=-1)
    with pytest.raises(ValueError, match="timeout"):
        queue.get(timeout=True)


def test_pack_assembler_validates_its_limits_and_clock():
    with pytest.raises(ValueError, match="must be positive"):
        PackAssembler(max_pack_bytes=0, max_records=10, max_linger_ns=100)

    assembler = PackAssembler(
        max_pack_bytes=1024 * 1024, max_records=10, max_linger_ns=100
    )
    with pytest.raises(ValueError, match="now_ns"):
        assembler.append(_record("capture-a"), now_ns=-1)


def test_pipeline_config_validates_bounds_and_admission_timeout():
    with pytest.raises(ValueError, match="bounds must be positive"):
        _config(max_queue_records=0)
    with pytest.raises(ValueError, match="admission_timeout"):
        _config(admission_timeout=-1.0)
    with pytest.raises(ValueError, match="admission_timeout"):
        _config(admission_timeout=True)


def test_pipeline_lifecycle_guards(tmp_path: Path):
    sink = DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    pipeline = HostCapturePipeline(_config(), sink, pack_id_factory=_ids())

    with pytest.raises(RuntimeError, match="not started"):
        pipeline.submit(_record("capture-a"))
    with pytest.raises(RuntimeError, match="not started"):
        pipeline.close()

    pipeline.start()
    with pytest.raises(RuntimeError, match="already been started"):
        pipeline.start()
    pipeline.close(timeout=2)


def test_pipeline_counts_records_dropped_under_overload(tmp_path: Path):
    from tests._faults import BlockingPackSink

    sink = BlockingPackSink(
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    )
    pipeline = HostCapturePipeline(
        _config(max_queue_records=1, max_pack_records=1),
        sink,
        pack_id_factory=_ids(),
    )
    pipeline.start()
    try:
        assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
        # Once the sink is entered, capture-a is out of the queue and the
        # persistence thread is stuck, so the next admissions are deterministic.
        assert sink.entered.wait(timeout=2)
        assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
        assert pipeline.submit(_record("capture-c")) is AdmissionResult.DROPPED
    finally:
        sink.release.set()
    snapshot = pipeline.close(timeout=2)

    assert snapshot.dropped_records == 1
    assert snapshot.admitted_records == 2


def test_pipeline_counts_admissions_that_time_out(tmp_path: Path):
    from tests._faults import BlockingPackSink

    sink = BlockingPackSink(
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    )
    pipeline = HostCapturePipeline(
        _config(
            max_queue_records=1,
            max_pack_records=1,
            overload_policy=OverloadPolicy.BLOCK,
            admission_timeout=0.0,
        ),
        sink,
        pack_id_factory=_ids(),
    )
    pipeline.start()
    try:
        assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
        assert sink.entered.wait(timeout=2)
        assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
        assert pipeline.submit(_record("capture-c")) is AdmissionResult.TIMED_OUT
    finally:
        sink.release.set()
    snapshot = pipeline.close(timeout=2)

    assert snapshot.timed_out_records == 1


def test_pipeline_counts_records_the_queue_cannot_ever_hold(tmp_path: Path):
    sink = DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    # The 8-byte payload fits a pack but never the queue.
    pipeline = HostCapturePipeline(
        _config(max_queue_bytes=4), sink, pack_id_factory=_ids()
    )
    pipeline.start()

    assert pipeline.submit(_record("capture-a")) is AdmissionResult.TOO_LARGE
    snapshot = pipeline.close(timeout=2)

    assert snapshot.oversized_records == 1
    assert snapshot.admitted_records == 0


def test_pipeline_counts_submissions_after_close(tmp_path: Path):
    sink = DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    pipeline = HostCapturePipeline(_config(), sink, pack_id_factory=_ids())
    pipeline.start()
    pipeline.close(timeout=2)

    assert pipeline.submit(_record("capture-a")) is AdmissionResult.CLOSED
    assert pipeline.snapshot().rejected_closed_records == 1


def test_close_times_out_while_the_sink_hangs(tmp_path: Path):
    from tests._faults import BlockingPackSink

    sink = BlockingPackSink(
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    )
    pipeline = HostCapturePipeline(
        _config(max_pack_records=1), sink, pack_id_factory=_ids()
    )
    pipeline.start()
    pipeline.submit(_record("capture-a"))
    assert sink.entered.wait(timeout=2)

    with pytest.raises(TimeoutError, match="did not stop"):
        pipeline.close(timeout=0.01)

    # Release the sink so the thread exits and a full close succeeds.
    sink.release.set()
    snapshot = pipeline.close(timeout=2)

    assert snapshot.packs_persisted == 1


def test_manual_flush_is_repeatable_and_does_not_close_pipeline(tmp_path: Path):
    pipeline = HostCapturePipeline(
        _config(max_linger_ns=60_000_000_000),
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local")),
        pack_id_factory=_ids(),
    )
    pipeline.start()

    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    assert pipeline.flush(timeout=2)
    first = pipeline.snapshot()
    assert first.persisted_records == 1
    assert first.flush_manual == 1
    assert pipeline.is_running

    assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
    assert pipeline.flush(timeout=2)
    second = pipeline.snapshot()
    assert second.persisted_records == 2
    assert second.flush_manual == 2
    assert pipeline.is_running

    closed = pipeline.close(timeout=2)
    assert closed.persisted_records == 2


def test_manual_flush_timeout_reuses_prefix_then_flushes_later_records(
    tmp_path: Path,
):
    from tests._faults import BlockingPackSink

    sink = BlockingPackSink(
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local"))
    )
    pipeline = HostCapturePipeline(
        _config(max_linger_ns=60_000_000_000),
        sink,
        pack_id_factory=_ids(),
    )
    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED

    assert not pipeline.flush(timeout=0.01)
    assert sink.entered.wait(timeout=2)
    assert pipeline.submit(_record("capture-b")) is AdmissionResult.ACCEPTED
    sink.release.set()

    assert pipeline.flush(timeout=2)
    snapshot = pipeline.snapshot()
    assert snapshot.persisted_records == 2
    assert snapshot.flush_manual == 2
    pipeline.close(timeout=2)


def test_manual_flush_after_close_returns_false_without_waiting(tmp_path: Path):
    """Flushing a closed pipeline must report False, never wait on a barrier.

    ``HostCapturePipeline`` is public API, and flush-after-close is an ordinary
    caller mistake -- ``record_adapter._flush_capture`` has no ``_closed`` guard
    of its own. The closed queue refuses the barrier, and nothing will ever
    drain it, so returning False is the only non-hanging answer.

    The barrier's wait is replaced rather than timed, so a regression fails
    this assertion immediately instead of depending on wall-clock time.
    """
    from dmi.storage.capture import pipeline as pipeline_module

    class _NeverWaitedEvent(threading.Event):
        def wait(self, timeout=None):
            pytest.fail("flush waited on a barrier the closed queue never accepted")

    class _NeverWaitedBarrier(pipeline_module._FlushBarrier):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("completed", _NeverWaitedEvent())
            super().__init__(*args, **kwargs)

    pipeline = HostCapturePipeline(
        _config(max_linger_ns=60_000_000_000),
        DirectPackSink(FilesystemPackStore(tmp_path, store_id="local")),
        pack_id_factory=_ids(),
    )
    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)
    assert snapshot.persisted_records == 1

    # Installed only now: the barrier this flush builds is the one that must
    # never be waited on.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline_module, "_FlushBarrier", _NeverWaitedBarrier)
        assert pipeline.flush(timeout=2) is False


def test_manual_flush_surfaces_worker_failure_without_hanging():
    pipeline = HostCapturePipeline(
        _config(max_linger_ns=60_000_000_000),
        _FailingSink(),
        pack_id_factory=_ids(),
    )
    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED

    with pytest.raises(PipelineFailedError, match="pipeline failed"):
        pipeline.flush(timeout=2)

    with pytest.raises(PipelineFailedError, match="pipeline failed"):
        pipeline.close(timeout=2)


def test_manual_flush_does_not_orphan_its_barrier_at_the_failure_seam(monkeypatch):
    """The worker must not decide "no waiter" before flush() has published one.

    ``flush`` reads ``_error``, then publishes its barrier, then enqueues it.
    The failure handler snapshots ``_pending_flush``, then closes the queue.
    Interleave the snapshot between flush's read and its publish and the
    barrier is published to nobody: the handler has already concluded there is
    no waiter, and the queue is still open, so the barrier is accepted onto a
    queue whose only consumer has stopped draining it.

    The window is a few bytecodes wide and unreachable unforced, so every step
    here is gated on an event rather than on timing, and the flush timeout is
    finite -- a regression must fail this assertion, never hang the suite.
    """
    from dmi.storage.capture import pipeline as pipeline_module

    persist_entered = threading.Event()
    persist_release = threading.Event()
    barrier_parked = threading.Event()
    barrier_resume = threading.Event()
    barrier_enqueued = threading.Event()
    close_entered = threading.Event()
    close_release = threading.Event()

    class _BlockingFailingSink:
        def persist(self, ready):
            persist_entered.set()
            assert persist_release.wait(timeout=10)
            raise OSError("sink unavailable")

    class _ParkedBarrier(pipeline_module._FlushBarrier):
        """Parks flush() between its ``_error`` read and the publish."""

        def __init__(self, *args, **kwargs):
            barrier_parked.set()
            assert barrier_resume.wait(timeout=10)
            super().__init__(*args, **kwargs)

    pipeline = HostCapturePipeline(
        _config(max_pack_records=1),
        _BlockingFailingSink(),
        pack_id_factory=_ids(),
    )
    queue = pipeline._queue
    put_barrier = queue.put_barrier
    close = queue.close

    def announcing_put_barrier(barrier):
        result = put_barrier(barrier)
        barrier_enqueued.set()
        return result

    def parked_close():
        # Parks the failure handler between its `_pending_flush` snapshot and
        # the close that would have turned a later put_barrier into CLOSED.
        close_entered.set()
        assert close_release.wait(timeout=10)
        close()

    monkeypatch.setattr(pipeline_module, "_FlushBarrier", _ParkedBarrier)
    monkeypatch.setattr(queue, "put_barrier", announcing_put_barrier)
    monkeypatch.setattr(queue, "close", parked_close)

    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    assert persist_entered.wait(timeout=10)

    outcome: list[object] = []

    def flush_once():
        try:
            outcome.append(pipeline.flush(timeout=2))
        except BaseException as exc:  # recorded, then asserted on below
            outcome.append(exc)

    flusher = threading.Thread(target=flush_once, name="orphan-flush")
    flusher.start()
    try:
        # flush() has read `_error` (None) and is parked before the publish.
        assert barrier_parked.wait(timeout=10)
        # Fail the worker: it latches `_error` and snapshots no waiter.
        persist_release.set()
        assert close_entered.wait(timeout=10)
        # Only now let flush() publish and enqueue onto the still-open queue.
        barrier_resume.set()
        assert barrier_enqueued.wait(timeout=10)
    finally:
        close_release.set()
        flusher.join(timeout=10)

    assert not flusher.is_alive()
    # The barrier will never be drained, so flush must report the failure it
    # can now observe instead of waiting out its timeout on a dead worker.
    assert isinstance(outcome[0], PipelineFailedError), outcome

    with pytest.raises(PipelineFailedError, match="pipeline failed"):
        pipeline.close(timeout=10)


def test_worker_failure_wakes_a_flush_already_waiting_on_its_barrier(monkeypatch):
    """The failure handler must complete a barrier that is already published.

    This is the sibling of the seam test above, and it needs the *opposite*
    interleaving. There, the handler snapshots ``_pending_flush`` before
    ``flush`` has published one, and the ``_error`` re-read is what saves the
    flush. Here it is the ordinary case: the sink raises on a record queued
    ahead of a barrier whose waiter is already asleep, so the handler's
    snapshot *does* see the barrier, and waking it is the only thing that can
    ever complete it.

    Gating the worker's failure on flush having entered ``completed.wait`` is
    what pins that arm: entering the wait proves the publish, a successful
    ``put_barrier`` and the ``_error`` re-read have all already happened with
    no error latched. The flush timeout stays finite so a regression fails the
    outcome assertion instead of hanging the suite.
    """
    from dmi.storage.capture import pipeline as pipeline_module

    persist_entered = threading.Event()
    persist_release = threading.Event()
    flush_parked = threading.Event()

    class _BlockingFailingSink:
        def persist(self, ready):
            persist_entered.set()
            assert persist_release.wait(timeout=10)
            raise OSError("sink unavailable")

    class _AnnouncingEvent(threading.Event):
        """Announces that flush() is about to sleep on its barrier."""

        def wait(self, timeout=None):
            flush_parked.set()
            return super().wait(timeout)

    class _AnnouncingBarrier(pipeline_module._FlushBarrier):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("completed", _AnnouncingEvent())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_FlushBarrier", _AnnouncingBarrier)

    pipeline = HostCapturePipeline(
        _config(max_pack_records=1),
        _BlockingFailingSink(),
        pack_id_factory=_ids(),
    )
    pipeline.start()
    assert pipeline.submit(_record("capture-a")) is AdmissionResult.ACCEPTED
    # The worker is parked inside persist for that record, so the barrier
    # queued below sits behind a record that is about to fail.
    assert persist_entered.wait(timeout=10)

    outcome: list[object] = []

    def flush_once():
        try:
            outcome.append(("returned", pipeline.flush(timeout=2)))
        except BaseException as exc:  # recorded, then asserted on below
            outcome.append(exc)

    flusher = threading.Thread(target=flush_once, name="waiting-flush")
    flusher.start()
    try:
        # flush() has published and enqueued its barrier, re-read `_error` as
        # None, and is now asleep on the barrier it owns.
        assert flush_parked.wait(timeout=10)
        # Fail the worker. Its snapshot sees this barrier, uncompleted.
        persist_release.set()
    finally:
        persist_release.set()
        flusher.join(timeout=10)

    assert not flusher.is_alive()
    # Nothing else will ever touch this barrier: the queue is closed and its
    # only consumer is gone. The failure handler's wake is the whole reason
    # flush reports the failure instead of waiting out its timeout.
    assert isinstance(outcome[0], PipelineFailedError), outcome

    with pytest.raises(PipelineFailedError, match="pipeline failed"):
        pipeline.close(timeout=10)
