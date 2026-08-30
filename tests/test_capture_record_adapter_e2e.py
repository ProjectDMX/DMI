"""Single-stream CUDA Ring -> Python capture-pack reference E2E."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import pytest
import torch

from dmi.records import PayloadSlice
from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]


class _CaptureHookRuntime:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self.metadata = None

    def should_emit(self, hook):
        return True

    def prepare_output(
        self,
        *,
        hook,
        output_index,
        output_id,
        output_spec,
        output,
    ):
        from dmi.api.v1 import ProducerPlanBuilder

        assert self.metadata is not None
        entry = ProducerPlanBuilder().record_output(
            output_id=output_id,
            output_spec=output_spec,
            output=output,
        )
        return self._runtime.emit_output(entry, self.metadata, output)


def _metadata(capture_id: str, tensor: torch.Tensor, *, step: int):
    from dmi.storage.capture import CaptureMetadata

    dtype = str(tensor.dtype).removeprefix("torch.")
    return CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-e2e",
        experiment_id="experiment-e2e",
        run_id="run-e2e",
        session_id="session-e2e",
        request_id=f"request-{step}",
        sequence_id="sequence-e2e",
        model_id="model-e2e",
        model_revision="revision-e2e",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="capture_tensor",
        layer_number=0,
        producer_rank=0,
        step_number=step,
        token_start=step,
        token_end=step + 1,
        batch_position=0,
        dtype=dtype,
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _read_pack(store, ref):
    from dmi.storage.capture import PackReader

    data = store.read_range(ref, 0, ref.object_bytes)
    reader = PackReader.from_bytes(data)
    descriptors = reader.descriptors(
        store_id=ref.store_id,
        object_key=ref.object_key,
    )
    return tuple(
        (descriptor, reader.read_payload(descriptor))
        for descriptor in descriptors
    )


def test_reference_adapter_ring_and_cpu_direct_reach_filesystem_pack(
    tmp_path: Path,
):
    from dmi.api.v1 import (
        HookPointV1,
        HookSpecV1,
        MonitoringEngine,
        RingConfig,
        TransportSpec,
    )
    from dmi.storage.capture import (
        CapturePackReferenceSink,
        CaptureRecordFormat,
        DirectPackSink,
        FilesystemPackStore,
        HostCapturePipeline,
        PipelineConfig,
    )

    store = FilesystemPackStore(tmp_path / "packs", store_id="local")
    pack_sink = DirectPackSink(store)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=8,
            max_queue_bytes=4 * 1024 * 1024,
            max_pack_bytes=4 * 1024 * 1024,
            max_pack_records=8,
            max_linger_ns=60_000_000_000,
        ),
        pack_sink,
    )
    pipeline.start()
    reference = CapturePackReferenceSink(pipeline)

    ring_config = RingConfig()
    ring_config.task_ring_entries = 32
    ring_config.payload_ring_bytes = 64 * 1024
    ring_config.pinned_staging_bytes = 64 * 1024
    engine = MonitoringEngine(model_id="capture-reference-e2e", ring_config=ring_config)

    try:
        runtime = engine.create_record_runtime(
            CaptureRecordFormat(),
            record_sink=reference.native_sink,
        )
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),))
        )
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)

        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        hook_runtime.metadata = _metadata("capture-ring", first, step=0)
        hook(first.cuda())
        engine.flush_and_wait(30.0)
        first_ref = pack_sink.last_ref
        assert first_ref is not None
        ((first_descriptor, first_bytes),) = _read_pack(store, first_ref)
        assert first_descriptor.capture_id == "capture-ring"
        assert first_bytes == first.numpy().tobytes()

        # This output exceeds the 64 KiB Ring and exercises the existing
        # CPU-direct fallback through the same adapter.  The first flush was
        # nonterminal, so a second record must still be accepted and durable.
        second = torch.arange(32 * 1024, dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-direct", second, step=1)
        hook(second.cuda())
        engine.flush_and_wait(30.0)
        second_ref = pack_sink.last_ref
        assert second_ref is not None and second_ref != first_ref
        ((second_descriptor, second_bytes),) = _read_pack(store, second_ref)
        assert second_descriptor.capture_id == "capture-direct"
        assert second_bytes == second.numpy().tobytes()

        scalar = torch.tensor(3.5, dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-scalar", scalar, step=2)
        hook(scalar.cuda())

        empty = torch.empty((0, 3), dtype=torch.bfloat16)
        hook_runtime.metadata = _metadata("capture-empty", empty, step=3)
        hook(empty.cuda())

        from dmi.storage.capture import CapturePayloadSlice

        joined = torch.tensor([10.0, 11.0, 20.0, 21.0], dtype=torch.float32)
        left = _metadata("capture-left", joined[:2], step=4)
        right = _metadata("capture-right", joined[2:], step=5)
        hook_runtime.metadata = (
            CapturePayloadSlice(left, offset_bytes=0),
            CapturePayloadSlice(right, offset_bytes=8),
        )
        hook(joined.cuda())
        engine.flush_and_wait(30.0)
        third_ref = pack_sink.last_ref
        assert third_ref is not None and third_ref not in (first_ref, second_ref)
        rows = _read_pack(store, third_ref)
        assert [(row.capture_id, payload) for row, payload in rows] == [
            ("capture-scalar", scalar.numpy().tobytes()),
            ("capture-empty", empty.view(torch.uint8).numpy().tobytes()),
            ("capture-left", joined[:2].numpy().tobytes()),
            ("capture-right", joined[2:].numpy().tobytes()),
        ]

    finally:
        engine.close()
        reference.close(timeout=30.0)

    snapshot = pipeline.snapshot()
    assert snapshot.persisted_records == 6
    assert snapshot.flush_manual == 3


def test_reference_callback_failure_is_stable_across_repeated_flushes(
    tmp_path: Path,
):
    from dmi.api.v1 import (
        HookPointV1,
        HookSpecV1,
        MonitoringEngine,
        RingConfig,
        TransportSpec,
    )
    from dmi.storage.capture import (
        CapturePackReferenceSink,
        CaptureRecordFormat,
        DirectPackSink,
        FilesystemPackStore,
        HostCapturePipeline,
        PipelineConfig,
        PipelineFailedError,
    )

    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=1,
            max_queue_bytes=4,
            max_pack_bytes=1024 * 1024,
            max_pack_records=4,
            max_linger_ns=60_000_000_000,
        ),
        DirectPackSink(FilesystemPackStore(tmp_path / "packs", store_id="local")),
    )
    pipeline.start()
    reference = CapturePackReferenceSink(pipeline)
    ring_config = RingConfig()
    ring_config.task_ring_entries = 16
    ring_config.payload_ring_bytes = 64 * 1024
    ring_config.pinned_staging_bytes = 64 * 1024
    engine = MonitoringEngine(
        model_id="capture-reference-failure", ring_config=ring_config
    )

    close_error = None
    try:
        runtime = engine.create_record_runtime(
            CaptureRecordFormat(),
            record_sink=reference.native_sink,
        )
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),))
        )
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        value = torch.tensor([1.0, 2.0], dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-rejected", value, step=0)
        hook(value.cuda())

        messages = []
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                engine.flush_and_wait(5.0)
            messages.append(str(caught.value))
        assert all("not durable: too_large" in message for message in messages)
    finally:
        engine.close()
        try:
            reference.close(timeout=5.0)
        except PipelineFailedError as exc:
            close_error = exc

    assert close_error is not None
    assert "oversized_records" in str(close_error)


class _NativeValidationTarget:
    def _attach(self):
        pass

    def _detach(self):
        pass

    def _submit_capture(self, metadata_json, payload):
        expected = getattr(torch, json.loads(metadata_json)["dtype"])
        if payload.dtype != expected:
            raise ValueError(f"semantic dtype mismatch: {payload.dtype} != {expected}")

    def _flush_capture(self, _timeout_s):
        return True

    def _rethrow_capture(self):
        pass


@pytest.mark.parametrize(
    ("payload_slice", "metadata_json", "message"),
    (
        (
            PayloadSlice(offset_bytes=1, nbytes=4, dtype=torch.float32, shape=(1,)),
            '{"dtype":"float32"}',
            "aligned",
        ),
        (
            PayloadSlice(
                offset_bytes=510, nbytes=4, dtype=torch.float32, shape=(1,)
            ),
            '{"dtype":"float32"}',
            "exceeds physical payload",
        ),
        (
            PayloadSlice(offset_bytes=0, nbytes=8, dtype=torch.float32, shape=(3,)),
            '{"dtype":"float32"}',
            "shape do not match declared bytes",
        ),
        (
            PayloadSlice(offset_bytes=0, nbytes=4, dtype=torch.float64, shape=(1,)),
            '{"dtype":"float64"}',
            "dtype and shape do not match declared bytes",
        ),
        (
            PayloadSlice(offset_bytes=0, nbytes=8, dtype=torch.float64, shape=(1,)),
            '{"dtype":"float32"}',
            "semantic dtype mismatch",
        ),
    ),
)
def test_reference_native_sink_rejects_wire_drift_stably(
    payload_slice,
    metadata_json,
    message,
):
    from dmi.records import RecordDescriptor
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.transport import native

    target = _NativeValidationTarget()
    sink = native.ReferencePythonCaptureSink(target)
    sink._attach_target()
    config = native.RingConfig()
    config.task_ring_entries = 8
    config.payload_ring_bytes = 256
    config.pinned_staging_bytes = 256
    engine = native.RingEngine.create_record(config, sink)
    engine.init()
    engine.start()
    source = torch.zeros(512, dtype=torch.uint8)
    descriptor = RecordDescriptor(
        "capture_pack_reference_v1",
        ((metadata_json, payload_slice),),
        output_id=7,
    )

    try:
        assert engine.reserve_record(((source.nbytes, False),)) == 2
        engine.push_record_descriptors((descriptor,), CaptureRecordFormat.schema)
        engine.submit_record_cpu_direct(source, source.nbytes)
        errors = []
        for _ in range(2):
            with pytest.raises(RuntimeError, match=message) as caught:
                engine.flush_records_and_wait(5.0)
            errors.append(str(caught.value))
        assert errors[0] == errors[1]
    finally:
        engine.stop()
        sink._detach_target()
        sink._release_target()


def _run_reference_gc_probe(root: Path) -> None:
    from dmi.engine import MonitoringEngine
    from dmi.hooks.producer_plan import ProducerPlanEntry
    from dmi.hooks.record import OutputStorage, RecordType, TransportType
    from dmi.storage.capture import (
        CapturePackReferenceSink,
        CaptureRecordFormat,
        DirectPackSink,
        FilesystemPackStore,
        HostCapturePipeline,
        PipelineConfig,
    )
    from dmi.transport import native
    from dmi.transport import ring as ring_transport

    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=4096,
            max_pack_bytes=4096,
            max_pack_records=4,
            max_linger_ns=60_000_000_000,
        ),
        DirectPackSink(FilesystemPackStore(root / "packs", store_id="local")),
    )
    pipeline.start()
    reference = CapturePackReferenceSink(pipeline)
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    original_submit = reference._target._submit_capture

    def blocking_submit(metadata_json, payload):
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("callback release did not arrive")
        try:
            return original_submit(metadata_json, payload)
        finally:
            completed.set()

    reference._target._submit_capture = blocking_submit
    config = native.RingConfig()
    config.task_ring_entries = 8
    config.payload_ring_bytes = 256
    config.pinned_staging_bytes = 256
    engine = MonitoringEngine(model_id="gc-probe", ring_config=config)
    runtime = engine.create_record_runtime(
        CaptureRecordFormat(), record_sink=reference.native_sink
    )
    source = torch.zeros(128, dtype=torch.float32)
    entry = ProducerPlanEntry(
        output_id=7,
        input_shape=(128,),
        output_shape=(128,),
        dtype=torch.float32,
        transport_type=TransportType.IDENTITY,
        transport_args=(),
        storage=OutputStorage.TENSOR,
        record_type=RecordType.PER_SAMPLE,
        reservation_upper_bytes=512,
    )
    descriptor = CaptureRecordFormat().encode(
        _metadata("capture-gc", source, step=0), entry
    )
    assert engine._ring_engine.reserve_record(((source.nbytes, False),)) == 2
    engine._ring_engine.push_record_descriptors(
        (descriptor,), CaptureRecordFormat.schema
    )
    engine._ring_engine.submit_record_cpu_direct(source, source.nbytes)
    assert entered.wait(5.0)

    def release_later():
        time.sleep(0.35)
        release.set()

    helper = threading.Thread(target=release_later)
    helper.start()
    ring_transport.deactivate()
    engine_ref = weakref.ref(engine)
    reference_ref = weakref.ref(reference)
    del runtime
    del engine
    gc.collect()
    helper.join(2.0)
    assert not helper.is_alive()
    assert completed.is_set()
    assert engine_ref() is None
    del reference
    gc.collect()
    assert reference_ref() is None
    pipeline.close(timeout=2.0)


def test_reference_gc_with_inflight_callback_does_not_deadlock(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(Path(__file__)), "--gc-probe", str(tmp_path)],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(repo_root), str(repo_root / "src"))),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--gc-probe":
    _run_reference_gc_probe(Path(sys.argv[2]))
