from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dmi.hooks.producer_plan import ProducerPlanEntry
from dmi.hooks.record import (
    HookPointV1,
    HookSpecV1,
    OutputStorage,
    RecordType,
    TransportSpec,
    TransportType,
)
from dmi.records import PayloadSlice, RecordRuntime
from dmi.storage.capture import (
    CaptureMetadata,
    CapturePackReferenceSink,
    CapturePayloadSlice,
    CaptureRecordFormat,
    HostCapturePipeline,
    PackReader,
    PipelineConfig,
    PipelineFailedError,
    record_adapter,
)

pytestmark = pytest.mark.cpu


def _metadata(
    capture_id: str,
    *,
    dtype: str = "float32",
    shape: tuple[int, ...] = (2,),
    step: int = 0,
) -> CaptureMetadata:
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
        dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _entry(
    *,
    shape: tuple[int, ...] = (2,),
    dtype: torch.dtype = torch.float32,
    reservation_bytes: int = 8,
) -> ProducerPlanEntry:
    return ProducerPlanEntry(
        output_id=17,
        input_shape=shape,
        output_shape=shape,
        dtype=dtype,
        transport_type=TransportType.IDENTITY,
        transport_args=(),
        storage=OutputStorage.TENSOR,
        record_type=RecordType.PER_SAMPLE,
        reservation_upper_bytes=reservation_bytes,
    )


class _CollectingSink:
    def __init__(self) -> None:
        self.ready = []

    def persist(self, ready) -> None:
        self.ready.append(ready)


class _FakeNativeReferenceSink:
    def __init__(self, target, layout) -> None:
        self.target = target
        self.layout = layout

    @property
    def attached(self) -> bool:
        return self.target.attached


def _pipeline(sink, *, queue_bytes: int = 64) -> HostCapturePipeline:
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=4,
            max_queue_bytes=queue_bytes,
            max_pack_bytes=1 << 20,
            max_pack_records=4,
            max_linger_ns=60_000_000_000,
        ),
        sink,
    )
    pipeline.start()
    return pipeline


@pytest.fixture
def fake_native_reference_sink(monkeypatch):
    # Set the module dictionary directly so its lazy __getattr__ does not try
    # to load the compiled extension in this CPU-only contract test.
    monkeypatch.setitem(
        record_adapter.native.__dict__,
        "ReferencePythonCaptureSink",
        _FakeNativeReferenceSink,
    )


def test_capture_record_format_encodes_single_and_multirow_payload_slices():
    record_format = CaptureRecordFormat()
    first = _metadata("capture-a")

    descriptor = record_format.encode(first, _entry())

    assert descriptor.layout == "capture_pack_reference_v1"
    assert descriptor.output_id == 17
    metadata_json, payload = descriptor.rows[0]
    assert json.loads(metadata_json) == first.to_mapping()
    assert payload == PayloadSlice(
        offset_bytes=0,
        nbytes=8,
        dtype=torch.float32,
        shape=(2,),
    )

    second = _metadata("capture-b", step=1)
    descriptor = record_format.encode(
        (
            CapturePayloadSlice(first, offset_bytes=0),
            CapturePayloadSlice(second, offset_bytes=8),
        ),
        _entry(shape=(4,), reservation_bytes=16),
    )

    assert [row[1].offset_bytes for row in descriptor.rows] == [0, 8]
    assert [json.loads(row[0])["capture_id"] for row in descriptor.rows] == [
        "capture-a",
        "capture-b",
    ]


def test_capture_record_format_rejects_shape_dtype_and_reservation_drift():
    record_format = CaptureRecordFormat()
    entry = _entry()

    with pytest.raises(ValueError, match="shape does not match"):
        record_format.encode(_metadata("capture-a", shape=(1,)), entry)
    with pytest.raises(ValueError, match="dtype does not match"):
        record_format.encode(_metadata("capture-a", dtype="float16"), entry)
    with pytest.raises(ValueError, match="exceeds producer reservation"):
        record_format.encode(
            (CapturePayloadSlice(_metadata("capture-a"), offset_bytes=4),),
            entry,
        )


def test_capture_record_format_rejects_device_gate_before_binding():
    class _Transport:
        def configure_record_schema(self, _schema):
            pass

    class _HookRuntime:
        def should_emit(self, _hook):
            return True

        def prepare_output(self, **_kwargs):
            return None

    runtime = RecordRuntime(_Transport(), CaptureRecordFormat())
    hook = HookPointV1(HookSpecV1("hook", (TransportSpec("out"),)))

    with pytest.raises(ValueError, match="does not support device-gated outputs"):
        runtime.bind_hook(
            hook,
            hook_runtime=_HookRuntime(),
            gate_tensor=torch.ones((), dtype=torch.int32),
            gate_value=1,
        )

    assert hook._output_ids == ()


def test_reference_sink_owns_bytes_and_flush_is_nonterminal_and_repeatable(
    fake_native_reference_sink,
):
    collecting = _CollectingSink()
    pipeline = _pipeline(collecting)
    sink = CapturePackReferenceSink(pipeline)
    native_sink = sink.native_sink
    assert sink.record_format.schema is CaptureRecordFormat.schema
    assert native_sink.layout == CaptureRecordFormat.LAYOUT_NAME
    callback = native_sink.target
    callback._attach()

    with pytest.raises(RuntimeError, match="close MonitoringEngine"):
        sink.close(timeout=2)

    first = torch.tensor([1.0, 2.0], dtype=torch.float32)
    expected = first.numpy().tobytes()
    callback._submit_capture(json.dumps(_metadata("capture-a").to_mapping()), first)
    first.zero_()

    assert callback._flush_capture(2.0) is True
    assert pipeline.is_running
    reader = PackReader.from_bytes(collecting.ready[0].pack.data)
    descriptor = reader.descriptors(store_id="memory", object_key="first")[0]
    assert reader.read_payload(descriptor) == expected

    second = torch.tensor([3.0, 4.0], dtype=torch.float32)
    callback._submit_capture(
        json.dumps(_metadata("capture-b", step=1).to_mapping()), second
    )
    assert callback._flush_capture(2.0) is True

    callback._detach()
    snapshot = sink.close(timeout=2)
    assert snapshot.persisted_records == 2
    assert snapshot.flush_manual == 2
    assert sink.close(timeout=2) == snapshot


def test_reference_sink_surfaces_any_non_durable_admission(
    fake_native_reference_sink,
):
    pipeline = _pipeline(_CollectingSink(), queue_bytes=4)
    sink = CapturePackReferenceSink(pipeline)
    native_sink = sink.native_sink
    callback = native_sink.target
    callback._attach()

    with pytest.raises(PipelineFailedError, match="not durable: too_large"):
        callback._submit_capture(
            json.dumps(_metadata("capture-a").to_mapping()),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
    with pytest.raises(PipelineFailedError, match="oversized_records"):
        callback._rethrow_capture()

    callback._detach()
    with pytest.raises(PipelineFailedError, match="oversized_records"):
        sink.close(timeout=2)
    with pytest.raises(PipelineFailedError, match="oversized_records"):
        sink.close(timeout=2)
    snapshot = pipeline.snapshot()
    assert snapshot.oversized_records == 1
    assert snapshot.persisted_records == 0


def test_capture_package_keeps_reference_adapter_and_native_backend_lazy():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dmi.storage.capture; "
                "assert 'dmi.storage.capture.record_adapter' not in sys.modules; "
                "assert 'dmi.transport.native' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
