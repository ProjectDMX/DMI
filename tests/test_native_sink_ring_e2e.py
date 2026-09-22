"""Single-GPU Ring -> NativePackSink: the production writer on a real engine.

The standalone torch suite (test_native_adapter_torch.py) drives
NativePackSink with synthetic envelopes and no engine. This is the boundary
it cannot exercise: BOTH extensions loaded in one process, the sink handed
to ``create_record_runtime`` as its ``record_sink``, records flowing from a
CUDA tensor through the Ring into a staged pack the Python reader opens.

Two things had to hold for this to run at all, and are asserted first:

* loading ``_dmi_native_sink`` beside ``_native_backend`` must not fail with
  pybind11's "type RecordSinkLease is already registered";
* ``NativePackSink`` must be an instance of ``_native_backend.RecordSink``
  with the inherited ``_acquire_engine`` -- the engine refuses anything
  else with "record_sink must be a native RecordSink". A module-local
  RecordSink registration imported fine and failed exactly there.

Build: make -C native && make -C native build/_dmi_native_sink PYTHON=...
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
SINK_BUILT = bool(
    sorted((REPO_ROOT / "native" / "build").glob("_dmi_native_sink*.so"))
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
    pytest.mark.skipif(
        not SINK_BUILT,
        reason="native/build/_dmi_native_sink*.so is not built; run "
        "`make -C native build/_dmi_native_sink PYTHON=<venv>/bin/python`",
    ),
]


class _CaptureHookRuntime:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self.metadata = None
        # RecordRuntime.emit_output's return value is the ONLY observable
        # that says which transport carried each record: it is OVERSIZED
        # exactly on the CPU-direct branch (records.py:333) and a ring
        # reservation otherwise. Persisted counts and payload bytes are
        # identical either way, so without this the test cannot tell a
        # record that went through the Ring from one that bypassed it.
        self.reservations = []

    def should_emit(self, hook):
        return True

    def prepare_output(self, *, hook, output_index, output_id, output_spec,
                       output):
        from dmi.api.v1 import ProducerPlanBuilder

        assert self.metadata is not None
        entry = ProducerPlanBuilder().record_output(
            output_id=output_id, output_spec=output_spec, output=output,
        )
        reservation = self._runtime.emit_output(entry, self.metadata, output)
        self.reservations.append(reservation)
        return reservation


def _metadata(capture_id: str, tensor: torch.Tensor, *, step: int):
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=capture_id, tenant_id="tenant-e2e",
        experiment_id="experiment-e2e", run_id="run-e2e",
        session_id="session-e2e", request_id=f"request-{step}",
        sequence_id="sequence-e2e", model_id="model-e2e",
        model_revision="revision-e2e", adapter_revision=None,
        capture_policy_version="policy-v1", hook_name="capture_tensor",
        layer_number=0, producer_rank=0, step_number=step, token_start=step,
        token_end=step + 1, batch_position=0,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _staged_records(spool_root: Path):
    from dmi.storage.capture import DurablePackSpool, PackReader

    out = []
    for entry in DurablePackSpool(spool_root, max_bytes=1 << 40).recover():
        with entry.open() as handle:
            reader = PackReader.from_bytes(handle.read())
        for descriptor in reader.descriptors(
                store_id="spool", object_key=entry.object_key):
            out.append((descriptor.metadata.capture_id,
                        reader.read_payload(descriptor)))
    return out


def _ring_config():
    from dmi.api.v1 import RingConfig

    config = RingConfig()
    config.task_ring_entries = 32
    config.payload_ring_bytes = 64 * 1024
    config.pinned_staging_bytes = 64 * 1024
    return config


def test_native_pack_sink_is_the_engines_record_sink(tmp_path: Path):
    """Both extensions load together and the sink IS a ring RecordSink."""
    from dmi.storage.capture.native_sink import (
        NativeSinkConfig, create_native_pack_sink,
    )
    from dmi.transport import native

    handle = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path / "spool")))
    assert isinstance(handle.native_sink, native.RecordSink)
    assert hasattr(handle.native_sink, "_acquire_engine")
    assert type(handle.native_sink).__mro__[1] is native.RecordSink
    from dmi.storage.capture.native_sink import _load_native_sink_extension

    assert _load_native_sink_extension().RING_TYPES_ARE_STANDINS is False


def test_records_flow_from_the_ring_into_a_native_pack(tmp_path: Path):
    """The explicit entry point: record_sink=create_native_pack_sink(...).

    Three records, and the point is that they do NOT all take the same
    route: two fit the 64 KiB payload ring and one does not. The persisted
    count and the payload bytes are the same whichever transport ran, so
    the per-record StepReservation is what pins the split.
    """
    from dmi.adapters.base import StepReservation
    from dmi.api.v1 import (
        HookPointV1, HookSpecV1, MonitoringEngine, TransportSpec,
    )
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.storage.capture.native_sink import (
        NativeSinkConfig, create_native_pack_sink,
    )

    spool_root = tmp_path / "spool"
    handle = create_native_pack_sink(NativeSinkConfig(
        spool_root=str(spool_root), max_pack_records=8,
        max_linger_ns=60_000_000_000))
    engine = MonitoringEngine(model_id="native-sink-e2e",
                              ring_config=_ring_config())
    try:
        runtime = engine.create_record_runtime(
            CaptureRecordFormat(), record_sink=handle.native_sink)
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)

        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        hook_runtime.metadata = _metadata("capture-ring", first, step=0)
        hook(first.cuda())

        # 128 KiB, above the 64 KiB Ring: the CPU-direct fallback, same
        # sink. Asserted, not just asserted-in-a-comment, below.
        second = torch.arange(32 * 1024, dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-direct", second, step=1)
        hook(second.cuda())

        scalar = torch.tensor(3.5, dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-scalar", scalar, step=2)
        hook(scalar.cuda())

        engine.flush_and_wait(30.0)
        assert handle.native_sink.flush_and_wait(30.0)
        handle.native_sink.rethrow_if_failed()
        snapshot = handle.native_sink.snapshot()
        assert snapshot["persisted_records"] == 3, snapshot
        assert snapshot["failures"] == 0, snapshot
        # 48 bytes, then 128 KiB, then 4 bytes against a 64 KiB payload
        # ring: reserved in the ring, CPU-direct, reserved in the ring.
        # RESERVED and not FLUSHED because neither ring record comes near
        # filling the ring, so nothing forces a drain.
        assert hook_runtime.reservations == [
            StepReservation.RESERVED,
            StepReservation.OVERSIZED,
            StepReservation.RESERVED,
        ], hook_runtime.reservations
    finally:
        engine.close()

    staged = dict(_staged_records(spool_root))
    assert staged == {
        "capture-ring": first.numpy().tobytes(),
        "capture-direct": second.numpy().tobytes(),
        "capture-scalar": scalar.numpy().tobytes(),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_graph_replays_preserve_payloads_across_ring_wraps(tmp_path: Path, dtype):
    from dmi.api.v1 import (
        HookPointV1, HookSpecV1, MonitoringEngine, ProducerPlanBuilder,
        TransportSpec,
    )
    from dmi.adapters.base import StepReservation
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.storage.capture.native_sink import NativeSinkConfig, create_native_pack_sink

    class GraphHookRuntime(_CaptureHookRuntime):
        builder = None

        def prepare_output(self, **kwargs):
            if self.builder is None:
                return super().prepare_output(**kwargs)
            self.builder.record_output(
                output_id=kwargs["output_id"], output_spec=kwargs["output_spec"],
                output=kwargs["output"],
            )
            return None

    spool_root = tmp_path / "spool"
    handle = create_native_pack_sink(NativeSinkConfig(
        spool_root=str(spool_root), max_pack_records=7,
        max_linger_ns=60_000_000_000))
    engine = MonitoringEngine(model_id="native-graph-e2e", ring_config=_ring_config())
    expected = {}
    try:
        runtime = engine.create_record_runtime(
            CaptureRecordFormat(), record_sink=handle.native_sink)
        hook = HookPointV1(HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = GraphHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        values = torch.arange(33 * 17).reshape(33, 17)
        static_input = values.to(device="cuda", dtype=dtype)

        def metadata_and_bytes(step):
            tensor = (values + step).to(dtype).t().contiguous()
            capture_id = f"graph-{step}"
            expected[capture_id] = tensor.view(torch.uint8).numpy().tobytes()
            return _metadata(capture_id, tensor, step=step)

        # Warm up the normal producer first, then record only its physical
        # plan during capture. Fresh FIFO metadata is published before replay.
        hook_runtime.metadata = metadata_and_bytes(0)
        hook(static_input.t())
        engine.flush_and_wait(30.0)
        hook_runtime.builder = ProducerPlanBuilder()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hook(static_input.t())
        plan = hook_runtime.builder.build()
        for step in range(1, 97):
            static_input.copy_((values + step).to(dtype))
            reservation = runtime.prepare_replay(plan, (metadata_and_bytes(step),))
            assert reservation in (StepReservation.RESERVED, StepReservation.FLUSHED)
            graph.replay()

        engine.flush_and_wait(30.0)
        assert handle.native_sink.flush_and_wait(30.0)
        handle.native_sink.rethrow_if_failed()
        snapshot = handle.native_sink.snapshot()
        assert snapshot["persisted_records"] == len(expected), snapshot
        assert snapshot["failures"] == 0, snapshot
    finally:
        engine.close()

    records = _staged_records(spool_root)
    assert len(records) == len(expected)
    assert dict(records) == expected


def test_the_capture_backend_selects_the_native_sink_by_config(tmp_path: Path):
    """The automatic entry point: storage_backend="capture" + capture_sink_config."""
    from dmi.api.v1 import (
        HookPointV1, HookSpecV1, MonitoringEngine, TransportSpec,
    )
    from dmi.config import MonitoringConfig
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.storage.capture.native_sink import NativeSinkConfig

    spool_root = tmp_path / "spool"
    config = MonitoringConfig(
        storage_backend="capture",
        capture_sink_config=NativeSinkConfig(
            spool_root=str(spool_root), max_pack_records=8,
            max_linger_ns=1_000_000_000),
    )
    engine = MonitoringEngine(config=config, model_id="native-sink-e2e",
                              ring_config=_ring_config())
    try:
        runtime = engine.create_record_runtime(CaptureRecordFormat())
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        tensor = torch.arange(16, dtype=torch.float32)
        hook_runtime.metadata = _metadata("capture-config", tensor, step=0)
        hook(tensor.cuda())
        engine.flush_and_wait(30.0)
    finally:
        engine.close()

    import time

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        staged = dict(_staged_records(spool_root))
        if staged:
            break
        time.sleep(0.2)
    assert staged == {"capture-config": tensor.numpy().tobytes()}
