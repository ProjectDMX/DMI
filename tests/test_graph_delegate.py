import pytest
import torch

from monitoring.graph_engine import GraphSafeEngine, GraphSlotResult, _decode_metadata_row
from monitoring.graph_monitor import GraphMonitor, METADATA_BYTES
from monitoring.monitor_native import parse_shadow_block, create_graph_delegate
from monitoring import _native_engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_alias_tensor_matches_source():
    device = torch.device("cuda")
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    tensor = torch.arange(0, 16, device=device, dtype=torch.float32).view(4, 4)

    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._latest_snapshot.copy_(monitor._gpu_buffer, non_blocking=False)
    metadata = monitor.metadata_view().clone()
    decoded = _decode_metadata_row(metadata[0])
    assert decoded is not None
    shape, stride, ndim, dtype_id, device_idx, data_ptr = decoded
    slot = GraphSlotResult(
        slot_id=0,
        name="identity",
        step_id=0,
        data_ptr=data_ptr,
        shape=shape,
        stride=stride,
        ndim=ndim,
        dtype_id=dtype_id,
        device_index=device_idx,
    )
    alias = torch.ops.graphmonitor_ops.alias_tensor(
        int(slot.data_ptr),
        list(slot.shape),
        list(slot.stride),
        int(slot.dtype_id),
        int(slot.device_index),
    )
    assert torch.allclose(alias.cpu(), tensor.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_parse_shadow_block_helper():
    device = torch.device("cuda")
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    tensor = torch.randn(2, 3, 5, device=device)

    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._latest_snapshot.copy_(monitor._gpu_buffer, non_blocking=False)
    metadata = monitor.metadata_view()
    spec = parse_shadow_block(metadata, [0], ["identity"])
    assert "tensors" in spec
    tensors = spec["tensors"]
    assert len(tensors) == 1
    parsed = tensors[0]
    assert torch.allclose(parsed.cpu(), tensor.cpu())
    assert spec["slice_dims"][0] == -2
    assert spec["slice_modes"][0] == 0
    assert spec["can_slice"][0] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_native_delegate_submit():
    device = torch.device("cuda")
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    tensor = torch.randn(2, 3, device=device)
    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._latest_snapshot.copy_(monitor._gpu_buffer, non_blocking=False)
    metadata = monitor.metadata_view()

    backend = _native_engine.create_engine(queue_size=8, cache_dtype=None, delay_steps=0)
    backend.begin_request(0)
    backend.begin_step(0, 2)  # StepPhase::kDecode

    delegate = create_graph_delegate(backend)
    try:
        delegate.submit_and_resolve(0, metadata, [0], ["identity"], None)
    finally:
        backend.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_delegate_submits_tasks(monkeypatch):
    device = torch.device("cuda")
    tensor = torch.randn(2, 3, device=device)
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._latest_snapshot.copy_(monitor._gpu_buffer, non_blocking=False)
    metadata = monitor.metadata_view()
    row = metadata[0]
    decoded = _decode_metadata_row(row)
    assert decoded is not None
    shape, stride, ndim, dtype_id, device_idx, data_ptr = decoded
    slot = GraphSlotResult(
        slot_id=0,
        name="identity",
        step_id=123,
        data_ptr=data_ptr,
        shape=shape,
        stride=stride,
        ndim=ndim,
        dtype_id=dtype_id,
        device_index=device_idx,
    )

    class DummyBackend:
        def __init__(self):
            self.calls = []

        def submit_and_resolve(self, step_id, metadata, slot_ids, hook_names, stream):
            self.calls.append((step_id, metadata, slot_ids, hook_names, stream))

    engine = GraphSafeEngine(device=device)
    backend = DummyBackend()
    engine.attach_backend_delegate(backend)
    engine._step_streams[slot.step_id] = None  # type: ignore[attr-defined]
    engine._metadata_snapshots[slot.step_id] = metadata  # type: ignore[attr-defined]
    engine._submit_to_delegate([slot])  # type: ignore[attr-defined]
    assert backend.calls, "delegate should have received a submit_and_resolve call"
    step_id, meta, slot_ids, hook_names, stream = backend.calls[0]
    assert step_id == slot.step_id
    assert stream is None
    spec = parse_shadow_block(meta, slot_ids, hook_names)
    assert len(spec["tensors"]) == 1
    captured_tensor = spec["tensors"][0]
    assert torch.allclose(captured_tensor.cpu(), tensor.cpu())
