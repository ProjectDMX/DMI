import pytest
import torch

from monitoring.graph_engine import GraphSafeEngine, GraphSlotResult, _decode_metadata_row
from monitoring.graph_monitor import GraphMonitor, METADATA_BYTES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_alias_tensor_matches_source():
    device = torch.device("cuda")
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    tensor = torch.arange(0, 16, device=device, dtype=torch.float32).view(4, 4)

    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._host_buffer.copy_(monitor._gpu_buffer, non_blocking=False)
    row = monitor.metadata_view()[0]
    decoded = _decode_metadata_row(row)
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
def test_delegate_submits_tasks(monkeypatch):
    device = torch.device("cuda")
    tensor = torch.randn(2, 3, device=device)
    model = torch.nn.Identity().to(device)
    monitor = GraphMonitor(model, max_slots=1)
    torch.ops.graphmonitor_ops.record(tensor, monitor._gpu_buffer, 0)
    monitor._host_buffer.copy_(monitor._gpu_buffer, non_blocking=False)
    row = monitor.metadata_view()[0]
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

        def submit_step(self, step_id, tasks, stream):
            self.calls.append((step_id, tasks, stream))

    engine = GraphSafeEngine(device=device)
    backend = DummyBackend()
    engine.attach_backend_delegate(backend)
    engine._step_streams[slot.step_id] = None  # type: ignore[attr-defined]
    engine._submit_to_delegate([slot])  # type: ignore[attr-defined]
    assert backend.calls, "delegate should have received a submit_step call"
    step_id, tasks, stream = backend.calls[0]
    assert step_id == slot.step_id
    assert stream is None
    assert len(tasks) == 1
    captured_tensor = tasks[0][0].tensor
    assert torch.allclose(captured_tensor.cpu(), tensor.cpu())
