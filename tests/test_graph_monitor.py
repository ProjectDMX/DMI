import struct

import pytest
import torch
import torch.nn as nn

from monitoring.graph_monitor import GraphMonitor, METADATA_BYTES
from monitoring.graph_engine import GraphSafeEngine

_HAS_TORCH_COMPILE = hasattr(torch, "compile")


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _parse_slot(metadata: torch.Tensor, slot_id: int):
    start = slot_id * METADATA_BYTES
    end = start + METADATA_BYTES
    slot_bytes = metadata[start:end].cpu().contiguous().numpy().tobytes()
    fields = struct.unpack("<Qqqqqqqqqiii44s", slot_bytes)
    return {
        "data_ptr": fields[0],
        "shape": fields[1:5],
        "stride": fields[5:9],
        "ndim": fields[9],
        "dtype_id": fields[10],
        "device_idx": fields[11],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_monitor_capture_and_replay_metadata():
    device = torch.device("cuda")
    model = TinyMLP().to(device).eval()
    monitor = GraphMonitor(
        model,
        max_slots=8,
        module_filter=lambda name, module: isinstance(module, nn.Linear),
        device=device,
    )
    slot_map = monitor.get_slot_mapping()
    assert len(slot_map) == 2
    assert slot_map[0].module_name.endswith("fc1")
    assert slot_map[1].module_name.endswith("fc2")

    static_input = torch.randn(2, 4, device=device)
    _ = model(static_input)  # warmup to allocate params
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        _ = model(static_input)
        monitor.finalize_capture()
    torch.cuda.current_stream().wait_stream(capture_stream)

    expected_device = torch.cuda.current_device()
    prev_ptr = None
    try:
        for step in range(1, 4):
            graph.replay()
            monitor.on_step_end(step, capture_stream)
            monitor.wait_for_step()

            slot0 = _parse_slot(monitor.metadata_buffer(), 0)
            slot1 = _parse_slot(monitor.metadata_buffer(), 1)

            assert slot0["ndim"] == 2
            assert slot0["shape"][0] == 2
            assert slot0["shape"][1] == 4
            assert slot0["stride"][0] == 4
            assert slot0["stride"][1] == 1
            assert slot0["device_idx"] == expected_device
            assert slot0["data_ptr"] != 0
            if prev_ptr is None:
                prev_ptr = slot0["data_ptr"]
            else:
                assert slot0["data_ptr"] == prev_ptr

            assert slot1["ndim"] == 2
            assert slot1["shape"][0] == 2
            assert slot1["shape"][1] == 4
            assert slot1["data_ptr"] != 0
    finally:
        monitor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_safe_engine_collect_results():
    device = torch.device("cuda")
    model = TinyMLP().to(device).eval()
    engine = GraphSafeEngine(
        module_filter=lambda name, module: isinstance(module, nn.Linear),
        max_slots=8,
        device=device,
    )
    engine.prepare_for_model(model)

    static_input = torch.randn(2, 4, device=device)
    capture_stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        _ = model(static_input)
        engine.finalize_capture()
    torch.cuda.current_stream().wait_stream(capture_stream)

    slot_map = engine.get_slot_mapping()
    try:
        for step in range(1, 3):
            engine.start_step()
            graph.replay()
            engine.end_step(stream=capture_stream)
            results = engine.collect_results(wait=True)
            assert len(results) == 2
            assert all(result.step_id == step for result in results)
            assert all(result.ndim == 2 for result in results)
            assert all(tuple(result.shape) == (2, 4) for result in results)
            names = sorted(result.name for result in results)
            expected = sorted(info.module_name for info in slot_map.values())
            assert names == expected
    finally:
        engine.close()


# ---- torch.compile (reduce-overhead) mode tests ----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile not available")
def test_graph_monitor_compile_mode_metadata():
    """Compile-mode hooks write correct metadata via torch.compile."""
    torch._dynamo.reset()
    device = torch.device("cuda")
    model = TinyMLP().to(device).eval()
    monitor = GraphMonitor(
        model,
        max_slots=8,
        module_filter=lambda name, module: isinstance(module, nn.Linear),
        device=device,
        graph_mode="compile",
    )
    slot_map = monitor.get_slot_mapping()
    assert len(slot_map) == 2

    compiled_forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(2, 4, device=device)

    # Warmup — triggers compilation and initial graph capture
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_forward(static_input)
    torch.cuda.synchronize()

    expected_device = torch.cuda.current_device()
    prev_ptr = None
    try:
        for step in range(1, 4):
            with torch.no_grad():
                _ = compiled_forward(static_input)
            monitor.on_step_end(step)
            monitor.wait_for_step()

            slot0 = _parse_slot(monitor.metadata_buffer(), 0)
            slot1 = _parse_slot(monitor.metadata_buffer(), 1)

            assert slot0["ndim"] == 2
            assert slot0["shape"][0] == 2
            assert slot0["shape"][1] == 4
            assert slot0["device_idx"] == expected_device
            assert slot0["data_ptr"] != 0
            if prev_ptr is None:
                prev_ptr = slot0["data_ptr"]
            else:
                assert slot0["data_ptr"] == prev_ptr

            assert slot1["ndim"] == 2
            assert slot1["shape"][0] == 2
            assert slot1["shape"][1] == 4
            assert slot1["data_ptr"] != 0
    finally:
        monitor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile not available")
def test_graph_safe_engine_compile_mode():
    """End-to-end: compile mode + GraphSafeEngine collect_results."""
    torch._dynamo.reset()
    device = torch.device("cuda")
    model = TinyMLP().to(device).eval()
    engine = GraphSafeEngine(
        module_filter=lambda name, module: isinstance(module, nn.Linear),
        max_slots=8,
        device=device,
        graph_mode="compile",
    )
    engine.prepare_for_model(model)

    compiled_forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(2, 4, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_forward(static_input)
    torch.cuda.synchronize()

    slot_map = engine.get_slot_mapping()
    try:
        for step in range(1, 3):
            engine.start_step()
            with torch.no_grad():
                _ = compiled_forward(static_input)
            engine.end_step()
            results = engine.collect_results(wait=True)
            assert len(results) == 2
            assert all(result.step_id == step for result in results)
            assert all(result.ndim == 2 for result in results)
            assert all(tuple(result.shape) == (2, 4) for result in results)
            names = sorted(result.name for result in results)
            expected = sorted(info.module_name for info in slot_map.values())
            assert names == expected
    finally:
        engine.close()
