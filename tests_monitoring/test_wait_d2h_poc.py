"""PoC: Validate cudaEventWaitExternal inside torch.compile CUDA Graph replay.

Core question: can a custom op call cudaStreamWaitEvent(stream, external_event,
cudaEventWaitExternal) during CUDA Graph capture, and have the event-wait node
correctly check the event's current state on every replay?

Tests:
  1. wait_d2h compiles into CUDA Graph without error
  2. Already-complete events pass through instantly (no hang)
  3. Pending events actually block the forward stream
  4. Full dual-frame D2H pipeline with per-slot barriers
"""
import struct
import time

import pytest
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn

from monitoring.graph_ops import load_graph_monitor_ops

METADATA_BYTES = 128
_HAS_TORCH_COMPILE = hasattr(torch, "compile")

# Disable tree-based CUDA Graph memory sharing for address isolation
inductor_config.triton.cudagraph_trees = False


def _parse_slot(metadata: torch.Tensor, slot_id: int):
    """Parse raw 128B metadata row into dict."""
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


class BarrierMLP(nn.Module):
    """TinyMLP with inline wait_d2h + record calls in forward."""

    def __init__(self, buf, num_slots, ops):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4, 4)
        self.buf = buf
        self.num_slots = num_slots
        self._ops = ops

    def forward(self, x: torch.Tensor, flag: int = 0) -> torch.Tensor:
        offset = flag * self.num_slots
        h = self.fc1(x)
        self._ops.wait_d2h(self.buf, 0 + offset)
        self._ops.record(h, self.buf, 0 + offset)
        h = self.act(h)
        out = self.fc2(h)
        self._ops.wait_d2h(self.buf, 1 + offset)
        self._ops.record(out, self.buf, 1 + offset)
        return out


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Reset Dynamo state between tests to avoid stale compilation caches."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ===========================================================================
# Test 1: wait_d2h compiles and captures into CUDA Graph
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_wait_d2h_compile_and_capture():
    """wait_d2h must be traced by Dynamo and captured into CUDA Graph."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()
    num_slots = 2
    buf = torch.zeros(
        2 * num_slots * METADATA_BYTES, dtype=torch.uint8, device=device
    )
    ops.init_d2h_events(2 * num_slots)

    model = BarrierMLP(buf, num_slots, ops).to(device).eval()
    compiled_fn = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False
    )
    static_input = torch.randn(2, 4, device=device)

    try:
        # Warmup flag=0 (Dynamo trace → Inductor compile → CUDA Graph capture)
        with torch.no_grad():
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled_fn(static_input, flag=0)
        torch.cuda.synchronize()

        assert out is not None and out.shape == (2, 4)
        assert not torch.isnan(out).any()

        # Verify metadata was written (record() worked after wait_d2h())
        host = buf.cpu()
        for i in range(num_slots):
            slot = _parse_slot(host, i)
            assert slot["data_ptr"] != 0, (
                f"Slot {i} data_ptr=0: record() after wait_d2h() did not fire"
            )

        # Warmup flag=1 (second CUDA Graph)
        with torch.no_grad():
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled_fn(static_input, flag=1)
        torch.cuda.synchronize()

        host = buf.cpu()
        for i in range(num_slots):
            slot = _parse_slot(host, num_slots + i)
            assert slot["data_ptr"] != 0, (
                f"Frame 1 slot {i} data_ptr=0"
            )

        print("[PASS] wait_d2h compiles and captures into CUDA Graph")
    finally:
        ops.destroy_d2h_events()


# ===========================================================================
# Test 2: Already-complete events → no hang
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_wait_d2h_no_hang():
    """Events never recorded (complete state) must not block replay."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()
    num_slots = 2
    buf = torch.zeros(
        2 * num_slots * METADATA_BYTES, dtype=torch.uint8, device=device
    )
    ops.init_d2h_events(2 * num_slots)

    model = BarrierMLP(buf, num_slots, ops).to(device).eval()
    compiled_fn = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False
    )
    static_input = torch.randn(2, 4, device=device)

    try:
        with torch.no_grad():
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=0)
        torch.cuda.synchronize()

        # 100 replays, events never recorded → should all pass instantly
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"Suspected hang: 100 replays took {elapsed:.2f}s"
        )
        print(f"[PASS] 100 replays (complete events): {elapsed*1000:.1f}ms")
    finally:
        ops.destroy_d2h_events()


# ===========================================================================
# Test 3: Pending event blocks forward stream
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_wait_d2h_blocks_on_pending():
    """wait_d2h must stall fwd_stream when event is not yet signaled."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()
    num_slots = 2
    buf = torch.zeros(
        2 * num_slots * METADATA_BYTES, dtype=torch.uint8, device=device
    )
    ops.init_d2h_events(2 * num_slots)

    model = BarrierMLP(buf, num_slots, ops).to(device).eval()
    compiled_fn = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False
    )
    static_input = torch.randn(2, 4, device=device)

    try:
        # Warmup
        with torch.no_grad():
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=0)
        torch.cuda.synchronize()

        # Baseline: 10 replays, events in complete state
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        with torch.no_grad():
            for _ in range(10):
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=0)
        t_end.record()
        torch.cuda.synchronize()
        baseline_ms = t_start.elapsed_time(t_end)

        # Blocked: enqueue slow D2H on copy_stream, record events AFTER
        copy_stream = torch.cuda.Stream(device=device)
        big_gpu = torch.randn(4096, 4096, device=device)
        big_cpu = torch.empty(4096, 4096, pin_memory=True)

        with torch.cuda.stream(copy_stream):
            for _ in range(10):
                big_cpu.copy_(big_gpu, non_blocking=True)
            # Record D2H events for slots 0,1 AFTER the slow copies
            ops.record_d2h_event(0)
            ops.record_d2h_event(1)

        t_start2 = torch.cuda.Event(enable_timing=True)
        t_end2 = torch.cuda.Event(enable_timing=True)
        t_start2.record()
        with torch.no_grad():
            # First replay must wait for copy_stream events
            torch.compiler.cudagraph_mark_step_begin()
            compiled_fn(static_input, flag=0)
        t_end2.record()
        torch.cuda.synchronize()
        blocked_ms = t_start2.elapsed_time(t_end2)

        per_replay_baseline = baseline_ms / 10
        print(f"  Baseline per-replay: {per_replay_baseline:.3f}ms, "
              f"Blocked single replay: {blocked_ms:.3f}ms")

        # The blocked replay should be noticeably slower
        assert blocked_ms > per_replay_baseline * 2, (
            f"wait_d2h did not block: baseline/replay={per_replay_baseline:.3f}ms, "
            f"blocked={blocked_ms:.3f}ms"
        )
        print(f"[PASS] wait_d2h blocks: {blocked_ms/per_replay_baseline:.1f}x "
              f"slower with pending events")
    finally:
        ops.destroy_d2h_events()


# ===========================================================================
# Test 4: Full dual-frame D2H pipeline with per-slot barriers
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_per_slot_d2h_pipeline():
    """Dual-frame pipeline: per-slot wait_d2h prevents data race during D2H."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()
    num_slots = 2
    buf = torch.zeros(
        2 * num_slots * METADATA_BYTES, dtype=torch.uint8, device=device
    )
    ops.init_d2h_events(2 * num_slots)

    model = BarrierMLP(buf, num_slots, ops).to(device).eval()
    compiled_fn = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False
    )
    static_input = torch.randn(2, 4, device=device)

    try:
        # Warmup both frames
        for flag in (0, 1):
            with torch.no_grad():
                for _ in range(4):
                    torch.compiler.cudagraph_mark_step_begin()
                    compiled_fn(static_input, flag=flag)
            torch.cuda.synchronize()

        # Parse metadata, create aliases
        host = buf.cpu()
        aliases = {}
        for frame in (0, 1):
            frame_aliases = {}
            for slot_local in range(num_slots):
                global_slot = frame * num_slots + slot_local
                slot = _parse_slot(host, global_slot)
                if slot["data_ptr"] == 0 or slot["ndim"] <= 0:
                    continue
                shape = list(slot["shape"][: slot["ndim"]])
                strides = list(slot["stride"][: slot["ndim"]])
                alias = ops.alias_tensor(
                    slot["data_ptr"], shape, strides,
                    slot["dtype_id"], slot["device_idx"],
                )
                frame_aliases[slot_local] = alias
            aliases[frame] = frame_aliases

        assert len(aliases[0]) > 0, "Frame 0 has no valid aliases"
        assert len(aliases[1]) > 0, "Frame 1 has no valid aliases"

        # Verify address isolation
        ptrs_0 = {a.data_ptr() for a in aliases[0].values()}
        ptrs_1 = {a.data_ptr() for a in aliases[1].values()}
        assert len(ptrs_0 & ptrs_1) == 0, "Frames share addresses!"

        # Pre-allocate pinned buffers
        pinned = {}
        for frame, fa in aliases.items():
            pinned[frame] = {
                sid: torch.empty_like(a, device="cpu", pin_memory=True)
                for sid, a in fa.items()
            }

        # Establish sync D2H reference
        ref = {}
        for frame in (0, 1):
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=frame)
            torch.cuda.synchronize()
            ref[frame] = {
                sid: a.cpu().clone() for sid, a in aliases[frame].items()
            }

        # Steady-state: 20 steps, alternating frames, per-slot D2H events
        copy_stream = torch.cuda.Stream(device=device)
        errors = []

        for step in range(20):
            flag = step % 2

            # Replay graph (wait_d2h barriers inside protect against D2H race)
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(static_input, flag=flag)

            # Async D2H on copy_stream with per-slot event records
            copy_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(copy_stream):
                for sid, alias in aliases[flag].items():
                    global_slot = flag * num_slots + sid
                    pinned[flag][sid].copy_(alias, non_blocking=True)
                    ops.record_d2h_event(global_slot)

            # Verify previous same-frame D2H
            if step >= 2:
                torch.cuda.synchronize()
                for sid, cpu_buf in pinned[flag].items():
                    if not torch.allclose(cpu_buf, ref[flag][sid], atol=1e-5):
                        diff = torch.max(
                            torch.abs(cpu_buf - ref[flag][sid])
                        ).item()
                        errors.append(
                            f"Step {step} frame {flag} slot {sid}: "
                            f"max_diff={diff:.6f}"
                        )

        assert len(errors) == 0, "D2H mismatches:\n" + "\n".join(errors)
        print(f"[PASS] Per-slot D2H pipeline: 20 steps, "
              f"{len(aliases[0])}+{len(aliases[1])} aliases, all correct")
    finally:
        ops.destroy_d2h_events()
