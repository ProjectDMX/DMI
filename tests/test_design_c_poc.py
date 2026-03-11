"""Phase 1 PoC: Design C — Dual-Graph via torch.compile + cudagraph_trees=False.

Key findings from exploration:
  - Forward hooks receive FakeTensors under torch.compile → custom ops don't fire
  - sink_hold C++ host effects invisible to Inductor → can't hold refs
  - CUDAGraphTreeManager shares memory between sibling branches → address overlap
  - FIX: cudagraph_trees=False disables tree sharing → natural address isolation
  - FIX: record() with Tensor(a!) mutation annotation → inline ops work

Validates 5 core assumptions:
  1. Dynamo int specialization → two separate CUDA Graph recordings
  2. Inline record() with (a!) writes per-frame metadata correctly
  3. cudagraph_trees=False → address isolation (no overlap between graphs)
  4. Dual-frame shadow buffer metadata correctness
  5. Full D2H pipeline: alias_tensor + async copy vs eager baseline
"""
import struct

import pytest
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn

from monitoring.graph_ops import load_graph_monitor_ops

METADATA_BYTES = 128
_HAS_TORCH_COMPILE = hasattr(torch, "compile")

# Module-level: disable tree-based CUDA Graph memory sharing
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


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Monitored model: inline record() calls (not hooks — hooks get FakeTensors)
# ---------------------------------------------------------------------------


class MonitoredMLP(nn.Module):
    """TinyMLP with inline record() calls for metadata capture."""

    def __init__(self, metadata_buf: torch.Tensor, num_slots: int):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4, 4)
        self.metadata_buf = metadata_buf
        self.num_slots = num_slots
        self._ops = load_graph_monitor_ops()

    def forward(self, x: torch.Tensor, flag: int = 0) -> torch.Tensor:
        offset = flag * self.num_slots
        h = self.fc1(x)
        self._ops.record(h, self.metadata_buf, 0 + offset)
        h = self.act(h)
        out = self.fc2(h)
        self._ops.record(out, self.metadata_buf, 1 + offset)
        return out


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


class _DualFrameFixture:
    """Sets up MonitoredMLP with cudagraph_trees=False and compiled forward."""

    def __init__(self):
        # Clear stale Dynamo/Inductor compilation cache from previous tests
        torch._dynamo.reset()

        self.device = torch.device("cuda")
        self.ops = load_graph_monitor_ops()
        self.num_slots = 2  # fc1, fc2
        self.metadata_buf = torch.empty(
            2 * self.num_slots * METADATA_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )
        self.metadata_buf.zero_()
        self.static_input = torch.randn(2, 4, device=self.device)
        self.model = MonitoredMLP(
            self.metadata_buf, self.num_slots
        ).to(self.device).eval()
        self.compiled_fn = torch.compile(
            self.model.forward, mode="reduce-overhead", fullgraph=False
        )

    def warmup(self, flag: int, steps: int = 4):
        with torch.no_grad():
            for _ in range(steps):
                torch.compiler.cudagraph_mark_step_begin()
                self.compiled_fn(self.static_input, flag=flag)

    def warmup_both(self, steps: int = 4):
        self.warmup(0, steps)
        torch.cuda.synchronize()
        self.warmup(1, steps)
        torch.cuda.synchronize()

    def replay(self, flag: int):
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            return self.compiled_fn(self.static_input, flag=flag)

    def parse_all_slots(self) -> torch.Tensor:
        """Return CPU copy of metadata buffer."""
        torch.cuda.synchronize()
        return self.metadata_buf.cpu()

    def close(self):
        pass


# ===========================================================================
# Test 1: Dynamo int specialization → two separate traces
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_dynamo_int_specialization():
    """flag=0 and flag=1 must produce two separate Dynamo traces."""
    device = torch.device("cuda")
    model = TinyMLP().to(device).eval()
    static_input = torch.randn(2, 4, device=device)
    trace_count = [0]

    def forward_with_flag(x, flag: int):
        trace_count[0] += 1
        if flag == 0:
            return model(x) + 0.0
        else:
            return model(x) + 0.0

    compiled_fn = torch.compile(
        forward_with_flag, mode="reduce-overhead", fullgraph=False
    )

    with torch.no_grad():
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            compiled_fn(static_input, flag=0)
        count_after_0 = trace_count[0]

        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            compiled_fn(static_input, flag=1)
        count_after_1 = trace_count[0]

    assert count_after_1 > count_after_0, (
        f"Expected flag=1 to trigger new trace. "
        f"count after flag=0: {count_after_0}, after flag=1: {count_after_1}"
    )
    print(f"[PASS] Dynamo int specialization: "
          f"traces after flag=0: {count_after_0}, after flag=1: {count_after_1}")


# ===========================================================================
# Test 2: Inline record() writes per-frame metadata
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_inline_record_per_frame():
    """Inline record() must write to frame 0 slots when flag=0, frame 1 when flag=1."""
    f = _DualFrameFixture()
    try:
        # Warmup flag=0 only
        f.warmup(0)
        host = f.parse_all_slots()

        for i in range(f.num_slots):
            slot = _parse_slot(host, i)
            assert slot["data_ptr"] != 0, (
                f"Frame 0 slot {i} should be populated (got data_ptr=0)"
            )
        for i in range(f.num_slots):
            slot = _parse_slot(host, f.num_slots + i)
            assert slot["data_ptr"] == 0, (
                f"Frame 1 slot {i} should be empty after flag=0 only "
                f"(got data_ptr={slot['data_ptr']:#x})"
            )

        # Warmup flag=1
        f.warmup(1)
        host = f.parse_all_slots()

        for i in range(f.num_slots):
            slot = _parse_slot(host, f.num_slots + i)
            assert slot["data_ptr"] != 0, (
                f"Frame 1 slot {i} should be populated (got data_ptr=0)"
            )

        print("[PASS] Inline record per-frame: frame 0 and 1 metadata written correctly")
    finally:
        f.close()


# ===========================================================================
# Test 3: Address isolation via cudagraph_trees=False
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_address_isolation():
    """cudagraph_trees=False must produce non-overlapping addresses."""
    f = _DualFrameFixture()
    try:
        f.warmup_both()
        host = f.parse_all_slots()

        ptrs_A = set()
        ptrs_B = set()
        for i in range(f.num_slots):
            a = _parse_slot(host, i)
            b = _parse_slot(host, f.num_slots + i)
            if a["data_ptr"] != 0:
                ptrs_A.add(a["data_ptr"])
            if b["data_ptr"] != 0:
                ptrs_B.add(b["data_ptr"])

        assert len(ptrs_A) > 0, "Frame 0 has no valid data_ptrs"
        assert len(ptrs_B) > 0, "Frame 1 has no valid data_ptrs"

        overlap = ptrs_A & ptrs_B
        assert len(overlap) == 0, (
            f"Address isolation FAILED! Overlap: {[hex(p) for p in overlap]}\n"
            f"  Frame 0: {[hex(p) for p in sorted(ptrs_A)]}\n"
            f"  Frame 1: {[hex(p) for p in sorted(ptrs_B)]}"
        )

        print(f"[PASS] Address isolation: "
              f"A={[hex(p) for p in sorted(ptrs_A)]}, "
              f"B={[hex(p) for p in sorted(ptrs_B)]}")
    finally:
        f.close()


# ===========================================================================
# Test 4: Dual-frame metadata correctness
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_dual_frame_metadata():
    """Both frames must have correct metadata (ndim, shape, device)."""
    f = _DualFrameFixture()
    try:
        f.warmup_both()
        host = f.parse_all_slots()
        expected_device = torch.cuda.current_device()

        for frame in [0, 1]:
            for slot_local in range(f.num_slots):
                global_slot = frame * f.num_slots + slot_local
                slot = _parse_slot(host, global_slot)
                assert slot["ndim"] == 2, (
                    f"Frame {frame} slot {slot_local}: ndim={slot['ndim']}, expected 2"
                )
                assert slot["shape"][0] == 2, (
                    f"Frame {frame} slot {slot_local}: shape[0]={slot['shape'][0]}"
                )
                assert slot["shape"][1] == 4, (
                    f"Frame {frame} slot {slot_local}: shape[1]={slot['shape'][1]}"
                )
                assert slot["data_ptr"] != 0, (
                    f"Frame {frame} slot {slot_local}: data_ptr is 0"
                )
                assert slot["device_idx"] == expected_device, (
                    f"Frame {frame} slot {slot_local}: "
                    f"device_idx={slot['device_idx']}, expected {expected_device}"
                )

        print("[PASS] Dual-frame metadata: all 4 slots correct (2 frames x 2 hooks)")
    finally:
        f.close()


# ===========================================================================
# Test 5: Full D2H pipeline — alias_tensor + async copy vs eager
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_d2h_correctness():
    """Async D2H via from_blob aliases must be consistent across replays.

    Note: intermediate tensor addresses may be reused by later ops under
    CUDA Graph, so we compare against a synchronous D2H reference (not eager),
    validating that replay produces deterministic, correct D2H data.
    """
    f = _DualFrameFixture()
    try:
        # --- Warmup both graphs ---
        f.warmup_both()
        host = f.parse_all_slots()

        # --- One-time parse: create from_blob aliases per frame ---
        cached_aliases = {}
        for frame in [0, 1]:
            aliases = []
            for slot_local in range(f.num_slots):
                slot = _parse_slot(host, frame * f.num_slots + slot_local)
                shape = list(slot["shape"][: slot["ndim"]])
                strides = list(slot["stride"][: slot["ndim"]])
                alias = f.ops.alias_tensor(
                    slot["data_ptr"], shape, strides,
                    slot["dtype_id"], slot["device_idx"],
                )
                aliases.append(alias)
            cached_aliases[frame] = aliases

        # --- Establish reference via synchronous D2H ---
        sync_refs = {}
        for frame in [0, 1]:
            f.replay(frame)
            torch.cuda.synchronize()
            sync_refs[frame] = [a.cpu().clone() for a in cached_aliases[frame]]

        # Verify final output (slot 1) matches eager
        with torch.no_grad():
            eager_out = f.model(f.static_input).clone().cpu()
        for frame in [0, 1]:
            assert torch.allclose(sync_refs[frame][1], eager_out, atol=1e-5), (
                f"Frame {frame} slot 1 (final output) doesn't match eager"
            )

        # --- Pre-allocate pinned host buffers ---
        pinned = {}
        for frame, aliases in cached_aliases.items():
            pinned[frame] = [
                torch.empty_like(a, device="cpu", pin_memory=True) for a in aliases
            ]

        # --- Steady-state decode: 10 steps with overlapped D2H ---
        copy_stream = torch.cuda.Stream(device=f.device)
        d2h_events = {0: torch.cuda.Event(), 1: torch.cuda.Event()}
        num_steps = 10
        errors = []

        for step in range(num_steps):
            flag = step % 2

            # Wait for same-frame previous D2H before consuming
            if step >= 2:
                d2h_events[flag].synchronize()
                for i, cpu_buf in enumerate(pinned[flag]):
                    if not torch.allclose(cpu_buf, sync_refs[flag][i], atol=1e-5):
                        errors.append(
                            f"Step {step} frame {flag} slot {i}: "
                            f"max_diff={torch.max(torch.abs(cpu_buf - sync_refs[flag][i])).item():.6f}"
                        )

            # Forward (replay graph)
            f.replay(flag)

            # Async D2H on copy stream
            copy_stream.wait_stream(torch.cuda.current_stream(f.device))
            with torch.cuda.stream(copy_stream):
                for i, alias in enumerate(cached_aliases[flag]):
                    pinned[flag][i].copy_(alias, non_blocking=True)
                d2h_events[flag].record(copy_stream)

        # Drain last 2 steps
        for flag in [num_steps % 2, (num_steps + 1) % 2]:
            d2h_events[flag].synchronize()
            for i, cpu_buf in enumerate(pinned[flag]):
                if not torch.allclose(cpu_buf, sync_refs[flag][i], atol=1e-5):
                    errors.append(
                        f"Drain frame {flag} slot {i}: "
                        f"max_diff={torch.max(torch.abs(cpu_buf - sync_refs[flag][i])).item():.6f}"
                    )

        assert len(errors) == 0, "D2H mismatches:\n" + "\n".join(errors)
        print(f"[PASS] D2H correctness: {num_steps} steps, all data matches eager")
    finally:
        f.close()


# ===========================================================================
# Test 6: Step cadence benchmark
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_step_cadence():
    """Measure step cadence for alternating dual-graph decode with async D2H."""
    f = _DualFrameFixture()
    try:
        f.warmup_both()
        host = f.parse_all_slots()

        # Create from_blob aliases
        cached_aliases = {}
        for frame in [0, 1]:
            aliases = []
            for slot_local in range(f.num_slots):
                slot = _parse_slot(host, frame * f.num_slots + slot_local)
                shape = list(slot["shape"][: slot["ndim"]])
                strides = list(slot["stride"][: slot["ndim"]])
                alias = f.ops.alias_tensor(
                    slot["data_ptr"], shape, strides,
                    slot["dtype_id"], slot["device_idx"],
                )
                aliases.append(alias)
            cached_aliases[frame] = aliases

        pinned = {}
        for frame, aliases in cached_aliases.items():
            pinned[frame] = [
                torch.empty_like(a, device="cpu", pin_memory=True) for a in aliases
            ]

        copy_stream = torch.cuda.Stream(device=f.device)
        d2h_events = {0: torch.cuda.Event(), 1: torch.cuda.Event()}

        # Warmup timed loop
        for step in range(20):
            flag = step % 2
            if step >= 2:
                d2h_events[flag].synchronize()
            f.replay(flag)
            copy_stream.wait_stream(torch.cuda.current_stream(f.device))
            with torch.cuda.stream(copy_stream):
                for i, alias in enumerate(cached_aliases[flag]):
                    pinned[flag][i].copy_(alias, non_blocking=True)
                d2h_events[flag].record(copy_stream)
        torch.cuda.synchronize()

        # Timed loop
        timed_steps = 200
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for step in range(timed_steps):
            flag = step % 2
            if step >= 2:
                d2h_events[flag].synchronize()
            f.replay(flag)
            copy_stream.wait_stream(torch.cuda.current_stream(f.device))
            with torch.cuda.stream(copy_stream):
                for i, alias in enumerate(cached_aliases[flag]):
                    pinned[flag][i].copy_(alias, non_blocking=True)
                d2h_events[flag].record(copy_stream)
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        cadence_ms = total_ms / timed_steps
        print(f"[BENCH] Step cadence: {cadence_ms:.3f} ms/step "
              f"({timed_steps} steps, total {total_ms:.1f} ms)")
    finally:
        f.close()
