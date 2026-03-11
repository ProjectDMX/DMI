"""Experiment: hook-based dual_compile (no inline _mon_record).

Validates that forward hooks on HookPoint modules work correctly with
torch.compile + cudagraph_trees=False for dual-frame address isolation.

Key questions:
1. Does Dynamo trace hooks and include record() in the FX graph?
2. Does Dynamo create guards on _mon_frame_offset for dual-frame specialization?
3. Does never_reuse_buffers prevent tensor data corruption?
4. Does _mon_buf=None cause Dynamo to re-trace without record ops?
"""
import struct

import pytest
import torch
import torch.nn as nn
import torch._dynamo
import torch._inductor.config as inductor_config

from monitoring.graph_ops import load_graph_monitor_ops

inductor_config.triton.cudagraph_trees = False

METADATA_BYTES = 128
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")


# ---------------------------------------------------------------------------
# Minimal model with HookPoint calls in forward (unlike production code)
# ---------------------------------------------------------------------------


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.monitor_activation = True

    def forward(self, x):
        return x


class TinyModel(nn.Module):
    """Two linear layers with HookPoint calls in forward."""

    def __init__(self, d=32):
        super().__init__()
        self.linear1 = nn.Linear(d, d, bias=False)
        self.hook_after_l1 = HookPoint()
        self.linear2 = nn.Linear(d, d, bias=False)
        self.hook_after_l2 = HookPoint()

    def forward(self, x):
        x = self.linear1(x)
        x = self.hook_after_l1(x)  # HookPoint CALLED in forward
        x = self.linear2(x)
        x = self.hook_after_l2(x)  # HookPoint CALLED in forward
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_slot(buf_cpu: torch.Tensor, slot_id: int):
    start = slot_id * METADATA_BYTES
    end = start + METADATA_BYTES
    if end > buf_cpu.numel():
        return None
    slot_bytes = buf_cpu[start:end].contiguous().numpy().tobytes()
    fields = _METADATA_STRUCT.unpack(slot_bytes)
    data_ptr = fields[0]
    ndim = fields[9]
    if data_ptr == 0 or ndim <= 0:
        return None
    return {
        "data_ptr": data_ptr,
        "shape": list(fields[1:5][:ndim]),
        "stride": list(fields[5:9][:ndim]),
        "ndim": ndim,
        "dtype_id": fields[10],
        "device_idx": fields[11],
    }


def _register_hooks(model, gpu_buffer, ops):
    """Register monitoring hooks on HookPoint modules.

    Sets _mon_buf and _mon_frame_offset on each HookPoint.
    Returns (hookpoints, slot_map, num_monitored).
    """
    hookpoints = []
    slot_map = {}
    slot_id = 0
    for name, module in model.named_modules():
        if not hasattr(module, "monitor_activation"):
            continue
        module._mon_buf = gpu_buffer
        module._mon_frame_offset = 0
        hookpoints.append(module)

        sid = slot_id

        def make_hook(s):
            def hook(mod, inp, out):
                buf = mod._mon_buf
                if buf is None:
                    return
                offset = mod._mon_frame_offset
                ops.record(out, buf, s + offset)
            return hook

        module.register_forward_hook(make_hook(sid))
        slot_map[sid] = name
        slot_id += 1
    return hookpoints, slot_map, slot_id


def _set_frame(hookpoints, frame, num_monitored):
    offset = frame * num_monitored
    for hp in hookpoints:
        hp._mon_frame_offset = offset


@pytest.fixture(autouse=True)
def _reset():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ===========================================================================
# Test 1: Dynamo traces hooks → record() in compiled graph
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hook_traced_by_dynamo():
    """record() called inside hook should produce valid metadata in GPU buffer."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()

    model = TinyModel(d=32).to(device).eval()
    max_slots = 16
    gpu_buffer = torch.empty(max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    gpu_buffer.zero_()

    hookpoints, slot_map, num_monitored = _register_hooks(model, gpu_buffer, ops)

    compiled_fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(1, 32, device=device)

    # Warmup
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Check metadata
    host = gpu_buffer.cpu()
    for sid in range(num_monitored):
        meta = _parse_slot(host, sid)
        assert meta is not None, f"Slot {sid} ({slot_map[sid]}): no metadata (hook not traced?)"
        print(f"  Slot {sid} ({slot_map[sid]}): ptr={hex(meta['data_ptr'])}, "
              f"shape={meta['shape'][:meta['ndim']]}")

    print(f"[PASS] Dynamo traced hooks: {num_monitored} slots have valid metadata")


# ===========================================================================
# Test 2: Dual-frame address isolation via _mon_frame_offset guard
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hook_dual_frame_isolation():
    """_mon_frame_offset guard should produce separate CUDA graphs with different addresses."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()

    model = TinyModel(d=32).to(device).eval()
    max_slots = 16
    gpu_buffer = torch.empty(2 * max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    gpu_buffer.zero_()

    hookpoints, slot_map, num_monitored = _register_hooks(model, gpu_buffer, ops)

    compiled_fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(1, 32, device=device)

    # Warmup frame 0
    _set_frame(hookpoints, 0, num_monitored)
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Warmup frame 1
    _set_frame(hookpoints, 1, num_monitored)
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Parse both frames
    host = gpu_buffer.cpu()
    ptrs_0, ptrs_1 = set(), set()

    for sid in range(num_monitored):
        meta = _parse_slot(host, sid)
        if meta:
            ptrs_0.add(meta["data_ptr"])
            print(f"  Frame 0, slot {sid} ({slot_map[sid]}): ptr={hex(meta['data_ptr'])}")

    for sid in range(num_monitored):
        meta = _parse_slot(host, sid + num_monitored)
        if meta:
            ptrs_1.add(meta["data_ptr"])
            print(f"  Frame 1, slot {sid} ({slot_map[sid]}): ptr={hex(meta['data_ptr'])}")

    assert len(ptrs_0) == num_monitored, f"Frame 0: expected {num_monitored} ptrs, got {len(ptrs_0)}"
    assert len(ptrs_1) == num_monitored, f"Frame 1: expected {num_monitored} ptrs, got {len(ptrs_1)}"

    overlap = ptrs_0 & ptrs_1
    assert len(overlap) == 0, (
        f"Address isolation FAILED! {len(overlap)} overlapping: "
        f"{[hex(p) for p in sorted(overlap)]}"
    )

    print(f"[PASS] Dual-frame isolation: {len(ptrs_0)}+{len(ptrs_1)} ptrs, 0 overlap")


# ===========================================================================
# Test 3: Tensor data correctness (never_reuse_buffers)
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hook_tensor_correctness():
    """Alias tensors created from hook-recorded metadata must contain correct data."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()

    model = TinyModel(d=32).to(device).eval()
    max_slots = 16
    gpu_buffer = torch.empty(max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    gpu_buffer.zero_()

    hookpoints, slot_map, num_monitored = _register_hooks(model, gpu_buffer, ops)

    compiled_fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(1, 32, device=device)

    # Warmup
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Create aliases from metadata
    host = gpu_buffer.cpu()
    aliases = {}
    for sid in range(num_monitored):
        meta = _parse_slot(host, sid)
        if meta:
            aliases[sid] = ops.alias_tensor(
                meta["data_ptr"],
                meta["shape"][:meta["ndim"]],
                meta["stride"][:meta["ndim"]],
                meta["dtype_id"],
                meta["device_idx"],
            )

    # Run with specific input
    test_input = torch.randn(1, 32, device=device)
    static_input.copy_(test_input)
    torch.compiler.cudagraph_mark_step_begin()
    with torch.no_grad():
        compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Compute reference (eager)
    with torch.no_grad():
        ref_l1 = model.linear1(test_input)
        ref_l2 = model.linear2(ref_l1)

    # Check each alias
    errors = []
    refs = {0: ref_l1, 1: ref_l2}
    for sid, alias in aliases.items():
        alias_cpu = alias.cpu()
        ref_cpu = refs[sid].cpu()
        if torch.allclose(alias_cpu, ref_cpu, atol=1e-5):
            print(f"  Slot {sid} ({slot_map[sid]}): PASS")
        else:
            diff = torch.max(torch.abs(alias_cpu - ref_cpu)).item()
            errors.append(f"slot {sid} ({slot_map[sid]}): max_diff={diff:.6f}")
            print(f"  Slot {sid} ({slot_map[sid]}): FAIL (max_diff={diff:.6f})")

    assert len(errors) == 0, "Tensor correctness FAILED:\n" + "\n".join(errors)
    print(f"[PASS] Tensor correctness: {len(aliases)} aliases verified")


# ===========================================================================
# Test 4: disable_record (_mon_buf=None → Dynamo re-trace without record)
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hook_disable_record():
    """Setting _mon_buf=None should cause Dynamo to re-trace without record ops."""
    device = torch.device("cuda")
    ops = load_graph_monitor_ops()

    model = TinyModel(d=32).to(device).eval()
    max_slots = 16
    gpu_buffer = torch.empty(max_slots * METADATA_BYTES, dtype=torch.uint8, device=device)
    gpu_buffer.zero_()

    hookpoints, slot_map, num_monitored = _register_hooks(model, gpu_buffer, ops)

    compiled_fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    static_input = torch.randn(1, 32, device=device)

    # Warmup WITH record
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Verify record happened
    host = gpu_buffer.cpu()
    has_data = any(_parse_slot(host, sid) is not None for sid in range(num_monitored))
    assert has_data, "Expected metadata after warmup with record"
    print("  Phase 1 (record enabled): metadata present")

    # Disable record
    for hp in hookpoints:
        hp._mon_buf = None
    gpu_buffer.zero_()

    # Re-warmup WITHOUT record
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_fwd(static_input)
    torch.cuda.synchronize()

    # Buffer should be empty
    host = gpu_buffer.cpu()
    for sid in range(num_monitored):
        meta = _parse_slot(host, sid)
        assert meta is None, (
            f"Slot {sid} has data after disable_record: ptr={hex(meta['data_ptr'])}"
        )

    print("  Phase 2 (record disabled): buffer empty")
    print(f"[PASS] disable_record: Dynamo re-traced without record ops")
