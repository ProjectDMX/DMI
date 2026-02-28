"""Phase 2 integration tests: Design C dual_compile mode with production classes.

Uses real HookedGPT2Model (tiny config) to verify GraphMonitor + GraphSafeEngine
dual_compile pipeline end-to-end.
"""
import struct

import pytest
import torch
import torch._dynamo
import torch._inductor.config as inductor_config

from transformers import StaticCache
from transformers.models.gpt2_p.configuration_gpt2 import GPT2Config
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

from monitoring.graph_monitor import GraphMonitor, METADATA_BYTES
from monitoring.graph_engine import GraphSafeEngine

_HAS_TORCH_COMPILE = hasattr(torch, "compile")
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")

# Disable tree-based CUDA Graph memory sharing for address isolation
inductor_config.triton.cudagraph_trees = False


def _tiny_gpt2_config():
    """Minimal GPT2 config for fast test execution."""
    return GPT2Config(
        vocab_size=256,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
        _attn_implementation="eager",
    )


def _parse_slot(metadata: torch.Tensor, slot_id: int):
    """Parse raw 128B metadata row into dict."""
    start = slot_id * METADATA_BYTES
    end = start + METADATA_BYTES
    if end > metadata.numel():
        return {"data_ptr": 0, "ndim": 0}
    slot_bytes = metadata[start:end].cpu().contiguous().numpy().tobytes()
    fields = _METADATA_STRUCT.unpack(slot_bytes)
    return {
        "data_ptr": fields[0],
        "shape": fields[1:5],
        "stride": fields[5:9],
        "ndim": fields[9],
        "dtype_id": fields[10],
        "device_idx": fields[11],
    }


def _module_filter(name, module):
    return hasattr(module, "monitor_activation")


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Reset Dynamo state between tests to avoid stale compilation caches."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ===========================================================================
# Test 1: dual_compile monitor setup
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dual_compile_monitor_setup():
    """GraphMonitor dual_compile: 2x buffer, _mon_frame_offset set, no hooks."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()
    model = HookedGPT2Model(config).to(device).eval()

    monitor = GraphMonitor(
        model,
        max_slots=256,
        module_filter=_module_filter,
        device=device,
        graph_mode="dual_compile",
    )
    try:
        slot_map = monitor.get_slot_mapping()
        num_slots = len(slot_map)
        assert num_slots > 0, "Expected at least one monitored slot"

        # Buffer should be 2x size
        expected_buf_size = 2 * 256 * METADATA_BYTES
        assert monitor._gpu_buffer.numel() == expected_buf_size, (
            f"Buffer size {monitor._gpu_buffer.numel()} != expected {expected_buf_size}"
        )

        # _num_monitored_slots should match
        assert monitor.num_monitored_slots == num_slots

        # No forward hooks registered
        assert len(monitor._handles) == 0, (
            f"Expected 0 hooks in dual_compile, got {len(monitor._handles)}"
        )

        # _mon_frame_offset should be set on parent modules
        assert len(monitor._frame_parents) > 0
        for parent in monitor._frame_parents:
            assert hasattr(parent, "_mon_frame_offset")
            assert parent._mon_frame_offset == 0

        # set_frame should update offsets
        monitor.set_frame(1)
        for parent in monitor._frame_parents:
            assert parent._mon_frame_offset == num_slots, (
                f"Expected offset {num_slots}, got {parent._mon_frame_offset}"
            )
        monitor.set_frame(0)
        for parent in monitor._frame_parents:
            assert parent._mon_frame_offset == 0

        print(f"[PASS] dual_compile monitor setup: {num_slots} slots, "
              f"{len(monitor._frame_parents)} parents, 0 hooks")
    finally:
        monitor.close()


# ===========================================================================
# Test 2: Address isolation between frames
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_dual_compile_address_isolation():
    """Warmup both frames → metadata addresses must not overlap."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()
    model = HookedGPT2Model(config).to(device).eval()

    monitor = GraphMonitor(
        model,
        max_slots=256,
        module_filter=_module_filter,
        device=device,
        graph_mode="dual_compile",
    )
    try:
        num_slots = monitor.num_monitored_slots
        static_input = torch.randint(0, 256, (1, 1), device=device)
        cache = StaticCache(config=config, max_cache_len=16)
        cache_pos = torch.tensor([0], device=device, dtype=torch.long)

        compiled_forward = torch.compile(
            model.forward, mode="reduce-overhead", fullgraph=False,
        )

        # Warmup frame 0
        monitor.set_frame(0)
        for _ in range(4):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
        torch.cuda.synchronize()

        # Warmup frame 1
        monitor.set_frame(1)
        for _ in range(4):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
        torch.cuda.synchronize()

        # Parse both frames
        meta_0 = monitor.parse_frame_metadata(0)
        meta_1 = monitor.parse_frame_metadata(1)

        assert len(meta_0) > 0, "Frame 0 has no valid metadata"
        assert len(meta_1) > 0, "Frame 1 has no valid metadata"

        ptrs_0 = {m["data_ptr"] for m in meta_0.values() if m["data_ptr"] != 0}
        ptrs_1 = {m["data_ptr"] for m in meta_1.values() if m["data_ptr"] != 0}

        overlap = ptrs_0 & ptrs_1
        assert len(overlap) == 0, (
            f"Address isolation FAILED! {len(overlap)} overlapping ptrs: "
            f"{[hex(p) for p in sorted(overlap)[:5]]}"
        )

        print(f"[PASS] Address isolation: frame0={len(ptrs_0)} ptrs, "
              f"frame1={len(ptrs_1)} ptrs, 0 overlap")
    finally:
        monitor.close()


# ===========================================================================
# Test 3: Full engine pipeline
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_dual_compile_engine_pipeline():
    """GraphSafeEngine dual_compile: warmup → finalize → start/end step loop → collect."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()
    model = HookedGPT2Model(config).to(device).eval()

    engine = GraphSafeEngine(
        module_filter=_module_filter,
        max_slots=256,
        device=device,
        graph_mode="dual_compile",
    )
    engine.prepare_for_model(model)

    static_input = torch.randint(0, 256, (1, 1), device=device)
    cache = StaticCache(config=config, max_cache_len=32)
    cache_pos = torch.tensor([0], device=device, dtype=torch.long)

    compiled_forward = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False,
    )

    try:
        # Warmup both frames
        for frame in (0, 1):
            engine.set_frame(frame)
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    compiled_forward(
                        static_input, use_cache=True, past_key_values=cache,
                        cache_position=cache_pos, return_dict=True,
                    )
            torch.cuda.synchronize()

        engine.finalize_dual_frame()
        assert engine._dual_frame_ready

        # Verify aliases created for both frames
        assert len(engine._frame_aliases[0]) > 0
        assert len(engine._frame_aliases[1]) > 0
        assert len(engine._pinned_buffers[0]) > 0
        assert len(engine._pinned_buffers[1]) > 0

        # Reset cache for decode loop
        cache.reset()

        # Run 10 decode steps
        for step in range(10):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        # Collect last frame results
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None, "Expected D2H results after 10 steps"
        assert len(results) > 0, "Expected at least one slot in results"

        # Verify tensors are on CPU and non-empty
        for slot_id, tensor in results.items():
            assert tensor.device == torch.device("cpu")
            assert tensor.numel() > 0

        print(f"[PASS] Engine pipeline: 10 steps, {len(results)} slots collected")
    finally:
        engine.close()


# ===========================================================================
# Test 4: D2H data correctness across replays
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_dual_compile_d2h_correctness():
    """D2H via aliases must produce consistent data across replays."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()
    model = HookedGPT2Model(config).to(device).eval()

    engine = GraphSafeEngine(
        module_filter=_module_filter,
        max_slots=256,
        device=device,
        graph_mode="dual_compile",
    )
    engine.prepare_for_model(model)

    static_input = torch.randint(0, 256, (1, 1), device=device)
    cache = StaticCache(config=config, max_cache_len=32)
    cache_pos = torch.tensor([0], device=device, dtype=torch.long)

    compiled_forward = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False,
    )

    try:
        # Warmup both frames
        for frame in (0, 1):
            engine.set_frame(frame)
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    compiled_forward(
                        static_input, use_cache=True, past_key_values=cache,
                        cache_position=cache_pos, return_dict=True,
                    )
            torch.cuda.synchronize()

        engine.finalize_dual_frame()
        cache.reset()

        # Collect reference: replay frame 0, sync D2H
        engine._monitor.set_frame(0)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_forward(
                static_input, use_cache=True, past_key_values=cache,
                cache_position=cache_pos, return_dict=True,
            )
        torch.cuda.synchronize()
        ref_0 = {sid: alias.cpu().clone() for sid, alias in engine._frame_aliases[0].items()}

        # Run 6 steps through engine pipeline
        cache.reset()
        collected = {}
        for step in range(6):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        # Collect frame 0 results (last even step)
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None

        # Check that frame 0 slot data matches reference
        errors = []
        for sid in ref_0:
            if sid not in results:
                continue
            if not torch.allclose(results[sid], ref_0[sid], atol=1e-4):
                diff = torch.max(torch.abs(results[sid] - ref_0[sid])).item()
                errors.append(f"slot {sid}: max_diff={diff:.6f}")

        assert len(errors) == 0, "D2H mismatches:\n" + "\n".join(errors)
        print(f"[PASS] D2H correctness: {len(ref_0)} slots verified")
    finally:
        engine.close()


# ===========================================================================
# Test 5: Backward compat — compile mode unchanged
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_backward_compat_compile_mode():
    """graph_mode='compile' behavior must be unchanged by dual_compile additions."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()
    model = HookedGPT2Model(config).to(device).eval()

    engine = GraphSafeEngine(
        module_filter=_module_filter,
        max_slots=256,
        device=device,
        graph_mode="compile",
    )
    engine.prepare_for_model(model)

    static_input = torch.randint(0, 256, (1, 1), device=device)
    cache = StaticCache(config=config, max_cache_len=16)
    cache_pos = torch.tensor([0], device=device, dtype=torch.long)

    compiled_forward = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            compiled_forward(
                static_input, use_cache=True, past_key_values=cache,
                cache_position=cache_pos, return_dict=True,
            )
    torch.cuda.synchronize()

    try:
        # Should not have dual_compile state
        assert not engine._dual_frame_ready
        assert len(engine._frame_aliases) == 0

        # Forward hooks should be registered (compile mode uses hooks)
        assert len(engine._monitor._handles) > 0

        # Standard collect_results should work
        for step in range(1, 3):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()
            results = engine.collect_results(wait=True)
            assert len(results) > 0, f"Step {step}: no results collected"

        print(f"[PASS] Backward compat: compile mode works with {len(results)} slots")
    finally:
        engine.close()
