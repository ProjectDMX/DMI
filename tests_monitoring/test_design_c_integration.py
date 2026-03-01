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


# ===========================================================================
# Test 6: Record elimination — disable_record after warmup
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_record_elimination():
    """After disable_record(), production graphs should have no record kernels
    but D2H via aliases must still produce correct data."""
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
        # Phase 1: warmup WITH record (metadata discovery)
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

        # Collect reference data WITH record
        cache.reset()
        engine._monitor.set_frame(0)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            compiled_forward(
                static_input, use_cache=True, past_key_values=cache,
                cache_position=cache_pos, return_dict=True,
            )
        torch.cuda.synchronize()
        ref_0 = {sid: alias.cpu().clone() for sid, alias in engine._frame_aliases[0].items()}

        # Phase 2: disable record, retrace production graphs
        engine.disable_record()

        # Verify _mon_buf cleared
        for parent in engine._monitor._frame_parents:
            assert parent._mon_buf is None, "_mon_buf should be None after disable_record"
            assert hasattr(parent, "_mon_frame_offset"), "_mon_frame_offset must be preserved"

        # Retrace both frames WITHOUT record
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

        # Run production decode loop (no record kernels in graph)
        cache.reset()
        for step in range(6):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        # Collect D2H results — aliases still valid because addresses are fixed
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None, "Expected D2H results after record elimination"
        assert len(results) > 0, "Expected at least one slot"

        # Verify data matches reference (same input → same activations)
        errors = []
        for sid in ref_0:
            if sid not in results:
                continue
            if not torch.allclose(results[sid], ref_0[sid], atol=1e-4):
                diff = torch.max(torch.abs(results[sid] - ref_0[sid])).item()
                errors.append(f"slot {sid}: max_diff={diff:.6f}")

        assert len(errors) == 0, "D2H mismatch after record elimination:\n" + "\n".join(errors)
        print(f"[PASS] Record elimination: {len(ref_0)} slots verified, D2H correct without record")
    finally:
        engine.close()


# ===========================================================================
# Test 7: Batched D2H SM kernel — correctness + selective mask
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="torch.compile required")
def test_batched_d2h_sm():
    """Batched D2H SM kernel produces correct data, and update_mask restricts copies."""
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
        assert engine._d2h_desc is not None, "BatchedD2HDescriptor should be created"

        # Phase 2: disable record, retrace
        engine.disable_record()
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

        # Run decode and collect D2H results (full mask)
        cache.reset()
        for step in range(6):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        full_results = engine.collect_dual_frame_results(wait=True)
        assert full_results is not None, "Expected D2H results"
        all_slots = set(full_results.keys())
        assert len(all_slots) > 2, f"Expected multiple slots, got {len(all_slots)}"
        print(f"  Full mask: {len(all_slots)} slots copied")

        # Verify data is non-zero (actual activations)
        for sid, tensor in full_results.items():
            assert tensor.abs().max().item() > 0, f"slot {sid} is all zeros"

        # Test selective mask: only copy first 2 slots
        subset = set(sorted(all_slots)[:2])
        engine.update_d2h_mask(subset)

        # Zero out pinned buffers to detect which slots were actually copied
        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        cache.reset()
        for step in range(6):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        masked_results = engine.collect_dual_frame_results(wait=True)
        assert masked_results is not None

        # Only subset slots should have data
        copied_slots = {sid for sid, t in masked_results.items() if t.abs().max().item() > 0}
        skipped_slots = all_slots - subset
        for sid in skipped_slots:
            if sid in masked_results:
                assert masked_results[sid].abs().max().item() == 0, \
                    f"Slot {sid} should NOT have been copied but has data"
        for sid in subset:
            if sid in masked_results:
                assert masked_results[sid].abs().max().item() > 0, \
                    f"Slot {sid} should have been copied but is zero"

        print(f"  Selective mask: subset={subset}, copied={copied_slots}, "
              f"skipped={skipped_slots}")

        # Restore full mask
        engine.update_d2h_mask(None)
        print(f"[PASS] Batched D2H SM: full + selective mask verified")
    finally:
        engine.close()


@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="requires torch.compile")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_select_hooks():
    """select_hooks() filters D2H by hook name patterns."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()  # n_layer=2
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

        # Phase 2: disable record, retrace
        engine.disable_record()
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

        # Print all hook names for debugging
        slot_mapping = engine.get_slot_mapping()
        all_names = {info.module_name for info in slot_mapping.values()}
        print(f"  All hooks ({len(all_names)}): {sorted(all_names)}")

        # --- Test 1: select by single pattern ---
        resid_post_slots = engine.select_hooks(["hook_resid_post"])
        resid_post_names = {slot_mapping[sid].module_name for sid in resid_post_slots}
        assert len(resid_post_slots) > 0, "hook_resid_post should match some hooks"
        assert all("hook_resid_post" in n for n in resid_post_names)
        # n_layer=2, so should be 2 hook_resid_post hooks
        assert len(resid_post_slots) == 2, \
            f"Expected 2 hook_resid_post (2 layers), got {len(resid_post_slots)}"
        print(f"  select_hooks(['hook_resid_post']): {len(resid_post_slots)} slots → {sorted(resid_post_names)}")

        # Verify D2H only copies selected slots
        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        cache.reset()
        for step in range(6):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None
        copied = {sid for sid, t in results.items() if t.abs().max().item() > 0}
        not_copied = set(results.keys()) - resid_post_slots
        for sid in not_copied:
            assert results[sid].abs().max().item() == 0, \
                f"Slot {sid} ({slot_mapping[sid].module_name}) should NOT be copied"
        for sid in resid_post_slots:
            if sid in results:
                assert results[sid].abs().max().item() > 0, \
                    f"Slot {sid} ({slot_mapping[sid].module_name}) should be copied"
        print(f"  D2H verification: copied={sorted(copied)}, expected={sorted(resid_post_slots)}")

        # --- Test 2: select by multiple patterns ---
        multi_slots = engine.select_hooks(["hook_resid_post", "hook_q"])
        multi_names = {slot_mapping[sid].module_name for sid in multi_slots}
        assert all(
            any(p in n for p in ["hook_resid_post", "hook_q"])
            for n in multi_names
        )
        print(f"  select_hooks(['hook_resid_post', 'hook_q']): {len(multi_slots)} slots")

        # --- Test 3: select None restores all ---
        all_slots = engine.select_hooks(None)
        assert all_slots == set(slot_mapping.keys()), \
            "select_hooks(None) should restore all slots"
        print(f"  select_hooks(None): restored {len(all_slots)} slots")

        # --- Test 4: no match returns empty ---
        empty = engine.select_hooks(["nonexistent_pattern_xyz"])
        assert len(empty) == 0, f"Expected 0 matches, got {len(empty)}"
        print(f"  select_hooks(['nonexistent']): {len(empty)} slots (expected 0)")

        print(f"[PASS] select_hooks: pattern filtering + D2H verified")
    finally:
        engine.close()


# ===========================================================================
# Test 9: per-request slicing
# ===========================================================================


def _coalesced_ranges(active_requests, stride_bytes):
    """Compute which request indices are covered by coalesced DMA ranges."""
    from monitoring.graph_engine import _coalesce_requests
    # Create a mock alias-like object to get coalesced ranges
    # We just need the sorted requests and stride info
    sorted_reqs = sorted(active_requests)
    threshold_bytes = 37500
    if not sorted_reqs:
        return set()
    covered = set()
    seg_start = sorted_reqs[0]
    seg_end = sorted_reqs[0]
    for r in sorted_reqs[1:]:
        gap_bytes = (r - seg_end - 1) * stride_bytes
        if gap_bytes <= threshold_bytes:
            seg_end = r
        else:
            covered.update(range(seg_start, seg_end + 1))
            seg_start = seg_end = r
    covered.update(range(seg_start, seg_end + 1))
    return covered


@pytest.mark.skipif(not _HAS_TORCH_COMPILE, reason="requires torch.compile")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_per_request_slicing():
    """Per-request D2H: only specified batch indices are copied."""
    device = torch.device("cuda")
    config = _tiny_gpt2_config()  # n_layer=2, n_embd=32
    BATCH_SIZE = 8
    model = HookedGPT2Model(config).to(device).eval()

    engine = GraphSafeEngine(
        module_filter=_module_filter,
        max_slots=256,
        device=device,
        graph_mode="dual_compile",
    )
    engine.prepare_for_model(model)

    static_input = torch.randint(0, 256, (BATCH_SIZE, 1), device=device)
    cache = StaticCache(config=config, max_batch_size=BATCH_SIZE, max_cache_len=32)
    cache_pos = torch.tensor([0], device=device, dtype=torch.long)

    compiled_forward = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=False,
    )

    def _warmup():
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

    def _run_steps(n=6):
        cache.reset()
        for _ in range(n):
            engine.start_step()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                compiled_forward(
                    static_input, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                )
            engine.end_step()

    try:
        # Phase 1: warmup with record
        _warmup()
        engine.finalize_dual_frame()

        # Phase 2: disable record, retrace
        engine.disable_record()
        _warmup()

        slot_mapping = engine.get_slot_mapping()
        all_slot_ids = set(slot_mapping.keys())
        # Get stride_bytes from an alias tensor for coalescing prediction
        sample_alias = next(iter(engine._frame_aliases[0].values()))
        stride_bytes = sample_alias.stride(0) * sample_alias.element_size()
        print(f"  Total hooks: {len(all_slot_ids)}, batch_size: {BATCH_SIZE}, "
              f"stride_bytes: {stride_bytes}")

        # --- Test 1: select_hooks with shared requests ---
        # Note: with tiny hidden_dim=32 (128 bytes/req), coalescing merges
        # nearby requests. Verify that active requests are copied and requests
        # OUTSIDE the coalesced ranges are not.
        active_requests = {0, 7}  # extremes of batch, gap = 5*128=640 < 37.5KB → coalesced
        coalesced = _coalesced_ranges(active_requests, stride_bytes)
        resid_slots = engine.select_hooks(["hook_resid_post"], requests=active_requests)
        assert len(resid_slots) == 2, f"Expected 2 resid_post, got {len(resid_slots)}"

        # Zero pinned buffers
        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        _run_steps(6)
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None

        for sid, tensor in results.items():
            if sid in resid_slots:
                # Active requests must be copied
                for r in active_requests:
                    assert tensor[r].abs().max().item() > 0, \
                        f"slot {sid} request {r} should be copied but is zero"
                # Requests outside coalesced range must be zero
                for r in range(BATCH_SIZE):
                    if r not in coalesced:
                        assert tensor[r].abs().max().item() == 0, \
                            f"slot {sid} request {r} outside coalesced range but has data"
            else:
                # Non-selected hooks should be all zero
                assert tensor.abs().max().item() == 0, \
                    f"slot {sid} should not be copied at all"

        print(f"  Test 1 PASS: requests={active_requests}, coalesced={coalesced}, "
              f"hooks={len(resid_slots)}")

        # --- Test 2: per-hook request sets (dict form) ---
        engine.select_hooks({
            "hook_resid_post": {0, 7},
            "hook_q": {2, 4},
        })

        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        _run_steps(6)
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None

        resid_slots_dict = {
            info.slot_id for info in slot_mapping.values()
            if "hook_resid_post" in info.module_name
        }
        q_slots_dict = {
            info.slot_id for info in slot_mapping.values()
            if "hook_q" in info.module_name
        }

        for sid, tensor in results.items():
            if sid in resid_slots_dict:
                for r in [0, 7]:
                    assert tensor[r].abs().max().item() > 0, \
                        f"resid slot {sid} request {r} should be copied"
            elif sid in q_slots_dict:
                for r in [2, 4]:
                    assert tensor[r].abs().max().item() > 0, \
                        f"hook_q slot {sid} request {r} should be copied"
            else:
                assert tensor.abs().max().item() == 0, \
                    f"slot {sid} should not be copied"

        print(f"  Test 2 PASS: per-hook request sets (resid→{{0,7}}, hook_q→{{2,4}})")

        # --- Test 3: coalescing (adjacent requests merged into 1 range) ---
        engine.select_hooks(["hook_resid_post"], requests={0, 1, 2, 3})

        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        _run_steps(6)
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None

        for sid in resid_slots:
            tensor = results[sid]
            for r in [0, 1, 2, 3]:
                assert tensor[r].abs().max().item() > 0, \
                    f"slot {sid} request {r} should be copied (coalesced)"
            for r in [4, 5, 6, 7]:
                assert tensor[r].abs().max().item() == 0, \
                    f"slot {sid} request {r} should NOT be copied"

        print(f"  Test 3 PASS: coalesced range {{0,1,2,3}} → requests 4-7 are zero")

        # --- Test 4: restore all ---
        engine.select_hooks(None)

        for frame in (0, 1):
            for sid, buf in engine._pinned_buffers[frame].items():
                buf.zero_()

        _run_steps(6)
        results = engine.collect_dual_frame_results(wait=True)
        assert results is not None
        for sid, tensor in results.items():
            assert tensor.abs().max().item() > 0, \
                f"slot {sid} should be copied after restore"

        print(f"  Test 4 PASS: select_hooks(None) restores full batch")

        print(f"[PASS] per-request slicing: all tests passed")
    finally:
        engine.close()
