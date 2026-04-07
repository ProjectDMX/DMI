from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Module-level profiling list for _prepare_wrapper timing.
# Enabled by RING_PROFILE_PREPARE=1.
# Accessible via monitoring.generate._prepare_profile_times
_prepare_profile_times: List[dict] = []


def print_prepare_profile() -> None:
    """Print summary of _prepare_wrapper profiling data."""
    if not _prepare_profile_times:
        print("[prepare_profile] No data collected. Set RING_PROFILE_PREPARE=1.")
        return
    keys = ["orig_prepare", "ring_step", "shape_compute", "prepare_step", "push_metas", "total"]
    n = len(_prepare_profile_times)
    print(f"[prepare_profile] {n} steps:")
    for k in keys:
        vals = [d.get(k, 0.0) for d in _prepare_profile_times if k in d]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mx = max(vals)
        print(f"  {k:20s}: avg={avg:.3f} ms  max={mx:.3f} ms  ({len(vals)} samples)")
    # Show prepare_step result distribution
    results = [d.get("result") for d in _prepare_profile_times if "result" in d]
    if results:
        from collections import Counter
        dist = Counter(results)
        labels = {0: "RING_OK", 1: "RING_FLUSHED", 2: "CPU_DIRECT", -1: "FORCE_CPU_DIRECT"}
        parts = [f"{labels.get(k, str(k))}={v}" for k, v in sorted(dist.items())]
        print(f"  {'prepare_step results':20s}: {', '.join(parts)}")


def _make_model_shape(model: Any) -> Optional[Any]:
    """Extract ModelShapeConfig from a HF model config. Returns None on failure."""
    try:
        from .ring_transport import ModelShapeConfig
        cfg = model.config
        hidden_dim   = getattr(cfg, "hidden_size",          getattr(cfg, "n_embd",  None))
        num_heads    = getattr(cfg, "num_attention_heads",  getattr(cfg, "n_head",  None))
        num_kv_heads = getattr(cfg, "num_key_value_heads",  num_heads)
        head_dim     = getattr(cfg, "head_dim",             None)
        if hidden_dim is None or num_heads is None:
            return None
        if head_dim is None:
            head_dim = int(hidden_dim) // int(num_heads)
        dtype = getattr(model, "dtype", None) or getattr(cfg, "torch_dtype", None)
        if dtype is None:
            import torch
            dtype = torch.float16
        vocab_size = getattr(cfg, "vocab_size", 0) or 0
        intermediate_dim = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or 0
        if not intermediate_dim and getattr(cfg, "model_type", "") == "gpt2":
            intermediate_dim = 4 * int(hidden_dim)
        return ModelShapeConfig(
            hidden_dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=dtype,
            vocab_size=int(vocab_size),
            intermediate_dim=int(intermediate_dim),
        )
    except Exception:
        return None


def _filter_specs(all_specs: List, engine: Any) -> List:
    """Apply engine HookSelection to all_specs, returning only enabled ones."""
    if engine is None or engine.config is None:
        return all_specs
    try:
        enabled = set(engine.config.hooks.compile(s.name for s in all_specs))
        return [s for s in all_specs if s.name in enabled]
    except Exception:
        return all_specs




def _install_monitoring_forward(model: Any) -> None:
    """Install monitored_forward wrapper and (if possible) ring forward hooks."""
    from . import ring_transport

    # --- Forward hooks (legacy registration for _forward_hook_names) ---
    # These register_forward_hook calls are no-ops for ring transport data
    # capture.  The actual producer kernel is called inside HookPoint.forward()
    # via torch.ops.ring.producer.  The hooks here only serve to populate
    # transport._forward_hook_names so capture_tensor() skips those hooks.
    transport = ring_transport.get_active()
    if transport is not None and hasattr(model, "get_hook_specs"):
        # Auto-detect model shape if not already set
        if transport._model_cfg is None:
            cfg = _make_model_shape(model)
            if cfg is not None:
                transport.set_model_cfg(cfg)

        if transport._model_cfg is not None:
            engine       = getattr(model, "monitoring_engine", None)
            all_specs    = model.get_hook_specs()
            active_specs = _filter_specs(all_specs, engine)

            # Apply hook selection preset (sets HookPoint.enabled per hook)
            hook_selection = getattr(transport, '_hook_selection', None)
            if hook_selection is not None:
                active_specs = ring_transport.apply_hook_selection(
                    active_specs, hook_selection, cfg=transport._model_cfg)

            handles: List = []
            ring_transport.install_ring_hooks(active_specs, handles)
            transport._active_specs        = active_specs
            transport._using_forward_hooks = True
            transport._forward_hook_names  = {s.hook_type for s in active_specs}
            model._ring_hook_handles       = handles

            # Startup validation: warn if pinned staging < GPU ring
            try:
                import warnings
                pcap = transport._ring_engine.payload_cap()
                scap = transport._ring_engine.staging_cap()
                if scap < pcap:
                    warnings.warn(
                        f"[ring_transport] Pinned staging ({scap / 1e6:.0f} MB) "
                        f"< GPU ring ({pcap / 1e6:.0f} MB). "
                        f"Effective capacity is staging-limited. "
                        f"Recommend --ring-pinned-mb >= --ring-payload-mb.",
                        stacklevel=2,
                    )
            except Exception:
                pass

            import os
            if os.environ.get("RING_DEBUG_SPECS"):
                print(f"[ring] install_ring_hooks: all={len(all_specs)} active={len(active_specs)}"
                      f" model_cfg={transport._model_cfg is not None}")

            # Ring transport does not need a model.forward wrapper.
            # Pre-forward work (prepare_step, metadata push) happens in
            # _prepare_wrapper which wraps prepare_inputs_for_generation.
            # HookPoint.forward() calls ring::producer directly.
            return

    # --- monitored_forward wrapper (legacy native backend only) ---
    # Ring transport returns above.  This wrapper is only installed when
    # using the legacy native backend (start_step/end_step lifecycle).
    wrapper = getattr(model, "_monitoring_forward_wrapper", None)
    current_forward = model.forward
    if wrapper is not None and current_forward is wrapper:
        return  # already installed

    existing_orig = getattr(current_forward, "_monitoring_orig_forward", None)
    if existing_orig is not None:
        model._monitoring_orig_forward    = existing_orig
        model._monitoring_forward_wrapper = current_forward
        return

    orig_forward = current_forward

    @functools.wraps(orig_forward)
    def monitored_forward(*f_args: Any, **f_kwargs: Any):
        engine = getattr(model, "monitoring_engine", None)

        # Phase detection (Python, outside compiled region)
        phase = "prefill" if f_kwargs.get("past_key_values") is None else "decode"
        try:
            input_ids = f_kwargs.get("input_ids")
            if (hasattr(input_ids, "dim")
                    and int(input_ids.dim()) >= 2
                    and int(input_ids.shape[1]) > 1):
                phase = "prefill"
        except Exception:
            pass

        if engine is not None:
            engine.start_step(phase=phase)

        try:
            return orig_forward(*f_args, **f_kwargs)
        finally:
            if engine is not None:
                engine.end_step()

    try:
        monitored_forward.__signature__ = inspect.signature(orig_forward)
    except (TypeError, ValueError):
        pass

    monitored_forward._monitoring_orig_forward = orig_forward
    model._monitoring_orig_forward             = orig_forward
    model._monitoring_forward_wrapper          = monitored_forward
    model._monitoring_had_instance_forward     = 'forward' in model.__dict__
    model.forward                              = monitored_forward


def _uninstall_monitoring_forward(model: Any) -> None:
    """Remove ring forward hooks and restore original model.forward."""
    from . import ring_transport

    # Remove register_forward_hook handles (legacy no-op hooks on submodules;
    # see _make_ring_hook in ring_transport.py)
    for h in getattr(model, "_ring_hook_handles", []):
        try:
            h.remove()
        except Exception:
            pass
    model._ring_hook_handles = []

    # Reset transport new-path state
    transport = ring_transport.get_active()
    if transport is not None:
        transport._using_forward_hooks = False
        transport._active_specs        = []
        transport._forward_hook_names  = set()

    # Restore original model.forward
    orig = getattr(model, "_monitoring_orig_forward", None)
    if orig is not None:
        had_instance = getattr(model, "_monitoring_had_instance_forward", True)
        if had_instance:
            # Forward was an instance attr before (e.g. compiled) -- restore it
            model.forward = orig
        else:
            # Forward was a class method -- delete instance attr so class MRO resumes
            model.__dict__.pop('forward', None)
        model._monitoring_orig_forward     = None
        model._monitoring_forward_wrapper  = None
        model.__dict__.pop('_monitoring_had_instance_forward', None)


def _install_prepare_wrapper(model: Any) -> None:
    """Wrap prepare_inputs_for_generation to push ring FIFO metas before every forward.

    This is the CUDA-graph-compatible injection point: HF calls
    prepare_inputs_for_generation at Python level before every model_forward,
    including during CUDA graph replay (where monitored_forward is NOT called).
    """
    from . import ring_transport

    orig_prepare = getattr(model, "prepare_inputs_for_generation", None)
    if orig_prepare is None:
        return
    # Already wrapped.
    if getattr(model, "_monitoring_orig_prepare", None) is not None:
        return

    import os as _os
    import time as _time
    _profile = _os.environ.get("RING_PROFILE_PREPARE", "") == "1"

    @functools.wraps(orig_prepare)
    def _prepare_wrapper(*args: Any, **kwargs: Any) -> Any:
        transport = ring_transport.get_active()
        if transport is None or transport.null_offload:
            return orig_prepare(*args, **kwargs)

        if _profile:
            _t0 = _time.perf_counter()

        model_inputs = orig_prepare(*args, **kwargs)

        if _profile:
            _t_orig = _time.perf_counter()

        engine    = getattr(model, "monitoring_engine", None)

        if (engine is not None
                and getattr(engine, "_using_ring_transport", False)
                and transport is not None
                and transport._using_forward_hooks):

            if isinstance(model_inputs, dict):
                input_ids_val   = model_inputs.get("input_ids")
                attention_mask  = model_inputs.get("attention_mask")
                past_key_values = model_inputs.get("past_key_values")
                cache_position  = model_inputs.get("cache_position")
            else:
                input_ids_val   = None
                attention_mask  = None
                past_key_values = None
                cache_position  = None

            # Compute kv_offsets for attention hook kv-dimension slicing.
            # Left-padded HF batches: both static and dynamic cache have the
            # same left-padding in the kv dimension (cache_position = arange(Pmax)).
            # kv_offset = pad_len = seq_len - real_len, computed once at prefill
            # from the 2D attention mask, then reused for decode steps.
            _kv_offsets = getattr(transport, '_prefill_kv_offsets', None)
            if attention_mask is not None and _kv_offsets is None:
                try:
                    _am = attention_mask
                    if isinstance(_am, dict):
                        _am = _am.get("full_attention", _am)
                    if hasattr(_am, 'dim'):
                        ndim = _am.dim()
                        if ndim == 2:
                            _seq_len = int(_am.shape[1])
                            _real_lens = _am.sum(dim=1).tolist()
                            _kv_offsets = [_seq_len - int(rl) for rl in _real_lens]
                        elif ndim == 4:
                            _batch = int(_am.shape[0])
                            _kv_offsets = []
                            for b in range(_batch):
                                row = _am[b, 0, -1, :]
                                pad = int((row < 0).long().argmin().item())
                                if pad == 0 and row[0] < 0:
                                    pad = int(row.shape[0])
                                _kv_offsets.append(pad)
                        if _kv_offsets is not None:
                            transport._prefill_kv_offsets = _kv_offsets
                except Exception:
                    pass

            _is_static_cache = past_key_values is not None and hasattr(past_key_values, 'max_cache_len')
            engine._prepare_ring_step(input_ids_val, attention_mask, past_key_values,
                                      cache_position=cache_position, kv_offsets=_kv_offsets)

            if _profile:
                _t_ring_step = _time.perf_counter()

            if input_ids_val is not None and hasattr(input_ids_val, "shape"):
                try:
                    import torch
                    import warnings

                    batch  = int(input_ids_val.shape[0])
                    q_len  = int(input_ids_val.shape[1])
                    is_static = _is_static_cache
                    kv_dim = ring_transport._get_kv_dim(past_key_values, q_len, is_static=is_static)
                    logits_to_keep = int(model_inputs.get("logits_to_keep", 0)) if isinstance(model_inputs, dict) else 0

                    # Compute per-hook tensor byte sizes (only enabled hooks)
                    hook_byte_sizes = []
                    model_cfg = transport._model_cfg
                    if model_cfg is not None:
                        for spec in transport._active_specs:
                            shape = ring_transport._compute_hook_shape(
                                spec.hook_type, model_cfg, batch, q_len, kv_dim,
                                logits_to_keep=logits_to_keep)
                            if shape:
                                dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
                                elem_size = torch._utils._element_size(dtype)
                                nbytes = elem_size
                                for d in shape:
                                    nbytes *= d
                                hook_byte_sizes.append(nbytes)
                            else:
                                hook_byte_sizes.append(0)

                    if _profile:
                        _t_shape = _time.perf_counter()

                    if not transport._force_cpu_direct:
                        step_total_bytes = sum(ring_transport.align_up_py(b, 16) for b in hook_byte_sizes)
                        n_hooks = len(hook_byte_sizes)
                        re = transport._ring_engine

                        # Single Python->C++ call: capacity check, conditional
                        # sync+flush, pre-allocate ring space.  Returns 0/1/2.
                        # GIL released inside; resolves CUDA stream lazily
                        # in C++ only when sync is needed (cases 1 and 2).
                        if _profile:
                            _t_pre_step = _time.perf_counter()
                        result = re.prepare_step(step_total_bytes, n_hooks)
                        transport.cpu_direct = (result == 2)
                        ring_transport.set_cpu_direct(result == 2)

                        if _profile:
                            _t_prepare = _time.perf_counter()
                            if result != 0:
                                _labels = {1: "RING_FLUSHED", 2: "CPU_DIRECT"}
                                print(
                                    f"[prepare_profile] FLUSH: result={_labels.get(result, result)} "
                                    f"step={step_total_bytes/1e6:.1f}MB "
                                    f"batch={batch} q_len={q_len} n_hooks={n_hooks} "
                                    f"took={(_t_prepare - _t_pre_step)*1000:.1f}ms",
                                    flush=True)

                        if result == 2:
                            # Case B: warn once per (batch, q_len) shape
                            shape_key = (batch, q_len)
                            if shape_key not in transport._warned_shapes:
                                transport._warned_shapes.add(shape_key)
                                pcap = re.payload_cap()
                                scap = re.staging_cap()
                                if step_total_bytes > pcap and step_total_bytes > scap:
                                    reason = (f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                                              f"and pinned staging ({scap / 1e6:.0f} MB)")
                                elif step_total_bytes > pcap:
                                    reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
                                else:
                                    reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
                                warnings.warn(
                                    f"[ring_transport] Step data ({step_total_bytes / 1e6:.1f} MB) {reason}. "
                                    f"Falling back to synced eager CPU offload for all {n_hooks} hooks.",
                                    stacklevel=2,
                                )
                    else:
                        # _force_cpu_direct -- cpu_direct already True,
                        # skip prepare_step (no ring interaction needed)
                        if _profile:
                            _t_prepare = _time.perf_counter()

                    # Push FIFO metadata for p2p thread.
                    # input_ids_val is guaranteed non-None here (guarded above).
                    # On non-first PP ranks input_ids would be None and we
                    # skip this entire block, so .dtype is always safe.
                    transport.pre_push_all_metas(batch, q_len, kv_dim,
                                                logits_to_keep=logits_to_keep,
                                                token_ids_dtype=input_ids_val.dtype)

                    if _profile:
                        _t_meta = _time.perf_counter()
                        _prepare_profile_times.append({
                            "orig_prepare": (_t_orig - _t0) * 1000,
                            "ring_step": (_t_ring_step - _t_orig) * 1000,
                            "shape_compute": (_t_shape - _t_ring_step) * 1000,
                            "prepare_step": (_t_prepare - _t_shape) * 1000,
                            "push_metas": (_t_meta - _t_prepare) * 1000,
                            "total": (_t_meta - _t0) * 1000,
                            "result": result if not transport._force_cpu_direct else -1,
                        })

                except Exception:
                    pass
        elif _profile:
            _t_end = _time.perf_counter()
            _prepare_profile_times.append({
                "orig_prepare": (_t_orig - _t0) * 1000,
                "total": (_t_end - _t0) * 1000,
            })

        return model_inputs


    model._monitoring_orig_prepare         = orig_prepare
    model.prepare_inputs_for_generation    = _prepare_wrapper


def _uninstall_prepare_wrapper(model: Any) -> None:
    """Restore the original prepare_inputs_for_generation."""
    orig = getattr(model, "_monitoring_orig_prepare", None)
    if orig is not None:
        model.prepare_inputs_for_generation = orig
        model._monitoring_orig_prepare      = None


def _compute_decode_step_bytes(transport: Any, batch: int,
                                kv_dim: int = 1) -> int:
    """Estimate total padded bytes for one decode step (q_len=1).

    This runs BEFORE generate() starts -- no StaticCache exists yet.
    The caller must estimate kv_dim from generate() kwargs to match
    what HF will create:

        HF generate() computes max_cache_len as:
            max_cache_length = generation_config.max_length - 1

        where max_length = input_length + max_new_tokens (or explicit max_length).
        StaticCache is then created with max_cache_len = max_cache_length.

        For attn_scores/pattern hooks, kv_dim = max_cache_len.
        For all other hooks, kv_dim is irrelevant (shapes don't depend on it).

    The per-step _prepare_wrapper uses the REAL kv_dim from the actual
    StaticCache object (past_key_values.max_cache_len), so FIFO metadata
    shapes are always correct.  This function is only used for the Phase 3
    upfront capacity check to decide whether to disable compilation.
    """
    import torch
    from . import ring_transport as _rt

    cfg = transport._model_cfg
    if cfg is None:
        return 0
    total = 0
    for spec in transport._active_specs:
        shape = _rt._compute_hook_shape(
            spec.hook_type, cfg, batch, q_len=1, kv_dim=kv_dim,
            logits_to_keep=1)
        if shape:
            dtype = spec.dtype if spec.dtype is not None else cfg.dtype
            elem_size = torch._utils._element_size(dtype)
            nbytes = elem_size
            for d in shape:
                nbytes *= d
            total += _rt.align_up_py(nbytes, 16)
    return total


def generate_with_monitoring(model: Any, *args: Any,
                             hook_selection: Optional[str] = None,
                             timing_out: Optional[dict] = None,
                             flush_after_generate: bool = False,
                             **kwargs: Any):
    """Run HF generate() with ring-transport monitoring hooks active.

    Hooks are installed before generate() and removed on return.

    Args:
        hook_selection: preset name controlling which hooks are enabled.
            "full" (default/None) -- all hooks
            "hf-only"  -- hidden states + attention weights + logits (matches HF output_hidden_states + output_attentions)
            "hidden-states" -- residual stream + embeddings + final LN
            "logits"   -- final logits only
            "attention" -- attention scores, pattern, Q, K, V, Z

    For CUDA graph capture, use HF's built-in CompileConfig::

        from transformers import CompileConfig
        generate_with_monitoring(
            model, ...,
            cache_implementation="static",
            compile_config=CompileConfig(mode="reduce-overhead", fullgraph=False),
        )

    If the model is externally compiled (torch.compile on model or
    model.forward) AND static cache is used, this function strips the
    external compilation and injects an equivalent CompileConfig so HF
    compiles only the decode path (prefill stays uncompiled).
    """
    from . import ring_transport
    import functools
    import time
    import types
    import warnings
    import torch

    # ------------------------------------------------------------------
    # Phase 1: Strip external compilation when static cache is active.
    #
    # WHY: HF generate() has a prefill/decode split:
    #   - Prefill runs via uncompiled self(...) (different shapes each call)
    #   - Decode runs via compiled model_forward from get_compiled_call()
    #     (stable shapes, CUDA graphs via mode="reduce-overhead")
    #
    # External torch.compile (model.forward = torch.compile(...) or
    # model = torch.compile(model)) wraps BOTH prefill and decode in
    # one compiled function.  When prefill needs cpu_direct (step data
    # exceeds ring), the .cpu() calls in _hook_cpu_direct cause graph
    # breaks.  With mode="reduce-overhead", this fragments the forward
    # into many tiny CUDA graph segments.  When decode then runs with a
    # different topology (no graph breaks, full ring path), the CUDA
    # graph tree's shared buffer pool is corrupted -> segfault.
    #
    # FIX: Strip external compilation and inject HF's CompileConfig.
    # HF's get_compiled_call() only compiles decode, leaving prefill
    # uncompiled.  Prefill can safely use cpu_direct (eager, no CUDA
    # graphs).  Decode uses ring path (no graph breaks, full CUDA graph).
    #
    # WHEN: Only when static cache is active (cache_implementation="static"
    # or external StaticCache in past_key_values).  Without static cache,
    # HF doesn't compile at all, so external compilation is harmless.
    # ------------------------------------------------------------------
    _saved_compiled_forward = None   # Case 1 restore
    _saved_cache_impl = None         # generation_config restore

    # Check if HF will attempt compilation.  Two triggers:
    # 1. cache_implementation="static" (in kwargs or generation_config)
    # 2. External StaticCache object in past_key_values (is_compileable=True)
    _cache_impl_static = (
        kwargs.get('cache_implementation') == 'static'
        or getattr(getattr(model, 'generation_config', None),
                   'cache_implementation', None) == 'static'
    )
    _pkv = kwargs.get('past_key_values')
    _external_compileable_cache = (
        _pkv is not None and hasattr(_pkv, 'is_compileable') and _pkv.is_compileable
    )
    has_static_cache = _cache_impl_static or _external_compileable_cache
    is_compiled_model = hasattr(model, '_orig_mod')
    check_target = getattr(model, '_orig_mod', model)
    is_compiled_forward = 'forward' in check_target.__dict__

    if has_static_cache and (is_compiled_model or is_compiled_forward):
        # Strip external compilation
        if is_compiled_model:
            model = model._orig_mod
        elif is_compiled_forward:
            _saved_compiled_forward = check_target.__dict__['forward']
            cls_forward = type(check_target).forward
            check_target.forward = types.MethodType(cls_forward, check_target)

        # Move cache_implementation from generation_config to kwargs so
        # HF creates StaticCache internally.  Keeps past_key_values
        # (external StaticCache) untouched if the user passed one.
        gen_cfg = getattr(model, 'generation_config', None)
        if gen_cfg is not None and getattr(gen_cfg, 'cache_implementation', None) is not None:
            if 'cache_implementation' not in kwargs:
                kwargs['cache_implementation'] = gen_cfg.cache_implementation
            _saved_cache_impl = gen_cfg.cache_implementation
            gen_cfg.cache_implementation = None

        # Inject CompileConfig if user didn't provide one
        if 'compile_config' not in kwargs:
            try:
                from transformers import CompileConfig
                kwargs['compile_config'] = CompileConfig(
                    mode="reduce-overhead", fullgraph=False)
            except ImportError:
                pass  # old transformers without CompileConfig

        warnings.warn(
            "[ring_transport] External torch.compile detected with static cache. "
            "Stripping external compilation; HF will compile decode only via "
            "CompileConfig (prefill stays uncompiled). Recommend: remove "
            'torch.compile() and pass compile_config=CompileConfig('
            'mode="reduce-overhead", fullgraph=False) to generate() directly.',
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Phase 2: Resolve target, install monitoring hooks.
    # ------------------------------------------------------------------
    # Set hook selection on transport so _install_monitoring_forward
    # applies it when filtering active_specs.
    transport = ring_transport.get_active()
    if transport is not None and hook_selection is not None:
        transport._hook_selection = hook_selection

    target = getattr(model, "_orig_mod", model)
    _restore_engine: Any = None
    if target is not model:
        outer_engine = getattr(model, "monitoring_engine", None)
        _restore_engine = getattr(target, "monitoring_engine", None)
        if outer_engine is not None:
            target.monitoring_engine = outer_engine

    _install_monitoring_forward(target)
    _install_prepare_wrapper(target)

    step_timestamps: List[float] = []
    _saved_prepare_with_monitoring = target.prepare_inputs_for_generation

    @functools.wraps(_saved_prepare_with_monitoring)
    def _timed_prepare(*_args: Any, **_kwargs: Any):
        step_timestamps.append(time.perf_counter())
        return _saved_prepare_with_monitoring(*_args, **_kwargs)

    if timing_out is not None:
        target.prepare_inputs_for_generation = _timed_prepare

    # ------------------------------------------------------------------
    # Phase 3: Check if decode fits in ring.
    #
    # WHY: If even a single decode step exceeds ring capacity, EVERY step
    # will use cpu_direct (.cpu() in HookPoint.forward).  Under CUDA
    # graphs (CompileConfig), this causes graph breaks at every hook,
    # fragmenting the forward into hundreds of tiny CUDA graph segments.
    # The stale-buffer detection in PyTorch's cudagraph_trees then
    # crashes on cross-segment tensor reuse.
    #
    # FIX: Detect this upfront by computing decode step size from
    # input_ids.shape[0] (batch).  If it exceeds effective_cap:
    #   1. Set _force_cpu_direct -- _prepare_wrapper skips prepare_step
    #      entirely (avoids per-step sync+flush overhead)
    #   2. Pop compile_config + cache_implementation from kwargs
    #   3. Set disable_compile=True -- catches HF auto-compile from
    #      external StaticCache (is_compileable=True)
    # Result: entire generate() runs eager, no CUDA graphs, no crash.
    # ------------------------------------------------------------------
    transport = ring_transport.get_active()
    if transport is not None and transport._model_cfg is not None and transport._active_specs:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
            batch = int(input_ids.shape[0])
            re = transport._ring_engine
            effective_cap = min(re.payload_cap(), re.staging_cap())

            # Estimate kv_dim by calling HF's own _prepare_generated_length
            # on a deepcopy of generation_config, then computing
            # max_cache_len = max_length - 1 (same as generate() line 2498).
            #
            # This MUST match HF's generate() behavior exactly:
            #   1. _prepare_generation_config() merges kwargs into gen config
            #   2. _prepare_generated_length() resolves max_length from
            #      max_new_tokens / max_length / default(20)
            #   3. max_cache_len = max_length - 1
            #   4. StaticCache is created with this max_cache_len
            #   5. attn_scores/pattern hooks have shape [..., kv_dim]
            #      where kv_dim = max_cache_len
            #
            # The per-step _prepare_wrapper reads the REAL kv_dim from the
            # actual StaticCache (past_key_values.max_cache_len).  This
            # estimate is only for Phase 3 upfront capacity check.
            input_len = int(input_ids.shape[1])
            try:
                gen_cfg_copy, _ = model._prepare_generation_config(
                    kwargs.get('generation_config'), **kwargs)
                gen_cfg_copy = model._prepare_generated_length(
                    generation_config=gen_cfg_copy,
                    has_default_max_length=(
                        kwargs.get("max_length") is None
                        and gen_cfg_copy.max_length is not None),
                    has_default_min_length=(
                        kwargs.get("min_length") is None
                        and gen_cfg_copy.min_length is not None),
                    model_input_name="input_ids",
                    inputs_tensor=input_ids,
                    input_ids_length=input_len,
                )
                kv_dim_estimate = int(gen_cfg_copy.max_length) - 1
            except Exception:
                max_new = int(kwargs.get('max_new_tokens', 20))
                kv_dim_estimate = input_len + max_new - 1
                warnings.warn(
                    "[ring_transport] Could not call model._prepare_generated_length "
                    "to estimate kv_dim for capacity check. Falling back to "
                    f"kv_dim={kv_dim_estimate} (input_len={input_len} + "
                    f"max_new_tokens={max_new} - 1). This may under-estimate "
                    "attn_scores/pattern hook sizes if max_position_embeddings "
                    "caps max_length.",
                    stacklevel=2,
                )

            decode_bytes = _compute_decode_step_bytes(
                transport, batch, kv_dim=kv_dim_estimate)

            if decode_bytes > effective_cap:
                transport.cpu_direct = True
                transport._force_cpu_direct = True
                ring_transport.set_cpu_direct(True)
                # Disable all compilation paths in HF generate():
                # 1. Pop compile_config from kwargs
                # 2. Pop cache_implementation (HF defaults to CompileConfig()
                #    when cache_implementation="static" even without explicit
                #    compile_config)
                # 3. Set disable_compile=True on generation_config (HF also
                #    auto-compiles when past_key_values is a compileable
                #    Cache object like StaticCache, regardless of
                #    cache_implementation)
                had_compile = bool(
                    kwargs.pop('compile_config', None) is not None
                    or kwargs.pop('cache_implementation', None) is not None
                )
                kwargs['disable_compile'] = True
                msg = (
                    f"[ring_transport] Decode step ({decode_bytes / 1e6:.1f} MB) "
                    f"exceeds ring capacity ({effective_cap / 1e6:.0f} MB). "
                    f"All hooks using synced CPU-direct offload."
                )
                if had_compile:
                    msg += " Disabled CUDA graph compilation for this generate() call."
                warnings.warn(msg, stacklevel=2)

    _t_generate_start = None
    _t_generate_end = None
    _flush_ms = 0.0
    result = None
    try:
        if timing_out is not None:
            torch.cuda.synchronize()
            _t_generate_start = time.perf_counter()
        result = model.generate(*args, **kwargs)
        if timing_out is not None:
            torch.cuda.synchronize()
            _t_generate_end = time.perf_counter()
            if flush_after_generate and transport is not None:
                ring_engine = getattr(transport, "_ring_engine", None)
                if ring_engine is not None:
                    _flush_ms = float(ring_engine.flush_and_wait()) / 1000.0
            total_ms = (_t_generate_end - _t_generate_start) * 1000.0
            timing_out.clear()
            timing_out["generate_total_ms"] = total_ms
            timing_out["flush_ms"] = _flush_ms
            timing_out["total_with_flush_ms"] = total_ms + _flush_ms
            timing_out["step_timestamps"] = list(step_timestamps)
            if len(step_timestamps) >= 2:
                prefill_ms = (step_timestamps[1] - step_timestamps[0]) * 1000.0
                decode_total_ms = (_t_generate_end - step_timestamps[1]) * 1000.0
                decode_steps = len(step_timestamps) - 1
                timing_out["prefill_ms"] = prefill_ms
                timing_out["decode_total_ms"] = decode_total_ms
                timing_out["decode_total_with_flush_ms"] = decode_total_ms + _flush_ms
                timing_out["decode_steps"] = decode_steps
                timing_out["decode_last_step_ms"] = (_t_generate_end - step_timestamps[-1]) * 1000.0
            elif len(step_timestamps) == 1:
                timing_out["prefill_ms"] = total_ms
                timing_out["decode_total_ms"] = 0.0
                timing_out["decode_total_with_flush_ms"] = _flush_ms
                timing_out["decode_steps"] = 0
                timing_out["decode_last_step_ms"] = 0.0
            else:
                timing_out["prefill_ms"] = total_ms
                timing_out["decode_total_ms"] = 0.0
                timing_out["decode_total_with_flush_ms"] = _flush_ms
                timing_out["decode_steps"] = 0
                timing_out["decode_last_step_ms"] = 0.0
        return result
    finally:
        if timing_out is not None:
            target.prepare_inputs_for_generation = _saved_prepare_with_monitoring
        _uninstall_monitoring_forward(target)
        _uninstall_prepare_wrapper(target)
        if target is not model:
            target.monitoring_engine = _restore_engine
        if _saved_compiled_forward is not None:
            check_target.forward = _saved_compiled_forward
        if _saved_cache_impl is not None:
            gen_cfg = getattr(model, 'generation_config', None)
            if gen_cfg is not None:
                gen_cfg.cache_implementation = _saved_cache_impl
        if transport is not None:
            transport.cpu_direct = False
            transport._force_cpu_direct = False
            ring_transport.set_cpu_direct(False)
            transport._hook_selection = None
            transport._prefill_kv_offsets = None


# ---------------------------------------------------------------------------
# generate_greedy: manual prefill + decode loop, no HF generate() overhead
# ---------------------------------------------------------------------------

@dataclass
class GreedyGenerateTimings:
    """Optional per-step timing data from generate_greedy."""
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    decode_steps: int = 0
    batch_size: int = 0
    prefill_tokens: int = 0
    step_ms: List[float] = field(default_factory=list)

    @property
    def prefill_tok_per_s(self) -> float:
        """Prefill throughput: prompt tokens processed per second."""
        if self.prefill_ms <= 0:
            return 0.0
        return self.batch_size * self.prefill_tokens / self.prefill_ms * 1000.0

    @property
    def decode_tok_per_s(self) -> float:
        """Decode throughput: new tokens generated per second."""
        if self.decode_ms <= 0 or self.decode_steps <= 0:
            return 0.0
        return self.batch_size * self.decode_steps / self.decode_ms * 1000.0

    @property
    def e2e_tok_per_s(self) -> float:
        """E2E throughput: all tokens (prompt + generated) per second."""
        if self.total_ms <= 0:
            return 0.0
        total_tokens = self.batch_size * (self.prefill_tokens + self.decode_steps)
        return total_tokens / self.total_ms * 1000.0

    @property
    def tpot_ms(self) -> float:
        """Time per output token (decode only)."""
        if self.decode_steps <= 0:
            return 0.0
        return self.decode_ms / self.decode_steps


def _prepare_ring_forward(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    past_key_values: Any = None,
    cache_position: Any = None,
    logits_to_keep: int = 0,
) -> None:
    """Call engine._prepare_ring_step + prepare_step + pre_push_all_metas."""
    from . import ring_transport
    import torch

    engine = getattr(model, "monitoring_engine", None)
    transport = ring_transport.get_active()
    if engine is None or transport is None:
        return
    if not getattr(engine, "_using_ring_transport", False):
        return
    if not transport._using_forward_hooks:
        return

    engine._prepare_ring_step(
        input_ids, attention_mask, past_key_values,
        cache_position=cache_position)

    batch = int(input_ids.shape[0])
    q_len = int(input_ids.shape[1])
    is_static = (past_key_values is not None
                 and hasattr(past_key_values, "max_cache_len"))
    kv_dim = ring_transport._get_kv_dim(
        past_key_values, q_len, is_static=is_static)

    hook_byte_sizes: List[int] = []
    model_cfg = transport._model_cfg
    if model_cfg is not None:
        for spec in transport._active_specs:
            shape = ring_transport._compute_hook_shape(
                spec.hook_type, model_cfg, batch, q_len, kv_dim,
                logits_to_keep=logits_to_keep)
            if shape:
                dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
                elem_size = torch._utils._element_size(dtype)
                nbytes = elem_size
                for d in shape:
                    nbytes *= d
                hook_byte_sizes.append(nbytes)
            else:
                hook_byte_sizes.append(0)

    if not transport._force_cpu_direct:
        step_total = sum(
            ring_transport.align_up_py(s, 16) for s in hook_byte_sizes)
        result = transport._ring_engine.prepare_step(
            step_total, len(hook_byte_sizes))
        transport.cpu_direct = (result == 2)

    transport.pre_push_all_metas(
        batch, q_len, kv_dim, logits_to_keep=logits_to_keep)


def generate_greedy(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    logits_to_keep: int = 0,
    cuda_graphs: bool = False,
    monitoring: bool = False,
    hook_selection: Optional[str] = None,
    timings: Optional[GreedyGenerateTimings] = None,
) -> List[Any]:
    """Greedy-argmax generate loop, no HF generate() overhead.

    Follows the same pattern as the hf_offload manual loop in
    benchmark/bench_ring_transport.py:
    - Optional torch.compile + StaticCache + CUDA graphs for decode
    - Without cuda_graphs: uses HF default (DynamicCache) for KV cache
    - Per-step CPU sync via token.cpu() (matches HF generate()'s implicit
      GPU→CPU sync from ``unfinished_sequences.max() == 0``)
    - Generated tokens moved to CPU per step

    Supports EOS stopping (after min_new_tokens), min_new_tokens, max_new_tokens.
    Greedy argmax only.  No beam search, no sampling.

    Args:
        model: HF model (AutoModelForCausalLM or similar).
        input_ids: [B, seq_len] input token IDs on CUDA.
        attention_mask: [B, seq_len] attention mask on CUDA.
        max_new_tokens: Maximum tokens to generate.
        min_new_tokens: Minimum tokens before EOS can stop generation.
        eos_token_id: EOS token ID.  None = never stop early.
        pad_token_id: Pad token ID (unused, kept for compat).
        logits_to_keep: 0 = all, 1 = last position only.
        cuda_graphs: If True, compile decode step with reduce-overhead + StaticCache.
                     If False, use HF default DynamicCache, no compilation.
        monitoring: If True, install ring transport hooks and call
                    _prepare_ring_forward before each forward pass.
        hook_selection: Hook selection preset (e.g. "hidden-states", "full").
                        Only used when monitoring=True.
        timings: If provided, filled with timing data.

    Returns:
        List of generated token ID tensors on CPU, one per batch element.
        Each tensor has shape [num_generated_tokens].
    """
    import torch

    device = input_ids.device
    B, Pmax = input_ids.shape

    _wants_position_ids = (
        "position_ids" in inspect.signature(model.forward).parameters)

    def _position_ids_from_mask(mask: Any) -> Any:
        pos = mask.long().cumsum(dim=-1) - 1
        pos.masked_fill_(mask == 0, 0)
        return pos

    # -- Monitoring setup --
    if monitoring:
        from . import ring_transport
        transport = ring_transport.get_active()
        if transport is not None and hook_selection is not None:
            transport._hook_selection = hook_selection
        _install_monitoring_forward(model)

    try:
        # -- Cache setup --
        if cuda_graphs:
            from transformers import StaticCache
            max_cache_len = Pmax + max_new_tokens + 4
            cache = StaticCache(
                config=model.config, batch_size=B,
                max_cache_len=max_cache_len, device=device,
                dtype=model.dtype,
            )
        else:
            cache = None  # DynamicCache, created by model on first call

        # -- Decode step function (same as hf_offload in bench_ring_transport) --
        def _decode_step_static(token, cache, cache_position):
            kwargs: Dict[str, Any] = {
                "input_ids": token,
                "use_cache": True,
                "past_key_values": cache,
                "cache_position": cache_position,
                "output_hidden_states": False,
                "output_attentions": False,
                "return_dict": True,
                "logits_to_keep": logits_to_keep,
            }
            if _wants_position_ids:
                kwargs["position_ids"] = cache_position.unsqueeze(0).expand(
                    token.shape[0], -1)
            return model(**kwargs)

        if cuda_graphs:
            compiled_decode = torch.compile(
                _decode_step_static, mode="reduce-overhead", fullgraph=False)
        else:
            compiled_decode = None

        do_timing = timings is not None
        torch.cuda.synchronize()
        t0 = time.perf_counter() if do_timing else 0.0

        # -- Prefill (uncompiled, variable-length input) --
        with torch.no_grad():
            prefill_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "output_hidden_states": False,
                "output_attentions": False,
                "return_dict": True,
                "logits_to_keep": logits_to_keep,
            }
            if cache is not None:
                prefill_kwargs["past_key_values"] = cache
                prefill_kwargs["cache_position"] = torch.arange(
                    Pmax, device=device, dtype=torch.long)
            if _wants_position_ids:
                prefill_kwargs["position_ids"] = _position_ids_from_mask(
                    attention_mask)

            if monitoring:
                _prepare_ring_forward(
                    model, input_ids, attention_mask,
                    past_key_values=prefill_kwargs.get("past_key_values"),
                    cache_position=prefill_kwargs.get("cache_position"),
                    logits_to_keep=logits_to_keep)

            out = model(**prefill_kwargs)

        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # .cpu() forces GPU→CPU sync (same effect as HF generate()'s
        # ``this_peer_finished = unfinished_sequences.max() == 0``)
        token_cpu = token.squeeze(-1).cpu()

        if cuda_graphs:
            cache_pos = torch.tensor(
                [Pmax], device=device, dtype=torch.long)
        else:
            cache = out.past_key_values

        t_decode_start = time.perf_counter() if do_timing else 0.0

        # -- EOS tracking on GPU (same ops as HF generate() lines 2820-2821) --
        unfinished_sequences = torch.ones(
            B, dtype=torch.long, device=device)

        # -- Decode loop (matches hf_offload overhead per step) --
        _prev_step_t = t_decode_start if do_timing else 0.0
        all_generated: List[Any] = [token.squeeze(-1)]
        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                if compiled_decode is not None:
                    if monitoring:
                        _prepare_ring_forward(
                            model, token, attention_mask,
                            past_key_values=cache,
                            cache_position=cache_pos,
                            logits_to_keep=logits_to_keep)
                    torch.compiler.cudagraph_mark_step_begin()
                    out = compiled_decode(token, cache, cache_pos)
                else:
                    decode_kwargs: Dict[str, Any] = {
                        "input_ids": token,
                        "past_key_values": cache,
                        "use_cache": True,
                        "output_hidden_states": False,
                        "output_attentions": False,
                        "return_dict": True,
                        "logits_to_keep": logits_to_keep,
                    }
                    if _wants_position_ids:
                        seq_pos = Pmax + step + 1
                        decode_kwargs["position_ids"] = torch.full(
                            (B, 1), seq_pos, device=device, dtype=torch.long)
                    if monitoring:
                        _prepare_ring_forward(
                            model, token, attention_mask,
                            past_key_values=cache,
                            logits_to_keep=logits_to_keep)
                    out = model(**decode_kwargs)
                    cache = out.past_key_values

                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if cuda_graphs:
                    cache_pos = cache_pos + 1

                all_generated.append(token.squeeze(-1))

                # EOS check on GPU — replaces hf_offload's token.max().item()
                # with same-cost GPU reduction, but also checks EOS.
                # (matches HF generate() lines 2820-2821)
                tokens_generated = step + 2
                if (eos_token_id is not None
                        and tokens_generated > min_new_tokens):
                    unfinished_sequences = unfinished_sequences & (
                        token.squeeze(-1) != eos_token_id).long()
                # GPU→CPU sync (same cost as token.max().item())
                this_peer_finished = unfinished_sequences.max() == 0

                if do_timing:
                    step_t = time.perf_counter()
                    timings.step_ms.append(
                        (step_t - _prev_step_t) * 1000.0)
                    _prev_step_t = step_t

                if this_peer_finished:
                    break

        torch.cuda.synchronize()

        if do_timing:
            t_end = time.perf_counter()
            timings.total_ms = (t_end - t0) * 1000.0
            timings.prefill_ms = (t_decode_start - t0) * 1000.0
            timings.decode_ms = (t_end - t_decode_start) * 1000.0
            timings.decode_steps = len(timings.step_ms)
            timings.batch_size = B
            timings.prefill_tokens = Pmax

        # Stack on GPU, move to CPU once, truncate per-sequence at EOS
        # (after timing — this is result collection, not part of generation)
        generated_ids = torch.stack(all_generated, dim=1).cpu()  # [B, T]
        results: List[Any] = []
        for b in range(B):
            seq = generated_ids[b]
            if eos_token_id is not None:
                eos_positions = (seq == eos_token_id).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    seq = seq[:int(eos_positions[0].item()) + 1]
            results.append(seq)

        return results

    finally:
        if monitoring:
            _uninstall_monitoring_forward(model)
