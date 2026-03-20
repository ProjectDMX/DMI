from __future__ import annotations

import functools
import inspect
from typing import Any, List, Optional

# Module-level profiling list for _prepare_wrapper timing.
# Accessible via monitoring.generate._prepare_profile_times
_prepare_profile_times: List[dict] = []


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
        return ModelShapeConfig(
            hidden_dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=dtype,
            vocab_size=int(vocab_size),
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
            handles: List = []
            ring_transport.install_ring_hooks(active_specs, handles)
            transport._active_specs        = active_specs
            transport._using_forward_hooks = True
            transport._forward_hook_names  = {s.name for s in active_specs}
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


    # --- monitored_forward wrapper ---
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

        using_ring = engine is not None and getattr(engine, "_using_ring_transport", False)
        if engine is not None and not using_ring:
            engine.start_step(phase=phase)

        try:
            return orig_forward(*f_args, **f_kwargs)
        finally:
            if engine is not None and not using_ring:
                engine.end_step()

    try:
        monitored_forward.__signature__ = inspect.signature(orig_forward)
    except (TypeError, ValueError):
        pass

    monitored_forward._monitoring_orig_forward = orig_forward
    model._monitoring_orig_forward             = orig_forward
    model._monitoring_forward_wrapper          = monitored_forward
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
        model.forward                      = orig
        model._monitoring_orig_forward     = None
        model._monitoring_forward_wrapper  = None


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

    @functools.wraps(orig_prepare)
    def _prepare_wrapper(*args: Any, **kwargs: Any) -> Any:
        model_inputs = orig_prepare(*args, **kwargs)

        engine    = getattr(model, "monitoring_engine", None)
        transport = ring_transport.get_active()

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

            engine._prepare_ring_step(input_ids_val, attention_mask, past_key_values, cache_position=cache_position)

            if input_ids_val is not None and hasattr(input_ids_val, "shape"):
                try:
                    import torch
                    import warnings

                    batch  = int(input_ids_val.shape[0])
                    q_len  = int(input_ids_val.shape[1])
                    is_static = past_key_values is not None and hasattr(past_key_values, 'max_cache_len')
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

                    if not transport._force_cpu_direct:
                        step_total_bytes = sum(ring_transport.align_up_py(b, 16) for b in hook_byte_sizes)
                        n_hooks = len(hook_byte_sizes)
                        re = transport._ring_engine

                        # Single Python->C++ call: capacity check, conditional
                        # sync+flush, pre-allocate ring space.  Returns 0/1/2.
                        # GIL released inside; resolves CUDA stream lazily
                        # in C++ only when sync is needed (cases 1 and 2).
                        result = re.prepare_step(step_total_bytes, n_hooks)
                        transport.cpu_direct = (result == 2)

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
                    # else: _force_cpu_direct -- cpu_direct already True,
                    # skip prepare_step (no ring interaction needed)

                    # Push FIFO metadata for p2p thread
                    transport.pre_push_all_metas(batch, q_len, kv_dim,
                                                logits_to_keep=logits_to_keep)
                except Exception:
                    pass

        return model_inputs


    model._monitoring_orig_prepare         = orig_prepare
    model.prepare_inputs_for_generation    = _prepare_wrapper


def _uninstall_prepare_wrapper(model: Any) -> None:
    """Restore the original prepare_inputs_for_generation."""
    orig = getattr(model, "_monitoring_orig_prepare", None)
    if orig is not None:
        model.prepare_inputs_for_generation = orig
        model._monitoring_orig_prepare      = None


def _compute_decode_step_bytes(transport: Any, batch: int) -> int:
    """Total padded bytes for one decode step (q_len=1) across all active hooks."""
    import torch
    from . import ring_transport as _rt

    cfg = transport._model_cfg
    if cfg is None:
        return 0
    total = 0
    for spec in transport._active_specs:
        shape = _rt._compute_hook_shape(
            spec.hook_type, cfg, batch, q_len=1, kv_dim=1,
            logits_to_keep=1)
        if shape:
            dtype = spec.dtype if spec.dtype is not None else cfg.dtype
            elem_size = torch._utils._element_size(dtype)
            nbytes = elem_size
            for d in shape:
                nbytes *= d
            total += _rt.align_up_py(nbytes, 16)
    return total


def generate_with_monitoring(model: Any, *args: Any, **kwargs: Any):
    """Run HF generate() with ring-transport monitoring hooks active.

    Hooks are installed before generate() and removed on return.

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
    import types
    import warnings

    # ------------------------------------------------------------------
    # Phase 1: Strip external compilation when static cache is active.
    #
    # External torch.compile wraps both prefill and decode, which causes
    # CUDA graph topology conflicts when prefill uses cpu_direct.  HF's
    # internal get_compiled_call() only compiles decode -- correct.
    #
    # We detect external compilation, strip it, move cache_implementation
    # into kwargs, and inject CompileConfig so HF handles the split.
    #
    # Without static cache, external compilation is harmless (no CUDA
    # graph capture), so we leave it alone.
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
    target = getattr(model, "_orig_mod", model)
    _restore_engine: Any = None
    if target is not model:
        outer_engine = getattr(model, "monitoring_engine", None)
        _restore_engine = getattr(target, "monitoring_engine", None)
        if outer_engine is not None:
            target.monitoring_engine = outer_engine

    _install_monitoring_forward(target)
    _install_prepare_wrapper(target)

    # ------------------------------------------------------------------
    # Phase 3: Check if decode fits in ring.  If not, force cpu_direct
    # for the entire generate() call and strip compile_config to prevent
    # HF from compiling decode (cpu_direct graph breaks crash CUDA graphs).
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
            decode_bytes = _compute_decode_step_bytes(transport, batch)

            if decode_bytes > effective_cap:
                transport.cpu_direct = True
                transport._force_cpu_direct = True
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

    try:
        return model.generate(*args, **kwargs)
    finally:
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
