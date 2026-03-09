from __future__ import annotations

import functools
import inspect
from typing import Any, List, Optional


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

    # --- Forward hooks (new CUDA-graph-compatible path) ---
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
    model.forward                              = monitored_forward


def _uninstall_monitoring_forward(model: Any) -> None:
    """Remove ring forward hooks and restore original model.forward."""
    from . import ring_transport

    # Remove register_forward_hook handles (ring producer hooks on submodules)
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
                    batch  = int(input_ids_val.shape[0])
                    q_len  = int(input_ids_val.shape[1])
                    kv_dim = ring_transport._get_kv_dim(past_key_values, q_len)
                    transport.pre_push_all_metas(batch, q_len, kv_dim)
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


def generate_with_monitoring(model: Any, *args: Any, **kwargs: Any):
    """Run HF generate() with ring-transport monitoring hooks active.

    Hooks are installed before generate() and removed on return.

    For CUDA graph capture, compile the model before calling:

        model = torch.compile(model, mode="reduce-overhead")
        generate_with_monitoring(model, ..., cache_implementation="static")
    """
    # When model is a torch.compile OptimizedModule, model.generate is a
    # bound method of _orig_mod (the inner uncompiled model).  Inside
    # generate, self(**model_inputs) calls _orig_mod.forward, not the
    # compiled wrapper forward.  We must therefore:
    #   1. Install monitored_forward on _orig_mod.forward.
    #   2. Copy monitoring_engine from the OptimizedModule to _orig_mod
    #      because monitored_forward's closure looks up engine via model.
    target = getattr(model, "_orig_mod", model)
    _restore_engine: Any = None
    if target is not model:
        outer_engine = getattr(model, "monitoring_engine", None)
        _restore_engine = getattr(target, "monitoring_engine", None)
        if outer_engine is not None:
            target.monitoring_engine = outer_engine
    _install_monitoring_forward(target)
    # Wrap prepare_inputs_for_generation so ring FIFO metas are pushed before
    # every model_forward — including CUDA graph replay steps where
    # monitored_forward's Python body is never executed.
    _install_prepare_wrapper(target)
    try:
        return model.generate(*args, **kwargs)
    finally:
        _uninstall_monitoring_forward(target)
        _uninstall_prepare_wrapper(target)
        if target is not model:
            target.monitoring_engine = _restore_engine
