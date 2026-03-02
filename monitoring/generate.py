from __future__ import annotations

import functools
import inspect
from typing import Any


def _install_monitoring_forward(model: Any) -> None:
    wrapper = getattr(model, "_monitoring_forward_wrapper", None)
    current_forward = model.forward
    if wrapper is not None and current_forward is wrapper:
        return

    existing_orig = getattr(current_forward, "_monitoring_orig_forward", None)
    if existing_orig is not None:
        model._monitoring_orig_forward = existing_orig
        model._monitoring_forward_wrapper = current_forward
        return

    orig_forward = current_forward

    @functools.wraps(orig_forward)
    def monitored_forward(*f_args: Any, **f_kwargs: Any):
        engine = getattr(model, "monitoring_engine", None)
        phase = "prefill" if f_kwargs.get("past_key_values") is None else "decode"
        try:
            input_ids = f_kwargs.get("input_ids")
            if hasattr(input_ids, "dim") and int(input_ids.dim()) >= 2 and int(input_ids.shape[1]) > 1:
                phase = "prefill"
        except Exception:
            pass
        if engine is not None:
            engine.start_step(phase=phase)
        try:
            model_out, _cache = model.run_with_cache(
                *f_args,
                forward_fn=orig_forward,
                **f_kwargs,
            )
            return model_out
        finally:
            if engine is not None:
                engine.end_step()

    try:
        monitored_forward.__signature__ = inspect.signature(orig_forward)
    except (TypeError, ValueError):
        pass

    monitored_forward._monitoring_orig_forward = orig_forward
    model._monitoring_orig_forward = orig_forward
    model._monitoring_forward_wrapper = monitored_forward
    model.forward = monitored_forward


def generate_with_monitoring(model: Any, *args: Any, **kwargs: Any):
    """Run HF generate() while keeping monitoring hooks active per step."""
    _install_monitoring_forward(model)
    return model.generate(*args, **kwargs)
