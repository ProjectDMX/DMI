"""Hugging Face model-shape derivation helpers.

``_make_model_shape_from_hf_config`` is the root-owned HF adapter's legacy
helper. External framework integrations use the public equivalent in
``dmi.api.v1``.

The helper takes a HuggingFace-shaped config object plus an optional dtype
override and returns a ``ModelShapeConfig``. TP fields default to
``tp_size=1, tp_rank=0``; each adapter's ``detect_parallel_ranks``
fills them in.

Dependency direction: ``dmi.adapters`` → ``dmi.transport`` for
``ModelShapeConfig``. Core transport code does not import framework adapters.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ...transport.ring import ModelShapeConfig


def _make_model_shape_from_hf_config(
    hf_config: Any,
    dtype: Optional[torch.dtype] = None,
) -> Optional[ModelShapeConfig]:
    """Build a ``ModelShapeConfig`` from a HuggingFace-shaped config object.

    Reads the standard transformer config fields:
      * hidden_size (or n_embd for GPT-2 family)
      * num_attention_heads (or n_head)
      * num_key_value_heads (default = num_attention_heads for MHA)
      * head_dim (default = hidden_size // num_attention_heads)
      * vocab_size
      * intermediate_size (or n_inner; falls back to 4 * hidden_size for GPT-2)
      * torch_dtype (overridden by the ``dtype`` argument when provided)

    Returns ``None`` if required fields are missing.
    """
    cfg = hf_config
    hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if hidden_dim is None or num_heads is None:
        return None
    if head_dim is None:
        head_dim = int(hidden_dim) // int(num_heads)
    if dtype is None:
        dtype = getattr(cfg, "torch_dtype", None)
    if dtype is None:
        dtype = torch.float16
    vocab_size = getattr(cfg, "vocab_size", 0) or 0
    num_experts = getattr(cfg, "num_experts", 0) or 0
    top_k = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "top_k", None)
        or 0
    )
    intermediate_dim = (
        getattr(cfg, "intermediate_size", None)
        or getattr(cfg, "n_inner", None)
        or 0
    )
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
        num_experts=int(num_experts),
        top_k=int(top_k),
        tp_size=1,
        tp_rank=0,
    )


__all__ = ["_make_model_shape_from_hf_config"]
