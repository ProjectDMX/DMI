"""Framework-neutral model-shape derivation for DMI API v1."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ...hooks.specs import ModelShapeConfig


def make_model_shape_from_hf_config(
    hf_config: Any,
    dtype: Optional[torch.dtype] = None,
) -> Optional[ModelShapeConfig]:
    """Build a model-shape description from a Hugging-Face-shaped config.

    The object need not inherit from a Transformers class. Only the standard
    configuration attributes read below are required.
    """

    hidden_dim = getattr(
        hf_config, "hidden_size", getattr(hf_config, "n_embd", None)
    )
    num_heads = getattr(
        hf_config,
        "num_attention_heads",
        getattr(hf_config, "n_head", None),
    )
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_dim = getattr(hf_config, "head_dim", None)
    if hidden_dim is None or num_heads is None:
        return None
    if head_dim is None:
        head_dim = int(hidden_dim) // int(num_heads)

    if dtype is None:
        dtype = getattr(hf_config, "torch_dtype", None)
    if dtype is None:
        dtype = torch.float16

    vocab_size = getattr(hf_config, "vocab_size", 0) or 0
    num_experts = getattr(hf_config, "num_experts", 0) or 0
    top_k = (
        getattr(hf_config, "num_experts_per_tok", None)
        or getattr(hf_config, "top_k", None)
        or 0
    )
    intermediate_dim = (
        getattr(hf_config, "intermediate_size", None)
        or getattr(hf_config, "n_inner", None)
        or 0
    )
    if not intermediate_dim and getattr(hf_config, "model_type", "") == "gpt2":
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


__all__ = ["make_model_shape_from_hf_config"]
