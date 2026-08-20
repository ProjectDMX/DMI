"""Framework-neutral model-shape derivation for integration API v1."""

from __future__ import annotations

from typing import Any, Optional

import torch

from monitoring.ring_transport import ModelShapeConfig


def effective_intermediate_dim(cfg: Any) -> int:
    """Return the tensor width at the post-activation MLP boundary."""

    intermediate_dim = (
        getattr(cfg, "intermediate_size", None)
        or getattr(cfg, "n_inner", None)
        or 0
    )
    if not intermediate_dim:
        return 0
    if getattr(cfg, "block_auto_adjust_ff_dim", False):
        intermediate_dim = int(2 * int(intermediate_dim) / 3)
        multiplier = getattr(cfg, "block_ffn_dim_multiplier", None)
        if multiplier is not None:
            intermediate_dim = int(multiplier * intermediate_dim)
        multiple_of = int(getattr(cfg, "block_multiple_of"))
        intermediate_dim = multiple_of * (
            (intermediate_dim + multiple_of - 1) // multiple_of
        )
    return int(intermediate_dim)


def make_model_shape_from_hf_config(
    hf_config: Any,
    dtype: Optional[torch.dtype] = None,
) -> Optional[ModelShapeConfig]:
    """Build a model-shape description from a Hugging-Face-shaped config.

    The object need not inherit from a Transformers class. Only the standard
    configuration attributes read below are required.
    """

    # Multimodal wrappers keep decoder geometry under ``text_config``.
    cfg = getattr(hf_config, "text_config", hf_config)
    hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    num_heads = getattr(
        cfg,
        "num_attention_heads",
        getattr(cfg, "n_head", None),
    )
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
    num_experts = (
        getattr(cfg, "num_experts", None)
        or getattr(cfg, "num_local_experts", None)
        or getattr(cfg, "n_routed_experts", None)
        or 0
    )
    top_k = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "top_k", None)
        or 0
    )
    intermediate_dim = effective_intermediate_dim(cfg)
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


__all__ = ["effective_intermediate_dim", "make_model_shape_from_hf_config"]
