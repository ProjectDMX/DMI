"""Framework-neutral model geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelShapeConfig:
    """Describes attention geometry for analytical shape computation."""

    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    vocab_size: int = 0
    intermediate_dim: int = 0
    num_experts: int = 0
    top_k: int = 0
    tp_size: int = 1
    tp_rank: int = 0


__all__ = ["ModelShapeConfig"]
