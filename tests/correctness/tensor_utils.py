# tests/correctness/tensor_utils.py
"""Tensor utilities for correctness tests: bitwise comparison and DB chunk merging."""

from __future__ import annotations

from typing import Any, List, Sequence

import torch


# ---------------------------------------------------------------------------
# Bitwise equality
# ---------------------------------------------------------------------------


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.detach().cpu().contiguous()
    b = b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# ---------------------------------------------------------------------------
# Model layer count
# ---------------------------------------------------------------------------


def get_num_layers_from_config(model: Any) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no .config")
    for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "num_attention_layers"):
        if hasattr(cfg, attr):
            v = int(getattr(cfg, attr))
            if v > 0:
                return v
    raise ValueError(f"could not find num layers in config attrs={dir(cfg)}")


# ---------------------------------------------------------------------------
# Act name predicates
# ---------------------------------------------------------------------------


def _is_attn_scores(act_name: str) -> bool:
    return act_name.endswith("attn.hook_attn_scores")


def _is_attn_pattern(act_name: str) -> bool:
    return act_name.endswith("attn.hook_pattern")


# ---------------------------------------------------------------------------
# DB chunk merging
# ---------------------------------------------------------------------------


class _OnDimSegments:
    """Non-attn: concat along token dimension 0 (DB tensors are [T, ...])."""

    def __init__(self, token_dim: int = 0):
        self._token_dim = token_dim
        self._chunks: List[torch.Tensor] = []

    def extend(self, chunks: Sequence[torch.Tensor]) -> None:
        self._chunks.extend(list(chunks))

    def read_and_merge(self) -> torch.Tensor:
        return torch.cat(self._chunks, dim=self._token_dim)


class _AttnMatrixSegments:
    """Attn matrices: expected chunk shape [H, q_chunk, k_up_to_now] (no batch)."""

    def __init__(self, fill_value: float):
        self._fill_value = float(fill_value)
        self._chunks: List[torch.Tensor] = []

    def extend(self, chunks: Sequence[torch.Tensor]) -> None:
        for t in chunks:
            # defensive: if some path still writes [1,H,Q,K]
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            self._chunks.append(t)

    def read_and_merge(self) -> torch.Tensor:
        td_inc, td_sum = 1, 2  # [H, Q, K]
        total_k = int(self._chunks[-1].shape[td_sum])
        padded: List[torch.Tensor] = []
        for t in self._chunks:
            if int(t.shape[td_sum]) > total_k:
                t = t.narrow(td_sum, 0, total_k)
            pad_len = total_k - int(t.shape[td_sum])
            if pad_len > 0:
                pad_shape = list(t.shape)
                pad_shape[td_sum] = pad_len
                pad_t = torch.full(pad_shape, self._fill_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad_t], dim=td_sum)
            padded.append(t)

        merged = torch.cat(padded, dim=td_inc)
        if int(merged.shape[td_inc]) > total_k:
            merged = merged.narrow(td_inc, 0, total_k)
        return merged


def merge_segments(chunks: Sequence[torch.Tensor], act_name: str) -> torch.Tensor:
    if _is_attn_scores(act_name):
        mgr = _AttnMatrixSegments(fill_value=float("-inf"))
        mgr.extend(chunks)
        return mgr.read_and_merge()
    if _is_attn_pattern(act_name):
        mgr = _AttnMatrixSegments(fill_value=0.0)
        mgr.extend(chunks)
        return mgr.read_and_merge()
    mgr2 = _OnDimSegments(token_dim=0)
    mgr2.extend(chunks)
    return mgr2.read_and_merge()
