"""Unit tests for TP-aware shape computation in _compute_hook_shape.

No GPU or distributed setup needed. These tests still import ring_transport's
native hook-definition layer, so they are not part of the no-native-build CPU
gate.
"""

import pytest
import torch

try:
    from monitoring.ring_transport import (
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_SCORES,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_POS_EMBED,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_FINAL_LOGITS,
        ModelShapeConfig,
        _compute_hook_shape,
    )
    _NATIVE_IMPORT_ERROR = None
except ImportError as exc:
    (
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_SCORES,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_POS_EMBED,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_FINAL_LOGITS,
        ModelShapeConfig,
        _compute_hook_shape,
    ) = (None,) * 21
    _NATIVE_IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.native_backend,
    pytest.mark.skipif(
        _NATIVE_IMPORT_ERROR is not None,
        reason=f"DMI native backend required: {_NATIVE_IMPORT_ERROR}",
    ),
]


def _cfg(tp_size=1):
    """Create a Qwen3-0.6B-like config with given TP degree.
    Note: Qwen3 has head_dim=128, so num_heads * head_dim = 2048 != hidden_dim=1024.
    """
    return ModelShapeConfig(
        hidden_dim=1024,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,  # Qwen3 uses explicit head_dim=128
        dtype=torch.bfloat16,
        vocab_size=151936,
        intermediate_dim=2816,
        tp_size=tp_size,
    )


# --------------------------------------------------------------------------
# tp_size=1: shapes should match pre-TP behavior (regression test)
# --------------------------------------------------------------------------

class TestTP1Regression:
    """With tp_size=1, shapes should be identical to pre-TP behavior."""

    def test_q_shape_tp1(self):
        # vLLM flattened
        assert _compute_hook_shape(HOOK_TYPE_Q, _cfg(1), 0, 10, 0) == [10, 16, 128]
        # HF batched
        assert _compute_hook_shape(HOOK_TYPE_Q, _cfg(1), 4, 10, 0) == [4, 10, 16, 128]

    def test_k_shape_tp1(self):
        assert _compute_hook_shape(HOOK_TYPE_K, _cfg(1), 0, 10, 0) == [10, 8, 128]

    def test_v_shape_tp1(self):
        assert _compute_hook_shape(HOOK_TYPE_V, _cfg(1), 0, 10, 0) == [10, 8, 128]

    def test_z_shape_tp1_vllm(self):
        # num_heads * head_dim = 16 * 128 = 2048 (not hidden_dim=1024)
        assert _compute_hook_shape(HOOK_TYPE_Z, _cfg(1), 0, 10, 0) == [10, 2048]

    def test_z_shape_tp1_hf(self):
        assert _compute_hook_shape(HOOK_TYPE_Z, _cfg(1), 4, 10, 0) == [4, 10, 16, 128]

    def test_hidden_dim_hooks_tp1(self):
        for ht in [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_ATTN_OUT,
                    HOOK_TYPE_RESID_MID, HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN,
                    HOOK_TYPE_MLP_OUT, HOOK_TYPE_RESID_FINAL, HOOK_TYPE_EMBED,
                    HOOK_TYPE_POS_EMBED, HOOK_TYPE_FINAL_LN]:
            assert _compute_hook_shape(ht, _cfg(1), 0, 10, 0) == [10, 1024], f"hook_type={ht}"

    def test_mlp_post_tp1(self):
        assert _compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(1), 0, 10, 0) == [10, 2816]

    def test_attn_scores_tp1(self):
        assert _compute_hook_shape(HOOK_TYPE_ATTN_SCORES, _cfg(1), 0, 10, 256) == [16, 10, 256]

    def test_token_ids_tp1(self):
        assert _compute_hook_shape(HOOK_TYPE_TOKEN_IDS, _cfg(1), 0, 10, 0) == [10]


# --------------------------------------------------------------------------
# tp_size=2: sharded hooks should halve, unsharded should stay full
# --------------------------------------------------------------------------

class TestTP2Sharding:
    """With tp_size=2, only Q/K/V/Z/mlp_post/attn_scores should change."""

    def test_q_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_Q, _cfg(2), 0, 10, 0) == [10, 8, 128]

    def test_k_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_K, _cfg(2), 0, 10, 0) == [10, 4, 128]

    def test_v_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_V, _cfg(2), 0, 10, 0) == [10, 4, 128]

    def test_z_halved_vllm(self):
        # (16 // 2) * 128 = 1024
        assert _compute_hook_shape(HOOK_TYPE_Z, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_z_halved_hf(self):
        assert _compute_hook_shape(HOOK_TYPE_Z, _cfg(2), 4, 10, 0) == [4, 10, 8, 128]

    def test_mlp_post_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(2), 0, 10, 0) == [10, 1408]

    def test_attn_scores_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_ATTN_SCORES, _cfg(2), 0, 10, 256) == [8, 10, 256]

    def test_attn_out_NOT_halved(self):
        """attn_out is after RowParallel all-reduce — full hidden_dim."""
        assert _compute_hook_shape(HOOK_TYPE_ATTN_OUT, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_mlp_out_NOT_halved(self):
        """mlp_out is after RowParallel all-reduce — full hidden_dim."""
        assert _compute_hook_shape(HOOK_TYPE_MLP_OUT, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_final_logits_NOT_halved(self):
        """final_logits is gathered — full vocab_size."""
        shape = _compute_hook_shape(HOOK_TYPE_FINAL_LOGITS, _cfg(2), 0, 10, 0, logits_to_keep=10)
        assert shape[-1] == 151936  # full vocab

    def test_residual_hooks_NOT_halved(self):
        """All residual/LN hooks are full hidden_dim regardless of TP."""
        for ht in [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_RESID_MID,
                    HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN, HOOK_TYPE_RESID_FINAL,
                    HOOK_TYPE_EMBED, HOOK_TYPE_FINAL_LN]:
            assert _compute_hook_shape(ht, _cfg(2), 0, 10, 0) == [10, 1024], f"hook_type={ht}"

    def test_token_ids_NOT_halved(self):
        assert _compute_hook_shape(HOOK_TYPE_TOKEN_IDS, _cfg(2), 0, 10, 0) == [10]


# --------------------------------------------------------------------------
# tp_size=4: verify further division
# --------------------------------------------------------------------------

class TestTP4:
    def test_q_quartered(self):
        assert _compute_hook_shape(HOOK_TYPE_Q, _cfg(4), 0, 10, 0) == [10, 4, 128]

    def test_k_quartered(self):
        assert _compute_hook_shape(HOOK_TYPE_K, _cfg(4), 0, 10, 0) == [10, 2, 128]

    def test_z_quartered_vllm(self):
        # (16 // 4) * 128 = 512
        assert _compute_hook_shape(HOOK_TYPE_Z, _cfg(4), 0, 10, 0) == [10, 512]

    def test_mlp_post_quartered(self):
        assert _compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(4), 0, 10, 0) == [10, 704]


# --------------------------------------------------------------------------
# GQA edge case: tp_size > num_kv_heads
# --------------------------------------------------------------------------

class TestGQAEdgeCase:
    """When tp_size > num_kv_heads, KV heads clamp to max(1, ...)."""

    def test_kv_heads_replicated(self):
        """num_kv_heads=8, tp=16 → max(1, 8//16) = max(1, 0) = 1."""
        cfg = _cfg(1)
        cfg.tp_size = 16
        assert _compute_hook_shape(HOOK_TYPE_K, cfg, 0, 10, 0) == [10, 1, 128]
        assert _compute_hook_shape(HOOK_TYPE_V, cfg, 0, 10, 0) == [10, 1, 128]

    def test_q_heads_still_divide(self):
        """num_heads=16, tp=16 → 16//16 = 1 Q head per rank."""
        cfg = _cfg(1)
        cfg.tp_size = 16
        assert _compute_hook_shape(HOOK_TYPE_Q, cfg, 0, 10, 0) == [10, 1, 128]

    def test_kv_exact_division(self):
        """num_kv_heads=8, tp=8 → 8//8 = 1 (not clamped, exact)."""
        cfg = _cfg(1)
        cfg.tp_size = 8
        assert _compute_hook_shape(HOOK_TYPE_K, cfg, 0, 10, 0) == [10, 1, 128]


# --------------------------------------------------------------------------
# Capacity: total reserved bytes should decrease with TP for sharded hooks
# --------------------------------------------------------------------------

class TestCapacityReduction:
    """Pre-reserved bytes for sharded hooks should scale inversely with TP."""

    def _bytes_for_hook(self, hook_type, tp_size, q_len=10):
        cfg = _cfg(tp_size)
        shape = _compute_hook_shape(hook_type, cfg, 0, q_len, 0)
        if not shape:
            return 0
        import math
        return math.prod(shape) * 2  # bfloat16 = 2 bytes

    def test_q_bytes_halved(self):
        b1 = self._bytes_for_hook(HOOK_TYPE_Q, 1)
        b2 = self._bytes_for_hook(HOOK_TYPE_Q, 2)
        assert b2 == b1 // 2

    def test_mlp_post_bytes_halved(self):
        b1 = self._bytes_for_hook(HOOK_TYPE_MLP_POST, 1)
        b2 = self._bytes_for_hook(HOOK_TYPE_MLP_POST, 2)
        assert b2 == b1 // 2

    def test_attn_out_bytes_unchanged(self):
        b1 = self._bytes_for_hook(HOOK_TYPE_ATTN_OUT, 1)
        b2 = self._bytes_for_hook(HOOK_TYPE_ATTN_OUT, 2)
        assert b2 == b1  # not sharded
