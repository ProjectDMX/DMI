"""Unit tests for tensor-parallel hook shape computation."""

from dataclasses import replace

import pytest
import torch

from dmi.hooks.specs import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_ATTN_SCORES,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_POS_EMBED,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TO_SHORT_NAME,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    ModelShapeConfig,
    compute_hook_shape,
)

pytestmark = pytest.mark.cpu


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
        assert compute_hook_shape(HOOK_TYPE_Q, _cfg(1), 0, 10, 0) == [10, 16, 128]
        # HF batched
        assert compute_hook_shape(HOOK_TYPE_Q, _cfg(1), 4, 10, 0) == [4, 10, 16, 128]

    def test_k_shape_tp1(self):
        assert compute_hook_shape(HOOK_TYPE_K, _cfg(1), 0, 10, 0) == [10, 8, 128]

    def test_v_shape_tp1(self):
        assert compute_hook_shape(HOOK_TYPE_V, _cfg(1), 0, 10, 0) == [10, 8, 128]

    def test_z_shape_tp1_vllm(self):
        # num_heads * head_dim = 16 * 128 = 2048 (not hidden_dim=1024)
        assert compute_hook_shape(HOOK_TYPE_Z, _cfg(1), 0, 10, 0) == [10, 2048]

    def test_z_shape_tp1_hf(self):
        assert compute_hook_shape(HOOK_TYPE_Z, _cfg(1), 4, 10, 0) == [4, 10, 16, 128]

    def test_hidden_dim_hooks_tp1(self):
        for ht in [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_ATTN_OUT,
                    HOOK_TYPE_RESID_MID, HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN,
                    HOOK_TYPE_MLP_OUT, HOOK_TYPE_RESID_FINAL, HOOK_TYPE_EMBED,
                    HOOK_TYPE_POS_EMBED, HOOK_TYPE_FINAL_LN]:
            assert compute_hook_shape(ht, _cfg(1), 0, 10, 0) == [10, 1024], f"hook_type={ht}"

    def test_mlp_post_tp1(self):
        assert compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(1), 0, 10, 0) == [10, 2816]

    def test_attn_scores_tp1(self):
        assert compute_hook_shape(HOOK_TYPE_ATTN_SCORES, _cfg(1), 0, 10, 256) == [16, 10, 256]

    def test_token_ids_tp1(self):
        assert compute_hook_shape(HOOK_TYPE_TOKEN_IDS, _cfg(1), 0, 10, 0) == [10]


# --------------------------------------------------------------------------
# tp_size=2: sharded hooks should halve, unsharded should stay full
# --------------------------------------------------------------------------

class TestTP2Sharding:
    """With tp_size=2, only Q/K/V/Z/mlp_post/attn_scores should change."""

    def test_q_halved(self):
        assert compute_hook_shape(HOOK_TYPE_Q, _cfg(2), 0, 10, 0) == [10, 8, 128]

    def test_k_halved(self):
        assert compute_hook_shape(HOOK_TYPE_K, _cfg(2), 0, 10, 0) == [10, 4, 128]

    def test_v_halved(self):
        assert compute_hook_shape(HOOK_TYPE_V, _cfg(2), 0, 10, 0) == [10, 4, 128]

    def test_z_halved_vllm(self):
        # (16 // 2) * 128 = 1024
        assert compute_hook_shape(HOOK_TYPE_Z, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_z_halved_hf(self):
        assert compute_hook_shape(HOOK_TYPE_Z, _cfg(2), 4, 10, 0) == [4, 10, 8, 128]

    def test_mlp_post_halved(self):
        assert compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(2), 0, 10, 0) == [10, 1408]

    def test_attn_scores_halved(self):
        assert compute_hook_shape(HOOK_TYPE_ATTN_SCORES, _cfg(2), 0, 10, 256) == [8, 10, 256]

    def test_attn_out_NOT_halved(self):
        """attn_out is after RowParallel all-reduce — full hidden_dim."""
        assert compute_hook_shape(HOOK_TYPE_ATTN_OUT, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_mlp_out_NOT_halved(self):
        """mlp_out is after RowParallel all-reduce — full hidden_dim."""
        assert compute_hook_shape(HOOK_TYPE_MLP_OUT, _cfg(2), 0, 10, 0) == [10, 1024]

    def test_final_logits_NOT_halved(self):
        """final_logits is gathered — full vocab_size."""
        shape = compute_hook_shape(HOOK_TYPE_FINAL_LOGITS, _cfg(2), 0, 10, 0, logits_to_keep=10)
        assert shape[-1] == 151936  # full vocab

    def test_residual_hooks_NOT_halved(self):
        """All residual/LN hooks are full hidden_dim regardless of TP."""
        for ht in [HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_RESID_MID,
                    HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN, HOOK_TYPE_RESID_FINAL,
                    HOOK_TYPE_EMBED, HOOK_TYPE_FINAL_LN]:
            assert compute_hook_shape(ht, _cfg(2), 0, 10, 0) == [10, 1024], f"hook_type={ht}"

    def test_token_ids_NOT_halved(self):
        assert compute_hook_shape(HOOK_TYPE_TOKEN_IDS, _cfg(2), 0, 10, 0) == [10]


# --------------------------------------------------------------------------
# tp_size=4: verify further division
# --------------------------------------------------------------------------

class TestTP4:
    def test_q_quartered(self):
        assert compute_hook_shape(HOOK_TYPE_Q, _cfg(4), 0, 10, 0) == [10, 4, 128]

    def test_k_quartered(self):
        assert compute_hook_shape(HOOK_TYPE_K, _cfg(4), 0, 10, 0) == [10, 2, 128]

    def test_z_quartered_vllm(self):
        # (16 // 4) * 128 = 512
        assert compute_hook_shape(HOOK_TYPE_Z, _cfg(4), 0, 10, 0) == [10, 512]

    def test_mlp_post_quartered(self):
        assert compute_hook_shape(HOOK_TYPE_MLP_POST, _cfg(4), 0, 10, 0) == [10, 704]


# --------------------------------------------------------------------------
# GQA edge case: tp_size > num_kv_heads
# --------------------------------------------------------------------------

class TestGQAEdgeCase:
    """When tp_size > num_kv_heads, KV heads clamp to max(1, ...)."""

    def test_kv_heads_replicated(self):
        """num_kv_heads=8, tp=16 → max(1, 8//16) = max(1, 0) = 1."""
        cfg = _cfg(1)
        cfg.tp_size = 16
        assert compute_hook_shape(HOOK_TYPE_K, cfg, 0, 10, 0) == [10, 1, 128]
        assert compute_hook_shape(HOOK_TYPE_V, cfg, 0, 10, 0) == [10, 1, 128]

    def test_q_heads_still_divide(self):
        """num_heads=16, tp=16 → 16//16 = 1 Q head per rank."""
        cfg = _cfg(1)
        cfg.tp_size = 16
        assert compute_hook_shape(HOOK_TYPE_Q, cfg, 0, 10, 0) == [10, 1, 128]

    def test_kv_exact_division(self):
        """num_kv_heads=8, tp=8 → 8//8 = 1 (not clamped, exact)."""
        cfg = _cfg(1)
        cfg.tp_size = 8
        assert compute_hook_shape(HOOK_TYPE_K, cfg, 0, 10, 0) == [10, 1, 128]


# --------------------------------------------------------------------------
# Capacity: total reserved bytes should decrease with TP for sharded hooks
# --------------------------------------------------------------------------

class TestCapacityReduction:
    """Pre-reserved bytes for sharded hooks should scale inversely with TP."""

    def _bytes_for_hook(self, hook_type, tp_size, q_len=10):
        cfg = _cfg(tp_size)
        shape = compute_hook_shape(hook_type, cfg, 0, q_len, 0)
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


# --------------------------------------------------------------------------
# final_logits row count and the zero-config degenerate returns
#
# The batched (batch > 0) final_logits branch caps logits_to_keep at q_len,
# while the packed branch (batch == 0) treats logits_to_keep as the request
# count and never clamps it against q_len.  The "unknown dimension" configs
# (vocab_size / intermediate_dim / num_experts / top_k == 0) must yield an
# empty shape so the caller skips pushing meta for that hook.
# --------------------------------------------------------------------------

_VOCAB = _cfg(1).vocab_size  # 151936 for the Qwen3-0.6B-like config above


class TestFinalLogitsRows:
    """Row count of the final_logits shape in both batch conventions."""

    def test_final_logits_batched_caps_logits_to_keep_at_q_len(self):
        """logits_to_keep > q_len must clamp to q_len, not overrun the step."""
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, _cfg(1), 2, 1, 0, logits_to_keep=4
        ) == [2, 1, _VOCAB]

    def test_final_logits_batched_below_q_len_and_zero(self):
        """Below the cap logits_to_keep wins; 0 means "all q_len rows"."""
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, _cfg(1), 2, 8, 0, logits_to_keep=1
        ) == [2, 1, _VOCAB]
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, _cfg(1), 2, 8, 0, logits_to_keep=0
        ) == [2, 8, _VOCAB]

    def test_final_logits_packed_uses_request_count(self):
        """Packed layout: dim 0 is the request count, with no batch dim."""
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, _cfg(1), 0, 10, 0, logits_to_keep=3
        ) == [3, _VOCAB]
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, _cfg(1), 0, 10, 0, logits_to_keep=0
        ) == [10, _VOCAB]

    def test_final_logits_empty_without_vocab_size(self):
        """vocab_size unknown -> empty shape in both batch conventions."""
        cfg = replace(_cfg(1), vocab_size=0)
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, cfg, 0, 10, 0, logits_to_keep=3
        ) == []
        assert compute_hook_shape(
            HOOK_TYPE_FINAL_LOGITS, cfg, 2, 4, 0, logits_to_keep=1
        ) == []


class TestUnknownDimensionsGiveEmptyShape:
    """Hooks whose trailing dim is not in the config are skipped."""

    def test_mlp_post_empty_when_intermediate_dim_unknown(self):
        cfg = replace(_cfg(1), intermediate_dim=0)
        assert compute_hook_shape(HOOK_TYPE_MLP_POST, cfg, 0, 10, 0) == []

    def test_routing_shapes_empty_when_moe_config_missing(self):
        moe = replace(_cfg(1), num_experts=8, top_k=2)
        # Positive batched cases (plain-CPU coverage; the MoE routing-hook
        # suite that also covers these needs the Transformers fork).
        assert compute_hook_shape(HOOK_TYPE_ROUTER_LOGITS, moe, 2, 5, 0) == [2, 5, 8]
        assert compute_hook_shape(HOOK_TYPE_TOPK_IDS, moe, 2, 5, 0) == [2, 5, 2]
        assert compute_hook_shape(HOOK_TYPE_TOPK_WEIGHTS, moe, 2, 5, 0) == [2, 5, 2]
        # Degenerate configs.
        no_experts = replace(moe, num_experts=0)
        assert compute_hook_shape(HOOK_TYPE_ROUTER_LOGITS, no_experts, 2, 5, 0) == []
        no_top_k = replace(moe, top_k=0)
        assert compute_hook_shape(HOOK_TYPE_TOPK_IDS, no_top_k, 2, 5, 0) == []
        assert compute_hook_shape(HOOK_TYPE_TOPK_WEIGHTS, no_top_k, 2, 5, 0) == []

    def test_unknown_hook_type_returns_empty_shape(self):
        """An unregistered hook type falls through to the empty shape."""
        unknown = max(HOOK_TYPE_TO_SHORT_NAME) + 1
        assert compute_hook_shape(unknown, _cfg(1), 0, 4, 0) == []
