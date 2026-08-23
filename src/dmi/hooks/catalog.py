"""Framework-neutral catalog of DMI hook definitions.

The numeric values are part of the native ABI and mirror ``HOOK_DEFS`` in
``native/csrc/ring/tensor_meta.h``. Keeping the catalog in pure Python
allows shape planning and adapter imports without loading the CUDA extension.
Native-enabled tests validate this table against the compiled definition.
"""

from __future__ import annotations

# Hook groups.
GROUP_ATTN, GROUP_MLP, GROUP_OTHER = 0, 1, 2

# Shape classes.
SHAPE_HIDDEN, SHAPE_QKV_Q, SHAPE_QKV_KV, SHAPE_QKV_Z = 0, 1, 2, 3
SHAPE_ATTN_WT, SHAPE_MLP_POST, SHAPE_TOKEN_IDS, SHAPE_LOGITS = 4, 5, 6, 7
SHAPE_ROUTER_LOGITS, SHAPE_TOPK_IDS, SHAPE_TOPK_WEIGHTS = 8, 9, 10

# Pipeline-parallel placement.
PP_ANY, PP_FIRST, PP_LAST = 0, 1, 2

# (id, act_name, short_name, per_layer, group, tp_sharded, shape_class,
#  pp_stage)
HOOK_DEFS: tuple[tuple[int, str, str, bool, int, bool, int, int], ...] = (
    (0, "hook_resid_pre", "resid_pre", True, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
    (1, "hook_ln1", "ln1", True, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
    (2, "hook_attn_out", "attn_out", True, GROUP_ATTN, False, SHAPE_HIDDEN, PP_ANY),
    (3, "hook_resid_mid", "resid_mid", True, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
    (4, "attn.hook_attn_scores", "attn_scores", True, GROUP_ATTN, True, SHAPE_ATTN_WT, PP_ANY),
    (5, "attn.hook_pattern", "pattern", True, GROUP_ATTN, True, SHAPE_ATTN_WT, PP_ANY),
    (6, "attn.hook_q", "q", True, GROUP_ATTN, True, SHAPE_QKV_Q, PP_ANY),
    (7, "attn.hook_k", "k", True, GROUP_ATTN, True, SHAPE_QKV_KV, PP_ANY),
    (8, "attn.hook_v", "v", True, GROUP_ATTN, True, SHAPE_QKV_KV, PP_ANY),
    (9, "attn.hook_z", "z", True, GROUP_ATTN, True, SHAPE_QKV_Z, PP_ANY),
    (11, "hook_ln2", "ln2", True, GROUP_OTHER, False, SHAPE_HIDDEN, PP_ANY),
    (12, "hook_mlp_in", "mlp_in", True, GROUP_MLP, False, SHAPE_HIDDEN, PP_ANY),
    (13, "hook_mlp_out", "mlp_out", True, GROUP_MLP, False, SHAPE_HIDDEN, PP_ANY),
    (20, "hook_mlp_post", "mlp_post", True, GROUP_MLP, True, SHAPE_MLP_POST, PP_ANY),
    (14, "hook_resid_final", "resid_final", False, GROUP_OTHER, False, SHAPE_HIDDEN, PP_LAST),
    (15, "hook_embed", "embed", False, GROUP_OTHER, False, SHAPE_HIDDEN, PP_FIRST),
    (16, "hook_pos_embed", "pos_embed", False, GROUP_OTHER, False, SHAPE_HIDDEN, PP_FIRST),
    (17, "hook_final_ln", "final_ln", False, GROUP_OTHER, False, SHAPE_HIDDEN, PP_LAST),
    (18, "token_ids", "token_ids", False, GROUP_OTHER, False, SHAPE_TOKEN_IDS, PP_FIRST),
    (19, "final_logits", "final_logits", False, GROUP_OTHER, False, SHAPE_LOGITS, PP_LAST),
    (21, "mlp.hook_router_logits", "router_logits", True, GROUP_OTHER, False, SHAPE_ROUTER_LOGITS, PP_ANY),
    (22, "mlp.hook_topk_ids", "topk_ids", True, GROUP_OTHER, False, SHAPE_TOPK_IDS, PP_ANY),
    (23, "mlp.hook_topk_weights", "topk_weights", True, GROUP_OTHER, False, SHAPE_TOPK_WEIGHTS, PP_ANY),
)


__all__ = [
    "HOOK_DEFS",
    "GROUP_ATTN",
    "GROUP_MLP",
    "GROUP_OTHER",
    "SHAPE_HIDDEN",
    "SHAPE_QKV_Q",
    "SHAPE_QKV_KV",
    "SHAPE_QKV_Z",
    "SHAPE_ATTN_WT",
    "SHAPE_MLP_POST",
    "SHAPE_TOKEN_IDS",
    "SHAPE_LOGITS",
    "SHAPE_ROUTER_LOGITS",
    "SHAPE_TOPK_IDS",
    "SHAPE_TOPK_WEIGHTS",
    "PP_ANY",
    "PP_FIRST",
    "PP_LAST",
]
