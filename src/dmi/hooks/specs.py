"""Hook specifications, shape planning, and catalog-derived metadata.

This module is framework- and transport-neutral. It defines the Python view of
DMI's native hook ABI and the analytical tensor shapes used by adapters and
transports.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .catalog import (
    GROUP_ATTN,
    GROUP_MLP,
    GROUP_OTHER,
    HOOK_DEFS,
    PP_ANY,
    PP_FIRST,
    PP_LAST,
    SHAPE_ATTN_WT,
    SHAPE_HIDDEN,
    SHAPE_LOGITS,
    SHAPE_MLP_POST,
    SHAPE_QKV_KV,
    SHAPE_QKV_Q,
    SHAPE_QKV_Z,
    SHAPE_TOKEN_IDS,
)

# ---------------------------------------------------------------------------
# Hook-type constants (values match C++ HookType enum in tensor_meta.h)
#
# Removed hook types (gaps in numbering are intentional):
#   10 (result):     removed because attn_out captures the same tensor.
#                    o_proj/c_proj output IS the attention block return value
#                    in all known architectures.  Use ATTN_OUT instead.
#   resid_post:      removed (was per-layer).  Replaced by RESID_FINAL (global).
#                    resid_post[i] == resid_pre[i+1] for all i < N-1, so
#                    per-layer capture was N-1 redundant D2D copies.
#                    RESID_FINAL captures the only unique value: last layer's
#                    residual stream before final norm.
#
# Duplicate hook types kept intentionally:
#   LN2 vs MLP_IN:  identical for dense models (norm output goes directly to
#                    MLP).  Differs for MoE models where a router sits between
#                    norm and expert MLP (MLP_IN is post-router, EP-sharded).
#
# TODO: per-model deduplication.  Some hook pairs (e.g. ln2/mlp_in in dense
# models) produce identical tensors.  A model-specific selection system could
# alias them so the same preset skips duplicates on dense models but captures
# both on MoE.  For now, both are always captured when selected.
# ---------------------------------------------------------------------------

# Hook definitions are kept in a pure-Python ABI catalog so importing this
# module does not require a compiled extension. Native-enabled tests compare it
# with the C++ table in ``tensor_meta.h``.
_HOOK_DEFS = HOOK_DEFS


class HookRowBasis(Enum):
    """Per-step cardinality that scales a registered hook's payload.

    ``TOKEN_ROWS`` means the shape scales with ``q_len``. In packed execution,
    padding-strip eligibility separately decides whether the adapter uses the
    actual token count or the padded execution count. ``REQUEST_ROWS`` means
    the packed shape scales with the request count supplied through
    ``logits_to_keep``. This enum does not describe tensor dimension order,
    rank ownership, or prefix-strip eligibility.
    """

    TOKEN_ROWS = auto()
    REQUEST_ROWS = auto()


# Auto-derive all mappings
_id_by_short: Dict[str, int] = {}       # "q" -> 6
_shape_class_by_type: Dict[int, int] = {}
for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS:
    _id_by_short[_short] = _id
    _shape_class_by_type[_id] = _sc
    # Inject HOOK_TYPE_Q, HOOK_TYPE_RESID_PRE, etc. into module namespace
    globals()[f"HOOK_TYPE_{_short.upper()}"] = _id

# Auto-derive act_name suffix sets per group (re-exported for tooling).
_ATTN_SUFFIXES: Tuple[str, ...] = tuple(
    _act for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _grp == GROUP_ATTN
)
_MLP_SUFFIXES: Tuple[str, ...] = tuple(
    _act for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _grp == GROUP_MLP
)

# Auto-derive property sets from HOOK_DEFS columns.
TP_SHARDED_TYPES: frozenset = frozenset(
    _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _tp
)
_HIDDEN_DIM_TYPES: frozenset = frozenset(
    _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _sc == SHAPE_HIDDEN
)
_ATTN_WT_TYPES: frozenset = frozenset(
    _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _sc == SHAPE_ATTN_WT
)
PP_FIRST_ONLY: frozenset = frozenset(
    _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _pp == PP_FIRST
)
PP_LAST_ONLY: frozenset = frozenset(
    _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS if _pp == PP_LAST
)

def hook_row_basis(hook_type: int) -> HookRowBasis:
    """Return the canonical row basis derived from `_HOOK_DEFS.shape_class`.

    The package hook catalog is the Python mapping source. Logit-shaped payloads
    are request-scaled; every other registered shape class is token-scaled. An
    unregistered hook type is a configuration error.
    """

    try:
        shape_class = _shape_class_by_type[hook_type]
    except KeyError as exc:
        raise ValueError(f"Unknown hook type: {hook_type!r}") from exc
    if shape_class == SHAPE_LOGITS:
        return HookRowBasis.REQUEST_ROWS
    return HookRowBasis.TOKEN_ROWS


# Hook selection (presets, resolution, and PP/TP filters) lives in
# ``dmi.hooks.selection`` and consumes the catalog-derived constants above.

# ---------------------------------------------------------------------------
# Two batch conventions used throughout this file
# ---------------------------------------------------------------------------
# - "batched" (batch > 0): tensors carry a leading batch dim; shapes are
#   [batch, q_len, ...].  This is what HF generate() produces.
# - "packed/flattened" (batch == 0): no leading batch dim; rows from every
#   active request are concatenated along dim 0 and q_len = total tokens
#   across requests.  This is what vLLM produces (one tensor per
#   scheduler step, requests cumsum'd into dim 0).
#
# Beyond this attribution block the rest of the file refers to the
# conventions by their neutral names ("batched" / "packed").
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hook-type -> short-name map (shared, derived from HOOK_DEFS).  Used for
# debug labels (logs, NVTX ranges, error messages).  Not part of any
# dispatch path -- kernel hook_type values come from HookSpec, never from
# string parsing.
# ---------------------------------------------------------------------------

HOOK_TYPE_TO_SHORT_NAME: Dict[int, str] = {
    _id: _short
    for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS
}


def align_up_py(x: int, a: int) -> int:
    """Python equivalent of ring::align_up (a must be a power of 2)."""
    return (x + a - 1) & ~(a - 1)


# ---------------------------------------------------------------------------
# ModelShapeConfig -- provided at hook-installation time
# ---------------------------------------------------------------------------

@dataclass
class ModelShapeConfig:
    """Describes attention geometry for analytical shape computation."""
    hidden_dim:   int
    num_heads:    int
    num_kv_heads: int   # == num_heads for MHA; < num_heads for GQA
    head_dim:     int
    dtype:        torch.dtype
    vocab_size:   int = 0  # required for final_logits shape
    intermediate_dim: int = 0  # MLP intermediate size (for mlp_post shape)
    num_experts:  int = 0  # router_logits final dim
    top_k:        int = 0  # topk_ids / topk_weights final dim
    tp_size:      int = 1  # tensor parallel world size
    tp_rank:      int = 0  # this rank's TP index


# ---------------------------------------------------------------------------
# HookSpec -- model self-describes its hooks in forward() firing order
# ---------------------------------------------------------------------------

@dataclass
class HookSpec:
    """One monitoring hook: type, layer, shape convention, and module reference."""
    hook_type: int                        # HOOK_TYPE_* -- determines shape formula
    module:    Optional[nn.Module]        # HookPoint, or None for model-wide specs
    layer_no:  int = -1                   # layer index (-1 for global hooks like embed, final_ln)
    dtype:     Optional[torch.dtype] = None  # override model dtype (e.g. int64 for token_ids)
    # True when the producer kernel may write fewer (or more) bytes than the
    # CPU-side shape estimate predicts -- e.g. EP hooks where the token count
    # routed to this rank varies per step.  Propagated to TensorMeta.flags as
    # META_FLAG_ALLOW_MISMATCH; consumer recomputes dim-0 from actual bytes.
    allow_token_cnt_mismatch: bool = False
    # True when this spec's shape has dim-0 = total_tokens in the framework's
    # packed-flat layout, or batch * q_len in the batched layout when q_len is
    # the variable axis.  Adapters that enable a padding-strip mode use this
    # flag to mark prefix-eligible specs.  Static property; ignored when no
    # adapter activates strip.
    dim0_is_actual_tokens: bool = False

def compute_hook_shape(
    hook_type: int,
    cfg: ModelShapeConfig,
    batch: int,
    q_len: int,
    kv_dim: int,
    logits_to_keep: int = 0,
) -> List[int]:
    """Return expected tensor shape for a given hook type and step dimensions.

    See the "two batch conventions" block at the top of this file for
    what ``batch == 0`` (packed/flattened) vs ``batch > 0`` (batched) mean.

    ASSUMPTION: hooked tensors have deterministic shapes given the same
    (batch, q_len, kv_dim, logits_to_keep) and model config.  This is
    guaranteed by the model architecture.

    Args:
        batch: batch size, or ``0`` for the packed/flattened convention.
        logits_to_keep: how many logit rows the model returns per step.
            ``0`` means "all q_len rows".  Frameworks that materialize
            only the last-token logits per request pass
            ``logits_to_keep > 0``.
    """
    # batch=0 means packed/flattened: shapes have no batch dimension.
    b = [batch] if batch > 0 else []

    tp = cfg.tp_size

    if hook_type in _HIDDEN_DIM_TYPES:
        return b + [q_len, cfg.hidden_dim]
    if hook_type == HOOK_TYPE_Q:
        return b + [q_len, cfg.num_heads // tp, cfg.head_dim]
    if hook_type in (HOOK_TYPE_K, HOOK_TYPE_V):
        kv_heads = max(1, cfg.num_kv_heads // tp)  # GQA: may replicate
        return b + [q_len, kv_heads, cfg.head_dim]
    if hook_type == HOOK_TYPE_Z:
        # Packed/flattened convention flattens heads into a single
        # trailing dim -> [q_len, num_heads * head_dim].
        # Batched convention keeps four dims -> [batch, q_len, num_heads, head_dim].
        if batch == 0:
            return [q_len, (cfg.num_heads // tp) * cfg.head_dim]
        return b + [q_len, cfg.num_heads // tp, cfg.head_dim]
    if hook_type in (HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN):
        return b + [cfg.num_heads // tp, q_len, kv_dim]
    if hook_type == HOOK_TYPE_MLP_POST:
        if cfg.intermediate_dim == 0:
            return []  # intermediate_dim unknown -- skip this hook
        return b + [q_len, cfg.intermediate_dim // tp]
    if hook_type == HOOK_TYPE_ROUTER_LOGITS:
        return (b + [q_len, cfg.num_experts]) if cfg.num_experts > 0 else []
    if hook_type == HOOK_TYPE_TOPK_IDS:
        return (b + [q_len, cfg.top_k]) if cfg.top_k > 0 else []
    if hook_type == HOOK_TYPE_TOPK_WEIGHTS:
        return (b + [q_len, cfg.top_k]) if cfg.top_k > 0 else []
    if hook_type == HOOK_TYPE_TOKEN_IDS:
        return b + [q_len]
    if hook_type == HOOK_TYPE_FINAL_LOGITS:
        # compute_logits returns fewer rows than q_len when the framework
        # only materializes the last-token logits per request.
        #
        # Batched (batch > 0): tensor is [batch, logits_to_keep, vocab].
        #   logits_to_keep is capped at q_len (defaults to q_len when 0).
        #
        # Packed/flattened (batch == 0): tensor is [num_reqs, vocab]
        #   (one logit per request).  Caller passes
        #   logits_to_keep=num_reqs so the meta shape becomes
        #   [num_reqs, vocab].  The p2p thread indexes by request
        #   position (not token offset) and adjusts the DB token range
        #   to (end_token-1, end_token).
        if batch > 0:
            logits_q = min(q_len, logits_to_keep) if logits_to_keep > 0 else q_len
        else:
            logits_q = logits_to_keep if logits_to_keep > 0 else q_len
        return (b + [logits_q, cfg.vocab_size]) if cfg.vocab_size > 0 else []
    return []  # unknown type -- push_meta skipped


# Private compatibility name used by integration API v1.
_compute_hook_shape = compute_hook_shape

_HOOK_TYPE_EXPORTS = tuple(
    f"HOOK_TYPE_{short.upper()}"
    for _id, _act, short, _per_layer, _group, _tp, _shape, _pp in HOOK_DEFS
)

__all__ = [
    "HookRowBasis",
    "HookSpec",
    "ModelShapeConfig",
    "HOOK_TYPE_TO_SHORT_NAME",
    "PP_FIRST_ONLY",
    "PP_LAST_ONLY",
    "TP_SHARDED_TYPES",
    "align_up_py",
    "compute_hook_shape",
    "hook_row_basis",
    *_HOOK_TYPE_EXPORTS,
]
