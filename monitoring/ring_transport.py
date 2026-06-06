"""Ring-based GPU-to-CPU tensor transport for monitoring.

Uses the ring producer/drain pipeline for GPU-to-CPU tensor transport.  Tensor metadata is pushed to the C++ TensorMetaFifo
(via push_meta) before the producer kernel is launched, so the C++ callback
thread can reconstruct and slice the tensor without ever touching Python or
the GIL.

New CUDA-graph-compatible path (activated when model_shape + get_hook_specs are available):
  - ring_producer_op: torch.library.custom_op wrapping ring_engine.hook()
  - register_forward_hook on HookPoint modules (PyTorch-native dispatch)
  - ModelShapeConfig + analytical shape computation (no warmup needed)
  - pre_push_all_metas called before orig_forward, outside compiled region

All transport now uses the CUDA-graph-compatible forward-hook path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.library
from torch import nn


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

# ---------------------------------------------------------------------------
# Hook type constants -- single source of truth is HOOK_DEFS in tensor_meta.h.
# All mappings are auto-derived from the C++ table at import time.
# To add a new hook: add one enum value + one HOOK_DEFS row in C++. Done.
# ---------------------------------------------------------------------------
from ._native_engine import _load_extension as _load_ext
_ext = _load_ext()
# (id, act_name, short_name, per_layer, group, tp_sharded, shape_class, pp_stage)
# group/shape_class/pp_stage are int enums matching the C++ definitions.
_HOOK_DEFS = _ext.HOOK_DEFS

# C++ enum mirrors -- keep in sync with tensor_meta.h
GROUP_ATTN, GROUP_MLP, GROUP_OTHER = 0, 1, 2
SHAPE_HIDDEN, SHAPE_QKV_Q, SHAPE_QKV_KV, SHAPE_QKV_Z = 0, 1, 2, 3
SHAPE_ATTN_WT, SHAPE_MLP_POST, SHAPE_TOKEN_IDS, SHAPE_LOGITS = 4, 5, 6, 7
PP_ANY, PP_FIRST, PP_LAST = 0, 1, 2

# Auto-derive all mappings
_id_by_short: Dict[str, int] = {}       # "q" -> 6
for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS:
    _id_by_short[_short] = _id
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

del _ext, _load_ext

# Hook selection (presets, resolve/apply, PP/TP filters) lives in
# monitoring/selection.py -- that module imports the C++-mirror constants
# above.  See the unified-adaptor refactor plan Sec.6 for rationale.

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
    module:    nn.Module                  # the HookPoint instance
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


# ---------------------------------------------------------------------------
# Module-level active transport
# ---------------------------------------------------------------------------

_active_transport: Optional["RingTransport"] = None


# ---------------------------------------------------------------------------
# register_fake for ring::producer C++ op
#
# ring::producer is registered via C++ TORCH_LIBRARY (ring_torch_op.cpp) with
# schema  Tensor(a!) -> Tensor(a!).  The fake impl is required for torch.compile
# shape propagation.  We register it after ensuring the .so is loaded.
# ---------------------------------------------------------------------------
try:
    from . import _native_engine as _ne
    _ne._load_extension()  # ensure .so is loaded -> registers ring::producer

    # Three fake impls, one per op.  Void schema; pure side-effect.
    # `ring_payload` is the shared `Tensor(a!)` mutation alias -- a view
    # of the engine's GPU payload buffer.  AOT autograd tracks the
    # mutation; successive producer calls form a real R/W chain through
    # this shared tensor, which prevents inductor from DCE-ing the op
    # AND from reordering successive producer launches relative to one
    # another.  No `_register_effectful_op` needed -- the alias is a
    # stronger guarantee than the effect-token hint.
    @torch.library.register_fake("ring::producer")
    def _ring_producer_fake(
        ring_payload: torch.Tensor, tensor: torch.Tensor,
        hook_type: int, hook_id: int,
    ) -> None:
        return None

    @torch.library.register_fake("ring::producer_prefix")
    def _ring_producer_prefix_fake(
        ring_payload: torch.Tensor, tensor: torch.Tensor,
        row_count: torch.Tensor, row_bytes: int,
        hook_type: int, hook_id: int,
    ) -> None:
        return None

    @torch.library.register_fake("ring::producer_chunked")
    def _ring_producer_chunked_fake(
        ring_payload: torch.Tensor, tensor: torch.Tensor,
        chunk_bytes: torch.Tensor,
        hook_type: int, hook_id: int,
    ) -> None:
        return None

    del _ne
except Exception:
    pass


# ---------------------------------------------------------------------------
# kv_dim computation -- cache-type-aware, called before each forward
# ---------------------------------------------------------------------------

def _get_kv_dim(past_key_values: Any, q_len: int, is_static: bool = False) -> int:
    """Return the PHYSICAL key-sequence dimension for shape computation.

    Returns the actual kv_dim that the attention kernel sees, not the logical
    sequence length.  This matters for static/sliding/hybrid caches where
    kv_dim = max_cache_len (fixed pre-allocated buffer), not the current
    token position.

    ASSUMPTION: hooked attention tensors (attn_scores, pattern) have shape
    [batch, heads, q_len, kv_dim] where kv_dim equals the physical cache
    dimension.  This is deterministic given the same input size and cache
    config -- required for correct FIFO metadata matching.

    Args:
        past_key_values: cache object (StaticCache, DynamicCache, or None)
        q_len: query sequence length for this forward step
        is_static: True if cache has fixed physical size (StaticCache,
            SlidingWindowCache, HybridCache).  Caller detects via
            hasattr(past_key_values, 'max_cache_len').
    """
    if past_key_values is None:
        return q_len
    if is_static:
        # Static/sliding/hybrid cache: kv_dim = physical cache size.
        # The attention kernel always sees the full buffer (masked).
        try:
            return int(past_key_values.max_cache_len)
        except Exception:
            pass
    # Dynamic cache: kv_dim = logical length after this step
    try:
        return past_key_values.get_seq_length() + q_len
    except Exception:
        return q_len


# ---------------------------------------------------------------------------
# Analytical shape computation
# ---------------------------------------------------------------------------

def _compute_hook_shape(
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


# ---------------------------------------------------------------------------
# Forward-hook installation
# ---------------------------------------------------------------------------

def install_ring_hooks(specs: List[HookSpec],
                       ring_payload: Optional[torch.Tensor] = None) -> None:
    """Bind HookPoints to ring transport.

    Idempotent: overwrites `_ring_hook_type` / `_ring_hook_id` /
    `_ring_payload` on each HookPoint from its spec + the engine's
    shared payload-view tensor.  Until this runs (and for any HookPoint
    not listed in `specs`), `_ring_hook_type is None` and
    HookPoint.forward() short-circuits without firing the producer.
    """
    for spec in specs:
        hp = spec.module
        hp._ring_hook_type = spec.hook_type
        hp._ring_hook_id = spec.layer_no
        hp._ring_payload = ring_payload


# ---------------------------------------------------------------------------
# RingTransport
# ---------------------------------------------------------------------------

class RingTransport:
    """Manages ring engine + per-step batch context for ring-mode monitoring.

    CUDA-graph-compatible path: install_ring_hooks + pre_push_all_metas.
    Activated when _model_cfg is set and _using_forward_hooks is True.
    """

    def __init__(self, ring_engine: Any) -> None:
        self._ring_engine = ring_engine

        # Cached torch.Tensor view of the engine's GPU payload buffer.
        # Used as the shared `Tensor(a!)` mutation alias passed to every
        # producer op call.  Same physical memory across hooks ->
        # successive producer calls form a real R/W chain in the FX
        # graph, which inductor cannot reorder.  Pinned at engine init;
        # the data_ptr is stable across cudagraph replays.
        self._ring_payload: torch.Tensor = ring_engine.payload_tensor()

        # Current step context -- set before each forward pass
        self._current_model_id: Optional[str] = None
        self._current_tp_rank: int = 0
        self._current_dp_rank: int = 0
        self._current_ep_rank: int = 0
        self._current_pp_rank: int = 0
        self._current_flattened: bool = False
        self._current_req_ids: Optional[List[str]] = None
        self._current_token_ranges: Optional[List[Tuple[int, int]]] = None
        self._current_dim0_offsets: Optional[List[int]] = None
        self._current_kv_offsets: Optional[List[int]] = None

        # When True: meta pushes are skipped so the FIFO stays empty.
        # Producer kernel still fires (for CUDA graph capture) but as no-ops.
        self.null_offload: bool = False

        # When True, HookPoint.forward takes the runtime safety-net branch
        # instead of the fast path:
        #   1. fits in current slack       -> reserve_one + ring
        #   2. fits after flushing the ring -> flush_and_wait + reserve_one + ring
        #   3. single tensor > ring        -> flush_and_wait + submit_cpu_direct
        # Owned by adaptor_base.before_forward (per-batch reassignment based
        # on prepare_step result and dynamic-spec presence).  Dispatch
        # wrappers and HookPoint.forward read only.
        self.force_eager: bool = False

        # New-path state
        self._model_cfg: Optional[ModelShapeConfig] = None
        self._active_specs: List[HookSpec] = []
        self._using_forward_hooks: bool = False

        # Runtime node-toggle gate. When False (default), the meta path is
        # unchanged -- every active spec pushes a meta. When True (set by
        # set_active_hooks), pre_push_all_metas pushes metas only for the
        # enabled-AND-captured subset, in LOCKSTEP with the device-side
        # cudaGraphNodeSetEnabled. The engine's enabled-set (with the #14
        # enabled-AND-captured guard) is the single source of truth.
        self._toggle_gate_active: bool = False
        # Single source of truth for "which specs actually fire this step".
        # When the toggle gate is active, this is the subset of _active_specs
        # whose nodes are enabled-AND-captured; recomputed once per
        # set_active_hooks (NOT per step). None => toggle inactive => all
        # _active_specs fire. capacity-reserve (adaptor_base._compute_step_plan),
        # meta-push AND the device node-enable must all read this same set, else
        # reserve() over-counts vs actual producer writes.
        self._effective_enabled_specs: "Optional[List[HookSpec]]" = None
        self._enabled_version: int = 0   # bumped whenever the enabled set changes
        # True when the active gate uses the LAZY device-apply path
        # (set_active_hooks_lazy). The replay guard reads this to decide whether
        # to ensure_graph_current (lazy) or only validate (eager).
        self._lazy_active: bool = False

        # Hook selection preset name (e.g. "full", "hidden-states", "logits").
        # Set by the active adapter before hook installation.
        self._hook_selection: Optional[str] = None

        # warn_once tracking for Case B fallback
        self._warned_shapes: set = set()

    def set_step_context(
        self,
        model_id: str,
        req_ids: List[str],
        token_ranges: List[Tuple[int, int]],
        dim0_offsets: Optional[List[int]] = None,
        kv_offsets: Optional[List[int]] = None,
        tp_rank: int = 0,
        dp_rank: int = 0,
        ep_rank: int = 0,
        pp_rank: int = 0,
        flattened: bool = False,
    ) -> None:
        """Called before each forward pass to provide per-step batch metadata.

        See the "two batch conventions" block at the top of this file for
        the ``batched`` / ``packed`` terminology.

        dim0_offsets: per-request offset in tensor dim 0.
            Batched: batch index (0, 1, 2, ...).  None = auto-generate range(len(req_ids)).
            Packed: token offset in the packed tensor
                (cumulative sum of scheduled tokens per request).
        kv_offsets: per-request kv-dimension start for attention hooks.
            Dynamic-cache batched: pad_len (real keys at the end, left-padded).
            Static-cache batched / packed: 0 (real keys at the start).
            None = auto-generate zeros.
        flattened: False = batched [batch, q_len, ...], True = packed [total_tokens, ...].
        """
        self._current_model_id = model_id
        self._current_tp_rank = tp_rank
        self._current_dp_rank = dp_rank
        self._current_ep_rank = ep_rank
        self._current_pp_rank = pp_rank
        self._current_flattened = flattened
        self._current_req_ids = req_ids
        self._current_token_ranges = token_ranges
        self._current_dim0_offsets = (
            dim0_offsets if dim0_offsets is not None
            else list(range(len(req_ids)))
        )
        self._current_kv_offsets = (
            kv_offsets if kv_offsets is not None
            else [0] * len(req_ids)
        )

    def set_model_cfg(self, cfg: ModelShapeConfig) -> None:
        """Set the model shape config for analytical shape computation."""
        self._model_cfg = cfg

    @property
    def effective_specs(self) -> "List[HookSpec]":
        """The specs that actually fire this step -- the single set that must
        drive capacity-reserve, meta-push and device node-enable in lockstep.
        Toggle active -> the precomputed enabled-AND-captured subset; else all
        active specs (legacy: every active hook fires every step)."""
        if self._toggle_gate_active and self._effective_enabled_specs is not None:
            return self._effective_enabled_specs
        return self._active_specs

    def pre_push_all_metas(self, batch: int, q_len: int, kv_dim: int,
                           logits_to_keep: int = 0,
                           token_ids_dtype: Optional[torch.dtype] = None,
                           actual_q_len: Optional[int] = None) -> None:
        """Push C++ FIFO metadata for all active specs before orig_forward.

        Called in the same order as install_ring_hooks() so FIFO pop order
        in the drain thread matches ring arrival order.
        Requires _model_cfg to be set via set_model_cfg() or enable_ring_transport().

        When ``actual_q_len`` is set AND a spec has
        ``dim0_is_actual_tokens=True``, the meta's shape uses
        ``actual_q_len`` in place of ``q_len`` -- so the meta describes
        the unpadded data the producer will actually write under
        padding-strip mode.  Other specs and the no-strip case use
        ``q_len`` (today's behavior).
        """
        if self.null_offload:
            return  # kernel launches happen; metas are intentionally skipped
        if self._model_cfg is None or not self._active_specs:
            return
        if self._current_model_id is None:
            return
        if self._current_req_ids is None or self._current_token_ranges is None:
            return
        if self._current_dim0_offsets is None:
            return

        hook_types = []
        layer_nos = []
        shapes = []
        dtypes = []
        flags = []
        # Iterate the single effective-enabled set (precomputed in
        # set_active_hooks). This IS the lockstep node-toggle gate: host meta set
        # == device enabled set == capacity-reserve set. Toggle-inactive ->
        # effective_specs is _active_specs (legacy behaviour, unchanged).
        for spec in self.effective_specs:
            spec_q_len = (actual_q_len if actual_q_len is not None
                          and spec.dim0_is_actual_tokens
                          else q_len)
            shape = _compute_hook_shape(
                spec.hook_type, self._model_cfg, batch, spec_q_len, kv_dim,
                logits_to_keep=logits_to_keep,
            )
            if not shape:
                continue
            if spec.dtype is not None:
                dtype = spec.dtype
            elif spec.hook_type == HOOK_TYPE_TOKEN_IDS and token_ids_dtype is not None:
                dtype = token_ids_dtype
            else:
                dtype = self._model_cfg.dtype
            hook_types.append(spec.hook_type)
            layer_nos.append(spec.layer_no)
            shapes.append(shape)
            dtypes.append(dtype)
            flags.append(1 if spec.allow_token_cnt_mismatch else 0)

        if hook_types:
            self._ring_engine.push_all_metas(
                hook_types, layer_nos, shapes, dtypes, flags,
                self._current_model_id,
                self._current_tp_rank,
                self._current_dp_rank,
                self._current_ep_rank,
                self._current_pp_rank,
                self._current_flattened,
                list(self._current_req_ids),
                list(self._current_token_ranges),
                list(self._current_dim0_offsets),
                list(self._current_kv_offsets) if self._current_kv_offsets else [],
            )

    def set_active_hooks(self, enabled: "Iterable[Tuple[int, int]]") -> None:
        """Set which hooks fire this step, as (hook_type, layer_no) pairs.

        EAGER apply: flips every bound exec NOW via cudaGraphNodeSetEnabled.
        CONTRACT: only safe at a QUIESCENT point (no replay of any bound exec in
        flight) -- e.g. once after warmup for a static config. apply_toggle does
        NOT wait on per-graph replay events, so mutating an exec that the GPU is
        still executing is UB. For DYNAMIC / per-step reconfigure use
        set_active_hooks_lazy (its replay-time ensure_graph_current waits on the
        prior replay event before mutating). It:
          1. set_enabled_hooks + apply_toggle on the engine -> device side
             (cudaGraphNodeSetEnabled on the captured producer nodes), and
          2. activates the host meta gate so pre_push_all_metas pushes metas only
             for the same enabled set.
        Both lanes read the engine's enabled-set (single source of truth).

        Requires the producer nodes to have been registered during capture
        (enable_toggle_capture(True) before capture) and the exec(s) bound via
        bind_graph_exec(). Hooks not captured are gated off (the engine's #14
        guard), so they neither fire nor get a meta.
        """
        eng = self._ring_engine
        # Guard (#1): activating the host gate while the device toggle is a no-op
        # (no exec bound / nothing captured) would filter metas while every
        # producer still fires at its default-enabled state -> desync. Fail loud.
        if eng.bound_graph_count() == 0 or eng.toggle_node_count() == 0:
            raise RuntimeError(
                "set_active_hooks: no producer nodes registered / no graph exec bound "
                "(toggle_node_count=%d, bound_graph_count=%d). Capture with "
                "enable_toggle_capture(True) and call bind_graph_exec() first."
                % (eng.toggle_node_count(), eng.bound_graph_count()))
        # Guard (#4): the gate keys on (hook_type,layer) globally; if captured
        # graphs have different hook sets, a hook in one graph but not the one
        # replayed this step would push a meta with no payload.
        if not eng.toggle_registry_uniform():
            raise RuntimeError(
                "set_active_hooks: captured graphs have non-uniform hook sets; the "
                "global meta gate cannot stay aligned. Ensure every captured graph "
                "registers the same hooks.")
        # Guard (#4): node-registry and exec-binding key sets must match exactly.
        # A partial/mismatched bind (a captured graph left unbound, or a bound
        # graph with no recorded nodes) would let some graph replay with
        # default-ON producers while the meta gate filters -> desync.
        if not eng.toggle_registry_complete():
            raise RuntimeError(
                "set_active_hooks: toggle registry incomplete -- the set of graphs "
                "with recorded producer nodes does not match the set of bound execs. "
                "Every captured graph must be bound (bind_graph_exec) and vice versa.")
        pairs = [(int(ht), int(ln)) for (ht, ln) in enabled]
        eng.set_enabled_hooks(pairs)
        err = eng.apply_toggle()
        if err != 0:
            raise RuntimeError(f"apply_toggle failed with CUDA error {err}")
        self._toggle_gate_active = True
        self._lazy_active = False
        self._recompute_effective_specs()

    def set_active_hooks_lazy(self, enabled: "Iterable[Tuple[int, int]]") -> None:
        """Phase 4 lazy reconfigure: update the enabled set + host meta gate, but
        DEFER the device apply to per-graph ensure_graph_current() (called just
        before each graph replays). set_enabled_hooks bumps the engine's
        target_version, marking every captured graph stale; the device flip then
        happens lazily, only on graphs actually replayed. Same guards + gate
        semantics as set_active_hooks, minus apply_toggle.
        """
        eng = self._ring_engine
        if eng.bound_graph_count() == 0 or eng.toggle_node_count() == 0:
            raise RuntimeError(
                "set_active_hooks_lazy: no producer nodes registered / no graph exec "
                "bound (toggle_node_count=%d, bound_graph_count=%d)."
                % (eng.toggle_node_count(), eng.bound_graph_count()))
        if not eng.toggle_registry_uniform():
            raise RuntimeError(
                "set_active_hooks_lazy: captured graphs have non-uniform hook sets.")
        if not eng.toggle_registry_complete():
            raise RuntimeError(
                "set_active_hooks_lazy: toggle registry incomplete -- node-registry and "
                "exec-binding key sets differ (some captured graph unbound, or some "
                "bound graph has no recorded nodes).")
        pairs = [(int(ht), int(ln)) for (ht, ln) in enabled]
        eng.set_enabled_hooks(pairs)        # bumps target_version; device apply deferred
        self._toggle_gate_active = True
        self._lazy_active = True
        self._recompute_effective_specs()

    def _recompute_effective_specs(self) -> None:
        """Recompute the single effective-enabled set (active ∩ enabled ∩
        registered) via ONE batched pybind call. Called by set_active_hooks[_lazy]
        only (the enabled set changes there; registered set is fixed after
        capture). capacity-reserve + meta-push both read this -> they can never
        diverge from the device enabled set. Bumps version to invalidate the
        worker's capacity cache."""
        mask = self._ring_engine.effective_enabled_mask(
            [(s.hook_type, s.layer_no) for s in self._active_specs])
        self._effective_enabled_specs = [
            s for s, m in zip(self._active_specs, mask) if m]
        self._enabled_version += 1

    def is_graph_ready(self, raw_graph: int) -> bool:
        """Read-only replay-time guard: True iff the graph has recorded producer
        nodes AND a bound exec. False for a graph vLLM captured at runtime after
        the warmup window closed -> the replay hook fails loud (#3)."""
        return self._ring_engine.is_graph_ready(raw_graph)

    def ensure_graph_current(self, raw_graph: int) -> int:
        """Lazy: apply the deferred toggle to the graph about to replay (no-op if
        already current). Call right before the graph's replay; raw_graph is the
        cudaGraph_t handle the graph was bound with."""
        return self._ring_engine.ensure_graph_current(raw_graph)

    def record_replay_event(self, raw_graph: int) -> int:
        """Lazy event guard: record a stream event after a graph's replay so a
        later ensure_graph_current() waits for it before mutating that exec.
        Returns the CUDA error (0 = ok); nonzero is FATAL (a missing event would
        let a later ensure mutate an executing exec)."""
        return self._ring_engine.record_replay_event(raw_graph)

    def clear_toggle(self) -> None:
        """Paired teardown: clear the engine's toggle registry AND deactivate the
        host gate, so neither lane is left half-configured (#2). Call before a
        re-capture or when disabling monitoring."""
        self._ring_engine.clear_toggle_registry()
        self._toggle_gate_active = False
        self._lazy_active = False
        self._effective_enabled_specs = None
        self._enabled_version += 1

    def submit_cpu_direct(self, cpu_tensor: torch.Tensor,
                          hook_type: int, hook_id: int) -> None:
        """Submit a CPU-tensor to the drain -> p2p pipeline.

        Called from HookPoint.forward()'s safety-net branch when a single
        tensor exceeds ring capacity.  The tensor is already in pageable
        CPU memory; it bypasses the ring and staging entirely.
        """
        self._ring_engine.submit_cpu_direct(cpu_tensor)



# ---------------------------------------------------------------------------
# Module-level transport management
# ---------------------------------------------------------------------------

def activate(transport: RingTransport) -> None:
    global _active_transport
    _active_transport = transport
    try:
        from . import _native_engine as _ne
        _ne.ring_set_active_engine(transport._ring_engine)
    except Exception:
        pass  # .so not built or binding unavailable; CUDA graph path skipped


def deactivate() -> None:
    global _active_transport
    _active_transport = None
    try:
        from . import _native_engine as _ne
        _ne.ring_clear_active_engine()
    except Exception:
        pass


def get_active() -> Optional[RingTransport]:
    return _active_transport
