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
from typing import Any, Dict, List, Optional, Tuple

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
# Hook type constants — single source of truth is HOOK_DEFS in tensor_meta.h.
# All mappings are auto-derived from the C++ table at import time.
# To add a new hook: add one enum value + one HOOK_DEFS row in C++. Done.
# ---------------------------------------------------------------------------
from ._native_engine import _load_extension as _load_ext
_ext = _load_ext()
# (id, act_name, short_name, per_layer, group, tp_sharded, shape_class, pp_stage)
# group/shape_class/pp_stage are int enums matching the C++ definitions.
_HOOK_DEFS = _ext.HOOK_DEFS

# C++ enum mirrors — keep in sync with tensor_meta.h
GROUP_ATTN, GROUP_MLP, GROUP_OTHER = 0, 1, 2
SHAPE_HIDDEN, SHAPE_QKV_Q, SHAPE_QKV_KV, SHAPE_QKV_Z = 0, 1, 2, 3
SHAPE_ATTN_WT, SHAPE_MLP_POST, SHAPE_TOKEN_IDS, SHAPE_LOGITS = 4, 5, 6, 7
PP_ANY, PP_FIRST, PP_LAST = 0, 1, 2

# Auto-derive all mappings
_id_by_short: Dict[str, int] = {}       # "q" → 6
for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS:
    _id_by_short[_short] = _id
    # Inject HOOK_TYPE_Q, HOOK_TYPE_RESID_PRE, etc. into module namespace
    globals()[f"HOOK_TYPE_{_short.upper()}"] = _id

# Auto-derive act_name suffix sets per group (used by config.py HookSelection).
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
# monitoring/selection.py — that module imports the C++-mirror constants
# above.  See the unified-adaptor refactor plan §6 for rationale.


# ---------------------------------------------------------------------------
# Hook-name -> hook_type helpers (auto-derived from HOOK_DEFS act_name)
# ---------------------------------------------------------------------------

# act_name is the suffix used in HookPoint names (e.g. "attn.hook_q", "token_ids")
_HOOK_SUFFIX_TO_TYPE: Dict[str, int] = {_act: _id for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in _HOOK_DEFS}


def align_up_py(x: int, a: int) -> int:
    """Python equivalent of ring::align_up (a must be a power of 2)."""
    return (x + a - 1) & ~(a - 1)


def _hook_type_from_name(hook_name: str) -> int:
    for suffix, htype in _HOOK_SUFFIX_TO_TYPE.items():
        if hook_name == suffix or hook_name.endswith("." + suffix):
            return htype
    return 0


def _layer_no_from_name(hook_name: str) -> int:
    """Extract layer index from 'blocks.N.xxx' or 'layers.N.xxx', returns -1 for global hooks."""
    parts = hook_name.split(".")
    if len(parts) >= 2 and parts[0] in ("blocks", "layers"):
        try:
            return int(parts[1])
        except ValueError:
            pass
    return -1


# Keep old name as alias for callers that still use it (HookPoint sets _ring_hook_id)
_hook_id_from_name = _layer_no_from_name


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

    @torch.library.register_fake("ring::producer")
    def _ring_producer_fake(
        tensor: torch.Tensor, hook_type: int, hook_id: int
    ) -> None:
        # Void schema: op is a pure side-effect (kernel launch), no output.
        # Marked effectful via _register_effectful_op so FX/inductor cannot DCE
        # the node even when its return value is unused.
        return None

    # Mark ring::producer as an ordered side-effect so torch.compile/inductor
    # preserves the node in the FX graph (prevents DCE on [num_users=0] nodes).
    try:
        from torch._higher_order_ops.effects import (
            _register_effectful_op, _EffectType,
        )
        _register_effectful_op(
            torch.ops.ring.producer.default, _EffectType.ORDERED
        )
    except Exception:
        pass  # older PyTorch without _EffectType; effectful path unavailable

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

    ASSUMPTION: hooked tensors have deterministic shapes given the same
    (batch, q_len, kv_dim, logits_to_keep) and model config.  This is
    guaranteed by the model architecture.

    Args:
        batch: batch size.  0 = flattened (vLLM): no batch dimension,
            q_len = total_tokens across all requests.
        logits_to_keep: HF generate() default is 1 (only last token logits).
            0 means keep all (q_len).
    """
    # batch=0 means flattened (vLLM): shapes have no batch dimension.
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
        # vLLM Attention.forward returns [N, hidden_size] (heads flattened).
        # HF returns [batch, q_len, num_heads, head_dim].
        if batch == 0:
            return [q_len, (cfg.num_heads // tp) * cfg.head_dim]
        return b + [q_len, cfg.num_heads // tp, cfg.head_dim]
    if hook_type in (HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN):
        return b + [cfg.num_heads // tp, q_len, kv_dim]
    if hook_type == HOOK_TYPE_MLP_POST:
        if cfg.intermediate_dim == 0:
            return []  # intermediate_dim unknown — skip this hook
        return b + [q_len, cfg.intermediate_dim // tp]
    if hook_type == HOOK_TYPE_TOKEN_IDS:
        return b + [q_len]
    if hook_type == HOOK_TYPE_FINAL_LOGITS:
        # compute_logits returns fewer rows than q_len when logits_to_keep
        # is set (HF generate default=1, vLLM always 1 per request).
        #
        # HF batched: tensor is [batch, logits_to_keep, vocab].
        #   logits_to_keep = min(q_len, logits_to_keep) capped to q_len.
        #
        # vLLM flattened: tensor is [num_reqs, vocab] (1 logit per
        #   request).  Caller passes logits_to_keep=num_reqs so the
        #   meta shape becomes [num_reqs, vocab].  The p2p thread
        #   indexes by request position (not token offset) and adjusts
        #   DB token range to (end_token-1, end_token).
        if batch > 0:
            logits_q = min(q_len, logits_to_keep) if logits_to_keep > 0 else q_len
        else:
            logits_q = logits_to_keep if logits_to_keep > 0 else q_len
        return (b + [logits_q, cfg.vocab_size]) if cfg.vocab_size > 0 else []
    return []  # unknown type -- push_meta skipped


# ---------------------------------------------------------------------------
# Forward-hook installation
# ---------------------------------------------------------------------------

def install_ring_hooks(specs: List[HookSpec]) -> None:
    """Set up HookPoints for ring transport.

    Sets _name, _ring_hook_type, _ring_hook_id on each HookPoint so
    HookPoint.forward() fires torch.ops.ring.producer.
    """
    for spec in specs:
        hp = spec.module
        if hasattr(hp, '_name') and hp._name is None:
            hp._name = f"hook_{spec.hook_type}_{spec.layer_no}"
            hp._ring_hook_type = spec.hook_type
            hp._ring_hook_id = spec.layer_no


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

        # Shared path flag for ALL hooks.  Reset to False at the start of
        # every pre-forward.  Set to True when the entire step's data exceeds
        # ring capacity (Case B).  When True, ALL enabled hooks use the
        # eager .cpu() path for that step.
        self.cpu_direct: bool = False

        # New-path state
        self._model_cfg: Optional[ModelShapeConfig] = None
        self._active_specs: List[HookSpec] = []
        self._using_forward_hooks: bool = False

        # When True, _prepare_wrapper skips prepare_step entirely --
        # all hooks use cpu_direct for the entire generate() call.
        # Set by generate_with_monitoring when decode doesn't fit in ring.
        self._force_cpu_direct: bool = False

        # Hook selection preset name (e.g. "full", "hf-only", "hidden-states").
        # Set by generate_with_monitoring before _install_monitoring_forward.
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

        dim0_offsets: per-request offset in tensor dim 0.
            HF: batch index (0, 1, 2, ...).  None = auto-generate range(len(req_ids)).
            vLLM: token offset in packed tensor (cumulative sum of scheduled tokens).
        kv_offsets: per-request kv-dimension start for attention hooks.
            HF dynamic cache: pad_len (real keys at the end, left-padded).
            HF static cache / vLLM: 0 (real keys at the start).
            None = auto-generate zeros.
        flattened: False = HF batched [batch, q_len, ...], True = vLLM packed [total_tokens, ...].
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

    def pre_push_all_metas(self, batch: int, q_len: int, kv_dim: int,
                           logits_to_keep: int = 0,
                           token_ids_dtype: Optional[torch.dtype] = None) -> None:
        """Push C++ FIFO metadata for all active specs before orig_forward.

        Called in the same order as install_ring_hooks() so FIFO pop order
        in the drain thread matches ring arrival order.
        Requires _model_cfg to be set via set_model_cfg() or enable_ring_transport().
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
        for spec in self._active_specs:
            shape = _compute_hook_shape(
                spec.hook_type, self._model_cfg, batch, q_len, kv_dim,
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

        if hook_types:
            self._ring_engine.push_all_metas(
                hook_types, layer_nos, shapes, dtypes,
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

    def submit_cpu_direct(self, cpu_tensor: torch.Tensor,
                          hook_type: int, hook_id: int) -> None:
        """Submit a CPU-direct tensor to the drain -> p2p pipeline.

        Called from HookPoint.forward() when cpu_direct=True.  The tensor
        is already in pageable CPU memory; it bypasses the ring and staging
        entirely.
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
        _ne.ring_set_cpu_direct(False)
    except Exception:
        pass


def set_cpu_direct(enabled: bool) -> None:
    """Set/clear the C++ cpu_direct flag for ring_producer_impl."""
    try:
        from . import _native_engine as _ne
        _ne.ring_set_cpu_direct(enabled)
    except Exception:
        pass


def get_active() -> Optional[RingTransport]:
    return _active_transport
