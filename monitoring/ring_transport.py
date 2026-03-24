"""Ring-based GPU-to-CPU tensor transport for monitoring.

Replaces NativeMonitoringEngine's pin-pool cudaMemcpy path with the ring
producer/drain pipeline.  Tensor metadata is pushed to the C++ TensorMetaFifo
(via push_meta) before the producer kernel is launched, so the C++ callback
thread can reconstruct and slice the tensor without ever touching Python or
the GIL.

New CUDA-graph-compatible path (activated when model_shape + get_hook_specs are available):
  - ring_producer_op: torch.library.custom_op wrapping ring_engine.hook()
  - register_forward_hook on HookPoint modules (PyTorch-native dispatch)
  - ModelShapeConfig + analytical shape computation (no warmup needed)
  - pre_push_all_metas called before orig_forward, outside compiled region

Legacy path (HookPoint.forward -> capture_tensor) still works as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.library
from torch import nn


# ---------------------------------------------------------------------------
# Hook-type constants (values match _HOOK_SUFFIX_TO_TYPE)
# ---------------------------------------------------------------------------

HOOK_TYPE_RESID_PRE   = 0
HOOK_TYPE_LN1         = 1
HOOK_TYPE_ATTN_OUT    = 2
HOOK_TYPE_RESID_MID   = 3
HOOK_TYPE_ATTN_SCORES = 4
HOOK_TYPE_PATTERN     = 5
HOOK_TYPE_Q           = 6
HOOK_TYPE_K           = 7
HOOK_TYPE_V           = 8
HOOK_TYPE_Z           = 9
HOOK_TYPE_RESULT      = 10
HOOK_TYPE_LN2         = 11
HOOK_TYPE_MLP_IN      = 12
HOOK_TYPE_MLP_OUT     = 13
HOOK_TYPE_RESID_POST  = 14
# Global residual stream right before final layer norm.
# Reuse the same ring hook-type id as resid_post since the tensor shape and
# slicing semantics are identical; it only differs by layer_no = -1.
HOOK_TYPE_RESID_FINAL = HOOK_TYPE_RESID_POST
HOOK_TYPE_EMBED       = 15
HOOK_TYPE_POS_EMBED   = 16
HOOK_TYPE_FINAL_LN    = 17
HOOK_TYPE_TOKEN_IDS   = 18
HOOK_TYPE_FINAL_LOGITS = 19

_HIDDEN_DIM_TYPES = frozenset({
    HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_MID, HOOK_TYPE_RESID_POST,
    HOOK_TYPE_ATTN_OUT,  HOOK_TYPE_MLP_IN,    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_LN1,       HOOK_TYPE_LN2,
    HOOK_TYPE_EMBED,     HOOK_TYPE_POS_EMBED,  HOOK_TYPE_FINAL_LN,
})

# ---------------------------------------------------------------------------
# Hook selection: composable presets + individual hook types
#
# Selection is a comma-separated string.  Each token is looked up in
# _HOOK_SELECTIONS (presets or individual hook names).  The final enabled
# set is the union of all tokens.
#
# Examples:
#   "full"                            -- all hooks
#   "no-attention-scores"             -- full minus attn_scores/pattern
#   "hidden-states,token_ids"         -- resid_pre + token_ids
#   "hidden-states,final_ln,logits"   -- resid_pre + final_ln + final_logits
#   "resid_pre,resid_post,embed"      -- just those three
# ---------------------------------------------------------------------------

_ALL_HOOK_TYPES = frozenset({
    HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_RESID_MID, HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN,
    HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z,
    HOOK_TYPE_RESULT, HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN, HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_RESID_POST, HOOK_TYPE_EMBED, HOOK_TYPE_POS_EMBED,
    HOOK_TYPE_FINAL_LN, HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_FINAL_LOGITS,
})

# -- Presets --
_HOOK_SELECTIONS: Dict[str, frozenset] = {
    "full": _ALL_HOOK_TYPES,
    # full minus attn_scores/pattern (FlashAttention never materializes them)
    "no-attention-scores": _ALL_HOOK_TYPES - {HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN},
    # What HF returns with output_hidden_states + output_attentions + logits
    "hf-only": frozenset({
        HOOK_TYPE_RESID_PRE, HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_PATTERN,
        HOOK_TYPE_FINAL_LOGITS,
    }),
}

# -- Individual hook type names (each maps to a single-element frozenset) --
_HOOK_TYPE_BY_NAME: Dict[str, int] = {
    "resid_pre":   HOOK_TYPE_RESID_PRE,
    "ln1":         HOOK_TYPE_LN1,
    "attn_out":    HOOK_TYPE_ATTN_OUT,
    "resid_mid":   HOOK_TYPE_RESID_MID,
    "attn_scores": HOOK_TYPE_ATTN_SCORES,
    "pattern":     HOOK_TYPE_PATTERN,
    "q":           HOOK_TYPE_Q,
    "k":           HOOK_TYPE_K,
    "v":           HOOK_TYPE_V,
    "z":           HOOK_TYPE_Z,
    "result":      HOOK_TYPE_RESULT,
    "ln2":         HOOK_TYPE_LN2,
    "mlp_in":      HOOK_TYPE_MLP_IN,
    "mlp_out":     HOOK_TYPE_MLP_OUT,
    "resid_post":  HOOK_TYPE_RESID_POST,
    "resid_final": HOOK_TYPE_RESID_FINAL,
    "embed":       HOOK_TYPE_EMBED,
    "pos_embed":   HOOK_TYPE_POS_EMBED,
    "final_ln":    HOOK_TYPE_FINAL_LN,
    "token_ids":   HOOK_TYPE_TOKEN_IDS,
    "final_logits": HOOK_TYPE_FINAL_LOGITS,
}
for _name, _htype in _HOOK_TYPE_BY_NAME.items():
    _HOOK_SELECTIONS[_name] = frozenset({_htype})

# -- Aliases --
_HOOK_SELECTIONS["hidden-states"] = _HOOK_SELECTIONS["resid_pre"]
_HOOK_SELECTIONS["hidden_states"] = _HOOK_SELECTIONS["resid_pre"]
_HOOK_SELECTIONS["logits"] = _HOOK_SELECTIONS["final_logits"]
_HOOK_SELECTIONS["token-ids"] = _HOOK_SELECTIONS["token_ids"]


def resolve_hook_selection(mode: str) -> frozenset:
    """Resolve a comma-separated hook selection string to a set of hook types.

    Each comma-separated token is looked up in _HOOK_SELECTIONS (presets
    and individual hook names).  The result is the union of all tokens.
    """
    result: set = set()
    for token in mode.split(","):
        token = token.strip()
        if not token:
            continue
        entry = _HOOK_SELECTIONS.get(token)
        if entry is None:
            raise ValueError(
                f"Unknown hook selection {token!r}. "
                f"Available: {sorted(_HOOK_SELECTIONS.keys())}")
        result |= entry
    if not result:
        raise ValueError(f"Empty hook selection: {mode!r}")
    return frozenset(result)


def apply_hook_selection(specs: List["HookSpec"], mode: str) -> List["HookSpec"]:
    """Filter specs and set HookPoint.enabled based on a selection string.

    Selection is a comma-separated string of preset names and/or individual
    hook type names.  The enabled set is the union of all tokens.

    Sets enabled=True on hooks in the set, enabled=False on others.
    Returns the filtered list of enabled specs (for _active_specs / metadata).
    """
    allowed = resolve_hook_selection(mode)

    enabled_specs = []
    for spec in specs:
        if spec.hook_type in allowed:
            spec.module.enabled = True
            enabled_specs.append(spec)
        else:
            spec.module.enabled = False
    return enabled_specs


# ---------------------------------------------------------------------------
# Hook-name -> (hook_type, hook_id) helpers  (legacy path)
# ---------------------------------------------------------------------------

_HOOK_SUFFIX_TO_TYPE: Dict[str, int] = {
    "hook_resid_pre":   HOOK_TYPE_RESID_PRE,
    "hook_ln1":         HOOK_TYPE_LN1,
    "hook_attn_out":    HOOK_TYPE_ATTN_OUT,
    "hook_resid_mid":   HOOK_TYPE_RESID_MID,
    "hook_attn_scores": HOOK_TYPE_ATTN_SCORES,
    "hook_pattern":     HOOK_TYPE_PATTERN,
    "hook_q":           HOOK_TYPE_Q,
    "hook_k":           HOOK_TYPE_K,
    "hook_v":           HOOK_TYPE_V,
    "hook_z":           HOOK_TYPE_Z,
    "hook_result":      HOOK_TYPE_RESULT,
    "hook_ln2":         HOOK_TYPE_LN2,
    "hook_mlp_in":      HOOK_TYPE_MLP_IN,
    "hook_mlp_out":     HOOK_TYPE_MLP_OUT,
    "hook_resid_post":  HOOK_TYPE_RESID_POST,
    "hook_resid_final": HOOK_TYPE_RESID_FINAL,
    "hook_embed":       HOOK_TYPE_EMBED,
    "hook_pos_embed":   HOOK_TYPE_POS_EMBED,
    "hook_final_ln":    HOOK_TYPE_FINAL_LN,
    "token_ids":        18,
    "final_logits":     19,
}


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
        logits_to_keep: HF generate() default is 1 (only last token logits).
            0 means keep all (q_len).
    """
    if hook_type in _HIDDEN_DIM_TYPES:
        return [batch, q_len, cfg.hidden_dim]
    if hook_type == HOOK_TYPE_Q:
        return [batch, q_len, cfg.num_heads, cfg.head_dim]
    if hook_type in (HOOK_TYPE_K, HOOK_TYPE_V):
        return [batch, q_len, cfg.num_kv_heads, cfg.head_dim]
    if hook_type == HOOK_TYPE_Z:
        return [batch, q_len, cfg.num_heads, cfg.head_dim]
    if hook_type == HOOK_TYPE_RESULT:
        # hook_result is after o_proj -> shape is [B, q_len, hidden_dim],
        # not [B, q_len, num_heads, head_dim] (differs on GQA models).
        return [batch, q_len, cfg.hidden_dim]
    if hook_type in (HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN):
        return [batch, cfg.num_heads, q_len, kv_dim]
    if hook_type == HOOK_TYPE_TOKEN_IDS:
        return [batch, q_len]
    if hook_type == HOOK_TYPE_FINAL_LOGITS:
        # HF generate() sets logits_to_keep=1 by default (only last token
        # needed for next-token prediction).  The actual tensor shape is
        # [batch, 1, vocab], not [batch, q_len, vocab].
        logits_q = min(q_len, logits_to_keep) if logits_to_keep > 0 else q_len
        return [batch, logits_q, cfg.vocab_size] if cfg.vocab_size > 0 else []
    return []  # unknown type -- push_meta skipped


# ---------------------------------------------------------------------------
# Forward-hook installation
# ---------------------------------------------------------------------------

def _make_ring_hook(hook_type: int, hook_id: int):
    """Return a PyTorch register_forward_hook callable for a HookPoint.

    Legacy / no-op: ring::producer is now called directly inside
    HookPoint.forward() via torch.ops.ring.producer (C++ TORCH_LIBRARY,
    captured in CUDA graph).  This hook is kept so _forward_hook_names
    remains populated and capture_tensor() correctly skips hooks handled
    by HookPoint.forward().  The actual GPU->ring data path is entirely
    in the C++ producer kernel; these Python hooks are never invoked for
    ring transport data capture.
    """
    def _hook(module: nn.Module, inp: Any, output: Any) -> None:
        pass
    return _hook


def install_ring_hooks(specs: List[HookSpec], handles_out: List) -> None:
    """Register ring producer forward hooks on each spec's module.

    Legacy: these hooks are no-ops (see _make_ring_hook).  They exist only
    to populate _forward_hook_names so capture_tensor() skips hooks that
    HookPoint.forward() handles via torch.ops.ring.producer.

    handles_out receives the RemovableHandle for each hook so callers
    can remove them later via handle.remove().
    """
    for spec in specs:
        handle = spec.module.register_forward_hook(
            _make_ring_hook(spec.hook_type, spec.layer_no)
        )
        handles_out.append(handle)


# ---------------------------------------------------------------------------
# RingTransport
# ---------------------------------------------------------------------------

class RingTransport:
    """Manages ring engine + per-step batch context for ring-mode monitoring.

    Two capture paths:
      - New (CUDA-graph-compatible): install_ring_hooks + pre_push_all_metas.
        Activated when _model_cfg is set and _using_forward_hooks is True.
      - Legacy: capture_tensor() called from HookPoint.forward().
        _using_forward_hooks=False makes capture_tensor active.
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

        # When True: push_meta / capture_tensor meta pushes are skipped so the
        # FIFO stays empty.  ring_producer_op still calls _ring_engine.hook()
        # (same kernel launch) so CUDA graph topology is identical to real mode.
        # Toggle via _ring_engine.set_null_mode() to control device-side behavior.
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
        # Names of hooks handled by register_forward_hook (populated at install time).
        # capture_tensor() skips these; any HookPoint whose name is not in this set
        # falls through to the legacy capture_tensor() path.
        self._forward_hook_names: set = set()

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

    def set_model_cfg(self, cfg: ModelShapeConfig) -> None:
        """Set the model shape config for analytical shape computation."""
        self._model_cfg = cfg

    def pre_push_all_metas(self, batch: int, q_len: int, kv_dim: int,
                           logits_to_keep: int = 0) -> None:
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
            dtype = spec.dtype if spec.dtype is not None else self._model_cfg.dtype
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
    except Exception:
        pass


def get_active() -> Optional[RingTransport]:
    return _active_transport
