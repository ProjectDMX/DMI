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

Legacy path (HookPoint.forward → capture_tensor) still works as a fallback.
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
# Hook-name → (hook_type, hook_id) helpers  (legacy path)
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
    "hook_embed":       HOOK_TYPE_EMBED,
    "hook_pos_embed":   HOOK_TYPE_POS_EMBED,
    "hook_final_ln":    HOOK_TYPE_FINAL_LN,
    "token_ids":        18,
    "final_logits":     19,
}


def _hook_type_from_name(hook_name: str) -> int:
    for suffix, htype in _HOOK_SUFFIX_TO_TYPE.items():
        if hook_name == suffix or hook_name.endswith("." + suffix):
            return htype
    return 0


def _hook_id_from_name(hook_name: str) -> int:
    """Extract layer index from 'blocks.N.xxx' or 'layers.N.xxx', returns 0 otherwise."""
    parts = hook_name.split(".")
    if len(parts) >= 2 and parts[0] in ("blocks", "layers"):
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# ModelShapeConfig — provided at hook-installation time
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
# HookSpec — model self-describes its hooks in forward() firing order
# ---------------------------------------------------------------------------

@dataclass
class HookSpec:
    """One monitoring hook: name, shape convention, and module reference."""
    name:      str                        # e.g. "blocks.3.attn.hook_attn_scores"
    hook_type: int                        # HOOK_TYPE_* — determines shape formula
    module:    nn.Module                  # the HookPoint instance
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
    _ne._load_extension()  # ensure .so is loaded → registers ring::producer

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
# kv_dim computation — cache-type-aware, called before each forward
# ---------------------------------------------------------------------------

def _get_kv_dim(past_key_values: Any, q_len: int) -> int:
    """Return the key-sequence dimension for shape computation.

    - None (prefill, no cache):  kv_dim = q_len
    - StaticCache:               kv_dim = max_seq_len (full pre-allocated buffer)
    - DynamicCache or other:     kv_dim = existing_len + q_len
    """
    if past_key_values is None:
        return q_len
    try:
        from transformers.cache_utils import StaticCache
        if isinstance(past_key_values, StaticCache):
            return past_key_values.key_cache[0].shape[-2]
    except Exception:
        pass
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
) -> List[int]:
    """Return expected tensor shape for a given hook type and step dimensions."""
    if hook_type in _HIDDEN_DIM_TYPES:
        return [batch, q_len, cfg.hidden_dim]
    if hook_type == HOOK_TYPE_Q:
        return [batch, q_len, cfg.num_heads, cfg.head_dim]
    if hook_type in (HOOK_TYPE_K, HOOK_TYPE_V):
        return [batch, q_len, cfg.num_kv_heads, cfg.head_dim]
    if hook_type in (HOOK_TYPE_Z, HOOK_TYPE_RESULT):
        return [batch, q_len, cfg.num_heads, cfg.head_dim]
    if hook_type in (HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN):
        return [batch, cfg.num_heads, q_len, kv_dim]
    if hook_type == HOOK_TYPE_TOKEN_IDS:
        return [batch, q_len]
    if hook_type == HOOK_TYPE_FINAL_LOGITS:
        return [batch, q_len, cfg.vocab_size] if cfg.vocab_size > 0 else []
    return []  # unknown type — push_meta skipped


# ---------------------------------------------------------------------------
# Forward-hook installation
# ---------------------------------------------------------------------------

def _make_ring_hook(hook_type: int, hook_id: int):
    """Return a PyTorch register_forward_hook callable for a HookPoint.

    No-op: ring::producer is now called directly inside HookPoint.forward()
    via torch.ops.ring.producer (C++ TORCH_LIBRARY, captured in CUDA graph).
    This hook is kept so _forward_hook_names remains populated and
    capture_tensor() correctly skips hooks handled by HookPoint.forward().
    """
    def _hook(module: nn.Module, inp: Any, output: Any) -> None:
        pass
    return _hook


def install_ring_hooks(specs: List[HookSpec], handles_out: List) -> None:
    """Register ring producer forward hooks on each spec's module.

    handles_out receives the RemovableHandle for each hook so callers
    can remove them later via handle.remove().
    """
    for spec in specs:
        hook_id = _hook_id_from_name(spec.name)
        handle = spec.module.register_forward_hook(
            _make_ring_hook(spec.hook_type, hook_id)
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

    def __init__(self, ring_engine: Any, drain_notify_on_forward: bool = True) -> None:
        self._ring_engine = ring_engine
        self.drain_notify_on_forward = drain_notify_on_forward

        # Current step context — set before each forward pass
        self._current_model_id: Optional[str] = None
        self._current_shard_rank: int = 0
        self._current_req_ids: Optional[List[str]] = None
        self._current_token_ranges: Optional[List[Tuple[int, int]]] = None

        # When True: push_meta / capture_tensor meta pushes are skipped so the
        # FIFO stays empty.  ring_producer_op still calls _ring_engine.hook()
        # (same kernel launch) so CUDA graph topology is identical to real mode.
        # Toggle via _ring_engine.set_null_mode() to control device-side behavior.
        self.null_offload: bool = False

        # New-path state
        self._model_cfg: Optional[ModelShapeConfig] = None
        self._active_specs: List[HookSpec] = []
        self._using_forward_hooks: bool = False
        # Names of hooks handled by register_forward_hook (populated at install time).
        # capture_tensor() skips only these, letting other HookPoints (e.g. token_ids,
        # final_logits) still go through the legacy path.
        self._forward_hook_names: set = set()

    def set_step_context(
        self,
        model_id: str,
        shard_rank: int,
        req_ids: List[str],
        token_ranges: List[Tuple[int, int]],
    ) -> None:
        """Called before each forward pass to provide per-step batch metadata."""
        self._current_model_id = model_id
        self._current_shard_rank = shard_rank
        self._current_req_ids = req_ids
        self._current_token_ranges = token_ranges

    def set_model_cfg(self, cfg: ModelShapeConfig) -> None:
        """Set the model shape config for analytical shape computation."""
        self._model_cfg = cfg

    def pre_push_all_metas(self, batch: int, q_len: int, kv_dim: int) -> None:
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

        for spec in self._active_specs:
            shape = _compute_hook_shape(
                spec.hook_type, self._model_cfg, batch, q_len, kv_dim
            )
            if not shape:
                continue
            dtype = spec.dtype if spec.dtype is not None else self._model_cfg.dtype
            self._ring_engine.push_meta(
                spec.name,
                self._current_model_id,
                self._current_shard_rank,
                list(self._current_req_ids),
                list(self._current_token_ranges),
                shape,
                dtype,
            )

    def capture_tensor(self, tensor: torch.Tensor, hook_name: str) -> None:
        """Legacy capture path: called from HookPoint.forward().

        Skipped for hooks that are covered by register_forward_hook (i.e. hooks
        whose names appear in _forward_hook_names).  Hooks absent from
        _forward_hook_names (e.g. token_ids, final_logits) still go through
        this path regardless of _using_forward_hooks.
        """
        if hook_name in self._forward_hook_names:
            return
        if self.null_offload:
            return  # kernel still launched by ring_producer_op; skip meta here
        if self._current_req_ids is None or self._current_token_ranges is None:
            return
        if self._current_model_id is None:
            return
        if not tensor.is_cuda or not tensor.is_contiguous():
            return

        hook_type     = _hook_type_from_name(hook_name)
        hook_id       = _hook_id_from_name(hook_name)
        shape         = list(tensor.shape)
        dtype         = tensor.dtype
        d_ptr         = tensor.data_ptr()
        nbytes        = tensor.nbytes
        stream_handle = torch.cuda.current_stream(tensor.device.index).cuda_stream

        self._ring_engine.push_meta(
            hook_name,
            self._current_model_id,
            self._current_shard_rank,
            list(self._current_req_ids),
            list(self._current_token_ranges),
            shape,
            dtype,
        )

        try:
            self._ring_engine.hook(d_ptr, nbytes, 0, hook_type, hook_id, stream_handle)
        except Exception:
            self._ring_engine.pop_last_meta()


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
