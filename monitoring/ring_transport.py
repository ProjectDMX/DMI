"""Ring-based GPU-to-CPU tensor transport for monitoring.

Replaces NativeMonitoringEngine's pin-pool cudaMemcpy path with the ring
producer/drain pipeline.  Tensor metadata is pushed to the C++ TensorMetaFifo
(via push_meta) before the producer kernel is launched, so the C++ callback
thread can reconstruct and slice the tensor without ever touching Python or
the GIL.

Usage:
  1. Create a RingEngine with a SubmitFn (or None for null mode).
  2. Call engine.enable_ring_transport(ring_config) to activate.
  3. HookPoint.forward() calls capture_tensor() via the active transport.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Hook-name → (hook_type, hook_id) helpers
# ---------------------------------------------------------------------------

_HOOK_SUFFIX_TO_TYPE: Dict[str, int] = {
    "hook_resid_pre": 0,
    "hook_ln1": 1,
    "hook_attn_out": 2,
    "hook_resid_mid": 3,
    "hook_attn_scores": 4,
    "hook_pattern": 5,
    "hook_q": 6,
    "hook_k": 7,
    "hook_v": 8,
    "hook_z": 9,
    "hook_result": 10,
    "hook_ln2": 11,
    "hook_mlp_in": 12,
    "hook_mlp_out": 13,
    "hook_resid_post": 14,
    "hook_embed": 15,
    "hook_pos_embed": 16,
    "hook_final_ln": 17,
    "token_ids": 18,
    "final_logits": 19,
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
# RingTransport
# ---------------------------------------------------------------------------

class RingTransport:
    """Manages ring engine + per-step batch context for ring-mode monitoring.

    capture_tensor() is called from the forward-pass thread.
    All tensor reconstruction and DB submission happens in C++ (no GIL).
    """

    def __init__(self, ring_engine: Any) -> None:
        self._ring_engine = ring_engine

        # Current step context — set before each forward pass
        self._current_model_id: Optional[str] = None
        self._current_shard_rank: int = 0
        self._current_req_ids: Optional[List[str]] = None
        self._current_token_ranges: Optional[List[Tuple[int, int]]] = None
        self._capture_enabled: bool = False

    def set_step_context(
        self,
        model_id: str,
        shard_rank: int,
        req_ids: List[str],
        token_ranges: List[Tuple[int, int]],
        capture_enabled: bool = True,
    ) -> None:
        """Called before each forward pass to provide per-step batch metadata."""
        self._current_model_id = model_id
        self._current_shard_rank = shard_rank
        self._current_req_ids = req_ids
        self._current_token_ranges = token_ranges
        self._capture_enabled = capture_enabled

    def capture_tensor(self, tensor: torch.Tensor, hook_name: str) -> None:
        """Called from a HookPoint during the forward pass.

        Pushes C++ FIFO metadata first (push_meta), then launches the ring
        producer kernel (hook) so the FIFO pop order in the callback thread
        matches arrival order.
        """
        if not self._capture_enabled:
            return
        if self._current_req_ids is None or self._current_token_ranges is None:
            return
        if self._current_model_id is None:
            return
        if not tensor.is_cuda or not tensor.is_contiguous():
            return

        hook_type = _hook_type_from_name(hook_name)
        hook_id   = _hook_id_from_name(hook_name)

        # Push metadata to the C++ FIFO before launching the kernel.
        # Extract all Python-owned values while still holding the GIL.
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
            # Kernel launch failed — undo the metadata push so FIFO stays in sync.
            self._ring_engine.pop_last_meta()


# ---------------------------------------------------------------------------
# Module-level active transport (accessed from HookPoint.forward())
# ---------------------------------------------------------------------------

_active_transport: Optional[RingTransport] = None


def activate(transport: RingTransport) -> None:
    global _active_transport
    _active_transport = transport


def deactivate() -> None:
    global _active_transport
    _active_transport = None


def get_active() -> Optional[RingTransport]:
    return _active_transport
