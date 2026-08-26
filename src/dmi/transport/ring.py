"""Ring-based GPU-to-CPU tensor transport for DMI capture.

Tensor metadata is pushed before the producer kernel launches, allowing the
native callback thread to reconstruct tensors without touching Python or the
GIL. Hook definitions and analytical shapes live in :mod:`dmi.hooks.specs`;
this module owns only native-op registration and ring runtime state.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.library

from ..hooks.dispatch import install_ring_hooks
from ..hooks.specs import *  # noqa: F401,F403 - compatibility re-exports
from ..hooks.specs import (
    _ATTN_SUFFIXES,
    _ATTN_WT_TYPES,
    _HIDDEN_DIM_TYPES,
    _MLP_SUFFIXES,
    _compute_hook_shape,
    _id_by_short,
    compute_hook_shape,
)


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
    from . import native as _ne
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

    @torch.library.register_fake("ring::record_producer")
    def _record_producer_fake(
        ring_payload: torch.Tensor,
        tensor: torch.Tensor,
        emit_gate: Optional[torch.Tensor] = None,
        emit_value: int = 0,
    ) -> None:
        return None

    @torch.library.register_fake("ring::record_producer_prefix")
    def _record_producer_prefix_fake(
        ring_payload: torch.Tensor,
        tensor: torch.Tensor,
        row_count: torch.Tensor,
        row_bytes: int,
        emit_gate: Optional[torch.Tensor] = None,
        emit_value: int = 0,
    ) -> None:
        return None

    @torch.library.register_fake("ring::record_producer_chunked")
    def _record_producer_chunked_fake(
        ring_payload: torch.Tensor,
        tensor: torch.Tensor,
        chunk_bytes: torch.Tensor,
        emit_gate: Optional[torch.Tensor] = None,
        emit_value: int = 0,
    ) -> None:
        return None

    @torch.library.register_fake("ring::record_producer_seq_prefix_pack")
    def _record_producer_seq_prefix_pack_fake(
        ring_payload: torch.Tensor,
        tensor: torch.Tensor,
        valid_count: torch.Tensor,
        valid_prefix_sum: torch.Tensor,
        feature_bytes: int,
        emit_gate: Optional[torch.Tensor] = None,
        emit_value: int = 0,
    ) -> None:
        return None

    @torch.library.register_fake("ring::record_producer_segmented_pack")
    def _record_producer_segmented_pack_fake(
        ring_payload: torch.Tensor,
        tensor: torch.Tensor,
        segment_start: torch.Tensor,
        segment_end: torch.Tensor,
        feature_bytes: int,
        emit_gate: Optional[torch.Tensor] = None,
        emit_value: int = 0,
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
        for spec in self._active_specs:
            spec_q_len = (actual_q_len if actual_q_len is not None
                          and spec.dim0_is_actual_tokens
                          else q_len)
            shape = compute_hook_shape(
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

    def submit_cpu_direct(self, cpu_tensor: torch.Tensor,
                          hook_type: int, hook_id: int) -> None:
        """Submit a CPU-tensor to the drain -> p2p pipeline.

        Called from HookPoint.forward()'s safety-net branch when a single
        tensor exceeds ring capacity.  The tensor is already in pageable
        CPU memory; it bypasses the ring and staging entirely.
        """
        self._ring_engine.submit_cpu_direct(cpu_tensor)

    # ------------------------------------------------------------------
    # Opt-in generic-record path.  None of these methods is used by the
    # released inference adapters.

    def _record_payload_tensor(self) -> torch.Tensor:
        """Return the stable payload alias to the core ``RecordRuntime``."""

        return self._ring_payload

    def configure_record_schema(self, schema: Any) -> None:
        """Install the immutable schema used to encode Python descriptors."""

        if hasattr(self, "_record_schema"):
            raise RuntimeError("record schema is already configured")
        self._record_schema = schema

    def reserve_record(self, reservation_items: Any) -> int:
        """Reserve ordered encoded-record producer entries."""

        return int(self._ring_engine.reserve_record(tuple(reservation_items)))

    def push_record_descriptors(self, descriptors: Any) -> None:
        """Publish descriptors in the exact order of their producer tasks."""

        self._ring_engine.push_record_descriptors(
            tuple(descriptors), self._record_schema
        )

    def submit_record_cpu_direct(self, output: Any, entry: Any) -> None:
        """Materialize one oversized physical output and submit it directly."""

        cpu_tensor = self._record_cpu_tensor(output, entry)
        self._ring_engine.submit_record_cpu_direct(
            cpu_tensor,
            int(cpu_tensor.numel()) * int(cpu_tensor.element_size()),
        )

    def flush_records_and_wait(self, timeout_s: float) -> None:
        """Durably finish all generic-record work or raise its async error."""

        completed = self._ring_engine.flush_records_and_wait(float(timeout_s))
        if completed is False:
            raise TimeoutError("timed out waiting for record ring completion")

    @staticmethod
    def _record_cpu_tensor(output: Any, entry: Any) -> torch.Tensor:
        """Apply the selected producer transformation for CPU-direct fallback."""

        from ..hooks.record import TransportType

        source = output.tensor.detach().cpu().contiguous()
        transport_type = entry.transport_type
        if transport_type is TransportType.IDENTITY:
            return source

        byte_source = source.view(torch.uint8).reshape(-1)
        if transport_type is TransportType.PREFIX_STRIP:
            row_count = int(output.producer_meta[0].detach().cpu().item())
            row_bytes = int(entry.transport_args[0])
            nbytes = min(byte_source.numel(), max(0, row_count) * row_bytes)
            return byte_source[:nbytes].clone()

        if transport_type is TransportType.CHUNKED:
            counts = output.producer_meta[0].detach().cpu().reshape(-1)
            chunks = int(counts.numel())
            if chunks <= 0 or byte_source.numel() % chunks != 0:
                raise ValueError("CHUNKED CPU fallback requires equal input chunks")
            chunk_bytes = byte_source.numel() // chunks
            pieces = []
            for index, value in enumerate(counts.tolist()):
                count = max(0, min(chunk_bytes, int(value)))
                begin = index * chunk_bytes
                pieces.append(byte_source[begin : begin + count])
            return torch.cat(pieces).contiguous() if pieces else byte_source[:0].clone()

        if transport_type is TransportType.SEQ_PREFIX_PACK:
            if source.dim() < 2:
                raise ValueError("SEQ_PREFIX_PACK requires [S, B, ...] input")
            counts = output.producer_meta[0].detach().cpu().reshape(-1)
            if source.size(1) != counts.numel():
                raise ValueError("SEQ_PREFIX_PACK valid-count length mismatch")
            pieces = []
            for batch_index, value in enumerate(counts.tolist()):
                count = max(0, min(source.size(0), int(value)))
                if count:
                    pieces.append(source[:count, batch_index, ...].contiguous())
            if pieces:
                return torch.cat(pieces, dim=0).contiguous()
            return torch.empty(
                (0, *tuple(source.shape[2:])),
                dtype=source.dtype,
            )

        if transport_type is TransportType.SEGMENTED_PACK:
            starts = output.producer_meta[0].detach().cpu().reshape(-1)
            ends = output.producer_meta[1].detach().cpu().reshape(-1)
            if starts.numel() == 0 or starts.numel() != ends.numel():
                raise ValueError("SEGMENTED_PACK start/end length mismatch")
            pieces = []
            rows = source.size(0)
            for start, end in zip(starts.tolist(), ends.tolist()):
                begin = max(0, min(rows, int(start)))
                finish = max(begin, min(rows, int(end)))
                if finish > begin:
                    pieces.append(source[begin:finish, ...].contiguous())
            if pieces:
                return torch.cat(pieces, dim=0).contiguous()
            return torch.empty(
                (0, *tuple(source.shape[1:])),
                dtype=source.dtype,
            )

        raise ValueError(f"Unsupported transport type {transport_type!r}")



# ---------------------------------------------------------------------------
# Module-level transport management
# ---------------------------------------------------------------------------

def activate(transport: RingTransport) -> None:
    global _active_transport
    _active_transport = transport
    try:
        from . import native as _ne
        _ne.ring_set_active_engine(transport._ring_engine)
    except Exception:
        pass  # .so not built or binding unavailable; CUDA graph path skipped


def deactivate() -> None:
    global _active_transport
    _active_transport = None
    try:
        from . import native as _ne
        _ne.ring_clear_active_engine()
    except Exception:
        pass


def get_active() -> Optional[RingTransport]:
    return _active_transport
