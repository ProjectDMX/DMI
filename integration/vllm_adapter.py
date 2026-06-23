"""vLLM integration: VLLMAdaptor + monitored GPU worker.

Phase 3a of the unified-adaptor refactor consolidates the vLLM-specific
orchestration that used to live in ``monitoring/vllm_integration.py``
into one file under ``integration/``.

Key pieces:

  * ``VLLMAdaptor`` -- concrete ``BackendAdaptor`` for vLLM models.
    Owns the framework-fragile pieces in localized methods:
      ``predict_padded_q_len`` (reads ``cudagraph_dispatcher._bs_to_padded_graph_size``);
      ``build_step_context`` (constructs the packed/flattened
      StepContext from a ``scheduler_output`` + ``model_runner`` pair);
      ``adapt_for_cpu_direct`` (swaps padded -> unpadded q_len when an
      oversize step forces eager dispatch + safety net for this batch);
      ``on_capacity_exceeded`` (no-op stub; transport.force_eager is
      owned by adaptor_base.before_forward and read by the
      ``_determine_batch_execution_and_padding`` wrapper);
      ``_warn_once_capacity`` (per-(total_q, num_reqs) shape warn).
  * ``DMXGPUWorker`` -- ~60-line vLLM ``Worker`` subclass that owns a
    ``VLLMAdaptor`` and delegates per-step work to it.  Architecture
    remap (GPT2LMHeadModel -> GPT2PLMHeadModel etc.) stays here.
  * Module-level ``register_preset("vllm-full", ...)`` -- relocated
    from ``monitoring/selection.py``'s default ``_HOOK_SELECTIONS``
    (deferred from Phase 1.5 per the unified-adaptor plan).  Lands as
    a side-effect of importing this module.

"""
from __future__ import annotations

import os
import re
import time
import warnings
from typing import Any, List, Optional, Tuple

import torch

from vllm.v1.worker.gpu_worker import Worker

from monitoring import ring_transport
from monitoring.adaptor_base import BackendAdaptor
from monitoring.ring_transport import (
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    align_up_py,
    install_ring_hooks,
)
from monitoring.selection import (
    apply_hook_selection,
    filter_by_pp_rank,
    filter_by_tp_rank,
    register_preset,
    _HOOK_SELECTIONS,
    _ALL_HOOK_TYPES,
)
# Import the C++-mirror constant we need for the "vllm-full" preset
# (full minus attention-weight matrices).  Import-time only.
from monitoring.ring_transport import _ATTN_WT_TYPES
from monitoring.step_context import StepContext

from integration.model_shape import _make_model_shape_from_hf_config


# ---------------------------------------------------------------------------
# vLLM-full preset registration (deferred from Phase 1.5).
#
# Moved out of monitoring/selection.py's default _HOOK_SELECTIONS so the
# core selection module is framework-neutral.  Registers when this
# module is imported -- which happens whenever DMXGPUWorker is loaded
# via worker_cls="integration.vllm_adapter.DMXGPUWorker".
#
# `register_preset` raises on duplicates, so re-import within the same
# process is a no-op (Python caches the module body).  Across separate
# processes (e.g. each TP rank in vLLM) each subprocess imports fresh
# and registers once.
# ---------------------------------------------------------------------------

if "vllm-full" not in _HOOK_SELECTIONS:
    register_preset("vllm-full", _ALL_HOOK_TYPES - _ATTN_WT_TYPES)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _cfg(ac: dict, key: str, env_key: str, default: Any) -> Any:
    """additional_config preferred, env var fallback, then default."""
    val = ac.get(key)
    if val is not None:
        return val
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if isinstance(default, bool):
            return env_val not in ("0", "false", "False", "")
        if isinstance(default, int):
            return int(env_val)
        return env_val
    return default


# ---------------------------------------------------------------------------
# In-process callable sink (hallu_monitor option-3 path)
# ---------------------------------------------------------------------------
# DMXGPUWorker normally builds a ClickHouse DMXHostEngine from additional_config.
# For an in-process consumer (e.g. hallu_monitor's ProbeWorker) there is no
# config-serializable way to hand a live Python object to the worker, so we
# stash a zero-arg factory in a module global. init_device() calls it when no
# dmx_db_host is set. Requires the in-process worker path
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) so this module global is shared; with TP>1
# every rank would call the factory, which must then return a per-rank sink.
_DMX_SINK_FACTORY: Optional[Any] = None


def register_sink_factory(factory: Any) -> None:
    """Register a zero-arg callable returning the host_engine sink (a ProbeWorker
    or any Python callable). Used instead of ClickHouse when dmx_db_host is unset.
    The sink's lifecycle (start/stop) is owned by the registrant, NOT the worker."""
    global _DMX_SINK_FACTORY
    _DMX_SINK_FACTORY = factory


def clear_sink_factory() -> None:
    global _DMX_SINK_FACTORY
    _DMX_SINK_FACTORY = None


_VLLM_REQ_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def normalize_vllm_request_id(req_id: str) -> str:
    """Strip the UUID suffix that vLLM V1 appends to request IDs."""
    return _VLLM_REQ_ID_SUFFIX.sub("", req_id)


# Architecture remap so vLLM's registry resolves to the hooked variant.
_ARCH_REMAP = {
    "GPT2LMHeadModel": "GPT2PLMHeadModel",
    "Qwen2MoeForCausalLM": "Qwen2MoePForCausalLM",
    "Qwen3ForCausalLM": "Qwen3PForCausalLM",
    "LlamaForCausalLM": "LlamaPForCausalLM",
}


# ---------------------------------------------------------------------------
# VLLMAdaptor
# ---------------------------------------------------------------------------


class VLLMAdaptor(BackendAdaptor):
    """``BackendAdaptor`` for vLLM models running under ``DMXGPUWorker``.

    vLLM's per-step protocol differs from HF's:
      * Tensors are packed/flattened (no batch dim; rows from each
        request are concatenated along dim 0).
      * Request IDs come from ``scheduler_output``, not auto-minted.
      * CUDA-graph dispatch may pad ``total_q`` up to the nearest
        capture size; meta shape must match the padded tensor.
      * When the ring is full, the worker falls back to eager for the
        next-determined batch (we set ``force_eager_next_batch``); the
        wrapped ``_determine_batch_execution_and_padding`` reads + clears
        the flag so the current batch runs unpadded, and the meta is
        rewritten via ``adapt_for_cpu_direct``.
    """

    def __init__(
        self, engine: Any, model_id: str, vllm_config: Any,
        *, gpu_padding_strip: bool = True,
    ) -> None:
        super().__init__(engine, model_id)
        self.vllm_config = vllm_config
        self._debug_step: bool = bool(os.environ.get("RING_DEBUG_STEP"))
        # Force-eager state lives on the active transport
        # (self.transport.force_eager).  Set by on_capacity_exceeded;
        # read + cleared by the _determine_batch_execution_and_padding
        # wrapper installed in DMXGPUWorker.init_device.
        # User opt-in; when True, ring transport stays in null mode after
        # warmup (kernels fire as no-ops, FIFO stays empty).
        self.user_wants_null_mode: bool = False
        # Stashed during build_step_context so adapt_for_cpu_direct can
        # restore the unpadded q_len without recomputing.
        self._last_total_q: int = 0
        # vLLM step counter (debug logging only).
        self._step_counter: int = 0
        # gpu_padding_strip mode: when True, the producer copies only
        # actual_q_len * row_bytes per eligible hook (instead of the
        # full padded captured tensor).  Default False = today's
        # behavior verbatim.  When True, attach_model allocates the
        # shared row_count tensor pair and wires each eligible
        # HookPoint's _strip_tensor + _strip_row_bytes; build_step_context
        # populates ctx.actual_q_len; before_forward does the per-step
        # 8-byte cudaMemcpyAsync of the new actual_q_len.
        self.gpu_padding_strip: bool = gpu_padding_strip
        self._row_count_dev: Optional[torch.Tensor] = None      # 1 int64 on GPU
        self._pinned_row_count: Optional[torch.Tensor] = None   # 1 int64 pinned host
        self._strip_eligible_hps: List[Any] = []                # for inspection

    # ------- abstract overrides ---------------------------------------

    def detect_model_shape(self, model: Any) -> ModelShapeConfig:
        # Use the vllm_config dtype (authoritative); fall back to the
        # HF config's torch_dtype if vllm_config is missing.
        vllm_dtype = getattr(
            getattr(self.vllm_config, "model_config", None), "dtype", None
        )
        cfg = _make_model_shape_from_hf_config(
            self.vllm_config.model_config.hf_config, dtype=vllm_dtype
        )
        if cfg is None:
            raise RuntimeError(
                "VLLMAdaptor.detect_model_shape: vllm_config.model_config.hf_config "
                "is missing hidden_size / num_attention_heads."
            )
        # vLLM TP world size comes from its parallel-state group (NOT
        # torch.distributed; vLLM constructs its own TP/PP groups).
        # ``_compute_hook_shape`` divides head/intermediate dims by
        # ``cfg.tp_size`` for sharded hooks (q, k, v, z, mlp_post,
        # attn_scores, pattern) -- so getting this right is load-bearing
        # for TP > 1: the producer-side meta shape must match the actual
        # sharded tensor, otherwise the drain thread rejects the row.
        from vllm.distributed.parallel_state import get_tp_group
        cfg.tp_size = max(1, get_tp_group().world_size)
        return cfg

    def detect_parallel_ranks(self) -> Tuple[int, int, int, int]:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        tp_rank = get_tp_group().rank_in_group
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        ep_rank = 0  # vLLM does not expose EP groups today; placeholder.
        pp_rank = get_pp_group().rank_in_group
        return (tp_rank, dp_rank, ep_rank, pp_rank)

    def is_pp_first(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_first_rank

    def is_pp_last(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_last_rank

    def on_capacity_exceeded(self, ctx: StepContext) -> None:
        # No-op.  transport.force_eager is owned by adaptor_base
        # before_forward (`force_eager = (result == 2) or needs_eager`).
        # Kept as a framework hook for subclasses that want to react to
        # overflow events (telemetry, custom logging, etc.).
        return

    def adapt_for_cpu_direct(self, ctx: StepContext) -> StepContext:
        # When the worker forces eager for this batch, the actual
        # tensor will not be padded -- swap meta q_len from padded
        # back to the unpadded total_q stashed during build_step_context.
        import dataclasses
        if self._last_total_q and ctx.q_len != self._last_total_q:
            return dataclasses.replace(ctx, q_len=self._last_total_q)
        return ctx

    def _warn_once_capacity(
        self, ctx: StepContext, total_bytes: int, n_hooks: int
    ) -> None:
        # Per-(total_q, num_reqs) shape warn.  Stored on adapter (not
        # transport) so multiple adapters in the same process don't
        # share state.
        shape_key = (ctx.q_len, len(ctx.req_ids))
        if shape_key in self._warned_shapes:
            return
        self._warned_shapes.add(shape_key)
        re = self.ring_engine
        if re is None:
            return
        pcap = re.payload_cap()
        scap = re.staging_cap()
        if total_bytes > pcap and total_bytes > scap:
            reason = (
                f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                f"and pinned staging ({scap / 1e6:.0f} MB)"
            )
        elif total_bytes > pcap:
            reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
        else:
            reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
        warnings.warn(
            f"[vllm_integration] Step data ({total_bytes / 1e6:.1f} MB) "
            f"{reason}. Falling back to eager dispatch + per-hook safety net "
            f"for {n_hooks} hooks.",
            stacklevel=2,
        )

    # ------- vLLM-specific helpers ------------------------------------

    def predict_padded_q_len(self, model_runner: Any, total_q: int) -> int:
        """Read vLLM's CUDA-graph capture size table and return the
        padded ``q_len`` the producer kernel will see.

        Risk surface: this reads ``cudagraph_dispatcher._bs_to_padded_graph_size``,
        a private list[int] indexed by token count.  On a vLLM upgrade,
        verify the attribute still exists and the per-step dispatch
        logic hasn't added new conditions that bypass CUDA graphs (LoRA,
        cascade, encoder, etc.).  Today the only per-step disabler is
        our own ``force_eager_next_batch``.
        """
        pad_table = getattr(
            getattr(model_runner, "cudagraph_dispatcher", None),
            "_bs_to_padded_graph_size", None,
        )
        if pad_table is not None and total_q < len(pad_table):
            return pad_table[total_q]
        return total_q

    def build_step_context(
        self, scheduler_output: Any, model_runner: Any
    ) -> Optional[StepContext]:
        """Construct the per-step ``StepContext`` from vLLM's
        scheduler_output + model_runner.

        The returned ``q_len`` is the CUDA-graph-padded total token
        count; ``adapt_for_cpu_direct`` swaps it back to ``total_q`` if
        the driver's prepare_step returns code 2 (oversize step).
        """
        total_tokens = scheduler_output.total_num_scheduled_tokens
        if total_tokens == 0:
            return None

        # Why vLLM has no post-EOS strip (unlike HFAdaptor):
        #   vLLM v1's scheduler removes finished requests from
        #   ``scheduler_output.num_scheduled_tokens`` before the next
        #   step.  The EOS-producing step's activation is real data
        #   (and is captured); subsequent steps for that request never
        #   appear in ``req_ids`` because the scheduler reassigned the
        #   slot.  No lockstep, no post-EOS noise to filter.
        #
        #   HF's batched ``generate()`` is lockstep: finished requests
        #   keep producing forward activations until the whole batch
        #   finishes or ``max_new_tokens`` hits, so HFAdaptor needs the
        #   per-request finished latch + zero-length token_range strip.
        #   Do NOT propagate that pattern here -- the scheduler is
        #   already filtering for us.

        self._step_counter += 1
        _step = self._step_counter

        num_scheduled = scheduler_output.num_scheduled_tokens  # dict[req_id, int]
        # ROW ORDER IS LOAD-BEARING. vLLM V1's model_runner lays out the flattened
        # hidden-tensor rows in INPUT_BATCH order (input_batch.req_ids), NOT in
        # num_scheduled_tokens dict order: _prepare_inputs builds
        #   num_scheduled_np = [num_scheduled[i] for i in input_batch.req_ids]
        #   req_indices = np.repeat(arange[:num_reqs], num_scheduled_np)
        # so rows are concatenated per request in input_batch order. Under
        # continuous batching the dict order diverges (freed-slot reuse/condense),
        # so iterating the dict attaches each request's metadata to ANOTHER
        # request's rows -> right metadata, WRONG activation. Order by input_batch.
        input_batch = getattr(model_runner, "input_batch", None)
        ib_req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
        if (ib_req_ids is not None and len(ib_req_ids) == len(num_scheduled)
                and all(r in num_scheduled for r in ib_req_ids)):
            req_ids = list(ib_req_ids)
        elif len(num_scheduled) <= 1:
            req_ids = list(num_scheduled.keys())  # single-req / no input_batch: order is unambiguous
        else:
            # #4: multiple requests but input_batch order is unusable -> dict
            # order would silently re-enter the L2 row<->request mismatch bug
            # (right metadata, WRONG activation). Fail loud instead.
            raise RuntimeError(
                "dmx: cannot determine input_batch row order for "
                f"{len(num_scheduled)} concurrent requests "
                f"(input_batch.req_ids={ib_req_ids!r}); refusing the dict-order "
                "fallback that mis-associates activations under continuous batching")
        num_reqs = len(req_ids)
        num_scheduled_per_req = [num_scheduled[rid] for rid in req_ids]
        total_q = total_tokens

        padded_q = self.predict_padded_q_len(model_runner, total_q)
        self._last_total_q = total_q

        # Per-request offsets and token ranges.
        computed_map: dict = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            computed_map[new_req.req_id] = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            computed_map[rid] = cached.num_computed_tokens[i]

        offset = 0
        req_id_list: List[str] = []
        token_ranges: List[Tuple[int, int]] = []
        dim0_offsets: List[int] = []
        for i in range(num_reqs):
            rid = req_ids[i]
            n = num_scheduled_per_req[i]
            pre_computed = computed_map.get(rid, 0)
            norm_id = normalize_vllm_request_id(rid)
            req_id_list.append(norm_id)
            token_ranges.append((pre_computed, pre_computed + n))
            dim0_offsets.append(offset)
            if self._debug_step:
                print(
                    f"[dmx_worker] step={_step} req[{i}] rid={norm_id} "
                    f"offset={offset} n={n} "
                    f"t_start={pre_computed} t_end={pre_computed + n} "
                    f"pre_computed={pre_computed} padded_q={padded_q}",
                    flush=True,
                )
            offset += n

        # Read input_ids dtype from model_runner buffer.  On non-first
        # PP ranks the buffer may not exist; pass None (the token_ids
        # hook is already filtered out by filter_by_pp_rank).
        ids_buf = getattr(model_runner, "input_ids", None)
        ids_dtype = ids_buf.gpu.dtype if ids_buf is not None else None

        tp_rank, dp_rank, ep_rank, pp_rank = self.detect_parallel_ranks()

        # logits_to_keep=num_reqs: vLLM's compute_logits returns one
        # logit per request shaped [num_reqs, vocab].  In flattened
        # mode _compute_hook_shape uses logits_to_keep directly as
        # dim0 (no batch dim), so the meta shape becomes
        # [num_reqs, vocab].  The p2p thread then slices row j for
        # request j and adjusts the DB token range to (end_token-1,
        # end_token) -- the single predicted position per request.
        return StepContext(
            model_id=str(self.model_id),
            flattened=True,
            req_ids=req_id_list,
            token_ranges=token_ranges,
            dim0_offsets=dim0_offsets,
            kv_offsets=[0] * num_reqs,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            ep_rank=ep_rank,
            pp_rank=pp_rank,
            batch=0,
            q_len=padded_q,
            kv_dim=0,
            logits_to_keep=num_reqs,
            token_ids_dtype=ids_dtype,
            # In gpu_padding_strip mode, eligible specs use this for shape
            # + reservation; non-eligible specs and gpu_padding_strip=False
            # ignore (None).
            actual_q_len=(total_q if self.gpu_padding_strip else None),
        )

    # ----- gpu_padding_strip integration -----

    def attach_model(self, model: Any, hook_selection: str = "full") -> None:
        """Standard attach, plus gpu_padding_strip pool setup when enabled.

        When `gpu_padding_strip=True`, allocate one shared int64[1] device
        tensor (`_row_count_dev`) and one pinned-host counterpart.  For
        every active spec with `dim0_is_actual_tokens=True`, point the
        HookPoint's `_strip_tensor` at the shared tensor and bake its
        per-spec `_strip_row_bytes` (CPU-known constant).  Every step
        we update the shared tensor's value in place (see
        before_forward); each HookPoint's captured producer_prefix
        call reads the freshly-written value and multiplies by its
        baked row_bytes.
        """
        super().attach_model(model, hook_selection)
        if not self.gpu_padding_strip:
            return
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"
        self._row_count_dev    = torch.empty(1, dtype=torch.int64, device=device)
        self._pinned_row_count = torch.empty(1, dtype=torch.int64, pin_memory=True)
        for spec in self.active_specs:
            if not spec.dim0_is_actual_tokens:
                continue
            hp = spec.module
            rb = _row_bytes_for_spec(spec, self.model_cfg)
            if rb <= 0:
                continue
            hp._strip_tensor    = self._row_count_dev
            hp._strip_row_bytes = rb
            self._strip_eligible_hps.append(hp)

    def before_forward(self, *raw) -> None:
        """Standard driver, plus per-step memcpy when gpu_padding_strip=True."""
        super().before_forward(*raw)
        if (self.gpu_padding_strip
                and self._pinned_row_count is not None
                and self._row_count_dev is not None
                and self._last_total_q > 0):
            # CPU: write the current step's actual_q_len to pinned host.
            # GPU: enqueue an async copy to the shared device scalar on
            # the model stream (captureable; values propagate to replays).
            self._pinned_row_count[0] = self._last_total_q
            self._row_count_dev.copy_(self._pinned_row_count, non_blocking=True)


def _row_bytes_for_spec(spec, model_cfg) -> int:
    """Bytes-per-token for `spec`'s shape under vLLM flat layout.

    Computed as prod(shape_excluding_total_tokens) * elem_size where
    the shape comes from _compute_hook_shape with batch=0, q_len=1.
    The "1" stands in for "one token's worth"; the result is the
    per-token byte stride.  Returns 0 if the spec produces an empty
    shape (skip).
    """
    shape = _compute_hook_shape(
        spec.hook_type, model_cfg, batch=0, q_len=1, kv_dim=0,
        logits_to_keep=0,
    )
    if not shape:
        return 0
    # In flat mode, dim-0 of the spec's shape IS the token count.
    # shape[0] should be 1 here (since we passed q_len=1); the
    # remaining dims times elem_size = bytes per token.
    nelem = 1
    for d in shape[1:]:
        nelem *= d
    dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
    elem_size = torch._utils._element_size(dtype)
    return int(nelem) * int(elem_size)


# ---------------------------------------------------------------------------
# DMXGPUWorker
# ---------------------------------------------------------------------------


class DMXGPUWorker(Worker):
    """vLLM ``Worker`` subclass that owns a ``VLLMAdaptor`` and
    delegates per-step work to it.

    Lifecycle:
      ``init_device``           -- super + build engine + adaptor + null mode +
                                    wrap _determine_batch_execution_and_padding.
      ``load_model``            -- arch remap + super + adaptor.attach_model.
      ``compile_or_warm_up_model`` -- super + clear null_mode (unless user
                                       opted-in via ``dmx_null_mode``).
      ``execute_model``         -- adaptor.before_forward + super.
      ``stop_monitoring``       -- adaptor.close (CUDA sync, ring stop,
                                    deactivate transport, host engine stop).
      ``shutdown``              -- best-effort stop_monitoring + super.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adaptor: Optional[VLLMAdaptor] = None
        self._dmx_host_engine: Any = None
        self._dmx_hook_selection: str = "vllm-full"
        # callable-sink (e.g. ProbeWorker) + per-request auto-finalize
        self._dmx_callable_sink: Any = None
        self._dmx_auto_finalize: bool = False
        # deferred finalize: a request's final-token slice may still be in the
        # ring when vLLM reports it finished. We barrier it to the sink (C1
        # drain_to_sink_and_wait) before finalizing so a "final"-pooled probe
        # scores the true last token. _dmx_pending_finalize maps req_id -> the
        # remaining step budget before we give up and best-effort finalize the
        # request as valid=False (so its verdict is visible, never silently lost).
        self._dmx_pending_finalize: dict = {}
        # ms the C1 barrier waits for a request's tail to reach the sink (0 = wait
        # forever -- avoid; a finite timeout keeps execute_model bounded under a
        # stalled sink). Steps to retry the barrier before giving up valid=False.
        self._dmx_finalize_drain_timeout_ms: int = 200
        self._dmx_finalize_max_steps: int = 8
        # B3 serving-cost telemetry for the finalize barrier (read via
        # dmx_sink_stats, published to Grafana at shutdown): cumulative count of
        # barrier timeouts + a bounded ring of recent barrier wait-times (s).
        self._dmx_finalize_timeouts: int = 0
        self._dmx_finalize_waits: list = []
        self._DMX_WAIT_CAP: int = 512

    def init_device(self) -> None:
        super().init_device()

        from monitoring import _native_engine as _ne
        from monitoring.engine import MonitoringEngine

        ac = self.vllm_config.additional_config
        if not isinstance(ac, dict):
            ac = {}

        self._dmx_hook_selection = _cfg(
            ac, "dmx_hook_selection", "DMX_HOOK_SELECTION", "vllm-full"
        )
        model_id = _cfg(ac, "dmx_model_id", "DMX_MODEL_ID", "")
        ring_payload_mb = _cfg(ac, "dmx_ring_payload_mb", "DMX_RING_PAYLOAD_MB", 4096)
        ring_pinned_mb = _cfg(ac, "dmx_ring_pinned_mb", "DMX_RING_PINNED_MB", 4096)
        ring_entries = _cfg(ac, "dmx_ring_task_entries", "DMX_RING_TASK_ENTRIES", 65536)
        null_mode = _cfg(ac, "dmx_null_mode", "DMX_NULL_MODE", False)
        db_host = _cfg(ac, "dmx_db_host", "DMX_DB_HOST", "")
        db_port = _cfg(ac, "dmx_db_port", "DMX_DB_PORT", 9000)
        db_database = _cfg(ac, "dmx_db_database", "DMX_DB_DATABASE", "default")
        db_table = _cfg(ac, "dmx_db_table", "DMX_DB_TABLE", "offload")
        ch_parallelism = int(
            _cfg(ac, "dmx_ch_parallelism", "DMX_CH_PARALLELISM", 10)
        )

        resolved_model_id = model_id or str(self.vllm_config.model_config.model)

        # Host engine: ClickHouse DMXHostEngine, or an in-process callable sink
        # (hallu_monitor ProbeWorker) registered via register_sink_factory().
        # db_host takes precedence; if neither is set the ring runs sink-less.
        host_engine = None
        use_callable_sink = (not db_host) and _DMX_SINK_FACTORY is not None
        # #5: fail loud when a callable sink is REQUIRED but none is registered
        # (e.g. forgot VLLM_ENABLE_V1_MULTIPROCESSING=0 so the module-global
        # factory isn't shared into this worker process) -- otherwise the ring
        # runs sink-less and silently produces no scores.
        require_callable_sink = bool(_cfg(
            ac, "dmx_require_callable_sink", "DMX_REQUIRE_CALLABLE_SINK", False))
        if require_callable_sink and not db_host and _DMX_SINK_FACTORY is None:
            raise RuntimeError(
                "dmx_require_callable_sink set but no sink factory registered "
                "(register_sink_factory + VLLM_ENABLE_V1_MULTIPROCESSING=0?)")
        if os.environ.get("DMX_DEBUG_SINK"):
            print(f"[dmx-debug] init_device: db_host={db_host!r} "
                  f"factory_set={_DMX_SINK_FACTORY is not None} "
                  f"use_callable_sink={use_callable_sink} "
                  f"hook_selection={self._dmx_hook_selection!r}", flush=True)
        if db_host:
            ch_cfg = _ne.ClickHouseClientConfig()
            ch_cfg.host = db_host
            ch_cfg.port = db_port
            ch_cfg.database = db_database
            ch_cfg.table = db_table
            ch_cfg.create_database_if_missing = True
            stage_cfg = _ne.StageConfig.clickhouse_insert(
                ch_cfg, parallelism=ch_parallelism, name="clickhouse_insert"
            )
            q = stage_cfg.input_queue
            q.max_batch_items = int(_cfg(
                ac, "dmx_ch_max_batch_items", "DMX_CH_MAX_BATCH_ITEMS", 1024))
            q.high_watermark_items = q.max_batch_items
            q.max_batch_size = int(_cfg(
                ac, "dmx_ch_max_batch_bytes", "DMX_CH_MAX_BATCH_BYTES",
                2048 * 1024 * 1024))
            q.high_watermark_size = q.max_batch_size
            host_engine = _ne.DMXHostEngine(stage_cfg)
            host_engine.start()
            self._dmx_host_engine = host_engine
        elif use_callable_sink:
            # NOT stored in self._dmx_host_engine: a callable sink is started/
            # stopped by whoever registered it, not by stop_monitoring().
            host_engine = _DMX_SINK_FACTORY()
            if host_engine is None:
                raise RuntimeError(
                    "registered dmx sink factory returned None")
            self._dmx_callable_sink = host_engine
            # auto-finalize each request when vLLM reports it finished, so
            # per-request ("final"/pooled) probes score at the response's last
            # token under continuous batching (no manual finalize_all needed).
            # Only meaningful for a finalize-capable sink (e.g. ProbeWorker).
            self._dmx_auto_finalize = bool(_cfg(
                ac, "dmx_auto_finalize", "DMX_AUTO_FINALIZE", True)) and \
                hasattr(host_engine, "finalize_request")
            # C1 finalize-barrier knobs (see _dmx_finalize_finished).
            self._dmx_finalize_drain_timeout_ms = int(_cfg(
                ac, "dmx_finalize_drain_timeout_ms",
                "DMX_FINALIZE_DRAIN_TIMEOUT_MS", 200))
            self._dmx_finalize_max_steps = int(_cfg(
                ac, "dmx_finalize_max_steps", "DMX_FINALIZE_MAX_STEPS", 8))

        ring_cfg = _ne.RingConfig()
        ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
        ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
        ring_cfg.task_ring_entries = ring_entries
        # The async ProbeWorker reads each slice later on another thread, so the
        # backend must hand it an OWNED tensor (not aliasing reused staging).
        if use_callable_sink:
            ring_cfg.clone_slices = True
        # Periodic drain flush (us). 0 = legacy (flush only when the ring fills
        # or on stop) -- fine for batch ClickHouse, but a live in-memory consumer
        # needs slices to keep arriving during serving, not just at shutdown.
        # Default 50ms for the callable sink, 0 otherwise.
        ring_cfg.drain_flush_timeout_us = int(_cfg(
            ac, "dmx_drain_flush_timeout_us", "DMX_DRAIN_FLUSH_TIMEOUT_US",
            50000 if use_callable_sink else 0))
        ring_cfg.insert_queue_max_bytes = int(_cfg(
            ac, "dmx_insert_queue_max_bytes", "DMX_INSERT_QUEUE_MAX_BYTES",
            4096 * 1024 * 1024))
        ring_cfg.insert_queue_max_items = int(_cfg(
            ac, "dmx_insert_queue_max_items", "DMX_INSERT_QUEUE_MAX_ITEMS",
            65536))

        # MonitoringEngine + ring transport.
        engine = MonitoringEngine(
            config=None,
            model_id=resolved_model_id,
            host_engine=host_engine,
            ring_config=ring_cfg,
        )

        # Build the adaptor.  Hooks aren't installed yet -- that
        # happens in load_model after the model is materialized.
        # gpu_padding_strip is on by default; flip it off via
        # additional_config["dmx_gpu_padding_strip"]=False or
        # DMX_GPU_PADDING_STRIP=0 if needed for debugging.
        gpu_padding_strip = _cfg(
            ac, "dmx_gpu_padding_strip", "DMX_GPU_PADDING_STRIP", True)
        self.adaptor = VLLMAdaptor(
            engine, resolved_model_id, self.vllm_config,
            gpu_padding_strip=bool(gpu_padding_strip))
        self.adaptor.user_wants_null_mode = bool(null_mode)
        if null_mode:
            self.adaptor.transport.null_offload = True

        # Enable null mode for warmup so producer kernels fire (CUDA
        # graph capture needs them) but no-op on the data path.  Cleared
        # after warmup in compile_or_warm_up_model unless the user
        # explicitly opted in to permanent null mode.
        self.adaptor.ring_engine.set_null_mode(True)

        # Wrap _determine_batch_execution_and_padding so the adapter
        # can request eager for the current batch via on_capacity_exceeded.
        adaptor = self.adaptor
        orig_fn = self.model_runner._determine_batch_execution_and_padding

        def _wrapped_determine(*args: Any, **kwargs: Any) -> Any:
            # Read-only.  transport.force_eager is owned by
            # before_forward (per-batch reassignment); clearing here
            # would hide this batch's force_eager from HookPoint.forward
            # since the dispatch wrapper fires BEFORE the model forward.
            if adaptor.transport.force_eager:
                kwargs["force_eager"] = True
            return orig_fn(*args, **kwargs)

        self.model_runner._determine_batch_execution_and_padding = _wrapped_determine

        # Backwards-compat attributes for external subclasses that read
        # the pre-refactor names (e.g. tests/compare_worker.py reads
        # `self._dmx_tp_rank` + `self._dmx_tp_size` to format per-rank
        # filenames).
        from vllm.distributed.parallel_state import get_tp_group
        tp_rank, dp_rank, ep_rank, pp_rank = self.adaptor.detect_parallel_ranks()
        self._dmx_tp_rank = tp_rank
        self._dmx_dp_rank = dp_rank
        self._dmx_ep_rank = ep_rank
        self._dmx_pp_rank = pp_rank
        self._dmx_tp_size = get_tp_group().world_size

    def load_model(self) -> None:
        # Remap the architecture string in the HF config so vLLM's
        # registry resolves to the hooked variant (Qwen3PForCausalLM
        # etc.).  Mutates vllm_config in place.
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model()

        # Now the model is materialized; install hooks via the adapter.
        if self.adaptor is None:
            return
        self.adaptor.attach_model(
            self.model_runner.model, hook_selection=self._dmx_hook_selection
        )

        # The step context (per-request row layout) MUST be built from the
        # POST-_update_states input_batch order. before_forward used to run in
        # execute_model BEFORE super().execute_model -> before the model_runner's
        # _update_states (remove_request/add_request/condense), so it saw the
        # PREVIOUS step's input_batch order while the forward lays rows out in
        # THIS step's order -> per-request metadata attached to the wrong row
        # under continuous batching. Wrap _update_states so the step context is
        # built right after it (input_batch now correct) and before the forward.
        mr = self.model_runner
        adaptor = self.adaptor
        _orig_update_states = mr._update_states

        def _dmx_update_states(scheduler_output):
            _orig_update_states(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens > 0:
                adaptor.before_forward(scheduler_output, mr)

        mr._update_states = _dmx_update_states

    def compile_or_warm_up_model(self) -> float:
        # Warmup runs with null_mode=True (set in init_device).
        # Producer kernels fire but are no-ops -- ring stays clean.
        result = super().compile_or_warm_up_model()

        # Warmup done.  Turn off null mode unless the user explicitly
        # asked for permanent null mode.  set_null_mode does
        # cudaDeviceSynchronize internally so it's safe to call here.
        if (
            self.adaptor is not None
            and not self.adaptor.user_wants_null_mode
            and self.adaptor.ring_engine is not None
        ):
            self.adaptor.ring_engine.set_null_mode(False)

        return result

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        # NOTE: before_forward is now driven from a _update_states wrapper
        # installed in load_model (so the step context is built from the
        # POST-update input_batch row order), NOT here.
        out = super().execute_model(scheduler_output)
        if self._dmx_auto_finalize:
            self._dmx_finalize_finished(scheduler_output)
        return out

    def _dmx_finalize_finished(self, scheduler_output: Any) -> None:
        """Finalize each finished request at its true final token via the C1
        barrier.

        vLLM reports a request finished, but its final-token decode slice may
        still be in the ring (GPU->p2p->sink lag). Finalizing immediately makes a
        "final"-pooled probe score an EARLIER (already-delivered) token. So once
        any request is pending we drive `drain_to_sink_and_wait` (C1): it flushes
        the GPU ring AND blocks until every flushed slice has reached the Python
        sink, so the worker is guaranteed to see [...all this req's slices,
        finalize] in order. When the barrier reports delivered we finalize all
        pending requests; on timeout we retry next step up to
        `_dmx_finalize_max_steps`, then best-effort finalize as valid=False so a
        verdict is never silently lost. A sink error (C0) -> same valid=False
        fail-safe immediately. Requests finishing in the last steps are swept in
        stop_monitoring."""
        sink = self._dmx_callable_sink
        finalize = getattr(sink, "finalize_request", None)
        if finalize is None:
            return
        fin = getattr(scheduler_output, "finished_req_ids", None)
        if fin:
            for rid in fin:
                nrid = normalize_vllm_request_id(rid)
                self._dmx_pending_finalize.setdefault(
                    nrid, self._dmx_finalize_max_steps)
        if not self._dmx_pending_finalize:
            return
        t0 = time.monotonic()
        status = self.dmx_drain_to_sink(self._dmx_finalize_drain_timeout_ms)
        self._dmx_record_finalize_wait(time.monotonic() - t0, timed_out=(status == 1))
        delivered = (status == 0)
        sink_failed = (status == 2)
        for rid in list(self._dmx_pending_finalize):
            if delivered:
                self._dmx_finalize_one(finalize, rid, valid=True)
                continue
            # not delivered: count down the retry budget; sink error gives up now
            self._dmx_pending_finalize[rid] -= 1
            if sink_failed or self._dmx_pending_finalize[rid] <= 0:
                self._dmx_finalize_one(finalize, rid, valid=False)

    def _dmx_finalize_one(self, finalize: Any, rid: str, valid: bool) -> None:
        """Finalize one request, dropping it from the pending set only AFTER the
        finalize is accepted. `valid=False` flags a best-effort finalize (barrier
        timed out / sink failed) so the verdict is visibly incomplete rather than
        silently trusted. If the finalize ENQUEUE itself fails, we keep the
        request pending so it's retried next step / in the stop sweep -- never
        silently dropped (preserves the "verdict never lost" invariant)."""
        try:
            try:
                finalize(rid, valid=valid)
            except TypeError:
                finalize(rid)  # sink without the valid kwarg (older ProbeWorker)
        except Exception as e:
            print(f"[dmx] finalize_request({rid}) raised: {e}; keeping pending "
                  "for retry", flush=True)
            return
        self._dmx_pending_finalize.pop(rid, None)

    def _dmx_record_finalize_wait(self, dt: float, timed_out: bool) -> None:
        """B3: accumulate finalize-barrier serving-cost telemetry."""
        if timed_out:
            self._dmx_finalize_timeouts += 1
        w = self._dmx_finalize_waits
        w.append(float(dt))
        if len(w) > self._DMX_WAIT_CAP:
            del w[:len(w) - self._DMX_WAIT_CAP]

    def _dmx_finalize_sweep(self) -> None:
        """Finalize every still-pending request before teardown. Barrier hard so
        their tails are delivered; whatever still hasn't arrived is finalized
        valid=False (visible, not lost)."""
        sink = self._dmx_callable_sink
        finalize = getattr(sink, "finalize_request", None)
        if finalize is None or not self._dmx_pending_finalize:
            return
        status = self.dmx_drain_to_sink(0)  # wait for the tail this last time
        delivered = (status == 0)
        for rid in list(self._dmx_pending_finalize):
            self._dmx_finalize_one(finalize, rid, valid=delivered)

    def dmx_drain_to_sink(self, timeout_ms: int = 0) -> int:
        """C1 barrier: flush the GPU ring AND block until every flushed slice has
        reached the Python sink. Returns 0=delivered, 1=timeout, 2=sink-error
        (C0), -1=no engine / method unavailable. Falls back to the GPU-only
        flush_and_wait if the .so predates drain_to_sink_and_wait."""
        if self.adaptor is None or self.adaptor.ring_engine is None:
            return -1
        re = self.adaptor.ring_engine
        drain = getattr(re, "drain_to_sink_and_wait", None)
        if drain is None:
            try:
                re.flush_and_wait()
            except Exception:
                pass
            return -1
        # PER-BARRIER sink-error detection: compare the cumulative exception
        # count across THIS drain, not against zero. submit_exceptions never
        # resets, so testing `> 0` would make one transient SubmitFn failure
        # poison every later finalize. A latched sink_failed (opt-in via
        # set_abort_on_sink_error) is the explicit fail-closed policy and is
        # honored permanently.
        exc_before = int(getattr(re, "submit_exceptions", 0) or 0)
        status = drain(int(timeout_ms))
        exc_after = int(getattr(re, "submit_exceptions", 0) or 0)
        if bool(getattr(re, "sink_failed", False)) or exc_after > exc_before:
            return 2
        return int(status)

    def dmx_flush(self) -> None:
        """Force the ring to drain all pending slices to the sink, NOW, without
        stopping. Barriers through the SubmitFn (C1) so an in-memory consumer
        deterministically has every produced slice on return, not just the
        GPU->p2p handoff. Call via collective_rpc('dmx_flush')."""
        self.dmx_drain_to_sink(self._dmx_finalize_drain_timeout_ms)

    def dmx_sink_stats(self) -> dict:
        """C0/B3 observability: cumulative SubmitFn-exception count + last error +
        latched-failed flag, plus the finalize-barrier serving-cost telemetry
        (timeout count + recent wait-times). Call via
        collective_rpc('dmx_sink_stats')."""
        re = self.adaptor.ring_engine if self.adaptor is not None else None
        base = {
            "submit_exceptions": 0, "last_sink_error": "", "sink_failed": False,
            "finalize_timeouts": int(self._dmx_finalize_timeouts),
            "finalize_waits": list(self._dmx_finalize_waits),
        }
        if re is None:
            return base
        base.update({
            "submit_exceptions": int(getattr(re, "submit_exceptions", 0) or 0),
            "last_sink_error": str(getattr(re, "last_sink_error", "") or ""),
            "sink_failed": bool(getattr(re, "sink_failed", False)),
        })
        return base

    def stop_monitoring(self) -> None:
        """Flush and stop DMI engine.  Reentrant: second call no-ops."""
        if self._dmx_auto_finalize:
            self._dmx_finalize_sweep()   # finalize last-step requests before teardown
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.adaptor is not None:
            engine = self.adaptor.engine
            ring_engine = self.adaptor.ring_engine
            if ring_engine is not None:
                try:
                    ring_engine.stop()
                except Exception:
                    pass
            try:
                ring_transport.deactivate()
            except Exception:
                pass
            try:
                engine.close()
            except Exception:
                pass
            self.adaptor = None

        if self._dmx_host_engine is not None:
            try:
                self._dmx_host_engine.stop()
            except Exception:
                pass
            self._dmx_host_engine = None

    def shutdown(self) -> None:
        import logging
        if self.adaptor is not None:
            logging.getLogger(__name__).warning(
                "DMI engine not explicitly stopped before shutdown. "
                "Data may be incomplete. Call stop_monitoring() first."
            )
        # Best-effort flush (may be killed by vLLM's 8 s deadline).
        self.stop_monitoring()
        super().shutdown()


__all__ = [
    "VLLMAdaptor",
    "DMXGPUWorker",
    "normalize_vllm_request_id",
]
