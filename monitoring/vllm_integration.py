"""vLLM integration: monitoring-aware GPU worker.

Always uses ring transport.  No legacy native backend path.

Usage:
  vllm serve Qwen/Qwen3-8B \
      --worker-cls monitoring.vllm_integration.DMXGPUWorker \
      --additional-config '{
          "dmx_hook_selection": "vllm-full",
          "dmx_ring_payload_mb": 4096,
          "dmx_ring_pinned_mb": 4096,
          "dmx_db_host": "localhost",
          "dmx_db_port": 9000
      }'
"""
from __future__ import annotations

import os
import re
from typing import Any

import torch

from vllm.v1.worker.gpu_worker import Worker


# ---------------------------------------------------------------------------
# Config: additional_config preferred, env var fallback
# ---------------------------------------------------------------------------

def _cfg(ac: dict, key: str, env_key: str, default: Any) -> Any:
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
# Request ID normalization
# ---------------------------------------------------------------------------

_VLLM_REQ_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def normalize_vllm_request_id(req_id: str) -> str:
    """Strip the UUID suffix that vLLM V1 appends to request IDs."""
    return _VLLM_REQ_ID_SUFFIX.sub("", req_id)


# ---------------------------------------------------------------------------
# PP rank filtering
# ---------------------------------------------------------------------------

def filter_by_pp_rank(specs: list, is_first_rank: bool, is_last_rank: bool) -> list:
    from .ring_transport import (
        HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_EMBED, HOOK_TYPE_POS_EMBED,
        HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS,
    )
    first_only = {HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_EMBED, HOOK_TYPE_POS_EMBED}
    last_only = {HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS}

    filtered = []
    for s in specs:
        if s.hook_type in first_only and not is_first_rank:
            s.module.enabled = False
            continue
        if s.hook_type in last_only and not is_last_rank:
            s.module.enabled = False
            continue
        filtered.append(s)
    return filtered


# ---------------------------------------------------------------------------
# TP rank filtering — skip unsharded hooks on non-zero ranks
# ---------------------------------------------------------------------------

_TP_SHARDED_TYPES: set = set()  # populated lazily

def _get_tp_sharded_types() -> set:
    if not _TP_SHARDED_TYPES:
        from .ring_transport import (
            HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z,
            HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN, HOOK_TYPE_MLP_POST,
        )
        _TP_SHARDED_TYPES.update({
            HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z,
            HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN, HOOK_TYPE_MLP_POST,
        })
    return _TP_SHARDED_TYPES

def filter_by_tp_rank(specs: list, tp_rank: int) -> list:
    """On non-zero TP ranks, keep only sharded hooks to avoid N× duplicate
    writes of identical unsharded data.  Rank 0 keeps all hooks."""
    if tp_rank == 0:
        return specs
    sharded = _get_tp_sharded_types()
    filtered = []
    for s in specs:
        if s.hook_type not in sharded:
            s.module.enabled = False
            continue
        filtered.append(s)
    return filtered


# ---------------------------------------------------------------------------
# DMXGPUWorker
# ---------------------------------------------------------------------------

class DMXGPUWorker(Worker):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dmx_model_id: str = ""
        self._dmx_tp_rank: int = 0
        self._dmx_dp_rank: int = 0
        self._dmx_ep_rank: int = 0
        self._dmx_pp_rank: int = 0
        self._dmx_force_eager: bool = False
        self._dmx_ring_engine: Any = None
        self._dmx_host_engine: Any = None

    def init_device(self) -> None:
        super().init_device()

        from . import ring_transport
        from .ring_transport import RingTransport
        from . import _native_engine as _ne

        ac = self.vllm_config.additional_config
        if not isinstance(ac, dict):
            ac = {}

        self._dmx_hook_selection = _cfg(ac, "dmx_hook_selection", "DMX_HOOK_SELECTION", "vllm-full")
        model_id        = _cfg(ac, "dmx_model_id",         "DMX_MODEL_ID",         "")
        ring_payload_mb = _cfg(ac, "dmx_ring_payload_mb",  "DMX_RING_PAYLOAD_MB",  4096)
        ring_pinned_mb  = _cfg(ac, "dmx_ring_pinned_mb",   "DMX_RING_PINNED_MB",   4096)
        ring_entries    = _cfg(ac, "dmx_ring_task_entries", "DMX_RING_TASK_ENTRIES", 65536)
        null_mode       = _cfg(ac, "dmx_null_mode",        "DMX_NULL_MODE",        False)
        db_host         = _cfg(ac, "dmx_db_host",          "DMX_DB_HOST",          "")
        db_port         = _cfg(ac, "dmx_db_port",          "DMX_DB_PORT",          9000)
        db_database     = _cfg(ac, "dmx_db_database",      "DMX_DB_DATABASE",      "default")
        db_table        = _cfg(ac, "dmx_db_table",         "DMX_DB_TABLE",         "offload")
        ch_parallelism  = int(_cfg(ac, "dmx_ch_parallelism", "DMX_CH_PARALLELISM",  10))

        self._dmx_model_id = model_id or str(self.vllm_config.model_config.model)

        # Ranks
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        self._dmx_tp_rank = get_tp_group().rank_in_group
        self._dmx_dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self._dmx_ep_rank = 0
        self._dmx_pp_rank = get_pp_group().rank_in_group

        # Host engine (ClickHouse)
        host_engine = None
        if db_host:
            ch_cfg = _ne.ClickHouseClientConfig()
            ch_cfg.host = db_host
            ch_cfg.port = db_port
            ch_cfg.database = db_database
            ch_cfg.table = db_table
            ch_cfg.create_database_if_missing = True
            stage_cfg = _ne.StageConfig.clickhouse_insert(
                ch_cfg, parallelism=ch_parallelism, name="clickhouse_insert")
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

        # Ring engine
        ring_cfg = _ne.RingConfig()
        ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
        ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
        ring_cfg.task_ring_entries = ring_entries
        ring_cfg.insert_queue_max_bytes = int(_cfg(
            ac, "dmx_insert_queue_max_bytes", "DMX_INSERT_QUEUE_MAX_BYTES",
            4096 * 1024 * 1024))
        ring_cfg.insert_queue_max_items = int(_cfg(
            ac, "dmx_insert_queue_max_items", "DMX_INSERT_QUEUE_MAX_ITEMS",
            65536))

        ring_engine = _ne.RingEngine(ring_cfg, host_engine)
        ring_engine.init()
        ring_engine.start()
        self._dmx_ring_engine = ring_engine

        transport = RingTransport(ring_engine)
        self._dmx_null_mode_user = null_mode
        if null_mode:
            transport.null_offload = True
        # Enable null_mode for warmup/profiling: producer kernels fire (needed
        # for CUDA graph capture) but no-op on the data path so the ring stays
        # clean.  Turned off after warmup in compile_or_warm_up_model.
        # set_null_mode internally does cudaDeviceSynchronize before/after to
        # avoid racing with non-blocking compute streams.
        ring_engine.set_null_mode(True)
        ring_transport.activate(transport)

        # Wrap model_runner for force_eager
        orig_fn = self.model_runner._determine_batch_execution_and_padding

        def _wrapped_determine(*args: Any, **kwargs: Any) -> Any:
            if self._dmx_force_eager:
                kwargs["force_eager"] = True
                self._dmx_force_eager = False
            return orig_fn(*args, **kwargs)

        self.model_runner._determine_batch_execution_and_padding = _wrapped_determine

    def load_model(self) -> None:
        # Remap HF architecture to our hooked variant so vLLM's registry
        # resolves to the model with HookPoints (e.g. GPT2PLMHeadModel).
        _ARCH_REMAP = {
            "GPT2LMHeadModel": "GPT2PLMHeadModel",
            "Qwen3ForCausalLM": "Qwen3PForCausalLM",
        }
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model()

        # Model is now available — set shape config and install hooks.
        # Hooks must be installed BEFORE warmup so CUDA graph capture
        # includes the producer kernel. Null mode is active (set in
        # init_device) so warmup producer calls are no-ops.
        from . import ring_transport
        from .ring_transport import apply_hook_selection, install_ring_hooks
        from .generate import _make_model_shape

        transport = ring_transport.get_active()
        if transport is None:
            return

        model = self.model_runner.model
        model_shape = _make_model_shape(model)
        if model_shape is not None:
            # _make_model_shape reads model.dtype which is unavailable on
            # vLLM model classes.  Override with the authoritative dtype
            # from vllm_config so meta shapes use the correct element size.
            model_shape.dtype = self.vllm_config.model_config.dtype
            # TP: set world size so _compute_hook_shape divides sharded dims.
            from vllm.distributed.parallel_state import get_tp_group
            model_shape.tp_size = get_tp_group().world_size
            transport.set_model_cfg(model_shape)

        if hasattr(model, "get_hook_specs"):
            all_specs = model.get_hook_specs()
            active_specs = apply_hook_selection(all_specs, self._dmx_hook_selection, cfg=model_shape)

            from vllm.distributed.parallel_state import get_pp_group
            pp_group = get_pp_group()
            active_specs = filter_by_pp_rank(
                active_specs,
                is_first_rank=pp_group.is_first_rank,
                is_last_rank=pp_group.is_last_rank,
            )

            # TP: non-zero ranks only keep sharded hooks to avoid N×
            # duplicate writes of identical unsharded data.
            active_specs = filter_by_tp_rank(active_specs, self._dmx_tp_rank)

            install_ring_hooks(active_specs)
            transport._active_specs = active_specs
            transport._using_forward_hooks = True

    def compile_or_warm_up_model(self) -> float:
        # Warmup runs with null_mode=True (set in init_device).
        # Producer kernels fire but are no-ops — ring stays clean.
        # We use null_mode (not HookPoint.enabled=False) because CUDA graph
        # capture needs the producer kernel to be launched during warmup so
        # it gets baked into the graph.  null_mode makes the kernel a no-op
        # on the data path while still being captured.
        result = super().compile_or_warm_up_model()

        # Warmup done. Turn off null mode (unless user explicitly wants it).
        # set_null_mode internally does cudaDeviceSynchronize before/after
        # to avoid racing with non-blocking compute streams.
        if not self._dmx_null_mode_user and self._dmx_ring_engine is not None:
            self._dmx_ring_engine.set_null_mode(False)

        return result

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        from . import ring_transport
        from .ring_transport import align_up_py, _compute_hook_shape

        total_tokens = scheduler_output.total_num_scheduled_tokens
        if not hasattr(self, '_dmx_step_counter'):
            self._dmx_step_counter = 0
        self._dmx_step_counter += 1
        _step = self._dmx_step_counter
        transport = ring_transport.get_active()

        if total_tokens == 0 or transport is None or transport.null_offload:
            return super().execute_model(scheduler_output)

        # Read batch info from scheduler_output (not input_batch, which
        # may not be populated yet for new requests at prefill time).
        num_scheduled = scheduler_output.num_scheduled_tokens  # dict[req_id, int]
        req_ids = list(num_scheduled.keys())
        num_reqs = len(req_ids)
        num_scheduled_per_req = list(num_scheduled.values())
        total_q = total_tokens

        # --- CUDA graph padding prediction ---
        # vLLM pads total_tokens to the nearest CUDA graph capture size.
        # The producer kernel sees the PADDED tensor, so both capacity
        # check and meta shapes must use the padded count.
        #
        # RISK: We read cudagraph_dispatcher._bs_to_padded_graph_size,
        # an internal list[int] indexed by token count.  This assumes:
        #   (a) The list exists and is populated after warmup.
        #   (b) For in-range tokens, if we don't set force_eager,
        #       CUDA graph WILL be dispatched and padding WILL happen.
        #       The only per-step condition that disables CUDA graph is
        #       our own force_eager (from cpu_direct).  Other conditions
        #       (LoRA, cascade, encoder) are session-level.
        # CHECK ON VLLM UPGRADE: verify _bs_to_padded_graph_size still
        # exists and the per-step dispatch logic hasn't added new
        # conditions that bypass CUDA graphs.
        padded_q = total_q
        pad_table = getattr(
            getattr(self.model_runner, 'cudagraph_dispatcher', None),
            '_bs_to_padded_graph_size', None)
        if pad_table is not None and total_q < len(pad_table):
            padded_q = pad_table[total_q]

        # Capacity check — use padded_q (matches real tensor size).
        cfg = transport._model_cfg
        is_cpu_direct = False
        if cfg is not None and transport._active_specs:
            step_bytes = 0
            n_hooks = 0
            for spec in transport._active_specs:
                shape = _compute_hook_shape(
                    spec.hook_type, cfg, batch=0, q_len=padded_q, kv_dim=0,
                    logits_to_keep=num_reqs)
                if not shape:
                    continue
                dtype = spec.dtype if spec.dtype is not None else cfg.dtype
                elem_size = torch._utils._element_size(dtype)
                nbytes = elem_size
                for d in shape:
                    nbytes *= d
                step_bytes += align_up_py(nbytes, 16)
                n_hooks += 1

            if n_hooks > 0:
                result = transport._ring_engine.prepare_step(step_bytes, n_hooks)
                is_cpu_direct = (result == 2)
                transport.cpu_direct = is_cpu_direct
                ring_transport.set_cpu_direct(is_cpu_direct)
                if is_cpu_direct:
                    self._dmx_force_eager = True
                    # Warn once per (total_q, num_reqs) shape
                    shape_key = (total_q, num_reqs)
                    if not hasattr(transport, '_warned_shapes'):
                        transport._warned_shapes = set()
                    if shape_key not in transport._warned_shapes:
                        transport._warned_shapes.add(shape_key)
                        import warnings
                        re = transport._ring_engine
                        pcap = re.payload_cap()
                        scap = re.staging_cap()
                        if step_bytes > pcap and step_bytes > scap:
                            reason = (f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                                      f"and pinned staging ({scap / 1e6:.0f} MB)")
                        elif step_bytes > pcap:
                            reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
                        else:
                            reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
                        warnings.warn(
                            f"[vllm_integration] Step data ({step_bytes / 1e6:.1f} MB) "
                            f"{reason}. Falling back to cpu_direct for {n_hooks} hooks.",
                            stacklevel=2,
                        )

        # Meta q_len: if cpu_direct, we set force_eager which disables
        # CUDA graph dispatch -> no padding -> tensor is unpadded.
        # Otherwise padding stays -> tensor matches padded_q.
        meta_q = total_q if is_cpu_direct else padded_q

        # Per-request metadata
        computed_map: dict = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            computed_map[new_req.req_id] = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            computed_map[rid] = cached.num_computed_tokens[i]

        offset = 0
        req_id_list = []
        token_ranges = []
        dim0_offsets = []
        for i in range(num_reqs):
            rid = req_ids[i]
            n = num_scheduled_per_req[i]
            pre_computed = computed_map.get(rid, 0)
            norm_id = normalize_vllm_request_id(rid)
            req_id_list.append(norm_id)
            token_ranges.append((pre_computed, pre_computed + n))
            dim0_offsets.append(offset)
            print(f"[dmx_worker] step={_step} req[{i}] rid={norm_id} offset={offset} n={n} "
                  f"t_start={pre_computed} t_end={pre_computed + n} "
                  f"pre_computed={pre_computed} padded_q={padded_q} meta_q={meta_q}",
                  flush=True)
            offset += n

        transport.set_step_context(
            model_id=self._dmx_model_id,
            req_ids=req_id_list,
            token_ranges=token_ranges,
            dim0_offsets=dim0_offsets,
            tp_rank=self._dmx_tp_rank,
            dp_rank=self._dmx_dp_rank,
            ep_rank=self._dmx_ep_rank,
            pp_rank=self._dmx_pp_rank,
            flattened=True,
        )

        if cfg is not None:
            # Read input_ids dtype from model_runner buffer.  On non-first
            # PP ranks the buffer may not exist; pass None (the token_ids
            # hook is already filtered out by filter_by_pp_rank).
            ids_buf = getattr(self.model_runner, 'input_ids', None)
            ids_dtype = ids_buf.gpu.dtype if ids_buf is not None else None

            # logits_to_keep=num_reqs: vLLM's compute_logits returns one
            # logit per request, shaped [num_reqs, vocab].  In flattened
            # mode _compute_hook_shape uses logits_to_keep directly as dim0
            # (no batch dim), so the meta shape becomes [num_reqs, vocab].
            # The p2p thread then slices row j for request j and adjusts
            # DB token range to (end_token-1, end_token) — the single
            # predicted position per request.
            transport.pre_push_all_metas(
                batch=0, q_len=meta_q, kv_dim=0,
                logits_to_keep=num_reqs,
                token_ids_dtype=ids_dtype)

        return super().execute_model(scheduler_output)

    def stop_monitoring(self) -> None:
        """Flush and stop DMI engine.  Callable via collective_rpc before
        vLLM shutdown to avoid the 8-second kill deadline.

        Reentrant: second call is a no-op (engine refs nulled after stop).
        """
        from . import ring_transport
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self._dmx_ring_engine is not None:
            try:
                self._dmx_ring_engine.stop()
            except Exception:
                pass
            self._dmx_ring_engine = None

        ring_transport.deactivate()

        if self._dmx_host_engine is not None:
            try:
                self._dmx_host_engine.stop()
            except Exception:
                pass
            self._dmx_host_engine = None

    def shutdown(self) -> None:
        import logging
        if self._dmx_ring_engine is not None:
            logging.getLogger(__name__).warning(
                "DMI engine not explicitly stopped before shutdown. "
                "Data may be incomplete. Call stop_monitoring() first.")
        # Best-effort flush (may be killed by vLLM's 8s deadline).
        self.stop_monitoring()
        super().shutdown()
