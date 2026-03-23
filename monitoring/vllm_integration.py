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
            stage_cfg = _ne.StageConfig.clickhouse_insert(ch_cfg)
            host_engine = _ne.DMXHostEngine(stage_cfg)
            host_engine.start()
            self._dmx_host_engine = host_engine

        # Ring engine
        ring_cfg = _ne.RingConfig()
        ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
        ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
        ring_cfg.task_ring_entries = ring_entries

        ring_engine = _ne.RingEngine(ring_cfg, host_engine)
        ring_engine.init()
        ring_engine.start()
        self._dmx_ring_engine = ring_engine

        # Transport (model shape + hooks set later in load_model)
        transport = RingTransport(ring_engine)
        if null_mode:
            transport.null_offload = True
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
        super().load_model()

        # Model is now available — set shape config and install hooks
        from . import ring_transport
        from .ring_transport import apply_hook_selection, install_ring_hooks
        from .generate import _make_model_shape

        transport = ring_transport.get_active()
        if transport is None:
            return

        model = self.model_runner.model
        model_shape = _make_model_shape(model)
        if model_shape is not None:
            transport.set_model_cfg(model_shape)

        if hasattr(model, "get_hook_specs"):
            all_specs = model.get_hook_specs()
            active_specs = apply_hook_selection(all_specs, self._dmx_hook_selection)

            from vllm.distributed.parallel_state import get_pp_group
            pp_group = get_pp_group()
            active_specs = filter_by_pp_rank(
                active_specs,
                is_first_rank=pp_group.is_first_rank,
                is_last_rank=pp_group.is_last_rank,
            )

            handles: list = []
            install_ring_hooks(active_specs, handles)
            transport._active_specs = active_specs
            transport._using_forward_hooks = True

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        from . import ring_transport
        from .ring_transport import align_up_py, _compute_hook_shape

        total_tokens = scheduler_output.total_num_scheduled_tokens
        transport = ring_transport.get_active()

        if total_tokens == 0 or transport is None or transport.null_offload:
            return super().execute_model(scheduler_output)

        # Read batch info
        input_batch = self.model_runner.input_batch
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids[:num_reqs]

        num_scheduled_per_req = []
        for rid in req_ids:
            num_scheduled_per_req.append(
                scheduler_output.num_scheduled_tokens.get(rid, 0))
        total_q = sum(num_scheduled_per_req)

        # Capacity check
        cfg = transport._model_cfg
        if cfg is not None and transport._active_specs:
            step_bytes = 0
            n_hooks = 0
            for spec in transport._active_specs:
                shape = _compute_hook_shape(
                    spec.hook_type, cfg, batch=1, q_len=total_q, kv_dim=0)
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
                transport.cpu_direct = (result == 2)
                if result == 2:
                    self._dmx_force_eager = True

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
            req_id_list.append(normalize_vllm_request_id(rid))
            token_ranges.append((pre_computed, pre_computed + n))
            dim0_offsets.append(offset)
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
            transport.pre_push_all_metas(
                batch=1, q_len=total_q, kv_dim=0, logits_to_keep=0)

        return super().execute_model(scheduler_output)

    def shutdown(self) -> None:
        from . import ring_transport
        ring_transport.deactivate()

        if self._dmx_ring_engine is not None:
            try:
                self._dmx_ring_engine.stop()
            except Exception:
                pass
            self._dmx_ring_engine = None

        if self._dmx_host_engine is not None:
            try:
                self._dmx_host_engine.stop()
            except Exception:
                pass
            self._dmx_host_engine = None

        super().shutdown()
