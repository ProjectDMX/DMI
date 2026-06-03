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
# Node-toggle (Phase 3b) helpers
# ---------------------------------------------------------------------------

def _parse_enabled_hooks(s: Any):
    """Parse 'hook_type:layer,hook_type:layer,...' -> [(ht, ln), ...]; '' -> None."""
    if not s:
        return None
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        ht, _, ln = tok.partition(":")
        out.append((int(ht), int(ln if ln != "" else -1)))
    return out or None


_CUDAGRAPH_KEEP_PATCHED = False


def _patch_cudagraph_keep_graph() -> None:
    """Monkeypatch torch.cuda.CUDAGraph to default keep_graph=True.

    vLLM's CUDAGraphWrapper creates `torch.cuda.CUDAGraph()` (keep_graph=False), which
    frees the captured template graph after instantiate -> the node handles DMI records
    via cudaStreamGetCaptureInfo would dangle. keep_graph=True keeps the template alive
    (host memory only; no device/latency cost). Idempotent, process-global; only applied
    when node-toggle is enabled. See docs/node_toggle_phase3_feasibility.md.
    """
    global _CUDAGRAPH_KEEP_PATCHED
    if _CUDAGRAPH_KEEP_PATCHED:
        return
    import torch
    Orig = torch.cuda.CUDAGraph
    if getattr(Orig, "_dmx_keep_graph", False):
        _CUDAGRAPH_KEEP_PATCHED = True
        return

    class _KeepGraphCUDAGraph(Orig):
        # Force keep_graph=True in BOTH __new__ and __init__: vLLM calls
        # `CUDAGraph()` with no args, so __init__ (builtin) would otherwise reset
        # keep_graph to its False default after __new__ set it. Subclass (not a
        # factory) so isinstance(x, torch.cuda.CUDAGraph) still holds.
        _dmx_keep_graph = True

        def __new__(cls, *args, **kwargs):
            kwargs["keep_graph"] = True
            return Orig.__new__(cls, **kwargs)

        def __init__(self, *args, **kwargs):
            kwargs["keep_graph"] = True
            super().__init__(**kwargs)

    torch.cuda.CUDAGraph = _KeepGraphCUDAGraph
    _CUDAGRAPH_KEEP_PATCHED = True


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
        # Runtime node-toggle (Phase 3b). Default off -> behaviour unchanged.
        # dmx_enabled_hooks: optional static enabled set "hook_type:layer,..." applied
        # once after warmup (for testing); empty -> toggle armed but gate inactive.
        node_toggle     = _cfg(ac, "dmx_node_toggle",      "DMX_NODE_TOGGLE",      False)
        enabled_hooks_s = _cfg(ac, "dmx_enabled_hooks",    "DMX_ENABLED_HOOKS",    "")

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
            ch_cfg.create_database_if_missing = True
            stage_cfg = _ne.StageConfig.clickhouse_insert(
                ch_cfg, parallelism=10, name="clickhouse_insert")
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
        # Drain flush timeout (us): bound export latency for LOW-VOLUME / sparse-hook
        # configs. should_flush() otherwise only fires when the ring fills, so a few
        # hooks (e.g. residual on the last few layers for hallucination monitoring)
        # on a multi-GB ring can delay export by minutes — the FEWER hooks, the WORSE
        # the latency. The drain-thread timeout flush runs on the drain's own D2H
        # stream (does NOT stall decode) and only commits already-published entries;
        # high-volume configs fill the ring before the timeout fires (unaffected) and
        # idle periods have nothing pending (no flush). 0 = legacy flush-when-full.
        # See docs/node_toggle_local_perf.md "Low-volume drain latency".
        ring_cfg.drain_flush_timeout_us = int(_cfg(
            ac, "dmx_drain_flush_timeout_us", "DMX_DRAIN_FLUSH_TIMEOUT_US", 50000))

        ring_engine = _ne.RingEngine(ring_cfg, host_engine)
        ring_engine.init()
        ring_engine.start()
        self._dmx_ring_engine = ring_engine

        transport = RingTransport(ring_engine)
        self._dmx_null_mode_user = null_mode
        if null_mode:
            transport.null_offload = True
        # Start in null mode — warmup/profiling will fire producer kernels
        # (needed for CUDA graph capture) but as no-ops so the ring stays clean.
        # Turned off after warmup in compile_or_warm_up_model.
        ring_engine.set_null_mode(True)
        ring_transport.activate(transport)

        # --- Runtime node-toggle (Phase 3b) ---
        # Only store config here. Capture recording + the keep_graph patch are
        # turned on around the REAL graph capture (compile_or_warm_up_model), not
        # here -- enabling them in init_device would also affect vLLM's pre-capture
        # memory profiling and any later captures (#1).
        self._dmx_node_toggle = bool(node_toggle)
        self._dmx_enabled_hooks = _parse_enabled_hooks(enabled_hooks_s)

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
            "LlamaForCausalLM": "LlamaPForCausalLM",
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

            handles: list = []
            install_ring_hooks(active_specs, handles)
            transport._active_specs = active_specs
            transport._using_forward_hooks = True

    def compile_or_warm_up_model(self) -> float:
        toggle = getattr(self, "_dmx_node_toggle", False) and self._dmx_ring_engine is not None
        if toggle:
            # Scope node recording + keep_graph to the REAL capture that
            # super().compile_or_warm_up_model() performs (memory profiling already
            # ran in determine_available_memory). Clear any stale registry first.
            self._dmx_ring_engine.clear_toggle_registry()
            self._dmx_ring_engine.enable_toggle_capture(True)
            _patch_cudagraph_keep_graph()

        # Warmup runs with null_mode=True (set in init_device).
        # Producer kernels fire but are no-ops — ring stays clean.
        result = super().compile_or_warm_up_model()

        # Warmup done. Turn off null mode (unless user explicitly wants it).
        if not self._dmx_null_mode_user and self._dmx_ring_engine is not None:
            self._dmx_ring_engine.set_null_mode(False)

        # Node-toggle: bind each captured graph's exec, then close the capture
        # window and (optionally) apply the static enabled set. Done here -- after
        # warmup, before serving -- so no replay is in flight (design-notes §1).
        if toggle:
            n_bound = self._dmx_bind_captured_graphs()
            self._dmx_ring_engine.enable_toggle_capture(False)  # close the window (#1)
            print(f"[DMX] node-toggle: bound {n_bound} captured graph(s), "
                  f"{self._dmx_ring_engine.toggle_node_count()} nodes registered")
            if self._dmx_enabled_hooks is not None:
                if n_bound == 0:
                    # Requested a subset but nothing bound (e.g. cudagraph_mode not
                    # FULL) -> refuse to serve silently with all hooks on (#2).
                    raise RuntimeError(
                        "dmx_node_toggle + dmx_enabled_hooks were set but no CUDA graph "
                        "was bound (is cudagraph_mode FULL/FULL_AND_PIECEWISE for decode?). "
                        "Refusing to serve with node-toggle silently inert.")
                from . import ring_transport
                transport = ring_transport.get_active()
                if transport is not None:
                    transport.set_active_hooks(self._dmx_enabled_hooks)
                    print(f"[DMX] node-toggle: active hooks set to {self._dmx_enabled_hooks}")
            elif n_bound == 0:
                print("[DMX] node-toggle: WARNING no graph bound; toggle is inert "
                      "(no dmx_enabled_hooks requested, so serving continues all-on).")

        return result

    def _dmx_bind_captured_graphs(self) -> int:
        """Find vLLM's captured CUDA graphs and bind each exec to the toggle
        registry. Returns the number bound. FULL-cudagraph path only (decode);
        the model is wrapped as CUDAGraphWrapper whose concrete_cudagraph_entries
        hold one torch.cuda.CUDAGraph per batch size (see
        docs/node_toggle_phase3b_vllm_investigation.md)."""
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
        except Exception:
            return 0
        model = getattr(self.model_runner, "model", None)
        n = 0
        # Unwrap nested wrappers (UBatchWrapper etc.) to find CUDAGraphWrapper(s).
        seen = set()
        stack = [model]
        while stack:
            obj = stack.pop()
            if id(obj) in seen or obj is None:
                continue
            seen.add(id(obj))
            if isinstance(obj, CUDAGraphWrapper):
                for entry in obj.concrete_cudagraph_entries.values():
                    g = getattr(entry, "cudagraph", None)
                    if g is None:
                        continue
                    # Don't re-instantiate if an exec already exists (that would
                    # destroy it, #3). raw_cuda_graph_exec() raises until the graph
                    # is instantiated -> instantiate exactly once in that case.
                    try:
                        exec_ptr = g.raw_cuda_graph_exec()
                    except RuntimeError:
                        g.instantiate()
                        exec_ptr = g.raw_cuda_graph_exec()
                    self._dmx_ring_engine.bind_graph_exec(g.raw_cuda_graph(), exec_ptr)
                    n += 1
            inner = getattr(obj, "runnable", None)
            if inner is not None:
                stack.append(inner)
        return n

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        from . import ring_transport
        from .ring_transport import align_up_py, _compute_hook_shape, HOOK_TYPE_TOKEN_IDS

        total_tokens = scheduler_output.total_num_scheduled_tokens
        if not hasattr(self, '_dmx_step_counter'):
            self._dmx_step_counter = 0
        self._dmx_step_counter += 1
        _step = self._dmx_step_counter
        transport = ring_transport.get_active()

        if total_tokens == 0 or transport is None or transport.null_offload:
            return super().execute_model(scheduler_output)

        # --- per-step host-path profiling (DMX_PROFILE) -------------------
        _prof_on = bool(os.environ.get("DMX_PROFILE"))
        if _prof_on:
            import time as _time, collections
            if not hasattr(self, "_dmx_prof"):
                self._dmx_prof = collections.defaultdict(float)
                self._dmx_prof_n = 0
            self._dmx_prof_n += 1
            _pf = _time.perf_counter
            _tmark = [_pf()]
            def _mark(label):
                now = _pf()
                self._dmx_prof[label] += now - _tmark[0]
                _tmark[0] = now
        else:
            def _mark(label):
                pass

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
        _mark("1_predict_padding")

        # Capacity check — use padded_q (matches real tensor size).
        cfg = transport._model_cfg
        # token_ids producer writes the REAL input_ids tensor (e.g. int64), not the
        # model dtype. Read its dtype once and feed the SAME dtype selection used by
        # pre_push_all_metas into the capacity loop below — otherwise reserve bytes
        # for the token_ids hook (included in the default vllm-full selection) would
        # use cfg.dtype and not match what the producer actually writes. None on
        # non-first PP ranks (token_ids hook filtered out there anyway).
        _ids_buf = getattr(self.model_runner, 'input_ids', None)
        token_ids_dtype = _ids_buf.gpu.dtype if _ids_buf is not None else None
        is_cpu_direct = False
        step_bytes = 0
        n_hooks = 0
        prep_ret = -1
        if cfg is not None and transport._active_specs:
            # (step_bytes, n_hooks) depend only on (padded_q, num_reqs) for a
            # fixed spec set + dtypes -> identical every decode step.  The
            # per-step _compute_hook_shape loop was the dominant host-floor
            # cost (see DMX_PROFILE: ~20-32 us/step), so cache it and reuse.
            # prepare_step still runs every step (ring fill level changes).
            _cap_cache = getattr(self, "_dmx_capacity_cache", None)
            if _cap_cache is None:
                _cap_cache = self._dmx_capacity_cache = {}
            # Key on _enabled_version so the cache invalidates when the toggle
            # enabled set changes. Reserve over only the EFFECTIVE enabled specs
            # (same set meta-push + device toggle use) — counting all _active_specs
            # would over-reserve bytes+hooks for disabled hooks (the integration
            # bug: reserve must match what producers actually write).
            _ckey = (padded_q, num_reqs, transport._enabled_version, token_ids_dtype)
            _cached = _cap_cache.get(_ckey)
            if _cached is None:
                step_bytes = 0
                n_hooks = 0
                for spec in transport.effective_specs:
                    shape = _compute_hook_shape(
                        spec.hook_type, cfg, batch=0, q_len=padded_q, kv_dim=0,
                        logits_to_keep=num_reqs)
                    if not shape:
                        continue
                    # Same dtype selection as pre_push_all_metas, so reserve bytes
                    # == produced bytes for every hook (token_ids uses the real
                    # input_ids dtype, not the model dtype).
                    if spec.dtype is not None:
                        dtype = spec.dtype
                    elif spec.hook_type == HOOK_TYPE_TOKEN_IDS and token_ids_dtype is not None:
                        dtype = token_ids_dtype
                    else:
                        dtype = cfg.dtype
                    elem_size = torch._utils._element_size(dtype)
                    nbytes = elem_size
                    for d in shape:
                        nbytes *= d
                    step_bytes += align_up_py(nbytes, 16)
                    n_hooks += 1
                _cap_cache[_ckey] = (step_bytes, n_hooks)
            else:
                step_bytes, n_hooks = _cached

            if n_hooks > 0:
                result = transport._ring_engine.prepare_step(step_bytes, n_hooks)
                prep_ret = result
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

        # No enabled hooks this step (e.g. node-toggle fully off): nothing will be
        # produced, so prepare_step() above was skipped. Explicitly clear any stale
        # cpu_direct / force_eager left by a prior step — a leftover force_eager would
        # run this step EAGER, firing producers outside the captured graph while the
        # meta push is empty -> desync. (force_eager is one-shot today, but the gate
        # set_cpu_direct flag is not; don't rely on that implicit invariant.)
        if n_hooks == 0:
            self._dmx_force_eager = False
            transport.cpu_direct = False
            ring_transport.set_cpu_direct(False)

        # Meta q_len: if cpu_direct, we set force_eager which disables
        # CUDA graph dispatch -> no padding -> tensor is unpadded.
        # Otherwise padding stays -> tensor matches padded_q.
        meta_q = total_q if is_cpu_direct else padded_q
        _mark("2_capacity_check")

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
            if os.environ.get("DMX_STEP_DEBUG"):
                print(f"[dmx_worker] step={_step} req[{i}] rid={norm_id} offset={offset} n={n} "
                      f"t_start={pre_computed} t_end={pre_computed + n} "
                      f"pre_computed={pre_computed} padded_q={padded_q} meta_q={meta_q}",
                      flush=True)
            offset += n
        _mark("3_build_req_meta")

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
        _mark("4_store_step_ctx")

        if cfg is not None:
            # token_ids_dtype computed once at the top of this method (same value
            # the capacity loop reserved with) -> reserve == push == produced.
            #
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
                token_ids_dtype=token_ids_dtype)
        _mark("5_push_meta_fifo")

        # --- over-reserve probe (DMX_PROBE): reserved (capacity) vs pushed (metas)
        # vs the live ring head/tail gap. Monotonic payload_gap growth == reserve()
        # over-counting vs actual producer writes (the node-toggle integration bug).
        if os.environ.get("DMX_PROBE"):
            p = getattr(self, "_dmx_probe", None)
            if p is None:
                p = self._dmx_probe = {
                    "n": 0, "win": int(os.environ.get("DMX_PROBE_WINDOW", "500")),
                    "prep": [0, 0, 0], "cpu_direct": 0,
                    "resv_hooks": 0, "pushed": 0, "last_gap": 0,
                }
            p["n"] += 1
            if 0 <= prep_ret <= 2:
                p["prep"][prep_ret] += 1
            p["cpu_direct"] += 1 if is_cpu_direct else 0
            p["resv_hooks"] += n_hooks
            p["pushed"] += getattr(transport, "_last_push_count", 0)
            if p["n"] % p["win"] == 0:
                st = transport._ring_engine.get_stats()
                pay_gap = st.cpu_payload_head - st.cpu_payload_tail_committed
                task_gap = st.cpu_task_head - st.cpu_task_tail_committed
                dgap = pay_gap - p["last_gap"]
                p["last_gap"] = pay_gap
                print(f"[PROBE] step={p['n']} payload_gap={pay_gap/1e6:.2f}MB "
                      f"(+{dgap/1e6:.2f}/win) task_gap={task_gap} "
                      f"prep[ok={p['prep'][0]},flush={p['prep'][1]},cpudir={p['prep'][2]}] "
                      f"resv_hooks/win={p['resv_hooks']} pushed/win={p['pushed']}",
                      flush=True)
                p["prep"] = [0, 0, 0]
                p["cpu_direct"] = 0
                p["resv_hooks"] = 0
                p["pushed"] = 0

        result = super().execute_model(scheduler_output)
        _mark("6_cudagraph_replay")
        return result

    def shutdown(self) -> None:
        from . import ring_transport

        if getattr(self, "_dmx_prof", None):
            n = max(1, self._dmx_prof_n)
            print(f"\n[DMX PROFILE] per-step host path over {self._dmx_prof_n} steps:",
                  flush=True)
            host = 0.0
            for k in sorted(self._dmx_prof):
                us = self._dmx_prof[k] / n * 1e6
                print(f"  {k:18s} {us:8.2f} us/step   "
                      f"(cum {self._dmx_prof[k] * 1e3:7.1f} ms)", flush=True)
                if not k.startswith("6_"):
                    host += us
            print(f"  {'HOST FLOOR (1-5)':18s} {host:8.2f} us/step", flush=True)

        # Sync CUDA stream to ensure last producer kernels complete, then
        # stop ring engine (flushes drain -> p2p -> host queue).
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

        super().shutdown()
