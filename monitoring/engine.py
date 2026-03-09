"""Monitoring engine wrapper backed only by the native C++/CUDA engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import AdvanceConfig, MonitoringConfig
from .task import CacheFuture, MonitoringTask


def _stream_to_handle(stream: Optional[torch.cuda.Stream]) -> Optional[int]:
    if stream is None:
        return None
    try:
        return int(stream.cuda_stream)
    except AttributeError:
        return None


def _serialize_task(task: MonitoringTask) -> tuple[Any, int, bool, bool, Any, Optional[torch.device]]:
    """Return a tuple payload consumable by the native backend."""

    return task.native_payload


@dataclass
class HostEngineConfig:
    """Configuration wrapper for the native DMXHostEngine pipeline.

    The DMXHostEngine lives in the native monitoring extension and is used to
    post-process per-step BackendFuture objects (e.g. wait on them, then write
    results to ClickHouse).

    Notes:
      - This path requires the native monitoring backend (CUDA +
        monitoring_native_backend extension).
      - The engine currently expects exactly **two** stages.
    """

    stages: Sequence[Any]
    start_on_init: bool = True


class MonitoringEngine:
    """High-level wrapper that routes monitoring tasks to native backend."""

    def __init__(
        self,
        *,
        async_enabled: bool = True,
        queue_size: int = 0,
        cache_dtype: Optional[torch.dtype] = None,
        delay_steps: int = 0,
        config: Optional[MonitoringConfig] = None,
        model_id: Optional[str] = None,
        host_engine: Optional[Any] = None,
        db_config: Optional[HostEngineConfig] = None,
    ) -> None:
        self.async_enabled = async_enabled
        self.cache_dtype = cache_dtype
        self._delay_steps = max(0, int(delay_steps))
        self.config = config
        self._debug_enabled = bool(self.config.debug) if self.config is not None else False
        self._nvtx_enabled = self._debug_enabled
        self._no_strip = False if config is None else config.no_strip
        self._request_capture_enabled = True
        self._capture_enabled = True
        self._hook_cache_config_id: Optional[int] = None
        self._hook_cache_key: Optional[int] = None
        self._hook_cache_list: Optional[List[str]] = None
        self._hook_cache_set: Optional[set[str]] = None
        self._sync_hook_debug_flag()

        self._current_step_id: int = 0
        self._pending_tasks: Dict[int, List[Tuple[MonitoringTask, CacheFuture]]] = {}
        self._pending_db_step: Optional[
            Tuple[str, int, List[str], List[Tuple[int, int]], Dict[str, Any]]
        ] = None
        self._model_id = model_id
        self._auto_batch_group_id = 0
        self._active_batch_request_ids: Optional[List[str]] = None
        self._active_batch_start_idx_per_request: Optional[List[int]] = None
        self._active_batch_finished_per_request: Optional[List[bool]] = None

        # Native backend only.
        self._native_backend: Optional[Any] = None
        self._using_native_backend = False

        # Host-side DB engine (optional; C++ backend only)
        self._host_engine: Optional[Any] = None
        self._host_engine_enabled = False

        # Ring transport (replaces NativeMonitoringEngine D2H when active)
        self._ring_transport: Optional[Any] = None
        self._using_ring_transport = False

        if not self.async_enabled:
            raise RuntimeError(
                "MonitoringEngine no longer supports async_enabled=False; Python backend was removed"
            )

        advance_cfg = self.config.advance if self.config is not None else AdvanceConfig()
        native_backend = _load_native_backend(
            queue_size,
            cache_dtype,
            self._delay_steps,
            advance_cfg,
        )
        if native_backend is None:
            raise RuntimeError(
                "Failed to initialize native monitoring backend; "
                "Python backend fallback has been removed"
            )
        self._native_backend = native_backend
        self._using_native_backend = True

        try:
            native_backend.begin_step(int(self._current_step_id), 0)  # initialize step tracking
        except Exception:
            pass
        if self.config is not None:
            self._apply_capture_schedule()
            self._apply_native_runtime_config()

        if host_engine is not None and db_config is not None:
            raise ValueError("Provide either host_engine or db_config, not both")

        if host_engine is not None or db_config is not None:
            if self._model_id is None:
                raise ValueError("model_id is required when host_engine integration is enabled")
            self._host_engine = host_engine
            if self._host_engine is None and db_config is not None:
                try:
                    from . import _native_engine

                    DMXHostEngine = _native_engine.DMXHostEngine  # type: ignore[attr-defined]
                except Exception as exc:
                    raise RuntimeError("Failed to import native DMXHostEngine") from exc
                stages = tuple(db_config.stages)
                if len(stages) != 1:
                    raise ValueError("db_config.stages must contain exactly 1 StageConfig object (clickhouse_insert)")

                try:
                    self._host_engine = DMXHostEngine(stages[0])  # type: ignore[call-arg]
                except Exception as exc:
                    raise RuntimeError("Failed to construct DMXHostEngine") from exc
            if self._host_engine is not None:
                try:
                    if db_config is None or db_config.start_on_init:
                        self._host_engine.start()
                except Exception as exc:
                    raise RuntimeError("Failed to start host_engine") from exc
                self._host_engine_enabled = True

        # Stats (optional) --------------------------------------------------
        self._stats_enabled = self._debug_enabled
        self._stats_hooks = 0
        self._stats_steps = 0
        self._stats_tasks = 0
        self._stats_native_submit_ms = 0.0
        # Fine-grained Python-side timings (ms)
        self._stats_py_serialize_ms = 0.0  # building tuple payloads in Python
        self._stats_py_bind_ms = 0.0  # binding tokens back to futures
        self._stats_py_resolve_ms = 0.0  # resolve_all + clear overhead
        self._stats_max_tasks_per_step = 0
        self._last_prepare_ms = 0.0
        self._stats_endstep_ms_total = 0.0
        self._stats_endstep_calls = 0
        self._stats_endstep_ms_max = 0.0

    # ------------------------------------------------------------------
    # Ring transport API

    def enable_ring_transport(
        self, ring_config: Any, model_shape: Optional[Any] = None
    ) -> None:
        """Switch to ring-based D2H transport.

        Creates a RingEngine with the C++ host engine as the submit target so
        tensor reconstruction, slicing, and DB submission all happen in C++
        without the GIL.

        Args:
            ring_config:  A _native_engine.RingConfig instance.
            model_shape:  Optional ModelShapeConfig for analytical shape computation.
                          When provided, the new CUDA-graph-compatible forward-hook
                          path is activated.  If None, shape is auto-detected from
                          model.config in _install_monitoring_forward.
        """
        from . import ring_transport as _rt
        from . import _native_engine  # type: ignore[attr-defined]

        # Pass the DMXHostEngine C++ object directly; RingEngine builds a
        # SubmitFn that calls submit_direct without touching Python/GIL.
        # Pass None for null/benchmark mode (no DB writes).
        host_cpp = None
        if self._host_engine is not None and isinstance(
            self._host_engine, _native_engine.DMXHostEngine
        ):
            host_cpp = self._host_engine

        ring_engine = _native_engine.RingEngine(ring_config, host_cpp)

        ring_engine.init()
        ring_engine.start()

        transport = _rt.RingTransport(ring_engine)
        if model_shape is not None:
            transport.set_model_cfg(model_shape)
        self._ring_engine = ring_engine
        self._ring_transport = transport
        self._using_ring_transport = True
        _rt.activate(transport)

    def _prepare_ring_step(self, input_ids: Any, attention_mask: Any, past_key_values: Any, cache_position: Any = None) -> None:
        """Precompute per-step batch context and set it on the ring transport.

        Called before the forward pass so ring hooks (firing during forward)
        can push correctly-keyed FIFO entries.
        """
        if not self._using_ring_transport or self._ring_transport is None:
            return
        if self._model_id is None:
            return
        if input_ids is None or not hasattr(input_ids, "shape"):
            return
        try:
            input_shape = tuple(input_ids.shape)
        except Exception:
            return
        if not input_shape:
            return
        try:
            batch_size = int(input_shape[0])
        except Exception:
            return
        if batch_size <= 0:
            return

        if cache_position is not None:
            try:
                is_prefill = int(cache_position[0]) == 0
            except Exception:
                is_prefill = past_key_values is None
        else:
            is_prefill = past_key_values is None
            try:
                if hasattr(input_ids, "dim") and int(input_ids.dim()) >= 2:
                    if int(input_ids.shape[1]) > 1:
                        is_prefill = True
            except Exception:
                pass

        current_ids = self._active_batch_request_ids
        need_reset = is_prefill or current_ids is None or len(current_ids) != batch_size
        if need_reset:
            gid = int(self._auto_batch_group_id)
            self._auto_batch_group_id += 1
            self._active_batch_request_ids = [f"{gid}:{i}" for i in range(batch_size)]
            self._active_batch_start_idx_per_request = [0] * batch_size
            self._active_batch_finished_per_request = [False] * batch_size

        req_ids = self._active_batch_request_ids
        starts = self._active_batch_start_idx_per_request
        finished = self._active_batch_finished_per_request
        if req_ids is None or starts is None or finished is None:
            return

        if isinstance(attention_mask, dict):
            assert "full_attention" in attention_mask, f"attention_mask dict missing 'full_attention' key: {list(attention_mask.keys())}"
            attention_mask = attention_mask["full_attention"]

        token_ranges: List[Tuple[int, int]] = []
        if is_prefill:
            if attention_mask is None or not hasattr(attention_mask, "dim"):
                return
            try:
                ndim = int(attention_mask.dim())
                if ndim == 2:
                    # Standard 2D mask [batch, seq_len]
                    lengths = (
                        attention_mask.sum(dim=1).tolist()
                        if not self._no_strip
                        else [attention_mask.shape[1]] * attention_mask.shape[0]
                    )
                elif ndim == 4 and len(input_shape) >= 2 and int(input_shape[1]) > 0:
                    # 4D causal mask [batch, 1, q_len, kv_dim] — used by static-cache generate.
                    # Values: 0.0 = attend, large negative = masked (NOT integer 0/1).
                    # Count non-masked positions among the first q_len key slots using the
                    # last query row (most permissive for left-padded causal sequences).
                    q_len_mask = int(input_shape[1])
                    lengths = (
                        (attention_mask[:, 0, -1, :q_len_mask] >= 0.0).sum(dim=-1).long().tolist()
                        if not self._no_strip
                        else [q_len_mask] * int(attention_mask.shape[0])
                    )
                else:
                    return
                lengths = [int(v) for v in lengths]
            except Exception:
                return
            if len(lengths) != batch_size:
                return
            for i in range(batch_size):
                start_i = int(starts[i])
                delta_i = int(lengths[i])
                if delta_i < 0:
                    delta_i = 0
                end_i = start_i + delta_i
                token_ranges.append((start_i, end_i))
                starts[i] = end_i
        else:
            eos_or_pad_ids: set[int] = set()
            eos_token_id = getattr(self.config, "eos_token_id", None) if self.config is not None else None
            pad_token_id = getattr(self.config, "pad_token_id", None) if self.config is not None else None
            if eos_token_id is not None:
                if isinstance(eos_token_id, (list, tuple, set)):
                    eos_or_pad_ids.update(int(v) for v in eos_token_id)
                else:
                    eos_or_pad_ids.add(int(eos_token_id))
            if pad_token_id is not None:
                eos_or_pad_ids.add(int(pad_token_id))
            last_ids = None
            if eos_or_pad_ids:
                try:
                    last_ids = input_ids[:, -1]
                except Exception:
                    last_ids = None

            for i in range(batch_size):
                start_i = int(starts[i])
                is_finished = bool(finished[i])
                if (not is_finished) and (last_ids is not None):
                    try:
                        if int(last_ids[i]) in eos_or_pad_ids:
                            is_finished = True
                    except Exception:
                        pass
                if is_finished:
                    finished[i] = True
                if is_finished and not self._no_strip:
                    token_ranges.append((start_i, start_i))
                else:
                    end_i = start_i + 1
                    token_ranges.append((start_i, end_i))
                    starts[i] = end_i

        self._ring_transport.set_step_context(
            model_id=str(self._model_id),
            shard_rank=0,
            req_ids=list(req_ids),
            token_ranges=token_ranges,
        )

    # ------------------------------------------------------------------
    # Public API

    def submit(self, task: MonitoringTask) -> CacheFuture:
        """Register a monitoring task for native backend processing."""

        if not task.is_cuda():
            raise RuntimeError("MonitoringTask tensor must be CUDA in C++-only mode")

        backend = self._native_backend
        if backend is None:
            raise RuntimeError("Native monitoring backend is not initialized")

        future = CacheFuture(task)

        step_id = self._current_step_id
        task.step_id = step_id

        bucket = self._pending_tasks.setdefault(step_id, [])
        bucket.append((task, future))
        if self._stats_enabled:
            self._stats_hooks += 1

        return future

    def start_step(self, phase: Optional[str] = None) -> None:
        """Mark the beginning of a decode/prefill step."""

        backend = self._native_backend
        if backend is None:
            return

        # Optional NVTX for Python-visible start_step
        try:
            from torch.cuda import nvtx as _nvtx  # type: ignore
        except Exception:
            _nvtx = None  # type: ignore

        if _nvtx is not None and self._nvtx_enabled:
            _nvtx.range_push("MonEng::PyStartStep")

        phase_code = 0
        phase_name = phase
        if phase == "prefill":
            phase_code = 1
        elif phase == "decode":
            phase_code = 2
        else:
            phase_name = "decode"

        if self.config is not None:
            schedule = self.config.schedule
            step_enabled = schedule.should_capture_step(self._current_step_id + 1, phase_name)
            self._capture_enabled = bool(self._request_capture_enabled and step_enabled)
        else:
            self._capture_enabled = True

        self._current_step_id += 1
        backend.begin_step(int(self._current_step_id), phase_code)

        if _nvtx is not None and self._nvtx_enabled:
            _nvtx.range_pop()

    def begin_request(self, request_id: int) -> None:
        """Mark the beginning of a request for request-level capture gating."""

        if self.config is not None:
            self._request_capture_enabled = bool(
                self.config.schedule.should_capture_request(int(request_id))
            )
        else:
            self._request_capture_enabled = True

        backend = self._native_backend
        if backend is not None:
            backend.begin_request(int(request_id))

    def is_capture_enabled(self) -> bool:
        return bool(self._capture_enabled)

    def prepare_for_model(
        self,
        model: Any,
        *,
        names_filter: Any = None,
        device: Optional[torch.device] = None,
        remove_batch_dim: bool = False,
        pos_slice: Any = None,
    ) -> float:
        """Initialize monitoring hooks for a model and return init time (ms)."""

        start = time.perf_counter()
        try:
            if model is not None and hasattr(model, "prepare_monitoring"):
                model.prepare_monitoring(
                    names_filter=names_filter,
                    device=device,
                    remove_batch_dim=remove_batch_dim,
                    pos_slice=pos_slice,
                )
        finally:
            self._last_prepare_ms = (time.perf_counter() - start) * 1e3
        return self._last_prepare_ms

    def get_compiled_hook_names(
        self,
        hook_names: Iterable[str],
        *,
        cache_key: Optional[int] = None,
    ) -> Optional[Tuple[List[str], set[str]]]:
        """Compile and cache enabled hook names based on config.hooks."""

        if self.config is None:
            return None

        cfg_id = id(self.config)
        if self._hook_cache_config_id != cfg_id:
            self._hook_cache_config_id = cfg_id
            self._hook_cache_key = None
            self._hook_cache_list = None
            self._hook_cache_set = None

        if (
            cache_key is not None
            and self._hook_cache_key == cache_key
            and self._hook_cache_list is not None
            and self._hook_cache_set is not None
        ):
            return self._hook_cache_list, self._hook_cache_set

        try:
            compiled_list = list(self.config.hooks.compile(hook_names))
        except Exception:
            return None

        self._hook_cache_key = cache_key
        self._hook_cache_list = compiled_list
        self._hook_cache_set = set(compiled_list)
        return self._hook_cache_list, self._hook_cache_set

    def _apply_capture_schedule(self) -> None:
        if not self._native_backend or self.config is None:
            return
        schedule = self.config.schedule
        self._native_backend.set_capture_schedule(
            int(schedule.step_stride),
            int(schedule.step_offset),
            int(schedule.warmup_steps),
            bool(schedule.capture_prefill),
            bool(schedule.capture_decode),
            int(schedule.request_stride),
            int(schedule.request_offset),
            int(schedule.warmup_requests),
        )

    def _sync_hook_debug_flag(self) -> None:
        """Propagate debug mode to hook_points module without hard import coupling."""

        try:
            from . import hook_points  # local import to avoid import cycle at module load

            setter = getattr(hook_points, "set_monitoring_debug", None)
            if setter is not None:
                setter(self._debug_enabled)
        except Exception:
            pass

    def _apply_native_runtime_config(self) -> None:
        if not self._native_backend or self.config is None:
            return
        cfg = self.config.native_partial_seal
        setter = getattr(self._native_backend, "set_partial_seal_config", None)
        if setter is None:
            return
        setter(
            bool(cfg.enabled),
            int(cfg.chunk_bytes),
            bool(cfg.cap_enabled),
            float(cfg.cap_ratio),
            int(cfg.driver_guard_mb),
        )

    def _register_db_step(
        self,
        cache_dict: Dict[str, Any],
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
    ) -> None:
        # Ring transport handles DB submission via drain callback; skip here.
        if self._using_ring_transport:
            return
        if not self._host_engine_enabled:
            return
        if self._model_id is None:
            return
        if not self._capture_enabled:
            return

        if input_ids is None or not hasattr(input_ids, "shape"):
            return

        try:
            input_shape = tuple(input_ids.shape)
        except Exception:
            return
        if not input_shape:
            return

        try:
            batch_size = int(input_shape[0])
        except Exception:
            return
        if batch_size <= 0:
            return

        if not isinstance(cache_dict, dict) or not cache_dict:
            return

        # Filter alias names to avoid duplicate DB entries.
        filtered: Dict[str, Any] = {
            k: v
            for k, v in cache_dict.items()
            if not k.startswith("h.") and not k.startswith("transformer.")
        }
        # Drop entries without future-like result.
        filtered = {k: v for k, v in filtered.items() if hasattr(v, "result")}

        if not filtered:
            return

        # Prefer shape-based prefill detection because some HF cache modes may pass
        # non-None past_key_values even on the first forward of a new generate() call.
        is_prefill = past_key_values is None
        try:
            if hasattr(input_ids, "dim") and int(input_ids.dim()) >= 2:
                if int(input_ids.shape[1]) > 1:
                    is_prefill = True
        except Exception:
            pass

        current_ids = self._active_batch_request_ids
        need_reset = is_prefill or current_ids is None or len(current_ids) != batch_size
        if need_reset:
            gid = int(self._auto_batch_group_id)
            self._auto_batch_group_id += 1
            self._active_batch_request_ids = [f"{gid}:{i}" for i in range(batch_size)]
            self._active_batch_start_idx_per_request = [0] * batch_size
            self._active_batch_finished_per_request = [False] * batch_size

        req_ids = self._active_batch_request_ids
        starts = self._active_batch_start_idx_per_request
        finished = self._active_batch_finished_per_request
        if req_ids is None or starts is None or finished is None:
            return

        token_ranges: List[Tuple[int, int]] = []
        if is_prefill:
            if attention_mask is None or not hasattr(attention_mask, "dim"):
                return
            if int(attention_mask.dim()) != 2:
                return
            try:
                lengths = (
                    attention_mask.sum(dim=1).tolist()
                    if not self._no_strip
                    else [attention_mask.shape[1]] * attention_mask.shape[0]
                )
            except Exception:
                return
            if len(lengths) != batch_size:
                return
            for i in range(batch_size):
                start_i = int(starts[i])
                delta_i = int(lengths[i])
                if delta_i < 0:
                    delta_i = 0
                end_i = start_i + delta_i
                token_ranges.append((start_i, end_i))
                starts[i] = end_i
        else:
            # Base heuristic: when decode input token is EOS/PAD-like, stop advancing this request.
            eos_or_pad_ids: set[int] = set()
            eos_token_id = getattr(self.config, "eos_token_id", None) if self.config is not None else None
            pad_token_id = getattr(self.config, "pad_token_id", None) if self.config is not None else None
            if eos_token_id is not None:
                if isinstance(eos_token_id, (list, tuple, set)):
                    eos_or_pad_ids.update(int(v) for v in eos_token_id)
                else:
                    eos_or_pad_ids.add(int(eos_token_id))
            if pad_token_id is not None:
                eos_or_pad_ids.add(int(pad_token_id))
            last_ids = None
            if eos_or_pad_ids:
                try:
                    last_ids = input_ids[:, -1]
                except Exception:
                    last_ids = None

            for i in range(batch_size):
                start_i = int(starts[i])
                is_finished = bool(finished[i])
                if (not is_finished) and (last_ids is not None):
                    try:
                        if int(last_ids[i]) in eos_or_pad_ids:
                            is_finished = True
                    except Exception:
                        pass
                if is_finished:
                    finished[i] = True
                if is_finished and not self._no_strip:
                    token_ranges.append((start_i, start_i))
                else:
                    # Not finished or no strip.
                    end_i = start_i + 1
                    token_ranges.append((start_i, end_i))
                    starts[i] = end_i

        self._pending_db_step = (
            str(self._model_id),
            0,  # shard_rank reserved for future TP/distributed.
            list(req_ids),
            token_ranges,
            filtered,
        )

    def _submit_pending_db_step(self) -> None:
        if self._using_ring_transport:
            return
        if not self._host_engine_enabled or self._host_engine is None:
            return

        payload = self._pending_db_step
        if payload is None:
            return
        model_id, shard_rank, req_ids, token_ranges, cache_dict = payload
        try:
            self._host_engine.submit(
                model_id,
                int(shard_rank),
                [req_ids],  # N=1 in V1
                [token_ranges],  # N=1 in V1
                [cache_dict],  # N=1 in V1
            )
        except Exception:
            pass
        finally:
            self._pending_db_step = None

    def end_step(self) -> None:
        """Seal the current step and hand it to the native backend."""

        backend = self._native_backend
        if backend is None:
            return

        # Optional NVTX for Python-visible end_step
        try:
            from torch.cuda import nvtx as _nvtx  # type: ignore
        except Exception:
            _nvtx = None  # type: ignore
        if _nvtx is not None and self._nvtx_enabled:
            _nvtx.range_push("MonEng::PyEndStep")

        step_id = self._current_step_id
        tasks = self._pending_tasks.pop(step_id, [])

        try:
            producer_stream = torch.cuda.current_stream()
        except RuntimeError:
            producer_stream = None

        _t_end0 = None
        if self._stats_enabled:
            _t_end0 = time.perf_counter()
        stream_handle = _stream_to_handle(producer_stream)

        if tasks:
            # Build tuple payloads (measure serialize cost) and submit.
            if self._stats_enabled:
                _t0 = time.perf_counter()
                task_specs = [_serialize_task(task) for task, _ in tasks]
                self._stats_py_serialize_ms += (time.perf_counter() - _t0) * 1000.0
                if len(tasks) > self._stats_max_tasks_per_step:
                    self._stats_max_tasks_per_step = len(tasks)
            else:
                task_specs = [_serialize_task(task) for task, _ in tasks]
            if self._stats_enabled:
                t0 = time.perf_counter()
                tokens = backend.submit_step(step_id, task_specs, stream_handle)
                self._stats_native_submit_ms += (time.perf_counter() - t0) * 1000.0
            else:
                tokens = backend.submit_step(step_id, task_specs, stream_handle)
            if self._stats_enabled:
                _t1 = time.perf_counter()
                for token, (_, future) in zip(tokens, tasks):
                    future.bind_backend(backend, token)
                self._stats_py_bind_ms += (time.perf_counter() - _t1) * 1000.0
            else:
                for token, (_, future) in zip(tokens, tasks):
                    future.bind_backend(backend, token)
            if self._stats_enabled:
                self._stats_steps += 1
                self._stats_tasks += len(tasks)
        else:
            # No Python-collected tasks for this step: seal the native-built step.
            if self._stats_enabled:
                t0 = time.perf_counter()
                backend.seal_step(step_id, stream_handle)
                self._stats_native_submit_ms += (time.perf_counter() - t0) * 1000.0
                self._stats_steps += 1
            else:
                backend.seal_step(step_id, stream_handle)
        self._submit_pending_db_step()
        if _nvtx is not None and self._nvtx_enabled:
            _nvtx.range_pop()
        if _t_end0 is not None:
            _dt = (time.perf_counter() - _t_end0) * 1000.0
            self._stats_endstep_ms_total += _dt
            self._stats_endstep_calls += 1
            if _dt > self._stats_endstep_ms_max:
                self._stats_endstep_ms_max = _dt

    def resolve_all(self) -> None:
        """Block until all pending tasks have been processed."""

        backend = self._native_backend
        if backend is None:
            return

        if self._pending_tasks:
            for step_id in sorted(self._pending_tasks.keys()):
                tasks = self._pending_tasks.pop(step_id)
                if self._stats_enabled:
                    _t0 = time.perf_counter()
                    task_specs = [_serialize_task(task) for task, _ in tasks]
                    self._stats_py_serialize_ms += (time.perf_counter() - _t0) * 1000.0
                    if len(tasks) > self._stats_max_tasks_per_step:
                        self._stats_max_tasks_per_step = len(tasks)
                    t0 = time.perf_counter()
                    tokens = backend.submit_step(step_id, task_specs, None)
                    self._stats_native_submit_ms += (time.perf_counter() - t0) * 1000.0
                    if tasks:
                        self._stats_steps += 1
                        self._stats_tasks += len(tasks)
                    _t1 = time.perf_counter()
                    for token, (_, future) in zip(tokens, tasks):
                        future.bind_backend(backend, token)
                    self._stats_py_bind_ms += (time.perf_counter() - _t1) * 1000.0
                else:
                    task_specs = [_serialize_task(task) for task, _ in tasks]
                    tokens = backend.submit_step(step_id, task_specs, None)
                    for token, (_, future) in zip(tokens, tasks):
                        future.bind_backend(backend, token)
        if self._stats_enabled:
            _t2 = time.perf_counter()
            backend.resolve_all()
            self.clear_completed_results()
            self._stats_py_resolve_ms += (time.perf_counter() - _t2) * 1000.0
        else:
            backend.resolve_all()
            self.clear_completed_results()

    def close(self) -> None:
        """Tear down backend resources."""

        self._pending_db_step = None
        self._active_batch_request_ids = None
        self._active_batch_start_idx_per_request = None
        self._active_batch_finished_per_request = None

        backend = self._native_backend
        if backend is None:
            return

        if self._stats_enabled:
            try:
                stats = backend.get_stats()
            except Exception:
                stats = None
            print(
                "[MonEng/Stats] hooks=",
                self._stats_hooks,
                " steps=",
                self._stats_steps,
                " tasks=",
                self._stats_tasks,
                " py_serialize_ms=",
                round(self._stats_py_serialize_ms, 3),
                " py_submit_ms=",
                round(self._stats_native_submit_ms, 3),
                " py_bind_ms=",
                round(self._stats_py_bind_ms, 3),
                " py_resolve_ms=",
                round(self._stats_py_resolve_ms, 3),
                " end_step_ms=",
                round(self._stats_endstep_ms_total, 3),
                " end_step_avg_ms=",
                round(
                    (self._stats_endstep_ms_total / self._stats_endstep_calls)
                    if self._stats_endstep_calls
                    else 0.0,
                    3,
                ),
                " end_step_max_ms=",
                round(self._stats_endstep_ms_max, 3),
                " max_tasks_per_step=",
                self._stats_max_tasks_per_step,
            )
            if stats is not None:
                try:
                    # Expect dict with microseconds
                    print(
                        "[Native/Stats] steps=",
                        int(stats.get("total_steps", 0)),
                        " tasks=",
                        int(stats.get("total_tasks", 0)),
                        " submit_ms=",
                        round(float(stats.get("submit_us", 0.0)) / 1000.0, 3),
                        " process_ms=",
                        round(float(stats.get("process_us", 0.0)) / 1000.0, 3),
                        " callback_ms=",
                        round(float(stats.get("callback_us", 0.0)) / 1000.0, 3),
                    )
                except Exception:
                    pass
            # Optional: slice mode stats
            try:
                from monitoring.hook_points import get_monitoring_hook_stats

                hook_stats = get_monitoring_hook_stats()
                if hook_stats:
                    print("[Hook/Stats]", hook_stats)
            except Exception:
                pass

        if self._using_ring_transport:
            try:
                ring_engine = getattr(self, "_ring_engine", None)
                if ring_engine is not None:
                    ring_engine.stop()
            except Exception:
                pass
            try:
                from . import ring_transport as _rt
                _rt.deactivate()
            except Exception:
                pass
            self._ring_transport = None
            self._ring_engine = None
            self._using_ring_transport = False

        if self._host_engine is not None:
            try:
                self._host_engine.stop()
            except Exception:
                pass
            self._host_engine = None
            self._host_engine_enabled = False

        self.clear_completed_results()
        backend.close()
        self._native_backend = None
        self._using_native_backend = False

    def clear_completed_results(self) -> None:
        """Clear completed results held by native backend to free memory."""

        if self._native_backend is not None:
            try:
                from torch.cuda import nvtx as _nvtx  # type: ignore
            except Exception:
                _nvtx = None  # type: ignore
            if _nvtx is not None and self._nvtx_enabled:
                _nvtx.range_push("MonEng::PyClearResults")
                try:
                    self._native_backend.clear_completed_results()
                finally:
                    _nvtx.range_pop()
            else:
                self._native_backend.clear_completed_results()


# ---------------------------------------------------------------------------
# Backend loader


def _load_native_backend(
    queue_size: int,
    cache_dtype: Optional[torch.dtype],
    delay_steps: int,
    advance: AdvanceConfig,
) -> Optional[Any]:
    """Attempt to load the native backend extension."""

    try:
        from . import _native_engine
    except Exception:
        return None

    try:
        return _native_engine.create_engine(  # type: ignore[attr-defined]
            queue_size=queue_size,
            cache_dtype=cache_dtype,
            delay_steps=delay_steps,
            pinpool_bins_kb=list(advance.pinpool_bins_kb),
            pinpool_max_mb=int(advance.pinpool_max_mb),
            host_copy_threads=int(advance.host_copy_threads),
            host_copy_queue_size=int(advance.host_copy_queue_size),
        )
    except Exception:
        return None


__all__ = ["MonitoringEngine"]
