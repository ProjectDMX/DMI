"""Monitoring engine wrapper backed only by the native C++/CUDA engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import AdvanceConfig, MonitoringConfig


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
        # Fine-grained Python-side timings (ms)
        self._last_prepare_ms = 0.0

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

    def _prepare_ring_step(self, input_ids: Any, attention_mask: Any, past_key_values: Any,
                           cache_position: Any = None, kv_offsets: Any = None) -> None:
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
                    # 4D causal mask [batch, 1, q_len, kv_dim] -- used by static-cache generate.
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

        import os
        if os.environ.get("RING_DEBUG_STEP"):
            print(f"[ring_step] prefill={is_prefill} token_ranges={token_ranges} finished={list(finished)}")
        self._ring_transport.set_step_context(
            model_id=str(self._model_id),
            req_ids=list(req_ids),
            token_ranges=token_ranges,
            kv_offsets=kv_offsets,
        )

    # ------------------------------------------------------------------
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
