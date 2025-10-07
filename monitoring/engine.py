"""Asynchronous monitoring engine for hook activations."""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional, Tuple

import torch

from .task import CacheFuture, MonitoringTask

try:  # Optional import to avoid circular dependency at runtime
    from transformer_lens.utils import Slice
except Exception:  # pragma: no cover - transformer_lens may be absent in some envs
    Slice = None


class MonitoringEngine:
    """Background processor that prepares hook activations asynchronously."""

    def __init__(
        self,
        *,
        async_enabled: bool = True,
        queue_size: int = 0,
        cache_dtype: Optional[torch.dtype] = None,
        delay_steps: int = 0,
        max_coalesce_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.async_enabled = async_enabled
        self._queue: "queue.Queue[Tuple[MonitoringTask, CacheFuture]]" = queue.Queue(maxsize=queue_size or 0)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._cache_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.Lock()
        self._current_step_id: int = 0
        self._last_waited_step: Optional[int] = None
        self.cache_dtype = cache_dtype
        self._step_buckets: dict[int, list[Tuple[MonitoringTask, CacheFuture]]] = {}
        self._sealed_steps: list[int] = []
        self._delay_steps = max(0, int(delay_steps))
        self._max_coalesce_bytes = int(max_coalesce_bytes)
        env_max = os.environ.get("MON_ENGINE_MAX_COALESCE_MB")
        if env_max:
            try:
                self._max_coalesce_bytes = int(env_max) * 1024 * 1024
            except Exception:
                if self._debug:
                    print(f"[MonEng] invalid MON_ENGINE_MAX_COALESCE_MB={env_max}")
        self._debug = bool(int(os.environ.get("MON_ENGINE_DEBUG", "0")))

    # ------------------------------------------------------------------
    # Public API

    def submit(self, task: MonitoringTask) -> CacheFuture:
        """Accept a monitoring task and schedule its processing."""

        future = CacheFuture(task)

        if not self._should_process_async(task):
            try:
                result = self._process_task(task)
            except BaseException as exc:  # pragma: no cover - propagate diagnostics
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

        self._ensure_started()
        assert self._cache_stream is not None

        # Assign step id for aggregated synchronization
        task.step_id = self._current_step_id
        # Bucket by step; processing will be triggered on end_step
        with self._lock:
            self._step_buckets.setdefault(task.step_id, []).append((task, future))
            if self._debug:
                size = len(self._step_buckets[task.step_id])
                print(f"[MonEng] submit step={task.step_id} bucket_size={size}")
        return future

    def start_step(self) -> None:
        """Mark the beginning of a producer step (no sync here)."""
        if not (self.async_enabled and torch.cuda.is_available()):
            return
        with self._lock:
            self._current_step_id += 1
            self._last_waited_step = None
            if self._debug:
                print(f"[MonEng] start_step -> current_step_id={self._current_step_id}")

    def end_step(self) -> None:
        """Seal the current step: establish stream dependency and enqueue ready steps."""
        if not (self.async_enabled and torch.cuda.is_available()):
            return
        with self._lock:
            # Establish dependency now that producer work for this step has been queued
            if self._cache_stream is not None:
                try:
                    self._cache_stream.wait_stream(torch.cuda.current_stream())
                except Exception:
                    pass
            # Mark the just-finished step as sealed
            self._sealed_steps.append(self._current_step_id)
            if self._debug:
                print(f"[MonEng] end_step sealed step_id={self._current_step_id} sealed_len={len(self._sealed_steps)} delay={self._delay_steps}")
            # Determine which sealed step is ready to process based on delay_steps
            while len(self._sealed_steps) > self._delay_steps:
                step_to_process = self._sealed_steps.pop(0)
                bucket = self._step_buckets.pop(step_to_process, [])
                if self._debug:
                    print(f"[MonEng] enqueue step={step_to_process} n_items={len(bucket)}")
                for item in bucket:
                    # Enqueue items; worker will coalesce per step upon retrieval
                    self._queue.put(item)

    def resolve_all(self, timeout: Optional[float] = None) -> None:
        """Block until all queued tasks have been processed."""

        if not self.async_enabled:
            return
        # Flush any sealed-but-not-enqueued steps
        with self._lock:
            # Enqueue sealed steps
            for step_id in list(self._sealed_steps):
                bucket = self._step_buckets.pop(step_id, [])
                if self._debug:
                    print(f"[MonEng] resolve_all enqueue sealed step={step_id} n_items={len(bucket)}")
                for item in bucket:
                    self._queue.put(item)
            self._sealed_steps.clear()
            # Enqueue any remaining buckets (eg, if end_step was not reached due to early exit)
            for step_id, bucket in list(self._step_buckets.items()):
                if not bucket:
                    continue
                if self._debug:
                    print(f"[MonEng] resolve_all enqueue leftover step={step_id} n_items={len(bucket)}")
                for item in bucket:
                    self._queue.put(item)
                self._step_buckets.pop(step_id, None)
        self._queue.join()

    def close(self) -> None:
        """Request graceful shutdown of background resources."""

        if not self.async_enabled:
            return

        with self._lock:
            if self._worker is None:
                return
            self._stop.set()
            self._queue.put((None, None))  # type: ignore[arg-type]
            self._worker.join()
            self._worker = None
            self._cache_stream = None
            self._stop.clear()

    # ------------------------------------------------------------------
    # Internal helpers

    def _should_process_async(self, task: MonitoringTask) -> bool:
        if not self.async_enabled:
            return False
        if not torch.cuda.is_available():
            return False
        if not task.is_cuda():
            return False
        return True

    def _ensure_started(self) -> None:
        with self._lock:
            if self._worker is not None:
                return
            # Use lowest priority stream to reduce interference with main compute
            try:
                min_pri, max_pri = torch.cuda.priority_range()
                self._cache_stream = torch.cuda.Stream(priority=max_pri)
            except Exception:
                self._cache_stream = torch.cuda.Stream()
            self._worker = threading.Thread(target=self._worker_loop, name="monitoring-engine", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        assert self._cache_stream is not None
        if self._debug:
            print("[MonEng] worker started")
        while not self._stop.is_set():
            task_item = self._queue.get()
            if task_item[0] is None:  # type: ignore[index]
                # mark sentinel done and exit
                self._queue.task_done()
                if self._debug:
                    print("[MonEng] worker got sentinel, exiting")
                break
            # Drain all items currently queued for this step (best-effort grouping)
            first_task, first_future = task_item  # type: ignore[assignment]
            step_id = first_task.step_id
            batch = [(first_task, first_future)]
            try:
                # Non-blocking drain of same-step items
                while True:
                    t2, f2 = self._queue.get_nowait()
                    if t2 is None:  # type: ignore
                        # push back the sentinel and stop
                        self._queue.put((t2, f2))  # type: ignore
                        break
                    if t2.step_id == step_id:
                        batch.append((t2, f2))
                    else:
                        # Not same step, push back and stop draining
                        self._queue.put((t2, f2))
                        break
            except queue.Empty:
                pass

            # Process batch with coalesced buffers
            try:
                if self._debug:
                    print(f"[MonEng] worker processing step={step_id} batch={len(batch)}")
                self._process_batch_async(batch)
            except BaseException as exc:  # pragma: no cover
                for _, fut in batch:
                    fut.set_exception(exc)
            finally:
                # Mark all batch items as done
                for _ in batch:
                    self._queue.task_done()

    def _process_batch_async(self, batch: list[Tuple[MonitoringTask, CacheFuture]]) -> None:
        assert self._cache_stream is not None
        with torch.cuda.stream(self._cache_stream):
            # Stream dependency already set at step start. Process each task independently
            # to avoid the extra D2D memcpy introduced by the large coalesced buffer path.
            for task, fut in batch:
                tensor = self._process_task(task)
                if self.cache_dtype is not None and tensor.dtype != self.cache_dtype:
                    tensor = tensor.to(self.cache_dtype)
                fut.set_result(tensor)

    def _process_task(self, task: MonitoringTask) -> torch.Tensor:
        tensor = task.tensor

        if task.target_device is not None and tensor.device != task.target_device:
            tensor = tensor.to(task.target_device, non_blocking=True)

        if task.metadata.get("remove_batch_dim"):
            tensor = tensor[0]

        can_slice = task.metadata.get("can_slice")
        if can_slice is None:
            can_slice = True

        if task.pos_slice is not None and can_slice:
            tensor = self._apply_pos_slice(tensor, task)

        if self.cache_dtype is not None and tensor.dtype != self.cache_dtype:
            tensor = tensor.to(self.cache_dtype)

        return tensor

    def _apply_pos_slice(self, tensor: torch.Tensor, task: MonitoringTask) -> torch.Tensor:
        slice_obj = task.pos_slice
        if slice_obj is None:
            return tensor

        if Slice is not None and isinstance(slice_obj, Slice):
            if getattr(slice_obj, "mode", None) == "identity":
                return tensor
            return slice_obj.apply(tensor, dim=task.slice_dim)

        if hasattr(slice_obj, "apply"):
            return slice_obj.apply(tensor, dim=task.slice_dim)

        # Fallback: assume slice_obj is fancy indexing descriptor
        return tensor[(slice_obj,)]  # pragma: no cover - defensive fallback

    # ------------------------------------------------------------------
    # Context manager convenience

    def __enter__(self) -> "MonitoringEngine":
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["MonitoringEngine"]
