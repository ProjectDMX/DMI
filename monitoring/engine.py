"""Asynchronous monitoring engine for hook activations."""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Deque, List, Optional, Tuple, Union

import torch

from .task import CacheFuture, MonitoringTask

try:  # Optional import to avoid circular dependency at runtime
    from transformer_lens.utils import Slice
except Exception:  # pragma: no cover - transformer_lens may be absent in some envs
    Slice = None


@dataclass
class _StepWork:
    """Container holding all monitoring tasks for a single decode step."""

    step_id: int
    tasks: List[Tuple[MonitoringTask, CacheFuture]]


class _StepQueue:
    """Lightweight single-producer/single-consumer queue backed by a ring buffer."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize if maxsize > 0 else None
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock) if self._maxsize else None
        self._all_tasks_done = threading.Condition(self._lock)
        if self._maxsize:
            self._buffer: Union[List[Optional[_StepWork]], Deque[Optional[_StepWork]]] = [None] * self._maxsize
            self._head = 0
            self._tail = 0
            self._size = 0
        else:
            self._buffer = deque()  # type: ignore[assignment]
        self._unfinished_tasks = 0

    def put(self, item: Union[_StepWork, object]) -> None:
        with self._lock:
            if self._maxsize:
                assert isinstance(self._buffer, list)
                assert self._not_full is not None
                while self._size == self._maxsize:
                    self._not_full.wait()
                self._buffer[self._tail] = item  # type: ignore[index]
                self._tail = (self._tail + 1) % self._maxsize
                self._size += 1
            else:
                assert isinstance(self._buffer, deque)
                self._buffer.append(item)  # type: ignore[arg-type]
            self._unfinished_tasks += 1
            self._not_empty.notify()

    def get(self) -> Union[_StepWork, object]:
        with self._lock:
            while True:
                if self._maxsize:
                    assert isinstance(self._buffer, list)
                    if self._size:
                        item = self._buffer[self._head]
                        self._buffer[self._head] = None
                        self._head = (self._head + 1) % self._maxsize
                        self._size -= 1
                        if self._not_full is not None:
                            self._not_full.notify()
                        return item  # type: ignore[return-value]
                else:
                    assert isinstance(self._buffer, deque)
                    if self._buffer:
                        return self._buffer.popleft()
                self._not_empty.wait()

    def task_done(self) -> None:
        with self._lock:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._all_tasks_done.notify_all()

    def join(self) -> None:
        with self._lock:
            while self._unfinished_tasks:
                self._all_tasks_done.wait()


class _SimpleStepQueue:
    """Wrapper around queue.SimpleQueue providing join/task_done semantics."""

    def __init__(self) -> None:
        self._queue: SimpleQueue = SimpleQueue()
        self._lock = threading.Lock()
        self._all_tasks_done = threading.Condition(self._lock)
        self._unfinished_tasks = 0

    def put(self, item: Union[_StepWork, object]) -> None:
        with self._lock:
            self._unfinished_tasks += 1
        self._queue.put(item)

    def get(self) -> Union[_StepWork, object]:
        return self._queue.get()

    def task_done(self) -> None:
        with self._lock:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._all_tasks_done.notify_all()

    def join(self) -> None:
        with self._lock:
            while self._unfinished_tasks:
                self._all_tasks_done.wait()


_QUEUE_SENTINEL = object()


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
        self._debug = bool(int(os.environ.get("MON_ENGINE_DEBUG", "0")))
        if queue_size > 0:
            self._queue: Union[_StepQueue, _SimpleStepQueue] = _StepQueue(queue_size)
        else:
            self._queue = _SimpleStepQueue()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._cache_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.Lock()
        self._current_step_id: int = 0
        self._last_waited_step: Optional[int] = None
        self.cache_dtype = cache_dtype
        self._step_buckets: dict[int, list[Tuple[MonitoringTask, CacheFuture]]] = {}
        self._sealed_steps: Deque[int] = deque()
        self._delay_steps = max(0, int(delay_steps))
        env_max = os.environ.get("MON_ENGINE_MAX_COALESCE_MB")
        if env_max:
            try:
                _ = int(env_max) * 1024 * 1024  # retained for backwards compatibility (no longer used)
            except Exception:
                if self._debug:
                    print(f"[MonEng] invalid MON_ENGINE_MAX_COALESCE_MB={env_max}")

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
        cache_stream = self._cache_stream
        if cache_stream is not None:
            try:
                cache_stream.wait_stream(torch.cuda.current_stream())
            except Exception:
                pass
        with self._lock:
            # Mark the just-finished step as sealed
            self._sealed_steps.append(self._current_step_id)
            if self._debug:
                print(f"[MonEng] end_step sealed step_id={self._current_step_id} sealed_len={len(self._sealed_steps)} delay={self._delay_steps}")
            # Determine which sealed step is ready to process based on delay_steps
            while len(self._sealed_steps) > self._delay_steps:
                step_to_process = self._sealed_steps.popleft()
                bucket = self._step_buckets.pop(step_to_process, [])
                if not bucket:
                    continue
                if self._debug:
                    print(
                        f"[MonEng] enqueue step={step_to_process} n_items={len(bucket)}"
                    )
                self._queue.put(_StepWork(step_to_process, bucket))

    def resolve_all(self, timeout: Optional[float] = None) -> None:
        """Block until all queued tasks have been processed."""

        if not self.async_enabled:
            return
        # Flush any sealed-but-not-enqueued steps
        with self._lock:
            sealed_ids = list(self._sealed_steps)
            self._sealed_steps.clear()
            for step_id in sealed_ids:
                bucket = self._step_buckets.pop(step_id, [])
                if not bucket:
                    continue
                if self._debug:
                    print(
                        f"[MonEng] resolve_all enqueue sealed step={step_id} n_items={len(bucket)}"
                    )
                self._queue.put(_StepWork(step_id, bucket))

            remaining_ids = list(self._step_buckets.keys())
            for step_id in remaining_ids:
                bucket = self._step_buckets.pop(step_id, [])
                if not bucket:
                    continue
                if self._debug:
                    print(
                        f"[MonEng] resolve_all enqueue leftover step={step_id} n_items={len(bucket)}"
                    )
                self._queue.put(_StepWork(step_id, bucket))
        self._queue.join()

    def close(self) -> None:
        """Request graceful shutdown of background resources."""

        if not self.async_enabled:
            return

        with self._lock:
            if self._worker is None:
                return
            self._stop.set()
            self._queue.put(_QUEUE_SENTINEL)
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
        while True:
            work = self._queue.get()
            if work is _QUEUE_SENTINEL:
                self._queue.task_done()
                if self._debug:
                    print("[MonEng] worker got sentinel, exiting")
                break
            assert isinstance(work, _StepWork)
            try:
                if self._debug:
                    print(
                        f"[MonEng] worker processing step={work.step_id} batch={len(work.tasks)}"
                    )
                self._process_step_work(work)
            except BaseException as exc:  # pragma: no cover
                for _, fut in work.tasks:
                    fut.set_exception(exc)
            finally:
                work.tasks.clear()
                self._queue.task_done()

    def _process_step_work(self, work: _StepWork) -> None:
        assert self._cache_stream is not None
        with torch.cuda.stream(self._cache_stream):
            for task, fut in work.tasks:
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
