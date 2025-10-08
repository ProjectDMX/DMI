"""Monitoring engine wrapper with Python fallback backend.

This module defines a thin Python-fronted MonitoringEngine that aggregates
per-step tasks and delegates all heavy lifting to a backend implementation.
In production the backend will be a native C++/CUDA extension; until that
arrives we retain a Python fallback that mirrors the established behaviour.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .task import CacheFuture, MonitoringTask

try:  # Optional import to avoid circular dependency at runtime
    from transformer_lens.utils import Slice
except Exception:  # pragma: no cover - transformer_lens may be absent in some envs
    Slice = None


_QUEUE_SENTINEL = object()


def _stream_to_handle(stream: Optional[torch.cuda.Stream]) -> Optional[int]:
    if stream is None:
        return None
    try:
        return int(stream.cuda_stream)
    except AttributeError:
        return None


def _serialize_slice(slice_obj: Any) -> Dict[str, Any]:
    if slice_obj is None:
        return {"mode": "identity"}

    if Slice is not None and isinstance(slice_obj, Slice):
        mode = getattr(slice_obj, "mode", "identity")
        data = getattr(slice_obj, "slice", slice(None))
    else:
        mode = "identity"
        data = slice_obj
        if isinstance(slice_obj, int):
            mode = "int"
        elif isinstance(slice_obj, slice):
            mode = "slice"
        elif isinstance(slice_obj, (list, tuple)):
            mode = "array"

    if mode == "identity":
        return {"mode": "identity"}

    if mode == "int":
        return {"mode": "int", "value": int(data)}

    if mode == "slice":
        return {
            "mode": "slice",
            "start": data.start,
            "stop": data.stop,
            "step": data.step,
        }

    if mode == "array":
        if hasattr(data, "tolist"):
            values = data.tolist()
        else:
            values = list(data)
        return {"mode": "array", "indices": [int(v) for v in values]}

    # Fallback – treat as identity
    return {"mode": "identity"}


def _serialize_task(task: MonitoringTask) -> Dict[str, Any]:
    metadata = task.metadata
    remove_batch_dim = bool(metadata.get("remove_batch_dim"))
    can_slice = metadata.get("can_slice")
    if can_slice is None:
        can_slice = True

    return {
        "tensor": task.tensor,
        "slice_dim": int(task.slice_dim),
        "remove_batch_dim": bool(remove_batch_dim),
        "can_slice": bool(can_slice),
        "slice": _serialize_slice(task.pos_slice),
        "target_device": task.target_device,
    }


class MonitoringEngine:
    """High-level wrapper that routes monitoring tasks to a backend."""

    def __init__(
        self,
        *,
        async_enabled: bool = True,
        queue_size: int = 0,
        cache_dtype: Optional[torch.dtype] = None,
        delay_steps: int = 0,
    ) -> None:
        self.async_enabled = async_enabled
        self.cache_dtype = cache_dtype
        self._delay_steps = max(0, int(delay_steps))
        self._debug = bool(int(os.environ.get("MON_ENGINE_DEBUG", "0")))

        self._current_step_id: int = 0
        self._pending_tasks: Dict[int, List[Tuple[MonitoringTask, CacheFuture]]] = {}

        # Backend references are populated on demand. When a native backend is
        # unavailable we fall back to the Python implementation.
        self._backend: Optional[Any] = None
        self._python_backend: Optional[_PythonBackend] = None
        self._native_backend: Optional[Any] = None
        self._using_native_backend = False

        if async_enabled and torch.cuda.is_available():
            native_backend = _load_native_backend(queue_size, cache_dtype, self._delay_steps)
            if native_backend is not None:
                self._backend = native_backend
                self._native_backend = native_backend
                self._using_native_backend = True
            else:
                python_backend = _PythonBackend(
                    queue_size=queue_size,
                    cache_dtype=cache_dtype,
                    delay_steps=self._delay_steps,
                    debug=self._debug,
                )
                self._backend = python_backend
                self._python_backend = python_backend

    # ------------------------------------------------------------------
    # Public API

    def submit(self, task: MonitoringTask) -> CacheFuture:
        """Register a monitoring task and queue it for processing."""

        future = CacheFuture(task)

        if not self._should_process_async(task):
            try:
                result = _process_task_sync(task, cache_dtype=self.cache_dtype)
            except BaseException as exc:  # pragma: no cover - diagnostics path
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

        assert self._backend is not None  # backend must exist in async path

        step_id = self._current_step_id
        task.step_id = step_id

        bucket = self._pending_tasks.setdefault(step_id, [])
        bucket.append((task, future))
        if self._debug:
            print(f"[MonEng] submit step={step_id} bucket_size={len(bucket)}")

        return future

    def start_step(self) -> None:
        """Mark the beginning of a decode/prefill step."""

        if not (self.async_enabled and torch.cuda.is_available()):
            return

        self._current_step_id += 1
        if self._debug:
            print(f"[MonEng] start_step -> step_id={self._current_step_id}")

    def end_step(self) -> None:
        """Seal the current step and hand it to the backend."""

        if not (self.async_enabled and torch.cuda.is_available()):
            return

        step_id = self._current_step_id
        tasks = self._pending_tasks.pop(step_id, [])

        if self._debug:
            print(
                f"[MonEng] end_step step_id={step_id} tasks={len(tasks)} delay={self._delay_steps}"
            )

        try:
            producer_stream = torch.cuda.current_stream()
        except RuntimeError:
            producer_stream = None

        if self._using_native_backend:
            backend = self._native_backend
            if backend is None:
                return
            task_specs = [_serialize_task(task) for task, _ in tasks]
            stream_handle = _stream_to_handle(producer_stream)
            tokens = backend.submit_step(step_id, task_specs, stream_handle)
            for token, (_, future) in zip(tokens, tasks):
                future.bind_backend(backend, token)
            return

        backend = self._python_backend
        if backend is None:
            return
        backend.submit_step(step_id, tasks, producer_stream)

    def resolve_all(self) -> None:
        """Block until all pending tasks have been processed."""

        if not self.async_enabled:
            return

        if self._using_native_backend:
            backend = self._native_backend
            if backend is None:
                return
            if self._pending_tasks:
                for step_id in sorted(self._pending_tasks.keys()):
                    tasks = self._pending_tasks.pop(step_id)
                    task_specs = [_serialize_task(task) for task, _ in tasks]
                    tokens = backend.submit_step(step_id, task_specs, None)
                    for token, (_, future) in zip(tokens, tasks):
                        future.bind_backend(backend, token)
            backend.resolve_all()
            return

        backend = self._python_backend
        if backend is None:
            return
        if self._pending_tasks:
            for step_id in sorted(self._pending_tasks.keys()):
                tasks = self._pending_tasks.pop(step_id)
                backend.submit_step(step_id, tasks, None)
        backend.resolve_all()

    def close(self) -> None:
        """Tear down backend resources."""

        if self._using_native_backend:
            backend = self._native_backend
            if backend is None:
                return
            backend.close()
            self._native_backend = None
            self._backend = None
            return

        backend = self._python_backend
        if backend is None:
            return
        backend.close()
        self._python_backend = None
        self._backend = None

    # ------------------------------------------------------------------
    # Internal helpers

    def _should_process_async(self, task: MonitoringTask) -> bool:
        if not self.async_enabled:
            return False
        if not torch.cuda.is_available():
            return False
        if not task.is_cuda():
            return False
        if self._backend is None:
            return False
        return True


# ---------------------------------------------------------------------------
# Backend loaders


def _load_native_backend(
    queue_size: int,
    cache_dtype: Optional[torch.dtype],
    delay_steps: int,
) -> Optional[Any]:
    """Attempt to load the native backend extension.

    Returns None when the extension is unavailable. The actual native backend
    will be implemented in subsequent phases; this loader prepares the hook.
    """

    try:
        from . import _native_engine  # pragma: no cover - module absent today
    except Exception:
        return None

    try:
        return _native_engine.create_engine(  # type: ignore[attr-defined]
            queue_size=queue_size,
            cache_dtype=cache_dtype,
            delay_steps=delay_steps,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Python fallback backend (mirrors existing behaviour)


@dataclass
class _StepWork:
    step_id: int
    tasks: List[Tuple[MonitoringTask, CacheFuture]]


class _StepQueue:
    """Single-producer/single-consumer queue with optional max size."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize if maxsize > 0 else None
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock) if self._maxsize else None
        if self._maxsize:
            self._buffer: Union[List[Optional[_StepWork]], Deque[Optional[_StepWork]]] = [None] * self._maxsize
            self._head = 0
            self._tail = 0
            self._size = 0
        else:
            self._buffer = deque()  # type: ignore[assignment]
        self._unfinished_tasks = 0
        self._all_tasks_done = threading.Condition(self._lock)

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
    """queue.SimpleQueue wrapper that adds join semantics."""

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


class _PythonBackend:
    """Existing Python implementation retained as fallback."""

    def __init__(
        self,
        *,
        queue_size: int,
        cache_dtype: Optional[torch.dtype],
        delay_steps: int,
        debug: bool,
    ) -> None:
        self.cache_dtype = cache_dtype
        self._debug = debug
        self._delay_steps = delay_steps

        if queue_size > 0:
            self._queue: Union[_StepQueue, _SimpleStepQueue] = _StepQueue(queue_size)
        else:
            self._queue = _SimpleStepQueue()

        self._worker: Optional[threading.Thread] = None
        self._cache_stream: Optional[torch.cuda.Stream] = None
        self._stop_event = threading.Event()

        self._sealed_steps: Deque[int] = deque()
        self._step_buckets: Dict[int, List[Tuple[MonitoringTask, CacheFuture]]] = {}
        self._lock = threading.Lock()

    # BackendProtocol ------------------------------------------------------------------

    def submit_step(
        self,
        step_id: int,
        tasks: Iterable[Tuple[MonitoringTask, CacheFuture]],
        producer_stream: Optional[torch.cuda.Stream],
    ) -> None:
        tasks_list = list(tasks)

        with self._lock:
            self._step_buckets[step_id] = tasks_list
            self._sealed_steps.append(step_id)

        if not tasks_list and self._debug:
            print(f"[MonEng/Py] sealed empty step={step_id}")

        if tasks_list:
            self._ensure_started()
            if self._cache_stream is not None and producer_stream is not None:
                try:
                    self._cache_stream.wait_stream(producer_stream)
                except Exception:
                    if self._debug:
                        print(f"[MonEng/Py] wait_stream failed for step={step_id}")

        self._enqueue_ready_steps()

    def resolve_all(self) -> None:
        self._enqueue_all_steps()
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._worker is None:
                return
            self._stop_event.set()
            self._queue.put(_QUEUE_SENTINEL)
            worker = self._worker
            self._worker = None
            cache_stream = self._cache_stream
            self._cache_stream = None

        if worker is not None:
            worker.join()

        if cache_stream is not None:
            del cache_stream

        self._stop_event.clear()

    # Internal helpers ----------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._worker is not None:
            return
        with self._lock:
            if self._worker is not None:
                return
            try:
                _, max_pri = torch.cuda.priority_range()
                self._cache_stream = torch.cuda.Stream(priority=max_pri)
            except Exception:
                self._cache_stream = torch.cuda.Stream()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="monitoring-engine",
                daemon=True,
            )
            self._worker.start()
            if self._debug:
                print("[MonEng/Py] worker started")

    def _enqueue_ready_steps(self) -> None:
        with self._lock:
            while len(self._sealed_steps) > self._delay_steps:
                step_id = self._sealed_steps.popleft()
                tasks = self._step_buckets.pop(step_id, [])
                if not tasks:
                    continue
                if self._debug:
                    print(f"[MonEng/Py] enqueue step={step_id} tasks={len(tasks)}")
                self._queue.put(_StepWork(step_id, tasks))

    def _enqueue_all_steps(self) -> None:
        with self._lock:
            while self._sealed_steps:
                step_id = self._sealed_steps.popleft()
                tasks = self._step_buckets.pop(step_id, [])
                if not tasks:
                    continue
                self._queue.put(_StepWork(step_id, tasks))

    def _worker_loop(self) -> None:
        assert self._cache_stream is not None
        cache_stream = self._cache_stream
        while True:
            work = self._queue.get()
            if work is _QUEUE_SENTINEL:
                self._queue.task_done()
                if self._debug:
                    print("[MonEng/Py] worker exiting")
                break

            assert isinstance(work, _StepWork)
            try:
                with torch.cuda.stream(cache_stream):
                    for task, future in work.tasks:
                        tensor = _process_task_sync(task, cache_dtype=self.cache_dtype)
                        future.set_result(tensor)
            except BaseException as exc:  # pragma: no cover - propagate diag
                for _, future in work.tasks:
                    future.set_exception(exc)
            finally:
                work.tasks.clear()
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Shared helpers


def _process_task_sync(
    task: MonitoringTask,
    *,
    cache_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tensor = task.tensor

    if task.target_device is not None and tensor.device != task.target_device:
        tensor = tensor.to(task.target_device, non_blocking=True)

    if task.metadata.get("remove_batch_dim"):
        tensor = tensor[0]

    can_slice = task.metadata.get("can_slice")
    if can_slice is None:
        can_slice = True

    if task.pos_slice is not None and can_slice:
        tensor = _apply_pos_slice(tensor, task)

    if cache_dtype is not None and tensor.dtype != cache_dtype:
        tensor = tensor.to(cache_dtype)

    return tensor


def _apply_pos_slice(tensor: torch.Tensor, task: MonitoringTask) -> torch.Tensor:
    slice_obj = task.pos_slice
    if slice_obj is None:
        return tensor

    if Slice is not None and isinstance(slice_obj, Slice):
        if getattr(slice_obj, "mode", None) == "identity":
            return tensor
        return slice_obj.apply(tensor, dim=task.slice_dim)

    if hasattr(slice_obj, "apply"):
        return slice_obj.apply(tensor, dim=task.slice_dim)

    return tensor[(slice_obj,)]  # pragma: no cover - defensive fallback


__all__ = ["MonitoringEngine"]
