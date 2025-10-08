"""Task and future primitives for the monitoring engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class MonitoringTask:
    """Container describing a single monitoring job for a hook activation."""

    name: str
    tensor: torch.Tensor
    pos_slice: Optional[Any] = None
    slice_dim: int = -2
    event: Optional[torch.cuda.Event] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_device: Optional[torch.device] = None
    step_id: Optional[int] = None

    def is_cuda(self) -> bool:
        return self.tensor.is_cuda


class CacheFuture:
    """Future wrapper that supports Python fallback and native backends."""

    __slots__ = (
        "_task",
        "_event",
        "_result",
        "_exception",
        "_backend",
        "_token",
    )

    def __init__(self, task: MonitoringTask):
        self._task = task
        self._event = threading.Event()
        self._result: Optional[torch.Tensor] = None
        self._exception: Optional[BaseException] = None
        self._backend: Optional[Any] = None
        self._token: Optional[int] = None

    @property
    def task(self) -> MonitoringTask:
        return self._task

    def bind_backend(self, backend: Any, token: int) -> None:
        """Attach a native backend handle to this future."""

        self._backend = backend
        self._token = int(token)

    def ready(self) -> bool:
        if self._backend is not None and self._token is not None:
            return bool(self._backend.future_ready(self._token))
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        if self._backend is not None and self._token is not None:
            result = self._backend.future_result(self._token, timeout)
            self._result = result
            self._exception = None
            self._event.set()
            # Once materialised, switch to cached mode for repeat calls.
            self._backend = None
            self._token = None
            return result

        if not self._event.wait(timeout):
            raise TimeoutError("CacheFuture.result timed out before data was ready")
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    def set_result(self, value: torch.Tensor) -> None:
        if self._backend is not None:
            raise RuntimeError("Cannot set result while bound to a native backend")
        self._result = value
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        if self._backend is not None:
            raise RuntimeError("Cannot set exception while bound to a native backend")
        self._exception = exc
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._backend is not None and self._token is not None:
            return bool(self._backend.future_wait(self._token, timeout))
        return self._event.wait(timeout)
