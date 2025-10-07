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
    """Simple future wrapper for asynchronously produced tensors."""

    __slots__ = ("_task", "_event", "_result", "_exception")

    def __init__(self, task: MonitoringTask):
        self._task = task
        self._event = threading.Event()
        self._result: Optional[torch.Tensor] = None
        self._exception: Optional[BaseException] = None

    @property
    def task(self) -> MonitoringTask:
        return self._task

    def ready(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        if not self._event.wait(timeout):
            raise TimeoutError("CacheFuture.result timed out before data was ready")
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    def set_result(self, value: torch.Tensor) -> None:
        self._result = value
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)
