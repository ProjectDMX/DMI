"""Task and future primitives for the monitoring engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from collections import Counter
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .utils import Slice

# A lightweight payload that can be consumed directly by the native backend.
SlicePayload = Tuple[Any, ...]
NativePayload = Tuple[object, int, bool, bool, SlicePayload, Optional[torch.device]]

# Optional slice mode statistics (guarded by env var)
_SLICE_STATS_ENABLED = bool(int(os.environ.get("MON_ENGINE_SLICE_STATS", "0")))
_slice_stats: Counter[str] = Counter()

def get_slice_stats() -> Dict[str, int]:
    return dict(_slice_stats)


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
    _native_payload: Optional[NativePayload] = field(init=False, default=None, repr=False)

    def is_cuda(self) -> bool:
        return self.tensor.is_cuda

    @property
    def native_payload(self) -> NativePayload:
        if self._native_payload is None or self._native_payload[0] is not self.tensor:
            self._native_payload = (
                self.tensor,
                int(self.slice_dim),
                bool(self.metadata.get("remove_batch_dim", False)),
                bool(self.metadata.get("can_slice", True)),
                _encode_slice(self.pos_slice),
                self.target_device,
            )
        return self._native_payload


def _encode_slice(slice_obj: Any) -> SlicePayload:
    """Encode slice metadata into a compact tuple for native consumption."""

    if slice_obj is None:
        if _SLICE_STATS_ENABLED:
            _slice_stats["identity"] += 1
        return ("identity",)

    if Slice is not None and isinstance(slice_obj, Slice):  # pragma: no cover - optional path
        mode = getattr(slice_obj, "mode", "identity")
        data = getattr(slice_obj, "slice", None)
        if mode == "identity":
            if _SLICE_STATS_ENABLED:
                _slice_stats["identity"] += 1
            return ("identity",)
        if mode == "int":
            if _SLICE_STATS_ENABLED:
                _slice_stats["int"] += 1
            return ("int", int(data) if data is not None else 0)
        if mode == "slice":
            if isinstance(data, slice):
                if _SLICE_STATS_ENABLED:
                    _slice_stats["slice"] += 1
                return ("slice", data.start, data.stop, data.step)
            return ("slice", None, None, None)
        if mode == "array":
            if _SLICE_STATS_ENABLED:
                _slice_stats["array"] += 1
            if data is None:
                values: Tuple[int, ...] = ()
            elif hasattr(data, "tolist"):
                values = tuple(int(v) for v in data.tolist())
            else:
                values = tuple(int(v) for v in data)
            return ("array", values)
        return ("identity",)

    slice_type = type(slice_obj)

    if slice_type is int:
        if _SLICE_STATS_ENABLED:
            _slice_stats["int"] += 1
        return ("int", int(slice_obj))

    if slice_type is slice:
        if _SLICE_STATS_ENABLED:
            _slice_stats["slice"] += 1
        return ("slice", slice_obj.start, slice_obj.stop, slice_obj.step)

    if slice_type in (list, tuple):
        if _SLICE_STATS_ENABLED:
            _slice_stats["array"] += 1
        return ("array", tuple(int(v) for v in slice_obj))

    if _SLICE_STATS_ENABLED:
        _slice_stats["identity"] += 1
    return ("identity",)


def _encode_slice_native(slice_obj: Any) -> SlicePayload:
    """Fast int-coded slice encoding for native backend.

    Codes: 0=identity, 1=int, 2=slice, 3=array
    """
    if slice_obj is None:
        if _SLICE_STATS_ENABLED:
            _slice_stats["identity"] += 1
        return (0,)

    if Slice is not None and isinstance(slice_obj, Slice):  # pragma: no cover
        mode = getattr(slice_obj, "mode", "identity")
        data = getattr(slice_obj, "slice", None)
        if mode == "identity":
            if _SLICE_STATS_ENABLED:
                _slice_stats["identity"] += 1
            return (0,)
        if mode == "int":
            if _SLICE_STATS_ENABLED:
                _slice_stats["int"] += 1
            return (1, int(data) if data is not None else 0)
        if mode == "slice":
            if isinstance(data, slice):
                if _SLICE_STATS_ENABLED:
                    _slice_stats["slice"] += 1
                return (2, data.start, data.stop, data.step)
            return (2, None, None, None)
        if mode == "array":
            if _SLICE_STATS_ENABLED:
                _slice_stats["array"] += 1
            if data is None:
                values: Tuple[int, ...] = ()
            elif hasattr(data, "tolist"):
                values = tuple(int(v) for v in data.tolist())
            else:
                values = tuple(int(v) for v in data)
            return (3, values)
        if _SLICE_STATS_ENABLED:
            _slice_stats["identity"] += 1
        return (0,)

    t = type(slice_obj)
    if t is int:
        if _SLICE_STATS_ENABLED:
            _slice_stats["int"] += 1
        return (1, int(slice_obj))
    if t is slice:
        if _SLICE_STATS_ENABLED:
            _slice_stats["slice"] += 1
        return (2, slice_obj.start, slice_obj.stop, slice_obj.step)
    if t in (list, tuple):
        if _SLICE_STATS_ENABLED:
            _slice_stats["array"] += 1
        return (3, tuple(int(v) for v in slice_obj))
    if _SLICE_STATS_ENABLED:
        _slice_stats["identity"] += 1
    return (0,)


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

    def discard(self) -> None:
        """Discard the result without retrieving it, freeing memory."""
        if self._backend is not None and self._token is not None:
            try:
                # Retrieve result to trigger C++ cleanup, but do not store it
                _ = self._backend.future_result(self._token, 0.001)
                self._backend = None
                self._token = None
            except Exception:
                # Already retrieved, timed out, or other error - that is fine
                pass


class BackendFuture:
    """Lightweight future for native backend tokens (no threading.Event).

    This avoids Python-side allocation overhead for the hot path where the
    native backend manages readiness and results.
    """

    __slots__ = ("_backend", "_token")

    def __init__(self, backend: Any, token: int) -> None:  # type: ignore[name-defined]
        self._backend = backend
        self._token = int(token)

    def ready(self) -> bool:
        return bool(self._backend.future_ready(self._token))

    def wait(self, timeout: Optional[float] = None) -> bool:
        return bool(self._backend.future_wait(self._token, timeout))

    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        return self._backend.future_result(self._token, timeout)
