"""Task and future primitives for the monitoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from .utils import Slice

# A lightweight payload that can be consumed directly by the native backend.
SlicePayload = Tuple[Any, ...]
NativePayload = Tuple[object, int, bool, bool, SlicePayload, Optional[torch.device]]


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
        return ("identity",)

    if Slice is not None and isinstance(slice_obj, Slice):  # pragma: no cover - optional path
        mode = getattr(slice_obj, "mode", "identity")
        data = getattr(slice_obj, "slice", None)
        if mode == "identity":
            return ("identity",)
        if mode == "int":
            return ("int", int(data) if data is not None else 0)
        if mode == "slice":
            if isinstance(data, slice):
                return ("slice", data.start, data.stop, data.step)
            return ("slice", None, None, None)
        if mode == "array":
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
        return ("int", int(slice_obj))

    if slice_type is slice:
        return ("slice", slice_obj.start, slice_obj.stop, slice_obj.step)

    if slice_type in (list, tuple):
        return ("array", tuple(int(v) for v in slice_obj))

    return ("identity",)


def _encode_slice_native(slice_obj: Any) -> SlicePayload:
    """Fast int-coded slice encoding for native backend.

    Codes: 0=identity, 1=int, 2=slice, 3=array
    """
    if slice_obj is None:
        return (0,)

    if Slice is not None and isinstance(slice_obj, Slice):  # pragma: no cover
        mode = getattr(slice_obj, "mode", "identity")
        data = getattr(slice_obj, "slice", None)
        if mode == "identity":
            return (0,)
        if mode == "int":
            return (1, int(data) if data is not None else 0)
        if mode == "slice":
            if isinstance(data, slice):
                return (2, data.start, data.stop, data.step)
            return (2, None, None, None)
        if mode == "array":
            if data is None:
                values: Tuple[int, ...] = ()
            elif hasattr(data, "tolist"):
                values = tuple(int(v) for v in data.tolist())
            else:
                values = tuple(int(v) for v in data)
            return (3, values)
        return (0,)

    t = type(slice_obj)
    if t is int:
        return (1, int(slice_obj))
    if t is slice:
        return (2, slice_obj.start, slice_obj.stop, slice_obj.step)
    if t in (list, tuple):
        return (3, tuple(int(v) for v in slice_obj))
    return (0,)


class CacheFuture:
    """Future wrapper backed by native backend token handles."""

    __slots__ = (
        "_task",
        "_result",
        "_backend",
        "_token",
    )

    def __init__(self, task: MonitoringTask):
        self._task = task
        self._result: Optional[torch.Tensor] = None
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
        if self._result is not None:
            return True
        if self._backend is None or self._token is None:
            return False
        return bool(self._backend.future_ready(self._token))

    def result(self, timeout: Optional[float] = None) -> torch.Tensor:
        if self._result is not None:
            return self._result
        if self._backend is None or self._token is None:
            raise RuntimeError("CacheFuture is not bound to native backend")
        result = self._backend.future_result(self._token, timeout)
        self._result = result
        # Once materialized, switch to cached mode for repeat calls.
        self._backend = None
        self._token = None
        return result

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._result is not None:
            return True
        if self._backend is None or self._token is None:
            raise RuntimeError("CacheFuture is not bound to native backend")
        return bool(self._backend.future_wait(self._token, timeout))

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


# -----------------------------------------------------------------------------
# Native BackendFuture
# -----------------------------------------------------------------------------


if TYPE_CHECKING:  # pragma: no cover
    from typing import Protocol

    class BackendFuture(Protocol):
        def __init__(self, backend: Any, token: int) -> None: ...
        def ready(self) -> bool: ...
        def wait(self, timeout: Optional[float] = None) -> bool: ...
        def result(
            self, timeout: Optional[float] = None, *, called_from_cpp: bool = False
        ) -> torch.Tensor: ...


_BACKENDFUTURE_CLS: Optional[type] = None


def _load_backendfuture_cls() -> type:
    global _BACKENDFUTURE_CLS
    if _BACKENDFUTURE_CLS is not None:
        return _BACKENDFUTURE_CLS

    # Lazy import so importing monitoring.task does not eagerly build/load the extension.
    from . import _native_engine

    mod = _native_engine._load_extension()
    cls = getattr(mod, "BackendFuture", None)
    if cls is None:
        raise ImportError("Native extension does not export BackendFuture")

    _BACKENDFUTURE_CLS = cls
    # Cache for future `from monitoring.task import BackendFuture` in hot paths.
    globals()["BackendFuture"] = cls
    return cls


def __getattr__(name: str):
    if name == "BackendFuture":
        return _load_backendfuture_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Make BackendFuture visible to dir() / IDEs even before it's resolved.
    return sorted(list(globals().keys()) + ["BackendFuture"])
