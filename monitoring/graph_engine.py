from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from .config import MonitoringConfig
from .graph_monitor import GraphMonitor, ModuleFilter, SlotInfo, METADATA_BYTES
from .task import CacheFuture, MonitoringTask

if TYPE_CHECKING:
    from .graph_consumer import GraphSlotConsumer
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")


@dataclass(frozen=True)
class GraphSlotResult:
    """Decoded metadata for a single captured slot."""

    slot_id: int
    name: str
    step_id: int
    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    ndim: int
    dtype_id: int
    device_index: int


class GraphSafeEngine:
    """Step-oriented monitoring engine backed by GraphMonitor metadata."""

    def __init__(
        self,
        *,
        config: Optional[MonitoringConfig] = None,
        module_filter: Optional[ModuleFilter] = None,
        max_slots: int = 4096,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self._module_filter = module_filter
        self._max_slots = max_slots
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.async_enabled = device.type == "cuda"
        self.cache_dtype: Optional[torch.dtype] = None  # API parity with MonitoringEngine
        self._monitor: Optional[GraphMonitor] = None
        self._slot_mapping: Dict[int, SlotInfo] = {}
        self._step_event_stream: Optional[torch.cuda.Stream] = None

        self._request_capture_enabled = True
        self._capture_enabled = True
        self._current_step_id = 0
        self._last_ready_step_id = 0
        self._prepare_ms = 0.0
        self._consumer: Optional["GraphSlotConsumer"] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None

    def prepare_for_model(
        self,
        model: nn.Module,
        *,
        module_filter: Optional[ModuleFilter] = None,
        **_: object,
    ) -> float:
        """Instantiate GraphMonitor for the provided model."""

        if self._device.type != "cuda":
            raise RuntimeError("GraphSafeEngine currently requires a CUDA device.")

        start = time.perf_counter()
        allowlist = self._compile_allowed_names(model)
        combined_filter = self._compose_filter(module_filter or self._module_filter, allowlist)
        self.close()
        self._monitor = GraphMonitor(
            model,
            max_slots=self._max_slots,
            module_filter=combined_filter,
            device=self._device,
        )
        self._slot_mapping = self._monitor.get_slot_mapping()
        self._prepare_ms = (time.perf_counter() - start) * 1e3
        return self._prepare_ms

    def finalize_capture(self) -> None:
        if self._monitor is None:
            raise RuntimeError("GraphSafeEngine.prepare_for_model must be called first.")
        self._monitor.finalize_capture()

    # ------------------------------------------------------------------
    # Step orchestration

    def begin_request(self, request_id: int) -> None:
        if self.config is None:
            self._request_capture_enabled = True
            return
        self._request_capture_enabled = bool(
            self.config.schedule.should_capture_request(int(request_id))
        )

    def start_step(self, phase: Optional[str] = None) -> None:
        phase_name = phase or "decode"
        next_step = self._current_step_id + 1
        if self.config is not None:
            enabled = self.config.schedule.should_capture_step(next_step, phase_name)
            self._capture_enabled = bool(self._request_capture_enabled and enabled)
        else:
            self._capture_enabled = True
        self._current_step_id = next_step

    def end_step(self, *, stream: Optional[torch.cuda.Stream] = None) -> None:
        if self._monitor is None:
            return
        if stream is None and torch.cuda.is_available():
            try:
                stream = torch.cuda.current_stream(device=self._device)
            except Exception:
                stream = None
        if stream is not None:
            self._monitor.on_step_end(stream)
            self._step_event_stream = stream
        else:
            self._monitor.on_step_end(None)
        self._last_ready_step_id = self._current_step_id

    def is_capture_enabled(self) -> bool:
        return bool(self._capture_enabled)

    # Legacy compatibility
    def resolve_all(self) -> None:
        slots = self._collect_and_consume(wait=True)
        if self._consumer is not None:
            self._consumer.flush_ready()

    def clear_completed_results(self) -> None:
        self._collect_and_consume(wait=False)

    # ------------------------------------------------------------------
    # Data access

    def collect_results(self, *, wait: bool = False) -> List[GraphSlotResult]:
        if self._monitor is None or not self._slot_mapping:
            return []
        if wait:
            self._monitor.wait_for_step()
        elif not self._monitor.is_step_ready():
            return []

        view = self._monitor.metadata_view()
        results: List[GraphSlotResult] = []
        ready_step = self._last_ready_step_id
        for slot_id, info in self._slot_mapping.items():
            if slot_id >= view.shape[0]:
                continue
            decoded = _decode_metadata_row(view[slot_id])
            if decoded is None:
                continue
            shape, stride, ndim, dtype_id, device_idx, data_ptr = decoded
            results.append(
                GraphSlotResult(
                    slot_id=slot_id,
                    name=info.module_name,
                    step_id=ready_step,
                    data_ptr=data_ptr,
                    shape=shape,
                    stride=stride,
                    ndim=ndim,
                    dtype_id=dtype_id,
                    device_index=device_idx,
                )
            )
        return results

    def consume_with(
        self,
        consumer: "GraphSlotConsumer",
        *,
        wait: bool = False,
    ) -> List[GraphSlotResult]:
        """Collect metadata for the latest step and feed it through a consumer."""

        slots = self.collect_results(wait=wait)
        if not slots:
            return []
        return consumer.consume_graph_slots(slots)

    def attach_consumer(self, consumer: "GraphSlotConsumer") -> None:
        self._consumer = consumer

    # ------------------------------------------------------------------

    def get_slot_mapping(self) -> Dict[int, SlotInfo]:
        return dict(self._slot_mapping)

    def _collect_and_consume(self, wait: bool) -> List[GraphSlotResult]:
        slots = self.collect_results(wait=wait)
        if not slots:
            return []
        if self._consumer is not None:
            slots = self._consumer.consume_graph_slots(slots)
        return slots

    def _compile_allowed_names(self, model: nn.Module) -> Optional[Set[str]]:
        if self.config is None:
            return None
        all_names = [name for name, _ in model.named_modules() if name]
        try:
            compiled = list(self.config.hooks.compile(all_names))
        except Exception:
            return None
        return set(compiled)

    @staticmethod
    def _compose_filter(
        base_filter: Optional[ModuleFilter],
        allowed_names: Optional[Set[str]],
    ) -> Optional[ModuleFilter]:
        if base_filter is None and allowed_names is None:
            return None

        def combined(name: str, module: nn.Module) -> bool:
            if allowed_names is not None and name not in allowed_names:
                return False
            if base_filter is not None and not base_filter(name, module):
                return False
            return True

        return combined


def _decode_metadata_row(row: torch.Tensor) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int, int]]:
    # row lives in pinned CPU memory; detach before converting to bytes
    data = row.detach().cpu().contiguous().numpy().tobytes()
    (
        data_ptr,
        s0, s1, s2, s3,
        st0, st1, st2, st3,
        ndim,
        dtype_id,
        device_idx,
        _pad,
    ) = _METADATA_STRUCT.unpack(data)

    if data_ptr == 0 or ndim <= 0:
        return None

    shape = tuple(int(v) for v in (s0, s1, s2, s3)[:ndim])
    stride = tuple(int(v) for v in (st0, st1, st2, st3)[:ndim])
    return (
        shape,
        stride,
        int(ndim),
        int(dtype_id),
        int(device_idx),
        int(data_ptr),
    )
