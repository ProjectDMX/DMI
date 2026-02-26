from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

from .config import MonitoringConfig
from .graph_monitor import GraphMonitor, ModuleFilter, SlotInfo, METADATA_BYTES

if TYPE_CHECKING:
    from .graph_consumer import GraphSlotConsumer
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")


class GraphDelegate(Protocol):
    """Protocol describing the minimal delegate interface."""

    def submit_and_resolve(
        self,
        step_id: int,
        metadata: torch.Tensor,
        slot_ids: Sequence[int],
        hook_names: Sequence[str],
        stream_handle: Optional[int],
    ) -> None:
        ...


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
        delegate: Optional[GraphDelegate] = None,
        graph_mode: str = "manual",
    ) -> None:
        self.config = config
        self._module_filter = module_filter
        self._graph_mode = graph_mode
        self._max_slots = max_slots
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.async_enabled = device.type == "cuda"
        self.cache_dtype: Optional[torch.dtype] = None  # API parity with MonitoringEngine
        self._monitor: Optional[GraphMonitor] = None
        self._slot_mapping: Dict[int, SlotInfo] = {}
        self._step_event_stream: Optional[torch.cuda.Stream] = None
        self._step_streams: Dict[int, Optional[torch.cuda.Stream]] = {}
        self._delegate: Optional[GraphDelegate] = delegate
        self._delegate_disabled = False
        self._metadata_snapshots: Dict[int, torch.Tensor] = {}

        self._request_capture_enabled = True
        self._capture_enabled = True
        self._current_step_id = 0
        self._prepare_ms = 0.0
        self._consumer: Optional["GraphSlotConsumer"] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None
        self._step_streams.clear()
        self._metadata_snapshots.clear()

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
            graph_mode=self._graph_mode,
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
        # Snapshot current metadata, then optionally zero the GPU buffer so
        # stale slots from manual CUDA Graph replay produce data_ptr=0.
        # Skip in compile mode: zero_() invalidates torch.compile's CUDA Graph
        # state, and compile mode re-executes all traced record() ops each step
        # so there are no stale slots.
        if stream is not None:
            self._monitor.on_step_end(self._current_step_id, stream)
            # if self._graph_mode == "manual":
            #     self._monitor.clear_gpu_buffer(stream)
            self._step_event_stream = stream
        else:
            self._monitor.on_step_end(self._current_step_id, None)
            # if self._graph_mode == "manual":
            #     self._monitor.clear_gpu_buffer()
        if self._delegate is not None and not self._delegate_disabled:
            self._step_streams[self._current_step_id] = stream

    def is_capture_enabled(self) -> bool:
        return bool(self._capture_enabled)

    # Legacy compatibility
    def drain_ready_results(self, *, wait: bool = False) -> bool:
        """Collect and submit any ready slots. Returns True if progress was made."""

        # Fast path: skip Python metadata decode when delegate handles everything
        if (
            self._delegate is not None
            and not self._delegate_disabled
            and (self._consumer is None or self._consumer._delay_steps == 0)
        ):
            return self._fast_drain_to_delegate(wait=wait)

        slots = self._collect_and_consume(wait=wait)
        if not slots:
            return False
        self._submit_to_delegate(slots)
        return True

    def _fast_drain_to_delegate(self, *, wait: bool) -> bool:
        """Bypass Python metadata decode — pass raw buffer directly to C++.

        Saves ~2ms/step by eliminating redundant _decode_metadata_row() calls;
        C++ parse_shadow_block() reads the same raw ShadowSlotRow bytes and
        performs identical validity filtering.
        """
        if self._monitor is None or not self._slot_mapping:
            return False

        ready = self._monitor.pop_ready_step(wait=wait)
        if ready is None:
            return False

        ready_step, buffer = ready
        view = buffer.view(-1, METADATA_BYTES)

        slot_ids = list(self._slot_mapping.keys())
        hook_names = [self._slot_mapping[sid].module_name for sid in slot_ids]

        stream = self._step_streams.pop(ready_step, None)
        if stream is None:
            stream = self._step_event_stream
        stream_handle = _stream_to_handle(stream)

        try:
            self._delegate.submit_and_resolve(
                ready_step, view, slot_ids, hook_names, stream_handle,
            )
        except Exception:
            self._delegate_disabled = True
            self._step_streams.clear()
            self._metadata_snapshots.clear()
            raise

        return True

    def resolve_all(self) -> None:
        slots = self._collect_and_consume(wait=True)
        if self._consumer is not None:
            flushed = self._consumer.flush_ready()
            if flushed:
                self._submit_to_delegate(flushed)
        if slots:
            self._submit_to_delegate(slots)

    def clear_completed_results(self) -> None:
        self.drain_ready_results(wait=False)

    # ------------------------------------------------------------------
    # Data access

    def collect_results(self, *, wait: bool = False) -> List[GraphSlotResult]:
        if self._monitor is None or not self._slot_mapping:
            return []

        ready = self._monitor.pop_ready_step(wait=wait)
        if ready is None:
            return []

        ready_step, buffer = ready
        view = buffer.view(-1, METADATA_BYTES)
        results: List[GraphSlotResult] = []
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
        if self._delegate is not None and not self._delegate_disabled and results:
            self._metadata_snapshots[ready_step] = view
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

    def attach_backend_delegate(self, delegate: Optional[GraphDelegate]) -> None:
        """Attach or replace the backend delegate for graph capture."""

        self._delegate = delegate
        self._delegate_disabled = False
        if delegate is None:
            self._step_streams.clear()
            self._metadata_snapshots.clear()

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

    # ------------------------------------------------------------------
    # Delegate + alias helpers

    def _submit_to_delegate(self, slots: Sequence[GraphSlotResult]) -> None:
        if not slots or self._delegate is None or self._delegate_disabled:
            return

        by_step: Dict[int, List[GraphSlotResult]] = {}
        for slot in slots:
            by_step.setdefault(slot.step_id, []).append(slot)

        for step_id, step_slots in by_step.items():
            stream = self._step_streams.pop(step_id, None)
            if stream is None:
                stream = self._step_event_stream
            metadata = self._metadata_snapshots.pop(step_id, None)
            if metadata is None:
                continue
            stream_handle = _stream_to_handle(stream)
            try:
                slot_ids = [slot.slot_id for slot in step_slots]
                hook_names = [slot.name for slot in step_slots]
                if not slot_ids:
                    continue
                self._delegate.submit_and_resolve(
                    step_id,
                    metadata,
                    slot_ids,
                    hook_names,
                    stream_handle,
                )
            except Exception:
                self._delegate_disabled = True
                self._step_streams.clear()
                self._metadata_snapshots.clear()
                raise


def _stream_to_handle(stream: Optional[torch.cuda.Stream]) -> Optional[int]:
    if stream is None:
        return None
    try:
        return int(stream.cuda_stream)
    except AttributeError:
        return None


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
