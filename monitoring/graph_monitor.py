from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .graph_ops import load_graph_monitor_ops

METADATA_BYTES = 128
_METADATA_STRUCT = struct.Struct("<Qqqqqqqqqiii44s")

ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class SlotInfo:
    slot_id: int
    module_name: str


@dataclass
class _StepSnapshot:
    step_id: int
    event: torch.cuda.Event
    buffer: torch.Tensor


class GraphMonitor:
    """Graph-safe monitor that records tensor metadata during CUDA Graph capture."""

    def __init__(
        self,
        model: nn.Module,
        *,
        max_slots: int = 2048,
        module_filter: Optional[ModuleFilter] = None,
        device: Optional[torch.device] = None,
        graph_mode: str = "manual",
    ) -> None:
        if graph_mode not in ("manual", "compile", "dual_compile"):
            raise ValueError(f"graph_mode must be 'manual', 'compile', or 'dual_compile', got {graph_mode!r}")
        if device is None:
            device = torch.device("cuda")
        if device.type != "cuda":
            raise ValueError("GraphMonitor currently supports CUDA devices only.")
        self._device = device
        self._graph_mode = graph_mode
        self._max_slots = max_slots
        self._ops = load_graph_monitor_ops()
        # dual_compile: 2x buffer for ping-pong frames
        buf_multiplier = 2 if graph_mode == "dual_compile" else 1
        self._gpu_buffer = torch.empty(
            buf_multiplier * max_slots * METADATA_BYTES, dtype=torch.uint8, device=device
        )
        self._host_template = torch.empty(
            buf_multiplier * max_slots * METADATA_BYTES, dtype=torch.uint8, pin_memory=True
        )
        self._latest_snapshot = self._allocate_snapshot()
        self._module_filter = module_filter
        self._slot_mapping: Dict[int, SlotInfo] = {}
        self._module_to_slot: Dict[nn.Module, int] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._capture_anchors: List[torch.Tensor] = []
        self._inline_attrs: Dict[nn.Module, List[str]] = {}
        self._pending_steps: Deque[_StepSnapshot] = deque()
        self._num_monitored_slots = 0
        self._frame_parents: List[nn.Module] = []
        self._register_hooks(model)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for parent, attrs in self._inline_attrs.items():
            for attr in attrs:
                if hasattr(parent, attr):
                    delattr(parent, attr)
        self._inline_attrs.clear()
        for parent in self._frame_parents:
            if hasattr(parent, "_mon_frame_offset"):
                delattr(parent, "_mon_frame_offset")
        self._frame_parents.clear()
        self._pending_steps.clear()

    def _register_hooks(self, model: nn.Module) -> None:
        parent_map: Dict[str, nn.Module] = {"": model}
        for name, mod in model.named_modules():
            if name:
                parent_map[name] = mod

        slot_id = 0
        seen_parents: Dict[nn.Module, bool] = {}
        for name, module in model.named_modules():
            if name == "":
                continue
            if module in self._module_to_slot:
                continue
            if self._module_filter is not None and not self._module_filter(name, module):
                continue
            if slot_id >= self._max_slots:
                break
            self._module_to_slot[module] = slot_id
            self._slot_mapping[slot_id] = SlotInfo(slot_id=slot_id, module_name=name)
            # dual_compile: no forward hooks (they receive FakeTensors under torch.compile)
            if self._graph_mode != "dual_compile":
                handle = module.register_forward_hook(self._make_hook(slot_id))
                self._handles.append(handle)
            # In compile/dual_compile mode, set inline attrs so _mon_record() works
            if self._graph_mode in ("compile", "dual_compile") and hasattr(module, "monitor_activation"):
                parts = name.rsplit(".", 1)
                parent_name = parts[0] if len(parts) > 1 else ""
                attr_name = parts[-1]
                parent = parent_map.get(parent_name)
                if parent is not None:
                    slot_attr = f"_mon_slot_{attr_name}"
                    parent._mon_buf = self._gpu_buffer
                    setattr(parent, slot_attr, slot_id)
                    if parent not in self._inline_attrs:
                        self._inline_attrs[parent] = ["_mon_buf"]
                    if slot_attr not in self._inline_attrs[parent]:
                        self._inline_attrs[parent].append(slot_attr)
                    # dual_compile: set _mon_frame_offset on parents
                    if self._graph_mode == "dual_compile" and parent not in seen_parents:
                        parent._mon_frame_offset = 0
                        self._frame_parents.append(parent)
                        seen_parents[parent] = True
            slot_id += 1
        self._num_monitored_slots = slot_id

    def _make_hook(self, slot_id: int) -> Callable[..., None]:
        if self._graph_mode in ("compile", "dual_compile"):
            return self._make_compile_hook(slot_id)
        return self._make_manual_hook(slot_id)

    def _make_manual_hook(self, slot_id: int) -> Callable[..., None]:
        """Hook for manual CUDA Graph capture mode."""

        def hook(module: nn.Module, inputs, output) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None or not tensor.is_cuda:
                return
            self._ops.record(tensor, self._gpu_buffer, slot_id)
            if torch.cuda.is_current_stream_capturing():
                self._capture_anchors.append(tensor)

        return hook

    def _make_compile_hook(self, slot_id: int) -> Callable[..., None]:
        """Hook for torch.compile(mode='reduce-overhead') mode.

        Calls record() and sink() as custom ops that Dynamo traces into the
        compiled graph.  No capture detection or anchor collection needed.
        """
        ops = self._ops
        gpu_buffer = self._gpu_buffer

        def hook(module: nn.Module, inputs, output) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None or not tensor.is_cuda:
                return
            ops.record(tensor, gpu_buffer, slot_id)
            ops.sink([tensor])

        return hook

    @staticmethod
    def _extract_tensor(obj) -> Optional[torch.Tensor]:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                tensor = GraphMonitor._extract_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(obj, dict):
            for value in obj.values():
                tensor = GraphMonitor._extract_tensor(value)
                if tensor is not None:
                    return tensor
            return None
        return None

    def _allocate_snapshot(self) -> torch.Tensor:
        return torch.empty(
            self._host_template.size(),
            dtype=self._host_template.dtype,
            pin_memory=True,
        )

    def clear_gpu_buffer(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Zero shadow buffer so unfired hooks produce data_ptr=0 (skipped by parser).

        Must run on the same stream as the preceding on_step_end() to ensure
        the D2H snapshot copy completes before the buffer is cleared.
        """
        if stream is not None:
            with torch.cuda.stream(stream):
                self._gpu_buffer.zero_()
        else:
            self._gpu_buffer.zero_()

    def finalize_capture(self) -> None:
        if self._graph_mode in ("compile", "dual_compile"):
            return  # compile modes: record() is inline / per-hook inside compiled graph
        if self._capture_anchors:
            self._ops.sink(self._capture_anchors)
            self._capture_anchors.clear()
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        """Copy latest metadata from GPU buffer into snapshot template."""
        self._latest_snapshot.copy_(self._gpu_buffer, non_blocking=True)

    def on_step_end(self, step_id: int, stream: Optional[torch.cuda.Stream] = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream(device=self._device)
        snapshot = self._allocate_snapshot()
        with torch.cuda.stream(stream):
            snapshot.copy_(self._gpu_buffer, non_blocking=True)
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(stream)
        self._pending_steps.append(_StepSnapshot(step_id=step_id, event=event, buffer=snapshot))
        self._latest_snapshot = snapshot

    def is_step_ready(self) -> bool:
        if not self._pending_steps:
            return False
        return self._pending_steps[0].event.query()

    def wait_for_step(self) -> None:
        if not self._pending_steps:
            return
        self._pending_steps[0].event.synchronize()

    def pop_ready_step(self, *, wait: bool = False) -> Optional[Tuple[int, torch.Tensor]]:
        if not self._pending_steps:
            return None
        entry = self._pending_steps[0]
        if wait:
            entry.event.synchronize()
        elif not entry.event.query():
            return None
        self._pending_steps.popleft()
        return entry.step_id, entry.buffer

    def get_slot_mapping(self) -> Dict[int, SlotInfo]:
        return dict(self._slot_mapping)

    def metadata_buffer(self) -> torch.Tensor:
        return self._latest_snapshot

    def metadata_view(self) -> torch.Tensor:
        return self._latest_snapshot.view(-1, METADATA_BYTES)

    def num_slots(self) -> int:
        return len(self._slot_mapping)

    # ------------------------------------------------------------------
    # dual_compile helpers

    @property
    def num_monitored_slots(self) -> int:
        return self._num_monitored_slots

    def set_frame(self, frame: int) -> None:
        """Set frame offset on all parent modules. Dynamo guards on the int value."""
        assert self._graph_mode == "dual_compile"
        offset = frame * self._num_monitored_slots
        for parent in self._frame_parents:
            parent._mon_frame_offset = offset

    def disable_record(self) -> None:
        """Clear _mon_buf on all parent modules so ops.record() is skipped.

        Dynamo re-traces to produce CUDA Graphs without record kernels.
        _mon_frame_offset is preserved so Dynamo still creates separate
        graphs per frame (address isolation for ping-pong D2H).
        """
        assert self._graph_mode == "dual_compile"
        for parent, attrs in self._inline_attrs.items():
            if hasattr(parent, "_mon_buf"):
                parent._mon_buf = None

    def enable_record(self) -> None:
        """Restore _mon_buf on all parent modules (re-enables ops.record())."""
        assert self._graph_mode == "dual_compile"
        for parent, attrs in self._inline_attrs.items():
            parent._mon_buf = self._gpu_buffer

    def parse_frame_metadata(self, frame: int) -> Dict[int, dict]:
        """Parse metadata for one frame from GPU buffer (synchronous)."""
        torch.cuda.synchronize()
        host = self._gpu_buffer.cpu()
        offset = frame * self._num_monitored_slots
        result: Dict[int, dict] = {}
        for slot_id in self._slot_mapping:
            global_slot = slot_id + offset
            row = _parse_metadata_row(host, global_slot)
            if row is not None:
                result[slot_id] = row
        return result

    def create_frame_aliases(self, frame: int) -> Dict[int, torch.Tensor]:
        """Create from_blob alias tensors for one frame (one-time at warmup)."""
        parsed = self.parse_frame_metadata(frame)
        aliases: Dict[int, torch.Tensor] = {}
        for slot_id, meta in parsed.items():
            aliases[slot_id] = self._ops.alias_tensor(
                meta["data_ptr"],
                list(meta["shape"]),
                list(meta["stride"]),
                meta["dtype_id"],
                meta["device_idx"],
            )
        return aliases


def _parse_metadata_row(host: torch.Tensor, slot_id: int) -> Optional[dict]:
    """Parse a single 128B metadata row from a CPU buffer."""
    start = slot_id * METADATA_BYTES
    end = start + METADATA_BYTES
    if end > host.numel():
        return None
    slot_bytes = host[start:end].contiguous().numpy().tobytes()
    fields = _METADATA_STRUCT.unpack(slot_bytes)
    data_ptr = fields[0]
    ndim = fields[9]
    if data_ptr == 0 or ndim <= 0:
        return None
    return {
        "data_ptr": data_ptr,
        "shape": list(fields[1:5][:ndim]),
        "stride": list(fields[5:9][:ndim]),
        "ndim": ndim,
        "dtype_id": fields[10],
        "device_idx": fields[11],
    }
