from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .graph_ops import load_graph_monitor_ops

METADATA_BYTES = 128

ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class SlotInfo:
    slot_id: int
    module_name: str


class GraphMonitor:
    """Graph-safe monitor that records tensor metadata during CUDA Graph capture."""

    def __init__(
        self,
        model: nn.Module,
        *,
        max_slots: int = 2048,
        module_filter: Optional[ModuleFilter] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        if device.type != "cuda":
            raise ValueError("GraphMonitor currently supports CUDA devices only.")
        self._device = device
        self._max_slots = max_slots
        self._ops = load_graph_monitor_ops()
        self._gpu_buffer = torch.empty(
            max_slots * METADATA_BYTES, dtype=torch.uint8, device=device
        )
        self._host_buffer = torch.empty(
            max_slots * METADATA_BYTES, dtype=torch.uint8, pin_memory=True
        )
        self._module_filter = module_filter
        self._slot_mapping: Dict[int, SlotInfo] = {}
        self._module_to_slot: Dict[nn.Module, int] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._capture_anchors: List[torch.Tensor] = []
        self._step_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self._register_hooks(model)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_hooks(self, model: nn.Module) -> None:
        slot_id = 0
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
            handle = module.register_forward_hook(self._make_hook(slot_id))
            self._handles.append(handle)
            slot_id += 1

    def _make_hook(self, slot_id: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs, output) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None or not tensor.is_cuda:
                return
            self._ops.record(tensor, self._gpu_buffer, slot_id)
            if torch.cuda.is_current_stream_capturing():
                self._capture_anchors.append(tensor)

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

    def finalize_capture(self) -> None:
        if self._capture_anchors:
            self._ops.sink(self._capture_anchors)
            self._capture_anchors.clear()
        self._host_buffer.copy_(self._gpu_buffer, non_blocking=True)

    def on_step_end(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        stream = stream or torch.cuda.current_stream()
        self._step_event.record(stream)

    def is_step_ready(self) -> bool:
        return self._step_event.query()

    def wait_for_step(self) -> None:
        self._step_event.synchronize()

    def get_slot_mapping(self) -> Dict[int, SlotInfo]:
        return dict(self._slot_mapping)

    def metadata_buffer(self) -> torch.Tensor:
        return self._host_buffer

    def metadata_view(self) -> torch.Tensor:
        return self._host_buffer.view(-1, METADATA_BYTES)

    def num_slots(self) -> int:
        return len(self._slot_mapping)
