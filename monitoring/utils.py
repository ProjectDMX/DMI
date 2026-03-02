"""Utility helpers for monitoring integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import torch

SliceInput = Union["Slice", int, slice, Sequence[int], torch.Tensor, None]


@dataclass
class Slice:
    """Minimal slice helper compatible with hook caching."""

    mode: str = "identity"
    slice: Optional[Any] = None

    def __init__(self, slice_obj: SliceInput = None) -> None:
        if isinstance(slice_obj, Slice):
            self.mode = slice_obj.mode
            self.slice = slice_obj.slice
            return
        if slice_obj is None:
            self.mode = "identity"
            self.slice = None
        elif isinstance(slice_obj, int):
            self.mode = "int"
            self.slice = int(slice_obj)
        elif isinstance(slice_obj, slice):
            self.mode = "slice"
            self.slice = slice_obj
        elif isinstance(slice_obj, (list, tuple, torch.Tensor)):
            self.mode = "array"
            self.slice = slice_obj
        else:
            self.mode = "identity"
            self.slice = None

    @staticmethod
    def unwrap(slice_obj: SliceInput) -> "Slice":
        """Normalize a slice input into a Slice instance."""
        if isinstance(slice_obj, Slice):
            return slice_obj
        return Slice(slice_obj)

    def apply(self, tensor: torch.Tensor, *, dim: int = -2) -> torch.Tensor:
        """Apply the slice along a given dimension."""
        if self.mode == "identity":
            return tensor
        if self.mode == "int":
            return tensor.select(dim, int(self.slice))
        if self.mode == "slice":
            idx = [slice(None)] * tensor.dim()
            idx[dim] = self.slice if isinstance(self.slice, slice) else slice(None)
            return tensor[tuple(idx)]
        if self.mode == "array":
            data = self.slice
            if isinstance(data, torch.Tensor):
                indices = data.to(device=tensor.device, dtype=torch.long)
            else:
                indices = torch.tensor(list(data), device=tensor.device, dtype=torch.long)
            return torch.index_select(tensor, dim, indices)
        return tensor
