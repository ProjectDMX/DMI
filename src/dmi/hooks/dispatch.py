"""Hook-to-native-producer dispatch and hook installation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from .specs import HookSpec


def dispatch_producer(
    ring_payload: torch.Tensor,
    tensor: torch.Tensor,
    strip_tensor: Optional[torch.Tensor],
    strip_row_bytes: int,
    hook_type: int,
    hook_id: int,
) -> None:
    """Dispatch a tensor to the producer matching its configured strip mode."""

    if strip_tensor is None:
        torch.ops.ring.producer(ring_payload, tensor, hook_type, hook_id)
    elif strip_row_bytes > 0:
        torch.ops.ring.producer_prefix(
            ring_payload, tensor, strip_tensor, strip_row_bytes, hook_type, hook_id
        )
    else:
        torch.ops.ring.producer_chunked(
            ring_payload, tensor, strip_tensor, hook_type, hook_id
        )


def install_ring_hooks(
    specs: Sequence[HookSpec],
    ring_payload: Optional[torch.Tensor] = None,
) -> None:
    """Bind hook specifications to their executable hook-point modules."""

    for spec in specs:
        hook_point = spec.module
        if hook_point is None:
            raise RuntimeError(
                "install_ring_hooks received an unbound model-wide HookSpec"
            )
        hook_point._ring_hook_type = spec.hook_type
        hook_point._ring_hook_id = spec.layer_no
        hook_point._ring_payload = ring_payload


__all__ = ["dispatch_producer", "install_ring_hooks"]
