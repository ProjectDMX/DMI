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


def uninstall_ring_hooks(specs: Sequence[HookSpec]) -> None:
    """Disarm hook points previously bound by :func:`install_ring_hooks`.

    ``HookPoint.forward`` gates the producer on ``_ring_hook_type is not
    None``, so clearing it is what actually stops the kernel launching; the
    payload is dropped with it so a detached model stops pinning the ring
    buffer. ``enabled`` is deliberately left alone -- it carries the hook
    SELECTION, which ``apply_hook_selection`` owns and a later re-attach
    reuses.

    Unlike ``install_ring_hooks`` this tolerates an unbound spec instead of
    raising: it runs from ``detach_model``'s teardown, often inside a
    ``finally``, where raising would displace whatever the caller was already
    handling.

    Cost, so it is not a surprise: ``_ring_hook_type`` is a plain int
    precisely so torch.compile bakes it as a compile-time constant, so
    flipping it to None invalidates a traced decode graph. Re-attaching costs
    a recompile. That is the price of not corrupting the ring on the next
    ordinary forward, and ``attach_model`` re-runs ``install_ring_hooks``
    anyway, so the pair stays symmetric.
    """
    for spec in specs:
        hook_point = spec.module
        if hook_point is None:
            continue
        hook_point._ring_hook_type = None
        hook_point._ring_payload = None


__all__ = ["dispatch_producer", "install_ring_hooks", "uninstall_ring_hooks"]
