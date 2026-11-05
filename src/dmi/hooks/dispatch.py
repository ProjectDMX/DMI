"""Hook-to-native-producer dispatch and hook installation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from ..hook_rows import clear_hook_row_features
from .specs import HEAD_TYPES, HookSpec


def dispatch_producer(
    ring_payload: torch.Tensor,
    tensor: torch.Tensor,
    strip_tensor: Optional[torch.Tensor],
    strip_row_bytes: int,
    hook_type: int,
    hook_id: int,
    gather_table: Optional[torch.Tensor] = None,
) -> None:
    """Dispatch a tensor to the producer matching its configured mode.

    A per-HookPoint row table (gather mode) takes precedence over strip
    mode: the table subsumes the prefix strip (padding rows are never
    listed), and ``strip_row_bytes`` doubles as the gather row stride.
    """

    if gather_table is not None:
        torch.ops.ring.producer_gather(
            ring_payload, tensor, gather_table, strip_row_bytes, hook_type, hook_id
        )
    elif strip_tensor is None:
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
    """Bind hook specifications to their executable hook-point modules.

    Installing starts a new attachment: any gather table or in-graph heads a
    previous attachment configured on these HookPoints are cleared (and the
    strip stride gather replaced is restored), so a plain re-attach runs the
    HookPoint's ordinary producer.  Integrations re-arm gather/heads after
    installing (``dmi.hook_rows.configure_hook_gather`` /
    ``configure_hook_heads``).

    An in-graph head spec (``HEAD_TYPES``) names the HookPoint whose rows it
    scores but is not that HookPoint's row producer: it binds the payload
    only and never replaces the row key, so a ``[row, head]`` spec pair keeps
    the row producer and a lone head spec leaves a head-only HookPoint.
    """

    for spec in specs:
        if spec.module is None:
            raise RuntimeError(
                "install_ring_hooks received an unbound model-wide HookSpec"
            )
    for spec in specs:
        # A new attachment: drop the previous row key and gather/head state,
        # so a HookPoint that now receives only a head spec is head-only.
        spec.module._ring_hook_type = None
        clear_hook_row_features(spec.module)
    for spec in specs:
        hook_point = spec.module
        hook_point._ring_payload = ring_payload
        if spec.hook_type in HEAD_TYPES:
            continue
        hook_point._ring_hook_type = spec.hook_type
        hook_point._ring_hook_id = spec.layer_no


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

    Gather and in-graph head state (row tables, head arena and accumulator)
    belongs to the attachment and is cleared here too, releasing those tensor
    references and restoring the strip stride gather replaced.
    """
    for spec in specs:
        hook_point = spec.module
        if hook_point is None:
            continue
        hook_point._ring_hook_type = None
        hook_point._ring_payload = None
        clear_hook_row_features(hook_point)


__all__ = ["dispatch_producer", "install_ring_hooks", "uninstall_ring_hooks"]
