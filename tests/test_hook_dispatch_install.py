"""Regression tests for ``install_ring_hooks`` attribute binding.

``dmi.hooks.dispatch.install_ring_hooks`` is the only writer of the three
``_ring_*`` attributes the native forward hooks read off a hook-point module.
It runs on the attach path (``dmi.adapters.base.BaseAdapter.attach_model``
calls it right after hook selection), but every caller in the tree sat behind
CUDA / a built native extension, so a total no-op body left the CPU gate
green.  The function itself needs neither: it only assigns three plain
attributes on ``spec.module``.  ``torch.ops.ring`` is reached from
``dispatch_producer``, a different function.  So a bare ``nn.Identity()``
standing in for a HookPoint exercises it end to end.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dmi.hooks.dispatch import install_ring_hooks
from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HookSpec

pytestmark = pytest.mark.cpu


def test_install_ring_hooks_binds_type_id_and_payload_to_each_hook_point() -> None:
    """Every spec must get its own hook type, layer id, and the ring payload.

    ``_ring_hook_id`` carries ``spec.layer_no`` verbatim, and that value is
    what per-layer reassembly groups on downstream: ``_reassemble_per_layer``
    in ``dmi/storage/internals.py`` buckets rows by ``key[3]``, the layer_no
    field.  An id that is stuck at one value (or never written at all) would
    therefore collapse every captured layer into a single bucket rather than
    fail loudly, so the per-module distinctness of the ids is asserted, not
    just their presence.
    """

    mods = [nn.Identity() for _ in range(3)]
    specs = [
        HookSpec(hook_type=HOOK_TYPE_RESID_PRE, module=m, layer_no=i)
        for i, m in enumerate(mods)
    ]
    payload = torch.zeros(4, dtype=torch.uint8)

    install_ring_hooks(specs, ring_payload=payload)

    assert [m._ring_hook_id for m in mods] == [0, 1, 2]
    assert [m._ring_hook_type for m in mods] == [HOOK_TYPE_RESID_PRE] * 3
    # Identity, not equality: the producer kernels mutate the ring payload in
    # place, so each hook point must hold the caller's tensor itself.
    for m in mods:
        assert m._ring_payload is payload


def test_install_ring_hooks_refuses_an_unbound_model_wide_spec() -> None:
    """A model-wide spec that never got a module bound is a config error.

    ``HookSpec.module`` is ``Optional``: model-wide specs (embed, final_ln)
    legitimately carry ``None`` until an adapter resolves them.  If such a
    spec reaches installation still unbound, the explicit guard must name the
    problem; without it the next line would raise a bare ``AttributeError``
    on ``None``.
    """

    spec = HookSpec(hook_type=HOOK_TYPE_RESID_PRE, module=None, layer_no=-1)

    with pytest.raises(RuntimeError, match="unbound model-wide HookSpec"):
        install_ring_hooks([spec])
