"""``detach_model`` must disarm the hook points it armed.

``generate_with_monitoring`` and ``generate_greedy_with_monitoring`` both call
``detach_model`` from a ``finally``, so it runs after every monitored generate.
Only ``install_ring_hooks`` ever wrote ``_ring_hook_type`` / ``_ring_payload``
and nothing cleared them, so the hooks stayed armed on the model object: the
next ordinary forward -- an eval pass, a perplexity computation, another
library calling the same model -- launched producer kernels with no
``prepare_step`` reservation and an empty metadata FIFO. Measured on a real
model before the fix: 27 producer dispatches and 29,638 unreserved bytes
written after detach, while the host still believed only the previous
generate's bytes were outstanding.
"""
import pytest
import torch

from dmi.adapters.huggingface.adapter import HuggingFaceAdapter
from dmi.hooks.dispatch import install_ring_hooks
from dmi.hooks.specs import HOOK_TYPE_Q, HOOK_TYPE_RESID_PRE, HookSpec

pytestmark = pytest.mark.cpu


class _FakeTransport:
    null_offload = False
    force_eager = False
    _model_cfg = None

    def __init__(self):
        self._active_specs = []
        self._using_forward_hooks = True
        self._ring_payload = torch.zeros(8, dtype=torch.uint8)


class _FakeEngine:
    def __init__(self):
        self._ring_transport = _FakeTransport()
        self._ring_engine = None


def _attached_adapter():
    """An adapter with armed specs, as attach_model would leave it."""
    adapter = HuggingFaceAdapter(_FakeEngine(), "test-model")
    specs = [
        HookSpec(hook_type=HOOK_TYPE_RESID_PRE, module=torch.nn.Identity(), layer_no=0),
        HookSpec(hook_type=HOOK_TYPE_Q, module=torch.nn.Identity(), layer_no=1),
    ]
    install_ring_hooks(specs, adapter.transport._ring_payload)
    adapter.active_specs = specs
    adapter.transport._active_specs = specs
    return adapter, specs


def test_detach_model_disarms_every_hook_point_it_armed():
    adapter, specs = _attached_adapter()
    assert all(s.module._ring_hook_type is not None for s in specs)

    adapter.detach_model(torch.nn.Identity())

    # `_ring_hook_type is None` is the condition HookPoint.forward returns on,
    # so this is what actually stops the producer kernel launching.
    assert all(s.module._ring_hook_type is None for s in specs)
    assert all(s.module._ring_payload is None for s in specs)


def test_detach_model_still_clears_the_transport_bookkeeping():
    """The pre-existing teardown must survive the added disarm."""
    adapter, _ = _attached_adapter()

    adapter.detach_model(torch.nn.Identity())

    assert adapter.transport._active_specs == []
    assert adapter.transport._using_forward_hooks is False


def test_detach_model_restores_the_prepare_wrapper():
    """Unchanged behaviour, pinned so the added disarm cannot displace it."""
    adapter, _ = _attached_adapter()
    model = torch.nn.Identity()
    sentinel = object()
    model._monitoring_orig_prepare = sentinel

    adapter.detach_model(model)

    assert model.prepare_inputs_for_generation is sentinel
    assert model._monitoring_orig_prepare is None


def test_detach_model_is_idempotent():
    """It runs from a `finally`, so a second call after an error path must not
    raise on already-cleared specs."""
    adapter, specs = _attached_adapter()

    adapter.detach_model(torch.nn.Identity())
    adapter.detach_model(torch.nn.Identity())

    assert all(s.module._ring_hook_type is None for s in specs)
