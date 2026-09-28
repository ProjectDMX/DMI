"""The eager safety net on a real legacy ring, through HookPoint and pybind.

tests/test_hook_point_eager_task_slots.py runs ``HookPoint.forward``'s
check-and-reserve against a fake engine, and
test_eager_safety_net_delivers_past_the_task_ring in
tests/native/ring/test_ring_engine.cu runs a C++ copy of it against the
native ring. Neither goes through point.py and the pybind surface together
(``available_task_slots``, ``reserve_one``, ``flush_and_wait`` and
``prepare_step``'s ``reserve``), so binding ``available_task_slots`` to the
wrong method passed every committed test while the model forward raised.
These drive the real composition: a ``MonitoringEngine``'s legacy ring, its
``RingTransport``, and real ``HookPoint`` modules on CUDA tensors.

Needs CUDA and the full native backend. The ring has no host engine, so its
consumer drops what it drains; delivering each payload byte for byte with
its own metadata is the native test's job.
"""
from __future__ import annotations

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

RING_BYTES = 64 * 1024


def _legacy_engine(task_entries):
    from dmi.api.v1 import RingConfig
    from dmi.engine import MonitoringEngine

    config = RingConfig()
    config.task_ring_entries = task_entries
    config.payload_ring_bytes = RING_BYTES
    config.pinned_staging_bytes = RING_BYTES
    engine = MonitoringEngine(model_id="eager-safety-net", ring_config=config)
    assert not engine._record_mode
    return engine


def _arm(hook, transport, hook_type, layer_no):
    """What install_ring_hooks does to a HookPoint."""
    hook._ring_hook_type = hook_type
    hook._ring_hook_id = layer_no
    hook._ring_payload = transport._ring_payload
    return hook


@pytest.mark.parametrize("task_entries", [1, 2, 4])
def test_an_oversized_step_fires_every_hook_through_a_small_task_ring(
        task_entries):
    """Nine hooks through 1, 2 and 4 task entries: the step is OVERSIZED on
    hook count alone, and the safety net must flush each time the entries
    run out rather than reserve past them."""
    from dmi.adapters.base import StepReservation
    from dmi.hooks.point import HookPoint
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, align_up_py

    hooks = 9
    engine = _legacy_engine(task_entries)
    transport = engine._ring_transport
    ring = engine._ring_engine
    try:
        assert ring.task_cap() == task_entries
        # Row counts differ per hook and 24 is not a multiple of 16, as in
        # the native delivery test.
        tensors = [
            torch.full((1 + layer % 3, 24), layer, dtype=torch.uint8,
                       device="cuda")
            for layer in range(hooks)
        ]
        ring.push_all_metas(
            [HOOK_TYPE_RESID_PRE] * hooks, list(range(hooks)),
            [list(t.shape) for t in tensors], [torch.uint8] * hooks,
            [0] * hooks, "eager-safety-net", 0, 0, 0, 0, False,
            ["0:0"], [(0, 1)], [0], [0])
        step_bytes = sum(align_up_py(t.nbytes, 16) for t in tensors)
        assert (ring.prepare_step(step_bytes, hooks)
                == StepReservation.OVERSIZED)

        transport.force_eager = True
        for layer, tensor in enumerate(tensors):
            hook = _arm(HookPoint(), transport, HOOK_TYPE_RESID_PRE, layer)
            assert hook(tensor) is tensor
            assert 0 <= ring.available_task_slots() <= task_entries
            assert ring.available_capacity() <= ring.payload_cap()

        ring.flush_and_wait()
        assert ring.available_task_slots() == task_entries
        assert ring.available_capacity() == ring.payload_cap()
    finally:
        transport.force_eager = False
        engine.close()


def test_needs_eager_steps_that_fit_leave_no_reservation_behind():
    """_spec_needs_eager turns the safety net on for steps that fit, and
    each hook then reserves its own entry. commit_step used to reserve the
    whole step as well: each step kept its hook count of task entries after
    the flush, and the second step's forward raised from reserve_one."""
    from types import SimpleNamespace

    from dmi.adapters.base import BackendAdapter, StepReservation
    from dmi.adapters.types import StepContext
    from dmi.hooks.point import HookPoint
    from dmi.hooks.specs import (
        HOOK_TYPE_RESID_PRE, HookSpec, ModelShapeConfig)

    class DynamicShapeAdapter(BackendAdapter):
        def detect_model_shape(self, model):
            return self._cfg

        def detect_parallel_ranks(self):
            return (0, 0, 0, 0)

        def is_pp_first(self):
            return True

        def is_pp_last(self):
            return True

        def build_step_context(self, *raw):
            return None

        def on_capacity_exceeded(self, ctx):
            pass

        def _spec_needs_eager(self, spec):
            return True

    task_entries = 4
    engine = _legacy_engine(task_entries)
    transport = engine._ring_transport
    ring = engine._ring_engine
    try:
        adapter = DynamicShapeAdapter(engine, "eager-safety-net")
        adapter._cfg = ModelShapeConfig(
            hidden_dim=16, num_heads=4, num_kv_heads=4, head_dim=4,
            dtype=torch.float16, vocab_size=32, intermediate_dim=0)
        hooks = [HookPoint(), HookPoint()]
        model = SimpleNamespace(get_hook_specs=lambda: [
            HookSpec(HOOK_TYPE_RESID_PRE, hook, layer)
            for layer, hook in enumerate(hooks)])
        adapter.attach_model(model)
        assert [spec.module for spec in adapter.active_specs] == hooks
        ctx = StepContext(
            model_id="eager-safety-net", flattened=False, req_ids=["0:0"],
            token_ranges=[(0, 4)], dim0_offsets=[0], kv_offsets=[0],
            batch=1, q_len=4, kv_dim=4)
        value = torch.ones(1, 4, 16, dtype=torch.float16, device="cuda")

        for step in range(3):
            assert adapter.commit_step(ctx) is StepReservation.RESERVED
            assert transport.force_eager
            for hook in hooks:
                hook(value)
            ring.flush_and_wait()
            assert ring.available_task_slots() == task_entries, step
            assert ring.available_capacity() == ring.payload_cap(), step
    finally:
        transport.force_eager = False
        engine.close()
