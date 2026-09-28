"""The eager safety net must not reserve more tasks than the task ring holds.

A legacy step with more firing hooks than ``task_ring_entries`` gets
``STEP_OVERSIZED`` from ``prepare_step`` (after a flush, so the ring is
empty), and the adapter sets ``force_eager``. Each hook then took the safety
net in ``HookPoint.forward``, which checked only payload BYTES before
``reserve_one`` -- and ``reserve_one`` advanced the task head with no
task-capacity check. With plenty of payload room, hook ``task_cap`` reserved
sequence ``task_cap``, whose producer release-stores into slot 0 while hook
0's READY word there is still unread (producers never read the tails). The
drain then either pairs the wrong size with hook 0's TensorMeta or clears the
new word and waits at that sequence forever while ``flush_and_wait`` reports
success. Found by TLA+ model checking of the payload ring.

The fake below keeps a legacy ring's head/tail counters, and like the real
drain a flush releases only what producers published. ``HookPoint.forward``
only takes the eager branch for a CUDA tensor, so a CPU tensor subclass poses
as one (as in tests/test_record_runtime.py) and ``dispatch_producer`` is
monkeypatched to publish; the transport is a real ``RingTransport``. The real
native ring runs the same checks in tests/test_eager_safety_net_gpu.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmi.adapters.base import BackendAdapter, StepReservation
from dmi.adapters.types import StepContext
from dmi.hooks.specs import (
    HOOK_TYPE_RESID_PRE,
    HookSpec,
    ModelShapeConfig,
    align_up_py,
)

pytestmark = pytest.mark.cpu


class _FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True


class _LegacyRingEngine:
    """Payload and task heads/tails of a legacy ring.

    Reservations advance the heads; a flush advances the tails by what
    producers published since the last one, as the native drain does, so a
    reservation nothing publishes stays outstanding.
    """

    def __init__(self, task_cap: int, payload_cap: int):
        self.task_cap = task_cap
        self.capacity = payload_cap
        self.task_head = self.task_tail = 0
        self.payload_head = self.payload_tail = 0
        self.published_tasks = self.published_bytes = 0
        self.events: list[str] = []
        self.max_outstanding_tasks = 0

    def payload_tensor(self) -> torch.Tensor:
        return torch.empty(64, dtype=torch.uint8)

    def payload_cap(self) -> int:
        return self.capacity

    def staging_cap(self) -> int:
        return self.capacity

    def available_capacity(self) -> int:
        return self.capacity - (self.payload_head - self.payload_tail)

    def available_task_slots(self) -> int:
        return max(0, self.task_cap - (self.task_head - self.task_tail))

    def outstanding(self) -> tuple[int, int]:
        return (self.task_head - self.task_tail,
                self.payload_head - self.payload_tail)

    def _reserve(self, nbytes: int, tasks: int) -> None:
        self.payload_head += nbytes
        self.task_head += tasks
        self.max_outstanding_tasks = max(
            self.max_outstanding_tasks, self.task_head - self.task_tail)

    def reserve_one(self, nbytes: int) -> None:
        self.events.append("reserve")
        self._reserve(align_up_py(nbytes, 16), 1)

    def prepare_step(self, step_total_bytes: int, num_hooks: int,
                     reserve: bool = True) -> int:
        """RingEnginePy::prepare_step's decision, on the same counters."""
        if step_total_bytes > self.capacity or num_hooks > self.task_cap:
            self.flush_and_wait()
            return int(StepReservation.OVERSIZED)
        result = StepReservation.RESERVED
        if (step_total_bytes > self.available_capacity()
                or num_hooks > self.available_task_slots()):
            self.flush_and_wait()
            result = StepReservation.FLUSHED
        if reserve:
            self._reserve(step_total_bytes, num_hooks)
        return int(result)

    def push_all_metas(self, *args) -> None:
        pass

    def publish(self, nbytes: int) -> None:
        self.published_tasks += 1
        self.published_bytes += align_up_py(nbytes, 16)

    def flush_and_wait(self) -> None:
        self.events.append("flush")
        self.task_tail += self.published_tasks
        self.payload_tail += self.published_bytes
        self.published_tasks = self.published_bytes = 0


def _activate(monkeypatch, engine, dispatched):
    from dmi.transport import ring as ring_transport
    from dmi.transport.ring import RingTransport

    transport = RingTransport(engine)
    monkeypatch.setattr(ring_transport, "_active_transport", transport)

    def dispatch(payload, x, *args):
        dispatched.append((payload, x, *args))
        engine.publish(x.nbytes)

    monkeypatch.setattr("dmi.hooks.point.dispatch_producer", dispatch)
    return transport


def _eager_hook(monkeypatch, engine, dispatched):
    from dmi.hooks.point import HookPoint

    transport = _activate(monkeypatch, engine, dispatched)
    transport.force_eager = True
    hook = HookPoint()
    hook._ring_hook_type = 1
    hook._ring_hook_id = 2
    hook._ring_payload = transport._ring_payload
    return hook


def test_a_step_with_more_hooks_than_task_entries_flushes_before_overflowing(
        monkeypatch):
    task_cap = 4
    engine = _LegacyRingEngine(task_cap=task_cap, payload_cap=1 << 20)
    dispatched = []
    hook = _eager_hook(monkeypatch, engine, dispatched)

    value = torch.arange(32, dtype=torch.uint8).as_subclass(_FakeCudaTensor)
    for _ in range(task_cap + 1):
        hook(value)

    assert len(dispatched) == task_cap + 1, "every hook still takes the ring"
    assert engine.max_outstanding_tasks <= task_cap, (
        "a reservation past task_cap overwrites an unread task slot")
    assert engine.events == ["reserve"] * task_cap + ["flush", "reserve"]


class _DynamicShapeAdapter(BackendAdapter):
    """Every hook needs eager dispatch, as for a dynamic-shape backend."""

    def detect_model_shape(self, model):
        raise NotImplementedError

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


def test_a_needs_eager_step_that_fits_is_reserved_once(monkeypatch):
    """_spec_needs_eager turns force_eager on even when the step fits, and
    each hook then reserves its own entry in the safety net. Had
    commit_step reserved the whole step too, every such step would hold
    hook_count entries (and its bytes) that no producer publishes and no
    flush frees, until the task check made a forward raise."""
    from dmi.hooks.point import HookPoint

    engine = _LegacyRingEngine(task_cap=4, payload_cap=1 << 20)
    dispatched = []
    transport = _activate(monkeypatch, engine, dispatched)
    adapter = _DynamicShapeAdapter(
        SimpleNamespace(_ring_transport=transport, _ring_engine=engine),
        "m")
    cfg = ModelShapeConfig(hidden_dim=16, num_heads=4, num_kv_heads=4,
                           head_dim=4, dtype=torch.float16, vocab_size=32,
                           intermediate_dim=0)
    adapter.model_cfg = cfg
    transport.set_model_cfg(cfg)
    hooks = [HookPoint(), HookPoint()]
    specs = [HookSpec(HOOK_TYPE_RESID_PRE, hook, layer)
             for layer, hook in enumerate(hooks)]
    for spec in specs:
        spec.module._ring_hook_type = spec.hook_type
        spec.module._ring_hook_id = spec.layer_no
        spec.module._ring_payload = transport._ring_payload
    adapter.active_specs = transport._active_specs = specs
    ctx = StepContext(model_id="m", flattened=False, req_ids=["0:0"],
                      token_ranges=[(0, 4)], dim0_offsets=[0],
                      kv_offsets=[0], batch=1, q_len=4, kv_dim=4)
    value = torch.zeros(1, 4, 16, dtype=torch.float16).as_subclass(
        _FakeCudaTensor)

    for step in range(3):
        assert adapter.commit_step(ctx) is StepReservation.RESERVED
        assert transport.force_eager
        for hook in hooks:
            hook(value)
        engine.flush_and_wait()
        assert engine.outstanding() == (0, 0), f"step {step} leaked"
    assert len(dispatched) == 3 * len(hooks)
