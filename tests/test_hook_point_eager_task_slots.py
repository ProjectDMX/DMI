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

The fake below keeps a legacy ring's head/tail counters. ``HookPoint.forward``
only takes the eager branch for a CUDA tensor, so a CPU tensor subclass poses
as one (as in tests/test_record_runtime.py) and ``dispatch_producer`` is
monkeypatched; the transport is a real ``RingTransport``.
"""
from __future__ import annotations

import pytest
import torch

from dmi.hooks.specs import align_up_py

pytestmark = pytest.mark.cpu


class _FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True


class _LegacyRingEngine:
    """Payload and task heads/tails of a legacy ring; flush drains both."""

    def __init__(self, task_cap: int, payload_cap: int):
        self.task_cap = task_cap
        self.capacity = payload_cap
        self.task_head = self.task_tail = 0
        self.payload_head = self.payload_tail = 0
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
        return self.task_cap - (self.task_head - self.task_tail)

    def reserve_one(self, nbytes: int) -> None:
        self.events.append("reserve")
        self.payload_head += align_up_py(nbytes, 16)
        self.task_head += 1
        self.max_outstanding_tasks = max(
            self.max_outstanding_tasks, self.task_head - self.task_tail)

    def flush_and_wait(self) -> None:
        self.events.append("flush")
        self.task_tail = self.task_head
        self.payload_tail = self.payload_head


def _eager_hook(monkeypatch, engine, dispatched):
    from dmi.hooks.point import HookPoint
    from dmi.transport import ring as ring_transport
    from dmi.transport.ring import RingTransport

    transport = RingTransport(engine)
    transport.force_eager = True
    monkeypatch.setattr(ring_transport, "_active_transport", transport)
    monkeypatch.setattr(
        "dmi.hooks.point.dispatch_producer",
        lambda *args: dispatched.append(args),
    )
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
