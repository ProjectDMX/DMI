"""The eager safety net of in-graph heads respects the ring's effective
capacity (min of payload ring and pinned staging): an entry the drain can
never move is refused before anything is reserved.  Fakes only."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu


class Engine:
    def __init__(self, available):
        self.available = available
        self.calls = []

    def available_capacity(self):
        return self.available

    def available_task_slots(self):
        return 8

    def flush_and_wait(self):
        self.calls.append("flush")

    def reserve_one(self, nbytes):
        self.calls.append(("reserve", nbytes))


def _hook(slots):
    from dmi.hooks.point import HookPoint

    hook = HookPoint()
    hook._head_hook_type, hook._head_hook_id, hook._head_slots = 24, 0, slots
    return hook


def test_head_output_above_the_effective_capacity_is_refused(monkeypatch):
    launches = []
    monkeypatch.setattr(torch.ops.ring, "producer_fused",
                        lambda *args: launches.append(args), raising=False)
    engine = Engine(available=1 << 20)
    transport = SimpleNamespace(effective_cap=256)
    x = torch.zeros(8, 4)                          # 8 rows x 16 slots x 4 B = 512 B
    with pytest.raises(RuntimeError, match="effective capacity"):
        _hook(16)._eager_head(transport, engine, None, x)
    assert engine.calls == [] and launches == []


def test_head_output_within_capacity_flushes_when_the_ring_is_full(monkeypatch):
    launches = []
    monkeypatch.setattr(torch.ops.ring, "producer_fused",
                        lambda *args: launches.append(args), raising=False)
    engine = Engine(available=64)
    transport = SimpleNamespace(effective_cap=1 << 20)
    _hook(4)._eager_head(transport, engine, None, torch.zeros(8, 4))
    assert engine.calls == ["flush", ("reserve", 128)] and len(launches) == 1
