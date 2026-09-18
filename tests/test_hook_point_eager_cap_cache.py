"""The eager safety net must not query ring capacities per forward.

``HookPoint.forward`` recomputed ``min(engine.payload_cap(),
engine.staging_cap())`` on every eager invocation -- two pybind crossings per
hook forward on a path the native header documents as "not called per-step"
(native/csrc/ring/ring_engine_py.h), and it did so even on the cpu-direct
strip branch that never uses the value. The caps are fixed for the life of an
engine, so the min is computed once per transport arming, cached on the
transport, and shared by every hook.

These mirror the eager-routing fakes of tests/test_producer_chunked_schema.py
but need no native extension: ``dispatch_producer`` is monkeypatched and the
counting engine stands in for the native one behind a real ``RingTransport``,
so the cache under test is the production one.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


class _CountingEagerRingEngine:
    def __init__(self, available: int, capacity: int, staging: int | None = None):
        self.available = available
        self.capacity = capacity
        self.staging = capacity if staging is None else staging
        self.reserved: list[int] = []
        self.flushes = 0
        self.payload_cap_calls = 0
        self.staging_cap_calls = 0

    def payload_tensor(self) -> torch.Tensor:
        return torch.empty(64, dtype=torch.uint8, device="cuda")

    def available_capacity(self) -> int:
        return self.available

    def payload_cap(self) -> int:
        self.payload_cap_calls += 1
        return self.capacity

    def staging_cap(self) -> int:
        self.staging_cap_calls += 1
        return self.staging

    def reserve_one(self, nbytes: int) -> None:
        self.reserved.append(nbytes)

    def flush_and_wait(self) -> None:
        self.flushes += 1


def _eager_hook(monkeypatch, engine, dispatched):
    from dmi.hooks.point import HookPoint
    from dmi.transport import ring as ring_transport
    from dmi.transport.ring import RingTransport

    transport = RingTransport(engine)
    transport.force_eager = True
    transport.direct = []
    transport.submit_cpu_direct = (
        lambda tensor, hook_type, hook_id: transport.direct.append(tensor))
    monkeypatch.setattr(ring_transport, "_active_transport", transport)
    monkeypatch.setattr(
        "dmi.hooks.point.dispatch_producer",
        lambda *args: dispatched.append(args),
    )
    hook = HookPoint()
    hook._ring_hook_type = 1
    hook._ring_hook_id = 2
    hook._ring_payload = transport._ring_payload
    return hook, transport


def test_repeated_forwards_query_the_ring_caps_at_most_once(monkeypatch):
    """The caps are engine-lifetime constants; per-step pybind crossings are
    exactly what the native header forbids for these getters."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = _CountingEagerRingEngine(available=4096, capacity=4096)
    dispatched = []
    hook, transport = _eager_hook(monkeypatch, engine, dispatched)

    value = torch.arange(17, dtype=torch.uint8, device="cuda")
    for _ in range(5):
        hook(value)

    assert len(dispatched) == 5, "routing must still take the ring path"
    assert transport.direct == []
    assert engine.payload_cap_calls <= 1
    assert engine.staging_cap_calls <= 1


def test_the_strip_cpu_direct_branch_never_queries_the_caps(monkeypatch):
    """Stripped producers bypass the ring entirely; computing the ceiling
    there was pure waste."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = _CountingEagerRingEngine(available=4096, capacity=4096)
    dispatched = []
    hook, transport = _eager_hook(monkeypatch, engine, dispatched)
    hook._strip_tensor = torch.tensor([1], dtype=torch.int64, device="cuda")
    hook._strip_row_bytes = 8

    value = torch.arange(17, dtype=torch.uint8, device="cuda")
    hook(value)

    assert len(transport.direct) == 1
    assert dispatched == []
    assert engine.payload_cap_calls == 0
    assert engine.staging_cap_calls == 0


def test_the_cached_ceiling_still_routes_over_staging_bytes_to_cpu_direct(monkeypatch):
    """Routing behavior is unchanged by the caching: a tensor larger than
    pinned staging must still bypass the ring, and one that fits must not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = _CountingEagerRingEngine(available=4096, capacity=4096, staging=64)
    dispatched = []
    hook, transport = _eager_hook(monkeypatch, engine, dispatched)

    hook(torch.arange(128, dtype=torch.uint8, device="cuda"))  # > staging
    hook(torch.arange(32, dtype=torch.uint8, device="cuda"))   # fits

    assert len(transport.direct) == 1, "over-staging tensor must go cpu-direct"
    assert len(dispatched) == 1, "fitting tensor must still take the ring"
    assert engine.reserved == [32]


def test_a_new_arming_queries_the_new_engines_caps(monkeypatch):
    """A re-attach arms a fresh transport over a possibly different engine;
    the cache is per-transport, so the new caps are read, not the old min."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine_one = _CountingEagerRingEngine(available=4096, capacity=4096)
    hook_one, transport_one = _eager_hook(monkeypatch, engine_one, [])
    hook_one(torch.arange(17, dtype=torch.uint8, device="cuda"))
    assert transport_one.effective_cap == 4096
    assert engine_one.payload_cap_calls == 1
    assert engine_one.staging_cap_calls == 1

    # Re-arming over an engine whose staging is smaller: the new transport
    # must publish the new ceiling, and the old engine is never consulted.
    engine_two = _CountingEagerRingEngine(
        available=4096, capacity=4096, staging=64)
    hook_two, transport_two = _eager_hook(monkeypatch, engine_two, [])
    hook_two(torch.arange(17, dtype=torch.uint8, device="cuda"))
    assert transport_two.effective_cap == 64
    assert engine_two.payload_cap_calls == 1
    assert engine_two.staging_cap_calls == 1
    assert engine_one.payload_cap_calls == 1, "old engine must not be re-read"
