"""Schema + dispatch tests for the three producer ops.

End-to-end strip verification (kernel actually copies fewer bytes and the
consumer receives stripped output) requires a real engine and ClickHouse.
These tests cover the framework-neutral wiring surface:
- the three torch ops are registered
- each accepts the right arg shape (smoke; no engine required)
- HookPoint's strip-mode attributes default to static path
- HookPoint dispatch is reachable via direct invocation
"""
from __future__ import annotations

import pytest
import torch

from dmi.transport.native import _load_extension

pytestmark = pytest.mark.native_backend


def setup_module(module):  # noqa: D401 -- pytest hook
    try:
        _load_extension()  # ensure .so loaded -> three ring ops registered
    except ImportError as exc:
        pytest.skip(f"DMI native backend required: {exc}", allow_module_level=True)


# --- Registration / wiring: no CUDA device required, but native backend needed.

def test_producer_op_registered():
    assert torch.ops.ring.producer.default is not None


def test_producer_prefix_op_registered():
    assert torch.ops.ring.producer_prefix.default is not None


def test_producer_chunked_op_registered():
    assert torch.ops.ring.producer_chunked.default is not None


@pytest.mark.parametrize(
    "name",
    [
        "record_producer",
        "record_producer_prefix",
        "record_producer_chunked",
        "record_producer_seq_prefix_pack",
        "record_producer_segmented_pack",
    ],
)
def test_additive_record_producer_ops_registered(name):
    assert getattr(torch.ops.ring, name).default is not None


def test_hook_point_strip_attrs_default_to_static():
    """HookPoint instances default to the static path."""
    from dmi.hooks.point import HookPoint
    hp = HookPoint()
    assert hp._strip_tensor is None
    assert hp._strip_row_bytes == 0


# --- Device smoke: allocate CUDA tensors and dispatch the op (GPU only) -------

@pytest.mark.gpu
def test_producer_static_smoke():
    """Static op accepts (Tensor(a!), Tensor, int, int).  C++ impl
    early-returns when no engine is active, so this is a pure schema
    smoke test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ring_payload = torch.zeros(64, dtype=torch.uint8, device="cuda")
    x = torch.zeros(16, dtype=torch.float32, device="cuda")
    torch.ops.ring.producer(ring_payload, x, 0, 0)


@pytest.mark.gpu
def test_producer_prefix_smoke():
    """Prefix op accepts (Tensor(a!), Tensor, Tensor, int, int, int)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ring_payload = torch.zeros(64, dtype=torch.uint8, device="cuda")
    x = torch.zeros(16, dtype=torch.float32, device="cuda")
    row_count = torch.tensor([2], dtype=torch.int64, device="cuda")
    torch.ops.ring.producer_prefix(ring_payload, x, row_count, 8, 0, 0)


@pytest.mark.gpu
def test_producer_chunked_smoke():
    """Chunked op accepts (Tensor(a!), Tensor, Tensor, int, int)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ring_payload = torch.zeros(64, dtype=torch.uint8, device="cuda")
    x = torch.zeros(64, dtype=torch.float32, device="cuda")
    chunk_bytes = torch.tensor([16, 32, 0, 8], dtype=torch.int64, device="cuda")
    torch.ops.ring.producer_chunked(ring_payload, x, chunk_bytes, 0, 0)


@pytest.mark.gpu
def test_record_producer_ops_smoke_without_active_engine():
    """New schemas accept their physical arguments without changing legacy ops."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ring_payload = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    x = torch.zeros((4, 2, 8), dtype=torch.float32, device="cuda")
    gate = torch.tensor([1], dtype=torch.int32, device="cuda")
    row_count = torch.tensor([2], dtype=torch.int64, device="cuda")
    chunk_bytes = torch.tensor([128, 128], dtype=torch.int64, device="cuda")
    valid_count = torch.tensor([3, 2], dtype=torch.int64, device="cuda")
    valid_prefix = torch.tensor([0, 3, 5], dtype=torch.int64, device="cuda")
    starts = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    ends = torch.tensor([2, 4], dtype=torch.int64, device="cuda")

    torch.ops.ring.record_producer(ring_payload, x, gate, 1)
    torch.ops.ring.record_producer_prefix(
        ring_payload, x, row_count, 64, gate, 1
    )
    torch.ops.ring.record_producer_chunked(
        ring_payload, x, chunk_bytes, gate, 1
    )
    torch.ops.ring.record_producer_seq_prefix_pack(
        ring_payload, x, valid_count, valid_prefix, 32, gate, 1
    )
    torch.ops.ring.record_producer_segmented_pack(
        ring_payload, x, starts, ends, 64, gate, 1
    )


@pytest.mark.gpu
def test_hook_point_strip_attrs_settable_for_prefix_mode():
    """Setting _strip_tensor + _strip_row_bytes > 0 selects prefix mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from dmi.hooks.point import HookPoint
    hp = HookPoint()
    rc = torch.tensor([3], dtype=torch.int64, device="cuda")
    hp._strip_tensor = rc
    hp._strip_row_bytes = 8
    assert hp._strip_tensor is rc
    assert hp._strip_row_bytes == 8


@pytest.mark.gpu
def test_hook_point_strip_attrs_settable_for_chunked_mode():
    """Setting _strip_tensor + _strip_row_bytes == 0 selects chunked mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from dmi.hooks.point import HookPoint
    hp = HookPoint()
    cb = torch.tensor([3, 7, 0, 5], dtype=torch.int64, device="cuda")
    hp._strip_tensor = cb
    # _strip_row_bytes stays at default 0
    assert hp._strip_tensor is cb
    assert hp._strip_row_bytes == 0


class _FakeEagerRingEngine:
    def __init__(self, available: int, capacity: int):
        self.available = available
        self.capacity = capacity
        self.reserved: list[int] = []
        self.flushes = 0

    def available_capacity(self) -> int:
        return self.available

    def payload_cap(self) -> int:
        return self.capacity

    def reserve_one(self, nbytes: int) -> None:
        self.reserved.append(nbytes)

    def flush_and_wait(self) -> None:
        self.flushes += 1


class _FakeEagerTransport:
    force_eager = True

    def __init__(self, engine: _FakeEagerRingEngine):
        self._ring_engine = engine
        self.direct: list[torch.Tensor] = []

    def submit_cpu_direct(self, tensor, hook_type, hook_id) -> None:
        self.direct.append(tensor)


@pytest.mark.gpu
def test_eager_ring_capacity_uses_padded_transport_size(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from dmi.hooks.point import HookPoint
    from dmi.transport import ring as ring_transport

    engine = _FakeEagerRingEngine(available=17, capacity=64)
    transport = _FakeEagerTransport(engine)
    monkeypatch.setattr(ring_transport, "_active_transport", transport)
    dispatched = []
    monkeypatch.setattr(
        "dmi.hooks.point.dispatch_producer",
        lambda *args: dispatched.append(args),
    )

    hook = HookPoint()
    hook._ring_hook_type = 1
    hook._ring_hook_id = 2
    hook._ring_payload = torch.empty(64, dtype=torch.uint8, device="cuda")
    value = torch.arange(17, dtype=torch.uint8, device="cuda")
    result = hook(value)

    assert result.data_ptr() == value.data_ptr()
    assert engine.flushes == 1
    assert engine.reserved == [17]
    assert len(dispatched) == 1
    assert transport.direct == []


@pytest.mark.gpu
def test_eager_stripped_v0_uses_cpu_direct_without_reservation(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from dmi.hooks.point import HookPoint
    from dmi.transport import ring as ring_transport

    engine = _FakeEagerRingEngine(available=4096, capacity=4096)
    transport = _FakeEagerTransport(engine)
    monkeypatch.setattr(ring_transport, "_active_transport", transport)
    dispatched = []
    monkeypatch.setattr(
        "dmi.hooks.point.dispatch_producer",
        lambda *args: dispatched.append(args),
    )

    hook = HookPoint()
    hook._ring_hook_type = 1
    hook._ring_hook_id = 2
    hook._ring_payload = torch.empty(64, dtype=torch.uint8, device="cuda")
    hook._strip_tensor = torch.tensor([1], dtype=torch.int64, device="cuda")
    hook._strip_row_bytes = 8
    value = torch.arange(17, dtype=torch.uint8, device="cuda")
    result = hook(value)

    assert result.data_ptr() == value.data_ptr()
    assert engine.flushes == 1
    assert engine.reserved == []
    assert dispatched == []
    assert len(transport.direct) == 1
    assert torch.equal(transport.direct[0], value.cpu())
