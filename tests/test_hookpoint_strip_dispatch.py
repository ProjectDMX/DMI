"""Routing tests for HookPoint's three-way strip-mode producer dispatch.

``_dispatch_producer`` selects one of three registered ring ops from two
per-HookPoint attributes.  The choice is invisible at the call site and the
three argument lists differ in both length and order, so a mis-route or a
swapped positional argument sends the wrong bytes to the consumer without
raising.  ``tests/test_producer_chunked_schema.py`` asserts that the
attributes *hold* the values that select a mode; nothing asserts that the
selection actually happens.

Both dispatch sites are covered: the fast path at the end of
``HookPoint.forward`` and the eager safety net's reserve-then-dispatch
branches, which forward the same strip state.

These tests replace the whole ``ring`` op namespace, so they behave
identically whether or not the native ops are registered.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

try:
    from monitoring import hook_points, ring_transport
    _NATIVE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on build environment
    hook_points = None
    ring_transport = None
    _NATIVE_IMPORT_ERROR = exc


pytestmark = [
    pytest.mark.native_backend,
    pytest.mark.skipif(
        _NATIVE_IMPORT_ERROR is not None,
        reason=f"DMI native backend required: {_NATIVE_IMPORT_ERROR}",
    ),
]

HOOK_TYPE = 7
HOOK_ID = 3
ROW_BYTES = 8


class _FakeCudaTensor:
    is_cuda = True

    def __init__(self, nbytes: int = 16) -> None:
        self.nbytes = nbytes

    def contiguous(self):
        return self

    def cpu(self):
        return ("cpu", self.nbytes)


class _RoomyEngine:
    """Grants every reservation so the eager net reaches its dispatch."""

    def available_capacity(self) -> int:
        return 1 << 30

    def payload_cap(self) -> int:
        return 1 << 30

    def staging_cap(self) -> int:
        return 1 << 30

    def reserve_one(self, nbytes: int) -> None:
        pass

    def flush_and_wait(self) -> None:
        pass


class _TightEngine:
    """Rejects everything so the eager net takes the cpu_direct bypass."""

    def available_capacity(self) -> int:
        return 0

    def payload_cap(self) -> int:
        return 0

    def staging_cap(self) -> int:
        return 0

    def flush_and_wait(self) -> None:
        pass


class _Transport:
    def __init__(self, engine, *, force_eager: bool) -> None:
        self._ring_engine = engine
        self.force_eager = force_eager
        self.direct = []

    def submit_cpu_direct(self, tensor, hook_type: int, hook_id: int) -> None:
        self.direct.append((tensor, hook_type, hook_id))


@pytest.fixture
def ring_ops(monkeypatch):
    """Replace the `ring` op namespace with recorders."""
    calls = []

    def _recorder(name):
        def _op(*args):
            calls.append((name, args))
        return _op

    monkeypatch.setattr(
        torch.ops,
        "ring",
        SimpleNamespace(
            producer=_recorder("producer"),
            producer_prefix=_recorder("producer_prefix"),
            producer_chunked=_recorder("producer_chunked"),
        ),
    )
    return calls


@pytest.fixture(autouse=True)
def restore_active_transport():
    previous = ring_transport._active_transport
    try:
        yield
    finally:
        ring_transport._active_transport = previous


@pytest.fixture(params=["fast", "eager"])
def dispatch_path(request):
    """Run through the fast path and the eager reserve-then-dispatch path."""
    if request.param == "fast":
        ring_transport._active_transport = None
    else:
        ring_transport._active_transport = _Transport(
            _RoomyEngine(), force_eager=True
        )
    return request.param


def _hook(payload, *, strip_tensor=None, strip_row_bytes=0):
    hp = hook_points.HookPoint()
    hp._ring_hook_type = HOOK_TYPE
    hp._ring_hook_id = HOOK_ID
    hp._ring_payload = payload
    hp._strip_tensor = strip_tensor
    hp._strip_row_bytes = strip_row_bytes
    return hp


def test_no_strip_tensor_routes_to_producer(ring_ops, dispatch_path):
    payload = object()
    tensor = _FakeCudaTensor()

    _hook(payload)(tensor)

    assert ring_ops == [("producer", (payload, tensor, HOOK_TYPE, HOOK_ID))]


def test_positive_row_bytes_routes_to_producer_prefix(ring_ops, dispatch_path):
    payload = object()
    tensor = _FakeCudaTensor()
    row_count = torch.tensor([4], dtype=torch.int64)

    _hook(payload, strip_tensor=row_count, strip_row_bytes=ROW_BYTES)(tensor)

    assert ring_ops == [
        (
            "producer_prefix",
            (payload, tensor, row_count, ROW_BYTES, HOOK_TYPE, HOOK_ID),
        )
    ]


def test_zero_row_bytes_routes_to_producer_chunked(ring_ops, dispatch_path):
    payload = object()
    tensor = _FakeCudaTensor()
    chunk_bytes = torch.tensor([8, 16], dtype=torch.int64)

    _hook(payload, strip_tensor=chunk_bytes, strip_row_bytes=0)(tensor)

    assert ring_ops == [
        (
            "producer_chunked",
            (payload, tensor, chunk_bytes, HOOK_TYPE, HOOK_ID),
        )
    ]


@pytest.mark.parametrize("strip_row_bytes", [0, ROW_BYTES])
def test_cpu_direct_bypass_sends_the_unstripped_tensor(
    ring_ops, strip_row_bytes
):
    """The oversized bypass deliberately does not apply the strip.

    ``HookPoint.forward`` documents this: the consumer's CPU-side slicing
    handles request-level demux for bypassed tensors.  No producer op runs at
    all, so the strip state must not leak into the submission.
    """
    transport = _Transport(_TightEngine(), force_eager=True)
    ring_transport._active_transport = transport
    tensor = _FakeCudaTensor(4096)

    _hook(
        object(),
        strip_tensor=torch.tensor([1], dtype=torch.int64),
        strip_row_bytes=strip_row_bytes,
    )(tensor)

    assert ring_ops == []
    assert transport.direct == [(("cpu", 4096), HOOK_TYPE, HOOK_ID)]
