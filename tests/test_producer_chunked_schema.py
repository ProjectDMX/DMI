"""Schema + dispatch tests for the three producer ops.

End-to-end strip verification (kernel actually copies fewer bytes
and consumer receives stripped output) requires real engine +
ClickHouse plumbing and lives in the vLLM regression sweep with
`padding_strip=True`.  These tests cover the wiring surface:
- the three torch ops are registered
- each accepts the right arg shape (smoke; no engine required)
- HookPoint's strip-mode attributes default to static path
- HookPoint dispatch is reachable via direct invocation
"""
from __future__ import annotations

import pytest
import torch

from monitoring._native_engine import _load_extension

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


def test_hook_point_strip_attrs_default_to_static():
    """HookPoint instances default to the static path."""
    from monitoring.hook_points import HookPoint
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
def test_hook_point_strip_attrs_settable_for_prefix_mode():
    """Setting _strip_tensor + _strip_row_bytes > 0 selects prefix mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from monitoring.hook_points import HookPoint
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
    from monitoring.hook_points import HookPoint
    hp = HookPoint()
    cb = torch.tensor([3, 7, 0, 5], dtype=torch.int64, device="cuda")
    hp._strip_tensor = cb
    # _strip_row_bytes stays at default 0
    assert hp._strip_tensor is cb
    assert hp._strip_row_bytes == 0
