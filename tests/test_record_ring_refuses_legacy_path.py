"""A record ring refuses every legacy producer entry, on a real engine.

After ``create_record_runtime`` the engine's ring is a record ring, and the
active transport points HookPoints at it. The legacy protocol shares that
ring but not its bookkeeping. ``push_step`` already refused legacy metadata;
the reservation and producer entries did not:

* ``prepare_step`` / ``reserve_one`` advanced the CPU heads with no record
  publication behind them. The space was never reclaimed, and each later
  ``reserve_record`` reclaim was registered against the wrong task sequence.
  HF's ``commit_step`` made exactly that reservation every step, then hit the
  ``push_step`` refusal.
* ``hook_no_notify*`` launched producers whose payloads carry no record
  descriptor, so the record consumer paired them with the next record's
  descriptor, or failed the ring when none was queued.
* ``submit_cpu_direct`` (the safety net's bypass) did the same from the host.

Each now throws before touching the ring. Above them, ``commit_step`` refuses
record mode, which also covers an adapter attached before
``create_record_runtime``: it still holds the stopped legacy ring, where the
step went through without any error.

The CPU contract for the HF side is tests/test_hf_capture_refusal.py; the
native unit test is test_record_ring_refuses_every_legacy_producer_entry in
tests/native/ring/test_ring_engine.cu.

Needs CUDA and the full native backend. No sink: the record ring is created
without one, which is enough to be a record ring and never reaches a sink.
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

REFUSED = "cannot be used on a record ring"
RECORD_MODE_REFUSAL = r"commit_step\(\): the engine is in record mode"


def _ring_config():
    from dmi.api.v1 import RingConfig

    config = RingConfig()
    config.task_ring_entries = 32
    config.payload_ring_bytes = 64 * 1024
    config.pinned_staging_bytes = 64 * 1024
    return config


@pytest.fixture
def record_engine():
    from dmi.engine import MonitoringEngine
    from dmi.storage.capture import CaptureRecordFormat

    engine = MonitoringEngine(model_id="record-ring-refusal",
                              ring_config=_ring_config())
    try:
        runtime = engine.create_record_runtime(CaptureRecordFormat())
        assert engine._record_mode
        yield engine
        del runtime
    finally:
        engine.close()


def _armed_hook_point(transport, strip):
    """A legacy HookPoint as install_ring_hooks leaves it, in a strip mode."""
    from dmi.hooks.point import HookPoint
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    hook = HookPoint()
    hook._ring_hook_type = HOOK_TYPE_RESID_PRE
    hook._ring_hook_id = 0
    hook._ring_payload = transport._ring_payload
    if strip == "prefix":
        hook._strip_tensor = torch.ones(1, dtype=torch.int64, device="cuda")
        hook._strip_row_bytes = 16
    elif strip == "chunked":
        hook._strip_tensor = torch.full((2,), 8, dtype=torch.int64,
                                        device="cuda")
        hook._strip_row_bytes = 0
    return hook


def test_prepare_step_on_a_record_ring_raises_and_reserves_nothing(
        record_engine):
    ring = record_engine._ring_engine
    before = ring.available_capacity()

    with pytest.raises(RuntimeError,
                       match=f"legacy step reservation {REFUSED}"):
        ring.prepare_step(64, 1)

    assert ring.available_capacity() == before


@pytest.mark.parametrize("strip", ["static", "prefix", "chunked"])
def test_a_legacy_hook_call_on_a_record_ring_raises(record_engine, strip):
    """The fast path: HookPoint -> ring::producer* -> hook_no_notify*."""
    transport = record_engine._ring_transport
    assert transport.capture_step and not transport.force_eager
    hook = _armed_hook_point(transport, strip)

    with pytest.raises(RuntimeError, match=f"legacy producer {REFUSED}"):
        hook(torch.ones(4, 4, device="cuda"))
    torch.cuda.synchronize()


@pytest.mark.parametrize(("strip", "refused"), [
    ("static", "legacy per-hook reservation"),
    ("prefix", "legacy CPU submission"),
])
def test_the_eager_safety_net_on_a_record_ring_raises(
        record_engine, strip, refused):
    """force_eager: reserve_one before the producer, or the CPU bypass."""
    transport = record_engine._ring_transport
    ring = record_engine._ring_engine
    before = ring.available_capacity()
    hook = _armed_hook_point(transport, strip)
    transport.force_eager = True
    try:
        with pytest.raises(RuntimeError, match=f"{refused} {REFUSED}"):
            hook(torch.ones(4, 4, device="cuda"))
    finally:
        transport.force_eager = False

    assert ring.available_capacity() == before


def test_hf_generation_after_create_record_runtime_raises(record_engine):
    """Audit link 7 on a real engine: the HF step fails loudly, and its
    reservation no longer leaks record-ring capacity first."""
    from dmi.adapters.huggingface.generation import generate_with_monitoring
    from tests.test_hf_capture_refusal import _TinyHookedLM

    ring = record_engine._ring_engine
    before = ring.available_capacity()
    model = _TinyHookedLM(record_engine)
    input_ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
        generate_with_monitoring(model, input_ids,
                                 attention_mask=torch.ones_like(input_ids))

    assert ring.available_capacity() == before


def test_an_adapter_attached_before_create_record_runtime_raises():
    """The attach_config-then-create_record_runtime order. The adapter keeps
    the legacy ring and transport it took at attach, and create_record_runtime
    has stopped that ring. Its prepare_step and metadata push went there and
    succeeded, so the step raised nothing until a HookPoint fired eagerly on
    the record ring; a replayed graph would not have fired at all."""
    from dmi.adapters.huggingface.adapter import HuggingFaceAdapter
    from dmi.engine import MonitoringEngine
    from dmi.storage.capture import CaptureRecordFormat
    from tests.test_hf_capture_refusal import _TinyHookedLM

    engine = MonitoringEngine(model_id="record-ring-refusal-stale",
                              ring_config=_ring_config())
    try:
        model = _TinyHookedLM(engine)
        adapter = HuggingFaceAdapter(engine, "record-ring-refusal-stale")
        adapter.attach_model(model)
        runtime = engine.create_record_runtime(CaptureRecordFormat())
        ring = engine._ring_engine
        assert adapter.ring_engine is not ring
        before = ring.available_capacity()
        input_ids = torch.tensor([[1, 2, 3]], device="cuda")

        try:
            with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
                model.prepare_inputs_for_generation(
                    input_ids, attention_mask=torch.ones_like(input_ids))
        finally:
            adapter.detach_model(model)

        assert ring.available_capacity() == before
        del runtime
    finally:
        engine.close()
