"""Reservation-side contract for dynamic actual-byte reclamation."""

from __future__ import annotations

import pytest
import torch

from dmi.hooks.record import HookOutput, TransportSpec, TransportType
from dmi.hooks.producer_plan import ProducerPlanBuilder
from dmi.transport.ring import RingTransport

pytestmark = pytest.mark.cpu


def test_dynamic_plan_reserves_upper_bound_not_current_device_count():
    output = HookOutput(
        torch.empty(64, dtype=torch.float32),
        (torch.tensor([8, 16, 0, 4], dtype=torch.int64),),
    )
    entry = ProducerPlanBuilder().record_output(
        output_id=1 << 16,
        output_spec=TransportSpec(
            "packed",
            transport_type=TransportType.CHUNKED,
            reservation_upper_bytes=256,
            output_shape=(-1,),
        ),
        output=output,
    )

    assert entry.reservation_upper_bytes == 256
    assert entry.aligned_reservation_bytes == 256
    assert entry.task_count == 1


def test_record_transport_delegates_ordered_per_entry_reservations():
    calls = []

    class _Ring:
        def payload_tensor(self):
            return torch.empty(0, dtype=torch.uint8)

        def reserve_record(self, items):
            calls.append(items)
            return 0

    transport = RingTransport(_Ring())

    items = ((64, False), (256, True))
    assert transport.reserve_record(items) == 0
    assert calls == [items]
