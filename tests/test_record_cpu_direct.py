"""CPU-direct record transformations must match CUDA producer bytes."""

from __future__ import annotations

import pytest
import torch

from dmi.hooks.record import HookOutput, TransportSpec, TransportType
from dmi.hooks.producer_plan import ProducerPlanBuilder
from dmi.transport.ring import RingTransport


pytestmark = pytest.mark.cpu


def _entry(
    output: HookOutput,
    *,
    transport_type: TransportType,
    feature_bytes: int,
):
    return ProducerPlanBuilder().record_output(
        output_id=1 << 16,
        output_spec=TransportSpec(
            "packed",
            transport_type=transport_type,
            feature_bytes=feature_bytes,
        ),
        output=output,
    )


def test_seq_prefix_cpu_direct_uses_flattened_feature_bytes() -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    output = HookOutput(
        source,
        (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 3], dtype=torch.int64),
        ),
    )
    entry = _entry(
        output,
        transport_type=TransportType.SEQ_PREFIX_PACK,
        feature_bytes=4,
    )

    actual = RingTransport._record_cpu_tensor(output, entry)
    expected = (
        torch.tensor([0, 1, 3], dtype=torch.float32)
        .view(torch.uint8)
        .reshape(-1)
    )

    assert actual.dtype is torch.uint8
    assert torch.equal(actual, expected)


def test_segmented_cpu_direct_uses_flattened_feature_bytes() -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    output = HookOutput(
        source,
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
        ),
    )
    entry = _entry(
        output,
        transport_type=TransportType.SEGMENTED_PACK,
        feature_bytes=4,
    )

    actual = RingTransport._record_cpu_tensor(output, entry)
    expected = (
        torch.tensor([0, 1], dtype=torch.float32)
        .view(torch.uint8)
        .reshape(-1)
    )

    assert actual.dtype is torch.uint8
    assert torch.equal(actual, expected)
