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
    **spec_args: int,
):
    return ProducerPlanBuilder().record_output(
        output_id=1 << 16,
        output_spec=TransportSpec(
            "packed",
            transport_type=transport_type,
            **spec_args,
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


def test_identity_cpu_direct_returns_the_source_tensor_unflattened() -> None:
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    output = HookOutput(source, ())
    entry = _entry(output, transport_type=TransportType.IDENTITY)

    actual = RingTransport._record_cpu_tensor(output, entry)

    assert actual.dtype is torch.float32
    assert actual.shape == (2, 2)
    assert torch.equal(actual, source)


@pytest.mark.parametrize(
    ("row_count", "expected_rows"),
    [(2, 2), (6, 4), (-1, 0)],
)
def test_prefix_strip_cpu_direct_clamps_row_count_like_the_cuda_kernel(
    row_count: int,
    expected_rows: int,
) -> None:
    source = torch.arange(4, dtype=torch.float32)
    output = HookOutput(source, (torch.tensor([row_count], dtype=torch.int64),))
    entry = _entry(
        output,
        transport_type=TransportType.PREFIX_STRIP,
        row_bytes=4,
    )

    actual = RingTransport._record_cpu_tensor(output, entry)
    expected = source[:expected_rows].view(torch.uint8).reshape(-1)

    assert actual.dtype is torch.uint8
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([2, 2], [0, 1, 4, 5]),
        ([6, 2], [0, 1, 2, 3, 4, 5]),
        ([-1, 4], [4, 5, 6, 7]),
    ],
)
def test_chunked_cpu_direct_clamps_each_count_into_its_own_chunk(
    counts: list[int],
    expected: list[int],
) -> None:
    source = torch.arange(8, dtype=torch.uint8)
    output = HookOutput(source, (torch.tensor(counts, dtype=torch.int64),))
    entry = _entry(output, transport_type=TransportType.CHUNKED)

    actual = RingTransport._record_cpu_tensor(output, entry)

    assert actual.dtype is torch.uint8
    assert actual.tolist() == expected


def test_chunked_cpu_direct_rejects_unequal_input_chunks() -> None:
    source = torch.arange(7, dtype=torch.uint8)
    output = HookOutput(source, (torch.tensor([1, 1], dtype=torch.int64),))
    entry = _entry(output, transport_type=TransportType.CHUNKED)

    with pytest.raises(ValueError, match="equal input chunks"):
        RingTransport._record_cpu_tensor(output, entry)
