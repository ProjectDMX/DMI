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


def _source_bytes(tensor: torch.Tensor) -> bytes:
    """The producer's view of ``tensor``: its contiguous payload bytes."""

    return bytes(
        tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()
    )


def _actual_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


# ----------------------------------------------------------------------
# Transcriptions of the three CUDA kernels these transforms must match.
# Kept deliberately literal -- byte slicing only, no torch -- so a drift in
# ring.py cannot be absorbed by sharing its helpers.


def _cuda_static(src: bytes) -> bytes:
    """``record_producer_static_kernel``: copy all ``nbytes`` of the source."""

    return src


def _cuda_prefix(src: bytes, row_count: int, row_bytes: int) -> bytes:
    """``record_producer_prefix_kernel``: clamp rows, else copy everything."""

    rows = max(0, row_count)
    actual = len(src)
    if row_bytes != 0 and rows <= len(src) // row_bytes:
        actual = rows * row_bytes
    return src[:actual]


def _cuda_chunked(src: bytes, counts: list[int]) -> bytes:
    """``record_producer_chunked_kernel``: per-chunk prefix, packed."""

    input_chunk = len(src) // len(counts)
    out = bytearray()
    for index, value in enumerate(counts):
        selected = min(max(0, value), input_chunk)
        begin = index * input_chunk
        out += src[begin : begin + selected]
    return bytes(out)


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


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((4,), torch.float32),
        ((3, 2), torch.float32),
        ((2, 2, 2), torch.float32),
        ((5, 7), torch.float32),
        ((2, 3, 4), torch.int16),
        ((1,), torch.float64),
    ],
)
def test_identity_cpu_direct_copies_every_source_byte(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    # IDENTITY is a whole-tensor copy: the producer sends `nbytes` and keeps
    # the source dtype and shape, unlike every other transform here, which
    # hands back a flat uint8 byte run.
    source = torch.arange(
        1, 1 + torch.Size(shape).numel(), dtype=dtype
    ).reshape(shape)
    output = HookOutput(source, ())
    entry = _entry(output, transport_type=TransportType.IDENTITY)

    actual = RingTransport._record_cpu_tensor(output, entry)

    assert actual.dtype is dtype
    assert tuple(actual.shape) == shape
    assert _actual_bytes(actual) == _cuda_static(_source_bytes(source))


@pytest.mark.parametrize(
    "rows_total, cols, row_bytes, row_count",
    [
        (4, 2, 8, 2),      # a strict prefix of the rows
        (4, 2, 8, 0),      # zero rows -> empty payload
        (4, 2, 8, -3),     # negative row count clamps to zero
        (4, 2, 8, 4),      # exactly n // row_bytes -> the whole payload
        (4, 2, 8, 5),      # one past n // row_bytes -> the whole payload
        (4, 2, 8, 9_999),  # far past the cap -> still the whole payload
        (6, 3, 12, 5),     # row_bytes wider than the trailing dimension
        (6, 3, 4, 7),      # row_bytes narrower than one source row
        (1, 1, 4, 1),      # single row, exact fit
    ],
)
def test_prefix_strip_cpu_direct_clamps_row_count(
    rows_total: int, cols: int, row_bytes: int, row_count: int
) -> None:
    source = torch.arange(rows_total * cols, dtype=torch.float32).reshape(
        rows_total, cols
    )
    output = HookOutput(source, (torch.tensor([row_count], dtype=torch.int64),))
    entry = _entry(
        output,
        transport_type=TransportType.PREFIX_STRIP,
        row_bytes=row_bytes,
    )

    actual = RingTransport._record_cpu_tensor(output, entry)

    assert actual.dtype is torch.uint8
    assert _actual_bytes(actual) == _cuda_prefix(
        _source_bytes(source), row_count, row_bytes
    )


@pytest.mark.parametrize(
    "numel, counts",
    [
        # Counts are BYTES, and each chunk is `numel * 4 // len(counts)` bytes
        # wide, so an over-cap count has to be larger than that width.
        (8, [4, 4]),        # a short prefix of each 16-byte chunk
        (8, [16, 16]),      # exactly the chunk width -> nothing dropped
        (8, [64, 64]),      # both counts over the width -> clamped to it
        (8, [0, 64]),       # a dropped chunk beside an over-cap one
        (8, [-5, 12]),      # a negative count clamps to zero
        (8, [40]),          # one chunk asking for more than its 32 bytes
        (12, [4, 8, 12]),   # unequal partial chunks pack contiguously
        (12, [40, 0, 4]),   # leading chunk over cap, middle chunk dropped
        (16, [7, 1]),       # counts that are not multiples of the item size
        (4, [1, 1, 1, 1]),  # one byte out of each 4-byte chunk
    ],
)
def test_chunked_cpu_direct_clamps_each_chunk(
    numel: int, counts: list[int]
) -> None:
    source = torch.arange(numel, dtype=torch.float32)
    output = HookOutput(source, (torch.tensor(counts, dtype=torch.int64),))
    entry = _entry(output, transport_type=TransportType.CHUNKED)

    actual = RingTransport._record_cpu_tensor(output, entry)

    assert actual.dtype is torch.uint8
    assert _actual_bytes(actual) == _cuda_chunked(_source_bytes(source), counts)


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
