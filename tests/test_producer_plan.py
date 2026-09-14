"""Tests for ordered, framework-neutral producer plans."""

from __future__ import annotations

import pytest
import torch
from dataclasses import replace

from dmi.hooks.record import HookOutput, OutputSizingMode, TransportSpec, TransportType
from dmi.hooks.producer_plan import (
    ProducerPlanBuilder, ProducerPlanEntry, _align_up, _dtype_element_size,
)

pytestmark = pytest.mark.cpu


def test_builder_preserves_order_and_derives_totals():
    builder = ProducerPlanBuilder()
    first = builder.record_output(
        output_id=100,
        output_spec=TransportSpec("a"),
        output=HookOutput(torch.empty(3, dtype=torch.float32)),
    )
    second = builder.record_output(
        output_id=101,
        output_spec=TransportSpec(
            "b",
            transport_type=TransportType.CHUNKED,
            reservation_upper_bytes=33,
            output_shape=(-1,),
        ),
        output=HookOutput(
            torch.empty(8, dtype=torch.float16),
            (torch.tensor([8, 8], dtype=torch.int64),),
        ),
    )

    plan = builder.build()

    assert plan.entries == (first, second)
    assert plan.total_reservation_bytes == 16 + 48
    assert plan.task_count == 2
    assert first.input_shape == (3,)
    assert second.output_shape == (-1,)


def test_plan_compatibility_reports_first_structural_difference():
    left = ProducerPlanBuilder()
    right = ProducerPlanBuilder()
    left.record_output(
        output_id=100,
        output_spec=TransportSpec("x"),
        output=HookOutput(torch.empty(2)),
    )
    right.record_output(
        output_id=100,
        output_spec=TransportSpec("x"),
        output=HookOutput(torch.empty(3)),
    )

    with pytest.raises(ValueError, match="entry 0"):
        left.build().assert_compatible(right.build())


def test_identity_bound_cannot_be_smaller_than_input():
    builder = ProducerPlanBuilder()
    with pytest.raises(ValueError, match="cannot be smaller"):
        builder.record_output(
            output_id=100,
            output_spec=TransportSpec("x", reservation_upper_bytes=1),
            output=HookOutput(torch.empty(2, dtype=torch.float32)),
        )


def test_plan_contains_no_framework_semantic_coordinates():
    entry = ProducerPlanBuilder().record_output(
        output_id=100,
        output_spec=TransportSpec("x"),
        output=HookOutput(torch.empty(1)),
    )
    forbidden = {
        "model_id",
        "layer_no",
        "dataset_id",
        "sample_id",
        "tp_rank",
        "dp_rank",
        "ep_rank",
        "pp_rank",
        "attempt_id",
        "invocation_id",
    }
    assert forbidden.isdisjoint(entry.__dataclass_fields__)


@pytest.mark.parametrize("dtype", [
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.complex64, torch.complex128,
])
def test_dtype_size_is_cached_without_caching_tensor_size(dtype, monkeypatch):
    entry = ProducerPlanBuilder().record_output(
        output_id=100, output_spec=TransportSpec("x"),
        output=HookOutput(torch.empty(3, dtype=dtype)),
    )
    expected = torch.empty((), dtype=dtype).element_size()
    _dtype_element_size.cache_clear()
    real_empty = torch.empty
    calls = []

    def counted_empty(*args, **kwargs):
        calls.append((args, kwargs))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counted_empty)
    for _ in range(5):
        assert entry.element_size == expected
    assert len(calls) == 1
    assert calls[0][1]["device"] == "cpu"
    changed = replace(entry, dtype=torch.float64)
    assert changed.element_size == 8
    assert len(calls) == (1 if dtype is torch.float64 else 2)


def test_explicit_alignment_does_not_require_native_backend(monkeypatch):
    import dmi.hooks.producer_plan as plans

    def forbidden():
        raise AssertionError("explicit arithmetic must not load native")

    monkeypatch.setattr(plans, "_payload_alignment", forbidden)
    assert _align_up(33, 16) == 48
    assert _align_up(0, 16) == 0


def test_runtime_sized_eager_entries_are_always_fresh_and_graphs_reject_them():
    spec = TransportSpec("ep_output", sizing_mode=OutputSizingMode.RUNTIME_SIZED)
    entries = []
    for count in (128, 128, 32, 128):
        output = HookOutput(torch.empty(count, 512, dtype=torch.bfloat16))
        entries.append(ProducerPlanEntry.from_output(
            output_id=100, output_spec=spec, output=output,
        ))
        with pytest.raises(ValueError, match="ep_output.*RUNTIME_SIZED.*eager"):
            ProducerPlanBuilder().record_output(output_id=100, output_spec=spec, output=output)
    assert len({id(entry) for entry in entries}) == 4
    assert [entry.aligned_reservation_bytes for entry in entries] == [
        131072, 131072, 32768, 131072]
    assert entries[0].input_shape == (128, 512)


def test_sizing_mode_requires_the_enum():
    with pytest.raises(TypeError, match="OutputSizingMode"):
        TransportSpec("bad", sizing_mode="runtime_sized")
