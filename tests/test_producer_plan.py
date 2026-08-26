"""Tests for ordered, framework-neutral producer plans."""

from __future__ import annotations

import pytest
import torch

from dmi.hooks.dynamic import HookOutput, TransportSpec, TransportType
from dmi.hooks.producer_plan import ProducerPlanBuilder

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
