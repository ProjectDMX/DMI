"""CPU contract tests for the additive record-hook path."""

from __future__ import annotations

from types import MethodType

import pytest
import torch

from dmi.adapters.base import StepReservation
from dmi.hooks.record import (
    HookOutput,
    HookPointV1,
    HookSpecV1,
    TransportSpec,
)

pytestmark = pytest.mark.cpu


class _Runtime:
    def __init__(self, *, eligible=True, reservation=None):
        self.eligible = eligible
        self.reservation = reservation
        self.prepared = []

    def should_emit(self, hook):
        return self.eligible

    def prepare_output(self, **kwargs):
        self.prepared.append(kwargs)
        return self.reservation


def _bind_for_cpu(hook, runtime):
    hook._output_ids = tuple(range(100, 100 + len(hook.spec.outputs)))
    hook._ring_payload = torch.empty(0, dtype=torch.uint8)
    hook._hook_runtime = runtime
    dispatched = []
    hook._dispatch = MethodType(
        lambda self, spec, output: dispatched.append((spec, output)), hook
    )
    return dispatched


def test_ineligible_hook_skips_preprocessing_and_outputs():
    calls = []
    spec = HookSpecV1(
        "hidden",
        (TransportSpec("summary"),),
        preprocess=lambda x: calls.append(x) or x,
    )
    hook = HookPointV1(spec)
    runtime = _Runtime(eligible=False)
    dispatched = _bind_for_cpu(hook, runtime)

    assert hook(torch.ones(2)) is None
    assert calls == []
    assert runtime.prepared == []
    assert dispatched == []


def test_preprocessing_outputs_preserve_declared_order():
    spec = HookSpecV1(
        "split",
        (TransportSpec("left"), TransportSpec("right")),
        preprocess=lambda x: (x + 1, x + 2),
    )
    hook = HookPointV1(spec)
    runtime = _Runtime()
    dispatched = _bind_for_cpu(hook, runtime)

    hook(torch.tensor([3.0]))

    assert [item["output_id"] for item in runtime.prepared] == [100, 101]
    assert [item["output_index"] for item in runtime.prepared] == [0, 1]
    assert [output.tensor.item() for _, output in dispatched] == [4.0, 5.0]


def test_reserved_output_dispatches():
    hook = HookPointV1(HookSpecV1("reserved", (TransportSpec("value"),)))
    runtime = _Runtime(reservation=StepReservation.RESERVED)
    dispatched = _bind_for_cpu(hook, runtime)

    hook(torch.ones(1))

    assert len(runtime.prepared) == 1
    assert len(dispatched) == 1


def test_oversized_output_skips_producer_dispatch():
    hook = HookPointV1(HookSpecV1("oversized", (TransportSpec("value"),)))
    runtime = _Runtime(reservation=StepReservation.OVERSIZED)
    dispatched = _bind_for_cpu(hook, runtime)

    hook(torch.ones(1))

    assert len(runtime.prepared) == 1
    assert dispatched == []


def test_no_preprocessing_requires_one_input_per_output():
    hook = HookPointV1(HookSpecV1("pair", (TransportSpec("a"),)))
    _bind_for_cpu(hook, _Runtime())

    with pytest.raises(ValueError, match="one input per declared output"):
        hook(torch.ones(1), torch.ones(1))


def test_hook_output_preserves_device_side_producer_metadata():
    meta = torch.tensor([2], dtype=torch.int64)
    output = HookOutput(torch.ones(4), (meta,))
    hook = HookPointV1(HookSpecV1("packed", (TransportSpec("value"),)))
    runtime = _Runtime()
    dispatched = _bind_for_cpu(hook, runtime)

    hook(output)

    assert runtime.prepared[0]["output"] is output
    assert dispatched[0][1].producer_meta == (meta,)


def test_wrong_preprocess_arity_fails_before_dispatch():
    hook = HookPointV1(
        HookSpecV1(
            "wrong",
            (TransportSpec("a"), TransportSpec("b")),
            preprocess=lambda x: (x,),
        )
    )
    dispatched = _bind_for_cpu(hook, _Runtime())

    with pytest.raises(ValueError, match="expected 2 outputs"):
        hook(torch.ones(1))
    assert dispatched == []


def test_record_gate_requires_exact_int32_contract():
    hook = HookPointV1(HookSpecV1("gated", (TransportSpec("value"),)))
    runtime = _Runtime()
    ring_payload = torch.empty(0, dtype=torch.uint8)

    with pytest.raises(TypeError, match="must use int32"):
        hook._bind_record_runtime(
            output_ids=(100,),
            ring_payload=ring_payload,
            hook_runtime=runtime,
            gate_tensor=torch.ones((), dtype=torch.int64),
            gate_value=1,
        )
