"""RecordRuntime ordering and customization tests using a fake transport."""

from __future__ import annotations

import pytest
import torch

from dmi.adapters.base import StepReservation
from dmi.hooks.dynamic import HookOutput, HookPointV1, HookSpecV1, TransportSpec
from dmi.hooks.producer_plan import ProducerPlan, ProducerPlanBuilder
from dmi.records import (
    PayloadSlice,
    RecordCellType,
    RecordColumn,
    RecordDescriptor,
    RecordLayout,
    RecordRuntime,
    RecordSchema,
)

pytestmark = pytest.mark.cpu


class _Transport:
    def __init__(self, result=StepReservation.RESERVED):
        self.result = result
        self.events = []
        self.payload = torch.empty(0, dtype=torch.uint8)

    def _record_payload_tensor(self):
        return self.payload

    def configure_record_schema(self, schema):
        self.schema = schema

    def reserve_record(self, nbytes, tasks):
        self.events.append(("reserve", nbytes, tasks))
        return int(self.result)

    def push_record_descriptors(self, descriptors):
        self.events.append(("descriptors", tuple(descriptors)))

    def submit_record_cpu_direct(self, output, entry):
        self.events.append(("direct", output, entry))

class _Format:
    def __init__(self, layout_name="tensor_rows"):
        self.schema = RecordSchema(
            (
                RecordLayout(
                    layout_name,
                    layout_name,
                    (
                        RecordColumn("tag", RecordCellType.STRING),
                        RecordColumn(
                            "payload",
                            RecordCellType.TENSOR,
                            "payload_dtype",
                            "payload_shape",
                            "payload_bytes",
                        ),
                    ),
                    primary_key=("tag",),
                    order_by=("tag",),
                ),
            )
        )

    def encode(self, metadata, entry):
        return RecordDescriptor(
            "tensor_rows",
            ((str(metadata), PayloadSlice(dtype=entry.dtype, shape=entry.output_shape)),),
            output_id=entry.output_id,
        )


class _EmptyFormat(_Format):
    def encode(self, metadata, entry):
        return RecordDescriptor(
            "tensor_rows",
            (),
            output_id=entry.output_id,
        )


class _PayloadFreeProducerFormat:
    schema = RecordSchema(
        (
            RecordLayout(
                "metadata_rows",
                "metadata_rows",
                (RecordColumn("tag", RecordCellType.STRING),),
                primary_key=("tag",),
                order_by=("tag",),
            ),
        )
    )

    def encode(self, metadata, entry):
        return RecordDescriptor(
            "metadata_rows",
            ((str(metadata),),),
            output_id=entry.output_id,
        )


class _HookRuntime:
    def should_emit(self, hook):
        return True

    def prepare_output(self, **kwargs):
        return None


def _runtime_and_entry(
    *,
    reservation=StepReservation.RESERVED,
    record_format=None,
):
    transport = _Transport(reservation)
    runtime = RecordRuntime(transport, record_format or _Format())
    hook = HookPointV1(HookSpecV1("hook", (TransportSpec("out"),)))
    runtime.bind_hook(hook, hook_runtime=_HookRuntime())
    output = HookOutput(torch.arange(4, dtype=torch.float32))
    entry = ProducerPlanBuilder().record_output(
        output_id=hook._output_ids[0],
        output_spec=hook.spec.outputs[0],
        output=output,
    )
    return runtime, transport, output, entry


def test_eager_descriptor_is_published_after_reservation_before_producer():
    runtime, transport, output, entry = _runtime_and_entry()

    result = runtime.emit_output(entry, "batch-7", output)

    assert result is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", 16, 1)
    assert transport.events[1][0] == "descriptors"
    assert len(transport.events) == 2


def test_oversized_eager_output_publishes_then_submits_directly():
    runtime, transport, output, entry = _runtime_and_entry(
        reservation=StepReservation.OVERSIZED
    )

    result = runtime.emit_output(entry, "batch-7", output)

    assert result is StepReservation.OVERSIZED
    assert [event[0] for event in transport.events] == [
        "reserve",
        "descriptors",
        "direct",
    ]


def test_replay_encodes_complete_batch_before_descriptor_publication():
    runtime, transport, output, entry = _runtime_and_entry()
    result = runtime.prepare_replay(ProducerPlan((entry,)), ("fresh",))

    assert result is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", 16, 1)
    descriptor = transport.events[1][1][0]
    assert descriptor.rows[0][0] == "fresh"


def test_replay_publishes_one_empty_descriptor_per_producer_occurrence():
    runtime, transport, _output, entry = _runtime_and_entry(
        record_format=_EmptyFormat()
    )

    result = runtime.prepare_replay(
        ProducerPlan((entry, entry)),
        ("first", "second"),
    )

    assert result is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", 32, 2)
    descriptors = transport.events[1][1]
    assert len(descriptors) == 2
    assert all(descriptor.rows == () for descriptor in descriptors)


def test_nonempty_producer_descriptor_requires_payload_slice():
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_PayloadFreeProducerFormat()
    )

    with pytest.raises(ValueError, match="must contain a PayloadSlice"):
        runtime.emit_output(entry, "batch-7", output)

    assert transport.events == []


def test_two_independent_formats_do_not_share_schema_or_output_registry():
    class _OtherFormat:
        def __init__(self):
            self.schema = RecordSchema(
                (
                    RecordLayout(
                        "other_rows",
                        "other_rows",
                        (
                            RecordColumn("value", RecordCellType.INT64),
                            RecordColumn(
                                "data",
                                RecordCellType.TENSOR,
                                "data_dtype",
                                "data_shape",
                                "data_bytes",
                            ),
                        ),
                        primary_key=("value",),
                        order_by=("value",),
                    ),
                )
            )

        def encode(self, metadata, entry):
            return RecordDescriptor(
                "other_rows",
                ((
                    int(metadata),
                    PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
                ),),
                output_id=entry.output_id,
            )

    transport_a = _Transport()
    transport_b = _Transport()
    format_a = _Format()
    format_b = _OtherFormat()
    runtime_a = RecordRuntime(transport_a, format_a)
    runtime_b = RecordRuntime(transport_b, format_b)
    hook_a = HookPointV1(HookSpecV1("a", (TransportSpec("out"),)))
    hook_b = HookPointV1(HookSpecV1("b", (TransportSpec("out"),)))

    runtime_a.bind_hook(hook_a, hook_runtime=_HookRuntime())
    runtime_b.bind_hook(hook_b, hook_runtime=_HookRuntime())

    output = HookOutput(torch.ones(1))
    entry_a = ProducerPlanBuilder().record_output(
        output_id=hook_a._output_ids[0],
        output_spec=hook_a.spec.outputs[0],
        output=output,
    )
    entry_b = ProducerPlanBuilder().record_output(
        output_id=hook_b._output_ids[0],
        output_spec=hook_b.spec.outputs[0],
        output=output,
    )
    runtime_a.emit_output(entry_a, "one", output)
    runtime_b.emit_output(entry_b, 2, output)

    assert hook_a._output_ids == hook_b._output_ids == (1 << 16,)
    assert transport_a.schema.layouts[0].name == "tensor_rows"
    assert transport_b.schema.layouts[0].name == "other_rows"
    assert transport_a.schema is format_a.schema
    assert transport_b.schema is format_b.schema


def test_binding_reuses_private_output_id_for_same_declared_name():
    transport = _Transport()
    runtime = RecordRuntime(transport, _Format())
    first = HookPointV1(HookSpecV1("first", (TransportSpec("shared"),)))
    second = HookPointV1(HookSpecV1("second", (TransportSpec("shared"),)))

    runtime.bind_hook(first, hook_runtime=_HookRuntime())
    runtime.bind_hook(second, hook_runtime=_HookRuntime())

    assert first._output_ids == second._output_ids == (1 << 16,)
