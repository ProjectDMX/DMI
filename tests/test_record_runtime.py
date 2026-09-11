"""RecordRuntime ordering and customization tests using a fake transport."""

from __future__ import annotations

import pytest
import torch

from dmi.adapters.base import StepReservation
from dmi.hooks.record import (
    HookOutput,
    HookPointV1,
    HookSpecV1,
    OutputStorage,
    TransportSpec,
    TransportType,
)
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
        self.null_offload = False
        self.pending_tasks = 0

    def _record_payload_tensor(self):
        return self.payload

    def configure_record_schema(self, schema):
        self.schema = schema

    def reserve_record(self, items):
        self.events.append(("reserve", tuple(items)))
        self.pending_tasks += len(items)
        return int(self.result)

    def push_record_descriptors(self, descriptors):
        self.events.append(("descriptors", tuple(descriptors)))

    def submit_record_cpu_direct(self, output, entry):
        self.events.append(("direct", output, entry))

    def flush_records_and_wait(self, timeout_s):
        self.events.append(("flush", timeout_s))
        if self.pending_tasks:
            raise TimeoutError("unmatched reserved record tasks")

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
    assert transport.events[0] == ("reserve", ((16, False),))
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
    assert transport.events[0] == ("reserve", ((16, False),))
    descriptor = transport.events[1][1][0]
    assert descriptor.rows[0][0] == "fresh"


def test_capture_disabled_skips_eager_and_replay_until_reenabled():
    runtime, transport, output, entry = _runtime_and_entry()
    plan = ProducerPlan((entry,))
    transport.null_offload = True

    assert runtime.emit_output(entry, "eager-disabled", output) is StepReservation.SKIPPED
    assert runtime.prepare_replay(plan, ("replay-disabled",)) is StepReservation.SKIPPED
    transport.flush_records_and_wait(0.25)
    assert transport.events == [("flush", 0.25)]

    transport.null_offload = False
    assert runtime.emit_output(entry, "enabled", output) is StepReservation.RESERVED
    assert [event[0] for event in transport.events] == [
        "flush",
        "reserve",
        "descriptors",
    ]


def test_replay_publishes_one_empty_descriptor_per_producer_occurrence():
    runtime, transport, _output, entry = _runtime_and_entry(
        record_format=_EmptyFormat()
    )

    result = runtime.prepare_replay(
        ProducerPlan((entry, entry)),
        ("first", "second"),
    )

    assert result is StepReservation.RESERVED
    assert transport.events[0] == (
        "reserve",
        ((16, False), (16, False)),
    )
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


def test_dynamic_producer_is_individually_marked_for_reclaim():
    transport = _Transport()
    runtime = RecordRuntime(transport, _Format())
    spec = TransportSpec(
        "out",
        transport_type=TransportType.CHUNKED,
        reservation_upper_bytes=64,
        output_shape=(-1,),
    )
    hook = HookPointV1(HookSpecV1("hook", (spec,)))
    runtime.bind_hook(hook, hook_runtime=_HookRuntime())
    output = HookOutput(
        torch.arange(4, dtype=torch.float32),
        (torch.tensor(16, dtype=torch.int64),),
    )
    entry = ProducerPlanBuilder().record_output(
        output_id=hook._output_ids[0],
        output_spec=spec,
        output=output,
    )

    runtime.emit_output(entry, "dynamic", output)

    assert transport.events[0] == ("reserve", ((64, True),))


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


def _tensor_column(name):
    return RecordColumn(
        name,
        RecordCellType.TENSOR,
        f"{name}_dtype",
        f"{name}_shape",
        f"{name}_bytes",
    )


class _CellFormat:
    """Emit one caller-supplied row against one caller-supplied layout.

    ``cells`` receives the producer plan entry so a test can pin a cell to
    the entry it is being validated against.  The first non-tensor column
    becomes the layout key, which keeps each cell test to the cells it
    actually exercises.
    """

    def __init__(self, columns, cells):
        columns = tuple(columns)
        key = next(
            column.name
            for column in columns
            if column.type is not RecordCellType.TENSOR
        )
        self.schema = RecordSchema(
            (
                RecordLayout(
                    "cell_rows",
                    "cell_rows",
                    columns,
                    primary_key=(key,),
                    order_by=(key,),
                ),
            )
        )
        self._cells = cells

    def encode(self, metadata, entry):
        return RecordDescriptor(
            "cell_rows",
            (tuple(self._cells(entry)),),
            output_id=entry.output_id,
        )


def test_two_open_payload_slices_are_refused():
    columns = (
        RecordColumn("tag", RecordCellType.STRING),
        _tensor_column("first"),
        _tensor_column("second"),
    )
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_CellFormat(
            columns,
            lambda entry: (
                "two-open",
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="at most one PayloadSlice may consume the remaining actual payload",
    ):
        runtime.emit_output(entry, "two-open", output)

    assert transport.events == []

    # One open slice plus one bounded slice is the accepted arrangement, so
    # the refusal above cannot be met by rejecting every multi-slice row.
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_CellFormat(
            columns,
            lambda entry: (
                "one-open",
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
                PayloadSlice(
                    offset_bytes=0,
                    nbytes=8,
                    dtype=entry.dtype,
                    shape=(-1,),
                ),
            ),
        )
    )

    assert runtime.emit_output(entry, "one-open", output) is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", ((16, False),))


def test_payload_slice_storage_must_match_producer_entry():
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_CellFormat(
            (RecordColumn("count", RecordCellType.INT64),),
            lambda entry: (
                PayloadSlice(
                    dtype=torch.int64,
                    storage=OutputStorage.SCALAR_INT,
                    nbytes=8,
                ),
            ),
        )
    )

    with pytest.raises(
        ValueError, match="storage does not match its producer plan entry"
    ):
        runtime.emit_output(entry, "scalar-int", output)

    assert transport.events == []


def test_payload_slice_dtype_drift_is_refused():
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_CellFormat(
            (RecordColumn("tag", RecordCellType.STRING), _tensor_column("payload")),
            lambda entry: (
                "drift",
                PayloadSlice(dtype=torch.float16, shape=entry.output_shape),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="PayloadSlice dtype changed: expected torch.float32, got torch.float16",
    ):
        runtime.emit_output(entry, "drift", output)

    assert transport.events == []


def test_bounded_payload_slice_may_not_exceed_the_reservation():
    columns = (
        RecordColumn("tag", RecordCellType.STRING),
        _tensor_column("payload"),
    )

    def _emit(offset_bytes, nbytes):
        runtime, transport, output, entry = _runtime_and_entry(
            record_format=_CellFormat(
                columns,
                lambda entry: (
                    "bounded",
                    PayloadSlice(
                        offset_bytes=offset_bytes,
                        nbytes=nbytes,
                        dtype=entry.dtype,
                        shape=(-1,),
                    ),
                ),
            )
        )
        assert entry.reservation_upper_bytes == 16
        return runtime, transport, output, entry

    runtime, transport, output, entry = _emit(8, 16)
    with pytest.raises(ValueError, match="exceeds producer reservation bound"):
        runtime.emit_output(entry, "bounded", output)
    assert transport.events == []

    runtime, transport, output, entry = _emit(17, None)
    with pytest.raises(ValueError, match="exceeds producer reservation bound"):
        runtime.emit_output(entry, "bounded", output)
    assert transport.events == []

    # The bound is inclusive: a slice covering the whole reservation exactly
    # must still reach the ring.
    runtime, transport, output, entry = _emit(0, 16)
    assert runtime.emit_output(entry, "bounded", output) is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", ((16, False),))


def _cell_runtime(column, value):
    """Bind one non-tensor column carrying ``value`` ahead of the payload."""

    return _runtime_and_entry(
        record_format=_CellFormat(
            (column, _tensor_column("payload")),
            lambda entry: (
                value,
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
            ),
        )
    )


def test_int32_cell_over_range_is_refused_before_reservation():
    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("n", RecordCellType.INT32), 2**31
    )

    with pytest.raises(TypeError, match="requires int32"):
        runtime.emit_output(entry, "over-range", output)

    assert transport.events == []


def test_int32_cell_rejects_bool():
    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("n", RecordCellType.INT32), True
    )

    with pytest.raises(TypeError, match="requires int32, got bool"):
        runtime.emit_output(entry, "bool", output)

    assert transport.events == []


def test_int64_cell_rejects_non_integer():
    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("n", RecordCellType.INT64), "not-an-int"
    )

    with pytest.raises(TypeError, match="requires int64, got str"):
        runtime.emit_output(entry, "str", output)

    assert transport.events == []


def test_float64_cell_rejects_non_numeric():
    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("value", RecordCellType.FLOAT64), "3.5"
    )

    with pytest.raises(TypeError, match="requires float64, got str"):
        runtime.emit_output(entry, "str", output)

    assert transport.events == []

    # An int, a float, and a scalar-float slice are all accepted float64
    # cells; the entry is a scalar-float producer so the slice matches it.
    transport = _Transport()
    runtime = RecordRuntime(
        transport,
        _CellFormat(
            (
                RecordColumn("whole", RecordCellType.FLOAT64),
                RecordColumn("fraction", RecordCellType.FLOAT64),
                RecordColumn("scalar", RecordCellType.FLOAT64),
            ),
            lambda entry: (
                1,
                2.5,
                PayloadSlice(
                    storage=OutputStorage.SCALAR_FLOAT,
                    dtype=torch.float64,
                    nbytes=8,
                ),
            ),
        ),
    )
    spec = TransportSpec("out", storage=OutputStorage.SCALAR_FLOAT)
    hook = HookPointV1(HookSpecV1("hook", (spec,)))
    runtime.bind_hook(hook, hook_runtime=_HookRuntime())
    output = HookOutput(torch.tensor([2.5], dtype=torch.float64))
    entry = ProducerPlanBuilder().record_output(
        output_id=hook._output_ids[0],
        output_spec=spec,
        output=output,
    )

    assert runtime.emit_output(entry, "numeric", output) is StepReservation.RESERVED
    assert transport.events[0][0] == "reserve"


def test_int64_array_cell_requires_a_tuple_of_in_range_ints():
    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("dims", RecordCellType.INT64_ARRAY), [1, 2]
    )

    with pytest.raises(TypeError, match="requires int64_array, got list"):
        runtime.emit_output(entry, "list", output)

    assert transport.events == []

    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("dims", RecordCellType.INT64_ARRAY), (1, 2**63)
    )

    with pytest.raises(TypeError, match="requires int64_array, got tuple"):
        runtime.emit_output(entry, "out-of-range", output)

    assert transport.events == []

    runtime, transport, output, entry = _cell_runtime(
        RecordColumn("dims", RecordCellType.INT64_ARRAY), (1, -2)
    )

    assert runtime.emit_output(entry, "in-range", output) is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", ((16, False),))


def test_tensor_column_requires_tensor_storage_slice():
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_CellFormat(
            (RecordColumn("tag", RecordCellType.STRING), _tensor_column("payload")),
            lambda entry: (
                "scalar-in-tensor-column",
                PayloadSlice(
                    storage=OutputStorage.SCALAR_INT,
                    dtype=torch.int64,
                    nbytes=8,
                ),
            ),
        )
    )

    with pytest.raises(TypeError, match="requires tensor"):
        runtime.emit_output(entry, "scalar-in-tensor-column", output)

    assert transport.events == []


class _FakeDeviceGate(torch.Tensor):
    """CPU stand-in for the single-element int32 CUDA gate bind_hook accepts."""

    @property
    def is_cuda(self) -> bool:
        return True

    @staticmethod
    def make():
        return torch.zeros(1, dtype=torch.int32).as_subclass(_FakeDeviceGate)


def test_device_gated_identity_output_is_reserved_with_reclaim():
    transport = _Transport()
    runtime = RecordRuntime(transport, _Format())
    hook = HookPointV1(HookSpecV1("hook", (TransportSpec("out"),)))
    runtime.bind_hook(
        hook,
        hook_runtime=_HookRuntime(),
        gate_tensor=_FakeDeviceGate.make(),
        gate_value=1,
    )
    output = HookOutput(torch.arange(4, dtype=torch.float32))
    entry = ProducerPlanBuilder().record_output(
        output_id=hook._output_ids[0],
        output_spec=hook.spec.outputs[0],
        output=output,
    )

    result = runtime.emit_output(entry, "batch-7", output)

    # The trailing flag is RecordReservationItem.needs_reclaim, and it is the
    # ring's only handling of a blocked gate: the gated IDENTITY kernel returns
    # before copying and before publishing, so unless the reservation asks for
    # it the drain thread never registers the pending task reclaim.  An
    # otherwise identical ungated output reserves (16, False) instead.
    assert result is StepReservation.RESERVED
    assert transport.events[0] == ("reserve", ((16, True),))


def test_replay_metadata_count_must_match_plan_entries():
    transport = _Transport()
    runtime = RecordRuntime(transport, _Format())
    hook = HookPointV1(HookSpecV1("hook", (TransportSpec("out"),)))
    runtime.bind_hook(hook, hook_runtime=_HookRuntime())
    output = HookOutput(torch.arange(4, dtype=torch.float32))
    builder = ProducerPlanBuilder()
    for _ in range(3):
        builder.record_output(
            output_id=hook._output_ids[0],
            output_spec=hook.spec.outputs[0],
            output=output,
        )
    plan = builder.build()

    # Short metadata must be refused rather than zipped: the body would
    # publish two descriptors while reserving three entries, breaking the
    # one-descriptor-per-reserved-record ordering contract.
    with pytest.raises(ValueError, match="expected 3, got 2"):
        runtime.prepare_replay(plan, ("a", "b"))

    assert transport.events == []


def test_producer_dtype_drift_is_refused_before_reservation():
    # A captured plan entry outlives the output it was derived from, so the
    # entry is re-checked against every output emitted through it.
    runtime, transport, _output, entry = _runtime_and_entry()
    drifted = HookOutput(torch.arange(4, dtype=torch.float64))

    with pytest.raises(ValueError, match="producer dtype changed"):
        runtime.emit_output(entry, "batch-7", drifted)

    assert transport.events == []


def test_producer_input_shape_drift_is_refused_before_reservation():
    runtime, transport, _output, entry = _runtime_and_entry()
    drifted = HookOutput(torch.arange(8, dtype=torch.float32))

    with pytest.raises(ValueError, match="producer input shape changed"):
        runtime.emit_output(entry, "batch-7", drifted)

    assert transport.events == []
