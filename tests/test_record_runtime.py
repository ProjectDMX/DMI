"""RecordRuntime ordering and customization tests using a fake transport."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dmi.adapters.base import StepReservation
from dmi.hooks import point as _hook_point, set_monitoring_debug
from dmi.hooks.record import (
    HookOutput,
    HookPointV1,
    HookSpecV1,
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


@pytest.fixture
def monitoring_debug(monkeypatch):
    monkeypatch.setattr(_hook_point, "_MONITORING_DEBUG", False)
    return set_monitoring_debug


class _Transport:
    def __init__(self, result=StepReservation.RESERVED):
        self.result = result
        self.d2h_pattern_result = True
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

    def define_d2h_window_pattern(self, **kwargs):
        self.events.append(("define_d2h_window_pattern", kwargs))
        return self.d2h_pattern_result

    def advance_boundary(self):
        self.events.append(("advance_boundary",))

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


def test_reclaim_dtype_size_does_not_allocate_after_warmup(monkeypatch):
    runtime = RecordRuntime(_Transport(), _Format())
    entry = ProducerPlanBuilder().record_output(
        output_id=1 << 16, output_spec=TransportSpec("x"),
        output=HookOutput(torch.empty(3, dtype=torch.float32)),
    )
    assert not runtime._needs_reclaim(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("reclaim allocated a scalar tensor after warmup")

    monkeypatch.setattr(torch, "empty", forbidden)
    for count in (3, 16, 2):
        assert not runtime._needs_reclaim(replace(
            entry, input_shape=(count,), output_shape=(count,),
            reservation_upper_bytes=count * entry.element_size))


def test_resolved_packed_reservations_do_not_mutate_the_entry(monitoring_debug):
    monitoring_debug(False)
    transport = _Transport()
    runtime = RecordRuntime(transport, _Format())
    output = HookOutput(torch.empty(1024, 2, 512, dtype=torch.bfloat16))
    entry = ProducerPlanBuilder().record_output(
        output_id=1 << 16,
        output_spec=TransportSpec("packed", transport_type=TransportType.SEQ_PREFIX_PACK,
                                  feature_bytes=1024, output_shape=(-1, 512)),
        output=output,
    )
    original = entry.reservation_upper_bytes
    for size in (393216, 786432, 0):
        runtime._emit_prepared_output(entry, str(size), output, reservation_bytes=size)
    assert [event[1] for event in transport.events if event[0] == "reserve"] == [
        ((393216, True),), ((786432, True),), ((0, True),)]
    assert entry.reservation_upper_bytes == original == 2097152
    assert [event[1][0].rows[0][0] for event in transport.events
            if event[0] == "descriptors"] == ["393216", "786432", "0"]
    with pytest.raises(ValueError, match="exceeds producer"):
        runtime._emit_prepared_output(entry, "bad", output, reservation_bytes=original + 16)
    with pytest.raises(ValueError, match="aligned"):
        runtime._emit_prepared_output(entry, "bad", output, reservation_bytes=1)


def test_only_owned_prepared_entry_skips_duplicate_validation(monkeypatch, monitoring_debug):
    runtime = RecordRuntime(_Transport(), _Format())
    output = HookOutput(torch.ones(4))
    entry = ProducerPlanBuilder().record_output(
        output_id=1 << 16, output_spec=TransportSpec("x"), output=output)

    def forbidden(*args):
        raise AssertionError("entry check")

    monkeypatch.setattr(runtime, "_validate_entry_output", forbidden)
    monitoring_debug(False)
    runtime._emit_prepared_output(entry, "ok", output, reservation_bytes=16)
    with pytest.raises(AssertionError, match="entry check"):
        runtime.emit_output(entry, "public", output)
    monitoring_debug(True)
    with pytest.raises(AssertionError, match="entry check"):
        runtime._emit_prepared_output(entry, "debug", output, reservation_bytes=16)


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


def test_nonempty_producer_descriptor_requires_payload_slice(monitoring_debug):
    monitoring_debug(True)
    runtime, transport, output, entry = _runtime_and_entry(
        record_format=_PayloadFreeProducerFormat()
    )

    with pytest.raises(ValueError, match="must contain a PayloadSlice"):
        runtime.emit_output(entry, "batch-7", output)

    assert transport.events == []


@pytest.mark.parametrize("replay", [False, True])
def test_descriptor_checks_are_skipped_without_debug(
    monitoring_debug, monkeypatch, replay,
):
    runtime, transport, output, entry = _runtime_and_entry()

    def unexpected_validation(*args, **kwargs):
        pytest.fail("descriptor validation ran with monitoring debug disabled")

    monkeypatch.setattr(runtime, "_validate_descriptor", unexpected_validation)
    monkeypatch.setattr(runtime, "_validate_payload_slices", unexpected_validation)
    monkeypatch.setattr(RecordDescriptor, "has_payload", property(unexpected_validation))

    if replay:
        result = runtime.prepare_replay(ProducerPlan((entry,)), ("batch-7",))
    else:
        result = runtime.emit_output(entry, "batch-7", output)

    assert result is StepReservation.RESERVED
    assert transport.events == [
        ("reserve", ((16, False),)),
        ("descriptors", (_Format().encode("batch-7", entry),)),
    ]


@pytest.mark.parametrize("replay", [False, True])
def test_debug_toggle_preserves_valid_descriptors_and_reservations(monitoring_debug, replay):
    runtime, transport, output, entry = _runtime_and_entry()
    events = []
    for enabled in (False, True, False):
        monitoring_debug(enabled)
        transport.events.clear()
        if replay:
            runtime.prepare_replay(ProducerPlan((entry,)), ("batch-7",))
        else:
            runtime.emit_output(entry, "batch-7", output)
        events.append(list(transport.events))
    assert events[0] == events[1] == events[2]


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("override, error, match", [
    ({"output_id": 0}, ValueError, "output_id does not match"),
    ({"rows": (("tag",),)}, ValueError, "requires 2"),
    ({"rows": ((0, PayloadSlice(dtype=torch.float32)),)}, TypeError, "requires string"),
    ({"rows": (("tag", PayloadSlice(dtype=torch.float64)),)}, ValueError, "dtype changed"),
    ({"rows": (("tag", PayloadSlice(dtype=torch.float32, nbytes=32)),)},
     ValueError, "exceeds producer reservation bound"),
])
def test_debug_rejects_invalid_descriptors_before_reservation(
    monitoring_debug, monkeypatch, replay, override, error, match,
):
    runtime, transport, output, entry = _runtime_and_entry()
    descriptor = replace(_Format().encode("batch-7", entry), **override)
    monkeypatch.setattr(runtime._format, "encode", lambda metadata, plan: descriptor)
    monitoring_debug(True)

    with pytest.raises(error, match=match):
        if replay:
            runtime.prepare_replay(ProducerPlan((entry,)), ("batch-7",))
        else:
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


def test_d2h_window_operations_delegate_without_exposing_transport_state():
    runtime, transport, _output, _entry = _runtime_and_entry()

    accepted = runtime.define_d2h_window_pattern(
        period=12,
        windows=((1, 3), (8, 10)),
        initial_counter=7,
    )
    runtime.advance_boundary()

    assert accepted is True
    assert transport.events == [
        (
            "define_d2h_window_pattern",
            {
                "period": 12,
                "windows": ((1, 3), (8, 10)),
                "initial_counter": 7,
            },
        ),
        ("advance_boundary",),
    ]


def test_d2h_window_definition_returns_terminal_fallback_state():
    runtime, transport, _output, _entry = _runtime_and_entry()
    transport.d2h_pattern_result = False

    accepted = runtime.define_d2h_window_pattern(
        period=12,
        windows=((1, 3), (8, 10)),
        initial_counter=7,
    )

    assert accepted is False


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
