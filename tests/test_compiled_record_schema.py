"""Configured transports compile schemas once and keep native rows owned."""

import gc
import json
from types import SimpleNamespace

import pytest
import torch

from dmi.records import (
    PayloadSlice, RecordCellType, RecordColumn, RecordDescriptor,
    RecordLayout, RecordSchema,
)
from dmi.transport import native
from dmi.transport.ring import RingTransport
from tests._requirements import require_cuda, require_native_backend


def _schema(name="values"):
    return RecordSchema((RecordLayout(
        name=name, table=name,
        columns=(RecordColumn("key", RecordCellType.INT32),
                 RecordColumn("tensor", RecordCellType.TENSOR,
                              dtype_column="dtype", shape_column="shape",
                              bytes_column="bytes")),
        primary_key=("key",), order_by=("key",),
    ),))


class _CountingSchema:
    def __init__(self, schema):
        self.schema = schema
        self.reads = 0
        self.index_granularity = schema.index_granularity

    @property
    def layouts(self):
        self.reads += 1
        assert self.reads == 1, "configured schema was converted again"
        return self.schema.layouts


@pytest.mark.cpu
def test_transport_keeps_independent_handles_and_configures_once(monkeypatch):
    compiled = []
    submissions = []

    def compile_schema(schema):
        handle = object()
        compiled.append((schema, handle))
        return handle

    monkeypatch.setattr(native, "_load_extension", lambda: SimpleNamespace(
        _compile_record_schema=compile_schema, PAYLOAD_ALIGN=16))
    transports = [object.__new__(RingTransport) for _ in range(2)]
    for i, transport in enumerate(transports):
        transport._ring_engine = SimpleNamespace(
            push_record_descriptors=lambda rows, handle: submissions.append((rows, handle)))
        transport.configure_record_schema(_schema(f"values_{i}"))
        for _ in range(10):
            transport.push_record_descriptors(())
        with pytest.raises(RuntimeError, match="already configured"):
            transport.configure_record_schema(_schema())
    assert len(compiled) == 2
    assert compiled[0][1] is not compiled[1][1]
    assert all(handle is compiled[0][1] for _, handle in submissions[:10])
    assert all(handle is compiled[1][1] for _, handle in submissions[10:])


@pytest.mark.native_backend
@require_native_backend()
def test_native_compilation_validates_and_owns_schema():
    backend = native._load_extension()
    source = _CountingSchema(_schema())
    handle = backend._compile_record_schema(source)
    assert isinstance(handle, backend._CompiledRecordSchema)
    assert source.reads == 1
    with pytest.raises(AttributeError):
        handle.schema = _schema("replacement")
    with pytest.raises(ValueError, match="index_granularity"):
        backend._compile_record_schema(SimpleNamespace(
            layouts=_schema().layouts, index_granularity=0))


@pytest.mark.native_backend
@require_native_backend()
def test_record_alignment_uses_native_constant():
    from dmi.hooks.producer_plan import _align_up, _payload_alignment

    assert native.PAYLOAD_ALIGN == native._load_extension().PAYLOAD_ALIGN
    assert _payload_alignment() == native.PAYLOAD_ALIGN
    for value in range(4 * native.PAYLOAD_ALIGN):
        expected = ((value + native.PAYLOAD_ALIGN - 1) // native.PAYLOAD_ALIGN
                    * native.PAYLOAD_ALIGN)
        assert _align_up(value) == expected


@pytest.mark.gpu
@require_cuda()
@require_native_backend()
@pytest.mark.parametrize("compiled", (False, True))
def test_native_submissions_keep_conversion_checks_and_owned_rows(tmp_path, compiled):
    backend = native._load_extension()
    semantic = _schema()
    source = _CountingSchema(semantic) if compiled else semantic
    schema = backend._compile_record_schema(source) if compiled else source
    sink = native.DropRecordSink(semantic, str(tmp_path), 0)
    config = native.RingConfig()
    config.task_ring_entries = 8
    config.payload_ring_bytes = 256
    config.pinned_staging_bytes = 256
    engine = native.RingEngine.create_record(config, sink)
    engine.init()
    engine.start()
    try:
        payload = torch.arange(128, dtype=torch.float32).view(torch.uint8)
        for key in range(3):
            descriptor = RecordDescriptor("values", ((key,
                PayloadSlice(dtype=torch.float32, shape=(128,))),))
            assert engine.reserve_record(((payload.nbytes, False),)) == 2
            engine.push_record_descriptors((descriptor,), schema)
            del descriptor
            gc.collect()
            engine.submit_record_cpu_direct(payload, payload.nbytes)
            assert engine.flush_records_and_wait(5000)

        with pytest.raises(ValueError, match="INT32.*out of range"):
            engine.push_record_descriptors((RecordDescriptor("values", ((1 << 40,
                PayloadSlice(dtype=torch.float32, shape=(128,))),)),), schema)
        with pytest.raises(ValueError, match="row width"):
            engine.push_record_descriptors((RecordDescriptor("values", ((1,),)),), schema)
        other = backend._compile_record_schema(_schema("other"))
        with pytest.raises((ValueError, RuntimeError), match="layout"):
            engine.push_record_descriptors((RecordDescriptor("values", ()),), other)
        if compiled:
            assert source.reads == 1
        del schema, source
        gc.collect()
    finally:
        engine.stop()
        sink.close()
    events = [json.loads(line) for line in
              (tmp_path / "rank_00000" / "events.jsonl").read_text().splitlines()]
    assert [row["key"] for event in events if event["type"] == "record"
            for row in event["rows"]] == [0, 1, 2]
