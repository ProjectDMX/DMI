"""The whole native capture chain on CPU: sink -> spool -> service -> reader.

The other live suites each start somewhere in the middle. The storage-service
suite stages its packs through the conformance_sink DRIVER, which feeds the
sink core JSON and base64 rather than tensors, and the adapter suite stops at
the spool. Nothing drove the real torch-facing NativePackSink into a real
catalog, so a disagreement between what the adapter stages and what the
service, the indexer and the reader expect had nowhere to show up.

Here the REAL sink (_dmi_native_sink, entered through its test-only
`attach` / `submit_envelope`, the same `submit` the ring calls) stages torch
CPU tensors into a spool, the in-process CaptureStorageService
(_dmi_native_store) uploads and indexes them into a live ClickHouse, and
NativeCaptureReader reads them back; the payload bytes must equal the tensors
that went in. No CUDA, no ring engine, no conformance driver, no Python on
the data path.

The object store is the in-repo signature-verifying fake S3. For the
multipart case -- a pack of 64 MiB or more crosses the client's multipart
threshold -- note that the fake enforces NOTHING about part sizes: it accepts
a part of any length, takes the part list on trust, and never checks the
ETags in the completion body. So the test checks the parts the client sent
itself against S3's rule: every part but the last is at least 5 MiB, or a
real store refuses CompleteMultipartUpload with EntityTooSmall. The real
object store DMI runs against, Garage, is exercised by the Garage suites.

Build: make -C native build/_dmi_native_sink build/_dmi_native_store \\
           PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from contextlib import contextmanager
from os import environ
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (
    ACCESS, BUCKET, REGION, SECRET, STATE, fake_s3,
)

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
SINK_BUILT = bool(sorted(BUILD.glob("_dmi_native_sink*.so")))
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))

# A per-test skipif rather than a module-level skip, and the sink is imported
# inside the tests rather than at module scope: this module is collected by
# the cpu job's live-glob walk too, which must be able to import it. In the
# live job both extensions are built, so this never fires there -- and if it
# did, that job's skip gate would fail it.
pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not (SINK_BUILT and STORE_BUILT),
        reason="the native sink and store modules are not built; run "
        "`make -C native build/_dmi_native_sink build/_dmi_native_store "
        "PYTHON=<venv>/bin/python`",
    ),
]

CLICKHOUSE_HOST = environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1")
CLICKHOUSE_HTTP_PORT = int(environ.get("DMI_CLICKHOUSE_HTTP_PORT", "8123"))
DATABASE = environ.get("DMI_CLICKHOUSE_DATABASE", "default")


LAYOUT = "capture_pack_reference_v1"
# at::ScalarType numeric values (c10/core/ScalarType.h -- stable ABI).
ATEN = {"float16": 5, "bfloat16": 15, "float32": 6, "int64": 4}

MiB = 1 << 20
# S3Config's defaults (native/csrc/store/s3_client.h), which the service
# does not override: a pack at or over the threshold goes up in parts.
MULTIPART_THRESHOLD = 64 * MiB
MULTIPART_CHUNK = 16 * MiB
# S3's floor for every part but the last.
S3_MIN_PART = 5 * MiB


def _native_sink():
    sys.path.insert(0, str(BUILD))
    try:
        import _dmi_native_sink
    finally:
        sys.path.remove(str(BUILD))
    return _dmi_native_sink


def _metadata(index: int, dtype: str, shape: tuple[int, ...]) -> dict:
    """The capture's metadata. The integer fields differ from one another
    across the records, so a parser that wires one field into another's
    slot reads back wrong rather than coincidentally right. producer_rank
    alone stays fixed: the sink seals a pack when the rank changes, and the
    pack counts below assume one rank."""
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=f"chain-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="resid_post", layer_number=index % 2,
        producer_rank=1, step_number=index, token_start=100 + index,
        token_end=102 + index, batch_position=index % 5, dtype=dtype,
        shape=shape,
        captured_at_ns=1_700_000_000_000_000_000 + index,
    ).to_mapping()


def _dtype_name(tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _raw(tensor):
    """The tensor's bytes as a flat uint8 tensor, in memory order."""
    import torch

    return tensor.contiguous().view(-1).view(torch.uint8)


class _Envelope:
    """One ring envelope: a flat payload and the rows that slice it."""

    def __init__(self):
        self.rows: list[dict] = []
        self.parts = []
        self.offset = 0
        self.expected: dict[str, object] = {}
        self.metadata: dict[str, dict] = {}

    def add(self, index: int, tensor):
        import torch

        # The sink refuses a slice that is not dtype-aligned, as the ring's
        # own layout never produces one: pad up to the element width.
        pad = -self.offset % tensor.element_size()
        if pad:
            self.parts.append(torch.zeros(pad, dtype=torch.uint8))
            self.offset += pad
        dtype = _dtype_name(tensor)
        length = tensor.numel() * tensor.element_size()
        metadata = _metadata(index, dtype, tuple(tensor.shape))
        self.rows.append({
            "metadata_json": json.dumps(metadata),
            "offset": self.offset, "length": length,
            "dtype": ATEN[dtype], "shape": list(tensor.shape),
        })
        self.parts.append(_raw(tensor))
        self.offset += length
        self.expected[metadata["capture_id"]] = tensor
        self.metadata[metadata["capture_id"]] = metadata

    def payload(self):
        import torch

        return torch.cat(self.parts)


def _open_sink(spool_root: Path, **overrides):
    config = dict(spool_root=str(spool_root), layout=LAYOUT,
                  # Only flush seals a pack: the multipart case must land as
                  # ONE pack, whatever the submission pace on a busy runner.
                  max_linger_ns=600 * 10**9)
    config.update(overrides)
    sink = _native_sink().NativePackSink(**config)
    return sink, sink.attach()


@contextmanager
def _catalog():
    """A fresh table prefix, dropped afterwards through the reference writer."""
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig, ClickHouseCatalogWriter,
    )

    client = clickhouse_driver.Client(
        host=CLICKHOUSE_HOST,
        port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000")))
    config = ClickHouseCatalogConfig(
        database=DATABASE, table_prefix=f"dmi_chain_{uuid.uuid4().hex}")
    try:
        yield config.table_prefix
    finally:
        ClickHouseCatalogWriter(client, config).drop_schema()


def _storage_config(endpoint, bucket, access, secret, prefix):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    return NativeCaptureStorageConfig(
        s3_endpoint=endpoint, s3_bucket=bucket, s3_region=REGION,
        s3_access_key=access, s3_secret_key=secret,
        s3_allow_insecure_http=True, clickhouse_host=CLICKHOUSE_HOST,
        clickhouse_port=CLICKHOUSE_HTTP_PORT, database=DATABASE,
        table_prefix=prefix, poll_interval_s=0.05)


def _run_chain(config, spool_root: Path, envelopes, *, sink_overrides=None):
    """Service first, as the engine orders it (its start sweeps the spool,
    which is safe only while no sink writes there), then the sink; flush
    both, and return the snapshots and what the reader reads back."""
    from dmi.storage.native_capture import (
        NativeCaptureReader, NativeCaptureStorage,
    )

    service = NativeCaptureStorage(config, spool_root=str(spool_root),
                                   spool_max_bytes=1 << 40, sweep_spool=True)
    service.start()
    try:
        sink, _lease = _open_sink(spool_root, **(sink_overrides or {}))
        for envelope in envelopes:
            sink.submit_envelope(LAYOUT, envelope.rows, envelope.payload())
        assert sink.flush_and_wait(120.0)
        sink.rethrow_if_failed()
        sink_snapshot = sink.snapshot()
        service.flush(120.0)
        service.rethrow_if_failed()
        service_snapshot = service.snapshot()
    finally:
        service.stop()

    reader = NativeCaptureReader(config)
    selection = reader.select(tenant_id="t")
    captures = {capture.descriptor["capture_id"]: capture
                for capture in reader.read(selection, byte_limit=1 << 30)}
    return sink_snapshot, service_snapshot, captures


def _catalog_form(name: str, value):
    """A submitted metadata value as NativeCaptureReader returns it: shape
    as a tuple, and the Nullable adapter_revision's NULL as the empty
    string (the native reader's sentinel for it; see parse_tsv_tuple)."""
    if name == "shape":
        return tuple(value)
    if name == "adapter_revision" and value is None:
        return ""
    return value


def _assert_read_back_exactly(captures, envelopes):
    """Every capture reads back with its bytes AND every metadata field it
    was submitted with. This chain is the only CI test that drives the torch
    sink's metadata parser (native/csrc/sink/record_row.cpp), so a field it
    mangles has to show up here."""
    expected, metadata = {}, {}
    for envelope in envelopes:
        expected.update(envelope.expected)
        metadata.update(envelope.metadata)
    assert sorted(captures) == sorted(expected)
    for capture_id, tensor in expected.items():
        capture = captures[capture_id]
        assert capture.payload == _raw(tensor).numpy().tobytes(), capture_id
        assert capture.descriptor["shape"] == tuple(tensor.shape)
        assert capture.descriptor["dtype"] == _dtype_name(tensor)
        # Every submitted field, not a chosen few: a field the reader stops
        # returning is a KeyError here rather than a silent pass.
        submitted = metadata[capture_id]
        read_back = {name: capture.descriptor[name] for name in submitted}
        assert read_back == {name: _catalog_form(name, value)
                             for name, value in submitted.items()}, capture_id


def _large_envelopes(count: int, elements: int):
    """`count` float32 captures of `elements` each, one envelope apiece."""
    import torch

    envelopes = []
    for index in range(count):
        envelope = _Envelope()
        generator = torch.Generator().manual_seed(index)
        envelope.add(index, torch.randn(elements, generator=generator))
        envelopes.append(envelope)
    return envelopes


def test_the_real_sink_reaches_the_catalog_and_reads_back_exactly(
        fake_s3, tmp_path):
    import torch

    # Several rows slicing one payload, as the ring presents them, in the
    # dtypes a model captures; three envelopes, so more than one pack.
    envelopes = []
    for batch in range(3):
        envelope = _Envelope()
        base = batch * 4
        envelope.add(base, torch.arange(6, dtype=torch.float16).reshape(2, 3)
                     + base)
        envelope.add(base + 1, torch.linspace(-1, 1, 8,
                                              dtype=torch.bfloat16))
        envelope.add(base + 2, torch.randn(4, 5, generator=torch.Generator()
                                           .manual_seed(batch)))
        envelope.add(base + 3, torch.arange(3, dtype=torch.int64) * base)
        envelopes.append(envelope)

    spool_root = tmp_path / "spool"
    with _catalog() as prefix:
        config = _storage_config(fake_s3, BUCKET, ACCESS, SECRET, prefix)
        sink_snapshot, snapshot, captures = _run_chain(
            config, spool_root, envelopes,
            sink_overrides={"max_pack_records": 5})

    assert sink_snapshot["persisted_records"] == 12, sink_snapshot
    assert sink_snapshot["dropped_records"] == 0, sink_snapshot
    assert sink_snapshot["failures"] == 0, sink_snapshot
    assert snapshot["uploaded_packs"] == 3, snapshot  # 5 + 5 + 2 records
    assert snapshot["indexed_packs"] == 3, snapshot
    assert snapshot["indexed_rows"] == 12, snapshot
    assert snapshot["pending_index"] == 0, snapshot
    assert list(spool_root.rglob("*.dmi-pack.ready")) == []
    _assert_read_back_exactly(captures, envelopes)
    for capture_id, capture in captures.items():
        expected = next(e.expected[capture_id] for e in envelopes
                        if capture_id in e.expected)
        assert torch.equal(capture.tensor(), expected), capture_id


def _one_large_pack(captures, expected_records):
    packs = {c.descriptor["object_key"]: c.descriptor for c in
             captures.values()}
    assert len(packs) == 1, sorted(packs)
    ((key, descriptor),) = packs.items()
    assert descriptor["pack_record_count"] == expected_records
    assert descriptor["object_bytes"] >= MULTIPART_THRESHOLD, descriptor
    return key, descriptor["object_bytes"]


# 17 captures of 4 MiB: one pack just over the 64 MiB threshold, so five
# parts -- four full 16 MiB chunks and a short last one.
LARGE_COUNT = 17
LARGE_ELEMENTS = MiB  # float32: 4 MiB each
LARGE_SINK = {"max_pack_bytes": 128 * MiB, "max_queue_bytes": 256 * MiB,
              "max_pack_records": 10_000}


def test_a_multipart_pack_through_the_fake_s3(fake_s3, tmp_path):
    envelopes = _large_envelopes(LARGE_COUNT, LARGE_ELEMENTS)
    with _catalog() as prefix:
        config = _storage_config(fake_s3, BUCKET, ACCESS, SECRET, prefix)
        sink_snapshot, snapshot, captures = _run_chain(
            config, tmp_path / "spool", envelopes, sink_overrides=LARGE_SINK)

    assert sink_snapshot["persisted_records"] == LARGE_COUNT, sink_snapshot
    assert sink_snapshot["dropped_records"] == 0, sink_snapshot
    assert snapshot["uploaded_packs"] == 1, snapshot
    assert snapshot["indexed_rows"] == LARGE_COUNT, snapshot
    key, object_bytes = _one_large_pack(captures, LARGE_COUNT)

    # The fake assembles whatever parts it is sent, so the S3 rules are
    # checked here, on what the client actually sent.
    with STATE.lock:
        calls = list(STATE.calls)
    # By part number, the last attempt at each winning: the log holds every
    # HTTP attempt, and a retried part would otherwise count twice.
    by_number = {}
    for call in calls:
        query = parse_qs(urlsplit(call["path"]).query)
        if call["method"] == "PUT" and "partNumber" in query:
            by_number[int(query["partNumber"][0])] = call["body_len"]
    assert sorted(by_number) == list(range(1, len(by_number) + 1)), by_number
    parts = [by_number[number] for number in sorted(by_number)]
    assert len(parts) == math.ceil(object_bytes / MULTIPART_CHUNK), parts
    assert sum(parts) == object_bytes
    assert all(size == MULTIPART_CHUNK for size in parts[:-1]), parts
    assert all(size >= S3_MIN_PART for size in parts[:-1]), parts
    assert STATE.objects[key]["etag"].endswith('-multipart"')
    _assert_read_back_exactly(captures, envelopes)
