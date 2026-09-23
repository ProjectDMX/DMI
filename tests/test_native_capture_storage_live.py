"""The native capture storage service against a real catalog.

Packs are staged by the native sink core (the conformance_sink driver), then
the in-process C++ service uploads them to a signature-verifying fake S3 and
indexes them into a live ClickHouse catalog, and NativeCaptureReader reads
them back. The Python capture reader -- the reference -- reads the same
catalog and bucket, and must agree.

Beyond the round trip, the three things the service adds over the pieces it
composes:

* an index failure after a verified upload -- the pack already gone from the
  spool -- keeps the pack owed, so flush does not report success until it
  lands (the catalog is cut with a TCP switch in front of ClickHouse);
* a pack a crashed process uploaded but never indexed is reconciled at the
  next start, and a foreign object in the bucket is skipped, not indexed;
* one publisher per catalog: a second service is refused while the first
  holds the lease, and takes over once it stops.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import uuid
from contextlib import contextmanager
from os import environ
from pathlib import Path

import pytest

# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, fake_s3,
)

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
SINK_DRIVER = BUILD / "conformance_sink"
STORE_DRIVER = BUILD / "conformance_store"
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))

pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not (STORE_BUILT and SINK_DRIVER.exists() and STORE_DRIVER.exists()),
        reason="the native store module and drivers are not built; run "
        "`make -C native build/_dmi_native_store build/conformance_sink "
        "build/conformance_store PYTHON=<venv>/bin/python`",
    ),
]

CLICKHOUSE_HOST = environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1")
CLICKHOUSE_HTTP_PORT = int(environ.get("DMI_CLICKHOUSE_HTTP_PORT", "8123"))
DATABASE = environ.get("DMI_CLICKHOUSE_DATABASE", "default")


class _Driver:
    def __init__(self, binary: Path):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=60)


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
        database=DATABASE, table_prefix=f"dmi_svc_{uuid.uuid4().hex}")
    try:
        yield client, config
    finally:
        ClickHouseCatalogWriter(client, config).drop_schema()


def _storage_config(endpoint, prefix, **overrides):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    fields = dict(
        s3_endpoint=endpoint, s3_bucket=BUCKET, s3_region=REGION,
        s3_access_key=ACCESS, s3_secret_key=SECRET,
        s3_allow_insecure_http=True, clickhouse_host=CLICKHOUSE_HOST,
        clickhouse_port=CLICKHOUSE_HTTP_PORT, database=DATABASE,
        table_prefix=prefix, poll_interval_s=0.05,
    )
    fields.update(overrides)
    return NativeCaptureStorageConfig(**fields)


def _service(config, spool_root: Path, *, sweep_spool=True):
    from dmi.storage.native_capture import NativeCaptureStorage

    return NativeCaptureStorage(config, spool_root=str(spool_root),
                                spool_max_bytes=1 << 40,
                                sweep_spool=sweep_spool)


def _record(index: int):
    """One float16 capture, its payload a distinct run of values."""
    import torch
    from dmi.storage.capture import CaptureMetadata

    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3) + index
    metadata = CaptureMetadata(
        capture_id=f"svc-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="resid_post", layer_number=index % 2, producer_rank=0,
        step_number=index, token_start=index, token_end=index + 1,
        batch_position=0, dtype="float16", shape=(2, 3),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )
    return metadata, tensor


def _stage(spool_root: Path, indexes, *, records_per_pack: int = 2):
    """Stage records through the native sink core; return the tensors."""
    sink = _Driver(SINK_DRIVER)
    tensors = {}
    try:
        assert sink.call(
            op="open", root=str(spool_root), max_bytes=1 << 40,
            max_queue_records=256, max_queue_bytes=1 << 24,
            max_pack_bytes=8 << 20, max_pack_records=records_per_pack,
            max_linger_ns=1_000_000_000, overload="drop_newest",
            admission_timeout=-1)["ok"]
        for index in indexes:
            metadata, tensor = _record(index)
            response = sink.call(
                op="submit", metadata=metadata.to_mapping(),
                payload_b64=base64.b64encode(
                    tensor.numpy().tobytes()).decode())
            assert response["admission"] == "accepted", response
            tensors[metadata.capture_id] = tensor
        assert sink.call(op="flush", timeout=30)["ok"]
        snapshot = sink.call(op="close", timeout=30)["snapshot"]
        assert snapshot["persisted_records"] == len(tensors), snapshot
    finally:
        sink.close()
    return tensors


def _ready(spool_root: Path):
    return sorted(spool_root.rglob("*.dmi-pack.ready"))


def _reader(config):
    from dmi.storage.native_capture import NativeCaptureReader

    return NativeCaptureReader(config)


def _read_all(config, **filters):
    reader = _reader(config)
    selection = reader.select(tenant_id="t", **filters)
    return {capture.descriptor["capture_id"]: capture
            for capture in reader.read(selection, byte_limit=1 << 24)}


def test_staged_packs_reach_the_catalog_and_read_back_exactly(fake_s3, tmp_path):
    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(5))
    assert len(_ready(spool_root)) == 3  # 2 + 2 + 1 records per pack

    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        service = _service(config, spool_root)
        service.start()
        try:
            service.flush(30.0)
            snapshot = service.snapshot()
        finally:
            service.stop()

        assert _ready(spool_root) == []
        assert snapshot["uploaded_packs"] == 3, snapshot
        assert snapshot["indexed_packs"] == 3, snapshot
        assert snapshot["indexed_rows"] == 5, snapshot
        assert snapshot["pending_index"] == 0, snapshot

        captures = _read_all(config)
        assert sorted(captures) == sorted(tensors)
        for capture_id, tensor in tensors.items():
            capture = captures[capture_id]
            assert capture.payload == tensor.numpy().tobytes()
            assert capture.descriptor["dtype"] == "float16"
            assert capture.descriptor["shape"] == (2, 3)
            import torch

            assert torch.equal(capture.tensor(), tensor)

        # Filters reach the catalog: layer 1 is the odd indexes.
        page = _reader(config).search(tenant_id="t", layer_numbers=[1])
        assert [item["capture_id"] for item in page.items] == [
            "svc-0001", "svc-0003"]


def test_the_reference_reader_sees_the_same_captures(fake_s3, tmp_path):
    from dmi.storage.capture import CaptureReader
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog, ClickHouseReaderConfig,
    )
    from dmi.storage.capture.model import CaptureQuery
    from dmi.storage.capture.s3 import S3PackStore, S3StoreConfig

    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(4))
    with _catalog() as (client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        service = _service(config, spool_root)
        service.start()
        try:
            service.flush(30.0)
        finally:
            service.stop()

        native_reader = _reader(config)
        native_selection = native_reader.select(tenant_id="t")
        native = {capture.descriptor["capture_id"]: capture.payload
                  for capture in native_reader.read(
                      native_selection, byte_limit=1 << 24)}

        store = S3PackStore.from_config(S3StoreConfig(
            endpoint_url=fake_s3, bucket=BUCKET, region=REGION,
            access_key_id=ACCESS, secret_access_key=SECRET,
            store_id=config.store_id, allow_insecure_http=True))
        python_reader = CaptureReader(
            ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(catalog)),
            {config.store_id: store})
        python_selection = python_reader.select(CaptureQuery(tenant_id="t"))
        python = {item.descriptor.metadata.capture_id: item.payload
                  for item in python_reader.hydrate(
                      python_selection, byte_limit=1 << 24)}

        assert native_selection.selection_id == python_selection.selection_id
        assert native_selection.capture_ids == python_selection.capture_ids
        assert native == python
        assert set(native) == set(tensors)


class _Switch:
    """A TCP forwarder in front of ClickHouse's HTTP port that can be cut."""

    def __init__(self, host: str, port: int):
        self._target = (host, port)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._up = True
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            if not self._up:
                client.close()
                continue
            try:
                upstream = socket.create_connection(self._target)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._sockets |= {client, upstream}
            for source, sink in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(source, sink),
                                 daemon=True).start()

    @staticmethod
    def _pump(source, sink):
        try:
            while data := source.recv(65536):
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for end in (source, sink):
                try:
                    end.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def cut(self):
        self._up = False
        with self._lock:
            sockets, self._sockets = self._sockets, set()
        for end in sockets:
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def restore(self):
        self._up = True

    def close(self):
        self.cut()
        self._listener.close()


def test_an_index_failure_keeps_the_pack_owed_until_it_lands(fake_s3, tmp_path):
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix,
                                 clickhouse_port=switch.port)
        service = _service(config, spool_root)
        service.start()  # schema and lease through the switch, while up
        try:
            switch.cut()
            tensors = _stage(spool_root, range(2))

            # The upload succeeds and removes the pack from the spool; the
            # index cannot. An empty spool is not a drained service.
            with pytest.raises(TimeoutError, match="1 uploaded but unindexed"):
                service.flush(1.5)
            snapshot = service.snapshot()
            assert _ready(spool_root) == []
            assert snapshot["uploaded_packs"] == 1, snapshot
            assert snapshot["indexed_packs"] == 0, snapshot
            assert snapshot["pending_index"] == 1, snapshot
            assert snapshot["last_error"], snapshot

            switch.restore()
            service.flush(30.0)
            snapshot = service.snapshot()
            assert snapshot["indexed_packs"] == 1, snapshot
            assert snapshot["pending_index"] == 0, snapshot
        finally:
            service.stop()
            switch.close()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_a_pack_uploaded_but_never_indexed_is_reconciled_at_start(
        fake_s3, tmp_path):
    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(3), records_per_pack=3)

    # The crash window: the uploader verified the pack and removed it from
    # the spool, and the process died before indexing it.
    store = _Driver(STORE_DRIVER)
    try:
        uploaded = store.call(
            op="upload_pending", endpoint=fake_s3, bucket=BUCKET,
            region=REGION, access=ACCESS, secret=SECRET, token=None,
            insecure=True, connect_timeout=5, read_timeout=15, max_attempts=4,
            store_id="s3", root=str(spool_root), spool_max_bytes=1 << 40,
            limit=-1, max_workers=4, max_in_flight_bytes=1 << 30)
        assert uploaded["ok"], uploaded
        # And a foreign object where packs live, which must not be indexed.
        foreign = store.call(
            op="put", endpoint=fake_s3, bucket=BUCKET, region=REGION,
            access=ACCESS, secret=SECRET, token=None, insecure=True,
            connect_timeout=5, read_timeout=15, max_attempts=4,
            key="v1/tenant=t/date=2023-11-14/session=s/rank=0/"
                "not-a-pack.dmi-pack",
            data_b64=base64.b64encode(b"not a pack").decode(),
            content_type="application/octet-stream", metadata={})
        assert foreign["ok"], foreign
    finally:
        store.close()
    assert _ready(spool_root) == []

    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        service = _service(config, spool_root)
        service.start()
        try:
            service.flush(30.0)
            snapshot = service.snapshot()
        finally:
            service.stop()

        assert snapshot["uploaded_packs"] == 0, snapshot
        assert snapshot["reconciled_packs"] == 1, snapshot
        assert snapshot["reconcile_skipped_objects"] == 1, snapshot
        assert sorted(_read_all(config)) == sorted(tensors)


def test_one_publisher_per_catalog(fake_s3, tmp_path):
    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        first = _service(config, tmp_path / "first")
        second = _service(config, tmp_path / "second")
        first.start()
        try:
            with pytest.raises(RuntimeError, match="held"):
                second.start()
        finally:
            first.stop()
        # Stopping released the lease, so the next process takes over.
        second.start()
        second.stop()


def test_the_loop_backs_off_while_the_object_store_is_down(tmp_path):
    """Each cycle re-lists the spool, and listing re-hashes every pending
    pack, so an outage must not become a hashing loop inside the capture
    process: failed cycles double the loop's wait."""
    import time

    from dmi.storage.native_capture import _load_native_store_extension

    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
    spool_root = tmp_path / "spool"
    _stage(spool_root, range(1))
    with _catalog() as (_client, catalog):
        native = _storage_config(dead, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="backoff-test",
            poll_interval_ns=20_000_000, max_backoff_ns=10_000_000_000,
            reconcile_on_start=False, s3_max_attempts=1,
            uploader_max_attempts=1)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            time.sleep(3.0)
            snapshot = service.snapshot()
        finally:
            service.stop()

    assert snapshot["upload_failures"] >= 2, snapshot
    # 20 ms doubling reaches ~2.5 s of waits in 7 cycles; a fixed 20 ms
    # poll would have run ~150.
    assert snapshot["cycles"] <= 15, snapshot
    assert len(_ready(spool_root)) == 1  # still staged, not lost
