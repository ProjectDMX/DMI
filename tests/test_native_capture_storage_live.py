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
import re
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from os import environ
from pathlib import Path

import pytest

# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, STATE, fake_s3, fake_s3_tls, private_ca,
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

        # The binding refuses bad limits itself, not only through the
        # Python wrapper: a negative byte_limit must never pass as unsigned.
        reader = _reader(config)
        native = reader.select(tenant_id="t")._native_dict()
        with pytest.raises(ValueError, match="byte_limit"):
            reader._reader.hydrate(native, -1, 1024)
        with pytest.raises(ValueError, match="request_limit"):
            reader._reader.hydrate(native, 1 << 24, 0)


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
    """A TCP forwarder in front of an HTTP server -- ClickHouse's HTTP port,
    or the fake S3 -- that can be cut (connections refused), stalled
    (connections accepted, never answered), made to hold lease INSERTs back
    and deliver them late, to stall only the requests a predicate picks, to
    hold the first request a predicate picks back for a while, or to hold
    every request back by a delay a function picks. Every native client
    opens one connection per request, so one request is one connection
    here."""

    def __init__(self, host: str, port: int):
        self._target = (host, port)
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._up = True
        self._stalled = False
        self._late_by = 0.0  # seconds a lease INSERT is held back; 0 = never
        # A predicate over one whole request (head and body): matching
        # requests are held open and never answered.
        self._stall_if = None
        # (predicate, seconds): the first matching request reaches the server
        # only that much later; its answer is relayed if anyone still listens.
        self._slow_once = None
        # A function of one whole request: the seconds it reaches the
        # server late (0 for at once); its answer is relayed as usual.
        self._delay_by = None
        # time.monotonic() of every request stall_requests() held.
        self.stalled: list[float] = []
        # time.monotonic() of every connection refused while cut.
        self.refused: list[float] = []
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        threading.Thread(target=self._accept, daemon=True).start()

    @classmethod
    def to_url(cls, url: str) -> "_Switch":
        host, port = url.split("://", 1)[1].rstrip("/").rsplit(":", 1)
        return cls(host, int(port))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _accept(self):
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            if self._stalled:
                with self._lock:
                    self._sockets.add(client)  # held open, never read
                continue
            if not self._up:
                self.refused.append(time.monotonic())
                client.close()
                continue
            if (self._late_by > 0 or self._stall_if is not None
                    or self._slow_once is not None
                    or self._delay_by is not None):
                threading.Thread(target=self._look_then_route,
                                 args=(client,), daemon=True).start()
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

    def _look_then_route(self, client):
        """Read one HTTP request and route it: a lease INSERT held back by
        deliver_lease_inserts_late() reaches the server only after that delay
        -- long after the client gave up on it -- and nobody hears the
        answer; a request stall_requests() picks is never answered; the first
        request slow_once() picks is forwarded late, and each request by
        what delay_requests() says; anything else goes straight through."""
        request = b""
        continued = False
        try:
            client.settimeout(2.0)
            while True:
                head, found, body = request.partition(b"\r\n\r\n")
                if found:
                    length = re.search(rb"(?i)content-length:\s*(\d+)", head)
                    if length is None or len(body) >= int(length.group(1)):
                        break
                    # libcurl holds a body over 1 KiB back until the server
                    # says 100 Continue, or for a second; answer for it, so
                    # routing a large statement does not add that second.
                    if not continued and re.search(
                            rb"(?i)\r\nexpect:\s*100-continue", head):
                        client.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
                        continued = True
                chunk = client.recv(65536)
                if not chunk:
                    break
                request += chunk
            client.settimeout(None)
        except OSError:
            client.close()
            return
        if continued:
            # The server must not answer 100 Continue a second time.
            head, _, body = request.partition(b"\r\n\r\n")
            request = (re.sub(rb"(?i)\r\nexpect:[^\r\n]*", b"", head)
                       + b"\r\n\r\n" + body)
        stall_if = self._stall_if
        if stall_if is not None and stall_if(request):
            with self._lock:
                self.stalled.append(time.monotonic())
                self._sockets.add(client)  # held open, never answered
            return
        slow = self._slow_once
        if slow is not None and slow[0](request):
            self._slow_once = None
            time.sleep(slow[1])
        delay_by = self._delay_by
        if delay_by is not None and (delay := delay_by(request)) > 0:
            time.sleep(delay)
        late_by = self._late_by
        late = (late_by > 0 and b"INSERT INTO" in request
                and b"_publisher_lease" in request)
        if late:
            time.sleep(late_by)
        try:
            upstream = socket.create_connection(self._target)
            upstream.sendall(request)
        except OSError:
            client.close()
            return
        if late:
            # Nobody is listening any more; the server runs it regardless.
            client.close()
            upstream.settimeout(10.0)
            try:
                while upstream.recv(65536):
                    pass
            except OSError:
                pass
            upstream.close()
            return
        with self._lock:
            self._sockets |= {client, upstream}
        for source, sink in ((client, upstream), (upstream, client)):
            threading.Thread(target=self._pump, args=(source, sink),
                             daemon=True).start()

    def deliver_lease_inserts_late(self, late_by: float):
        self._late_by = late_by

    def stall_requests(self, predicate):
        """Hold every new request `predicate(request_bytes)` picks open and
        never answer it; the rest go through."""
        self._stall_if = predicate

    def slow_once(self, predicate, seconds: float):
        """Forward the first request `predicate` picks `seconds` late."""
        self._slow_once = (predicate, seconds)

    def delay_requests(self, delay_by):
        """Forward every request `delay_by(request_bytes)` seconds late."""
        self._delay_by = delay_by

    def cut(self):
        self._up = False
        with self._lock:
            sockets, self._sockets = self._sockets, set()
        for end in sockets:
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def stall(self):
        """Accept new connections and never answer them. Live ones are
        dropped, so a keep-alive client has to reconnect into the stall."""
        self.cut()
        self._up = True
        self._stalled = True

    def restore(self):
        self._up = True
        self._stalled = False
        self._late_by = 0.0
        self._stall_if = None
        self._slow_once = None
        self._delay_by = None

    def close(self):
        """Refuse new connections and drop the live ones. Closing the
        listener does not wake a thread blocked in accept(), which still
        takes one more queued connection: the stall settings go first, so
        that connection is refused rather than held open for its client's
        whole timeout (a stop()'s lease release after a stall() did)."""
        self._stalled = False
        self._stall_if = None
        self._slow_once = None
        self._delay_by = None
        self._late_by = 0.0
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


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_no_new_upload_while_an_uploaded_pack_is_owed(fake_s3, tmp_path):
    """An uploaded pack leaves the durable spool for an in-memory retry
    list. With the catalog down, uploading kept moving packs there, and a
    process that exited then lost them to everything but a reconcile --
    which reconcile_on_start=False, the shared-bucket setting, never runs.
    New packs now stay in the spool until the owed index lands."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix,
                                 clickhouse_port=switch.port,
                                 reconcile_on_start=False)
        service = _service(config, spool_root)
        service.start()
        try:
            switch.cut()
            tensors = _stage(spool_root, range(2))
            _wait_for(lambda: service.snapshot()["pending_index"] == 1)
            assert _ready(spool_root) == []

            tensors.update(_stage(spool_root, range(2, 4)))
            with pytest.raises(TimeoutError, match="1 uploaded but unindexed"):
                service.flush(1.0)  # cycles, each retrying the owed index
            snapshot = service.snapshot()
            assert len(_ready(spool_root)) == 1, snapshot
            assert snapshot["uploaded_packs"] == 1, snapshot
            assert snapshot["pending_index"] == 1, snapshot

            switch.restore()
            service.flush(30.0)
            snapshot = service.snapshot()
        finally:
            service.stop()
            switch.close()

        assert _ready(spool_root) == []
        assert snapshot["uploaded_packs"] == 2, snapshot
        assert snapshot["indexed_packs"] == 2, snapshot
        assert snapshot["pending_index"] == 0, snapshot
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


def test_a_failed_head_is_an_error_not_a_foreign_object(fake_s3, tmp_path):
    """A pack whose HEAD failed was counted as a foreign object, skipped
    silently: nothing said the pass had missed a pack it could not read."""
    from dmi.storage.native_capture import _load_native_store_extension

    # The fake S3 answers 403 for anything under fault/forbidden/. The list
    # is not faulted, so the reconciler finds the pack and cannot HEAD it.
    key = f"fault/forbidden/{uuid.uuid4()}.dmi-pack"
    with STATE.lock:
        STATE.objects[key] = {"body": b"x", "meta": {}, "content_type": "",
                              "etag": '"0"'}
    with _catalog() as (_client, catalog):
        native = _storage_config(fake_s3, catalog.table_prefix)._native_dict()
        native.update(spool_root=str(tmp_path / "spool"), holder="head-test",
                      reconcile_prefix="fault/", s3_max_attempts=1)
        service = _load_native_store_extension().StorageService(native)
        service.start()  # reconciles once
        try:
            snapshot = service.snapshot()
        finally:
            service.stop()

    assert snapshot["reconcile_passes"] == 1, snapshot
    assert snapshot["reconcile_skipped_objects"] == 0, snapshot
    assert snapshot["reconcile_head_errors"] == 1, snapshot
    assert "HEAD" in snapshot["last_error"] and key in snapshot["last_error"]


def test_one_publisher_per_catalog(fake_s3, tmp_path):
    with _catalog() as (_client, catalog):
        # No start wait: the refusal is the point here, and waiting out the
        # holder is covered by the restart tests.
        config = _storage_config(fake_s3, catalog.table_prefix,
                                 start_lease_wait_s=0.0)
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


def test_a_pack_too_big_to_index_is_set_aside_and_the_rest_still_index(
        fake_s3, tmp_path):
    """A pack that can never fit the indexer's batch budget used to stall the
    whole queue: it was retried first on every cycle, and everything queued
    behind it was parked with it. It is now rejected once, flush says so,
    and later packs index normally."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    _stage(spool_root, range(40), records_per_pack=40)  # ~40 descriptors
    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        native = config._native_dict()
        native.update(
            spool_root=str(spool_root), holder="poison-test",
            poll_interval_ns=50_000_000, reconcile_on_start=False,
            # One descriptor fits, forty do not.
            indexer_max_estimated_bytes=4000)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            with pytest.raises(RuntimeError, match="cannot be indexed"):
                service.flush(3.0)
            small = _stage(spool_root, range(100, 101))
            assert service.flush(10.0)
            snapshot = service.snapshot()
        finally:
            service.stop()

        assert snapshot["rejected_packs"] == 1, snapshot
        assert snapshot["indexed_packs"] == 1, snapshot
        assert snapshot["pending_index"] == 0, snapshot
        assert sorted(_read_all(config)) == sorted(small)


def test_a_batch_over_the_budget_splits_until_every_pack_indexes(
        fake_s3, tmp_path):
    """Packs that each fit the indexer's batch budget can still overflow it
    together; the service halves such a batch until every half fits, and
    sets nothing aside."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(20), records_per_pack=2)
    assert len(_ready(spool_root)) == 10
    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3, catalog.table_prefix)
        native = config._native_dict()
        native.update(
            spool_root=str(spool_root), holder="split-test",
            poll_interval_ns=50_000_000, reconcile_on_start=False,
            # A two-record pack renders ~720 bytes: two fit, ten do not.
            indexer_max_estimated_bytes=2000)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            service.flush(30.0)
            snapshot = service.snapshot()
        finally:
            service.stop()

        assert _ready(spool_root) == []
        assert snapshot["uploaded_packs"] == 10, snapshot
        assert snapshot["indexed_packs"] == 10, snapshot
        assert snapshot["indexed_rows"] == 20, snapshot
        assert snapshot["batch_splits"] > 0, snapshot
        assert snapshot["rejected_packs"] == 0, snapshot
        assert snapshot["pending_index"] == 0, snapshot
        assert sorted(_read_all(config)) == sorted(tensors)


def test_flush_returns_on_time_when_the_catalog_stops_answering(
        fake_s3, tmp_path):
    """flush(timeout) waited for the cycle lock with no deadline, and the
    catalog client had no timeouts, so one ClickHouse connection that never
    answered held flush -- and flush_and_wait and close -- indefinitely."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        native = _storage_config(
            fake_s3, catalog.table_prefix,
            clickhouse_port=switch.port)._native_dict()
        native.update(spool_root=str(spool_root), holder="stall-test",
                      poll_interval_ns=20_000_000, reconcile_on_start=False,
                      clickhouse_request_timeout_s=5.0)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            switch.stall()
            _stage(spool_root, range(2))
            time.sleep(0.5)  # the background cycle is now waiting on ClickHouse

            outcome = {}

            def _flush():
                started = time.monotonic()
                outcome["drained"] = service.flush(1.0)
                outcome["elapsed"] = time.monotonic() - started

            waiter = threading.Thread(target=_flush, daemon=True)
            waiter.start()
            waiter.join(timeout=10.0)
            assert not waiter.is_alive(), "flush(1.0) still blocked after 10 s"
            assert outcome["drained"] is False
            assert outcome["elapsed"] < 3.0, outcome
        finally:
            switch.close()  # releases the stalled connections
            service.stop()


def test_a_flush_against_a_black_hole_catalog_overruns_by_one_request(
        fake_s3, tmp_path):
    """A flush's own cycle ran the periodic reconcile when it fell due: past
    the index pass that failed on a catalog that accepts connections and
    never answers, it listed the bucket and asked the catalog again, one
    more request timeout past the deadline. A flush cycle skips the
    reconcile now (the loop runs it), and its catalog requests, each
    bounded, are never cut mid-flight: flush(1.0) returns within one second
    plus the one request in flight at the deadline."""
    from dmi.storage.native_capture import _load_native_store_extension

    request_timeout = 4.0
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        native = _storage_config(
            fake_s3, catalog.table_prefix,
            clickhouse_port=switch.port)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="black-hole-test",
            # The loop sleeps through the test, so the flush runs the cycle.
            poll_interval_ns=60_000_000_000, reconcile_on_start=False,
            # Every cycle is due a reconcile.
            reconcile_interval_ns=1_000_000,
            # No renewal falls due while the test runs.
            lease_ttl_ns=30_000_000_000, publish_timeout_ns=5_000_000_000,
            clickhouse_request_timeout_s=request_timeout)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            _stage(spool_root, range(2))
            switch.stall()
            started = time.monotonic()
            drained = service.flush(1.0)
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
        finally:
            switch.close()  # releases the stalled connections
            service.stop()

    assert drained is False
    assert snapshot["reconcile_passes"] == 0, snapshot
    assert elapsed < 1.0 + request_timeout + 1.0, elapsed


def test_the_stop_after_a_flush_runs_no_further_cycle(fake_s3, tmp_path):
    """close() runs flush(budget) and then stop(). A loop that woke while
    the flush held the cycle lock took the lock the moment the flush let
    go -- before stop() could say anything -- and ran a whole cycle of its
    own, which stop() then joined: against a catalog that accepts
    connections and never answers, one more request timeout on top of the
    flush's and the lease release's. The loop now waits out its interval
    from the end of the last cycle, anyone's, so a stop() right after a
    flush finds it waiting and it leaves at once."""
    from dmi.storage.native_capture import _load_native_store_extension

    request_timeout = 4.0
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        native = _storage_config(
            fake_s3, catalog.table_prefix,
            clickhouse_port=switch.port)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="stop-after-flush",
            # Wakes while the flush below holds the cycle lock.
            poll_interval_ns=3_000_000_000, reconcile_on_start=False,
            # No renewal falls due while the test runs (a third of the TTL).
            lease_ttl_ns=60_000_000_000, publish_timeout_ns=5_000_000_000,
            clickhouse_request_timeout_s=request_timeout)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        stopped = False
        try:
            # Right after one of the loop's cycles, so that it next wakes
            # while the flush's cycle waits on the catalog.
            _wait_for(lambda: service.snapshot()["cycles"] >= 1,
                      timeout_s=10.0)
            _stage(spool_root, range(2))
            switch.stall()
            started = time.monotonic()
            drained = service.flush(1.0)
            flushed = time.monotonic() - started
            cycles = service.snapshot()["cycles"]
            started = time.monotonic()
            service.stop()
            stopped = True
            stop_s = time.monotonic() - started
            snapshot = service.snapshot()
        finally:
            switch.close()  # releases the stalled connections
            if not stopped:
                service.stop()

    assert drained is False
    assert flushed < 1.0 + request_timeout + 1.0, flushed
    # No cycle after the flush's: stop() waited for the lease release
    # alone, one request timeout against this catalog, not two.
    assert snapshot["cycles"] == cycles, (cycles, stop_s, snapshot)
    assert stop_s < request_timeout + 1.5, (stop_s, snapshot)


def test_a_flush_out_of_time_does_not_hash_the_spool(fake_s3, tmp_path):
    """Past its deadline a flush's cycle starts no upload, but it still
    listed the spool through the uploader, and a listing re-hashes every
    staged pack: flush(0) -- what flush_and_wait and close() pass once the
    sink's flush has spent the budget -- held its caller for as long as
    hashing the whole backlog took. It now asks only whether any pack is
    staged, by name. One sparse 1 GiB pack stands in for a backlog."""
    import hashlib

    from dmi.storage.native_capture import _load_native_store_extension

    size = 1 << 30
    digest = hashlib.sha256()
    zeros = bytes(1 << 24)
    for _ in range(size // len(zeros)):
        digest.update(zeros)
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    ready = spool_root / (f"{uuid.uuid4()}.1.1.{digest.hexdigest()}"
                          ".dmi-pack.ready")
    with open(ready, "wb") as sparse:
        sparse.truncate(size)  # no blocks on disk
    with _catalog() as (_client, catalog):
        native = _storage_config(fake_s3, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="flush-out-of-time",
            # The loop sleeps through the test, and nothing uploads.
            poll_interval_ns=60_000_000_000, reconcile_on_start=False,
            sweep_spool_on_start=False,
            uploader_max_in_flight_bytes=2 * size)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            started = time.monotonic()
            drained = service.flush(0.0)
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
        finally:
            service.stop()

    assert drained is False  # a pack is staged
    assert elapsed < 0.25, (elapsed, snapshot)
    assert snapshot["uploaded_packs"] == 0, snapshot
    # Not tried, so not cancelled either: it was never listed.
    assert snapshot["cancelled_uploads"] == 0, snapshot
    assert ready.exists()


def _put(request: bytes) -> bool:
    return request.startswith(b"PUT ")


def test_a_flush_returns_on_time_while_an_upload_stalls(fake_s3, tmp_path):
    """The flush's cycle uploads through an object store that accepted the
    PUT and never answers. It waited out the S3 read timeout on every
    attempt; its deadline now cancels the upload, and the pack stays in the
    spool, which is where a pack waits for a store that is not answering."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (_client, catalog):
        native = _storage_config(s3.url, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="stalled-upload-flush",
            # The loop sleeps through the test, so the flush runs the cycle.
            poll_interval_ns=60_000_000_000, reconcile_on_start=False,
            s3_read_timeout_s=30)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            s3.stall_requests(_put)
            tensors = _stage(spool_root, range(2))
            outcome = {}

            def _flush():
                started = time.monotonic()
                outcome["drained"] = service.flush(1.0)
                outcome["elapsed"] = time.monotonic() - started

            waiter = threading.Thread(target=_flush, daemon=True)
            waiter.start()
            waiter.join(timeout=15.0)
            assert not waiter.is_alive(), "flush(1.0) still blocked after 15 s"
            assert s3.stalled, "the upload never reached the store"
            assert outcome["drained"] is False
            assert outcome["elapsed"] < 3.0, outcome
            assert len(_ready(spool_root)) == 1
            snapshot = service.snapshot()
            # Cut short, not failed: nothing for the backoff to count.
            assert snapshot["upload_failures"] == 0, snapshot
            assert snapshot["cancelled_uploads"] == 1, snapshot
            assert snapshot["uploaded_packs"] == 0, snapshot

            s3.restore()
            assert service.flush(30.0)
            # A flush out of time uploads nothing, but still finds a drained
            # spool drained.
            assert service.flush(0.0)
        finally:
            s3.close()
            service.stop()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_stop_returns_promptly_while_an_upload_stalls(fake_s3, tmp_path):
    """stop() joined a loop whose cycle was inside a PUT the store never
    answers, so it waited out the S3 read timeout on every attempt, with the
    lease held. It cancels the upload now: the pack stays in the spool, the
    lease is released, and the next process uploads the pack."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (_client, catalog):
        native = _storage_config(s3.url, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="stalled-upload-stop",
            poll_interval_ns=20_000_000, reconcile_on_start=False,
            s3_read_timeout_s=60)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        stopper = None
        try:
            s3.stall_requests(_put)
            tensors = _stage(spool_root, range(2))
            _wait_for(lambda: s3.stalled, timeout_s=10.0)
            outcome = {}

            def _stop():
                started = time.monotonic()
                service.stop()
                outcome["elapsed"] = time.monotonic() - started

            stopper = threading.Thread(target=_stop, daemon=True)
            stopper.start()
            stopper.join(timeout=15.0)
            assert not stopper.is_alive(), "stop() still blocked after 15 s"
            assert outcome["elapsed"] < 3.0, outcome
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "released", snapshot
            assert snapshot["upload_failures"] == 0, snapshot
            assert snapshot["cancelled_uploads"] == 1, snapshot
            assert len(_ready(spool_root)) == 1
        finally:
            s3.close()  # releases the stalled PUT
            # Never a second stop() beside one still running.
            if stopper is None:
                service.stop()
            else:
                stopper.join(timeout=120.0)

        direct = _storage_config(fake_s3, catalog.table_prefix)
        successor = _service(direct, spool_root)
        successor.start()
        try:
            successor.flush(30.0)
        finally:
            successor.stop()
        assert _ready(spool_root) == []
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_a_flush_returns_on_time_while_the_index_reads_stall(
        fake_s3, tmp_path):
    """The store takes the PUTs, then stops answering GETs. The flush's
    cycle indexed what it had uploaded through a client no cancel could
    cut, and a pack whose read failed only moved the pass on to the next
    pack: 3 packs cost 3 x (4 attempts x s3_read_timeout_s + backoff), 40 s
    at a 3 s read timeout and 8 minutes a pack at the defaults. The reads
    are cut one request timeout past the deadline now, and the pass ends at
    the first read the store does not answer: the packs stay owed, none is
    counted against or set aside, and they index once the store answers."""
    from dmi.storage.native_capture import _load_native_store_extension

    request_timeout = 4.0
    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (_client, catalog):
        native = _storage_config(s3.url, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="stalled-read-flush",
            # The loop sleeps through the test, so the flush runs the cycle.
            poll_interval_ns=60_000_000_000, reconcile_on_start=False,
            s3_read_timeout_s=3,
            clickhouse_request_timeout_s=request_timeout)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            tensors = _stage(spool_root, range(6))  # three packs
            s3.stall_requests(_ranged_get)  # only the indexer's reads
            outcome = {}

            def _flush():
                started = time.monotonic()
                outcome["drained"] = service.flush(1.0)
                outcome["elapsed"] = time.monotonic() - started

            waiter = threading.Thread(target=_flush, daemon=True)
            waiter.start()
            waiter.join(timeout=90.0)
            assert not waiter.is_alive(), "flush(1.0) still blocked after 90 s"
            snapshot = service.snapshot()
            assert outcome["drained"] is False
            assert outcome["elapsed"] < 1.0 + request_timeout + 1.5, (
                outcome, snapshot)
            # One read's attempts, not every pack's.
            assert 1 <= len(s3.stalled) <= 2, (s3.stalled, snapshot)
            assert snapshot["uploaded_packs"] == 3, snapshot
            assert snapshot["pending_index"] == 3, snapshot
            assert snapshot["indexed_packs"] == 0, snapshot
            # Cut short by the deadline, not the packs' fault.
            assert snapshot["index_failures"] == 0, snapshot
            assert snapshot["rejected_packs"] == 0, snapshot
            assert _ready(spool_root) == []

            s3.cut()  # releases the stalled reads
            s3.restore()
            assert service.flush(30.0)
        finally:
            s3.close()
            service.stop()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_stop_returns_promptly_while_the_index_reads_stall(fake_s3, tmp_path):
    """stop() cancelled the uploads but joined a cycle whose index pass
    read, through the uncancellable client, every pack it had uploaded: a
    GET the store never answers cost 4 attempts x s3_read_timeout_s each.
    stop() cuts the reads too now. The packs it leaves unindexed are in the
    bucket, where the next start's reconcile finds them."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (_client, catalog):
        native = _storage_config(s3.url, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="stalled-read-stop",
            poll_interval_ns=20_000_000, reconcile_on_start=False,
            s3_read_timeout_s=3)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        stopper = None
        try:
            s3.stall_requests(_ranged_get)
            tensors = _stage(spool_root, range(6))
            _wait_for(lambda: s3.stalled, timeout_s=10.0)
            outcome = {}

            def _stop():
                started = time.monotonic()
                service.stop()
                outcome["elapsed"] = time.monotonic() - started

            stopper = threading.Thread(target=_stop, daemon=True)
            stopper.start()
            stopper.join(timeout=90.0)
            assert not stopper.is_alive(), "stop() still blocked after 90 s"
            snapshot = service.snapshot()
            assert outcome["elapsed"] < 3.0, (outcome, snapshot)
            assert snapshot["lease_state"] == "released", snapshot
            assert snapshot["rejected_packs"] == 0, snapshot
        finally:
            s3.close()  # releases the stalled reads
            if stopper is None:
                service.stop()
            else:
                stopper.join(timeout=120.0)

        direct = _storage_config(fake_s3, catalog.table_prefix)
        successor = _service(direct, spool_root)  # reconciles at start
        successor.start()
        try:
            successor.flush(30.0)
        finally:
            successor.stop()
        assert _ready(spool_root) == []
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_an_object_store_read_outage_sets_no_pack_aside(fake_s3, tmp_path):
    """A read the store never answered counted against the pack, as if the
    pack itself were unreadable: after max_index_attempts cycles of an
    outage the pack was set aside for good -- left in the bucket, out of
    the flush boundary, and reported by flush() as never indexable. A pass
    that meets an unanswered read now ends there and counts nothing
    against the pack."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (_client, catalog):
        native = _storage_config(s3.url, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="read-outage",
            poll_interval_ns=20_000_000, max_backoff_ns=200_000_000,
            reconcile_on_start=False, s3_read_timeout_s=1,
            s3_max_attempts=1, max_index_attempts=2)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            s3.stall_requests(_ranged_get)
            tensors = _stage(spool_root, range(2))  # one pack
            # Three passes over the pack, each ended by an unanswered read.
            _wait_for(lambda: service.snapshot()["index_failures"] >= 3,
                      timeout_s=30.0)
            snapshot = service.snapshot()
            assert snapshot["rejected_packs"] == 0, snapshot
            assert snapshot["pending_index"] == 1, snapshot

            s3.cut()
            s3.restore()
            assert service.flush(30.0)  # raised "set aside" before
            snapshot = service.snapshot()
            assert snapshot["indexed_packs"] == 1, snapshot
            assert snapshot["rejected_packs"] == 0, snapshot
        finally:
            s3.close()
            service.stop()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_a_flush_against_a_slow_catalog_indexes_one_batch_past_its_deadline(
        fake_s3, tmp_path):
    """A flush's cycle checked its deadline only while uploading: past it,
    it still indexed everything it had uploaded, batch after batch. Against
    a catalog that answers slowly but inside the request timeout no request
    fails, so the overrun grew with the batches -- 8 one-pack batches at
    0.4 s a statement ran a 1 s flush for 59 s. Past the deadline a
    flush now starts no further batch: it finishes the one in flight (or
    indexes one batch of what it uploaded) and leaves the rest owed, for
    the loop, or a later flush, to index. (Its index reads are cut one
    request timeout past the deadline too, which ends the pass; the request
    timeout here outlasts all eight batches, so that is not what ends it.)"""
    from dmi.storage.native_capture import _load_native_store_extension

    request_timeout = 60.0
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        native = _storage_config(
            fake_s3, catalog.table_prefix,
            clickhouse_port=switch.port)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="slow-catalog-flush",
            # The loop sleeps through the test, so the flush runs the cycle.
            poll_interval_ns=60_000_000_000, reconcile_on_start=False,
            indexer_max_packs=1,
            clickhouse_request_timeout_s=request_timeout)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            tensors = _stage(spool_root, range(16))  # eight packs
            switch.delay_requests(_slow_catalog(0.4, 0.4))
            started = time.monotonic()
            drained = service.flush(1.0)
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
            # One batch past the deadline -- about 7 s of statements at
            # 0.4 s each, the lease's included -- not eight (59 s before).
            assert elapsed < 1.0 + 12.0, (elapsed, snapshot)
            assert drained is False
            assert snapshot["uploaded_packs"] == 8, snapshot
            assert snapshot["indexed_packs"] == 1, snapshot
            assert snapshot["pending_index"] == 7, snapshot
            # Left owed by the deadline: neither failed nor set aside.
            assert snapshot["index_failures"] == 0, snapshot

            switch.restore()
            assert service.flush(60.0)
            assert service.snapshot()["indexed_packs"] == 8
        finally:
            switch.close()
            service.stop()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_only_the_loop_reconciles_never_a_flush(fake_s3, tmp_path):
    """A flush's cycles skip the periodic reconcile, which lists the whole
    bucket and asks the catalog about every page -- work no deadline
    bounds; the loop runs it. A reconcile is due on every cycle here, and
    the loop's first wake is 2 s off: the flush drains without one, and the
    loop's first cycle runs it."""
    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    with _catalog() as (_client, catalog):
        native = _storage_config(fake_s3, catalog.table_prefix)._native_dict()
        native.update(
            spool_root=str(spool_root), holder="flush-no-reconcile",
            poll_interval_ns=2_000_000_000, reconcile_on_start=False,
            reconcile_interval_ns=1_000_000)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        try:
            tensors = _stage(spool_root, range(2))
            assert service.flush(30.0)
            snapshot = service.snapshot()
            assert snapshot["indexed_packs"] == 1, snapshot
            assert snapshot["reconcile_passes"] == 0, snapshot
            _wait_for(lambda: service.snapshot()["reconcile_passes"] >= 1,
                      timeout_s=10.0)
        finally:
            service.stop()

        direct = _storage_config(fake_s3, catalog.table_prefix)
        assert sorted(_read_all(direct)) == sorted(tensors)


def test_dropping_a_running_service_does_not_hold_the_gil(fake_s3, tmp_path):
    """A service collected without stop() stops itself in its destructor,
    joining a cycle that may be waiting on the catalog. That ran with the
    GIL held, so every other Python thread froze until the join returned."""
    import gc

    from dmi.storage.native_capture import _load_native_store_extension

    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    ticks = []
    done = threading.Event()

    def _tick():
        while not done.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.01)

    with _catalog() as (_client, catalog):
        native = _storage_config(
            fake_s3, catalog.table_prefix,
            clickhouse_port=switch.port)._native_dict()
        native.update(spool_root=str(spool_root), holder="gil-test",
                      poll_interval_ns=20_000_000, reconcile_on_start=False,
                      clickhouse_request_timeout_s=2.0)
        service = _load_native_store_extension().StorageService(native)
        service.start()
        ticker = threading.Thread(target=_tick, daemon=True)
        try:
            switch.stall()
            _stage(spool_root, range(2))
            time.sleep(0.5)  # the background cycle is now waiting on ClickHouse
            ticker.start()
            time.sleep(0.1)
            before = len(ticks)
            started = time.monotonic()
            del service
            gc.collect()
            elapsed = time.monotonic() - started
            during = len(ticks) - before
        finally:
            done.set()
            ticker.join(timeout=10.0)
            switch.close()

    assert elapsed > 0.5, elapsed  # destruction really waited on the cycle
    # A thread ticking every 10 ms, free to run, ticks dozens of times.
    assert during >= 0.3 * elapsed / 0.01, (during, elapsed)


def test_the_lease_holds_through_an_object_store_outage(tmp_path):
    """The lease was renewed only inside a cycle, and failed cycles back off
    up to max_backoff -- past the lease TTL. With the object store down and
    ClickHouse healthy, a second publisher could take the catalog."""
    from dmi.storage.native_capture import _load_native_store_extension

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
    _stage(tmp_path / "first", range(1))  # a pack whose upload keeps failing
    module = _load_native_store_extension()
    with _catalog() as (_client, catalog):
        def _native(spool, holder):
            native = _storage_config(dead, catalog.table_prefix)._native_dict()
            native.update(
                spool_root=str(spool), holder=holder,
                poll_interval_ns=20_000_000, max_backoff_ns=10_000_000_000,
                lease_ttl_ns=3_000_000_000, publish_timeout_ns=1_000_000_000,
                clock_skew_ns=0, reconcile_on_start=False,
                s3_max_attempts=1, uploader_max_attempts=1)
            return native

        first = module.StorageService(_native(tmp_path / "first", "first"))
        rival = module.StorageService(_native(tmp_path / "rival", "rival"))
        first.start()
        try:
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline:
                try:
                    rival.start()
                except RuntimeError as refused:
                    assert "held" in str(refused)
                else:
                    rival.stop()
                    pytest.fail("a second publisher took the lease")
                time.sleep(0.5)
            snapshot = first.snapshot()
            first.rethrow_if_failed()
        finally:
            first.stop()

    assert snapshot["upload_failures"] >= 1, snapshot
    assert snapshot["lease_renewals"] >= 3, snapshot


def test_capture_round_trips_through_a_verified_tls_catalog(fake_s3, tmp_path):
    """The whole path -- schema, lease, index, publish, search, resolve --
    over https to the catalog, verified against a private CA.

    A TLS terminator with a freshly generated CA stands in front of the local
    ClickHouse HTTP port, so the server itself is untouched. The same catalog
    refuses the service when the client is not given that CA: https never
    falls back to trusting whatever answers."""
    from tests._private_ca import TlsTerminator, make_private_ca

    ca = make_private_ca(tmp_path / "ca")
    terminator = TlsTerminator(ca, (CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT))
    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(4))
    try:
        with _catalog() as (_client, catalog):
            tls = dict(clickhouse_scheme="https", clickhouse_host="127.0.0.1",
                       clickhouse_port=terminator.port,
                       # The local server's passwordless `default`, named so
                       # the credential headers are on every request.
                       clickhouse_user="default")

            untrusted = _storage_config(fake_s3, catalog.table_prefix, **tls)
            with pytest.raises(RuntimeError, match="(?i)certificate"):
                _service(untrusted, spool_root).start()
            # Counted on the terminator's thread, which may learn of the
            # refused handshake a moment after the client has given up.
            deadline = time.monotonic() + 5.0
            while (terminator.failed_handshakes < 1
                   and time.monotonic() < deadline):
                time.sleep(0.01)
            assert terminator.failed_handshakes >= 1

            config = _storage_config(fake_s3, catalog.table_prefix,
                                     clickhouse_ca_file=str(ca.ca_file), **tls)
            service = _service(config, spool_root)
            service.start()
            try:
                service.flush(30.0)
                snapshot = service.snapshot()
            finally:
                service.stop()
            assert snapshot["indexed_packs"] == 2, snapshot
            assert snapshot["indexed_rows"] == 4, snapshot

            captures = _read_all(config)
            assert sorted(captures) == sorted(tensors)
            for capture_id, tensor in tensors.items():
                assert captures[capture_id].payload == tensor.numpy().tobytes()
            # The CA as a hashed directory reaches the same catalog.
            by_path = _storage_config(fake_s3, catalog.table_prefix,
                                      clickhouse_ca_path=str(ca.ca_path), **tls)
            assert sorted(_read_all(by_path)) == sorted(tensors)
    finally:
        terminator.close()


# --- the publisher lease through ClickHouse errors and restarts ---------------


def test_a_catalog_cut_spanning_a_renewal_and_a_publish_recovers(
        fake_s3, tmp_path):
    """A ClickHouse error of unknown outcome -- a renewal or a publish that
    cannot reach the server -- quarantines the catalog writer and drops its
    lease without a tombstone. When the quarantine ended the service found
    no lease, latched "publisher lease lost", and indexing stopped for good,
    although ClickHouse was back and nobody else held the catalog. It now
    waits the quarantine out and takes a fresh lease."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=3.0, publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            tensors = _stage(spool_root, range(2))
            service.flush(30.0)
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "held", snapshot
            assert snapshot["failed"] is False, snapshot
            assert snapshot["lease_reacquisitions"] == 0, snapshot
            assert snapshot["quarantined_until"] == 0.0, snapshot

            switch.cut()
            cut_at = time.monotonic()
            # Staged during the cut: uploaded, then owed, since the index
            # (and its publish) cannot reach the catalog.
            tensors.update(_stage(spool_root, range(2, 4)))
            # The renewal due every ttl/3 fails and quarantines the lease.
            _wait_for(lambda: service.snapshot()["lease_state"] == "quarantined",
                      timeout_s=5.0)
            snapshot = service.snapshot()
            assert snapshot["quarantined_until"] > time.monotonic(), snapshot
            assert snapshot["running"] is True, snapshot
            assert snapshot["index_failures"] >= 1, snapshot  # it tried
            assert snapshot["pending_index"] == 1, snapshot
            with pytest.raises(TimeoutError, match="1 uploaded but unindexed"):
                service.flush(0.5)
            # Longer than ttl/3, so the cut spans a renewal and a publish.
            time.sleep(max(0.0, cut_at + 2.5 - time.monotonic()))
            switch.restore()

            # The background loop recovers on its own, without a flush.
            _wait_for(lambda: service.snapshot()["indexed_packs"] == 2,
                      timeout_s=15.0)
            snapshot = service.snapshot()
            assert snapshot["failed"] is False, snapshot
            assert snapshot["running"] is True, snapshot
            assert snapshot["lease_state"] == "held", snapshot
            assert snapshot["lease_reacquisitions"] >= 1, snapshot
            assert snapshot["quarantined_until"] == 0.0, snapshot

            tensors.update(_stage(spool_root, range(4, 6)))
            service.flush(30.0)
            service.rethrow_if_failed()
            snapshot = service.snapshot()
        finally:
            service.stop()
            switch.close()

        assert snapshot["indexed_packs"] == 3, snapshot
        assert snapshot["pending_index"] == 0, snapshot
        direct = _storage_config(fake_s3, catalog.table_prefix)
        captures = _read_all(direct)
        assert sorted(captures) == sorted(tensors)
        for capture_id, tensor in tensors.items():
            assert captures[capture_id].payload == tensor.numpy().tobytes()


def test_a_quarantined_service_leaves_new_packs_in_the_spool(
        fake_s3, tmp_path):
    """While quarantined the service cannot index, so it must not upload
    either: an uploaded but unindexed pack is remembered only in memory, and
    a crash then orphans it in the bucket when reconcile_on_start is off.
    Before, a quarantined cycle with nothing owed uploaded the whole spool
    into that in-memory list; now new packs stay in the durable spool until
    the lease is back."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=3.0, publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            switch.cut()
            # Nothing is owed: the renewal alone fails and quarantines.
            _wait_for(lambda: service.snapshot()["lease_state"] == "quarantined",
                      timeout_s=5.0)
            assert service.snapshot()["pending_index"] == 0

            tensors = _stage(spool_root, range(2))
            assert len(_ready(spool_root)) == 1
            # flush() drives cycles; none of them may upload.
            with pytest.raises(TimeoutError):
                service.flush(0.5)
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "quarantined", snapshot
            assert snapshot["uploaded_packs"] == 0, snapshot
            assert snapshot["pending_index"] == 0, snapshot
            assert len(_ready(spool_root)) == 1

            switch.restore()
            service.flush(30.0)
            service.rethrow_if_failed()
            snapshot = service.snapshot()
        finally:
            service.stop()
            switch.close()

        assert snapshot["uploaded_packs"] == 1, snapshot
        assert snapshot["indexed_packs"] == 1, snapshot
        assert _ready(spool_root) == []
        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert sorted(captures) == sorted(tensors)


def _lease_row_live(client, prefix) -> bool:
    """Whether the newest lease row on the catalog still keeps rivals out."""
    table = f"`{DATABASE}`.`{prefix}_publisher_lease`"
    return client.execute(
        f"SELECT max(expires_at_ns) > toUnixTimestamp64Nano(now64(9)) "
        f"FROM {table} WHERE term = (SELECT max(term) FROM {table})")[0][0] == 1


def _sample_lease(service, client, prefix, seconds, *, origin=None,
                  every=0.1):
    """(t, lease_state, row live) every `every` s for `seconds`, t measured
    from `origin` (time.monotonic(); the first sample by default). The state
    is read before the row, so a sample that says "held" over a dead row
    means the service called the lease held after the row had expired."""
    log = []
    origin = time.monotonic() if origin is None else origin
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        state = service.snapshot()["lease_state"]
        log.append((round(time.monotonic() - origin, 2), state,
                    _lease_row_live(client, prefix)))
        time.sleep(every)
    return log


def _held_but_dead(log):
    return [sample for sample in log if sample[1] == "held" and not sample[2]]


def _quarantined_at(snapshot, ttl_s: float) -> float:
    """When the writer quarantined, on time.monotonic()'s clock: the
    quarantine ends one TTL after it began, on the same steady clock. More
    precise than the first sample that saw it."""
    assert snapshot["lease_state"] == "quarantined", snapshot
    return snapshot["quarantined_until"] - ttl_s


def _lease_row_left_s(client, prefix) -> float:
    """Seconds the newest lease row has left on the server's clock; negative
    once it has expired."""
    table = f"`{DATABASE}`.`{prefix}_publisher_lease`"
    return client.execute(
        f"SELECT (toInt64(max(expires_at_ns)) - "
        f"toInt64(toUnixTimestamp64Nano(now64(9)))) / 1e9 "
        f"FROM {table} WHERE term = (SELECT max(term) FROM {table})")[0][0]


def test_passes_over_committed_packs_do_not_hold_off_the_renewal(
        fake_s3, tmp_path):
    """The lease thread renews a third of the TTL after the last renewal,
    and the service counted an index pass as one whenever it indexed or
    SKIPPED a pack. A pass over packs the catalog had already committed
    publishes nothing, so nothing renewed the row -- yet each such pass
    restarted the renewal clock. Packs that keep arriving already committed
    (re-staged after a crash between upload and spool removal, or reconciled
    first at start) held the renewal off indefinitely, and the row expired
    under a service still reporting the lease held. The schedule now runs
    from when the claim that stamped the row was sent, whatever the passes
    do."""
    import os

    spool_root = tmp_path / "spool"
    _stage(spool_root, range(2))
    (ready,) = _ready(spool_root)
    kept = tmp_path / ready.name
    os.link(ready, kept)  # the spool's copy goes once it is uploaded
    with _catalog() as (client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, reconcile_on_start=False,
            lease_ttl_s=3.0, publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            service.flush(30.0)
            before = service.snapshot()
            assert before["indexed_packs"] == 1, before

            # Two TTLs of cycles that each upload the committed pack again
            # (the uploader finds it in the store and verifies it) and index
            # it: every pass skips it.
            log, left = [], []
            origin = time.monotonic()
            while time.monotonic() < origin + 6.0:
                try:
                    os.link(kept, ready)
                except FileExistsError:
                    pass
                state = service.snapshot()["lease_state"]
                log.append((round(time.monotonic() - origin, 2), state,
                            _lease_row_live(client, catalog.table_prefix)))
                left.append(_lease_row_left_s(client, catalog.table_prefix))
                time.sleep(0.1)
            after = service.snapshot()
            service.rethrow_if_failed()
        finally:
            service.stop()

        assert after["indexed_packs"] == 1, after
        assert after["uploaded_packs"] - before["uploaded_packs"] >= 10, after
        assert not _held_but_dead(log), log
        # A renewal falls due a third of the TTL after the last and the lease
        # thread looks every sixth, so the row never has less than about half
        # the TTL left; a third leaves room for a slow request.
        assert min(left) > 1.0, left
        # One every 1 to 1.5 s over the 6 s.
        assert after["lease_renewals"] - before["lease_renewals"] >= 3, after


def test_a_stalled_renewal_gives_up_while_the_lease_row_is_still_live(
        fake_s3, tmp_path):
    """One renewal is three ClickHouse requests, and each was bounded only by
    the client's request timeout -- 60 s by default, against a 15 s TTL --
    while the lease thread held the lease lock. A ClickHouse that accepted
    the renewal's connection and never answered let the row expire with the
    service still reporting the lease held, so a rival could take the
    catalog before any error surfaced. Every request made under the lease is
    now bounded by the lease deadline -- when the claim that stamped the row
    was sent, plus the TTL, less the skew and a margin -- so the stalled
    renewal fails, and the writer quarantines, while its row still keeps
    rivals out."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=3.0, publish_timeout_s=1,
            # Longer than the TTL, so only the lease deadline can end the
            # stalled request in time.
            clickhouse_request_timeout_s=20.0)
        service = _service(config, spool_root)
        service.start()
        try:
            assert service.snapshot()["lease_state"] == "held"
            stalled_at = time.monotonic()
            switch.stall()
            # Past the row's whole TTL, so the row has expired by the end.
            log = _sample_lease(service, client, catalog.table_prefix, 4.0,
                                origin=stalled_at)
            assert not _held_but_dead(log), log
            snapshot = service.snapshot()
            # At the lease deadline, 2.9 s after start()'s claim was sent,
            # a moment before stalled_at: before the row can have expired.
            # From the quarantine itself, not the first sample to see it,
            # which could come a sampling interval later.
            quarantined_at = _quarantined_at(snapshot, 3.0) - stalled_at
            assert 2.0 < quarantined_at < 2.95, (quarantined_at, log)
            assert "lease renewal failed" in snapshot["last_error"], snapshot
            assert "Timeout" in snapshot["last_error"], snapshot
            assert snapshot["failed"] is False, snapshot

            # The quarantine is the recoverable kind: the lease comes back.
            switch.restore()
            _wait_for(lambda: service.snapshot()["lease_state"] == "held",
                      timeout_s=10.0)
            service.rethrow_if_failed()
        finally:
            switch.close()  # releases the stalled connections
            service.stop()


def _catalog_insert_but_the_lease(request: bytes) -> bool:
    body = request.partition(b"\r\n\r\n")[2]
    return body.startswith(b"INSERT") and b"_publisher_lease` (term" not in body


def test_a_claim_that_wrote_nothing_goes_again_a_tick_later(
        fake_s3, tmp_path):
    """A claim that failed before its INSERT went out -- here every
    connection is refused -- wrote nothing, so it does not quarantine the
    writer; and so that flush()'s fast cycles do not hammer a catalog that
    cannot answer, the next claim waits a lease tick (a sixth of the TTL),
    not a cycle."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        ttl = 6.0  # a 1 s tick
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=ttl, publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            _stage(spool_root, range(2))
            switch.cut()
            # The renewal fails and quarantines; once that ends, every claim
            # fails to connect, and writes nothing.
            _wait_for(lambda: service.snapshot()["lease_state"]
                      == "quarantined", timeout_s=ttl)
            _wait_for(lambda: service.snapshot()["lease_state"]
                      == "reacquiring", timeout_s=ttl + 2)

            def _flush():
                try:
                    service.flush(4.0)
                except Exception:  # noqa: BLE001 -- only the traffic counts
                    pass

            flusher = threading.Thread(target=_flush, daemon=True)
            watched_from = time.monotonic()
            flusher.start()
            flusher.join(timeout=10)
            watched = time.monotonic() - watched_from
            refused = [t for t in switch.refused if t >= watched_from]
            snapshot = service.snapshot()
        finally:
            switch.close()
            service.stop()

        # Not quarantined by claims that wrote nothing.
        assert snapshot["lease_state"] == "reacquiring", snapshot
        # One claim a tick, of up to three connection attempts each (the
        # client retries a refused connection): about 12 in 4 s, where
        # flush()'s cycles, 50 ms apart, would make several times that.
        assert len(refused) <= 3 * (watched / 1.0 + 2), (len(refused), watched)


def _replay_guard_read(request: bytes) -> bool:
    body = request.partition(b"\r\n\r\n")[2]
    return (body.startswith(b"SELECT store_id, toString(pack_id) FROM")
            and b"_pack_inventory" in body)


@pytest.mark.parametrize("stalled", [_catalog_insert_but_the_lease,
                                     _replay_guard_read],
                         ids=["insert", "read"])
def test_a_stall_inside_the_index_pass_gives_up_while_the_lease_row_is_live(
        fake_s3, tmp_path, stalled):
    """The index pass holds the lease lock across its catalog requests -- the
    version claim, the descriptor INSERTs, the publish and its read-backs --
    and those were bounded only by the client's request timeout. One that
    stalled kept the lease thread from renewing: the row expired while the
    snapshot still said "held", until the 20 s request timeout. The lease
    deadline now bounds every request sent under the lease, not only the
    lease's own, so the stalled request fails, and the lease quarantines,
    while the row is still live. A stalled INSERT has an unknown outcome and
    quarantines the writer itself; a stalled read (the replay guard) does
    not, and it is the lease lock's scope that abandons the lease its
    deadline passed on, rather than go on calling it held."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=3.0, publish_timeout_s=1,
            clickhouse_request_timeout_s=20.0)
        service = _service(config, spool_root)
        service.start()
        try:
            # Every such request stalls, while the lease's own go through,
            # so the lease thread alone could keep the row alive -- if it
            # got the lock.
            switch.stall_requests(stalled)
            tensors = _stage(spool_root, range(2))
            _wait_for(lambda: switch.stalled, timeout_s=10.0)
            log = _sample_lease(service, client, catalog.table_prefix, 2.5,
                                origin=switch.stalled[0])
            _wait_for(lambda: service.snapshot()["lease_state"]
                      == "quarantined", timeout_s=2.0)
            snapshot = service.snapshot()
            log += _sample_lease(service, client, catalog.table_prefix, 3.0,
                                 origin=switch.stalled[0])
            assert not _held_but_dead(log), log
            # By the lease deadline, at most 2.9 s after the stalled
            # request went out (the lease renews before each request that
            # finds it due).
            quarantined_at = _quarantined_at(snapshot, 3.0) - switch.stalled[0]
            assert quarantined_at < 2.95, (quarantined_at, log)
            assert snapshot["failed"] is False

            switch.restore()
            service.flush(30.0)
            service.rethrow_if_failed()
        finally:
            switch.close()
            service.stop()

        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert sorted(captures) == sorted(tensors)


def _ranged_get(request: bytes) -> bool:
    head = request.partition(b"\r\n\r\n")[0].lower()
    return head.startswith(b"get ") and b"\r\nrange:" in head


def test_a_stalled_object_store_read_does_not_hold_up_the_lease(
        fake_s3, tmp_path):
    """The index pass read each pack's footer from the object store with the
    lease lock held, so a GET that stalled -- bounded only by the S3 read
    timeout, 120 s by default -- kept the lease thread from renewing, and the
    row expired while the snapshot said "held". The pass now takes the lock
    only for its catalog requests and reads packs without it, so the lease
    keeps renewing through a stalled read."""
    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (client, catalog):
        config = _storage_config(
            s3.url, catalog.table_prefix, reconcile_on_start=False,
            lease_ttl_s=3.0, publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            # The uploader PUTs and HEADs a new pack; only the indexer's
            # footer reads are ranged GETs.
            s3.stall_requests(_ranged_get)
            tensors = _stage(spool_root, range(2))
            _wait_for(lambda: s3.stalled, timeout_s=10.0)
            renewals = service.snapshot()["lease_renewals"]
            # Two TTLs: without renewals the row would be long dead.
            log = _sample_lease(service, client, catalog.table_prefix, 6.0,
                                origin=s3.stalled[0])
            assert all(state == "held" and live
                       for _, state, live in log), log
            assert service.snapshot()["lease_renewals"] >= renewals + 3

            # Fail the stalled read: the pack stays owed and indexes after.
            s3.cut()
            s3.restore()
            service.flush(30.0)
            service.rethrow_if_failed()
        finally:
            s3.close()
            service.stop()

        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert sorted(captures) == sorted(tensors)


def test_a_pass_whose_lease_changed_while_it_read_rereads_the_replay_guard(
        fake_s3, tmp_path):
    """Reading packs without the lease lock opens a window the whole-pass
    lock did not have: the lease can be lost and taken afresh while the pass
    reads, and another publisher can hold the catalog meanwhile and index
    the same packs (its reconcile finds them uploaded and uncommitted). The
    replay guard the pass read first no longer holds then, and a commit that
    trusted it would publish the packs a second time, at a higher version.
    The commit reads the guard again when the lease is not the one it was
    read under."""
    from tests.test_native_catalog_lease_live import CatalogDriver, _open

    spool_root = tmp_path / "spool"
    _stage(spool_root, range(4))  # two packs
    store = _Driver(STORE_DRIVER)
    try:
        uploaded = store.call(
            op="upload_pending", endpoint=fake_s3, bucket=BUCKET,
            region=REGION, access=ACCESS, secret=SECRET, token=None,
            insecure=True, connect_timeout=5, read_timeout=15, max_attempts=4,
            store_id="s3", root=str(spool_root), spool_max_bytes=1 << 40,
            limit=-1, max_workers=4, max_in_flight_bytes=1 << 30)
        assert uploaded["ok"], uploaded
    finally:
        store.close()
    refs = uploaded["refs"]
    assert len(refs) == 2, refs

    with _catalog() as (client, catalog):
        driver = CatalogDriver()
        try:
            _open(driver, catalog.table_prefix)
            assert driver.call(op="ensure_schema")["ok"]
            assert driver.call(op="acquire", holder="indexer")["ok"]
            result = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
                rival_indexes_before_commit=True)
            assert result["ok"], result
            assert result["result"]["indexed_packs"] == 0, result
            assert result["result"]["skipped_packs"] == 2, result
            assert driver.call(op="release")["ok"]
        finally:
            driver.close()

        table = f"`{DATABASE}`.`{catalog.table_prefix}_capture_raw`"
        # The rival's one publish, not a second copy of every descriptor.
        assert client.execute(
            f"SELECT count(), uniqExact(index_version) FROM {table}") == [
                (4, 1)]
        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert len(captures) == 4


@pytest.mark.parametrize("after_conflict", [None, "transport", "lease_refused"])
def test_a_conflicted_publish_reports_the_conflict_unless_the_lease_was_lost(
        fake_s3, tmp_path, after_conflict):
    """A publish that finds a second writer's row at its own version is
    visible and must not be retried, so the indexer records its packs in the
    inventory and raises kPublishConflict: a supervisor matching on it learns
    that something else is writing the prefix.

    If that inventory INSERT then fails in transport, the conflict is still
    what the pass reports, with the failure in its message (catalog.py's
    `raise conflict from commit_failure`). Left to propagate, the transport
    error replaced it.

    A lease refusal on that request is not a transport error, though. Under
    the storage service every request runs behind the lease scope's hook,
    which renews first once a renewal is due, and a rival claiming the lease
    -- which is when a second writer turns up -- refuses it. index_bounded
    tells a lost lease by its error kind and rethrows it, so that the pass it
    cut short is owed (storage_service.cpp); relabelled a conflict, the loss
    was handled as an ordinary failed batch. The refusal keeps its kind and
    carries the conflict in its message.

    The conflict is real: a foreign watermark row lands at the pass's version
    after the pass's own and before its owners read-back
    (`conflict_at_publish`, in conformance_catalog's `index` op).
    """
    from tests.test_native_catalog_lease_live import CatalogDriver, _open

    spool_root = tmp_path / "spool"
    _stage(spool_root, range(4))  # two packs
    store = _Driver(STORE_DRIVER)
    try:
        uploaded = store.call(
            op="upload_pending", endpoint=fake_s3, bucket=BUCKET,
            region=REGION, access=ACCESS, secret=SECRET, token=None,
            insecure=True, connect_timeout=5, read_timeout=15, max_attempts=4,
            store_id="s3", root=str(spool_root), spool_max_bytes=1 << 40,
            limit=-1, max_workers=4, max_in_flight_bytes=1 << 30)
        assert uploaded["ok"], uploaded
    finally:
        store.close()
    refs = uploaded["refs"]
    assert len(refs) == 2, refs

    with _catalog() as (client, catalog):
        driver = CatalogDriver()
        try:
            _open(driver, catalog.table_prefix)
            assert driver.call(op="ensure_schema")["ok"]
            assert driver.call(op="acquire", holder="indexer")["ok"]
            seam = {"conflict_at_publish": True}
            if after_conflict is not None:
                seam["after_conflict"] = after_conflict
            result = driver.call(
                op="index", refs=refs, endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
                **seam)
        finally:
            driver.close()

        def table(name):
            return f"`{DATABASE}`.`{catalog.table_prefix}_{name}`"

        # The conflict: two publishes at one version, the pass's own among
        # them with its whole manifest, so its packs are visible.
        conflicted = client.execute(
            f"SELECT index_version FROM {table('index_watermark')} "
            "GROUP BY index_version HAVING uniqExact(publish_id) = 2")
        assert len(conflicted) == 1, conflicted
        assert client.execute(
            f"SELECT count() FROM {table('snapshot_manifest')} "
            "WHERE index_version = %(version)s",
            {"version": conflicted[0][0]}) == [(2,)]
        recorded = client.execute(
            f"SELECT count() FROM {table('pack_inventory_raw')}")[0][0]

        assert not result["ok"], result
        message = result["message"]
        if after_conflict is None:
            assert result["error"] == "SnapshotPublishConflictError", result
            assert recorded == 2, result
        elif after_conflict == "transport":
            assert result["error"] == "SnapshotPublishConflictError", result
            assert "was published by this writer" in message, result
            assert ("recording its packs in the inventory then failed too"
                    in message), result
            assert "simulated connection reset" in message, result
            assert recorded == 0, result
        else:
            assert result["error"] == "PublisherLeaseHeldError", result
            assert "is contested" in message, result
            assert "was published by this writer" in message, result
            assert recorded == 0, result


def _lease_head_read(request: bytes) -> bool:
    return b"SELECT term, toString(lease_id)" in request


@pytest.mark.parametrize("schema", ["installed", "fresh"])
def test_start_survives_a_first_lease_read_slower_than_its_bound(
        fake_s3, tmp_path, schema):
    """A claim made without a lease has min(clickhouse_request_timeout_s,
    lease_ttl_s / 3) per request, and a cold server's first read can take
    longer. start() failed outright when its claim timed out; it now retries
    one that did, as it retries one another holder refused, until
    start_lease_wait_s runs out. A head read that timed out wrote nothing,
    so it does not quarantine the writer and the retry need not wait. On a
    fresh catalog the first lease read is the schema install's claim, and
    that is the first start against a new catalog -- when the server is
    likeliest to be cold -- so it is retried the same way, within the
    install lease's own wait."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1)
        if schema == "installed":
            # The schema first, directly: then the service's own claim is
            # the first lease read through the switch.
            warm = _service(_storage_config(fake_s3, catalog.table_prefix,
                                            **knobs), tmp_path / "warm")
            warm.start()
            warm.stop()

        switch.slow_once(_lease_head_read, 1.5)  # past the 1 s claim bound
        service = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            **knobs), tmp_path / "spool")
        started = time.monotonic()
        service.start()
        try:
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "held", snapshot
            # Retried at once (the install claim after its 0.5 s retry
            # sleep), without waiting out a 3 s quarantine.
            assert 1.0 <= elapsed < 3.0, elapsed
            if schema == "installed":
                # The schema install records no error of its own.
                assert "Timeout" in snapshot["last_error"], snapshot
        finally:
            service.stop()
            switch.close()


def _first(predicate):
    """A predicate that picks only the first request `predicate` does."""
    picked = []

    def _pick(request: bytes) -> bool:
        if picked or not predicate(request):
            return False
        picked.append(time.monotonic())
        return True

    return _pick


@pytest.mark.parametrize("delivered", [False, True],
                         ids=["never-delivered", "delivered-late"])
def test_start_waits_out_the_quarantine_its_own_claim_left(
        fake_s3, tmp_path, delivered):
    """A claim INSERT that timed out at start() may still land, so it
    quarantines the writer for a TTL -- and start() failed at once whenever
    that quarantine ended past start_lease_wait_s, which at the default wait
    (lease_ttl_s + publish_timeout_s + clock_skew_s) it always did once the
    INSERT had used its claim bound: a slow INSERT at start, a cold
    replicated catalog's quorum INSERT say, still failed start(). The
    quarantine is now waited out even past the wait, once, like the live
    row of a crashed predecessor it may be, and the claim made again after
    it -- refused by the late row, if it landed, until that expires."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1)
        warm = _service(_storage_config(fake_s3, catalog.table_prefix,
                                        **knobs), tmp_path / "warm")
        warm.start()
        warm.stop()

        if delivered:
            # Reaches the server after the 1 s claim bound: the row lands,
            # and is this service's own, live for a TTL.
            switch.slow_once(_lease_insert, 1.5)
        else:
            switch.stall_requests(_first(_lease_insert))
        service = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            **knobs), tmp_path / "spool")
        started = time.monotonic()
        service.start()
        try:
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "held", snapshot
            # The claim timed out at 1 s and quarantined until 4 s; the
            # default wait, 3 + 1 = 4 s, ended before a claim could follow.
            assert 3.9 < elapsed < 7.0, elapsed
        finally:
            service.stop()
            switch.close()


def _listing(request: bytes) -> bool:
    line = request.partition(b"\r\n")[0]
    return line.startswith(b"GET ") and b"list-type=2" in line


def _upload_behind_the_service(spool_root: Path, endpoint: str) -> list:
    """Upload what the spool holds without indexing it -- the crash window
    between an upload and its index -- and return the refs."""
    store = _Driver(STORE_DRIVER)
    try:
        uploaded = store.call(
            op="upload_pending", endpoint=endpoint, bucket=BUCKET,
            region=REGION, access=ACCESS, secret=SECRET, token=None,
            insecure=True, connect_timeout=5, read_timeout=15, max_attempts=4,
            store_id="s3", root=str(spool_root), spool_max_bytes=1 << 40,
            limit=-1, max_workers=4, max_in_flight_bytes=1 << 30)
        assert uploaded["ok"], uploaded
    finally:
        store.close()
    assert _ready(spool_root) == []
    return uploaded["refs"]


def _sample_during(call, service, client, prefix, *, every=0.1):
    """Run call() while sampling (t, lease_state, row live) from another
    thread, as _sample_lease does; the row is read only while the service
    says "held", since the lease table may not exist before that."""
    log = []
    done = threading.Event()
    origin = time.monotonic()

    def _sample():
        while not done.is_set():
            state = service.snapshot()["lease_state"]
            live = (_lease_row_live(client, prefix) if state == "held"
                    else None)
            log.append((round(time.monotonic() - origin, 2), state, live))
            time.sleep(every)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        call()
    finally:
        done.set()
        sampler.join(timeout=10)
    return log


@pytest.mark.parametrize("crash_window_pack", [True, False])
def test_the_lease_renews_while_start_reconciles(
        fake_s3, tmp_path, crash_window_pack):
    """start() takes the lease, then sweeps the spool and reconciles the
    bucket, and only then started the lease thread: nothing renewed the
    lease while a large bucket was listed. Once its deadline passed, the
    first lease-locked step abandoned it, so a crash-window pack found after
    that could not be indexed and start() failed, where main re-claimed its
    own lapsed row and indexed it; with nothing to index, start() returned
    with the lease quarantined, and the snapshot said "held" over a dead row
    all through the listing. The lease thread now runs from the moment the
    lease is taken. One slow listing page (6 s, two TTLs) stands in for a
    bucket of many pages."""
    spool_root = tmp_path / "spool"
    tensors = {}
    if crash_window_pack:
        tensors = _stage(spool_root, range(2))
        _upload_behind_the_service(spool_root, fake_s3)
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (client, catalog):
        config = _storage_config(s3.url, catalog.table_prefix,
                                 lease_ttl_s=3.0, publish_timeout_s=1)
        service = _service(config, spool_root)
        s3.slow_once(_listing, 6.0)
        try:
            log = _sample_during(service.start, service, client,
                                 catalog.table_prefix)
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "held", snapshot
            assert snapshot["reconcile_passes"] == 1, snapshot
            assert snapshot["reconciled_packs"] == int(crash_window_pack), \
                snapshot
            assert snapshot["indexed_packs"] == int(crash_window_pack), \
                snapshot
            assert snapshot["lease_renewals"] >= 3, snapshot
            assert not _held_but_dead(log), log
            service.flush(30.0)
            service.rethrow_if_failed()
        finally:
            service.stop()
            s3.close()

        if crash_window_pack:
            captures = _read_all(_storage_config(fake_s3,
                                                 catalog.table_prefix))
            assert sorted(captures) == sorted(tensors)


def _multipart_part_or_abort(request: bytes) -> bool:
    line = request.partition(b"\r\n")[0]
    return ((line.startswith(b"PUT ") and b"partNumber=" in line)
            or (line.startswith(b"DELETE ") and b"uploadId=" in line))


def test_the_lease_renews_until_the_loops_last_cycle_is_done(
        fake_s3, tmp_path):
    """stop() woke the loop and the lease thread together, and the lease
    thread left at once while the loop's last cycle still ran: the snapshot
    said "held" over a row that had expired meanwhile, and stop() found its
    lease abandoned instead of releasing it. The lease thread now stops only
    after the loop has.

    stop() cuts the cycle's object-store requests short, so the cycle needs
    something to do that no cancel cuts, outside the lease lock (inside it,
    each request renews the lease when due). A multipart upload that stop()
    cuts is aborted with a request of its own, one attempt of up to 5 s --
    here held for all of it, past the 3 s lease -- while the lease thread
    must go on renewing. (It used to be an index read held past the lease
    deadline; stop() cuts reads now.) The pack is a sparse file of zeros
    named for its checksum, over the client's 64 MiB multipart threshold:
    it is only ever uploaded, and stays staged."""
    import hashlib

    size = 65 << 20
    digest = hashlib.sha256()
    zeros = bytes(1 << 20)
    for _ in range(size // len(zeros)):
        digest.update(zeros)
    spool_root = tmp_path / "spool"
    s3 = _Switch.to_url(fake_s3)
    with _catalog() as (client, catalog):
        config = _storage_config(s3.url, catalog.table_prefix,
                                 reconcile_on_start=False, lease_ttl_s=3.0,
                                 publish_timeout_s=1)
        service = _service(config, spool_root)
        service.start()
        try:
            s3.stall_requests(_multipart_part_or_abort)
            ready = spool_root / (f"{uuid.uuid4()}.1.1.{digest.hexdigest()}"
                                  ".dmi-pack.ready")
            with open(ready, "wb") as sparse:
                sparse.truncate(size)
            _wait_for(lambda: s3.stalled, timeout_s=10.0)  # a part, held
            started = time.monotonic()
            log = _sample_during(service.stop, service, client,
                                 catalog.table_prefix)
            elapsed = time.monotonic() - started
            snapshot = service.snapshot()
        finally:
            service.stop()
            s3.close()

        assert not _held_but_dead(log), log
        assert snapshot["lease_state"] == "released", (snapshot, log)
        # The part was cut and the abort held to its 5 s bound: the loop's
        # last cycle outlived the lease, which kept renewing throughout.
        assert len(s3.stalled) == 2, s3.stalled
        assert 4.5 < elapsed < 9.0, (elapsed, snapshot)
        assert snapshot["lease_renewals"] >= 3, snapshot
        assert snapshot["cancelled_uploads"] == 1, snapshot
        assert snapshot["upload_failures"] == 0, snapshot
        assert _ready(spool_root) == [ready]


def _slow_catalog(insert_s: float, read_s: float):
    """A delay_requests() function: a catalog slow but healthy, every INSERT
    answered insert_s late and every SELECT read_s late."""
    def _delay(request: bytes) -> float:
        body = request.partition(b"\r\n\r\n")[2]
        if body.startswith(b"INSERT"):
            return insert_s
        if body.startswith(b"SELECT"):
            return read_s
        return 0.0
    return _delay


def test_a_slow_but_healthy_catalog_keeps_its_lease_through_index_passes(
        fake_s3, tmp_path):
    """Every request made under the lease lock has to be answered by the
    lease deadline, and a pass renewed only at the indexer's keep_lease hook,
    before its catalog writes. Between two hooks ran several requests with
    no chance to renew -- allocate_version's two max() reads, its claim
    INSERT and read-back; the publish's watermark INSERT, read-backs and the
    inventory commit -- so against a catalog slow but healthy the renewal
    at the next hook started with too little time left, timed out, and
    quarantined the lease, pass after pass: flush() never drained. main
    drained it, bounding nothing. The renewal is now checked before every
    request made under the lease lock, reads included. Scaled 1/5: 0.6 s
    INSERTs and 0.2 s reads against a 3 s TTL are 3 s and 1 s at the 15 s
    default."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1, clickhouse_request_timeout_s=20.0)
        warm = _service(_storage_config(fake_s3, catalog.table_prefix,
                                        **knobs), tmp_path / "warm")
        warm.start()
        warm.stop()

        switch.delay_requests(_slow_catalog(0.6, 0.2))
        service = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            **knobs), spool_root)
        service.start()
        try:
            tensors = _stage(spool_root, range(4))  # two packs
            log = _sample_during(lambda: service.flush(60.0), service,
                                 client, catalog.table_prefix, every=0.2)
            snapshot = service.snapshot()
            service.rethrow_if_failed()
        finally:
            switch.close()
            service.stop()

        assert {state for _, state, _ in log} == {"held"}, log
        assert not _held_but_dead(log), log
        assert snapshot["indexed_packs"] == 2, snapshot
        assert snapshot["lease_timeouts"] == 0, snapshot
        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert sorted(captures) == sorted(tensors)


def test_a_catalog_too_slow_to_keep_its_lease_says_so(fake_s3, tmp_path):
    """The lease-timeout streak reset on every successful claim, and counted
    only claims and renewals that timed out. Against a catalog too slow to
    keep a lease -- each claim goes through, then a renewal or a request of
    the pass runs out of lease, the writer quarantines, and the next claim
    goes through again -- the count went 1, 0, 1, 0 and the knobs were
    never named. Every timeout that costs the lease now counts, and the
    count clears only once a lease has been held for 2 x TTL. Scaled 1/5:
    0.9 s INSERTs and 0.4 s reads against a 3 s TTL."""
    spool_root = tmp_path / "spool"
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1, clickhouse_request_timeout_s=20.0)
        warm = _service(_storage_config(fake_s3, catalog.table_prefix,
                                        **knobs), tmp_path / "warm")
        warm.start()
        warm.stop()

        switch.delay_requests(_slow_catalog(0.9, 0.4))
        service = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            **knobs), spool_root)
        service.start()
        states = []
        stop = threading.Event()

        def _sample():
            while not stop.is_set():
                states.append(service.snapshot()["lease_state"])
                time.sleep(0.1)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()
        try:
            _stage(spool_root, range(4))

            def _flush():
                try:
                    service.flush(40.0)  # never drains; stop() ends it
                except Exception:  # noqa: BLE001 -- "not started" once stopped
                    pass

            flusher = threading.Thread(target=_flush, daemon=True)
            flusher.start()
            _wait_for(lambda: service.snapshot()["lease_timeout_error"],
                      timeout_s=30.0)
            snapshot = service.snapshot()
        finally:
            stop.set()
            switch.close()
            service.stop()
            sampler.join(timeout=5)

        # The claims went through: this is the lease lost after each.
        assert "held" in states and "quarantined" in states, states
        assert snapshot["lease_timeouts"] >= 3, snapshot
        assert snapshot["failed"] is False, snapshot
        for knob in ("lease_ttl_s", "clock_skew_s",
                     "clickhouse_request_timeout_s", "lease_ttl_s / 3"):
            assert knob in snapshot["lease_timeout_error"], snapshot
        # last_error is whatever failed last -- the knobs when the count
        # reached three, the pass the lost lease failed a moment later.
        assert "lease" in snapshot["last_error"], snapshot


def _lease_insert(request: bytes) -> bool:
    body = request.partition(b"\r\n\r\n")[2]
    return body.startswith(b"INSERT") and b"_publisher_lease` (term" in body


def test_a_lease_lost_while_start_reconciles_is_retaken_and_the_pass_rerun(
        fake_s3, tmp_path):
    """A lease lost while start() reconciled failed start(): the pass's
    commit found no lease and start() rethrew, although a quarantine is the
    recoverable kind everywhere else. Here the lease thread's renewal stalls
    while the listing is slow, so the lease quarantines before the
    crash-window pack is found. start() now returns, and the loop takes a
    fresh lease once the quarantine ends and runs the reconcile it owes."""
    spool_root = tmp_path / "spool"
    tensors = _stage(spool_root, range(2))
    _upload_behind_the_service(spool_root, fake_s3)
    s3 = _Switch.to_url(fake_s3)
    catalog_switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(lease_ttl_s=3.0, publish_timeout_s=1,
                     clickhouse_request_timeout_s=20.0)
        # The schema first, directly, so that the service's claim is the
        # first lease INSERT through the switch.
        warm = _service(_storage_config(fake_s3, catalog.table_prefix,
                                        reconcile_on_start=False, **knobs),
                        tmp_path / "warm")
        warm.start()
        warm.stop()

        seen = []

        def _renewal(request: bytes) -> bool:
            if not _lease_insert(request):
                return False
            seen.append(request)
            return len(seen) > 1  # the claim goes through, renewals stall

        catalog_switch.stall_requests(_renewal)
        s3.slow_once(_listing, 2.5)  # past the first renewal, due at 1 s
        service = _service(_storage_config(
            s3.url, catalog.table_prefix,
            clickhouse_port=catalog_switch.port, **knobs), spool_root)
        service.start()
        try:
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "quarantined", snapshot
            assert snapshot["indexed_packs"] == 0, snapshot
            assert "reconcile at start lost the publisher lease" in \
                snapshot["last_error"], snapshot

            catalog_switch.restore()
            _wait_for(lambda: service.snapshot()["reconciled_packs"] == 1,
                      timeout_s=15.0)
            service.flush(30.0)
            snapshot = service.snapshot()
            assert snapshot["indexed_packs"] == 1, snapshot
            assert snapshot["lease_state"] == "held", snapshot
            service.rethrow_if_failed()
        finally:
            service.stop()
            catalog_switch.close()
            s3.close()

        captures = _read_all(_storage_config(fake_s3, catalog.table_prefix))
        assert sorted(captures) == sorted(tensors)


def test_lease_requests_that_keep_timing_out_name_the_knobs_that_bound_them(
        fake_s3, tmp_path):
    """On a catalog too slow for its lease the service went round -- a claim
    timed out, quarantined, was refused by its own late row, started over --
    and all the snapshot ever said was "curl: Timeout was reached". Once
    three lease requests in a row time out, the snapshot names the bound
    and the knobs that set it, until a lease has been held for 2 x TTL
    again."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            reconcile_on_start=False, lease_ttl_s=3.0, publish_timeout_s=1,
            clickhouse_request_timeout_s=20.0)
        service = _service(config, tmp_path / "spool")
        service.start()
        try:
            snapshot = service.snapshot()
            assert snapshot["lease_timeouts"] == 0, snapshot
            assert snapshot["lease_timeout_error"] == "", snapshot

            # The renewal times out at the lease deadline, then every claim
            # at its own bound.
            switch.stall()
            _wait_for(lambda: service.snapshot()["lease_timeout_error"],
                      timeout_s=15.0)
            snapshot = service.snapshot()
            assert snapshot["lease_timeouts"] >= 3, snapshot
            assert snapshot["failed"] is False, snapshot
            for text in (snapshot["lease_timeout_error"],
                         snapshot["last_error"]):
                for knob in ("lease_ttl_s", "clickhouse_request_timeout_s",
                             "lease_ttl_s / 3"):
                    assert knob in text, snapshot

            switch.restore()
            _wait_for(lambda: service.snapshot()["lease_timeout_error"] == "",
                      timeout_s=15.0)
            snapshot = service.snapshot()
            assert snapshot["lease_state"] == "held", snapshot
            assert snapshot["lease_timeouts"] == 0, snapshot
            service.rethrow_if_failed()
        finally:
            switch.close()
            service.stop()


def test_the_lease_statements_carry_a_server_side_cap(fake_s3, tmp_path):
    """The claim INSERT went out with no max_execution_time, unlike every
    fenced publish statement. A request the client gave up on could still
    land its claim row later -- after the quarantine that was meant to
    outlive it had ended. The cap makes the server abandon the INSERT no
    later than the client does: the time left before the request's
    deadline, to the millisecond, and lock_acquire_timeout the same."""
    with _catalog() as (client, catalog):
        # The production knobs: a 15 s TTL.
        config = _storage_config(fake_s3, catalog.table_prefix,
                                 reconcile_on_start=False, holder="cap-test")
        service = _service(config, tmp_path / "spool")
        service.start()
        service.stop()  # the release tombstone is a lease INSERT too

        rows = []
        for _ in range(25):
            client.execute("SYSTEM FLUSH LOGS")
            rows = client.execute(
                "SELECT query, Settings['max_execution_time'], "
                "Settings['timeout_overflow_mode'], "
                "Settings['lock_acquire_timeout'] FROM system.query_log "
                "WHERE type = 'QueryFinish' AND query LIKE 'INSERT%' "
                f"AND query LIKE '%{catalog.table_prefix}_publisher_lease%'")
            ours = [query for query, *_ in rows if "'cap-test'" in query]
            if len(ours) >= 2:
                break
            time.sleep(0.2)
        # The service's claim and its tombstone, beside the schema check's.
        assert len(ours) == 2, rows
        for query, cap, overflow, lock in rows:
            if "now_ns + toUInt64" in query:
                # A claim with no lease held: min(clickhouse_request_timeout_s,
                # lease_ttl_s / 3) = 5 s, less the microseconds since.
                assert 4.9 < float(cap) <= 5.0, (query, cap)
            else:
                # A tombstone, under the lease: what is left of 15 s less
                # the 0.1 s margin since the claim was sent.
                assert 12.0 < float(cap) < 14.9, (query, cap)
            assert lock == cap, (query, lock)
            # The log lists only settings that differ from the default, and
            # throw is the default; break would insert what had been read.
            assert overflow in ("", "throw"), (query, overflow)


def _latch_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if "indexing stopped" in line]


def test_a_rival_that_takes_over_during_a_cut_still_latches(
        fake_s3, tmp_path, capfd):
    """Recovery must not paper over a real takeover: when another publisher
    holds the catalog for longer than two lease TTLs, the service stops,
    says so once on stderr, and flush raises naming the holder."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1)
        first = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            holder="first-publisher", **knobs), tmp_path / "first")
        rival = _service(_storage_config(
            fake_s3, catalog.table_prefix, holder="rival-publisher",
            start_lease_wait_s=10.0, **knobs), tmp_path / "rival")
        first.start()
        try:
            switch.cut()
            rival.start()  # waits out the lease the cut first cannot renew
            rival_started = time.monotonic()
            switch.restore()

            _wait_for(lambda: first.snapshot()["failed"], timeout_s=20.0)
            latched_at = time.monotonic()
            snapshot = first.snapshot()
            assert snapshot["running"] is False, snapshot
            assert snapshot["lease_state"] == "failed", snapshot
            assert "rival-publisher" in snapshot["last_error"], snapshot
            # Only a foreign lease that outlives two TTLs latches.
            assert latched_at - rival_started >= 2 * 3.0 - 0.5
            with pytest.raises(RuntimeError, match="rival-publisher"):
                first.flush(1.0)
            with pytest.raises(RuntimeError, match="rival-publisher"):
                first.rethrow_if_failed()

            rival.flush(10.0)
            rival_snapshot = rival.snapshot()
            assert rival_snapshot["lease_state"] == "held", rival_snapshot
            assert rival_snapshot["failed"] is False, rival_snapshot
        finally:
            first.stop()
            rival.stop()
            switch.close()

    lines = _latch_lines(capfd.readouterr().err)
    assert len(lines) == 1, lines
    assert "rival-publisher" in lines[0]


def test_a_rival_that_stops_within_two_ttls_does_not_latch(
        fake_s3, tmp_path, capfd):
    """A foreign lease that is gone again before two TTLs pass is a
    handover, not a takeover: the service takes the lease back."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    spool_root = tmp_path / "first"
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1)
        config = _storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            holder="first-publisher", **knobs)
        first = _service(config, spool_root)
        rival = _service(_storage_config(
            fake_s3, catalog.table_prefix, holder="rival-publisher",
            start_lease_wait_s=10.0, **knobs), tmp_path / "rival")
        first.start()
        try:
            switch.cut()
            rival.start()
            switch.restore()
            time.sleep(2.0)
            rival.stop()  # releases with a tombstone

            tensors = _stage(spool_root, range(2))
            first.flush(20.0)
            snapshot = first.snapshot()
        finally:
            first.stop()
            rival.stop()
            switch.close()

        assert snapshot["failed"] is False, snapshot
        assert snapshot["lease_state"] == "held", snapshot
        assert snapshot["lease_reacquisitions"] >= 1, snapshot
        assert sorted(_read_all(_storage_config(
            fake_s3, catalog.table_prefix))) == sorted(tensors)
    assert _latch_lines(capfd.readouterr().err) == []


def test_a_late_landing_claim_of_its_own_does_not_latch_the_service(
        fake_s3, tmp_path, capfd):
    """The lease INSERT's server-side cap cannot cover a request held up in
    transit: the server starts the clock only when the statement reaches
    it. Such a claim lands after the client gave up and quarantined, and its
    row then refuses the service's own next claim. Refusals by a rival that
    had since left and by that row of our own added up to 2 x TTL, and the
    service latched "held by another publisher" against itself. A refusal
    by the service's own claim row now restarts the refusal clock."""
    switch = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
    with _catalog() as (_client, catalog):
        knobs = dict(reconcile_on_start=False, lease_ttl_s=3.0,
                     publish_timeout_s=1)
        first = _service(_storage_config(
            fake_s3, catalog.table_prefix, clickhouse_port=switch.port,
            holder="first-publisher", **knobs), tmp_path / "first")
        rival = _service(_storage_config(
            fake_s3, catalog.table_prefix, holder="rival-publisher",
            start_lease_wait_s=10.0, **knobs), tmp_path / "rival")
        first.start()
        try:
            # A cut quarantines the first service; the rival takes over.
            switch.cut()
            rival.start()
            switch.restore()
            # The first service's quarantine ends and the rival refuses it:
            # the refusal clock starts.
            _wait_for(lambda: first.snapshot()["lease_state"] == "reacquiring",
                      timeout_s=10.0)
            refused_at = time.monotonic()
            # Well inside 2 x TTL of refusals the rival leaves (a handover),
            # and the first service's next claim is delivered 1.5 s late:
            # the 1 s bound on a claim made without a lease (lease_ttl_s / 3)
            # gives up first, and it quarantines.
            time.sleep(max(0.0, refused_at + 4.0 - time.monotonic()))
            switch.deliver_lease_inserts_late(1.5)
            rival.stop()
            _wait_for(lambda: first.snapshot()["lease_state"] == "quarantined",
                      timeout_s=3.0)
            switch.restore()
            # The late claim lands inside the quarantine and outlives it by
            # about a second, refusing the first service's next claim more
            # than 2 x TTL after the rival's first refusal.
            _wait_for(lambda: first.snapshot()["lease_state"] == "held"
                      or first.snapshot()["failed"], timeout_s=10.0)
            snapshot = first.snapshot()
            assert time.monotonic() - refused_at > 2 * 3.0, snapshot
            assert snapshot["failed"] is False, snapshot
            assert snapshot["lease_state"] == "held", snapshot
            first.rethrow_if_failed()
        finally:
            first.stop()
            rival.stop()
            switch.close()

    assert _latch_lines(capfd.readouterr().err) == []


_HOLD_LEASE = """
import json, sys, time
from dmi.storage.native_capture import (
    NativeCaptureStorage, NativeCaptureStorageConfig)

service = NativeCaptureStorage(
    NativeCaptureStorageConfig(**json.loads(sys.argv[1])),
    spool_root=sys.argv[2], spool_max_bytes=1 << 40, sweep_spool=True)
service.start()
print("started", flush=True)
time.sleep(600)
"""


def test_a_restart_within_the_ttl_of_a_killed_predecessor_succeeds(
        fake_s3, tmp_path):
    """A SIGKILLed publisher leaves its lease live for up to a TTL, and a
    restart in that window failed at once with the lease held. start() now
    waits, within start_lease_wait_s, for the lease to expire."""
    import os
    import signal
    import sys

    with _catalog() as (_client, catalog):
        knobs = dict(
            s3_endpoint=fake_s3, s3_bucket=BUCKET, s3_region=REGION,
            s3_access_key=ACCESS, s3_secret_key=SECRET,
            s3_allow_insecure_http=True, clickhouse_host=CLICKHOUSE_HOST,
            clickhouse_port=CLICKHOUSE_HTTP_PORT, database=DATABASE,
            table_prefix=catalog.table_prefix, reconcile_on_start=False,
            lease_ttl_s=5.0, publish_timeout_s=1)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src")] + ([env["PYTHONPATH"]]
                                   if env.get("PYTHONPATH") else []))
        predecessor = subprocess.Popen(
            [sys.executable, "-c", _HOLD_LEASE,
             json.dumps(dict(knobs, holder="crashed-publisher")),
             str(tmp_path / "predecessor")],
            stdout=subprocess.PIPE, text=True, env=env)
        try:
            assert predecessor.stdout.readline().strip() == "started"
        finally:
            predecessor.send_signal(signal.SIGKILL)
            predecessor.wait(timeout=30)
        assert predecessor.returncode == -signal.SIGKILL

        from dmi.storage.native_capture import NativeCaptureStorageConfig

        successor = _service(NativeCaptureStorageConfig(
            **knobs, holder="restarted-publisher", start_lease_wait_s=8.0),
            tmp_path / "successor")
        started = time.monotonic()
        successor.start()
        elapsed = time.monotonic() - started
        try:
            snapshot = successor.snapshot()
        finally:
            successor.stop()

    assert snapshot["running"] is True, snapshot
    assert snapshot["lease_state"] == "held", snapshot
    # It really waited on the dead holder's lease, and no longer than it.
    assert 1.0 < elapsed < 8.0, elapsed


def test_a_start_refused_the_lease_leaves_the_spool_unswept(fake_s3, tmp_path):
    """start() swept the spool before taking the lease, so a second process
    pointed at a live process's spool deleted its in-progress .open files
    and only then found the catalog held. The lease now comes first."""
    with _catalog() as (_client, catalog):
        first = _service(_storage_config(
            fake_s3, catalog.table_prefix, reconcile_on_start=False),
            tmp_path / "first")
        spool_root = tmp_path / "second"
        spool_root.mkdir()
        in_progress = spool_root / f".{uuid.uuid4()}.1234.open"
        in_progress.write_bytes(b"a pack being written")
        second = _service(_storage_config(
            fake_s3, catalog.table_prefix, reconcile_on_start=False,
            start_lease_wait_s=0.0), spool_root)
        first.start()
        try:
            with pytest.raises(RuntimeError, match="held"):
                second.start()
            assert in_progress.exists()
        finally:
            first.stop()
        second.start()  # the lease is free now, and the sweep runs
        second.stop()
        assert not in_progress.exists()


# --- https with a private CA ----------------------------------------------------


def test_a_64_mib_pack_over_https_with_a_private_ca_hydrates_exactly(
        fake_s3_tls, private_ca, tmp_path):
    """s3_ca_file reaches both halves: the service's upload and the reader.

    17 x 4 MiB records stage as one pack over the client's 64 MiB multipart
    threshold. The service uploads it over TLS to a store whose certificate
    only the private CA vouches for, indexes it, and the reader hydrates
    every capture byte-equal over the same TLS. A reader without the CA is
    refused by the store's certificate.
    """
    import random

    from dmi.storage.capture import CaptureMetadata

    ca_file, _ca_path, _cert, _key = private_ca
    records, record_bytes = 17, 4 << 20
    spool_root = tmp_path / "spool"
    payloads = {}
    sink = _Driver(SINK_DRIVER)
    try:
        assert sink.call(
            op="open", root=str(spool_root), max_bytes=1 << 40,
            max_queue_records=records, max_queue_bytes=2 * records * record_bytes,
            max_pack_bytes=2 * records * record_bytes,
            max_pack_records=records, max_linger_ns=60_000_000_000,
            overload="drop_newest", admission_timeout=-1)["ok"]
        for index in range(records):
            metadata = CaptureMetadata(
                capture_id=f"tls-{index:04d}", tenant_id="t",
                experiment_id="e", run_id="r", session_id="s",
                request_id=f"q{index}", sequence_id=f"n{index}",
                model_id="m", model_revision="mr", adapter_revision=None,
                capture_policy_version="v", hook_name="resid_post",
                layer_number=0, producer_rank=0, step_number=index,
                token_start=index, token_end=index + 1, batch_position=0,
                dtype="uint8", shape=(record_bytes,),
                captured_at_ns=1_700_000_000_000_000_000 + index)
            payload = random.Random(index).randbytes(record_bytes)
            response = sink.call(op="submit", metadata=metadata.to_mapping(),
                                 payload_b64=base64.b64encode(payload).decode())
            assert response["admission"] == "accepted", response
            payloads[metadata.capture_id] = payload
        assert sink.call(op="flush", timeout=60)["ok"]
        assert sink.call(op="close", timeout=60)["snapshot"][
            "persisted_records"] == records
    finally:
        sink.close()
    [pack] = _ready(spool_root)
    assert pack.stat().st_size >= 64 << 20

    with _catalog() as (_client, catalog):
        config = _storage_config(fake_s3_tls, catalog.table_prefix,
                                 s3_allow_insecure_http=False,
                                 s3_ca_file=ca_file)
        service = _service(config, spool_root)
        service.start()
        try:
            service.flush(60.0)
            snapshot = service.snapshot()
        finally:
            service.stop()
        assert snapshot["uploaded_packs"] == 1, snapshot
        assert snapshot["indexed_rows"] == records, snapshot
        parts = [call for call in STATE.calls
                 if call["method"] == "PUT" and "partNumber=" in call["path"]]
        assert len(parts) >= 5, len(parts)  # multipart, 16 MiB parts

        reader = _reader(config)
        selection = reader.select(tenant_id="t")
        captures = {capture.descriptor["capture_id"]: capture.payload
                    for capture in reader.read(selection, byte_limit=1 << 30)}
        assert sorted(captures) == sorted(payloads)
        for capture_id, payload in payloads.items():
            assert captures[capture_id] == payload, capture_id

        untrusted = _reader(_storage_config(
            fake_s3_tls, catalog.table_prefix, s3_allow_insecure_http=False))
        with pytest.raises(Exception, match="(?i)certificate"):
            untrusted.read(untrusted.select(tenant_id="t"),
                           byte_limit=1 << 30)
