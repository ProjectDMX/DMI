"""The engine's wiring of the native capture storage service.

The service itself is C++ and runs against a real object store and catalog
in test_native_capture_storage_live.py. This suite pins what the engine
promises around it, with the native modules faked: the service starts
before the sink opens the spool it sweeps, ``flush_and_wait`` waits for the
catalog as well as the spool, ``close`` drains and releases the lease, and a
failed attach does not leave the lease held.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from dmi.engine import MonitoringEngine

pytestmark = pytest.mark.cpu


def _storage_config(**overrides):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    fields = dict(
        s3_endpoint="https://s3.example.test", s3_bucket="bucket",
        s3_access_key="AKIA-test", s3_secret_key="secret-test",
    )
    fields.update(overrides)
    return NativeCaptureStorageConfig(**fields)


# --- configuration -----------------------------------------------------------


def test_storage_config_needs_the_sink_config_whose_spool_it_drains():
    from dmi.config import MonitoringConfig

    with pytest.raises(ValueError, match="capture_sink_config"):
        MonitoringConfig(storage_backend="persistent",
                         capture_storage_config=_storage_config())


def test_storage_config_is_accepted_beside_a_sink_config(tmp_path):
    from dmi.config import MonitoringConfig
    from dmi.storage.capture.native_sink import NativeSinkConfig

    storage = _storage_config()
    config = MonitoringConfig(
        storage_backend="persistent",
        capture_sink_config=NativeSinkConfig(spool_root=str(tmp_path)),
        capture_storage_config=storage,
    )
    assert config.capture_storage_config is storage


def test_plain_http_endpoint_must_be_opted_into():
    with pytest.raises(ValueError, match="s3_allow_insecure_http"):
        _storage_config(s3_endpoint="http://127.0.0.1:3900")
    config = _storage_config(s3_endpoint="http://127.0.0.1:3900",
                             s3_allow_insecure_http=True)
    assert config.s3_endpoint == "http://127.0.0.1:3900"


def test_credentials_stay_out_of_the_repr():
    text = repr(_storage_config(s3_session_token="token-test"))
    assert "secret-test" not in text
    assert "AKIA-test" not in text
    assert "token-test" not in text


@pytest.mark.parametrize("name", ["s3_bucket", "s3_secret_key", "table_prefix"])
def test_required_text_fields_refuse_empty(name):
    with pytest.raises(ValueError, match=name):
        _storage_config(**{name: ""})


def test_a_poll_interval_below_a_millisecond_is_refused():
    # int(poll_interval_s * 1e9) is the native wait; a value that rounds to
    # a zero wait would spin the service thread.
    with pytest.raises(ValueError, match="poll_interval_s"):
        _storage_config(poll_interval_s=1e-10)
    with pytest.raises(ValueError, match="poll_interval_s"):
        _storage_config(poll_interval_s=0.0009)
    assert _storage_config(poll_interval_s=0.001).poll_interval_s == 0.001


@pytest.mark.parametrize("name", [
    "poll_interval_s", "reconcile_interval_s", "close_flush_timeout_s",
    "clickhouse_connect_timeout_s", "clickhouse_request_timeout_s"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_intervals_and_timeouts_must_be_finite(name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        _storage_config(**{name: value})


@pytest.mark.parametrize("endpoint", [
    "s3.example.test", "127.0.0.1:3900", "ftp://s3.example.test",
    "HTTPS://s3.example.test"])
def test_the_endpoint_needs_an_http_or_https_scheme(endpoint):
    # The native client treats a scheme-less endpoint as plain HTTP, then
    # refuses it at every request as insecure.
    with pytest.raises(ValueError, match="http:// or https://"):
        _storage_config(s3_endpoint=endpoint)


def test_https_with_the_insecure_flag_is_refused():
    # The native client refuses every request of that pairing: the flag
    # admits plain http:// endpoints and never downgrades TLS.
    with pytest.raises(ValueError, match="s3_allow_insecure_http"):
        _storage_config(s3_endpoint="https://s3.example.test",
                        s3_allow_insecure_http=True)


def _fake_reader(monkeypatch):
    from dmi.storage import native_capture

    calls = []

    class _Reader:
        def __init__(self, config):
            pass

        def resolve(self, selection):
            calls.append("resolve")
            return []

        def hydrate(self, selection, byte_limit, request_limit):
            calls.append("hydrate")
            return []

    monkeypatch.setattr(
        native_capture, "_load_native_store_extension",
        lambda: SimpleNamespace(CaptureReader=_Reader, SEARCH_ITEM_COLUMNS=()))
    selection = native_capture.NativeCaptureSelection(
        selection_id="s", capture_ids=(), catalog_watermark="w",
        filter_hash="f", tenant_id="t")
    return native_capture.NativeCaptureReader(_storage_config()), selection, calls


@pytest.mark.parametrize("limits, match", [
    (dict(byte_limit=-1), "byte_limit"),
    (dict(byte_limit=1 << 20, request_limit=0), "request_limit"),
    (dict(byte_limit=1 << 20, request_limit=-3), "request_limit"),
])
def test_read_refuses_bad_limits_before_any_request(monkeypatch, limits, match):
    reader, selection, calls = _fake_reader(monkeypatch)
    with pytest.raises(ValueError, match=match):
        reader.read(selection, **limits)
    assert calls == []


def test_read_accepts_a_zero_byte_limit(monkeypatch):
    reader, selection, calls = _fake_reader(monkeypatch)
    assert reader.read(selection, byte_limit=0) == ()
    assert calls == ["resolve", "hydrate"]


def test_engine_refuses_a_storage_config_of_the_wrong_type():
    config = SimpleNamespace(storage_backend="persistent", capture_sink_config=None,
                             capture_storage_config={"s3_bucket": "b"})
    with pytest.raises(TypeError, match="NativeCaptureStorageConfig"):
        MonitoringEngine(config=config, enable_ring_transport=False)


# --- the engine around a faked service ----------------------------------------


class _FakeService:
    """Stands in for _dmi_native_store.StorageService."""

    def __init__(self, events, config):
        self.events = events
        self.config = config
        self.flush_results = []
        self.flush_error = None
        events.append(("service", "construct"))

    def start(self):
        self.events.append(("service", "start"))

    def flush(self, timeout_s):
        self.events.append(("service", "flush", timeout_s))
        if self.flush_error is not None:
            raise self.flush_error
        return self.flush_results.pop(0) if self.flush_results else True

    def stop(self):
        self.events.append(("service", "stop"))

    def snapshot(self):
        return {"pending_index": 2, "last_error": "index failed: boom"}

    def rethrow_if_failed(self):
        pass


def _capture_engine(monkeypatch, tmp_path, *, fail_ring=False):
    """An engine under storage_backend="persistent" with both native modules
    faked. Returns (engine, events, services)."""
    from dmi.storage.capture.native_sink import NativeSinkConfig

    events, services = [], []
    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = SimpleNamespace(null_offload=False,
                                             force_eager=False)
    engine._ring_engine = SimpleNamespace(stop=lambda: None)
    engine._ring_config = object()
    engine._storage_backend = "persistent"
    engine._capture_sink_config = NativeSinkConfig(
        spool_root=str(tmp_path / "spool"), spool_max_bytes=1 << 30)
    engine._capture_storage_config = _storage_config()

    class _Lease:
        def release(self):
            pass

    class _RecordSink:
        def _acquire_engine(self):
            return _Lease()

    class _NativePackSink(_RecordSink):
        def __init__(self, **kwargs):
            events.append(("sink", "open", kwargs["spool_root"]))

    def _service(config):
        service = _FakeService(events, config)
        services.append(service)
        return service

    def _load_named_extension(name):
        if name == "_dmi_native_store":
            return SimpleNamespace(StorageService=_service,
                                   SEARCH_ITEM_COLUMNS=())
        return SimpleNamespace(NativePackSink=_NativePackSink)

    class _NewRing:
        def payload_tensor(self):
            return object()

        def init(self):
            if fail_ring:
                raise RuntimeError("ring init failed")

        def start(self):
            pass

        def stop(self):
            events.append(("ring", "stop"))

    class _FakeTransport:
        def __init__(self, native_ring):
            self.null_offload = False
            self.force_eager = False

        def configure_record_schema(self, schema):
            pass

        def flush_records_and_wait(self, timeout_s):
            events.append(("sink", "flush", timeout_s))

    native = ModuleType("dmi.transport.native")
    native.RecordSink = _RecordSink
    native._load_named_extension = _load_named_extension
    class _RingEngine(_NewRing):
        """enable_ring_transport's plain ring; create_record's record ring."""

        def __init__(self, config, host):
            events.append(("ring", "create"))

        create_record = staticmethod(lambda config, target: _NewRing())

    native.RingEngine = _RingEngine
    ring = ModuleType("dmi.transport.ring")
    ring.RingTransport = _FakeTransport
    ring.activate = lambda transport: None
    ring.deactivate = lambda: None
    monkeypatch.setitem(sys.modules, "dmi.transport.native", native)
    monkeypatch.setitem(sys.modules, "dmi.transport.ring", ring)
    import dmi.transport

    monkeypatch.setattr(dmi.transport, "native", native, raising=False)
    return engine, events, services


def _record_format():
    from dmi.records import RecordCellType, RecordColumn, RecordLayout, RecordSchema

    class _Format:
        schema = RecordSchema((RecordLayout(
            "events", "events",
            (RecordColumn("event_id", RecordCellType.INT64),),
            primary_key=("event_id",), order_by=("event_id",)),))

        def encode(self, metadata, entry):
            raise AssertionError("not part of runtime construction")

    return _Format()


def test_the_service_starts_before_the_sink_opens_the_spool(monkeypatch, tmp_path):
    engine, events, services = _capture_engine(monkeypatch, tmp_path)

    engine.create_record_runtime(_record_format())

    spool_root = str(tmp_path / "spool")
    assert events[:3] == [
        ("service", "construct"),
        ("service", "start"),
        ("sink", "open", spool_root),
    ]
    config = services[0].config
    assert config["spool_root"] == spool_root
    assert config["spool_max_bytes"] == 1 << 30
    assert config["sweep_spool_on_start"] is True
    assert config["holder"]  # a generated lease holder, never empty


def test_an_explicit_sink_leaves_the_spool_unswept(monkeypatch, tmp_path):
    engine, events, services = _capture_engine(monkeypatch, tmp_path)
    import dmi.transport

    engine.create_record_runtime(
        _record_format(), record_sink=dmi.transport.native.RecordSink())

    assert services[0].config["sweep_spool_on_start"] is False


def test_a_failed_attach_stops_the_service_it_started(monkeypatch, tmp_path):
    engine, events, _services = _capture_engine(monkeypatch, tmp_path,
                                                fail_ring=True)

    with pytest.raises(RuntimeError, match="ring init failed"):
        engine.create_record_runtime(_record_format())

    assert ("service", "stop") in events
    assert engine._capture_storage is None


def test_flush_waits_for_the_catalog_after_the_sink(monkeypatch, tmp_path):
    engine, events, _services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    events.clear()

    engine.flush_and_wait(30.0)

    assert [event[:2] for event in events] == [
        ("sink", "flush"), ("service", "flush")]
    assert 0.0 <= events[1][2] <= 30.0


def test_flush_reports_packs_that_did_not_reach_the_catalog(monkeypatch, tmp_path):
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    services[0].flush_results = [False]

    with pytest.raises(TimeoutError, match="2 uploaded but unindexed.*boom"):
        engine.flush_and_wait(1.0)


def test_close_flushes_the_sink_before_the_ring_stops(monkeypatch, tmp_path):
    """The sink's open pack is in memory until a flush seals it, and stopping
    the ring releases the sink without one. So close() flushes the sink
    first; only then can draining the service reach the tail."""
    engine, events, _services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    events.clear()

    engine.close()

    assert [event[:2] for event in events] == [
        ("sink", "flush"), ("ring", "stop"),
        ("service", "flush"), ("service", "stop")]
    # One budget for the whole drain: the service gets what the sink left.
    assert 59.0 <= events[0][2] <= 60.0
    assert 0.0 <= events[2][2] <= 60.0
    assert engine._capture_storage is None


def test_close_still_stops_when_the_sink_flush_fails(monkeypatch, tmp_path):
    engine, events, _services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())

    def _failing_flush(timeout_s):
        events.append(("sink", "flush", timeout_s))
        raise TimeoutError("timed out waiting for durable record completion")

    engine._ring_transport.flush_records_and_wait = _failing_flush
    events.clear()

    engine.close()

    assert [event[:2] for event in events] == [
        ("sink", "flush"), ("ring", "stop"),
        ("service", "flush"), ("service", "stop")]


def test_close_releases_the_lease_even_when_the_drain_fails(monkeypatch, tmp_path):
    engine, events, services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    services[0].flush_error = RuntimeError("catalog unreachable")
    events.clear()

    engine.close()

    assert events[-1] == ("service", "stop")


def test_replacing_a_record_ring_drains_and_stops_the_service(
        monkeypatch, tmp_path):
    """enable_ring_transport over a record ring used to leave the service
    running and holding the catalog lease, with the sink unsealed: the open
    pack's records were dropped, and the next create_record_runtime built a
    second service that its own process's lease refused."""
    engine, events, services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    events.clear()

    engine.enable_ring_transport(object())

    assert [event[:2] for event in events] == [
        ("sink", "flush"), ("ring", "stop"),
        ("service", "flush"), ("service", "stop"), ("ring", "create")]
    assert 59.0 <= events[0][2] <= 60.0
    assert engine._capture_storage is None
    assert engine._record_mode is False

    # A second record runtime starts its own service; nothing still holds
    # the lease it takes.
    engine.create_record_runtime(_record_format())
    assert len(services) == 2
    assert engine._capture_storage is not None
