"""The engine's wiring of the native capture storage service.

The service itself is C++ and runs against a real object store and catalog
in test_native_capture_storage_live.py. This suite pins what the engine
promises around it, with the native modules faked: the engine takes its own
rank directory's owner lock before anything opens it, the service starts
before the sink opens the spool it sweeps, both open that directory
held_by_caller, ``flush_and_wait`` waits for the catalog as well as the
spool, ``close`` drains and releases the lease and only then the spool
lock, and a failed attach leaves neither the lease nor the lock held.
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


# --- the catalog connection: scheme, credentials, TLS --------------------------


def test_catalog_passwords_stay_out_of_the_repr():
    text = repr(_storage_config(
        clickhouse_scheme="https", clickhouse_user="writer",
        clickhouse_password="writer-pw-test",
        clickhouse_reader_user="reader",
        clickhouse_reader_password="reader-pw-test"))
    assert "writer-pw-test" not in text
    assert "reader-pw-test" not in text
    # The account names are not secrets, and help say which one failed.
    assert "clickhouse_user='writer'" in text


def test_the_catalog_defaults_are_the_local_plain_http_server():
    config = _storage_config()
    assert (config.clickhouse_scheme, config.clickhouse_host,
            config.clickhouse_port) == ("http", "127.0.0.1", 8123)
    native = config._native_dict()
    assert native["clickhouse_scheme"] == "http"
    assert native["clickhouse_user"] == ""
    assert native["clickhouse_password"] == ""
    assert native["clickhouse_allow_insecure_http"] is False


def test_the_catalog_connection_reaches_the_native_dict():
    config = _storage_config(
        clickhouse_scheme="https", clickhouse_host="ch.example.test",
        clickhouse_port=8443, clickhouse_user="writer",
        clickhouse_password="writer-pw", clickhouse_ca_file="/etc/ca.pem",
        clickhouse_ca_path="/etc/ca.d")
    native = config._native_dict()
    assert {k: native[k] for k in (
        "clickhouse_scheme", "clickhouse_host", "clickhouse_port",
        "clickhouse_user", "clickhouse_password", "clickhouse_ca_file",
        "clickhouse_ca_path", "clickhouse_allow_insecure_http")} == {
        "clickhouse_scheme": "https", "clickhouse_host": "ch.example.test",
        "clickhouse_port": 8443, "clickhouse_user": "writer",
        "clickhouse_password": "writer-pw", "clickhouse_ca_file": "/etc/ca.pem",
        "clickhouse_ca_path": "/etc/ca.d",
        "clickhouse_allow_insecure_http": False}
    # The reader account is the reader's alone; the service never sees it.
    assert not any(k.startswith("clickhouse_reader") for k in native)


def test_the_reader_account_replaces_the_writer_account_for_the_reader():
    config = _storage_config(
        clickhouse_scheme="https", clickhouse_user="writer",
        clickhouse_password="writer-pw", clickhouse_reader_user="reader",
        clickhouse_reader_password="reader-pw")
    native = config._native_reader_dict()
    assert (native["clickhouse_user"], native["clickhouse_password"]) == (
        "reader", "reader-pw")
    without = _storage_config(clickhouse_scheme="https",
                              clickhouse_user="writer",
                              clickhouse_password="writer-pw")
    native = without._native_reader_dict()
    assert (native["clickhouse_user"], native["clickhouse_password"]) == (
        "writer", "writer-pw")


def test_a_catalog_password_over_plain_http_must_be_opted_into():
    with pytest.raises(ValueError, match="clickhouse_allow_insecure_http"):
        _storage_config(clickhouse_user="writer", clickhouse_password="pw")
    with pytest.raises(ValueError, match="clickhouse_allow_insecure_http"):
        _storage_config(clickhouse_reader_user="reader",
                        clickhouse_reader_password="pw")
    config = _storage_config(clickhouse_user="writer",
                             clickhouse_password="pw",
                             clickhouse_allow_insecure_http=True)
    assert config.clickhouse_allow_insecure_http is True
    # A user name alone is not a secret: the local passwordless default.
    assert _storage_config(clickhouse_user="default").clickhouse_user == "default"


def test_the_catalog_insecure_flag_never_downgrades_https():
    with pytest.raises(ValueError, match="clickhouse_allow_insecure_http"):
        _storage_config(clickhouse_scheme="https",
                        clickhouse_allow_insecure_http=True)


@pytest.mark.parametrize("scheme", ["", "ftp", "HTTP", "https://", None])
def test_the_catalog_scheme_is_http_or_https(scheme):
    with pytest.raises((ValueError, TypeError), match="clickhouse_scheme"):
        _storage_config(clickhouse_scheme=scheme)


@pytest.mark.parametrize("host", [
    "writer:pw@ch.example.test", "@ch.example.test", "https://ch.example.test",
    "ch.example.test:8443", "ch.example.test/db", "ch.example.test?x=1",
    "ch.example.test#x", "ch example", "", "[::1",
])
def test_the_catalog_host_is_only_a_host(host):
    with pytest.raises(ValueError, match="clickhouse_host"):
        _storage_config(clickhouse_host=host)


def test_a_bracketed_ipv6_catalog_host_is_accepted():
    assert _storage_config(clickhouse_host="[::1]").clickhouse_host == "[::1]"


@pytest.mark.parametrize("name", ["clickhouse_ca_file", "clickhouse_ca_path"])
def test_a_catalog_ca_needs_https(name):
    with pytest.raises(ValueError, match=name):
        _storage_config(**{name: "/etc/ca.pem"})
    assert getattr(_storage_config(clickhouse_scheme="https",
                                   **{name: "/etc/ca.pem"}), name)


@pytest.mark.parametrize("fields, match", [
    (dict(clickhouse_password="pw"), "clickhouse_user"),
    (dict(clickhouse_reader_password="pw"), "clickhouse_reader_user"),
    (dict(clickhouse_user="a\nb"), "clickhouse_user"),
    (dict(clickhouse_password="a\r\nb", clickhouse_user="u"),
     "clickhouse_password"),
    (dict(clickhouse_reader_user="a\x00b"), "clickhouse_reader_user"),
])
def test_catalog_credentials_are_checked(fields, match):
    with pytest.raises(ValueError, match=match):
        _storage_config(clickhouse_scheme="https", **fields)


@pytest.mark.parametrize("name", [
    "clickhouse_user", "clickhouse_password", "clickhouse_ca_file",
    "clickhouse_reader_user", "clickhouse_reader_password"])
def test_catalog_text_options_must_be_text(name):
    with pytest.raises(TypeError, match=name):
        _storage_config(clickhouse_scheme="https", **{name: 7})


def test_the_request_timeout_rule_follows_the_configured_publish_timeout():
    # Twice publish_timeout_s, not twice a fixed 5 s: a 7 s publish cap needs
    # a 14 s request timeout, and 12 s would abandon a publish mid-flight.
    with pytest.raises(ValueError, match="twice publish_timeout_s"):
        _storage_config(publish_timeout_s=7.0,
                        clickhouse_request_timeout_s=12.0)
    assert _storage_config(
        publish_timeout_s=7.0,
        clickhouse_request_timeout_s=14.0).clickhouse_request_timeout_s == 14.0


def test_the_catalog_request_timeout_outlasts_a_publish():
    """A publish runs server-side for up to its 5 s publish timeout; a client
    that gives up sooner reports an outcome it does not know, and the writer
    quarantines itself over a statement that may well have committed."""
    with pytest.raises(ValueError, match="clickhouse_request_timeout_s"):
        _storage_config(clickhouse_request_timeout_s=9.9)
    assert _storage_config(
        clickhouse_request_timeout_s=10.0).clickhouse_request_timeout_s == 10.0


# --- a private CA for an https object store ----------------------------------


def test_a_private_ca_reaches_the_native_service_and_reader(monkeypatch):
    from dmi.storage import native_capture

    seen = []

    class _Native:
        def __init__(self, config):
            seen.append(dict(config))

    monkeypatch.setattr(
        native_capture, "_load_native_store_extension",
        lambda: SimpleNamespace(StorageService=_Native, CaptureReader=_Native,
                                SEARCH_ITEM_COLUMNS=()))
    config = _storage_config(s3_ca_file="/etc/dmi/ca.pem",
                             s3_ca_path="/etc/dmi/ca.d")
    native_capture.NativeCaptureStorage(config, spool_root="/tmp/spool",
                                        spool_max_bytes=1 << 30,
                                        sweep_spool=True)
    native_capture.NativeCaptureReader(config)
    assert [(d["s3_ca_file"], d["s3_ca_path"]) for d in seen] == \
        [("/etc/dmi/ca.pem", "/etc/dmi/ca.d")] * 2
    # Unset means libcurl's default trust store: empty, not absent.
    default = _storage_config()._native_dict()
    assert (default["s3_ca_file"], default["s3_ca_path"]) == ("", "")


@pytest.mark.parametrize("name", ["s3_ca_file", "s3_ca_path"])
def test_a_ca_on_a_plain_http_endpoint_is_refused(name):
    # It would read as "this is TLS" while credentials go in the clear; the
    # native client refuses the same pairing.
    with pytest.raises(ValueError, match=f"{name}.*https"):
        _storage_config(s3_endpoint="http://127.0.0.1:3900",
                        s3_allow_insecure_http=True, **{name: "/etc/dmi/ca"})


@pytest.mark.parametrize("name", ["s3_ca_file", "s3_ca_path"])
def test_a_ca_option_must_be_a_string(name):
    with pytest.raises(TypeError, match=name):
        _storage_config(**{name: None})


def _native_store_or_skip():
    from dmi.storage import native_capture

    try:
        return native_capture._load_native_store_extension()
    except ImportError:
        pytest.skip("_dmi_native_store is not built")


@pytest.mark.parametrize("name", ["s3_ca_file", "s3_ca_path"])
def test_the_native_module_names_a_missing_ca_at_construction(tmp_path, name):
    """The real bindings read the CA fields into the client's config.

    No store or catalog is contacted: the client's own validation refuses
    the path when the service or reader is built, not at the first upload.
    """
    module = _native_store_or_skip()
    missing = str(tmp_path / "no-such-ca")
    native = _storage_config(**{name: missing})._native_dict()
    with pytest.raises(ValueError, match="no-such-ca"):
        module.CaptureReader(native)
    with pytest.raises(ValueError, match="no-such-ca"):
        module.StorageService({**native, "spool_root": str(tmp_path / "spool")})


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


RANK_DIRECTORY = "{base}/0123456789ab/r{rank}-0a1b2c3d"


class _FakeSpoolLock:
    """Stands in for _dmi_native_store.SpoolOwnerLock."""

    def __init__(self, events, directory, allow_shared_filesystem=False):
        self.events = events
        self.directory = directory
        self.allow_shared_filesystem = allow_shared_filesystem
        self.held = True
        events.append(("lock", "acquire", directory))

    def release_and_remove_if_empty(self):
        self.held = False
        self.events.append(("lock", "release"))
        return True

    def release(self):
        self.release_and_remove_if_empty()


def _capture_engine(monkeypatch, tmp_path, *, fail_ring=False,
                    fail_start=False, spool_owner=None, storage=True):
    """An engine under storage_backend="persistent" with both native modules
    faked. Returns (engine, events, services); ``events`` also records the
    spool locks and the sinks' keyword arguments (``sinks``)."""
    from dmi.storage.capture.native_sink import NativeSinkConfig

    events, services = [], []
    events_sinks: list = []
    locks: list = []
    engine = MonitoringEngine(enable_ring_transport=False)
    engine._ring_transport = SimpleNamespace(null_offload=False,
                                             force_eager=False)
    engine._ring_engine = SimpleNamespace(stop=lambda: None)
    engine._ring_config = object()
    engine._storage_backend = "persistent"
    engine._capture_sink_config = NativeSinkConfig(
        spool_root=str(tmp_path / "spool"), spool_max_bytes=1 << 30)
    engine._capture_storage_config = _storage_config() if storage else None

    class _Lease:
        def release(self):
            pass

    class _RecordSink:
        def _acquire_engine(self):
            return _Lease()

    class _NativePackSink(_RecordSink):
        def __init__(self, **kwargs):
            events.append(("sink", "open", kwargs["spool_root"]))
            events_sinks.append(kwargs)

    def _service(config):
        service = _FakeService(events, config)
        if fail_start:
            def _refuse():
                events.append(("service", "start"))
                raise RuntimeError("publisher lease held elsewhere")
            service.start = _refuse
        services.append(service)
        return service

    def _lock(directory, allow_shared_filesystem=False):
        lock = _FakeSpoolLock(events, directory, allow_shared_filesystem)
        locks.append(lock)
        return lock

    def _rank_directory(base, database, table_prefix, store_id, rank):
        events.append(("layout", database, table_prefix, store_id, rank))
        return RANK_DIRECTORY.format(base=base, rank=rank)

    def _load_named_extension(name):
        if name == "_dmi_native_store":
            return SimpleNamespace(StorageService=_service,
                                   SpoolOwnerLock=_lock,
                                   spool_rank_directory=_rank_directory,
                                   spool_owner=lambda directory: spool_owner,
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

        create_record = staticmethod(
            lambda config, target, **_options: _NewRing())

    native.RingEngine = _RingEngine
    ring = ModuleType("dmi.transport.ring")
    ring.RingTransport = _FakeTransport
    ring.activate = lambda transport: None
    ring.deactivate = lambda: None
    monkeypatch.setitem(sys.modules, "dmi.transport.native", native)
    monkeypatch.setitem(sys.modules, "dmi.transport.ring", ring)
    import dmi.transport

    monkeypatch.setattr(dmi.transport, "native", native, raising=False)
    engine._test_sinks = events_sinks
    engine._test_locks = locks
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
    """The engine's own directory is locked first; the service starts (and
    sweeps it) before the sink opens it; both open it under that lock."""
    monkeypatch.delenv("RANK", raising=False)
    engine, events, services = _capture_engine(monkeypatch, tmp_path)

    engine.create_record_runtime(_record_format())

    directory = RANK_DIRECTORY.format(base=tmp_path / "spool", rank=0)
    storage = _storage_config()
    assert events[:5] == [
        ("layout", storage.database, storage.table_prefix, storage.store_id,
         0),
        ("lock", "acquire", directory),
        ("service", "construct"),
        ("service", "start"),
        ("sink", "open", directory),
    ]
    config = services[0].config
    assert config["spool_root"] == directory
    assert config["spool_max_bytes"] == 1 << 30
    assert config["sweep_spool_on_start"] is True
    assert config["spool_owner_lock"] == "held_by_caller"
    assert config["adopt_sibling_spools"] is True
    assert config["spool_allow_shared_filesystem"] is False
    assert config["holder"]  # a generated lease holder, never empty
    (sink,) = engine._test_sinks
    assert sink["owner_lock"] == "held_by_caller"
    assert engine._test_locks[0].held


def test_the_spool_directory_is_named_for_the_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "3")
    engine, events, services = _capture_engine(monkeypatch, tmp_path)

    engine.create_record_runtime(_record_format())

    assert services[0].config["spool_root"] == RANK_DIRECTORY.format(
        base=tmp_path / "spool", rank=3)


@pytest.mark.parametrize("rank", ["", "-1", "x", "1.5"])
def test_a_rank_that_is_not_a_rank_names_rank_zero(monkeypatch, tmp_path, rank):
    monkeypatch.setenv("RANK", rank)
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)

    engine.create_record_runtime(_record_format())

    assert services[0].config["spool_root"].endswith("/r0-0a1b2c3d")


def test_the_shared_filesystem_override_reaches_the_lock_and_both_spools(
        monkeypatch, tmp_path):
    from dmi.storage.capture.native_sink import NativeSinkConfig

    engine, _events, services = _capture_engine(monkeypatch, tmp_path)
    engine._capture_sink_config = NativeSinkConfig(
        spool_root=str(tmp_path / "spool"),
        spool_allow_shared_filesystem=True)

    engine.create_record_runtime(_record_format())

    assert engine._test_locks[0].allow_shared_filesystem is True
    assert services[0].config["spool_allow_shared_filesystem"] is True
    assert engine._test_sinks[0]["allow_shared_filesystem"] is True


def test_a_service_that_fails_to_start_releases_the_spool_lock(
        monkeypatch, tmp_path):
    engine, events, _services = _capture_engine(monkeypatch, tmp_path,
                                                fail_start=True)

    with pytest.raises(RuntimeError, match="lease held elsewhere"):
        engine.create_record_runtime(_record_format())

    assert events[-1] == ("lock", "release")
    assert not engine._test_locks[0].held
    assert engine._capture_storage is None
    assert not any(event[0] == "sink" for event in events)


def test_an_explicit_sink_leaves_the_spool_unswept(monkeypatch, tmp_path):
    engine, events, services = _capture_engine(monkeypatch, tmp_path)
    import dmi.transport

    engine.create_record_runtime(
        _record_format(), record_sink=dmi.transport.native.RecordSink())

    config = services[0].config
    assert config["sweep_spool_on_start"] is False
    # An explicit sink writes where it was built to: the configured root,
    # which the engine neither lays out nor adopts siblings around. Nothing
    # holds it here, so the service takes its lock.
    assert config["spool_root"] == str(tmp_path / "spool")
    assert config["adopt_sibling_spools"] is False
    assert config["spool_owner_lock"] == "take"
    assert not any(event[0] == "lock" for event in events)


def test_an_explicit_sink_holding_the_spool_shares_it_with_the_service(
        monkeypatch, tmp_path):
    """A NativePackSink the caller built takes the spool's owner lock; the
    engine's service in the same process must open beside it, not take it
    again (that would be refused, naming this very process)."""
    import os
    import socket

    engine, _events, services = _capture_engine(
        monkeypatch, tmp_path,
        spool_owner={"host": socket.gethostname(), "pid": os.getpid()})
    import dmi.transport

    engine.create_record_runtime(
        _record_format(), record_sink=dmi.transport.native.RecordSink())

    assert services[0].config["spool_owner_lock"] == "held_by_caller"


def test_an_explicit_sink_on_a_spool_another_process_owns_is_taken(
        monkeypatch, tmp_path):
    """Owned by another process: the service takes it, and so is refused
    by the native spool naming that holder."""
    engine, _events, services = _capture_engine(
        monkeypatch, tmp_path, spool_owner={"host": "elsewhere", "pid": 1})
    import dmi.transport

    engine.create_record_runtime(
        _record_format(), record_sink=dmi.transport.native.RecordSink())

    assert services[0].config["spool_owner_lock"] == "take"


def test_a_sink_without_a_service_owns_its_spool_itself(monkeypatch, tmp_path):
    """No capture_storage_config: packs stay in the spool for something else
    to drain, and the sink is the directory's one owner -- no layout, no
    engine lock."""
    engine, events, services = _capture_engine(monkeypatch, tmp_path,
                                               storage=False)

    engine.create_record_runtime(_record_format())

    assert services == []
    (sink,) = engine._test_sinks
    assert sink["spool_root"] == str(tmp_path / "spool")
    assert sink["owner_lock"] == "take"
    assert not any(event[0] in ("lock", "layout") for event in events)


def test_a_failed_attach_stops_the_service_it_started(monkeypatch, tmp_path):
    engine, events, _services = _capture_engine(monkeypatch, tmp_path,
                                                fail_ring=True)

    with pytest.raises(RuntimeError, match="ring init failed"):
        engine.create_record_runtime(_record_format())

    assert events[-2:] == [("service", "stop"), ("lock", "release")]
    assert engine._capture_storage is None
    assert not engine._test_locks[0].held


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


def test_flush_says_when_a_dead_spool_is_still_to_adopt(monkeypatch, tmp_path):
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    services[0].flush_results = [False]
    services[0].snapshot = lambda: {
        "pending_index": 0, "adoption_owed": True,
        "last_error": "adopting dead spool /x: upload failed"}

    with pytest.raises(TimeoutError,
                       match="dead process's spool still to adopt.*upload"):
        engine.flush_and_wait(1.0)


def test_close_flushes_the_sink_before_the_ring_stops(monkeypatch, tmp_path):
    """The sink's open pack is in memory until a flush seals it, and stopping
    the ring releases the sink without one. So close() flushes the sink
    first; only then can draining the service reach the tail."""
    engine, events, _services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    events.clear()

    engine.close()

    # The spool lock last: after the sink and the service are both done.
    assert [event[:2] for event in events] == [
        ("sink", "flush"), ("ring", "stop"),
        ("service", "flush"), ("service", "stop"), ("lock", "release")]
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
        ("service", "flush"), ("service", "stop"), ("lock", "release")]


def test_close_releases_the_lease_even_when_the_drain_fails(monkeypatch, tmp_path):
    engine, events, services = _capture_engine(monkeypatch, tmp_path)
    engine.create_record_runtime(_record_format())
    services[0].flush_error = RuntimeError("catalog unreachable")
    events.clear()

    engine.close()

    # Whatever did not drain stays in the directory for the next process on
    # the node to adopt; the lock goes either way.
    assert events[-2:] == [("service", "stop"), ("lock", "release")]


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
        ("service", "flush"), ("service", "stop"), ("lock", "release"),
        ("ring", "create")]
    assert 59.0 <= events[0][2] <= 60.0
    assert engine._capture_storage is None
    assert engine._record_mode is False

    # A second record runtime starts its own service in a directory of its
    # own; nothing still holds the lease it takes, or the lock.
    engine.create_record_runtime(_record_format())
    assert len(services) == 2
    assert engine._capture_storage is not None
    assert [lock.held for lock in engine._test_locks] == [False, True]


# --- the publisher lease knobs -------------------------------------------------


def test_lease_knobs_default_to_a_fifteen_second_lease():
    config = _storage_config()
    assert config.lease_ttl_s == 15.0
    assert config.publish_timeout_s == 5.0
    assert config.clock_skew_s == 0.0
    # None waits out a predecessor for the TTL plus the publish timeout.
    assert config.start_lease_wait_s is None


@pytest.mark.parametrize("ttl, publish, skew", [
    (5.0, 5.0, 0.0),    # no margin at all
    (5.09, 5.0, 0.0),   # under the 0.1 s the renewed lease needs
    (6.0, 5.0, 0.95),   # the skew bound spends the rest
    (3.0, 4.0, 0.0),    # the statement cap outlives the lease
])
def test_the_lease_must_outlast_the_publish_fence(ttl, publish, skew):
    # The native writer's rule, checked here so a bad pairing fails at
    # construction rather than at start(): the TTL must exceed the publish
    # timeout plus the skew bound by at least 0.1 s.
    with pytest.raises(ValueError, match="lease_ttl_s"):
        _storage_config(lease_ttl_s=ttl, publish_timeout_s=publish,
                        clock_skew_s=skew)


@pytest.mark.parametrize("ttl, publish, skew", [
    (5.1, 5.0, 0.0), (3.0, 1.0, 0.0), (7.0, 5.0, 1.5), (2, 1, 0)])
def test_a_lease_that_outlasts_the_fence_is_accepted(ttl, publish, skew):
    config = _storage_config(lease_ttl_s=ttl, publish_timeout_s=publish,
                             clock_skew_s=skew)
    assert config.lease_ttl_s == ttl


@pytest.mark.parametrize("ttl, publish, skew", [
    (30.0, 5.0, 15.0),   # the skew spends the renewal's whole half
    (15.0, 5.0, 7.3),    # 100 ms left for a renewal
    (3.0, 1.0, 1.25),    # 150 ms, at the live suites' TTL
    (3.0, 1.0, 1.21),    # 190 ms: just under the 200 ms floor
])
def test_the_skew_must_leave_a_renewal_time_to_finish(ttl, publish, skew):
    # Each clears the fence margin, but a renewal starts up to lease_ttl_s/2
    # after the claim that stamped the row was sent, and has to be answered
    # by the lease deadline, lease_ttl_s less the skew and a 0.1 s margin
    # after that send. The native service refuses these too.
    with pytest.raises(ValueError, match="clock_skew_s") as refusal:
        _storage_config(lease_ttl_s=ttl, publish_timeout_s=publish,
                        clock_skew_s=skew)
    assert "lease_ttl_s" in str(refusal.value)


@pytest.mark.parametrize("ttl, publish, skew", [
    (15.0, 5.0, 0.0),    # the defaults
    (15.0, 5.0, 7.2),    # exactly the 200 ms floor
    (15.0, 5.0, 5.0),    # 2.4 s, where (TTL / 3 - skew) / 4 leaves 0
    (3.0, 1.0, 1.2),     # the floor at a 3 s TTL
    (7.0, 5.0, 1.5),
])
def test_a_skew_that_leaves_a_renewal_its_time_is_accepted(ttl, publish, skew):
    config = _storage_config(lease_ttl_s=ttl, publish_timeout_s=publish,
                             clock_skew_s=skew)
    assert config.clock_skew_s == skew


def test_the_request_timeout_does_not_bound_the_lease():
    # Requests under the lease are bounded by the lease deadline as well, so
    # a generic request timeout longer than the TTL is not refused.
    config = _storage_config(lease_ttl_s=3.0, publish_timeout_s=1,
                             clickhouse_request_timeout_s=120.0)
    assert config.clickhouse_request_timeout_s == 120.0


@pytest.mark.parametrize("publish", [1.5, 0.5, 4.999])
def test_the_publish_timeout_is_whole_seconds(publish):
    # The native writer sends it as max_execution_time in whole seconds; a
    # fraction would truncate, and 0 is no limit at all.
    with pytest.raises(ValueError, match="publish_timeout_s must be a whole"):
        _storage_config(lease_ttl_s=30.0, publish_timeout_s=publish)


@pytest.mark.parametrize("name, value", [
    ("lease_ttl_s", 0.0), ("lease_ttl_s", -3.0), ("publish_timeout_s", 0),
    ("publish_timeout_s", -1.0), ("clock_skew_s", -0.5),
    ("start_lease_wait_s", -1.0)])
def test_lease_knobs_refuse_out_of_range_values(name, value):
    with pytest.raises(ValueError, match=name):
        _storage_config(**{name: value})


@pytest.mark.parametrize("name", [
    "lease_ttl_s", "publish_timeout_s", "clock_skew_s", "start_lease_wait_s"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_lease_knobs_must_be_finite(name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        _storage_config(**{name: value})


def test_a_zero_start_wait_is_accepted():
    assert _storage_config(start_lease_wait_s=0.0).start_lease_wait_s == 0.0


def test_the_service_gets_the_lease_knobs_in_nanoseconds(monkeypatch, tmp_path):
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)

    engine.create_record_runtime(_record_format())

    config = services[0].config
    assert config["lease_ttl_ns"] == 15_000_000_000
    assert config["publish_timeout_ns"] == 5_000_000_000
    assert config["clock_skew_ns"] == 0
    assert config["start_lease_wait_ns"] == 20_000_000_000


def test_the_default_start_wait_outlasts_a_skewed_predecessor(monkeypatch,
                                                              tmp_path):
    # A replica whose clock lags by clock_skew_s still sees the crashed
    # predecessor's row as live for that long after lease_ttl_s.
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)
    engine._capture_storage_config = _storage_config(clock_skew_s=2.0)

    engine.create_record_runtime(_record_format())

    assert services[0].config["start_lease_wait_ns"] == 22_000_000_000


def test_explicit_lease_knobs_reach_the_service(monkeypatch, tmp_path):
    engine, _events, services = _capture_engine(monkeypatch, tmp_path)
    engine._capture_storage_config = _storage_config(
        lease_ttl_s=3, publish_timeout_s=1.0, clock_skew_s=0.25,
        start_lease_wait_s=0.5)

    engine.create_record_runtime(_record_format())

    config = services[0].config
    assert config["lease_ttl_ns"] == 3_000_000_000
    assert config["publish_timeout_ns"] == 1_000_000_000
    assert config["clock_skew_ns"] == 250_000_000
    assert config["start_lease_wait_ns"] == 500_000_000
    for name in ("lease_ttl_ns", "publish_timeout_ns", "clock_skew_ns",
                 "start_lease_wait_ns"):
        assert type(config[name]) is int, name


def test_the_reader_is_not_handed_the_lease_knobs(monkeypatch):
    # A reader takes no lease; the knobs are the service's alone.
    from dmi.storage import native_capture

    seen = {}

    class _Reader:
        def __init__(self, config):
            seen.update(config)

    monkeypatch.setattr(
        native_capture, "_load_native_store_extension",
        lambda: SimpleNamespace(CaptureReader=_Reader, SEARCH_ITEM_COLUMNS=()))
    native_capture.NativeCaptureReader(_storage_config(lease_ttl_s=3.0,
                                                       publish_timeout_s=1))
    assert not {"lease_ttl_ns", "publish_timeout_ns", "clock_skew_ns",
                "start_lease_wait_ns"} & set(seen)
