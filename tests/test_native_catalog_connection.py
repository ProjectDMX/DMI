"""The native catalog client's connection: auth, TLS, retries, bounds.

The catalog's ClickHouse client used to speak plain ``http://host:port`` with
no credentials, so capture storage could not reach a secured ClickHouse at
all. These tests pin what it now puts on the wire, against a local fake
ClickHouse that records every request (no server needed):

* credentials travel as ``X-ClickHouse-User`` / ``X-ClickHouse-Key`` headers
  and never in the URL, and a password is refused over plain HTTP unless the
  caller opts in;
* ``https`` always verifies the peer and its name, against the system roots
  or a private CA given as a file or a hashed directory;
* a statement that never reached the server (connection refused) is retried
  whatever it is; a read is retried on a transport error or a 5xx; a write is
  never retried once it may have reached the server, since its outcome is
  unknown; and a timeout is not retried at all, so one request timeout stays
  the bound it claims to be;
* reads ask the server to finish the query before answering
  (``wait_end_of_query=1``), so an error part-way through a result arrives as
  an error status instead of a truncated body that parses as rows.

Most of it drives the client through the ``conformance_catalog`` driver's
``execute`` op; the last section drives the ``_dmi_native_store`` bindings
and ``NativeCaptureStorageConfig`` to show the options reach both the
storage service and the reader.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tests._private_ca import FakeClickHouse, make_private_ca

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
DRIVER = BUILD / "conformance_catalog"
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`"),
]

OPEN = {
    "op": "open",
    "database": "default",
    "table_prefix": "connection_test",
    "lease_ttl_ns": 30_000_000_000,
    "publish_timeout_ns": 5_000_000_000,
    "clock_skew_ns": 0,
    "allocation_attempts": 3,
}

PASSWORD = "pw-5e1f0c7a-do-not-log"
READ = "SELECT 1"
WRITE = "INSERT INTO `default`.`connection_test_t` VALUES (1)"


@pytest.fixture(scope="module")
def ca(tmp_path_factory):
    return make_private_ca(tmp_path_factory.mktemp("private-ca"))


class Driver:
    """One conformance_catalog process; the env points it at a dead port,
    so a test reaches a server only through the connection it opens."""

    def __init__(self):
        env = dict(os.environ)
        env["DMI_CLICKHOUSE_HOST"] = "127.0.0.1"
        env["DMI_CLICKHOUSE_HTTP_PORT"] = "1"
        self.proc = subprocess.Popen(
            [str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def open(self, **connection) -> dict:
        return self.call(**OPEN, **connection)

    def execute(self, query: str) -> dict:
        return self.call(op="execute", query=query)

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=60)


@pytest.fixture
def driver():
    d = Driver()
    yield d
    d.close()


def _opened(driver, **connection):
    response = driver.open(**connection)
    assert response["ok"], response
    return driver


def _query(request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.path).query)


# --- credentials ---------------------------------------------------------------


def test_credentials_travel_as_headers_and_never_in_the_url(driver):
    with FakeClickHouse(lambda r: (200, b"1\n")) as fake:
        _opened(driver, clickhouse_port=fake.port,
                clickhouse_user="catalog_writer", clickhouse_password=PASSWORD,
                clickhouse_allow_insecure_http=True)
        response = driver.execute(READ)
    assert response["ok"], response
    assert response["rows"] == [["1"]]
    [request] = fake.requests
    assert request.headers["x-clickhouse-user"] == "catalog_writer"
    assert request.headers["x-clickhouse-key"] == PASSWORD
    assert PASSWORD not in request.path
    assert "catalog_writer" not in request.path
    assert "authorization" not in request.headers
    assert request.body == READ.encode()


def test_no_credentials_send_no_auth_headers(driver):
    with FakeClickHouse() as fake:
        _opened(driver, clickhouse_port=fake.port)
        assert driver.execute(READ)["ok"]
    [request] = fake.requests
    assert "x-clickhouse-user" not in request.headers
    assert "x-clickhouse-key" not in request.headers


def test_a_user_without_a_password_sends_only_the_user(driver):
    with FakeClickHouse() as fake:
        _opened(driver, clickhouse_port=fake.port, clickhouse_user="reader")
        assert driver.execute(READ)["ok"]
    [request] = fake.requests
    assert request.headers["x-clickhouse-user"] == "reader"
    assert "x-clickhouse-key" not in request.headers


def test_a_password_over_plain_http_is_refused_unless_opted_into(driver):
    with FakeClickHouse() as fake:
        response = driver.open(clickhouse_port=fake.port,
                               clickhouse_user="catalog_writer",
                               clickhouse_password=PASSWORD)
        assert not response["ok"]
        assert "allow_insecure_http" in response["message"], response
        assert PASSWORD not in response["message"]
    assert fake.requests == []


@pytest.mark.parametrize("host", [
    "catalog_writer:secret@127.0.0.1",   # userinfo
    "@127.0.0.1",
    "http://127.0.0.1",                  # a scheme belongs in the scheme
    "127.0.0.1:8123",                    # a port belongs in the port
    "127.0.0.1/db",
    "127.0.0.1?user=x",
    "127.0.0.1#x",
    "127.0.0.1 ",
    "",
    "[::1",
])
def test_a_host_that_is_not_just_a_host_is_refused(driver, host):
    response = driver.open(clickhouse_host=host)
    assert not response["ok"], response
    assert "clickhouse host" in response["message"], response
    assert "secret" not in response["message"]


def test_a_bracketed_ipv6_host_is_accepted(driver):
    assert driver.open(clickhouse_host="[::1]")["ok"]


@pytest.mark.parametrize("connection, needle", [
    (dict(clickhouse_scheme="ftp"), "scheme"),
    (dict(clickhouse_scheme="HTTPS"), "scheme"),
    (dict(clickhouse_scheme="https", clickhouse_allow_insecure_http=True),
     "allow_insecure_http"),
    (dict(clickhouse_ca_file="/etc/ssl/certs/ca-certificates.crt"),
     "https"),
    (dict(clickhouse_ca_path="/etc/ssl/certs"), "https"),
    (dict(clickhouse_scheme="https", clickhouse_password=PASSWORD),
     "user"),
    (dict(clickhouse_user="a\r\nX-Injected: 1"), "user"),
    (dict(clickhouse_scheme="https", clickhouse_user="u",
          clickhouse_password="a\nb"), "password"),
    (dict(clickhouse_max_attempts=0), "max_attempts"),
])
def test_inconsistent_connection_options_are_refused(driver, connection, needle):
    response = driver.open(**connection)
    assert not response["ok"], response
    assert needle in response["message"], response
    assert PASSWORD not in response["message"]


# --- what a read asks of the server --------------------------------------------


def test_a_read_waits_for_the_end_of_the_query_and_a_write_does_not(driver):
    with FakeClickHouse() as fake:
        _opened(driver, clickhouse_port=fake.port)
        assert driver.execute(READ)["ok"]
        assert driver.execute(WRITE)["ok"]
    read, write = fake.requests
    assert _query(read).get("wait_end_of_query") == ["1"]
    assert "wait_end_of_query" not in _query(write)
    # A URL setting: the statement bytes are exactly what the caller sent.
    assert read.body == READ.encode()
    assert write.body == WRITE.encode()


@pytest.mark.parametrize("statement", [
    "SELECT 1", "  select 1", "\nWITH 1 AS x SELECT x", "SHOW TABLES",
    "DESCRIBE TABLE t", "EXISTS TABLE t", "CHECK GRANT SHOW TABLES ON t",
])
def test_statements_that_only_read_are_classified_as_reads(driver, statement):
    with FakeClickHouse() as fake:
        _opened(driver, clickhouse_port=fake.port)
        assert driver.execute(statement)["ok"]
    assert _query(fake.requests[0]).get("wait_end_of_query") == ["1"]


@pytest.mark.parametrize("statement", [
    "INSERT INTO t SELECT 1", "CREATE TABLE t (x UInt8) ENGINE = Memory",
    "ALTER TABLE t DELETE WHERE 1", "DROP TABLE t", "TRUNCATE TABLE t",
    "SELECTED", "(SELECT 1)", "SYSTEM FLUSH LOGS",
])
def test_everything_else_is_treated_as_a_write(driver, statement):
    with FakeClickHouse() as fake:
        _opened(driver, clickhouse_port=fake.port)
        assert driver.execute(statement)["ok"]
    assert "wait_end_of_query" not in _query(fake.requests[0])


# --- retries -------------------------------------------------------------------


def _failing(times: int, answer):
    """Answer `answer` for the first `times` requests, then 200 "1"."""
    seen = []

    def respond(request):
        seen.append(request)
        return answer if len(seen) <= times else (200, b"1\n")
    return respond


def test_a_read_is_retried_after_a_server_error(driver):
    with FakeClickHouse(_failing(2, (503, b"overloaded"))) as fake:
        _opened(driver, clickhouse_port=fake.port)
        response = driver.execute(READ)
    assert response["ok"], response
    assert response["rows"] == [["1"]]
    assert len(fake.requests) == 3


def test_a_read_gives_up_after_its_attempts(driver):
    with FakeClickHouse(lambda r: (500, b"still broken")) as fake:
        _opened(driver, clickhouse_port=fake.port, clickhouse_max_attempts=4)
        response = driver.execute(READ)
    assert not response["ok"]
    assert "500" in response["message"] and "still broken" in response["message"]
    assert len(fake.requests) == 4


def test_a_write_is_not_retried_after_a_server_error(driver):
    with FakeClickHouse(_failing(1, (503, b"overloaded"))) as fake:
        _opened(driver, clickhouse_port=fake.port)
        response = driver.execute(WRITE)
    assert not response["ok"]
    assert "503" in response["message"]
    assert len(fake.requests) == 1


def test_a_read_is_retried_when_the_connection_drops_and_a_write_is_not(driver):
    with FakeClickHouse(_failing(1, None)) as fake:
        _opened(driver, clickhouse_port=fake.port)
        response = driver.execute(READ)
        assert response["ok"], response
        assert len(fake.requests) == 2
    with FakeClickHouse(_failing(1, None)) as fake:
        _opened(driver, clickhouse_port=fake.port)
        response = driver.execute(WRITE)
        assert not response["ok"]
        assert len(fake.requests) == 1


def test_a_client_error_is_not_retried_even_for_a_read(driver):
    with FakeClickHouse(lambda r: (404, b"Code: 60. UNKNOWN_TABLE")) as fake:
        _opened(driver, clickhouse_port=fake.port)
        response = driver.execute(READ)
    assert not response["ok"]
    assert len(fake.requests) == 1


def test_a_timed_out_read_is_not_retried(driver):
    """Retrying a timeout would multiply the bound the timeout promises."""
    release = threading.Event()

    def stall(request):
        release.wait(10)
        return (200, b"1\n")

    with FakeClickHouse(stall) as fake:
        _opened(driver, clickhouse_port=fake.port,
                clickhouse_request_timeout_ms=300)
        started = time.monotonic()
        response = driver.execute(READ)
        elapsed = time.monotonic() - started
        time.sleep(0.3)  # a retry would have landed by now
        release.set()
    assert not response["ok"]
    assert "timeout" in response["message"].lower() or \
        "timed out" in response["message"].lower(), response
    assert len(fake.requests) == 1
    assert elapsed < 2.0, elapsed


def test_a_refused_connection_is_retried_even_for_a_write(driver):
    """Nothing reached the server, so even a write is safe to send again.

    The port refuses the first attempt and starts listening while the client
    backs off: the write lands exactly once, on a later attempt.
    """
    port_holder = socket.socket()
    port_holder.bind(("127.0.0.1", 0))  # bound, not listening: refused
    port = port_holder.getsockname()[1]
    _opened(driver, clickhouse_port=port, clickhouse_max_attempts=5)

    received = []

    def serve():
        time.sleep(0.05)
        port_holder.listen(1)
        conn, _ = port_holder.accept()
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _, body = data.partition(b"\r\n\r\n")
            length = next(int(line.split(b":")[1]) for line in
                          head.split(b"\r\n")
                          if line.lower().startswith(b"content-length"))
            while len(body) < length:
                body += conn.recv(65536)
            received.append(body)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    response = driver.execute(WRITE)
    server.join(timeout=10)
    port_holder.close()
    assert response["ok"], response
    assert received == [WRITE.encode()]


def test_a_refused_connection_gives_up_after_its_attempts(driver):
    port_holder = socket.socket()
    port_holder.bind(("127.0.0.1", 0))
    port = port_holder.getsockname()[1]
    try:
        _opened(driver, clickhouse_port=port, clickhouse_max_attempts=3)
        started = time.monotonic()
        response = driver.execute(WRITE)
        elapsed = time.monotonic() - started
    finally:
        port_holder.close()
    assert not response["ok"]
    assert "3 attempts" in response["message"], response
    assert elapsed < 5.0, elapsed


# --- TLS -----------------------------------------------------------------------


def test_https_verifies_the_server_against_a_private_ca_file(driver, ca):
    with FakeClickHouse(lambda r: (200, b"1\n"), tls=ca) as fake:
        _opened(driver, clickhouse_scheme="https", clickhouse_port=fake.port,
                clickhouse_ca_file=str(ca.ca_file),
                clickhouse_user="catalog_writer", clickhouse_password=PASSWORD)
        response = driver.execute(READ)
    assert response["ok"], response
    assert response["rows"] == [["1"]]
    [request] = fake.requests
    assert request.headers["x-clickhouse-key"] == PASSWORD
    assert PASSWORD not in request.path
    assert (fake.handshakes, fake.failed_handshakes) == (1, 0)


def test_https_verifies_the_server_against_a_private_ca_directory(driver, ca):
    with FakeClickHouse(lambda r: (200, b"1\n"), tls=ca) as fake:
        _opened(driver, clickhouse_scheme="https", clickhouse_port=fake.port,
                clickhouse_ca_path=str(ca.ca_path))
        response = driver.execute(READ)
    assert response["ok"], response


def test_https_refuses_a_server_its_roots_do_not_vouch_for(driver, ca):
    """Without the private CA the handshake fails: no request is sent, the
    password never leaves, and a refused certificate is not retried."""
    with FakeClickHouse(tls=ca) as fake:
        _opened(driver, clickhouse_scheme="https", clickhouse_port=fake.port,
                clickhouse_user="catalog_writer", clickhouse_password=PASSWORD)
        response = driver.execute(READ)
    assert not response["ok"]
    assert "certificate" in response["message"].lower(), response
    assert fake.requests == []
    assert (fake.handshakes, fake.failed_handshakes) == (1, 1)


def test_https_refuses_a_certificate_for_another_name(driver, ca):
    with FakeClickHouse(tls=ca, wrong_name=True) as fake:
        _opened(driver, clickhouse_scheme="https", clickhouse_port=fake.port,
                clickhouse_ca_file=str(ca.ca_file))
        response = driver.execute(READ)
    assert not response["ok"]
    assert fake.requests == []


def test_https_does_not_fall_back_to_plain_http(driver, ca):
    with FakeClickHouse() as fake:  # a plain HTTP server
        _opened(driver, clickhouse_scheme="https", clickhouse_port=fake.port,
                clickhouse_ca_file=str(ca.ca_file),
                clickhouse_user="catalog_writer", clickhouse_password=PASSWORD)
        response = driver.execute(READ)
    assert not response["ok"]
    assert fake.requests == []


# --- the bindings and the config -------------------------------------------------

needs_store = pytest.mark.skipif(
    not STORE_BUILT,
    reason="the native store module is not built; run "
    "`make -C native build/_dmi_native_store PYTHON=<venv>/bin/python`")


def _store_dict(ca, port, **overrides):
    native = {
        "s3_endpoint": "http://127.0.0.1:1", "s3_bucket": "bucket",
        "s3_access_key": "AKIA-test", "s3_secret_key": "secret-test",
        "s3_allow_insecure_http": True,
        "clickhouse_scheme": "https", "clickhouse_host": "127.0.0.1",
        "clickhouse_port": port, "clickhouse_ca_file": str(ca.ca_file),
        "clickhouse_user": "catalog_writer", "clickhouse_password": PASSWORD,
        "database": "default", "table_prefix": "connection_test",
    }
    native.update(overrides)
    return native


@needs_store
def test_the_reader_binding_reaches_a_verified_tls_catalog(ca):
    from dmi.storage.native_capture import _load_native_store_extension

    module = _load_native_store_extension()
    with FakeClickHouse(lambda r: (400, b"Code: 62. refused"), tls=ca) as fake:
        reader = module.CaptureReader(_store_dict(ca, fake.port))
        with pytest.raises(RuntimeError, match="400"):
            reader.search({"tenant_id": "t"})
    assert fake.requests, "the reader never reached the catalog"
    request = fake.requests[0]
    assert request.headers["x-clickhouse-user"] == "catalog_writer"
    assert request.headers["x-clickhouse-key"] == PASSWORD
    assert PASSWORD not in request.path
    assert _query(request).get("wait_end_of_query") == ["1"]


@needs_store
def test_the_service_binding_reaches_a_verified_tls_catalog(ca, tmp_path):
    from dmi.storage.native_capture import _load_native_store_extension

    module = _load_native_store_extension()
    with FakeClickHouse(lambda r: (400, b"Code: 62. refused"), tls=ca) as fake:
        service = module.StorageService(_store_dict(
            ca, fake.port, spool_root=str(tmp_path / "spool"),
            holder="connection-test", reconcile_on_start=False))
        with pytest.raises(Exception, match="400"):
            service.start()
        service.stop()
    assert fake.requests, "the service never reached the catalog"
    request = fake.requests[0]
    assert request.headers["x-clickhouse-user"] == "catalog_writer"
    assert request.headers["x-clickhouse-key"] == PASSWORD
    assert PASSWORD not in request.path


@needs_store
def test_the_bindings_refuse_a_password_over_plain_http(ca):
    from dmi.storage.native_capture import _load_native_store_extension

    module = _load_native_store_extension()
    native = _store_dict(ca, 1, clickhouse_scheme="http")
    native.pop("clickhouse_ca_file")
    with pytest.raises(RuntimeError, match="allow_insecure_http"):
        module.CaptureReader(native)
