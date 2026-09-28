"""The publisher lease's requests run under a deadline of their own.

A lease renewal is three ClickHouse requests (head read, claim INSERT,
read-back), and each was bounded only by the catalog client's generic
request timeout -- 60 s by default, against a 15 s lease TTL. A server that
accepted a renewal and never answered let the row expire before any error
surfaced, and a claim INSERT the client gave up on carried no server-side
cap, so the server could still land it afterwards, outliving the quarantine
meant to cover it.

While a lease is held, its requests now share the lease deadline: when the
claim that stamped the row was sent, plus the TTL, less the clock skew and a
0.1 s margin. Without one, each request of a claim is bounded by
min(request timeout, TTL / 3). The lease INSERTs carry the time left as
max_execution_time and lock_acquire_timeout, and cap a quorum wait by it.

These drive the native coordinator (``native/build/conformance_catalog``)
against a scripted HTTP server standing in for ClickHouse, so the request
bound and the settings each lease statement carries are pinned on the CPU
gate. The same behaviour against a real server, through the storage
service, is in tests/test_native_capture_storage_live.py.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
DRIVER = BUILD / "conformance_catalog"
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]

HEAD = "SELECT term, toString(lease_id)"
READ_BACK = "SELECT toString(lease_id), acquired_at_ns"


class _FakeClickHouse:
    """Answers the lease protocol's statements as an empty catalog would,
    records every request, and can stall a statement: accept it and never
    answer until the test ends."""

    def __init__(self):
        self.requests: list[tuple[dict[str, str], str]] = []
        self.head_rows = ""  # the head read's TSV answer; empty catalog
        self.stall: str | None = None  # a statement prefix to never answer
        # (statement prefix, seconds): answer the first such statement late.
        self.delay: tuple[str, float] | None = None
        # statement prefix -> seconds: answer every such statement late.
        self.delays: dict[str, float] = {}
        # (statement prefix, seconds): answer every such statement that late,
        # with a 503 -- a failure the client retries for a read.
        self.unavailable: tuple[str, float] | None = None
        # Called before the head read is answered, from the thread serving
        # it.
        self.before_head_answer = None
        # When each statement arrived, by prefix of its body.
        self.arrivals: list[tuple[float, str]] = []
        self._released = threading.Event()
        self._claimed = ""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 -- the http.server hook name
                body = self.rfile.read(
                    int(self.headers["Content-Length"])).decode()
                settings = {key: values[-1] for key, values in
                            parse_qs(urlsplit(self.path).query).items()}
                fake.requests.append((settings, body))
                fake.arrivals.append((time.monotonic(), body))
                if fake.stall is not None and body.startswith(fake.stall):
                    fake._released.wait(30)
                    return
                if fake.delay is not None and body.startswith(fake.delay[0]):
                    seconds = fake.delay[1]
                    fake.delay = None
                    time.sleep(seconds)
                for prefix, seconds in list(fake.delays.items()):
                    if body.startswith(prefix):
                        time.sleep(seconds)
                unavailable = fake.unavailable
                if unavailable is not None and body.startswith(unavailable[0]):
                    time.sleep(unavailable[1])
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if body.startswith("INSERT"):
                    fake._claimed = re.search(
                        r"toUUID\('([^']+)'\)", body).group(1)
                    answer = ""
                elif body.startswith(HEAD):
                    answer = fake.head_rows
                    if fake.before_head_answer is not None:
                        fake.before_head_answer()
                elif body.startswith(READ_BACK):
                    answer = f"{fake._claimed}\t1\t2\n"
                else:
                    answer = ""
                payload = answer.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever,
                         daemon=True).start()

    def inserts(self) -> list[dict[str, str]]:
        return [settings for settings, body in self.requests
                if body.startswith("INSERT")]

    def close(self):
        self._released.set()
        self._server.shutdown()
        self._server.server_close()


class _Driver:
    def __init__(self, port: int):
        env = dict(os.environ, DMI_CLICKHOUSE_HOST="127.0.0.1",
                   DMI_CLICKHOUSE_HTTP_PORT=str(port))
        self.proc = subprocess.Popen(
            [str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def open(self, **overrides) -> dict:
        fields = {"op": "open", "database": "default",
                  "table_prefix": "dmi_lease_bound",
                  "lease_ttl_ns": 15_000_000_000,
                  "publish_timeout_ns": 5_000_000_000, "clock_skew_ns": 0,
                  "allocation_attempts": 3, **overrides}
        return self.call(**fields)

    def close(self):
        self.proc.kill()
        self.proc.wait(timeout=30)


@pytest.fixture
def fake():
    server = _FakeClickHouse()
    yield server
    server.close()


@pytest.fixture
def driver(fake):
    process = _Driver(fake.port)
    yield process
    process.close()


@pytest.mark.parametrize("ttl_ns, publish_ns", [
    (30_000_000_000, 5_000_000_000),
    (15_000_000_000, 5_000_000_000),  # the service default
    (3_000_000_000, 1_000_000_000),
    (1_200_000_000, 1_000_000_000),
])
def test_the_lease_inserts_carry_the_time_left_as_a_server_cap(
        fake, driver, ttl_ns, publish_ns):
    """A claim the client gave up on must not land afterwards: the server
    abandons the INSERT at max_execution_time, and waits for a table lock no
    longer (lock_acquire_timeout), never later than the client's own
    deadline. A claim with no lease held has min(request timeout, TTL / 3);
    the tombstone, sent under the lease, what is left of the TTL less the
    0.1 s margin. To the millisecond, rounded down: whole seconds cut up to
    a second off a healthy INSERT the client would still have waited for."""
    assert driver.open(lease_ttl_ns=ttl_ns, publish_timeout_ns=publish_ns)["ok"]
    claimed = driver.call(op="claim", holder="h", lease_id=str(uuid.uuid4()))
    assert claimed["ok"], claimed
    assert driver.call(op="release")["ok"]

    claim, tombstone = fake.inserts()
    claim_bound = min(60.0, ttl_ns / 3e9)
    lease_left = ttl_ns / 1e9 - 0.1
    for settings, bound in ((claim, claim_bound), (tombstone, lease_left)):
        cap = float(settings["max_execution_time"])
        assert bound - 0.05 < cap <= bound, settings
        assert settings["lock_acquire_timeout"] == \
            settings["max_execution_time"], settings
        # break would insert whatever had been read by then.
        assert settings.get("timeout_overflow_mode") == "throw", settings
        assert "insert_quorum" not in settings, settings
        assert settings["max_execution_time"] != "0", settings  # no limit


def test_a_quorum_lease_insert_waits_no_longer_than_its_deadline(
        fake, driver):
    """max_execution_time does not cover the quorum wait, so the quorum
    timeout is capped too, by the time left: past the client's deadline
    nobody is listening. publish_timeout still caps it where that is less."""
    assert driver.open(lease_ttl_ns=12_000_000_000,
                       publish_timeout_ns=5_000_000_000,
                       clock_skew_ns=1_000_000, insert_quorum=2)["ok"]
    assert driver.call(op="claim", holder="h",
                       lease_id=str(uuid.uuid4()))["ok"]
    assert driver.call(op="release")["ok"]

    claim, tombstone = fake.inserts()
    # No lease yet: TTL / 3 = 4 s left, less than the 5 s publish timeout.
    assert 3900 < int(claim["insert_quorum_timeout"]) <= 4000, claim
    assert 3.9 < float(claim["max_execution_time"]) <= 4.0, claim
    assert claim["insert_quorum"] == "2", claim
    # Under the lease, nearly 12 s are left: publish_timeout caps it.
    assert tombstone["insert_quorum_timeout"] == "5000", tombstone


def _timed(call):
    outcome = {}

    def _run():
        started = time.monotonic()
        outcome["response"] = call()
        outcome["elapsed"] = time.monotonic() - started

    waiter = threading.Thread(target=_run, daemon=True)
    waiter.start()
    waiter.join(timeout=10.0)
    assert not waiter.is_alive(), "the stalled lease request was not bounded"
    return outcome["response"], outcome["elapsed"]


@pytest.mark.parametrize("statement", [HEAD, "INSERT", READ_BACK])
def test_a_claim_request_never_answered_fails_within_the_claim_bound(
        fake, driver, statement):
    """With no lease held, each request of a claim gives up after
    min(request timeout, TTL / 3) -- 1 s at a 3 s TTL -- not after the
    client's 60 s request timeout, and the error names the knobs."""
    assert driver.open(lease_ttl_ns=3_000_000_000,
                       publish_timeout_ns=1_000_000_000)["ok"]
    fake.stall = statement
    response, elapsed = _timed(lambda: driver.call(
        op="claim", holder="h", lease_id=str(uuid.uuid4())))
    assert not response["ok"], response
    assert response["error"] == "ClickHouseError", response
    assert "Timeout" in response["message"], response
    for knob in ("clickhouse_request_timeout_s", "lease_ttl_s / 3"):
        assert knob in response["message"], response
    assert 0.9 < elapsed < 1.5, elapsed


def test_a_renewal_may_use_all_the_time_its_row_has_left(fake, driver):
    """A fixed bound per request, small enough for several requests to fit
    the renewal window (TTL / 12), would quarantine a healthy but slow
    catalog. A renewal's requests share the lease deadline instead, so one
    slow request may use the row's whole remaining life -- and a stalled one
    still fails 0.1 s before the row expires."""
    assert driver.open(lease_ttl_ns=3_000_000_000,
                       publish_timeout_ns=1_000_000_000)["ok"]
    claimed_at = time.monotonic()
    assert driver.call(op="claim", holder="h",
                       lease_id=str(uuid.uuid4()))["ok"]
    fake.stall = HEAD
    response, _ = _timed(lambda: driver.call(op="renew"))
    elapsed = time.monotonic() - claimed_at
    assert not response["ok"], response
    assert "Timeout" in response["message"], response
    for knob in ("lease_ttl_s", "clock_skew_s", "clickhouse_request_timeout_s"):
        assert knob in response["message"], response
    assert 2.5 < elapsed < 3.0, elapsed


def test_retries_stop_at_the_deadline(fake, driver):
    """A read the server answers with a 503 is retried, up to three attempts
    with a backoff between them, and each attempt had the whole request
    timeout: a claim's head read could take three times its bound, and more.
    Under a deadline no attempt, and no backoff, runs past it."""
    assert driver.open(lease_ttl_ns=3_000_000_000,
                       publish_timeout_ns=1_000_000_000)["ok"]
    # 0.4 s an attempt: the second ends at 0.9 s, and the 0.2 s backoff
    # before a third would end past the 1 s claim bound.
    fake.unavailable = (HEAD, 0.4)
    response, elapsed = _timed(lambda: driver.call(
        op="claim", holder="h", lease_id=str(uuid.uuid4())))
    assert not response["ok"], response
    assert "503" in response["message"], response
    assert "not retried" in response["message"], response
    assert "lease_ttl_s / 3" in response["message"], response
    assert elapsed < 1.05, elapsed
    heads = [body for _, body in fake.requests if body.startswith(HEAD)]
    assert len(heads) == 2, heads


@pytest.mark.parametrize("statement, quarantined", [
    (HEAD, False), ("INSERT", True), (READ_BACK, True),
])
def test_a_claim_quarantines_only_once_its_insert_may_have_landed(
        fake, driver, statement, quarantined):
    """acquire() quarantined the writer for a TTL on any ClickHouse error. A
    claim whose head read failed has sent no INSERT, so no row of it can
    land and there is nothing to wait out -- a claim that timed out at start
    could be retried at once instead of failing, or waiting a TTL. One whose
    INSERT went out has an unknown outcome and still quarantines.
    (Deliberately unlike the Python oracle, which quarantines on any
    error.)"""
    assert driver.open(lease_ttl_ns=3_000_000_000,
                       publish_timeout_ns=1_000_000_000)["ok"]
    fake.stall = statement
    response, _ = _timed(lambda: driver.call(op="acquire", holder="h"))
    assert not response["ok"], response
    assert "Timeout" in response["message"], response
    assert driver.call(op="quarantined")["quarantined"] is quarantined


@pytest.mark.parametrize("op", ["claim", "acquire"])
def test_a_claim_is_confirmed_by_the_deadline_of_the_lease_it_takes(
        fake, driver, op):
    """Each request of a claim made without a lease had the claim bound,
    min(request timeout, TTL / 3), from when it started -- the read-back
    after the INSERT another full one. The lease the claim takes has until
    sent + TTL - skew - 0.1 s. With a skew over TTL / 3 - 0.1 s (which
    validation accepts), an INSERT and read-back each inside the bound could
    confirm the claim after its own deadline: a success the service then
    abandoned at once, blaming a renewal that never happened, and no
    timeout counted. The requests after the INSERT are bounded by that
    deadline too, so such a claim fails as a timeout that names the knobs,
    and -- its INSERT sent -- quarantines."""
    assert driver.open(lease_ttl_ns=3_000_000_000,
                       publish_timeout_ns=1_000_000_000,
                       clock_skew_ns=1_200_000_000)["ok"]
    # 0.9 s each, inside the 1 s claim bound; 1.8 s together, past the
    # 3 - 1.2 - 0.1 = 1.7 s the new lease has.
    fake.delays = {"INSERT": 0.9, READ_BACK: 0.9}
    fields = {"holder": "h"}
    if op == "claim":
        fields["lease_id"] = str(uuid.uuid4())
    response, _ = _timed(lambda: driver.call(op=op, **fields))
    arrived = next(at for at, body in fake.arrivals
                   if body.startswith("INSERT"))
    assert not response["ok"], response
    assert response["error"] == "ClickHouseError", response
    assert "Timeout" in response["message"], response
    for knob in ("lease_ttl_s", "clock_skew_s"):
        assert knob in response["message"], response
    # Given up at the lease deadline, 1.7 s after the INSERT went out.
    assert 1.55 < time.monotonic() - arrived < 1.9
    if op == "acquire":
        assert driver.call(op="quarantined")["quarantined"] is True


def test_a_slow_but_healthy_head_read_does_not_fail_the_claim(fake, driver):
    """A head read with select_sequential_consistency on a loaded catalog
    can take seconds, which a fixed TTL / 12 per request -- 1.25 s at the
    service defaults -- would fail. A claim's request has min(request
    timeout, TTL / 3) = 5 s."""
    assert driver.open()["ok"]
    fake.delay = (HEAD, 2.0)
    claimed = driver.call(op="claim", holder="h", lease_id=str(uuid.uuid4()))
    assert claimed["ok"], claimed


def test_renewals_do_not_crowd_earlier_claims_out_of_the_history(
        fake, driver):
    """The history of recent claim lease_ids is what tells the service's own
    late row from a rival's. Each renewal pushed its (unchanged) lease_id
    again, so sixteen renewals pushed out a claim that could still land."""
    assert driver.open()["ok"]
    earlier = str(uuid.uuid4())
    assert driver.call(op="claim", holder="h", lease_id=earlier)["ok"]
    driver.call(op="discard_local_lease")  # as a quarantine drops it
    assert driver.call(op="acquire", holder="h")["ok"]
    for _ in range(20):
        assert driver.call(op="renew")["ok"]
    driver.call(op="discard_local_lease")

    fake.head_rows = f"1\t{earlier}\th\t{2 * 10**18}\t{10**18}\n"
    refused = driver.call(op="acquire", holder="h")
    assert refused["error"] == "PublisherLeaseHeldError", refused
    assert refused["own_claims"] is True, refused


def test_a_refusal_by_the_writers_own_claim_row_is_attributed_to_it(
        fake, driver):
    """A claim whose request gave up can still land, and then refuses the
    writer's next claim. The refusal names it as the writer's own, which the
    storage service keeps out of its 2 x TTL latch; a rival's row is not."""
    assert driver.open()["ok"]
    mine = str(uuid.uuid4())
    assert driver.call(op="claim", holder="h", lease_id=mine)["ok"]
    driver.call(op="discard_local_lease")  # as a quarantine drops it

    # Our earlier claim row is the live head; a fresh lease_id is refused.
    fake.head_rows = f"1\t{mine}\th\t{2 * 10**18}\t{10**18}\n"
    refused = driver.call(op="acquire", holder="h")
    assert refused["error"] == "PublisherLeaseHeldError", refused
    assert refused["own_claims"] is True, refused

    # Contested between our row and a rival's: not ours alone.
    rival = str(uuid.uuid4())
    fake.head_rows = (f"1\t{rival}\trival\t{2 * 10**18}\t{10**18}\n"
                      f"1\t{mine}\th\t{2 * 10**18}\t{10**18}\n")
    refused = driver.call(op="acquire", holder="h")
    assert refused["error"] == "PublisherLeaseHeldError", refused
    assert refused["own_claims"] is False, refused

    # A coordinator that never claimed `mine` sees a rival in it.
    other = _Driver(fake.port)
    try:
        assert other.open()["ok"]
        fake.head_rows = f"1\t{mine}\th\t{2 * 10**18}\t{10**18}\n"
        refused = other.call(op="acquire", holder="h2")
        assert refused["error"] == "PublisherLeaseHeldError", refused
        assert refused["own_claims"] is False, refused
    finally:
        other.close()


class _Gate:
    """A TCP forwarder in front of the fake that can stop listening for a
    while, so that connections are refused -- a restarting catalog, or a
    load balancer whose backend went away -- and then listen again on the
    same port. One request is one connection for the native client."""

    def __init__(self, target_port: int):
        self._target = target_port
        self.port = 0
        self._listener: socket.socket | None = None
        self._lock = threading.Lock()
        self._open()

    def _open(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen(64)
        self.port = listener.getsockname()[1]
        with self._lock:
            self._listener = listener
        threading.Thread(target=self._accept, args=(listener,),
                         daemon=True).start()

    def _accept(self, listener):
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self._target))
            for source, sink in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(source, sink),
                                 daemon=True).start()

    @staticmethod
    def _pump(source, sink):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for end in (source, sink):
                try:
                    end.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _stop(self):
        with self._lock:
            listener, self._listener = self._listener, None
        if listener is not None:
            # shutdown() first: a close() alone leaves the socket listening
            # while the accept thread still blocks on it.
            try:
                listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            listener.close()

    def refuse_for(self, seconds: float):
        """Stop listening now, and listen again `seconds` later."""
        self._stop()
        timer = threading.Timer(seconds, self._open)
        timer.daemon = True
        timer.start()

    def close(self):
        self._stop()


@pytest.fixture
def gate(fake):
    forwarder = _Gate(fake.port)
    yield forwarder
    forwarder.close()


@pytest.fixture
def gated_driver(gate):
    process = _Driver(gate.port)
    yield process
    process.close()


def test_a_lease_insert_retried_after_refused_connections_carries_the_time_then_left(
        fake, gate, gated_driver):
    """A lease INSERT whose connection is refused is retried, since nothing
    reached the server. The caps rode in the URL built once, before the
    first attempt, so the attempt that got through told the server it had
    the time left before the first: the server could run the claim past
    the client's deadline by the refused attempts and their backoff. Each
    attempt now carries the time left when it goes out."""
    assert gated_driver.open(lease_ttl_ns=3_000_000_000,
                             publish_timeout_ns=1_000_000_000)["ok"]
    answered = []

    def _refuse():
        answered.append(time.monotonic())
        gate.refuse_for(0.25)

    fake.before_head_answer = _refuse
    claimed = gated_driver.call(op="claim", holder="h",
                                lease_id=str(uuid.uuid4()))
    assert claimed["ok"], claimed
    arrived, _ = next(arrival for arrival in fake.arrivals
                      if arrival[1].startswith("INSERT"))
    (settings,) = fake.inserts()
    # Refused at once, then 0.1 s and 0.2 s of backoff: the third attempt.
    assert arrived - answered[0] > 0.25, (arrived, answered)
    cap = float(settings["max_execution_time"])
    assert settings["lock_acquire_timeout"] == settings["max_execution_time"]
    # The claim bound, 1 s at a 3 s TTL, counts from the INSERT's first
    # attempt, just after the head read was answered.
    assert arrived + cap <= answered[0] + 1.0 + 0.02, (arrived, cap, answered)


def test_a_lease_insert_that_never_connected_wrote_nothing(
        fake, gate, gated_driver):
    """A claim INSERT whose every attempt was refused its connection cannot
    have reached the server: its outcome is known, so the writer is not
    quarantined and may claim again at once."""
    assert gated_driver.open(lease_ttl_ns=3_000_000_000,
                             publish_timeout_ns=1_000_000_000)["ok"]
    fake.before_head_answer = lambda: gate.refuse_for(1.5)
    refused = gated_driver.call(op="acquire", holder="h")
    assert not refused["ok"], refused
    assert refused["error"] == "ClickHouseError", refused
    assert "connect" in refused["message"].lower(), refused
    assert not fake.inserts(), fake.requests
    fake.before_head_answer = None
    time.sleep(1.6)
    assert gated_driver.call(op="quarantined")["quarantined"] is False
    assert gated_driver.call(op="acquire", holder="h")["ok"]


# --- the storage service's configuration --------------------------------------


def _service_config(tmp_path, **overrides):
    config = {
        "s3_endpoint": "http://127.0.0.1:1", "s3_bucket": "bucket",
        "s3_access_key": "AKIA-test", "s3_secret_key": "secret-test",
        "s3_allow_insecure_http": True, "store_id": "s3",
        "clickhouse_host": "127.0.0.1", "clickhouse_port": 1,
        "database": "default", "table_prefix": "dmi_lease_bound",
        "spool_root": str(tmp_path / "spool"), "holder": "h",
        "lease_ttl_ns": 3_000_000_000, "publish_timeout_ns": 1_000_000_000,
        "clock_skew_ns": 0,
    }
    config.update(overrides)
    return config


@pytest.mark.skipif(not STORE_BUILT,
                    reason="the native store module is not built")
@pytest.mark.parametrize("ttl_ns, skew_ns", [
    (3_000_000_000, 1_250_000_000),    # 150 ms left for a renewal
    (3_000_000_000, 1_210_000_000),    # 190 ms: under the 200 ms floor
    (30_000_000_000, 15_000_000_000),  # nothing left at all
])
def test_the_service_refuses_a_skew_that_leaves_a_renewal_no_time(
        tmp_path, ttl_ns, skew_ns):
    # A renewal starts up to lease_ttl / 2 after the claim that stamped the
    # row was sent, and must be answered by the lease deadline, lease_ttl
    # less the skew and the 0.1 s margin after that send.
    from dmi.storage.native_capture import _load_native_store_extension

    module = _load_native_store_extension()
    with pytest.raises(ValueError, match="clock_skew_ns") as refusal:
        module.StorageService(_service_config(
            tmp_path, lease_ttl_ns=ttl_ns, clock_skew_ns=skew_ns))
    assert "lease_ttl_ns" in str(refusal.value)


@pytest.mark.skipif(not STORE_BUILT,
                    reason="the native store module is not built")
@pytest.mark.parametrize("ttl_ns, skew_ns, request_s", [
    (15_000_000_000, 0, 60.0),        # the defaults
    (3_000_000_000, 1_200_000_000, 60.0),  # exactly the floor
    (15_000_000_000, 5_000_000_000, 60.0),  # (TTL / 3 - skew) / 4 is 0
    (3_000_000_000, 0, 120.0),        # a request timeout past the TTL
])
def test_the_service_accepts_a_renewal_that_fits(tmp_path, ttl_ns, skew_ns,
                                                 request_s):
    from dmi.storage.native_capture import _load_native_store_extension

    module = _load_native_store_extension()
    module.StorageService(_service_config(
        tmp_path, lease_ttl_ns=ttl_ns, clock_skew_ns=skew_ns,
        clickhouse_request_timeout_s=request_s))
