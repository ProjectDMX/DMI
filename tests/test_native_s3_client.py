"""A2a/A2b: the native S3 client against a fake S3 with fault injection.

Every request the C++ client sends is signature-verified server-side with
botocore (the oracle): the server rebuilds the request, freezes botocore's
clock to the received x-amz-date, re-signs, and requires an identical
Authorization header. A 403 on any test therefore means a wire-level signing
divergence, not just a failed assertion.

No network beyond localhost; no Garage needed. Live-Garage equivalence stays
behind the `garage` marker in the Python suite.

Build: make -C native build/conformance_store
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVER = REPO_ROOT / "native" / "build" / "conformance_store"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_store is not built; run "
        "`make -C native build/conformance_store`",
    ),
]

import botocore.auth  # noqa: E402
import botocore.awsrequest  # noqa: E402
import botocore.credentials  # noqa: E402

ACCESS = "test-access"
SECRET = "test-secret"
BUCKET = "test-bucket"
REGION = "us-east-1"


class FakeS3State:
    def __init__(self):
        self.lock = threading.Lock()
        self.objects: dict[str, dict] = {}
        self.uploads: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.fault_counts: dict[str, int] = {}


STATE = FakeS3State()


def _verify_signature(handler, body: bytes) -> bool:
    """Re-sign the received request with botocore; require identical auth.

    Mirrors S3 verification: only the headers named in the request's own
    SignedHeaders list enter the rebuilt canonical request. libcurl adds
    unsigned headers (Accept, Content-Length) that must not participate.
    The path is percent-DECODED once first, exactly like S3: the client
    wires the encoded form (tenant%3Dt) and S3 verifies against the key
    (tenant=t). Feeding the encoded path to botocore would double-encode
    (%25) and reject every key carrying a reserved character.
    """
    import datetime as datetime_module
    import re
    from urllib.parse import unquote, urlsplit

    received_auth = handler.headers.get("Authorization", "")
    signed = re.search(r"SignedHeaders=([^,]+)", received_auth)
    if not signed:
        return False
    wanted = set(signed.group(1).split(";"))
    amz_date = handler.headers.get("x-amz-date", "")
    try:
        frozen = datetime_module.datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return False
    headers = {k.lower(): v for k, v in handler.headers.items()
               if k.lower() in wanted}
    parts = urlsplit(handler.path)
    decoded = unquote(parts.path)
    url = f"http://{handler.headers.get('Host', 'localhost')}{decoded}"
    if parts.query:
        url += "?" + parts.query
    credentials = botocore.credentials.Credentials(ACCESS, SECRET, None)
    auth = botocore.auth.SigV4Auth(credentials, "s3", REGION)
    request = botocore.awsrequest.AWSRequest(
        method=handler.command, url=url, data=body, headers=headers
    )
    with mock.patch.object(
        botocore.auth, "get_current_datetime", return_value=frozen
    ):
        auth.add_auth(request)
    return request.prepare().headers.get("Authorization") == received_auth


class FakeS3Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _record(self, body: bytes):
        with STATE.lock:
            STATE.calls.append(
                {"method": self.command, "path": self.path,
                 "headers": dict(self.headers), "body_len": len(body)}
            )

    def _send(self, status: int, headers: dict, body: bytes = b""):
        payload = body
        self.send_response(status)
        # HEAD responses carry the OBJECT's size with an empty body: never
        # emit a second Content-Length for the (absent) payload.
        if not any(name.lower() == "content-length" for name in headers):
            self.send_header("Content-Length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD" and payload:
            self.wfile.write(payload)

    def _reject_unsigned(self, body: bytes) -> bool:
        if not _verify_signature(self, body):
            self._send(403, {}, b"bad signature")
            return True
        return False

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _split(self):
        from urllib.parse import unquote

        parts = urlsplit(self.path)
        # Decode %XX once for routing/storage, exactly like S3: the client
        # wires the encoded form and S3 routes on the key. (unquote, never
        # unquote_plus: '+' in a path is a literal plus, not a space.)
        path = unquote(parts.path)
        query = {}
        for part in parts.query.split("&"):
            if not part:
                continue
            name, _, value = part.partition("=")
            # S3 query values arrive percent-encoded (prefix=v1%2F); the
            # server must decode before comparing, exactly like S3 does.
            query[unquote(name)] = unquote(value)
        return path, query

    def _maybe_fault(self, key: str):
        """Returns a (status, body) override, or None for normal handling.

        Fault keys match by segment: any key under fault/<name>/ behaves the
        same, so spool-layout keys (which must end in <pack_id>.dmi-pack)
        can trigger faults too.
        """
        def under(name: str) -> bool:
            return key == name or key.startswith(name + "/")

        with STATE.lock:
            n = STATE.fault_counts.get(key, 0)
            STATE.fault_counts[key] = n + 1
        if under("fault/once-500") and n == 0:
            return 500, b"boom"
        if under("fault/slow-once-500") and n == 0:
            # Outlive the one-second resolution of x-amz-date, so a retry
            # that reuses the first attempt's signature is visible.
            time.sleep(1.2)
            return 500, b"boom"
        if under("fault/always-500"):
            return 500, b"boom"
        if under("fault/forbidden"):
            return 403, b"no"
        if under("fault/hang"):
            time.sleep(5)
            return None
        return None

    def _route(self):
        path, query = self._split()
        segments = path.lstrip("/").split("/", 1)
        if len(segments) != 2 or segments[0] != BUCKET:
            self._send(404, {}, b"no bucket")
            return
        key = segments[1]
        body = self._read_body()
        self._record(body)
        if self._reject_unsigned(body):
            return
        fault = self._maybe_fault(key)
        if fault is not None:
            status, payload = fault
            self._send(status, {}, payload)
            return

        if self.command == "PUT" and "partNumber" in query:
            self._handle_put_part(key, query, body)
        elif self.command == "PUT":
            self._handle_put(key, body)
        elif self.command == "GET" and "uploadId" not in query and \
                ("list-type" in query or path.rstrip("/") == f"/{BUCKET}"):
            self._handle_list(query)
        elif self.command == "GET":
            self._handle_get(key)
        elif self.command == "HEAD":
            self._handle_head(key)
        elif self.command == "DELETE" and "uploadId" in query:
            self._handle_abort(key, query)
        elif self.command == "DELETE":
            self._handle_delete(key)
        elif self.command == "POST" and "uploads" in query:
            self._handle_create(key)
        elif self.command == "POST":
            self._handle_complete(key, query, body)
        else:
            self._send(400, {}, b"nope")

    do_GET = do_PUT = do_HEAD = do_DELETE = do_POST = _route

    # -- handlers ---------------------------------------------------------
    def _meta_out(self, meta: dict) -> dict:
        return {f"x-amz-meta-{k}": v for k, v in meta.items()}

    def _handle_put(self, key: str, body: bytes):
        meta = {k[11:]: v for k, v in self.headers.items()
                if k.lower().startswith("x-amz-meta-")}
        with STATE.lock:
            STATE.objects[key] = {
                "body": body,
                "meta": meta,
                "content_type": self.headers.get("Content-Type", ""),
                "etag": f'"{hashlib.md5(body).hexdigest()}"',
            }
        self._send(200, {"ETag": STATE.objects[key]["etag"]})

    def _handle_get(self, key: str):
        with STATE.lock:
            obj = STATE.objects.get(key)
        if obj is None:
            self._send(404, {}, b"missing")
            return
        body = obj["body"]
        if key == "fault/short":
            # Lie the way a truncated transfer does: 200 with a body shorter
            # than the requested range, correctly framed so the connection
            # does not hang — the client must refuse by length, not by hang.
            truncated = body[: len(body) // 2] or body[:1]
            self._send(200, {"Content-Length": str(len(truncated)),
                             "ETag": obj["etag"]}, truncated)
            return
        headers = {"ETag": obj["etag"],
                   **self._meta_out(obj["meta"])}
        range_header = self.headers.get("Range")
        if range_header:
            unit, _, spec = range_header.partition("=")
            start, _, end = spec.partition("-")
            start, end = int(start), int(end)
            headers["Content-Range"] = (
                f"bytes {start}-{end}/{len(body)}")
            self._send(206, headers, body[start : end + 1])
        else:
            self._send(200, headers, body)

    def _handle_head(self, key: str):
        with STATE.lock:
            obj = STATE.objects.get(key)
        if obj is None:
            self._send(404, {})
            return
        self._send(200, {"Content-Length": str(len(obj["body"])),
                         "ETag": obj["etag"],
                         **self._meta_out(obj["meta"])})

    def _handle_delete(self, key: str):
        with STATE.lock:
            STATE.objects.pop(key, None)
        self._send(204, {})

    def _handle_list(self, query: dict):
        prefix = query.get("prefix", "")
        max_keys = int(query.get("max-keys", "1000"))
        token = query.get("continuation-token", "")
        with STATE.lock:
            keys = sorted(k for k in STATE.objects if k.startswith(prefix))
        start = 0
        if token:
            start = int(token)
        page = keys[start : start + max_keys]
        truncated = start + max_keys < len(keys)
        xml = ['<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
               f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"]
        if truncated:
            xml.append(f"<NextContinuationToken>{start + max_keys}</NextContinuationToken>")
        for key in page:
            with STATE.lock:
                obj = STATE.objects[key]
            xml.append(f"<Contents><Key>{key}</Key>"
                       f"<Size>{len(obj['body'])}</Size>"
                       f"<ETag>{obj['etag']}</ETag></Contents>")
        xml.append("</ListBucketResult>")
        self._send(200, {"Content-Type": "application/xml"},
                   "".join(xml).encode())

    def _handle_create(self, key: str):
        with STATE.lock:
            upload_id = f"upload-{len(STATE.uploads)}"
            STATE.uploads[upload_id] = {"key": key, "parts": {},
                                        "meta": {k[11:]: v for k, v in self.headers.items()
                                                 if k.lower().startswith("x-amz-meta-")}}
        self._send(200, {}, f"<InitiateMultipartUploadResult><UploadId>{upload_id}</UploadId></InitiateMultipartUploadResult>".encode())

    def _handle_put_part(self, key: str, query: dict, body: bytes):
        upload = STATE.uploads.get(query.get("uploadId", ""))
        if upload is None or upload["key"] != key:
            self._send(404, {}, b"no upload")
            return
        part = int(query.get("partNumber", "0"))
        with STATE.lock:
            upload["parts"][part] = body
        self._send(200, {"ETag": f'"{hashlib.md5(body).hexdigest()}"'})

    def _handle_complete(self, key: str, query: dict, body: bytes):
        import re

        with STATE.lock:
            upload = STATE.uploads.pop(query.get("uploadId", ""), None)
        if upload is None or upload["key"] != key:
            self._send(404, {}, b"no upload")
            return
        xml = body.decode("utf-8")
        if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
            self._send(400, {}, b"unsafe xml")
            return
        try:
            numbers = [
                int(part)
                for part in re.findall(
                    r"<(?:\w+:)?PartNumber>\s*(\d+)\s*</(?:\w+:)?PartNumber>",
                    xml,
                )
            ]
        except ValueError:
            self._send(400, {}, b"invalid xml")
            return
        if not numbers:
            self._send(400, {}, b"invalid xml")
            return
        ordered = sorted(numbers)
        # S3's rule: every part but the last is at least 5 MiB.
        if any(len(upload["parts"][n]) < 5 * 1024 * 1024
               for n in ordered[:-1]):
            self._send(400, {}, b"<Error><Code>EntityTooSmall</Code></Error>")
            return
        assembled = b"".join(upload["parts"][n] for n in ordered)
        meta = upload["meta"]
        with STATE.lock:
            STATE.objects[key] = {
                "body": assembled,
                "meta": meta,
                "content_type": "",
                "etag": f'"{hashlib.md5(assembled).hexdigest()}-multipart"',
            }
        self._send(200, {}, f"<CompleteMultipartUploadResult><ETag>{STATE.objects[key]['etag']}</ETag></CompleteMultipartUploadResult>".encode())

    def _handle_abort(self, key: str, query: dict):
        with STATE.lock:
            STATE.uploads.pop(query.get("uploadId", ""), None)
        self._send(204, {})


def _reset_state():
    STATE.objects.clear()
    STATE.uploads.clear()
    STATE.calls.clear()
    STATE.fault_counts.clear()


@pytest.fixture()
def fake_s3():
    _reset_state()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# --- TLS ---------------------------------------------------------------------
#
# The same signature-verifying fake, behind TLS with a server certificate
# issued by a private CA generated here. Nothing about the CA is installed
# anywhere: a client trusts it only when told to (ca_file / ca_path).


def _openssl(*args: str, cwd: Path) -> str:
    return subprocess.run(["openssl", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture(scope="session")
def private_ca(tmp_path_factory):
    """(ca_file, ca_path, server_cert, server_key) for 127.0.0.1."""
    if shutil.which("openssl") is None:
        pytest.skip("the openssl CLI is needed to mint the test CA")
    root = tmp_path_factory.mktemp("private-ca")
    # Own config files, not the system openssl.cnf: its v3_ca section adds a
    # basicConstraints of its own, and a duplicated extension makes OpenSSL
    # reject the CA ("unable to get local issuer certificate").
    (root / "ca.cnf").write_text(
        "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=v3_ca\n"
        "[dn]\nCN=DMI test private CA\n"
        "[v3_ca]\nbasicConstraints=critical,CA:TRUE\n"
        "keyUsage=critical,keyCertSign,cRLSign\n"
        "subjectKeyIdentifier=hash\n")
    (root / "server.cnf").write_text(
        "[req]\nprompt=no\ndistinguished_name=dn\n[dn]\nCN=127.0.0.1\n")
    (root / "server.ext").write_text(
        "basicConstraints=CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=IP:127.0.0.1,DNS:localhost\n"
        "authorityKeyIdentifier=keyid\n")
    _openssl("req", "-config", "ca.cnf", "-x509", "-newkey", "rsa:2048",
             "-nodes", "-keyout", "ca.key", "-out", "ca.pem", "-days", "2",
             cwd=root)
    _openssl("req", "-config", "server.cnf", "-new", "-newkey", "rsa:2048",
             "-nodes", "-keyout", "server.key", "-out", "server.csr",
             cwd=root)
    _openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem",
             "-CAkey", "ca.key", "-CAcreateserial", "-out", "server.pem",
             "-days", "2", "-extfile", "server.ext", cwd=root)
    _openssl("verify", "-CAfile", "ca.pem", "server.pem", cwd=root)
    # CURLOPT_CAPATH reads an OpenSSL-hashed directory: <subject hash>.0.
    ca_dir = root / "ca-dir"
    ca_dir.mkdir()
    subject_hash = _openssl("x509", "-hash", "-noout", "-in", "ca.pem",
                            cwd=root).strip()
    shutil.copy(root / "ca.pem", ca_dir / f"{subject_hash}.0")
    return (str(root / "ca.pem"), str(ca_dir), str(root / "server.pem"),
            str(root / "server.key"))


class _QuietTLSServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A client that refuses the certificate aborts the handshake; that is
        # the outcome under test, not a server fault worth a traceback.
        if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError)):
            return
        super().handle_error(request, client_address)


@pytest.fixture()
def fake_s3_tls(private_ca):
    """https://127.0.0.1:<port> served with the private CA's certificate."""
    _reset_state()
    _ca_file, _ca_path, cert, key = private_ca
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = _QuietTLSServer(("127.0.0.1", 0), FakeS3Handler)
    # The handshake runs lazily, on the handler's thread, so a client that
    # hangs or aborts it cannot stall the accept loop.
    server.socket = context.wrap_socket(server.socket, server_side=True,
                                        do_handshake_on_connect=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"https://127.0.0.1:{server.server_port}"
    server.shutdown()


def _base(endpoint: str, **overrides) -> dict:
    request = {
        "endpoint": endpoint,
        "bucket": BUCKET,
        "region": REGION,
        "access": ACCESS,
        "secret": SECRET,
        "token": None,
        "insecure": True,
        "connect_timeout": 5,
        "read_timeout": 10,
        "max_attempts": 4,
    }
    # Overrides win (e.g. a fault test that shortens timeouts).
    for name in ("connect_timeout", "read_timeout", "max_attempts",
                 "multipart_threshold", "multipart_chunk"):
        if name in overrides:
            request[name] = overrides.pop(name)
    request.update(overrides)
    return request


def _call(op: str, **fields) -> dict:
    request = {"op": op, **fields}
    proc = subprocess.run(
        [str(DRIVER)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"driver produced no output: {proc.stderr}"
    return json.loads(lines[0])


def test_put_get_head_delete_round_trip(fake_s3):
    payload = bytes(range(256)) * 64
    meta = {"dmi-format": "dmi-pack-v1", "dmi-sha256": "ab" * 32}
    put = _call("put", **_base(fake_s3), key="packs/a.dmi-pack",
                data_b64=base64.b64encode(payload).decode(), metadata=meta,
                content_type="application/vnd.dmi.pack")
    assert put["ok"], put
    assert put["attempts"] == 1

    head = _call("head", **_base(fake_s3), key="packs/a.dmi-pack")
    assert head["ok"] and head["found"] and head["size"] == len(payload)
    assert head["metadata"] == meta

    get = _call("get", **_base(fake_s3), key="packs/a.dmi-pack",
                offset=100, length=500)
    assert get["ok"] and base64.b64decode(get["data_b64"]) == payload[100:600]

    full = _call("get", **_base(fake_s3), key="packs/a.dmi-pack",
                 offset=0, length=len(payload))
    assert full["ok"] and base64.b64decode(full["data_b64"]) == payload

    assert _call("delete", **_base(fake_s3),
                 key="packs/a.dmi-pack")["ok"]
    missing = _call("head", **_base(fake_s3), key="packs/a.dmi-pack")
    assert missing["ok"] and not missing["found"]


MIB = 1024 * 1024


def test_put_multipart_round_trip(fake_s3):
    """Real part sizes: S3 refuses a part under 5 MiB unless it is the last.

    12 MiB in 5 MiB parts is two full parts and a 2 MiB tail -- the
    smallest shape with both a minimum-size part and a short final one.
    """
    payload = bytes((i * 7) & 0xFF for i in range(12 * MIB))
    put = _call("put", **_base(fake_s3), key="packs/big.dmi-pack",
                data_b64=base64.b64encode(payload).decode(), metadata={},
                content_type="application/vnd.dmi.pack",
                multipart_threshold=5 * MIB, multipart_chunk=5 * MIB)
    assert put["ok"], put
    parts = [call["body_len"] for call in STATE.calls
             if call["method"] == "PUT" and "partNumber=" in call["path"]]
    assert parts == [5 * MIB, 5 * MIB, 2 * MIB]
    echo = _call("get", **_base(fake_s3), key="packs/big.dmi-pack",
                 offset=0, length=len(payload))
    assert echo["ok"] and base64.b64decode(echo["data_b64"]) == payload


def test_multipart_part_under_5_mib_is_refused_before_any_request(fake_s3):
    """The client refuses the part size, as the Python store does (s3.py).

    Left to the server, a sub-5 MiB part fails only at
    CompleteMultipartUpload, after every part was sent -- and a payload
    under the threshold never notices. The refusal is the configuration's,
    so it lands on every operation, before any request.
    """
    payload = bytes(3 * MIB)
    put = _call("put", **_base(fake_s3), key="packs/small-parts.dmi-pack",
                data_b64=base64.b64encode(payload).decode(), metadata={},
                content_type="application/vnd.dmi.pack",
                multipart_threshold=MIB, multipart_chunk=MIB)
    assert not put["ok"], put
    assert "multipart_chunk_bytes" in put["what"], put
    assert str(5 * MIB) in put["what"], put
    head = _call("head", **_base(fake_s3), key="anything",
                 multipart_chunk=5 * MIB - 1)
    assert not head["ok"] and "multipart_chunk_bytes" in head["what"], head
    assert STATE.calls == []
    # Exactly 5 MiB is S3's minimum, and is accepted.
    assert _call("head", **_base(fake_s3), key="anything",
                 multipart_chunk=5 * MIB)["ok"]


def test_list_pagination(fake_s3):
    for i in range(5):
        ack = _call("put", **_base(fake_s3), key=f"v1/pack-{i}",
                    data_b64=base64.b64encode(b"x").decode(), metadata={},
                    content_type="application/octet-stream")
        assert ack["ok"], ack
    first = _call("list", **_base(fake_s3), prefix="v1/", delimiter="",
                  max_keys=2, continuation="")
    assert first["ok"] and len(first["objects"]) == 2 and first["truncated"]
    second = _call("list", **_base(fake_s3), prefix="v1/", delimiter="",
                   max_keys=10, continuation=first["next_token"])
    assert second["ok"] and len(second["objects"]) == 3
    assert not second["truncated"]
    assert [o["key"] for o in first["objects"] + second["objects"]] == \
        [f"v1/pack-{i}" for i in range(5)]


def test_retry_then_success_on_500(fake_s3):
    put = _call("put", **_base(fake_s3), key="fault/once-500",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert put["ok"], put
    assert put["attempts"] == 2


def _header(call: dict, name: str) -> str:
    return next(v for k, v in call["headers"].items() if k.lower() == name)


def test_every_attempt_is_signed_afresh(fake_s3):
    """A retry carries its own x-amz-date and signature, both still valid.

    SigV4 binds the signature to x-amz-date, and S3 refuses a request whose
    date is more than 15 minutes off. Signing once before the loop replayed
    the first attempt's date on every retry, so a slow first attempt (up to
    read_timeout_s each, plus backoff) aged every later one. The first
    attempt here outlives a second before failing with a 500: a fresh
    signature must carry a later date. Both attempts passed the server's
    botocore re-signing check -- the 500 is only reached after it -- and
    the second one stored the object.
    """
    put = _call("put", **_base(fake_s3), key="fault/slow-once-500",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert put["ok"], put
    assert put["attempts"] == 2
    attempts = [call for call in STATE.calls
                if call["path"].endswith("/fault/slow-once-500")]
    assert len(attempts) == 2, attempts
    first, second = attempts
    assert _header(second, "x-amz-date") > _header(first, "x-amz-date")
    assert _header(second, "authorization") != \
        _header(first, "authorization")
    assert STATE.objects["fault/slow-once-500"]["body"] == b"data"


def test_no_retry_on_403(fake_s3):
    put = _call("put", **_base(fake_s3), key="fault/forbidden",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert not put["ok"]
    assert put["attempts"] == 1


def test_gives_up_after_max_attempts(fake_s3):
    put = _call("put", **_base(fake_s3, max_attempts=3),
                key="fault/always-500",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert not put["ok"]
    assert put["attempts"] == 3


def test_short_range_body_is_an_error(fake_s3):
    payload = b"0123456789abcdef"
    assert _call("put", **_base(fake_s3), key="fault/short",
                 data_b64=base64.b64encode(payload).decode(), metadata={},
                 content_type="application/octet-stream")["ok"]
    # Ask for the full length; the server truncates.
    get = _call("get", **_base(fake_s3), key="fault/short",
                offset=0, length=len(payload))
    assert not get["ok"] and "range" in get["what"].lower()


def test_timeout_retries_then_fails(fake_s3):
    put = _call("put", **_base(fake_s3, read_timeout=1, max_attempts=2),
                key="fault/hang",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert not put["ok"]
    assert put["attempts"] == 2


# --- integer bounds ------------------------------------------------------------
#
# The transport's integer fields are 64-bit on the wire, and FindInt reported
# an unrepresentable literal as -1. Every site here reads -1 as something
# legal: the timeouts and attempt counts treat "> 0" as "given" and silently
# fall back to 5s/120s/4 attempts, the multipart sizes keep their defaults,
# and a range offset or length casts it to 18446744073709551615 -- so the
# caller's own bound is quietly replaced by a different one.


@pytest.mark.parametrize(
    "field",
    ("connect_timeout", "read_timeout", "max_attempts",
     "multipart_threshold", "multipart_chunk"),
)
def test_transport_config_refuses_an_out_of_range_integer(field):
    """No server needed: the refusal has to land before any request goes out."""
    request = _base("http://127.0.0.1:1", **{field: 2**64 + 1})
    response = _call("head", key="anything", **request)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert field in response["what"], response


@pytest.mark.parametrize(
    "field, value",
    [
        ("offset", 2**64 + 1),
        ("length", 2**64 + 1),
        ("offset", int("9" * 40)),
        ("length", -(2**63) - 1),
    ],
)
def test_get_refuses_an_out_of_range_range(fake_s3, field, value):
    payload = b"0123456789abcdef"
    assert _call("put", **_base(fake_s3), key="bounds/obj",
                 data_b64=base64.b64encode(payload).decode(), metadata={},
                 content_type="application/octet-stream")["ok"]
    fields = {"offset": 0, "length": len(payload)}
    fields[field] = value
    get = _call("get", **_base(fake_s3), key="bounds/obj", **fields)
    assert not get["ok"], get
    assert "out of range" in get["what"], get
    assert field in get["what"], get


def test_list_refuses_an_out_of_range_max_keys(fake_s3):
    response = _call("list", **_base(fake_s3), prefix="", delimiter="",
                     max_keys=2**64 + 1, continuation="")
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert "max_keys" in response["what"], response


def test_the_64_bit_boundaries_still_reach_the_transport(fake_s3):
    """INT64_MAX and 2**64 - 1 parse; only wider literals are refused.

    A length of 2**64 - 1 is a legal 64-bit value, so it must reach the
    client and be refused (if at all) by the range check there -- not by the
    JSON scan, and not by being turned into something else.
    """
    payload = b"0123456789abcdef"
    assert _call("put", **_base(fake_s3), key="bounds/limits",
                 data_b64=base64.b64encode(payload).decode(), metadata={},
                 content_type="application/octet-stream")["ok"]
    for length in (2**63 - 1, 2**64 - 1):
        get = _call("get", **_base(fake_s3), key="bounds/limits",
                    offset=0, length=length)
        assert not get["ok"], (length, get)
        assert "out of range" not in get["what"], (length, get)
    # The ordinary read still works, byte for byte.
    get = _call("get", **_base(fake_s3), key="bounds/limits", offset=0,
                length=len(payload))
    assert get["ok"], get
    assert base64.b64decode(get["data_b64"]) == payload


# --- https with a private CA --------------------------------------------------


def _tls(endpoint: str, **overrides) -> dict:
    return _base(endpoint, insecure=False, **overrides)


@pytest.mark.parametrize("trust", ["ca_file", "ca_path"])
def test_https_round_trip_trusts_a_private_ca(fake_s3_tls, private_ca, trust):
    ca_file, ca_path, _cert, _key = private_ca
    anchor = {"ca_file": ca_file} if trust == "ca_file" else {"ca_path": ca_path}
    payload = bytes(range(256)) * 16
    meta = {"dmi-format": "dmi-pack-v1"}
    put = _call("put", **_tls(fake_s3_tls, **anchor), key="tls/a",
                data_b64=base64.b64encode(payload).decode(), metadata=meta,
                content_type="application/octet-stream")
    assert put["ok"], put
    head = _call("head", **_tls(fake_s3_tls, **anchor), key="tls/a")
    assert head["ok"] and head["found"] and head["metadata"] == meta, head
    get = _call("get", **_tls(fake_s3_tls, **anchor), key="tls/a",
                offset=0, length=len(payload))
    assert get["ok"], get
    assert base64.b64decode(get["data_b64"]) == payload


def test_https_without_the_private_ca_is_refused(fake_s3_tls):
    """libcurl's default trust store does not know the CA: no request lands.

    A certificate failure is not transient, so it is not retried.
    """
    put = _call("put", **_tls(fake_s3_tls), key="tls/untrusted",
                data_b64=base64.b64encode(b"data").decode(), metadata={},
                content_type="application/octet-stream")
    assert not put["ok"], put
    assert "certificate" in put["what"].lower(), put
    assert put["attempts"] == 1, put
    assert STATE.calls == [] and STATE.objects == {}


def test_ca_options_on_plain_http_are_refused(fake_s3, private_ca):
    ca_file, ca_path, _cert, _key = private_ca
    for anchor in ({"ca_file": ca_file}, {"ca_path": ca_path}):
        head = _call("head", **_base(fake_s3, **anchor), key="anything")
        assert not head["ok"], head
        assert "https" in head["what"], head
    assert STATE.calls == []


def test_a_missing_ca_is_named_before_any_request(fake_s3_tls, tmp_path):
    missing = tmp_path / "no-such-ca.pem"
    head = _call("head", **_tls(fake_s3_tls, ca_file=str(missing)),
                 key="anything")
    assert not head["ok"] and str(missing) in head["what"], head
    head = _call("head", **_tls(fake_s3_tls, ca_path=str(missing)),
                 key="anything")
    assert not head["ok"] and str(missing) in head["what"], head
    assert STATE.calls == []
