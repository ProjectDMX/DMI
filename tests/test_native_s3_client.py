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
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVER = REPO_ROOT / "native" / "build" / "conformance_store"

pytestmark = pytest.mark.cpu

if not DRIVER.exists():
    pytest.skip(
        "native/build/conformance_store is not built; run "
        "`make -C native build/conformance_store`",
        allow_module_level=True,
    )

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
    """
    import datetime as datetime_module
    import re

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
    url = f"http://{handler.headers.get('Host', 'localhost')}{handler.path}"
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
        query = {}
        for part in parts.query.split("&"):
            if not part:
                continue
            name, _, value = part.partition("=")
            # S3 query values arrive percent-encoded (prefix=v1%2F); the
            # server must decode before comparing, exactly like S3 does.
            query[unquote(name)] = unquote(value)
        return parts.path, query

    def _maybe_fault(self, key: str):
        """Returns a (status, body) override, or None for normal handling."""
        with STATE.lock:
            n = STATE.fault_counts.get(key, 0)
            STATE.fault_counts[key] = n + 1
        if key == "fault/once-500" and n == 0:
            return 500, b"boom"
        if key == "fault/always-500":
            return 500, b"boom"
        if key == "fault/forbidden":
            return 403, b"no"
        if key == "fault/hang":
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
        with STATE.lock:
            upload = STATE.uploads.pop(query.get("uploadId", ""), None)
        if upload is None or upload["key"] != key:
            self._send(404, {}, b"no upload")
            return
        root = ET.fromstring(body)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        numbers = [int(p.find("s3:PartNumber", ns).text) for p in root.findall("s3:Part", ns)]
        assembled = b"".join(upload["parts"][n] for n in sorted(numbers))
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


@pytest.fixture()
def fake_s3():
    STATE.objects.clear()
    STATE.uploads.clear()
    STATE.calls.clear()
    STATE.fault_counts.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeS3Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
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


def test_put_multipart_round_trip(fake_s3):
    payload = bytes((i * 7) & 0xFF for i in range(3 * 1024 * 1024))
    put = _call("put", **_base(fake_s3), key="packs/big.dmi-pack",
                data_b64=base64.b64encode(payload).decode(), metadata={},
                content_type="application/vnd.dmi.pack",
                multipart_threshold=1024 * 1024, multipart_chunk=1024 * 1024)
    assert put["ok"], put
    echo = _call("get", **_base(fake_s3), key="packs/big.dmi-pack",
                 offset=0, length=len(payload))
    assert echo["ok"] and base64.b64decode(echo["data_b64"]) == payload


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
