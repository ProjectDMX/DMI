"""A2a conformance: the native SigV4 signer must match botocore exactly.

botocore is the oracle (the Python S3 store signs through it). The test
freezes botocore's clock, signs a battery of generated requests both ways,
and requires identical Authorization headers and canonical requests.

The native side runs through ``native/build/conformance_sign`` (no pybind,
no network, CPU gate).

Build: make -C native build/conformance_sign
"""

from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVER = REPO_ROOT / "native" / "build" / "conformance_sign"

pytestmark = pytest.mark.cpu

if not DRIVER.exists():
    pytest.skip(
        "native/build/conformance_sign is not built; run "
        "`make -C native build/conformance_sign`",
        allow_module_level=True,
    )

import botocore.auth  # noqa: E402
import botocore.awsrequest  # noqa: E402
import botocore.credentials  # noqa: E402
import botocore.utils  # noqa: E402

FROZEN = datetime_module.datetime(2026, 9, 5, 12, 0, 0)
DATESTAMP = "20260905"
AMZ_DATE = "20260905T120000Z"


class _FrozenDateTime(datetime_module.datetime):
    @classmethod
    def utcnow(cls):
        return FROZEN


def _botocore_sign(
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    region: str,
    service: str,
    access: str,
    secret: str,
    token: str | None,
) -> tuple[str, str]:
    credentials = botocore.credentials.Credentials(access, secret, token)
    auth = botocore.auth.SigV4Auth(credentials, service, region)
    request = botocore.awsrequest.AWSRequest(
        method=method, url=url, data=body, headers=dict(headers)
    )
    # botocore.auth binds get_current_datetime at import
    # (from botocore.compat import ...), so freeze it in ITS namespace.
    # Capture the canonical request AFTER add_auth: _modify_request_before_signing
    # injects x-amz-date (and the session token), and the canonical form must
    # reflect exactly what was signed. Strip only the Authorization header —
    # canonical_request would otherwise fold the signature into itself.
    with mock.patch.object(
        botocore.auth, "get_current_datetime", return_value=FROZEN
    ):
        auth.add_auth(request)
    prepared = request.prepare()
    signed_headers = {
        k: v for k, v in request.headers.items() if k.lower() != "authorization"
    }
    canon_request = botocore.awsrequest.AWSRequest(
        method=method, url=url, data=body, headers=signed_headers
    )
    return prepared.headers["Authorization"], auth.canonical_request(canon_request)


def _native_sign(
    method: str,
    path: str,
    query: list,
    headers: dict,
    payload_hash: str,
    access: str,
    secret: str,
    token: str | None,
    region: str,
    service: str,
) -> tuple[str, str]:
    request = {
        "op": "sign",
        "method": method,
        "path": path,
        "query": query,
        "headers": headers,
        "payload_hash": payload_hash,
        "access": access,
        "secret": secret,
        "token": token,
        "region": region,
        "service": service,
        "datestamp": DATESTAMP,
        "amz_date": AMZ_DATE,
    }
    proc = subprocess.run(
        [str(DRIVER)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    response = json.loads(proc.stdout.strip())
    assert response["ok"], response
    return response["authorization"], response["canonical"]


def _check(
    method: str,
    host: str,
    path: str,
    query_string: str,
    extra_headers: dict,
    body: bytes,
    region: str = "us-east-1",
    service: str = "s3",
    token: str | None = None,
):
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {
        "host": host,
        "x-amz-date": AMZ_DATE,
        "x-amz-content-sha256": payload_hash,
        **extra_headers,
    }
    url = f"https://{host}{path}"
    if query_string:
        url += "?" + query_string
    expected_auth, expected_canon = _botocore_sign(
        method, url, headers, body, region, service,
        "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", token,
    )
    # The native driver takes the raw path and raw query pairs (it encodes).
    pairs = []
    for part in query_string.split("&"):
        if not part:
            continue
        name, _, value = part.partition("=")
        from urllib.parse import unquote

        pairs.append([unquote(name), unquote(value)])
    got_auth, got_canon = _native_sign(
        method, path, pairs, headers, payload_hash,
        "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", token,
        region, service,
    )
    assert got_canon == expected_canon, (
        f"canonical request differs for {method} {path}?{query_string}\n"
        f"botocore: {expected_canon!r}\n"
        f"native:   {got_canon!r}"
    )
    assert got_auth == expected_auth, (
        f"authorization differs for {method} {path}?{query_string}\n"
        f"botocore: {expected_auth!r}\n"
        f"native:   {got_auth!r}"
    )


def test_sign_get_object_path_style():
    _check("GET", "s3.example.com", "/bucket/packs/golden.dmi-pack", "",
           {}, b"")


def test_sign_put_object_with_metadata():
    _check(
        "PUT", "s3.example.com", "/bucket/v1/tenant=t/pack.dmi-pack", "",
        {
            "content-type": "application/vnd.dmi.pack",
            "x-amz-meta-dmi-format": "dmi-pack-v1",
            "x-amz-meta-dmi-sha256": "ab" * 32,
        },
        b"hello-payload",
    )


def test_sign_head_object():
    _check("HEAD", "s3.example.com:3900", "/bucket/some-key", "", {}, b"")


def test_sign_delete_object():
    _check("DELETE", "s3.example.com", "/bucket/old-pack", "", {}, b"")


def test_sign_multipart_subresources():
    _check("POST", "s3.example.com", "/bucket/big-pack", "uploads=", {}, b"")
    _check(
        "PUT", "s3.example.com", "/bucket/big-pack",
        "partNumber=3&uploadId=abc123", {}, b"x" * 1024,
    )
    _check(
        "POST", "s3.example.com", "/bucket/big-pack",
        "uploadId=abc123", {"content-type": "application/xml"},
        b"<CompleteMultipartUpload/>",
    )


def test_sign_list_v2_query():
    _check(
        "GET", "s3.example.com", "/bucket",
        "list-type=2&prefix=v1%2Ftenant%3Dt&max-keys=64&delimiter=%2F"
        "&continuation-token=abc&encoding-type=url&fetch-owner=true",
        {}, b"",
    )


def test_sign_nasty_key_characters():
    _check(
        "GET", "s3.example.com",
        "/bucket/v1/tenant=a b+~/caf\u00e9\u65e5\u672c/end=",
        "", {}, b"",
    )


def test_sign_session_token():
    _check(
        "GET", "s3.example.com", "/bucket/key", "",
        {}, b"", token="session-token-value",
    )


def test_sign_other_region_and_service():
    _check(
        "GET", "iam.amazonaws.com", "/", "Action=ListUsers&Version=2010-05-08",
        {}, b"", region="us-east-1", service="iam",
    )


def test_sign_header_whitespace_normalization():
    _check(
        "PUT", "s3.example.com", "/bucket/key", "",
        {"x-amz-meta-note": "  padded\t internal   spaces  "},
        b"body",
    )


def test_sign_empty_and_large_bodies():
    _check("PUT", "s3.example.com", "/bucket/empty", "", {}, b"")
    _check("PUT", "s3.example.com", "/bucket/big", "", {}, b"z" * (1024 * 1024))
