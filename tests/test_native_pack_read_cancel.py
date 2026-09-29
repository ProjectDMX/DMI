"""How a pack's failed index read is booked: the store's, or a cancel's.

The storage service reads a pack's trailer and footer through an S3 client
holding a Cancellation -- stop()'s, and a flush's read deadline. A read the
store did not answer ends the index pass: recorded as the cycle's error,
counted towards the backoff. A read the cancel cut is no failure at all:
the pack is deferred, nothing recorded. read_pack_descriptor_rows told the
two apart by asking the Cancellation, when the read had failed, whether it
was set -- which a flush's deadline makes true from the moment it passes,
whatever ended the read. A store failure answered in the gap between the
deadline and libcurl's next progress poll was then booked as a cancel, and
the flush's TimeoutError named an older error. It asks the read itself now,
as the uploader does since b61db70.

That gap is under a poll interval wide, so the driver stands in for it:
conformance_catalog's session-less read_pack_rows op cancels its client's
Cancellation once an exchange has returned (cancel_after_exchange), through
the S3 client's test seam, or arms its deadline cancel_after_ms from the
call. No catalog is involved.

Build: make -C native build/conformance_catalog
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "native" / "build" / "conformance_catalog"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]

from tests.test_native_s3_client import (  # noqa: E402
    ACCESS,
    BUCKET,
    REGION,
    SECRET,
    fake_s3,
)


def _read(endpoint: str, key: str, **fields) -> dict:
    request = {
        "op": "read_pack_rows", "endpoint": endpoint, "bucket": BUCKET,
        "region": REGION, "access": ACCESS, "secret": SECRET,
        "insecure": True, "read_timeout": 10, "max_attempts": 1,
        "ref": {
            "pack_id": str(uuid.uuid4()), "store_id": "s3",
            "object_key": key, "object_bytes": 4096, "checksum": "0" * 64,
            "record_count": 1,
        },
        **fields,
    }
    proc = subprocess.run([str(DRIVER)], input=json.dumps(request) + "\n",
                          capture_output=True, text=True, timeout=60)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"driver produced no output: {proc.stderr}"
    return json.loads(lines[0])


@pytest.mark.parametrize("request_cut", [False, True],
                         ids=["failed-on-its-own", "cut-by-the-cancel"])
def test_a_read_counts_as_cancelled_only_when_the_cancel_ended_it(
        fake_s3, request_cut):
    """fault/always-500 answers the trailer read 500 at once, and the
    Cancellation is cancelled as that answer comes back -- a cancel that
    comes in while the failure is reported: the store failed the read, and
    it is booked so. fault/hang holds the read for 5 s, and a deadline
    0.2 s in cuts it: that is a cancel."""
    if request_cut:
        key = f"fault/hang/{uuid.uuid4()}.dmi-pack"
        armed = {"cancel_after_ms": 200}
    else:
        key = f"fault/always-500/{uuid.uuid4()}.dmi-pack"
        armed = {"cancel_after_exchange": True}
    started = time.monotonic()
    result = _read(fake_s3, key, **armed)
    elapsed = time.monotonic() - started
    assert not result["ok"], result
    assert result["error"] == "StoreUnavailable", result
    assert result["cancelled"] is request_cut, (result, elapsed)
    if request_cut:
        assert "cancel" in result["message"], result
        assert elapsed < 2.5, (elapsed, result)
    else:
        assert "HTTP 500" in result["message"], result
