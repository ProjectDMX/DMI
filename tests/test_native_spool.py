"""A3a: the native spool stages packs the Python spool can drain, and vice versa.

The on-disk contract (.open → .ready, ready-file naming, sha256 validation,
quarantine, capacity) is shared byte-for-byte, so the cross-implementation
tests below are the actual conformance gate — not same-implementation
round trips alone.

Build: make -C native build/conformance_spool
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi.storage.capture import (  # noqa: E402
    CaptureMetadata,
    CaptureRecord,
    DurablePackSpool,
    PackReader,
    PackWriter,
    SpoolFullError,
)
from tests.tools.golden_workload import PACK_ID, _corpus  # noqa: E402

DRIVER = REPO_ROOT / "native" / "build" / "conformance_spool"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_spool is not built; run "
        "`make -C native build/conformance_spool`",
    ),
]

MAX_BYTES = 256 * 1024 * 1024
# The key must end in "<pack_id>.dmi-pack" — both spools enforce this — and
# its tenant= segment must name the pack's own tenant (the catalog's
# foreign-tenant binding refuses anything else at read time).
OBJECT_KEY = (
    "v1/tenant=tenant-golden/date=2026-09-05/session=session-golden/rank=0/"
    "018f0000-0000-7000-8000-000000000f01.dmi-pack"
)


def _golden_pack() -> tuple[bytes, str, int]:
    writer = PackWriter(
        pack_id=PACK_ID, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024,
    )
    for record in _corpus():
        writer.append(record)
    sealed = writer.seal()
    return sealed.data, sealed.checksum, len(_corpus())


def _call(**fields) -> dict:
    proc = subprocess.run(
        [str(DRIVER)],
        input=json.dumps(fields) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"driver produced no output: {proc.stderr}"
    return json.loads(lines[0])


def test_native_spool_stages_what_python_drains(tmp_path):
    data, checksum, count = _golden_pack()
    response = _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )
    assert response["ok"], response
    staged = response["staged"]
    assert staged["object_key"] == OBJECT_KEY
    assert staged["object_bytes"] == len(data)

    # The Python spool recovers the native-written entry and reads it back
    # through the pack path — the rollback direction.
    spool = DurablePackSpool(tmp_path, max_bytes=MAX_BYTES)
    recovered = spool.recover()
    assert len(recovered) == 1
    entry = recovered[0]
    assert entry.checksum == checksum
    assert entry.object_key == OBJECT_KEY
    with entry.open() as handle:
        descriptors = PackReader.from_bytes(handle.read()).descriptors(
            store_id="spool", object_key=OBJECT_KEY
        )
    assert len(descriptors) == count
    assert descriptors[0].metadata.capture_id == "golden-00"


def test_python_spool_stages_what_native_recovers(tmp_path):
    from dmi.storage.capture import FilesystemPackStore

    data, checksum, count = _golden_pack()
    writer = PackWriter(
        pack_id=PACK_ID, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024,
    )
    for record in _corpus():
        writer.append(record)
    sealed = writer.seal()

    spool = DurablePackSpool(tmp_path, max_bytes=MAX_BYTES)
    staged = spool.stage(sealed, OBJECT_KEY)

    response = _call(op="recover", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert response["ok"], response
    assert len(response["staged"]) == 1
    entry = response["staged"][0]
    assert entry["pack_id"] == str(PACK_ID)
    assert entry["checksum"] == checksum
    assert entry["object_key"] == OBJECT_KEY
    assert entry["object_bytes"] == len(data)
    assert Path(entry["path"]).read_bytes() == data


def test_stage_is_idempotent(tmp_path):
    data, checksum, count = _golden_pack()
    fields = dict(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )
    first = _call(**fields)
    second = _call(**fields)
    assert first["ok"] and second["ok"]
    assert first["staged"] == second["staged"]
    assert second["snapshot"] == {"entries": 1, "bytes": len(data)}


def test_capacity_is_enforced(tmp_path):
    data, checksum, count = _golden_pack()
    response = _call(
        op="stage", root=str(tmp_path), max_bytes=len(data) - 1,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )
    assert not response["ok"] and "limit" in response["status"]
    # The Python side agrees: same bytes, same bound.
    spool = DurablePackSpool(tmp_path, max_bytes=len(data) - 1)
    writer = PackWriter(
        pack_id=PACK_ID, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024,
    )
    for record in _corpus():
        writer.append(record)
    with pytest.raises(SpoolFullError):
        spool.stage(writer.seal(), OBJECT_KEY)


def test_stage_rejects_a_key_escaping_through_a_symlinked_directory(tmp_path):
    # Mirror of test_capture_spool.py's symlink test: a key whose parent is a
    # symlink out of the root resolves outside it, so the write must be
    # refused. Accepting it writes the pack outside the spool and reports it
    # durably staged, while recursive_directory_iterator (which does not
    # follow directory symlinks) can never find it again to upload -- the
    # pack is lost and its bytes stay charged against the spool budget.
    data, checksum, count = _golden_pack()
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "spool"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    response = _call(
        op="stage", root=str(root), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum,
        object_key=f"link/{PACK_ID}.dmi-pack",
        data_b64=base64.b64encode(data).decode(),
    )
    assert not response["ok"], response
    assert response["status"] == "invalid argument", response
    assert "escapes the spool root" in response["what"], response
    assert list(outside.rglob("*")) == []
    assert response["snapshot"] == {"entries": 0, "bytes": 0}

    # The Python spool refuses the same key for the same reason.
    spool = DurablePackSpool(root, max_bytes=MAX_BYTES)
    writer = PackWriter(
        pack_id=PACK_ID, created_at_ns=1_700_000_000_000_000_000,
        max_pack_bytes=8 * 1024 * 1024,
    )
    for record in _corpus():
        writer.append(record)
    with pytest.raises(ValueError, match="escapes the spool root"):
        spool.stage(writer.seal(), f"link/{PACK_ID}.dmi-pack")
    assert list(outside.rglob("*")) == []


def test_corrupt_ready_file_is_quarantined(tmp_path):
    data, checksum, count = _golden_pack()
    assert _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )["ok"]
    readies = list(tmp_path.rglob("*.dmi-pack.ready"))
    assert len(readies) == 1
    with readies[0].open("r+b") as handle:
        handle.seek(100)
        handle.write(b"corrupt!")
    response = _call(op="recover", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert response["ok"] and response["staged"] == []
    assert response["snapshot"] == {"entries": 0, "bytes": 0}
    quarantined = list(tmp_path.rglob("*.quarantined"))
    assert len(quarantined) == 1


def test_stale_open_files_are_removed_on_recovery(tmp_path):
    data, checksum, count = _golden_pack()
    assert _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )["ok"]
    stale = tmp_path / "v1" / ".deadbeef.1234.open"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")
    response = _call(op="recover", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert response["ok"] and len(response["staged"]) == 1
    assert not stale.exists()


def test_remove_unaccounts(tmp_path):
    data, checksum, count = _golden_pack()
    staged = _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )["staged"]
    assert _call(op="remove", root=str(tmp_path), max_bytes=MAX_BYTES,
                 staged=staged)["ok"]
    snap = _call(op="snapshot", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert snap["snapshot"]["entries"] == 0
    assert not list(tmp_path.rglob("*.dmi-pack.ready"))


# --- integer bounds ------------------------------------------------------------
#
# Every integer field on this protocol is 64-bit, and a literal that does not
# fit has no value it could carry. FindInt handed each site -1, which
# `max_bytes` read as "not given, use the 1 TiB default" and the StagedPack
# counters cast to 18446744073709551615 -- so a defaulted or wrapped value came
# back reported as success. Refuse instead, in the ok:false shape the driver
# already answers a bad argument with.


@pytest.mark.parametrize(
    "field, value",
    [
        # 2**64 + 1 wraps to 1: a plausible-looking timestamp is worse than a
        # refusal.
        ("created_at_ns", 2**64 + 1),
        ("record_count", 2**64 + 1),
        # A digit run far longer than any 64-bit value.
        ("created_at_ns", int("9" * 40)),
        # Below INT64_MIN by one; the negative branch has the wider limit.
        ("created_at_ns", -(2**63) - 1),
        # max_bytes' -1 became the default, so a capacity limit the caller
        # asked for was never applied.
        ("max_bytes", 2**64 + 1),
        ("max_bytes", -(2**63) - 1),
    ],
)
def test_stage_refuses_an_out_of_range_integer(tmp_path, field, value):
    data, checksum, count = _golden_pack()
    fields = dict(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )
    fields[field] = value
    response = _call(**fields)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    assert field in response["what"], response
    assert not list(tmp_path.rglob("*.dmi-pack.ready"))


def test_remove_refuses_an_out_of_range_integer_in_the_staged_object(tmp_path):
    """The nested StagedPack is decoded by the same helper, so it refuses too."""
    data, checksum, count = _golden_pack()
    staged = _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=1_700_000_000_000_000_000,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )["staged"]
    staged["object_bytes"] = 2**64 + 1
    response = _call(op="remove", root=str(tmp_path), max_bytes=MAX_BYTES,
                     staged=staged)
    assert not response["ok"], response
    assert "out of range" in response["what"], response
    # Nothing was removed on a refusal: the entry is still accounted for.
    snap = _call(op="snapshot", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert snap["snapshot"]["entries"] == 1, snap


@pytest.mark.parametrize("value", [2**64 - 1, 2**63, 2**63 - 1, 1, 0])
def test_stage_keeps_a_created_at_ns_of_any_64_bit_width(tmp_path, value):
    """The union range must survive the parse bit for bit, not int64's half.

    `created_at_ns` rides the ready-file name, so staging a value above
    INT64_MAX and recovering it is what proves the two's-complement bit
    pattern is recovered exactly rather than clamped or refused.
    """
    data, checksum, count = _golden_pack()
    response = _call(
        op="stage", root=str(tmp_path), max_bytes=MAX_BYTES,
        pack_id=str(PACK_ID), created_at_ns=value,
        record_count=count, checksum=checksum, object_key=OBJECT_KEY,
        data_b64=base64.b64encode(data).decode(),
    )
    assert response["ok"], response
    assert response["staged"]["created_at_ns"] == value, response
    recovered = _call(op="recover", root=str(tmp_path), max_bytes=MAX_BYTES)
    assert recovered["ok"], recovered
    assert recovered["staged"][0]["created_at_ns"] == value, recovered
