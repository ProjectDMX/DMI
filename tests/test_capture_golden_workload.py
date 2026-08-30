"""The capture-storage conformance manifest.

Phase 6 compares golden workloads "by identity, logical bytes, checksums,
decoded tensors, and query results". `tests/tools/golden_workload.py` produces
exactly that as one JSON document, and the checked-in manifest beside this file
is what today's implementation produces.

Because the Python implementation is the reference rather than the production
writer, this manifest is the contract: a native writer is conformant when the
same corpus yields the same document. These tests keep the reference honest --
if any of it drifts, the diff names the capture and field that moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.tools.golden_workload import (
    MANIFEST_VERSION,
    build_manifest,
    compare,
)


pytestmark = pytest.mark.cpu


GOLDEN = Path(__file__).parent / "data" / "capture_golden_manifest.json"


def test_the_implementation_still_matches_the_recorded_manifest():
    expected = json.loads(GOLDEN.read_text())

    differences = compare(expected, build_manifest())

    assert differences == [], (
        "capture storage no longer matches the conformance manifest.\n"
        "If the change is intended, regenerate with:\n"
        f"  python tests/tools/golden_workload.py generate --out {GOLDEN}\n"
        + "\n".join(f"  {line}" for line in differences[:20])
    )


def test_the_manifest_is_reproducible():
    # Two runs must agree, or the manifest cannot serve as a contract at all.
    assert compare(build_manifest(), build_manifest()) == []


def test_the_manifest_covers_every_supported_dtype():
    from dmi.storage.capture.model import _DTYPE_BYTES

    manifest = build_manifest()

    covered = {capture["dtype"] for capture in manifest["captures"]}
    assert covered == set(_DTYPE_BYTES), "a dtype would ship unverified"


def test_the_manifest_records_what_phase_six_has_to_compare():
    manifest = build_manifest()

    assert manifest["manifest_version"] == MANIFEST_VERSION
    # identity, logical bytes, checksums, decoded tensors, query results
    assert manifest["pack"]["sha256"]
    assert manifest["hydration"]["logical_bytes"] > 0
    for capture in manifest["captures"]:
        assert capture["payload_sha256"] and capture["payload_crc32"]
        assert capture["decoded_sha256"]
        assert capture["summary"]["version"] == 1


def test_a_changed_payload_is_caught_by_the_comparison():
    manifest = build_manifest()
    tampered = json.loads(json.dumps(manifest))
    tampered["captures"][0]["decoded_sha256"] = "0" * 64

    differences = compare(manifest, tampered)

    # The diff must name the field, not just report inequality.
    assert len(differences) == 1
    assert "captures[0].decoded_sha256" in differences[0]
