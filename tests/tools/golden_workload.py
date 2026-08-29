#!/usr/bin/env python3
"""Generate and verify a capture-storage conformance manifest.

Phase 6 has to compare golden workloads "by identity, logical bytes, checksums,
decoded tensors, and query results". This produces exactly that, as one JSON
document, from a deterministic corpus.

The Python implementation is the reference, not the production writer, so the
manifest is the contract rather than the code: a native writer is conformant if
and only if the same corpus produces the same manifest. Everything in it is
language-neutral -- byte counts, hex digests, and integers. No pickles, no
Python types, nothing that presumes the producer is Python.

    # record what today's implementation produces
    python tests/tools/golden_workload.py generate --out golden.json

    # later, or from another implementation, check nothing moved
    python tests/tools/golden_workload.py verify --manifest golden.json

`verify` re-runs the corpus and diffs field by field, so a mismatch names the
capture and field that changed rather than just failing.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi.storage.capture import (  # noqa: E402
    CaptureMetadata,
    CaptureRecord,
    CaptureQuery,
    CaptureReader,
    CatalogIndexer,
    FilesystemPackStore,
    PackIndex,
    PackWriter,
    decode_tensor,
    summarize_tensor,
)

MANIFEST_VERSION = 1
PACK_ID = UUID("018f0000-0000-7000-8000-000000000f01")

# One capture per dtype the format accepts, so conformance covers every decode
# path rather than float32 alone. Payload bytes are generated from a fixed
# formula, not a random source, so any implementation can reproduce the corpus.
_DTYPES = (
    ("bool", 1),
    ("uint8", 1),
    ("int8", 1),
    ("int16", 2),
    ("float16", 2),
    ("bfloat16", 2),
    ("int32", 4),
    ("float32", 4),
    ("int64", 8),
    ("float64", 8),
)
_ELEMENTS = 16


def _payload(dtype: str, width: int, index: int) -> bytes:
    """Deterministic bytes for one capture, defined by position not by RNG."""
    raw = bytearray()
    for element in range(_ELEMENTS):
        seed = (index * 131 + element * 17 + 7) & 0xFF
        raw.extend(bytes([(seed + byte) & 0xFF for byte in range(width)]))
    if dtype == "bool":
        # A bool payload must contain only 0 or 1, or numpy comparisons and the
        # format's own round trip stop agreeing.
        return bytes(1 if value & 1 else 0 for value in raw)
    return bytes(raw)


def _metadata(index: int, dtype: str) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"golden-{index:02d}",
        tenant_id="tenant-golden",
        experiment_id="experiment-golden",
        run_id="run-golden",
        session_id="session-golden",
        request_id=f"request-{index}",
        sequence_id=f"sequence-{index}",
        model_id="model-golden",
        model_revision="revision-1",
        adapter_revision=None if index % 2 else f"adapter-{index}",
        capture_policy_version="policy-v1",
        hook_name="resid_pre" if index % 2 else "attn_out",
        layer_number=index,
        producer_rank=100 + index,
        batch_position=900 + index,
        step_number=100_000 + index,
        token_start=200_000 + index,
        token_end=300_000 + index,
        dtype=dtype,
        shape=(_ELEMENTS,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )


def _corpus() -> list[CaptureRecord]:
    return [
        CaptureRecord(
            metadata=_metadata(index, dtype),
            payload=_payload(dtype, width, index),
        )
        for index, (dtype, width) in enumerate(_DTYPES)
    ]


class _Catalog:
    """Stands in for the ClickHouse catalog so the manifest needs no server."""

    def __init__(self, descriptors):
        self._descriptors = tuple(descriptors)

    def search(self, query: CaptureQuery):
        from dmi.storage.capture import CapturePage

        return CapturePage(
            items=self._descriptors[: query.limit], next_cursor=None, watermark="1"
        )

    def get_by_ids(self, capture_ids, *, tenant_id, watermark):
        wanted = set(capture_ids)
        return tuple(
            i
            for i in self._descriptors
            if i.capture_id in wanted and i.metadata.tenant_id == tenant_id
        )


def build_manifest() -> dict[str, Any]:
    """Run the corpus through the storage path and describe the result."""
    records = _corpus()
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        writer = PackWriter(
            pack_id=PACK_ID,
            created_at_ns=1_700_000_000_000_000_000,
            max_pack_bytes=8 * 1024 * 1024,
        )
        for record in records:
            writer.append(record)
        sealed = writer.seal()

        store = FilesystemPackStore(root, store_id="golden")
        ref = store.put(sealed, "packs/golden.dmi-pack")

        # Read descriptors back through the footer path, exactly as the
        # indexer does -- not from the writer's in-memory state.
        descriptors = PackIndex.from_store(store, ref).descriptors()
        reader = CaptureReader(
            _Catalog(descriptors), {"golden": store}, max_coalesce_gap_bytes=0
        )
        selection = reader.select(CaptureQuery(limit=len(records)))
        hydrated = reader.hydrate(selection, byte_limit=8 << 20)
        estimate = reader.estimate(selection)

        captures = []
        for item in hydrated:
            descriptor = item.descriptor
            metadata, locator = descriptor.metadata, descriptor.locator
            decoded = decode_tensor(descriptor, item.payload)
            summary = summarize_tensor(descriptor, item.payload)
            captures.append(
                {
                    "capture_id": metadata.capture_id,
                    "dtype": metadata.dtype,
                    "shape": list(metadata.shape),
                    "logical_bytes": metadata.logical_bytes,
                    # identity + checksums
                    "payload_sha256": sha256(item.payload).hexdigest(),
                    "payload_crc32": locator.checksum,
                    # decoded tensor, hashed in a byte order the format fixes
                    "decoded_sha256": sha256(decoded.tobytes()).hexdigest(),
                    "decoded_dtype": str(decoded.dtype),
                    # placement inside the pack
                    "offset": locator.offset,
                    "stored_length": locator.stored_length,
                    "decoded_length": locator.decoded_length,
                    # the summary contract
                    "summary": {
                        "version": summary.summary_version,
                        "element_count": summary.element_count,
                        "finite_count": summary.finite_count,
                        "nan_count": summary.nan_count,
                        "inf_count": summary.inf_count,
                        "zero_fraction": round(summary.zero_fraction, 12),
                        "mean": round(summary.mean, 9),
                        "minimum": round(summary.minimum, 9),
                        "maximum": round(summary.maximum, 9),
                        "abs_max": round(summary.abs_max, 9),
                        "l2_norm": round(summary.l2_norm, 9),
                    },
                }
            )

        return {
            "manifest_version": MANIFEST_VERSION,
            "pack": {
                "pack_id": ref.pack_id,
                "object_bytes": ref.object_bytes,
                "sha256": ref.checksum,
                "record_count": ref.record_count,
            },
            "hydration": {
                "capture_count": estimate.capture_count,
                "object_count": estimate.object_count,
                "request_count": estimate.request_count,
                "logical_bytes": estimate.logical_bytes,
                "stored_bytes": estimate.stored_bytes,
                "request_bytes": estimate.request_bytes,
            },
            "captures": captures,
        }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Field-by-field differences, so a mismatch names what moved."""
    left, right = _flatten(expected), _flatten(actual)
    differences = []
    for key in sorted(set(left) | set(right)):
        if key not in left:
            differences.append(f"{key}: unexpected -> {right[key]!r}")
        elif key not in right:
            differences.append(f"{key}: missing (expected {left[key]!r})")
        elif left[key] != right[key]:
            differences.append(f"{key}: expected {left[key]!r}, got {right[key]!r}")
    return differences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="write a manifest")
    generate.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify", help="check the corpus against a manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = build_manifest()
    if args.command == "generate":
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out} ({len(manifest['captures'])} captures)")
        return 0

    expected = json.loads(args.manifest.read_text())
    differences = compare(expected, manifest)
    if not differences:
        print(f"conformant: {len(manifest['captures'])} captures match {args.manifest}")
        return 0
    print(f"NON-CONFORMANT: {len(differences)} difference(s)", file=sys.stderr)
    for line in differences[:40]:
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
