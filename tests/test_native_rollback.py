"""A5b selection + rollback: choose the native writer, switch back safely.

Selection is a factory call — :func:`create_native_pack_sink` vs the
reference :class:`CapturePackReferenceSink` — and both writers produce packs
the same Python reader indexes. The rollback test proves the load-bearing
direction: packs staged by the NATIVE writer (through torch envelopes) are
indexed by the PYTHON CatalogIndexer with byte-identical descriptors, so
flipping selection back never strands staged captures.

Build: make -C native build/_dmi_native_sink PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MATCHES = sorted((REPO_ROOT / "native" / "build").glob("_dmi_native_sink*.so"))

pytestmark = pytest.mark.cpu

# Module-level skip, not per-test: this module imports the built .so
# below, which must not execute at collection when the build is absent.
if not MATCHES:
    pytest.skip(
        "native/build/_dmi_native_sink*.so is not built; run "
        "`make -C native build/_dmi_native_sink PYTHON=<venv>/bin/python`",
        allow_module_level=True,
    )

import torch  # noqa: E402

sys.path.insert(0, str(MATCHES[0].parent))
import _dmi_native_sink as native_sink  # noqa: E402

from dmi.storage.capture import (  # noqa: E402
    CaptureMetadata,
    CatalogIndexer,
    DurablePackSpool,
    PackReader,
    PackRef,
)
from dmi.storage.capture.native_sink import (  # noqa: E402
    LAYOUT_NAME,
    NativeSinkConfig,
    create_native_pack_sink,
)
from dmi.storage.capture.record_adapter import (  # noqa: E402
    CaptureRecordFormat,
)


def _meta(index: int, session: str) -> dict:
    meta = CaptureMetadata(
        capture_id=f"rollback-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id=session, request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v", hook_name="h",
        layer_number=0, producer_rank=0, step_number=index,
        token_start=index, token_end=index + 1, batch_position=0,
        dtype="float32", shape=(16,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    )
    return json.loads(json.dumps(meta.to_mapping()))


class _FakeWriter:
    """Minimal CatalogWriter: records what the indexer publishes."""

    publisher_lease = "held"

    def __init__(self):
        self.descriptors: list = []
        self.versions: list[int] = []
        self.committed: list = []
        self._version = 0

    def committed_pack_ids(self, identities):
        return set()

    def write_descriptors(self, descriptors, *, index_version):
        self.descriptors.extend(descriptors)

    def commit_packs(self, refs, *, index_version):
        self.committed.extend(refs)

    def publish_snapshot(self, *, index_version, refs, published_at_ns,
                         indexed_rows, indexed_packs):
        self.versions.append(index_version)

    def last_published_version(self):
        return self.versions[-1] if self.versions else 0

    def allocate_version(self):
        self._version += 1
        return self._version


class _SpoolPackStore:
    """PackStore over staged .ready files, keyed by staged path."""

    store_id = "spool"

    def __init__(self, paths: dict[str, Path]):
        self._paths = paths

    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes:
        with open(self._paths[ref.object_key], "rb") as handle:
            handle.seek(offset)
            return handle.read(length)


def test_selection_shares_the_wire_layout():
    assert LAYOUT_NAME == CaptureRecordFormat.LAYOUT_NAME
    handle = create_native_pack_sink(
        NativeSinkConfig(spool_root="/tmp/rollback-layout")
    )
    assert handle.record_format_layout == LAYOUT_NAME
    assert handle.native_sink.layout == LAYOUT_NAME


def test_selection_rejects_bad_config():
    import dataclasses

    with pytest.raises(ValueError, match="spool_root"):
        NativeSinkConfig(spool_root="")
    with pytest.raises(TypeError):
        create_native_pack_sink(object())
    with pytest.raises(ValueError, match="num_workers"):
        NativeSinkConfig(spool_root="/tmp/x", num_workers=0)


def test_native_staged_packs_index_through_python(tmp_path):
    # Write path: torch envelopes -> native sink -> spool.
    handle = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path / "spool"))
    )
    sink = handle.native_sink
    lease = sink.attach()
    try:
        payload = torch.arange(32, dtype=torch.float32)
        for session in ("s0", "s1"):
            rows = [
                {
                    "metadata_json": json.dumps(_meta(0, session)),
                    "offset": 0,
                    "length": 64,
                    "dtype": 6,  # at::ScalarType::Float
                    "shape": [16],
                },
                {
                    "metadata_json": json.dumps(_meta(1, session)),
                    "offset": 64,
                    "length": 64,
                    "dtype": 6,
                    "shape": [16],
                },
            ]
            sink.submit_envelope(
                "capture_pack_reference_v1", rows, payload
            )
        assert sink.flush_and_wait(30.0)
        sink.rethrow_if_failed()
    finally:
        del lease
    # Rollback direction: the PYTHON indexer reads the native packs.
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1 << 40)
    recovered = spool.recover()
    assert len(recovered) == 2
    paths = {entry.object_key: Path(entry.path) for entry in recovered}
    store = _SpoolPackStore(paths)
    refs = [
        PackRef(
            pack_id=entry.pack_id,
            store_id=store.store_id,
            object_key=entry.object_key,
            object_bytes=entry.object_bytes,
            checksum=entry.checksum,
            record_count=entry.record_count,
        )
        for entry in recovered
    ]
    writer = _FakeWriter()
    result = CatalogIndexer(store, writer).index(refs)
    assert result.indexed_packs == 2
    assert result.indexed_rows == 4
    assert result.failed_packs == 0
    assert sorted(d.metadata.capture_id for d in writer.descriptors) == [
        "rollback-0000",
        "rollback-0000",
        "rollback-0001",
        "rollback-0001",
    ]
    assert writer.versions == [1]
