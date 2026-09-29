"""B6 live: a SIGKILLed capture process's spool is adopted by its successor.

A capture process -- here a child running what the engine composes: one
SpoolOwnerLock on its own rank directory of the section 2.3 layout, the
storage service and the REAL native pack sink both opened held_by_caller --
captures, gets part of it indexed, stages the rest, and is SIGKILLed with
packs still in its spool and the publisher lease still live. A new
incarnation on the same node and catalog, in a directory of its own, then
starts: it waits out the dead lease, sweeps its own directory, takes the
dead directory's owner lock (the kernel dropped it with the process),
sweeps its stale .open file, uploads and indexes every ready pack, and
removes the directory. Every capture of the dead process reads back from
the catalog with its bytes. A live sibling -- another process's spool,
lock held -- is left alone.

When the object store is down at start, the dead directory stays as it
was (its packs are durable there), flush() does not report drained, and
the loop adopts it once the store is back.

Needs ClickHouse on 127.0.0.1:8123/9000 and the native sink and store
modules: make -C native build/_dmi_native_sink build/_dmi_native_store
PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, fake_s3,
)

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
SINK_BUILT = bool(sorted(BUILD.glob("_dmi_native_sink*.so")))
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))

pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not (SINK_BUILT and STORE_BUILT),
        reason="the native sink and store modules are not built; run "
        "`make -C native build/_dmi_native_sink build/_dmi_native_store "
        "PYTHON=<venv>/bin/python`",
    ),
]

CLICKHOUSE_HOST = os.environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1")
CLICKHOUSE_HTTP_PORT = int(os.environ.get("DMI_CLICKHOUSE_HTTP_PORT", "8123"))
DATABASE = os.environ.get("DMI_CLICKHOUSE_DATABASE", "default")
LAYOUT = "capture_pack_reference_v1"
# Short, so the successor's wait for the dead process's lease stays short.
LEASE = dict(lease_ttl_s=3.0, publish_timeout_s=1.0)
INDEXED_BY_THE_DEAD = range(0, 4)   # flushed to the catalog before the kill
STAGED_BY_THE_DEAD = range(4, 10)   # only in its spool when it dies
RECORDS_PER_PACK = 2


def _storage_config(endpoint: str, prefix: str, **overrides):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    fields = dict(
        s3_endpoint=endpoint, s3_bucket=BUCKET, s3_region=REGION,
        s3_access_key=ACCESS, s3_secret_key=SECRET,
        s3_allow_insecure_http=True, clickhouse_host=CLICKHOUSE_HOST,
        clickhouse_port=CLICKHOUSE_HTTP_PORT, database=DATABASE,
        table_prefix=prefix, poll_interval_s=0.05, **LEASE)
    fields.update(overrides)
    return NativeCaptureStorageConfig(**fields)


def _store():
    from dmi.storage.native_capture import _load_native_store_extension

    return _load_native_store_extension()


def _claim(base: Path, config):
    """What the engine does first: a fresh rank directory, locked."""
    store = _store()
    directory = store.spool_rank_directory(
        str(base), config.database, config.table_prefix, config.store_id, 0)
    return store.SpoolOwnerLock(directory)


def _service(config, directory: str, **options):
    from dmi.storage.native_capture import NativeCaptureStorage

    return NativeCaptureStorage(
        config, spool_root=directory, spool_max_bytes=1 << 30,
        sweep_spool=True, spool_owner_lock="held_by_caller",
        adopt_sibling_spools=True, **options)


def _envelope(indexes):
    from tests.test_native_capture_chain_live import _Envelope

    import torch

    envelope = _Envelope()
    for index in indexes:
        envelope.add(index, torch.arange(6, dtype=torch.float16) + index)
    return envelope


def _dead_capture_process(base: str, endpoint: str, prefix: str) -> None:
    """The child: capture, index some, stage the rest, then wait to die."""
    import torch  # noqa: F401 -- the sink extension links against it

    sys.path.insert(0, str(BUILD))
    import _dmi_native_sink

    # The loop must not upload the second batch before the kill: with this
    # poll interval only the explicit flush below runs a cycle. The lease
    # thread renews meanwhile, so the process dies holding the lease.
    config = _storage_config(endpoint, prefix, poll_interval_s=3600.0)
    lock = _claim(Path(base), config)
    service = _service(config, lock.directory)
    service.start()
    sink = _dmi_native_sink.NativePackSink(
        spool_root=lock.directory, layout=LAYOUT,
        max_pack_records=RECORDS_PER_PACK, max_linger_ns=600 * 10**9,
        owner_lock="held_by_caller")
    _lease = sink.attach()
    first = _envelope(INDEXED_BY_THE_DEAD)
    sink.submit_envelope(LAYOUT, first.rows, first.payload())
    assert sink.flush_and_wait(60.0)
    service.flush(60.0)
    second = _envelope(STAGED_BY_THE_DEAD)
    sink.submit_envelope(LAYOUT, second.rows, second.payload())
    assert sink.flush_and_wait(60.0)
    sink.rethrow_if_failed()
    print(json.dumps({"directory": lock.directory}), flush=True)
    time.sleep(3600)


def _spawn_dead_process(base: Path, endpoint: str, prefix: str):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + str(REPO),
               CUDA_VISIBLE_DEVICES="")
    child = subprocess.Popen(
        [sys.executable, "-m", "tests.test_native_spool_adoption_live",
         str(base), endpoint, prefix],
        cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    line = child.stdout.readline()
    if not line:
        child.wait(timeout=30)
        raise AssertionError(f"the capture process failed: {child.stderr.read()}")
    return child, Path(json.loads(line)["directory"])


def _sigkill(child) -> None:
    os.kill(child.pid, signal.SIGKILL)
    child.communicate(timeout=30)
    assert child.returncode == -signal.SIGKILL


def _stale_open_file(directory: Path) -> Path:
    """A stage the dead process had in flight: its temp file, left behind."""
    (ready,) = sorted(directory.rglob("*.dmi-pack.ready"))[:1]
    stale = ready.parent / ".018f0000-0000-7000-8000-00000000dead.0badf00d.open"
    stale.write_bytes(b"half a pack")
    return stale


def _claim_staging(parent: Path, name: str, *, age_s: float) -> Path:
    """The staging copy a claim killed before its rename leaves behind."""
    staging = parent / f".{name}.0badf00d.creating"
    staging.mkdir()
    (staging / ".owner.lock").write_text("host 1\n")
    then = time.time() - age_s
    os.utime(staging, (then, then))
    return staging


def _read_all(config) -> dict:
    from dmi.storage.native_capture import NativeCaptureReader

    reader = NativeCaptureReader(config)
    selection = reader.select(tenant_id="t")
    return {capture.descriptor["capture_id"]: capture.payload
            for capture in reader.read(selection, byte_limit=1 << 24)}


def _expected() -> dict:
    expected = {}
    for indexes in (INDEXED_BY_THE_DEAD, STAGED_BY_THE_DEAD):
        envelope = _envelope(indexes)
        for capture_id, tensor in envelope.expected.items():
            expected[capture_id] = tensor.contiguous().view(-1).numpy().tobytes()
    return expected


def _catalog():
    from tests.test_native_capture_chain_live import _catalog as chain_catalog

    return chain_catalog()


def test_a_sigkilled_process_spool_is_adopted_by_its_successor(
        fake_s3, tmp_path):
    base = tmp_path / "spool"
    with _catalog() as prefix:
        config = _storage_config(fake_s3, prefix)
        child, dead = _spawn_dead_process(base, fake_s3, prefix)
        try:
            staged = sorted(dead.rglob("*.dmi-pack.ready"))
            assert len(staged) == len(STAGED_BY_THE_DEAD) // RECORDS_PER_PACK
        finally:
            _sigkill(child)
        stale = _stale_open_file(dead)
        assert _store().spool_owner(str(dead)) is None  # died with it
        # A claim killed before it renamed its directory into place leaves
        # the staging copy (lock file inside). An old one is cleared; a
        # fresh one may be a claim in progress, and is left alone.
        old_claim = _claim_staging(dead.parent, "r0-0badf00d", age_s=3600)
        fresh_claim = _claim_staging(dead.parent, "r0-00c0ffee", age_s=0)

        # Another process's live spool, bound for the same catalog.
        live = _claim(base, config)
        (Path(live.directory) / "marker").write_text("live")

        lock = _claim(base, config)
        service = _service(config, lock.directory)
        started = time.monotonic()
        service.start()  # waits out the dead lease, then adopts
        try:
            snapshot = service.snapshot()
            assert time.monotonic() - started < 30
            assert snapshot["adopted_spools"] == 1, snapshot
            assert snapshot["adopted_packs"] == len(staged), snapshot
            assert snapshot["adoption_owed"] is False, snapshot
            assert not stale.exists()
            assert not dead.exists()
            assert not old_claim.exists()
            assert fresh_claim.exists()
            service.flush(60.0)
            assert _read_all(config) == _expected()
        finally:
            service.stop()
        assert (Path(live.directory) / "marker").read_text() == "live"
        assert live.held
        assert lock.release_and_remove_if_empty()
        live.release()


def test_a_dead_spool_waits_in_place_while_the_object_store_is_down(
        fake_s3, tmp_path):
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        child, dead = _spawn_dead_process(base, fake_s3, prefix)
        _sigkill(child)
        staged = sorted(dead.rglob("*.dmi-pack.ready"))
        assert staged

        switch = _Switch.to_url(fake_s3)
        switch.cut()
        config = _storage_config(switch.url, prefix)
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is True, snapshot
            assert snapshot["adopted_spools"] == 0, snapshot
            # Nothing left the dead spool: its packs are durable there.
            assert sorted(dead.rglob("*.dmi-pack.ready")) == staged
            with pytest.raises(TimeoutError):
                service.flush(2.0)

            switch.restore()
            service.flush(60.0)
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is False, snapshot
            assert snapshot["adopted_spools"] == 1, snapshot
            assert not dead.exists()
            assert _read_all(config) == _expected()
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


if __name__ == "__main__":
    _dead_capture_process(*sys.argv[1:4])
