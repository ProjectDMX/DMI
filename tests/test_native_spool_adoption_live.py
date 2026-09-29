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
the loop adopts it once the store is back. A sibling whose owner is still
alive at start and dies later is adopted by a later pass. stop() cuts an
adoption as it cuts the service's own work -- its uploads, and its listing
of a dead backlog -- and leaves the dead directory with every pack in it.

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
import threading
import time
from pathlib import Path

import pytest

# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, STATE, fake_s3,
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
STAGED_BY_THE_LIVE = range(20, 24)  # a live sibling's, never adopted
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
        str(base), config._spool_destination(), 0)
    return store.SpoolOwnerLock(directory)


def _service(config, directory: str, **options):
    from dmi.storage.native_capture import NativeCaptureStorage

    return NativeCaptureStorage(
        config, spool_root=directory, spool_max_bytes=1 << 30,
        sweep_spool=True, spool_owner_lock="held_by_caller",
        adopt_sibling_spools=True, **options)


def _envelope(indexes, width: int = 6):
    from tests.test_native_capture_chain_live import _Envelope

    import torch

    envelope = _Envelope()
    for index in indexes:
        envelope.add(index, torch.arange(width, dtype=torch.float16) + index)
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


def _wait_for(predicate, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for the service")
        time.sleep(0.05)


def _adopted(service, spools: int):
    """Whether `spools` dead siblings have been adopted and nothing more is
    to adopt, as of the last cycle's end."""
    def done():
        snapshot = service.snapshot()
        return (snapshot["adopted_spools"] == spools
                and not snapshot["adoption_owed"])
    return done


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

        # A live sibling bound for the same catalog: ready packs its sink
        # staged, and a stage it has in flight. Its lock is held by this
        # process, so an adopter that took it for dead would get past the
        # held_by_caller check and sweep it; only its liveness saves it.
        live = _claim(base, config)
        live_directory = Path(live.directory)
        _stage_into(live.directory, STAGED_BY_THE_LIVE)
        live_packs = sorted(live_directory.rglob("*.dmi-pack.ready"))
        assert len(live_packs) == len(STAGED_BY_THE_LIVE) // RECORDS_PER_PACK
        live_open = live_packs[0].parent / (
            ".018f0000-0000-7000-8000-00000000beef.0badf00d.open")
        live_open.write_bytes(b"half a pack")

        lock = _claim(base, config)
        service = _service(config, lock.directory)
        started = time.monotonic()
        service.start()  # waits out the dead lease; the loop adopts
        try:
            _wait_for(_adopted(service, 1))
            snapshot = service.snapshot()
            assert time.monotonic() - started < 30
            assert snapshot["adopted_spools"] == 1, snapshot
            assert snapshot["adopted_packs"] == len(staged), snapshot
            assert snapshot["adoption_owed"] is False, snapshot
            assert not stale.exists()
            assert not dead.exists()
            assert not old_claim.exists()
            assert fresh_claim.exists()
            assert snapshot["live_siblings"] == 1, snapshot
            service.flush(60.0)
            # Only the dead process's captures reach the catalog.
            assert _read_all(config) == _expected()
        finally:
            service.stop()
        # The live sibling is as it was: nothing swept, nothing uploaded.
        assert sorted(live_directory.rglob("*.dmi-pack.ready")) == live_packs
        assert live_open.read_bytes() == b"half a pack"
        live_ids = {path.name.split(".")[0] for path in live_packs}
        with STATE.lock:
            uploaded = list(STATE.objects)
        assert not [key for key in uploaded
                    if any(pack_id in key for pack_id in live_ids)]
        assert live.held
        assert lock.release_and_remove_if_empty()
        live.release()


def test_a_dead_spool_waits_in_place_while_the_object_store_is_down(
        fake_s3, tmp_path):
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        # Both processes reach the store through the switch: the endpoint is
        # part of the catalog key, so a successor spelling it differently
        # would not see the dead directory as its sibling.
        switch = _Switch.to_url(fake_s3)
        child, dead = _spawn_dead_process(base, switch.url, prefix)
        _sigkill(child)
        staged = sorted(dead.rglob("*.dmi-pack.ready"))
        assert staged

        # Let the dead process's lease lapse, so start() has none to wait
        # out and its time is its own.
        time.sleep(LEASE["lease_ttl_s"] + 1.0)
        switch.cut()
        config = _storage_config(switch.url, prefix)
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        started = time.monotonic()
        service.start()
        try:
            # start() leaves the dead backlog to the loop: it does not sit
            # through every dead pack's retry chain against a store that
            # refuses them (about 8 s a round of four packs).
            assert time.monotonic() - started < 3.0
            # Nor does flush(), which is about this process's own records:
            # it returns within its deadline, drained or not.
            flushed = time.monotonic()
            try:
                service.flush(0.5)
            except TimeoutError:
                pass  # a loop cycle still in an upload round holds the cycle
            assert time.monotonic() - flushed < 2.0
            _wait_for(lambda: service.snapshot()["upload_failures"] > 0)
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is True, snapshot
            assert snapshot["adopted_spools"] == 0, snapshot
            # Nothing left the dead spool: its packs are durable there.
            assert sorted(dead.rglob("*.dmi-pack.ready")) == staged

            switch.restore()
            _wait_for(_adopted(service, 1), 90.0)
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is False, snapshot
            assert not dead.exists()
            service.flush(60.0)
            assert _read_all(config) == _expected()
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


def _stage_into(directory: str, indexes, width: int = 6) -> None:
    """Stage records into a directory this process holds, as its own sink
    would: the REAL native pack sink, held_by_caller."""
    import torch  # noqa: F401 -- the sink extension links against it

    sys.path.insert(0, str(BUILD))
    try:
        import _dmi_native_sink
    finally:
        sys.path.remove(str(BUILD))
    sink = _dmi_native_sink.NativePackSink(
        spool_root=directory, layout=LAYOUT,
        max_pack_records=RECORDS_PER_PACK, max_linger_ns=600 * 10**9,
        owner_lock="held_by_caller")
    lease = sink.attach()
    envelope = _envelope(indexes, width)
    sink.submit_envelope(LAYOUT, envelope.rows, envelope.payload())
    assert sink.flush_and_wait(60.0)
    sink.rethrow_if_failed()
    del lease, sink


def test_no_dead_spool_is_uploaded_while_an_adopted_pack_is_owed(
        fake_s3, tmp_path):
    """Owner decision 7, inside adoption. Two dead siblings; the catalog
    goes away just after the first one's packs reach the object store, so
    they cannot be indexed and are owed in memory. The second sibling's
    packs must then stay in its spool, where a crash cannot lose them --
    not be uploaded into a list only this process remembers. Once the
    catalog is back, both are indexed."""
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        # The servers are part of the catalog key, so the siblings are
        # claimed through the same switch URLs as the service reaches.
        clickhouse = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
        store = _Switch.to_url(fake_s3)
        config = _storage_config(store.url, prefix,
                                 clickhouse_host="127.0.0.1",
                                 clickhouse_port=clickhouse.port)
        halves = (range(0, 4), range(4, 10))
        siblings = []
        for indexes in halves:
            sibling = _claim(base, config)
            siblings.append(Path(sibling.directory))
            _stage_into(sibling.directory, indexes)
            sibling.release()  # its owner is gone
        first, second = sorted(siblings)  # adopted in this order
        second_packs = sorted(second.rglob("*.dmi-pack.ready"))
        assert second_packs

        cut = threading.Event()

        def _cut_the_catalog_at_the_first_upload(request: bytes) -> float:
            if request.startswith(b"PUT ") and not cut.is_set():
                cut.set()
                clickhouse.cut()
            return 0.0

        store.delay_requests(_cut_the_catalog_at_the_first_upload)
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            _wait_for(lambda: service.snapshot()["pending_index"] > 0)
            snapshot = service.snapshot()
            assert cut.is_set()
            assert not sorted(first.rglob("*.dmi-pack.ready")), snapshot
            # Nothing of the second left its spool while the first's packs
            # were owed.
            assert sorted(second.rglob("*.dmi-pack.ready")) == second_packs
            assert snapshot["adoption_owed"] is True, snapshot
            with pytest.raises(TimeoutError):
                service.flush(2.0)
            assert sorted(second.rglob("*.dmi-pack.ready")) == second_packs

            clickhouse.restore()
            service.flush(60.0)
            _wait_for(_adopted(service, 2), 60.0)
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is False, snapshot
            assert not first.exists() and not second.exists()
            service.flush(60.0)
            expected = {}
            for indexes in halves:
                for capture_id, tensor in _envelope(indexes).expected.items():
                    expected[capture_id] = (
                        tensor.contiguous().view(-1).numpy().tobytes())
            assert _read_all(config) == expected
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


def test_a_dead_spool_this_service_can_never_upload_is_left_and_reported(
        fake_s3, tmp_path):
    """A pack this service can never upload -- here larger than its
    uploader_max_in_flight_bytes, which a crashed run with a larger bound
    left behind -- blocks its dead directory for good: retrying it only
    re-hashes it and keeps the loop backing off. It is reported once and
    left in place for a process that can (or a person), the siblings after
    it are adopted, and nothing is owed or retried."""
    base = tmp_path / "spool"
    with _catalog() as prefix:
        # A 4096-byte bound: the wide captures' packs exceed it, the narrow
        # ones' fit.
        config = _storage_config(fake_s3, prefix,
                                 uploader_max_in_flight_bytes=4096)
        directories = []
        for indexes, width in ((range(0, 4), 4096), (range(4, 10), 6)):
            sibling = _claim(base, config)
            directories.append(Path(sibling.directory))
            _stage_into(sibling.directory, indexes, width)
            sibling.release()  # its owner is gone
        blocked, adoptable = directories
        blocked_packs = sorted(blocked.rglob("*.dmi-pack.ready"))
        assert all(path.stat().st_size > 4096 for path in blocked_packs)

        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            _wait_for(_adopted(service, 1))
            snapshot = service.snapshot()
            assert snapshot["blocked_siblings"] == [str(blocked)], snapshot
            assert "in-flight byte limit" in snapshot["last_error"], snapshot
            assert sorted(blocked.rglob("*.dmi-pack.ready")) == blocked_packs
            # Let go of, and marked in its lock file -- for a person, and for
            # the sinks on the node, which charge a dead directory against
            # their budgets only while an adoption can drain it.
            assert _store().spool_owner(str(blocked)) is None
            record = (blocked / ".owner.lock").read_text()
            assert ("\nblocked: it holds a pack this service can never "
                    "upload") in record, record
            assert "in-flight byte limit" in record, record
            assert not adoptable.exists()
            # Not retried: no more failed uploads, and no backoff -- the
            # loop keeps its poll interval.
            failures, cycles = (snapshot["upload_failures"],
                                snapshot["cycles"])
            time.sleep(1.0)
            snapshot = service.snapshot()
            assert snapshot["upload_failures"] == failures, snapshot
            assert snapshot["cycles"] >= cycles + 5, snapshot
            assert snapshot["adoption_owed"] is False, snapshot
            started = time.monotonic()
            service.flush(10.0)
            assert time.monotonic() - started < 2.0
            expected = {
                capture_id: tensor.contiguous().view(-1).numpy().tobytes()
                for capture_id, tensor in _envelope(
                    range(4, 10)).expected.items()}
            assert _read_all(config) == expected
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


def test_packs_left_outside_the_layout_are_adopted_once_moved_as_told(
        fake_s3, tmp_path, caplog):
    """A sink-only run (or an engine from before the layout) leaves its
    packs in spool_root itself, where nothing adopts them. The claim's
    warning names a directory of the layout to move them into; moved there
    with their paths kept, they are adopted like a dead process's."""
    import logging
    import shutil

    import torch  # noqa: F401 -- the sink extension links against it

    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    base = tmp_path / "spool"
    with _catalog() as prefix:
        config = _storage_config(fake_s3, prefix)
        sys.path.insert(0, str(BUILD))
        try:
            import _dmi_native_sink
        finally:
            sys.path.remove(str(BUILD))
        sink = _dmi_native_sink.NativePackSink(
            spool_root=str(base), layout=LAYOUT,
            max_pack_records=RECORDS_PER_PACK, max_linger_ns=600 * 10**9)
        lease = sink.attach()
        envelope = _envelope(STAGED_BY_THE_DEAD)
        sink.submit_envelope(LAYOUT, envelope.rows, envelope.payload())
        assert sink.flush_and_wait(60.0)
        del lease, sink
        flat = sorted((base / "v1").rglob("*.dmi-pack.ready"))
        assert len(flat) == len(STAGED_BY_THE_DEAD) // RECORDS_PER_PACK

        with caplog.at_level(logging.WARNING,
                             logger="dmi.storage.native_capture"):
            claim = claim_spool_directory(
                NativeSinkConfig(spool_root=str(base)), config)
        (warning,) = [record.getMessage() for record in caplog.records
                      if "outside" in record.getMessage()]
        key = Path(claim.directory).parent.name
        target = base / key / "r0-00000000"
        assert f"{len(flat)} ready pack(s)" in warning
        assert str(target) in warning

        target.mkdir()
        shutil.move(str(base / "v1"), str(target / "v1"))
        service = _service(config, claim.directory)
        service.start()
        try:
            _wait_for(_adopted(service, 1))
            assert not target.exists()
            service.flush(60.0)
            expected = {
                capture_id: tensor.contiguous().view(-1).numpy().tobytes()
                for capture_id, tensor in envelope.expected.items()}
            assert _read_all(config) == expected
        finally:
            service.stop()
        claim.release()


def test_a_sibling_whose_owner_dies_after_start_is_adopted_by_a_recheck(
        fake_s3, tmp_path):
    """A sibling still owned when the service starts -- a predecessor still
    inside its close(), another rank that dies later -- is left alone then.
    The service looks again every adoption_recheck_interval_ns while it
    has such a sibling, and adopts it once its owner is gone, rather than
    leaving its packs for the next restart on the node."""
    base = tmp_path / "spool"
    with _catalog() as prefix:
        config = _storage_config(fake_s3, prefix)
        sibling = _claim(base, config)
        sibling_directory = Path(sibling.directory)
        _stage_into(sibling.directory, STAGED_BY_THE_DEAD)
        staged = sorted(sibling_directory.rglob("*.dmi-pack.ready"))
        assert len(staged) == len(STAGED_BY_THE_DEAD) // RECORDS_PER_PACK

        lock = _claim(base, config)
        native = config._native_dict()
        native.update(
            spool_root=lock.directory, spool_max_bytes=1 << 30,
            holder="recheck-test", poll_interval_ns=50_000_000,
            sweep_spool_on_start=True, spool_owner_lock="held_by_caller",
            adopt_sibling_spools=True,
            adoption_recheck_interval_ns=200_000_000,
            **config._lease_native())
        service = _store().StorageService(native)
        service.start()
        try:
            _wait_for(lambda: not service.snapshot()["adoption_owed"])
            snapshot = service.snapshot()
            assert snapshot["live_siblings"] == 1, snapshot
            assert snapshot["adopted_spools"] == 0, snapshot
            assert snapshot["adoption_owed"] is False, snapshot
            assert sorted(sibling_directory.rglob(
                "*.dmi-pack.ready")) == staged

            sibling.release()  # its owner is gone
            deadline = time.monotonic() + 30
            while ((service.snapshot()["adopted_spools"] == 0
                    or service.snapshot()["live_siblings"] != 0)
                   and time.monotonic() < deadline):
                time.sleep(0.05)
            snapshot = service.snapshot()
            assert snapshot["adopted_spools"] == 1, snapshot
            assert snapshot["adopted_packs"] == len(staged), snapshot
            assert snapshot["live_siblings"] == 0, snapshot
            assert not sibling_directory.exists()
            assert service.flush(60.0)
            expected = {
                capture_id: tensor.contiguous().view(-1).numpy().tobytes()
                for capture_id, tensor in _envelope(
                    STAGED_BY_THE_DEAD).expected.items()}
            assert _read_all(config) == expected
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


def test_flush_adopts_nothing_while_the_loop_is_idle(fake_s3, tmp_path):
    """flush() covers this process's records, and its cycles never adopt:
    a dead backlog is the loop's. With the object store down, a flush that
    adopted sat through a dead pack's retry chain, past its deadline by a
    round (minutes against a stalled store). Here the loop is idle -- its
    first round has failed and its poll interval is an hour -- so the flush
    holds the cycle itself and would be seen adopting: uploading (and
    failing) dead packs and overrunning its deadline."""
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        store = _Switch.to_url(fake_s3)
        config = _storage_config(store.url, prefix, poll_interval_s=3600.0)
        sibling = _claim(base, config)
        dead = Path(sibling.directory)
        _stage_into(sibling.directory, STAGED_BY_THE_DEAD)
        sibling.release()  # its owner is gone
        staged = sorted(dead.rglob("*.dmi-pack.ready"))
        assert staged
        store.cut()
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            # The loop's first cycle, which start() kicks, begins adopting
            # and fails its first round; then it waits out its interval.
            _wait_for(lambda: service.snapshot()["upload_failures"] > 0
                      and service.snapshot()["cycles"] >= 1, 60.0)
            before = service.snapshot()
            assert before["adoption_owed"] is True, before
            flushed = time.monotonic()
            service.flush(0.5)  # nothing of its own: drained
            assert time.monotonic() - flushed < 2.0
            after = service.snapshot()
            assert after["upload_failures"] == before["upload_failures"], after
            assert after["adopted_packs"] == 0, after
            assert after["adoption_owed"] is True, after
            assert sorted(dead.rglob("*.dmi-pack.ready")) == staged
        finally:
            service.stop()
        lock.release_and_remove_if_empty()


def test_stop_cuts_an_adoption_stalled_on_its_uploads(fake_s3, tmp_path):
    """An adoption uploads through the service's own upload client and
    Cancellation, so stop() cuts it as it cuts the service's own uploads.
    Here every PUT is held open and never answered: before, the adoption's
    uploader went round a pack's retries on a client stop() did not cancel,
    and stop() sat through read timeouts. Now stop() returns at once, every
    pack of the dead spool is still in it (none was uploaded, none lost),
    and the directory is left, unlocked, for the next incarnation -- which
    adopts it."""
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        store = _Switch.to_url(fake_s3)
        config = _storage_config(store.url, prefix, s3_read_timeout_s=30,
                                 reconcile_on_start=False)
        sibling = _claim(base, config)
        dead = Path(sibling.directory)
        _stage_into(sibling.directory, STAGED_BY_THE_DEAD)
        sibling.release()  # its owner is gone
        staged = sorted(dead.rglob("*.dmi-pack.ready"))
        assert len(staged) == len(STAGED_BY_THE_DEAD) // RECORDS_PER_PACK
        store.stall_requests(lambda request: request.startswith(b"PUT "))
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            _wait_for(lambda: store.stalled, 30.0)  # an adopted PUT in flight
            stopping = time.monotonic()
            service.stop()
            assert time.monotonic() - stopping < 5.0
            snapshot = service.snapshot()
            assert snapshot["cancelled_uploads"] >= 1, snapshot
            assert snapshot["upload_failures"] == 0, snapshot
            assert snapshot["adopted_packs"] == 0, snapshot
            assert "adopting dead spool" not in snapshot["last_error"], snapshot
            assert sorted(dead.rglob("*.dmi-pack.ready")) == staged
            assert _store().spool_owner(str(dead)) is None
        finally:
            service.stop()
        lock.release_and_remove_if_empty()

        store.restore()
        successor_lock = _claim(base, config)
        successor = _service(config, successor_lock.directory)
        successor.start()
        try:
            _wait_for(_adopted(successor, 1), 60.0)
            assert not dead.exists()
            successor.flush(60.0)
            expected = {
                capture_id: tensor.contiguous().view(-1).numpy().tobytes()
                for capture_id, tensor in _envelope(
                    STAGED_BY_THE_DEAD).expected.items()}
            assert _read_all(config) == expected
        finally:
            successor.stop()
            store.close()
        successor_lock.release_and_remove_if_empty()


def _sparse_backlog(directory: Path, count: int, size: int) -> list[Path]:
    """`count` ready packs of `size` zero bytes, sparse and named for their
    checksum: validating them hashes every byte, which takes seconds, while
    they take no disk."""
    import hashlib
    import uuid

    digest = hashlib.sha256()
    zeros = bytes(1 << 20)
    for _ in range(size // len(zeros)):
        digest.update(zeros)
    packs = directory / "v1"
    packs.mkdir(parents=True, exist_ok=True)
    backlog = []
    for _ in range(count):
        ready = packs / f"{uuid.uuid4()}.1.1.{digest.hexdigest()}.dmi-pack.ready"
        with open(ready, "wb") as sparse:
            sparse.truncate(size)
        backlog.append(ready)
    return sorted(backlog)


def test_stop_cuts_an_adoption_listing_a_dead_backlog(fake_s3, tmp_path):
    """An adoption lists a dead spool once, validating every pack, which
    over a backlog takes seconds: 32 sparse packs of 256 MiB, here. That
    listing stops between packs once stop() cancels the uploads, as the
    service's own listing does, so stop() does not wait for the rest of
    it; the directory keeps every pack, and nothing of it was uploaded."""
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        store = _Switch.to_url(fake_s3)
        config = _storage_config(store.url, prefix, reconcile_on_start=False)
        sibling = _claim(base, config)
        dead = Path(sibling.directory)
        sibling.release()  # its owner is gone
        backlog = _sparse_backlog(dead, 32, 256 << 20)
        # Were the listing not cut, no pack may reach the fake store's
        # memory: every PUT is held back, and stop() cuts those.
        store.stall_requests(lambda request: request.startswith(b"PUT "))
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        service.start()
        try:
            # The loop's first cycle, which start() kicks, begins the
            # listing; the whole of it takes several seconds.
            _wait_for(lambda: _store().spool_owner(str(dead)) is not None,
                      10.0)
            time.sleep(0.3)
            stopping = time.monotonic()
            service.stop()
            elapsed = time.monotonic() - stopping
            assert elapsed < 2.5, elapsed
            snapshot = service.snapshot()
            assert snapshot["adopted_packs"] == 0, snapshot
            assert store.stalled == [], store.stalled
            assert sorted(dead.rglob("*.dmi-pack.ready")) == backlog
            assert _store().spool_owner(str(dead)) is None
        finally:
            service.stop()
            store.close()
        lock.release_and_remove_if_empty()


def test_a_latched_service_lets_go_of_the_sibling_it_was_adopting(
        fake_s3, tmp_path):
    """A service whose catalog another publisher keeps for 2 x TTL latches,
    and its loop stops for good -- but the dead sibling it was half-way
    through adopting stayed locked by this process, with nobody working on
    it, until the engine's close(), possibly hours of capture later. Every
    other process on the node read it as live meanwhile. The latched loop
    lets go of it, at once."""
    from dmi.storage.native_capture import NativeCaptureStorage
    from tests.test_native_capture_storage_live import _Switch

    base = tmp_path / "spool"
    with _catalog() as prefix:
        # The servers are part of the catalog key: the sibling is claimed
        # through the same switch URLs as the service reaches.
        clickhouse = _Switch(CLICKHOUSE_HOST, CLICKHOUSE_HTTP_PORT)
        store = _Switch.to_url(fake_s3)
        config = _storage_config(store.url, prefix,
                                 clickhouse_host="127.0.0.1",
                                 clickhouse_port=clickhouse.port,
                                 reconcile_on_start=False)
        sibling = _claim(base, config)
        dead = Path(sibling.directory)
        _stage_into(sibling.directory, STAGED_BY_THE_DEAD)
        sibling.release()  # its owner is gone
        staged = sorted(dead.rglob("*.dmi-pack.ready"))
        store.cut()  # so the adoption stays half done
        lock = _claim(base, config)
        service = _service(config, lock.directory)
        rival = NativeCaptureStorage(
            _storage_config(fake_s3, prefix, holder="rival-publisher",
                            start_lease_wait_s=10.0,
                            reconcile_on_start=False),
            spool_root=str(tmp_path / "rival"), spool_max_bytes=1 << 30,
            sweep_spool=True)
        service.start()
        try:
            _wait_for(lambda: service.snapshot()["upload_failures"] > 0, 60.0)
            owner = _store().spool_owner(str(dead))
            assert owner is not None and owner["pid"] == os.getpid(), owner

            clickhouse.cut()  # the service can renew no more
            rival.start()  # waits out the lease the cut service cannot renew
            clickhouse.restore()
            _wait_for(lambda: service.snapshot()["failed"], 30.0)
            _wait_for(lambda: _store().spool_owner(str(dead)) is None, 5.0)
            snapshot = service.snapshot()
            assert snapshot["adoption_owed"] is False, snapshot
            assert snapshot["adopted_packs"] == 0, snapshot
            assert sorted(dead.rglob("*.dmi-pack.ready")) == staged
        finally:
            service.stop()
            rival.stop()
            clickhouse.close()
            store.close()
        lock.release_and_remove_if_empty()


if __name__ == "__main__":
    _dead_capture_process(*sys.argv[1:4])
