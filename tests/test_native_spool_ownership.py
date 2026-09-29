"""B6 through the Python surface: one owner per spool directory.

``_dmi_native_store`` binds the spool's owner lock (``SpoolOwnerLock``), the
section 2.3 layout (``spool_rank_directory``) and who owns a directory
(``spool_owner``); the storage service and the native pack sink each open
their spool with ``owner_lock="take"`` or ``"held_by_caller"``.

The regression this pins is the in-process one: a sink and a storage
service on ONE directory in ONE process. Were each to take the lock, the
second would be refused by the first -- flock binds to an open file
description, not to the process -- so the engine holds one SpoolOwnerLock
and opens both held_by_caller. The service is constructed, not started:
start() needs a catalog, which the live suites bring
(test_native_spool_adoption_live.py, test_native_capture_chain_live.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"
STORE_BUILT = bool(sorted(BUILD.glob("_dmi_native_store*.so")))
SINK_BUILT = bool(sorted(BUILD.glob("_dmi_native_sink*.so")))

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not STORE_BUILT,
        reason="the native store module is not built; run `make -C native "
        "build/_dmi_native_store PYTHON=<venv>/bin/python`"),
]

LAYOUT = "capture_pack_reference_v1"
NFS_SUPER_MAGIC = 0x6969


def _store():
    from dmi.storage.native_capture import _load_native_store_extension

    return _load_native_store_extension()


def _config(**overrides):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    fields = dict(
        s3_endpoint="http://127.0.0.1:9", s3_bucket="bucket",
        s3_access_key="AKIA-test", s3_secret_key="secret-test",
        s3_allow_insecure_http=True, clickhouse_port=9)
    fields.update(overrides)
    return NativeCaptureStorageConfig(**fields)


def _rank_directory(base: Path, config, rank: int = 0) -> str:
    return _store().spool_rank_directory(
        str(base), config._spool_destination(), rank)


# Where a spool's packs go: the catalog's server and names, the store's.
DESTINATION = dict(
    clickhouse_host="ch", clickhouse_port=8123, database="db",
    table_prefix="prefix", s3_endpoint="http://s3:9000", s3_bucket="bucket",
    store_id="s3")


def _service(config, spool_root, **options):
    from dmi.storage.native_capture import NativeCaptureStorage

    return NativeCaptureStorage(config, spool_root=str(spool_root),
                                spool_max_bytes=1 << 30, sweep_spool=True,
                                **options)


def _sink(spool_root, **options):
    sys.path.insert(0, str(BUILD))
    try:
        import _dmi_native_sink
    finally:
        sys.path.remove(str(BUILD))
    return _dmi_native_sink.NativePackSink(
        spool_root=str(spool_root), layout=LAYOUT, max_pack_records=4,
        **options)


def _stage_one(sink) -> None:
    """One float16 capture through the sink's ring-facing submit."""
    import torch
    from dmi.storage.capture import CaptureMetadata

    tensor = torch.arange(6, dtype=torch.float16)
    metadata = CaptureMetadata(
        capture_id="own-0", tenant_id="t", experiment_id="e", run_id="r",
        session_id="s", request_id="q", sequence_id="n", model_id="m",
        model_revision="mr", adapter_revision=None,
        capture_policy_version="v", hook_name="resid_post", layer_number=0,
        producer_rank=0, step_number=0, token_start=0, token_end=1,
        batch_position=0, dtype="float16", shape=(6,),
        captured_at_ns=1_700_000_000_000_000_000,
    ).to_mapping()
    lease = sink.attach()
    sink.submit_envelope(LAYOUT, [{
        "metadata_json": json.dumps(metadata), "offset": 0, "length": 12,
        "dtype": 5, "shape": [6]}], tensor.view(torch.uint8))
    assert sink.flush_and_wait(30.0)
    sink.rethrow_if_failed()
    del lease


@pytest.mark.skipif(not SINK_BUILT, reason="the native sink module is not built")
def test_the_real_sink_and_service_share_one_spool_in_one_process(tmp_path):
    pytest.importorskip("torch")
    config = _config()
    directory = _rank_directory(tmp_path / "spool", config)
    with _store().SpoolOwnerLock(directory) as lock:
        service = _service(config, directory,
                           spool_owner_lock="held_by_caller",
                           adopt_sibling_spools=True)
        sink = _sink(directory, owner_lock="held_by_caller")
        _stage_one(sink)
        assert len(list(Path(directory).rglob("*.dmi-pack.ready"))) == 1
        snapshot = service.snapshot()
        assert snapshot["adopted_spools"] == 0
        assert snapshot["adoption_owed"] is False
        # Neither Spool took a lock of its own: the one holder is this
        # process, through `lock`.
        assert _store().spool_owner(directory)["pid"] == os.getpid()
        del sink, service
        assert lock.held
    assert _store().spool_owner(directory) is None


@pytest.mark.skipif(not SINK_BUILT, reason="the native sink module is not built")
def test_two_takes_in_one_process_refuse_each_other(tmp_path):
    """What a per-Spool lock would do to the engine's sink and service."""
    pytest.importorskip("torch")
    directory = tmp_path / "spool"
    service = _service(_config(), directory)  # take
    with pytest.raises(RuntimeError, match=f"owned by pid {os.getpid()}"):
        _sink(directory)  # take
    del service
    _sink(directory)  # the service's lock went with it


def test_a_second_process_is_refused_naming_the_holder(tmp_path):
    directory = tmp_path / "spool"
    with _store().SpoolOwnerLock(str(directory)):
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import _dmi_native_store as m\n"
            "try:\n"
            "    m.SpoolOwnerLock(sys.argv[2])\n"
            "except m.SpoolOwnedError as e:\n"
            "    print(e)\n")
        result = subprocess.run(
            [sys.executable, "-c", probe, str(BUILD), str(directory)],
            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert f"owned by pid {os.getpid()} on host {socket.gethostname()}" in (
        result.stdout), result.stdout


def test_a_forked_worker_does_not_keep_a_dead_owners_lock(tmp_path):
    """A child forked without exec -- a fork-started DataLoader or
    multiprocessing worker -- shares the lock's open file description. It
    used to keep the lock after its parent was SIGKILLed, so the dead
    parent's directory read as owned, naming the dead pid, and was never
    adopted."""
    directory = tmp_path / "spool"
    script = (
        "import os, sys, time; sys.path.insert(0, sys.argv[1]);"
        "import _dmi_native_store as m\n"
        "lock = m.SpoolOwnerLock(sys.argv[2])\n"
        "worker = os.fork()\n"
        "if worker == 0:\n"
        "    time.sleep(3600)\n"
        "    os._exit(0)\n"
        "print(worker, flush=True)\n"
        "time.sleep(3600)\n")
    owner = subprocess.Popen(
        [sys.executable, "-c", script, str(BUILD), str(directory)],
        stdout=subprocess.PIPE, text=True)
    worker = int(owner.stdout.readline())
    try:
        assert _store().spool_owner(str(directory))["pid"] == owner.pid
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=30)
        os.kill(worker, 0)  # the worker outlives its parent
        assert _store().spool_owner(str(directory)) is None
        with _store().SpoolOwnerLock(str(directory)):
            pass
    finally:
        try:
            os.kill(worker, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=30)


def test_a_removed_lock_file_leaves_a_live_directory_owned(tmp_path):
    """Something other than the owner -- an age-based cleaner such as
    systemd-tmpfiles, a person -- can remove .owner.lock from under a live
    owner. Judged by that file alone, another process then saw nobody
    holding the directory, took it, and could sweep the owner's stages in
    flight. The owner locks the directory itself as well."""
    directory = tmp_path / "spool"
    with _store().SpoolOwnerLock(str(directory)) as lock:
        (directory / ".owner.lock").unlink()
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import _dmi_native_store as m\n"
            "print(m.spool_owner(sys.argv[2]) is not None)\n"
            "try:\n"
            "    m.SpoolOwnerLock(sys.argv[2])\n"
            "    print('taken')\n"
            "except m.SpoolOwnedError:\n"
            "    print('owned')\n")
        result = subprocess.run(
            [sys.executable, "-c", probe, str(BUILD), str(directory)],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "owned"], result.stdout
        assert lock.held
    assert _store().spool_owner(str(directory)) is None
    with _store().SpoolOwnerLock(str(directory)):
        pass


def test_spool_owner_names_the_holder_while_it_holds(tmp_path):
    store = _store()
    directory = str(tmp_path / "spool")
    assert store.spool_owner(directory) is None
    lock = store.SpoolOwnerLock(directory)
    assert lock.held
    assert store.spool_owner(directory) == {
        "host": socket.gethostname(), "pid": os.getpid()}
    lock.release()
    assert not lock.held
    assert store.spool_owner(directory) is None


def test_the_owned_error_is_a_runtime_error(tmp_path):
    store = _store()
    assert issubclass(store.SpoolOwnedError, RuntimeError)
    with store.SpoolOwnerLock(str(tmp_path / "spool")):
        with pytest.raises(store.SpoolOwnedError):
            store.SpoolOwnerLock(str(tmp_path / "spool"))


def test_a_nested_spool_directory_is_refused(tmp_path):
    store = _store()
    with store.SpoolOwnerLock(str(tmp_path / "outer")):
        with pytest.raises(ValueError, match="nested"):
            store.SpoolOwnerLock(str(tmp_path / "outer" / "inner"))
        with pytest.raises(RuntimeError, match="nested"):
            _service(_config(), tmp_path / "outer" / "inner")


def test_a_spool_root_a_sink_once_owned_still_takes_rank_directories(tmp_path):
    """A sink-only run (or the explicit-record_sink rollback) takes
    spool_root itself and leaves its lock file there. The next default run
    claims a rank directory under it, which that unheld lock file must not
    refuse as nested."""
    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    root = tmp_path / "root"
    _store().SpoolOwnerLock(str(root)).release()
    assert (root / ".owner.lock").exists()
    claim = claim_spool_directory(NativeSinkConfig(spool_root=str(root)),
                                  _config())
    try:
        assert claim.held
        assert Path(claim.directory).parent.parent == root
    finally:
        claim.release()


@pytest.mark.skipif(not SINK_BUILT, reason="the native sink module is not built")
def test_a_dead_siblings_bytes_count_against_the_sinks_budget(tmp_path):
    """Each process start claims a fresh rank directory. A sink that
    charged only its own let every crash-restart add a whole
    spool_max_bytes while uploads were blocked; charge_dead_siblings (what
    the engine passes) counts what the dead incarnations beside it still
    hold, as the one directory every restart reused did before."""
    pytest.importorskip("torch")
    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    sink_config = NativeSinkConfig(spool_root=str(tmp_path / "root"))
    dead = claim_spool_directory(sink_config, _config())
    dead_directory = Path(dead.directory)
    (dead_directory / "v1").mkdir()
    (dead_directory / "v1" / "left.dmi-pack.ready").write_bytes(
        b"\0" * (1 << 20))
    dead._lock.release()  # killed: the kernel let go, the packs stay
    budget = (1 << 20) + 256  # room for the dead bytes, not for a pack

    claim = claim_spool_directory(sink_config, _config())
    try:
        sink = _sink(claim.directory, owner_lock="held_by_caller",
                     spool_max_bytes=budget, max_pack_bytes=budget,
                     charge_dead_siblings=True)
        with pytest.raises(RuntimeError, match="spool byte limit exceeded"):
            _stage_one(sink)
        del sink
        uncharged = _sink(claim.directory, owner_lock="held_by_caller",
                          spool_max_bytes=budget, max_pack_bytes=budget)
        _stage_one(uncharged)  # the whole budget, as if nothing were left
        del uncharged
    finally:
        claim.release()


def test_a_claim_warns_of_packs_left_outside_the_layout(tmp_path, caplog):
    """Before the per-process layout the engine spooled into
    <spool_root>/v1/... and its next start swept and uploaded whatever a
    crashed run left there; a sink-only run still writes there. Nothing
    adopts packs outside <spool_root>/<key>/r<rank>-<inc>/ now, so a claim
    says so -- how many, where, and how to hand them to adoption -- rather
    than leave them silently."""
    import logging

    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    root = tmp_path / "root"
    sink = NativeSinkConfig(spool_root=str(root))
    # Packs inside the layout are adoption's, and no reason to warn.
    first = claim_spool_directory(sink, _config())
    (Path(first.directory) / "v1").mkdir()
    (Path(first.directory) / "v1" / "in.dmi-pack.ready").write_bytes(b"x")
    first._lock.release()
    with caplog.at_level(logging.WARNING, logger="dmi.storage.native_capture"):
        claim = claim_spool_directory(sink, _config())
    claim.release()
    assert not [r for r in caplog.records if "outside" in r.getMessage()]

    flat = root / "v1" / "tenant=t" / "date=2026-09-01"
    flat.mkdir(parents=True)
    for name in ("a", "b"):
        (flat / f"{name}.dmi-pack.ready").write_bytes(b"pack")
    (flat / ".c.0badf00d.open").write_bytes(b"half")  # not a pack
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dmi.storage.native_capture"):
        claim = claim_spool_directory(sink, _config())
    key = Path(claim.directory).parent.name
    claim.release()
    (warning,) = [r.getMessage() for r in caplog.records
                  if "outside" in r.getMessage()]
    assert "2 ready pack(s)" in warning
    assert str(flat) in warning
    assert f"{root}/{key}/r0-00000000" in warning


def test_a_dropped_spool_claim_keeps_its_directory_owned(tmp_path):
    """Only release() lets go of a claim. An engine dropped without close()
    drops its claim, while the ring and the sink it activated may still be
    capturing into the directory: garbage collection must not unlock it for
    another process to adopt. The kernel lets go when the process exits."""
    import gc

    from dmi.storage import native_capture
    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    claim = claim_spool_directory(
        NativeSinkConfig(spool_root=str(tmp_path / "root")), _config())
    directory = claim.directory
    del claim
    gc.collect()
    try:
        assert _store().spool_owner(directory)["pid"] == os.getpid()
    finally:
        for kept in list(native_capture._HELD_SPOOL_CLAIMS):
            if kept.directory == directory:
                kept.release()
    assert _store().spool_owner(directory) is None


def test_releasing_a_claim_removes_its_directory_once_drained(tmp_path):
    """SpoolClaim.release() is how the engine lets go of its directory: a
    drained one is removed with its lock file, one still holding a pack
    stays for the next process on the node to adopt."""
    from dmi.storage import native_capture
    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    sink = NativeSinkConfig(spool_root=str(tmp_path / "root"))
    claim = claim_spool_directory(sink, _config())
    drained = Path(claim.directory)
    (drained / "v1" / "tenant=t").mkdir(parents=True)
    assert claim.release() is True
    assert not drained.exists()
    assert not claim.held
    assert claim not in native_capture._HELD_SPOOL_CLAIMS

    claim = claim_spool_directory(sink, _config())
    kept = Path(claim.directory)
    (kept / "v1").mkdir()
    (kept / "v1" / "left.dmi-pack.ready").write_bytes(b"pack")
    assert claim.release() is False
    assert (kept / "v1" / "left.dmi-pack.ready").exists()
    assert _store().spool_owner(str(kept)) is None
    assert claim not in native_capture._HELD_SPOOL_CLAIMS


@pytest.mark.parametrize("f_type, name", [
    (0x6969, "NFS"), (0x0BD00BD0, "Lustre"), (0x19830326, "BeeGFS"),
    (0xFF534D42, "CIFS"), (0xFE534D42, "SMB2"), (0x65735546, "FUSE"),
    (0x47504653, "GPFS"), (0x01021997, "9p"), (0x5346414F, "AFS"),
    (0x6B414653, "AFS"), (0x20030528, "OrangeFS")])
def test_each_shared_filesystem_is_refused_by_name(tmp_path, f_type, name):
    store = _store()
    store._set_spool_filesystem_type_for_testing(f_type)
    try:
        with pytest.raises(ValueError, match=f"is on {name} .*node-local"):
            store.SpoolOwnerLock(str(tmp_path / "shared"))
        with store.SpoolOwnerLock(str(tmp_path / "shared"),
                                  allow_shared_filesystem=True):
            pass
    finally:
        store._set_spool_filesystem_type_for_testing(None)


def test_a_shared_filesystem_is_refused_unless_allowed(tmp_path):
    store = _store()
    store._set_spool_filesystem_type_for_testing(NFS_SUPER_MAGIC)
    try:
        with pytest.raises(ValueError, match="NFS.*node-local"):
            store.SpoolOwnerLock(str(tmp_path / "nfs"))
        with pytest.raises(RuntimeError, match="node-local"):
            _service(_config(), tmp_path / "nfs")
        with store.SpoolOwnerLock(str(tmp_path / "nfs"),
                                  allow_shared_filesystem=True):
            pass
        _service(_config(), tmp_path / "nfs-allowed",
                 spool_allow_shared_filesystem=True)
    finally:
        store._set_spool_filesystem_type_for_testing(None)
    with store.SpoolOwnerLock(str(tmp_path / "local")):
        pass


def test_the_rank_directory_layout(tmp_path):
    store = _store()
    key = hashlib.sha256(
        b"db/prefix/s3\nclickhouse ch:8123\ns3 http://s3:9000/bucket"
    ).hexdigest()[:12]
    assert store.spool_catalog_key(DESTINATION) == key
    assert store.spool_rank_directory(
        "/base", DESTINATION, 3, "0a1b2c3d") == f"/base/{key}/r3-0a1b2c3d"
    fresh = {store.spool_rank_directory("/base", DESTINATION, 0)
             for _ in range(16)}
    assert len(fresh) == 16  # a fresh incarnation each time
    for path in fresh:
        assert path.startswith(f"/base/{key}/r0-")
    with pytest.raises(ValueError, match="incarnation"):
        store.spool_rank_directory("/base", DESTINATION, 0, "XYZ")
    incomplete = dict(DESTINATION)
    del incomplete["s3_bucket"]
    with pytest.raises(KeyError, match="s3_bucket"):
        store.spool_catalog_key(incomplete)
    assert _config()._spool_destination() == {
        name: getattr(_config(), name) for name in DESTINATION}


def test_the_catalog_key_names_the_servers_not_only_the_names(tmp_path):
    """The key was sha256(database/table_prefix/store_id), so two
    deployments on one node with the default names (default, dmi, s3) but
    different ClickHouse servers and buckets shared a key: whichever started
    first adopted the other's dead directories, uploading its packs to its
    own bucket and indexing them into its own catalog. Every server and
    name the packs go to is in the key now."""
    from dmi.storage.native_capture import (
        NativeSinkConfig, claim_spool_directory,
    )

    store = _store()
    key = store.spool_catalog_key(DESTINATION)
    for field, value in (("clickhouse_host", "ch2"), ("clickhouse_port", 8124),
                         ("s3_endpoint", "http://s3b:9000"),
                         ("s3_bucket", "bucket2")):
        assert store.spool_catalog_key({**DESTINATION, field: value}) != key

    staging = _config(clickhouse_host="ch-staging", s3_bucket="staging")
    production = _config(clickhouse_host="ch-production",
                         s3_bucket="production")
    sink = NativeSinkConfig(spool_root=str(tmp_path / "spool"))
    staged = claim_spool_directory(sink, staging)
    produced = claim_spool_directory(sink, production)
    try:
        assert (Path(staged.directory).parent
                != Path(produced.directory).parent)
        # A staging service cannot adopt from under production's key.
        with pytest.raises(ValueError, match="adopt_sibling_spools"):
            _service(staging, produced.directory,
                     spool_owner_lock="held_by_caller",
                     adopt_sibling_spools=True)
    finally:
        staged.release()
        produced.release()


def test_adoption_needs_a_rank_directory_under_this_catalogs_key(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="adopt_sibling_spools"):
        _service(config, tmp_path / "spool", adopt_sibling_spools=True)
    # A rank directory, but under another catalog's key: adopting its
    # siblings would index that catalog's packs into this one.
    other = _store().spool_rank_directory(
        str(tmp_path / "base"),
        {**config._spool_destination(), "table_prefix": "another_prefix"}, 0)
    with pytest.raises(ValueError, match="adopt_sibling_spools"):
        _service(config, other, adopt_sibling_spools=True)
    mine = _rank_directory(tmp_path / "base", config)
    assert _service(config, mine, adopt_sibling_spools=True).snapshot()[
        "adopted_packs"] == 0


def test_an_unknown_owner_lock_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="spool_owner_lock"):
        _service(_config(), tmp_path / "spool", spool_owner_lock="share")


def test_held_by_caller_with_nothing_held_is_refused(tmp_path):
    """The directory exists -- so this is the check that nothing holds its
    lock, not a failure to resolve a path -- first with no lock file, then
    with one nobody holds."""
    directory = tmp_path / "spool"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="held_by_caller, but nothing holds"):
        _service(_config(), directory, spool_owner_lock="held_by_caller")
    _store().SpoolOwnerLock(str(directory)).release()
    assert (directory / ".owner.lock").exists()
    with pytest.raises(RuntimeError, match="held_by_caller, but nothing holds"):
        _service(_config(), directory, spool_owner_lock="held_by_caller")


def test_held_by_caller_still_refuses_a_shared_filesystem(tmp_path):
    """The node-local check applies to a Spool opened held_by_caller too:
    the caller's lock was taken on a local disk, but what the Spool opens
    is judged again."""
    store = _store()
    directory = tmp_path / "spool"
    with store.SpoolOwnerLock(str(directory)):
        store._set_spool_filesystem_type_for_testing(NFS_SUPER_MAGIC)
        try:
            with pytest.raises(RuntimeError, match="is on NFS .*node-local"):
                _service(_config(), directory,
                         spool_owner_lock="held_by_caller")
            _service(_config(), directory, spool_owner_lock="held_by_caller",
                     spool_allow_shared_filesystem=True)
        finally:
            store._set_spool_filesystem_type_for_testing(None)


class _OtherProcessHolder:
    """Another process holding a directory's SpoolOwnerLock until closed."""

    def __init__(self, directory):
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import _dmi_native_store as m\n"
            "lock = m.SpoolOwnerLock(sys.argv[2])\n"
            "print('held', flush=True)\n"
            "sys.stdin.read()\n")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", script, str(BUILD), str(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert self.proc.stdout.readline().strip() == "held"

    @property
    def pid(self) -> int:
        return self.proc.pid

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def test_held_by_caller_beside_another_processs_lock_is_refused(tmp_path):
    """held_by_caller is for the process that holds the lock. The service
    and the sink opened that way beside ANOTHER process's lock are refused,
    naming it, rather than sweeping and uploading from its directory."""
    directory = tmp_path / "spool"
    holder = _OtherProcessHolder(directory)
    try:
        with pytest.raises(RuntimeError,
                           match=f"held_by_caller.*pid {holder.pid}"):
            _service(_config(), directory, spool_owner_lock="held_by_caller")
        if SINK_BUILT:
            pytest.importorskip("torch")
            with pytest.raises(RuntimeError,
                               match=f"held_by_caller.*pid {holder.pid}"):
                _sink(directory, owner_lock="held_by_caller")
    finally:
        holder.close()


def test_a_drained_directory_is_removed_with_its_lock(tmp_path):
    store = _store()
    directory = tmp_path / "spool"
    lock = store.SpoolOwnerLock(str(directory))
    (directory / "v1" / "tenant=t").mkdir(parents=True)
    assert lock.release_and_remove_if_empty()
    assert not directory.exists()
    lock = store.SpoolOwnerLock(str(directory))
    (directory / "kept.quarantined").write_bytes(b"x")
    assert not lock.release_and_remove_if_empty()
    assert (directory / "kept.quarantined").exists()
    assert not lock.held
