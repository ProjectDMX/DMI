from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    DurablePackSink,
    DurablePackSpool,
    FilesystemPackStore,
    FlushReason,
    PackWriter,
    ReadyPack,
    SpoolFullError,
    SpoolUploader,
    HostCapturePipeline,
    PipelineConfig,
    AdmissionResult,
)


pytestmark = pytest.mark.cpu


def _sealed(pack_id: str, capture_id: str):
    metadata = CaptureMetadata(
        capture_id=capture_id,
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    writer = PackWriter(
        pack_id=UUID(pack_id),
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=1024 * 1024,
    )
    writer.append(CaptureRecord(metadata=metadata, payload=b"\x00" * 8))
    return writer.seal(), metadata


class _AmbiguousStore(FilesystemPackStore):
    def __init__(self, root: Path):
        super().__init__(root, store_id="remote")
        self.fail_once = True

    def put(self, pack, object_key):
        ref = super().put(pack, object_key)
        if self.fail_once:
            self.fail_once = False
            raise OSError("connection lost after commit")
        return ref


def test_spool_stage_is_atomic_bounded_and_restart_recoverable(tmp_path: Path):
    first, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    second, _ = _sealed(
        "018f0000-0000-7000-8000-000000000002", "capture-b"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(first.data))
    sink = DurablePackSink(spool)

    staged = sink.persist(ReadyPack(first, metadata, FlushReason.SHUTDOWN))

    assert staged.path.suffix == ".ready"
    assert staged.created_at_ns == first.created_at_ns
    assert staged.path.exists()
    assert not tuple((tmp_path / "spool").rglob("*.open"))
    assert spool.snapshot().bytes == len(first.data)
    with pytest.raises(SpoolFullError, match="spool byte limit"):
        sink.persist(ReadyPack(second, metadata, FlushReason.SHUTDOWN))

    recovered = DurablePackSpool(
        tmp_path / "spool", max_bytes=len(first.data)
    ).recover()
    assert recovered == (staged,)
    assert recovered[0].created_at_ns == first.created_at_ns


def test_spool_rejects_an_object_key_it_cannot_recover_exactly(tmp_path: Path):
    pack, _ = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)

    with pytest.raises(ValueError, match="pack ID"):
        spool.stage(pack, "v1/custom-name.dmi-pack")


def test_spool_upload_retry_resolves_an_ambiguous_remote_commit(tmp_path: Path):
    pack, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = DurablePackSink(spool).persist(
        ReadyPack(pack, metadata, FlushReason.SHUTDOWN)
    )
    remote = _AmbiguousStore(tmp_path / "objects")
    uploader = SpoolUploader(spool, remote)

    with pytest.raises(OSError, match="after commit"):
        uploader.upload(staged)
    assert staged.path.exists()

    ref = uploader.upload(staged)

    assert not staged.path.exists()
    assert remote.stat(ref).checksum == pack.checksum
    assert spool.snapshot().bytes == 0


def test_a_staged_pack_is_always_re_hashed_on_its_way_out_of_the_spool(
    tmp_path: Path,
):
    """The disk round trip is verified before upload, and nothing may skip it.

    A staged pack's bytes come back from the filesystem, so ``StagedPack``
    must never claim the ``verified_bytes`` fast path however confidently the
    spool wrote them: re-hashing what actually came off disk is the check that
    catches bit-rot, a truncated write, or a tampered spool between staging
    and upload. That check is the whole reason spool mode is durable.
    """

    from dmi.storage.capture import PackIntegrityError, VerifiedPackSource

    pack, metadata = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = DurablePackSink(spool).persist(
        ReadyPack(pack, metadata, FlushReason.SHUTDOWN)
    )
    assert not isinstance(staged, VerifiedPackSource)

    store = FilesystemPackStore(tmp_path / "objects", store_id="local")
    staged.path.write_bytes(b"\xff" * staged.object_bytes)

    with pytest.raises(PackIntegrityError, match="checksum does not match"):
        SpoolUploader(spool, store).upload(staged)

    # The corrupt pack is still spooled: a failed upload must not be mistaken
    # for a delivered one.
    assert staged.path.exists()


def test_spool_recovery_quarantines_corrupt_ready_pack(tmp_path: Path):
    corrupt_pack, corrupt_metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    healthy_pack, healthy_metadata = _sealed(
        "018f0000-0000-7000-8000-000000000002", "capture-b"
    )
    root = tmp_path / "spool"
    spool = DurablePackSpool(
        root, max_bytes=(len(corrupt_pack.data) + len(healthy_pack.data)) * 2
    )
    sink = DurablePackSink(spool)
    staged = sink.persist(ReadyPack(corrupt_pack, corrupt_metadata, FlushReason.SHUTDOWN))
    healthy = sink.persist(ReadyPack(healthy_pack, healthy_metadata, FlushReason.SHUTDOWN))
    with staged.path.open("r+b") as handle:
        handle.seek(64)
        handle.write(b"\xff")

    recovered = DurablePackSpool(
        root, max_bytes=(len(corrupt_pack.data) + len(healthy_pack.data)) * 2
    ).recover()

    # One corrupt file must not block the healthy pack behind it: it is
    # sidelined with its bytes intact, and only the healthy entry returns.
    assert [entry.pack_id for entry in recovered] == [healthy.pack_id]
    assert not staged.path.exists()
    quarantined = staged.path.with_suffix(".quarantined")
    assert quarantined.exists()
    assert quarantined.stat().st_size == staged.object_bytes


def test_spool_recovery_removes_incomplete_open_files(tmp_path: Path):
    root = tmp_path / "spool"
    root.mkdir()
    incomplete = root / ".interrupted.open"
    incomplete.write_bytes(b"partial")

    spool = DurablePackSpool(root, max_bytes=1024)
    assert spool.snapshot().bytes == len(b"partial")

    recovered = spool.recover()

    assert recovered == ()
    assert not incomplete.exists()
    assert spool.snapshot().bytes == 0


def test_durable_pipeline_commits_to_spool_without_remote_storage(tmp_path: Path):
    pack, metadata = _sealed(
        "018f0000-0000-7000-8000-000000000001", "capture-a"
    )
    record = CaptureRecord(metadata=metadata, payload=b"\x00" * 8)
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    pipeline = HostCapturePipeline(
        PipelineConfig(
            max_queue_records=2,
            max_queue_bytes=16,
            max_pack_bytes=1024 * 1024,
            max_pack_records=2,
            max_linger_ns=1_000_000_000,
        ),
        DurablePackSink(spool),
        pack_id_factory=lambda: UUID(
            "018f0000-0000-7000-8000-000000000003"
        ),
    )

    pipeline.start()
    assert pipeline.submit(record) is AdmissionResult.ACCEPTED
    snapshot = pipeline.close(timeout=2)

    assert snapshot.persisted_records == 1
    assert len(spool.recover()) == 1


# --- staging edge cases ------------------------------------------------------------


class _LyingSpoolPack:
    """Metadata copied from a real pack, bytes that contradict its checksum."""

    def __init__(self, sealed):
        self.pack_id = sealed.pack_id
        self.created_at_ns = sealed.created_at_ns
        self.record_count = sealed.record_count
        self.checksum = sealed.checksum
        self._data = b"\x00" * len(sealed.data)

    @property
    def object_bytes(self) -> int:
        return len(self._data)

    def open(self):
        from io import BytesIO

        return BytesIO(self._data)


def test_transport_classifier_returns_false_without_botocore(monkeypatch):
    import sys

    from dmi.storage.capture.spool import _is_transient_transport_error

    monkeypatch.setitem(sys.modules, "botocore.exceptions", None)

    assert _is_transient_transport_error(RuntimeError("unreachable")) is False


def test_a_new_spool_root_is_fsynced_at_construction(tmp_path: Path, monkeypatch):
    import os

    synced: list[Path] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr("dmi.storage.capture.filesystem.os.fsync", record)
    root = (tmp_path / "spool").resolve()

    DurablePackSpool(root, max_bytes=1024)

    # Same root-creation gap as FilesystemPackStore: the spool's whole
    # durability promise hangs off a root mkdir() alone never made durable.
    assert synced == [root, root.parent]


def test_spool_rejects_a_non_positive_byte_budget(tmp_path: Path):
    with pytest.raises(ValueError, match="max_bytes"):
        DurablePackSpool(tmp_path / "spool", max_bytes=0)


def test_stage_is_idempotent_for_identical_content(tmp_path: Path):
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    key = f"packs/{pack.pack_id}.dmi-pack"

    first = spool.stage(pack, key)
    second = spool.stage(pack, key)

    assert first == second
    assert spool.snapshot().entries == 1


def test_stage_rejects_a_conflicting_intent_for_the_same_pack_id(tmp_path: Path):
    from dataclasses import replace

    from dmi.storage.capture import PackConflictError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 4)
    key = f"packs/{pack.pack_id}.dmi-pack"
    spool.stage(pack, key)

    with pytest.raises(PackConflictError, match="different pack intent"):
        spool.stage(replace(pack, created_at_ns=pack.created_at_ns + 1), key)


def test_stage_rejects_a_ready_file_holding_different_bytes(tmp_path: Path):
    from dmi.storage.capture import PackConflictError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    key = f"packs/{pack.pack_id}.dmi-pack"
    staged = spool.stage(pack, key)
    staged.path.write_bytes(b"\xff" * staged.object_bytes)

    with pytest.raises(PackConflictError, match="different content"):
        spool.stage(pack, key)


def test_stage_survives_losing_the_link_race_to_an_identical_writer(
    tmp_path: Path, monkeypatch
):
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)

    def concurrent_link(src, dst, *args, **kwargs):
        # The other writer wins the race with identical content.
        Path(dst).write_bytes(pack.data)
        raise FileExistsError(dst)

    monkeypatch.setattr("dmi.storage.capture.spool.os.link", concurrent_link)

    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")

    assert staged.object_bytes == len(pack.data)
    assert staged.path.exists()


def test_stage_cleans_up_its_temp_file_when_the_source_lies(tmp_path: Path):
    from dmi.storage.capture import PackIntegrityError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)

    with pytest.raises(PackIntegrityError, match="checksum"):
        spool.stage(_LyingSpoolPack(pack), f"packs/{pack.pack_id}.dmi-pack")

    assert not tuple(spool.root.rglob("*.open"))
    assert spool.snapshot().entries == 0


def test_stage_rejects_a_key_escaping_through_a_symlinked_directory(tmp_path: Path):
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    outside = tmp_path / "outside"
    outside.mkdir()
    (spool.root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the spool root"):
        spool.stage(pack, f"link/{pack.pack_id}.dmi-pack")


# --- recovery edge cases -------------------------------------------------------------


def test_recover_quarantines_a_ready_file_with_an_invalid_name(tmp_path: Path):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024)
    bogus = spool.root / "bogus.dmi-pack.ready"
    bogus.write_bytes(b"junk")

    assert spool.recover() == ()
    assert not bogus.exists()
    assert bogus.with_suffix(".quarantined").exists()


def test_recover_skips_entries_removed_by_a_concurrent_uploader(
    tmp_path: Path, monkeypatch
):
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")

    def gone(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(DurablePackSpool, "_checksum", staticmethod(gone))

    # Completed work elsewhere, not corruption: skipped, never quarantined.
    assert spool.recover() == ()
    assert staged.path.exists()


def test_quarantine_tolerates_a_file_already_gone(tmp_path: Path):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024)

    spool._quarantine(spool.root / "ghost.dmi-pack.ready")

    assert tuple(spool.root.iterdir()) == ()


def test_recover_does_not_erase_accounting_for_a_pack_staged_mid_pass(
    tmp_path: Path, monkeypatch
):
    """recover() validates without the lock, so a stage() may land mid-pass.

    The pass's aggregate is then stale: assigning it unconditionally erases
    the concurrent stage's accounting, and the next over-budget stage()
    overfills the spool instead of raising SpoolFullError.
    """
    import threading

    first, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    second, _ = _sealed("018f0000-0000-7000-8000-000000000002", "capture-b")
    third, _ = _sealed("018f0000-0000-7000-8000-000000000003", "capture-c")
    budget = len(first.data) + len(second.data) + len(third.data) - 1
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=budget)
    spool.stage(first, f"packs/{first.pack_id}.dmi-pack")

    in_checksum = threading.Event()
    release = threading.Event()
    real_checksum = FilesystemPackStore._checksum

    def gated(path):
        in_checksum.set()
        assert release.wait(timeout=10)
        return real_checksum(path)

    monkeypatch.setattr(DurablePackSpool, "_checksum", staticmethod(gated))

    recovering = threading.Thread(target=spool.recover)
    recovering.start()
    try:
        assert in_checksum.wait(timeout=10)
        # The reviewer's repro: recovery is blocked mid-checksum, a second
        # pack is staged, then recovery finishes and assigns its aggregate.
        spool.stage(second, f"packs/{second.pack_id}.dmi-pack")
    finally:
        release.set()
        recovering.join(timeout=10)
    assert not recovering.is_alive()

    snapshot = spool.snapshot()
    assert snapshot.entries == 2
    assert snapshot.bytes == len(first.data) + len(second.data)
    with pytest.raises(SpoolFullError, match="spool byte limit"):
        spool.stage(third, f"packs/{third.pack_id}.dmi-pack")


def test_recover_falls_back_to_a_locked_pass_under_constant_churn(
    tmp_path: Path, monkeypatch
):
    """When every optimistic pass is invalidated, the last one holds the lock.

    A corrupt ready file planted during the final optimistic pass also proves
    the locked pass still quarantines -- without trying to retake the lock it
    already holds.
    """
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")
    bogus = spool.root / "bogus.dmi-pack.ready"
    real_checksum = FilesystemPackStore._checksum
    calls = {"count": 0}

    def churning(path):
        calls["count"] += 1
        # Simulate an accounting mutation landing during every unlocked
        # pass. Once the final pass holds the lock this acquire fails, so
        # that pass cannot be invalidated.
        if spool._lock.acquire(blocking=False):
            try:
                spool._mutation_generation += 1
            finally:
                spool._lock.release()
        if calls["count"] == 8:
            bogus.write_bytes(b"junk")
        return real_checksum(path)

    monkeypatch.setattr(DurablePackSpool, "_checksum", staticmethod(churning))

    recovered = spool.recover()

    assert calls["count"] == 9, "8 optimistic passes, then the locked one"
    assert [entry.pack_id for entry in recovered] == [staged.pack_id]
    assert not bogus.exists()
    assert bogus.with_suffix(".quarantined").exists()
    assert spool.snapshot().bytes == len(pack.data)


# --- removal edge cases ----------------------------------------------------------------


def test_remove_validates_the_staged_path(tmp_path: Path):
    from dataclasses import replace

    from dmi.storage.capture import PackIntegrityError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")

    link = tmp_path / "link.dmi-pack.ready"
    link.symlink_to(staged.path)
    with pytest.raises(PackIntegrityError, match="regular spool file"):
        spool.remove(replace(staged, path=link))

    outside = tmp_path / staged.path.name
    outside.write_bytes(pack.data)
    with pytest.raises(ValueError, match="outside the spool root"):
        spool.remove(replace(staged, path=outside))


def test_remove_tolerates_an_entry_already_uploaded_elsewhere(tmp_path: Path):
    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")
    staged.path.unlink()

    spool.remove(staged)  # a second removal is a no-op, not an error


def test_remove_rejects_an_entry_whose_identity_changed(tmp_path: Path):
    from dataclasses import replace

    from dmi.storage.capture import PackIntegrityError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")

    with pytest.raises(PackIntegrityError, match="identity changed"):
        spool.remove(replace(staged, created_at_ns=staged.created_at_ns + 1))


def test_remove_rejects_an_entry_whose_size_changed(tmp_path: Path, monkeypatch):
    from dmi.storage.capture import PackIntegrityError

    pack, _ = _sealed("018f0000-0000-7000-8000-000000000001", "capture-a")
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=len(pack.data) * 2)
    staged = spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")
    monkeypatch.setattr(DurablePackSpool, "_entry", lambda self, path: staged)
    with staged.path.open("ab") as handle:
        handle.write(b"x")

    with pytest.raises(PackIntegrityError, match="size changed"):
        spool.remove(staged)


def test_entry_rejects_non_regular_files_and_unparseable_names(tmp_path: Path):
    from dmi.storage.capture import PackIntegrityError

    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024)

    with pytest.raises(PackIntegrityError, match="regular spool file"):
        spool._entry(spool.root)

    stray = spool.root / "stray.dmi-pack.ready"
    stray.write_bytes(b"x")
    with pytest.raises(PackIntegrityError, match="invalid ready-pack name"):
        spool._entry(stray)


# --- serial uploader -----------------------------------------------------------------


def test_spool_uploader_drains_pending_entries_with_a_limit(tmp_path: Path):
    spool = DurablePackSpool(tmp_path / "spool", max_bytes=1024 * 1024)
    for index in (1, 2, 3):
        pack, _ = _sealed(
            f"018f0000-0000-7000-8000-{index:012d}", f"capture-{index}"
        )
        spool.stage(pack, f"packs/{pack.pack_id}.dmi-pack")
    uploader = SpoolUploader(
        spool, FilesystemPackStore(tmp_path / "remote", store_id="remote")
    )

    with pytest.raises(ValueError, match="limit"):
        uploader.upload_pending(limit=0)

    first = uploader.upload_pending(limit=2)
    rest = uploader.upload_pending()

    assert len(first) == 2
    assert len(rest) == 1
    assert spool.snapshot().entries == 0
