from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import errno
import math
import os
from pathlib import Path
import random
import re
import tempfile
import threading
import time
from typing import BinaryIO, Callable, Mapping

from .filesystem import (
    FilesystemPackStore,
    copy_pack_source,
    fsync_new_root,
    fsync_path_to_root,
    validate_object_key,
    validate_pack_source,
)
from .model import PackConflictError, PackIntegrityError, PackRef, PackSource, PackStore
from .pipeline import ReadyPack, object_key_for


_PERMANENT_ERRNOS = frozenset(
    (
        errno.EACCES,
        errno.EPERM,
        errno.ENOENT,
        errno.EROFS,
        errno.EISDIR,
        errno.ENOTDIR,
    )
)


def _is_transient_transport_error(exc: Exception) -> bool:
    """True for botocore connection-level failures, which carry no response."""

    try:
        from botocore.exceptions import ConnectionError as _TransportError
    except ImportError:
        return False
    return isinstance(exc, _TransportError)


_READY_NAME = re.compile(
    r"^(?P<pack_id>[0-9a-f-]{36})\.(?P<created>[0-9]+)\."
    r"(?P<records>[0-9]+)\."
    r"(?P<checksum>[0-9a-f]{64})\.dmi-pack\.ready$"
)


class SpoolFullError(RuntimeError):
    pass


# Optimistic recovery passes before recover() falls back to holding the lock
# for one final, necessarily consistent pass.
_RECOVER_ATTEMPTS = 8


@dataclass(frozen=True, slots=True)
class SpoolSnapshot:
    entries: int
    bytes: int
    peak_bytes: int
    max_bytes: int


@dataclass(frozen=True, slots=True)
class StagedPack:
    pack_id: str
    created_at_ns: int
    record_count: int
    checksum: str
    object_key: str
    path: Path
    object_bytes: int

    def open(self) -> BinaryIO:
        return FilesystemPackStore._open_regular(self.path)


class DurablePackSpool:
    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root).resolve()
        fsync_new_root(self.root)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # Incremented under the lock by every accounting mutation (stage,
        # remove, quarantine, recovery commit). recover() validates entries
        # without the lock, so this is what tells it whether the aggregate it
        # computed is still the truth when it comes back to assign it.
        self._mutation_generation = 0
        ready = tuple(self.root.rglob("*.dmi-pack.ready"))
        stale = tuple(self.root.rglob("*.open"))
        self._bytes = sum(path.lstat().st_size for path in (*ready, *stale))
        self._entries = len(ready)
        self._peak_bytes = self._bytes

    def stage(self, pack: PackSource, object_key: str) -> StagedPack:
        validate_pack_source(pack)
        key = validate_object_key(object_key)
        if key.name != f"{pack.pack_id}.dmi-pack":
            raise ValueError("spool object key must end with the pack ID")
        ready = self._ready_path(key, pack)
        with self._lock:
            if ready.exists():
                return self._existing(pack, object_key, ready)
            conflicts = tuple(ready.parent.glob(f"{pack.pack_id}.*.dmi-pack.ready"))
            if conflicts:
                raise PackConflictError(
                    f"spool already contains a different pack intent: {pack.pack_id}"
                )
            if self._bytes + pack.object_bytes > self.max_bytes:
                raise SpoolFullError(
                    f"spool byte limit exceeded: "
                    f"{self._bytes + pack.object_bytes} > {self.max_bytes}"
                )
            ready.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{pack.pack_id}.",
                    suffix=".open",
                    dir=ready.parent,
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    copy_pack_source(pack, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temp_path, ready)
                except FileExistsError:
                    return self._existing(pack, object_key, ready)
                temp_path.unlink()
                temp_path = None
                # The key's tenant/date/session directories may all be new;
                # each level must be fsynced or power loss can drop the
                # subtree after stage() reported the pack durable.
                fsync_path_to_root(ready.parent, self.root)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            self._bytes += pack.object_bytes
            self._entries += 1
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            self._mutation_generation += 1
            return self._entry(ready)

    def recover(self) -> tuple[StagedPack, ...]:
        # Per-entry validation re-hashes every ready file, so it must run
        # without the lock -- holding it would block stage() behind a full
        # re-hash of the backlog. That leaves the final accounting assignment
        # racing any stage()/remove() that lands mid-pass: assigning the
        # pass's aggregate would erase their updates and defeat max_bytes.
        # Each pass therefore records the mutation generation with its
        # listing and only assigns if nothing moved; otherwise it re-runs,
        # and after _RECOVER_ATTEMPTS optimistic tries the last pass holds
        # the lock throughout, which cannot race but may briefly stall
        # staging behind the re-hash.
        for _ in range(_RECOVER_ATTEMPTS):
            with self._lock:
                generation = self._mutation_generation
                paths = self._begin_recovery_locked()
            entries = self._validate_ready(paths, locked=False)
            with self._lock:
                if self._mutation_generation != generation:
                    continue
                self._commit_recovery_locked(entries)
            return tuple(entries)
        with self._lock:
            paths = self._begin_recovery_locked()
            entries = self._validate_ready(paths, locked=True)
            self._commit_recovery_locked(entries)
        return tuple(entries)

    def _begin_recovery_locked(self) -> tuple[Path, ...]:
        stale = tuple(self.root.rglob("*.open"))
        for path in stale:
            path.unlink(missing_ok=True)
        for parent in {path.parent for path in stale}:
            self._fsync_directory(parent)
        return tuple(sorted(self.root.rglob("*.dmi-pack.ready")))

    def _validate_ready(
        self, paths: tuple[Path, ...], *, locked: bool
    ) -> list[StagedPack]:
        entries: list[StagedPack] = []
        for path in paths:
            # Validate each entry independently. One corrupt file must not
            # block every healthy pack behind it -- it is quarantined for
            # offline inspection instead -- and a concurrent uploader may
            # legitimately remove entries while this pass runs, which is
            # completed work rather than corruption.
            try:
                entry = self._entry(path)
                checksum = self._checksum(path)
            except FileNotFoundError:
                continue
            except (PackIntegrityError, ValueError):
                if path.exists():
                    self._quarantine(path, locked=locked)
                continue
            if checksum != entry.checksum:
                self._quarantine(path, locked=locked)
                continue
            entries.append(entry)
        return entries

    def _commit_recovery_locked(self, entries: list[StagedPack]) -> None:
        self._bytes = sum(entry.object_bytes for entry in entries)
        self._entries = len(entries)
        self._peak_bytes = max(self._peak_bytes, self._bytes)
        self._mutation_generation += 1

    def _quarantine(self, path: Path, *, locked: bool = False) -> None:
        """Sideline a ready file that failed integrity validation.

        The bytes are kept for diagnosis but the ``.ready`` suffix is
        dropped, so later passes neither upload the file nor fail on it.
        Quarantined files are no longer counted against ``max_bytes``;
        cleaning them up is an operator action.

        ``locked`` says whether the caller already holds ``self._lock``:
        removing a ready file is an accounting mutation, so the generation
        must move either way, and the lock is not reentrant.
        """

        target = path.with_suffix(".quarantined")
        try:
            os.replace(path, target)
        except FileNotFoundError:
            return
        self._fsync_directory(path.parent)
        if locked:
            self._mutation_generation += 1
        else:
            with self._lock:
                self._mutation_generation += 1

    def remove(self, staged: StagedPack) -> None:
        if staged.path.is_symlink():
            raise PackIntegrityError("ready pack must be a regular spool file")
        path = staged.path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("staged pack is outside the spool root")
        with self._lock:
            if not path.exists():
                return
            current = self._entry(path)
            if current != staged:
                raise PackIntegrityError("staged pack identity changed before removal")
            if path.stat().st_size != staged.object_bytes:
                raise PackIntegrityError("staged pack size changed before removal")
            path.unlink()
            self._fsync_directory(path.parent)
            self._bytes -= staged.object_bytes
            self._entries -= 1
            self._mutation_generation += 1

    def snapshot(self) -> SpoolSnapshot:
        with self._lock:
            return SpoolSnapshot(
                entries=self._entries,
                bytes=self._bytes,
                peak_bytes=self._peak_bytes,
                max_bytes=self.max_bytes,
            )

    def _ready_path(self, key, pack: PackSource) -> Path:
        name = (
            f"{pack.pack_id}.{pack.created_at_ns}.{pack.record_count}."
            f"{pack.checksum}.dmi-pack.ready"
        )
        path = self.root.joinpath(*key.parent.parts, name)
        if not path.parent.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("object key escapes the spool root")
        return path

    def _entry(self, path: Path) -> StagedPack:
        resolved = path.resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or not resolved.is_relative_to(self.root)
        ):
            raise PackIntegrityError("ready pack must be a regular spool file")
        match = _READY_NAME.fullmatch(path.name)
        if match is None:
            raise PackIntegrityError(f"invalid ready-pack name: {path.name}")
        relative = path.relative_to(self.root)
        pack_id = match.group("pack_id")
        object_key = str(relative.parent / f"{pack_id}.dmi-pack")
        staged = StagedPack(
            pack_id=pack_id,
            created_at_ns=int(match.group("created")),
            record_count=int(match.group("records")),
            checksum=match.group("checksum"),
            object_key=object_key,
            path=path,
            object_bytes=path.stat().st_size,
        )
        validate_pack_source(staged)
        return staged

    def _existing(
        self, pack: PackSource, object_key: str, ready: Path
    ) -> StagedPack:
        staged = self._entry(ready)
        if (
            staged.object_key != object_key
            or staged.object_bytes != pack.object_bytes
            or staged.checksum != pack.checksum
            or self._checksum(ready) != pack.checksum
        ):
            raise PackConflictError(f"spool contains different content: {object_key}")
        return staged

    @staticmethod
    def _checksum(path: Path) -> str:
        return FilesystemPackStore._checksum(path)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class DurablePackSink:
    def __init__(self, spool: DurablePackSpool) -> None:
        self._spool = spool

    def persist(self, ready: ReadyPack) -> StagedPack:
        return self._spool.stage(ready.pack, object_key_for(ready))


class SpoolUploader:
    def __init__(self, spool: DurablePackSpool, store: PackStore) -> None:
        self._spool = spool
        self._store = store

    def upload(self, staged: StagedPack) -> PackRef:
        # put() already verifies the source and the resulting object; a
        # post-upload stat() would only re-read metadata put() itself wrote
        # (a full re-hash on filesystem stores, a tautological HeadObject on
        # S3), so the ref put() returns is trusted as-is.
        ref = self._store.put(staged, staged.object_key)
        self._spool.remove(staged)
        return ref

    def upload_pending(self, *, limit: int | None = None) -> tuple[PackRef, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        pending = self._spool.recover()
        if limit is not None:
            pending = pending[:limit]
        return tuple(self.upload(staged) for staged in pending)


@dataclass(frozen=True, slots=True)
class ParallelUploadConfig:
    max_workers: int
    max_in_flight_bytes: int
    max_attempts: int = 4
    base_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 10
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        for name in ("max_workers", "max_in_flight_bytes", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "base_backoff_seconds",
            "max_backoff_seconds",
            "jitter_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_backoff_seconds < self.base_backoff_seconds:
            raise ValueError("max_backoff_seconds must cover base_backoff_seconds")
        if self.jitter_ratio > 1:
            raise ValueError("jitter_ratio must not exceed one")


@dataclass(frozen=True, slots=True)
class UploadFailure:
    pack_id: str
    object_key: str
    attempts: int
    error_type: str


@dataclass(frozen=True, slots=True)
class UploadSnapshot:
    attempted_packs: int
    uploaded_packs: int
    uploaded_bytes: int
    failed_packs: int
    retries: int
    peak_active_uploads: int
    peak_in_flight_bytes: int
    event_callback_failures: int
    duration_count: int
    duration_total_ns: int
    duration_max_ns: int


@dataclass(frozen=True, slots=True)
class UploadBatchResult:
    refs: tuple[PackRef, ...]
    failures: tuple[UploadFailure, ...]
    snapshot: UploadSnapshot


@dataclass(frozen=True, slots=True)
class UploadEvent:
    event: str
    fields: Mapping[str, int | str]


@dataclass(frozen=True, slots=True)
class _UploadOutcome:
    ref: PackRef | None
    failure: UploadFailure | None
    attempts: int
    duration_ns: int


class ParallelSpoolUploader:
    def __init__(
        self,
        spool: DurablePackSpool,
        store: PackStore,
        config: ParallelUploadConfig,
        *,
        event_callback: Callable[[UploadEvent], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._spool = spool
        self._uploader = SpoolUploader(spool, store)
        self._config = config
        self._event_callback = event_callback
        self._sleep = sleep
        self._random_value = random_value
        self._clock_ns = clock_ns
        self._event_callback_failures = 0
        self._event_lock = threading.Lock()

    def upload_pending(self, *, limit: int | None = None) -> UploadBatchResult:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be positive")
        pending = list(self._spool.recover())
        if limit is not None:
            pending = pending[:limit]
        oversized = next(
            (
                staged
                for staged in pending
                if staged.object_bytes > self._config.max_in_flight_bytes
            ),
            None,
        )
        if oversized is not None:
            raise ValueError(
                f"pack {oversized.pack_id} exceeds the in-flight byte limit"
            )

        active_bytes = 0
        peak_active = 0
        peak_bytes = 0
        # Outcomes are stored by list position, never by pack_id: recover()
        # walks the whole tree, so two ready files sharing a pack_id (however
        # unlikely) must not overwrite each other's outcome -- that could
        # report a failed upload as succeeded.
        ordered: list[_UploadOutcome | None] = [None] * len(pending)
        futures: dict[Future[_UploadOutcome], tuple[int, StagedPack]] = {}
        remaining = list(enumerate(pending))
        with ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="dmi-pack-upload",
        ) as executor:
            while remaining or futures:
                while len(futures) < self._config.max_workers:
                    selected = next(
                        (
                            (slot, entry)
                            for slot, entry in enumerate(remaining)
                            if active_bytes + entry[1].object_bytes
                            <= self._config.max_in_flight_bytes
                        ),
                        None,
                    )
                    if selected is None:
                        break
                    slot, (index, staged) = selected
                    remaining.pop(slot)
                    future = executor.submit(self._upload, staged)
                    futures[future] = (index, staged)
                    active_bytes += staged.object_bytes
                    peak_active = max(peak_active, len(futures))
                    peak_bytes = max(peak_bytes, active_bytes)
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, staged = futures.pop(future)
                    active_bytes -= staged.object_bytes
                    ordered[index] = future.result()

        assert all(outcome is not None for outcome in ordered)
        refs = tuple(
            outcome.ref
            for outcome in ordered
            if outcome is not None and outcome.ref is not None
        )
        failures = tuple(
            outcome.failure
            for outcome in ordered
            if outcome is not None and outcome.failure is not None
        )
        durations = [
            outcome.duration_ns for outcome in ordered if outcome is not None
        ]
        snapshot = UploadSnapshot(
            attempted_packs=len(ordered),
            uploaded_packs=len(refs),
            uploaded_bytes=sum(ref.object_bytes for ref in refs),
            failed_packs=len(failures),
            retries=sum(max(0, outcome.attempts - 1) for outcome in ordered),
            peak_active_uploads=peak_active,
            peak_in_flight_bytes=peak_bytes,
            event_callback_failures=self._event_callback_failures,
            duration_count=len(durations),
            duration_total_ns=sum(durations),
            duration_max_ns=max(durations, default=0),
        )
        return UploadBatchResult(refs=refs, failures=failures, snapshot=snapshot)

    def _upload(self, staged: StagedPack) -> _UploadOutcome:
        started = self._clock_ns()
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                ref = self._uploader.upload(staged)
            except Exception as exc:
                if attempt == self._config.max_attempts or not self._retryable(exc):
                    failure = UploadFailure(
                        pack_id=staged.pack_id,
                        object_key=staged.object_key,
                        attempts=attempt,
                        error_type=type(exc).__name__,
                    )
                    self._emit(
                        "pack_upload_failed",
                        pack_id=staged.pack_id,
                        attempt=attempt,
                        error_type=failure.error_type,
                    )
                    return _UploadOutcome(
                        ref=None,
                        failure=failure,
                        attempts=attempt,
                        duration_ns=max(0, self._clock_ns() - started),
                    )
                self._emit(
                    "pack_upload_retry",
                    pack_id=staged.pack_id,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
                self._sleep(self._backoff(attempt))
                continue
            self._emit(
                "pack_upload_committed",
                pack_id=staged.pack_id,
                attempt=attempt,
                object_bytes=staged.object_bytes,
            )
            return _UploadOutcome(
                ref=ref,
                failure=None,
                attempts=attempt,
                duration_ns=max(0, self._clock_ns() - started),
            )
        raise AssertionError(  # pragma: no cover - loop always returns or raises
            "unreachable upload attempt state"
        )

    def _backoff(self, attempt: int) -> float:
        base = min(
            self._config.max_backoff_seconds,
            self._config.base_backoff_seconds * 2 ** (attempt - 1),
        )
        return min(
            self._config.max_backoff_seconds,
            base + base * self._config.jitter_ratio * self._random_value(),
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, OSError):
            # Local I/O errors are mostly transient (EIO under load, EAGAIN,
            # interrupted syscalls), but a deterministic failure must not
            # burn every attempt at full backoff.
            return exc.errno not in _PERMANENT_ERRNOS
        if _is_transient_transport_error(exc):
            # Botocore connection failures carry no HTTP response for the
            # shape-based classification below, yet they are the archetypal
            # retryable error -- without this branch a network blip is
            # classified permanent while local permission errors retried.
            return True
        response = getattr(exc, "response", None)
        if not isinstance(response, Mapping):
            return False
        metadata = response.get("ResponseMetadata", {})
        error = response.get("Error", {})
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
        code = error.get("Code") if isinstance(error, Mapping) else None
        return status in {408, 429} or (
            isinstance(status, int) and status >= 500
        ) or code in {
            "InternalError",
            "RequestTimeout",
            "ServiceUnavailable",
            "SlowDown",
            "Throttling",
        }

    def _emit(self, event: str, **fields: int | str) -> None:
        if self._event_callback is None:
            return
        with self._event_lock:
            try:
                self._event_callback(UploadEvent(event=event, fields=fields))
            except Exception:
                self._event_callback_failures += 1
