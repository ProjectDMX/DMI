from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
    validate_object_key,
    validate_pack_source,
)
from .model import PackConflictError, PackIntegrityError, PackRef, PackSource, PackStore
from .pipeline import ReadyPack, object_key_for


_READY_NAME = re.compile(
    r"^(?P<pack_id>[0-9a-f-]{36})\.(?P<created>[0-9]+)\."
    r"(?P<records>[0-9]+)\."
    r"(?P<checksum>[0-9a-f]{64})\.dmi-pack\.ready$"
)


class SpoolFullError(RuntimeError):
    pass


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
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
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
                self._fsync_directory(ready.parent)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            self._bytes += pack.object_bytes
            self._entries += 1
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            return self._entry(ready)

    def recover(self) -> tuple[StagedPack, ...]:
        with self._lock:
            stale = tuple(self.root.rglob("*.open"))
            for path in stale:
                path.unlink(missing_ok=True)
            for parent in {path.parent for path in stale}:
                self._fsync_directory(parent)
            paths = tuple(sorted(self.root.rglob("*.dmi-pack.ready")))
            self._bytes = sum(path.lstat().st_size for path in paths)
            self._entries = len(paths)
            self._peak_bytes = max(self._peak_bytes, self._bytes)
        entries = tuple(self._entry(path) for path in paths)
        for entry in entries:
            if self._checksum(entry.path) != entry.checksum:
                raise PackIntegrityError(f"spool checksum mismatch: {entry.pack_id}")
        return entries

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
        ref = self._store.put(staged, staged.object_key)
        info = self._store.stat(ref)
        if info.size != staged.object_bytes or info.checksum != staged.checksum:
            raise PackIntegrityError("remote object verification failed")
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
        outcomes: dict[str, _UploadOutcome] = {}
        futures: dict[Future[_UploadOutcome], StagedPack] = {}
        remaining = pending.copy()
        with ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="dmi-pack-upload",
        ) as executor:
            while remaining or futures:
                while len(futures) < self._config.max_workers:
                    selected = next(
                        (
                            (index, staged)
                            for index, staged in enumerate(remaining)
                            if active_bytes + staged.object_bytes
                            <= self._config.max_in_flight_bytes
                        ),
                        None,
                    )
                    if selected is None:
                        break
                    index, staged = selected
                    remaining.pop(index)
                    future = executor.submit(self._upload, staged)
                    futures[future] = staged
                    active_bytes += staged.object_bytes
                    peak_active = max(peak_active, len(futures))
                    peak_bytes = max(peak_bytes, active_bytes)
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    staged = futures.pop(future)
                    active_bytes -= staged.object_bytes
                    outcomes[staged.pack_id] = future.result()

        ordered = [outcomes[staged.pack_id] for staged in pending]
        refs = tuple(outcome.ref for outcome in ordered if outcome.ref is not None)
        failures = tuple(
            outcome.failure
            for outcome in ordered
            if outcome.failure is not None
        )
        durations = [outcome.duration_ns for outcome in ordered]
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
        raise AssertionError("unreachable upload attempt state")

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
