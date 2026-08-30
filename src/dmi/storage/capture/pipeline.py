from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import math
import threading
import time
from typing import Callable, Mapping, Protocol
from urllib.parse import quote
from uuid import UUID, uuid4

from .model import (
    CaptureMetadata,
    CaptureRecord,
    DuplicateCaptureError,
    PackRef,
    PackSource,
    PackStore,
)
from .pack import PackCapacityError, PackWriter, SealedPack


class AdmissionResult(str, Enum):
    ACCEPTED = "accepted"
    DROPPED = "dropped"
    TIMED_OUT = "timed_out"
    TOO_LARGE = "too_large"
    CLOSED = "closed"


class OverloadPolicy(str, Enum):
    BLOCK = "block"
    DROP_NEWEST = "drop_newest"


class FlushReason(str, Enum):
    SIZE = "size"
    RECORDS = "records"
    LINGER = "linger"
    SESSION = "session"
    MANUAL = "manual"
    SHUTDOWN = "shutdown"


class OversizedRecordError(ValueError):
    pass


class PipelineFailedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    records: int
    bytes: int
    peak_records: int
    peak_bytes: int
    closed: bool


class BoundedRecordQueue:
    def __init__(self, *, max_records: int, max_bytes: int) -> None:
        if (
            type(max_records) is not int
            or type(max_bytes) is not int
            or max_records <= 0
            or max_bytes <= 0
        ):
            raise ValueError("queue limits must be positive")
        self._max_records = max_records
        self._max_bytes = max_bytes
        self._items: deque[CaptureRecord | _FlushBarrier] = deque()
        self._records = 0
        self._bytes = 0
        self._peak_records = 0
        self._peak_bytes = 0
        self._closed = False
        self._condition = threading.Condition()

    def put(
        self,
        record: CaptureRecord,
        *,
        policy: OverloadPolicy,
        timeout: float | None = None,
    ) -> AdmissionResult:
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative")
        if not isinstance(policy, OverloadPolicy):
            raise ValueError("unknown overload policy")
        record_bytes = len(record.payload)
        if record_bytes > self._max_bytes:
            return AdmissionResult.TOO_LARGE
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    return AdmissionResult.CLOSED
                if self._fits(record_bytes):
                    self._items.append(record)
                    self._records += 1
                    self._bytes += record_bytes
                    self._peak_records = max(self._peak_records, self._records)
                    self._peak_bytes = max(self._peak_bytes, self._bytes)
                    self._condition.notify()
                    return AdmissionResult.ACCEPTED
                if policy is OverloadPolicy.DROP_NEWEST:
                    return AdmissionResult.DROPPED
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return AdmissionResult.TIMED_OUT
                self._condition.wait(remaining)

    def put_barrier(self, barrier: _FlushBarrier) -> AdmissionResult:
        """Append one control item without consuming record capacity."""

        with self._condition:
            if self._closed:
                return AdmissionResult.CLOSED
            self._items.append(barrier)
            self._condition.notify()
            return AdmissionResult.ACCEPTED

    def get(
        self, timeout: float | None = None
    ) -> CaptureRecord | _FlushBarrier | None:
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            item = self._items.popleft()
            if isinstance(item, CaptureRecord):
                self._records -= 1
                self._bytes -= len(item.payload)
            self._condition.notify_all()
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self) -> QueueSnapshot:
        with self._condition:
            return QueueSnapshot(
                records=self._records,
                bytes=self._bytes,
                peak_records=self._peak_records,
                peak_bytes=self._peak_bytes,
                closed=self._closed,
            )

    def _fits(self, record_bytes: int) -> bool:
        return (
            self._records < self._max_records
            and self._bytes + record_bytes <= self._max_bytes
        )


@dataclass(frozen=True, slots=True)
class ReadyPack:
    pack: SealedPack
    first_metadata: CaptureMetadata
    reason: FlushReason


class PackAssembler:
    def __init__(
        self,
        *,
        max_pack_bytes: int,
        max_records: int,
        max_linger_ns: int,
        pack_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        limits = (max_pack_bytes, max_records, max_linger_ns)
        if any(type(value) is not int or value <= 0 for value in limits):
            raise ValueError("pack limits and linger must be positive")
        self._max_pack_bytes = max_pack_bytes
        self._max_records = max_records
        self._max_linger_ns = max_linger_ns
        self._pack_id_factory = pack_id_factory
        self._writer: PackWriter | None = None
        self._first_metadata: CaptureMetadata | None = None
        self._scope: tuple[str, str, int] | None = None
        self._opened_ns: int | None = None

    def append(self, record: CaptureRecord, *, now_ns: int) -> tuple[ReadyPack, ...]:
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        emitted: list[ReadyPack] = []
        # Enforce max_linger_ns on the append path as well: the drain loop
        # only checks expiry when the queue goes idle, so under continuous
        # traffic an open pack would otherwise linger unbounded.
        emitted.extend(self.flush_expired(now_ns=now_ns))
        scope = self._record_scope(record)
        if self._writer is None:
            writer = self._writer_with(record)
            self._adopt(writer, record.metadata, scope=scope, now_ns=now_ns)
        elif scope != self._scope:
            writer = self._writer_with(record)
            emitted.extend(self.flush(FlushReason.SESSION))
            self._adopt(writer, record.metadata, scope=scope, now_ns=now_ns)
        else:
            try:
                self._writer.append(record)
            except PackCapacityError:
                writer = self._writer_with(record)
                reason = (
                    FlushReason.RECORDS
                    if self._writer.record_count >= self._max_records
                    else FlushReason.SIZE
                )
                emitted.extend(self.flush(reason))
                self._adopt(writer, record.metadata, scope=scope, now_ns=now_ns)
        assert self._writer is not None
        if self._writer.record_count >= self._max_records:
            emitted.extend(self.flush(FlushReason.RECORDS))
        return tuple(emitted)

    def _writer_with(self, record: CaptureRecord) -> PackWriter:
        writer = PackWriter(
            pack_id=self._pack_id_factory(),
            created_at_ns=record.metadata.captured_at_ns,
            max_pack_bytes=self._max_pack_bytes,
            max_records=self._max_records,
        )
        try:
            writer.append(record)
        except PackCapacityError as exc:
            raise OversizedRecordError(
                f"capture {record.metadata.capture_id} does not fit an empty pack"
            ) from exc
        return writer

    def _adopt(
        self,
        writer: PackWriter,
        metadata: CaptureMetadata,
        *,
        scope: tuple[str, str, int],
        now_ns: int,
    ) -> None:
        self._writer = writer
        self._first_metadata = metadata
        self._scope = scope
        self._opened_ns = now_ns

    def flush_expired(self, *, now_ns: int) -> tuple[ReadyPack, ...]:
        if self._opened_ns is None or now_ns - self._opened_ns < self._max_linger_ns:
            return ()
        return self.flush(FlushReason.LINGER)

    def seconds_until_expiry(self, *, now_ns: int) -> float | None:
        if self._opened_ns is None:
            return None
        remaining = self._max_linger_ns - (now_ns - self._opened_ns)
        return max(0, remaining) / 1_000_000_000

    def flush(self, reason: FlushReason) -> tuple[ReadyPack, ...]:
        if self._writer is None:
            return ()
        assert self._first_metadata is not None
        ready = ReadyPack(self._writer.seal(), self._first_metadata, reason)
        self._reset()
        return (ready,)

    def _reset(self) -> None:
        self._writer = None
        self._first_metadata = None
        self._scope = None
        self._opened_ns = None

    @staticmethod
    def _record_scope(record: CaptureRecord) -> tuple[str, str, int]:
        metadata = record.metadata
        return metadata.tenant_id, metadata.session_id, metadata.producer_rank


def object_key_for(ready: ReadyPack) -> str:
    metadata = ready.first_metadata
    captured = datetime.fromtimestamp(
        metadata.captured_at_ns / 1_000_000_000, tz=timezone.utc
    )
    return (
        f"v1/tenant={_key_component(metadata.tenant_id)}/"
        f"date={captured:%Y-%m-%d}/"
        f"session={_key_component(metadata.session_id)}/"
        f"rank={metadata.producer_rank}/"
        f"{ready.pack.pack_id}.dmi-pack"
    )


def _key_component(value: str) -> str:
    # quote() treats "~" as always-safe per RFC 3986 and ignores `safe` for it,
    # but the object-key pattern does not allow it. Left alone, an identifier
    # containing "~" produces a key every store rejects, and the sink failure
    # is fatal to the whole pipeline.
    encoded = quote(value, safe="-_.=").replace("~", "%7E")
    if len(encoded.encode()) <= 160:
        return encoded
    return "sha256-" + sha256(value.encode()).hexdigest()


class PackSink(Protocol):
    def persist(self, ready: ReadyPack) -> PackRef | PackSource: ...


class DirectPackSink:
    def __init__(self, store: PackStore) -> None:
        self._store = store
        self.last_ref: PackRef | None = None

    def persist(self, ready: ReadyPack) -> PackRef:
        ref = self._store.put(ready.pack, object_key_for(ready))
        self.last_ref = ref
        return ref


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    max_queue_records: int
    max_queue_bytes: int
    max_pack_bytes: int
    max_pack_records: int
    max_linger_ns: int
    overload_policy: OverloadPolicy = OverloadPolicy.DROP_NEWEST
    admission_timeout: float | None = None

    def __post_init__(self) -> None:
        positive = (
            "max_queue_records",
            "max_queue_bytes",
            "max_pack_bytes",
            "max_pack_records",
            "max_linger_ns",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in positive
        ):
            raise ValueError("pipeline bounds must be positive")
        if not isinstance(self.overload_policy, OverloadPolicy):
            raise ValueError("unknown overload policy")
        if self.admission_timeout is not None and (
            isinstance(self.admission_timeout, bool)
            or not math.isfinite(self.admission_timeout)
            or self.admission_timeout < 0
        ):
            raise ValueError("admission_timeout must be non-negative")


@dataclass(frozen=True, slots=True)
class HistogramSnapshot:
    bounds_ns: tuple[int, ...]
    counts: tuple[int, ...]
    count: int
    total_ns: int
    max_ns: int


class _Histogram:
    _BOUNDS_NS = (
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
    )

    def __init__(self) -> None:
        self._counts = [0] * (len(self._BOUNDS_NS) + 1)
        self._count = 0
        self._total_ns = 0
        self._max_ns = 0

    def observe(self, duration_ns: int) -> None:
        bucket = next(
            (
                index
                for index, bound in enumerate(self._BOUNDS_NS)
                if duration_ns <= bound
            ),
            len(self._BOUNDS_NS),
        )
        self._counts[bucket] += 1
        self._count += 1
        self._total_ns += duration_ns
        self._max_ns = max(self._max_ns, duration_ns)

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            bounds_ns=self._BOUNDS_NS,
            counts=tuple(self._counts),
            count=self._count,
            total_ns=self._total_ns,
            max_ns=self._max_ns,
        )


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    submitted_records: int
    admitted_records: int
    admitted_bytes: int
    dropped_records: int
    timed_out_records: int
    oversized_records: int
    duplicate_records: int
    rejected_closed_records: int
    persisted_records: int
    packs_persisted: int
    packed_bytes: int
    flush_size: int
    flush_records: int
    flush_linger: int
    flush_session: int
    flush_manual: int
    flush_shutdown: int
    failures: int
    event_callback_failures: int
    queue_records: int
    queue_bytes: int
    queue_peak_records: int
    queue_peak_bytes: int
    admission_duration: HistogramSnapshot
    persist_duration: HistogramSnapshot


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    event: str
    fields: Mapping[str, int | str]


@dataclass(slots=True)
class _FlushBarrier:
    target_admitted: int
    completed: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class HostCapturePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        sink: PackSink,
        *,
        pack_id_factory: Callable[[], UUID] = uuid4,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        event_callback: Callable[[PipelineEvent], None] | None = None,
    ) -> None:
        self._config = config
        self._sink = sink
        self._clock_ns = clock_ns
        self._event_callback = event_callback
        self._queue = BoundedRecordQueue(
            max_records=config.max_queue_records,
            max_bytes=config.max_queue_bytes,
        )
        self._assembler = PackAssembler(
            max_pack_bytes=config.max_pack_bytes,
            max_records=config.max_pack_records,
            max_linger_ns=config.max_linger_ns,
            pack_id_factory=pack_id_factory,
        )
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending_flush: _FlushBarrier | None = None
        self._error: BaseException | None = None
        self._counters = {
            "submitted_records": 0,
            "admitted_records": 0,
            "admitted_bytes": 0,
            "dropped_records": 0,
            "timed_out_records": 0,
            "oversized_records": 0,
            "duplicate_records": 0,
            "rejected_closed_records": 0,
            "persisted_records": 0,
            "packs_persisted": 0,
            "packed_bytes": 0,
            "flush_size": 0,
            "flush_records": 0,
            "flush_linger": 0,
            "flush_session": 0,
            "flush_manual": 0,
            "flush_shutdown": 0,
            "failures": 0,
            "event_callback_failures": 0,
        }
        self._admission_duration = _Histogram()
        self._persist_duration = _Histogram()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("pipeline has already been started")
            self._thread = threading.Thread(
                target=self._run, name="dmi-capture-persistence", daemon=True
            )
            self._thread.start()

    @property
    def is_running(self) -> bool:
        """Whether the persistence worker is alive and still accepts records."""

        with self._lock:
            thread = self._thread
            error = self._error
        return (
            thread is not None
            and thread.is_alive()
            and error is None
            and not self._queue.snapshot().closed
        )

    def submit(self, record: CaptureRecord) -> AdmissionResult:
        with self._lock:
            if self._thread is None:
                raise RuntimeError("pipeline is not started")
            error = self._error
            self._counters["submitted_records"] += 1
        if error is not None:
            raise PipelineFailedError("capture pipeline failed") from error
        if len(record.payload) > self._config.max_pack_bytes:
            # Queue admission bounds max_queue_bytes, which is unrelated to
            # max_pack_bytes. Without this, a payload no pack could ever hold is
            # admitted here and only rejected on the persistence thread, where
            # it reads as a fatal pipeline error.
            with self._lock:
                self._counters["oversized_records"] += 1
            return AdmissionResult.TOO_LARGE
        started = self._clock_ns()
        result = self._queue.put(
            record,
            policy=self._config.overload_policy,
            timeout=self._config.admission_timeout,
        )
        duration = self._clock_ns() - started
        with self._lock:
            self._admission_duration.observe(max(0, duration))
            if result is AdmissionResult.ACCEPTED:
                self._counters["admitted_records"] += 1
                self._counters["admitted_bytes"] += len(record.payload)
            elif result is AdmissionResult.DROPPED:
                self._counters["dropped_records"] += 1
            elif result is AdmissionResult.TIMED_OUT:
                self._counters["timed_out_records"] += 1
            elif result is AdmissionResult.TOO_LARGE:
                self._counters["oversized_records"] += 1
            elif result is AdmissionResult.CLOSED:
                self._counters["rejected_closed_records"] += 1
        return result

    def close(self, *, timeout: float | None = None) -> PipelineSnapshot:
        self._queue.close()
        with self._lock:
            thread = self._thread
        if thread is None:
            raise RuntimeError("pipeline is not started")
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("capture pipeline did not stop before timeout")
        with self._lock:
            error = self._error
        if error is not None:
            raise PipelineFailedError("capture pipeline failed") from error
        return self.snapshot()

    def flush(self, *, timeout: float | None = None) -> bool:
        """Persist all records admitted before this non-closing barrier.

        The capture pipeline remains open after a successful flush.  Calls are
        serialized and reuse a timed-out in-flight barrier, so repeated
        timeouts cannot grow an unbounded control queue.
        """

        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            if self._thread is None:
                raise RuntimeError("pipeline is not started")
            requested_target = self._counters["admitted_records"]
        acquired = self._flush_lock.acquire(
            timeout=-1 if deadline is None else max(0.0, deadline - time.monotonic())
        )
        if not acquired:
            return False
        try:
            with self._lock:
                error = self._error
                pending = self._pending_flush
            if error is not None:
                raise PipelineFailedError("capture pipeline failed") from error

            if pending is not None:
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                if not pending.completed.wait(remaining):
                    return False
                if pending.error is not None:
                    raise PipelineFailedError("capture pipeline failed") from pending.error
                with self._lock:
                    if self._pending_flush is pending:
                        self._pending_flush = None
                if pending.target_admitted >= requested_target:
                    return True

            pending = _FlushBarrier(target_admitted=requested_target)
            with self._lock:
                # Publish the waiter before the queue item.  Otherwise the
                # worker could fail an earlier record in the small window
                # between enqueue and assignment, leaving this waiter asleep.
                self._pending_flush = pending
            if self._queue.put_barrier(pending) is AdmissionResult.CLOSED:
                with self._lock:
                    error = self._error
                    if self._pending_flush is pending:
                        self._pending_flush = None
                if error is not None:
                    raise PipelineFailedError("capture pipeline failed") from error
                return False

            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            if not pending.completed.wait(remaining):
                return False
            if pending.error is not None:
                raise PipelineFailedError("capture pipeline failed") from pending.error
            with self._lock:
                if self._pending_flush is pending:
                    self._pending_flush = None
            return True
        finally:
            self._flush_lock.release()

    def raise_if_failed(self) -> None:
        """Raise the pipeline's latched asynchronous failure, if any."""

        with self._lock:
            error = self._error
        if error is not None:
            raise PipelineFailedError("capture pipeline failed") from error

    def snapshot(self) -> PipelineSnapshot:
        queue = self._queue.snapshot()
        with self._lock:
            return PipelineSnapshot(
                **self._counters,
                queue_records=queue.records,
                queue_bytes=queue.bytes,
                queue_peak_records=queue.peak_records,
                queue_peak_bytes=queue.peak_bytes,
                admission_duration=self._admission_duration.snapshot(),
                persist_duration=self._persist_duration.snapshot(),
            )

    def _run(self) -> None:
        try:
            while True:
                now_ns = self._clock_ns()
                timeout = self._assembler.seconds_until_expiry(now_ns=now_ns)
                item = self._queue.get(timeout=timeout)
                if item is None:
                    queue = self._queue.snapshot()
                    if queue.closed and queue.records == 0:
                        self._persist(self._assembler.flush(FlushReason.SHUTDOWN))
                        return
                    self._persist(
                        self._assembler.flush_expired(now_ns=self._clock_ns())
                    )
                    continue
                if isinstance(item, _FlushBarrier):
                    try:
                        self._persist(self._assembler.flush(FlushReason.MANUAL))
                    except BaseException as exc:
                        item.error = exc
                        item.completed.set()
                        raise
                    item.completed.set()
                    continue
                record = item
                try:
                    packs = self._assembler.append(record, now_ns=self._clock_ns())
                except OversizedRecordError:
                    # Framing overhead can push a payload that cleared admission
                    # past max_pack_bytes. PackAssembler is written to survive
                    # this with its buffered pack intact, so drop the one record
                    # and keep going rather than failing the pipeline.
                    with self._lock:
                        self._counters["oversized_records"] += 1
                    self._emit(
                        "record_oversized",
                        capture_id=record.metadata.capture_id,
                        bytes=len(record.payload),
                    )
                    continue
                except DuplicateCaptureError:
                    # A repeated capture_id in one open pack (e.g. a producer
                    # retry after an ambiguous admission) is a fault of that
                    # one record. The buffered pack is untouched, so drop the
                    # duplicate and keep going rather than failing the whole
                    # pipeline and losing every queued record.
                    with self._lock:
                        self._counters["duplicate_records"] += 1
                    self._emit(
                        "record_duplicate",
                        capture_id=record.metadata.capture_id,
                    )
                    continue
                self._persist(packs)
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._counters["failures"] += 1
                pending = self._pending_flush
            self._queue.close()
            if pending is not None and not pending.completed.is_set():
                pending.error = exc
                pending.completed.set()
            self._emit("pipeline_failed", error_type=type(exc).__name__)

    def _persist(self, packs: tuple[ReadyPack, ...]) -> None:
        for ready in packs:
            started = self._clock_ns()
            self._sink.persist(ready)
            duration = max(0, self._clock_ns() - started)
            reason_key = f"flush_{ready.reason.value}"
            with self._lock:
                self._persist_duration.observe(duration)
                self._counters["persisted_records"] += ready.pack.record_count
                self._counters["packs_persisted"] += 1
                self._counters["packed_bytes"] += len(ready.pack.data)
                self._counters[reason_key] += 1
            self._emit(
                "pack_persisted",
                pack_id=ready.pack.pack_id,
                reason=ready.reason.value,
                records=ready.pack.record_count,
                bytes=len(ready.pack.data),
            )

    def _emit(self, event: str, **fields: int | str) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(PipelineEvent(event=event, fields=fields))
        except Exception:
            with self._lock:
                self._counters["event_callback_failures"] += 1
