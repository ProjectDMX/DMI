# batching_queue.py
from __future__ import annotations

import time
import queue as _queue
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, replace
from threading import Condition, Lock
from typing import Any, Deque, Generic, List, Optional, TypeVar
import logging


SizeT = TypeVar("SizeT")
ItemT = TypeVar("ItemT", bound="SizedWorkItem")


class SizedWorkItem(ABC):
    @abstractmethod
    def size(self):
        """Return the item's size in arbitrary units (e.g., bytes)."""
        raise NotImplementedError


class ItemTooLargeError(ValueError):
    """Raised when an item exceeds a configured per-item size limit."""


class QueueClosedError(RuntimeError):
    """
    Raised when attempting to enqueue into a closed WatermarkBatchingQueue.

    Subclasses RuntimeError for backward compatibility with code that was catching RuntimeError
    and inspecting the message.
    """


@dataclass(frozen=True, slots=True)
class _Entry(Generic[ItemT, SizeT]):
    item: ItemT
    size: SizeT  # NOTE: Must not change after enqueue.
    # NOTE: size == 0 can make queue waiting forever without timeout.
    enqueued_at: float


# ---------------------------
# Queue profiling (optional)
# ---------------------------

@dataclass(frozen=True, slots=True)
class QueueProfilingConfig:
    """
    Configuration for queue profiling.

    If both enable_stats and enable_timing are False, profiling is disabled.

    Timing uses time.perf_counter() and is aggregated (totals), not per-call traces.
    """
    enable_stats: bool = False
    enable_timing: bool = False

    # Granular timing switches
    timing_profile_enqueue: bool = True
    timing_profile_dequeue: bool = True
    timing_profile_wait: bool = True
    timing_profile_batch_build: bool = True


@dataclass(slots=True)
class _QueueProfile:
    # counts
    enqueue_calls: int = 0
    enqueue_full_errors: int = 0
    enqueue_closed_errors: int = 0

    dequeue_calls: int = 0
    dequeue_empty_errors: int = 0  # includes non-blocking and user-timeout empties
    dequeued_batches: int = 0
    dequeued_items: int = 0
    dequeue_returned_empty_on_close: int = 0

    close_calls: int = 0
    notify_all_calls: int = 0

    # timing totals (seconds)
    enqueue_total_s: float = 0.0
    enqueue_wait_s: float = 0.0
    enqueue_wait_calls: int = 0

    dequeue_total_s: float = 0.0
    dequeue_wait_s: float = 0.0
    dequeue_wait_calls: int = 0
    dequeue_batch_build_s: float = 0.0


# TODO: Optimization on notify_all, when to notify a correct number of threads.
# Can know threads counts on both sides in advance.
class WatermarkBatchingQueue(Generic[ItemT, SizeT]):
    """
    Thread-safe micro-batching queue with backpressure (high-watermark) and deadlock avoidance.

    -------------------------
    SIZE TYPE ASSUMPTION
    -------------------------
    SizeT must be "0-compatible": totals can start at 0, and `+`, `-`, and comparisons with
    0/thresholds work.

    -------------------------
    FLUSH TRIGGERS (OR)
    -------------------------
    At least one of these must be enabled / not None:

      - min_batch_items: buffered_items >= min_batch_items
      - min_batch_size:  buffered_size  >= min_batch_size     (ACTUAL sum of item.size())
      - max_linger_s:    oldest_age     >= max_linger_s

    -------------------------
    LIMITS
    -------------------------
    (None removes that limit)

      - max_batch_items: cap on items returned by dequeue_batch()
      - max_batch_size:  cap on total size returned by dequeue_batch()
      - high_watermark_items: backpressure threshold on buffered item count
      - high_watermark_size:  backpressure threshold on buffered total size

    Per-item size constraint (hard; checked on enqueue):
      - If max_batch_size is set: item.size() must be <= max_batch_size
        (So a single item can always be emitted in some batch.)

    -------------------------
    SEMANTICS
    -------------------------
    (1) Dequeue is allowed when:
          flush_triggers_met OR high_watermark_reached
        where "high watermark reached" uses >= (not ==), so temporary overshoot still allows draining.

    (2) Deadlock-avoidance "nudge":
        If an enqueue would be blocked by a high-watermark limit, we allow EXACTLY ONE such enqueue
        *only if* it changes dequeue_allowed from False -> True.
        This prevents the classic size-granularity deadlock (e.g., 95/100 full, next item is 10).

    Close semantics:
      - close() prevents further enqueues, and allows draining remaining items regardless of triggers/watermarks.

    -------------------------
    PROFILING (OPTIONAL)
    -------------------------
    If profiling is enabled (QueueProfilingConfig), the queue aggregates:
      - counts for enqueue/dequeue/errors/notify/close
      - timing totals (perf_counter) for enqueue/dequeue/waits/batch-build
    """

    def __init__(
        self,
        *,
        min_batch_items: Optional[int] = None,
        min_batch_size: Optional[SizeT] = None,
        max_linger_s: Optional[float] = None,
        max_batch_items: Optional[int] = None,
        max_batch_size: Optional[SizeT] = None,
        high_watermark_items: Optional[int] = None,
        high_watermark_size: Optional[SizeT] = None,
        logger: Optional[logging.Logger] = None,
        # optional metadata/profiling
        name: str = "",
        profiling: Optional[QueueProfilingConfig] = None,
    ) -> None:
        if min_batch_items is None and min_batch_size is None and max_linger_s is None:
            raise ValueError("At least one of min_batch_items/min_batch_size/max_linger_s must be set")

        # Flush triggers
        self._min_batch_items: Optional[int] = min_batch_items
        if self._min_batch_items is not None and self._min_batch_items <= 0:
            raise ValueError("min_batch_items must be > 0 or None")

        self._min_batch_size: Optional[SizeT] = min_batch_size
        if self._min_batch_size is not None and self._min_batch_size < 0:  # assumes 0-compatible
            raise ValueError("min_batch_size must be >= 0 or None")

        self._max_linger_s: Optional[float] = max_linger_s
        if self._max_linger_s is not None and self._max_linger_s < 0:
            raise ValueError("max_linger_s must be >= 0 or None")

        # Batch caps
        self._max_batch_items: Optional[int] = max_batch_items
        if self._max_batch_items is not None and self._max_batch_items <= 0:
            raise ValueError("max_batch_items must be > 0 or None")

        self._max_batch_size: Optional[SizeT] = max_batch_size
        if self._max_batch_size is not None and self._max_batch_size <= 0:  # assumes 0-compatible
            raise ValueError("max_batch_size must be > 0 or None")

        # Backpressure watermarks
        self._high_watermark_items: Optional[int] = high_watermark_items
        if self._high_watermark_items is not None and self._high_watermark_items <= 0:
            raise ValueError("high_watermark_items must be > 0 or None")

        self._high_watermark_size: Optional[SizeT] = high_watermark_size
        if self._high_watermark_size is not None and self._high_watermark_size <= 0:  # assumes 0-compatible
            raise ValueError("high_watermark_size must be > 0 or None")

        self._name: str = name

        self._logger = logger

        # NOTE: Avoid deadlock, ensuring False to True change on dequeueable.
        # Now min_batch_size must be > 0 if not None.
        if self._min_batch_size is not None and self._min_batch_size == 0:
            if self._min_batch_items is None:
                raise ValueError(
                    "min_batch_size=0 is only allowed when min_batch_items=1 "
                    "(they're equivalent on non-empty queues). "
                    "Set min_batch_items=1 and omit min_batch_size."
                )
            else:
                if self._min_batch_items == 1:
                    if self._logger is not None:
                        self._logger.warning(
                            "WatermarkBatchingQueue(%s): min_batch_size=0 is redundant with min_batch_items=1; "
                            "treating min_batch_size as None.",
                            self._name,
                        )
                    self._min_batch_size = None
                else:
                    raise ValueError(
                        f"min_batch_size=0 requires min_batch_items=1; got min_batch_items={self._min_batch_items}"
                    )

        self._prof_cfg: Optional[QueueProfilingConfig] = profiling
        self._prof: Optional[_QueueProfile] = None
        if profiling is not None and (profiling.enable_stats or profiling.enable_timing):
            self._prof = _QueueProfile()
            if profiling.enable_timing and not profiling.enable_stats:
                assert self._prof_cfg is not None
                # NOTE: Following code collects stats call numbers when _prof is not None based on this.
                self._prof_cfg = replace(self._prof_cfg, enable_stats=True)
                if self._logger is not None:
                    self._logger.warning("profiling configuration enable_stats is set to True when enable_timing is True")

        self._lock = Lock()
        self._cv = Condition(self._lock)

        self._q: Deque[_Entry[ItemT, SizeT]] = deque()
        self._buffered_items: int = 0
        self._buffered_size: Any = 0  # ACTUAL total; relies on 0-compatibility
        self._closed: bool = False

    # ---------------------------
    # Lifecycle / stats
    # ---------------------------

    @property
    def name(self) -> str:
        return self._name

    def close(self) -> None:
        """Close queue. Enqueue will fail; Dequeue can drain remaining items."""
        with self._cv:
            self._closed = True
            if self._prof is not None:
                self._prof.close_calls += 1
                self._prof.notify_all_calls += 1
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def __len__(self) -> int:
        with self._lock:
            return self._buffered_items

    def buffered_size(self) -> SizeT:
        """Total ACTUAL size currently buffered (sum of item.size())."""
        with self._lock:
            return self._buffered_size

    # ---------------------------
    # Profiling API
    # ---------------------------

    def profiling(self) -> dict[str, Any]:
        """
        Return a snapshot of queue profiling. Returns {} if profiling disabled.
        Includes counts, totals, and average times (if timing enabled).
        """
        with self._lock:
            # Capture local references under lock for safety and cleaner code
            prof = self._prof
            cfg = self._prof_cfg

            if prof is None or cfg is None:
                return {}

            # 1. Base Counts (Always available if stats enabled)
            out: dict[str, Any] = {
                "name": self._name,
                "counts": {
                    "enqueue_calls": prof.enqueue_calls,
                    "enqueue_full_errors": prof.enqueue_full_errors,
                    "enqueue_closed_errors": prof.enqueue_closed_errors,
                    "dequeue_calls": prof.dequeue_calls,
                    "dequeue_empty_errors": prof.dequeue_empty_errors,
                    "dequeued_batches": prof.dequeued_batches,
                    "dequeued_items": prof.dequeued_items,
                    "dequeue_returned_empty_on_close": prof.dequeue_returned_empty_on_close,
                    "close_calls": prof.close_calls,
                    "notify_all_calls": prof.notify_all_calls,
                    "enqueue_wait_calls": prof.enqueue_wait_calls,
                    "dequeue_wait_calls": prof.dequeue_wait_calls,
                },
            }

            # 2. Timing (Totals & Averages)
            if cfg.enable_timing:
                timing: dict[str, float] = {}
                # NOTE: It reports avg per call, not per succesful call.
                # Enqueue Timing
                if cfg.timing_profile_enqueue:
                    timing["enqueue_total_s"] = prof.enqueue_total_s
                    if prof.enqueue_calls > 0:
                        timing["enqueue_avg_s"] = prof.enqueue_total_s / prof.enqueue_calls

                # Dequeue Timing
                if cfg.timing_profile_dequeue:
                    timing["dequeue_total_s"] = prof.dequeue_total_s
                    if prof.dequeue_calls > 0:
                        timing["dequeue_avg_s"] = prof.dequeue_total_s / prof.dequeue_calls

                # Wait Timing
                if cfg.timing_profile_wait:
                    timing["enqueue_wait_s"] = prof.enqueue_wait_s
                    if prof.enqueue_wait_calls > 0:
                        timing["enqueue_wait_avg_s"] = prof.enqueue_wait_s / prof.enqueue_wait_calls

                    timing["dequeue_wait_s"] = prof.dequeue_wait_s
                    if prof.dequeue_wait_calls > 0:
                        timing["dequeue_wait_avg_s"] = prof.dequeue_wait_s / prof.dequeue_wait_calls

                # Batch Build Timing
                if cfg.timing_profile_batch_build:
                    timing["dequeue_batch_build_s"] = prof.dequeue_batch_build_s
                    if prof.dequeued_batches > 0:
                        timing["dequeue_batch_build_avg_s"] = prof.dequeue_batch_build_s / prof.dequeued_batches

                out["timing_s"] = timing

            return out

    def reset_profiling(self) -> None:
        """Reset queue profiling counters/timers. No-op if profiling disabled."""
        # (Also avoids reading _prof outside lock.)
        with self._lock:
            if self._prof is None:
                return
            self._prof = _QueueProfile()

    # ---------------------------
    # Public API
    # ---------------------------

    def enqueue(self, item: ItemT, *, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Enqueue one item.

        Blocks (if block=True) when high-watermark limits would be exceeded (with "nudge" exception).
        Raises queue.Full on timeout / non-blocking failure.
        Raises ItemTooLargeError if the item exceeds max_batch_size (when configured).
        """
        sz: SizeT = item.size()
        if sz <= 0:  # assumes 0-compatible
            # NOTE: Does not allow 0 sized items for now to avoid deadlock.
            raise ValueError("item.size() must be > 0")

        # Hard per-item cap needed to guarantee we can always emit at least one item per batch.
        if self._max_batch_size is not None and sz > self._max_batch_size:
            raise ItemTooLargeError("item.size() exceeds max_batch_size")

        deadline: Optional[float] = None
        if timeout is not None:
            if timeout < 0:
                raise ValueError("timeout must be >= 0 or None")
            deadline = time.monotonic() + timeout

        with self._cv:
            # read profiling pointers under lock to avoid races with reset_profiling().

            t0 = None
            do_time_total = bool(self._prof and self._prof_cfg and self._prof_cfg.enable_timing and self._prof_cfg.timing_profile_enqueue)
            do_time_wait = bool(self._prof and self._prof_cfg and self._prof_cfg.enable_timing and self._prof_cfg.timing_profile_wait)
            if do_time_total:
                t0 = time.perf_counter()

            if self._prof is not None:
                self._prof.enqueue_calls += 1

            try:
                if self._closed:
                    if self._prof is not None:
                        self._prof.enqueue_closed_errors += 1
                    raise QueueClosedError("Queue is closed")

                while not self._can_enqueue(sz):
                    if self._closed:
                        if self._prof is not None:
                            self._prof.enqueue_closed_errors += 1
                        raise QueueClosedError("Queue is closed")
                    if not block:
                        if self._prof is not None:
                            self._prof.enqueue_full_errors += 1
                        raise _queue.Full

                    if deadline is None:
                        if do_time_wait:
                            w0 = time.perf_counter()
                            self._cv.wait()
                            self._prof.enqueue_wait_calls += 1
                            self._prof.enqueue_wait_s += (time.perf_counter() - w0)
                        else:
                            if self._prof is not None:
                                self._prof.enqueue_wait_calls += 1
                            self._cv.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            if self._prof is not None:
                                self._prof.enqueue_full_errors += 1
                            raise _queue.Full
                        if do_time_wait:
                            w0 = time.perf_counter()
                            self._cv.wait(remaining)
                            self._prof.enqueue_wait_calls += 1
                            self._prof.enqueue_wait_s += (time.perf_counter() - w0)
                        else:
                            if self._prof is not None:
                                self._prof.enqueue_wait_calls += 1
                            self._cv.wait(remaining)

                # close() may have happened while we were waiting, and _can_enqueue() may now be True.
                # Re-check closed after the wait-loop before actually enqueuing.
                if self._closed:
                    if self._prof is not None:
                        self._prof.enqueue_closed_errors += 1
                    raise QueueClosedError("Queue is closed")

                self._do_enqueue(item=item, sz=sz)
                if self._prof is not None:
                    self._prof.notify_all_calls += 1
                self._cv.notify_all()

            finally:
                if do_time_total and t0 is not None and self._prof is not None:
                    self._prof.enqueue_total_s += (time.perf_counter() - t0)

    def dequeue_batch(self, *, block: bool = True, timeout: Optional[float] = None) -> List[ItemT]:
        """
        Dequeue one batch.

        Blocks (if block=True) until:
          - dequeue is allowed by (flush triggers OR high-watermark reached), and
          - queue is non-empty (unless closed, in which case returns []).

        Raises queue.Empty if user timeout is reached.
        Returns [] if closed and empty.
        """
        # NOTE: Empty list is a signal for closed.
        deadline: Optional[float] = None
        if timeout is not None:
            if timeout < 0:
                raise ValueError("timeout must be >= 0 or None")
            deadline = time.monotonic() + timeout

        with self._cv:
            # read profiling pointers under lock to avoid races with reset_profiling().

            t0 = None
            do_time_total = bool(self._prof and self._prof_cfg and self._prof_cfg.enable_timing and self._prof_cfg.timing_profile_dequeue)
            do_time_wait = bool(self._prof and self._prof_cfg and self._prof_cfg.enable_timing and self._prof_cfg.timing_profile_wait)
            do_time_build = bool(self._prof and self._prof_cfg and self._prof_cfg.enable_timing and self._prof_cfg.timing_profile_batch_build)

            if do_time_total:
                t0 = time.perf_counter()
            if self._prof is not None:
                self._prof.dequeue_calls += 1

            try:
                while True:
                    if self._q:
                        now = time.monotonic()
                        if self._dequeue_allowed(now, cnt=self._buffered_items, size=self._buffered_size):
                            break

                        if not block:
                            if self._prof is not None:
                                self._prof.dequeue_empty_errors += 1
                            # NOTE: Here Empty means cannot get anything right now.
                            raise _queue.Empty

                        wait_for = self._compute_wait_for_dequeue(now=now, deadline=deadline)

                        # ----------------------------
                        # If wait_for <= 0 due to max_linger_s expiring, we must NOT raise queue.Empty.
                        # We only raise queue.Empty when the user-provided deadline is exceeded.
                        # ----------------------------
                        if wait_for is not None and wait_for <= 0:
                            if deadline is not None and now >= deadline:
                                if self._prof is not None:
                                    self._prof.dequeue_empty_errors += 1
                                raise _queue.Empty
                            # Linger (or other internal trigger) matured: loop to re-check _dequeue_allowed().
                            continue

                        if do_time_wait:
                            w0 = time.perf_counter()
                            self._cv.wait(wait_for)
                            self._prof.dequeue_wait_calls += 1
                            self._prof.dequeue_wait_s += (time.perf_counter() - w0)
                        else:
                            if self._prof is not None:
                                self._prof.dequeue_wait_calls += 1
                            self._cv.wait(wait_for)
                        continue

                    # empty
                    if self._closed:
                        if self._prof is not None:
                            self._prof.dequeue_returned_empty_on_close += 1
                        return []
                    if not block:
                        if self._prof is not None:
                            self._prof.dequeue_empty_errors += 1
                        raise _queue.Empty

                    now = time.monotonic()
                    wait_for = self._compute_wait_for_dequeue(now=now, deadline=deadline)
                    if wait_for is not None and wait_for <= 0:
                        if self._prof is not None:
                            self._prof.dequeue_empty_errors += 1
                        raise _queue.Empty

                    if do_time_wait:
                        w0 = time.perf_counter()
                        self._cv.wait(wait_for)
                        self._prof.dequeue_wait_calls += 1
                        self._prof.dequeue_wait_s += (time.perf_counter() - w0)
                    else:
                        if self._prof is not None:
                            self._prof.dequeue_wait_calls += 1
                        self._cv.wait(wait_for)

                # Build a batch (capped by max_batch_items/max_batch_size)
                b0 = time.perf_counter() if do_time_build else None

                batch: List[ItemT] = []
                batch_size: Any = 0  # relies on 0-compatibility

                while self._q:
                    if self._max_batch_items is not None and len(batch) >= self._max_batch_items:
                        break

                    head = self._q[0]

                    if self._max_batch_size is not None:
                        max_batch_size = self._max_batch_size

                        # Normally item.size <= max_batch_size (enforced in enqueue), but keep defensive.
                        if not batch and head.size > max_batch_size:
                            entry = self._q.popleft()
                            self._do_dequeue(entry)
                            batch.append(entry.item)
                            break

                        if (batch_size + head.size) > max_batch_size:
                            break

                    entry = self._q.popleft()
                    self._do_dequeue(entry)
                    batch.append(entry.item)
                    batch_size = batch_size + entry.size

                if do_time_build and b0 is not None and self._prof is not None:
                    self._prof.dequeue_batch_build_s += (time.perf_counter() - b0)

                if self._prof is not None:
                    self._prof.dequeued_batches += 1
                    self._prof.dequeued_items += len(batch)
                    self._prof.notify_all_calls += 1
                self._cv.notify_all()
                return batch

            finally:
                if do_time_total and t0 is not None and self._prof is not None:
                    self._prof.dequeue_total_s += (time.perf_counter() - t0)

    # ---------------------------
    # Internals
    # ---------------------------

    def _do_enqueue(self, *, item: ItemT, sz: SizeT) -> None:
        self._q.append(_Entry(item=item, size=sz, enqueued_at=time.monotonic()))
        self._buffered_items += 1
        self._buffered_size = self._buffered_size + sz

    def _do_dequeue(self, entry: _Entry[ItemT, SizeT]) -> None:
        self._buffered_items -= 1
        self._buffered_size = self._buffered_size - entry.size

    def _high_watermark_reached(self, *, cnt: int, size: SizeT) -> bool:
        if self._high_watermark_items is not None and cnt >= self._high_watermark_items:
            return True
        if self._high_watermark_size is not None and size >= self._high_watermark_size:
            return True
        return False

    def _flush_triggers_met(
        self,
        now: float,
        *,
        cnt: int,
        size: SizeT,
        oldest_enqueued_at: Optional[float],
    ) -> bool:
        if self._min_batch_items is not None and cnt >= self._min_batch_items:
            return True
        if self._min_batch_size is not None and size >= self._min_batch_size:
            return True
        if self._max_linger_s is not None and oldest_enqueued_at is not None:
            if (now - oldest_enqueued_at) >= self._max_linger_s:
                return True
        return False

    def _dequeue_allowed(self, now: float, *, cnt: int, size: SizeT) -> bool:
        if self._closed:
            return True

        oldest_enqueued_at = self._q[0].enqueued_at if self._q else None

        if self._flush_triggers_met(
            now,
            cnt=cnt,
            size=size,
            oldest_enqueued_at=oldest_enqueued_at,
        ):
            return True

        if self._high_watermark_reached(cnt=cnt, size=size):
            return True

        return False

    def _can_enqueue(self, sz: SizeT) -> bool:
        """
        Backpressure policy with "nudge":

          - Normally: refuse if (cnt+1 > high_watermark_items) or (size+sz > high_watermark_size)
          - Nudge: allow overflow iff it newly enables dequeue.
        """
        if self._high_watermark_items is None and self._high_watermark_size is None:
            return True

        exceed_cnt = False
        exceed_size = False

        if self._high_watermark_items is not None and (self._buffered_items + 1) > self._high_watermark_items:
            exceed_cnt = True
        if self._high_watermark_size is not None and (self._buffered_size + sz) > self._high_watermark_size:
            exceed_size = True

        if not exceed_cnt and not exceed_size:
            return True

        now = time.monotonic()

        allowed_before = self._dequeue_allowed(now, cnt=self._buffered_items, size=self._buffered_size)

        after_cnt = self._buffered_items + 1
        after_size = self._buffered_size + sz

        oldest_after = self._q[0].enqueued_at if self._q else now

        triggers_after = self._flush_triggers_met(
            now,
            cnt=after_cnt,
            size=after_size,
            oldest_enqueued_at=oldest_after,
        )
        max_after = self._high_watermark_reached(cnt=after_cnt, size=after_size)
        allowed_after = self._closed or triggers_after or max_after

        if (not allowed_before) and allowed_after:
            return True

        return False

    def _compute_wait_for_dequeue(self, *, now: float, deadline: Optional[float]) -> Optional[float]:
        """
        Returns how long Condition.wait() should sleep before re-checking, considering:
          - user timeout (deadline)
          - max_linger_s (oldest item reaching age threshold), if enabled
        If neither applies, returns None to wait indefinitely until notify().
        """
        waits: List[float] = []

        if deadline is not None:
            waits.append(deadline - now)

        if self._max_linger_s is not None and self._q:
            oldest = self._q[0].enqueued_at
            waits.append((oldest + self._max_linger_s) - now)

        if not waits:
            return None

        w = min(waits)
        return 0.0 if w < 0 else w
