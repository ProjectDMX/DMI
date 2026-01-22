# engine.py
from __future__ import annotations

import logging
import queue as _queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, List, Optional, ParamSpec, Protocol, Sequence, Tuple

try:
    from .batching_queue import (
        SizedWorkItem,
        WatermarkBatchingQueue,
        QueueProfilingConfig,
        ItemTooLargeError,
        QueueClosedError,
    )
except ImportError:
    from batching_queue import (
        SizedWorkItem,
        WatermarkBatchingQueue,
        QueueProfilingConfig,
        ItemTooLargeError,
        QueueClosedError,
    )


# -----------------------------
# Public config types
# -----------------------------

StageProcessFn = Callable[[Sequence[SizedWorkItem]], List[SizedWorkItem]]
StageThreadInitFn = Callable[[int, Any], None]    # thread_init(thread_idx, thread_config)
StageThreadCleanupFn = Callable[[], None]    # thread_cleanup()

P = ParamSpec("P")


class InputHandlerFn(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> List[SizedWorkItem]: ...


class ThreadExceptionPolicy(str, Enum):
    STOP_ENGINE = "stop_engine"
    CONTINUE = "continue"


class OnFullPolicy(str, Enum):
    RAISE = "raise"
    DROP = "drop"
    RETRY = "retry"
    ABORT = "abort"


class OnClosedPolicy(str, Enum):
    RAISE = "raise"
    DROP = "drop"


# TODO: CHeck this policy.
@dataclass(frozen=True, slots=True)
class EnqueuePolicy:
    block: bool = True
    timeout_s: Optional[float] = None

    on_full: OnFullPolicy = OnFullPolicy.RAISE
    max_retries: int = 0
    retry_backoff_s: float = 0.001

    on_closed: OnClosedPolicy = OnClosedPolicy.RAISE
    drop_if_stopping: bool = True


@dataclass(frozen=True, slots=True)
class QueueConfig:
    min_batch_items: Optional[int] = 1
    min_batch_size: Optional[Any] = None
    max_linger_s: Optional[float] = None

    max_batch_items: Optional[int] = None
    max_batch_size: Optional[Any] = None

    high_watermark_items: Optional[int] = None
    high_watermark_size: Optional[Any] = None

    def build_queue(
        self,
        *,
        name: str = "",
        logger: Optional[logging.Logger] = None,
        profiling: Optional[QueueProfilingConfig] = None,
    ) -> WatermarkBatchingQueue:
        return WatermarkBatchingQueue(
            min_batch_items=self.min_batch_items,
            min_batch_size=self.min_batch_size,
            max_linger_s=self.max_linger_s,
            max_batch_items=self.max_batch_items,
            max_batch_size=self.max_batch_size,
            high_watermark_items=self.high_watermark_items,
            high_watermark_size=self.high_watermark_size,
            name=name,
            logger=logger,
            profiling=profiling,
        )


@dataclass(frozen=True, slots=True)
class StageConfig:
    name: str
    parallelism: int
    process_fn: StageProcessFn
    thread_init_config: Any

    # Per-worker-thread hooks (called inside the worker thread):
    # - thread_init(thread_idx, thread_config): can record worker_id, initialize per-thread resources, etc.
    # - thread_cleanup(): cleanup per-thread resources (TLS/thread_local makes thread_idx unnecessary here)
    thread_init: Optional[StageThreadInitFn] = None
    thread_cleanup: Optional[StageThreadCleanupFn] = None

    input_queue: QueueConfig = field(default_factory=QueueConfig)
    ingress_policy: EnqueuePolicy = field(default_factory=EnqueuePolicy)

    thread_name_prefix: Optional[str] = None
    daemon_threads: Optional[bool] = None


@dataclass(frozen=True, slots=True)
class EngineConfig:
    name: str = "pipelined-engine"
    daemon_threads: bool = True
    exception_policy: ThreadExceptionPolicy = ThreadExceptionPolicy.STOP_ENGINE

    close_queues_on_abort: bool = True
    suppress_closed_queue_errors_during_shutdown: bool = True
    worker_dequeue_timeout_s: Optional[float] = None

    # Engine metrics
    enable_stats: bool = False
    enable_timing: bool = False
    timing_profile_input_handler: bool = True
    timing_profile_dequeue: bool = True
    timing_profile_process: bool = True
    timing_profile_enqueue: bool = True
    timing_profile_output: bool = True

    # Queue profiling config
    # If None, inherit from engine enable_stats/enable_timing.
    enable_queue_stats: Optional[bool] = None
    enable_queue_timing: Optional[bool] = None

    queue_timing_profile_enqueue: bool = True
    queue_timing_profile_dequeue: bool = True
    queue_timing_profile_wait: bool = True
    queue_timing_profile_batch_build: bool = True


@dataclass(frozen=True, slots=True)
class ThreadFailure:
    stage: str
    thread_name: str
    where: str
    exc_repr: str
    traceback: str


OutputHandlerFn = Callable[[List[SizedWorkItem]], None]


# -----------------------------
# Internal metrics containers
# -----------------------------

@dataclass(slots=True)
class _IngestStats:
    submits: int = 0
    items_returned: int = 0

    # call-counts for timing averages (collected when engine timing is enabled)
    input_handler_calls: int = 0
    enqueue_calls: int = 0

    input_handler_s: float = 0.0
    enqueue_s: float = 0.0


@dataclass(slots=True)
class _QueueStats:
    enqueued: int = 0
    dropped: int = 0
    full_errors: int = 0
    closed_errors: int = 0
    too_large_errors: int = 0
    retries: int = 0


@dataclass(slots=True)
class _StageStats:
    batches: int = 0
    items_in: int = 0
    items_out: int = 0

    # call-counts for timing averages (collected when stage timing is enabled)
    dequeue_calls: int = 0
    process_calls: int = 0
    enqueue_calls: int = 0

    dequeue_s: float = 0.0
    dequeue_idle_s: float = 0.0
    dequeue_timeouts: int = 0  # also the denominator for dequeue_idle_avg_s

    process_s: float = 0.0
    enqueue_s: float = 0.0
    output_s: float = 0.0
    output_calls: int = 0  # denominator for output_avg_s
    output_items: int = 0


# -----------------------------
# Engine implementation
# -----------------------------

class PipelinedEngine(Generic[P]):
    def __init__(
        self,
        stages: Sequence[StageConfig],
        *,
        input_handler: InputHandlerFn[P],
        config: Optional[EngineConfig] = None,
        logger: Optional[logging.Logger] = None,
        output_handler: Optional[OutputHandlerFn] = None,
    ) -> None:
        if not stages:
            raise ValueError("stages must be non-empty")

        self._stages: List[StageConfig] = list(stages)
        self._input_handler = input_handler
        self._cfg: EngineConfig = config or EngineConfig()
        self._log: logging.Logger = logger or logging.getLogger(__name__)
        self._output_handler: Optional[OutputHandlerFn] = output_handler

        # Validate config footguns
        if self._cfg.worker_dequeue_timeout_s is not None and self._cfg.worker_dequeue_timeout_s <= 0:
            raise ValueError("worker_dequeue_timeout_s must be > 0 or None")
        if (not self._cfg.close_queues_on_abort) and (self._cfg.worker_dequeue_timeout_s is None):
            raise ValueError(
                "close_queues_on_abort=False requires worker_dequeue_timeout_s, "
                "otherwise workers can block forever in dequeue_batch()."
            )

        # Queue profiling config derived from EngineConfig
        q_stats = self._cfg.enable_queue_stats
        q_timing = self._cfg.enable_queue_timing
        if q_stats is None:
            q_stats = self._cfg.enable_stats
        if q_timing is None:
            q_timing = self._cfg.enable_timing

        # Keep engine behavior consistent with WatermarkBatchingQueue:
        # timing requires call counts, and the queue will auto-enable stats internally.
        if q_timing and not q_stats:
            q_stats = True
            self._log.warning(
                "EngineConfig: enable_queue_stats is set to True because enable_queue_timing is True "
                "(queue timing requires call counts)."
            )

        self._queue_prof_cfg: Optional[QueueProfilingConfig] = None
        if q_stats or q_timing:
            self._queue_prof_cfg = QueueProfilingConfig(
                enable_stats=bool(q_stats),
                enable_timing=bool(q_timing),
                timing_profile_enqueue=self._cfg.queue_timing_profile_enqueue,
                timing_profile_dequeue=self._cfg.queue_timing_profile_dequeue,
                timing_profile_wait=self._cfg.queue_timing_profile_wait,
                timing_profile_batch_build=self._cfg.queue_timing_profile_batch_build,
            )

        # Validate stages
        seen = set()
        for i, s in enumerate(self._stages):
            if not s.name:
                raise ValueError(f"Stage at index {i} must have a non-empty name")
            if s.name in seen:
                raise ValueError(f"Duplicate stage name: {s.name!r}")
            seen.add(s.name)
            if s.parallelism <= 0:
                raise ValueError(f"Stage {s.name!r} parallelism must be > 0")
            if s.ingress_policy.max_retries < 0:
                raise ValueError(f"Stage {s.name!r} ingress_policy.max_retries must be >= 0")
            if s.ingress_policy.timeout_s is not None and s.ingress_policy.timeout_s < 0:
                raise ValueError(f"Stage {s.name!r} ingress_policy.timeout_s must be >= 0 or None")
            if s.ingress_policy.retry_backoff_s < 0:
                raise ValueError(f"Stage {s.name!r} ingress_policy.retry_backoff_s must be >= 0")

            # Validate queue config early so errors are stage-attributed (mirrors WatermarkBatchingQueue checks).
            qc = s.input_queue
            if qc.min_batch_items is None and qc.min_batch_size is None and qc.max_linger_s is None:
                raise ValueError(
                    f"Stage {s.name!r} input_queue must set at least one of "
                    "min_batch_items/min_batch_size/max_linger_s"
                )
            if qc.min_batch_items is not None and qc.min_batch_items <= 0:
                raise ValueError(f"Stage {s.name!r} input_queue.min_batch_items must be > 0 or None")
            if qc.min_batch_size is not None and qc.min_batch_size < 0:
                raise ValueError(f"Stage {s.name!r} input_queue.min_batch_size must be >= 0 or None")
            if qc.min_batch_size is not None and qc.min_batch_size == 0:
                if qc.min_batch_items is None:
                    raise ValueError(
                        f"Stage {s.name!r} input_queue.min_batch_size=0 requires input_queue.min_batch_items=1"
                    )
                if qc.min_batch_items != 1:
                    raise ValueError(
                        f"Stage {s.name!r} input_queue.min_batch_size=0 requires input_queue.min_batch_items=1; "
                        f"got input_queue.min_batch_items={qc.min_batch_items}"
                    )
            if qc.max_linger_s is not None and qc.max_linger_s < 0:
                raise ValueError(f"Stage {s.name!r} input_queue.max_linger_s must be >= 0 or None")
            if qc.max_batch_items is not None and qc.max_batch_items <= 0:
                raise ValueError(f"Stage {s.name!r} input_queue.max_batch_items must be > 0 or None")
            if qc.max_batch_size is not None and qc.max_batch_size <= 0:
                raise ValueError(f"Stage {s.name!r} input_queue.max_batch_size must be > 0 or None")
            if qc.high_watermark_items is not None and qc.high_watermark_items <= 0:
                raise ValueError(f"Stage {s.name!r} input_queue.high_watermark_items must be > 0 or None")
            if qc.high_watermark_size is not None and qc.high_watermark_size <= 0:
                raise ValueError(f"Stage {s.name!r} input_queue.high_watermark_size must be > 0 or None")

        # Abort-mode footgun guard:
        # If we do NOT close queues on abort, producers might block forever in enqueue()
        # unless enqueue has a bounded timeout or is non-blocking.
        if not self._cfg.close_queues_on_abort:
            for s in self._stages:
                pol = s.ingress_policy
                if pol.block and pol.timeout_s is None:
                    raise ValueError(
                        "close_queues_on_abort=False requires every stage.ingress_policy to have "
                        "timeout_s set (or block=False). Otherwise abort/stop can deadlock producers "
                        "blocked in enqueue()."
                    )

        # One input queue per stage (pass profiling config + name)
        self._queues: List[WatermarkBatchingQueue] = [
            s.input_queue.build_queue(
                name=f"{self._cfg.name}.{s.name}.in",
                logger=self._log,
                profiling=self._queue_prof_cfg,
            )
            for s in self._stages
        ]

        self._threads: List[List[threading.Thread]] = [[] for _ in self._stages]

        self._lock = threading.Lock()
        self._started = False
        self._input_closed = False
        self._stop_event = threading.Event()

        self._failures_lock = threading.Lock()
        self._failures: List[ThreadFailure] = []

        # One-time warnings (avoid spamming logs from worker threads)
        self._warn_lock = threading.Lock()
        self._warned_no_output_handler: set[str] = set()

        # Engine metrics
        self._metrics_lock = threading.Lock()
        self._ingest_stats = _IngestStats()
        self._queue_stats_by_stage: dict[str, _QueueStats] = {s.name: _QueueStats() for s in self._stages}
        self._stage_stats_by_stage: dict[str, _StageStats] = {s.name: _StageStats() for s in self._stages}

    # -----------------------------
    # Lifecycle
    # -----------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._stop_event.is_set():
                raise RuntimeError("Engine is stopped/aborted and cannot be restarted")

            for stage_idx, stage in enumerate(self._stages):
                for t_idx in range(stage.parallelism):
                    tname = self._thread_name(stage_idx=stage_idx, thread_idx=t_idx)
                    daemon = stage.daemon_threads if stage.daemon_threads is not None else self._cfg.daemon_threads
                    th = threading.Thread(
                        target=self._stage_worker,
                        name=tname,
                        args=(stage_idx, t_idx, stage.thread_init_config),
                        daemon=daemon,
                    )
                    self._threads[stage_idx].append(th)

            for stage_threads in self._threads:
                for th in stage_threads:
                    th.start()

            self._started = True

    def close_input(self) -> None:
        with self._lock:
            if self._input_closed:
                return
            self._input_closed = True
            self._queues[0].close()

    def stop(self, *, graceful: bool = True, timeout: Optional[float] = None) -> None:
        if graceful:
            end_at: Optional[float] = None
            if timeout is not None:
                if timeout < 0:
                    raise ValueError("timeout must be >= 0 or None")
                end_at = time.monotonic() + timeout

            self.close_input()
            self.join(timeout=timeout)

            # If join timed out, escalate to abort to ensure we wake blocked workers/producers.
            if any(th.is_alive() for stage_threads in self._threads for th in stage_threads):
                remaining: Optional[float] = None
                if end_at is not None:
                    remaining = max(0.0, end_at - time.monotonic())

                alive = [th.name for stage_threads in self._threads for th in stage_threads if th.is_alive()]
                show = ", ".join(alive[:10])
                more = "" if len(alive) <= 10 else f" (+{len(alive) - 10} more)"
                self._log.warning(
                    "Graceful stop did not complete within timeout; escalating to abort. "
                    "alive_threads=%d [%s%s]",
                    len(alive),
                    show,
                    more,
                )
                self.abort(timeout=remaining)
            else:
                # join() sets this on successful completion, but set again for clarity/idempotence.
                self._stop_event.set()
        else:
            self.abort(timeout=timeout)

    def request_abort(self) -> None:
        with self._lock:
            if self._stop_event.is_set():
                self._input_closed = True
                return
            self._stop_event.set()
            self._input_closed = True
            if self._cfg.close_queues_on_abort:
                for q in self._queues:
                    try:
                        q.close()
                    except Exception:
                        pass

    def abort(self, *, timeout: Optional[float] = None) -> None:
        self.request_abort()
        self._join_all_threads(timeout=timeout)

    def join(self, *, timeout: Optional[float] = None) -> None:
        if not self._started:
            return

        end_at: Optional[float] = None
        if timeout is not None:
            if timeout < 0:
                raise ValueError("timeout must be >= 0 or None")
            end_at = time.monotonic() + timeout

        timed_out = False

        for stage_idx in range(len(self._stages)):
            for th in self._threads[stage_idx]:
                if end_at is None:
                    th.join()
                    continue

                remaining = end_at - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break

                th.join(timeout=remaining)
                if th.is_alive():
                    timed_out = True
                    break

            if timed_out:
                break

            # Close next stage's input queue once this stage is fully joined, so the pipeline drains forward.
            if stage_idx + 1 < len(self._queues):
                q_next = self._queues[stage_idx + 1]
                if not q_next.closed:
                    q_next.close()

            if end_at is not None and time.monotonic() >= end_at:
                timed_out = True
                break

        if timed_out:
            # Don't mark stopped here; caller can decide whether to abort.
            alive = [th.name for stage_threads in self._threads for th in stage_threads if th.is_alive()]
            if alive:
                show = ", ".join(alive[:10])
                more = "" if len(alive) <= 10 else f" (+{len(alive) - 10} more)"
                self._log.debug(
                    "join(timeout=%s) did not complete; %d worker thread(s) still alive [%s%s].",
                    timeout,
                    len(alive),
                    show,
                    more,
                )
            return

        self._stop_event.set()

    def __enter__(self) -> "PipelinedEngine[P]":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(graceful=(exc_type is None))

    # -----------------------------
    # Public interface
    # -----------------------------

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("Engine not started. Call start() first.")
            if self._stop_event.is_set():
                raise RuntimeError("Engine is stopping/aborted")
            if self._input_closed:
                raise RuntimeError("Input is closed")

        t0 = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_input_handler) else None
        items = self._input_handler(*args, **kwargs)
        if items is None:
            items = []
        if t0 is not None:
            self._metrics_add_ingest(
                input_handler_s=(time.perf_counter() - t0),
                input_handler_calls=1,
            )

        if self._cfg.enable_stats:
            self._metrics_add_ingest(submits=1, items_returned=len(items))

        t1 = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_enqueue) else None
        try:
            self._enqueue_many(
                target_queue=self._queues[0],
                items=items,
                policy=self._stages[0].ingress_policy,
                target_stage_name=self._stages[0].name,
            )
        except QueueClosedError as e:
            raise RuntimeError("Input is closed") from e
        except RuntimeError as e:
            # Backward compatibility if an older queue implementation still raises RuntimeError("Queue is closed").
            if "Queue is closed" in str(e):
                raise RuntimeError("Input is closed") from e
            raise
        finally:
            if t1 is not None:
                self._metrics_add_ingest(
                    enqueue_s=(time.perf_counter() - t1),
                    enqueue_calls=1,
                )

    @property
    def failures(self) -> Tuple[ThreadFailure, ...]:
        with self._failures_lock:
            return tuple(self._failures)

    def raise_if_failed(self) -> None:
        fs = self.failures
        if not fs:
            return
        msg = "\n".join(f"[{f.stage}::{f.thread_name}::{f.where}] {f.exc_repr}" for f in fs[-10:])
        raise RuntimeError(f"Engine has recorded {len(fs)} worker failure(s). Recent:\n{msg}")

    # -----------------------------
    # Profiling / metrics reporting
    # -----------------------------

    def queue_profiling(self) -> dict[str, Any]:
        """
        Full queue profiling snapshots (counts + timing) for each stage input queue.
        Returns {} if queue profiling is disabled.
        """
        if self._queue_prof_cfg is None:
            return {}
        out: dict[str, Any] = {}
        for stage, q in zip(self._stages, self._queues):
            out[stage.name] = q.profiling()
        return out

    def stats(self) -> dict[str, Any]:
        """
        Engine-level counters + (optionally) queue-internal counters.

        Returns {} if neither engine stats nor queue stats are enabled.
        """
        want_engine = self._cfg.enable_stats
        want_queue = bool(self._queue_prof_cfg and self._queue_prof_cfg.enable_stats)

        if not (want_engine or want_queue):
            return {}

        engine_part: dict[str, Any] = {}
        if want_engine:
            with self._metrics_lock:
                engine_part = {
                    "ingest": {
                        "submits": self._ingest_stats.submits,
                        "items_returned": self._ingest_stats.items_returned,
                    },
                    "queue_by_stage": {
                        k: {
                            "enqueued": v.enqueued,
                            "dropped": v.dropped,
                            "full_errors": v.full_errors,
                            "closed_errors": v.closed_errors,
                            "too_large_errors": v.too_large_errors,
                            "retries": v.retries,
                        }
                        for k, v in self._queue_stats_by_stage.items()
                    },
                    "stage_by_stage": {
                        k: {
                            "batches": v.batches,
                            "items_in": v.items_in,
                            "items_out": v.items_out,
                            "output_calls": v.output_calls,
                            "output_items": v.output_items,
                            "dequeue_timeouts": v.dequeue_timeouts,
                        }
                        for k, v in self._stage_stats_by_stage.items()
                    },
                }

        if want_queue:
            q_counts: dict[str, Any] = {}
            for stage, q in zip(self._stages, self._queues):
                snap = q.profiling()
                if snap:
                    q_counts[stage.name] = snap.get("counts", {})
            engine_part["queue_internal_counts_by_stage"] = q_counts

        return engine_part

    def timings(self) -> dict[str, Any]:
        """
        Engine-level timing totals + averages (if timing enabled) + (optionally) queue-internal timing.

        Returns {} if neither engine timing nor queue timing are enabled.
        """
        want_engine = self._cfg.enable_timing
        want_queue = bool(self._queue_prof_cfg and self._queue_prof_cfg.enable_timing)

        if not (want_engine or want_queue):
            return {}

        engine_part: dict[str, Any] = {}

        if want_engine:
            with self._metrics_lock:
                ing = {
                    "input_handler_s": self._ingest_stats.input_handler_s,
                    "enqueue_s": self._ingest_stats.enqueue_s,
                }
                if self._ingest_stats.input_handler_calls > 0:
                    ing["input_handler_avg_s"] = self._ingest_stats.input_handler_s / self._ingest_stats.input_handler_calls
                if self._ingest_stats.enqueue_calls > 0:
                    ing["enqueue_avg_s"] = self._ingest_stats.enqueue_s / self._ingest_stats.enqueue_calls

                stage_out: dict[str, Any] = {}
                for k, v in self._stage_stats_by_stage.items():
                    sdict: dict[str, Any] = {
                        "dequeue_s": v.dequeue_s,
                        "dequeue_idle_s": v.dequeue_idle_s,
                        "process_s": v.process_s,
                        "enqueue_s": v.enqueue_s,
                        "output_s": v.output_s,
                    }

                    if v.dequeue_calls > 0:
                        sdict["dequeue_avg_s"] = v.dequeue_s / v.dequeue_calls
                    if v.dequeue_timeouts > 0:
                        sdict["dequeue_idle_avg_s"] = v.dequeue_idle_s / v.dequeue_timeouts
                    if v.process_calls > 0:
                        sdict["process_avg_s"] = v.process_s / v.process_calls
                    if v.enqueue_calls > 0:
                        sdict["enqueue_avg_s"] = v.enqueue_s / v.enqueue_calls
                    if v.output_calls > 0:
                        sdict["output_avg_s"] = v.output_s / v.output_calls

                    stage_out[k] = sdict

                engine_part = {
                    "ingest": ing,
                    "stage_by_stage": stage_out,
                }

        if want_queue:
            q_timing: dict[str, Any] = {}
            for stage, q in zip(self._stages, self._queues):
                snap = q.profiling()
                if snap:
                    q_timing[stage.name] = snap.get("timing_s", {})
            engine_part["queue_internal_timing_by_stage"] = q_timing

        return engine_part

    def profiling(self) -> dict[str, Any]:
        """
        Convenience: return everything (engine stats, engine timings, and full queue profiling snapshots).
        """
        return {
            "engine_stats": self.stats(),
            "engine_timings": self.timings(),
            "queues": self.queue_profiling(),
        }

    def reset_metrics(self) -> None:
        """Reset engine metrics + queue profiling (if enabled)."""
        with self._metrics_lock:
            self._ingest_stats = _IngestStats()
            self._queue_stats_by_stage = {s.name: _QueueStats() for s in self._stages}
            self._stage_stats_by_stage = {s.name: _StageStats() for s in self._stages}

        # reset queues outside engine metrics lock (avoid lock-order inversion)
        if self._queue_prof_cfg is not None:
            for q in self._queues:
                q.reset_profiling()

    # -----------------------------
    # Internals
    # -----------------------------

    def _thread_name(self, *, stage_idx: int, thread_idx: int) -> str:
        stage = self._stages[stage_idx]
        prefix = stage.thread_name_prefix or f"{self._cfg.name}.{stage.name}"
        return f"{prefix}.t{thread_idx}"

    def _record_failure(self, failure: ThreadFailure) -> None:
        with self._failures_lock:
            self._failures.append(failure)

    def _maybe_abort_on_failure(self) -> None:
        if self._cfg.exception_policy == ThreadExceptionPolicy.STOP_ENGINE:
            self.request_abort()

    # ---- engine metric helpers ----

    def _metrics_add_ingest(
        self,
        *,
        submits: int = 0,
        items_returned: int = 0,
        input_handler_s: float = 0.0,
        enqueue_s: float = 0.0,
        input_handler_calls: int = 0,
        enqueue_calls: int = 0,
    ) -> None:
        if not (self._cfg.enable_stats or self._cfg.enable_timing):
            return
        with self._metrics_lock:
            self._ingest_stats.submits += submits
            self._ingest_stats.items_returned += items_returned
            self._ingest_stats.input_handler_s += input_handler_s
            self._ingest_stats.enqueue_s += enqueue_s
            self._ingest_stats.input_handler_calls += input_handler_calls
            self._ingest_stats.enqueue_calls += enqueue_calls

    def _metrics_queue_add(
        self,
        stage_name: str,
        *,
        enqueued: int = 0,
        dropped: int = 0,
        full_errors: int = 0,
        closed_errors: int = 0,
        too_large_errors: int = 0,
        retries: int = 0,
    ) -> None:
        if not self._cfg.enable_stats:
            return
        with self._metrics_lock:
            qs = self._queue_stats_by_stage[stage_name]
            qs.enqueued += enqueued
            qs.dropped += dropped
            qs.full_errors += full_errors
            qs.closed_errors += closed_errors
            qs.too_large_errors += too_large_errors
            qs.retries += retries

    def _metrics_stage_add(
        self,
        stage_name: str,
        *,
        batches: int = 0,
        items_in: int = 0,
        items_out: int = 0,
        dequeue_s: float = 0.0,
        dequeue_idle_s: float = 0.0,
        dequeue_timeouts: int = 0,
        process_s: float = 0.0,
        enqueue_s: float = 0.0,
        output_s: float = 0.0,
        output_calls: int = 0,
        output_items: int = 0,
        dequeue_calls: int = 0,
        process_calls: int = 0,
        enqueue_calls: int = 0,
    ) -> None:
        if not (self._cfg.enable_stats or self._cfg.enable_timing):
            return
        with self._metrics_lock:
            ss = self._stage_stats_by_stage[stage_name]
            ss.batches += batches
            ss.items_in += items_in
            ss.items_out += items_out
            ss.dequeue_s += dequeue_s
            ss.dequeue_idle_s += dequeue_idle_s
            ss.dequeue_timeouts += dequeue_timeouts
            ss.process_s += process_s
            ss.enqueue_s += enqueue_s
            ss.output_s += output_s
            ss.output_calls += output_calls
            ss.output_items += output_items
            ss.dequeue_calls += dequeue_calls
            ss.process_calls += process_calls
            ss.enqueue_calls += enqueue_calls

    # ---- enqueue helpers ----

    def _enqueue_many(
        self,
        *,
        target_queue: WatermarkBatchingQueue,
        items: List[SizedWorkItem],
        policy: EnqueuePolicy,
        target_stage_name: str,
    ) -> None:
        if not items:
            return
        if policy.drop_if_stopping and self._stop_event.is_set():
            self._metrics_queue_add(target_stage_name, dropped=len(items))
            return
        for item in items:
            self._enqueue_one(
                target_queue=target_queue,
                item=item,
                policy=policy,
                target_stage_name=target_stage_name,
            )

    def _enqueue_one(
        self,
        *,
        target_queue: WatermarkBatchingQueue,
        item: SizedWorkItem,
        policy: EnqueuePolicy,
        target_stage_name: str,
    ) -> None:
        if policy.drop_if_stopping and self._stop_event.is_set():
            self._metrics_queue_add(target_stage_name, dropped=1)
            return

        attempts = 0
        while True:
            try:
                target_queue.enqueue(item, block=policy.block, timeout=policy.timeout_s)
                self._metrics_queue_add(target_stage_name, enqueued=1)
                return

            except ItemTooLargeError:
                # Not recoverable via retries; honor DROP/ABORT if configured, otherwise raise.
                self._metrics_queue_add(target_stage_name, too_large_errors=1)
                if policy.on_full == OnFullPolicy.DROP:
                    self._metrics_queue_add(target_stage_name, dropped=1)
                    return
                if policy.on_full == OnFullPolicy.ABORT:
                    self.request_abort()
                raise

            except _queue.Full:
                self._metrics_queue_add(target_stage_name, full_errors=1)

                if policy.on_full == OnFullPolicy.DROP:
                    self._metrics_queue_add(target_stage_name, dropped=1)
                    return
                if policy.on_full == OnFullPolicy.ABORT:
                    self.request_abort()
                    raise
                if policy.on_full == OnFullPolicy.RETRY:
                    if attempts >= policy.max_retries:
                        raise
                    attempts += 1
                    self._metrics_queue_add(target_stage_name, retries=1)
                    if policy.retry_backoff_s > 0:
                        time.sleep(policy.retry_backoff_s)
                    if policy.drop_if_stopping and self._stop_event.is_set():
                        self._metrics_queue_add(target_stage_name, dropped=1)
                        return
                    continue
                raise

            except QueueClosedError:
                self._metrics_queue_add(target_stage_name, closed_errors=1)
                if (
                    policy.on_closed == OnClosedPolicy.DROP
                    or (self._cfg.suppress_closed_queue_errors_during_shutdown and self._stop_event.is_set())
                ):
                    self._metrics_queue_add(target_stage_name, dropped=1)
                    return
                raise

            except RuntimeError as e:
                # Backward compatibility for older queue implementation(s)
                if "Queue is closed" in str(e):
                    self._metrics_queue_add(target_stage_name, closed_errors=1)
                    if (
                        policy.on_closed == OnClosedPolicy.DROP
                        or (
                            self._cfg.suppress_closed_queue_errors_during_shutdown
                            and self._stop_event.is_set()
                        )
                    ):
                        self._metrics_queue_add(target_stage_name, dropped=1)
                        return
                raise

    # ---- worker loop ----

    def _warn_no_output_handler(self, stage_name: str, n_items: int) -> None:
        # Warn only once per stage to avoid log spam.
        first = False
        with self._warn_lock:
            if stage_name not in self._warned_no_output_handler:
                self._warned_no_output_handler.add(stage_name)
                first = True
        if first:
            self._log.warning(
                "Stage %s produced %d output item(s) but no output_handler is set; dropping outputs.",
                stage_name,
                n_items,
            )

    def _stage_worker(self, stage_idx: int, thread_idx: int, thread_config: Any) -> None:
        stage = self._stages[stage_idx]
        stage_name = stage.name

        try:
            # Per-thread initialization hook (runs in the worker thread).
            if stage.thread_init is not None:
                stage.thread_init(thread_idx, thread_config)

            in_q = self._queues[stage_idx]

            is_last = (stage_idx == (len(self._stages) - 1))
            next_q: Optional[WatermarkBatchingQueue] = None
            next_stage_name: Optional[str] = None
            next_policy: Optional[EnqueuePolicy] = None

            if not is_last:
                next_q = self._queues[stage_idx + 1]
                next_stage = self._stages[stage_idx + 1]
                next_stage_name = next_stage.name
                next_policy = next_stage.ingress_policy

            while True:
                if self._stop_event.is_set():
                    return

                t_deq = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_dequeue) else None
                try:
                    batch = in_q.dequeue_batch(block=True, timeout=self._cfg.worker_dequeue_timeout_s)
                except _queue.Empty:
                    if t_deq is not None:
                        self._metrics_stage_add(
                            stage_name,
                            dequeue_idle_s=(time.perf_counter() - t_deq),
                            dequeue_timeouts=1,
                        )
                    elif self._cfg.enable_stats:
                        self._metrics_stage_add(stage_name, dequeue_timeouts=1)
                    continue
                except Exception as e:
                    tb = traceback.format_exc()
                    self._record_failure(
                        ThreadFailure(
                            stage=stage_name,
                            thread_name=threading.current_thread().name,
                            where="dequeue_batch",
                            exc_repr=repr(e),
                            traceback=tb,
                        )
                    )
                    self._log.exception("Worker failure in dequeue_batch (stage=%s)", stage_name)
                    self._maybe_abort_on_failure()
                    return

                if t_deq is not None:
                    self._metrics_stage_add(
                        stage_name,
                        dequeue_s=(time.perf_counter() - t_deq),
                        dequeue_calls=1,
                    )

                if self._stop_event.is_set():
                    return
                if not batch:
                    return  # closed and empty

                if self._cfg.enable_stats:
                    self._metrics_stage_add(stage_name, batches=1, items_in=len(batch))

                t_proc = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_process) else None
                try:
                    outs = stage.process_fn(batch)
                    if outs is None:
                        outs = []
                except Exception as e:
                    tb = traceback.format_exc()
                    self._record_failure(
                        ThreadFailure(
                            stage=stage_name,
                            thread_name=threading.current_thread().name,
                            where="process_fn",
                            exc_repr=repr(e),
                            traceback=tb,
                        )
                    )
                    self._log.exception("Worker failure in process_fn (stage=%s)", stage_name)
                    self._maybe_abort_on_failure()
                    if self._cfg.exception_policy == ThreadExceptionPolicy.STOP_ENGINE:
                        return
                    continue
                finally:
                    if t_proc is not None:
                        self._metrics_stage_add(
                            stage_name,
                            process_s=(time.perf_counter() - t_proc),
                            process_calls=1,
                        )

                if self._cfg.enable_stats:
                    self._metrics_stage_add(stage_name, items_out=len(outs))

                if not outs:
                    continue

                if is_last:
                    if self._output_handler is None:
                        if outs:
                            self._warn_no_output_handler(stage_name, len(outs))
                        continue

                    t_out = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_output) else None
                    out_success = False
                    try:
                        self._output_handler(outs)
                        out_success = True
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._record_failure(
                            ThreadFailure(
                                stage=stage_name,
                                thread_name=threading.current_thread().name,
                                where="output_handler",
                                exc_repr=repr(e),
                                traceback=tb,
                            )
                        )
                        self._log.exception("Worker failure in output_handler (stage=%s)", stage_name)
                        self._maybe_abort_on_failure()
                        if self._cfg.exception_policy == ThreadExceptionPolicy.STOP_ENGINE:
                            return
                        continue
                    finally:
                        # Count attempts for timing denominators even on failure; count items only on success.
                        if self._cfg.enable_stats or (t_out is not None):
                            self._metrics_stage_add(
                                stage_name,
                                output_calls=1,
                                output_items=(len(outs) if (out_success and self._cfg.enable_stats) else 0),
                            )
                        if t_out is not None:
                            self._metrics_stage_add(stage_name, output_s=(time.perf_counter() - t_out))

                else:
                    assert next_q is not None and next_policy is not None and next_stage_name is not None
                    t_enq = time.perf_counter() if (self._cfg.enable_timing and self._cfg.timing_profile_enqueue) else None
                    try:
                        self._enqueue_many(
                            target_queue=next_q,
                            items=outs,
                            policy=next_policy,
                            target_stage_name=next_stage_name,
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._record_failure(
                            ThreadFailure(
                                stage=stage_name,
                                thread_name=threading.current_thread().name,
                                where="enqueue_outputs",
                                exc_repr=repr(e),
                                traceback=tb,
                            )
                        )
                        self._log.exception(
                            "Worker failure enqueuing outputs (stage=%s -> %s)", stage_name, next_stage_name
                        )
                        self._maybe_abort_on_failure()
                        if self._cfg.exception_policy == ThreadExceptionPolicy.STOP_ENGINE:
                            return
                        continue
                    finally:
                        if t_enq is not None:
                            self._metrics_stage_add(
                                stage_name,
                                enqueue_s=(time.perf_counter() - t_enq),
                                enqueue_calls=1,
                            )

        except Exception as e:
            # Catch failures in thread_init (or any unexpected error before the main loop).
            tb = traceback.format_exc()
            self._record_failure(
                ThreadFailure(
                    stage=stage_name,
                    thread_name=threading.current_thread().name,
                    where="thread_init",
                    exc_repr=repr(e),
                    traceback=tb,
                )
            )
            self._log.exception("Worker failure in thread_init (stage=%s)", stage_name)
            self._maybe_abort_on_failure()
            return

        finally:
            # Per-thread cleanup hook (runs in the worker thread).
            if stage.thread_cleanup is not None:
                try:
                    stage.thread_cleanup()
                except Exception as e:
                    tb = traceback.format_exc()
                    self._record_failure(
                        ThreadFailure(
                            stage=stage_name,
                            thread_name=threading.current_thread().name,
                            where="thread_cleanup",
                            exc_repr=repr(e),
                            traceback=tb,
                        )
                    )
                    self._log.exception("Worker failure in thread_cleanup (stage=%s)", stage_name)
                    self._maybe_abort_on_failure()

    def _join_all_threads(self, *, timeout: Optional[float] = None) -> None:
        if not self._started:
            return

        end_at = None
        if timeout is not None:
            if timeout < 0:
                raise ValueError("timeout must be >= 0 or None")
            end_at = time.monotonic() + timeout

        cur = threading.current_thread()

        for stage_threads in self._threads:
            for th in stage_threads:
                if th is cur:
                    continue
                if end_at is None:
                    th.join()
                else:
                    remaining = end_at - time.monotonic()
                    if remaining <= 0:
                        return
                    th.join(timeout=remaining)
                    if th.is_alive():
                        return
