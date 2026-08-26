"""Extension points for scalar metrics and derived artifacts.

The phase plan asks for *extension points*, not a fixed catalogue of extra
statistics: a study should be able to add a metric without changing the storage
library. Both registries are bounded, and a failing extension is recorded as a
typed failure rather than allowed to fail the summary -- the same containment
the indexer applies to its event callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from typing import TYPE_CHECKING, Callable, Protocol

from .model import CaptureStorageError
from .summary import ArtifactRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


DEFAULT_MAX_EXTENSIONS = 32
DEFAULT_TIME_BUDGET_NS = 2_000_000_000


class ExtensionError(CaptureStorageError):
    """An extension registry was used incorrectly."""


@dataclass(frozen=True, slots=True)
class ScalarMetric:
    """A named, versioned reduction from a decoded tensor to one float."""

    name: str
    version: int
    compute: Callable[["np.ndarray"], float]

    def __post_init__(self) -> None:
        _validate_identity(self.name, self.version, label="metric name")


@dataclass(frozen=True, slots=True)
class ArtifactProducer:
    """A named, versioned derivation from a decoded tensor to stored bytes.

    Producers return bytes and a content type; they never touch a store
    themselves, so the framework stays in control of what gets written and
    where.
    """

    kind: str
    version: int
    produce: Callable[["np.ndarray"], tuple[bytes, str]]

    def __post_init__(self) -> None:
        _validate_identity(self.kind, self.version, label="artifact kind")


@dataclass(frozen=True, slots=True)
class ExtensionFailure:
    """One extension that raised or overran, kept out of the summary path."""

    name: str
    version: int
    error_type: str
    message: str
    elapsed_ns: int


class ArtifactSink(Protocol):
    """Where produced artifact bytes are written."""

    def put(
        self, *, capture_id: str, kind: str, version: int, data: bytes, content_type: str
    ) -> ArtifactRef: ...


def _validate_identity(name: str, version: int, *, label: str) -> None:
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ExtensionError(f"{label} must be a non-empty string within 128 characters")
    if type(version) is not int or version < 1:
        raise ExtensionError(f"{label} version must be a positive integer")


class ExtensionRegistry:
    """A bounded set of scalar metrics and artifact producers."""

    def __init__(
        self,
        *,
        max_extensions: int = DEFAULT_MAX_EXTENSIONS,
        time_budget_ns: int = DEFAULT_TIME_BUDGET_NS,
        timer_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if type(max_extensions) is not int or max_extensions <= 0:
            raise ValueError("max_extensions must be a positive integer")
        if type(time_budget_ns) is not int or time_budget_ns <= 0:
            raise ValueError("time_budget_ns must be a positive integer")
        self._max_extensions = max_extensions
        self._time_budget_ns = time_budget_ns
        self._timer_ns = timer_ns
        self._metrics: dict[str, ScalarMetric] = {}
        self._producers: dict[str, ArtifactProducer] = {}

    @property
    def max_extensions(self) -> int:
        return self._max_extensions

    @property
    def time_budget_ns(self) -> int:
        return self._time_budget_ns

    @property
    def metrics(self) -> tuple[ScalarMetric, ...]:
        return tuple(self._metrics.values())

    @property
    def producers(self) -> tuple[ArtifactProducer, ...]:
        return tuple(self._producers.values())

    def __len__(self) -> int:
        return len(self._metrics) + len(self._producers)

    def register_metric(self, metric: ScalarMetric) -> ScalarMetric:
        if metric.name in self._metrics:
            raise ExtensionError(f"metric already registered: {metric.name}")
        self._guard_capacity()
        self._metrics[metric.name] = metric
        return metric

    def register_producer(self, producer: ArtifactProducer) -> ArtifactProducer:
        if producer.kind in self._producers:
            raise ExtensionError(f"artifact producer already registered: {producer.kind}")
        self._guard_capacity()
        self._producers[producer.kind] = producer
        return producer

    def _guard_capacity(self) -> None:
        if len(self) >= self._max_extensions:
            raise ExtensionError(
                f"registry exceeds max_extensions: {self._max_extensions}"
            )

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        array: "np.ndarray",
        *,
        capture_id: str,
        sink: ArtifactSink | None = None,
    ) -> tuple[dict[str, float], tuple[ArtifactRef, ...], tuple[ExtensionFailure, ...]]:
        """Run every extension over one decoded tensor.

        Returns the scalar results, the artifact references that were written,
        and the failures. An extension that raises, returns the wrong type, or
        overruns the time budget contributes a failure and nothing else.

        The time budget is checked after each call rather than enforced by
        preemption -- a runaway extension is reported, not interrupted.
        """
        scalars: dict[str, float] = {}
        artifacts: list[ArtifactRef] = []
        failures: list[ExtensionFailure] = []

        for metric in self._metrics.values():
            started = self._timer_ns()
            try:
                value = metric.compute(array)
                elapsed = self._timer_ns() - started
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"metric returned {type(value).__name__}, expected a float"
                    )
                self._guard_budget(elapsed)
                scalars[metric.name] = float(value)
            except Exception as exc:
                failures.append(
                    _failure(metric.name, metric.version, exc, self._timer_ns() - started)
                )

        for producer in self._producers.values():
            if sink is None:
                continue
            started = self._timer_ns()
            try:
                result = producer.produce(array)
                elapsed = self._timer_ns() - started
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or not isinstance(result[0], (bytes, bytearray))
                    or not isinstance(result[1], str)
                ):
                    raise TypeError(
                        "artifact producer must return (bytes, content_type)"
                    )
                self._guard_budget(elapsed)
                artifacts.append(
                    sink.put(
                        capture_id=capture_id,
                        kind=producer.kind,
                        version=producer.version,
                        data=bytes(result[0]),
                        content_type=result[1],
                    )
                )
            except Exception as exc:
                failures.append(
                    _failure(
                        producer.kind, producer.version, exc, self._timer_ns() - started
                    )
                )

        return scalars, tuple(artifacts), tuple(failures)

    def _guard_budget(self, elapsed_ns: int) -> None:
        if elapsed_ns > self._time_budget_ns:
            raise TimeoutError(
                f"extension exceeded its time budget: {elapsed_ns}ns > "
                f"{self._time_budget_ns}ns"
            )


def _failure(
    name: str, version: int, exc: BaseException, elapsed_ns: int
) -> ExtensionFailure:
    return ExtensionFailure(
        name=name,
        version=version,
        error_type=type(exc).__name__,
        message=str(exc)[:512],
        elapsed_ns=elapsed_ns,
    )
