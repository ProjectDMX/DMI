"""Opt-in reference bridge from the generic record ring to capture packs.

This module is intentionally not the production capture writer.  Every record
crosses the Python GIL and is copied once into immutable ``bytes`` before queue
admission.  It exists to exercise the pack/catalog/read path end to end while a
future native ``RecordSink`` can target the same pack format directly.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ...hooks.producer_plan import ProducerPlanEntry
from ...hooks.record import OutputStorage
from ...records import (
    PayloadSlice,
    RecordCellType,
    RecordColumn,
    RecordDescriptor,
    RecordLayout,
    RecordSchema,
)
from ...transport import native
from .model import CaptureMetadata, CaptureRecord
from .pipeline import (
    AdmissionResult,
    HostCapturePipeline,
    PipelineFailedError,
    PipelineSnapshot,
)

_LOSS_COUNTERS = (
    "dropped_records",
    "timed_out_records",
    "oversized_records",
    "duplicate_records",
    "rejected_closed_records",
    "failures",
)


def _torch_dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise ValueError(f"unsupported torch dtype: {dtype!r}")
    return name.removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class CapturePayloadSlice:
    """One logical capture stored in a byte range of a producer payload."""

    metadata: CaptureMetadata
    offset_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, CaptureMetadata):
            raise TypeError("metadata must be a CaptureMetadata")
        if type(self.offset_bytes) is not int or self.offset_bytes < 0:
            raise ValueError("offset_bytes must be a non-negative integer")


class CaptureRecordFormat:
    """Encode capture metadata beside tensor slices in one physical payload.

    Wire v1 supports one or more fixed-shape tensor slices.  Every row carries
    canonical metadata JSON plus the exact byte range that belongs to it.
    """

    # The current RecordEnvelope has no explicit device-gate outcome.  In
    # particular, a gated-off empty tensor is indistinguishable from a real
    # empty capture, so this reference format rejects gates at bind time.
    supports_device_gate = False
    LAYOUT_NAME = "capture_pack_reference_v1"

    schema = RecordSchema(
        (
            RecordLayout(
                name=LAYOUT_NAME,
                table=LAYOUT_NAME,
                columns=(
                    RecordColumn("metadata_json", RecordCellType.STRING),
                    RecordColumn(
                        "payload",
                        RecordCellType.TENSOR,
                        dtype_column="payload_dtype",
                        shape_column="payload_shape",
                        bytes_column="payload_bytes",
                    ),
                ),
                primary_key=("metadata_json",),
                order_by=("metadata_json",),
            ),
        )
    )

    def encode(
        self,
        metadata: CaptureMetadata | Sequence[CapturePayloadSlice],
        entry: ProducerPlanEntry,
    ) -> RecordDescriptor:
        if not isinstance(entry, ProducerPlanEntry):
            raise TypeError("entry must be a ProducerPlanEntry")
        if entry.storage is not OutputStorage.TENSOR:
            raise ValueError("capture packs require tensor materialization")
        if isinstance(metadata, CaptureMetadata):
            slices = (CapturePayloadSlice(metadata),)
            expected_shape = entry.output_shape
            if len(metadata.shape) != len(expected_shape) or any(
                planned != -1 and planned != actual
                for planned, actual in zip(expected_shape, metadata.shape)
            ):
                raise ValueError(
                    "capture metadata shape does not match producer output: "
                    f"{metadata.shape} != {expected_shape}"
                )
        else:
            slices = tuple(metadata)
            if not slices or not all(
                isinstance(item, CapturePayloadSlice) for item in slices
            ):
                raise TypeError(
                    "metadata must be CaptureMetadata or CapturePayloadSlice values"
                )
        rows = []
        for item in slices:
            capture = item.metadata
            if _torch_dtype_name(entry.dtype) != capture.dtype:
                raise ValueError(
                    "capture metadata dtype does not match producer: "
                    f"{capture.dtype} != {entry.dtype}"
                )
            end = item.offset_bytes + capture.logical_bytes
            if end > entry.reservation_upper_bytes:
                raise ValueError("capture payload slice exceeds producer reservation")
            encoded = json.dumps(
                capture.to_mapping(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            rows.append((
                encoded,
                PayloadSlice(
                    offset_bytes=item.offset_bytes,
                    nbytes=capture.logical_bytes,
                    dtype=entry.dtype,
                    shape=capture.shape,
                ),
            ))
        return RecordDescriptor(
            layout=self.LAYOUT_NAME,
            rows=tuple(rows),
            output_id=entry.output_id,
        )


class _CapturePackTarget:
    """Python callback target retained by the native reference sink.

    This object deliberately does not own the native sink.  Keeping the target
    separate from the public wrapper avoids a Python -> C++ -> Python reference
    cycle when callers forget an explicit close.
    """

    def __init__(self, pipeline: HostCapturePipeline) -> None:
        if not isinstance(pipeline, HostCapturePipeline):
            raise TypeError("pipeline must be a HostCapturePipeline")
        if not pipeline.is_running:
            raise RuntimeError("pipeline must be started before creating the sink")
        self._pipeline = pipeline
        self._baseline = pipeline.snapshot()
        if any(
            getattr(self._baseline, name) != 0
            for name in (
                "submitted_records",
                "persisted_records",
                "queue_records",
                "failures",
            )
        ):
            raise ValueError("pipeline must be empty and dedicated to this sink")
        self._lock = threading.Lock()
        self._attached = False
        self._closed = False
        self._close_error: BaseException | None = None
        self._close_snapshot: PipelineSnapshot | None = None
        self._accepted = 0

    def close(self, *, timeout: float | None = None) -> PipelineSnapshot:
        with self._lock:
            if self._attached:
                raise RuntimeError("close MonitoringEngine before the reference sink")
            if self._closed:
                error = self._close_error
                snapshot = self._close_snapshot
                if error is not None:
                    raise error
                if snapshot is None:
                    raise RuntimeError("reference capture sink closed without a result")
                return snapshot
            target = self._accepted
        try:
            snapshot = self._pipeline.close(timeout=timeout)
            self._validate_snapshot(snapshot, target)
        except TimeoutError:
            raise
        except BaseException as exc:
            with self._lock:
                self._closed = True
                self._close_error = exc
            raise
        with self._lock:
            self._closed = True
            self._close_snapshot = snapshot
        return snapshot

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._attached

    # Called only by ReferencePythonCaptureSink while holding the GIL.
    def _attach(self) -> None:
        if not self._pipeline.is_running:
            raise RuntimeError("reference capture pipeline is not running")
        self._pipeline.raise_if_failed()
        current = self._pipeline.snapshot()
        if current != self._baseline:
            raise RuntimeError(
                "reference capture pipeline must remain empty before attachment"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("reference capture sink is closed")
            if self._attached:
                raise RuntimeError("reference capture sink is already attached")
            self._attached = True

    def _detach(self) -> None:
        with self._lock:
            self._attached = False

    def _submit_capture(self, metadata_json: str, payload: torch.Tensor) -> None:
        with self._lock:
            if not self._attached or self._closed:
                raise RuntimeError("reference capture sink is not active")
        try:
            mapping = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid capture metadata JSON") from exc
        if not isinstance(mapping, dict):
            raise ValueError("capture metadata JSON must contain an object")
        metadata = CaptureMetadata.from_mapping(mapping)
        if not isinstance(payload, torch.Tensor) or payload.device.type != "cpu":
            raise TypeError("reference capture payload must be a CPU tensor")
        if not payload.is_contiguous():
            raise ValueError("reference capture payload must be contiguous")
        if _torch_dtype_name(payload.dtype) != metadata.dtype:
            raise ValueError("capture metadata dtype does not match payload")
        if tuple(payload.shape) != metadata.shape:
            raise ValueError("capture metadata shape does not match payload")

        immutable = payload.reshape(-1).view(torch.uint8).numpy().tobytes()
        result = self._pipeline.submit(CaptureRecord(metadata, immutable))
        if result is not AdmissionResult.ACCEPTED:
            raise PipelineFailedError(
                f"reference capture admission was not durable: {result.value}"
            )
        with self._lock:
            self._accepted += 1

    def _flush_capture(self, timeout_s: float) -> bool:
        with self._lock:
            target = self._accepted
        if not self._pipeline.flush(timeout=timeout_s):
            return False
        self._pipeline.raise_if_failed()
        self._validate_snapshot(self._pipeline.snapshot(), target)
        return True

    def _rethrow_capture(self) -> None:
        self._pipeline.raise_if_failed()
        self._validate_losses(self._pipeline.snapshot())

    def _validate_snapshot(self, snapshot: PipelineSnapshot, target: int) -> None:
        self._validate_losses(snapshot)
        persisted = snapshot.persisted_records - self._baseline.persisted_records
        if persisted != target:
            raise PipelineFailedError(
                "reference capture durability mismatch: "
                f"accepted={target}, persisted={persisted}"
            )

    def _validate_losses(self, snapshot: PipelineSnapshot) -> None:
        for name in _LOSS_COUNTERS:
            if getattr(snapshot, name) != getattr(self._baseline, name):
                raise PipelineFailedError(
                    f"reference capture pipeline reported {name}"
                )


class CapturePackReferenceSink:
    """Reference-only adapter from one native record ring to capture packs.

    The supplied pipeline must be dedicated to this adapter and started before
    construction.  Close the owning ``MonitoringEngine`` first, then call
    :meth:`close`; this ordering prevents a producer from submitting into a
    closed Python queue.

    This is a correctness reference, not the future production writer: every
    envelope acquires the GIL and copies each declared payload slice to
    immutable Python ``bytes`` before queue admission.
    """

    def __init__(self, pipeline: HostCapturePipeline) -> None:
        self._target = _CapturePackTarget(pipeline)
        self._record_format = CaptureRecordFormat()
        self._native_sink = native.ReferencePythonCaptureSink(
            self._target,
            self._record_format.LAYOUT_NAME,
        )

    @property
    def record_format(self) -> CaptureRecordFormat:
        """The format paired with this sink's versioned wire contract."""

        return self._record_format

    @property
    def native_sink(self) -> Any:
        """Native ``RecordSink`` passed explicitly to ``create_record_runtime``."""

        return self._native_sink

    def close(self, *, timeout: float | None = None) -> PipelineSnapshot:
        """Terminally close the pipeline after checked engine completion."""

        if self._native_sink.attached:
            raise RuntimeError("close MonitoringEngine before the reference sink")
        return self._target.close(timeout=timeout)


__all__ = [
    "CapturePackReferenceSink",
    "CapturePayloadSlice",
    "CaptureRecordFormat",
]
