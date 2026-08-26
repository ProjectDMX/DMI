"""Framework-neutral dynamic hooks for the opt-in record runtime.

The released :class:`dmi.hooks.point.HookPoint` path remains independent of
this module.  ``HookPointV1`` is a side-effect-only tap: an integration owns
eligibility and metadata association, while DMI owns preprocessing, ordered
physical outputs, and record-producer dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from ..adapters.base import StepReservation


class TransportType(str, Enum):
    """Physical transformation performed while copying one logical output."""

    IDENTITY = "identity"
    PREFIX_STRIP = "prefix_strip"
    CHUNKED = "chunked"
    SEQ_PREFIX_PACK = "seq_prefix_pack"
    SEGMENTED_PACK = "segmented_pack"


class OutputStorage(IntEnum):
    """Host materialization requested for a producer output."""

    TENSOR = 0
    SCALAR_FLOAT = 1
    SCALAR_INT = 2


class RecordType(IntEnum):
    """Logical row granularity supplied to an integration's record format."""

    PER_SAMPLE = 0
    PER_ITERATION = 1
    PER_EXECUTION = 2


@dataclass(frozen=True, slots=True)
class HookOutput:
    """One tensor output and any device tensors consumed by its transport."""

    tensor: torch.Tensor
    producer_meta: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("HookOutput.tensor must be a Tensor")
        object.__setattr__(self, "producer_meta", tuple(self.producer_meta))
        if not all(isinstance(item, torch.Tensor) for item in self.producer_meta):
            raise TypeError("HookOutput producer metadata must contain only tensors")


@dataclass(frozen=True, slots=True)
class TransportSpec:
    """Static physical contract for one ordered hook output.

    ``reservation_upper_bytes`` is optional for fixed-size outputs, whose
    bound is the captured tensor size.  Dynamic producers may provide a larger
    conservative bound.  ``output_shape`` may contain one ``-1`` dimension,
    which the record consumer infers from the actual produced byte count.
    """

    name: str
    transport_type: TransportType = TransportType.IDENTITY
    storage: OutputStorage = OutputStorage.TENSOR
    record_type: RecordType = RecordType.PER_SAMPLE
    reservation_upper_bytes: int | None = None
    output_shape: tuple[int, ...] | None = None
    row_bytes: int | None = None
    feature_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TransportSpec.name must not be empty")
        if not isinstance(self.transport_type, TransportType):
            raise TypeError("transport_type must be a TransportType")
        if not isinstance(self.storage, OutputStorage):
            raise TypeError("storage must be an OutputStorage")
        if not isinstance(self.record_type, RecordType):
            raise TypeError("record_type must be a RecordType")
        if self.output_shape is not None:
            object.__setattr__(
                self, "output_shape", tuple(int(dim) for dim in self.output_shape)
            )
        if self.reservation_upper_bytes is not None and self.reservation_upper_bytes < 0:
            raise ValueError("reservation_upper_bytes must be non-negative")
        if self.output_shape is not None:
            dynamic_dims = sum(int(dim) == -1 for dim in self.output_shape)
            if dynamic_dims > 1 or any(int(dim) < -1 for dim in self.output_shape):
                raise ValueError("output_shape supports at most one -1 dimension")
        if self.transport_type is TransportType.PREFIX_STRIP:
            if self.row_bytes is None or self.row_bytes <= 0:
                raise ValueError("PREFIX_STRIP requires a positive row_bytes")
        elif self.row_bytes is not None:
            raise ValueError("row_bytes is valid only for PREFIX_STRIP")
        if self.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            if self.feature_bytes is None or self.feature_bytes <= 0:
                raise ValueError(
                    f"{self.transport_type.value} requires a positive feature_bytes"
                )
        elif self.feature_bytes is not None:
            raise ValueError(
                "feature_bytes is valid only for sequence or segmented packing"
            )


@dataclass(frozen=True, slots=True)
class HookSpecV1:
    """Declarative definition of one dynamic hook and its ordered outputs."""

    name: str
    outputs: tuple[TransportSpec, ...]
    preprocess: Callable[..., Any] | None = None
    enabled_by: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HookSpecV1.name must not be empty")
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "enabled_by", frozenset(self.enabled_by))
        if not self.outputs:
            raise ValueError("HookSpecV1 requires at least one output")
        if not all(isinstance(output, TransportSpec) for output in self.outputs):
            raise TypeError("HookSpecV1.outputs must contain TransportSpec values")
        names = [output.name for output in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError("HookSpecV1 output names must be unique")


@runtime_checkable
class HookRuntime(Protocol):
    """Integration-owned association policy called by ``HookPointV1``.

    ``prepare_output`` runs before the producer launch.  Eager integrations
    normally call :meth:`RecordRuntime.emit_output` there; capture integrations
    append to a :class:`ProducerPlanBuilder` and return ``None``.  Returning an
    ``OVERSIZED`` reservation tells the hook that the runtime already completed
    the CPU-direct transfer and that no producer kernel should be launched.
    """

    def should_emit(self, hook: "HookPointV1") -> bool:
        """Return eligibility before preprocessing executes."""

    def prepare_output(
        self,
        *,
        hook: "HookPointV1",
        output_index: int,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation | None:
        """Associate or record one physical output before producer dispatch."""


def _normalize_outputs(value: Any, count: int) -> tuple[Any, ...]:
    if count == 1:
        return (value,)
    if not isinstance(value, (tuple, list)):
        raise ValueError(
            f"HookPointV1 expected {count} outputs, got non-sequence "
            f"{type(value).__name__}"
        )
    if len(value) != count:
        raise ValueError(f"HookPointV1 expected {count} outputs, got {len(value)}")
    return tuple(value)


def _as_hook_output(value: Any) -> HookOutput:
    if isinstance(value, HookOutput):
        return value
    if isinstance(value, torch.Tensor):
        return HookOutput(value)
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        producer_meta = tuple(value[1:])
        if not all(isinstance(item, torch.Tensor) for item in producer_meta):
            raise TypeError("HookOutput producer metadata must contain only tensors")
        return HookOutput(value[0], producer_meta)
    raise TypeError(
        "HookPointV1 output must be a Tensor, HookOutput, or a tuple beginning "
        "with a Tensor"
    )


class HookPointV1(nn.Module):
    """Variadic side-effect hook with optional preprocessing and multiple outputs."""

    def __init__(self, spec: HookSpecV1 | None = None) -> None:
        super().__init__()
        self.enabled = True
        self.spec = spec
        self._output_ids: tuple[int, ...] = ()
        self._ring_payload: torch.Tensor | None = None
        self._hook_runtime: HookRuntime | None = None
        self._gate_tensor: torch.Tensor | None = None
        self._gate_value = 0

    def _bind_record_runtime(
        self,
        *,
        output_ids: Sequence[int],
        ring_payload: torch.Tensor,
        hook_runtime: HookRuntime,
        gate_tensor: torch.Tensor | None,
        gate_value: int,
    ) -> None:
        spec = self.spec
        if spec is None:
            raise RuntimeError("HookPointV1 must have a HookSpecV1 before binding")
        ids = tuple(int(value) for value in output_ids)
        if len(ids) != len(spec.outputs):
            raise ValueError("output_ids length must match HookSpecV1.outputs")
        if gate_tensor is not None:
            if gate_tensor.numel() != 1:
                raise ValueError("gate_tensor must contain exactly one value")
            if gate_tensor.dtype is not torch.int32:
                raise TypeError("gate_tensor must use int32")
            if not gate_tensor.is_cuda:
                raise TypeError("gate_tensor must be a CUDA tensor")
            if not gate_tensor.is_contiguous():
                raise ValueError("gate_tensor must be contiguous")
        self._output_ids = ids
        self._ring_payload = ring_payload
        self._hook_runtime = hook_runtime
        self._gate_tensor = gate_tensor
        self._gate_value = int(gate_value)

    def forward(self, *inputs: Any) -> None:
        spec = self.spec
        runtime = self._hook_runtime
        if not self.enabled or spec is None or runtime is None:
            return None
        if not runtime.should_emit(self):
            return None

        if spec.preprocess is None:
            if len(inputs) != len(spec.outputs):
                raise ValueError(
                    "HookPointV1 without preprocessing requires one input per "
                    f"declared output; expected {len(spec.outputs)}, got {len(inputs)}"
                )
            outputs = tuple(inputs)
        else:
            outputs = _normalize_outputs(
                spec.preprocess(*inputs), len(spec.outputs)
            )
        for index, (output_spec, output_id, value) in enumerate(
            zip(spec.outputs, self._output_ids, outputs)
        ):
            output = _as_hook_output(value)
            reservation = runtime.prepare_output(
                hook=self,
                output_index=index,
                output_id=output_id,
                output_spec=output_spec,
                output=output,
            )
            if reservation is not StepReservation.OVERSIZED:
                self._dispatch(output_spec, output)
        return None

    def _dispatch(self, spec: TransportSpec, output: HookOutput) -> None:
        ring_payload = self._ring_payload
        if ring_payload is None:
            raise RuntimeError("HookPointV1 is not bound to a record runtime")
        tensor = output.tensor.contiguous()
        gate = self._gate_tensor
        gate_value = self._gate_value

        if spec.transport_type is TransportType.IDENTITY:
            torch.ops.ring.record_producer(
                ring_payload, tensor, gate, gate_value
            )
            return
        if spec.transport_type is TransportType.PREFIX_STRIP:
            row_count = self._producer_tensor(output, 0, "row_count")
            torch.ops.ring.record_producer_prefix(
                ring_payload,
                tensor,
                row_count,
                int(spec.row_bytes),
                gate,
                gate_value,
            )
            return
        if spec.transport_type is TransportType.CHUNKED:
            chunk_bytes = self._producer_tensor(output, 0, "chunk_bytes")
            torch.ops.ring.record_producer_chunked(
                ring_payload, tensor, chunk_bytes, gate, gate_value
            )
            return
        if spec.transport_type is TransportType.SEQ_PREFIX_PACK:
            valid_count = self._producer_tensor(output, 0, "valid_count")
            valid_prefix_sum = self._producer_tensor(output, 1, "valid_prefix_sum")
            torch.ops.ring.record_producer_seq_prefix_pack(
                ring_payload,
                tensor,
                valid_count,
                valid_prefix_sum,
                int(spec.feature_bytes),
                gate,
                gate_value,
            )
            return
        if spec.transport_type is TransportType.SEGMENTED_PACK:
            starts = self._producer_tensor(output, 0, "segment_start")
            ends = self._producer_tensor(output, 1, "segment_end")
            torch.ops.ring.record_producer_segmented_pack(
                ring_payload,
                tensor,
                starts,
                ends,
                int(spec.feature_bytes),
                gate,
                gate_value,
            )
            return
        raise ValueError(f"Unsupported transport type {spec.transport_type!r}")

    @staticmethod
    def _producer_tensor(output: HookOutput, index: int, name: str) -> torch.Tensor:
        try:
            value = output.producer_meta[index]
        except IndexError as exc:
            raise ValueError(f"record producer requires {name} tensor") from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        return value


__all__ = [
    "HookOutput",
    "HookPointV1",
    "HookRuntime",
    "HookSpecV1",
    "OutputStorage",
    "RecordType",
    "TransportSpec",
    "TransportType",
]
