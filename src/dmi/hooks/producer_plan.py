"""Ordered physical producer plans for the opt-in record runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .dynamic import HookOutput, OutputStorage, RecordType, TransportSpec, TransportType


def _align_up(value: int, alignment: int = 16) -> int:
    return (int(value) + alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True, slots=True)
class ProducerPlanEntry:
    """Physical state required to reserve and emit one producer output."""

    output_id: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: torch.dtype
    transport_type: TransportType
    transport_args: tuple[int, ...]
    storage: OutputStorage
    record_type: RecordType
    reservation_upper_bytes: int
    task_count: int = 1

    def __post_init__(self) -> None:
        if self.output_id < 0:
            raise ValueError("output_id must be non-negative")
        if self.reservation_upper_bytes < 0:
            raise ValueError("reservation_upper_bytes must be non-negative")
        if self.task_count != 1:
            raise ValueError("each ProducerPlanEntry currently represents one task")

    @property
    def aligned_reservation_bytes(self) -> int:
        return _align_up(self.reservation_upper_bytes)


@dataclass(frozen=True, slots=True)
class ProducerPlan:
    """Immutable producer order learned during one graph capture."""

    entries: tuple[ProducerPlanEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    @property
    def total_reservation_bytes(self) -> int:
        return sum(entry.aligned_reservation_bytes for entry in self.entries)

    @property
    def task_count(self) -> int:
        return sum(entry.task_count for entry in self.entries)

    @property
    def signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                entry.output_id,
                entry.input_shape,
                entry.output_shape,
                entry.dtype,
                entry.transport_type,
                entry.transport_args,
                entry.storage,
                entry.record_type,
                entry.reservation_upper_bytes,
                entry.task_count,
            )
            for entry in self.entries
        )

    def assert_compatible(self, other: "ProducerPlan") -> None:
        if self.signature == other.signature:
            return
        if len(self.entries) != len(other.entries):
            raise ValueError(
                "producer plan entry-count mismatch: "
                f"expected {len(self.entries)}, got {len(other.entries)}"
            )
        for index, (expected, actual) in enumerate(
            zip(self.signature, other.signature)
        ):
            if expected != actual:
                raise ValueError(
                    f"producer plan mismatch at entry {index}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        raise ValueError("producer plan mismatch")


class ProducerPlanBuilder:
    """Append producer entries in the order in which hooks execute."""

    def __init__(self) -> None:
        self._entries: list[ProducerPlanEntry] = []

    def record_output(
        self,
        *,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> ProducerPlanEntry:
        tensor = output.tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("HookOutput.tensor must be a Tensor")
        input_shape = tuple(int(dim) for dim in tensor.shape)
        output_shape = (
            input_shape
            if output_spec.output_shape is None
            else tuple(int(dim) for dim in output_spec.output_shape)
        )
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        upper_bytes = (
            tensor_bytes
            if output_spec.reservation_upper_bytes is None
            else int(output_spec.reservation_upper_bytes)
        )
        if upper_bytes < tensor_bytes and output_spec.transport_type is TransportType.IDENTITY:
            raise ValueError(
                "IDENTITY reservation_upper_bytes cannot be smaller than the tensor"
            )
        transport_args = self._transport_args(output_spec)
        entry = ProducerPlanEntry(
            output_id=int(output_id),
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=tensor.dtype,
            transport_type=output_spec.transport_type,
            transport_args=transport_args,
            storage=output_spec.storage,
            record_type=output_spec.record_type,
            reservation_upper_bytes=upper_bytes,
        )
        self._entries.append(entry)
        return entry

    def build(self) -> ProducerPlan:
        return ProducerPlan(tuple(self._entries))

    @staticmethod
    def _transport_args(spec: TransportSpec) -> tuple[int, ...]:
        if spec.transport_type is TransportType.PREFIX_STRIP:
            return (int(spec.row_bytes),)
        if spec.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            return (int(spec.feature_bytes),)
        return ()


__all__ = ["ProducerPlan", "ProducerPlanBuilder", "ProducerPlanEntry"]
