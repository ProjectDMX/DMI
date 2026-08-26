"""Integration-defined record formats and the opt-in record runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
import re
from typing import Any, Generic, Protocol, Sequence, TypeVar, runtime_checkable

import torch

from .adapters.base import StepReservation
from .hooks.record import (
    HookOutput,
    HookPointV1,
    HookRuntime,
    OutputStorage,
    TransportType,
)
from .hooks.producer_plan import ProducerPlan, ProducerPlanEntry


MetadataT = TypeVar("MetadataT")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DYNAMIC_OUTPUT_ID_BASE = 1 << 16


class RecordCellType(str, Enum):
    """Cell types supported by the schema-driven record sink."""

    STRING = "string"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT64 = "float64"
    INT64_ARRAY = "int64_array"
    TENSOR = "tensor"


@dataclass(frozen=True, slots=True)
class RecordColumn:
    """One logical record column.

    A tensor column expands to explicit dtype, shape, and bytes columns in the
    physical table.  Non-tensor columns use only ``name``.
    """

    name: str
    type: RecordCellType
    dtype_column: str | None = None
    shape_column: str | None = None
    bytes_column: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.name, "column name")
        if not isinstance(self.type, RecordCellType):
            raise TypeError("RecordColumn.type must be a RecordCellType")
        tensor_names = (self.dtype_column, self.shape_column, self.bytes_column)
        if self.type is RecordCellType.TENSOR:
            if any(name is None for name in tensor_names):
                raise ValueError(
                    "TENSOR columns require dtype_column, shape_column, and "
                    "bytes_column"
                )
            for name in tensor_names:
                _require_identifier(str(name), "tensor physical column name")
            if len(set(tensor_names)) != 3:
                raise ValueError("tensor physical column names must be distinct")
        elif any(name is not None for name in tensor_names):
            raise ValueError("physical tensor columns are valid only for TENSOR")


@dataclass(frozen=True, slots=True)
class RecordLayout:
    """One named table layout accepted by a record schema."""

    name: str
    table: str
    columns: tuple[RecordColumn, ...]
    primary_key: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.name, "layout name")
        _require_table_identifier(self.table)
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "primary_key", tuple(self.primary_key))
        object.__setattr__(self, "order_by", tuple(self.order_by))
        if not self.columns:
            raise ValueError("RecordLayout requires at least one column")
        if not all(isinstance(column, RecordColumn) for column in self.columns):
            raise TypeError("RecordLayout.columns must contain RecordColumn values")
        if not self.primary_key:
            raise ValueError("RecordLayout.primary_key must not be empty")
        if not self.order_by:
            raise ValueError("RecordLayout.order_by must not be empty")
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("RecordLayout column names must be unique")
        physical_names: set[str] = set()
        for column in self.columns:
            names_for_column = (
                (
                    str(column.dtype_column),
                    str(column.shape_column),
                    str(column.bytes_column),
                )
                if column.type is RecordCellType.TENSOR
                else (column.name,)
            )
            for physical_name in names_for_column:
                if physical_name in physical_names:
                    raise ValueError(
                        "RecordLayout physical column names must be unique"
                    )
                physical_names.add(physical_name)
        keyable = {
            column.name
            for column in self.columns
            if column.type is not RecordCellType.TENSOR
        }
        for key in (*self.primary_key, *self.order_by):
            if key not in keyable:
                raise ValueError(
                    f"record key {key!r} must name a non-tensor logical column"
                )


@dataclass(frozen=True, slots=True)
class RecordSchema:
    """Immutable set of layouts copied into a schema-driven sink."""

    layouts: tuple[RecordLayout, ...]
    index_granularity: int = 8192

    def __post_init__(self) -> None:
        object.__setattr__(self, "layouts", tuple(self.layouts))
        if not self.layouts:
            raise ValueError("RecordSchema requires at least one layout")
        if not all(isinstance(layout, RecordLayout) for layout in self.layouts):
            raise TypeError("RecordSchema.layouts must contain RecordLayout values")
        if self.index_granularity <= 0:
            raise ValueError("index_granularity must be positive")
        layout_names = [layout.name for layout in self.layouts]
        table_names = [layout.table for layout in self.layouts]
        if len(layout_names) != len(set(layout_names)):
            raise ValueError("RecordSchema layout names must be unique")
        if len(table_names) != len(set(table_names)):
            raise ValueError("RecordSchema table names must be unique")

    def layout(self, name: str) -> RecordLayout:
        for layout in self.layouts:
            if layout.name == name:
                return layout
        raise KeyError(f"record layout {name!r} is not declared by this schema")


@dataclass(frozen=True, slots=True)
class PayloadSlice:
    """Materialize one cell from a byte range of the associated payload."""

    offset_bytes: int = 0
    nbytes: int | None = None
    storage: OutputStorage = OutputStorage.TENSOR
    dtype: torch.dtype | None = None
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.offset_bytes < 0:
            raise ValueError("PayloadSlice.offset_bytes must be non-negative")
        if self.nbytes is not None and self.nbytes < 0:
            raise ValueError("PayloadSlice.nbytes must be non-negative")
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if self.dtype is None:
            raise ValueError("PayloadSlice requires dtype")
        if not isinstance(self.storage, OutputStorage):
            raise TypeError("PayloadSlice.storage must be an OutputStorage")
        if self.storage is OutputStorage.TENSOR:
            if sum(dim == -1 for dim in self.shape) > 1 or any(
                dim < -1 for dim in self.shape
            ):
                raise ValueError("PayloadSlice.shape supports at most one -1")
        elif self.shape:
            raise ValueError("scalar PayloadSlice must not declare a shape")


RecordCell = str | int | float | tuple[int, ...] | PayloadSlice


@dataclass(frozen=True, slots=True)
class RecordDescriptor:
    """Pre-encoded rows associated with one FIFO payload or host record."""

    layout: str
    rows: tuple[tuple[RecordCell, ...], ...]
    output_id: int | None = None

    def __post_init__(self) -> None:
        if not self.layout:
            raise ValueError("RecordDescriptor.layout must not be empty")
        object.__setattr__(
            self,
            "rows",
            tuple(tuple(row) for row in self.rows),
        )
        if self.output_id is not None and self.output_id < 0:
            raise ValueError("RecordDescriptor.output_id must be non-negative")

    @property
    def has_payload(self) -> bool:
        return any(
            isinstance(cell, PayloadSlice)
            for row in self.rows
            for cell in row
        )


@runtime_checkable
class RecordFormat(Protocol[MetadataT]):
    """Integration-owned conversion from semantic metadata to encoded rows."""

    @property
    def schema(self) -> RecordSchema:
        """Schema containing every layout returned by :meth:`encode`."""

    def encode(
        self,
        metadata: MetadataT,
        entry: ProducerPlanEntry,
    ) -> RecordDescriptor:
        """Encode one metadata object before its producer is launched."""


class RecordRuntime(Generic[MetadataT]):
    """Non-owning facade over one ring's opt-in encoded-record path."""

    def __init__(self, transport: Any, record_format: RecordFormat[MetadataT]) -> None:
        if not isinstance(record_format, RecordFormat):
            raise TypeError("record_format must implement RecordFormat")
        if not isinstance(record_format.schema, RecordSchema):
            raise TypeError("record_format.schema must be a RecordSchema")
        self._transport = transport
        self._format = record_format
        self._transport.configure_record_schema(record_format.schema)
        self._next_output_id = _DYNAMIC_OUTPUT_ID_BASE
        self._output_name_to_id: dict[str, int] = {}
        self._output_specs_by_name: dict[str, Any] = {}
        self._bound_output_ids: set[int] = set()
        self._bound_hooks: set[int] = set()
        self._device_gated_output_ids: set[int] = set()

    def bind_hook(
        self,
        hook: HookPointV1,
        *,
        hook_runtime: HookRuntime,
        gate_tensor: torch.Tensor | None = None,
        gate_value: int = 0,
    ) -> None:
        """Register outputs and bind one hook to this runtime."""

        if not isinstance(hook, HookPointV1):
            raise TypeError("hook must be a HookPointV1")
        if not isinstance(hook_runtime, HookRuntime):
            raise TypeError("hook_runtime must implement HookRuntime")
        if id(hook) in self._bound_hooks:
            raise RuntimeError("HookPointV1 is already bound to this runtime")
        spec = hook.spec
        if spec is None:
            raise RuntimeError("HookPointV1 must have a HookSpecV1 before binding")
        for output in spec.outputs:
            existing = self._output_specs_by_name.get(output.name)
            if existing is not None and existing != output:
                raise ValueError(
                    f"output name {output.name!r} is already bound with a "
                    "different TransportSpec"
                )
        next_output_id = self._next_output_id
        new_ids: dict[str, int] = {}
        output_ids_list = []
        for output in spec.outputs:
            output_id = self._output_name_to_id.get(output.name)
            if output_id is None:
                output_id = next_output_id
                next_output_id += 1
                new_ids[output.name] = output_id
            output_ids_list.append(output_id)
        output_ids = tuple(output_ids_list)
        hook._bind_record_runtime(
            output_ids=output_ids,
            ring_payload=self._transport._record_payload_tensor(),
            hook_runtime=hook_runtime,
            gate_tensor=gate_tensor,
            gate_value=gate_value,
        )
        self._next_output_id = next_output_id
        for output in spec.outputs:
            if output.name in new_ids:
                self._output_name_to_id[output.name] = new_ids[output.name]
                self._output_specs_by_name[output.name] = output
        self._bound_output_ids.update(output_ids)
        if gate_tensor is not None:
            self._device_gated_output_ids.update(output_ids)
        self._bound_hooks.add(id(hook))

    def emit_output(
        self,
        entry: ProducerPlanEntry,
        metadata: MetadataT,
        output: HookOutput,
    ) -> StepReservation:
        """Reserve, publish a descriptor, and prepare one eager producer.

        For an oversized entry the transport has already flushed older work;
        this method publishes the descriptor and completes the matching
        CPU-direct submission.  The hook therefore skips its producer launch.
        """

        self._validate_entry_output(entry, output)
        descriptor = self._encode(metadata, entry)
        reservation = StepReservation(
            self._transport.reserve_record(self._reservation_items((entry,)))
        )
        self._transport.push_record_descriptors((descriptor,))
        if reservation is StepReservation.OVERSIZED:
            self._transport.submit_record_cpu_direct(output, entry)
        return reservation

    def prepare_replay(
        self,
        plan: ProducerPlan,
        metadata: Sequence[MetadataT],
    ) -> StepReservation:
        """Bind fresh descriptors to a captured physical plan and reserve it."""

        if not isinstance(plan, ProducerPlan):
            raise TypeError("plan must be a ProducerPlan")
        metadata_items = tuple(metadata)
        if len(metadata_items) != len(plan.entries):
            raise ValueError(
                "replay metadata count must match producer plan entries: "
                f"expected {len(plan.entries)}, got {len(metadata_items)}"
            )
        if not plan.entries:
            return StepReservation.SKIPPED
        descriptors = tuple(
            self._encode(item, entry)
            for item, entry in zip(metadata_items, plan.entries)
        )
        reservation = StepReservation(
            self._transport.reserve_record(self._reservation_items(plan.entries))
        )
        if reservation is not StepReservation.OVERSIZED:
            self._transport.push_record_descriptors(descriptors)
        return reservation

    def _reservation_items(
        self,
        entries: Sequence[ProducerPlanEntry],
    ) -> tuple[tuple[int, bool], ...]:
        return tuple(
            (entry.aligned_reservation_bytes, self._needs_reclaim(entry))
            for entry in entries
        )

    def _needs_reclaim(self, entry: ProducerPlanEntry) -> bool:
        tensor_bytes = (
            prod(entry.input_shape)
            * torch.empty((), dtype=entry.dtype).element_size()
        )
        tensor_aligned_bytes = (tensor_bytes + 15) & ~15
        return not (
            entry.transport_type is TransportType.IDENTITY
            and entry.output_id not in self._device_gated_output_ids
            and entry.aligned_reservation_bytes == tensor_aligned_bytes
        )

    def _encode(
        self,
        metadata: MetadataT,
        entry: ProducerPlanEntry,
    ) -> RecordDescriptor:
        if entry.output_id not in self._bound_output_ids:
            raise ValueError(
                f"producer output_id {entry.output_id} is not bound to this runtime"
            )
        descriptor = self._format.encode(metadata, entry)
        self._validate_descriptor(descriptor, expected_output_id=entry.output_id)
        if descriptor.rows and not descriptor.has_payload:
            raise ValueError("producer RecordDescriptor must contain a PayloadSlice")
        self._validate_payload_slices(descriptor, entry)
        return descriptor

    def _validate_descriptor(
        self,
        descriptor: RecordDescriptor,
        *,
        expected_output_id: int | None,
    ) -> None:
        if not isinstance(descriptor, RecordDescriptor):
            raise TypeError("RecordFormat.encode must return a RecordDescriptor")
        if descriptor.output_id != expected_output_id:
            raise ValueError(
                "RecordDescriptor.output_id does not match its producer: "
                f"expected {expected_output_id!r}, got {descriptor.output_id!r}"
            )
        layout = self._format.schema.layout(descriptor.layout)
        for row_index, row in enumerate(descriptor.rows):
            if len(row) != len(layout.columns):
                raise ValueError(
                    f"record row {row_index} has {len(row)} cells; "
                    f"layout {layout.name!r} requires {len(layout.columns)}"
                )
            for column, value in zip(layout.columns, row):
                _validate_cell(column, value)

    @staticmethod
    def _validate_entry_output(
        entry: ProducerPlanEntry,
        output: HookOutput,
    ) -> None:
        tensor = output.tensor
        if tensor.dtype != entry.dtype:
            raise ValueError(
                f"producer dtype changed: expected {entry.dtype}, got {tensor.dtype}"
            )
        shape = tuple(int(dim) for dim in tensor.shape)
        if shape != entry.input_shape:
            raise ValueError(
                f"producer input shape changed: expected {entry.input_shape}, got {shape}"
            )

    @staticmethod
    def _validate_payload_slices(
        descriptor: RecordDescriptor,
        entry: ProducerPlanEntry,
    ) -> None:
        slices = [
            cell
            for row in descriptor.rows
            for cell in row
            if isinstance(cell, PayloadSlice)
        ]
        if sum(payload.nbytes is None for payload in slices) > 1:
            raise ValueError(
                "at most one PayloadSlice may consume the remaining actual payload"
            )
        for payload in slices:
            if payload.storage is not entry.storage:
                raise ValueError(
                    "PayloadSlice storage does not match its producer plan entry"
                )
            if payload.dtype != entry.dtype:
                raise ValueError(
                    f"PayloadSlice dtype changed: expected {entry.dtype}, "
                    f"got {payload.dtype}"
                )
            end = (
                None
                if payload.nbytes is None
                else payload.offset_bytes + payload.nbytes
            )
            if payload.offset_bytes > entry.reservation_upper_bytes or (
                end is not None and end > entry.reservation_upper_bytes
            ):
                raise ValueError("PayloadSlice exceeds producer reservation bound")


def _require_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")


def _require_table_identifier(value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid table identifier: {value!r}")


def _validate_cell(column: RecordColumn, value: RecordCell) -> None:
    cell_type = column.type
    if cell_type is RecordCellType.STRING:
        valid = isinstance(value, str)
    elif cell_type is RecordCellType.INT32:
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(1 << 31) <= value < (1 << 31)
        )
    elif cell_type is RecordCellType.INT64:
        valid = (
            (
                isinstance(value, int)
                and not isinstance(value, bool)
                and -(1 << 63) <= value < (1 << 63)
            )
            or (
                isinstance(value, PayloadSlice)
                and value.storage is OutputStorage.SCALAR_INT
            )
        )
    elif cell_type is RecordCellType.FLOAT64:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ) or (
            isinstance(value, PayloadSlice)
            and value.storage is OutputStorage.SCALAR_FLOAT
        )
    elif cell_type is RecordCellType.INT64_ARRAY:
        valid = isinstance(value, tuple) and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and -(1 << 63) <= item < (1 << 63)
            for item in value
        )
    else:
        valid = (
            isinstance(value, PayloadSlice)
            and value.storage is OutputStorage.TENSOR
        )
    if not valid:
        raise TypeError(
            f"column {column.name!r} requires {cell_type.value}, "
            f"got {type(value).__name__}"
        )


__all__ = [
    "PayloadSlice",
    "RecordCell",
    "RecordCellType",
    "RecordColumn",
    "RecordDescriptor",
    "RecordFormat",
    "RecordLayout",
    "RecordRuntime",
    "RecordSchema",
]
