"""Validation tests for integration-defined record schemas and descriptors."""

from __future__ import annotations

import pytest
import torch

from dmi.hooks.dynamic import OutputStorage
from dmi.records import (
    PayloadSlice,
    RecordCellType,
    RecordColumn,
    RecordDescriptor,
    RecordLayout,
    RecordSchema,
)

pytestmark = pytest.mark.cpu


def test_tensor_column_requires_explicit_physical_columns():
    with pytest.raises(ValueError, match="require dtype_column"):
        RecordColumn("payload", RecordCellType.TENSOR)

    column = RecordColumn(
        "payload",
        RecordCellType.TENSOR,
        dtype_column="payload_dtype",
        shape_column="payload_shape",
        bytes_column="payload_bytes",
    )
    assert column.bytes_column == "payload_bytes"


def test_layout_rejects_colliding_physical_column_names():
    with pytest.raises(ValueError, match="physical column names"):
        RecordLayout(
            "collision",
            "collision",
            (
                RecordColumn("payload_dtype", RecordCellType.STRING),
                RecordColumn(
                    "payload",
                    RecordCellType.TENSOR,
                    "payload_dtype",
                    "payload_shape",
                    "payload_bytes",
                ),
            ),
            primary_key=("payload_dtype",),
            order_by=("payload_dtype",),
        )


def test_schema_has_named_immutable_layouts():
    layout = RecordLayout(
        "tensor_rows",
        "tensor_rows",
        (
            RecordColumn("record_id", RecordCellType.INT64),
            RecordColumn(
                "payload",
                RecordCellType.TENSOR,
                "payload_dtype",
                "payload_shape",
                "payload_bytes",
            ),
        ),
        primary_key=("record_id",),
        order_by=("record_id",),
    )
    schema = RecordSchema((layout,))

    assert schema.layout("tensor_rows") is layout
    with pytest.raises(KeyError):
        schema.layout("missing")


def test_payload_slice_accepts_one_inferred_dimension():
    payload = PayloadSlice(dtype=torch.float32, shape=(-1, 8))
    descriptor = RecordDescriptor(
        "tensor_rows",
        ((1, payload),),
        output_id=100,
    )
    assert descriptor.has_payload

    with pytest.raises(ValueError, match="at most one -1"):
        PayloadSlice(dtype=torch.float32, shape=(-1, -1))


def test_descriptor_accepts_zero_rows():
    descriptor = RecordDescriptor("tensor_rows", (), output_id=100)

    assert descriptor.rows == ()
    assert not descriptor.has_payload


def test_scalar_payload_slice_has_no_tensor_shape():
    with pytest.raises(ValueError, match="must not declare a shape"):
        PayloadSlice(
            storage=OutputStorage.SCALAR_FLOAT,
            dtype=torch.float32,
            shape=(1,),
        )
