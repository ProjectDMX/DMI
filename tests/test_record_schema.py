from __future__ import annotations

import pytest

from dmi.api.v1 import (
    RecordCellType,
    RecordColumn,
    RecordLayout,
    RecordSchema,
)


def _metadata_layout(name: str = "metadata", table: str = "metadata_records"):
    return RecordLayout(
        name=name,
        table=table,
        columns=(
            RecordColumn("name", RecordCellType.STRING),
            RecordColumn("rank", RecordCellType.INT32),
            RecordColumn("step", RecordCellType.INT64),
            RecordColumn("score", RecordCellType.FLOAT64),
            RecordColumn("extent", RecordCellType.INT64_ARRAY),
        ),
        primary_key=("name", "rank", "step"),
        order_by=("name", "rank", "step"),
    )


def test_schema_accepts_independent_metadata_and_tensor_layouts():
    tensor_layout = RecordLayout(
        name="payload",
        table="payload_records",
        columns=(
            RecordColumn("record_id", RecordCellType.INT64),
            RecordColumn(
                "payload",
                RecordCellType.TENSOR,
                dtype_column="payload_dtype",
                shape_column="payload_shape",
                bytes_column="payload_bytes",
            ),
        ),
        primary_key=("record_id",),
        order_by=("record_id",),
    )
    schema = RecordSchema((_metadata_layout(), tensor_layout), index_granularity=4096)

    assert schema.layout("metadata") is schema.layouts[0]
    assert schema.layout("payload") is tensor_layout
    assert schema.index_granularity == 4096


def test_tensor_column_requires_all_physical_column_names():
    with pytest.raises(ValueError, match="require dtype_column"):
        RecordColumn("payload", RecordCellType.TENSOR)


def test_layout_rejects_missing_or_undeclared_keys():
    with pytest.raises(ValueError, match="primary_key"):
        RecordLayout(
            name="missing_keys",
            table="missing_keys",
            columns=(RecordColumn("value", RecordCellType.INT64),),
        )

    with pytest.raises(ValueError, match="table identifier"):
        RecordLayout(
            name="qualified_table",
            table="other_database.records",
            columns=(RecordColumn("value", RecordCellType.INT64),),
            primary_key=("value",),
            order_by=("value",),
        )

    with pytest.raises(ValueError, match="must name"):
        RecordLayout(
            name="bad_key",
            table="bad_key",
            columns=(RecordColumn("value", RecordCellType.INT64),),
            primary_key=("absent",),
            order_by=("value",),
        )


def test_schema_rejects_duplicate_layout_or_table_identity():
    layout = _metadata_layout()
    with pytest.raises(ValueError, match="layout names"):
        RecordSchema((layout, _metadata_layout(table="other_table")))
    with pytest.raises(ValueError, match="table names"):
        RecordSchema((layout, _metadata_layout(name="other_layout")))
