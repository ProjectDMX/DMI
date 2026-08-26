from __future__ import annotations

from contextlib import contextmanager
import os
import uuid

import pytest
import torch

from tests._requirements import require_clickhouse, require_native_backend


pytestmark = [
    pytest.mark.clickhouse,
    pytest.mark.native_backend,
    require_clickhouse(),
    require_native_backend(),
]


@contextmanager
def _engine_table_cleanup(engine, client, tables, *, graceful):
    def finish():
        if not engine.stop(graceful, 10.0):
            raise AssertionError("ClickHouse test engine did not stop within 10s")

        drop_failures = []
        for table in tables:
            try:
                client.execute(f"DROP TABLE IF EXISTS `{table}`")
            except Exception as exc:
                drop_failures.append((table, exc))
        if len(drop_failures) == 1:
            raise drop_failures[0][1]
        if drop_failures:
            detail = "; ".join(
                f"{table}: {error}" for table, error in drop_failures
            )
            raise RuntimeError(f"failed to drop ClickHouse test tables: {detail}") from (
                drop_failures[0][1]
            )

    try:
        yield
    except BaseException as body_error:
        try:
            finish()
        except BaseException as cleanup_error:
            raise body_error from cleanup_error
        raise
    else:
        finish()


def test_schema_driven_rows_round_trip_with_tensor_expansion():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    tensor_table = f"dmi_record_tensor_{uuid.uuid4().hex[:12]}"
    schema = RecordSchema(
        (
            RecordLayout(
                name="tensor",
                table=tensor_table,
                columns=(
                    RecordColumn("run", RecordCellType.STRING),
                    RecordColumn("rank", RecordCellType.INT32),
                    RecordColumn("step", RecordCellType.INT64),
                    RecordColumn("score", RecordCellType.FLOAT64),
                    RecordColumn("extent", RecordCellType.INT64_ARRAY),
                    RecordColumn(
                        "payload",
                        RecordCellType.TENSOR,
                        dtype_column="payload_dtype",
                        shape_column="payload_shape",
                        bytes_column="payload_bytes",
                    ),
                ),
                primary_key=("run", "rank", "step"),
                order_by=("run", "rank", "step"),
            ),
        ),
        index_granularity=1024,
    )

    host = os.environ.get("DMX_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    client = clickhouse_driver.Client(host=host, port=port)
    config = ClickHouseClientConfig()
    config.host = host
    config.port = port
    config.database = "default"
    stage = StageConfig.clickhouse_records(config, schema, parallelism=2)
    engine = DMXHostEngine(stage)
    payload = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    with _engine_table_cleanup(engine, client, (tensor_table,), graceful=True):
        engine.start()
        assert engine.wait_until_ready(10.0)
        engine.submit_record(
            "tensor",
            ["run-a", 2, 11, 1.5, [2, 3], payload],
            ["string", "int32", "int64", "float64", "int64_array", "tensor"],
            nbytes=payload.nbytes,
        )
        assert engine.flush_and_wait(10.0)

        tensor_rows = client.execute(
            f"SELECT run, rank, step, score, extent, payload_dtype, payload_shape, "
            f"hex(payload_bytes) FROM `{tensor_table}`"
        )
        assert len(tensor_rows) == 1
        tensor_row = tensor_rows[0]
        assert tensor_row[:7] == (
            "run-a", 2, 11, 1.5, [2, 3], "torch.float", [2, 3]
        )
        assert tensor_row[7] == payload.numpy().tobytes().hex().upper()


@pytest.mark.parametrize(
    ("table_definition", "expected_failure"),
    (
        pytest.param(
            "(`event_id` Int64, `run` String, `score` Float64) "
            "ENGINE = MergeTree PRIMARY KEY (`run`) "
            "ORDER BY (`run`, `event_id`) "
            "SETTINGS index_granularity = 1024",
            "columns/types",
            id="physical-column-order",
        ),
        pytest.param(
            "(`run` String, `event_id` UInt64, `score` Float64) "
            "ENGINE = MergeTree PRIMARY KEY (`run`) "
            "ORDER BY (`run`, `event_id`) "
            "SETTINGS index_granularity = 1024",
            "columns/types",
            id="columns-and-types",
        ),
        pytest.param(
            "(`run` String, `event_id` Int64, `score` Float64) "
            "ENGINE = MergeTree PRIMARY KEY (`run`, `event_id`) "
            "ORDER BY (`run`, `event_id`) "
            "SETTINGS index_granularity = 1024",
            "primary key",
            id="primary-key",
        ),
        pytest.param(
            "(`run` String, `event_id` Int64, `score` Float64) "
            "ENGINE = MergeTree PRIMARY KEY (`run`) "
            "ORDER BY (`run`, `score`, `event_id`) "
            "SETTINGS index_granularity = 1024",
            "ORDER BY",
            id="order-by",
        ),
        pytest.param(
            "(`run` String, `event_id` Int64, `score` Float64) "
            "ENGINE = MergeTree PRIMARY KEY (`run`) "
            "ORDER BY (`run`, `event_id`) "
            "SETTINGS index_granularity = 2048",
            "index_granularity",
            id="index-granularity",
        ),
    ),
)
def test_schema_driven_stage_rejects_live_table_mismatch_before_ready(
    table_definition, expected_failure
):
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    table = f"dmi_record_mismatch_{uuid.uuid4().hex[:12]}"
    host = os.environ.get("DMX_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    client = clickhouse_driver.Client(host=host, port=port)

    schema = RecordSchema(
        (
            RecordLayout(
                name="event",
                table=table,
                columns=(
                    RecordColumn("run", RecordCellType.STRING),
                    RecordColumn("event_id", RecordCellType.INT64),
                    RecordColumn("score", RecordCellType.FLOAT64),
                ),
                primary_key=("run",),
                order_by=("run", "event_id"),
            ),
        ),
        index_granularity=1024,
    )
    config = ClickHouseClientConfig()
    config.host = host
    config.port = port
    config.database = "default"
    engine = DMXHostEngine(
        StageConfig.clickhouse_records(config, schema, parallelism=1)
    )

    with _engine_table_cleanup(engine, client, (table,), graceful=False):
        client.execute(f"CREATE TABLE `{table}` {table_definition}")
        engine.start()
        assert not engine.wait_until_ready(10.0)
        assert engine.clickhouse_metrics().ready_workers == 0
        failures = engine.failures()
        assert failures
        assert expected_failure in failures[0].exc_what


def test_schema_driven_stage_revalidates_after_same_process_table_recreation():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    from dmi.api.v1 import RecordCellType, RecordColumn, RecordLayout, RecordSchema
    from dmi.transport.native import ClickHouseClientConfig, DMXHostEngine, StageConfig

    table = f"dmi_record_recreated_{uuid.uuid4().hex[:12]}"
    host = os.environ.get("DMX_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    client = clickhouse_driver.Client(host=host, port=port)
    config = ClickHouseClientConfig()
    config.host = host
    config.port = port
    config.database = "default"

    def schema_for(cell_type):
        return RecordSchema(
            (
                RecordLayout(
                    name="event",
                    table=table,
                    columns=(RecordColumn("event_id", cell_type),),
                    primary_key=("event_id",),
                    order_by=("event_id",),
                ),
            ),
            index_granularity=1024,
        )

    first = DMXHostEngine(
        StageConfig.clickhouse_records(
            config, schema_for(RecordCellType.INT64), parallelism=1
        )
    )
    with _engine_table_cleanup(first, client, (table,), graceful=True):
        first.start()
        assert first.wait_until_ready(10.0)

    second = DMXHostEngine(
        StageConfig.clickhouse_records(
            config, schema_for(RecordCellType.INT32), parallelism=1
        )
    )
    with _engine_table_cleanup(second, client, (table,), graceful=True):
        second.start()
        assert second.wait_until_ready(10.0)
        assert client.execute(f"DESCRIBE TABLE `{table}`")[0][:2] == (
            "event_id",
            "Int32",
        )
