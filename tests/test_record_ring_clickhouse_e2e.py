"""Generic record-runtime E2E through the CUDA ring and ClickHouse."""

from __future__ import annotations

import os
import uuid

import pytest
import torch

from tests._requirements import (
    require_clickhouse,
    require_cuda,
    require_native_backend,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
    pytest.mark.native_backend,
    require_cuda(),
    require_clickhouse(),
    require_native_backend(),
]

_TABLE_PREFIX = "dmi_phase2_record_ring_e2e"


class _TensorRecordFormat:
    def __init__(self, schema):
        self.schema = schema

    def encode(self, metadata, entry):
        from dmi.api.v1 import PayloadSlice, RecordDescriptor

        run_id, record_id = metadata
        return RecordDescriptor(
            "tensor_record",
            ((
                run_id,
                record_id,
                PayloadSlice(dtype=entry.dtype, shape=entry.output_shape),
            ),),
            output_id=entry.output_id,
        )


class _EagerHookRuntime:
    def __init__(self, runtime, metadata):
        self._runtime = runtime
        self._metadata = metadata

    def should_emit(self, hook):
        return True

    def prepare_output(
        self,
        *,
        hook,
        output_index,
        output_id,
        output_spec,
        output,
    ):
        from dmi.api.v1 import ProducerPlanBuilder

        entry = ProducerPlanBuilder().record_output(
            output_id=output_id,
            output_spec=output_spec,
            output=output,
        )
        return self._runtime.emit_output(entry, self._metadata, output)


def test_record_producer_ring_descriptor_and_clickhouse_readback():
    clickhouse_driver = pytest.importorskip("clickhouse_driver")
    from dmi.api.v1 import (
        ClickHouseClientConfig,
        DMXHostEngine,
        HookPointV1,
        HookSpecV1,
        MonitoringEngine,
        RecordCellType,
        RecordColumn,
        RecordLayout,
        RecordSchema,
        RingConfig,
        StageConfig,
        TransportSpec,
    )

    db_host = os.environ.get("DMX_DB_HOST", "127.0.0.1")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    database = os.environ.get("DMX_DB_DATABASE", "default")
    quoted_database = database.replace("`", "``")
    table = f"{_TABLE_PREFIX}_{uuid.uuid4().hex[:12]}"
    qualified_table = f"`{quoted_database}`.`{table}`"
    admin = clickhouse_driver.Client(host=db_host, port=db_port)
    admin.execute(f"CREATE DATABASE IF NOT EXISTS `{quoted_database}`")

    schema = RecordSchema(
        (
            RecordLayout(
                name="tensor_record",
                table=table,
                columns=(
                    RecordColumn("run_id", RecordCellType.STRING),
                    RecordColumn("record_id", RecordCellType.INT64),
                    RecordColumn(
                        "payload",
                        RecordCellType.TENSOR,
                        dtype_column="payload_dtype",
                        shape_column="payload_shape",
                        bytes_column="payload_bytes",
                    ),
                ),
                primary_key=("run_id", "record_id"),
                order_by=("run_id", "record_id"),
            ),
        ),
        index_granularity=1024,
    )

    clickhouse_config = ClickHouseClientConfig()
    clickhouse_config.host = db_host
    clickhouse_config.port = db_port
    clickhouse_config.database = database
    clickhouse_config.create_database_if_missing = True
    clickhouse_config.drop_existing_database = False
    host = DMXHostEngine(
        StageConfig.clickhouse_records(
            clickhouse_config,
            schema,
            parallelism=1,
            name="phase2_record_ring_e2e",
        )
    )

    ring_config = RingConfig()
    ring_config.task_ring_entries = 64
    ring_config.payload_ring_bytes = 1024 * 1024
    ring_config.pinned_staging_bytes = 1024 * 1024

    engine = None
    try:
        engine = MonitoringEngine(
            model_id="phase2-generic-record-e2e",
            host_engine=host,
            ring_config=ring_config,
        )
        assert host.wait_until_ready(10.0)

        runtime = engine.create_record_runtime(_TensorRecordFormat(schema))
        hook = HookPointV1(
            HookSpecV1("phase2_tensor", (TransportSpec("payload"),))
        )
        hook_runtime = _EagerHookRuntime(runtime, ("run-a", 7))
        runtime.bind_hook(hook, hook_runtime=hook_runtime)

        scalar_hook = HookPointV1(
            HookSpecV1("phase2_scalar_tensor", (TransportSpec("scalar_payload"),))
        )
        scalar_hook_runtime = _EagerHookRuntime(runtime, ("run-a", 8))
        runtime.bind_hook(scalar_hook, hook_runtime=scalar_hook_runtime)

        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected_scalar = torch.tensor(3.5, dtype=torch.float32)
        hook(expected.cuda())
        scalar_hook(expected_scalar.cuda())
        engine.flush_and_wait(30.0)

        client = clickhouse_driver.Client(
            host=db_host,
            port=db_port,
            database=database,
        )
        rows = client.execute(
            f"SELECT run_id, record_id, payload_dtype, payload_shape, "
            f"hex(payload_bytes) FROM `{table}` ORDER BY run_id, record_id"
        )
        assert rows == [
            (
                "run-a",
                7,
                "torch.float",
                [3, 4],
                expected.numpy().tobytes().hex().upper(),
            ),
            (
                "run-a",
                8,
                "torch.float",
                [],
                expected_scalar.numpy().tobytes().hex().upper(),
            ),
        ]
    finally:
        try:
            if engine is not None:
                engine.close()
            else:
                host.stop(False, 10.0)
        finally:
            admin.execute(f"DROP TABLE IF EXISTS {qualified_table}")
