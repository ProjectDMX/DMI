"""GPU -> Ring -> NativePackSink -> storage service -> catalog -> reader.

The whole native capture storage path on a real engine, selected by config
alone: ``storage_backend="capture"`` with ``capture_sink_config`` and
``capture_storage_config``. ``flush_and_wait`` returning is the promise under
test -- every record captured before it is queryable in the catalog and reads
back byte-identical to the CUDA tensor it came from -- with no Python between
the ring and the catalog.

Needs CUDA, the native backend, ``_dmi_native_sink`` and ``_dmi_native_store``,
and ClickHouse (HTTP on DMI_CLICKHOUSE_HTTP_PORT, default 8123); the object
store is the in-process signature-verifying fake.
"""

from __future__ import annotations

import uuid
from os import environ
from pathlib import Path

import pytest
import torch

from tests._requirements import (
    require_clickhouse, require_cuda, require_native_backend,
)
# Module-level so the fake-S3 fixture registers in this module.
from tests.test_native_s3_client import (  # noqa: E402
    ACCESS, BUCKET, REGION, SECRET, fake_s3,
)

BUILD = Path(__file__).resolve().parents[1] / "native" / "build"
EXTENSIONS_BUILT = all(
    sorted(BUILD.glob(f"{name}*.so"))
    for name in ("_dmi_native_sink", "_dmi_native_store"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
    pytest.mark.native_backend,
    require_cuda(),
    require_clickhouse(),
    require_native_backend(),
    pytest.mark.skipif(
        not EXTENSIONS_BUILT,
        reason="_dmi_native_sink / _dmi_native_store are not built; run "
        "`make -C native build/_dmi_native_sink build/_dmi_native_store "
        "PYTHON=<venv>/bin/python`",
    ),
]


class _CaptureHookRuntime:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self.metadata = None

    def should_emit(self, hook):
        return True

    def prepare_output(self, *, hook, output_index, output_id, output_spec,
                       output):
        from dmi.api.v1 import ProducerPlanBuilder

        entry = ProducerPlanBuilder().record_output(
            output_id=output_id, output_spec=output_spec, output=output)
        return self._runtime.emit_output(entry, self.metadata, output)


def _metadata(capture_id: str, tensor: torch.Tensor, *, step: int):
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=capture_id, tenant_id="tenant-gpu",
        experiment_id="experiment-gpu", run_id="run-gpu",
        session_id="session-gpu", request_id=f"request-{step}",
        sequence_id="sequence-gpu", model_id="model-gpu",
        model_revision="revision-gpu", adapter_revision=None,
        capture_policy_version="policy-v1", hook_name="capture_tensor",
        layer_number=step, producer_rank=0, step_number=step,
        token_start=step, token_end=step + 1, batch_position=0,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _ring_config():
    from dmi.api.v1 import RingConfig

    config = RingConfig()
    config.task_ring_entries = 32
    config.payload_ring_bytes = 64 * 1024
    config.pinned_staging_bytes = 64 * 1024
    return config


def test_flush_and_wait_means_queryable_and_byte_identical(fake_s3, tmp_path):
    from dmi.api.v1 import HookPointV1, HookSpecV1, MonitoringEngine, TransportSpec
    from dmi.config import MonitoringConfig
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig, ClickHouseCatalogWriter,
    )
    from dmi.storage.capture.native_sink import NativeSinkConfig
    from dmi.storage.native_capture import (
        NativeCaptureReader, NativeCaptureStorageConfig,
    )

    database = environ.get("DMI_CLICKHOUSE_DATABASE", "default")
    storage = NativeCaptureStorageConfig(
        s3_endpoint=fake_s3, s3_bucket=BUCKET, s3_region=REGION,
        s3_access_key=ACCESS, s3_secret_key=SECRET,
        s3_allow_insecure_http=True,
        clickhouse_host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        clickhouse_port=int(environ.get("DMI_CLICKHOUSE_HTTP_PORT", "8123")),
        database=database, table_prefix=f"dmi_gpu_{uuid.uuid4().hex}",
        poll_interval_s=0.05,
    )
    config = MonitoringConfig(
        storage_backend="capture",
        capture_sink_config=NativeSinkConfig(
            spool_root=str(tmp_path / "spool"), max_pack_records=2,
            max_linger_ns=60_000_000_000),
        capture_storage_config=storage,
    )
    # Three dtypes, and one record above the 64 KiB payload ring, so both
    # the ring and the CPU-direct fallback feed the same sink.
    tensors = {
        "ring-f32": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "ring-bf16": torch.linspace(-2, 2, 64).to(torch.bfloat16),
        "direct-f16": torch.randn(64, 1024).to(torch.float16),
        "ring-i64": torch.arange(-5, 5, dtype=torch.int64),
    }
    engine = MonitoringEngine(config=config, model_id="native-storage-gpu",
                              ring_config=_ring_config())
    try:
        runtime = engine.create_record_runtime(CaptureRecordFormat())
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        for step, (capture_id, tensor) in enumerate(tensors.items()):
            hook_runtime.metadata = _metadata(capture_id, tensor, step=step)
            hook(tensor.cuda())

        engine.flush_and_wait(60.0)

        # Queryable now -- not eventually. No polling.
        reader = NativeCaptureReader(storage)
        selection = reader.select(tenant_id="tenant-gpu")
        captures = {capture.descriptor["capture_id"]: capture
                    for capture in reader.read(selection, byte_limit=1 << 26)}
        assert sorted(captures) == sorted(tensors)
        for capture_id, tensor in tensors.items():
            capture = captures[capture_id]
            assert capture.payload == tensor.view(torch.uint8).numpy().tobytes()
            assert torch.equal(capture.tensor(), tensor), capture_id
        snapshot = engine._capture_storage.snapshot()
        assert snapshot["indexed_rows"] == len(tensors), snapshot
        assert snapshot["pending_index"] == 0, snapshot
    finally:
        engine.close()
        import clickhouse_driver

        ClickHouseCatalogWriter(
            clickhouse_driver.Client(
                host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
                port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000"))),
            ClickHouseCatalogConfig(database=database,
                                    table_prefix=storage.table_prefix),
        ).drop_schema()
    assert not sorted((tmp_path / "spool").rglob("*.dmi-pack.ready"))


def test_close_alone_delivers_the_tail_to_the_catalog(fake_s3, tmp_path):
    """No flush_and_wait: close() must still seal the sink's open pack and
    drain it into the catalog. The 60 s linger means nothing but a flush can
    seal it, and 3 records against max_pack_records=2 leave one record in the
    open pack when close() runs."""
    from dmi.api.v1 import HookPointV1, HookSpecV1, MonitoringEngine, TransportSpec
    from dmi.config import MonitoringConfig
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.storage.capture.clickhouse_catalog import (
        ClickHouseCatalogConfig, ClickHouseCatalogWriter,
    )
    from dmi.storage.capture.native_sink import NativeSinkConfig
    from dmi.storage.native_capture import (
        NativeCaptureReader, NativeCaptureStorageConfig,
    )

    database = environ.get("DMI_CLICKHOUSE_DATABASE", "default")
    storage = NativeCaptureStorageConfig(
        s3_endpoint=fake_s3, s3_bucket=BUCKET, s3_region=REGION,
        s3_access_key=ACCESS, s3_secret_key=SECRET,
        s3_allow_insecure_http=True,
        clickhouse_host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
        clickhouse_port=int(environ.get("DMI_CLICKHOUSE_HTTP_PORT", "8123")),
        database=database, table_prefix=f"dmi_gpu_{uuid.uuid4().hex}",
        poll_interval_s=0.05,
    )
    config = MonitoringConfig(
        storage_backend="capture",
        capture_sink_config=NativeSinkConfig(
            spool_root=str(tmp_path / "spool"), max_pack_records=2,
            max_linger_ns=60_000_000_000),
        capture_storage_config=storage,
    )
    tensors = {f"tail-{i}": torch.arange(8, dtype=torch.float32) + i
               for i in range(3)}
    engine = MonitoringEngine(config=config, model_id="native-storage-gpu",
                              ring_config=_ring_config())
    try:
        runtime = engine.create_record_runtime(CaptureRecordFormat())
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = _CaptureHookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        for step, (capture_id, tensor) in enumerate(tensors.items()):
            hook_runtime.metadata = _metadata(capture_id, tensor, step=step)
            hook(tensor.cuda())
        torch.cuda.synchronize()

        engine.close()  # no flush_and_wait

        reader = NativeCaptureReader(storage)
        selection = reader.select(tenant_id="tenant-gpu")
        captures = {capture.descriptor["capture_id"]: capture
                    for capture in reader.read(selection, byte_limit=1 << 26)}
        assert sorted(captures) == sorted(tensors)
        for capture_id, tensor in tensors.items():
            assert torch.equal(captures[capture_id].tensor(), tensor), capture_id
    finally:
        engine.close()
        import clickhouse_driver

        ClickHouseCatalogWriter(
            clickhouse_driver.Client(
                host=environ.get("DMI_CLICKHOUSE_HOST", "127.0.0.1"),
                port=int(environ.get("DMI_CLICKHOUSE_PORT", "9000"))),
            ClickHouseCatalogConfig(database=database,
                                    table_prefix=storage.table_prefix),
        ).drop_schema()
