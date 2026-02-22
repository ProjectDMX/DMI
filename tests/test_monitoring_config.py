import pytest

from monitoring import MonitoringConfig, MonitoringEngine, NativePartialSealConfig


def test_native_partial_seal_config_validation():
    with pytest.raises(ValueError, match="cap_ratio"):
        NativePartialSealConfig(cap_ratio=0.0)
    with pytest.raises(ValueError, match="driver_guard_mb"):
        NativePartialSealConfig(driver_guard_mb=-1)
    with pytest.raises(ValueError, match="chunk_bytes"):
        NativePartialSealConfig(chunk_bytes=-1)


def test_monitoring_config_as_dict_includes_native_partial_seal():
    cfg = MonitoringConfig(
        native_partial_seal=NativePartialSealConfig(
            enabled=True,
            chunk_bytes=32 * 1024 * 1024,
            cap_enabled=True,
            cap_ratio=0.9,
            driver_guard_mb=512,
        )
    )
    data = cfg.as_dict()
    assert data["native_partial_seal"] == {
        "enabled": True,
        "chunk_bytes": 32 * 1024 * 1024,
        "cap_enabled": True,
        "cap_ratio": 0.9,
        "driver_guard_mb": 512,
    }


def test_apply_native_runtime_config_calls_backend_setter():
    cfg = MonitoringConfig(
        native_partial_seal=NativePartialSealConfig(
            enabled=False,
            chunk_bytes=8 * 1024 * 1024,
            cap_enabled=True,
            cap_ratio=0.75,
            driver_guard_mb=256,
        )
    )
    engine = MonitoringEngine(async_enabled=False, config=cfg)

    class _DummyBackend:
        def __init__(self):
            self.args = None

        def set_partial_seal_config(self, *args):
            self.args = args

    backend = _DummyBackend()
    engine._native_backend = backend
    engine._apply_native_runtime_config()
    assert backend.args == (False, 8 * 1024 * 1024, True, 0.75, 256)
