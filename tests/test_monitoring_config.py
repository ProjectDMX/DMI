import pytest

from monitoring import AdvanceConfig, MonitoringConfig, MonitoringEngine, NativePartialSealConfig


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


def test_advance_config_defaults():
    cfg = AdvanceConfig()
    assert cfg.pinpool_bins_kb == (256, 512, 1024, 2048, 4096, 8192)
    assert cfg.pinpool_max_mb == 512
    assert cfg.host_copy_threads == 0
    assert cfg.host_copy_queue_size == 512


def test_advance_config_filters_invalid_bins_and_keeps_tuple():
    cfg = AdvanceConfig(pinpool_bins_kb=(0, 512, -1, 1024))
    assert cfg.pinpool_bins_kb == (512, 1024)
    assert isinstance(cfg.pinpool_bins_kb, tuple)


def test_advance_config_rejects_empty_or_invalid_bins():
    with pytest.raises(ValueError, match="at least one positive integer"):
        AdvanceConfig(pinpool_bins_kb=())
    with pytest.raises(ValueError, match="at least one positive integer"):
        AdvanceConfig(pinpool_bins_kb=(0, -1))


def test_advance_config_rejects_invalid_numeric_fields():
    with pytest.raises(ValueError, match="pinpool_max_mb must be > 0"):
        AdvanceConfig(pinpool_max_mb=0)
    with pytest.raises(ValueError, match="host_copy_threads must be >= 0"):
        AdvanceConfig(host_copy_threads=-1)
    with pytest.raises(ValueError, match="host_copy_queue_size must be > 0"):
        AdvanceConfig(host_copy_queue_size=0)


def test_monitoring_config_as_dict_includes_advance_debug_and_no_strip():
    cfg = MonitoringConfig(
        advance=AdvanceConfig(
            pinpool_bins_kb=(512,),
            pinpool_max_mb=256,
            host_copy_threads=2,
            host_copy_queue_size=64,
        ),
        debug=True,
        no_strip=True,
    )
    data = cfg.as_dict()
    assert data["advance"] == {
        "pinpool_bins_kb": [512],
        "pinpool_max_mb": 256,
        "host_copy_threads": 2,
        "host_copy_queue_size": 64,
    }
    assert data["debug"] is True
    assert data["no_strip"] is True


def test_monitoring_config_default_debug_false():
    cfg = MonitoringConfig()
    assert cfg.debug is False
    assert cfg.as_dict()["debug"] is False


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
