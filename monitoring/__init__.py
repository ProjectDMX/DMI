"""Monitoring engine package for asynchronous hook processing."""

from .engine import MonitoringEngine, HostEngineConfig
from .config import CaptureSchedule, HookSelection, MonitoringConfig
from .task import CacheFuture, MonitoringTask

_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "ThreadFailure",
)

def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        from . import _native_engine
        return getattr(_native_engine, name)
    raise AttributeError(name)


__all__ = [
    "MonitoringEngine",
    "HostEngineConfig",
    "MonitoringTask",
    "CacheFuture",
    "CaptureSchedule",
    "HookSelection",
    "MonitoringConfig",
    *_NATIVE_EXPORTS,
]
