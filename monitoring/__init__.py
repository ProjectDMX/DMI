"""Monitoring engine package for asynchronous hook processing."""

from .engine import MonitoringEngine, HostEngineConfig
from .graph_engine import GraphSafeEngine, GraphSlotResult
from .graph_consumer import GraphSlotConsumer
from .graph_monitor import GraphMonitor, SlotInfo
from .config import AdvanceConfig, CaptureSchedule, HookSelection, MonitoringConfig, NativePartialSealConfig
from .task import CacheFuture, MonitoringTask

_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "ThreadFailure",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
)

def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        from . import _native_engine
        return getattr(_native_engine, name)
    raise AttributeError(name)


__all__ = [
    "MonitoringEngine",
    "GraphSafeEngine",
    "GraphSlotConsumer",
    "GraphSlotResult",
    "GraphMonitor",
    "SlotInfo",
    "HostEngineConfig",
    "MonitoringTask",
    "CacheFuture",
    "CaptureSchedule",
    "HookSelection",
    "NativePartialSealConfig",
    "AdvanceConfig",
    "MonitoringConfig",
    *_NATIVE_EXPORTS,
]
