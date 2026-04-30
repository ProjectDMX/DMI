"""Configuration helpers for monitoring capture scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CaptureSchedule:
    """Schedule for step-level and request-level capture."""

    step_stride: int = 1
    step_offset: int = 0
    warmup_steps: int = 0
    capture_prefill: bool = True
    capture_decode: bool = True

    request_stride: int = 1
    request_offset: int = 0
    warmup_requests: int = 0

    def __post_init__(self) -> None:
        if self.step_stride < 1:
            raise ValueError("step_stride must be >= 1.")
        if self.request_stride < 1:
            raise ValueError("request_stride must be >= 1.")
        if self.step_offset < 0 or self.request_offset < 0:
            raise ValueError("offsets must be >= 0.")
        if self.warmup_steps < 0 or self.warmup_requests < 0:
            raise ValueError("warmup values must be >= 0.")

    def should_capture_request(self, request_id: int) -> bool:
        if request_id < self.warmup_requests:
            return False
        effective = request_id - self.warmup_requests
        if effective < self.request_offset:
            return False
        return (effective - self.request_offset) % self.request_stride == 0

    def should_capture_step(self, step_id: int, phase: Literal["prefill", "decode"]) -> bool:
        if phase not in ("prefill", "decode"):
            raise ValueError(f"Unsupported phase: {phase}")
        if phase == "prefill" and not self.capture_prefill:
            return False
        if phase == "decode" and not self.capture_decode:
            return False
        if step_id < self.warmup_steps:
            return False
        effective = step_id - self.warmup_steps
        if effective < self.step_offset:
            return False
        return (effective - self.step_offset) % self.step_stride == 0


@dataclass
class MonitoringConfig:
    """Bundle capture schedule and runtime flags for the monitoring engine."""

    schedule: CaptureSchedule = field(default_factory=CaptureSchedule)
    debug: bool = False
    no_strip: bool = field(default=False)
