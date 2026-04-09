"""Configuration helpers for monitoring capture selection and scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence, Tuple


def _get_group_suffixes(group: str) -> Tuple[str, ...]:
    """Return act_name suffixes for a hook group, derived from C++ HOOK_DEFS."""
    from .ring_transport import _ATTN_SUFFIXES, _MLP_SUFFIXES
    if group == "attn":
        return _ATTN_SUFFIXES
    elif group == "mlp":
        return _MLP_SUFFIXES
    raise ValueError(f"Unknown hook group: {group!r}")


def _matches_suffix(name: str, suffixes: Sequence[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


@dataclass
class HookSelection:
    """Select which hooks to enable for capture."""

    mode: Literal["full", "attention", "mlp", "custom"] = "full"
    include: Optional[Sequence[str]] = None
    exclude: Optional[Sequence[str]] = None

    def compile(self, hook_names: Iterable[str]) -> list[str]:
        """Return the ordered list of hook names that should be enabled."""

        names = list(hook_names)

        if self.mode == "full":
            selected = list(names)
        elif self.mode == "attention":
            selected = [name for name in names if _matches_suffix(name, _get_group_suffixes("attn"))]
        elif self.mode == "mlp":
            selected = [name for name in names if _matches_suffix(name, _get_group_suffixes("mlp"))]
        elif self.mode == "custom":
            if self.include is None:
                raise ValueError("HookSelection(mode='custom') requires include to be provided.")
            include_set = set(self.include)
            selected = [name for name in names if name in include_set]
        else:
            raise ValueError(f"Unsupported hook selection mode: {self.mode}")

        if self.include is not None and self.mode != "custom":
            include_set = set(self.include)
            selected = [name for name in selected if name in include_set]

        if self.exclude:
            exclude_set = set(self.exclude)
            selected = [name for name in selected if name not in exclude_set]

        return selected

    def is_full(self) -> bool:
        return self.mode == "full" and self.include is None and not self.exclude


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
    """Bundle hook selection and capture schedule for the monitoring engine."""

    hooks: HookSelection = field(default_factory=HookSelection)
    schedule: CaptureSchedule = field(default_factory=CaptureSchedule)
    debug: bool = False
    no_strip: bool = field(default=False)
