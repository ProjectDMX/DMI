from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HookDef:
    name: str
    target: Optional[str]
    dtype: Optional[str]
    anchor_before: Optional[str] = None
    anchor_kind: str = "stmt"
    anchor_header: Optional[str] = None
    anchor_branch: Optional[str] = None
    block_ordinal: int = 0
    class_name: str = ""


@dataclass
class SpecClass:
    name: str
    hooks: list[HookDef] = field(default_factory=list)


@dataclass
class SpecInfo:
    source: str
    framework: str
    classes: list[SpecClass] = field(default_factory=list)


@dataclass
class HookInsertionPlan:
    insertion: Optional[tuple[int, str]] = None
    search_from: Optional[int] = None
    warning: Optional[str] = None
