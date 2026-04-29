"""DSL for DMI hook spec files.

Spec files import H and spec from here.  Neither function does anything
at runtime — the compiler reads them via AST only.
"""
from __future__ import annotations
from typing import Any


def H(name: str, target: Any = None, *, dtype: Any = None) -> None:  # noqa: N802
    """Hook insertion marker.  Never called at runtime."""


def spec(source: str, framework: str = "hf"):
    """Decorator declaring which source file this spec targets."""
    def _decorator(cls):
        return cls
    return _decorator
