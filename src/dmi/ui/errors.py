"""Errors the configurator's CLI surface treats as clean user-facing lines."""

from __future__ import annotations


class UIDependencyError(RuntimeError):
    """The optional UI dependencies are missing or unusable.

    A distinct subclass so the CLI can print the install command without
    catching bare ``RuntimeError`` -- which would also swallow genuine bugs
    (``RecursionError`` is one) as one stderr line.
    """


__all__ = ["UIDependencyError"]
