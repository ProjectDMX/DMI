"""Contracts shared by public-API DMI black-box tests."""

from __future__ import annotations

from typing import Any


_TRANSPARENCY_FIELDS = (
    "schema_version",
    "model",
    "cudagraph",
    "tensor_parallel_size",
    "prompts",
    "prompt_token_ids",
    "token_ids",
    "texts",
    "finish_reasons",
    "stop_reasons",
)


def transparency_mismatches(
    baseline: dict[str, Any],
    monitored: dict[str, Any],
) -> list[str]:
    """Return public output fields changed by enabling DMI monitoring."""

    missing = [
        field
        for field in _TRANSPARENCY_FIELDS
        if field not in baseline or field not in monitored
    ]
    if missing:
        return [f"missing required field: {field}" for field in missing]

    return [
        field
        for field in _TRANSPARENCY_FIELDS
        if baseline[field] != monitored[field]
    ]
