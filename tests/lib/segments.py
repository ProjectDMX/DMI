"""Chunked-segment merge helpers (plan §7).

Thin wrapper over :mod:`monitoring.segment_merger` so the matrix, the
pytest wrappers, and the numeric study all merge chunked ring/CH segments
through one entry point.  Re-exports the canonical implementation rather
than reimplementing the per-act-name merge rules.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from monitoring.segment_merger import (  # noqa: F401  (re-exported)
    merge_segments,
    segment_manager,
    parse_internal_id,
    get_delta_token_len,
)


def merge_request_chunks(
    chunks: List[Tuple[int, int, torch.Tensor]], act_name: str,
    *, drop_token_cnt_to: Optional[int] = None,
) -> torch.Tensor:
    """Sort ``(start, end, tensor)`` chunks by start token and merge them.

    Convenience over :func:`merge_segments` for the
    ``group_by_request`` output shape (lists of ``(s, e, t)`` triples).
    """
    ordered = sorted(chunks, key=lambda c: c[0])
    return merge_segments(
        [t for _, _, t in ordered], act_name, drop_token_cnt_to=drop_token_cnt_to)
