"""Alignment helpers shared by the comparators / matrix (plan §7).

Consolidates the left-pad strip, EOS trim, and ``request_id`` parsing that
several comparators reimplement.  Pure-CPU, ``torch``-only, unit-tested
without CUDA.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import torch

# request_id canonical form is "<group_id>:<row_index>" -- the group is the
# batched generate() call, the row the position within that batch.
_REQUEST_ID_RE = re.compile(r"^(\d+):(\d+)$")

# vLLM appends a "-<8 hex>" UUID suffix to request ids; the ref/disk workers
# strip it so monitored and reference rows key the same.
_VLLM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")


def parse_request_id(req_id: str) -> Tuple[int, int]:
    """Parse ``"<group_id>:<row_index>"`` into ``(group_id, row_index)``.

    Raises ``ValueError`` on an unexpected format so a malformed id surfaces
    loudly rather than silently sorting wrong.
    """
    m = _REQUEST_ID_RE.match(req_id)
    if not m:
        raise ValueError(f"unexpected request_id format: {req_id!r}")
    return int(m.group(1)), int(m.group(2))


def normalize_request_id(req_id: str) -> str:
    """Strip a trailing vLLM ``-<8hex>`` UUID suffix, if present."""
    return _VLLM_SUFFIX_RE.sub("", req_id)


def strip_left_pad(ids_row: torch.Tensor, attn_row: torch.Tensor) -> torch.Tensor:
    """Drop left-padding from a single sequence using its attention mask.

    Returns the last ``attn_row.sum()`` ids (HF left-pads, so the real
    tokens are the trailing run).  Empty (all-pad) rows return an empty
    slice.
    """
    true_len = int(attn_row.sum().item())
    if true_len <= 0:
        return ids_row[:0]
    return ids_row[-true_len:]


def trim_eos(ids: torch.Tensor, eos_id: int,
             *, keep_eos: bool = False) -> torch.Tensor:
    """Trim a 1-D id sequence at the first EOS token.

    With ``keep_eos=False`` (default) the EOS itself is dropped; with
    ``keep_eos=True`` it is retained.  If no EOS is present the sequence is
    returned unchanged.
    """
    flat = ids.reshape(-1)
    hits = torch.nonzero(flat == eos_id, as_tuple=False)
    if hits.numel() == 0:
        return flat
    first = int(hits[0].item())
    return flat[: first + 1] if keep_eos else flat[:first]


def align_to_min_len(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim two tensors to a common length along dim 0.

    Used before a value comparison when reference and monitored captures
    cover slightly different token spans (e.g. generate() drops the final
    never-forwarded token).
    """
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def logits_align_skip(db_len: int, ref_len: int) -> int:
    """Rows to skip at the head of a DB ``final_logits`` block to align to ref.

    The DB (``logits_to_keep=0``) stores every position
    ``[prompt_0..prompt_{N-1}, decode_0..decode_{G-1}]``; ``generate()``'s
    ``output_logits`` yields ``[prefill_last, decode_0..decode_{G-2}]``.
    The prefill-last DB row sits at ``prompt_len - 1 = db_len - ref_len - 1``.
    """
    return max(0, db_len - ref_len - 1)
