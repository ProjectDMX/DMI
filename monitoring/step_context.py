"""StepContext: per-forward-pass batch metadata produced by adapters.

Carved out as part of the unified-adaptor refactor (Phase 1).  Every
``BackendAdaptor.build_step_context`` returns one of these objects; the
driver feeds it to ``RingTransport.set_step_context`` and
``RingTransport.pre_push_all_metas``.

Today this is built for HF (batched, ``batch>0``) and vLLM
(packed/flattened, ``batch=0``).  Future framework adapters target the same
contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class StepContext:
    """Per-step batch metadata fed to RingTransport before each forward pass.

    Mirrors the inputs of two RingTransport calls:
      - ``set_step_context``: model_id, req_ids, token_ranges, dim0_offsets,
        kv_offsets, tp/dp/ep/pp ranks, flattened.
      - ``pre_push_all_metas``: batch, q_len, kv_dim, logits_to_keep,
        token_ids_dtype.
    """

    model_id:        str
    flattened:       bool
    req_ids:         List[str]
    token_ranges:    List[Tuple[int, int]]
    dim0_offsets:    List[int]
    kv_offsets:      List[int]
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    pp_rank: int = 0
    batch:           int = 0
    q_len:           int = 0
    kv_dim:          int = 0
    logits_to_keep:  int = 0
    token_ids_dtype: Optional[torch.dtype] = None
    # When set, the UNPADDED total-token count for the step (vLLM:
    # total_q before CUDA-graph padding to q_len; HF: same as q_len
    # typically).  Used by adapters that enable padding-strip mode:
    # specs with `dim0_is_actual_tokens=True` substitute this value
    # for `q_len` in shape / byte-budget computation, so the meta
    # describes the unpadded data and reservation is sized to actual.
    # None means today's behavior (use q_len everywhere).
    actual_q_len:    Optional[int] = None

    def transport_kwargs(self) -> dict:
        """Return kwargs accepted by ``RingTransport.set_step_context``."""
        return dict(
            model_id=self.model_id,
            req_ids=self.req_ids,
            token_ranges=self.token_ranges,
            dim0_offsets=self.dim0_offsets,
            kv_offsets=self.kv_offsets,
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            ep_rank=self.ep_rank,
            pp_rank=self.pp_rank,
            flattened=self.flattened,
        )


__all__ = ["StepContext"]
