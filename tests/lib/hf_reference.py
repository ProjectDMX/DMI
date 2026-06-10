"""HF reference rollouts for the matrix / wrappers (plan §7).

Re-exports the ROL (manual KV-cache rollout: full logits + hidden states +
attn patterns) and GEN (``generate()`` token_ids + decode scores) reference
helpers under stable public names.  The canonical implementation still
lives in :mod:`tests.hf_reference`; this shim gives the shared ``tests.lib``
namespace a single import point without moving the 700-line module (that
relocation is deferred to the legacy-removal PR so this one stays additive).
"""
from __future__ import annotations

from tests.hf_reference import (  # noqa: F401  (re-exported)
    _HFRef as HFRef,
    _HFGenRef as HFGenRef,
    _hf_greedy_rollout_collect_all_batched as rollout_collect_all,
    _hf_generate_collect_scores_batched as generate_collect_scores,
    _hf_generate_collect_hidden_states_batched as generate_collect_hidden_states,
    _load_hf_refs_from_disk as load_refs_from_disk,
)

__all__ = [
    "HFRef",
    "HFGenRef",
    "rollout_collect_all",
    "generate_collect_scores",
    "generate_collect_hidden_states",
    "load_refs_from_disk",
]
