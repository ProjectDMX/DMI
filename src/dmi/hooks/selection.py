"""Hook selection policy: presets, resolution, and PP/TP filters.

Catalog-derived constants and specification types live in
:mod:`dmi.hooks.specs`; this module decides which hooks to enable.

Adapters extend the preset table at import time via ``register_preset``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

# These describe the universe of hooks without importing a transport runtime.
from .specs import (
    _id_by_short,
    _ATTN_WT_TYPES,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_PATTERN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    PP_FIRST_ONLY,
    PP_LAST_ONLY,
    TP_SHARDED_TYPES,
)

if TYPE_CHECKING:
    from .specs import HookSpec, ModelShapeConfig


# ---------------------------------------------------------------------------
# Hook selection: composable presets + individual hook types
#
# Selection is a comma-separated string.  Each token is looked up in
# _HOOK_SELECTIONS (presets or individual hook names).  The final enabled
# set is the union of all tokens.
#
# Examples:
#   "full"                            -- all hooks
#   "vllm-full"                       -- full minus attn_scores/pattern
#   "hidden-states,token_ids"         -- resid_pre + token_ids
#   "hidden-states,final_ln,logits"   -- resid_pre + final_ln + final_logits
#   "resid_pre,resid_final,embed"     -- just those three
# ---------------------------------------------------------------------------

_ALL_HOOK_TYPES = frozenset(_id_by_short.values())

# -- Presets --
_HOOK_SELECTIONS: Dict[str, frozenset] = {
    "full": _ALL_HOOK_TYPES,
    # Framework-specific integrations may register additional presets through
    # register_preset without modifying this core table.
    # What HF returns with output_hidden_states + output_attentions + logits
    "hf-only": frozenset({
        HOOK_TYPE_RESID_PRE, HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_PATTERN,
        HOOK_TYPE_FINAL_LOGITS,
    }),
}

# -- Individual hook type names (auto-derived from HOOK_DEFS) --
for _name, _htype in _id_by_short.items():
    _HOOK_SELECTIONS[_name] = frozenset({_htype})

# -- Aliases --
_HOOK_SELECTIONS["hidden-states"] = _HOOK_SELECTIONS["resid_pre"]
_HOOK_SELECTIONS["hidden_states"] = _HOOK_SELECTIONS["resid_pre"]
_HOOK_SELECTIONS["logits"] = _HOOK_SELECTIONS["final_logits"]
_HOOK_SELECTIONS["token-ids"] = _HOOK_SELECTIONS["token_ids"]


def register_preset(name: str, hook_types: frozenset) -> None:
    """Register a new hook-selection preset.

    Used by integration adapters to add framework-specific presets at import
    time without modifying core code.  Strict-by-default: raises
    ``ValueError`` if ``name`` is already registered.  Callers that want to
    override an existing preset must delete the existing entry first.
    """
    if name in _HOOK_SELECTIONS:
        raise ValueError(
            f"Preset {name!r} already registered. "
            f"Delete the existing entry from _HOOK_SELECTIONS before "
            f"calling register_preset.")
    _HOOK_SELECTIONS[name] = hook_types


def resolve_hook_selection(mode: str) -> frozenset:
    """Resolve a comma-separated hook selection string to a set of hook types.

    Each comma-separated token is looked up in _HOOK_SELECTIONS (presets
    and individual hook names).  The result is the union of all tokens.
    """
    result: set = set()
    for token in mode.split(","):
        token = token.strip()
        if not token:
            continue
        entry = _HOOK_SELECTIONS.get(token)
        if entry is None:
            raise ValueError(
                f"Unknown hook selection {token!r}. "
                f"Available: {sorted(_HOOK_SELECTIONS.keys())}")
        result |= entry
    if not result:
        raise ValueError(f"Empty hook selection: {mode!r}")
    return frozenset(result)


def select_hook_specs(
    specs: List["HookSpec"],
    mode: str,
    cfg: Optional["ModelShapeConfig"] = None,
) -> List["HookSpec"]:
    """Purely filter specs by selection and available shape configuration."""
    allowed = resolve_hook_selection(mode)
    unavailable = set()
    if cfg is not None:
        if cfg.intermediate_dim == 0:
            unavailable.add(HOOK_TYPE_MLP_POST)
        if cfg.num_experts == 0:
            unavailable.add(HOOK_TYPE_ROUTER_LOGITS)
        if cfg.top_k == 0:
            unavailable.add(HOOK_TYPE_TOPK_IDS)
            unavailable.add(HOOK_TYPE_TOPK_WEIGHTS)
    return [
        spec
        for spec in specs
        if spec.hook_type in allowed and spec.hook_type not in unavailable
    ]


def apply_hook_selection(
    specs: List["HookSpec"],
    mode: str,
    cfg: Optional["ModelShapeConfig"] = None,
) -> List["HookSpec"]:
    """Filter specs and set HookPoint.enabled based on a selection string.

    Selection is a comma-separated string of preset names and/or individual
    hook type names.  The enabled set is the union of all tokens.

    If cfg is provided, hooks requiring unavailable model config fields
    (e.g. mlp_post when intermediate_dim is unknown) are skipped.

    Sets enabled=True on hooks in the set, enabled=False on others.
    Returns the filtered list of enabled specs (for _active_specs / metadata).
    """
    enabled_specs = select_hook_specs(specs, mode, cfg)
    enabled_ids = {id(spec) for spec in enabled_specs}
    unavailable = set()
    if cfg is not None:
        if cfg.intermediate_dim == 0:
            unavailable.add(HOOK_TYPE_MLP_POST)
        if cfg.num_experts == 0:
            unavailable.add(HOOK_TYPE_ROUTER_LOGITS)
        if cfg.top_k == 0:
            unavailable.add(HOOK_TYPE_TOPK_IDS)
            unavailable.add(HOOK_TYPE_TOPK_WEIGHTS)

    for spec in specs:
        if spec.module is None:
            raise RuntimeError(
                "apply_hook_selection requires bound executable HookSpecs"
            )
        spec.module.enabled = id(spec) in enabled_ids
        if spec.hook_type in unavailable:
            import warnings
            warnings.warn(
                f"[apply_hook_selection] Skipping hook_type={spec.hook_type} "
                f"layer={spec.layer_no}: required model config unavailable "
                f"(e.g. intermediate_dim=0)")
    return enabled_specs


# ---------------------------------------------------------------------------
# PP / TP filters (used by adapters during attach_model)
# ---------------------------------------------------------------------------

def hook_belongs_to_pp_rank(
    spec: "HookSpec", is_first_rank: bool, is_last_rank: bool
) -> bool:
    """Return whether a hook belongs to the given PP-stage role."""
    if spec.hook_type in PP_FIRST_ONLY and not is_first_rank:
        return False
    if spec.hook_type in PP_LAST_ONLY and not is_last_rank:
        return False
    return True


def hook_belongs_to_tp_rank(spec: "HookSpec", tp_rank: int) -> bool:
    """Return whether a hook belongs to the given TP-rank role."""
    return tp_rank == 0 or spec.hook_type in TP_SHARDED_TYPES


def filter_by_pp_rank(specs: list, is_first_rank: bool, is_last_rank: bool) -> list:
    """Drop hooks that are PP-stage-restricted and not on this rank."""
    filtered = []
    for s in specs:
        if not hook_belongs_to_pp_rank(s, is_first_rank, is_last_rank):
            if s.module is None:
                raise RuntimeError(
                    "filter_by_pp_rank requires bound executable HookSpecs"
                )
            s.module.enabled = False
            continue
        filtered.append(s)
    return filtered


def filter_by_tp_rank(specs: list, tp_rank: int) -> list:
    """On non-zero TP ranks, keep only sharded hooks to avoid Nx duplicate
    writes of identical unsharded data.  Rank 0 keeps all hooks."""
    if tp_rank == 0:
        return specs
    filtered = []
    for s in specs:
        if not hook_belongs_to_tp_rank(s, tp_rank):
            if s.module is None:
                raise RuntimeError(
                    "filter_by_tp_rank requires bound executable HookSpecs"
                )
            s.module.enabled = False
            continue
        filtered.append(s)
    return filtered


__all__ = [
    "register_preset",
    "resolve_hook_selection",
    "select_hook_specs",
    "apply_hook_selection",
    "hook_belongs_to_pp_rank",
    "hook_belongs_to_tp_rank",
    "filter_by_pp_rank",
    "filter_by_tp_rank",
]
