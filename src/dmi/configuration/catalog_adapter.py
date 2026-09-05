"""Project the DMI hook catalog into UI-facing metadata.

``dmi.hooks.catalog.HOOK_DEFS`` stays the single source of truth for which
observations exist. This module adds only what the catalog deliberately does
not carry: human labels, a presentation grouping, and per-model availability.

Two deviations from the raw catalog are intentional:

* **Presentation grouping.** ``router_logits``, ``topk_ids`` and
  ``topk_weights`` are catalog group ``GROUP_OTHER``, but users look for them
  under MoE. The mapping below regroups them without touching the catalog.
* **Unknown hooks still surface.** A hook added to ``HOOK_DEFS`` that is
  missing from the label table falls back to its catalog group and short name
  rather than disappearing from the UI.

Availability mirrors the suppression rules already implemented in
``dmi.hooks.selection.select_hook_specs`` -- they are restated here against a
descriptor topology (design time, no model loaded) rather than reimplemented
with different semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ..hooks.catalog import GROUP_ATTN, GROUP_MLP, HOOK_DEFS
from .schema import ModelTopology

# Presentation groups, in the order the UI should render them.
GROUP_ATTENTION = "attention"
GROUP_MLP_UI = "mlp"
GROUP_MOE = "moe"
GROUP_RESIDUAL = "residual"
GROUP_GLOBAL = "global"

GROUP_ORDER = (
    GROUP_ATTENTION,
    GROUP_MLP_UI,
    GROUP_MOE,
    GROUP_RESIDUAL,
    GROUP_GLOBAL,
)

GROUP_LABELS = {
    GROUP_ATTENTION: "Attention",
    GROUP_MLP_UI: "MLP",
    GROUP_MOE: "MoE",
    GROUP_RESIDUAL: "Residual stream",
    GROUP_GLOBAL: "Global",
}

# short_name -> (presentation group, human label)
_PRESENTATION: dict[str, tuple[str, str]] = {
    "q": (GROUP_ATTENTION, "Q"),
    "k": (GROUP_ATTENTION, "K"),
    "v": (GROUP_ATTENTION, "V"),
    "pattern": (GROUP_ATTENTION, "Attention pattern"),
    "attn_scores": (GROUP_ATTENTION, "Attention scores (pre-softmax)"),
    "z": (GROUP_ATTENTION, "Attention head output"),
    "attn_out": (GROUP_ATTENTION, "Attention output"),
    "mlp_in": (GROUP_MLP_UI, "MLP input"),
    "mlp_post": (GROUP_MLP_UI, "MLP post-activation"),
    "mlp_out": (GROUP_MLP_UI, "MLP output"),
    "router_logits": (GROUP_MOE, "Router logits"),
    "topk_ids": (GROUP_MOE, "Expert IDs"),
    "topk_weights": (GROUP_MOE, "Expert weights"),
    "resid_pre": (GROUP_RESIDUAL, "Residual stream (pre-layer)"),
    "resid_mid": (GROUP_RESIDUAL, "Residual stream (mid-layer)"),
    "ln1": (GROUP_RESIDUAL, "Layer norm 1"),
    "ln2": (GROUP_RESIDUAL, "Layer norm 2"),
    "embed": (GROUP_GLOBAL, "Token embedding"),
    "pos_embed": (GROUP_GLOBAL, "Positional embedding"),
    "token_ids": (GROUP_GLOBAL, "Token IDs"),
    "resid_final": (GROUP_GLOBAL, "Residual stream (final)"),
    "final_ln": (GROUP_GLOBAL, "Final normalization"),
    "final_logits": (GROUP_GLOBAL, "Final logits"),
}

# Fallback grouping for a catalog hook with no presentation entry.
_CATALOG_GROUP_FALLBACK = {
    GROUP_ATTN: GROUP_ATTENTION,
    GROUP_MLP: GROUP_MLP_UI,
}


@dataclass(frozen=True)
class HookInfo:
    """One observation, as the UI needs to see it."""

    id: str
    label: str
    group: str
    per_layer: bool
    available: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _presentation_for(short_name: str, catalog_group: int) -> tuple[str, str]:
    entry = _PRESENTATION.get(short_name)
    if entry is not None:
        return entry
    return _CATALOG_GROUP_FALLBACK.get(catalog_group, GROUP_GLOBAL), short_name


def _unavailable_reason(
    short_name: str, topology: Optional[ModelTopology]
) -> Optional[str]:
    """Mirror of the suppression rules in ``select_hook_specs``."""
    if topology is None:
        return None
    if short_name == "mlp_post" and topology.intermediate_size == 0:
        return "Model has no MLP intermediate dimension"
    if short_name == "router_logits" and topology.num_experts == 0:
        return "Model has no experts"
    if short_name in ("topk_ids", "topk_weights") and topology.top_k == 0:
        return "Model has no top-k expert routing"
    if short_name == "final_logits" and topology.vocab_size == 0:
        return "Model descriptor has no vocab_size, so final logits have no shape"
    return None


def hook_ids() -> tuple[str, ...]:
    """Every hook short name in the catalog, in catalog order."""
    return tuple(short for _id, _act, short, _pl, _g, _tp, _sc, _pp in HOOK_DEFS)


def per_layer_hook_ids() -> frozenset[str]:
    """Hooks that exist once per layer, and so respect a layer range."""
    return frozenset(
        short
        for _id, _act, short, per_layer, _g, _tp, _sc, _pp in HOOK_DEFS
        if per_layer
    )


def describe_hooks(topology: Optional[ModelTopology] = None) -> list[HookInfo]:
    """Describe every catalog hook, flagging what this model cannot produce.

    Passing ``topology=None`` yields the model-independent catalog with
    everything marked available.
    """
    described: list[HookInfo] = []
    for _id, _act, short, per_layer, group, _tp, _sc, _pp in HOOK_DEFS:
        ui_group, label = _presentation_for(short, group)
        reason = _unavailable_reason(short, topology)
        described.append(
            HookInfo(
                id=short,
                label=label,
                group=ui_group,
                per_layer=per_layer,
                available=reason is None,
                reason=reason,
            )
        )
    return described


def grouped_hooks(topology: Optional[ModelTopology] = None) -> dict[str, list[HookInfo]]:
    """``describe_hooks`` bucketed by presentation group, in render order."""
    buckets: dict[str, list[HookInfo]] = {group: [] for group in GROUP_ORDER}
    for info in describe_hooks(topology):
        buckets.setdefault(info.group, []).append(info)
    return {group: hooks for group, hooks in buckets.items() if hooks}


def catalog_payload(topology: Optional[ModelTopology] = None) -> dict:
    """JSON-ready catalog for ``GET /api/catalog``."""
    return {
        "groups": [
            {
                "id": group,
                "label": GROUP_LABELS.get(group, group),
                "hooks": [info.to_dict() for info in hooks],
            }
            for group, hooks in grouped_hooks(topology).items()
        ]
    }


__all__ = [
    "GROUP_ATTENTION",
    "GROUP_MLP_UI",
    "GROUP_MOE",
    "GROUP_RESIDUAL",
    "GROUP_GLOBAL",
    "GROUP_ORDER",
    "GROUP_LABELS",
    "HookInfo",
    "hook_ids",
    "per_layer_hook_ids",
    "describe_hooks",
    "grouped_hooks",
    "catalog_payload",
]
