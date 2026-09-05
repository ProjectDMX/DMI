"""Structural metadata that drives the architecture visualization.

The diagram represents *configurable semantic objects*, not every tensor edge
in a real transformer. Each node bundles the observations a user thinks of
together -- "Attention", "MLP", "Final norm & logits" -- and the renderer draws
whatever this module describes. Adding an architecture type means adding a node
list here, not editing SVG.

Nodes are a presentation concept and deliberately do not match catalog groups
one-to-one: the catalog's ``GROUP_OTHER`` splits across the input node, the
residual node, and the output node depending on where users look for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .catalog_adapter import describe_hooks
from .schema import ModelDescriptor, ModelTopology

SCOPE_GLOBAL = "global"
SCOPE_LAYER = "layer"


@dataclass(frozen=True)
class ArchitectureNode:
    """One selectable block in the diagram."""

    id: str
    label: str
    scope: str
    hooks: tuple[str, ...]


_DECODER_NODES: tuple[ArchitectureNode, ...] = (
    ArchitectureNode(
        id="input",
        label="Input & embedding",
        scope=SCOPE_GLOBAL,
        hooks=("token_ids", "embed", "pos_embed"),
    ),
    ArchitectureNode(
        id="residual",
        label="Residual stream",
        scope=SCOPE_LAYER,
        hooks=("resid_pre", "ln1", "resid_mid", "ln2"),
    ),
    ArchitectureNode(
        id="attention",
        label="Attention",
        scope=SCOPE_LAYER,
        hooks=("q", "k", "v", "pattern", "attn_scores", "z", "attn_out"),
    ),
    ArchitectureNode(
        id="mlp",
        label="MLP",
        scope=SCOPE_LAYER,
        hooks=("mlp_in", "mlp_post", "mlp_out"),
    ),
    ArchitectureNode(
        id="moe",
        label="MoE router",
        scope=SCOPE_LAYER,
        hooks=("router_logits", "topk_ids", "topk_weights"),
    ),
    ArchitectureNode(
        id="output",
        label="Final norm & logits",
        scope=SCOPE_GLOBAL,
        hooks=("resid_final", "final_ln", "final_logits"),
    ),
)

_NODES_BY_ARCHITECTURE = {"decoder_transformer": _DECODER_NODES}


def nodes_for(architecture: str) -> tuple[ArchitectureNode, ...]:
    try:
        return _NODES_BY_ARCHITECTURE[architecture]
    except KeyError as exc:
        raise ValueError(
            f"No architecture layout registered for {architecture!r}. "
            f"Known: {', '.join(sorted(_NODES_BY_ARCHITECTURE))}."
        ) from exc


def architecture_payload(descriptor: ModelDescriptor) -> dict:
    """JSON-ready diagram description for one model.

    Every catalog hook appears in exactly one node. Any hook with no assigned
    node is collected into a trailing "Other observations" node, so extending
    ``HOOK_DEFS`` can never make an observation unreachable from the UI.

    Nodes a model cannot produce are still emitted, with ``available: False``
    and a per-hook reason. A dense model shows a greyed-out MoE block rather
    than silently omitting it, so the absence is explained rather than
    mysterious.
    """
    topology: ModelTopology = descriptor.topology
    info_by_id = {info.id: info for info in describe_hooks(topology)}

    payload_nodes = []
    assigned: set[str] = set()

    for node in nodes_for(descriptor.model.architecture):
        hooks = [info_by_id[h] for h in node.hooks if h in info_by_id]
        assigned.update(info.id for info in hooks)
        if not hooks:
            continue
        payload_nodes.append(
            {
                "id": node.id,
                "label": node.label,
                "scope": node.scope,
                "available": any(info.available for info in hooks),
                "hooks": [info.to_dict() for info in hooks],
            }
        )

    leftovers = [info for hook_id, info in info_by_id.items() if hook_id not in assigned]
    if leftovers:
        # Split by scope rather than assuming per-layer. The renderer draws
        # layer-scoped nodes inside the "Transformer layer x N" group, so a
        # global hook landing here with SCOPE_LAYER would look as though it
        # were captured once per layer and as though the layer range applied
        # to it.
        for scope, label in (
            (SCOPE_LAYER, "Other observations"),
            (SCOPE_GLOBAL, "Other model-wide observations"),
        ):
            in_scope = [
                info
                for info in leftovers
                if (SCOPE_LAYER if info.per_layer else SCOPE_GLOBAL) == scope
            ]
            if not in_scope:
                continue
            payload_nodes.append(
                {
                    "id": "other" if scope == SCOPE_LAYER else "other-global",
                    "label": label,
                    "scope": scope,
                    "available": any(info.available for info in in_scope),
                    "hooks": [info.to_dict() for info in in_scope],
                }
            )

    return {
        "architecture": descriptor.model.architecture,
        "num_layers": topology.num_layers,
        "is_moe": topology.is_moe,
        "nodes": payload_nodes,
    }


def model_payload(descriptor: ModelDescriptor) -> dict:
    """JSON-ready model description for ``GET /api/model``."""
    return {
        "id": descriptor.model.id,
        "name": descriptor.model.name,
        "architecture": descriptor.model.architecture,
        "topology": {
            "num_layers": descriptor.topology.num_layers,
            "hidden_size": descriptor.topology.hidden_size,
            "num_attention_heads": descriptor.topology.num_attention_heads,
            "num_kv_heads": descriptor.topology.num_kv_heads,
            "head_dim": descriptor.topology.effective_head_dim,
            "intermediate_size": descriptor.topology.intermediate_size,
            "num_experts": descriptor.topology.num_experts,
            "top_k": descriptor.topology.top_k,
            "vocab_size": descriptor.topology.vocab_size,
        },
        "architecture_layout": architecture_payload(descriptor),
    }


__all__ = [
    "SCOPE_GLOBAL",
    "SCOPE_LAYER",
    "ArchitectureNode",
    "nodes_for",
    "architecture_payload",
    "model_payload",
]
