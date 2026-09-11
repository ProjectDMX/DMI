"""Bridge structured observations to DMI's string-based hook selection.

The existing interface is ``BackendAdapter.attach_model(model,
hook_selection: str)``, where the selection is a comma-separated list of hook
names and presets ("q,k,v,pattern", "vllm-full"). That interface is not going
away -- integrations pass it today, notably as the vLLM ``additional_config``
key ``dmx_hook_selection``.

Everything backwards-compatible lives here, so the rest of the configuration
layer never has to think in strings.
"""

from __future__ import annotations

from .schema import ObservationConfig


def to_legacy_hook_selection(observations: ObservationConfig) -> str:
    """Render structured observations as a selection string.

    Only the hook set survives -- a selection string cannot express a layer
    range, which is exactly why layer filtering is a separate spec-level
    operation (``dmi.hooks.selection.filter_by_layers``) rather than invented
    syntax such as ``pattern@8-15``.
    """
    if not observations.hooks:
        raise ValueError(
            "Cannot build a hook selection from an empty observation set."
        )
    # dict.fromkeys keeps first-seen order while removing duplicates; the
    # selector unions tokens anyway, but a clean string is easier to read in
    # logs and integration config.
    return ",".join(dict.fromkeys(observations.hooks))


def from_legacy_hook_selection(selection: str) -> ObservationConfig:
    """Expand a selection string into structured observations.

    Presets are resolved through the existing selector, so ``"vllm-full"``
    expands to the hooks that preset actually means in this build rather than
    to a copy of the table maintained here.

    The result carries ``layers=None``: a selection string never constrained
    layers, and inventing a range would change what the configuration means.
    """
    from ..hooks.catalog import HOOK_DEFS  # local: keeps import cost off callers
    from ..hooks.selection import resolve_hook_selection

    short_by_id = {
        hook_id: short for hook_id, _act, short, _pl, _g, _tp, _sc, _pp in HOOK_DEFS
    }
    catalog_order = {short: index for index, short in enumerate(short_by_id.values())}

    hook_types = resolve_hook_selection(selection)
    hooks = sorted(
        (short_by_id[hook_type] for hook_type in hook_types),
        key=catalog_order.__getitem__,
    )
    return ObservationConfig(hooks=hooks, layers=None)


__all__ = ["to_legacy_hook_selection", "from_legacy_hook_selection"]
