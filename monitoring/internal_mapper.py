"""Retrieve captured internals from the store and present them like HF's
model output: one field per internal, mirroring its HF counterpart.

Backend-agnostic -- it works off the ClickHouse rows DMI writes, regardless of
whether the run came from the HF or vLLM path.

Adding an internal: write a reassembler ``rows -> value`` (or reuse one) and add
a ``field -> (act_name, reassembler)`` entry to ``_FIELDS``. ``get_internal``
needs no change.
"""
from __future__ import annotations

import os

import torch

from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly


def _default_reader() -> CHClickhouseDriverReadOnly:
    return CHClickhouseDriverReadOnly(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
    )


def _left_pad_stack(per_request: list[torch.Tensor]) -> torch.Tensor:
    """Stack ragged per-request tensors [seq_i, hidden] into [batch, seq, hidden],
    left-padding shorter requests with zeros so real tokens stay right-aligned --
    the layout HF's left-padded batch produces."""
    seq = max(t.shape[0] for t in per_request)
    batched = torch.zeros(len(per_request), seq, per_request[0].shape[1],
                          dtype=per_request[0].dtype)
    for i, t in enumerate(per_request):
        batched[i, seq - t.shape[0]:] = t
    return batched


def _reassemble_per_layer(rows: list) -> tuple[torch.Tensor, ...]:
    """Reassemble a per-layer hook (residual stream, mlp, ...).

    rows: (key, tensor) pairs for one act_name, where
    key = (model_id, request_id, act_name, layer_no, shard_rank, start, end).
    Group by layer, concatenate each request's chunks along the token axis, then
    left-pad-stack the requests. Returns a tuple ordered by layer."""
    layers: dict[int, dict[str, list]] = {}
    for key, tensor in rows:
        layers.setdefault(key[3], {}).setdefault(key[1], []).append((key[5], tensor))
    out = []
    for layer in sorted(layers):
        per_request = [
            torch.cat([t for _, t in sorted(chunks)], dim=0)
            for _, chunks in sorted(layers[layer].items())
        ]
        out.append(_left_pad_stack(per_request))
    return tuple(out)


# Public field name -> (capture act_name, reassembler). The reassembler takes the
# (key, tensor) rows for its act_name and returns the field value. Add new
# internals (attention, logits, kv, ...) here, each with its own reassembler.
_FIELDS = {
    "hidden_states": ("blocks.hook_resid_pre", _reassemble_per_layer),
}


class Internal:
    """Captured internals for one run, presented like HF's model output: each
    field mirrors its HF counterpart (e.g. ``hidden_states`` is a tuple indexed
    by layer, each entry [batch, seq, hidden]). Only fields actually captured are
    present -- see ``available``; accessing an uncaptured field raises."""

    def __init__(self, fields: dict):
        self._fields = fields

    @property
    def available(self) -> list[str]:
        return sorted(self._fields)

    def __getattr__(self, name: str):
        fields = self.__dict__.get("_fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(
            f"{name!r} was not captured in this run. Available: {sorted(fields)}. "
            f"Pass it via hook_selection= when generating."
        )


def get_internal(source, reader: CHClickhouseDriverReadOnly | None = None) -> Internal:
    """Retrieve a run's captured internals.

    ``source`` is either the ``generate_with_monitoring`` output (its
    ``model_id`` is used) or a ``model_id`` string. ``reader`` defaults to a
    local ClickHouse connection (``DMX_DB_HOST`` / ``DMX_DB_PORT``); pass one to
    read a run from another process or host."""
    model_id = getattr(source, "model_id", source)
    reader = reader or _default_reader()
    rows_by_act: dict[str, list] = {}
    for key, tensor in reader.prefix_get((model_id,)):
        rows_by_act.setdefault(key[2], []).append((key, tensor))
    fields = {
        field: reassemble(rows_by_act[act])
        for field, (act, reassemble) in _FIELDS.items()
        if act in rows_by_act
    }
    return Internal(fields)
