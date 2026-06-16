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
import time
from dataclasses import dataclass

import torch

from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
from monitoring.segment_merger import merge_segments


def _default_reader() -> CHClickhouseDriverReadOnly:
    return CHClickhouseDriverReadOnly(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
    )


def _left_pad_stack(per_request: list[torch.Tensor]) -> torch.Tensor:
    """Stack ragged per-request tensors [seq_i, ...] into [batch, seq, ...],
    left-padding shorter requests with zeros so real tokens stay right-aligned --
    the layout HF's left-padded batch produces."""
    seq = max(t.shape[0] for t in per_request)
    batched = torch.zeros(len(per_request), seq, *per_request[0].shape[1:],
                          dtype=per_request[0].dtype)
    for i, t in enumerate(per_request):
        batched[i, seq - t.shape[0]:] = t
    return batched


def _request_sort_key(request_id: str) -> tuple:
    parts = request_id.split(":")
    key = []
    for part in parts:
        try:
            key.append((0, int(part)))
        except ValueError:
            key.append((1, part))
    return tuple(key)


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
            for _, chunks in sorted(layers[layer].items(), key=lambda item: _request_sort_key(item[0]))
        ]
        out.append(_left_pad_stack(per_request))
    return tuple(out)


def _left_pad_stack_attention(per_request: list[torch.Tensor]) -> torch.Tensor:
    """Stack ragged per-request attention tensors into [batch, heads, seq, seq].

    Each request tensor is [heads, query_tokens, key_tokens]. Shorter requests
    are left-padded on both query and key axes.
    """
    heads = per_request[0].shape[0]
    seq = max(t.shape[-1] for t in per_request)
    batched = torch.zeros(len(per_request), heads, seq, seq,
                          dtype=per_request[0].dtype)
    for i, t in enumerate(per_request):
        q_len = t.shape[-2]
        k_len = t.shape[-1]
        batched[i, :, seq - q_len:, seq - k_len:] = t
    return batched


def _reassemble_attention_per_layer(rows: list, act_name: str) -> tuple[torch.Tensor, ...]:
    """Reassemble attention-matrix rows as a tuple ordered by layer."""
    layers: dict[int, dict[str, list]] = {}
    for key, tensor in rows:
        layers.setdefault(key[3], {}).setdefault(key[1], []).append((key[5], tensor))
    out = []
    for layer in sorted(layers):
        per_request = [
            merge_segments(
                [t for _, t in sorted(chunks)],
                act_name,
            )
            for _, chunks in sorted(layers[layer].items(), key=lambda item: _request_sort_key(item[0]))
        ]
        out.append(_left_pad_stack_attention(per_request))
    return tuple(out)


def _attention_reassembler(act_name: str):
    def _reassemble(rows: list) -> tuple[torch.Tensor, ...]:
        return _reassemble_attention_per_layer(rows, act_name)
    return _reassemble


def _reassemble_global(rows: list) -> torch.Tensor:
    """Reassemble a non-layered field into [batch, seq, ...]."""
    requests: dict[str, list] = {}
    for key, tensor in rows:
        requests.setdefault(key[1], []).append((key[5], tensor))
    per_request = [
        torch.cat([t for _, t in sorted(chunks)], dim=0)
        for _, chunks in sorted(requests.items(), key=lambda item: _request_sort_key(item[0]))
    ]
    if per_request[0].ndim == 1:
        seq = max(t.shape[0] for t in per_request)
        batched = torch.zeros(len(per_request), seq, dtype=per_request[0].dtype)
        for i, t in enumerate(per_request):
            batched[i, seq - t.shape[0]:] = t
        return batched
    return _left_pad_stack(per_request)


# Public field name -> (capture act_name, reassembler). The reassembler takes the
# (key, tensor) rows for its act_name and returns the field value. Add new
# internals (attention, logits, kv, ...) here, each with its own reassembler.
_FIELDS = {
    "attention_output": ("blocks.hook_attn_out", _reassemble_per_layer),
    "attention_scores": (
        "blocks.attn.hook_attn_scores",
        _attention_reassembler("blocks.attn.hook_attn_scores"),
    ),
    "attention_values": ("blocks.attn.hook_z", _reassemble_per_layer),
    "hidden_states": ("blocks.hook_resid_pre", _reassemble_per_layer),
    "attentions": (
        "blocks.attn.hook_pattern",
        _attention_reassembler("blocks.attn.hook_pattern"),
    ),
    "embeddings": ("hook_embed", _reassemble_global),
    "expert_ids": ("blocks.mlp.hook_topk_ids", _reassemble_per_layer),
    "expert_weights": ("blocks.mlp.hook_topk_weights", _reassemble_per_layer),
    "final_hidden": ("hook_final_ln", _reassemble_global),
    "final_residual": ("hook_resid_final", _reassemble_global),
    "k": ("blocks.attn.hook_k", _reassemble_per_layer),
    "ln1": ("blocks.hook_ln1", _reassemble_per_layer),
    "ln2": ("blocks.hook_ln2", _reassemble_per_layer),
    "logits": ("final_logits", _reassemble_global),
    "middle_residual": ("blocks.hook_resid_mid", _reassemble_per_layer),
    "mlp_activation": ("blocks.hook_mlp_post", _reassemble_per_layer),
    "mlp_input": ("blocks.hook_mlp_in", _reassemble_per_layer),
    "mlp_output": ("blocks.hook_mlp_out", _reassemble_per_layer),
    "position_embeddings": ("hook_pos_embed", _reassemble_global),
    "q": ("blocks.attn.hook_q", _reassemble_per_layer),
    "router_logits": ("blocks.mlp.hook_router_logits", _reassemble_per_layer),
    "token_ids": ("token_ids", _reassemble_global),
    "v": ("blocks.attn.hook_v", _reassemble_per_layer),
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


class IncompleteInternalError(RuntimeError):
    """Raised when a required internal field is present but incomplete."""


@dataclass(frozen=True)
class _Requirement:
    count: int
    retry: bool = False
    timeout_s: float | None = 30.0
    poll_s: float = 0.25


class InternalRequirements:
    """Reusable strictness policy for lazy internal fields.

    ``count`` validates ``len(field_value)``. For per-layer fields such as
    ``hidden_states``, that means layer count, not token completeness.
    """

    def __init__(self, counts: dict[str, int | _Requirement] | None = None) -> None:
        self._requirements = {
            field: self._coerce_requirement(value)
            for field, value in dict(counts or {}).items()
        }

    @staticmethod
    def _coerce_requirement(value: int | _Requirement) -> _Requirement:
        if isinstance(value, _Requirement):
            return value
        return _Requirement(count=int(value))

    def require(
        self,
        field: str,
        *,
        count: int,
        retry: bool = False,
        timeout_s: float | None = 30.0,
        poll_s: float = 0.25,
    ) -> "InternalRequirements":
        if count < 0:
            raise ValueError("count must be non-negative")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        self._requirements[field] = _Requirement(
            count=int(count),
            retry=bool(retry),
            timeout_s=timeout_s,
            poll_s=float(poll_s),
        )
        return self

    def copy(self) -> "InternalRequirements":
        return InternalRequirements(self._requirements)

    def expected_count(self, field: str) -> int | None:
        requirement = self._requirements.get(field)
        if requirement is None:
            return None
        return requirement.count

    def requirement(self, field: str) -> _Requirement | None:
        return self._requirements.get(field)


class _LazyInternal:
    """Lazy proxy for captured internals.

    Successful field access caches that field. Failed loads and incomplete
    fields are passed through unchanged and are not cached, so later accesses
    retry.
    """

    def __init__(
        self,
        model_id: str,
        reader: CHClickhouseDriverReadOnly | None = None,
        requirements: InternalRequirements | None = None,
        request_ids: tuple[str, ...] | list[str] | None = None,
        token_ranges: dict[str, tuple[tuple[int, int], ...] | list[tuple[int, int]]] | None = None,
    ) -> None:
        self._model_id = model_id
        self._reader = reader
        self._requirements = (
            requirements.copy() if requirements is not None else InternalRequirements()
        )
        self._request_ids = tuple(request_ids or ())
        self._token_ranges = {
            rid: tuple((int(start), int(end)) for start, end in ranges)
            for rid, ranges in dict(token_ranges or {}).items()
        }
        self._field_cache: dict[str, object] = {}

    def require(
        self,
        field: str,
        *,
        count: int,
        retry: bool = False,
        timeout_s: float | None = 30.0,
        poll_s: float = 0.25,
    ) -> "_LazyInternal":
        self._requirements.require(
            field,
            count=count,
            retry=retry,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
        return self

    def clear_cache(self, field: str | None = None) -> None:
        if field is None:
            self._field_cache.clear()
            return
        self._field_cache.pop(field, None)

    def _count_value(self, field: str, value: object) -> int:
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise IncompleteInternalError(
                f"{field} cannot be validated for model_id={self._model_id!r}: "
                "value has no length."
            ) from exc

    def _incomplete_error(
        self,
        field: str,
        requirement: _Requirement,
        *,
        found: int | None = None,
        timeout: bool = False,
        cause: Exception | None = None,
    ) -> IncompleteInternalError:
        if found is None:
            detail = f"expected {requirement.count} entries, found none"
        else:
            detail = f"expected {requirement.count} entries, found {found}"
        if timeout:
            timeout_text = (
                "without timeout"
                if requirement.timeout_s is None
                else f"within {requirement.timeout_s:.3g}s"
            )
            detail = f"{detail} {timeout_text}"
        message = (
            f"{field} is incomplete for model_id={self._model_id!r}: "
            f"{detail}."
        )
        error = IncompleteInternalError(message)
        error.field = field  # type: ignore[attr-defined]
        error.expected = requirement.count  # type: ignore[attr-defined]
        error.found = found  # type: ignore[attr-defined]
        if cause is not None:
            error.__cause__ = cause
        return error

    def _validate(
        self,
        field: str,
        value: object,
        requirement: _Requirement | None = None,
    ) -> None:
        requirement = requirement or self._requirements.requirement(field)
        if requirement is None:
            return
        found = self._count_value(field, value)
        if found != requirement.count:
            raise self._incomplete_error(field, requirement, found=found)

    def _load_field_once(
        self,
        field: str,
        requirement: _Requirement | None,
    ) -> object:
        value = self._read_field(field)
        self._validate(field, value, requirement)
        self._field_cache[field] = value
        return value

    def _reader_or_default(self) -> CHClickhouseDriverReadOnly:
        return self._reader or _default_reader()

    def _read_rows_for_field(self, field: str) -> list:
        if field not in _FIELDS:
            raise AttributeError(
                f"{field!r} is not a supported DMI internal field. "
                f"Available mapped fields: {sorted(_FIELDS)}."
            )
        act, _ = _FIELDS[field]
        reader = self._reader_or_default()
        if self._request_ids:
            rows = []
            for request_id in self._request_ids:
                rows.extend(reader.prefix_get((self._model_id, request_id, act)))
            return rows
        return [
            (key, tensor)
            for key, tensor in reader.prefix_get((self._model_id,))
            if key[2] == act
        ]

    def _read_field(self, field: str) -> object:
        if field == "token_mask":
            return self._build_token_mask()
        if field not in _FIELDS:
            raise AttributeError(
                f"{field!r} was not captured because it is not a supported "
                "DMI internal field. "
                f"Available mapped fields: {sorted(_FIELDS) + ['token_mask']}."
            )
        act, reassemble = _FIELDS[field]
        rows = self._read_rows_for_field(field)
        if not rows:
            raise AttributeError(
                f"{field!r} was not captured in this run. "
                f"Pass the corresponding hook via hook_selection= when generating."
            )
        return reassemble(rows)

    def _build_token_mask(self) -> torch.Tensor:
        if not self._request_ids or not self._token_ranges:
            raise AttributeError(
                "'token_mask' is not available because this internal handle "
                "does not have per-generate request token ranges."
            )
        max_len = 0
        for rid in self._request_ids:
            ranges = self._token_ranges.get(rid, ())
            for _, end in ranges:
                max_len = max(max_len, int(end))
        mask = torch.zeros(len(self._request_ids), max_len, dtype=torch.bool)
        for batch_i, rid in enumerate(self._request_ids):
            ranges = self._token_ranges.get(rid, ())
            final_len = max((int(end) for _, end in ranges), default=0)
            offset = max_len - final_len
            for start, end in ranges:
                start_i = int(start)
                end_i = int(end)
                if end_i > start_i:
                    mask[batch_i, offset + start_i: offset + end_i] = True
        return mask

    def _load_field_with_retry(
        self,
        field: str,
        requirement: _Requirement,
    ) -> object:
        deadline = (
            None
            if requirement.timeout_s is None
            else time.monotonic() + requirement.timeout_s
        )
        last_error: Exception | None = None
        last_found: int | None = None

        while True:
            try:
                return self._load_field_once(field, requirement)
            except IncompleteInternalError as exc:
                last_error = exc
                last_found = getattr(exc, "found", None)
                if field in self._field_cache:
                    self._field_cache.pop(field, None)
            except AttributeError as exc:
                last_error = exc

            if deadline is not None and time.monotonic() >= deadline:
                raise self._incomplete_error(
                    field,
                    requirement,
                    found=last_found,
                    timeout=True,
                    cause=last_error,
                )
            time.sleep(requirement.poll_s)

    def _load_field(self, field: str) -> object:
        requirement = self._requirements.requirement(field)
        if field in self._field_cache:
            value = self._field_cache[field]
            try:
                self._validate(field, value, requirement)
            except Exception:
                self._field_cache.pop(field, None)
                if requirement is None or not requirement.retry:
                    raise
                return self._load_field_with_retry(field, requirement)
            return value
        if requirement is not None and requirement.retry:
            return self._load_field_with_retry(field, requirement)
        return self._load_field_once(field, requirement)

    @property
    def available(self) -> list[str]:
        if not self._request_ids:
            return get_internal(self._model_id, self._reader).available
        fields = []
        for field in sorted(_FIELDS):
            if self._read_rows_for_field(field):
                fields.append(field)
        if self._token_ranges:
            fields.append("token_mask")
        return fields

    def __getattr__(self, name: str):
        return self._load_field(name)

    def __repr__(self) -> str:
        state = f"cached={sorted(self._field_cache)}" if self._field_cache else "pending"
        return f"<DMIInternal model_id={self._model_id!r} state={state}>"


def make_lazy_internal(
    model_id: str,
    reader: CHClickhouseDriverReadOnly | None = None,
    requirements: InternalRequirements | None = None,
    request_ids: tuple[str, ...] | list[str] | None = None,
    token_ranges: dict[str, tuple[tuple[int, int], ...] | list[tuple[int, int]]] | None = None,
) -> _LazyInternal:
    return _LazyInternal(
        model_id,
        reader,
        requirements=requirements,
        request_ids=request_ids,
        token_ranges=token_ranges,
    )


def get_internal(model_id: str, reader: CHClickhouseDriverReadOnly | None = None) -> Internal:
    """Retrieve a run's captured internals.

    ``model_id`` identifies the captured run. ``reader`` defaults to a local
    ClickHouse connection (``DMX_DB_HOST`` / ``DMX_DB_PORT``); pass one to read
    a run from another process or host.
    """
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
