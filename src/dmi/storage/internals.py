"""Retrieve captured internals from storage and expose HF-style fields.
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

from .clickhouse import CHClickhouseDriverReadOnly
from .reassembly import merge_segments


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


# k and v are the one sharded class whose TP split cannot be identified from
# the stored rows -- see the refusal in `_merged_shard_chunks`.
_KV_ACTS = ("attn.hook_k", "attn.hook_v")


def _shard_axis(act_name: str) -> int:
    """The axis a TP shard of ``act_name`` was cut along.

    Derived from where the TOKEN axis sits, which is what
    ``reassembly.segment_manager`` already encodes -- so the two stay one
    decision rather than two that can drift apart:

    * An attention matrix row is ``[heads, q_tokens, kv_tokens]``: that manager
      uses ``token_dim_incremental=2-dim_diff`` and
      ``token_dim_sum_to_now=3-dim_diff``, i.e. dims 1 and 2 once the batch dim
      is absent. TP splits HEADS, so the join is dim 0.
    * Every other row is token-major (``token_dim=1-dim_diff`` = dim 0), and TP
      splits the axis immediately after tokens -- dim 1.

    Dim 1 is deliberately NOT spelled ``-1``. ``compute_hook_shape``
    (``dmi/hooks/specs.py``) stores q, k, v and batched z as
    ``[tokens, heads, head_dim]``, where the split axis is heads in the MIDDLE.
    Only packed z and mlp_post are two-dimensional, and there dim 1 already IS
    the trailing axis. Spelling it -1 joins head_dim on the three-dimensional
    layouts, producing a tensor with the right element count, the wrong shape
    and spliced heads -- and raising nothing.
    """
    if act_name.endswith(("attn.hook_attn_scores", "attn.hook_pattern")):
        return 0
    return 1


def _ordered_chunks(
    chunks: list,
    *,
    act_name: str,
    request_id: str,
    merge_shards: bool = False,
) -> list:
    """One request's chunks in token order, refusing a duplicate start token.

    Sorting ``(start_token, tensor)`` pairs without an explicit key makes
    Python fall through to comparing the TENSORS whenever two starts tie --
    which raises "Boolean value of Tensor with more than one value is
    ambiguous" for a normal payload, and, worse, compares fine for a
    one-element payload and silently concatenates the two colliding captures.

    A tie means the same (request, layer, start) arrived twice, which in
    practice means two TP shards: ``filter_by_tp_rank`` deliberately keeps
    sharded hooks on every rank, and all ranks write one table with
    ``shard_rank`` telling the rows apart. Reassembling along the token axis
    cannot merge shards -- they are the same tokens, not more of them -- so
    refusing is the default answer. The caller picks a ``shard_rank``, exactly
    as the repo's own comparison tooling does, or opts in to ``merge_shards``
    to join the slices back along their own axis (see ``_shard_axis``).
    """
    # One pass over the starts, not ``starts.count(s)`` per start: a long
    # chunked capture (hundreds of rows per request/layer) otherwise paid
    # O(n^2) to build a diagnostic that only names the repeats.
    by_start: dict[int, list] = {}
    for start, rank, tensor in chunks:
        by_start.setdefault(start, []).append((rank, tensor))
    duplicated = sorted(start for start, group in by_start.items() if len(group) > 1)
    if duplicated and merge_shards:
        return _merged_shard_chunks(
            by_start, act_name=act_name, request_id=request_id)
    if duplicated:
        raise RuntimeError(
            f"{act_name}: duplicate capture chunks for request {request_id!r} "
            f"at start token(s) {duplicated} -- the same tokens were captured "
            "more than once, which usually means rows from several TP ranks "
            "(distinguished by shard_rank) or from separate runs sharing one "
            "model_id. Reassembly cannot merge them along the token axis; "
            "select a single shard_rank before reading, or pass "
            "merge_shards=True to join TP shards along their own axis."
        )
    return [group[0][1] for _, group in sorted(by_start.items())]


def _merged_shard_chunks(
    by_start: dict, *, act_name: str, request_id: str
) -> list:
    """Join each start token's TP shards into one tensor, in rank order.

    Ordering by ``shard_rank`` is what makes the result match the unsharded
    tensor: rank r holds the r-th slice of the split axis, so concatenating in
    rank order is the inverse of the split. A dict is not ordered by rank, and
    row arrival order is whatever the reader returned, so the sort is load
    bearing rather than tidiness.
    """
    if act_name.endswith(_KV_ACTS):
        # K and V are the one class where "one row per rank" does not imply a
        # split. `compute_hook_shape` computes
        # `kv_heads = max(1, cfg.num_kv_heads // tp)`, so a GQA model with
        # fewer KV heads than ranks gives EVERY rank the same kv head. The
        # stored rows are identical in shape either way, and the row key
        # carries neither `num_kv_heads` nor `tp_size`, so nothing here can
        # tell a genuine split from replication. Joining a replicated K would
        # return a tensor with duplicated heads that the model never produced,
        # which is worse than refusing.
        raise RuntimeError(
            f"{act_name}: request {request_id!r} cannot be merged across TP "
            "ranks -- GQA may replicate the KV heads rather than split them "
            "(kv_heads = max(1, num_kv_heads // tp_size)), and the stored rows "
            "do not record num_kv_heads or tp_size, so a replicated K/V is "
            "indistinguishable from a split one. Merging would duplicate "
            "heads. Select a single shard_rank instead."
        )
    axis = _shard_axis(act_name)
    out = []
    rank_tuples = set()
    for start in sorted(by_start):
        group = sorted(by_start[start], key=lambda rank_tensor: rank_tensor[0])
        ranks = tuple(rank for rank, _ in group)
        if len(set(ranks)) != len(ranks):
            # Two rows from the SAME rank are not shards of one tensor --
            # that is the "separate runs sharing one model_id" case, and
            # concatenating them would fabricate a wider tensor than the
            # model ever produced. Refuse it even under merge_shards.
            raise RuntimeError(
                f"{act_name}: request {request_id!r} has repeated shard_rank "
                f"{sorted(r for r in ranks if ranks.count(r) > 1)} at start "
                f"token {start} -- merge_shards joins one row per rank, so "
                "this is a duplicate capture (separate runs sharing one "
                "model_id), not a TP split. Select a single shard_rank."
            )
        rank_tuples.add(ranks)
        out.append(group[0][1] if len(group) == 1
                   else torch.cat([tensor for _, tensor in group], dim=axis))
    if len(rank_tuples) > 1:
        # Every start token must carry the same rank set, or the merged
        # tensors disagree on width and the token-axis concatenation that
        # follows would either throw or silently ragged-join.
        raise RuntimeError(
            f"{act_name}: request {request_id!r} has inconsistent shard_rank "
            f"sets across start tokens ({sorted(rank_tuples)}) -- some tokens "
            "are missing a rank's rows, so the shards cannot be joined into "
            "one tensor. Select a single shard_rank."
        )
    return out


def _reassemble_per_layer(rows: list, *, merge_shards: bool = False) -> tuple[torch.Tensor, ...]:
    """Reassemble a per-layer hook (residual stream, mlp, ...).

    rows: (key, tensor) pairs for one act_name, where
    key = (model_id, request_id, act_name, layer_no, shard_rank, start, end).
    Group by layer, concatenate each request's chunks along the token axis, then
    left-pad-stack the requests. Returns a tuple ordered by layer."""
    layers: dict[int, dict[str, list]] = {}
    # key[2] is the act_name; every row here belongs to one act by construction.
    act_name = rows[0][0][2] if rows else "<unknown act>"
    for key, tensor in rows:
        layers.setdefault(key[3], {}).setdefault(key[1], []).append(
            (key[5], key[4], tensor))
    out = []
    for layer in sorted(layers):
        per_request = [
            torch.cat(
                _ordered_chunks(chunks, act_name=act_name, request_id=request_id,
                                merge_shards=merge_shards),
                dim=0,
            )
            for request_id, chunks in sorted(layers[layer].items(), key=lambda item: _request_sort_key(item[0]))
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


def _reassemble_attention_per_layer(
    rows: list, act_name: str, *, merge_shards: bool = False
) -> tuple[torch.Tensor, ...]:
    """Reassemble attention-matrix rows as a tuple ordered by layer."""
    layers: dict[int, dict[str, list]] = {}
    for key, tensor in rows:
        layers.setdefault(key[3], {}).setdefault(key[1], []).append(
            (key[5], key[4], tensor))
    out = []
    for layer in sorted(layers):
        per_request = [
            merge_segments(
                _ordered_chunks(chunks, act_name=act_name, request_id=request_id,
                                merge_shards=merge_shards),
                act_name,
            )
            for request_id, chunks in sorted(layers[layer].items(), key=lambda item: _request_sort_key(item[0]))
        ]
        out.append(_left_pad_stack_attention(per_request))
    return tuple(out)


def _attention_reassembler(act_name: str):
    def _reassemble(rows: list, *, merge_shards: bool = False) -> tuple[torch.Tensor, ...]:
        return _reassemble_attention_per_layer(
            rows, act_name, merge_shards=merge_shards)
    return _reassemble


def _reassemble_global(rows: list, *, merge_shards: bool = False) -> torch.Tensor:
    """Reassemble a non-layered field into [batch, seq, ...]."""
    requests: dict[str, list] = {}
    act_name = rows[0][0][2] if rows else "<unknown act>"
    for key, tensor in rows:
        requests.setdefault(key[1], []).append((key[5], key[4], tensor))
    per_request = [
        torch.cat(
            _ordered_chunks(chunks, act_name=act_name, request_id=request_id,
                            merge_shards=merge_shards),
            dim=0,
        )
        for request_id, chunks in sorted(requests.items(), key=lambda item: _request_sort_key(item[0]))
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
class InternalRequirement:
    count: int
    retry: bool = False
    timeout_s: float | None = 30.0
    poll_s: float = 0.25
    match_token_ranges: bool = False


# Backward-compatible private name used before integration API v1.
_Requirement = InternalRequirement


class InternalRequirements:
    """Reusable strictness policy for lazy internal fields.

    ``count`` validates ``len(field_value)``. For per-layer fields such as
    ``hidden_states``, that means layer count, not token completeness.
    ``match_token_ranges=True`` additionally validates that captured row ranges
    match this generate call's expected request token ranges.
    """

    def __init__(
        self,
        counts: dict[str, int | InternalRequirement] | None = None,
    ) -> None:
        self._requirements = {
            field: self._coerce_requirement(value)
            for field, value in dict(counts or {}).items()
        }

    @staticmethod
    def _coerce_requirement(
        value: int | InternalRequirement,
    ) -> InternalRequirement:
        if isinstance(value, InternalRequirement):
            return value
        return InternalRequirement(count=int(value))

    def require(
        self,
        field: str,
        *,
        count: int,
        retry: bool = False,
        timeout_s: float | None = 30.0,
        poll_s: float = 0.25,
        match_token_ranges: bool = False,
    ) -> "InternalRequirements":
        if count < 0:
            raise ValueError("count must be non-negative")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        self._requirements[field] = InternalRequirement(
            count=int(count),
            retry=bool(retry),
            timeout_s=timeout_s,
            poll_s=float(poll_s),
            match_token_ranges=bool(match_token_ranges),
        )
        return self

    def copy(self) -> "InternalRequirements":
        return InternalRequirements(self._requirements)

    def expected_count(self, field: str) -> int | None:
        requirement = self._requirements.get(field)
        if requirement is None:
            return None
        return requirement.count

    def requirement(self, field: str) -> InternalRequirement | None:
        return self._requirements.get(field)


class LazyInternal:
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
        shard_rank: int | None = None,
        merge_shards: bool = False,
    ) -> None:
        if shard_rank is not None and merge_shards:
            raise ValueError(
                "shard_rank and merge_shards are mutually exclusive: "
                "shard_rank selects one TP rank's slice, merge_shards joins "
                "every rank's slices into the full tensor."
            )
        self._model_id = model_id
        self._reader = reader
        self._shard_rank = shard_rank
        self._merge_shards = merge_shards
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
        match_token_ranges: bool = False,
    ) -> "LazyInternal":
        self._requirements.require(
            field,
            count=count,
            retry=retry,
            timeout_s=timeout_s,
            poll_s=poll_s,
            match_token_ranges=match_token_ranges,
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
        requirement: InternalRequirement,
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
        requirement: InternalRequirement | None = None,
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
        requirement: InternalRequirement | None,
    ) -> object:
        value = self._read_field(field, requirement)
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
        else:
            rows = [
                (key, tensor)
                for key, tensor in reader.prefix_get((self._model_id,))
                if key[2] == act
            ]
        if self._shard_rank is not None:
            rows = [(key, tensor) for key, tensor in rows
                    if key[4] == self._shard_rank]
        return rows

    def _expected_non_empty_ranges(self, request_id: str) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(start), int(end))
            for start, end in self._token_ranges.get(request_id, ())
            if int(end) > int(start)
        )

    def _actual_ranges_for_request(
        self,
        rows: list,
        request_id: str,
        layer: int | None,
    ) -> tuple[tuple[int, int], ...]:
        ranges = {
            (int(key[5]), int(key[6]))
            for key, _ in rows
            if key[1] == request_id and (layer is None or int(key[3]) == layer)
            and int(key[6]) > int(key[5])
        }
        return tuple(sorted(ranges))

    def _validate_token_ranges(
        self,
        field: str,
        rows: list,
        requirement: InternalRequirement | None,
    ) -> None:
        if (
            requirement is None
            or not requirement.match_token_ranges
            or not self._request_ids
            or not self._token_ranges
        ):
            return
        layers = sorted({int(key[3]) for key, _ in rows if int(key[3]) >= 0})
        # Fast check for per-layer fields: validating the last layer catches the
        # common "writer still pending" case without scanning every layer.
        check_layer = layers[-1] if layers else None
        for request_id in self._request_ids:
            expected = self._expected_non_empty_ranges(request_id)
            actual = self._actual_ranges_for_request(rows, request_id, check_layer)
            if actual != expected:
                raise IncompleteInternalError(
                    f"{field} token ranges are incomplete for "
                    f"model_id={self._model_id!r}, request_id={request_id!r}: "
                    f"expected {list(expected)}, found {list(actual)}."
                )

    def _read_field(
        self,
        field: str,
        requirement: InternalRequirement | None = None,
    ) -> object:
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
        self._validate_token_ranges(field, rows, requirement)
        return reassemble(rows, merge_shards=self._merge_shards)

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
        requirement: InternalRequirement,
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
            return get_internal(
                self._model_id, self._reader, shard_rank=self._shard_rank,
                merge_shards=self._merge_shards,
            ).available
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


# Backward-compatible private name used before integration API v1.
_LazyInternal = LazyInternal


def make_lazy_internal(
    model_id: str,
    reader: CHClickhouseDriverReadOnly | None = None,
    requirements: InternalRequirements | None = None,
    request_ids: tuple[str, ...] | list[str] | None = None,
    token_ranges: dict[str, tuple[tuple[int, int], ...] | list[tuple[int, int]]] | None = None,
    shard_rank: int | None = None,
    merge_shards: bool = False,
) -> LazyInternal:
    return LazyInternal(
        model_id,
        reader,
        requirements=requirements,
        request_ids=request_ids,
        token_ranges=token_ranges,
        shard_rank=shard_rank,
        merge_shards=merge_shards,
    )


def get_internal(
    model_id: str,
    reader: CHClickhouseDriverReadOnly | None = None,
    shard_rank: int | None = None,
    merge_shards: bool = False,
) -> Internal:
    """Retrieve a run's captured internals.

    ``model_id`` identifies the captured run. ``reader`` defaults to a local
    ClickHouse connection (``DMX_DB_HOST`` / ``DMX_DB_PORT``); pass one to read
    a run from another process or host.

    Under tensor parallelism a sharded activation is written once per rank,
    every rank holding a different slice of the same tokens. The two arguments
    below are the two ways to read such a run, and they are mutually
    exclusive:

    ``shard_rank`` keeps one rank's rows and returns THAT RANK'S SLICE -- a
    tensor narrower than the model's, on the split axis. None keeps every row,
    in which case reassembly refuses the collision by name rather than
    guessing.

    ``merge_shards=True`` joins every rank's slices back into the full tensor,
    concatenating in ascending ``shard_rank`` order along the axis TP split:
    dim 0 (heads) for an attention matrix, and otherwise the axis right after
    tokens -- heads for the three-dimensional q and batched z, the feature axis
    for two-dimensional packed z and mlp_post. ``k`` and ``v`` are refused,
    because GQA may replicate KV heads across ranks rather than split them and
    the stored rows cannot tell the two apart. Unsharded fields are
    unaffected, since they have no
    collision to join.
    """
    reader = reader or _default_reader()
    if shard_rank is not None and merge_shards:
        raise ValueError(
            "shard_rank and merge_shards are mutually exclusive: "
            "shard_rank selects one TP rank's slice, merge_shards joins "
            "every rank's slices into the full tensor."
        )
    rows_by_act: dict[str, list] = {}
    for key, tensor in reader.prefix_get((model_id,)):
        if shard_rank is not None and key[4] != shard_rank:
            continue
        rows_by_act.setdefault(key[2], []).append((key, tensor))
    fields = {
        field: reassemble(rows_by_act[act], merge_shards=merge_shards)
        for field, (act, reassemble) in _FIELDS.items()
        if act in rows_by_act
    }
    return Internal(fields)
