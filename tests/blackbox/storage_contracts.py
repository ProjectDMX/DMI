"""Black-box contracts for DMI activation rows stored by vLLM runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class StorageRow:
    """Public storage fields needed to validate one DMI activation segment."""

    request_id: str
    act_name: str
    layer_no: int
    shard_rank: int
    start_token_idx: int
    end_token_idx: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def family(self) -> tuple[str, int, int]:
        return self.act_name, self.layer_no, self.shard_rank

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.request_id,
            *self.family,
            self.start_token_idx,
            self.end_token_idx,
        )


@dataclass(frozen=True)
class _FamilyContract:
    dtype: str
    shape_tail: tuple[int, ...]
    coverage: str


def _expand_contract(
    contract: dict[str, Any],
) -> dict[tuple[str, int, int], _FamilyContract]:
    num_layers = contract.get("num_layers")
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
        raise ValueError("storage contract num_layers must be a positive integer")
    hooks = contract.get("hooks")
    if not isinstance(hooks, list) or not hooks:
        raise ValueError("storage contract hooks must be a non-empty list")

    expanded: dict[tuple[str, int, int], _FamilyContract] = {}
    for index, hook in enumerate(hooks):
        if not isinstance(hook, dict):
            raise ValueError(f"storage contract hooks[{index}] must be an object")
        act_name = hook.get("act_name")
        if not isinstance(act_name, str) or not act_name:
            raise ValueError(f"storage contract hooks[{index}].act_name is invalid")
        layers = hook.get("layers")
        if layers == "all":
            layer_numbers = range(num_layers)
        elif isinstance(layers, list) and all(
            isinstance(layer, int) and not isinstance(layer, bool)
            for layer in layers
        ):
            layer_numbers = layers
        else:
            raise ValueError(f"storage contract hooks[{index}].layers is invalid")
        shard_ranks = hook.get("shard_ranks", [0])
        if not isinstance(shard_ranks, list) or not shard_ranks or not all(
            isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
            for rank in shard_ranks
        ):
            raise ValueError(
                f"storage contract hooks[{index}].shard_ranks is invalid"
            )
        dtype = hook.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"storage contract hooks[{index}].dtype is invalid")
        shape_tail = hook.get("shape_tail")
        if not isinstance(shape_tail, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in shape_tail
        ):
            raise ValueError(
                f"storage contract hooks[{index}].shape_tail is invalid"
            )
        coverage = hook.get("coverage", "tokens")
        if coverage not in {"tokens", "decisions"}:
            raise ValueError(f"storage contract hooks[{index}].coverage is invalid")

        family_contract = _FamilyContract(
            dtype=dtype,
            shape_tail=tuple(shape_tail),
            coverage=coverage,
        )
        for layer_no in layer_numbers:
            for shard_rank in shard_ranks:
                family = (act_name, layer_no, shard_rank)
                if family in expanded:
                    raise ValueError(f"duplicate storage family contract: {family}")
                expanded[family] = family_contract
    return expanded


def _segment_mismatches(
    rows: list[StorageRow],
    *,
    label: str,
    require_start_zero: bool,
) -> tuple[list[str], int | None]:
    ordered = sorted(rows, key=lambda row: (row.start_token_idx, row.end_token_idx))
    mismatches: list[str] = []
    if not ordered:
        return [f"{label}: no rows"], None
    if require_start_zero and ordered[0].start_token_idx != 0:
        mismatches.append(f"{label}: coverage starts at {ordered[0].start_token_idx}, not 0")
    cursor = ordered[0].start_token_idx
    for row in ordered:
        if row.start_token_idx != cursor:
            kind = "overlap" if row.start_token_idx < cursor else "gap"
            mismatches.append(
                f"{label}: {kind} before [{row.start_token_idx}, {row.end_token_idx})"
            )
        cursor = max(cursor, row.end_token_idx)
    return mismatches, cursor


def storage_contract_mismatches(
    rows: Iterable[StorageRow],
    contract: dict[str, Any],
) -> list[str]:
    """Return completeness, shape, dtype, and token-range contract violations."""

    materialized = list(rows)
    expected = _expand_contract(contract)
    expected_request_count = contract.get("expected_request_count")
    if (
        not isinstance(expected_request_count, int)
        or isinstance(expected_request_count, bool)
        or expected_request_count < 1
    ):
        raise ValueError(
            "storage contract expected_request_count must be a positive integer"
        )
    if not materialized:
        return ["storage: no rows"]

    mismatches: list[str] = []
    identities = Counter(row.identity for row in materialized)
    duplicates = sum(count - 1 for count in identities.values() if count > 1)
    if duplicates:
        mismatches.append(f"storage: {duplicates} duplicate row identities")

    request_ids = sorted({row.request_id for row in materialized})
    if len(request_ids) != expected_request_count:
        mismatches.append(
            "storage: request count "
            f"{len(request_ids)} != {expected_request_count}"
        )
    actual_families = {row.family for row in materialized}
    for family in sorted(set(expected) - actual_families):
        mismatches.append(f"storage: missing family {family}")
    for family in sorted(actual_families - set(expected)):
        mismatches.append(f"storage: unexpected family {family}")

    grouped: dict[tuple[str, tuple[str, int, int]], list[StorageRow]] = defaultdict(list)
    for row in materialized:
        family_contract = expected.get(row.family)
        label = f"storage.{row.request_id}.{row.family}"
        if row.start_token_idx < 0 or row.end_token_idx <= row.start_token_idx:
            mismatches.append(
                f"{label}: invalid range "
                f"[{row.start_token_idx}, {row.end_token_idx})"
            )
        span = row.end_token_idx - row.start_token_idx
        if not row.shape or row.shape[0] != span:
            mismatches.append(
                f"{label}: shape[0] {row.shape[0] if row.shape else None} "
                f"!= token span {span}"
            )
        if family_contract is not None:
            if row.dtype != family_contract.dtype:
                mismatches.append(
                    f"{label}: dtype {row.dtype} != {family_contract.dtype}"
                )
            if row.shape[1:] != family_contract.shape_tail:
                mismatches.append(
                    f"{label}: shape tail {row.shape[1:]} "
                    f"!= {family_contract.shape_tail}"
                )
        grouped[(row.request_id, row.family)].append(row)

    for request_id in request_ids:
        request_families = {
            family for candidate_request, family in grouped if candidate_request == request_id
        }
        for family in sorted(set(expected) - request_families):
            mismatches.append(f"storage.{request_id}: missing family {family}")

        token_ends: set[int] = set()
        decision_ends: set[int] = set()
        for family in sorted(request_families & set(expected)):
            family_contract = expected[family]
            label = f"storage.{request_id}.{family}"
            segment_errors, end = _segment_mismatches(
                grouped[(request_id, family)],
                label=label,
                require_start_zero=family_contract.coverage == "tokens",
            )
            mismatches.extend(segment_errors)
            if end is not None:
                if family_contract.coverage == "tokens":
                    token_ends.add(end)
                else:
                    decision_ends.add(end)
        if len(token_ends) != 1:
            mismatches.append(
                f"storage.{request_id}: inconsistent token coverage ends {sorted(token_ends)}"
            )
        if decision_ends and token_ends and decision_ends != token_ends:
            mismatches.append(
                f"storage.{request_id}: decision coverage ends "
                f"{sorted(decision_ends)} != token coverage ends {sorted(token_ends)}"
            )
    return mismatches
