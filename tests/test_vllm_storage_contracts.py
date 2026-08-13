"""CPU tests for black-box DMI storage completeness contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.blackbox.storage_contracts import (
    StorageRow,
    storage_contract_mismatches,
)


pytestmark = pytest.mark.cpu


_CONTRACT = {
    "schema_version": 1,
    "expected_request_count": 1,
    "num_layers": 1,
    "hooks": [
        {
            "act_name": "hook_embed",
            "layers": [-1],
            "dtype": "torch.bfloat16",
            "shape_tail": [4],
        },
        {
            "act_name": "blocks.attn.hook_q",
            "layers": "all",
            "dtype": "torch.bfloat16",
            "shape_tail": [2, 2],
        },
        {
            "act_name": "final_logits",
            "layers": [-1],
            "dtype": "torch.bfloat16",
            "shape_tail": [8],
            "coverage": "decisions",
        },
    ],
}


def _row(
    act_name: str,
    layer_no: int,
    start: int,
    end: int,
    shape_tail: tuple[int, ...],
    *,
    request_id: str = "request-0",
    dtype: str = "torch.bfloat16",
) -> StorageRow:
    return StorageRow(
        request_id=request_id,
        act_name=act_name,
        layer_no=layer_no,
        shard_rank=0,
        start_token_idx=start,
        end_token_idx=end,
        dtype=dtype,
        shape=(end - start, *shape_tail),
    )


def _valid_rows() -> list[StorageRow]:
    rows = []
    for start, end in ((0, 2), (2, 3)):
        rows.extend(
            [
                _row("hook_embed", -1, start, end, (4,)),
                _row("blocks.attn.hook_q", 0, start, end, (2, 2)),
            ]
        )
    rows.extend(
        [
            _row("final_logits", -1, 1, 2, (8,)),
            _row("final_logits", -1, 2, 3, (8,)),
        ]
    )
    return rows


def test_storage_contract_accepts_complete_contiguous_public_rows() -> None:
    assert storage_contract_mismatches(_valid_rows(), _CONTRACT) == []


def test_storage_contract_rejects_duplicate_and_missing_families() -> None:
    rows = _valid_rows()
    rows.append(rows[0])
    rows = [row for row in rows if row.act_name != "blocks.attn.hook_q"]

    mismatches = storage_contract_mismatches(rows, _CONTRACT)

    assert "storage: 1 duplicate row identities" in mismatches
    assert "storage: missing family ('blocks.attn.hook_q', 0, 0)" in mismatches
    assert (
        "storage.request-0: missing family ('blocks.attn.hook_q', 0, 0)"
        in mismatches
    )


def test_storage_contract_rejects_shape_dtype_and_token_gaps() -> None:
    rows = _valid_rows()
    rows[0] = StorageRow(
        **{
            **rows[0].__dict__,
            "dtype": "torch.float",
            "shape": (2, 5),
        }
    )
    rows[2] = StorageRow(
        **{
            **rows[2].__dict__,
            "start_token_idx": 3,
            "end_token_idx": 4,
            "shape": (1, 4),
        }
    )

    mismatches = storage_contract_mismatches(rows, _CONTRACT)

    assert any("dtype torch.float != torch.bfloat16" in item for item in mismatches)
    assert any("shape tail (5,) != (4,)" in item for item in mismatches)
    assert any(": gap before [3, 4)" in item for item in mismatches)
    assert any("inconsistent token coverage ends" in item for item in mismatches)


def test_storage_contract_rejects_duplicate_expectations() -> None:
    contract = deepcopy(_CONTRACT)
    contract["hooks"].append(deepcopy(contract["hooks"][0]))

    with pytest.raises(ValueError, match="duplicate storage family contract"):
        storage_contract_mismatches(_valid_rows(), contract)
