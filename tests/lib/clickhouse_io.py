"""ClickHouse read helpers for the comparators / matrix (plan §7).

Consolidates the row-decode logic, dtype table, and short-hook -> CH
``act_name`` map that ``vllm_identical_comparator``, ``compare_disk_vs_ch``,
and ``vllm_rowcnt_comparator`` each reimplement.

``clickhouse_driver`` is imported lazily inside the functions so importing
this module stays CPU/offline-friendly (the unit suite never reaches a DB).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# CH stores the torch dtype as its ``str(dtype)``; map back on read.
DTYPE_MAP: Dict[str, torch.dtype] = {
    "torch.bfloat16": torch.bfloat16, "torch.float": torch.float32,
    "torch.float32": torch.float32, "torch.half": torch.float16,
    "torch.float16": torch.float16, "torch.int": torch.int32,
    "torch.int32": torch.int32, "torch.long": torch.int64,
    "torch.int64": torch.int64, "torch.uint8": torch.uint8,
    "torch.int8": torch.int8, "torch.short": torch.int16,
    "torch.double": torch.float64, "torch.bool": torch.bool,
}

# Short hook name (the ``_buf_<name>`` suffix / disk filename stem) -> the
# ClickHouse ``act_name``.  Must match tensor_meta.h hook_type_name() and the
# p2p make_act_name() convention.
HOOK_TO_CH_ACT: Dict[str, str] = {
    "resid_pre": "blocks.hook_resid_pre",
    "ln1": "blocks.hook_ln1",
    "q": "blocks.attn.hook_q",
    "k": "blocks.attn.hook_k",
    "v": "blocks.attn.hook_v",
    "z": "blocks.attn.hook_z",
    "attn_scores": "blocks.attn.hook_attn_scores",
    "pattern": "blocks.attn.hook_pattern",
    "attn_out": "blocks.hook_attn_out",
    "resid_mid": "blocks.hook_resid_mid",
    "ln2": "blocks.hook_ln2",
    "mlp_in": "blocks.hook_mlp_in",
    "mlp_out": "blocks.hook_mlp_out",
    "mlp_post": "blocks.hook_mlp_post",
    "embed": "hook_embed",
    "pos_embed": "hook_pos_embed",
    "resid_final": "hook_resid_final",
    "final_ln": "hook_final_ln",
    "final_logits": "final_logits",
    "token_ids": "token_ids",
    "router_logits": "blocks.mlp.hook_router_logits",
    "topk_ids": "blocks.mlp.hook_topk_ids",
    "topk_weights": "blocks.mlp.hook_topk_weights",
}

# A CH row key: (req_id, act_name, layer_no, shard_rank, start_token, end_token).
RowKey = Tuple[str, str, int, int, int, int]


def _decode(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


def read_offload_rows(
    db_host: str, db_port: int, *,
    database: str = "default", table: str = "offload",
) -> Tuple[Dict[RowKey, torch.Tensor], int]:
    """Read every row from ``<database>.<table>`` into a keyed dict.

    Returns ``(rows_by_key, num_rows)`` where the key is :data:`RowKey` and
    the value the decoded CPU tensor.  Raises whatever ``clickhouse_driver``
    raises on a connection / query error -- callers decide whether that is a
    soft "db unreachable" skip or a hard failure.
    """
    import clickhouse_driver

    client = clickhouse_driver.Client(db_host, port=db_port)
    raw_rows = client.execute(
        "SELECT model_id, request_id, act_name, layer_no, shard_rank, "
        "start_token_idx, end_token_idx, dtype, shape, bytes "
        f"FROM {database}.{table}",
        settings={"strings_as_bytes": True},
    )

    out: Dict[RowKey, torch.Tensor] = {}
    for row in raw_rows:
        _, req_id, act_name, layer_no, shard_rank, s, e, dtype_str, shape, payload = row
        dt = DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        out[(_decode(req_id), _decode(act_name), int(layer_no),
             int(shard_rank), int(s), int(e))] = t
    return out, len(raw_rows)


def per_hook_counts(rows_by_key: Dict[RowKey, torch.Tensor]) -> Dict[str, int]:
    """Count rows per ``act_name`` (input to the ``row_count`` standard)."""
    counts: Dict[str, int] = {}
    for key in rows_by_key:
        act = key[1]
        counts[act] = counts.get(act, 0) + 1
    return counts


def group_by_request(
    rows_by_key: Dict[RowKey, torch.Tensor],
) -> Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]]:
    """Regroup CH rows as ``req_id -> (layer_no, act_name) -> [(s, e, t)]``.

    ``act_name`` is canonicalised by stripping a leading ``"blocks."`` so a
    per-layer hook keys as ``(layer_no, "hook_resid_pre")`` and a global hook
    as ``(-1, "final_logits")``.  Segments are left unsorted; merge callers
    sort by start token.
    """
    grouped: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]] = {}
    for (req_id, act_name, layer_no, _shard, s, e), t in rows_by_key.items():
        if act_name.startswith("blocks."):
            canon = act_name[len("blocks."):]
            lno = layer_no
        else:
            canon = act_name
            lno = -1
        grouped.setdefault(req_id, {}).setdefault((lno, canon), []).append(
            (s, e, t.detach().cpu()))
    return grouped
