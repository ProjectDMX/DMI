"""Shared comparator: compare .pt ref files on disk vs ClickHouse rows.

Used by both vllm_compare_runner.py and hf_compare_runner.py.
No CUDA needed — runs entirely on CPU.
"""
import re
import sys
from pathlib import Path

import torch


_DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16, "torch.float": torch.float32,
    "torch.half": torch.float16, "torch.float16": torch.float16,
    "torch.int": torch.int32, "torch.long": torch.int64,
    "torch.uint8": torch.uint8, "torch.int8": torch.int8,
    "torch.short": torch.int16, "torch.double": torch.float64,
    "torch.bool": torch.bool,
}

# Map short hook name → ClickHouse act_name (must match tensor_meta.h + p2p make_act_name).
_BUF_TO_CH_ACT = {
    "resid_pre": "blocks.hook_resid_pre",
    "ln1": "blocks.hook_ln1",
    "q": "blocks.attn.hook_q",
    "k": "blocks.attn.hook_k",
    "v": "blocks.attn.hook_v",
    "z": "blocks.attn.hook_z",
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
}

_PT_RE = re.compile(
    r"^(?P<hook>\w+?)(?:_L(?P<layer>\d+))?_T(?P<start>\d+)_(?P<end>\d+)"
    r"(?:_SR(?P<shard>\d+))?\.pt$"
)

_decode = lambda v: v.decode() if isinstance(v, bytes) else v


def _bytes_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    a_c = a.contiguous()
    b_c = b.contiguous()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    dt = int_dtype.get(a_c.element_size())
    if dt is None:
        return bytes(a_c.untyped_storage()) == bytes(b_c.untyped_storage())
    return torch.equal(a_c.view(dt), b_c.view(dt))


def read_clickhouse(db_host: str, db_port: int,
                    database: str = "default", table: str = "offload"):
    """Read all rows from ClickHouse, return dict keyed by (req_id, act_name, layer, shard, start, end)."""
    import clickhouse_driver
    ch_client = clickhouse_driver.Client(db_host, port=db_port)
    raw_rows = ch_client.execute(
        f"SELECT model_id, request_id, act_name, layer_no, shard_rank, "
        f"start_token_idx, end_token_idx, dtype, shape, bytes "
        f"FROM {database}.{table}",
        settings={"strings_as_bytes": True})

    ch_data: dict[tuple, torch.Tensor] = {}
    for row in raw_rows:
        _, req_id, act_name, layer_no, shard_rank, s, e, dtype_str, shape, payload = row
        dt = _DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        ch_data[(_decode(req_id), _decode(act_name), int(layer_no),
                 int(shard_rank), int(s), int(e))] = t

    return ch_data, len(raw_rows)


def compare(ref_dir: str, ch_data: dict, num_ch_rows: int) -> tuple[int, int]:
    """Compare .pt files in ref_dir against ch_data. Returns (passed, failed)."""
    ref_path = Path(ref_dir)
    passed = 0
    failed = 0
    not_found = 0

    for req_dir in sorted(ref_path.iterdir()):
        if not req_dir.is_dir():
            continue
        req_id = req_dir.name
        for pt_file in sorted(req_dir.iterdir()):
            m = _PT_RE.match(pt_file.name)
            if not m:
                continue
            hook = m.group("hook")
            layer = int(m.group("layer")) if m.group("layer") is not None else -1
            shard = int(m.group("shard")) if m.group("shard") is not None else 0
            start = int(m.group("start"))
            end = int(m.group("end"))

            ref_t = torch.load(str(pt_file), weights_only=True, map_location="cpu")

            ch_act = _BUF_TO_CH_ACT.get(hook, hook)
            ch_key = (req_id, ch_act, layer, shard, start, end)
            ch_t = ch_data.get(ch_key)

            if ch_t is None:
                # Collect all segments for this (req, act, layer, shard)
                candidates = sorted(
                    [(k, v) for k, v in ch_data.items()
                     if k[0] == req_id and k[1] == ch_act and k[2] == layer
                     and k[3] == shard],
                    key=lambda kv: kv[0][4],  # sort by start_token
                )
                if len(candidates) == 1:
                    ch_t = candidates[0][1]
                elif len(candidates) > 1:
                    # Concatenate segments covering [start, end)
                    segments = [(k[4], k[5], v) for k, v in candidates
                                if k[4] < end and k[5] > start]
                    if segments:
                        ch_t = torch.cat([v for _, _, v in segments], dim=0)

            label = f"{req_id}/{hook}"
            if layer >= 0:
                label += f"_L{layer}"
            label += f"_T{start}_{end}"

            if ch_t is None:
                print(f"  [FAIL] {label} -- not found in ClickHouse", flush=True)
                failed += 1
                not_found += 1
                continue

            ch_t = ch_t.cpu()
            if ref_t.shape != ch_t.shape:
                print(f"  [FAIL] {label} -- shape {list(ref_t.shape)} vs {list(ch_t.shape)}",
                      flush=True)
                failed += 1
                continue

            if ref_t.dtype != ch_t.dtype:
                print(f"  [FAIL] {label} -- dtype mismatch: ref={ref_t.dtype} ch={ch_t.dtype}",
                      flush=True)
                failed += 1
                continue

            if _bytes_identical(ref_t, ch_t):
                passed += 1
            else:
                diff = (ref_t.float() - ch_t.float()).abs().max().item()
                print(f"  [FAIL] {label} -- max_abs_diff={diff:.6e}", flush=True)
                failed += 1

    total = passed + failed
    print(f"\n[compare] {num_ch_rows} rows in ClickHouse", flush=True)
    print(f"[compare] Results: {passed}/{total} passed, {failed} failed", flush=True)
    if not_found > 0:
        print(f"[compare] ({not_found} not found in ClickHouse)", flush=True)

    return passed, failed
