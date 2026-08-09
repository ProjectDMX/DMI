"""Standalone comparator: read reference tensors from disk + monitored data
from ClickHouse, compare, write results to JSON.

No CUDA needed — runs entirely on CPU.

Usage:
    python -m tests.hf_comparator \
        --ref-dir /tmp/hf_ref \
        --mon-dir /tmp/hf_mon \
        --result-file /tmp/hf_result.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--mon-dir", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--standard", default="allclose",
                   help="comparison standard (only 'allclose' is implemented)")
    args = p.parse_args()

    if args.standard != "allclose":
        raise SystemExit(
            f"hf_comparator: standard={args.standard!r} is not implemented; "
            "only 'allclose' is supported"
        )

    from tests.hf_reference import (
        _HFRef,
        _load_hf_refs_from_disk,
        _parse_request_id,
        _strip_left_pad,
    )
    from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
    from monitoring.segment_merger import merge_segments

    # Tolerance: eager comparison is tight (0.01), CUDA graph vs eager
    # allows bf16 rounding (0.5) due to different accumulation order
    # from torch.compile + static cache.
    tolerance = float(os.environ.get("E2E_TOLERANCE", "0.01"))
    print(f"  Tolerance: {tolerance}", flush=True)

    # Load reference
    hf_refs = _load_hf_refs_from_disk(args.ref_dir)
    ref_meta = torch.load(os.path.join(args.ref_dir, "meta.pt"), weights_only=False)
    batch_size = int(ref_meta["batch_size"])
    eos_id = int(ref_meta["eos_token_id"])

    # Load monitored metadata
    with open(os.path.join(args.mon_dir, "meta.json")) as f:
        mon_meta = json.load(f)

    model_id = mon_meta["model_id"]

    # Read from ClickHouse
    ch = CHClickhouseDriverReadOnly(
        host=mon_meta["db_host"],
        port=mon_meta["db_port"],
        database=mon_meta["db_database"],
        table=mon_meta["db_table"],
        decode_strings=True,
    )
    try:
        rows = ch.prefix_get((model_id,), return_full_key_tuple=True)
    finally:
        ch.close()

    if not rows:
        _fail(args.result_file, f"No rows in ClickHouse for model_id={model_id}")
        return

    # Group DB data by (request_id, (layer_no, act_name))
    shard_ranks = sorted({int(k[4]) for k, _ in rows})
    chosen_rank = 0 if 0 in shard_ranks else shard_ranks[0]
    rows = [(k, t) for k, t in rows if int(k[4]) == chosen_rank]

    grouped: Dict[str, Dict[Tuple[int, str], List[Tuple[int, int, torch.Tensor]]]] = {}
    for full_key, t_raw in rows:
        _, req_id, act_name_raw, layer_no_raw, _, s, e = full_key
        act_name = str(act_name_raw)
        layer_no = int(layer_no_raw)
        # Canonicalize: "blocks.hook_resid_pre" layer=5 -> (5, "hook_resid_pre")
        if act_name.startswith("blocks."):
            act_name = act_name[len("blocks."):]
        else:
            layer_no = -1
        t = t_raw.detach().cpu()
        grouped.setdefault(str(req_id), {}).setdefault((layer_no, act_name), []).append(
            (int(s), int(e), t)
        )

    request_ids = sorted(grouped.keys(), key=_parse_request_id)
    results = {"tests": [], "passed": 0, "failed": 0, "total_rows": len(rows)}

    def _check(name: str, passed: bool, detail: str = ""):
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg, flush=True)

    # Basic checks
    _check("rows_found", len(rows) > 0, f"{len(rows)} rows")
    _check("requests_found", len(request_ids) > 0, f"{len(request_ids)} requests")

    if len(request_ids) != batch_size:
        _check("request_count", False,
               f"expected {batch_size}, got {len(request_ids)}")

    # Per-request comparison
    for req_idx, req_id in enumerate(request_ids):
        _, local_i = _parse_request_id(req_id)
        if local_i >= len(hf_refs):
            _check(f"req_{req_id}_in_range", False,
                   f"local_i={local_i} >= len(hf_refs)={len(hf_refs)}")
            continue
        ref = hf_refs[local_i]
        hooks_map = grouped[req_id]

        # Token IDs
        if (-1, "token_ids") in hooks_map:
            chunks = sorted(hooks_map[(-1, "token_ids")], key=lambda x: x[0])
            db_tok = merge_segments([t for _, _, t in chunks], "token_ids").to(torch.long).view(-1)
            ref_tok = ref.token_ids
            min_len = min(db_tok.numel(), ref_tok.numel())
            if min_len > 0:
                match = torch.equal(db_tok[:min_len], ref_tok[:min_len])
                _check(f"req_{local_i}_token_ids",
                       match and db_tok.numel() == ref_tok.numel(),
                       f"db_len={db_tok.numel()} ref_len={ref_tok.numel()} prefix_match={match}")
            else:
                _check(f"req_{local_i}_token_ids", False, "empty")

        # Hidden states (resid_pre per layer)
        # ref.hidden_states[0] = embedding output
        # ref.hidden_states[i+1] = output of layer i (= input to layer i+1 = resid_pre[i+1])
        # Our hook_resid_pre at layer i captures INPUT to layer i.
        # So ref.hidden_states[i] should match our resid_pre at layer i (for i >= 1)
        # And ref.hidden_states[0] should match our hook_embed.
        n_ref_layers = len(ref.hidden_states)

        # Skip hook_embed comparison: ref hidden_states[0] = token_embed + pos_embed,
        # but our hook_embed captures token_embed only.  Not comparable.

        # Compare resid_pre per layer
        for layer_i in range(1, n_ref_layers):
            key = (layer_i, "hook_resid_pre")
            if key not in hooks_map:
                # Layer might be on different PP rank
                continue
            chunks = sorted(hooks_map[key], key=lambda x: x[0])
            db_t = merge_segments([t for _, _, t in chunks], "hook_resid_pre")
            ref_t = ref.hidden_states[layer_i]
            min_len = min(db_t.shape[0], ref_t.shape[0])
            diff = (db_t[:min_len].float() - ref_t[:min_len].float()).abs().max().item()
            _check(f"req_{local_i}_resid_pre_L{layer_i}", diff <= tolerance,
                   f"max_abs_diff={diff:.6e}")

        # Compare final_logits.
        # DB with logits_to_keep=0 stores ALL positions per step:
        #   [prompt_pos_0, ..., prompt_pos_N-1, decode_0, ..., decode_G-1]
        # ref.final_logits from generate(output_logits=True):
        #   [prefill_last_pos, decode_0, ..., decode_G-2]  (drops last)
        # Align: skip first (prompt_len - 1) rows in DB, then compare.
        if (-1, "final_logits") in hooks_map:
            chunks = sorted(hooks_map[(-1, "final_logits")], key=lambda x: x[0])
            db_logits = merge_segments([t for _, _, t in chunks], "final_logits")
            ref_logits = ref.final_logits
            if ref_logits.numel() > 0 and db_logits.numel() > 0:
                # db has [prompt_len + gen_len, vocab] (all positions).
                # ref has [keep_gen, vocab] where keep_gen = gen_len - 1
                # (generate() drops last token that was never forwarded).
                # db[prompt_len - 1] = prefill last-position logit = ref[0].
                # So skip = prompt_len - 1 = db_len - ref_len - 1.
                skip = max(0, db_logits.shape[0] - ref_logits.shape[0] - 1)
                db_aligned = db_logits[skip:]
                G = min(ref_logits.shape[0], db_aligned.shape[0])
                diff = (db_aligned[:G].float() - ref_logits[:G].float()).abs().max().item()
                _check(f"req_{local_i}_final_logits", diff <= tolerance,
                       f"max_abs_diff={diff:.6e} skip={skip} "
                       f"db_shape={list(db_logits.shape)} ref_shape={list(ref_logits.shape)}")

    # Write results
    with open(args.result_file, "w") as f:
        json.dump(results, f, indent=2)

    status = "ALL PASSED" if results["failed"] == 0 else f"{results['failed']} FAILED"
    print(f"\n  {status} ({results['passed']}/{results['passed'] + results['failed']} checks)", flush=True)


def _fail(result_file, msg):
    print(f"  FAIL: {msg}", flush=True)
    with open(result_file, "w") as f:
        json.dump({"tests": [{"name": "setup", "passed": False, "detail": msg}],
                    "passed": 0, "failed": 1}, f)


if __name__ == "__main__":
    main()
