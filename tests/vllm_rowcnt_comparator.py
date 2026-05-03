"""Standalone comparator for vLLM: read reference hidden states from disk +
monitored data from ClickHouse, compare, write results to JSON.

No CUDA needed — runs entirely on CPU.

Usage:
    python -m tests.vllm_rowcnt_comparator \
        --ref-dir /tmp/vllm_ref \
        --mon-dir /tmp/vllm_mon \
        --result-file /tmp/vllm_result.json
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--mon-dir", required=True)
    p.add_argument("--result-file", required=True)
    args = p.parse_args()

    from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
    from monitoring.segment_merger import merge_segments

    tolerance = float(os.environ.get("E2E_TOLERANCE", "0.01"))
    print(f"  Tolerance: {tolerance}", flush=True)

    # Load monitored metadata
    with open(os.path.join(args.mon_dir, "meta.json")) as f:
        mon_meta = json.load(f)
    db_host = mon_meta["db_host"]
    db_port = mon_meta["db_port"]

    results = {"tests": [], "passed": 0, "failed": 0}

    def _check(name, passed, detail=""):
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

    # Read ClickHouse — get all rows (table was dropped before monitored run)
    import clickhouse_driver
    ch_client = clickhouse_driver.Client(db_host, port=db_port)
    try:
        raw_rows = ch_client.execute(
            "SELECT model_id, request_id, act_name, layer_no, shard_rank, "
            "start_token_idx, end_token_idx, dtype, shape, bytes "
            "FROM default.offload",
            settings={"strings_as_bytes": True})
    except Exception as e:
        _check("db_readable", False, str(e))
        with open(args.result_file, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Decode tensors
    _DTYPE_MAP = {
        "torch.bfloat16": torch.bfloat16, "torch.float": torch.float32,
        "torch.half": torch.float16, "torch.float16": torch.float16,
        "torch.int": torch.int32, "torch.long": torch.int64,
        "torch.uint8": torch.uint8, "torch.int8": torch.int8,
        "torch.short": torch.int16, "torch.double": torch.float64,
        "torch.bool": torch.bool,
    }
    rows = []
    _decode = lambda v: v.decode() if isinstance(v, bytes) else v
    for row in raw_rows:
        model_id_val, req_id, act_name, layer_no, shard_rank, s, e, dtype_str, shape, payload = row
        dt = _DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        key = (_decode(model_id_val), _decode(req_id), _decode(act_name), layer_no, shard_rank, s, e)
        rows.append((key, t))

    _check("rows_found", len(rows) > 0, f"{len(rows)} rows")

    # Group by act_name for row-count validation
    per_hook: Dict[str, int] = {}
    for k, _ in rows:
        act = str(k[2])
        per_hook[act] = per_hook.get(act, 0) + 1

    print(f"\n  Total DB rows: {len(rows)}")
    for name, cnt in sorted(per_hook.items()):
        print(f"    {name}: {cnt}")

    # Per-layer consistency
    per_layer_names = [n for n in per_hook if n.startswith("blocks.")]
    per_layer_counts = [per_hook[n] for n in per_layer_names]

    _check("per_layer_types", len(per_layer_names) >= 10,
           f"{len(per_layer_names)} types")

    if per_layer_counts:
        expected = per_layer_counts[0]
        all_equal = all(c == expected for c in per_layer_counts)
        _check("per_layer_equal_count", all_equal,
               f"counts: {set(per_layer_counts)}")

    # Global hooks
    _check("final_logits_present", "final_logits" in per_hook)

    # Valid token ranges
    bad = [(str(k[2]), int(k[5]), int(k[6])) for k, _ in rows if int(k[5]) >= int(k[6])]
    _check("valid_token_ranges", len(bad) == 0, f"bad: {bad[:3]}")

    # -----------------------------------------------------------------------
    # Value comparison (if reference exists)
    # -----------------------------------------------------------------------
    ref_meta_path = os.path.join(args.ref_dir, "meta.json")
    if os.path.exists(ref_meta_path):
        with open(ref_meta_path) as f:
            ref_meta = json.load(f)

        if ref_meta.get("skipped"):
            print("\n  Value comparison skipped (model not supported by extract_hidden_states)")
        else:
            layer_ids = ref_meta["layer_ids"]
            ref_data = torch.load(
                os.path.join(args.ref_dir, "ref_hidden_states.pt"),
                weights_only=False)

            print(f"\n  Value comparison: {len(ref_data)} ref requests, layers={layer_ids}")

            # Group DB resid_pre by request
            from integration.vllm_adapter import normalize_vllm_request_id

            db_grouped: Dict[str, Dict[int, List]] = {}
            for k, t in rows:
                act = str(k[2])
                if act != "blocks.hook_resid_pre":
                    continue
                req_id = str(k[1])
                layer_no = int(k[3])
                if layer_no not in layer_ids:
                    continue
                s, e = int(k[5]), int(k[6])
                db_grouped.setdefault(req_id, {}).setdefault(layer_no, []).append(
                    (s, e, t.detach().cpu()))

            for ref_req_id, ref_hs in ref_data.items():
                norm_id = normalize_vllm_request_id(ref_req_id)
                ring_layers = db_grouped.get(norm_id)
                if ring_layers is None:
                    _check(f"req_{norm_id}_found", False, "not in DB")
                    continue

                # ref_hs: [total_tokens, num_extracted_layers, hidden_size]
                for layer_idx, layer_no in enumerate(layer_ids):
                    ring_chunks = ring_layers.get(layer_no)
                    if ring_chunks is None:
                        continue
                    ring_chunks.sort(key=lambda x: x[0])
                    ring_merged = merge_segments(
                        [t for _, _, t in ring_chunks], "hook_resid_pre")

                    ref_layer = ref_hs[:, layer_idx, :].float()
                    ring_layer = ring_merged.float()
                    min_len = min(ref_layer.shape[0], ring_layer.shape[0])
                    diff = (ref_layer[:min_len] - ring_layer[:min_len]).abs().max().item()
                    _check(f"req_{norm_id}_L{layer_no}", diff <= tolerance,
                           f"max_abs_diff={diff:.6e}")

    # Write results
    with open(args.result_file, "w") as f:
        json.dump(results, f, indent=2)

    status = "ALL PASSED" if results["failed"] == 0 else f"{results['failed']} FAILED"
    print(f"\n  {status} ({results['passed']}/{results['passed'] + results['failed']} checks)")


if __name__ == "__main__":
    main()
