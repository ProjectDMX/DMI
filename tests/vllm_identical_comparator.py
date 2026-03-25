"""Standalone comparator: read ref tensors from disk + monitored tensors
from ClickHouse, perform bitwise comparison, write results to JSON.

No CUDA needed — runs entirely on CPU.

Usage:
    python -m tests.vllm_identical_comparator \
        --ref-config /tmp/ref/ref_config.json \
        --mon-dir /tmp/vllm_mon \
        --result-file /tmp/result.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch


def bitwise_check(a: torch.Tensor, b: torch.Tensor, label: str) -> dict:
    """Compare two tensors for bitwise equality.

    Returns dict with 'name', 'passed', 'detail'.
    Asserts shape, dtype, device match before comparing bytes.
    """
    if a.shape != b.shape:
        return {"name": label, "passed": False,
                "detail": f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"}
    if a.dtype != b.dtype:
        return {"name": label, "passed": False,
                "detail": f"dtype mismatch: {a.dtype} vs {b.dtype}"}
    if a.device != b.device:
        return {"name": label, "passed": False,
                "detail": f"device mismatch: {a.device} vs {b.device}"}

    # Bitwise comparison via raw bytes
    a_bytes = a.contiguous().view(torch.uint8)
    b_bytes = b.contiguous().view(torch.uint8)
    if torch.equal(a_bytes, b_bytes):
        return {"name": label, "passed": True, "detail": "BITWISE EQUAL"}

    # Not bitwise equal — report max abs diff
    diff = (a.float() - b.float()).abs().max().item()
    return {"name": label, "passed": False,
            "detail": f"max_abs_diff={diff:.6e}"}


def compare_logprobs(orig_path: str, ref_path: str, results: dict,
                     _check_fn) -> None:
    """Compare full-vocab logprobs between original and ref model.

    Informational only — never marks tests as failed.
    Reports per-position max_abs_diff if not bitwise identical.
    """
    orig_data = torch.load(orig_path, weights_only=False, map_location="cpu")
    ref_data = torch.load(ref_path, weights_only=False, map_location="cpu")

    print("\n  === Step 0: Logprob sanity check (original vs ref) ===")

    for prompt_idx in sorted(orig_data.keys()):
        orig = orig_data[prompt_idx]
        ref = ref_data.get(prompt_idx)
        if ref is None:
            print(f"    prompt[{prompt_idx}]: ref missing — skipped")
            continue

        # Compare token IDs first
        if orig["token_ids"] != ref["token_ids"]:
            diff_pos = next(
                i for i, (a, b) in enumerate(
                    zip(orig["token_ids"], ref["token_ids"]))
                if a != b)
            print(f"    prompt[{prompt_idx}]: token_ids DIFFER at position "
                  f"{diff_pos} (orig={orig['token_ids'][diff_pos]} "
                  f"ref={ref['token_ids'][diff_pos]})")
            continue

        orig_lp = orig["logprobs"]
        ref_lp = ref["logprobs"]
        if orig_lp is None or ref_lp is None:
            print(f"    prompt[{prompt_idx}]: logprobs not available — skipped")
            continue

        # Trim to same length
        min_len = min(orig_lp.shape[0], ref_lp.shape[0])
        min_vocab = min(orig_lp.shape[1], ref_lp.shape[1])
        a = orig_lp[:min_len, :min_vocab]
        b = ref_lp[:min_len, :min_vocab]

        # Bitwise check
        a_bytes = a.contiguous().view(torch.uint8)
        b_bytes = b.contiguous().view(torch.uint8)
        if torch.equal(a_bytes, b_bytes):
            print(f"    prompt[{prompt_idx}]: BITWISE EXACT ({min_len} positions)")
            continue

        # Not bitwise exact — report per-position max_abs_diff
        for t in range(min_len):
            at = a[t]
            bt = b[t]
            at_bytes = at.contiguous().view(torch.uint8)
            bt_bytes = bt.contiguous().view(torch.uint8)
            if torch.equal(at_bytes, bt_bytes):
                continue
            diff = (at - bt).abs().max().item()
            print(f"    prompt[{prompt_idx}] pos[{t}]: "
                  f"max_abs_diff={diff:.6e}")

    print("  === Step 0 complete (informational, never fails) ===\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-config", required=True)
    p.add_argument("--mon-dir", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--orig-logprobs", default=None,
                   help="Path to original model logprobs .pt file")
    p.add_argument("--ref-logprobs", default=None,
                   help="Path to ref model logprobs .pt file")
    args = p.parse_args()

    with open(args.ref_config) as f:
        ref_cfg = json.load(f)
    ref_dir = ref_cfg["output_dir"]
    enabled_hooks = set(ref_cfg["enabled_hooks"])

    # Load monitored metadata
    with open(os.path.join(args.mon_dir, "meta.json")) as f:
        mon_meta = json.load(f)
    db_host = mon_meta["db_host"]
    db_port = mon_meta["db_port"]

    results = {"tests": [], "passed": 0, "failed": 0}

    def _check(name: str, passed: bool, detail: str = "") -> None:
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

    # Step 0: Logprob sanity check (informational, never fails)
    if args.orig_logprobs and args.ref_logprobs:
        compare_logprobs(args.orig_logprobs, args.ref_logprobs, results, _check)

    # Read ClickHouse
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
        _write_results(results, args.result_file)
        return

    _check("db_readable", True, f"{len(raw_rows)} rows")

    # Decode CH tensors into dict: (req_id, act_name, layer_no, start, end) -> tensor
    _DTYPE_MAP = {
        "torch.bfloat16": torch.bfloat16, "torch.float": torch.float32,
        "torch.half": torch.float16, "torch.float16": torch.float16,
        "torch.int": torch.int32, "torch.long": torch.int64,
        "torch.uint8": torch.uint8, "torch.int8": torch.int8,
        "torch.short": torch.int16, "torch.double": torch.float64,
        "torch.bool": torch.bool,
    }

    _decode = lambda v: v.decode() if isinstance(v, bytes) else v
    ch_data: dict[tuple, torch.Tensor] = {}
    for row in raw_rows:
        _, req_id, act_name, layer_no, _, s, e, dtype_str, shape, payload = row
        dt = _DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        ch_data[(_decode(req_id), _decode(act_name), int(layer_no), int(s), int(e))] = t

    # Map hook names to CH act_name format
    # CH uses "blocks.hook_resid_pre" for per-layer, "final_logits" for global
    # Must match tensor_meta.h hook_type_name() + p2p_thread make_act_name().
    # Per-layer: "blocks." + hook_type_name  (layer_no in separate DB column)
    # Global: hook_type_name as-is (layer_no = -1 in DB stored as 0)
    _HOOK_TO_CH_PREFIX = {
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
        "embed": "hook_embed",
        "pos_embed": "hook_pos_embed",
        "resid_final": "hook_resid_final",
        "final_ln": "hook_final_ln",
        "final_logits": "final_logits",
        "token_ids": "token_ids",
    }

    # Scan ref directory for .pt files
    ref_path = Path(ref_dir)
    # Pattern: {req_id}/{hook_name}_L{layer}_T{start}_{end}.pt
    #       or {req_id}/{hook_name}_T{start}_{end}.pt
    _PT_RE = re.compile(
        r"^(?P<hook>\w+?)(?:_L(?P<layer>\d+))?_T(?P<start>\d+)_(?P<end>\d+)\.pt$"
    )

    ref_files: list[tuple[str, str, int, int, int, str]] = []
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
            start = int(m.group("start"))
            end = int(m.group("end"))
            ref_files.append((req_id, hook, layer, start, end, str(pt_file)))

    _check("ref_files_found", len(ref_files) > 0, f"{len(ref_files)} files")

    # Compare each ref tensor against CH
    for req_id, hook, layer, start, end, pt_path in ref_files:
        ref_t = torch.load(pt_path, weights_only=True, map_location="cpu")

        # Find matching CH tensor
        ch_act = _HOOK_TO_CH_PREFIX.get(hook, hook)
        ch_layer = layer  # -1 for global hooks, matches CH layer_no

        # For final_logits: DB stores one row per request at (end_token-1, end_token)
        # Ref stores the full logits tensor for the step.
        # We need to find the matching CH entry.
        ch_key = (req_id, ch_act, ch_layer, start, end)
        ch_t = ch_data.get(ch_key)

        if ch_t is None:
            # Try finding with any token range for this (req_id, act, layer)
            candidates = [
                (k, v) for k, v in ch_data.items()
                if k[0] == req_id and k[1] == ch_act and k[2] == ch_layer
            ]
            if len(candidates) == 1:
                ch_t = candidates[0][1]
            elif len(candidates) > 1:
                # Multiple segments — find overlapping ones and concatenate
                segments = [(k[3], k[4], v) for k, v in candidates]
                segments.sort(key=lambda x: x[0])
                # Find segments that overlap with [start, end)
                matching = [v for s, e, v in segments if s < end and e > start]
                if matching:
                    ch_t = torch.cat(matching, dim=0)
                    # Trim to match ref range
                    # Find offset of start in concatenated segments
                    seg_start = min(s for s, e, v in segments if s < end and e > start)
                    trim_start = start - seg_start
                    ch_t = ch_t[trim_start:trim_start + ref_t.shape[0]]

        label = f"{req_id}/{hook}"
        if layer >= 0:
            label += f"_L{layer}"
        label += f"_T{start}_{end}"

        if ch_t is None:
            _check(label, False, "not found in ClickHouse")
            continue

        # Both on CPU for comparison
        ch_t = ch_t.cpu()
        result = bitwise_check(ref_t, ch_t, label)
        _check(result["name"], result["passed"], result["detail"])

    _write_results(results, args.result_file)


def _write_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    total = results["passed"] + results["failed"]
    status = "ALL PASSED" if results["failed"] == 0 else f"{results['failed']} FAILED"
    print(f"\n  {status} ({results['passed']}/{total} checks)")


if __name__ == "__main__":
    main()
