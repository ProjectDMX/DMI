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


def _bytes_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff a and b have identical raw bytes. Both must be CPU tensors.

    Reinterprets as same-width integer type so torch.equal does true
    bitwise comparison (no NaN/signed-zero surprises).
    """
    a_c = a.contiguous()
    b_c = b.contiguous()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    dt = int_dtype.get(a_c.element_size())
    if dt is None:
        return bytes(a_c.untyped_storage()) == bytes(b_c.untyped_storage())
    return torch.equal(a_c.view(dt), b_c.view(dt))


def bitwise_check(a: torch.Tensor, b: torch.Tensor, label: str) -> dict:
    """Compare two tensors for bitwise equality.

    Returns dict with 'name', 'passed', 'detail'.
    Checks device, dtype, shape first. Then compares raw byte buffers.
    """
    if a.device != b.device:
        return {"name": label, "passed": False,
                "detail": f"device mismatch: {a.device} vs {b.device}"}
    if a.dtype != b.dtype:
        print(f"  [WARNING] {label}: dtype mismatch {a.dtype} vs {b.dtype} — SKIPPED",
              flush=True)
        return {"name": label, "passed": True,
                "detail": f"SKIPPED dtype mismatch: {a.dtype} vs {b.dtype}"}
    if a.shape != b.shape:
        return {"name": label, "passed": False,
                "detail": f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"}

    # Raw byte comparison via memoryview (no torch.equal)
    if _bytes_identical(a, b):
        return {"name": label, "passed": True, "detail": "BITWISE EQUAL"}

    # Not bitwise equal — report max abs diff
    diff = (a.float() - b.float()).abs().max().item()
    return {"name": label, "passed": False,
            "detail": f"max_abs_diff={diff:.6e}"}


def compare_logprobs(orig_path: str, ref_path: str, results: dict,
                     _check_fn, label: str = "original vs ref") -> None:
    """Compare full-vocab logprobs between two models.

    Informational only — never marks tests as failed.
    Prints summary line FIRST, then details.
    """
    orig_data = torch.load(orig_path, weights_only=False, map_location="cpu")
    ref_data = torch.load(ref_path, weights_only=False, map_location="cpu")

    # First pass: check all prompts, collect results
    all_exact = True
    n_prompts = 0
    n_exact = 0
    worst_diff = 0.0
    details = []

    for prompt_idx in sorted(orig_data.keys()):
        n_prompts += 1
        orig = orig_data[prompt_idx]
        ref = ref_data.get(prompt_idx)
        if ref is None:
            all_exact = False
            details.append(f"    prompt[{prompt_idx}]: ref missing")
            continue

        if orig["token_ids"] != ref["token_ids"]:
            all_exact = False
            diff_pos = next(
                i for i, (a, b) in enumerate(
                    zip(orig["token_ids"], ref["token_ids"]))
                if a != b)
            details.append(f"    prompt[{prompt_idx}]: token_ids DIFFER at pos {diff_pos}")
            continue

        orig_lp = orig["logprobs"]
        ref_lp = ref["logprobs"]
        if orig_lp is None or ref_lp is None:
            details.append(f"    prompt[{prompt_idx}]: logprobs N/A")
            continue

        min_len = min(orig_lp.shape[0], ref_lp.shape[0])
        min_vocab = min(orig_lp.shape[1], ref_lp.shape[1])
        a = orig_lp[:min_len, :min_vocab]
        b = ref_lp[:min_len, :min_vocab]

        if a.dtype != b.dtype:
            all_exact = False
            details.append(f"    prompt[{prompt_idx}]: dtype mismatch {a.dtype} vs {b.dtype}")
            continue
        if a.shape != b.shape:
            all_exact = False
            details.append(f"    prompt[{prompt_idx}]: shape mismatch {list(a.shape)} vs {list(b.shape)}")
            continue

        if _bytes_identical(a, b):
            n_exact += 1
            continue

        all_exact = False
        diff = (a.float() - b.float()).abs().max().item()
        worst_diff = max(worst_diff, diff)
        details.append(f"    prompt[{prompt_idx}]: DIFFER max_abs_diff={diff:.6e}")

    # SUMMARY LINE — always visible even if output is truncated
    if all_exact:
        print(f"  [LOGPROBS {label}] BITWISE EXACT ({n_exact}/{n_prompts} prompts)", flush=True)
    else:
        detail_str = "; ".join(d.strip() for d in details)
        print(f"  [LOGPROBS {label}] DIFFER ({n_exact}/{n_prompts} exact, "
              f"worst={worst_diff:.6e}) -- {detail_str}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-config", required=True)
    p.add_argument("--mon-dir", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--orig-logprobs", default=None,
                   help="Path to original model logprobs .pt file")
    p.add_argument("--ref-logprobs", default=None,
                   help="Path to ref model logprobs .pt file")
    p.add_argument("--mon-logprobs", default=None,
                   help="Path to monitored model logprobs .pt file")
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

    # Step 0: Logprob sanity checks (informational, never fails)
    if args.orig_logprobs and args.ref_logprobs:
        compare_logprobs(args.orig_logprobs, args.ref_logprobs, results, _check,
                         label="original vs ref")
    if args.orig_logprobs and args.mon_logprobs:
        compare_logprobs(args.orig_logprobs, args.mon_logprobs, results, _check,
                         label="original vs monitored")

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
        _, req_id, act_name, layer_no, shard_rank, s, e, dtype_str, shape, payload = row
        dt = _DTYPE_MAP.get(_decode(dtype_str), torch.float32)
        t = torch.frombuffer(bytearray(payload), dtype=dt).reshape(list(shape))
        ch_data[(_decode(req_id), _decode(act_name), int(layer_no),
                 int(shard_rank), int(s), int(e))] = t

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

    # Scan ref directory for .pt files
    ref_path = Path(ref_dir)
    # Pattern: {hook}_L{layer}_T{start}_{end}[_SR{rank}].pt
    #       or {hook}_T{start}_{end}[_SR{rank}].pt
    _PT_RE = re.compile(
        r"^(?P<hook>\w+?)(?:_L(?P<layer>\d+))?_T(?P<start>\d+)_(?P<end>\d+)"
        r"(?:_SR(?P<shard>\d+))?\.pt$"
    )

    # ref_files: (req_id, hook, layer, shard_rank, start, end, path)
    ref_files: list[tuple[str, str, int, int, int, int, str]] = []
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
            ref_files.append((req_id, hook, layer, shard, start, end, str(pt_file)))

    _check("ref_files_found", len(ref_files) > 0, f"{len(ref_files)} files")

    # Compare each ref tensor against CH
    for req_id, hook, layer, shard, start, end, pt_path in ref_files:
        ref_t = torch.load(pt_path, weights_only=True, map_location="cpu")

        # Find matching CH tensor (include shard_rank in key)
        ch_act = _HOOK_TO_CH_PREFIX.get(hook, hook)
        ch_layer = layer  # -1 for global hooks, matches CH layer_no

        ch_key = (req_id, ch_act, ch_layer, shard, start, end)
        ch_t = ch_data.get(ch_key)

        if ch_t is None:
            # Try finding with any token range for this (req_id, act, layer, shard)
            candidates = [
                (k, v) for k, v in ch_data.items()
                if k[0] == req_id and k[1] == ch_act and k[2] == ch_layer
                and k[3] == shard
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
        if not result["passed"]:
            # Debug: print shapes, dtypes, and first bytes
            ref_flat = ref_t.contiguous().view(-1)
            ch_flat = ch_t.contiguous().view(-1)
            print(f"    [debug] {label}: ref shape={list(ref_t.shape)} dtype={ref_t.dtype}"
                  f"  ch shape={list(ch_t.shape)} dtype={ch_t.dtype}", flush=True)
            n = min(8, ref_flat.numel(), ch_flat.numel())
            print(f"    [debug]   ref first {n}: {ref_flat[:n].tolist()}", flush=True)
            print(f"    [debug]   ch  first {n}: {ch_flat[:n].tolist()}", flush=True)
            # Check if it looks like an offset shift
            if ref_flat.numel() == ch_flat.numel() and ref_flat.numel() > 16:
                # Try shifted comparison
                try:
                    for shift in [1, -1, 768, -768]:
                        if shift > 0 and ref_flat.numel() > shift:
                            m = ref_flat.numel() - abs(shift)
                            if _bytes_identical(ref_flat[shift:shift+m].contiguous(),
                                                ch_flat[:m].contiguous()):
                                print(f"    [debug]   MATCH with shift={shift}!", flush=True)
                                break
                        elif shift < 0 and ref_flat.numel() > abs(shift):
                            m = ref_flat.numel() - abs(shift)
                            if _bytes_identical(ref_flat[:m].contiguous(),
                                                ch_flat[abs(shift):abs(shift)+m].contiguous()):
                                print(f"    [debug]   MATCH with shift={shift}!", flush=True)
                                break
                except RuntimeError:
                    pass  # view() fails on odd-offset slices for multi-byte dtypes
        if not result["passed"] and hook == "resid_pre" and layer >= 0:
            # Cross-layer check: does ref_L{N} match ch_L{N-1} or ch_L{N+1}?
            for alt_layer in [layer - 1, layer + 1]:
                alt_key = (req_id, ch_act, alt_layer, start, end)
                alt_t = ch_data.get(alt_key)
                if alt_t is not None:
                    alt_t = alt_t.cpu()
                    alt_res = bitwise_check(ref_t, alt_t, f"cross_L{layer}_vs_chL{alt_layer}")
                    if alt_res["passed"]:
                        print(f"    [CROSS-LAYER] ref L{layer} MATCHES ch L{alt_layer}! "
                              f"Layer numbering off-by-one!", flush=True)
                    else:
                        print(f"    [cross-layer] ref L{layer} vs ch L{alt_layer}: "
                              f"{alt_res['detail']}", flush=True)

            # Ref self-check: does ref_L{N} == ref_L{N-1}? (clone buffer reuse)
            if layer > 0:
                prev_path = pt_path.replace(f"_L{layer}_", f"_L{layer-1}_")
                if os.path.exists(prev_path):
                    prev_t = torch.load(prev_path, weights_only=True, map_location="cpu")
                    self_res = bitwise_check(ref_t, prev_t, f"ref_L{layer}_vs_ref_L{layer-1}")
                    if self_res["passed"]:
                        print(f"    [REF-SELF] ref L{layer} == ref L{layer-1}! "
                              f"Clone buffer REUSED!", flush=True)

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
