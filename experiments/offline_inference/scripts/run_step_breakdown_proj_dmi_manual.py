#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from common import build_tokenizer, ensure_dir, make_output_path, resolve_model_id, write_json
from run_step_breakdown_microbench import (
    _close_proj_dmi_microbench,
    _compile_decode_step,
    _create_proj_dmi_engine_for_model,
    _dmi_prefill_kwargs,
    _flush_proj_dmi_ring,
    _forward_accepts_position_ids,
    _load_hooked_model,
    _make_static_cache,
    _make_synthetic_batch,
    _mean,
    _position_ids_from_attention_mask,
    _prepare_ring_forward,
    _run_unmonitored_seed_prefill,
    build_step_breakdown_parser,
    device_sync,
)


def _measure_dmi_manual_decode(
    model: Any,
    model_id: str,
    args: Any,
    encoded: Dict[str, torch.Tensor],
    *,
    engine: Any,
    decode_steps: int,
    compiled_decode: Any = None,
) -> tuple[float, float, float, float, float, float, Dict[str, int]]:
    device = encoded["input_ids"].device
    prefill_tokens = int(encoded["input_ids"].shape[1])
    batch_size = int(encoded["input_ids"].shape[0])
    decode_iters = max(int(decode_steps) - 1, 0)
    if decode_iters == 0:
        ring_stats = _flush_proj_dmi_ring(engine)
        return 0.0, 0.0, float(ring_stats["flush_wait_us"]) / 1000.0, 0.0, 0.0, float(ring_stats["flush_wait_us"]) / 1000.0, ring_stats

    cache = _make_static_cache(
        model,
        batch_size=batch_size,
        max_cache_len=prefill_tokens + int(decode_steps) + 4,
        device=device,
    )
    prefill_kwargs: Dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "past_key_values": cache,
        "cache_position": torch.arange(prefill_tokens, device=device, dtype=torch.long),
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
        "logits_to_keep": 1,
    }
    if _forward_accepts_position_ids(model):
        prefill_kwargs["position_ids"] = _position_ids_from_attention_mask(encoded["attention_mask"])

    seed_outputs = _run_unmonitored_seed_prefill(model, engine, prefill_kwargs)
    next_tokens = seed_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).clone()
    del seed_outputs

    decode_mask = torch.zeros(
        (batch_size, prefill_tokens + decode_iters),
        device=device,
        dtype=encoded["attention_mask"].dtype,
    )
    decode_mask[:, :prefill_tokens] = encoded["attention_mask"]

    wants_position_ids = _forward_accepts_position_ids(model)
    step_compute_times_ms: List[float] = []
    device_sync(device)
    t_total0 = time.perf_counter()
    for step_idx in range(decode_iters):
        decode_mask[:, prefill_tokens + step_idx] = 1
        decode_cache_position = torch.tensor([prefill_tokens + step_idx], device=device, dtype=torch.long)
        decode_position_ids = (
            torch.full((batch_size, 1), prefill_tokens + step_idx, device=device, dtype=torch.long)
            if wants_position_ids
            else None
        )
        _prepare_ring_forward(
            model=model,
            input_ids=next_tokens,
            attention_mask=decode_mask,
            past_key_values=cache,
            cache_position=decode_cache_position,
            logits_to_keep=1,
        )
        step_t0 = time.perf_counter()
        if compiled_decode is None:
            decode_kwargs: Dict[str, Any] = {
                "input_ids": next_tokens,
                "attention_mask": decode_mask,
                "past_key_values": cache,
                "cache_position": decode_cache_position,
                "use_cache": True,
                "return_dict": True,
                "output_hidden_states": False,
                "output_attentions": False,
                "logits_to_keep": 1,
            }
            if decode_position_ids is not None:
                decode_kwargs["position_ids"] = decode_position_ids
            outputs = model(**decode_kwargs)
        else:
            torch.compiler.cudagraph_mark_step_begin()
            outputs = compiled_decode(next_tokens, decode_mask, cache, decode_cache_position, decode_position_ids)
        torch.cuda.current_stream().synchronize()
        step_compute_t = time.perf_counter()
        step_compute_times_ms.append((step_compute_t - step_t0) * 1000.0)
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).clone()
        del outputs

    ring_stats = _flush_proj_dmi_ring(engine)
    resolve_end_t = time.perf_counter()
    flush_ms = float(ring_stats["flush_wait_us"]) / 1000.0
    decode_total_compute_ms = (step_compute_t - t_total0) * 1000.0
    decode_total_total_ms = decode_total_compute_ms
    print(
        f"  [proj_dmi_manual] per-step compute ms: {['%.3f' % t for t in step_compute_times_ms]}  "
        f"sum={sum(step_compute_times_ms):.3f}  seq_compute={decode_total_compute_ms:.3f}  total+flush={(resolve_end_t - t_total0) * 1000.0:.3f}",
        flush=True,
    )
    last_compute_ms = step_compute_times_ms[-1]
    last_total_ms = last_compute_ms
    last_total_with_flush_ms = last_total_ms + flush_ms
    decode_total_total_with_flush_ms = decode_total_total_ms + flush_ms
    return (
        last_compute_ms,
        last_total_ms,
        last_total_with_flush_ms,
        decode_total_compute_ms,
        decode_total_total_ms,
        decode_total_total_with_flush_ms,
        ring_stats,
    )


def main() -> None:
    parser = build_step_breakdown_parser(include_baseline_arg=False, default_baseline="proj_dmi")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = resolve_model_id(args.model)
    tokenizer = build_tokenizer(model_id, local_files_only=bool(args.local_files_only))
    device = torch.device("cuda")
    encoded = _make_synthetic_batch(
        tokenizer,
        batch_size=int(args.batch_size),
        prefill_tokens=int(args.prefill_tokens),
        device=device,
    )

    baseline_label = args.baseline_label or "proj_dmi_manual"
    compile_enabled = not bool(args.disable_compile)
    model = _load_hooked_model(model_id, local_files_only=bool(args.local_files_only)).to(device).eval()

    prefill_compute_runs: List[float] = []
    prefill_total_runs: List[float] = []
    decode_compute_runs: List[float] = []
    decode_total_runs: List[float] = []
    decode_total_with_flush_runs: List[float] = []
    decode_seq_compute_runs: List[float] = []
    decode_seq_total_runs: List[float] = []
    decode_seq_total_with_flush_runs: List[float] = []
    prefill_ring_stats_runs: List[Dict[str, int]] = []
    decode_ring_stats_runs: List[Dict[str, int]] = []

    dmi_compiled_prefill = None
    dmi_compiled_decode = _compile_decode_step(model, output_hidden_states=False, logits_to_keep=1) if compile_enabled else None

    from run_step_breakdown_microbench import _measure_dmi_prefill

    prefill_engine = _create_proj_dmi_engine_for_model(
        model_id=model_id,
        model=model,
        proj_dmi_mode=str(args.proj_dmi_mode),
        hook_selection=str(args.hook_selection),
        args=args,
    )
    try:
        with torch.no_grad():
            for _ in range(int(args.warmup)):
                _measure_dmi_prefill(model, model_id, args, encoded, engine=prefill_engine, compiled_prefill=dmi_compiled_prefill)
                device_sync(device)
        print(f"Warmup done ({int(args.warmup)} iterations).", flush=True)
        with torch.no_grad():
            for _ in range(int(args.iters)):
                prefill_compute_ms, prefill_total_ms, prefill_ring_stats = _measure_dmi_prefill(
                    model, model_id, args, encoded, engine=prefill_engine, compiled_prefill=dmi_compiled_prefill
                )
                prefill_compute_runs.append(prefill_compute_ms)
                prefill_total_runs.append(prefill_total_ms)
                prefill_ring_stats_runs.append(prefill_ring_stats)
    finally:
        _close_proj_dmi_microbench(model, prefill_engine)

    decode_engine = _create_proj_dmi_engine_for_model(
        model_id=model_id,
        model=model,
        proj_dmi_mode=str(args.proj_dmi_mode),
        hook_selection=str(args.hook_selection),
        args=args,
    )
    try:
        with torch.no_grad():
            for _ in range(int(args.warmup)):
                _measure_dmi_manual_decode(
                    model,
                    model_id,
                    args,
                    encoded,
                    engine=decode_engine,
                    decode_steps=int(args.decode_steps),
                    compiled_decode=dmi_compiled_decode,
                )
                device_sync(device)
        with torch.no_grad():
            for _ in range(int(args.iters)):
                (
                    decode_compute_ms,
                    decode_total_ms,
                    decode_total_with_flush_ms,
                    decode_seq_compute_ms,
                    decode_seq_total_ms,
                    decode_seq_total_with_flush_ms,
                    decode_ring_stats,
                ) = _measure_dmi_manual_decode(
                    model,
                    model_id,
                    args,
                    encoded,
                    engine=decode_engine,
                    decode_steps=int(args.decode_steps),
                    compiled_decode=dmi_compiled_decode,
                )
                decode_compute_runs.append(decode_compute_ms)
                decode_total_runs.append(decode_total_ms)
                decode_total_with_flush_runs.append(decode_total_with_flush_ms)
                decode_seq_compute_runs.append(decode_seq_compute_ms)
                decode_seq_total_runs.append(decode_seq_total_ms)
                decode_seq_total_with_flush_runs.append(decode_seq_total_with_flush_ms)
                decode_ring_stats_runs.append(decode_ring_stats)
    finally:
        _close_proj_dmi_microbench(model, decode_engine)

    prefill_summary = {
        "compute_ms": _mean(prefill_compute_runs),
        "total_ms": _mean(prefill_total_runs),
    }
    prefill_summary["transfer_tail_ms"] = max(0.0, prefill_summary["total_ms"] - prefill_summary["compute_ms"])

    decode_summary = {
        "compute_ms": _mean(decode_compute_runs),
        "total_ms": _mean(decode_total_runs),
        "total_with_final_flush_ms": _mean(decode_total_with_flush_runs),
    }
    decode_summary["transfer_tail_ms"] = max(0.0, decode_summary["total_ms"] - decode_summary["compute_ms"])
    decode_summary["final_flush_tail_ms"] = max(0.0, decode_summary["total_with_final_flush_ms"] - decode_summary["total_ms"])

    decode_seq_summary = {
        "compute_ms": _mean(decode_seq_compute_runs),
        "total_ms": _mean(decode_seq_total_runs),
        "total_with_final_flush_ms": _mean(decode_seq_total_with_flush_runs),
    }
    decode_seq_summary["transfer_tail_ms"] = max(0.0, decode_seq_summary["total_ms"] - decode_seq_summary["compute_ms"])
    decode_seq_summary["final_flush_tail_ms"] = max(0.0, decode_seq_summary["total_with_final_flush_ms"] - decode_seq_summary["total_ms"])

    payload = {
        "baseline": "proj_dmi",
        "baseline_label": baseline_label,
        "model": str(args.model),
        "model_id": model_id,
        "batch_size": int(args.batch_size),
        "prefill_tokens": int(args.prefill_tokens),
        "decode_steps": int(args.decode_steps),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "hook_selection": str(args.hook_selection),
        "repeat_index": int(args.repeat_index),
        "compile_enabled": bool(compile_enabled),
        "measurement_kind": "prefill_decode_step_breakdown",
        "timing_semantics": {
            "compute_ms": "Wall time until the GPU main compute stream is synchronized.",
            "total_ms": "Wall time until pending payloads are flushed to the ring transport tail.",
            "decode_steps_note": "decode_sequence_total uses a manual cached decode loop after an unmonitored seed prefill.",
        },
        "prefill": {
            **prefill_summary,
            "compute_runs_ms": [float(v) for v in prefill_compute_runs],
            "total_runs_ms": [float(v) for v in prefill_total_runs],
            "ring_stats_runs": prefill_ring_stats_runs,
        },
        "decode_last_step": decode_summary,
        "decode_last_step_runs_ms": {
            "compute_runs_ms": [float(v) for v in decode_compute_runs],
            "total_runs_ms": [float(v) for v in decode_total_runs],
            "total_with_final_flush_runs_ms": [float(v) for v in decode_total_with_flush_runs],
            "ring_stats_runs": decode_ring_stats_runs,
        },
        "decode_sequence_total": decode_seq_summary,
        "decode_sequence_total_runs_ms": {
            "compute_runs_ms": [float(v) for v in decode_seq_compute_runs],
            "total_runs_ms": [float(v) for v in decode_seq_total_runs],
            "total_with_final_flush_runs_ms": [float(v) for v in decode_seq_total_with_flush_runs],
        },
        "hook_count": None,
        "ring_payload_mb": int(args.ring_payload_mb),
        "ring_pinned_mb": int(args.ring_pinned_mb),
    }

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)
    output_path = make_output_path(
        results_dir=results_dir,
        baseline=baseline_label,
        model=args.model,
        sample_file=f"synthetic_prefill_{int(args.prefill_tokens)}.jsonl",
        batch_size=int(args.batch_size),
        repeat_index=int(args.repeat_index),
    )
    write_json(output_path, payload)
    print(f"Saved {output_path}")
    print(
        f"[{baseline_label}] bs={int(args.batch_size)} "
        f"prefill compute={prefill_summary['compute_ms']:.3f} total={prefill_summary['total_ms']:.3f} "
        f"decode_last compute={decode_summary['compute_ms']:.3f} total={decode_summary['total_ms']:.3f} total+flush={decode_summary['total_with_final_flush_ms']:.3f} "
        f"decode_seq compute={decode_seq_summary['compute_ms']:.3f} total={decode_seq_summary['total_ms']:.3f} total+flush={decode_seq_summary['total_with_final_flush_ms']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
