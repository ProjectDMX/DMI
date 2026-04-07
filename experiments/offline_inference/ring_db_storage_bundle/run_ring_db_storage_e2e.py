#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for _path in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "transformers" / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from common import DEFAULT_RESULTS_DIR, build_tokenizer, ensure_dir, make_output_path, resolve_model_id, write_json


DEFAULT_HOOK_SELECTION = "hidden-states,final_ln,logits"


class _NullHostEngine:
    def start(self) -> None:
        pass

    def stop(self, *args, **kwargs) -> None:
        pass

    def join(self, *args, **kwargs) -> bool:
        return True

    def close_input(self) -> None:
        pass

    def request_abort(self) -> None:
        pass

    def failures(self) -> list:
        return []

    def raise_if_failed(self) -> None:
        pass

    def submit_direct(self, *args, **kwargs) -> None:
        pass


def _make_synthetic_batch(
    tokenizer: Any,
    *,
    batch_size: int,
    prefill_tokens: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    fill_token_id = int(
        tokenizer.eos_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None
        else tokenizer.pad_token_id
    )
    input_ids = torch.full((batch_size, prefill_tokens), fill_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, prefill_tokens), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_hooked_model(model_id: str, *, local_files_only: bool):
    if "qwen3" in model_id.lower():
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM

        model_cls = HookedQwen3ForCausalLM
    elif "llama" in model_id.lower():
        from transformers.models.llama.modeling_llama import HookedLlamaForCausalLM

        model_cls = HookedLlamaForCausalLM
    elif model_id.lower() == "gpt2":
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

        model_cls = HookedGPT2LMHeadModel
    else:
        raise ValueError(
            f"No local Hooked* model implementation found for {model_id}. "
            "Proj-DMI is currently wired for repo HookedQwen3/Llama/GPT2 models."
        )
    return model_cls.from_pretrained(
        pretrained_model_name_or_path=model_id,
        attn_implementation="eager",
        dtype=torch.float16,
        local_files_only=local_files_only,
    )


def _build_ring_cfg(args: argparse.Namespace) -> object:
    from monitoring._native_engine import RingConfig

    rc = RingConfig()
    rc.task_ring_entries = int(args.ring_task_entries)
    rc.payload_ring_bytes = int(args.ring_payload_mb) * 1024 * 1024
    rc.pinned_staging_bytes = int(args.ring_pinned_mb) * 1024 * 1024
    rc.drain_poll_timeout_us = int(args.drain_poll_timeout_us)
    rc.drain_flush_task_ratio = float(args.drain_flush_task_ratio)
    rc.drain_flush_payload_ratio = float(args.drain_flush_payload_ratio)
    rc.drain_flush_entry_threshold = int(args.drain_flush_entry_threshold)
    rc.drain_flush_byte_threshold = int(args.drain_flush_byte_threshold)
    rc.drain_flush_timeout_us = int(args.drain_flush_timeout_us)
    rc.clone_slices = bool(args.clone_slices)
    rc.insert_queue_max_bytes = int(args.ch_queue_max_size_mb) * 1024 * 1024
    rc.insert_queue_max_items = int(args.ch_queue_max_items)
    return rc


def _build_db_host_cfg(args: argparse.Namespace):
    from monitoring import HostEngineConfig
    from monitoring._native_engine import ClickHouseClientConfig, StageConfig

    ch = ClickHouseClientConfig()
    ch.host = args.db_host
    ch.port = int(args.db_port)
    ch.username = args.db_user
    ch.password = args.db_password
    ch.database = args.db_database
    ch.table = args.db_table
    ch.secure = False
    ch.client_side_compress = "none"
    ch.client_settings = None
    ch.create_database_if_missing = True
    ch.drop_existing_database = False
    ch.index_granularity = 8192

    stage = StageConfig.clickhouse_insert(ch, parallelism=int(args.ch_parallelism), name="clickhouse_insert")
    q = stage.input_queue
    q.max_batch_items = int(args.ch_queue_max_items)
    q.high_watermark_items = int(args.ch_queue_max_items)
    q.max_batch_size = int(args.ch_queue_max_size_mb) * 1024 * 1024
    q.high_watermark_size = int(args.ch_queue_max_size_mb) * 1024 * 1024
    return HostEngineConfig(stages=[stage])


def _timed_close(engine: Any) -> Dict[str, float]:
    result = {"ring_ms": 0.0, "db_ms": 0.0, "cleanup_ms": 0.0}

    ring_engine = getattr(engine, "_ring_engine", None)
    if ring_engine is not None:
        t0 = time.perf_counter()
        try:
            ring_engine.stop()
        except Exception:
            pass
        result["ring_ms"] = (time.perf_counter() - t0) * 1000.0

    host_engine = getattr(engine, "_host_engine", None)
    if host_engine is not None:
        t0 = time.perf_counter()
        try:
            host_engine.stop()
        except Exception:
            pass
        result["db_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    try:
        engine.close()
    except Exception:
        pass
    result["cleanup_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DMI ring_db e2e benchmark for comparing ClickHouse tmpfs vs disk storage."
    )
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--storage-label", default="disk")
    parser.add_argument("--proj-dmi-mode", choices=["ring_null", "ring_db"], default="ring_db")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR / "ring_db_storage_e2e"))
    parser.add_argument("--hook-selection", default=DEFAULT_HOOK_SELECTION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")

    parser.add_argument("--ring-task-entries", type=int, default=65536)
    parser.add_argument("--ring-payload-mb", type=int, default=4096)
    parser.add_argument("--ring-pinned-mb", type=int, default=4096)
    parser.add_argument("--drain-poll-timeout-us", type=int, default=100)
    parser.add_argument("--drain-flush-task-ratio", type=float, default=0.0)
    parser.add_argument("--drain-flush-payload-ratio", type=float, default=0.0)
    parser.add_argument("--drain-flush-entry-threshold", type=int, default=0)
    parser.add_argument("--drain-flush-byte-threshold", type=int, default=0)
    parser.add_argument("--drain-flush-timeout-us", type=int, default=0)
    parser.add_argument("--clone-slices", action="store_true")

    parser.add_argument("--ch-parallelism", type=int, default=10)
    parser.add_argument("--ch-queue-max-items", type=int, default=1024)
    parser.add_argument("--ch-queue-max-size-mb", type=int, default=2048)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--db-user", default="default")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-database", default="default")
    parser.add_argument("--db-table", default="offload_bench")
    return parser.parse_args()


def _build_monitoring_engine(args: argparse.Namespace, model_id: str):
    from monitoring import AdvanceConfig, MonitoringConfig, MonitoringEngine, NativePartialSealConfig
    from monitoring.config import CaptureSchedule, HookSelection

    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True,
            chunk_bytes=256 * 1024,
            cap_enabled=True,
            cap_ratio=0.8,
            driver_guard_mb=1024,
        ),
        advance=AdvanceConfig(),
    )

    if str(args.proj_dmi_mode) == "ring_db":
        engine = MonitoringEngine(
            async_enabled=True,
            config=mon_cfg,
            model_id=model_id,
            db_config=_build_db_host_cfg(args),
        )
    else:
        engine = MonitoringEngine(
            async_enabled=True,
            config=mon_cfg,
            model_id=model_id,
            host_engine=_NullHostEngine(),
        )
    engine.enable_ring_transport(_build_ring_cfg(args))
    return engine


def _compile_kwargs(disable_compile: bool) -> Dict[str, Any]:
    if disable_compile:
        return {}
    try:
        from transformers import CompileConfig
    except ImportError:
        return {}
    return {
        "cache_implementation": "static",
        "compile_config": CompileConfig(mode="reduce-overhead", fullgraph=False),
    }


def _run_generate(
    model: Any,
    encoded: Dict[str, torch.Tensor],
    hook_selection: str,
    decode_steps: int,
    compile_kwargs: Dict[str, Any],
    *,
    warmup_null: bool,
) -> Dict[str, Any]:
    from monitoring.generate import generate_with_monitoring

    timing_out: Dict[str, Any] = {}
    with torch.no_grad():
        generate_with_monitoring(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=int(decode_steps),
            min_new_tokens=int(decode_steps),
            do_sample=False,
            hook_selection=str(hook_selection),
            timing_out=(None if warmup_null else timing_out),
            flush_after_generate=(not warmup_null),
            **compile_kwargs,
        )
    return timing_out


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    model_id = resolve_model_id(args.model)
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)

    model = _load_hooked_model(model_id, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    encoded = _make_synthetic_batch(
        tokenizer,
        batch_size=int(args.batch_size),
        prefill_tokens=int(args.prefill_tokens),
        device=device,
    )
    compile_kwargs = _compile_kwargs(bool(args.disable_compile))

    results_root = Path(args.results_dir)
    ensure_dir(results_root)
    run_dir = results_root / f"{args.model.replace('/', '_')}__{args.storage_label}__{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)

    print("=" * 72, flush=True)
    print(
        f"storage={args.storage_label} batches={args.num_batches} "
        f"mode={args.proj_dmi_mode} model={args.model} bs={args.batch_size} prefill={args.prefill_tokens} decode={args.decode_steps}",
        flush=True,
    )
    print("=" * 72, flush=True)

    records: List[Dict[str, Any]] = []
    total_tokens_per_batch = int(args.batch_size) * (int(args.prefill_tokens) + int(args.decode_steps))
    total_tokens_all = total_tokens_per_batch * int(args.num_batches)

    engine = _build_monitoring_engine(args, model_id)
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    ring_engine = getattr(engine, "_ring_engine", None)
    ring_transport = getattr(engine, "_ring_transport", None)
    close_t: Dict[str, float] | None = None

    try:
        if ring_engine is not None:
            ring_engine.set_null_mode(True)
        if ring_transport is not None:
            ring_transport.null_offload = True
        for _ in range(int(args.warmup)):
            _run_generate(
                model,
                encoded,
                args.hook_selection,
                int(args.decode_steps),
                compile_kwargs,
                warmup_null=True,
            )
        if ring_engine is not None:
            ring_engine.set_null_mode(False)
        if ring_transport is not None:
            ring_transport.null_offload = False

        torch.cuda.synchronize()
        run_t0 = time.perf_counter()
        batch_iter = range(1, int(args.num_batches) + 1)
        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=int(args.num_batches), desc=f"{args.storage_label} batches", unit="batch")
        for batch_index in batch_iter:
            batch_start_ms = (time.perf_counter() - run_t0) * 1000.0
            timing_out = _run_generate(
                model,
                encoded,
                args.hook_selection,
                int(args.decode_steps),
                compile_kwargs,
                warmup_null=False,
            )
            batch_forward_ms = float(timing_out.get("generate_total_ms", 0.0))
            batch_flush_ms = float(timing_out.get("total_with_flush_ms", batch_forward_ms))
            record = {
                "storage_label": str(args.storage_label),
                "model": str(args.model),
                "model_id": str(model_id),
                "batch_size": int(args.batch_size),
                "prefill_tokens": int(args.prefill_tokens),
                "decode_steps": int(args.decode_steps),
                "hook_selection": str(args.hook_selection),
                "compile_enabled": not bool(args.disable_compile),
                "proj_dmi_mode": str(args.proj_dmi_mode),
                "batch_index": int(batch_index),
                "batch_forward_done_ms": batch_forward_ms,
                "batch_flush_done_ms": batch_flush_ms,
                "run_forward_done_ms": batch_start_ms + batch_forward_ms,
                "run_flush_done_ms": batch_start_ms + batch_flush_ms,
                "batch_forward_tok_s": (float(total_tokens_per_batch) / batch_forward_ms * 1000.0) if batch_forward_ms > 0 else 0.0,
                "batch_flush_tok_s": (float(total_tokens_per_batch) / batch_flush_ms * 1000.0) if batch_flush_ms > 0 else 0.0,
                "db_database": str(args.db_database),
                "db_table": str(args.db_table),
            }
            out_path = make_output_path(
                results_dir=run_dir,
                baseline="proj_dmi_ring_db_storage_e2e",
                model=model_id,
                sample_file=f"synthetic_prefill_{int(args.prefill_tokens)}_decode_{int(args.decode_steps)}.jsonl",
                batch_size=int(args.batch_size),
                repeat_index=int(batch_index),
            )
            write_json(out_path, record)
            record["path"] = str(out_path)
            records.append(record)

        final_forward_done_ms = float(records[-1]["run_forward_done_ms"]) if records else 0.0
        final_flush_done_ms = float(records[-1]["run_flush_done_ms"]) if records else 0.0
        phase_t1 = time.perf_counter()
        close_t = _timed_close(engine)
        phase_close_ms = (time.perf_counter() - phase_t1) * 1000.0
        print(f"[finalize] engine close path finished in {phase_close_ms:.3f} ms", flush=True)
    finally:
        model.monitoring_engine = None
        engine = None
        ring_engine = None
        ring_transport = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    final_db_done_ms = final_flush_done_ms + float(close_t["ring_ms"]) + float(close_t["db_ms"])

    summary = {
        "storage_label": str(args.storage_label),
        "model": str(args.model),
        "batch_size": int(args.batch_size),
        "prefill_tokens": int(args.prefill_tokens),
        "decode_steps": int(args.decode_steps),
        "hook_selection": str(args.hook_selection),
        "compile_enabled": not bool(args.disable_compile),
        "proj_dmi_mode": str(args.proj_dmi_mode),
        "num_batches": len(records),
        "final_forward_done_ms": final_forward_done_ms,
        "final_flush_done_ms": final_flush_done_ms,
        "final_db_done_ms": final_db_done_ms,
        "ring_close_tail_ms": float(close_t["ring_ms"]),
        "db_close_tail_ms": float(close_t["db_ms"]),
        "cleanup_ms": float(close_t["cleanup_ms"]),
        "batch_forward_done_ms_median": float(statistics.median(r["batch_forward_done_ms"] for r in records)),
        "batch_flush_done_ms_median": float(statistics.median(r["batch_flush_done_ms"] for r in records)),
        "batch_forward_tok_s_median": float(statistics.median(r["batch_forward_tok_s"] for r in records)),
        "batch_flush_tok_s_median": float(statistics.median(r["batch_flush_tok_s"] for r in records)),
        "run_forward_tok_s": (float(total_tokens_all) / final_forward_done_ms * 1000.0) if final_forward_done_ms > 0 else 0.0,
        "run_flush_tok_s": (float(total_tokens_all) / final_flush_done_ms * 1000.0) if final_flush_done_ms > 0 else 0.0,
        "run_db_tok_s": (float(total_tokens_all) / final_db_done_ms * 1000.0) if final_db_done_ms > 0 else 0.0,
        "records": records,
    }

    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fieldnames = [
        "storage_label",
        "model",
        "batch_size",
        "prefill_tokens",
        "decode_steps",
        "compile_enabled",
        "num_batches",
        "final_forward_done_ms",
        "final_flush_done_ms",
        "final_db_done_ms",
        "ring_close_tail_ms",
        "db_close_tail_ms",
        "cleanup_ms",
        "batch_forward_done_ms_median",
        "batch_flush_done_ms_median",
        "batch_forward_tok_s_median",
        "batch_flush_tok_s_median",
        "run_forward_tok_s",
        "run_flush_tok_s",
        "run_db_tok_s",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: summary[k] for k in fieldnames})

    print("-" * 72, flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print(f"summary_csv : {summary_csv}", flush=True)
    print(
        f"final forward={summary['final_forward_done_ms']:.3f} ms  "
        f"flush={summary['final_flush_done_ms']:.3f} ms  "
        f"db_done={summary['final_db_done_ms']:.3f} ms",
        flush=True,
    )
    print("-" * 72, flush=True)


if __name__ == "__main__":
    main()
