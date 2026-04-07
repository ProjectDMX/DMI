#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

import torch
from tqdm import tqdm

from common import (
    BatchMetrics,
    add_shared_args,
    batch_target_lengths,
    build_rendered_prompts,
    build_tokenizer,
    compile_generate_kwargs,
    device_sync,
    iter_batches,
    load_jsonl_examples,
    make_output_path,
    warmup_decode_tokens,
    maybe_sort_by_length,
    parse_pad_buckets,
    parsed_limit,
    resolve_model_id,
    summarize_run,
    tokenize_batch,
    make_bucket_warmup_inputs,
    warmup_batches,
    write_json,
)


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
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )


def _build_ring_cfg(args) -> object:
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


def _build_db_host_cfg(args):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Proj-DMI offline baseline with ring transport.")
    add_shared_args(parser)
    parser.add_argument("--proj-dmi-mode", choices=["ring_null", "ring_db"], default="ring_null")
    parser.add_argument("--hook-selection", default="hidden-states")
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
    parser.add_argument("--db-table", default="offload")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from monitoring import AdvanceConfig, MonitoringConfig, MonitoringEngine, NativePartialSealConfig
    from monitoring.config import CaptureSchedule, HookSelection
    from monitoring.generate import generate_with_monitoring

    model_id = resolve_model_id(args.model)
    compile_enabled = not args.disable_compile
    device = torch.device("cuda")
    default_hook_sel = "hidden-states,final_ln,logits" if args.capture_mode == "hs_logits" else "hidden-states,final_ln"
    hook_sel = str(args.hook_selection) if str(args.hook_selection) != "hidden-states" else default_hook_sel

    examples = load_jsonl_examples(args.sample_file, limit=parsed_limit(args))
    tokenizer = build_tokenizer(model_id, local_files_only=args.local_files_only)
    rendered = maybe_sort_by_length(
        build_rendered_prompts(tokenizer, examples),
        enabled=not args.no_sort_by_length,
    )

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

    if args.proj_dmi_mode == "ring_db":
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

    model = _load_hooked_model(model_id, local_files_only=args.local_files_only)
    model.to(device).eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    batch_metrics = []
    gen_kwargs = compile_generate_kwargs(compile_enabled)
    pad_buckets = parse_pad_buckets(args.pad_buckets)

    bucket_inputs = (
        make_bucket_warmup_inputs(
            tokenizer,
            pad_buckets,
            args.batch_size,
            device,
            active_tokens=(int(args.max_input_tokens) if int(args.max_input_tokens) > 0 else max(pad_buckets)),
        )
        if compile_enabled and pad_buckets
        else []
    )
    with torch.no_grad():
        for bi in bucket_inputs:
            _ = generate_with_monitoring(
                model, input_ids=bi["input_ids"], attention_mask=bi["attention_mask"],
                max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                hook_selection=hook_sel, **gen_kwargs,
            )
            device_sync(device)
        for warmup_batch in warmup_batches(rendered, args.batch_size, count=2):
            warmup_texts = [item["prompt_text"] for item in warmup_batch]
            warmup_encoded = tokenize_batch(tokenizer, warmup_texts, pad_buckets=pad_buckets,
                pad_to_multiple_of=int(args.pad_to_multiple_of), max_input_tokens=int(args.max_input_tokens))
            _ = generate_with_monitoring(
                model, input_ids=warmup_encoded["input_ids"].to(device),
                attention_mask=warmup_encoded["attention_mask"].to(device),
                max_new_tokens=warmup_decode_tokens(warmup_batch, int(args.max_new_tokens)), do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                hook_selection=hook_sel, **gen_kwargs,
            )
            device_sync(device)
    print(f"Warmup done ({len(bucket_inputs)} buckets + 2 real batches).", flush=True)

    try:
        total_batches = math.ceil(len(rendered) / args.batch_size)
        with torch.no_grad():
            t0 = time.perf_counter()
            for batch_index, batch in tqdm(enumerate(iter_batches(rendered, args.batch_size)), total=total_batches, desc="proj_dmi"):
                texts = [item["prompt_text"] for item in batch]
                encoded = tokenize_batch(
                    tokenizer,
                    texts,
                    pad_buckets=pad_buckets,
                    pad_to_multiple_of=int(args.pad_to_multiple_of),
                    max_input_tokens=int(args.max_input_tokens),
                )
                target_lengths = batch_target_lengths(batch, int(args.max_new_tokens))
                batch_max_new_tokens = max(target_lengths)

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                input_tokens = int(attention_mask.sum().item())
                padded_tokens = int(input_ids.numel())

                device_sync(device)
                batch_t0 = time.perf_counter()
                _ = generate_with_monitoring(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=batch_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    hook_selection=hook_sel,
                    **gen_kwargs,
                )
                device_sync(device)
                batch_t1 = time.perf_counter()
                batch_metrics.append(
                    BatchMetrics(
                        batch_index=batch_index,
                        batch_size=len(batch),
                        input_tokens=input_tokens,
                        padded_tokens=padded_tokens,
                        target_generated_tokens=sum(target_lengths),
                        actual_generated_tokens=len(batch) * batch_max_new_tokens,
                        seconds=batch_t1 - batch_t0,
                    )
                )
            total_seconds = time.perf_counter() - t0
    finally:
        engine.close()

    payload = summarize_run(
        baseline="proj_dmi",
        model=args.model,
        model_id=model_id,
        sample_file=args.sample_file,
        repeat_index=args.repeat_index,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        sort_by_length=not args.no_sort_by_length,
        compile_enabled=compile_enabled,
        dataset_size=len(rendered),
        total_seconds=total_seconds,
        batch_metrics=batch_metrics,
        extra={
            "local_files_only": bool(args.local_files_only),
            "hook_selection": hook_sel,
            "capture_mode": str(args.capture_mode),
            "proj_dmi_mode": args.proj_dmi_mode,
            "ring_task_entries": int(args.ring_task_entries),
            "ring_payload_mb": int(args.ring_payload_mb),
            "ring_pinned_mb": int(args.ring_pinned_mb),
            "ch_queue_max_items": int(args.ch_queue_max_items),
            "ch_queue_max_size_mb": int(args.ch_queue_max_size_mb),
            "pad_buckets": pad_buckets,
            "pad_to_multiple_of": int(args.pad_to_multiple_of),
            "max_input_tokens": int(args.max_input_tokens),
            "decode_length_mode": "per_sample_target",
            "max_new_tokens_cap": int(args.max_new_tokens),
        },
    )

    out_path = make_output_path(
        results_dir=args.results_dir,
        baseline="proj_dmi",
        model=args.model,
        sample_file=args.sample_file,
        batch_size=args.batch_size,
        repeat_index=args.repeat_index,
    )
    write_json(out_path, payload)
    print(f"Saved {out_path}")
    print(
        f"[proj_dmi:{args.proj_dmi_mode}] prompts/s={payload['prompts_per_s']:.3f} "
        f"target_tok/s={payload['target_generated_tokens_per_s']:.3f} "
        f"compute_tok/s={payload['actual_generated_tokens_per_s']:.3f}"
    )


if __name__ == "__main__":
    main()
