import argparse
import contextlib
import json
import math
import os
import time
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import (
    ClickHouseClientConfig,
    EnqueuePolicy,
    HostEngineConfig,
    MonitoringConfig,
    MonitoringEngine,
    OnClosedPolicy,
    OnFullPolicy,
    QueueConfig,
    StageConfig,
)
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

def _load_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _iter_batches(items: List[str], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield idx // batch_size, items[idx : idx + batch_size]


def _build_db_config() -> ClickHouseClientConfig:
    cfg = ClickHouseClientConfig()
    cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    cfg.username = os.environ.get("DMX_DB_USER", "default")
    cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    cfg.database = os.environ.get("DMX_DB_DATABASE", "default")
    cfg.table = os.environ.get("DMX_DB_TABLE", "offload")
    cfg.secure = False
    # Expects a string: "none" | "lz4" | "zstd" | "true" | "false"
    cfg.client_side_compress = "none"
    cfg.client_settings = None
    cfg.create_database_if_missing = True
    # Previous version dropped the DB each run; keep that default here,
    # but allow overriding via env var.
    cfg.drop_existing_database = bool(int(os.environ.get("DMX_DB_DROP_EXISTING", "1")))
    cfg.index_granularity = 8192
    return cfg

@contextlib.contextmanager
def _nvtx_range(name: str):
    try:
        if not torch.cuda.is_available():
            yield
            return
        from torch.cuda import nvtx  # type: ignore
    except Exception:
        yield
        return
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def _build_queue_config(*, high_watermark_items: int | None = None) -> QueueConfig:
    # Keep the benchmark effectively "unbounded" by default (matches old script),
    # so long runs don't fail due to queue backpressure.
    q = QueueConfig()
    q.min_batch_items = 1
    q.high_watermark_items = high_watermark_items
    return q


def _build_ingress_policy() -> EnqueuePolicy:
    p = EnqueuePolicy()
    p.block = False
    p.on_full = OnFullPolicy.RAISE
    p.on_closed = OnClosedPolicy.RAISE
    return p


def _build_host_config(db_cfg: ClickHouseClientConfig) -> HostEngineConfig:
    # Stage 1: wait on BackendFuture + parse payloads
    stage_one = StageConfig.process_future(parallelism=1, name="process_future")
    # Stage 2: insert into ClickHouse
    stage_two = StageConfig.clickhouse_insert(db_cfg, parallelism=10, name="clickhouse_insert")

    stage_one.input_queue = _build_queue_config()
    stage_two.input_queue = _build_queue_config()

    ingress = _build_ingress_policy()
    stage_one.ingress_policy = ingress
    stage_two.ingress_policy = ingress

    # MonitoringEngine currently expects exactly two stages.
    return HostEngineConfig(stages=[stage_one, stage_two])

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring generate benchmark")
    parser.add_argument("--prompts", default="benchmark/data/prompts.txt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--no-db", action="store_true", help="Disable host_engine DB submission.")
    args = parser.parse_args()

    # Native backend must be enabled for MonitoringEngine + (optional) DB futures.
    os.environ.setdefault("MON_NATIVE_TO_CPU", "1")
    os.environ.setdefault("MON_NATIVE_CALLBACK", "1")
    os.environ.setdefault("MON_NATIVE_BUILDER", "1")
    os.environ.setdefault("MON_NATIVE_BATCH", "0")
    # Enable pinned and memcpy pool.
    os.environ.setdefault("MON_NATIVE_PINNED", "1")
    os.environ.setdefault("MON_NATIVE_PINPOOL", "1")
    os.environ.setdefault("MON_NATIVE_HOST_COPY_THREADS", "5")

    if args.no_db:
        # No DB path: disable auto-cleanup; we'll clear once after the run.
        os.environ["MON_NATIVE_AUTOCLEAR"] = "0"
    else:
        # Prevent native backend from clearing futures before host_engine consumes them.
        os.environ["MON_NATIVE_AUTOCLEAR"] = "0"

    if not torch.cuda.is_available():
        raise RuntimeError("Monitoring benchmark requires CUDA + native backend.")

    prompts = _load_prompts(args.prompts)
    device = torch.device(args.device)

    cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )

    host_cfg = None
    if not args.no_db:
        db_cfg = _build_db_config()
        host_cfg = _build_host_config(db_cfg)

    engine = MonitoringEngine(
        async_enabled=True,
        config=cfg,
        model_id=args.model,
        db_config=host_cfg,
    )

    model = HookedGPT2LMHeadModel.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    model.to(device).eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    per_batch = []
    total_tokens = 0
    start = time.perf_counter()
    loop_end = None
    host_timings = None

    try:
        total_batches = math.ceil(len(prompts) / args.batch_size)
        use_nvtx = os.environ.get("BENCH_NVTX", "0") == "1"
        nvtx_ctx = _nvtx_range("monitoring_generate") if use_nvtx else contextlib.nullcontext()
        with nvtx_ctx, torch.no_grad():
            for batch_idx, batch_prompts in tqdm(
                _iter_batches(prompts, args.batch_size),
                total=total_batches,
                desc="monitoring_generate",
            ):
                encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                t0 = time.perf_counter()
                _ = generate_with_monitoring(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                )
                t1 = time.perf_counter()

                batch_seconds = t1 - t0
                batch_tokens = int(input_ids.shape[0] * args.max_new_tokens)
                total_tokens += batch_tokens
                per_batch.append(
                    {
                        "batch_idx": batch_idx,
                        "batch_size": int(input_ids.shape[0]),
                        "seconds": batch_seconds,
                        "tokens": batch_tokens,
                        "tokens_per_s": batch_tokens / batch_seconds if batch_seconds > 0 else None,
                    }
                )   
        loop_end = time.perf_counter()
        engine.resolve_all()
        t_r_ed = time.perf_counter()
    finally:
        if engine._host_engine is not None:
            try:
                host_timings = engine._host_engine.timings()
            except Exception:
                host_timings = None
        if args.no_db:
            try:
                engine.resolve_all()
                engine.clear_completed_results()
            except Exception:
                pass
        engine.close()

    if loop_end is None:
        loop_end = time.perf_counter()
    total_seconds = time.perf_counter() - start
    total_to_cpu = t_r_ed - start
    main_seconds = loop_end - start
    result = {
        "backend": "monitoring",
        "model": args.model,
        "device": str(device),
        "prompts": len(prompts),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "main_seconds": main_seconds,
        "to_cpu_seconds": total_to_cpu,
        "total_seconds": total_seconds,
        "total_tokens": total_tokens,
        "tokens_per_s": total_tokens / total_seconds if total_seconds > 0 else None,
        "per_batch": per_batch,
    }
    if host_timings:
        result["host_engine_timings"] = host_timings
        ingest = host_timings.get("ingest", {})
        if ingest:
            print(
                "[HostEng/Timing] input_handler_s=",
                round(float(ingest.get("input_handler_s", 0.0)), 6),
                " enqueue_s=",
                round(float(ingest.get("enqueue_s", 0.0)), 6),
                " input_handler_avg_s=",
                round(float(ingest.get("input_handler_avg_s", 0.0)), 6),
                " enqueue_avg_s=",
                round(float(ingest.get("enqueue_avg_s", 0.0)), 6),
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
