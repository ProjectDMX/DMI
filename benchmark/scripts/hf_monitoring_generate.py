import argparse
import json
import os
import time
from typing import List

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import HostEngineConfig, MonitoringConfig, MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring

from dmx_host.engine import EngineConfig, QueueConfig, StageConfig
import dmx_host.dmx_interface as dmx_interface


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


def _build_db_config():
    try:
        import dmx_host.clickhouse_client as clickhouse_client
    except Exception:
        import clickhouse_client  # type: ignore

    cfg = clickhouse_client.ClickHouseClientConfig()
    cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    cfg.username = os.environ.get("DMX_DB_USER", "default")
    cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    cfg.database = os.environ.get("DMX_DB_DATABASE", "default")
    cfg.table = os.environ.get("DMX_DB_TABLE", "offload")
    cfg.secure = False
    cfg.client_side_compress = False
    cfg.client_settings = None
    cfg.create_database_if_missing = True
    cfg.drop_existing_database = False
    cfg.index_granularity = 8192
    return cfg


def _build_host_config(db_cfg):
    stage_one = StageConfig(
        name="stage_one",
        parallelism=1,
        process_fn=dmx_interface.stage_one_parsing_and_wait,
        thread_init_config=None,
        thread_init=dmx_interface.stage_one_thread_init,
        thread_cleanup=dmx_interface.stage_one_thread_cleanup,
        input_queue=QueueConfig(1, None, None, None, None, 400, None),
    )
    try:
        import dmx_host.clickhouse_client as clickhouse_client
    except Exception:
        import clickhouse_client  # type: ignore
    stage_two = StageConfig(
        name="stage_two",
        parallelism=1,
        process_fn=clickhouse_client.clickhouse_insert,
        thread_init_config=db_cfg,
        thread_init=clickhouse_client.clickhouse_init,
        thread_cleanup=clickhouse_client.clickhouse_cleanup,
        input_queue=QueueConfig(1, None, None, None, None, 400, None),
    )
    return HostEngineConfig(
        stages=[stage_one, stage_two],
        input_handler=dmx_interface.input_handler_v1,
        engine_config=EngineConfig(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring generate benchmark")
    parser.add_argument("--prompts", default="benchmark/data/prompts.txt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--no-db", action="store_true", help="Disable host_engine DB submission.")
    args = parser.parse_args()

    # Native backend must be enabled for DB futures.
    os.environ.setdefault("MON_NATIVE_TO_CPU", "1")
    os.environ.setdefault("MON_NATIVE_CALLBACK", "1")
    os.environ.setdefault("MON_NATIVE_BUILDER", "1")
    os.environ.setdefault("MON_NATIVE_BATCH", "0")
    # When DB is disabled there is no consumer for futures; enable autoclear to avoid unbounded growth.
    if args.no_db:
        os.environ["MON_NATIVE_AUTOCLEAR"] = "1"
    else:
        # Prevent native backend from clearing futures before host_engine consumes them.
        os.environ.setdefault("MON_NATIVE_AUTOCLEAR", "0")

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

    per_batch = []
    total_tokens = 0
    start = time.perf_counter()
    loop_end = None

    try:
        with torch.no_grad():
            for batch_idx, batch_prompts in _iter_batches(prompts, args.batch_size):
                encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # if device.type == "cuda":
                #     torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = generate_with_monitoring(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # if device.type == "cuda":
                #     torch.cuda.synchronize()
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
    finally:
        engine.close()

    if loop_end is None:
        loop_end = time.perf_counter()
    total_seconds = time.perf_counter() - start
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
        "total_seconds": total_seconds,
        "total_tokens": total_tokens,
        "tokens_per_s": total_tokens / total_seconds if total_seconds > 0 else None,
        "per_batch": per_batch,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
