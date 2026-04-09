"""Standalone script: run HOOKED model with ring transport monitoring.

Activations go to ClickHouse. Saves model_id to disk for the comparator.

Usage:
    python -m tests.hf_monitored_runner --output-dir /tmp/hf_mon
"""
import argparse
import json
import os
import uuid

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args, _ = p.parse_known_args()

    from monitoring import MonitoringEngine
    from monitoring._native_engine import ClickHouseClientConfig
    from monitoring.generate import generate_with_monitoring
    from transformers import AutoTokenizer

    # Config from env (same vars as the test)
    batch_size = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    cuda_graphs = os.environ.get("E2E_CUDA_GRAPHS", "0") == "1"
    _MODEL_ALIASES = {"qwen3": "Qwen/Qwen3-4B"}
    model_key = os.environ.get("E2E_MODEL", "gpt2")
    hf_model_id = _MODEL_ALIASES.get(model_key, model_key)

    device = torch.device("cuda")

    # Load HOOKED model
    if "qwen3" in hf_model_id.lower() or "qwen" in hf_model_id.lower():
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM
        model_cls = HookedQwen3ForCausalLM
    else:
        from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel
        model_cls = HookedGPT2LMHeadModel

    model = model_cls.from_pretrained(
        hf_model_id, attn_implementation="eager", torch_dtype=torch.float16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    prompts = [("Hello " * (i + 1)).strip() for i in range(batch_size)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ClickHouse config
    ch_cfg = ClickHouseClientConfig()
    ch_cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    ch_cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    ch_cfg.username = os.environ.get("DMX_DB_USER", "default")
    ch_cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    ch_cfg.database = os.environ.get("DMX_DB_DATABASE", "default")
    ch_cfg.table = os.environ.get("DMX_DB_TABLE", "offload")
    ch_cfg.secure = False
    ch_cfg.client_side_compress = "none"
    ch_cfg.create_database_if_missing = bool(int(os.environ.get("DMX_DB_CREATE_IF_MISSING", "1")))
    ch_cfg.drop_existing_database = bool(int(os.environ.get("DMX_DB_DROP_EXISTING", "1")))
    ch_cfg.index_granularity = 8192

    # Monitoring config
    from monitoring import MonitoringConfig, AdvanceConfig, NativePartialSealConfig
    from monitoring.config import CaptureSchedule, HookSelection

    hook_mode = os.environ.get("E2E_HOOK_MODE", "full")
    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode=hook_mode),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True, chunk_bytes=int(os.environ.get("E2E_CHUNK_BYTES", str(256 * 1024))),
            cap_enabled=True, cap_ratio=0.8, driver_guard_mb=1024,
        ),
        advance=AdvanceConfig(),
    )
    if hasattr(mon_cfg, "eos_token_id"):
        mon_cfg.eos_token_id = eos_id
    if hasattr(mon_cfg, "pad_token_id"):
        mon_cfg.pad_token_id = pad_id

    # Host engine config
    from monitoring import HostEngineConfig
    from monitoring._native_engine import StageConfig
    ch_parallelism = int(os.environ.get("DMX_CH_PARALLELISM", "10"))
    stage = StageConfig.clickhouse_insert(ch_cfg, parallelism=ch_parallelism, name="ch_insert")
    q = stage.input_queue
    q.max_batch_items = int(os.environ.get("DMX_CH_MAX_BATCH_ITEMS", "1024"))
    q.high_watermark_items = q.max_batch_items
    q.max_batch_size = int(os.environ.get("DMX_CH_MAX_BATCH_BYTES", str(2048 * 1024 * 1024)))
    q.high_watermark_size = q.max_batch_size
    host_cfg = HostEngineConfig(stages=[stage])

    # Ring config
    from monitoring._native_engine import RingConfig
    ring_cfg = RingConfig()
    ring_cfg.task_ring_entries = int(os.environ.get("E2E_RING_TASK_ENTRIES", "16384"))
    ring_cfg.payload_ring_bytes = int(os.environ.get("E2E_RING_PAYLOAD_BYTES", str(4 * 1024**3)))
    ring_cfg.pinned_staging_bytes = int(os.environ.get("E2E_RING_PINNED_BYTES", str(4 * 1024**3)))
    ring_cfg.drain_poll_timeout_us = int(os.environ.get("E2E_DRAIN_POLL_TIMEOUT_US", "100"))
    ring_cfg.clone_slices = int(os.environ.get("E2E_CLONE_SLICES", "0")) != 0
    ring_cfg.insert_queue_max_bytes = int(os.environ.get("E2E_INSERT_QUEUE_MAX_BYTES", str(512 * 1024**2)))
    ring_cfg.insert_queue_max_items = int(os.environ.get("E2E_INSERT_QUEUE_MAX_ITEMS", "4096"))

    unique_run_model_id = f"e2e_correctness_hf::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        async_enabled=True, config=mon_cfg, model_id=unique_run_model_id, db_config=host_cfg
    )
    engine.enable_ring_transport(ring_cfg)
    model.monitoring_engine = engine

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        logits_to_keep=0,
    )
    if cuda_graphs:
        gen_kwargs["cache_implementation"] = "static"

    try:
        with torch.no_grad():
            _ = generate_with_monitoring(model, **gen_kwargs)
    finally:
        engine.close()

    # Save metadata for comparator
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": unique_run_model_id,
            "hf_model_id": hf_model_id,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "db_host": ch_cfg.host,
            "db_port": ch_cfg.port,
            "db_database": ch_cfg.database,
            "db_table": ch_cfg.table,
        }, f)

    print(f"[hf_monitored_runner] Done, model_id={unique_run_model_id}", flush=True)


if __name__ == "__main__":
    main()
