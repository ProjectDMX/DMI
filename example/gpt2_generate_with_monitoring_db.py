import os

import torch
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
from monitoring.config import CaptureSchedule
from monitoring.generate import generate_with_monitoring


def _build_db_config() -> ClickHouseClientConfig:
    cfg = ClickHouseClientConfig()
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


def _build_queue_config(*, high_watermark_items: int = 400) -> QueueConfig:
    q = QueueConfig()
    q.min_batch_items = 1
    q.high_watermark_items = int(high_watermark_items)
    return q


def _build_ingress_policy() -> EnqueuePolicy:
    p = EnqueuePolicy()
    p.block = False
    p.on_full = OnFullPolicy.RAISE
    p.on_closed = OnClosedPolicy.RAISE
    return p


def _build_host_config(db_cfg: ClickHouseClientConfig, *, debug: bool = False) -> HostEngineConfig:
    stage_one = StageConfig.process_future(parallelism=1, name="process_future", debug=bool(debug))
    stage_two = StageConfig.clickhouse_insert(db_cfg, parallelism=1, name="clickhouse_insert")

    stage_one.input_queue = _build_queue_config(high_watermark_items=400)
    stage_two.input_queue = _build_queue_config(high_watermark_items=400)

    ingress = _build_ingress_policy()
    stage_one.ingress_policy = ingress
    stage_two.ingress_policy = ingress

    return HostEngineConfig(stages=[stage_one, stage_two])

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA + native backend.")

    model_id = "gpt2"
    device = torch.device("cuda")

    # Minimal capture set that still supports DB pipeline.
    cfg = MonitoringConfig(
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )

    db_cfg = _build_db_config()
    host_cfg = _build_host_config(db_cfg, debug=bool(cfg.debug))
    engine = MonitoringEngine(
        config=cfg,
        model_id=model_id,
        db_config=host_cfg,
    )

    model = HookedGPT2LMHeadModel.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model.to(device).eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "The future of AI is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = generate_with_monitoring(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_to_keep=0,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    engine.close()


if __name__ == "__main__":
    main()
