import os
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import HostEngineConfig, MonitoringConfig, MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring

from dmx_host.engine import EngineConfig, QueueConfig, StageConfig
import dmx_host.dmx_interface as dmx_interface


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
    # Native backend must be enabled for DB futures.
    os.environ.setdefault("MON_NATIVE_TO_CPU", "1")
    os.environ.setdefault("MON_NATIVE_CALLBACK", "1")
    os.environ.setdefault("MON_NATIVE_BUILDER", "1")
    os.environ.setdefault("MON_NATIVE_BATCH", "0")
    # Prevent native backend from clearing futures before host_engine consumes them.
    os.environ.setdefault("MON_NATIVE_AUTOCLEAR", "0")

    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA + native backend.")

    model_id = "gpt2"
    device = torch.device("cuda")

    # Minimal capture set that still supports DB pipeline.
    cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )

    db_cfg = _build_db_config()
    host_cfg = _build_host_config(db_cfg)
    engine = MonitoringEngine(
        async_enabled=True,
        config=cfg,
        model_id=model_id,
        # db_config=host_cfg,
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
