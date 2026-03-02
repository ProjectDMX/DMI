import argparse

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import HostEngineConfig, MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig

from dmx_host.engine import EngineConfig, QueueConfig, StageConfig
import dmx_host.dmx_interface as dmx_interface


def _drop_stage(items):
    # Minimal sink: consume stage_one output without DB writes.
    print(f"[host_engine] stage_two items={len(items)}")
    return []


def _build_host_config(with_db: bool, db_cfg=None):
    stage_one = StageConfig(
        name="stage_one",
        parallelism=1,
        process_fn=dmx_interface.stage_one_parsing_and_wait,
        thread_init_config=None,
        thread_init=dmx_interface.stage_one_thread_init,
        thread_cleanup=dmx_interface.stage_one_thread_cleanup,
        input_queue=QueueConfig(1, None, None, None, None, 400, None),
    )
    if with_db:
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
    else:
        stage_two = StageConfig(
            name="stage_two",
            parallelism=1,
            process_fn=_drop_stage,
            thread_init_config=None,
            thread_init=None,
            thread_cleanup=None,
            input_queue=QueueConfig(1, None, None, None, None, 400, None),
        )
    return HostEngineConfig(
        stages=[stage_one, stage_two],
        input_handler=dmx_interface.input_handler_v1,
        engine_config=EngineConfig(),
    )


def _build_db_config(args):
    try:
        import dmx_host.clickhouse_client as clickhouse_client
    except Exception:
        import clickhouse_client  # type: ignore
    cfg = clickhouse_client.ClickHouseClientConfig()
    cfg.host = args.db_host
    cfg.port = args.db_port
    cfg.username = args.db_username
    cfg.password = args.db_password
    cfg.database = args.db_database
    cfg.table = args.db_table
    cfg.secure = False
    cfg.client_side_compress = False
    cfg.client_settings = None
    cfg.create_database_if_missing = True
    cfg.drop_existing_database = args.drop_db
    cfg.index_granularity = 8192
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-db", action="store_true")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--db-username", default="default")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-database", default="default")
    parser.add_argument("--db-table", default="offload")
    parser.add_argument("--drop-db", action="store_true")
    args = parser.parse_args()

    model_id = "gpt2"
    prompt = "The future of AI is"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This test requires CUDA (native backend).")

    cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(
            step_stride=1,
            step_offset=0,
            warmup_steps=0,
            capture_prefill=True,
            capture_decode=False,
            request_stride=1,
            request_offset=0,
            warmup_requests=0,
        ),
    )

    db_cfg = _build_db_config(args) if args.with_db else None
    host_cfg = _build_host_config(args.with_db, db_cfg)
    engine = MonitoringEngine(
        async_enabled=True,
        config=cfg,
        model_id=model_id,
        db_config=host_cfg,
    )

    model = HookedGPT2LMHeadModel.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    tok = AutoTokenizer.from_pretrained(model_id)
    tokens = tok.encode(prompt, return_tensors="pt").to(device)

    engine.start_step(phase="prefill")
    _outputs, _cache_dict = model.run_with_cache(
        tokens,
        use_cache=True,
        past_key_values=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    )
    engine.end_step()
    engine.close()


if __name__ == "__main__":
    main()
