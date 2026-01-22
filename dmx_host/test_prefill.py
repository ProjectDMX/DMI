import torch
from transformers import AutoTokenizer, PreTrainedModel
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

from monitoring import MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig

from datetime import datetime
import clickhouse_client

from engine import PipelinedEngine, StageConfig, QueueConfig, EngineConfig
import dmx_interface
import torch



# MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 MON_NATIVE_BATCH=1 PYTHONPATH=<path_to_backend_dmx_repo>:<path_to_host_side_repo>/src:$PYTHONPATH python3 integrated_prefill.py clickhouse-driver


"""
def make_logits_from_last_hidden_states_and_embeddings(base_model: PreTrainedModel, last_hidden_states):
    embedding = base_model.get_input_embeddings().weight # Only work for models that use input_embeddings as output_embeddings.
    logits = last_hidden_states @ embedding.T
    return logits
"""

    

def generate_prefill_futures(offload_engine, tokenizer, base_model, prompt: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = offload_engine

    token = tokenizer.encode(prompt, return_tensors="pt").to(device)

    
    
    hf_hooked_model = base_model

    

    

    engine.begin_request(0)
    engine.start_step(phase="prefill")
    outputs, cache_dict = hf_hooked_model.run_with_cache(
        token,
        use_cache=True,
        past_key_values=None,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )

    engine.end_step()


    cache_dict.keys()
    # print(cache_dict['blocks.0.hook_resid_pre'].shape)

    # engine.clear_completed_results()
    # print(outputs)
    # print(type(outputs))
    # print(cache_dict.keys())
    return cache_dict



def get_str_timestemp(fake: bool) -> str:
    if fake:
        return '<timestamp>'
    else:
        return datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")

if __name__ == "__main__":
    
    model_id = "gpt2"
    
    prompt = "The future of AI is"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(
            step_stride=1,
            step_offset=0,
            warmup_steps=0,
            capture_prefill=True,
            capture_decode=True,
            request_stride=1,
            request_offset=0,
            warmup_requests=0,
        ),
    )

    engine = MonitoringEngine(async_enabled=device.type == "cuda", config=config)
    print(f"native: {getattr(engine, '_using_native_backend', False)}")
    
    hf_hooked_model = HookedGPT2Model.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_hooked_model.to(device)
    hf_hooked_model.eval()
    hf_hooked_model.monitoring_engine = engine
    init_ms = engine.prepare_for_model(hf_hooked_model)
    
    

    # Run the model and get logits and activations
    timestamp = get_str_timestemp(fake=True)
    # t1 = time.perf_counter()
    activations = generate_prefill_futures(engine, tokenizer, hf_hooked_model, prompt)
    # t2 = time.perf_counter()
    # print(f"Prefill offload time is {t2 - t1}")

    request_num_id = 0
    request_id = f"{request_num_id}.{timestamp}"



    cfg = clickhouse_client.ClickHouseClientConfig()
    cfg.host = "localhost"
    cfg.port = 9000
    cfg.username = "default"
    cfg.password = ""
    cfg.database = "default"
    cfg.table = "offload"
    cfg.secure = False
    cfg.client_side_compress = False   # or True / "lz4" / "zstd"
    cfg.client_settings = None         # or {"async_insert": 1, "wait_for_async_insert": 0}
    cfg.create_database_if_missing = True
    cfg.drop_existing_database = True
    cfg.index_granularity = 8192
    
    
    stage_one_config = StageConfig("stage_one", 1, dmx_interface.stage_one_parsing_and_wait, 
                                   None, dmx_interface.stage_one_thread_init, dmx_interface.stage_one_thread_cleanup, 
                                   QueueConfig(1, None, None, None, None, 400, None))
    statge_two_config = StageConfig("stage_two", 1, clickhouse_client.clickhouse_insert, 
                                    cfg,
                                    clickhouse_client.clickhouse_init, 
                                    clickhouse_client.clickhouse_cleanup,
                                    QueueConfig(1, None, None, None, None, 400, None))
    engine_config = EngineConfig()
    engine = PipelinedEngine([stage_one_config, statge_two_config], input_handler=dmx_interface.input_handler_v1,
                             config=engine_config
                             )

    key = (model_id, request_id)
    engine.start()
    engine.submit([key], [0], [activations])
    engine.stop()
