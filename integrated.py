import torch
from transformers import AutoTokenizer, PreTrainedModel
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

from monitoring import MonitoringEngine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig

from datetime import datetime
import os
import shutil
import sys
import time



from transformerlens_interface import put_batch, TransformerLensHandle
from hostengine import init_host_engine, terminate_host_engine

# PYTHONPATH=<path_to_backend_dmx_repo>:<path_to_host_side_repo>/src:$PYTHONPATH python3 integrated.py clickhouse-driver


def make_logits_from_last_hidden_states_and_embeddings(base_model: PreTrainedModel, last_hidden_states):
    embedding = base_model.get_input_embeddings().weight # Only work for models that use input_embeddings as output_embeddings.
    logits = last_hidden_states @ embedding.T
    return logits


    

def generate_prefill_data(offload_engine, tokenizer, base_model, prompt: str):
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
    


    """
     ## decode the next token
    next_token = token[:, -1:]
    engine.start_step(phase="decode")
    outputs, cache_dict = hf_hooked_model.run_with_cache(
        next_token,
        use_cache=True,
        past_key_values=outputs.past_key_values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    engine.end_step()
    """
   

    count = 0
    for name, fut in cache_dict.items():
        if fut is None:
            continue
        cache_dict[name] = fut.result()
        if cache_dict[name] is not None:
            count += 1

    assert count == len(cache_dict.items()), "Some are missing"
    print("All tensor are ready")


    cache_dict.keys()
    # print(cache_dict['blocks.0.hook_resid_pre'].shape)

    engine.clear_completed_results()
    cache_dict["token_ids"] = token
    # print(outputs)
    logits = make_logits_from_last_hidden_states_and_embeddings(hf_hooked_model, outputs["last_hidden_state"])
    cache_dict["final_logits"] = logits
    # print(type(outputs))
    print(cache_dict.keys())
    return cache_dict



def get_str_timestemp(fake: bool) -> str:
    if fake:
        return '<timestamp>'
    else:
        return datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")

if __name__ == "__main__":
    assert len(sys.argv) == 2
    key_format = ('model_id', 'request_id', 'layer_no', 'act_name', 'start_token_idx', 'end_token_idx')
    key_types = ('TEXT', 'TEXT', 'INT', 'TEXT', 'INT', 'INT')
    backend = sys.argv[1].lower()
    db_connector = None
    if backend == "filesys":
        dir_name = "filesys"

        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        from db_connector.filesys import FileSystem
        db_connector = FileSystem(dir_name, '__', TransformerLensHandle.get_value_key_suffixes())
    elif backend == "leveldb":
        dir_name = "leveldb"
        
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        from db_connector.leveldb import LocalLevelDB
        db_connector = LocalLevelDB(dir_name, True, '__', TransformerLensHandle.get_value_key_suffixes())
    elif backend == "sqlite":
        file_name = "sqlite"
        
        if os.path.exists(file_name):
            os.remove(file_name)
        from db_connector.sql import LocalSQLite
        db_connector = LocalSQLite(file_name, "offload", key_format, ("start_token_idx", ), TransformerLensHandle.get_value_key_suffixes(),
                                   key_types + TransformerLensHandle.get_value_sql_types())
    elif backend == "mysql":
        from db_connector.sql import RemoteMySQL
        db_connector = RemoteMySQL(primary_key_column_names=key_format, value_column_names=TransformerLensHandle.get_value_key_suffixes(), 
                                   type_of_columns=key_types + TransformerLensHandle.get_value_sql_types(), drop_existing_database=True)
    elif backend == "clickhouse":
        from db_connector.clickhouse import CHConnect
        db_connector = CHConnect(primary_key_column_names=key_format, value_column_names=TransformerLensHandle.get_value_key_suffixes(), 
                                 type_of_columns=key_types + TransformerLensHandle.get_value_sql_types(), delete_before_insert=False, drop_existing_database=True)
        # Set drop existing database only for testing.
    elif backend == "clickhouse-driver":
        from db_connector.clickhouse import CHDriver
        db_connector = CHDriver(primary_key_column_names=key_format, value_column_names=TransformerLensHandle.get_value_key_suffixes(), 
                                 type_of_columns=key_types + TransformerLensHandle.get_value_sql_types(), delete_before_insert=False, drop_existing_database=True)
    else:
        raise NotImplementedError()
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
    t1 = time.perf_counter()
    activations = generate_prefill_data(engine, tokenizer, hf_hooked_model, prompt)
    t2 = time.perf_counter()
    print(f"Prefill offload time is {t2 - t1}")

    request_num_id = 0
    request_id = f"{request_num_id}.{timestamp}"

    
    engine = init_host_engine(2 * 1024 * 1024 * 1024, db_connector)
    key = (model_id, request_id)
    put_batch(engine, [key], [0], [activations], True)
    
    terminate_host_engine(engine)
