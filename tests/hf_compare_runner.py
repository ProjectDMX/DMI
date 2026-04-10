"""HF transport correctness test: single run with compare model.

The compare model has BOTH HookPoints (ring::producer -> ClickHouse) and
.copy_() capture (-> pre-allocated buffers) in the same compiled graph.
After generate(), saves buffers to disk and compares vs ClickHouse for
bitwise equality.

Supports TP via torchrun:
    torchrun --nproc_per_node=2 -m tests.hf_compare_runner

Usage:
    E2E_MODEL=gpt2 E2E_CUDA_GRAPHS=0 python -m tests.hf_compare_runner
    E2E_MODEL=qwen3 E2E_TP_SIZE=2 torchrun --nproc_per_node=2 -m tests.hf_compare_runner
"""
import os
import sys
import tempfile
import uuid

import torch

_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}


def main():
    from monitoring import MonitoringEngine, MonitoringConfig, HostEngineConfig
    from monitoring._native_engine import ClickHouseClientConfig, StageConfig, RingConfig
    from monitoring.config import CaptureSchedule, HookSelection
    from monitoring.generate import generate_with_monitoring
    from transformers import AutoTokenizer

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    hf_model_id = _MODEL_ALIASES.get(model_key, model_key)
    batch_size = int(os.environ.get("E2E_BATCH_SIZE", "4"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "8"))
    cuda_graphs = os.environ.get("E2E_CUDA_GRAPHS", "0") == "1"
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))

    # Init distributed for TP
    tp_rank = 0
    if tp_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        tp_rank = dist.get_rank()
        torch.cuda.set_device(tp_rank)

    device = torch.device("cuda")

    # Load compare model
    if "qwen3" in hf_model_id.lower() or "qwen" in hf_model_id.lower():
        from transformers.models.qwen3_compare.modeling_qwen3 import CompareQwen3ForCausalLM
        model_cls = CompareQwen3ForCausalLM
    else:
        from transformers.models.gpt2_compare.modeling_gpt2 import CompareGPT2LMHeadModel
        model_cls = CompareGPT2LMHeadModel

    load_kwargs = dict(attn_implementation="eager", torch_dtype=torch.float16)
    if tp_size > 1:
        load_kwargs["tp_plan"] = "auto"

    model = model_cls.from_pretrained(hf_model_id, **load_kwargs).to(device).eval()
    if tp_size > 1:
        import torch.distributed as dist
        dist.barrier()

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

    max_total_len = int(input_ids.shape[1]) + max_new_tokens + 16
    model.allocate_compare_buffers(batch_size, max_total_len, dtype=torch.float16, tp_size=tp_size)

    # ClickHouse setup — drop per-rank tables
    import clickhouse_driver
    ch_client = clickhouse_driver.Client(db_host, port=db_port)
    table_name = f"offload_rank{tp_rank}" if tp_size > 1 else "offload"
    try:
        ch_client.execute(f"DROP TABLE IF EXISTS default.{table_name}")
    except Exception:
        pass
    if tp_size > 1:
        import torch.distributed as dist
        dist.barrier()

    ch_cfg = ClickHouseClientConfig()
    ch_cfg.host = db_host
    ch_cfg.port = db_port
    ch_cfg.username = os.environ.get("DMX_DB_USER", "default")
    ch_cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    ch_cfg.database = "default"
    ch_cfg.table = f"offload_rank{tp_rank}" if tp_size > 1 else "offload"
    ch_cfg.secure = False
    ch_cfg.client_side_compress = "none"
    ch_cfg.client_settings = None
    ch_cfg.create_database_if_missing = True
    ch_cfg.drop_existing_database = True
    ch_cfg.index_granularity = 8192

    stage = StageConfig.clickhouse_insert(ch_cfg, parallelism=10, name="ch_insert")
    q = stage.input_queue
    q.max_batch_items = 1024
    q.high_watermark_items = 1024
    q.max_batch_size = 2048 * 1024 * 1024
    q.high_watermark_size = 2048 * 1024 * 1024
    host_cfg = HostEngineConfig(stages=[stage])

    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "512"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "512"))
    ring_cfg = RingConfig()
    ring_cfg.task_ring_entries = 16384
    ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
    ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
    ring_cfg.drain_poll_timeout_us = 100
    ring_cfg.clone_slices = False

    mon_cfg = MonitoringConfig(
        hooks=HookSelection(mode="full"),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )
    if hasattr(mon_cfg, "eos_token_id"):
        mon_cfg.eos_token_id = eos_id
    if hasattr(mon_cfg, "pad_token_id"):
        mon_cfg.pad_token_id = pad_id

    unique_run_model_id = f"hf_compare::{uuid.uuid4().hex}"[:120]
    engine = MonitoringEngine(
        config=mon_cfg, model_id=unique_run_model_id, db_config=host_cfg
    )
    engine.enable_ring_transport(ring_cfg)
    model.monitoring_engine = engine
    if tp_size > 1:
        import torch.distributed as dist
        dist.barrier()

    mode = "cudagraph" if cuda_graphs else "eager"
    if tp_rank == 0:
        print(f"[compare] model={model_key} mode={mode} tp={tp_size} "
              f"batch={batch_size} tokens={max_new_tokens}", flush=True)

    # Shared compare dir across ranks — rank 0 creates, broadcasts to all
    if tp_rank == 0:
        compare_dir = tempfile.mkdtemp(prefix="hf_compare_ref_")
        print(f"[compare] ref_dir={compare_dir}", flush=True)
    else:
        compare_dir = ""
    if tp_size > 1:
        import torch.distributed as dist
        obj_list = [compare_dir]
        dist.broadcast_object_list(obj_list, src=0)
        compare_dir = obj_list[0]

    _step_saver = _StepSaver(model, engine, compare_dir, tp_rank, tp_size)
    _save_handle = model.register_forward_hook(
        lambda mod, inp, out: _step_saver.save_step()
    )

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
            outputs = generate_with_monitoring(model, **gen_kwargs)
    finally:
        _save_handle.remove()
        engine.close()

    generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
    if tp_rank == 0:
        for i in range(batch_size):
            n_gen = (generated_ids[i] != pad_id).sum().item() - (input_ids[i] != pad_id).sum().item()
            print(f"  prompt[{i}]: {n_gen} tokens generated")

    # Barrier so all ranks finish writing ref files before comparison
    if tp_size > 1:
        import torch.distributed as dist
        dist.barrier()

    # Only rank 0 compares — merge all per-rank CH tables
    if tp_rank == 0:
        print("\n[compare] Comparing disk vs ClickHouse...", flush=True)
        from tests.compare_disk_vs_ch import read_clickhouse, compare
        ch_data_all = {}
        total_rows = 0
        if tp_size > 1:
            for r in range(tp_size):
                data, n = read_clickhouse(db_host, db_port,
                                          table=f"offload_rank{r}")
                ch_data_all.update(data)
                total_rows += n
        else:
            ch_data_all, total_rows = read_clickhouse(db_host, db_port)
        passed, failed = compare(compare_dir, ch_data_all, total_rows)

        if failed > 0:
            sys.exit(1)
        else:
            print("[compare] ALL PASSED", flush=True)


_TP_SHARDED_HOOKS = {"q", "k", "v", "z", "mlp_post"}


class _StepSaver:
    """Save .copy_() buffers to disk after each forward step.

    Each step overwrites buf[:B, :seq_len]. We read the engine's per-request
    token ranges (set by _prepare_ring_step before each forward) to know
    the correct (start, end) for each request, then save the buffer slice.
    """

    def __init__(self, model, engine, compare_dir: str,
                 tp_rank: int = 0, tp_size: int = 1):
        self.model = model
        self.engine = engine
        self.compare_dir = compare_dir
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.step = 0

    def save_step(self):
        req_ids = self.engine._active_batch_request_ids
        starts = self.engine._active_batch_start_idx_per_request
        if req_ids is None or starts is None:
            return

        inner_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        cache_pos = getattr(inner_model, "_cache_pos", None)
        if cache_pos is None:
            return

        bufs = self.model.get_ref_buffers()
        if not bufs:
            return

        seq_len = int(cache_pos.shape[0])
        is_prefill = int(cache_pos[0]) == 0
        batch_size = len(req_ids)

        for i in range(batch_size):
            req_id = req_ids[i]
            # starts[i] was already advanced by _prepare_ring_step
            t_end = int(starts[i])
            if is_prefill:
                t_start = 0
                real_tokens = t_end  # prompt_len for this request
                pad_len = seq_len - real_tokens  # left-padding
            else:
                t_start = t_end - 1
                real_tokens = 1
                pad_len = 0

            if real_tokens <= 0:
                continue

            req_dir = os.path.join(self.compare_dir, req_id)
            os.makedirs(req_dir, exist_ok=True)

            sr = f"_SR{self.tp_rank}" if self.tp_size > 1 else ""

            for name, buf in bufs.items():
                if "_L" in name:
                    hook_name = name.rsplit("_L", 1)[0]
                    layer = int(name.rsplit("_L", 1)[1])
                else:
                    hook_name = name
                    layer = -1

                if hook_name == "final_logits":
                    continue

                # Non-zero TP ranks only save sharded hooks
                if self.tp_rank != 0 and hook_name not in _TP_SHARDED_HOOKS:
                    continue

                # Strip left-padding: real data is at buf[i, pad_len : pad_len + real_tokens]
                chunk = buf[i, pad_len:pad_len + real_tokens].cpu().clone()

                if layer >= 0:
                    fname = f"{hook_name}_L{layer}_T{t_start}_{t_end}{sr}.pt"
                else:
                    fname = f"{hook_name}_T{t_start}_{t_end}{sr}.pt"

                torch.save(chunk, os.path.join(req_dir, fname))

        self.step += 1


if __name__ == "__main__":
    main()
