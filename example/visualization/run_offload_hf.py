"""Demo offload (HF): run Qwen3-0.6B with monitoring on the IOI prompt.

Writes activations into ClickHouse under the fixed model_id
``demo_hf``.  Wipes only that slot before insert, so re-runs are
idempotent and the vLLM run (``demo_vllm``) is never touched.

The visualization notebook (``visualize.ipynb``) reads from the
same fixed slot -- no model_id paste step.

Pre-requisites: project installed per main README, ClickHouse running
on ``DMX_DB_HOST`` (default ``localhost:9000``).

Usage:
    python example/visualization/run_offload_hf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap: add the repo root to sys.path so `monitoring` /
# `integration` / `transformers` (the patched fork) resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

MODEL_ID = "demo_hf"
HF_MODEL = "Qwen/Qwen3-0.6B"

# Hooks that the notebook needs.  ``token_ids`` is required for the
# notebook to recover token labels; ``pattern`` powers plot 1
# (attention heatmaps), ``resid_pre`` powers plot 2 (residual norm),
# ``final_logits`` powers plots 3 + 5 (per-token confidence + top-k
# alternatives), ``mlp_post`` powers plot 4 (top neuron activations).
HOOK_SELECTION = "token_ids,pattern,resid_pre,final_logits,mlp_post"


def _read_prompt() -> str:
    here = Path(__file__).resolve().parent
    return (here / "prompt.txt").read_text().strip()


def _build_db_config():
    from monitoring._native_engine import ClickHouseClientConfig

    cfg = ClickHouseClientConfig()
    cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    cfg.username = os.environ.get("DMX_DB_USER", "default")
    cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    cfg.database = os.environ.get("DMX_DB_DATABASE", "default")
    cfg.table = os.environ.get("DMX_DB_TABLE", "offload")
    cfg.secure = False
    cfg.client_side_compress = "none"
    cfg.create_database_if_missing = True
    cfg.drop_existing_database = False  # keep the DB; we wipe just our model_id
    cfg.index_granularity = 8192
    return cfg


def _wipe_my_rows(db_cfg) -> None:
    """ALTER TABLE ... DELETE only this script's model_id rows.

    Idempotent: re-running the script never accumulates stale rows
    under the same model_id.  Other users of the table (other
    model_ids, including ``demo_vllm``) are untouched.

    Safe even when the table doesn't exist yet (first run).
    """
    import clickhouse_driver

    client = clickhouse_driver.Client(
        host=db_cfg.host, port=db_cfg.port,
        user=db_cfg.username, password=db_cfg.password,
    )
    try:
        client.execute(
            f"ALTER TABLE {db_cfg.database}.{db_cfg.table} "
            f"DELETE WHERE model_id = %(m)s",
            {"m": MODEL_ID},
        )
    except Exception as exc:
        # First run: table doesn't exist; the engine will create it.
        msg = str(exc)
        if "doesn't exist" in msg.lower() or "unknown table" in msg.lower():
            return
        raise


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA.")

    from monitoring import HostEngineConfig, MonitoringConfig, MonitoringEngine
    from monitoring._native_engine import StageConfig
    from monitoring.config import CaptureSchedule
    from transformers import AutoTokenizer
    from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM

    from integration.hf_adapter import generate_with_monitoring

    db_cfg = _build_db_config()
    _wipe_my_rows(db_cfg)

    device = torch.device("cuda")

    # Load the hooked Qwen3 variant.  ``attn_implementation="eager"``
    # is required so the attention-weight tensor is materialized for
    # the ``hook_pattern`` HookPoint to capture.
    print(f"[demo] Loading {HF_MODEL} on CUDA in fp16 ...", flush=True)
    model = HookedQwen3ForCausalLM.from_pretrained(
        HF_MODEL, torch_dtype=torch.float16, attn_implementation="eager",
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Engine + ring transport.  ``MonitoringEngine.__init__`` auto-enables
    # the ring transport with default sizes; no manual ``RingConfig`` step.
    stage = StageConfig.clickhouse_insert(db_cfg, parallelism=4, name="ch_insert")
    host_cfg = HostEngineConfig(stages=[stage])
    engine = MonitoringEngine(
        config=MonitoringConfig(schedule=CaptureSchedule()),
        model_id=MODEL_ID,
        db_config=host_cfg,
    )
    model.monitoring_engine = engine

    prompt = _read_prompt()
    print(f"[demo] Prompt: {prompt!r}", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = generate_with_monitoring(
                model, **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                hook_selection=HOOK_SELECTION,
            )
    finally:
        engine.close()

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"[demo] Output:  {decoded!r}", flush=True)
    print(f"[demo] model_id = {MODEL_ID}  (open visualize.ipynb to render)", flush=True)


if __name__ == "__main__":
    main()
