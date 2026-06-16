"""One-file HF example: generate, read DMI internals, print layer norms.

Prerequisites:
  - ClickHouse is running on DMX_DB_HOST:DMX_DB_PORT, default localhost:9000.
  - The project is installed with the patched transformers/HF hooks.
  - CUDA is available.

Usage:
    python example/hf_internal_norms/run_hf_internal_norms.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

MODEL_ID = "example_hf_internal_norms"
HF_MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8


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
    cfg.drop_existing_database = False
    cfg.index_granularity = 8192
    return cfg


def _wipe_my_rows(db_cfg) -> None:
    import clickhouse_driver

    client = clickhouse_driver.Client(
        host=db_cfg.host,
        port=db_cfg.port,
        user=db_cfg.username,
        password=db_cfg.password,
    )
    try:
        client.execute(
            f"ALTER TABLE {db_cfg.database}.{db_cfg.table} "
            f"DELETE WHERE model_id = %(model_id)s",
            {"model_id": MODEL_ID},
        )
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in ("doesn't exist", "unknown table", "could not find table")):
            return
        print(f"[example] WARNING: row cleanup failed: {exc}", file=sys.stderr)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    from integration.hf_adapter import generate_with_monitoring_dict
    from monitoring import HostEngineConfig, MonitoringConfig, MonitoringEngine
    from monitoring._native_engine import StageConfig
    from monitoring.config import CaptureSchedule
    from monitoring.internal_mapper import InternalRequirements
    from transformers import AutoTokenizer
    from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM

    db_cfg = _build_db_config()
    _wipe_my_rows(db_cfg)

    device = torch.device("cuda")
    print(f"[example] Loading {HF_MODEL} on CUDA in fp16 ...", flush=True)
    model = HookedQwen3ForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stage = StageConfig.clickhouse_insert(db_cfg, parallelism=4, name="ch_insert")
    engine = MonitoringEngine(
        config=MonitoringConfig(schedule=CaptureSchedule()),
        model_id=MODEL_ID,
        db_config=HostEngineConfig(stages=[stage]),
    )
    model.monitoring_engine = engine

    expected_layers = model.config.num_hidden_layers
    requirements = InternalRequirements().require(
        "hidden_states",
        count=expected_layers,
        retry=True,
        timeout_s=30.0,
        poll_s=0.25,
    )
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            out = generate_with_monitoring_dict(
                model,
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                hook_selection="hidden-states",
                internal_requirements=requirements,
            )
    finally:
        engine.close()

    decoded = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    print(f"[example] Output: {decoded!r}")

    hidden_states = out.dmi_internal.hidden_states
    token_mask = out.dmi_internal.token_mask
    assert len(hidden_states) == expected_layers
    assert all(t.device.type == "cpu" for t in hidden_states)
    assert token_mask.device.type == "cpu"

    print("[example] Per-layer hidden-state activation norm:")
    for layer, tensor in enumerate(hidden_states):
        norm = tensor.float().norm(dim=-1)[token_mask].mean().item()
        print(f"  layer {layer:02d}: {norm:.6f}")


if __name__ == "__main__":
    main()
