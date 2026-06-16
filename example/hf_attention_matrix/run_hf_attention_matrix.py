"""One-file HF example: capture DMI attentions and save an attention matrix.

Prerequisites:
  - ClickHouse is running on DMX_DB_HOST:DMX_DB_PORT, default localhost:9000.
  - The project is installed with the patched transformers/HF hooks.
  - CUDA is available.
  - matplotlib is installed.

Usage:
    python example/hf_attention_matrix/run_hf_attention_matrix.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

MODEL_ID = "example_hf_attention_matrix"
HF_MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8
LAYER = 0
HEAD = 0
OUTPUT_PNG = Path(__file__).resolve().parent / "attention_layer00_head00.png"


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


def _token_labels(tokenizer, sequence: torch.Tensor, n_tokens: int) -> list[str]:
    ids = sequence[:n_tokens].detach().cpu().tolist()
    labels = tokenizer.convert_ids_to_tokens(ids)
    return [label.replace("\u0120", " ").replace("\n", "\\n") for label in labels]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    import matplotlib.pyplot as plt

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
        attn_implementation="eager",
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
        "attentions",
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
                hook_selection="pattern",
                internal_requirements=requirements,
            )
    finally:
        engine.close()

    attentions = out.dmi_internal.attentions
    token_mask = out.dmi_internal.token_mask
    assert len(attentions) == expected_layers

    matrix = attentions[LAYER][0, HEAD].float()
    real_tokens = token_mask[0]
    matrix = matrix[real_tokens][:, real_tokens]
    labels = _token_labels(tokenizer, out.sequences[0], matrix.shape[0])

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.numpy(), cmap="viridis", vmin=0.0)
    ax.set_title(f"Layer {LAYER}, head {HEAD}")
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    if len(labels) <= 32:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=160)
    plt.close(fig)

    decoded = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    print(f"[example] Output: {decoded!r}")
    print(f"[example] Saved attention matrix to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
