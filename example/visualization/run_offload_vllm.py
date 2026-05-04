"""Demo offload (vLLM): run Qwen3-0.6B with monitoring on the IOI prompt.

Writes activations into ClickHouse under the fixed model_id
``demo_vllm``.  Wipes only that slot before insert, so re-runs are
idempotent and the HF run (``demo_hf``) is never touched.

Note that vLLM does NOT capture ``attn.hook_pattern`` -- vLLM's
attention backends fuse attention weights into the kernel and don't
expose them as standalone tensors.  The notebook detects this and
skips the attention-pattern plot when reading from this slot; the
other 4 plots (residual norm, per-token confidence, top neuron
activations, top-k alternatives) render normally.

Pre-requisites: project installed per main README, ClickHouse running
on ``DMX_DB_HOST`` (default ``localhost:9000``).

Usage:
    python example/visualization/run_offload_vllm.py
"""
from __future__ import annotations

import os
import sys

# Required workaround for the void+effectful ring::producer op + vLLM's
# AOT compile-cache serialization (see todo.md item 10).
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

from pathlib import Path

# Bootstrap: add the repo root to sys.path so `monitoring` /
# `integration` resolve when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_ID = "demo_vllm"
HF_MODEL = "Qwen/Qwen3-0.6B"

# ``pattern`` is dropped (vLLM exclusion) -- everything else matches
# the HF demo's hook selection so the notebook can render the same
# plots.
HOOK_SELECTION = "token_ids,resid_pre,final_logits"


def _read_prompt() -> str:
    here = Path(__file__).resolve().parent
    return (here / "prompt.txt").read_text().strip()


def _wipe_my_rows(db_host: str, db_port: int, database: str, table: str) -> None:
    """ALTER TABLE ... DELETE only this script's model_id rows.  Safe on
    first run when the table doesn't exist yet."""
    import clickhouse_driver

    client = clickhouse_driver.Client(host=db_host, port=db_port)
    try:
        client.execute(
            f"ALTER TABLE {database}.{table} DELETE WHERE model_id = %(m)s",
            {"m": MODEL_ID},
        )
    except Exception as exc:
        msg = str(exc)
        if "doesn't exist" in msg.lower() or "unknown table" in msg.lower():
            return
        raise


def main() -> None:
    # Don't import torch here -- vLLM forks the engine subprocess and any
    # torch.cuda probe in the parent (e.g. ``torch.cuda.is_available()``)
    # would lazy-init CUDA, after which the forked subprocess can't
    # re-init.  The LLM itself owns all CUDA setup.
    from vllm import LLM, SamplingParams

    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_database = os.environ.get("DMX_DB_DATABASE", "default")
    db_table = os.environ.get("DMX_DB_TABLE", "offload")

    _wipe_my_rows(db_host, db_port, db_database, db_table)

    prompt = _read_prompt()
    print(f"[demo] Prompt: {prompt!r}", flush=True)

    llm = LLM(
        model=HF_MODEL,
        worker_cls="integration.vllm_adapter.DMXGPUWorker",
        additional_config={
            "dmx_model_id": MODEL_ID,
            "dmx_hook_selection": HOOK_SELECTION,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
            "dmx_db_database": db_database,
            "dmx_db_table": db_table,
        },
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )

    out = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    decoded = out[0].outputs[0].text

    # vLLM's worker shutdown should call stop_monitoring (DMXGPUWorker
    # registers it).  Force a final flush by deleting the LLM.
    del llm
    import torch
    torch.cuda.empty_cache()

    print(f"[demo] Output:  {decoded!r}", flush=True)
    print(f"[demo] model_id = {MODEL_ID}  (open visualize.ipynb to render)", flush=True)


if __name__ == "__main__":
    main()
