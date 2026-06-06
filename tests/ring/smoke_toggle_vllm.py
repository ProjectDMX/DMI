"""Real FULL-decode vLLM smoke gate for node-toggle (offline LLM, in-process worker).

Loads Qwen3-0.6B with the DMXGPUWorker, cudagraph_mode=FULL, node-toggle ON with
a PARTIAL enabled set (last 4 of 28 hidden-state hooks), full pipeline -> ClickHouse.
Exercises the real lifecycle: capture window -> producer-node recording -> bind
captured graphs -> set_active_hooks (eager) -> per-replay graph guard -> meta gate
+ reserve in lockstep -> drain/p2p -> DB.

This is a SELF-CHECKING GATE: it DROPs its table up front (so any rows seen are
provably from THIS run -- no historical pollution), then asserts the export
landed rows for EXACTLY the enabled layers with the right shape, and exits
NONZERO on any failure. SMOKE_OK + exit 0 only on full success.

Note: vLLM may auto-downgrade cudagraph_mode FULL -> FULL_AND_PIECEWISE when the
attention backend lacks full-graph support; this validates the FULL-DECODE path
(the graphs node-toggle binds), not full FULL_AND_PIECEWISE coverage.

Requires: fork vllm on PYTHONPATH, the monitoring .so, a cached model
(SMOKE_MODEL), ClickHouse on localhost:9000.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER HF_HUB_OFFLINE=1 \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 SMOKE_MODEL=<path> \
      PYTHONPATH=$PWD/integration/vllm:$PWD <venv>/bin/python tests/ring/smoke_toggle_vllm.py
"""
import gc
import os
import subprocess
import sys
import time

MODEL = os.environ["SMOKE_MODEL"]
DB_TABLE = "smoke_toggle"
ENABLED = "0:24,0:25,0:26,0:27"          # last 4 layers of Qwen3-0.6B (28 layers)
ENABLED_LAYERS = {24, 25, 26, 27}
HIDDEN = 1024                              # Qwen3-0.6B hidden dim; rows are [q_len, HIDDEN]
                                           # (q_len=1 for decode, >1 for prefill chunks)


def ch(query: str) -> str:
    """Run a clickhouse-client query; FAIL the smoke if the CLI errors."""
    r = subprocess.run(["clickhouse-client", "--query", query],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"clickhouse-client failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout.strip()


def fail(msg: str):
    print(f"SMOKE_FAIL: {msg}", flush=True)
    sys.exit(1)


# --- fresh table: any rows seen afterwards are provably from THIS run (#2) ---
ch(f"DROP TABLE IF EXISTS {DB_TABLE}")

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    worker_cls="integration.vllm_adapter.DMXGPUWorker",
    enforce_eager=False,                  # we WANT cuda graphs
    max_model_len=2048,
    gpu_memory_utilization=0.40,
    compilation_config={"cudagraph_mode": "FULL"},
    additional_config={
        "dmx_hook_selection": "hidden-states",
        "dmx_node_toggle": True,
        "dmx_enabled_hooks": ENABLED,
        "dmx_ring_payload_mb": 1024,
        "dmx_ring_pinned_mb": 1024,
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000,
        "dmx_db_table": DB_TABLE,
        "dmx_drain_flush_timeout_us": 2000,   # flush sparse data within ~2ms
        "dmx_ch_max_batch_items": 16,         # low insert watermark for the smoke
    },
)

out = llm.generate(
    ["The capital of France is", "Large language models are",
     "In the beginning", "A short story about a robot:"],
    SamplingParams(temperature=0.0, max_tokens=48),
)
texts = [o.outputs[0].text for o in out]
for t in texts:
    print(f"SMOKE_GEN: {t!r}", flush=True)
if not all(t.strip() for t in texts):
    fail("a prompt produced empty output (generation/desync problem)")

# Clean shutdown so the async export pipeline flushes to ClickHouse.
del llm
gc.collect()
time.sleep(3.0)

# --- strict verification: rows for EXACTLY the enabled layers, right shape ---
raw = ch(f"SELECT layer_no, count() FROM {DB_TABLE} GROUP BY layer_no ORDER BY layer_no FORMAT TSV")
print("SMOKE_DB_RAW:\n" + (raw or "<empty>"), flush=True)
rows = [ln.split("\t") for ln in raw.splitlines() if ln]
got_layers = sorted(int(c[0]) for c in rows)
total = sum(int(c[1]) for c in rows)

if total == 0:
    fail("ClickHouse received 0 rows (export/flush problem)")
if set(got_layers) != ENABLED_LAYERS:
    fail(f"delivered layers {got_layers} != enabled {sorted(ENABLED_LAYERS)} (lockstep desync)")

import re
shapes = ch(f"SELECT DISTINCT shape FROM {DB_TABLE} FORMAT TSV").splitlines()
# hidden-states rows are [q_len, HIDDEN]: q_len=1 (decode) or >1 (prefill chunk).
bad = [s for s in shapes if not re.fullmatch(rf"\[\d+,{HIDDEN}\]", s)]
if bad:
    fail(f"shapes with wrong hidden dim (expected [*,{HIDDEN}]): {bad}")

print(f"SMOKE_DB_OK: {total} rows, layers={got_layers}, shapes={shapes} "
      f"(partial-toggle lockstep on a real model)", flush=True)

# Leave the table clean for the next run.
ch(f"DROP TABLE IF EXISTS {DB_TABLE}")
print("SMOKE_OK", flush=True)
sys.exit(0)
