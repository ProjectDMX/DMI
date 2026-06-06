"""Real FULL vLLM server smoke for node-toggle (offline LLM, in-process worker).

Loads Qwen3-0.6B with the DMXGPUWorker, cudagraph_mode=FULL, node-toggle ON with
a PARTIAL enabled set (last 4 of 28 hidden-state hooks), full pipeline -> ClickHouse.
Exercises the real lifecycle: capture window -> producer-node recording -> bind
captured graphs -> set_active_hooks (eager) -> per-replay graph guard -> meta gate
+ reserve in lockstep -> drain/p2p -> DB. A desync or guard failure crashes here.

Asserts (via markers grepped by the runner):
  - bound >=1 captured graphs, nodes registered,
  - active hooks set to the partial set,
  - replay guard active (mode=eager) -- the #3 guard runs in vivo,
  - generation produces text (no desync/crash),
  - ClickHouse received rows ONLY for the enabled layers (partial-toggle lockstep).
"""
import os
import sys

MODEL = os.environ["SMOKE_MODEL"]
DB_TABLE = "smoke_toggle"
ENABLED = "0:24,0:25,0:26,0:27"          # last 4 layers of Qwen3-0.6B (28 layers)
ENABLED_LAYERS = {24, 25, 26, 27}

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
for o in out:
    print(f"SMOKE_GEN: {o.outputs[0].text!r}", flush=True)

# Clean shutdown so the async export pipeline flushes to ClickHouse (low-volume
# runs won't hit the insert batch watermark, so a flush-on-close is required).
import gc, time
del llm
gc.collect()
time.sleep(3.0)

# Verify rows landed for ONLY the enabled layers (partial-toggle lockstep in vivo)
# via the clickhouse-client CLI (clickhouse_driver isn't in the venv).
import subprocess
q = (f"SELECT layer_no, count() FROM {DB_TABLE} GROUP BY layer_no ORDER BY layer_no "
     f"FORMAT TSV")
res = subprocess.run(["clickhouse-client", "--query", q], capture_output=True, text=True)
print("SMOKE_DB_RAW:\n" + (res.stdout.strip() or "<empty>"), flush=True)
got = sorted(int(l.split("\t")[0]) for l in res.stdout.strip().splitlines() if l)
if set(got) == ENABLED_LAYERS:
    print("SMOKE_DB_OK: rows for EXACTLY the enabled layers (partial-toggle lockstep)", flush=True)
elif got:
    print(f"SMOKE_DB_MISMATCH: got {got} expected {sorted(ENABLED_LAYERS)}", flush=True)
else:
    print("SMOKE_DB_EMPTY", flush=True)
print("SMOKE_OK", flush=True)
