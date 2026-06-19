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
        # NOTE: deliberately NOT setting dmx_drain_flush_timeout_us -- this
        # exercises the new toggle-on default (50ms) so the default export
        # behaviour is covered (not just an explicit override).
        "dmx_ch_max_batch_items": 16,         # low insert watermark for the smoke
    },
)

# Capture the C++ p2p thread's stderr during generate + shutdown flush.
# A "shape/bytes mismatch" warning is the visible symptom of a meta<->payload
# desync (e.g. an eager/prefill step over-firing producers vs the toggle
# subset). The prompts below MUST exercise prefill (every request prefills),
# so a regression there surfaces here. Redirect fd 2 (fprintf writes the C
# FILE* stderr, not sys.stderr), then replay it so visibility is preserved.
import tempfile
_saved_fd2 = os.dup(2)
_cap = tempfile.NamedTemporaryFile(mode="w+", suffix=".p2pstderr", delete=False)
os.dup2(_cap.fileno(), 2)
try:
    out = llm.generate(
        ["The capital of France is", "Large language models are",
         "In the beginning", "A short story about a robot:"],
        SamplingParams(temperature=0.0, max_tokens=48),
    )
    # Drain the export pipeline while stderr is still captured so any
    # late p2p warning is caught too.
    del llm
    gc.collect()
    time.sleep(3.0)
finally:
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)
_cap.flush(); _cap.seek(0)
_captured = _cap.read()
_cap.close()
sys.stderr.write(_captured)          # preserve worker stderr visibility
_n_mismatch = _captured.count("shape/bytes mismatch")
if _n_mismatch:
    fail(f"{_n_mismatch} p2p 'shape/bytes mismatch' warning(s) -- meta/payload "
         f"desync (prefill/eager over-fire vs toggle subset). The export data is "
         f"corrupt even where layer labels look right.")

texts = [o.outputs[0].text for o in out]
for t in texts:
    print(f"SMOKE_GEN: {t!r}", flush=True)
if not all(t.strip() for t in texts):
    fail("a prompt produced empty output (generation/desync problem)")

# --- verification: decode is gated to the enabled set; prefill is ungated ---
# Toggle gates only DECODE-graph producers; prefill runs eager (full active
# set). So in the DB:
#   - decode rows (shape [1,HIDDEN], one token/step) must be EXACTLY the
#     enabled layers -- the decode gate, no leakage, no missing layer.
#   - prefill rows (shape [N>1,HIDDEN], the prompt) carry the full active
#     set -- enabled layers must be present among them.
import re
raw = ch(f"SELECT layer_no, shape, count() FROM {DB_TABLE} "
         f"GROUP BY layer_no, shape ORDER BY layer_no FORMAT TSV")
print("SMOKE_DB_RAW:\n" + (raw or "<empty>"), flush=True)
rows = [ln.split("\t") for ln in raw.splitlines() if ln]
total = sum(int(c[2]) for c in rows)
if total == 0:
    fail("ClickHouse received 0 rows (export/flush problem)")

bad = [c[1] for c in rows if not re.fullmatch(rf"\[\d+,{HIDDEN}\]", c[1])]
if bad:
    fail(f"shapes with wrong hidden dim (expected [*,{HIDDEN}]): {bad}")

def _dim0(shape):
    return int(shape[1:-1].split(",")[0])

decode_layers = sorted({int(c[0]) for c in rows if _dim0(c[1]) == 1})
prefill_layers = sorted({int(c[0]) for c in rows if _dim0(c[1]) > 1})
all_layers = sorted({int(c[0]) for c in rows})

# Decode gate: decode-shaped rows are EXACTLY the enabled layers.
if set(decode_layers) != ENABLED_LAYERS:
    fail(f"decode-shaped rows layers {decode_layers} != enabled "
         f"{sorted(ENABLED_LAYERS)} -- decode gate leaked or dropped a layer")
# Prefill is ungated: the enabled layers must show up in prefill too, and
# prefill legitimately carries more layers (full active set, not a desync).
if not ENABLED_LAYERS <= set(prefill_layers):
    fail(f"enabled layers missing from prefill-shaped rows {prefill_layers} "
         f"(prefill should carry the full active set ungated)")

print(f"SMOKE_DB_OK: {total} rows; decode_layers={decode_layers} (==enabled, gated), "
      f"prefill_layers={all_layers and prefill_layers} (full active set, ungated) "
      f"-- decode-gated + prefill-passthrough lockstep on a real model", flush=True)

# Leave the table clean for the next run.
ch(f"DROP TABLE IF EXISTS {DB_TABLE}")
print("SMOKE_OK", flush=True)
sys.exit(0)
