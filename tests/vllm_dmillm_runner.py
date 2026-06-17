"""Subprocess runner: DMILLM offline -> read each RequestOutput's per-request
.dmi_internal back from ClickHouse, write pass/fail to a result file.

Run as a subprocess (the pytest parent must not touch CUDA before forking the
vLLM engine). See tests/test_vllm_dmillm_e2e.py.

Usage:
    python -m tests.vllm_dmillm_runner --result-file /tmp/r.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")


def _shapes(outputs):
    return [(len(o.dmi_internal.hidden_states), o.dmi_internal.hidden_states[0].shape[1])
            for o in outputs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--result-file", required=True)
    args = ap.parse_args()

    import torch
    import clickhouse_driver
    from vllm import SamplingParams
    from integration.vllm_adapter import DMILLM

    host = os.environ.get("DMX_DB_HOST", "localhost")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    model_id = f"test_dmillm::{os.getpid()}"
    client = clickhouse_driver.Client(host=host, port=port)
    client.execute("ALTER TABLE default.offload DELETE WHERE model_id=%(m)s", {"m": model_id})

    llm = DMILLM(
        args.model,
        additional_config={
            "dmx_model_id": model_id, "dmx_hook_selection": "resid_pre",
            "dmx_db_host": host, "dmx_db_port": port,
        },
        max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.5,
    )
    prompts = ["The capital of France is", "Hello"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))

    # No-flush read: rely on the 100ms drain-flush timeout (default since the
    # drain-flush change) to land captures in ClickHouse without stop_monitoring.
    time.sleep(2.0)
    no_stop = _shapes(outputs)
    for o in outputs:
        o.dmi_internal.clear_cache()

    # Explicit flush, then re-read as the authoritative baseline.
    llm.collective_rpc("stop_monitoring")
    time.sleep(0.5)
    internals = [o.dmi_internal.hidden_states for o in outputs]
    after_stop = [(len(hs), hs[0].shape[1]) for hs in internals]

    tests = [
        {"name": "outputs_are_native",
         "passed": len(outputs) == 2 and all(o.outputs[0].text for o in outputs),
         "detail": [o.outputs[0].text for o in outputs]},
        # Auto-drain: internals reach ClickHouse without an explicit stop_monitoring.
        {"name": "readable_without_stop_monitoring",
         "passed": no_stop == after_stop and all(layers > 0 for layers, _ in no_stop),
         "detail": {"no_stop": no_stop, "after_stop": after_stop}},
        # Per-request: each is its own [1, seq, hidden] (batch dim 1).
        {"name": "per_request_hidden_states",
         "passed": all(len(hs) > 0 and hs[0].dim() == 3 and hs[0].shape[0] == 1
                       for hs in internals),
         "detail": after_stop},
        # Ragged prompts -> independent seq lengths (no cross-padding).
        {"name": "per_request_isolated_lengths",
         "passed": len({s for _, s in after_stop}) > 1,
         "detail": [s for _, s in after_stop]},
        {"name": "available_lists_hidden_states",
         "passed": "hidden_states" in outputs[0].dmi_internal.available,
         "detail": outputs[0].dmi_internal.available},
    ]

    client.execute("ALTER TABLE default.offload DELETE WHERE model_id=%(m)s", {"m": model_id})
    with open(args.result_file, "w") as f:
        json.dump({"tests": tests}, f)

    del llm
    torch.cuda.empty_cache()
    sys.exit(0 if all(t["passed"] for t in tests) else 1)


if __name__ == "__main__":
    main()
