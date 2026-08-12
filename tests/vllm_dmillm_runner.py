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
import uuid

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
    from transformers import AutoConfig
    from vllm import SamplingParams
    from integration.vllm_adapter import DMILLM

    host = os.environ.get("DMX_DB_HOST", "localhost")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    model_id = f"test_dmillm::{uuid.uuid4().hex}"[:120]
    client = clickhouse_driver.Client(host=host, port=port)
    client.execute(
        "ALTER TABLE default.offload DELETE WHERE model_id=%(m)s",
        {"m": model_id},
    )

    expected_layers = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    llm = None
    stopped = False
    try:
        llm = DMILLM(
            args.model,
            additional_config={
                "dmx_model_id": model_id,
                "dmx_hook_selection": "resid_pre",
                "dmx_db_host": host,
                "dmx_db_port": port,
                # This test intentionally verifies bounded asynchronous reads.
                # The production default is 0 (flush on pressure or stop).
                "dmx_drain_flush_timeout_us": 100_000,
            },
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.5,
        )
        prompts = ["The capital of France is", "Hello"]
        outputs = llm.generate(
            prompts, SamplingParams(temperature=0.0, max_tokens=8)
        )

        # Replace fixed sleeps with the public bounded-retry contract. This
        # rejects both missing fields and partial layer sets.
        for output in outputs:
            output.dmi_internal.require(
                "hidden_states",
                count=expected_layers,
                retry=True,
                timeout_s=15.0,
                poll_s=0.1,
                match_token_ranges=True,
            )
        before_stop = _shapes(outputs)
        for output in outputs:
            output.dmi_internal.clear_cache()

        # Explicit flush, then re-read as the authoritative baseline.
        llm.collective_rpc("stop_monitoring")
        stopped = True
        internals = [output.dmi_internal.hidden_states for output in outputs]
        after_stop = [(len(hs), hs[0].shape[1]) for hs in internals]

        tests = [
            {"name": "outputs_are_native",
             "passed": len(outputs) == 2 and all(o.outputs[0].text for o in outputs),
             "detail": [o.outputs[0].text for o in outputs]},
            {"name": "bounded_async_read_is_complete",
             "passed": before_stop == after_stop
                       and all(layers == expected_layers
                               for layers, _ in before_stop),
             "detail": {"before_stop": before_stop,
                        "after_stop": after_stop}},
            # Per-request: each is its own [1, seq, hidden] (batch dim 1).
            {"name": "per_request_hidden_states",
             "passed": all(len(hs) > 0 and hs[0].dim() == 3
                           and hs[0].shape[0] == 1 for hs in internals),
             "detail": after_stop},
            # Ragged prompts -> independent seq lengths (no cross-padding).
            {"name": "per_request_isolated_lengths",
             "passed": len({s for _, s in after_stop}) > 1,
             "detail": [s for _, s in after_stop]},
            {"name": "available_lists_hidden_states",
             "passed": "hidden_states" in outputs[0].dmi_internal.available,
             "detail": outputs[0].dmi_internal.available},
        ]

        with open(args.result_file, "w") as f:
            json.dump({"tests": tests}, f)
        success = all(test["passed"] for test in tests)
    finally:
        try:
            if llm is not None and not stopped:
                llm.collective_rpc("stop_monitoring")
        finally:
            client.execute(
                "ALTER TABLE default.offload DELETE WHERE model_id=%(m)s",
                {"m": model_id},
                settings={"mutations_sync": 1},
            )
            if llm is not None:
                del llm
            torch.cuda.empty_cache()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
