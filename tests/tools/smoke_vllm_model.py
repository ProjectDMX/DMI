"""Run a deterministic baseline or DMI vLLM smoke and save token IDs.

This tool needs one CUDA GPU and local model weights, but not ClickHouse. It is
intended for version/model port validation before the full transport suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "monitored"), required=True)
    parser.add_argument("--model", default="qwen2")
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON corpus with a top-level prompts list.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--ring-mb", type=int, default=64)
    parser.add_argument("--hook-selection", default="resid_pre")
    parser.add_argument("--cudagraph", action="store_true")
    return parser.parse_args()


def _load_prompts(cases_path: Path | None) -> list[str]:
    if cases_path is None:
        return ["The capital of France is", "2 + 2 ="]

    payload = json.loads(cases_path.read_text())
    prompts = payload.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) for prompt in prompts)
    ):
        raise ValueError(f"{cases_path} must contain a non-empty string prompts list")
    return prompts


def _optional_list(value):
    return None if value is None else list(value)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    from vllm import LLM, SamplingParams

    model_id = MODEL_ALIASES.get(args.model, args.model)
    kwargs = {
        "model": model_id,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "enforce_eager": not args.cudagraph,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.mode == "monitored":
        kwargs.update(
            worker_cls="integration.vllm_adapter.DMXGPUWorker",
            additional_config={
                "dmx_hook_selection": args.hook_selection,
                "dmx_ring_payload_mb": args.ring_mb,
                "dmx_ring_pinned_mb": args.ring_mb,
                "dmx_db_host": "",
            },
        )

    prompts = _load_prompts(args.cases)
    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        payload = {
            "schema_version": 1,
            "mode": args.mode,
            "model": model_id,
            "cudagraph": args.cudagraph,
            "tensor_parallel_size": args.tensor_parallel_size,
            "prompts": prompts,
            "prompt_token_ids": [
                _optional_list(output.prompt_token_ids) for output in outputs
            ],
            "token_ids": [list(output.outputs[0].token_ids) for output in outputs],
            "texts": [output.outputs[0].text for output in outputs],
            "finish_reasons": [
                output.outputs[0].finish_reason for output in outputs
            ],
            "stop_reasons": [output.outputs[0].stop_reason for output in outputs],
        }
    finally:
        try:
            if args.mode == "monitored":
                llm.collective_rpc("stop_monitoring")
        finally:
            del llm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
