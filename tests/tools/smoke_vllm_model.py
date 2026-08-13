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
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "ernie45": "baidu/ERNIE-4.5-0.3B-PT",
    "falcon_h1": "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "gemma3": "shibatch/tinygemma3-2m",
    "gpt2": "gpt2",
    "granite": "ibm-granite/granite-4.1-3b",
    "jamba": "ai21labs/AI21-Jamba2-3B",
    "lfm2": "tiny-random/lfm2",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "openaccess-ai-collective/tiny-mistral",
    "minicpm4": "openbmb/MiniCPM4.1-8B",
    "olmo3": "allenai/Olmo-3-7B-Instruct",
    "phi3": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "monitored"), required=True)
    parser.add_argument("--model", default="qwen2")
    parser.add_argument("--model-subfolder")
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
    parser.add_argument(
        "--decision-logprobs",
        type=int,
        default=0,
        help="Public top-logprob count retained to justify greedy branch ambiguity.",
    )
    parser.add_argument("--cudagraph", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow a checkpoint's pinned remote configuration code.",
    )
    parser.add_argument(
        "--revision",
        help="Exact model/tokenizer revision used for reproducible qualification.",
    )
    return parser.parse_args()


def _load_corpus(cases_path: Path | None) -> dict:
    if cases_path is None:
        return {
            "schema_version": 2,
            "name": "default-smoke",
            "generator": {"name": "manual", "version": 1},
            "executions": ["batch", "reversed"],
            "cases": [
                {
                    "case_id": "default-capital",
                    "checklist_ids": ["P02", "P03", "P04", "P05"],
                    "input": {"form": "text", "text": "The capital of France is"},
                    "sampling": {"temperature": 0.0, "max_tokens": 8},
                    "dimensions": {"prompt_form": "text"},
                    "oracles": ["differential", "reverse-batch-order"],
                    "kills": ["compare-only-decoded-text"],
                },
                {
                    "case_id": "default-math",
                    "checklist_ids": ["P02", "P03", "P04", "P05"],
                    "input": {"form": "text", "text": "2 + 2 ="},
                    "sampling": {"temperature": 0.0, "max_tokens": 8},
                    "dimensions": {"prompt_form": "text"},
                    "oracles": ["differential", "reverse-batch-order"],
                    "kills": ["compare-only-decoded-text"],
                },
            ],
            "omitted_combinations": [],
        }

    payload = json.loads(cases_path.read_text())
    from tests.blackbox.case_generation import validate_case_corpus

    try:
        validate_case_corpus(payload)
    except ValueError as error:
        raise ValueError(f"{cases_path}: {error}") from error
    return payload


def _optional_list(value):
    return None if value is None else list(value)


def _materialize_case(case: dict, tokenizer, default_max_tokens: int):
    input_spec = case.get("input")
    if not isinstance(input_spec, dict):
        raise ValueError(f"{case.get('case_id')} has no input object")
    form = input_spec.get("form")
    text = input_spec.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{case.get('case_id')} input.text must be a string")
    if form == "text":
        prompt = text
    elif form == "token_ids_from_text":
        prompt = list(tokenizer.encode(text))
    else:
        raise ValueError(f"{case.get('case_id')} has unsupported input form {form!r}")

    sampling = dict(case.get("sampling", {}))
    sampling.setdefault("temperature", 0.0)
    sampling.setdefault("max_tokens", default_max_tokens)
    allowed = {"temperature", "max_tokens", "top_p", "top_k", "min_tokens"}
    unexpected = sorted(set(sampling) - allowed)
    if unexpected:
        raise ValueError(
            f"{case.get('case_id')} has unsupported sampling fields {unexpected}"
        )
    return prompt, sampling


def _serialize_request(case_id: str, output) -> dict:
    return {
        "case_id": case_id,
        "request_id": output.request_id,
        "prompt": output.prompt,
        "prompt_token_ids": _optional_list(output.prompt_token_ids),
        "finished": output.finished,
        "outputs": [
            {
                "index": completion.index,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "cumulative_logprob": completion.cumulative_logprob,
                "decision_logprobs": _serialize_logprob_trace(
                    getattr(completion, "logprobs", None)
                ),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
            for completion in output.outputs
        ],
    }


def _serialize_logprob_trace(logprobs) -> list[list[dict]] | None:
    """Serialize public vLLM top-logprobs without retaining runtime objects."""

    if logprobs is None:
        return None
    return [
        [
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": None if value.rank is None else int(value.rank),
                "decoded_token": value.decoded_token,
            }
            for token_id, value in sorted(step.items())
        ]
        for step in logprobs
    ]


def main() -> None:
    args = _parse_args()
    if args.decision_logprobs < 0:
        raise ValueError("--decision-logprobs must be non-negative")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    from vllm import LLM, SamplingParams

    model_id = MODEL_ALIASES.get(args.model, args.model)
    from tests.model_artifacts import resolve_model_artifact

    model_id = resolve_model_artifact(model_id, args.model_subfolder)
    kwargs = {
        "model": model_id,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "enforce_eager": not args.cudagraph,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
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

    corpus = _load_corpus(args.cases)
    llm = LLM(**kwargs)
    try:
        tokenizer = llm.get_tokenizer()
        materialized = {
            case["case_id"]: _materialize_case(
                case, tokenizer, args.max_tokens
            )
            for case in corpus["cases"]
        }
        canonical_order = [case["case_id"] for case in corpus["cases"]]
        execution_orders = {
            "batch": canonical_order,
            "reversed": list(reversed(canonical_order)),
        }
        executions = []
        for execution_id in corpus["executions"]:
            case_order = execution_orders[execution_id]
            prompts = [materialized[case_id][0] for case_id in case_order]
            sampling_params = [
                SamplingParams(
                    **materialized[case_id][1],
                    **(
                        {"logprobs": args.decision_logprobs}
                        if args.decision_logprobs
                        else {}
                    ),
                )
                for case_id in case_order
            ]
            outputs = llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != len(case_order):
                raise RuntimeError(
                    f"public LLM.generate returned {len(outputs)} outputs for "
                    f"{len(case_order)} inputs"
                )
            executions.append(
                {
                    "execution_id": execution_id,
                    "case_order": case_order,
                    "results": [
                        _serialize_request(case_id, output)
                        for case_id, output in zip(case_order, outputs)
                    ],
                }
            )
        payload = {
            "schema_version": 2,
            "mode": args.mode,
            "model": model_id,
            "cudagraph": args.cudagraph,
            "tensor_parallel_size": args.tensor_parallel_size,
            "decision_logprobs": args.decision_logprobs,
            "corpus": corpus,
            "executions": executions,
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
