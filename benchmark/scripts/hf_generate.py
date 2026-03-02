import argparse
import contextlib
import json
import math
import os
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

_MODEL_ALIASES = {
    # Convenience alias used in benchmark CLI.
    "qwen3": "Qwen/Qwen3-4B",
}


def _resolve_model_id(model: str) -> str:
    return _MODEL_ALIASES.get(model.lower(), model)


def _load_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _iter_batches(items: List[str], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield idx // batch_size, items[idx : idx + batch_size]


@contextlib.contextmanager
def _nvtx_range(name: str):
    try:
        if not torch.cuda.is_available():
            yield
            return
        from torch.cuda import nvtx  # type: ignore
    except Exception:
        yield
        return
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HF generate benchmark")
    parser.add_argument("--prompts", default="benchmark/data/prompts.txt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    model_id = _resolve_model_id(args.model)

    prompts = _load_prompts(args.prompts)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", torch_dtype=torch.float16)
    model.to(device).eval()

    per_batch = []
    total_tokens = 0
    start = time.perf_counter()

    total_batches = math.ceil(len(prompts) / args.batch_size)
    use_nvtx = os.environ.get("BENCH_NVTX", "0") == "1"
    nvtx_ctx = _nvtx_range("hf_generate") if use_nvtx else contextlib.nullcontext()
    with nvtx_ctx, torch.no_grad():
        for batch_idx, batch_prompts in tqdm(
            _iter_batches(prompts, args.batch_size),
            total=total_batches,
            desc="hf_generate",
        ):
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_seconds = t1 - t0
            batch_tokens = int(input_ids.shape[0] * args.max_new_tokens)
            total_tokens += batch_tokens
            per_batch.append(
                {
                    "batch_idx": batch_idx,
                    "batch_size": int(input_ids.shape[0]),
                    "seconds": batch_seconds,
                    "tokens": batch_tokens,
                    "tokens_per_s": batch_tokens / batch_seconds if batch_seconds > 0 else None,
                }
            )

    total_seconds = time.perf_counter() - start
    result = {
        "backend": "hf",
        "model": model_id,
        "device": str(device),
        "prompts": len(prompts),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "total_seconds": total_seconds,
        "total_tokens": total_tokens,
        "tokens_per_s": total_tokens / total_seconds if total_seconds > 0 else None,
        "per_batch": per_batch,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
