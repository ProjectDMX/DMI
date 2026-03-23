#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATA_DIR = Path("benchmark/data/offline_e2e")

MODEL_ALIASES: Dict[str, str] = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


@dataclass
class PromptExample:
    dataset: str
    sample_id: int
    entry_id: str
    source_conversation_id: str
    approx_prompt_tokens: int
    approx_target_tokens: int
    target_text: str
    messages: List[Dict[str, str]]


@dataclass
class BatchMetrics:
    batch_index: int
    batch_size: int
    input_tokens: int
    padded_tokens: int
    target_generated_tokens: int
    actual_generated_tokens: int
    seconds: float


def resolve_model_id(model: str) -> str:
    return MODEL_ALIASES.get(model.lower(), model)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl_examples(path: str | Path, limit: int | None = None) -> List[PromptExample]:
    out: List[PromptExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(PromptExample(**raw))
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"no prompt examples found in {path}")
    return out


def apply_chat_template_or_fallback(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    chunks: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().upper()
        content = str(msg.get("content", "")).strip()
        if content:
            chunks.append(f"{role}: {content}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


def build_rendered_prompts(tokenizer: Any, examples: Sequence[PromptExample]) -> List[Dict[str, Any]]:
    rendered: List[Dict[str, Any]] = []
    for example in examples:
        target_tok = tokenizer(
            example.target_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        target_len = len(target_tok["input_ids"])
        messages = list(example.messages) + [
            {"role": "user", "content": f"Please respond with exactly {target_len} tokens."},
        ]
        prompt_text = apply_chat_template_or_fallback(tokenizer, messages)
        prompt_tok = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_tok["input_ids"])
        rendered.append(
            {
                "example": example,
                "prompt_text": prompt_text,
                "prompt_len": int(prompt_len),
                "target_len": int(target_len),
            }
        )
    return rendered


def maybe_sort_by_length(rendered: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:
    if not enabled:
        return rendered
    return sorted(
        rendered,
        key=lambda item: (item["prompt_len"], item["target_len"], item["example"].entry_id),
    )


def iter_batches(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def warmup_batches(items: Sequence[Dict[str, Any]], batch_size: int, count: int = 2) -> List[List[Dict[str, Any]]]:
    batches = list(iter_batches(items, batch_size))[:count]
    if not batches:
        raise ValueError("no batches available for warmup")
    while len(batches) < count:
        batches.append(list(batches[-1]))
    return batches


def build_tokenizer(model_id: str, *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def compile_generate_kwargs(enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {}
    kwargs: Dict[str, Any] = {"cache_implementation": "static"}
    try:
        from transformers import CompileConfig

        kwargs["compile_config"] = CompileConfig(mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass
    return kwargs


def parse_pad_buckets(spec: str) -> List[int]:
    values: List[int] = []
    for chunk in str(spec).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("pad bucket sizes must be > 0")
        values.append(value)
    return sorted(set(values))


def resolve_padded_length(
    current_length: int,
    *,
    pad_buckets: Sequence[int],
    pad_to_multiple_of: int,
) -> int:
    target = int(current_length)
    if pad_buckets:
        for bucket in pad_buckets:
            if bucket >= target:
                target = int(bucket)
                break
    if pad_to_multiple_of > 0:
        target = ((target + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    return target


def tokenize_batch(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    pad_buckets: Sequence[int],
    pad_to_multiple_of: int,
    max_input_tokens: int,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=(max_input_tokens > 0),
        max_length=(max_input_tokens if max_input_tokens > 0 else None),
        pad_to_multiple_of=(pad_to_multiple_of if pad_to_multiple_of > 0 and not pad_buckets else None),
    )
    if not pad_buckets:
        return encoded

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    current_length = int(input_ids.shape[1])
    target_length = resolve_padded_length(
        current_length,
        pad_buckets=pad_buckets,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    if target_length <= current_length:
        return encoded

    pad_width = target_length - current_length
    encoded["input_ids"] = F.pad(input_ids, (pad_width, 0), value=int(tokenizer.pad_token_id))
    encoded["attention_mask"] = F.pad(attention_mask, (pad_width, 0), value=0)
    if "token_type_ids" in encoded:
        encoded["token_type_ids"] = F.pad(encoded["token_type_ids"], (pad_width, 0), value=0)
    return encoded


def batch_target_lengths(batch: Sequence[Dict[str, Any]], max_new_tokens_cap: int) -> List[int]:
    target_lengths = [int(item["target_len"]) for item in batch]
    if max_new_tokens_cap > 0:
        target_lengths = [min(length, int(max_new_tokens_cap)) for length in target_lengths]
    target_lengths = [max(length, 1) for length in target_lengths]
    return target_lengths


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    ensure_dir(Path(path).resolve().parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_run(
    *,
    baseline: str,
    model: str,
    model_id: str,
    sample_file: str,
    repeat_index: int,
    batch_size: int,
    max_new_tokens: int,
    sort_by_length: bool,
    compile_enabled: bool,
    dataset_size: int,
    total_seconds: float,
    batch_metrics: Sequence[BatchMetrics],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    input_tokens = sum(item.input_tokens for item in batch_metrics)
    padded_tokens = sum(item.padded_tokens for item in batch_metrics)
    target_generated_tokens = sum(item.target_generated_tokens for item in batch_metrics)
    actual_generated_tokens = sum(item.actual_generated_tokens for item in batch_metrics)
    prompts_per_s = dataset_size / total_seconds if total_seconds > 0 else None
    target_generated_tokens_per_s = (
        target_generated_tokens / total_seconds if total_seconds > 0 else None
    )
    actual_generated_tokens_per_s = (
        actual_generated_tokens / total_seconds if total_seconds > 0 else None
    )
    payload: Dict[str, Any] = {
        "baseline": baseline,
        "model": model,
        "model_id": model_id,
        "sample_file": str(sample_file),
        "repeat_index": int(repeat_index),
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "sort_by_length": bool(sort_by_length),
        "compile_enabled": bool(compile_enabled),
        "dataset_size": int(dataset_size),
        "total_seconds": float(total_seconds),
        "prompts_per_s": prompts_per_s,
        "generated_tokens_budget": int(target_generated_tokens),
        "generated_tokens_per_s": target_generated_tokens_per_s,
        "target_generated_tokens": int(target_generated_tokens),
        "target_generated_tokens_per_s": target_generated_tokens_per_s,
        "actual_generated_tokens": int(actual_generated_tokens),
        "actual_generated_tokens_per_s": actual_generated_tokens_per_s,
        "input_tokens": int(input_tokens),
        "input_tokens_per_s": (input_tokens / total_seconds if total_seconds > 0 else None),
        "padded_tokens": int(padded_tokens),
        "padding_overhead_tokens": int(max(padded_tokens - input_tokens, 0)),
        "batch_metrics": [asdict(item) for item in batch_metrics],
    }
    if extra:
        payload.update(extra)
    return payload


def make_output_path(
    *,
    results_dir: str | Path,
    baseline: str,
    model: str,
    sample_file: str | Path,
    batch_size: int,
    repeat_index: int,
) -> Path:
    sample_stem = Path(sample_file).stem
    filename = (
        f"{baseline}__{model.replace('/', '_')}__{sample_stem}"
        f"__bs{batch_size}__rep{repeat_index}__{timestamp_tag()}.json"
    )
    return Path(results_dir) / filename


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Optional cap on each sample target output length; 0 means uncapped.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--no-sort-by-length", action="store_true")
    parser.add_argument("--pad-to-multiple-of", type=int, default=0)
    parser.add_argument("--pad-buckets", default="")
    parser.add_argument("--max-input-tokens", type=int, default=0)


def parsed_limit(args: argparse.Namespace) -> int | None:
    return None if int(args.limit) <= 0 else int(args.limit)
