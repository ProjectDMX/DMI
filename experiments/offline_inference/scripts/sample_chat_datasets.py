#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import tiktoken
from datasets import Dataset


DEFAULT_OUT_DIR = Path("benchmark/data/offline_e2e")
DEFAULT_SEEDS = [3407, 3408, 3409]


def _default_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


DEFAULT_HF_HOME = _default_hf_home()
DEFAULT_SHAREGPT_PATH = (
    DEFAULT_HF_HOME
    / "hub"
    / "datasets--anon8231489123--ShareGPT_Vicuna_unfiltered"
    / "snapshots"
    / "192ab2185289094fc556ec8ce5ce1e8e587154ca"
    / "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_WILDCHAT_DIR = Path(
    os.environ.get(
        "HF_DATASETS_CACHE",
        str(DEFAULT_HF_HOME / "datasets"),
    )
).expanduser() / "allenai___wild_chat-1_m" / "default" / "0.0.0" / "7d6490e462285cf85d91eabea0f9a954fbddcd1f"


def _canonical_text(messages: Iterable[Dict[str, str]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message["role"]).strip().upper()
        content = str(message["content"]).strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _approx_prompt_token_count(enc: tiktoken.Encoding, messages: List[Dict[str, str]]) -> int:
    return len(enc.encode(_canonical_text(messages), disallowed_special=()))


def _approx_text_token_count(enc: tiktoken.Encoding, text: str) -> int:
    return len(enc.encode(text, disallowed_special=()))


def _extract_prompt_and_target(messages: List[Dict[str, str]]) -> Dict[str, object] | None:
    cleaned = [
        {"role": str(msg["role"]), "content": str(msg["content"]).strip()}
        for msg in messages
        if str(msg.get("content", "")).strip()
    ]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()
    if len(cleaned) < 2:
        return None
    if cleaned[-1]["role"] != "assistant" or cleaned[-2]["role"] != "user":
        return None
    prompt_messages = cleaned[:-1]
    target_text = cleaned[-1]["content"]
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        return None
    if len(prompt_messages[-1]["content"]) < 8:
        return None
    if len(target_text) < 2:
        return None
    return {
        "messages": prompt_messages,
        "target_text": target_text,
    }


def _normalize_sharegpt_row(row: Dict[str, object]) -> Dict[str, object] | None:
    turns = row.get("conversations")
    if not isinstance(turns, list):
        return None
    messages: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        content = str(turn.get("value", "")).strip()
        if role == "human" and content:
            messages.append({"role": "user", "content": content})
        elif role == "gpt" and content:
            messages.append({"role": "assistant", "content": content})
    extracted = _extract_prompt_and_target(messages)
    if extracted is None:
        return None
    return {
        "source_conversation_id": str(row.get("id", "")),
        "messages": extracted["messages"],
        "target_text": extracted["target_text"],
    }


def _normalize_wildchat_row(row: Dict[str, object]) -> Dict[str, object] | None:
    turns = row.get("conversation")
    if not isinstance(turns, list):
        return None
    messages: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": str(role), "content": content})
    extracted = _extract_prompt_and_target(messages)
    if extracted is None:
        return None
    return {
        "source_conversation_id": str(row.get("conversation_hash", "")),
        "messages": extracted["messages"],
        "target_text": extracted["target_text"],
    }


def _finalize_rows(
    *,
    dataset_name: str,
    sample_id: int,
    records: Sequence[Dict[str, object]],
    enc: tiktoken.Encoding,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        messages = list(record["messages"])
        target_text = str(record["target_text"])
        rows.append(
            {
                "dataset": dataset_name,
                "sample_id": int(sample_id),
                "entry_id": f"{dataset_name}_sample{sample_id}_{idx:04d}",
                "source_conversation_id": str(record["source_conversation_id"]),
                "approx_prompt_tokens": _approx_prompt_token_count(enc, messages),
                "approx_target_tokens": _approx_text_token_count(enc, target_text),
                "target_text": target_text,
                "messages": messages,
            }
        )
    rows.sort(key=lambda item: (item["approx_prompt_tokens"], item["approx_target_tokens"], item["entry_id"]))
    return rows


def _sample_sharegpt(raw_rows: List[Dict[str, object]], seeds: List[int], sample_size: int, enc):
    total = len(raw_rows)
    out: Dict[int, List[Dict[str, object]]] = {}
    for sample_id, seed in enumerate(seeds, start=1):
        rng = random.Random(seed)
        chosen: List[Dict[str, object]] = []
        seen_indices = set()
        while len(chosen) < sample_size:
            need = sample_size - len(chosen)
            draw = min(total, max(need * 3, 3000))
            for index in rng.sample(range(total), draw):
                if index in seen_indices:
                    continue
                seen_indices.add(index)
                normalized = _normalize_sharegpt_row(raw_rows[index])
                if normalized is not None:
                    chosen.append(normalized)
                    if len(chosen) >= sample_size:
                        break
            if len(seen_indices) >= total and len(chosen) < sample_size:
                raise RuntimeError("not enough valid ShareGPT rows to sample from")
        out[sample_id] = _finalize_rows(
            dataset_name="sharegpt",
            sample_id=sample_id,
            records=chosen[:sample_size],
            enc=enc,
        )
    return out, total


def _load_wildchat_shards(wildchat_dir: Path):
    shard_paths = sorted(wildchat_dir.glob("wild_chat-1_m-train-*.arrow"))
    if not shard_paths:
        raise FileNotFoundError(f"no WildChat shards found under {wildchat_dir}")
    shards = []
    prefix = [0]
    for path in shard_paths:
        ds = Dataset.from_file(str(path))
        length = len(ds)
        shards.append((path, length))
        prefix.append(prefix[-1] + length)
    return shards, prefix


def _resolve_global_index(prefix: Sequence[int], global_index: int) -> tuple[int, int]:
    shard_idx = bisect.bisect_right(prefix, global_index) - 1
    local_index = global_index - prefix[shard_idx]
    return shard_idx, local_index


def _sample_wildchat(shards, prefix, seeds: List[int], sample_size: int, enc):
    total = prefix[-1]
    out: Dict[int, List[Dict[str, object]]] = {}
    for sample_id, seed in enumerate(seeds, start=1):
        rng = random.Random(seed)
        chosen: List[Dict[str, object]] = []
        seen_indices = set()
        while len(chosen) < sample_size:
            need = sample_size - len(chosen)
            draw = min(total, max(need * 3, 3000))
            candidate_indices = [idx for idx in rng.sample(range(total), draw) if idx not in seen_indices]
            for idx in candidate_indices:
                seen_indices.add(idx)
            by_shard = defaultdict(list)
            for global_index in candidate_indices:
                shard_idx, local_index = _resolve_global_index(prefix, global_index)
                by_shard[shard_idx].append(local_index)
            for shard_idx in sorted(by_shard):
                ds = Dataset.from_file(str(shards[shard_idx][0]))
                for local_index in by_shard[shard_idx]:
                    normalized = _normalize_wildchat_row(ds[int(local_index)])
                    if normalized is not None:
                        chosen.append(normalized)
                        if len(chosen) >= sample_size:
                            break
                if len(chosen) >= sample_size:
                    break
            if len(seen_indices) >= total and len(chosen) < sample_size:
                raise RuntimeError("not enough valid WildChat rows to sample from")
        out[sample_id] = _finalize_rows(
            dataset_name="wildchat",
            sample_id=sample_id,
            records=chosen[:sample_size],
            enc=enc,
        )
    return out, total


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample fixed chat benchmark datasets locally.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--sharegpt-path",
        default=os.environ.get("SHAREGPT_PATH", str(DEFAULT_SHAREGPT_PATH)),
        help="Path to ShareGPT_V3_unfiltered_cleaned_split.json.",
    )
    parser.add_argument(
        "--wildchat-dir",
        default=os.environ.get("WILDCHAT_DIR", str(DEFAULT_WILDCHAT_DIR)),
        help="Directory containing wild_chat-1_m-train-*.arrow shards.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds, one sample file per seed.",
    )
    args = parser.parse_args()

    seeds = [int(chunk.strip()) for chunk in args.seeds.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    enc = tiktoken.get_encoding("cl100k_base")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sharegpt_path = Path(args.sharegpt_path).expanduser()
    wildchat_dir = Path(args.wildchat_dir).expanduser()

    with open(sharegpt_path, "r", encoding="utf-8") as handle:
        sharegpt_raw = json.load(handle)
    sharegpt_samples, sharegpt_total = _sample_sharegpt(
        sharegpt_raw,
        seeds=seeds,
        sample_size=int(args.sample_size),
        enc=enc,
    )

    wildchat_shards, wildchat_prefix = _load_wildchat_shards(wildchat_dir)
    wildchat_samples, wildchat_total = _sample_wildchat(
        wildchat_shards,
        wildchat_prefix,
        seeds=seeds,
        sample_size=int(args.sample_size),
        enc=enc,
    )

    manifest = {
        "sharegpt_source_rows": sharegpt_total,
        "wildchat_source_rows": wildchat_total,
        "sample_size": int(args.sample_size),
        "seeds": seeds,
        "files": [],
    }

    for sample_id, seed in enumerate(seeds, start=1):
        sharegpt_path = out_dir / f"sharegpt_{args.sample_size}_sample{sample_id}.jsonl"
        wildchat_path = out_dir / f"wildchat_{args.sample_size}_sample{sample_id}.jsonl"
        _write_jsonl(sharegpt_path, sharegpt_samples[sample_id])
        _write_jsonl(wildchat_path, wildchat_samples[sample_id])
        manifest["files"].append(
            {
                "dataset": "sharegpt",
                "sample_id": sample_id,
                "seed": seed,
                "path": str(sharegpt_path),
                "count": len(sharegpt_samples[sample_id]),
            }
        )
        manifest["files"].append(
            {
                "dataset": "wildchat",
                "sample_id": sample_id,
                "seed": seed,
                "path": str(wildchat_path),
                "count": len(wildchat_samples[sample_id]),
            }
        )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote samples to {out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
