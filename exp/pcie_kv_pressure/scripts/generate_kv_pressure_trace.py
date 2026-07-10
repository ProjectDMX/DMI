#!/usr/bin/env python3
"""Generate a high KV store/retrieve pressure trace.

The output is compatible with the HiddenCache workload replay format:
`prefix_bank.jsonl` stores long reusable prefixes and `trace.jsonl` stores
scheduled OpenAI-compatible requests.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-prefixes", type=int, default=None)
    p.add_argument("--prefix-len-tokens", type=int, default=None)
    p.add_argument("--retrieve-repeats", type=int, default=None)
    p.add_argument("--reuse-gap-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def merge_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "num_prefixes": args.num_prefixes,
        "prefix_len_tokens": args.prefix_len_tokens,
        "retrieve_repeats": args.retrieve_repeats,
        "reuse_gap_s": args.reuse_gap_s,
        "seed": args.seed,
    }
    out = dict(cfg)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def as_int(cfg: dict[str, Any], key: str) -> int:
    return int(cfg[key])


def as_float(cfg: dict[str, Any], key: str) -> float:
    return float(cfg[key])


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")


def make_approx_token_text(label: str, target_tokens: int) -> str:
    """Build deterministic text with roughly `target_tokens` whitespace tokens."""

    base = safe_label(label)
    vocab = (
        "system",
        "analysis",
        "context",
        "memory",
        "request",
        "response",
        "evidence",
        "retrieval",
        "planning",
        "tool",
        "state",
        "result",
        "summary",
        "decision",
        "trace",
        "token",
    )
    offset = sum(ord(ch) for ch in base) % len(vocab)
    words = [vocab[(offset + i) % len(vocab)] for i in range(max(1, target_tokens))]
    words[0] = base
    lines = [" ".join(words[i : i + 64]) for i in range(0, len(words), 64)]
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def jitter(base_ts: float, rng: random.Random, width_s: float) -> float:
    if width_s <= 0:
        return base_ts
    return max(0.0, base_ts + rng.uniform(-width_s, width_s))


def build_prefix_bank(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix_len = as_int(cfg, "prefix_len_tokens")
    for idx in range(as_int(cfg, "num_prefixes")):
        workflow_id = f"wf_{idx:05d}"
        agent_id = "kv_pressure"
        prefix_id = f"{workflow_id}/{agent_id}"
        rows.append(
            {
                "prefix_id": prefix_id,
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "target_prefix_len_tokens": prefix_len,
                "actual_prefix_len_tokens": prefix_len,
                "prefix_text": make_approx_token_text(f"hot prefix {prefix_id}", prefix_len),
            }
        )
    return rows


def add_event(
    events: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    prefix: dict[str, Any],
    timestamp_s: float,
    phase: str,
    step_id: int,
    last_seen: dict[str, float],
) -> None:
    prefix_id = str(prefix["prefix_id"])
    prior_ts = last_seen.get(prefix_id)
    expected_reuse = prior_ts is not None
    suffix_len = as_int(cfg, "suffix_len_tokens")
    max_tokens = as_int(cfg, "max_tokens")
    event_id = f"evt_{len(events):08d}"
    approx_prefix_kv_bytes = as_int(cfg, "prefix_len_tokens") * as_int(cfg, "kv_bytes_per_token")
    expected_action = "retrieve_then_store" if expected_reuse else "store"
    event = {
        "event_id": event_id,
        "timestamp_s": round(timestamp_s, 6),
        "workflow_id": prefix["workflow_id"],
        "agent_id": prefix["agent_id"],
        "prefix_id": prefix_id,
        "prompt_prefix_id": prefix_id,
        "step_id": step_id,
        "branch_id": 0,
        "parent_event_id": None,
        "prefix_len_tokens": as_int(cfg, "prefix_len_tokens"),
        "suffix_len_tokens": suffix_len,
        "max_tokens": max_tokens,
        "temperature": float(cfg.get("temperature", 0.0)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "reactivation_gap_s": None if prior_ts is None else round(timestamp_s - prior_ts, 6),
        "expected_prefix_reuse": expected_reuse,
        "expected_kv_action": expected_action,
        "phase": phase,
        "kv_pressure_phase": phase,
        "approx_prefix_kv_bytes": approx_prefix_kv_bytes,
        "approx_prefix_kv_mb": approx_prefix_kv_bytes / 1024 / 1024,
        "is_branch_fanout": False,
        "suffix_text": make_approx_token_text(f"suffix {event_id} {phase}", suffix_len),
    }
    events.append(event)
    last_seen[prefix_id] = timestamp_s


def build_events(cfg: dict[str, Any], prefixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(as_int(cfg, "seed"))
    store_rps = max(as_float(cfg, "store_rps"), 0.001)
    retrieve_rps = max(as_float(cfg, "retrieve_rps"), 0.001)
    retrieve_repeats = as_int(cfg, "retrieve_repeats")
    reuse_gap_s = as_float(cfg, "reuse_gap_s")
    jitter_s = as_float(cfg, "timestamp_jitter_s")

    events: list[dict[str, Any]] = []
    last_seen: dict[str, float] = {}
    burst_size = int(cfg.get("store_burst_size", 0) or 0)
    burst_gap_s = float(cfg.get("store_burst_gap_s", 0.0) or 0.0)
    burst_spacing_s = float(cfg.get("store_burst_spacing_s", 0.0) or 0.0)

    for idx, prefix in enumerate(prefixes):
        if burst_size > 0:
            burst_idx, within_burst = divmod(idx, burst_size)
            ts = burst_idx * burst_gap_s + within_burst * burst_spacing_s
        else:
            ts = idx / store_rps
        ts = jitter(ts, rng, jitter_s)
        add_event(
            events,
            cfg=cfg,
            prefix=prefix,
            timestamp_s=ts,
            phase="cold_store",
            step_id=0,
            last_seen=last_seen,
        )

    cold_store_end = max(last_seen.values(), default=0.0)
    base = cold_store_end + reuse_gap_s
    for repeat in range(retrieve_repeats):
        order = list(prefixes)
        rng.shuffle(order)
        for idx, prefix in enumerate(order):
            ts = base + repeat * (len(prefixes) / retrieve_rps) + idx / retrieve_rps
            ts = jitter(ts, rng, jitter_s)
            add_event(
                events,
                cfg=cfg,
                prefix=prefix,
                timestamp_s=ts,
                phase=f"hot_retrieve_{repeat}",
                step_id=repeat + 1,
                last_seen=last_seen,
            )

    return sorted(events, key=lambda row: (row["timestamp_s"], row["event_id"]))


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(read_json(args.config), args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = build_prefix_bank(cfg)
    events = build_events(cfg, prefixes)

    write_jsonl(out_dir / "prefix_bank.jsonl", prefixes)
    write_jsonl(out_dir / "trace.jsonl", events)

    total_approx_kv_bytes = sum(int(event["approx_prefix_kv_bytes"]) for event in events)
    metadata = {
        "config": cfg,
        "files": {
            "prefix_bank": "prefix_bank.jsonl",
            "trace": "trace.jsonl",
        },
        "num_prefixes": len(prefixes),
        "num_events": len(events),
        "cold_store_events": sum(1 for event in events if event["expected_kv_action"] == "store"),
        "retrieve_events": sum(1 for event in events if event["expected_prefix_reuse"]),
        "duration_s": max((float(event["timestamp_s"]) for event in events), default=0.0),
        "approx_total_prefix_kv_gb_referenced": total_approx_kv_bytes / 1024 / 1024 / 1024,
        "notes": (
            "First wave introduces unique long prefixes to force KV store/offload. "
            "Later waves reuse the same prefixes after reuse_gap_s to force KV retrieve/load."
        ),
    }
    write_json(out_dir / "metadata.json", metadata)
    # The shared vLLM launcher expects this conventional name for its copied
    # trace manifest.  Keep metadata.json for backward compatibility.
    write_json(out_dir / "trace_summary.json", metadata)
    print(json.dumps({"out_dir": str(out_dir), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
