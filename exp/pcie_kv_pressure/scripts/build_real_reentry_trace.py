#!/usr/bin/env python3
"""Build real-prompt warm/evict/replay traces via HiddenCache RealReentry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from common import write_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hiddencache_dir() -> Path:
    return repo_root().parent / "hiddencache"


def default_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hiddencache-dir", type=Path, default=default_hiddencache_dir())
    p.add_argument("--mode", choices=("length-bucket", "reentry"), default="length-bucket")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--hf-home", type=Path, default=default_hf_home())
    p.add_argument("--no-resolve-tokenizer-cache", action="store_true")
    p.add_argument("--allow-tokenizer-download", action="store_true")
    p.add_argument("--tokenizer-trust-remote-code", action="store_true")
    p.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    p.add_argument("--datasets", default="arena-hard-v2,swe-bench,livecodebench,mt-bench")
    p.add_argument("--include-all-turns", action="store_true")
    p.add_argument("--seed", type=int, default=980406)
    p.add_argument("--warm-max-tokens", type=int, default=1)
    p.add_argument("--evict-max-tokens", type=int, default=1)
    p.add_argument("--replay-max-tokens", type=int, default=128)
    p.add_argument("--warm-qps", type=float, default=2.0)
    p.add_argument("--evict-qps", type=float, default=4.0)
    p.add_argument("--replay-qps", type=float, default=1.0)
    p.add_argument("--phase-gap-s", type=float, default=120.0)
    p.add_argument("--timestamp-jitter-s", type=float, default=0.05)

    # length-bucket mode.
    p.add_argument(
        "--bucket-spec",
        default="128:512:40,512:1024:40,1024:2048:40,2048:4096:20,4096:8192:5",
    )
    p.add_argument("--strict-buckets", action="store_true")

    # reentry mode.
    p.add_argument("--target-count", type=int, default=150)
    p.add_argument("--evict-count", type=int, default=600)
    p.add_argument("--min-prompt-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=12000)
    return p.parse_args()


def resolve_tokenizer_path(args: argparse.Namespace) -> str:
    value = str(args.tokenizer)
    if args.no_resolve_tokenizer_cache or args.allow_tokenizer_download:
        return value
    direct = Path(value).expanduser()
    if direct.exists():
        return str(direct)
    if "/" not in value:
        return value

    cache_name = "models--" + value.replace("/", "--")
    snapshots = args.hf_home / "hub" / cache_name / "snapshots"
    if not snapshots.exists():
        return value
    candidates = sorted([path for path in snapshots.iterdir() if path.is_dir()])
    if not candidates:
        return value
    return str(candidates[-1])


def script_for_mode(hiddencache_dir: Path, mode: str) -> Path:
    name = "build_length_bucket_trace.py" if mode == "length-bucket" else "build_reentry_trace.py"
    return hiddencache_dir / "RealReentry" / "scripts" / name


def default_out_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return repo_root() / "exp" / "pcie_kv_pressure" / "traces" / f"real_{args.mode}_{stamp}"


def common_builder_args(args: argparse.Namespace, out_dir: Path) -> list[str]:
    dataset_root = args.hiddencache_dir / "DeepSpec" / "eval_datasets"
    tokenizer = resolve_tokenizer_path(args)
    cmd = [
        "--dataset-root",
        str(dataset_root),
        "--datasets",
        args.datasets,
        "--tokenizer",
        tokenizer,
        "--endpoint",
        args.endpoint,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--warm-max-tokens",
        str(args.warm_max_tokens),
        "--evict-max-tokens",
        str(args.evict_max_tokens),
        "--replay-max-tokens",
        str(args.replay_max_tokens),
        "--warm-qps",
        str(args.warm_qps),
        "--evict-qps",
        str(args.evict_qps),
        "--replay-qps",
        str(args.replay_qps),
        "--phase-gap-s",
        str(args.phase_gap_s),
        "--timestamp-jitter-s",
        str(args.timestamp_jitter_s),
    ]
    if not args.allow_tokenizer_download:
        cmd.append("--tokenizer-local-files-only")
    if args.tokenizer_trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    if args.include_all_turns:
        cmd.append("--include-all-turns")
    return cmd


def build_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    builder = script_for_mode(args.hiddencache_dir, args.mode)
    if not builder.exists():
        raise FileNotFoundError(builder)
    cmd = ["python", str(builder)]
    cmd.extend(common_builder_args(args, out_dir))
    if args.mode == "length-bucket":
        cmd.extend(["--bucket-spec", args.bucket_spec])
        cmd.extend(["--evict-count", str(args.evict_count)])
        if args.strict_buckets:
            cmd.append("--strict-buckets")
    else:
        cmd.extend(["--target-count", str(args.target_count)])
        cmd.extend(["--evict-count", str(args.evict_count)])
        cmd.extend(["--min-prompt-tokens", str(args.min_prompt_tokens)])
        cmd.extend(["--max-prompt-tokens", str(args.max_prompt_tokens)])
    return cmd


def read_summary(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "trace_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, out_dir)
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if not args.allow_tokenizer_download:
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
    subprocess.run(cmd, check=True, cwd=str(args.hiddencache_dir), env=env)

    manifest = {
        "source": "HiddenCache RealReentry",
        "hiddencache_dir": str(args.hiddencache_dir),
        "mode": args.mode,
        "command": cmd,
        "requested_tokenizer": args.tokenizer,
        "resolved_tokenizer": resolve_tokenizer_path(args),
        "out_dir": str(out_dir),
        "files": {
            "prefix_bank": "prefix_bank.jsonl",
            "trace": "trace.jsonl",
            "trace_warm": "trace_warm.jsonl",
            "trace_evict": "trace_evict.jsonl",
            "trace_replay": "trace_replay.jsonl",
            "selected_prompts": "selected_prompts.jsonl",
            "trace_summary": "trace_summary.json",
        },
        "summary": read_summary(out_dir),
    }
    write_json(out_dir / "dmi_real_reentry_manifest.json", manifest)
    print(json.dumps({"out_dir": str(out_dir), "manifest": str(out_dir / "dmi_real_reentry_manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
