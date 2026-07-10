#!/usr/bin/env python3
"""Create a run manifest for the PCIe pressure scaffold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import create_manifest, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", default="synthetic-pcie-pressure")
    p.add_argument("--config", default="")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_data = read_json(args.config) if args.config else {}
    extra = {
        "notes": args.notes,
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "DRY_RUN",
                "GPUS",
                "DIRECTION",
                "BLOCK_MB",
                "INFLIGHT",
                "TARGET_MB_S",
                "DURATION_S",
                "VLLM_METRICS_URL",
                "LMCACHE_METRICS_URLS",
                "DMI_METRICS_URLS",
            )
            if os.environ.get(key) is not None
        },
        "config": config_data,
    }
    manifest = create_manifest(
        args.run_dir,
        run_id=args.run_id,
        mode=args.mode,
        config_path=args.config or None,
        extra=extra,
    )
    if args.config:
        write_json(Path(args.run_dir) / "config.resolved.json", config_data)
    print(json.dumps({"run_dir": args.run_dir, "run_id": args.run_id, "manifest": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
