#!/usr/bin/env python3
"""Fail unless selected physical NVIDIA GPUs remain idle for every sample."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical indices")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-memory-mb", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    return parser.parse_args()


def _nvidia_smi(*fields: str, compute: bool = False) -> list[list[str]]:
    query = "--query-compute-apps=" if compute else "--query-gpu="
    output = subprocess.check_output(
        [
            "nvidia-smi",
            query + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        [item.strip() for item in row]
        for row in csv.reader(io.StringIO(output))
        if row
    ]


def _sample(selected: set[int], max_memory: int, max_utilization: int) -> list[str]:
    gpu_rows = _nvidia_smi("index", "uuid", "memory.used", "utilization.gpu")
    by_index = {
        int(index): {
            "uuid": uuid,
            "memory": int(memory),
            "utilization": int(utilization),
        }
        for index, uuid, memory, utilization in gpu_rows
    }
    missing = selected - set(by_index)
    if missing:
        return [f"unknown GPU indices: {sorted(missing)}"]

    compute_rows = _nvidia_smi("gpu_uuid", "pid", "process_name", compute=True)
    compute_by_uuid: dict[str, list[str]] = {}
    for uuid, pid, process_name in compute_rows:
        compute_by_uuid.setdefault(uuid, []).append(f"pid={pid} {process_name}")

    errors = []
    for index in sorted(selected):
        state = by_index[index]
        processes = compute_by_uuid.get(state["uuid"], [])
        if processes:
            errors.append(f"GPU {index} has compute processes: {', '.join(processes)}")
        if state["memory"] >= max_memory:
            errors.append(
                f"GPU {index} uses {state['memory']} MiB (limit < {max_memory} MiB)"
            )
        if state["utilization"] >= max_utilization:
            errors.append(
                f"GPU {index} utilization is {state['utilization']}% "
                f"(limit < {max_utilization}%)"
            )
    return errors


def main() -> None:
    args = _parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.interval < 0:
        raise SystemExit("--interval must be non-negative")
    try:
        selected = {int(item) for item in args.gpus.split(",") if item.strip()}
    except ValueError as error:
        raise SystemExit("--gpus must contain integer indices") from error
    if len(selected) < 1:
        raise SystemExit("--gpus must select at least one GPU")

    for sample in range(1, args.samples + 1):
        errors = _sample(
            selected,
            max_memory=args.max_memory_mb,
            max_utilization=args.max_utilization,
        )
        if errors:
            detail = "\n".join(f"  - {error}" for error in errors)
            raise SystemExit(f"GPU idle check failed at sample {sample}:\n{detail}")
        print(f"sample {sample}/{args.samples}: GPUs {sorted(selected)} idle", flush=True)
        if sample < args.samples:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
