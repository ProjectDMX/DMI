"""Validate one completed vLLM DMI capture against an external JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.blackbox.storage_contracts import StorageRow, storage_contract_mismatches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--db-database", default="default")
    parser.add_argument("--db-table", default="offload")
    return parser.parse_args()


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def main() -> None:
    args = _parse_args()
    contract = json.loads(args.contract.read_text())

    import clickhouse_driver

    client = clickhouse_driver.Client(args.db_host, port=args.db_port)
    raw_rows = client.execute(
        "SELECT request_id, act_name, layer_no, shard_rank, "
        "start_token_idx, end_token_idx, dtype, shape "
        f"FROM {args.db_database}.{args.db_table} "
        "WHERE model_id = %(model_id)s",
        {"model_id": args.model_id},
        settings={"strings_as_bytes": True},
    )
    rows = [
        StorageRow(
            request_id=_decode(request_id),
            act_name=_decode(act_name),
            layer_no=int(layer_no),
            shard_rank=int(shard_rank),
            start_token_idx=int(start),
            end_token_idx=int(end),
            dtype=_decode(dtype),
            shape=tuple(int(size) for size in shape),
        )
        for request_id, act_name, layer_no, shard_rank, start, end, dtype, shape
        in raw_rows
    ]
    mismatches = storage_contract_mismatches(rows, contract)
    result = {
        "model_id": args.model_id,
        "row_count": len(rows),
        "status": "passed" if not mismatches else "failed",
        "mismatches": mismatches,
    }
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
