# tests/correctness/db_reader.py
"""ClickHouse reader and tensor decoding utilities for correctness tests."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

BytesLike = Union[bytes, bytearray, memoryview]

TORCH_DTYPES_NAME2TYPE: Dict[str, torch.dtype] = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.short": torch.int16,
    "torch.int32": torch.int32,
    "torch.int": torch.int32,
    "torch.int64": torch.int64,
    "torch.long": torch.int64,
    "torch.bool": torch.bool,
}


# ---------------------------------------------------------------------------
# Raw bytes helpers
# ---------------------------------------------------------------------------


def _to_bytes(v: Any) -> bytes:
    if v is None:
        raise TypeError("cannot convert None to bytes")
    if isinstance(v, bytes):
        return v
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, bytearray):
        return bytes(v)
    # ClickHouse may return Array(UInt8) as list[int]
    if isinstance(v, (list, tuple)) and (not v or isinstance(v[0], int)):
        return bytes(int(x) & 0xFF for x in v)
    if isinstance(v, str):
        return v.encode("utf-8", errors="surrogateescape")
    raise TypeError(f"expected bytes-like, got {type(v)!r}")


# ---------------------------------------------------------------------------
# Tensor decoding (v1: json+bytes, v2: dtype+shape+bytes)
# ---------------------------------------------------------------------------


def torch_decode_v1(meta: BytesLike, payload: Any) -> torch.Tensor:
    meta_b = _to_bytes(meta)
    meta_json = json.loads(meta_b.decode("utf-8", errors="surrogateescape"))
    dtype_str = meta_json["dtype"]
    shape = tuple(int(x) for x in meta_json["shape"])
    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    data_b = _to_bytes(payload)
    return torch.frombuffer(data_b, dtype=dtype).clone().view(*shape)


def torch_decode_v2(dtype_cell: Any, shape_cell: Any, payload: Any) -> torch.Tensor:
    if isinstance(dtype_cell, (bytes, bytearray, memoryview)):
        dtype_str = dtype_cell.decode("utf-8", errors="surrogateescape")
    else:
        dtype_str = str(dtype_cell)

    if isinstance(shape_cell, (bytes, bytearray, memoryview)):
        shape_raw = shape_cell.decode("utf-8", errors="surrogateescape")
        try:
            shape = tuple(int(x) for x in ast.literal_eval(shape_raw))
        except Exception:
            shape = tuple(int(x) for x in re.findall(r"\d+", shape_raw))
    elif isinstance(shape_cell, (list, tuple)):
        shape = tuple(int(x) for x in shape_cell)
    else:
        shape_raw = str(shape_cell)
        try:
            shape = tuple(int(x) for x in ast.literal_eval(shape_raw))
        except Exception:
            shape = tuple(int(x) for x in re.findall(r"\d+", shape_raw))

    dtype = TORCH_DTYPES_NAME2TYPE[dtype_str]
    data_b = _to_bytes(payload)
    return torch.frombuffer(data_b, dtype=dtype).clone().view(*shape)


# ---------------------------------------------------------------------------
# ClickHouse config + reader
# ---------------------------------------------------------------------------


@dataclass
class _DBConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    table: str
    secure: bool = False


class _ClickHouseReader:
    def __init__(self, cfg: _DBConfig):
        from clickhouse_driver import Client as CHClient

        settings = {"strings_as_bytes": 1}
        self._client = CHClient(
            host=cfg.host,
            port=int(cfg.port),
            user=cfg.username,
            password=cfg.password,
            database=cfg.database,
            secure=bool(cfg.secure),
            settings=settings,
        )
        self._db = cfg.database
        self._table = cfg.table
        self._columns = self._describe_columns()
        self._value_cols = self._infer_value_columns(self._columns)
        self._has_shard_rank = "shard_rank" in self._columns

    def _describe_columns(self) -> List[str]:
        rows = self._client.execute(f"DESCRIBE TABLE {self._db}.{self._table}")
        cols: List[str] = []
        for r in rows:
            if not r:
                continue
            c = r[0]
            if isinstance(c, (bytes, bytearray, memoryview)):
                c = c.decode("utf-8", errors="surrogateescape")
            else:
                c = str(c)
            if c.startswith("b'") and c.endswith("'"):
                c = c[2:-1]
            cols.append(c)
        return cols

    @staticmethod
    def _infer_value_columns(columns: Sequence[str]) -> Tuple[str, ...]:
        if "json" in columns and "bytes" in columns:
            return ("json", "bytes")
        if "dtype" in columns and "shape" in columns and "bytes" in columns:
            return ("dtype", "shape", "bytes")
        raise RuntimeError(f"Could not infer value columns. Columns present: {list(columns)}")

    def fetch_all_rows_for_model(self, *, model_id: str) -> List[Any]:
        shard_expr = "shard_rank" if self._has_shard_rank else "0"
        value_select = ", ".join(self._value_cols)
        q = (
            "SELECT model_id, request_id, act_name, layer_no, start_token_idx, end_token_idx, "
            f"{shard_expr} as shard_rank, {value_select} "
            f"FROM {self._db}.{self._table} "
            "WHERE model_id = %(model_id)s "
            "ORDER BY request_id, act_name, layer_no, start_token_idx, end_token_idx"
        )
        rows = self._client.execute(q, {"model_id": model_id})
        out = []
        for r in rows:
            _model_raw, req_raw, act_raw, layer_no, s, e, shard_rank, *vals = r
            req_id = (
                req_raw.decode("utf-8", errors="surrogateescape")
                if isinstance(req_raw, (bytes, bytearray, memoryview))
                else str(req_raw)
            )
            act_name = (
                act_raw.decode("utf-8", errors="surrogateescape")
                if isinstance(act_raw, (bytes, bytearray, memoryview))
                else str(act_raw)
            )
            out.append((req_id, act_name, int(layer_no), int(s), int(e), int(shard_rank), vals))
        return out

    def decode_tensor(self, vals: Sequence[Any]) -> torch.Tensor:
        if len(vals) == 2:
            meta, payload = vals
            return torch_decode_v1(meta, payload)
        if len(vals) == 3:
            dtype_cell, shape_cell, payload = vals
            return torch_decode_v2(dtype_cell, shape_cell, payload)
        raise RuntimeError(f"Unexpected value columns length: {len(vals)}")
