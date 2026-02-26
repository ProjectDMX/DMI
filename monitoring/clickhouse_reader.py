import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, overload, Literal

import torch


class CHClickhouseDriverReadOnly:
    """
    Read-only ClickHouse interface using `clickhouse-driver` (native protocol).

    - Forces clickhouse-driver: strings_as_bytes=1
    - Provides `prefix_get` for fast primary-key prefix lookups (decodes strings based on init config).
    - Provides `custom_select` for arbitrary reads (NEVER decodes strings; returns raw bytes).
    - Reconstructs tensors directly from (dtype, shape, bytes) schema via `torch_decode()`.
    """

    _HTTP_PORTS = (8123, 8443)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,  # native TCP; if secure, commonly 9440
        username: str = "default",
        password: str = "",
        database: str = "default",
        table: str = "offload",
        secure: bool = False,
        client_settings: Optional[Dict[str, Union[str, int, bool]]] = None,
        primary_key_column_names: Tuple[str, ...] = (
            "model_id",
            "request_id",
            "act_name",
            "layer_no",
            "shard_rank",      # Added from new schema
            "start_token_idx",
            "end_token_idx",
        ),
        order_by_column_names: Optional[Tuple[str, ...]] = None,
        value_column_names: Tuple[str, ...] = ("dtype", "shape", "bytes"),  # Updated schema
        decode_strings: bool = True,
        **_,
    ):
        """
        Args:
            decode_strings (bool): If True (default), automatically decodes byte strings 
                                   into standard Python strings for the primary key 
                                   tuples returned by `prefix_get`. This setting has 
                                   NO effect on `custom_select`.
        """
        from clickhouse_driver import Client as ClickHouseClient
        if ClickHouseClient is None:
            raise ImportError("clickhouse-driver is required. Install with: pip install clickhouse-driver")

        if port in self._HTTP_PORTS:
            raise ValueError(
                f"This class uses clickhouse-driver (native protocol). Port {port} looks like an HTTP port.\n"
                f"Use the native TCP port (commonly 9000; TLS commonly 9440 with secure=True)."
            )

        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._database = database
        self._table = self._validate_ident(table)

        self._primary_key_column_names = tuple(self._validate_ident(c) for c in primary_key_column_names)
        self._order_by_column_names = (
            self._primary_key_column_names
            if order_by_column_names is None
            else tuple(self._validate_ident(c) for c in order_by_column_names)
        )
        self._value_column_names = tuple(self._validate_ident(c) for c in value_column_names)
        if len(self._value_column_names) != 3:
            raise ValueError(
                "value_column_names must be exactly 3 columns: (dtype, shape, bytes). "
                f"Got: {self._value_column_names}"
            )

        self._full_column_names = self._primary_key_column_names + self._value_column_names
        self._pk_count = len(self._primary_key_column_names)

        self._secure = secure
        self._client_settings = dict(client_settings or {})
        self._client_settings["strings_as_bytes"] = 1  # force binary safety
        self._decode_strings = decode_strings

        self._client: Optional[ClickHouseClient] = None

        # Pre-build SELECT templates for prefix_len = 0..pk_count
        self._prefix_select_sql_with_key = [
            self._build_select_sql(
                db=self._database,
                table=self._table,
                pk_names=self._primary_key_column_names[:prefix_len],
                select_col_names=self._full_column_names,  # pk + values
                order_by=self._order_by_column_names,
            )
            for prefix_len in range(self._pk_count + 1)
        ]

        self._prefix_select_sql_values_only = [
            self._build_select_sql(
                db=self._database,
                table=self._table,
                pk_names=self._primary_key_column_names[:prefix_len],
                select_col_names=self._value_column_names,  # values only
                order_by=self._order_by_column_names,
            )
            for prefix_len in range(self._pk_count + 1)
        ]

    # ---------- helpers ----------

    @staticmethod
    def _validate_ident(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid identifier: {name!r}")
        return name

    @staticmethod
    def _backtick(name: str) -> str:
        return f"`{name}`"

    @classmethod
    def _build_select_sql(
        cls,
        db: str,
        table: str,
        pk_names: Sequence[str],
        select_col_names: Sequence[str],
        order_by: Optional[Sequence[str]],
    ) -> str:
        sel = ", ".join(cls._backtick(c) for c in select_col_names)
        where_parts = [f"{cls._backtick(c)} = %({c})s" for c in pk_names]
        sql = f"SELECT {sel} FROM {cls._backtick(db)}.{cls._backtick(table)}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        if order_by:
            sql += " ORDER BY " + ", ".join(cls._backtick(c) for c in order_by)
        return sql

    @staticmethod
    def _decode_key_cell(v: Any) -> Any:
        if isinstance(v, memoryview):
            v = v.tobytes()
        elif isinstance(v, bytearray):
            v = bytes(v)
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="surrogateescape")
        return v

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from clickhouse_driver import Client as ClickHouseClient
        self._client = ClickHouseClient(
            host=self._host,
            port=self._port,
            user=self._username,
            password=self._password,
            database=self._database,
            secure=self._secure,
            settings=self._client_settings,
        )

    # ---------- PUBLIC STATIC DECODERS ----------

    @staticmethod
    def bytes_to_torch_dtype(dtype_bytes: Union[bytes, bytearray, memoryview, str]) -> torch.dtype:
        """Converts a raw ClickHouse string/bytes representation of a dtype to a torch.dtype."""
        if isinstance(dtype_bytes, (memoryview, bytearray)):
            dtype_bytes = bytes(dtype_bytes)
        
        if isinstance(dtype_bytes, bytes):
            dtype_str = dtype_bytes.decode('utf-8')
        else:
            dtype_str = dtype_bytes

        if not dtype_str.startswith("torch."):
            raise ValueError(f"Expected dtype string starting with 'torch.', got {dtype_str!r}")
        
        return getattr(torch, dtype_str.split(".")[1])

    @classmethod
    def torch_decode(
        cls, 
        dtype_bytes: Union[bytes, str], 
        shape: Sequence[int], 
        payload_bytes: Union[bytes, bytearray, memoryview]
    ) -> torch.Tensor:
        """Reconstructs the PyTorch tensor from raw ClickHouse columns."""
        pt_dtype = cls.bytes_to_torch_dtype(dtype_bytes)
        
        if isinstance(payload_bytes, (memoryview, bytearray)):
            payload_bytes = bytes(payload_bytes)

        # Zero-copy buffer reconstruction
        tensor_1d = torch.frombuffer(payload_bytes, dtype=pt_dtype)
        return tensor_1d.reshape(tuple(shape))


    # ---------- PUBLIC QUERY API ----------

    @overload
    def prefix_get(
        self, prefix_key: tuple, *, return_full_key_tuple: Literal[True] = True
    ) -> List[Tuple[tuple, torch.Tensor]]: ...

    @overload
    def prefix_get(self, prefix_key: tuple, *, return_full_key_tuple: Literal[False]) -> List[torch.Tensor]: ...

    def prefix_get(
        self, prefix_key: tuple, *, return_full_key_tuple: bool = True
    ) -> Union[List[Tuple[tuple, torch.Tensor]], List[torch.Tensor]]:
        """
        Fetch rows matching a primary-key prefix.

        If `decode_strings` was True at initialization (default), any string columns 
        in the returned `full_key_tuple` will be decoded to standard Python strings. 
        If False, they remain as raw bytes.

        If return_full_key_tuple=True:
          returns List[(full_key_tuple, tensor)] and SELECTs (pk + values).
        If return_full_key_tuple=False:
          returns List[tensor] and SELECTs (values only).
        """
        self._ensure_client()

        if not isinstance(prefix_key, tuple):
            raise TypeError(f"prefix_key must be a tuple, got {type(prefix_key)!r}")
        if len(prefix_key) == 0:
            raise ValueError("prefix_get requires a non-empty prefix_key")
        if len(prefix_key) > self._pk_count:
            raise ValueError(f"prefix_key too long: got {len(prefix_key)} max {self._pk_count}")

        prefix_len = len(prefix_key)
        pk_names = self._primary_key_column_names[:prefix_len]

        # With strings_as_bytes=1, ClickHouse String params should be bytes for exact matching
        params = {name: prefix_key[i] for i, name in enumerate(pk_names)}

        if return_full_key_tuple:
            sql = self._prefix_select_sql_with_key[prefix_len]
            rows = self._client.execute(sql, params)  # type: ignore[union-attr]
            if not rows:
                return []

            out: List[Tuple[tuple, torch.Tensor]] = []
            pk_count = self._pk_count
            for row in rows:
                if self._decode_strings:
                    key_tuple = tuple(self._decode_key_cell(x) for x in row[:pk_count])
                else:
                    key_tuple = tuple(row[:pk_count])
                
                # Extract the 3 value columns
                dtype_bytes = row[pk_count]
                shape = row[pk_count + 1]
                payload_bytes = row[pk_count + 2]
                
                tensor = self.torch_decode(dtype_bytes, shape, payload_bytes)
                out.append((key_tuple, tensor))
            return out

        # values-only path
        sql = self._prefix_select_sql_values_only[prefix_len]
        rows = self._client.execute(sql, params)  # type: ignore[union-attr]
        if not rows:
            return []

        out_v: List[torch.Tensor] = []
        for dtype_bytes, shape, payload_bytes in rows:
            out_v.append(self.torch_decode(dtype_bytes, shape, payload_bytes))
        return out_v

    def custom_select(
        self, 
        query: str, 
        params: Optional[dict] = None
    ) -> List[Tuple[Any, ...]]:
        """
        Execute a custom SELECT query.

        Because `strings_as_bytes=1` is forced by this driver, ALL ClickHouse String 
        columns (including metadata strings and tensor payloads) will be returned as 
        raw bytes. 

        This method DOES NOT decode any strings automatically. You must manually decode 
        metadata strings, and use `CHClickhouseDriverReadOnly.torch_decode()` to 
        reconstruct tensors from the returned bytes.

        Args:
            query: The SELECT query to run.
            params: Optional parameter dict.
        """
        if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE):
            raise ValueError(
                "custom_select strictly requires a SELECT query. "
                "For absolute security, ensure the ClickHouse user has a read-only profile."
            )

        self._ensure_client()
        params = params or {}

        rows = self._client.execute(query, params)  # type: ignore[union-attr]
        return rows if rows else []

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None

    # ---------- Properties ----------

    @property
    def database(self) -> str:
        return self._database

    @property
    def table(self) -> str:
        return self._table

    @property
    def primary_keys_columns(self) -> Tuple[str, ...]:
        return self._primary_key_column_names

    @property
    def value_columns(self) -> Tuple[str, ...]:
        return self._value_column_names

    @property
    def columns(self) -> Tuple[str, ...]:
        return self._full_column_names