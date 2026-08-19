"""Online Qwen3 storage verification for the vLLM adaptor.

This is a manually invoked GPU/ClickHouse integration gate.  It serves the
same prefix-cache workload through ``Qwen3RefForCausalLM`` (D2D buffers saved
to disk) and ``DMXGPUWorker`` (ring offload to ClickHouse), then requires exact
stable HTTP-field parity (excluding ``created``) and bitwise equality for every
captured tensor.

Example::

    CUDA_VISIBLE_DEVICES=0 python -m tests.vllm_online_storage --tp 1 --pp 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


_MODEL = "Qwen/Qwen3-0.6B"
_NUM_LAYERS = 28
_TP0_LAYER_HOOKS = 7
_TP_SHARDED_LAYER_HOOKS = 5
_GLOBAL_HOOKS = 5
_REQUESTS = (
    ("dmi-online-cold", 0, (0, 80)),
    ("dmi-online-warm", 64, (64, 80)),
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _hook_count_per_request(tp_size: int) -> int:
    """Physical rows across all TP/PP workers for one request."""
    return (
        _NUM_LAYERS
        * (_TP0_LAYER_HOOKS + _TP_SHARDED_LAYER_HOOKS * tp_size)
        + _GLOBAL_HOOKS
    )


def _json_request(
    url: str,
    body: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, method="GET" if data is None else "POST")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    return None if not payload else json.loads(payload)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _wait_ready(process: subprocess.Popen[Any], port: int, log_path: Path) -> None:
    deadline = time.monotonic() + 240
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with {process.returncode}:\n{_tail(log_path)}"
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f"vLLM server did not become ready:\n{_tail(log_path)}")


def _stop_process(process: subprocess.Popen[Any]) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=40)
        return int(process.returncode)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    return int(process.returncode)


@contextmanager
def _server(
    *,
    worker_cls: str,
    port: int,
    log_path: Path,
    env: dict[str, str],
    tp_size: int,
    pp_size: int,
    additional_config: dict[str, Any] | None = None,
) -> Iterator[subprocess.Popen[Any]]:
    master_port = _free_port()
    while master_port == port:
        master_port = _free_port()
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        _MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--master-port",
        str(master_port),
        "--worker-cls",
        worker_cls,
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.5",
        "--shutdown-timeout",
        "30",
        "--enforce-eager",
        "--no-async-scheduling",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--block-size",
        "16",
    ]
    if additional_config is not None:
        command.extend(["--additional-config", json.dumps(additional_config)])

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(process, port, log_path)
            yield process
        finally:
            body_failed = sys.exc_info()[0] is not None
            return_code = _stop_process(process)
            if return_code != 0 and not body_failed:
                raise RuntimeError(
                    f"vLLM server exited with {return_code}:\n{_tail(log_path)}"
                )


def _completion(port: int, base_id: str) -> dict[str, Any]:
    body = {
        "model": _MODEL,
        "prompt": list(range(100, 180)),
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "return_token_ids": True,
        "request_id": base_id,
    }
    result = _json_request(
        f"http://127.0.0.1:{port}/v1/completions",
        body,
        headers={"X-Request-Id": base_id},
        timeout=60,
    )
    if not isinstance(result, dict) or "error" in result:
        raise AssertionError(f"completion failed: {result!r}")
    return result


def _run_workload(port: int) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for base_id, expected_cached, _ in _REQUESTS:
        result = _completion(port, base_id)
        expected_id = f"cmpl-{base_id}"
        assert result["id"] == expected_id, result
        usage = result["usage"]
        assert usage["prompt_tokens"] == 80, usage
        assert usage["completion_tokens"] == 1, usage
        details = usage.get("prompt_tokens_details") or {}
        assert details.get("cached_tokens") == expected_cached, usage
        responses[expected_id] = result
    return responses


def _stable_response(response: dict[str, Any]) -> dict[str, Any]:
    stable = dict(response)
    stable.pop("created", None)
    return stable


def _assert_http_parity(
    reference: dict[str, dict[str, Any]],
    monitored: dict[str, dict[str, Any]],
) -> None:
    assert set(reference) == set(monitored)
    for request_id in sorted(reference):
        assert _stable_response(reference[request_id]) == _stable_response(
            monitored[request_id]
        )


def _quote_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return f"`{value}`"


def _assert_database(
    client: Any,
    *,
    database: str,
    table: str,
    model_id: str,
    hook_count: int,
    tp_size: int,
) -> None:
    fq_table = f"{_quote_identifier(database)}.{_quote_identifier(table)}"
    rows = client.execute(
        "SELECT request_id, act_name, layer_no, shard_rank, "
        "start_token_idx, end_token_idx, shape, bytes "
        f"FROM {fq_table} WHERE model_id = %(model_id)s",
        {"model_id": model_id},
        settings={"strings_as_bytes": True},
    )
    assert len(rows) == 2 * hook_count, len(rows)

    decoded: list[tuple[str, str, int, int, int, int, list[int], bytes]] = []
    for request_id, act_name, layer, shard, start, end, shape, payload in rows:
        rid = request_id.decode() if isinstance(request_id, bytes) else request_id
        act = act_name.decode() if isinstance(act_name, bytes) else act_name
        decoded.append(
            (
                rid,
                act,
                int(layer),
                int(shard),
                int(start),
                int(end),
                list(shape),
                bytes(payload),
            )
        )

    keys = [row[:6] for row in decoded]
    assert len(set(keys)) == len(keys), "duplicate ClickHouse primary-key rows"
    assert all(end > start for _, _, _, _, start, end, _, _ in decoded)
    assert all(shape and shape[0] == end - start for *_, start, end, shape, _ in decoded)
    assert all(payload for *_, payload in decoded)

    # The public completion ID is ``cmpl-{base_id}``; vLLM appends ``-0``
    # for the first (and here only) prompt before submitting it to the engine.
    expected_ids = {f"cmpl-{base_id}-0" for base_id, _, _ in _REQUESTS}
    assert {row[0] for row in decoded} == expected_ids
    for base_id, _, expected_range in _REQUESTS:
        request_id = f"cmpl-{base_id}-0"
        request_rows = [row for row in decoded if row[0] == request_id]
        hook_ids = {(row[1], row[2], row[3]) for row in request_rows}
        assert len(request_rows) == hook_count
        assert len(hook_ids) == hook_count
        expected_shards = {0: _hook_count_per_request(1)}
        if tp_size == 2:
            expected_shards[1] = _NUM_LAYERS * _TP_SHARDED_LAYER_HOOKS
        assert Counter(row[3] for row in request_rows) == Counter(
            expected_shards
        )
        for row in request_rows:
            row_range = (row[4], row[5])
            if row[1] == "final_logits":
                assert row_range == (79, 80)
            else:
                assert row_range == expected_range

        resid = [row for row in request_rows if row[1] == "blocks.hook_resid_pre"]
        assert len(resid) == 28
        assert len({row[2] for row in resid}) == 28
        assert {(row[4], row[5]) for row in resid} == {expected_range}
        expected_token_rows = 28 * (expected_range[1] - expected_range[0])
        assert sum(row[5] - row[4] for row in resid) == expected_token_rows
        assert sum(row[6][0] for row in resid) == expected_token_rows


def _wait_for_rows(
    client: Any,
    fq_table: str,
    model_id: str,
    expected_rows: int,
) -> None:
    deadline = time.monotonic() + 30
    last_count = 0
    while time.monotonic() < deadline:
        last_count = int(
            client.execute(
                f"SELECT count() FROM {fq_table} WHERE model_id = %(model_id)s",
                {"model_id": model_id},
            )[0][0]
        )
        if last_count == expected_rows:
            return
        if last_count > expected_rows:
            break
        time.sleep(0.25)
    raise AssertionError(f"expected {expected_rows} rows, found {last_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--db-database", default="default")
    parser.add_argument("--db-table")
    parser.add_argument("--keep-table", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    args = parser.parse_args()

    if (args.tp, args.pp) not in {(1, 1), (2, 1), (1, 2)}:
        parser.error(
            "validated topologies are --tp 1 --pp 1, --tp 2 --pp 1, "
            "and --tp 1 --pp 2"
        )
    hook_count = _hook_count_per_request(args.tp)
    expected_rows = 2 * hook_count

    project_root = Path(__file__).resolve().parents[1]
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).resolve()
        if artifact_dir.exists() and any(artifact_dir.iterdir()):
            raise RuntimeError(f"artifact directory is not empty: {artifact_dir}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifact_dir = Path(
            tempfile.mkdtemp(prefix="dmi_vllm_online_storage_")
        ).resolve()
    ref_dir = artifact_dir / "ref"
    ref_dir.mkdir(exist_ok=True)
    ref_config = ref_dir / "ref_config.json"

    run_tag = uuid.uuid4().hex[:12]
    table = args.db_table or f"dmi_vllm_online_tp{args.tp}_pp{args.pp}_{run_tag}"
    database = args.db_database
    _quote_identifier(database)
    _quote_identifier(table)
    if not table.startswith("dmi_vllm_online_"):
        raise ValueError("test table name must start with 'dmi_vllm_online_'")
    model_id = f"dmi-online-qwen3-tp{args.tp}-pp{args.pp}-{run_tag}"
    fq_table = f"{_quote_identifier(database)}.{_quote_identifier(table)}"

    import clickhouse_driver

    client = clickhouse_driver.Client(args.db_host, port=args.db_port)
    existing = client.execute(
        "SELECT count() FROM system.tables WHERE database=%(database)s "
        "AND name=%(table)s",
        {"database": database, "table": table},
    )[0][0]
    if existing:
        raise RuntimeError(f"refusing to reuse existing table {database}.{table}")

    qwen3_ref = (
        project_root
        / "integration/vllm/vllm/model_executor/models/qwen3_ref.py"
    )
    submodule_root = project_root / "integration/vllm"
    source_status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(qwen3_ref)],
        cwd=submodule_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if source_status.stdout.strip():
        raise RuntimeError(f"refusing to overwrite modified source: {qwen3_ref}")
    backup = artifact_dir / "qwen3_ref.py.bak"
    shutil.copy2(qwen3_ref, backup)

    base_env = dict(os.environ)
    base_env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    base_env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    base_env["VLLM_SERVER_DEV_MODE"] = "1"
    base_env["HF_HUB_OFFLINE"] = "1"
    prior_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(project_root) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )

    try:
        from integration.vllm.vllm.model_executor.models.enable_ref_hooks import (
            enable_ref_hooks,
        )

        enable_ref_hooks(
            model_file=str(qwen3_ref),
            hooks="vllm-full",
            max_len=128,
            output_dir=str(ref_dir),
            config_out=str(ref_config),
        )

        ref_env = dict(base_env)
        ref_env["REF_CONFIG"] = str(ref_config)
        ref_port = _free_port()
        print(f"[online-storage] reference server port={ref_port}", flush=True)
        with _server(
            worker_cls="tests.ref_disk_worker.RefDiskWorker",
            port=ref_port,
            log_path=artifact_dir / "reference-server.log",
            env=ref_env,
            tp_size=args.tp,
            pp_size=args.pp,
        ):
            reference_responses = _run_workload(ref_port)

        shutil.copy2(backup, qwen3_ref)
        ref_files = list(ref_dir.glob("cmpl-dmi-online-*/*.pt"))
        assert len(ref_files) == expected_rows, len(ref_files)

        monitored_port = _free_port()
        print(f"[online-storage] monitored server port={monitored_port}", flush=True)
        monitored_config = {
            "dmx_model_id": model_id,
            "dmx_hook_selection": "vllm-full",
            "dmx_ring_payload_mb": 256,
            "dmx_ring_pinned_mb": 256,
            "dmx_db_host": args.db_host,
            "dmx_db_port": args.db_port,
            "dmx_db_database": database,
            "dmx_db_table": table,
            "dmx_ch_parallelism": 1,
            "dmx_drain_flush_timeout_us": 100000,
        }
        with _server(
            worker_cls="integration.vllm_adapter.DMXGPUWorker",
            port=monitored_port,
            log_path=artifact_dir / "monitored-server.log",
            env=base_env,
            tp_size=args.tp,
            pp_size=args.pp,
            additional_config=monitored_config,
        ):
            monitored_responses = _run_workload(monitored_port)
            stop_result = _json_request(
                f"http://127.0.0.1:{monitored_port}/collective_rpc",
                {"method": "stop_monitoring", "timeout": 30},
                timeout=40,
            )
            assert stop_result == {"results": [None] * (args.tp * args.pp)}, (
                stop_result
            )
            _wait_for_rows(client, fq_table, model_id, expected_rows)
            _assert_database(
                client,
                database=database,
                table=table,
                model_id=model_id,
                hook_count=hook_count,
                tp_size=args.tp,
            )

        _assert_http_parity(reference_responses, monitored_responses)

        from tests.compare_disk_vs_ch import compare, read_clickhouse

        ch_data, num_ch_rows = read_clickhouse(
            args.db_host,
            args.db_port,
            database=database,
            table=table,
        )
        passed, failed = compare(str(ref_dir), ch_data, num_ch_rows)
        assert failed == 0
        assert passed == expected_rows

        summary = {
            "model": _MODEL,
            "model_id": model_id,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "database": database,
            "table": table,
            "http_requests": len(monitored_responses),
            "clickhouse_rows": num_ch_rows,
            "reference_files": len(ref_files),
            "bitwise_equal": passed,
            "cold_range": [0, 80],
            "warm_prefix_cache_range": [64, 80],
        }
        (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        print("[online-storage] PASS", flush=True)
    finally:
        shutil.copy2(backup, qwen3_ref)
        table_exists = client.execute(
            "SELECT count() FROM system.tables WHERE database=%(database)s "
            "AND name=%(table)s",
            {"database": database, "table": table},
        )[0][0]
        if table_exists and not args.keep_table:
            client.execute(f"DROP TABLE IF EXISTS {fq_table} SYNC")
            print(f"[online-storage] removed test table {database}.{table}")
        print(f"[online-storage] artifacts={artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
