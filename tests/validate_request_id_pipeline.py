import argparse
import ast
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, LogitsProcessor
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import (
    ClickHouseClientConfig,
    EnqueuePolicy,
    HostEngineConfig,
    MonitoringConfig,
    MonitoringEngine,
    NativePartialSealConfig,
    OnClosedPolicy,
    OnFullPolicy,
    QueueConfig,
    StageConfig,
)
from monitoring.config import CaptureSchedule, HookSelection
from monitoring.generate import generate_with_monitoring


class _ForceFirstRequestEosLogitsProcessor(LogitsProcessor):
    """Force batch item 0 to emit EOS at decode time, to exercise finished-request path."""

    def __init__(self, eos_token_id: int, prompt_width: int) -> None:
        self.eos_token_id = int(eos_token_id)
        self.prompt_width = int(prompt_width)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Apply only on decode steps (after prefill token has been generated).
        if input_ids.shape[1] <= self.prompt_width:
            return scores
        if scores.shape[0] <= 0:
            return scores
        scores[0, :] = torch.finfo(scores.dtype).min
        scores[0, self.eos_token_id] = 0.0
        return scores


def _load_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _iter_batches(items: List[str], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield idx // batch_size, items[idx : idx + batch_size]


def _build_db_config(drop_existing: bool) -> ClickHouseClientConfig:
    cfg = ClickHouseClientConfig()
    cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    cfg.username = os.environ.get("DMX_DB_USER", "default")
    cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    cfg.database = os.environ.get("DMX_DB_DATABASE", "default")
    cfg.table = os.environ.get("DMX_DB_TABLE", "offload")
    cfg.secure = False
    cfg.client_side_compress = "none"
    cfg.client_settings = None
    cfg.create_database_if_missing = True
    cfg.drop_existing_database = drop_existing
    cfg.index_granularity = 8192
    return cfg


def _build_queue_config() -> QueueConfig:
    q = QueueConfig()
    q.min_batch_items = 1
    q.high_watermark_items = None
    return q


def _build_ingress_policy() -> EnqueuePolicy:
    p = EnqueuePolicy()
    p.block = False
    p.on_full = OnFullPolicy.RAISE
    p.on_closed = OnClosedPolicy.RAISE
    return p


def _build_host_config(db_cfg: ClickHouseClientConfig) -> HostEngineConfig:
    stage_one = StageConfig.process_future(parallelism=1, name="process_future")
    stage_two = StageConfig.clickhouse_insert(db_cfg, parallelism=1, name="clickhouse_insert")
    stage_one.input_queue = _build_queue_config()
    stage_two.input_queue = _build_queue_config()
    ingress = _build_ingress_policy()
    stage_one.ingress_policy = ingress
    stage_two.ingress_policy = ingress
    return HostEngineConfig(stages=[stage_one, stage_two])


def _run_clickhouse_query(db_cfg: ClickHouseClientConfig, query: str) -> str:
    cmd = [
        "clickhouse-client",
        "--host",
        db_cfg.host,
        "--port",
        str(db_cfg.port),
        "--user",
        db_cfg.username,
        "--database",
        db_cfg.database,
        "--query",
        query,
    ]
    if db_cfg.password:
        cmd.extend(["--password", db_cfg.password])
    try:
        out = subprocess.check_output(cmd, text=True)
        return out
    except FileNotFoundError as exc:
        raise RuntimeError(
            "clickhouse-client is not found. Please install it or run this check on a machine "
            "with ClickHouse client CLI."
        ) from exc


def _parse_ranges_tsv(tsv: str) -> Dict[str, List[Tuple[int, int]]]:
    rows = [line.strip() for line in tsv.splitlines() if line.strip()]
    if len(rows) <= 1:
        return {}
    data = defaultdict(list)
    for line in rows[1:]:
        req_id, start_s, end_s = line.split("\t")
        data[req_id].append((int(start_s), int(end_s)))
    for req_id in data:
        data[req_id].sort(key=lambda x: (x[0], x[1]))
    return dict(data)


def _parse_shapes_tsv(tsv: str) -> Dict[str, List[List[int]]]:
    rows = [line.strip() for line in tsv.splitlines() if line.strip()]
    if len(rows) <= 1:
        return {}
    data = defaultdict(list)
    for line in rows[1:]:
        req_id, shape_s = line.split("\t", 1)
        shape = ast.literal_eval(shape_s)
        if not isinstance(shape, list):
            raise AssertionError(f"{req_id}: invalid shape payload: {shape_s}")
        data[req_id].append([int(v) for v in shape])
    return dict(data)


def _effective_generated_len(tokens: List[int], eos_token_id: int, pad_token_id: int) -> int:
    stop_ids = {int(eos_token_id), int(pad_token_id)}
    for idx, token in enumerate(tokens):
        if int(token) in stop_ids:
            return idx + 1
    return len(tokens)


def _validate_ranges(
    actual: Dict[str, List[Tuple[int, int]]],
    expected_prefill_len: Dict[str, int],
    generated_tokens: Dict[str, List[int]],
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    expected_ids = set(expected_prefill_len.keys())
    actual_ids = set(actual.keys())
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise AssertionError(
            f"request_id mismatch: missing={missing[:8]} extra={extra[:8]}"
        )

    for req_id in sorted(expected_ids):
        ranges = actual[req_id]
        if not ranges:
            raise AssertionError(f"{req_id}: no ranges in DB")

        first_start, first_end = ranges[0]
        if first_start != 0:
            raise AssertionError(f"{req_id}: first start_token_idx is {first_start}, expected 0")
        first_len = first_end - first_start
        if first_len != int(expected_prefill_len[req_id]):
            raise AssertionError(
                f"{req_id}: prefill len mismatch, db={first_len}, expected={expected_prefill_len[req_id]}"
            )
        if first_end <= first_start:
            raise AssertionError(f"{req_id}: first range is invalid ({first_start}, {first_end})")

        prev_end = first_end
        for idx, (start_i, end_i) in enumerate(ranges[1:], start=1):
            if end_i <= start_i:
                raise AssertionError(f"{req_id}: invalid range at step {idx}: ({start_i}, {end_i})")
            if start_i != prev_end:
                raise AssertionError(
                    f"{req_id}: non-contiguous range at step {idx}: start={start_i}, prev_end={prev_end}"
                )
            prev_end = end_i

        effective_generated = _effective_generated_len(
            generated_tokens.get(req_id, []),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # HF decoder-only generate:
        # - token#1 comes from prefill forward logits
        # - remaining tokens come from decode forwards
        expected_decode = max(0, int(effective_generated) - 1)
        expected_rows = 1 + expected_decode
        if len(ranges) != expected_rows:
            raise AssertionError(
                f"{req_id}: row count mismatch, db={len(ranges)} expected={expected_rows} "
                f"(generated_effective={effective_generated}, decode_expected={expected_decode})"
            )


def _validate_shape_rank(
    shape_map: Dict[str, List[List[int]]],
    expected_request_ids: List[str],
    expected_rank: int,
    act_name: str,
) -> None:
    missing = [req_id for req_id in expected_request_ids if req_id not in shape_map]
    if missing:
        raise AssertionError(f"{act_name}: missing shape rows for request_ids={missing[:8]}")
    for req_id in expected_request_ids:
        for shape in shape_map[req_id]:
            if len(shape) != expected_rank:
                raise AssertionError(
                    f"{act_name}: request_id={req_id} unexpected rank={len(shape)} shape={shape}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate per-request request_id/token range correctness via full DB pipeline."
    )
    parser.add_argument("--prompts", default="benchmark/data/prompts_varlen_validation.txt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument(
        "--exercise-eos-path",
        action="store_true",
        help="Force request[0] in each batch to emit EOS during decode, to exercise finished-request path.",
    )
    parser.add_argument(
        "--with-attn-hook",
        action="store_true",
        help="Also validate one attention hook path (blocks.0.attn.hook_attn_scores).",
    )
    parser.add_argument(
        "--keep-existing-db",
        action="store_true",
        help="Do not drop/recreate DB before run (default drops for clean validation).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script.")

    os.environ.setdefault("MON_NATIVE_TO_CPU", "1")
    os.environ.setdefault("MON_NATIVE_CALLBACK", "1")
    os.environ.setdefault("MON_NATIVE_BUILDER", "1")
    os.environ.setdefault("MON_NATIVE_BATCH", "0")
    os.environ.setdefault("MON_NATIVE_PINNED", "1")
    os.environ.setdefault("MON_NATIVE_PINPOOL", "1")
    os.environ.setdefault("MON_NATIVE_HOST_COPY_THREADS", "4")
    os.environ.setdefault("MON_NATIVE_AUTOCLEAR", "0")

    prompts = _load_prompts(args.prompts)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    expected_prefill_len: Dict[str, int] = {}
    for batch_idx, batch_prompts in _iter_batches(prompts, args.batch_size):
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        lens = encoded["attention_mask"].sum(dim=1).tolist()
        for i, ln in enumerate(lens):
            expected_prefill_len[f"{batch_idx}:{i}"] = int(ln)
    all_request_ids = sorted(expected_prefill_len.keys())

    hook_list = ["final_logits"]
    if args.with_attn_hook:
        hook_list.append("blocks.0.attn.hook_attn_scores")

    cfg = MonitoringConfig(
        hooks=HookSelection(mode="custom", include=hook_list),
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
        native_partial_seal=NativePartialSealConfig(
            enabled=True,
            chunk_bytes=64 * 1024 * 1024,
            cap_enabled=True,
            cap_ratio=0.8,
            driver_guard_mb=1024,
        ),
    )
    # Enable decode-finished tracking in engine._register_db_step.
    cfg.eos_token_id = int(tokenizer.eos_token_id)
    cfg.pad_token_id = int(tokenizer.pad_token_id)

    db_cfg = _build_db_config(drop_existing=(not args.keep_existing_db))
    host_cfg = _build_host_config(db_cfg)

    engine = MonitoringEngine(
        async_enabled=True,
        config=cfg,
        model_id=args.model,
        db_config=host_cfg,
    )
    model = HookedGPT2LMHeadModel.from_pretrained(
        args.model,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    ).to(device).eval()
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    generated_tokens: Dict[str, List[int]] = {}
    try:
        with torch.no_grad():
            for batch_idx, batch_prompts in _iter_batches(prompts, args.batch_size):
                encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                prompt_width = int(input_ids.shape[1])
                logits_processor = None
                if args.exercise_eos_path:
                    logits_processor = [
                        _ForceFirstRequestEosLogitsProcessor(
                            eos_token_id=int(tokenizer.eos_token_id),
                            prompt_width=prompt_width,
                        )
                    ]
                output_ids = generate_with_monitoring(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processor,
                )
                batch_out = output_ids.detach().cpu()
                for i in range(int(batch_out.shape[0])):
                    req_id = f"{batch_idx}:{i}"
                    generated_tokens[req_id] = [int(v) for v in batch_out[i, prompt_width:].tolist()]
    finally:
        engine.close()

    q = (
        "SELECT request_id, start_token_idx, end_token_idx "
        f"FROM {db_cfg.database}.{db_cfg.table} "
        f"WHERE model_id = '{args.model}' AND act_name = 'final_logits' "
        "ORDER BY request_id, start_token_idx, end_token_idx "
        "FORMAT TabSeparatedWithNames"
    )
    tsv = _run_clickhouse_query(db_cfg, q)
    actual = _parse_ranges_tsv(tsv)
    _validate_ranges(
        actual,
        expected_prefill_len,
        generated_tokens,
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
    )

    q_shape_logits = (
        "SELECT request_id, shape "
        f"FROM {db_cfg.database}.{db_cfg.table} "
        f"WHERE model_id = '{args.model}' AND act_name = 'final_logits' "
        "ORDER BY request_id, start_token_idx, end_token_idx "
        "FORMAT TabSeparatedWithNames"
    )
    shape_logits = _parse_shapes_tsv(_run_clickhouse_query(db_cfg, q_shape_logits))
    _validate_shape_rank(shape_logits, all_request_ids, expected_rank=2, act_name="final_logits")

    if args.with_attn_hook:
        q_attn_ranges = (
            "SELECT request_id, start_token_idx, end_token_idx "
            f"FROM {db_cfg.database}.{db_cfg.table} "
            f"WHERE model_id = '{args.model}' AND act_name = 'blocks.attn.hook_attn_scores' "
            "ORDER BY request_id, start_token_idx, end_token_idx "
            "FORMAT TabSeparatedWithNames"
        )
        attn_ranges = _parse_ranges_tsv(_run_clickhouse_query(db_cfg, q_attn_ranges))
        if not attn_ranges:
            raise AssertionError("blocks.attn.hook_attn_scores has no DB rows")
        for req_id, req_rows in attn_ranges.items():
            for start_i, end_i in req_rows:
                if end_i <= start_i:
                    raise AssertionError(
                        f"blocks.attn.hook_attn_scores invalid range: request_id={req_id} "
                        f"({start_i}, {end_i})"
                    )

        q_attn_shape = (
            "SELECT request_id, shape "
            f"FROM {db_cfg.database}.{db_cfg.table} "
            f"WHERE model_id = '{args.model}' AND act_name = 'blocks.attn.hook_attn_scores' "
            "ORDER BY request_id, start_token_idx, end_token_idx "
            "FORMAT TabSeparatedWithNames"
        )
        shape_attn = _parse_shapes_tsv(_run_clickhouse_query(db_cfg, q_attn_shape))
        _validate_shape_rank(
            shape_attn,
            sorted(attn_ranges.keys()),
            expected_rank=3,
            act_name="blocks.attn.hook_attn_scores",
        )

    if args.exercise_eos_path:
        shortened = [
            req_id
            for req_id, toks in generated_tokens.items()
            if _effective_generated_len(
                toks,
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            ) < args.max_new_tokens
        ]
        if not shortened:
            raise AssertionError(
                "--exercise-eos-path enabled, but no request finished early (effective_generated < max_new_tokens)"
            )

    total_rows = sum(len(v) for v in actual.values())
    print(
        f"[PASS] validated {len(actual)} request_ids, {total_rows} rows, "
        f"batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}, "
        f"with_attn_hook={args.with_attn_hook}, exercise_eos_path={args.exercise_eos_path}"
    )


if __name__ == "__main__":
    main()
