from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from monitoring import MonitoringConfig, MonitoringEngine


class _DummyFuture:
    def result(self, timeout: Any = None) -> None:  # pragma: no cover - interface shim
        return None


@dataclass
class _DummyHostEngine:
    calls: list[tuple[Any, ...]]

    def __init__(self) -> None:
        self.calls = []

    def submit(
        self,
        model_id: str,
        shard_rank: int,
        request_ids: list[list[str]],
        token_ranges: list[list[tuple[int, int]]],
        cache_dicts: list[dict[str, Any]],
    ) -> None:
        self.calls.append(
            (
                model_id,
                shard_rank,
                [list(x) for x in request_ids],
                [list(x) for x in token_ranges],
                [{k: v for k, v in d.items()} for d in cache_dicts],
            )
        )


class _DummyNativeBackend:
    def begin_step(self, step_id: int, phase_code: int) -> None:
        return None


def _build_engine_for_db_step_unit(monkeypatch) -> tuple[MonitoringEngine, _DummyHostEngine]:
    monkeypatch.setattr(
        "monitoring.engine._load_native_backend",
        lambda *args, **kwargs: _DummyNativeBackend(),
    )
    engine = MonitoringEngine(async_enabled=True, model_id="unit-test-model")
    host = _DummyHostEngine()

    # Unit-test mode: validate Python DB-step logic with a no-op native backend.
    engine._host_engine = host
    engine._host_engine_enabled = True
    engine._capture_enabled = True
    return engine, host


def _submit_step(
    engine: MonitoringEngine,
    cache_dict: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
) -> None:
    engine._register_db_step(cache_dict, input_ids, attention_mask, past_key_values)
    engine._submit_pending_db_step()


def test_request_id_reset_on_prefill_and_batch_change(monkeypatch) -> None:
    engine, host = _build_engine_for_db_step_unit(monkeypatch)

    cache_dict = {
        "final_logits": _DummyFuture(),
        # alias entry should be filtered out
        "h.final_logits": _DummyFuture(),
        # no result() -> should be filtered out
        "bad_entry": object(),
    }

    # Batch 0 prefill (size=4)
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.ones((4, 6), dtype=torch.long),
        attention_mask=torch.tensor(
            [[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
            dtype=torch.long,
        ),
        past_key_values=None,
    )

    # Batch 0 decode (size=4)
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.tensor([[10], [11], [12], [13]], dtype=torch.long),
        attention_mask=torch.ones((4, 1), dtype=torch.long),
        past_key_values=object(),
    )

    # Batch 1 prefill (size=2) with non-None past_key_values to verify
    # shape-based prefill fallback (seq_len>1).
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.ones((2, 5), dtype=torch.long),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.long),
        past_key_values=object(),
    )

    assert len(host.calls) == 3

    # call[0]: prefill batch 0
    _, _, reqs0, ranges0, cache0 = host.calls[0]
    assert reqs0 == [["0:0", "0:1", "0:2", "0:3"]]
    assert ranges0 == [[(0, 6), (0, 5), (0, 4), (0, 3)]]
    assert list(cache0[0].keys()) == ["final_logits"]

    # call[1]: decode batch 0
    _, _, reqs1, ranges1, _ = host.calls[1]
    assert reqs1 == [["0:0", "0:1", "0:2", "0:3"]]
    assert ranges1 == [[(6, 7), (5, 6), (4, 5), (3, 4)]]

    # call[2]: prefill batch 1 must reset IDs and starts.
    _, _, reqs2, ranges2, _ = host.calls[2]
    assert reqs2 == [["1:0", "1:1"]]
    assert ranges2 == [[(0, 4), (0, 2)]]


def test_request_id_eos_finished_stops_decode_growth(monkeypatch) -> None:
    engine, host = _build_engine_for_db_step_unit(monkeypatch)
    cfg = MonitoringConfig()
    cfg.eos_token_id = 99
    cfg.pad_token_id = 0
    engine.config = cfg

    cache_dict = {"final_logits": _DummyFuture()}

    # Prefill
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.long),
        past_key_values=None,
    )
    # Decode 1: first request sees EOS token in decode input; it should be marked finished.
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.tensor([[99], [42]], dtype=torch.long),
        attention_mask=torch.ones((2, 1), dtype=torch.long),
        past_key_values=object(),
    )
    # Decode 2: finished request must no longer advance.
    _submit_step(
        engine,
        cache_dict,
        input_ids=torch.tensor([[99], [43]], dtype=torch.long),
        attention_mask=torch.ones((2, 1), dtype=torch.long),
        past_key_values=object(),
    )

    assert len(host.calls) == 3

    _, _, _, prefill_ranges, _ = host.calls[0]
    _, _, _, decode1_ranges, _ = host.calls[1]
    _, _, _, decode2_ranges, _ = host.calls[2]

    assert prefill_ranges == [[(0, 3), (0, 2)]]
    # First request sees EOS token in decode input, so it no longer advances.
    assert decode1_ranges == [[(3, 3), (2, 3)]]
    assert decode2_ranges == [[(3, 3), (3, 4)]]
