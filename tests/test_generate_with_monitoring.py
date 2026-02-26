import inspect
import time

import pytest
import torch

from transformers import GPT2Config
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2LMHeadModel

from monitoring import MonitoringConfig, MonitoringEngine
from monitoring.config import HookSelection
from monitoring.generate import generate_with_monitoring
from monitoring.task import BackendFuture


def _build_small_lm() -> HookedGPT2LMHeadModel:
    config = GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    config.eos_token_id = 0
    config.bos_token_id = 1
    config.pad_token_id = 0
    return HookedGPT2LMHeadModel(config).eval()


def test_generate_with_monitoring_calls_run_with_cache():
    model = _build_small_lm()
    cfg = MonitoringConfig(hooks=HookSelection(mode="custom", include=["final_logits"]))
    engine = MonitoringEngine(async_enabled=False, config=cfg)
    model.monitoring_engine = engine

    calls = {"count": 0}
    orig_run_with_cache = model.run_with_cache

    def wrapped_run_with_cache(*args, **kwargs):
        calls["count"] += 1
        return orig_run_with_cache(*args, **kwargs)

    model.run_with_cache = wrapped_run_with_cache  # type: ignore[assignment]

    input_ids = torch.randint(0, model.config.vocab_size, (1, 3))
    attention_mask = torch.ones_like(input_ids)
    output_ids = generate_with_monitoring(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
    )

    assert output_ids.shape[1] == input_ids.shape[1] + 2
    assert calls["count"] >= 1
    assert getattr(model, "_monitoring_forward_wrapper", None) is model.forward


def test_generate_with_monitoring_preserves_forward_signature():
    model = _build_small_lm()
    orig_sig = inspect.signature(model.forward)

    _ = generate_with_monitoring(
        model,
        input_ids=torch.randint(0, model.config.vocab_size, (1, 2)),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
    )

    new_sig = inspect.signature(model.forward)
    assert "input_ids" in new_sig.parameters
    assert orig_sig == new_sig


def _wait_native_drain_without_resolve(engine: MonitoringEngine, timeout_s: float = 120.0) -> None:
    backend = getattr(engine, "_native_backend", None)
    assert backend is not None
    deadline = time.perf_counter() + timeout_s
    last_dbg = None
    while time.perf_counter() < deadline:
        last_dbg = backend.debug_state()
        if (
            int(last_dbg.get("pending_tasks", -1)) == 0
            and int(last_dbg.get("queue_size", -1)) == 0
            and int(last_dbg.get("open_steps", -1)) == 0
            and int(last_dbg.get("sealed_steps", -1)) == 0
        ):
            return
        time.sleep(0.05)
    pytest.fail(f"native backend did not drain without resolve_all: {last_dbg}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generate_and_forward_collect_cpp_futures_and_consume_results():
    model = _build_small_lm().to("cuda").eval()
    cfg = MonitoringConfig(hooks=HookSelection(mode="full"))
    engine = MonitoringEngine(async_enabled=True, config=cfg)
    model.monitoring_engine = engine
    engine.prepare_for_model(model)

    all_step_futures = []
    orig_run_with_cache = model.run_with_cache

    def wrapped_run_with_cache(*args, **kwargs):
        model_out, cache = orig_run_with_cache(*args, **kwargs)
        step_futures = [v for v in cache.values() if hasattr(v, "result")]
        if step_futures:
            all_step_futures.append(step_futures)
        return model_out, cache

    model.run_with_cache = wrapped_run_with_cache  # type: ignore[assignment]

    input_ids = torch.randint(0, model.config.vocab_size, (1, 4), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    try:
        out_ids = generate_with_monitoring(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=model.config.pad_token_id,
        )
        assert out_ids.shape[1] == input_ids.shape[1] + 4

        _ = model(input_ids=input_ids, attention_mask=attention_mask)

        assert all_step_futures, "expected futures captured from monitored run_with_cache calls"

        consumed = 0
        for step_futures in all_step_futures:
            for future in step_futures:
                if isinstance(future, BackendFuture):
                    tensor = future.result(30.0, True)
                else:
                    tensor = future.result(timeout=30.0)
                assert isinstance(tensor, torch.Tensor)
                consumed += 1

        assert consumed > 0
        _wait_native_drain_without_resolve(engine)
    finally:
        engine.close()
