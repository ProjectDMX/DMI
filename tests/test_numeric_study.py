"""Tests for the per-hook numeric-difference study (plan §9 / Phase 5).

Two layers, mirroring ``test_per_hook_isolation.py``:

1. **CPU unit tests** (always run): exercise the pure comparison / alert /
   report core (``compute_drift``, ``format_table``, serialization, threshold
   + standard selection) on small hand-built CPU tensors. These guard the
   study's verdict logic -- including its use of the shared ``tests.lib.compare``
   standards -- independently of any GPU rollout.

2. **GPU smoke** (marked ``numeric`` + ``gpu`` + ``slow``, opt-in): run the
   real study for a couple of observational hooks and assert the machinery
   produces serializable per-hook records and a self-consistent eager verdict.
"""
from __future__ import annotations

import json

import pytest

from tests._requirements import require_cuda
from tests.lib.compare import Check
from tests.numeric_study import (
    HookDrift,
    StudyResult,
    VocabDiff,
    compute_drift,
    cuda_graph_threshold,
    format_table,
    run_study,
    standard_for_mode,
)


# ---------------------------------------------------------------------------
# CPU unit tests for the comparison core
# ---------------------------------------------------------------------------


def _cap(logits):
    import torch

    t = torch.tensor(logits, dtype=torch.float32)
    return {"logits": t, "token_ids": t.argmax(dim=-1).to(torch.int64)}


@pytest.mark.cpu
class TestComputeDrift:
    def test_identical_eager_passes_bitwise(self):
        base = _cap([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]])
        d = compute_drift("q", base, base, mode="eager", threshold=0.15, topk=3)
        assert d.check is not None and d.check.name == "logits_bitwise"
        assert d.check.passed is True
        assert d.check.max_abs == 0.0
        assert d.first_diff_pos == -1
        assert d.token_ids_diverged is False
        assert d.topk_vocab_diffs == []
        assert d.alert is False
        assert d.n_positions == 2 and d.vocab_size == 3

    def test_eager_any_drift_alerts(self):
        base = _cap([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0]])
        mon = _cap([[5.0, 1.0, 0.0], [0.0, 4.0, 1.25]])  # tiny perturb, no argmax flip
        d = compute_drift("k", base, mon, mode="eager", threshold=0.15, topk=2)
        assert d.check.passed is False
        assert d.first_diff_pos == 1
        assert d.check.max_abs == pytest.approx(0.25, abs=1e-6)
        assert d.token_ids_diverged is False
        assert d.alert is True
        assert any("eager" in r for r in d.alert_reasons)
        # Top-k drift at the first differing position points at the perturbed id.
        assert d.topk_vocab_diffs[0].token_id == 2
        assert d.topk_vocab_diffs[0].abs_diff == pytest.approx(0.25, abs=1e-6)

    def test_cuda_graph_below_threshold_no_alert(self):
        base = _cap([[5.0, 1.0, 0.0]])
        mon = _cap([[5.0, 1.05, 0.0]])  # max_abs 0.05 < 0.15
        d = compute_drift("q", base, mon, mode="cuda_graph", threshold=0.15, topk=1)
        assert d.check.name == "logits_allclose"
        assert d.check.passed is True
        assert d.alert is False

    def test_cuda_graph_above_threshold_alerts(self):
        base = _cap([[5.0, 1.0, 0.0]])
        mon = _cap([[5.0, 1.5, 0.0]])  # max_abs 0.5 > 0.15, no argmax flip
        d = compute_drift("q", base, mon, mode="cuda_graph", threshold=0.15, topk=1)
        assert d.check.passed is False
        assert d.alert is True
        assert any("threshold" in r for r in d.alert_reasons)

    def test_argmax_flip_always_alerts_even_under_cuda_graph(self):
        base = _cap([[5.0, 1.0, 0.0]])
        mon = _cap([[1.0, 9.0, 0.0]])  # argmax 0 -> 1
        d = compute_drift("resid_pre", base, mon, mode="cuda_graph", threshold=100.0, topk=1)
        assert d.token_ids_diverged is True
        assert d.n_token_diff == 1
        assert d.alert is True
        assert any("token ids diverged" in r for r in d.alert_reasons)

    def test_shape_mismatch_alerts(self):
        import torch

        base = _cap([[1.0, 2.0, 3.0]])
        mon = {
            "logits": torch.zeros((1, 4), dtype=torch.float32),
            "token_ids": torch.zeros((1,), dtype=torch.int64),
        }
        d = compute_drift("q", base, mon, mode="eager", threshold=0.15, topk=1)
        assert d.shape_mismatch is True
        assert d.alert is True
        assert any("identity swap" in r for r in d.alert_reasons)

    def test_empty_capture_alerts(self):
        import torch

        empty = {
            "logits": torch.zeros((0, 3), dtype=torch.float32),
            "token_ids": torch.zeros((0,), dtype=torch.int64),
        }
        d = compute_drift("q", empty, empty, mode="eager", threshold=0.15, topk=1)
        assert d.error is not None
        assert d.alert is True

    def test_missing_logits_alerts(self):
        d = compute_drift("q", {"token_ids": None}, {"token_ids": None},
                          mode="eager", threshold=0.15, topk=1)
        assert d.error is not None
        assert d.alert is True


@pytest.mark.cpu
class TestSelectorsAndReport:
    def test_standard_for_mode(self):
        assert standard_for_mode("eager") == "bitwise"
        assert standard_for_mode("cuda_graph") == "allclose"

    def test_threshold_lookup_and_fallback(self):
        assert cuda_graph_threshold("qwen2_moe", "float16") == 0.25
        assert cuda_graph_threshold("nonesuch", "float64") == pytest.approx(0.15)

    def test_format_table_contains_hooks_and_verdict(self):
        result = StudyResult(
            framework="hf", model="qwen3", mode="eager", variant="p",
            dtype="float16", standard="bitwise", threshold=0.15, topk=2,
        )
        ok = HookDrift(hook="resid_pre", check=Check("logits_bitwise", True, max_abs=0.0))
        bad = HookDrift(
            hook="q",
            check=Check("logits_bitwise", False, max_abs=0.3, mean_abs=0.01, detail="max_abs=3e-01"),
            first_diff_pos=1, alert=True, alert_reasons=["non-bitwise drift in eager mode"],
            topk_vocab_diffs=[VocabDiff(7, 1.0, 1.3, 0.3)],
        )
        result.hooks.extend([ok, bad])
        table = format_table(result)
        assert "resid_pre" in table
        assert "ALERT" in table
        assert "tok 7" in table
        assert "1/2 hooks clean" in table

    def test_result_to_dict_is_json_serializable(self):
        result = StudyResult(
            framework="hf", model="qwen3", mode="cuda_graph", variant="compare",
            dtype="float16", standard="allclose", threshold=0.15, topk=1,
        )
        result.hooks.append(
            HookDrift(
                hook="q",
                check=Check("logits_allclose", False, max_abs=0.2, mean_abs=0.01),
                topk_vocab_diffs=[VocabDiff(1, 0.0, 0.2, 0.2)],
                alert=True, alert_reasons=["exceeds cuda-graph threshold"],
            )
        )
        payload = result.to_dict()
        s = json.dumps(payload)  # must not raise
        rt = json.loads(s)
        assert rt["any_alert"] is True
        assert rt["variant"] == "compare"
        assert rt["hooks"][0]["check"]["name"] == "logits_allclose"
        assert rt["hooks"][0]["topk_vocab_diffs"][0]["token_id"] == 1


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


@pytest.mark.numeric
@pytest.mark.gpu
@pytest.mark.slow
@require_cuda()
def test_numeric_study_hf_qwen3_eager_smoke(tmp_path):
    """Eager observational hooks must not drift the logits vs the unhooked
    baseline; the study reports per-hook records and the eager verdict is
    self-consistent (alert iff the bitwise check failed)."""
    hooks = ["resid_pre", "final_logits"]
    result = run_study(
        framework="hf", model="qwen3", mode="eager",
        hooks=hooks, out_dir=tmp_path, max_new_tokens=4,
    )

    # Always surface the table, even on pass.
    print("\n" + format_table(result))

    assert [h.hook for h in result.hooks] == hooks
    json.dumps(result.to_dict())  # serializable

    for h in result.hooks:
        assert h.error is None, f"{h.hook} rollout errored: {h.error}"
        assert h.check is not None
        # Eager verdict must be self-consistent with the bitwise outcome.
        assert h.alert == (not h.check.passed), (
            f"{h.hook}: alert={h.alert} but check.passed={h.check.passed}"
        )
    assert not result.any_alert, (
        "eager observational hooks drifted vs the unhooked baseline:\n"
        + format_table(result)
    )
