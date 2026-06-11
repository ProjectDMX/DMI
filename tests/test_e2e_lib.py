"""CPU unit coverage for the shared E2E lib + matrix expansion (plan §7, §8).

These are pure-CPU (torch-only) — no CUDA / ClickHouse / vLLM / weights — and
guard the de-duplicated comparison/align/report logic plus the matrix's cell
expansion and env translation (exercised via ``--dry-run`` internals).
"""
from __future__ import annotations

import json

import pytest
import torch

from tests.lib import align, compare
from tests.lib.compare import Check
from tests.lib.report import (
    CellResult,
    checks_from_legacy_result,
    human_table,
    read_jsonl,
    write_jsonl,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# compare.py — the four standards
# ---------------------------------------------------------------------------


class TestCompareStandards:
    def test_bitwise_equal_records_zero_drift(self):
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        c = compare.bitwise(a, a.clone(), "x")
        assert c.passed and c.max_abs == 0.0 and c.mean_abs == 0.0
        assert c.first_diff_pos is None

    def test_bitwise_detects_diff_and_reports_first_pos(self):
        a = torch.arange(6, dtype=torch.float32)
        b = a.clone()
        b[2] = 99.0
        c = compare.bitwise(a, b, "x")
        assert not c.passed
        assert c.first_diff_pos == 2
        assert c.max_abs == pytest.approx(99.0 - 2.0)

    def test_bitwise_shape_mismatch_fails_closed(self):
        c = compare.bitwise(torch.zeros(4), torch.zeros(5), "x")
        assert not c.passed and "shape mismatch" in c.detail

    def test_bitwise_dtype_mismatch_fails_closed(self):
        c = compare.bitwise(torch.zeros(4, dtype=torch.float32),
                            torch.zeros(4, dtype=torch.float16), "x")
        assert not c.passed and "dtype mismatch" in c.detail

    def test_allclose_passes_within_tol_but_records_drift(self):
        a = torch.zeros(8)
        b = a + 1e-4
        c = compare.allclose(a, b, "x", atol=1e-3)
        assert c.passed
        # drift recorded even on a pass
        assert c.max_abs == pytest.approx(1e-4, abs=1e-9)
        assert c.mean_abs == pytest.approx(1e-4, abs=1e-9)

    def test_allclose_fails_outside_tol(self):
        a = torch.zeros(8)
        b = a + 1.0
        c = compare.allclose(a, b, "x", atol=1e-3)
        assert not c.passed and c.first_diff_pos == 0

    def test_transport_bitwise_is_exact(self):
        a = torch.randn(4, 4)
        assert compare.transport_bitwise(a, a.clone()).passed
        assert not compare.transport_bitwise(a, a + 1e-6).passed

    def test_row_count_ok(self):
        counts = {f"blocks.hook_{i}": 5 for i in range(12)}
        counts["final_logits"] = 3
        c = compare.row_count(counts)
        assert c.passed, c.detail

    def test_row_count_uneven_fails(self):
        counts = {f"blocks.hook_{i}": 5 for i in range(12)}
        counts["blocks.hook_0"] = 4
        counts["final_logits"] = 3
        c = compare.row_count(counts)
        assert not c.passed and "uneven" in c.detail

    def test_row_count_missing_final_logits_fails(self):
        counts = {f"blocks.hook_{i}": 5 for i in range(12)}
        c = compare.row_count(counts)
        assert not c.passed and "final_logits" in c.detail

    def test_row_count_too_few_types_fails(self):
        c = compare.row_count({"blocks.hook_0": 1, "final_logits": 1})
        assert not c.passed

    def test_compare_tensors_dispatch_and_unknown(self):
        a = torch.zeros(3)
        assert compare.compare_tensors(a, a.clone(), "bitwise").passed
        with pytest.raises(ValueError):
            compare.compare_tensors(a, a, "row_count")

    def test_bytes_identical_neg_zero(self):
        # -0.0 and 0.0 are equal numerically but differ bitwise.
        a = torch.tensor([0.0])
        b = torch.tensor([-0.0])
        assert torch.equal(a, b)              # numerically equal
        assert not compare.bytes_identical(a, b)  # but not byte-identical


# ---------------------------------------------------------------------------
# align.py
# ---------------------------------------------------------------------------


class TestAlign:
    def test_parse_request_id(self):
        assert align.parse_request_id("3:7") == (3, 7)
        with pytest.raises(ValueError):
            align.parse_request_id("nope")

    def test_normalize_request_id_strips_vllm_suffix(self):
        assert align.normalize_request_id("12:0-deadbeef") == "12:0"
        assert align.normalize_request_id("12:0") == "12:0"

    def test_strip_left_pad(self):
        ids = torch.tensor([0, 0, 5, 6, 7])
        attn = torch.tensor([0, 0, 1, 1, 1])
        assert torch.equal(align.strip_left_pad(ids, attn), torch.tensor([5, 6, 7]))

    def test_strip_left_pad_all_padding(self):
        ids = torch.tensor([0, 0])
        attn = torch.tensor([0, 0])
        assert align.strip_left_pad(ids, attn).numel() == 0

    def test_trim_eos_drops_and_keeps(self):
        ids = torch.tensor([1, 2, 9, 3])
        assert torch.equal(align.trim_eos(ids, 9), torch.tensor([1, 2]))
        assert torch.equal(align.trim_eos(ids, 9, keep_eos=True), torch.tensor([1, 2, 9]))

    def test_trim_eos_absent_returns_all(self):
        ids = torch.tensor([1, 2, 3])
        assert torch.equal(align.trim_eos(ids, 9), ids)

    def test_align_to_min_len(self):
        a = torch.arange(5)
        b = torch.arange(3)
        ra, rb = align.align_to_min_len(a, b)
        assert ra.numel() == rb.numel() == 3

    def test_logits_align_skip(self):
        # skip = max(0, db_len - ref_len - 1): the DB keeps every position
        # while generate() yields gen-1 rows, so the head offset is the
        # prefill span. Never negative when ref is longer than db.
        assert align.logits_align_skip(db_len=9, ref_len=4) == 4
        assert align.logits_align_skip(db_len=2, ref_len=5) == 0


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------


class TestReport:
    def test_cellresult_finalize_and_record(self):
        cr = CellResult("vllm", "qwen3", "eager", "transport_bitwise", "vllm-full", tp=1)
        cr.checks = [Check("token_ids", True), Check("layer.0.q", True, max_abs=0.0)]
        cr.finalize()
        rec = cr.to_record()
        assert rec["passed"] is True
        assert rec["backend"] == "vllm" and rec["hook_selection"] == "vllm-full"
        assert len(rec["checks"]) == 2
        assert rec["checks"][1]["max_abs"] == 0.0

    def test_cellresult_fails_when_any_check_fails(self):
        cr = CellResult("hf", "gpt2", "eager", "bitwise", "vllm-full")
        cr.checks = [Check("a", True), Check("b", False, detail="boom")]
        cr.finalize()
        assert cr.passed is False

    def test_cellresult_no_checks_is_not_passed(self):
        cr = CellResult("hf", "gpt2", "eager", "bitwise", "vllm-full").finalize()
        assert cr.passed is False

    def test_cellresult_error_sets_failed(self):
        cr = CellResult("hf", "gpt2", "eager", "bitwise", "vllm-full")
        cr.error = "runner crashed"
        cr.checks = [Check("a", True)]
        cr.finalize()
        assert cr.passed is False
        assert cr.to_record()["error"] == "runner crashed"

    def test_legacy_result_adapter(self):
        legacy = {"tests": [
            {"name": "rows_found", "passed": True, "detail": "10 rows"},
            {"name": "x", "passed": False, "detail": "max_abs=1e-3"},
        ]}
        checks = checks_from_legacy_result(legacy)
        assert [c.name for c in checks] == ["rows_found", "x"]
        assert checks[0].passed and not checks[1].passed

    def test_jsonl_roundtrip(self, tmp_path):
        crs = []
        for passed in (True, False):
            cr = CellResult("vllm", "gpt2", "eager", "row_count", "vllm-full")
            cr.checks = [Check("c", passed)]
            crs.append(cr.finalize())
        out = tmp_path / "e2e.jsonl"
        write_jsonl(crs, str(out))
        recs = read_jsonl(str(out))
        assert len(recs) == 2
        assert recs[0]["passed"] is True and recs[1]["passed"] is False
        # each line is valid standalone JSON
        for line in out.read_text().splitlines():
            json.loads(line)

    def test_human_table_renders_counts(self):
        crs = [
            CellResult("vllm", "gpt2", "eager", "bitwise", "vllm-full").finalize(),
        ]
        crs[0].checks = [Check("c", True)]
        crs[0].finalize()
        table = human_table(crs)
        assert "backend" in table and "1/1 cells passed" in table


# ---------------------------------------------------------------------------
# e2e_matrix — cell expansion + env translation (the dry-run surface)
# ---------------------------------------------------------------------------


class TestMatrixExpansion:
    def _args(self, **over):
        from tests.e2e_matrix import build_parser
        argv = []
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return build_parser().parse_args(argv)

    def test_cartesian_product_count(self):
        from tests.e2e_matrix import build_cells
        args = self._args(backend="hf,vllm", model="gpt2,qwen3",
                          mode="eager,cuda_graph", standard="row_count")
        cells = build_cells(args)
        assert len(cells) == 2 * 2 * 2 * 1

    def test_env_translates_hook_selection(self):
        from tests.e2e_matrix import build_cells, cell_env
        args = self._args(backend="vllm", model="qwen3", mode="cuda_graph",
                          standard="transport_bitwise", hooks="q")
        cell = build_cells(args)[0]
        env = cell_env(cell, args, base={})
        # public -> internal contract (plan §2)
        assert env["E2E_HOOK_SELECTION"] == "q"
        assert env["DMX_HOOK_SELECTION"] == "q"
        # cuda_graph -> not eager
        assert env["E2E_ENFORCE_EAGER"] == "0"
        assert env["E2E_CUDA_GRAPHS"] == "1"
        assert env["E2E_MODEL"] == "qwen3"

    def test_eager_sets_enforce_eager(self):
        from tests.e2e_matrix import build_cells, cell_env
        args = self._args(backend="hf", model="gpt2", mode="eager", standard="bitwise")
        env = cell_env(build_cells(args)[0], args, base={})
        assert env["E2E_ENFORCE_EAGER"] == "1"
        assert env["E2E_CUDA_GRAPHS"] == "0"

    def test_plan_vllm_identical_dispatch(self):
        from tests.e2e_matrix import build_cells, plan_cell
        args = self._args(backend="vllm", model="qwen3", mode="eager",
                          standard="transport_bitwise", hooks="vllm-full")
        steps, comparator, _result = plan_cell(build_cells(args)[0], "/run")
        labels = [s.label for s in steps]
        assert labels == ["enable_ref_hooks", "vllm_ref", "vllm_monitored", "compare"]
        assert comparator == "tests.vllm_identical_comparator"

    def test_plan_vllm_rowcount_dispatch(self):
        from tests.e2e_matrix import build_cells, plan_cell
        args = self._args(backend="vllm", model="gpt2", mode="eager",
                          standard="row_count")
        steps, comparator, _ = plan_cell(build_cells(args)[0], "/run")
        assert [s.label for s in steps] == ["vllm_monitored", "compare"]
        assert comparator == "tests.vllm_rowcnt_comparator"

    def test_plan_hf_dispatch(self):
        from tests.e2e_matrix import build_cells, plan_cell
        args = self._args(backend="hf", model="gpt2", mode="eager", standard="allclose")
        steps, comparator, _ = plan_cell(build_cells(args)[0], "/run")
        assert [s.label for s in steps] == ["hf_ref", "hf_monitored", "compare"]
        assert comparator == "tests.hf_comparator"

    def test_unknown_backend_raises(self):
        from tests.e2e_matrix import Cell, plan_cell
        with pytest.raises(ValueError):
            plan_cell(Cell("nope", "gpt2", "eager", "bitwise", "vllm-full"), "/run")

    def test_main_dry_run_no_side_effects(self, capsys):
        from tests.e2e_matrix import main
        rc = main(["--backend", "hf,vllm", "--model", "gpt2",
                   "--standard", "row_count", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 cell(s) planned" in out

    def test_main_empty_axis_returns_2(self):
        from tests.e2e_matrix import main
        assert main(["--backend", "", "--dry-run"]) == 2


class TestWrapperTranslation:
    """matrix_argv_from_env / run_single — the thin-wrapper surface (plan §5)."""

    def test_env_to_argv_defaults(self):
        from tests.e2e_matrix import matrix_argv_from_env, build_parser, build_cells
        argv = matrix_argv_from_env("vllm", "bitwise", env={})
        cell = build_cells(build_parser().parse_args(argv))[0]
        assert cell.backend == "vllm" and cell.standard == "bitwise"
        assert cell.model == "gpt2" and cell.mode == "eager"
        assert cell.hooks == "vllm-full" and cell.ring_mb == 4096

    def test_env_enforce_eager_maps_mode(self):
        from tests.e2e_matrix import matrix_argv_from_env, build_parser, build_cells
        env = {"E2E_ENFORCE_EAGER": "0", "E2E_MODEL": "qwen3"}
        cell = build_cells(build_parser().parse_args(
            matrix_argv_from_env("vllm", "row_count", env=env)))[0]
        assert cell.mode == "cuda_graph" and cell.model == "qwen3"

    def test_explicit_mode_overrides_enforce_eager(self):
        from tests.e2e_matrix import matrix_argv_from_env, build_parser, build_cells
        # HF cuda-graph wrapper forces mode even though E2E_ENFORCE_EAGER=1.
        env = {"E2E_ENFORCE_EAGER": "1"}
        cell = build_cells(build_parser().parse_args(
            matrix_argv_from_env("hf", "allclose", mode="cuda_graph", env=env)))[0]
        assert cell.mode == "cuda_graph"

    def test_hook_selection_precedence(self):
        from tests.e2e_matrix import matrix_argv_from_env
        # public E2E_HOOK_SELECTION wins over internal DMX_HOOK_SELECTION
        argv = matrix_argv_from_env("vllm", "bitwise", env={
            "E2E_HOOK_SELECTION": "q", "DMX_HOOK_SELECTION": "k"})
        assert argv[argv.index("--hooks") + 1] == "q"
        # falls back to DMX_HOOK_SELECTION when public unset
        argv = matrix_argv_from_env("vllm", "bitwise", env={"DMX_HOOK_SELECTION": "k"})
        assert argv[argv.index("--hooks") + 1] == "k"

    def test_default_tolerance_passthrough(self):
        from tests.e2e_matrix import matrix_argv_from_env
        argv = matrix_argv_from_env("hf", "allclose", mode="cuda_graph",
                                    default_tolerance="0.5", env={})
        assert argv[argv.index("--tolerance") + 1] == "0.5"
        # explicit env overrides the default
        argv = matrix_argv_from_env("hf", "allclose", default_tolerance="0.5",
                                    env={"E2E_TOLERANCE": "0.01"})
        assert argv[argv.index("--tolerance") + 1] == "0.01"

    def test_run_single_rejects_multi_cell(self, monkeypatch):
        # matrix_argv_from_env always yields one cell; guard the invariant.
        from tests import e2e_matrix
        args = e2e_matrix.build_parser().parse_args(
            ["--backend", "hf,vllm", "--standard", "row_count"])
        monkeypatch.setattr(e2e_matrix, "run_cell", lambda *a, **k: None)
        with pytest.raises(ValueError, match="exactly 1 cell"):
            # build_cells gives 2 -> run_single must refuse
            cells = e2e_matrix.build_cells(args)
            assert len(cells) == 2
            e2e_matrix.run_single(["--backend", "hf,vllm", "--standard", "row_count"])
