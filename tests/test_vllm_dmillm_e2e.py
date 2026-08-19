"""End-to-end test for DMILLM (offline vLLM).

The vLLM engine is launched in a subprocess (tests.vllm_dmillm_runner) so the
pytest parent never initializes CUDA before the engine forks. The runner drives
DMILLM, reads each RequestOutput's per-request .dmi_internal back from
ClickHouse, and writes pass/fail to a result file we assert on here.

Requires CUDA, a reachable ClickHouse, vLLM, and the model in cache.

ClickHouse connection: DMX_DB_HOST / DMX_DB_PORT (default localhost:9000).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.vllm,
    pytest.mark.clickhouse,
    pytest.mark.e2e,
]


@pytest.mark.skipif(not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_vllm_dmillm_e2e():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = tempfile.mkdtemp(prefix="vllm_dmillm_")
    result_file = os.path.join(run_dir, "result.json")
    env = dict(os.environ)
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    try:
        r = subprocess.run(
            [sys.executable, "-m", "tests.vllm_dmillm_runner", "--result-file", result_file],
            env=env, capture_output=True, text=True, cwd=project_root,
        )
        if not os.path.exists(result_file):
            pytest.skip(
                f"DMILLM runner could not run (vLLM / ClickHouse / deps missing?):\n"
                f"{r.stderr[-1500:]}"
            )
        with open(result_file) as f:
            results = json.load(f)
        failed = [t for t in results["tests"] if not t["passed"]]
        assert not failed, f"failed checks: {failed}"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
