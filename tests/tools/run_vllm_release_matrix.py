#!/usr/bin/env python3
"""Run the version-pinned DMI-vLLM release matrix with retained evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDLE_CHECK = PROJECT_ROOT / "tests/tools/check_gpu_idle.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.process_group import run_process_group


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    command: tuple[str, ...]
    environment: dict[str, str]
    gpu_count: int = 0
    model_id: str | None = None
    phase: str = "focused"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Two physical GPU indices")
    parser.add_argument(
        "--phase",
        choices=("focused", "public", "storage", "all"),
        default="all",
    )
    parser.add_argument("--expected-vllm-version", default="0.27.1")
    parser.add_argument("--case-timeout", type=float, default=3600.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=2.0)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a prerequisite-blocked run in --artifact-dir.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def _selected_gpus(value: str) -> tuple[str, str]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError("--gpus must name exactly two distinct physical indices")
    if not all(item.isdigit() for item in values):
        raise ValueError("--gpus must contain integer physical indices")
    return values[0], values[1]


def _blackbox_case(
    model_key: str,
    model_id: str,
    *,
    tp_size: int,
    memory_utilization: float,
) -> MatrixCase:
    return MatrixCase(
        case_id=f"public-{model_key}-tp{tp_size}-eager-graph",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "tests/test_vllm_blackbox.py",
        ),
        environment={
            "DMI_BLACKBOX_MODEL": model_key,
            "DMI_BLACKBOX_TP_SIZE": str(tp_size),
            "DMI_BLACKBOX_GPU_MEMORY_UTILIZATION": str(memory_utilization),
            "DMI_BLACKBOX_CUDAGRAPH": "1",
            "DMI_BLACKBOX_SEED": "20260812",
            "DMI_BLACKBOX_GENERATED_CASES": "6",
        },
        gpu_count=tp_size,
        model_id=model_id,
        phase="public",
    )


def _storage_case(
    model_key: str,
    model_id: str,
    mode: str,
    tp_size: int,
    *,
    hook_selection: str = "vllm-full",
    ref_max_len: int = 8192,
) -> MatrixCase:
    return MatrixCase(
        case_id=f"storage-{model_key}-{mode}-tp{tp_size}",
        command=(
            "bash",
            "tests/tools/run_tp_compare_vllm.sh",
            model_key,
            mode,
            str(tp_size),
        ),
        environment={
            "DMX_HOOK_SELECTION": hook_selection,
            "E2E_REF_MAX_LEN": str(ref_max_len),
            "E2E_RING_PAYLOAD_MB": "512",
            "E2E_RING_PINNED_MB": "512",
            "E2E_GPU_MEM_UTIL": "0.85" if tp_size == 2 else "0.6",
        },
        gpu_count=tp_size,
        model_id=model_id,
        phase="storage",
    )


def build_cases(phase: str) -> list[MatrixCase]:
    focused = MatrixCase(
        case_id="focused-cpu-contracts",
        command=(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_vllm_version_compat.py",
            "tests/test_vllm_027_model_contracts.py",
            "tests/test_qwen2_p_inventory.py",
            "tests/test_vllm_blackbox_contract.py",
            "tests/test_vllm_comparator_contract.py",
            "tests/test_moe_v1_routing_hooks.py",
            "tests/test_vllm_request_order_fix.py",
            "tests/test_clickhouse_test_utils.py",
            "tests/test_gpu_idle_check.py",
            "tests/test_blackbox_case_generation.py",
            "tests/test_process_group.py",
            "tests/test_vllm_release_matrix.py",
        ),
        environment={},
    )
    public = [
        _blackbox_case("gpt2", "gpt2", tp_size=1, memory_utilization=0.5),
        _blackbox_case(
            "qwen2",
            "Qwen/Qwen2.5-0.5B-Instruct",
            tp_size=1,
            memory_utilization=0.5,
        ),
        _blackbox_case(
            "qwen3", "Qwen/Qwen3-0.6B", tp_size=1, memory_utilization=0.5
        ),
        _blackbox_case(
            "llama",
            "meta-llama/Llama-3.1-8B-Instruct",
            tp_size=2,
            memory_utilization=0.8,
        ),
        _blackbox_case(
            "qwen2_moe",
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            tp_size=2,
            memory_utilization=0.85,
        ),
    ]
    storage: list[MatrixCase] = []
    for model_key, model_id in (
        ("gpt2", "gpt2"),
        ("qwen3", "Qwen/Qwen3-0.6B"),
    ):
        for mode in ("eager", "cudagraph"):
            for tp_size in (1, 2):
                storage.append(_storage_case(model_key, model_id, mode, tp_size))
    for model_key, model_id in (
        ("llama", "meta-llama/Llama-3.1-8B-Instruct"),
        ("qwen2_moe", "Qwen/Qwen1.5-MoE-A2.7B-Chat"),
    ):
        for mode in ("eager", "cudagraph"):
            storage.append(
                _storage_case(
                    model_key,
                    model_id,
                    mode,
                    2,
                    hook_selection="resid_pre",
                    ref_max_len=512,
                )
            )

    if phase == "focused":
        return [focused]
    if phase == "public":
        return [focused, *public]
    if phase == "storage":
        return [focused, *storage]
    return [focused, *public, *storage]


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _model_in_cache(model_id: str) -> bool:
    if os.path.exists(model_id):
        return True
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub",
        )
    )
    return (cache_root / ("models--" + model_id.replace("/", "--"))).is_dir()


def _check_clickhouse() -> None:
    host = os.environ.get("DMX_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    with socket.create_connection((host, port), timeout=2.0):
        pass


def _runtime_identity(expected_version: str) -> dict[str, str]:
    import torch
    import vllm

    if vllm.__version__ != expected_version:
        raise RuntimeError(
            f"expected vLLM {expected_version}, found {vllm.__version__}"
        )
    return {
        "python": sys.version.split()[0],
        "vllm_version": vllm.__version__,
        "vllm_path": str(Path(vllm.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def _idle_command(gpus: tuple[str, ...], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(IDLE_CHECK),
        "--gpus",
        ",".join(gpus),
        "--samples",
        str(args.idle_samples),
        "--interval",
        str(args.idle_interval),
    ]


def _case_environment(case: MatrixCase, gpus: tuple[str, str]) -> dict[str, str]:
    selected = gpus[: case.gpu_count] if case.gpu_count else ()
    environment = dict(os.environ)
    environment.update(
        {
            "DMI_MATRIX_PYTHON": sys.executable,
            "HF_HUB_OFFLINE": "1",
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), environment.get("PATH", "")]
            ).rstrip(os.pathsep),
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        }
    )
    if selected:
        visible = ",".join(selected)
        environment["CUDA_VISIBLE_DEVICES"] = visible
        environment["E2E_GPUS"] = visible
    environment.update(case.environment)
    return environment


def _artifact_dir(path: Path | None, *, resume: bool) -> Path:
    if resume:
        if path is None:
            raise ValueError("--resume requires --artifact-dir")
        if not path.is_dir():
            raise ValueError(f"resume artifact directory does not exist: {path}")
        return path.resolve()
    if path is not None:
        path.mkdir(parents=True, exist_ok=False)
        return path.resolve()
    return Path(tempfile.mkdtemp(prefix="dmi_vllm_0271_matrix_"))


def _validate_passed_result(result: dict[str, Any]) -> None:
    case_id = result.get("case_id", "<unknown>")
    if result.get("status") != "passed" or result.get("returncode") != 0:
        raise ValueError(f"cannot resume past non-passing case {case_id}")
    log = result.get("log")
    if not isinstance(log, str) or not Path(log).is_file():
        raise ValueError(f"resume evidence log is missing for {case_id}")
    summary = result.get("pytest_summary")
    if summary is not None and any(
        int(summary.get(field, 0))
        for field in ("failures", "errors", "skipped")
    ):
        raise ValueError(f"resume pytest evidence is not clean for {case_id}")


def _resume_manifest(
    manifest_path: Path,
    *,
    cases: list[MatrixCase],
    root_commit: str,
    integration_commit: str,
    runtime: dict[str, str],
    gpus: tuple[str, str],
    phase: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read resume manifest: {error}") from error

    expected = {
        "schema_version": 1,
        "project_root": str(PROJECT_ROOT),
        "root_commit": root_commit,
        "vllm_integration_commit": integration_commit,
        "runtime": runtime,
        "physical_gpus": list(gpus),
        "phase": phase,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"resume manifest {field} mismatch: "
                f"expected {value!r}, found {manifest.get(field)!r}"
            )

    results = manifest.get("results")
    if not isinstance(results, list):
        raise ValueError("resume manifest results must be a list")
    if results and results[-1].get("status") == "blocked-prerequisite":
        results.pop()
    if len(results) > len(cases):
        raise ValueError("resume manifest contains more results than matrix cases")
    for index, result in enumerate(results):
        expected_case = cases[index]
        if result.get("case_id") != expected_case.case_id:
            raise ValueError(
                f"resume case order mismatch at {index + 1}: expected "
                f"{expected_case.case_id}, found {result.get('case_id')}"
            )
        _validate_passed_result(result)

    resumed_at = manifest.setdefault("resumed_at", [])
    if not isinstance(resumed_at, list):
        raise ValueError("resume manifest resumed_at must be a list")
    resumed_at.append(datetime.now(timezone.utc).isoformat())
    for field in ("finished_at", "passed", "failed", "blocked"):
        manifest.pop(field, None)
    return manifest


def _is_pytest_command(command: tuple[str, ...]) -> bool:
    return len(command) >= 3 and command[1:3] == ("-m", "pytest")


def _pytest_summary(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [
        suite
        for suite in root.iter("testsuite")
        if any(child.tag == "testcase" for child in suite)
    ]
    return {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


_FATAL_RUNTIME_LOG_MARKERS = (
    "WorkerProc hit an exception.",
    "EngineCore failed to start.",
)


def _fatal_runtime_log_markers(output: str) -> tuple[str, ...]:
    """Return runtime failures that must invalidate a zero process exit code."""

    return tuple(marker for marker in _FATAL_RUNTIME_LOG_MARKERS if marker in output)


def main() -> None:
    args = _parse_args()
    try:
        gpus = _selected_gpus(args.gpus)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = build_cases(args.phase)
    if args.list:
        for case in cases:
            print(f"{case.case_id}\tgpus={case.gpu_count}\t{' '.join(case.command)}")
        return

    root_status = _git(PROJECT_ROOT, "status", "--short")
    if root_status:
        raise SystemExit("release matrix requires a clean root worktree")
    integration = PROJECT_ROOT / "integration/vllm"
    integration_status = _git(integration, "status", "--short")
    if integration_status:
        raise SystemExit("release matrix requires a clean vLLM worktree")

    runtime = _runtime_identity(args.expected_vllm_version)
    if any(case.phase == "storage" for case in cases):
        _check_clickhouse()
    missing_models = sorted(
        {
            case.model_id
            for case in cases
            if case.model_id is not None and not _model_in_cache(case.model_id)
        }
    )
    if missing_models:
        raise SystemExit(f"model artifacts are not cached: {missing_models}")

    try:
        artifact_dir = _artifact_dir(args.artifact_dir, resume=args.resume)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest_path = artifact_dir / "manifest.json"
    root_commit = _git(PROJECT_ROOT, "rev-parse", "HEAD")
    integration_commit = _git(integration, "rev-parse", "HEAD")
    if args.resume:
        try:
            manifest = _resume_manifest(
                manifest_path,
                cases=cases,
                root_commit=root_commit,
                integration_commit=integration_commit,
                runtime=runtime,
                gpus=gpus,
                phase=args.phase,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "root_commit": root_commit,
            "vllm_integration_commit": integration_commit,
            "runtime": runtime,
            "physical_gpus": list(gpus),
            "phase": args.phase,
            "results": [],
        }

    blocked = False
    attempt = len(manifest.get("resumed_at", [])) + 1
    for index, case in enumerate(cases, start=1):
        if index <= len(manifest["results"]):
            print(
                f"[{index}/{len(cases)}] {case.case_id} (resume: already passed)",
                flush=True,
            )
            continue
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        result: dict[str, Any] = {
            **asdict(case),
            "attempt": attempt,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if blocked:
            result.update(status="blocked", reason="required focused gate failed")
            manifest["results"].append(result)
            continue
        if case.gpu_count:
            selected = gpus[: case.gpu_count]
            idle = subprocess.run(
                _idle_command(selected, args),
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if idle.returncode != 0:
                result.update(
                    status="blocked-prerequisite",
                    reason=idle.stdout + idle.stderr,
                )
                manifest["results"].append(result)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(result["reason"], file=sys.stderr)
                break

        environment = _case_environment(case, gpus)
        if case.phase == "public":
            raw_artifact_root = (
                artifact_dir / "raw" / case.case_id / f"attempt-{attempt}"
            )
            environment["DMI_BLACKBOX_ARTIFACT_DIR"] = str(raw_artifact_root)
            result["raw_artifact_root"] = str(raw_artifact_root)
        command = list(case.command)
        junit_path = None
        if _is_pytest_command(case.command):
            junit_path = artifact_dir / (
                f"{index:02d}-{case.case_id}-attempt-{attempt}.xml"
            )
            command.append(f"--junitxml={junit_path}")
        started = time.monotonic()
        fatal_markers: tuple[str, ...] = ()
        try:
            completed = run_process_group(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                timeout=args.case_timeout,
            )
            output = completed.stdout + completed.stderr
            returncode = completed.returncode
            status = "passed" if returncode == 0 else "failed"
            fatal_markers = _fatal_runtime_log_markers(output)
            if status == "passed" and fatal_markers:
                status = "failed-runtime-log"
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + (error.stderr or "")
            returncode = None
            status = "timeout"
        log_path = artifact_dir / (
            f"{index:02d}-{case.case_id}-attempt-{attempt}.log"
        )
        log_path.write_text(output)
        pytest_summary = (
            _pytest_summary(junit_path)
            if junit_path is not None and junit_path.is_file()
            else None
        )
        if (
            status == "passed"
            and pytest_summary is not None
            and pytest_summary["skipped"]
        ):
            status = "blocked-prerequisite"
        result.update(
            status=status,
            returncode=returncode,
            executed_command=command,
            pytest_summary=pytest_summary,
            fatal_runtime_log_markers=list(fatal_markers),
            duration_s=round(time.monotonic() - started, 3),
            log=str(log_path),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest["results"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(output[-4000:], flush=True)
        if status != "passed":
            if case.phase == "focused":
                blocked = True
            if args.fail_fast:
                break

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["passed"] = sum(
        result["status"] == "passed" for result in manifest["results"]
    )
    manifest["failed"] = sum(
        result["status"] in {"failed", "failed-runtime-log", "timeout"}
        for result in manifest["results"]
    )
    manifest["blocked"] = sum(
        result["status"].startswith("blocked") for result in manifest["results"]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"evidence: {manifest_path}")
    if manifest["failed"] or manifest["blocked"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
