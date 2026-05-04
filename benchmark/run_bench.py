import argparse
import json
import os
import shlex
import subprocess
import sys
import time


def _run_script(script: str, args: list[str], env: dict[str, str] | None = None) -> dict:
    import tempfile
    # Use temp file to capture JSON result
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name
    try:
        cmd = [sys.executable, script] + args + ["--json-out", tmp_path]
        subprocess.run(cmd, check=True, env=env)
        with open(tmp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _detect_clickhouse_pid() -> str:
    try:
        out = subprocess.check_output(["pidof", "-s", "clickhouse-server"], text=True).strip()
    except Exception:
        out = ""
    return out


def _start_monitors(
    gpu_csv: str,
    cpu_csv: str,
    interval: float,
    gpu_id: int,
    extra_pids: str,
) -> list[subprocess.Popen[str]]:
    return [
        subprocess.Popen(
            [
                sys.executable,
                "benchmark/scripts/monitor_gpu_mem.py",
                "--gpu-id",
                str(gpu_id),
                "--interval",
                str(interval),
                "--output",
                gpu_csv,
            ]
        ),
        subprocess.Popen(
            [
                sys.executable,
                "benchmark/scripts/monitor_cpu_mem.py",
                "--pid",
                str(os.getpid()),
                "--interval",
                str(interval),
                "--output",
                cpu_csv,
                "--include-children",
                "--extra-pids",
                extra_pids,
            ]
        ),
    ]


def _stop_monitors(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _now_epoch() -> float:
    return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF vs monitoring benchmarks")
    parser.add_argument("--prompts", default="benchmark/data/prompts.txt")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--out-dir", default="benchmark/results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-db", action="store_true", help="Disable host_engine DB submission.")
    parser.add_argument("--monitor-mem", action="store_true")
    parser.add_argument("--mem-interval", type=float, default=0.1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-extra-pids", default="")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument(
        "--monitor-extra-args",
        default="",
        help="Extra CLI args passed through to benchmark/scripts/hf_monitoring_generate.py",
    )
    parser.add_argument(
        "--no-clickhouse-pid",
        action="store_true",
        help="Do not auto-include clickhouse-server PID in CPU RSS sampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    summary_json = os.path.join(args.out_dir, f"summary_{tag}.json")
    gpu_csv_hf = os.path.join(args.out_dir, f"gpu_mem_hf_{tag}.csv")
    cpu_csv_hf = os.path.join(args.out_dir, f"cpu_mem_hf_{tag}.csv")
    gpu_csv_mon = os.path.join(args.out_dir, f"gpu_mem_monitoring_{tag}.csv")
    cpu_csv_mon = os.path.join(args.out_dir, f"cpu_mem_monitoring_{tag}.csv")

    common_args = [
        "--prompts",
        args.prompts,
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.do_sample:
        common_args.append("--do-sample")
    monitor_extra_args = shlex.split(args.monitor_extra_args) if args.monitor_extra_args else []

    base_env = os.environ.copy()
    if args.nvtx:
        base_env["BENCH_NVTX"] = "1"

    if args.monitor_mem:
        procs = _start_monitors(
            gpu_csv_hf,
            cpu_csv_hf,
            args.mem_interval,
            args.gpu_id,
            args.cpu_extra_pids,
        )
        try:
            hf_start_ts = _now_epoch()
            hf_result = _run_script("benchmark/scripts/hf_generate.py", common_args, env=base_env)
            hf_end_ts = _now_epoch()
        finally:
            _stop_monitors(procs)
    else:
        hf_start_ts = _now_epoch()
        hf_result = _run_script("benchmark/scripts/hf_generate.py", common_args, env=base_env)
        hf_end_ts = _now_epoch()

    # Run monitoring with DB (unless --no-db is specified)
    mon_result = None
    mon_start_ts = None
    mon_end_ts = None
    if not args.no_db:
        mon_args = common_args[:] + monitor_extra_args
        if args.monitor_mem:
            extra_pids = args.cpu_extra_pids
            if not args.no_clickhouse_pid:
                clickhouse_pid = _detect_clickhouse_pid()
                if clickhouse_pid:
                    extra_pids = f"{extra_pids},{clickhouse_pid}" if extra_pids else clickhouse_pid
            procs = _start_monitors(
                gpu_csv_mon,
                cpu_csv_mon,
                args.mem_interval,
                args.gpu_id,
                extra_pids,
            )
        try:
            mon_start_ts = _now_epoch()
            mon_result = _run_script("benchmark/scripts/hf_monitoring_generate.py", mon_args, env=base_env)
            mon_end_ts = _now_epoch()
        finally:
            _stop_monitors(procs)

    # Run monitoring with --no-db
    gpu_csv_mon_nodb = os.path.join(args.out_dir, f"gpu_mem_monitoring_nodb_{tag}.csv")
    cpu_csv_mon_nodb = os.path.join(args.out_dir, f"cpu_mem_monitoring_nodb_{tag}.csv")
    mon_nodb_args = common_args[:] + monitor_extra_args + ["--no-db"]
    if args.monitor_mem:
        procs = _start_monitors(
            gpu_csv_mon_nodb,
            cpu_csv_mon_nodb,
            args.mem_interval,
            args.gpu_id,
            args.cpu_extra_pids,  # No clickhouse PID needed for no-db
        )
        try:
            mon_nodb_start_ts = _now_epoch()
            mon_nodb_result = _run_script("benchmark/scripts/hf_monitoring_generate.py", mon_nodb_args, env=base_env)
            mon_nodb_end_ts = _now_epoch()
        finally:
            _stop_monitors(procs)
    else:
        mon_nodb_start_ts = _now_epoch()
        mon_nodb_result = _run_script("benchmark/scripts/hf_monitoring_generate.py", mon_nodb_args, env=base_env)
        mon_nodb_end_ts = _now_epoch()

    summary = {
        "tag": tag,
        "prompts": args.prompts,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "hf_start_ts": hf_start_ts,
        "hf_end_ts": hf_end_ts,
        "monitoring_start_ts": mon_start_ts,
        "monitoring_end_ts": mon_end_ts,
        "monitoring_nodb_start_ts": mon_nodb_start_ts,
        "monitoring_nodb_end_ts": mon_nodb_end_ts,
        "hf": hf_result,
        "monitoring": mon_result,
        "monitoring_nodb": mon_nodb_result,
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # print(f"hf result: {hf_json}")
    # print(f"monitoring result: {mon_json}")
    # print(f"summary: {summary_json}")
    # if args.monitor_mem:
    #     print(f"hf gpu mem: {gpu_csv_hf}")
    #     print(f"hf cpu mem: {cpu_csv_hf}")
    #     print(f"monitoring gpu mem: {gpu_csv_mon}")
    #     print(f"monitoring cpu mem: {cpu_csv_mon}")


if __name__ == "__main__":
    main()
