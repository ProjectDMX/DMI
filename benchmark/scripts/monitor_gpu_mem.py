import argparse
import csv
import signal
import subprocess
import time


_running = True


def _handle_signal(_signum, _frame):
    global _running
    _running = False


def _query_gpu(gpu_id: int):
    cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=timestamp,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    # Format: "2025/01/26 12:34:56.789, 123, 24576"
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 3:
        raise ValueError(f"unexpected nvidia-smi output: {out}")
    return parts[0], float(parts[1]), float(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "epoch_s", "gpu_id", "memory_used_mb", "memory_total_mb"])

        while _running:
            try:
                ts, used_mb, total_mb = _query_gpu(args.gpu_id)
            except Exception:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                used_mb = -1.0
                total_mb = -1.0
            writer.writerow([ts, time.time(), args.gpu_id, used_mb, total_mb])
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
