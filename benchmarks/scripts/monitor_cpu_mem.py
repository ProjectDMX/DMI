import argparse
import csv
import signal
import time
from typing import Iterable, List, Set


_running = True


def _handle_signal(_signum, _frame):
    global _running
    _running = False


def _read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except FileNotFoundError:
        return 0
    except PermissionError:
        return 0
    return 0


def _list_children_pids(root_pid: int) -> Set[int]:
    children: Set[int] = set()
    stack = [root_pid]
    while stack:
        current = stack.pop()
        try:
            with open(f"/proc/{current}/task/{current}/children", "r", encoding="utf-8") as handle:
                data = handle.read().strip()
        except FileNotFoundError:
            continue
        if not data:
            continue
        for token in data.split():
            try:
                pid = int(token)
            except ValueError:
                continue
            if pid not in children:
                children.add(pid)
                stack.append(pid)
    return children


def _parse_extra_pids(raw: str) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _iter_pids(root_pid: int, include_children: bool, extra_pids: Iterable[int]) -> List[int]:
    pids = {root_pid}
    if include_children:
        pids.update(_list_children_pids(root_pid))
    for pid in extra_pids:
        pids.add(pid)
    return sorted(pids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor CPU RSS for a process tree")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-children", action="store_true")
    parser.add_argument("--extra-pids", default="")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    extra_pids = _parse_extra_pids(args.extra_pids)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "epoch_s", "root_pid", "rss_mb", "pid_count"])

        while _running:
            pids = _iter_pids(args.pid, args.include_children, extra_pids)
            rss_kb = 0
            for pid in pids:
                rss_kb += _read_rss_kb(pid)
            rss_mb = rss_kb / 1024.0
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([ts, time.time(), args.pid, rss_mb, len(pids)])
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
