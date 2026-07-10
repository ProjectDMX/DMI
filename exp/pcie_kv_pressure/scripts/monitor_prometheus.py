#!/usr/bin/env python3
"""Scrape Prometheus endpoints and record selected samples as JSONL."""

from __future__ import annotations

import argparse
import re
import signal
import time
import urllib.request
from pathlib import Path
from typing import Any

from common import append_jsonl, event_base, load_manifest


METRIC_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+eE0-9.]+)$")
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
STOP = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="prometheus")
    p.add_argument("--url", action="append", default=[])
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--keep-prefix", action="append", default=[])
    p.add_argument("--timeout-s", type=float, default=5.0)
    return p.parse_args()


def stop_handler(*_: Any) -> None:
    global STOP
    STOP = True


def parse_metrics(text: str, keep_prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, label_text, value_text = match.groups()
        if keep_prefixes and not any(name.startswith(prefix) for prefix in keep_prefixes):
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        rows.append({"metric": name, "labels": dict(LABEL_RE.findall(label_text or "")), "value": value})
    return rows


def scrape(url: str, timeout_s: float, keep: tuple[str, ...]) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return "ok", parse_metrics(text, keep), None
    except Exception as exc:
        return "error", [], str(exc)


def main() -> None:
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    args = parse_args()
    manifest = load_manifest(args.run_dir)
    out = Path(args.out)
    keep = tuple(args.keep_prefix)
    urls = args.url
    if not urls:
        raise SystemExit("--url is required")

    while not STOP:
        for url in urls:
            status, samples, error = scrape(url, args.timeout_s, keep)
            row = event_base(manifest, source=args.source, event_type="metrics_sample")
            row.update({"url": url, "status": status, "sample_count": len(samples), "samples": samples})
            if error:
                row["error"] = error
            append_jsonl(out, row)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
