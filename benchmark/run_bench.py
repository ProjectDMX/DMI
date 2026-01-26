import argparse
import json
import os
import subprocess
import sys
import time


def _run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script] + args
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF vs monitoring benchmarks")
    parser.add_argument("--prompts", default="benchmark/data/prompts.txt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--out-dir", default="benchmark/results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-db", action="store_true", help="Disable host_engine DB submission.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    hf_json = os.path.join(args.out_dir, f"hf_{tag}.json")
    mon_json = os.path.join(args.out_dir, f"monitoring_{tag}.json")
    summary_json = os.path.join(args.out_dir, f"summary_{tag}.json")

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

    _run_script("benchmark/scripts/hf_generate.py", common_args + ["--json-out", hf_json])
    mon_args = common_args + ["--json-out", mon_json]
    if args.no_db:
        mon_args.append("--no-db")
    _run_script("benchmark/scripts/hf_monitoring_generate.py", mon_args)

    with open(hf_json, "r", encoding="utf-8") as handle:
        hf_result = json.load(handle)
    with open(mon_json, "r", encoding="utf-8") as handle:
        mon_result = json.load(handle)

    summary = {
        "tag": tag,
        "prompts": args.prompts,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "hf": hf_result,
        "monitoring": mon_result,
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"hf result: {hf_json}")
    print(f"monitoring result: {mon_json}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
