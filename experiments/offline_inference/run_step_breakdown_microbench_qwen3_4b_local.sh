#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/nengneng/miniconda3/envs/proj-dmx/bin/python}"
GPU="${GPU:-1}"
MODEL="${MODEL:-qwen3-4b}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/offline_inference/results/step_breakdown_qwen3_4b_local}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/${MODEL//\//_}_${RUN_TAG}"
PREFILL_TOKENS="${PREFILL_TOKENS:-128}"
DECODE_STEPS="${DECODE_STEPS:-10}"
ITERS="${ITERS:-1}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HOOK_SELECTION="${HOOK_SELECTION:-hidden-states,final_ln,logits}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"
RING_PAYLOAD_MB="${RING_PAYLOAD_MB:-10240}"
RING_PINNED_MB="${RING_PINNED_MB:-10240}"
RING_TASK_ENTRIES="${RING_TASK_ENTRIES:-131072}"

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${RESULTS_DIR}"

run_one() {
  local baseline="$1"
  local repeat_index="$2"
  shift 2
  local extra_args=("$@")
  echo "============================================================"
  echo "baseline=${baseline} bs=${BATCH_SIZE} repeat=${repeat_index}/${REPEATS} ${extra_args[*]:-}"
  echo "============================================================"
  PYTHON_BIN="${PYTHON_BIN}" bash experiments/offline_inference/run_step_breakdown_baseline.sh "${baseline}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --prefill-tokens "${PREFILL_TOKENS}" \
    --decode-steps "${DECODE_STEPS}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeat-index "${repeat_index}" \
    --results-dir "${RESULTS_DIR}" \
    ${LOCAL_FILES_ONLY} \
    --hook-selection "${HOOK_SELECTION}" \
    --proj-dmi-mode ring_null \
    --ring-payload-mb "${RING_PAYLOAD_MB}" \
    --ring-pinned-mb "${RING_PINNED_MB}" \
    --ring-task-entries "${RING_TASK_ENTRIES}" \
    --drain-flush-task-ratio 0.0 \
    --drain-flush-payload-ratio 0.0 \
    --drain-flush-timeout-us 1000 \
    --db-host localhost \
    --db-port 9000 \
    --db-user default \
    --db-database default \
    --db-table offload \
    "${extra_args[@]}"
}

for repeat_index in $(seq 1 "${REPEATS}"); do
  run_one hf_ideal "${repeat_index}"
  run_one hf_api "${repeat_index}"
  run_one torch_hooks "${repeat_index}"
  run_one proj_dmi_manual "${repeat_index}" --baseline-label proj_dmi_manual
  run_one proj_dmi_manual "${repeat_index}" --disable-compile --baseline-label proj_dmi_manual_eager
done

"${PYTHON_BIN}" - <<'PY' "${RESULTS_DIR}"
import csv
import json
import statistics
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
rows = []
for path in sorted(results_dir.glob("*.json")):
    with path.open() as f:
        data = json.load(f)
    decode = data.get("decode_last_step") or data.get("decode_1") or {}
    decode_seq = data.get("decode_sequence_total") or {}
    rows.append(
        {
            "baseline": data.get("baseline_label", data["baseline"]),
            "baseline_key": data["baseline"],
            "baseline_label": data.get("baseline_label", data["baseline"]),
            "repeat_index": int(data.get("repeat_index", 1)),
            "prefill_compute_ms": float(data["prefill"]["compute_ms"]),
            "prefill_total_ms": float(data["prefill"]["total_ms"]),
            "prefill_transfer_tail_ms": float(data["prefill"]["transfer_tail_ms"]),
            "decode_compute_ms": float(decode["compute_ms"]),
            "decode_total_ms": float(decode["total_ms"]),
            "decode_total_with_final_flush_ms": float(decode.get("total_with_final_flush_ms", decode["total_ms"])),
            "decode_transfer_tail_ms": float(decode["transfer_tail_ms"]),
            "decode_seq_compute_ms": float(decode_seq["compute_ms"]),
            "decode_seq_total_ms": float(decode_seq["total_ms"]),
            "decode_seq_total_with_final_flush_ms": float(decode_seq.get("total_with_final_flush_ms", decode_seq["total_ms"])),
            "decode_seq_transfer_tail_ms": float(decode_seq["transfer_tail_ms"]),
            "path": str(path),
        }
    )

if not rows:
    raise SystemExit("No JSON results found for summary.")

by_baseline = {}
for row in rows:
    by_baseline.setdefault(row["baseline"], []).append(row)

summary_rows = []
for baseline, items in sorted(by_baseline.items()):
    def med(key: str) -> float:
        return float(statistics.median(item[key] for item in items))

    summary_rows.append(
        {
            "baseline": baseline,
            "num_repeats": len(items),
            "prefill_compute_ms_median": med("prefill_compute_ms"),
            "prefill_total_ms_median": med("prefill_total_ms"),
            "prefill_transfer_tail_ms_median": med("prefill_transfer_tail_ms"),
            "decode_compute_ms_median": med("decode_compute_ms"),
            "decode_total_ms_median": med("decode_total_ms"),
            "decode_total_with_final_flush_ms_median": med("decode_total_with_final_flush_ms"),
            "decode_transfer_tail_ms_median": med("decode_transfer_tail_ms"),
            "decode_seq_compute_ms_median": med("decode_seq_compute_ms"),
            "decode_seq_total_ms_median": med("decode_seq_total_ms"),
            "decode_seq_total_with_final_flush_ms_median": med("decode_seq_total_with_final_flush_ms"),
            "decode_seq_transfer_tail_ms_median": med("decode_seq_transfer_tail_ms"),
        }
    )

summary_json = results_dir / "summary_median.json"
summary_csv = results_dir / "summary_median.csv"
with summary_json.open("w") as f:
    json.dump({"results_dir": str(results_dir), "rows": summary_rows}, f, indent=2)

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print("============================================================")
print(f"Median summary written to: {summary_json}")
print(f"Median summary written to: {summary_csv}")
print("Summary:")
for row in summary_rows:
    print(
        f"  {row['baseline']:12s} "
        f"prefill compute/total={row['prefill_compute_ms_median']:.3f}/{row['prefill_total_ms_median']:.3f} ms  "
        f"decode_last compute/total/total+flush={row['decode_compute_ms_median']:.3f}/{row['decode_total_ms_median']:.3f}/{row['decode_total_with_final_flush_ms_median']:.3f} ms  "
        f"decode_seq compute/total/total+flush={row['decode_seq_compute_ms_median']:.3f}/{row['decode_seq_total_ms_median']:.3f}/{row['decode_seq_total_with_final_flush_ms_median']:.3f} ms"
    )
print("============================================================")
PY
