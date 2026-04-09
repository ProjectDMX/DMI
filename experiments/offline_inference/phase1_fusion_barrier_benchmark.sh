#!/usr/bin/env bash
# Phase 1 benchmark: Measure fusion barrier overhead and selective monitoring scaling.
#
# Compares 4 baselines on the same model/batch/decode config:
#   hf_ideal         — pure forward, no HookPoint (true zero baseline)
#   proj_dmi_manual  — DMI with Ring² null sink, all hooks (fusion barrier + ring overhead)
#   proj_dmi_manual  — DMI with Ring² null sink, hidden-states only (selective 36 hooks)
#   hf_api           — HF built-in hidden_states extraction (reference baseline)
#
# Key output: fusion_barrier_overhead = proj_dmi_manual(null) - hf_ideal
#
# Usage:
#   bash experiments/offline_inference/phase1_fusion_barrier_benchmark.sh
#
# Environment variables:
#   MODEL        — model name (default: qwen3-4b)
#   BATCH_SIZE   — batch size (default: 64)
#   GPU          — CUDA device (default: 0)
#   REPEATS      — number of measurement repeats (default: 5)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-qwen3-4b}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/offline_inference/results/phase1_fusion_barrier}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/${MODEL//\//_}_${RUN_TAG}"
PREFILL_TOKENS="${PREFILL_TOKENS:-32}"
DECODE_STEPS="${DECODE_STEPS:-128}"
ITERS="${ITERS:-1}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"
RING_PAYLOAD_MB="${RING_PAYLOAD_MB:-10240}"
RING_PINNED_MB="${RING_PINNED_MB:-10240}"
RING_TASK_ENTRIES="${RING_TASK_ENTRIES:-131072}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_MODULE_LOADING=EAGER
mkdir -p "${RESULTS_DIR}"

COMMON_RING_ARGS=(
  --ring-payload-mb "${RING_PAYLOAD_MB}"
  --ring-pinned-mb "${RING_PINNED_MB}"
  --ring-task-entries "${RING_TASK_ENTRIES}"
  --drain-flush-task-ratio 0.0
  --drain-flush-payload-ratio 0.0
  --drain-flush-timeout-us 1000
  --db-host localhost
  --db-port 9000
  --db-user default
  --db-database default
  --db-table offload
)

run_one() {
  local baseline="$1"
  local repeat_index="$2"
  shift 2
  local extra_args=("$@")
  echo "============================================================"
  echo "baseline=${baseline} ${extra_args[*]:-} bs=${BATCH_SIZE} repeat=${repeat_index}/${REPEATS}"
  echo "============================================================"
  PYTHON_BIN="${PYTHON_BIN}" bash experiments/offline_inference/scripts/run_step_breakdown_baseline.sh "${baseline}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --prefill-tokens "${PREFILL_TOKENS}" \
    --decode-steps "${DECODE_STEPS}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeat-index "${repeat_index}" \
    --results-dir "${RESULTS_DIR}" \
    ${LOCAL_FILES_ONLY} \
    "${extra_args[@]}"
}

for repeat_index in $(seq 1 "${REPEATS}"); do
  # Baseline 1: hf_ideal (no hooks, true zero)
  run_one hf_ideal "${repeat_index}"

  # Baseline 2: HF API (output_hidden_states=True, sync D2H)
  run_one hf_api "${repeat_index}"

  # Baseline 3: DMI full hooks (hidden-states,final_ln,logits), ring_null
  run_one proj_dmi_manual "${repeat_index}" \
    --baseline-label "dmi_full_null" \
    --hook-selection "hidden-states,final_ln,logits" \
    --proj-dmi-mode ring_null \
    "${COMMON_RING_ARGS[@]}"

  # Baseline 4: DMI selective (hidden-states only ~36 hooks), ring_null
  run_one proj_dmi_manual "${repeat_index}" \
    --baseline-label "dmi_hs_only_null" \
    --hook-selection "hidden-states" \
    --proj-dmi-mode ring_null \
    "${COMMON_RING_ARGS[@]}"

  # Baseline 5: DMI full hooks, ring_null, with torch.compile disabled (eager)
  # This isolates CUDA graph overhead from fusion barrier overhead
  run_one proj_dmi_manual "${repeat_index}" \
    --baseline-label "dmi_full_null_eager" \
    --hook-selection "hidden-states,final_ln,logits" \
    --proj-dmi-mode ring_null \
    --disable-compile \
    "${COMMON_RING_ARGS[@]}"
done

# Summary
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
            "repeat_index": int(data.get("repeat_index", 1)),
            "prefill_compute_ms": float(data["prefill"]["compute_ms"]),
            "prefill_total_ms": float(data["prefill"]["total_ms"]),
            "decode_compute_ms": float(decode.get("compute_ms", 0)),
            "decode_total_ms": float(decode.get("total_ms", 0)),
            "decode_total_with_flush_ms": float(decode.get("total_with_final_flush_ms", decode.get("total_ms", 0))),
            "decode_seq_compute_ms": float(decode_seq.get("compute_ms", 0)),
            "decode_seq_total_ms": float(decode_seq.get("total_ms", 0)),
        }
    )

if not rows:
    raise SystemExit("No JSON results found.")

by_baseline = {}
for row in rows:
    by_baseline.setdefault(row["baseline"], []).append(row)

print("\n" + "=" * 80)
print("Phase 1 Summary: Fusion Barrier Overhead")
print("=" * 80)

ref_decode = None
for baseline, items in sorted(by_baseline.items()):
    med_decode = statistics.median(item["decode_seq_compute_ms"] for item in items)
    med_prefill = statistics.median(item["prefill_compute_ms"] for item in items)
    if baseline == "hf_ideal":
        ref_decode = med_decode
    overhead = ""
    if ref_decode and ref_decode > 0 and baseline != "hf_ideal":
        pct = (med_decode - ref_decode) / ref_decode * 100
        overhead = f"  overhead vs hf_ideal: {pct:+.2f}%"
    print(f"  {baseline:25s}  prefill={med_prefill:8.3f} ms  decode_seq={med_decode:8.3f} ms{overhead}")

print("=" * 80)

# Write CSV
summary_rows = []
for baseline, items in sorted(by_baseline.items()):
    summary_rows.append({
        "baseline": baseline,
        "n": len(items),
        "prefill_compute_ms": statistics.median(item["prefill_compute_ms"] for item in items),
        "decode_seq_compute_ms": statistics.median(item["decode_seq_compute_ms"] for item in items),
        "decode_seq_total_ms": statistics.median(item["decode_seq_total_ms"] for item in items),
    })
csv_path = results_dir / "phase1_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nCSV: {csv_path}")
PY
