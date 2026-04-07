#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/_local_env.sh"
offline_e2e_setup_local_env

MODEL="${MODEL:-qwen3-14b}"
DATASET="${DATASET:-wildchat}"
CAPTURE_MODE="${CAPTURE_MODE:-hs_logits}"
MAX_BS_CAP="${MAX_BS_CAP:-500}"
PAD_BUCKETS="${PAD_BUCKETS:-64,128,256,512}"
RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results/max_batch_memory_qwen3_14b_$(date '+%Y%m%d_%H%M%S')}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"
RING_MB="${RING_MB:-2048}"
RING_TASK_ENTRIES="${RING_TASK_ENTRIES:-131072}"
RING_FLUSH="${RING_FLUSH:-0.15}"
SEARCH_STOP_GAP="${SEARCH_STOP_GAP:-5}"

SEARCH_HIGH_HF_MONITOR_GENERATE="${SEARCH_HIGH_HF_MONITOR_GENERATE:-32}"
SEARCH_HIGH_HF_MANUAL_COMPILE="${SEARCH_HIGH_HF_MANUAL_COMPILE:-128}"
SEARCH_HIGH_DMI="${SEARCH_HIGH_DMI:-128}"
SEARCH_HIGH_HF_UPPER_COMPILE="${SEARCH_HIGH_HF_UPPER_COMPILE:-183}"
SEARCH_HIGH_TORCH_HOOKS="${SEARCH_HIGH_TORCH_HOOKS:-192}"
SEARCH_HIGH_NNSIGHT="${SEARCH_HIGH_NNSIGHT:-192}"
SEARCH_HIGH_HF_UPPER_EAGER="${SEARCH_HIGH_HF_UPPER_EAGER:-183}"

if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

case "${DATASET}" in
  wildchat)
    MAX_INPUT=250
    MAX_OUTPUT=1000
    ;;
  sharegpt)
    MAX_INPUT=200
    MAX_OUTPUT=750
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

BASELINES=(
  "hf_monitor_generate_eager"
  "hf_monitor_manual_compile"
  "proj_dmi_compile_ring2g"
  "hf_upper_bound_compile"
  "torch_hooks_eager"
  "nnsight_eager"
  "proj_dmi_eager_ring2g"
  "hf_upper_bound_eager"
)

mkdir -p "${RESULTS_DIR}"
LOG_DIR="${RESULTS_DIR}/logs"
ATTEMPT_DIR="${RESULTS_DIR}/attempts"
mkdir -p "${LOG_DIR}" "${ATTEMPT_DIR}"

SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
ATTEMPTS_CSV="${RESULTS_DIR}/attempts.csv"
SYNTHETIC_FILE="${RESULTS_DIR}/synthetic_${DATASET}_${MAX_INPUT}_${MAX_OUTPUT}_${MAX_BS_CAP}.jsonl"

echo "baseline,ring_mb,max_batch_size,last_ok_bs,first_fail_bs,target_tok_s,compute_tok_s,prompts_s,total_seconds,search_total_seconds,search_log_file,search_json" > "${SUMMARY_CSV}"
echo "baseline,ring_mb,batch_size,status,target_tok_s,compute_tok_s,prompts_s,total_seconds,log_file,json_file" > "${ATTEMPTS_CSV}"

generate_synthetic_sample() {
  "${PYTHON_BIN}" - <<'PY' "${SYNTHETIC_FILE}" "${MAX_BS_CAP}" "${DATASET}" "${MAX_INPUT}" "${MAX_OUTPUT}"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
count = int(sys.argv[2])
dataset = sys.argv[3]
max_input = int(sys.argv[4])
max_output = int(sys.argv[5])

prompt_text = ("synthetic prompt token " * (max_input * 12)).strip()
target_text = ("synthetic target token " * (max_output * 12)).strip()

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i in range(count):
        row = {
            "dataset": dataset,
            "sample_id": i + 1,
            "entry_id": f"synthetic_{i}",
            "source_conversation_id": f"synthetic_{i}",
            "approx_prompt_tokens": max_input,
            "approx_target_tokens": max_output,
            "target_text": target_text,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        f.write(json.dumps(row) + "\n")
PY
}

cleanup_gpu() {
  "${PYTHON_BIN}" -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" >/dev/null 2>&1 || true
  sleep 2
}

is_oom_log() {
  local log_file="$1"
  grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM' "${log_file}"
}

next_probe_up() {
  local bs="$1"
  if [ "${bs}" -lt 128 ]; then
    echo 128
  elif [ "${bs}" -lt 256 ] && [ "${MAX_BS_CAP}" -gt 128 ]; then
    if [ "${MAX_BS_CAP}" -lt 256 ]; then
      echo "${MAX_BS_CAP}"
    else
      echo 256
    fi
  elif [ "${bs}" -lt "${MAX_BS_CAP}" ]; then
    echo "${MAX_BS_CAP}"
  else
    echo 0
  fi
}

get_fallbacks() {
  local baseline="$1"
  if [ "${baseline}" = "hf_monitor_generate_eager" ]; then
    echo "16 8 4 2 1"
  else
    echo "128 64 32 16 8 4 2 1"
  fi
}

get_start_probe() {
  local baseline="$1"
  case "${baseline}" in
    hf_monitor_generate_eager) echo "${SEARCH_HIGH_HF_MONITOR_GENERATE}" ;;
    hf_monitor_manual_compile) echo "${SEARCH_HIGH_HF_MANUAL_COMPILE}" ;;
    proj_dmi_compile_ring2g|proj_dmi_eager_ring2g) echo "${SEARCH_HIGH_DMI}" ;;
    hf_upper_bound_compile) echo "${SEARCH_HIGH_HF_UPPER_COMPILE}" ;;
    torch_hooks_eager) echo "${SEARCH_HIGH_TORCH_HOOKS}" ;;
    nnsight_eager) echo "${SEARCH_HIGH_NNSIGHT}" ;;
    hf_upper_bound_eager) echo "${SEARCH_HIGH_HF_UPPER_EAGER}" ;;
    *)
      echo "Unknown baseline ${baseline}" >&2
      exit 1
      ;;
  esac
}

build_command() {
  local baseline="$1"
  local bs="$2"
  local run_dir="$3"

  COMMON=(
    --model "${MODEL}"
    --batch-size "${bs}"
    --sample-file "${SYNTHETIC_FILE}"
    --max-input-tokens "${MAX_INPUT}"
    --max-new-tokens "${MAX_OUTPUT}"
    --limit "${bs}"
    --pad-buckets "${PAD_BUCKETS}"
    --results-dir "${run_dir}"
    --capture-mode "${CAPTURE_MODE}"
  )

  if [ -n "${LOCAL_FILES_ONLY}" ]; then
    COMMON+=("${LOCAL_FILES_ONLY}")
  fi

  RING_MB_USED=""
  case "${baseline}" in
    hf_monitor_generate_eager)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_monitor.py "${COMMON[@]}")
      ;;
    hf_monitor_manual_compile)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_monitor_manual.py "${COMMON[@]}")
      ;;
    proj_dmi_compile_ring2g)
      RING_MB_USED="${RING_MB}"
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --ring-payload-mb "${RING_MB_USED}" --ring-pinned-mb "${RING_MB_USED}" --ring-task-entries "${RING_TASK_ENTRIES}" --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}")
      ;;
    hf_upper_bound_compile)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}")
      ;;
    torch_hooks_eager)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_torch_hooks.py "${COMMON[@]}" --disable-compile)
      ;;
    nnsight_eager)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_nnsight.py "${COMMON[@]}")
      ;;
    proj_dmi_eager_ring2g)
      RING_MB_USED="${RING_MB}"
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --disable-compile --ring-payload-mb "${RING_MB_USED}" --ring-pinned-mb "${RING_MB_USED}" --ring-task-entries "${RING_TASK_ENTRIES}" --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}")
      ;;
    hf_upper_bound_eager)
      CMD=("${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}" --disable-compile)
      ;;
    *)
      echo "Unknown baseline ${baseline}" >&2
      exit 1
      ;;
  esac
}

append_attempt() {
  local baseline="$1"
  local ring_mb="$2"
  local bs="$3"
  local status="$4"
  local target_toks="$5"
  local compute_toks="$6"
  local prompts_s="$7"
  local total_seconds="$8"
  local log_file="$9"
  local json_file="${10}"
  echo "${baseline},${ring_mb},${bs},${status},${target_toks},${compute_toks},${prompts_s},${total_seconds},${log_file},${json_file}" >> "${ATTEMPTS_CSV}"
}

append_summary() {
  local baseline="$1"
  local ring_mb="$2"
  local max_bs="$3"
  local low_ok="$4"
  local high_fail="$5"
  local target_toks="$6"
  local compute_toks="$7"
  local prompts_s="$8"
  local total_seconds="$9"
  local search_total_seconds="${10}"
  local log_file="${11}"
  local json_file="${12}"
  echo "${baseline},${ring_mb},${max_bs},${low_ok},${high_fail},${target_toks},${compute_toks},${prompts_s},${total_seconds},${search_total_seconds},${log_file},${json_file}" >> "${SUMMARY_CSV}"
}

run_attempt() {
  local baseline="$1"
  local bs="$2"
  local safe_label="${baseline// /_}"
  safe_label="${safe_label//\//_}"
  local tag="${safe_label}__bs${bs}"
  local run_dir="${ATTEMPT_DIR}/${tag}"
  local log_file="${LOG_DIR}/${tag}.log"
  mkdir -p "${run_dir}"

  build_command "${baseline}" "${bs}" "${run_dir}"

  echo ""
  echo "=== ${baseline} | bs=${bs} ring=${RING_MB_USED:-na} ==="

  local status="FAIL"
  local target_toks=""
  local compute_toks=""
  local prompts_s=""
  local total_seconds=""
  local json_file=""

  if "${CMD[@]}" >"${log_file}" 2>&1; then
    status="OK"
  else
    if is_oom_log "${log_file}"; then
      status="OOM"
    else
      status="FAIL"
    fi
  fi

  cat "${log_file}"

  json_file=$("${PYTHON_BIN}" - <<'PY' "${run_dir}"
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
files = sorted(run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
print(files[0] if files else "")
PY
)

  if [ -n "${json_file}" ] && [ -f "${json_file}" ]; then
    read -r target_toks compute_toks prompts_s total_seconds <<EOF
$("${PYTHON_BIN}" - <<'PY' "${json_file}"
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
vals = [
    d.get("target_generated_tokens_per_s", ""),
    d.get("actual_generated_tokens_per_s", ""),
    d.get("prompts_per_s", ""),
    d.get("total_seconds", ""),
]
print(*vals)
PY
)
EOF
  else
    target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
    compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
  fi

  append_attempt "${baseline}" "${RING_MB_USED}" "${bs}" "${status}" "${target_toks}" "${compute_toks}" "${prompts_s}" "${total_seconds}" "${log_file}" "${json_file}"
  cleanup_gpu

  ATTEMPT_STATUS="${status}"
  ATTEMPT_TARGET_TOKS="${target_toks}"
  ATTEMPT_COMPUTE_TOKS="${compute_toks}"
  ATTEMPT_PROMPTS_S="${prompts_s}"
  ATTEMPT_TOTAL_SECONDS="${total_seconds}"
  ATTEMPT_LOG_FILE="${log_file}"
  ATTEMPT_JSON_FILE="${json_file}"
}

search_baseline() {
  local baseline="$1"
  local start_probe
  start_probe=$(get_start_probe "${baseline}")
  local fallbacks
  fallbacks=$(get_fallbacks "${baseline}")

  local search_start_ts
  search_start_ts=$(date +%s)
  local low_ok=0
  local high_fail=0
  local best_ring_mb=""
  local best_target_toks=""
  local best_compute_toks=""
  local best_prompts_s=""
  local best_total_seconds=""
  local best_log_file=""
  local best_json_file=""

  run_attempt "${baseline}" "${start_probe}"
  if [ "${ATTEMPT_STATUS}" = "OK" ]; then
    low_ok="${start_probe}"
    best_ring_mb="${RING_MB_USED}"
    best_target_toks="${ATTEMPT_TARGET_TOKS}"
    best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
    best_prompts_s="${ATTEMPT_PROMPTS_S}"
    best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
    best_log_file="${ATTEMPT_LOG_FILE}"
    best_json_file="${ATTEMPT_JSON_FILE}"

    local next_probe
    next_probe=$(next_probe_up "${low_ok}")
    while [ "${next_probe}" -gt 0 ]; do
      run_attempt "${baseline}" "${next_probe}"
      if [ "${ATTEMPT_STATUS}" = "OK" ]; then
        low_ok="${next_probe}"
        best_ring_mb="${RING_MB_USED}"
        best_target_toks="${ATTEMPT_TARGET_TOKS}"
        best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
        best_prompts_s="${ATTEMPT_PROMPTS_S}"
        best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
        best_log_file="${ATTEMPT_LOG_FILE}"
        best_json_file="${ATTEMPT_JSON_FILE}"
        if [ "${low_ok}" -ge "${MAX_BS_CAP}" ]; then
          high_fail=0
          break
        fi
        next_probe=$(next_probe_up "${low_ok}")
      else
        high_fail="${next_probe}"
        break
      fi
    done
  else
    high_fail="${start_probe}"
    for bs in ${fallbacks}; do
      if [ "${bs}" -ge "${start_probe}" ]; then
        continue
      fi
      run_attempt "${baseline}" "${bs}"
      if [ "${ATTEMPT_STATUS}" = "OK" ]; then
        low_ok="${bs}"
        best_ring_mb="${RING_MB_USED}"
        best_target_toks="${ATTEMPT_TARGET_TOKS}"
        best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
        best_prompts_s="${ATTEMPT_PROMPTS_S}"
        best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
        best_log_file="${ATTEMPT_LOG_FILE}"
        best_json_file="${ATTEMPT_JSON_FILE}"
        break
      fi
      high_fail="${bs}"
    done
  fi

  if [ "${low_ok}" -gt 0 ] && [ "${high_fail}" -gt $((low_ok + SEARCH_STOP_GAP)) ]; then
    while [ $((high_fail - low_ok)) -gt "${SEARCH_STOP_GAP}" ]; do
      local mid=$(((low_ok + high_fail) / 2))
      run_attempt "${baseline}" "${mid}"
      if [ "${ATTEMPT_STATUS}" = "OK" ]; then
        low_ok="${mid}"
        best_ring_mb="${RING_MB_USED}"
        best_target_toks="${ATTEMPT_TARGET_TOKS}"
        best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
        best_prompts_s="${ATTEMPT_PROMPTS_S}"
        best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
        best_log_file="${ATTEMPT_LOG_FILE}"
        best_json_file="${ATTEMPT_JSON_FILE}"
      else
        high_fail="${mid}"
      fi
    done
  fi

  local search_end_ts
  search_end_ts=$(date +%s)
  local search_total_seconds=$((search_end_ts - search_start_ts))

  append_summary "${baseline}" "${best_ring_mb}" "${low_ok}" "${low_ok}" "${high_fail}" "${best_target_toks}" "${best_compute_toks}" "${best_prompts_s}" "${best_total_seconds}" "${search_total_seconds}" "${best_log_file}" "${best_json_file}"
}

echo "============================================================"
echo "Host     : $(hostname)"
echo "Date     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python   : $("${PYTHON_BIN}" --version 2>&1)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' -)"
echo "Model    : ${MODEL}"
echo "Dataset  : ${DATASET}"
echo "Capture  : ${CAPTURE_MODE}"
echo "Input    : ${MAX_INPUT}"
echo "Output   : ${MAX_OUTPUT}"
echo "Max cap  : ${MAX_BS_CAP}"
echo "Ring MB  : ${RING_MB}"
echo "Results  : ${RESULTS_DIR}"
echo "============================================================"

generate_synthetic_sample

for baseline in "${BASELINES[@]}"; do
  echo ""
  echo "############################################################"
  echo "Searching max batch size for ${baseline}"
  echo "############################################################"
  search_baseline "${baseline}"
done

echo ""
echo "============================================================"
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir : ${RESULTS_DIR}"
echo "Summary CSV : ${SUMMARY_CSV}"
echo "Attempts CSV: ${ATTEMPTS_CSV}"
echo "============================================================"
