#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/sixian/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ring_offload}"

if [ ! -f "$CONDA_SH" ]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

MODEL="${E2E_MODEL:-qwen2_moe}"
HOOKS="${E2E_HOOKS:-vllm-full}"
TP="${E2E_TP_SIZE:-2}"
MAX_MODEL_LEN="${E2E_MAX_MODEL_LEN:-128}"
REF_MAX_LEN="${E2E_REF_MAX_LEN:-512}"
MAX_BATCHED="${E2E_MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEM_UTIL="${E2E_GPU_MEM_UTIL:-0.8}"
DTYPE="${E2E_DTYPE:-bfloat16}"
ENFORCE_EAGER="${E2E_ENFORCE_EAGER:-1}"
ENABLE_EP="${E2E_ENABLE_EP:-0}"
ALL2ALL_BACKEND="${E2E_ALL2ALL_BACKEND:-}"
DB_HOST="${DMX_DB_HOST:-localhost}"
DB_PORT="${DMX_DB_PORT:-9000}"
RING_PAYLOAD_MB="${E2E_RING_PAYLOAD_MB:-4096}"
RING_PINNED_MB="${E2E_RING_PINNED_MB:-4096}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_BASE="${E2E_ARTIFACT_BASE:-$ROOT_DIR/artifacts}"
RUN_DIR="${E2E_ARTIFACT_DIR:-$ARTIFACT_BASE/vllm_qwen2_moe_pipeline_$TIMESTAMP}"

mkdir -p "$RUN_DIR/ref" "$RUN_DIR/mon"

export LD_PRELOAD="${LD_PRELOAD:-/home/sixian/miniforge3/envs/ring_offload/lib/libstdc++.so.6}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export E2E_MODEL="$MODEL"
export E2E_TP_SIZE="$TP"
export E2E_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export E2E_REF_MAX_LEN="$REF_MAX_LEN"
export E2E_MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED"
export E2E_GPU_MEM_UTIL="$GPU_MEM_UTIL"
export E2E_DTYPE="$DTYPE"
export E2E_ENFORCE_EAGER="$ENFORCE_EAGER"
export E2E_ENABLE_EP="$ENABLE_EP"
if [ -n "$ALL2ALL_BACKEND" ]; then
  export E2E_ALL2ALL_BACKEND="$ALL2ALL_BACKEND"
fi
export DMX_HOOK_SELECTION="$HOOKS"
export DMX_DB_HOST="$DB_HOST"
export DMX_DB_PORT="$DB_PORT"
export E2E_RING_PAYLOAD_MB="$RING_PAYLOAD_MB"
export E2E_RING_PINNED_MB="$RING_PINNED_MB"

REF_MODEL_FILE="$ROOT_DIR/integration/vllm/vllm/model_executor/models/qwen2_moe_ref.py"
REF_CONFIG="$RUN_DIR/ref/ref_config.json"
RESULT_JSON="$RUN_DIR/result.json"

{
  echo "============================================"
  echo "vLLM MoE pipeline"
  echo "model=$MODEL"
  echo "hooks=$HOOKS"
  echo "tp=$TP"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "ref_max_len=$REF_MAX_LEN"
  echo "max_num_batched_tokens=$MAX_BATCHED"
  echo "gpu_memory_utilization=$GPU_MEM_UTIL"
  echo "enable_ep=$ENABLE_EP"
  echo "all2all_backend=${ALL2ALL_BACKEND:-default}"
  echo "dtype=$DTYPE"
  echo "run_dir=$RUN_DIR"
  echo "============================================"
} | tee "$RUN_DIR/summary.log"

python "$ROOT_DIR/integration/vllm/vllm/model_executor/models/enable_ref_hooks.py" \
  --model-file "$REF_MODEL_FILE" \
  --hooks "$HOOKS" \
  --max-len "$REF_MAX_LEN" \
  --output-dir "$RUN_DIR/ref" \
  --config-out "$REF_CONFIG" \
  | tee "$RUN_DIR/enable_ref_hooks.log"

REF_CONFIG="$REF_CONFIG" \
python -m tests.vllm_ref_runner \
  --output-dir "$RUN_DIR/ref" \
  2>&1 | tee "$RUN_DIR/stdout_ref_runner.log"

python -m tests.vllm_monitored_runner \
  --output-dir "$RUN_DIR/mon" \
  2>&1 | tee "$RUN_DIR/stdout_mon_runner.log"

python -m tests.vllm_identical_comparator \
  --ref-config "$REF_CONFIG" \
  --mon-dir "$RUN_DIR/mon" \
  --result-file "$RESULT_JSON" \
  2>&1 | tee "$RUN_DIR/comparator.log"

echo "done: $RUN_DIR"
