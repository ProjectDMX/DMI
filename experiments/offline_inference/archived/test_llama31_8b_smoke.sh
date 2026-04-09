#!/usr/bin/env bash
set -euo pipefail

SCRATCH="${SCRATCH:-}"
if [ -z "${SCRATCH}" ] || [ "${SCRATCH}" = "YOUR_SCRATCH_PATH" ]; then
    echo "ERROR: SCRATCH is not set to a valid scratch directory." >&2
    echo "Please export SCRATCH=/path/to/your/scratch before running this script." >&2
    exit 1
fi
ENV_DIR=${SCRATCH}/proj-dmx
PROJECT=${SCRATCH}/DMI/DMI

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "ERROR: ${ENV_DIR}/bin/python not found"
  exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_HOME="${ENV_DIR}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${PROJECT}/transformers/src:${PROJECT}"
export TL_ENABLE_NVTX=1
export HF_HOME="${SCRATCH}/hf_cache"

if [ ! -d "${HF_HOME}" ]; then
  echo "ERROR: HF cache not found at ${HF_HOME}"
  exit 1
fi

cd "${PROJECT}/experiments/offline_inference"

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/test_llama31_8b_${RUN_TAG}"
mkdir -p "${RESULTS_DIR}"

MODEL="llama3.1-8b"
BATCH_SIZE=1
LIMIT=1
MAX_INPUT=32
MAX_OUTPUT=16
PAD_BUCKETS="64"
SHAREGPT_SAMPLE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
WILDCHAT_SAMPLE="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
INTERNAL_HOOKS="q,k,v,z,mlp_in,mlp_out,resid_mid"

run_cmd() {
  local label="$1"
  shift
  echo ""
  echo "=== ${label} ==="
  "$@"
}

for sample_file in "${SHAREGPT_SAMPLE}" "${WILDCHAT_SAMPLE}"; do
  dataset="$(basename "${sample_file}" .jsonl)"
  echo ""
  echo "############################################################"
  echo "Smoke test dataset=${dataset}"
  echo "############################################################"

  COMMON=(
    --model "${MODEL}"
    --batch-size "${BATCH_SIZE}"
    --sample-file "${sample_file}"
    --local-files-only
    --max-input-tokens "${MAX_INPUT}"
    --max-new-tokens "${MAX_OUTPUT}"
    --limit "${LIMIT}"
    --pad-buckets "${PAD_BUCKETS}"
    --results-dir "${RESULTS_DIR}"
  )

  run_cmd "hs_logits/proj_dmi_compile/${dataset}" \
    python scripts/run_proj_dmi.py "${COMMON[@]}" --capture-mode hs_logits --proj-dmi-mode ring_null \
      --ring-payload-mb 5120 --ring-pinned-mb 5120 --ring-task-entries 65536

  run_cmd "internal_hooks/proj_dmi_compile/${dataset}" \
    python scripts/run_proj_dmi.py "${COMMON[@]}" --capture-mode hs --proj-dmi-mode ring_null \
      --hook-selection "${INTERNAL_HOOKS}" \
      --ring-payload-mb 5120 --ring-pinned-mb 5120 --ring-task-entries 65536
done

echo ""
echo "Smoke tests completed."
echo "Results dir: ${RESULTS_DIR}"
