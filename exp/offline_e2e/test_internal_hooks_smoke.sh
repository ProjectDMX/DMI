#!/usr/bin/env bash
set -euo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
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

cd "${PROJECT}/exp/offline_e2e"

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/test_internal_hooks_${RUN_TAG}"
mkdir -p "${RESULTS_DIR}"

MODEL="${MODEL:-qwen3-4b}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-1}"
MAX_INPUT="${MAX_INPUT:-32}"
MAX_OUTPUT="${MAX_OUTPUT:-16}"
PAD_BUCKETS="${PAD_BUCKETS:-64}"
SHAREGPT_SAMPLE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
WILDCHAT_SAMPLE="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
INTERNAL_HOOKS="${INTERNAL_HOOKS:-q,k,v,z,mlp_in,mlp_out,resid_mid}"

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
  echo "Internal-hooks smoke test model=${MODEL} dataset=${dataset}"
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
    --capture-mode hs
  )

  run_cmd "internal_hooks/hf_upper_bound_compile/${dataset}" \
    python run_hf_upper_bound.py "${COMMON[@]}"

  run_cmd "internal_hooks/torch_hooks_eager/${dataset}" \
    python run_torch_hooks.py "${COMMON[@]}" --hook-selection "${INTERNAL_HOOKS}" --disable-compile

  run_cmd "internal_hooks/nnsight_eager/${dataset}" \
    python run_nnsight.py "${COMMON[@]}" --hook-selection "${INTERNAL_HOOKS}" --disable-compile

  run_cmd "internal_hooks/proj_dmi_compile/${dataset}" \
    python run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null \
      --hook-selection "${INTERNAL_HOOKS}" \
      --ring-payload-mb 5120 --ring-pinned-mb 5120 --ring-task-entries 65536
done

echo ""
echo "Internal-hooks smoke tests completed."
echo "Results dir: ${RESULTS_DIR}"
