#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/_local_env.sh"
offline_e2e_setup_local_env

MODELS_CSV="${MODELS_CSV:-qwen3-4b,qwen3-14b}"
DATASETS="${DATASETS:-sharegpt,wildchat}"
BATCH_SIZES="${BATCH_SIZES:-16,32,64}"
RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results/full_sweep_internal_hooks_$(date '+%Y%m%d_%H%M%S')}"
for model in ${MODELS_CSV//,/ }; do
  for dataset in ${DATASETS//,/ }; do
    case "${dataset}" in
      sharegpt) max_input=200; max_output=750 ;;
      wildchat) max_input=250; max_output=1000 ;;
      *) echo "Unknown dataset: ${dataset}" >&2; exit 1 ;;
    esac
    bash experiments/offline_inference/scripts/run_internal_hooks_compare.sh \
      --model "${model}" \
      --datasets "${dataset}" \
      --baselines "hf_upper_bound,torch_hooks,nnsight,proj_dmi" \
      --batch-sizes "${BATCH_SIZES}" \
      --limit 128 \
      --max-input-tokens "${max_input}" \
      --max-new-tokens "${max_output}" \
      --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid" \
      --proj-dmi-mode ring_null \
      --proj-dmi-retry \
      --results-dir "${RESULTS_DIR}" \
      --local-files-only
  done
done
