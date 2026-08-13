#!/usr/bin/env bash
# Four-H100 DMI-vLLM acceptance sweep. This is deliberately scheduler-neutral
# so a cluster-specific sbatch file only needs to allocate four GPUs and invoke
# this wrapper.
set -euo pipefail

usage() {
    echo "Usage: $0 GPU_LIST ARTIFACT_DIR [--resume]" >&2
    echo "Example: $0 0,1,2,3 /scratch/dmi-vllm-0271-h100" >&2
}

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && ${3:-} != "--resume" ) ]]; then
    usage
    exit 2
fi

gpu_list=$1
artifact_dir=$2
IFS=',' read -r -a gpu_ids <<< "$gpu_list"
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
    echo "GPU_LIST must contain exactly four comma-separated GPU indices" >&2
    exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
        echo "invalid GPU index: $gpu_id" >&2
        exit 2
    fi
done

python_bin=${DMI_MATRIX_PYTHON:-.venv/bin/python}
if [[ ! -x $python_bin ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "Set DMI_MATRIX_PYTHON to the vLLM 0.27.1 environment interpreter." >&2
    exit 2
fi

case_timeout=${DMI_CASE_TIMEOUT_SECONDS:-7200}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
resume_args=()
if [[ ${3:-} == "--resume" ]]; then
    resume_args=(--resume)
fi

exec "$python_bin" tests/tools/run_vllm_release_matrix.py \
    --phase h100-tp4 \
    --gpus "$gpu_list" \
    --case-timeout "$case_timeout" \
    --artifact-dir "$artifact_dir" \
    "${resume_args[@]}"
