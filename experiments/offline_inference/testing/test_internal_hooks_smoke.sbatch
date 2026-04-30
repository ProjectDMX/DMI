#!/bin/bash
#SBATCH --job-name=test-hooks-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRATCH="${SCRATCH:-}"
if [ -z "${SCRATCH}" ] || [ "${SCRATCH}" = "YOUR_SCRATCH_PATH" ]; then
    echo "ERROR: SCRATCH is not set to a valid scratch directory." >&2
    echo "Please export SCRATCH=/path/to/your/scratch before running this script." >&2
    exit 1
fi
PROJECT=${SCRATCH}/DMI/DMI

bash "${PROJECT}/experiments/offline_inference/test_internal_hooks_smoke.sh"
