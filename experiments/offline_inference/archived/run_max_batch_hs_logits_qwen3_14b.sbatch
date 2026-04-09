#!/bin/bash
#SBATCH --job-name=maxbs-14b-hslogits
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

set -uo pipefail

SCRATCH="${SCRATCH:-YOUR_SCRATCH_PATH}"
PROJECT=${SCRATCH}/DMI/DMI

bash "${PROJECT}/experiments/offline_inference/run_max_batch_hs_logits_qwen3_14b.sh"
