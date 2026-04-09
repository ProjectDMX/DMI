#!/bin/bash
#SBATCH --job-name=smoke-maxbs-14b
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

set -uo pipefail

SCRATCH="${SCRATCH:-YOUR_SCRATCH_PATH}"
PROJECT=${SCRATCH}/DMI/DMI

SMOKE=1 bash "${PROJECT}/experiments/offline_inference/run_max_batch_hs_logits_qwen3_14b.sh"
