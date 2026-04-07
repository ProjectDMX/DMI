#!/bin/bash
#SBATCH --job-name=test-llama31-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
PROJECT=${SCRATCH}/DMI/DMI

bash "${PROJECT}/experiments/offline_inference/test_llama31_8b_smoke.sh"
