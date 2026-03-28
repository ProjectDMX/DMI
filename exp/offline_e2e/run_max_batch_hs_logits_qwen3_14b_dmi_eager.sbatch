#!/bin/bash
#SBATCH --job-name=maxbs-14b-dmi-eager
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

set -uo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
PROJECT=${SCRATCH}/DMI/DMI

bash "${PROJECT}/exp/offline_e2e/run_max_batch_hs_logits_qwen3_14b_dmi_eager.sh"
