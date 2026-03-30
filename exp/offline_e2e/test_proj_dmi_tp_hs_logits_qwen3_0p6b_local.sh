#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv_tp_smoke/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}/transformers/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/exp/offline_e2e/results/tp_smoke}"
mkdir -p "${RESULTS_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "${PROJECT_ROOT}/exp/offline_e2e/run_proj_dmi.py" \
  --model "Qwen/Qwen3-0.6B" \
  --sample-file "${PROJECT_ROOT}/benchmark/data/offline_e2e/sharegpt_1000_sample1.jsonl" \
  --batch-size 2 \
  --limit 2 \
  --max-input-tokens 32 \
  --max-new-tokens 8 \
  --capture-mode hs_logits \
  --hook-selection "hidden-states,final_ln,logits" \
  --proj-dmi-mode ring_null \
  --tp-size 2 \
  --disable-compile \
  --local-files-only \
  --results-dir "${RESULTS_DIR}"
