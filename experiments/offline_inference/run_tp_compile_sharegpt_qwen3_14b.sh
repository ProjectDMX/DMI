#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_env.sh"
offline_e2e_setup_local_env

RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results/tp_compile_sharegpt_qwen3_14b_$(date '+%Y%m%d_%H%M%S')}"
COMMON=(--model qwen3-14b --batch-size 64 --sample-file benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl --local-files-only --max-input-tokens 200 --max-new-tokens 750 --limit 128 --pad-buckets 64,128,256,512 --results-dir "${RESULTS_DIR}" --capture-mode hs_logits)

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TP2:-0,1}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}" --tp-size 2
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TP4:-0,1,2,3}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}" --tp-size 4
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null --ring-payload-mb 30720 --ring-pinned-mb 30720 --ring-task-entries 131072 --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TP2:-0,1}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null --tp-size 2 --ring-payload-mb 30720 --ring-pinned-mb 30720 --ring-task-entries 131072 --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TP4:-0,1,2,3}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null --tp-size 4 --ring-payload-mb 30720 --ring-pinned-mb 30720 --ring-task-entries 131072 --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15
