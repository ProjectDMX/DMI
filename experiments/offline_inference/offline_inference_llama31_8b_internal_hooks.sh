#!/usr/bin/env bash
set -euo pipefail

MODELS_CSV="llama3.1-8b" bash experiments/offline_inference/offline_inference_qwen_internal_hooks.sh "$@"
