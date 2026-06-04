#!/bin/bash
#SBATCH --job-name=atog-llama8b
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=cur_sbatch_log/atog-llama8b-%j.out
#SBATCH --error=cur_sbatch_log/atog-llama8b-%j.err

# ===========================================================================
# Node-toggle online experiment (Llama-3.1-8B-Instruct, 32 layers).
#
# Sweeps the runtime node-toggle at three granularities and adaptive-benches
# each, to show monitoring overhead scales with #enabled hooks (vs DMI's fixed
# always-on transport tax):
#   off       0/32 hooks  -> monitoring armed but idle (~baseline)
#   partial   4/32 (last layers) -> realistic online hallucination-monitor case
#   full      32/32       -> all hidden-states (== always-on DMI / ring_null)
#
# Baseline (no DMI) = adapt_bl_llama8b.sbatch; full == adapt_dmi_llama8b workload
# via the toggle path. Node-toggle needs FULL CUDA graphs for decode
# (cudagraph_mode FULL) so producer nodes are captured + bindable; DMXGPUWorker
# raises if a subset is requested but nothing bound. Static config -> eager apply
# (lazy dmx_lazy_toggle only matters for per-step reconfigure, not this bench).
# ===========================================================================

cd ~/scratch.zaoxing-prj

ENV_PYTHON=$(pwd)/envs/vllm-h100/bin/python3.10
SITE=$(pwd)/dmi-env/lib/python3.10/site-packages
DMI=$(pwd)/DMI
export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/transformers/src:$SITE
export LD_LIBRARY_PATH=$(pwd)/dmi-env/lib:$(pwd)/DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VLLM_TARGET_DEVICE=cuda
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_MODULE_LOADING=EAGER
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache
export VLLM_CACHE_ROOT=$(pwd)/vllm_cache

mkdir -p cur_sbatch_log toggle_results

MODEL_PATH=$(ls -d hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1 | sed 's:/$::')
PORT=8046

# enabled-hook strings ("hook_type:layer", hook_type 0 = hidden-states/resid_pre)
OFF_HOOKS="0:99"                                   # nonexistent layer -> all disabled
PARTIAL_HOOKS="0:28,0:29,0:30,0:31"                # last 4 of 32 layers
FULL_HOOKS=$(printf "0:%d," $(seq 0 31) | sed 's/,$//')

run_one() {
    local tag="$1" enabled="$2"
    local DMI_ARGS="{\"dmx_hook_selection\": \"hidden-states\", \"dmx_node_toggle\": true, \"dmx_enabled_hooks\": \"$enabled\", \"dmx_ring_payload_mb\": 4096, \"dmx_ring_pinned_mb\": 4096}"

    echo "========== Toggle [$tag]  enabled=[$enabled]  port=$PORT =========="

    $ENV_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --max-model-len 4096 \
        --no-enable-prefix-caching \
        --port $PORT \
        --compilation-config '{"cudagraph_mode": "FULL"}' \
        --worker-cls monitoring.vllm_integration.DMXGPUWorker \
        --additional-config "$DMI_ARGS" &
    SERVER_PID=$!

    echo "Waiting for server (PID=$SERVER_PID)..."
    for i in $(seq 1 1800); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"; break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process died (toggle [$tag])"; return 1
        fi
        sleep 1
    done
    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server failed to start (toggle [$tag])"; kill $SERVER_PID 2>/dev/null; return 1
    fi

    $ENV_PYTHON adaptive_bench.py \
        --base-url http://localhost:$PORT \
        --model "$MODEL_PATH" \
        --tag "llama8b_$tag" \
        --result-dir toggle_results \
        --result-prefix "toggle_" \
        --low-rate 64 --high-rate 256 \
        --multiplier 5 \
        --resolution 4 \
        --duration 30 \
        --cooldown 30

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5
}

run_one off     "$OFF_HOOKS"
run_one partial "$PARTIAL_HOOKS"
run_one full    "$FULL_HOOKS"

echo "========== Done (results in toggle_results/) =========="
