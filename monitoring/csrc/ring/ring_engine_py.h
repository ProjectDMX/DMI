// ring/ring_engine_py.h — Plain C++ interface for RingEngine, usable from g++.
//
// All CUDA/ATen details are hidden behind a pimpl so bindings.cpp (compiled
// with g++) can include this header without needing CUDA or nvcc compilation.
//
// Implementation is in ring_engine_py.cu (compiled with nvcc).

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ring/tensor_meta.h"   // TensorMeta, TensorMetaFifo

// Forward-declare at::Tensor so SubmitFn can use it without including ATen here.
namespace at { class Tensor; }

namespace ring_py {

// Plain C++ mirror of ring::RingConfig — no CUDA types.
struct RingConfig {
    uint32_t task_ring_entries          = 1024;
    uint64_t payload_ring_bytes         = 256ULL * 1024 * 1024;
    uint64_t pinned_staging_bytes       = 0;
    uint64_t drain_poll_timeout_us      = 100;
    // Drain flush thresholds
    float    drain_flush_task_ratio     = 0.0f;
    float    drain_flush_payload_ratio  = 0.0f;
    uint64_t drain_flush_entry_threshold = 0;
    uint64_t drain_flush_byte_threshold  = 0;
    uint64_t drain_flush_timeout_us      = 0;
    // Bypass budget for large tensors (bytes).
    uint64_t bypass_budget_bytes        = 256ULL * 1024 * 1024;
    // Clone per-request slices
    bool     clone_slices               = false;
    // ClickHouse insert queue limits
    uint64_t insert_queue_max_bytes     = 512ULL * 1024 * 1024;
    uint64_t insert_queue_max_items     = 4096;
};

// Called by the p2p thread for each per-request tensor slice.
using SubmitFn = std::function<void(
    const std::string& model_id,
    int32_t            shard_rank,
    const std::string& req_id,
    const std::string& act_name,
    int32_t            layer_no,
    int32_t            start_token,
    int32_t            end_token,
    at::Tensor         slice)>;

// Opaque RAII engine.
class RingEnginePy {
public:
    explicit RingEnginePy(RingConfig cfg, SubmitFn submit_fn);
    ~RingEnginePy();

    RingEnginePy(const RingEnginePy&)            = delete;
    RingEnginePy& operator=(const RingEnginePy&) = delete;

    void init(uint64_t stream_handle = 0);
    void start();
    void stop();

    // Allocate condition tensor after hook count is known.
    void init_hooks(uint32_t num_hooks);

    // Prepare forward: compute condition grants, enqueue H2D on main stream.
    // Runs on Python thread (GIL released).  Non-blocking, no sync.
    void prepare_forward(const std::vector<uint64_t>& hook_tensor_bytes,
                         uint64_t stream_handle);

    // Enable/disable null mode (same kernel launch, no ring writes).
    void set_null_mode(bool enabled);

    // Push metadata for the next tensor about to be hooked.
    void push_meta(TensorMeta meta);
    void push_all_metas(const std::vector<TensorMeta>& metas);
    void pop_last_meta();

    // Launch producer kernel with condition tensor gating.
    // Inserts cudaStreamWaitValue32 before kernel, and for large tensors
    // also inserts post-kernel wait for condition == 0 (ack).
    void hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                        uint32_t hook_type,
                        uint64_t stream_handle);

    // Lightweight wake-up for the drain thread.
    void notify_drain();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
