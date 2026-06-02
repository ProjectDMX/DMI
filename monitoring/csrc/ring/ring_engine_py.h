// ring/ring_engine_py.h -- Plain C++ interface for RingEngine, usable from g++.
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
#include <utility>
#include <vector>

#include "ring/tensor_meta.h"   // TensorMeta, TensorMetaFifo

// Forward-declare at::Tensor so SubmitFn can use it without including ATen here.
namespace at { class Tensor; }

namespace ring_py {

// Plain C++ mirror of ring::RingConfig -- no CUDA types.
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
    // Clone per-request slices
    bool     clone_slices               = false;
    // ClickHouse insert queue limits
    uint64_t insert_queue_max_bytes     = 4096ULL * 1024 * 1024;
    uint64_t insert_queue_max_items     = 65536;
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

struct RingFlushStats {
    uint64_t pending_entries{0};
    uint64_t pending_bytes{0};
    uint64_t cpu_payload_head{0};
    uint64_t cpu_payload_tail_committed{0};
    uint64_t total_flushes{0};
    uint64_t last_flush_entries{0};
    uint64_t last_flush_bytes{0};
    uint64_t last_flush_complete_monotonic_us{0};
    uint64_t last_force_flush_wait_us{0};
};

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
    uint64_t flush_and_wait();
    RingFlushStats get_stats() const;

    // Enable/disable null mode (same kernel launch, no ring writes).
    void set_null_mode(bool enabled);

    // Push a complete step: context (heap-allocated, ownership transferred)
    // + all hook metas.  Single lock on the FIFO.
    void push_step(StepContext* ctx, std::vector<TensorMeta>& metas);

    // Launch producer kernel unconditionally (no condition gating).
    // Space must be guaranteed by pre-forward capacity check.
    void hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                        uint32_t hook_type,
                        uint64_t stream_handle);

    // --- Runtime node-toggle (Phase B). cudaGraph_t / cudaGraphNode_t /
    //     cudaGraphExec_t are passed as uint64_t handles to keep this header
    //     CUDA-free (it is included by g++-compiled bindings.cpp). The enabled
    //     hook set is the SINGLE SOURCE OF TRUTH: apply_toggle() (device) and
    //     is_hook_enabled() (host meta gate) both read it. See
    //     docs/node_toggle_design_notes.md §1.
    //
    // enable_toggle_capture(true) makes the producer op record its kernel node
    // (via cudaStreamGetCaptureInfo) during graph capture; default off ->
    // behaviour unchanged.
    void enable_toggle_capture(bool enabled);
    bool toggle_capture_enabled() const;
    // Called from the producer op during capture (C++ only, not bound).
    void register_capture_node(uint64_t graph, int hook_type, int layer_no, uint64_t node);
    // Associate the instantiated exec (torch raw_cuda_graph_exec) with the graph
    // (torch raw_cuda_graph) the nodes were recorded under.
    void bind_graph_exec(uint64_t graph, uint64_t exec);
    // Set the enabled (hook_type, layer_no) set (the single source).
    void set_enabled_hooks(const std::vector<std::pair<int,int>>& enabled);
    // Apply the enabled set to every bound exec via cudaGraphNodeSetEnabled.
    // Caller guarantees the prior replay has completed (design-notes §1).
    int  apply_toggle();
    // Host meta-gate query: true if the hook is enabled (or toggle inactive).
    bool is_hook_enabled(int hook_type, int layer_no) const;
    uint64_t toggle_node_count() const;
    uint64_t bound_graph_count() const;          // #bound execs (0 => apply is a no-op)
    uint64_t last_apply_count() const;           // #SetEnabled calls in last apply_toggle
    bool     toggle_registry_uniform() const;    // all graphs share the same hook set
    // Full reset: registry, exec map, enabled set, toggle_active, and capture flag.
    void clear_toggle_registry();

    // Lightweight wake-up for the drain thread.
    void notify_drain();

    // Pre-forward capacity check + conditional flush.
    //
    // Called once per generate() step from _prepare_wrapper (GIL released).
    // Reads ring/staging counters internally and decides whether this step's
    // tensors fit in the ring (Case A) or must fall back to CPU-direct
    // copies (Case B).
    //
    // When a flush is needed (Case A ring-full, or Case B), this method
    // synchronises the CUDA current stream (via at::cuda::getCurrentCUDAStream)
    // and asks the drain thread to flush.  The caller does NOT need to pass a
    // stream handle -- the C++ side resolves it only when sync is required.
    //
    // Returns:
    //   0  RING_OK        -- Case A, ring has space.  No sync, no flush.
    //                        Forward may use ring::producer normally.
    //   1  RING_FLUSHED   -- Case A, ring was full.  Synced + flushed.
    //                        Ring now has space; forward may use ring::producer.
    //   2  CPU_DIRECT     -- Case B, step exceeds effective ring capacity.
    //                        Synced + flushed.  All hooks must use .cpu() path.
    //
    // For cases 0 and 1, advances cpu_payload_head_ and cpu_task_head_
    // under mgmt_mu_ to pre-allocate ring space for this step's producers.
    // Also resets the internal hook index counter for hook_no_notify.
    static constexpr int STEP_RING_OK      = 0;
    static constexpr int STEP_RING_FLUSHED = 1;
    static constexpr int STEP_CPU_DIRECT   = 2;

    int prepare_step(uint64_t step_total_bytes, uint32_t num_hooks);

    // Submit a CPU-direct tensor to drain -> p2p pipeline.
    void submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes);

    // Capacity queries (for startup warning only -- not called per-step).
    uint64_t payload_cap() const;
    uint64_t staging_cap() const;
    uint64_t task_cap() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
