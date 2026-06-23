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

    // Enable/disable null mode (same kernel launch, no ring writes).
    void set_null_mode(bool enabled);

    // Push a complete step: context (heap-allocated, ownership transferred)
    // + all hook metas.  Single lock on the FIFO.
    void push_step(StepContext* ctx, std::vector<TensorMeta>& metas);

    // Launch producer kernel unconditionally (no condition gating).
    // Space must be guaranteed by pre-forward capacity check.
    // Three variants matching the three torch ops.

    // Static: copies all `nbytes`; today's behavior.
    void hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                        uint32_t hook_type,
                        uint64_t stream_handle);

    // Prefix: reads row_count[0] from device, copies
    // (row_count[0] * row_bytes) bytes.  Used by ring::producer_prefix.
    void hook_no_notify_prefix(uint64_t d_ptr, uint64_t nbytes_upper,
                                uint64_t row_count_dev_ptr,
                                uint64_t row_bytes,
                                uint32_t hook_type,
                                uint64_t stream_handle);

    // Chunked: K>1 chunked-suffix; per-chunk bytes from
    // chunk_bytes_dev_ptr[K].  Used by ring::producer_chunked.
    void hook_no_notify_chunked(uint64_t d_ptr, uint64_t nbytes_upper,
                                 uint64_t chunk_bytes_dev_ptr,
                                 uint32_t K,
                                 uint32_t hook_type,
                                 uint64_t stream_handle);

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
    // Step's total bytes > ring capacity (or n_hooks > task ring entries):
    // even an empty ring can't hold it.  Caller falls back to the per-hook
    // safety net in HookPoint.forward via transport.force_eager = True.
    static constexpr int STEP_OVERSIZED    = 2;

    int prepare_step(uint64_t step_total_bytes, uint32_t num_hooks);

    // Submit a CPU-direct tensor to drain -> p2p pipeline.
    void submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes);

    // Capacity queries (for startup warning only -- not called per-step).
    uint64_t payload_cap() const;
    uint64_t staging_cap() const;
    uint64_t task_cap() const;

    // Return a torch.Tensor view of the GPU payload buffer (uint8,
    // length = payload_cap()).  No copy, no ownership transfer -- the
    // buffer continues to be owned by the engine.  Used as the
    // `Tensor(a!)` mutation alias passed to every producer op call,
    // which gives AOT autograd a real R/W dependency that prevents
    // inductor from reordering successive producer launches.
    at::Tensor payload_tensor() const;

    // ---- Runtime queries / actions used by the safety-net branch in
    //      HookPoint.forward (eager-only path).  Never called during
    //      CUDA-graph capture or replay.

    // Free bytes in the payload ring not currently reserved and not
    // pending drain.  CPU-only read.
    uint64_t available_capacity() const;

    // Per-hook reservation: claim `nbytes` of payload ring + 1 task entry
    // for an upcoming producer kernel launch.  Used by the safety net
    // when force_eager is on and the spec is dynamic-shape.  Advances
    // cpu_payload_head/cpu_task_head atomically.
    void reserve_one(uint64_t nbytes);

    // Synchronise the current CUDA stream + force drain to process all
    // outstanding entries.  Blocking; the Python binding releases the
    // GIL.  Used by safety-net branches that need to free ring space or
    // ensure FIFO ordering before consuming the next meta out-of-band.
    void flush_and_wait();

    // Like flush_and_wait(), but additionally barrier through the
    // p2p -> SubmitFn stage so every slice produced so far has reached the
    // sink on return.  Returns 0 = delivered, 1 = timed out (0 = wait forever).
    // The adapter calls this (not flush_and_wait) before finalizing a finished
    // request so its committed-token slice is guaranteed delivered.  "Delivered"
    // = every submit_fn was called, NOT that each succeeded; check
    // submit_exceptions()/sink_failed() for per-request sink errors.
    int drain_to_sink_and_wait(uint32_t timeout_ms);

    // Fail-loud sink error surface, observable from Python.
    uint64_t    submit_exceptions() const;   // count of SubmitFn exceptions
    std::string last_sink_error()   const;   // most recent error message
    void set_abort_on_sink_error(bool enabled);
    bool sink_failed() const;                // armed AND >=1 exception seen

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
