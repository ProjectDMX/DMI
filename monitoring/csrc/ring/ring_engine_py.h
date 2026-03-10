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

#include "ring/tensor_meta.h"   // TensorMeta, TensorMetaFifo

// Forward-declare at::Tensor so SubmitFn can use it without including ATen here.
namespace at { class Tensor; }

namespace ring_py {

// Plain C++ mirror of ring::RingConfig — no CUDA types.
struct RingConfig {
    uint32_t task_ring_entries          = 1024;
    uint64_t payload_ring_bytes         = 256ULL * 1024 * 1024;
    uint64_t chunk_bytes                = 64ULL * 1024 * 1024;
    uint64_t pinned_pool_bytes          = 256ULL * 1024 * 1024;
    // 0 = INFINITE (spin, backpressure), 1 = TIMEOUT_DROP
    int      wait_policy                = 0;
    uint64_t no_progress_timeout_cycles = 1'000'000'000ULL;
    // 0 = COUNTER_ONLY, 1 = DROP_TASK (emit a drop entry to the callback)
    int      drop_reporting             = 1;
    // Drain thread poll timeout in µs (0 = no timeout, infinite wait).
    uint64_t drain_poll_timeout_us      = 0;
    // Whether to call notify_drain() before each forward pass from Python.
    bool     drain_notify_on_forward    = true;
};

// Called by the callback thread for each per-request tensor slice.
// Runs without the GIL — all arguments are C++ types.
using SubmitFn = std::function<void(
    const std::string& model_id,
    int32_t            shard_rank,
    const std::string& req_id,
    const std::string& act_name,
    int32_t            layer_no,
    int32_t            start_token,
    int32_t            end_token,
    at::Tensor         slice)>;

// Opaque RAII engine.  All CUDA/ATen details hidden behind the pimpl.
class RingEnginePy {
public:
    // submit_fn: called for each per-request slice after assembly.
    //   Pass an empty SubmitFn{} for null/benchmark mode (no DB writes).
    explicit RingEnginePy(RingConfig cfg, SubmitFn submit_fn);
    ~RingEnginePy();

    RingEnginePy(const RingEnginePy&)            = delete;
    RingEnginePy& operator=(const RingEnginePy&) = delete;

    // Initialise GPU buffers.  Call once before start().
    // stream_handle: raw cudaStream_t cast to uint64_t (0 = default stream).
    void init(uint64_t stream_handle = 0);

    void start();
    void stop();   // blocks until drain thread has flushed and exited

    // Enable/disable null mode.  When enabled, producer_kernel launches with
    // the same parameters but returns immediately (no ring writes, no FIFO
    // consumption).  Call this outside any CUDA graph capture region.
    // Use for warmup iterations so graph topology matches the real capture.
    void set_null_mode(bool enabled);

    // Flush barrier: block until all ring data enqueued on `stream_handle`
    // (before this call) has been fully drained and all on_tensor callbacks
    // have returned.  Does NOT stop or restart the engine.
    //
    // Caller must ensure that all producer kernels for the tensors they want
    // to wait on were launched on `stream_handle` (or a stream serialised
    // before it).  Internally:
    //   1. cudaStreamSynchronize(stream) — ensures GPU kernels + hostfuncs done
    //   2. Reads task_head (stable after sync) as the drain target
    //   3. Wakes the drain thread and waits for all D2H copies + assembler calls
    //   4. Waits for the callback thread queue to empty (all on_tensor done)
    void flush(uint64_t stream_handle = 0);

    // Push metadata for the next tensor about to be hooked.
    // Must be called before hook() so the FIFO pop order matches arrival order.
    void push_meta(TensorMeta meta);

    // Undo the last push_meta (called if hook() throws after push_meta).
    void pop_last_meta();

    // Launch the producer kernel + hostfunc for one tensor.
    //   d_ptr          — CUDA device pointer cast to uint64_t
    //   nbytes         — tensor byte size
    //   logical_task_id — opaque user ID (unused by current pipeline; pass 0)
    //   hook_type/id   — user-defined classification fields
    //   stream_handle  — raw cudaStream_t cast to uint64_t
    void hook(uint64_t d_ptr, uint64_t nbytes,
              uint64_t logical_task_id,
              uint32_t hook_type, uint32_t hook_id,
              uint64_t stream_handle);

    // Launch producer kernel WITHOUT the hostfunc notification.
    // Use this from ring_torch_op.cpp (CUDA graph path) where the hostfunc
    // would be captured as a host node causing ~18μs GPU→CPU→GPU round-trip
    // per hook per decode step.  The drain thread is notified separately
    // via notify_drain() after each forward pass.
    void hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                        uint64_t logical_task_id,
                        uint32_t hook_type, uint32_t hook_id,
                        uint64_t stream_handle);

    // Lightweight wake-up for the drain thread.  Non-blocking: sets a flag
    // and signals the condition variable so the drain thread starts
    // processing any tasks that have landed in the ring.
    //
    // Call this from Python after each forward pass (outside the CUDA graph)
    // so ring data is streamed out during generation rather than batched at
    // stop() time.  Without this, the drain thread sleeps indefinitely when
    // hook_no_notify() is used (no cudaLaunchHostFunc to wake it).
    void notify_drain();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
