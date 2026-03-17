// ring/ring_engine_py.cu — Pimpl implementation of RingEnginePy.
// Compiled with nvcc so it can instantiate ring::RingEngine (needs CUDA).

#include "ring_engine_py.h"
#include "ring/ring_engine.h"
#include "ring/drain_thread.h"
#include "ring/ring_state.h"
#include "ring/ring_config.h"
#include "ring/tensor_meta.h"
#include "ring/ring_torch_op.h"
#include "ring/producer.cuh"   // COND_* constants, launch_producer
#include "ring/ring_debug.h"
#include <cuda.h>              // cuStreamWaitValue32, CU_STREAM_WAIT_VALUE_*

// Forward-declare symbols from producer.cu
namespace ring {
void set_ring_null_mode(bool enabled);
}  // namespace ring

namespace ring_py {

// ---------------------------------------------------------------------------
struct RingEnginePy::Impl {
    TensorMetaFifo   fifo;
    ring::RingEngine engine;
    uint32_t         current_hook_idx{0};

    Impl(ring::RingConfig cfg, SubmitFn sf)
        : engine(std::move(cfg), fifo, std::move(sf))
    {}
};

// ---------------------------------------------------------------------------
static ring::RingConfig convert(const RingConfig& c) {
    ring::RingConfig r{};
    r.task_ring_entries           = c.task_ring_entries;
    r.payload_ring_bytes          = c.payload_ring_bytes;
    r.pinned_staging_bytes        = c.pinned_staging_bytes;
    r.drain_poll_timeout_us       = c.drain_poll_timeout_us;
    r.drain_flush.task_ratio      = c.drain_flush_task_ratio;
    r.drain_flush.payload_ratio   = c.drain_flush_payload_ratio;
    r.drain_flush.entry_threshold = c.drain_flush_entry_threshold;
    r.drain_flush.byte_threshold  = c.drain_flush_byte_threshold;
    r.drain_flush.timeout_us      = c.drain_flush_timeout_us;
    r.bypass_budget_bytes         = c.bypass_budget_bytes;
    r.clone_slices                = c.clone_slices;
    r.insert_queue_max_bytes      = c.insert_queue_max_bytes;
    r.insert_queue_max_items      = c.insert_queue_max_items;
    return r;
}

// ---------------------------------------------------------------------------
RingEnginePy::RingEnginePy(RingConfig cfg, SubmitFn submit_fn) {
    impl_ = std::make_unique<Impl>(convert(cfg), std::move(submit_fn));
}

RingEnginePy::~RingEnginePy() = default;

void RingEnginePy::init(uint64_t stream_handle) {
    impl_->engine.init(reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::start() {
    ring_diag_reset_host_counters();
    impl_->engine.start();
}

void RingEnginePy::stop() {
    impl_->engine.stop();
#if RING_DEBUG
    ring_diag_print_host_counters();
#endif
}

void RingEnginePy::init_hooks(uint32_t num_hooks) {
    RING_DBG("[ring_engine] init_hooks(%u)\n", num_hooks);
    impl_->engine.init_hooks(num_hooks);
}

void RingEnginePy::prepare_forward(const std::vector<uint64_t>& hook_tensor_bytes,
                                   uint64_t stream_handle) {
    impl_->current_hook_idx = 0;
    cudaStream_t main_stream = reinterpret_cast<cudaStream_t>(stream_handle);
    impl_->engine.drain_thread().prepare_forward(hook_tensor_bytes, main_stream);
}

void RingEnginePy::set_null_mode(bool enabled) {
    ring::set_ring_null_mode(enabled);
    // Also tell drain thread so prepare_forward grants unconditionally
    // in null mode (avoids capacity deadlock with large batches).
    // When turning off, reset all conditions to 0 on the main stream.
    cudaStream_t main_stream = nullptr;  // default stream
    impl_->engine.drain_thread().set_null_mode(enabled, main_stream);
}

void RingEnginePy::push_meta(TensorMeta meta) {
    impl_->fifo.push(std::move(meta));
}

void RingEnginePy::push_all_metas(const std::vector<TensorMeta>& metas) {
    for (auto& m : metas)
        impl_->fifo.push(m);
}

void RingEnginePy::pop_last_meta() {
    TensorMeta discard;
    impl_->fifo.pop_last(discard);
}

// ---------------------------------------------------------------------------
// hook_no_notify — condition-gated producer launch.
//
// Condition values (see producer.cuh):
//   0 = COND_RESET    (not ready / kernel reset)
//   1 = COND_PENDING  (large kernel done, drain must ack)
//   3 = COND_GRANT_TASK_ONLY (large tensor: task slot available)
//   4 = COND_GRANT_FULL      (normal tensor: task slot + ring space)
//
// Normal:  wait(>=4) → kernel → kernel writes 0
// Large:   wait(>=3) → kernel → kernel writes 1 → wait(==0) → drain acks
// ---------------------------------------------------------------------------
void RingEnginePy::hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                                  uint32_t hook_type,
                                  uint64_t stream_handle)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    uint32_t hook_idx = impl_->current_hook_idx++;

    uint64_t padded = ring::align_up(nbytes, ring::PAYLOAD_ALIGN);
    uint64_t payload_cap = impl_->engine.payload_cap();
    uint64_t staging_cap = impl_->engine.staging_cap();
    // Bypass ring+staging if tensor exceeds EITHER buffer.
    bool large_bypass = (padded > payload_cap) || (padded > staging_cap);

    uint32_t* d_cond = impl_->engine.d_condition();
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    if (hook_idx < 3 || (hook_idx % 100 == 0)) {
        RING_DBG("[hook_no_notify] idx=%u nbytes=%lu padded=%lu large=%d "
                "d_cond=%p num_hooks=%u\n",
                hook_idx, (unsigned long)nbytes, (unsigned long)padded,
                (int)large_bypass, (void*)d_cond, impl_->engine.num_hooks());
    }

    if (d_cond && hook_idx < impl_->engine.num_hooks()) {
        CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(d_cond + hook_idx);
        CUresult rc;
        if (large_bypass) {
            rc = cuStreamWaitValue32(cu_stream, addr,
                                ring::COND_GRANT_TASK_ONLY,
                                CU_STREAM_WAIT_VALUE_GEQ);
        } else {
            rc = cuStreamWaitValue32(cu_stream, addr,
                                ring::COND_GRANT_FULL,
                                CU_STREAM_WAIT_VALUE_GEQ);
        }
        if (rc != CUDA_SUCCESS && hook_idx < 3) {
            const char* err_str = nullptr;
            cuGetErrorString(rc, &err_str);
            RING_DBG("[hook_no_notify] cuStreamWaitValue32 FAILED idx=%u rc=%d: %s\n",
                    hook_idx, (int)rc, err_str ? err_str : "unknown");
        }
    } else if (hook_idx < 3) {
        RING_DBG("[hook_no_notify] WARNING: skipping wait — d_cond=%p idx=%u num=%u\n",
                (void*)d_cond, hook_idx, impl_->engine.num_hooks());
    }

    ring::launch_producer(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, hook_type, large_bypass,
        d_cond, hook_idx, stream);

    if (large_bypass && d_cond && hook_idx < impl_->engine.num_hooks()) {
        CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(d_cond + hook_idx);
        cuStreamWaitValue32(cu_stream, addr,
                            ring::COND_RESET,
                            CU_STREAM_WAIT_VALUE_EQ);
    }
}

void RingEnginePy::notify_drain() {
    impl_->engine.drain_thread().notify();
}

}  // namespace ring_py
