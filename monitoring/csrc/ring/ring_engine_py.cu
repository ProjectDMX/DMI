// ring/ring_engine_py.cu -- Pimpl implementation of RingEnginePy.
// Compiled with nvcc so it can instantiate ring::RingEngine (needs CUDA).

#include "ring_engine_py.h"
#include "ring/ring_engine.h"
#include "ring/drain_thread.h"
#include "ring/ring_state.h"
#include "ring/ring_config.h"
#include "ring/tensor_meta.h"
#include "ring/ring_torch_op.h"
#include "ring/producer.cuh"
#include "ring/ring_debug.h"
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream

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

uint64_t RingEnginePy::flush_and_wait() {
    return impl_->engine.drain_thread().force_flush_and_wait_timed();
}

RingFlushStats RingEnginePy::get_stats() const {
    const ring::FlushStats native = impl_->engine.drain_thread().get_stats();
    RingFlushStats stats{};
    stats.pending_entries = native.pending_entries;
    stats.pending_bytes = native.pending_bytes;
    stats.cpu_payload_head = native.cpu_payload_head;
    stats.cpu_payload_tail_committed = native.cpu_payload_tail_committed;
    stats.total_flushes = native.total_flushes;
    stats.last_flush_entries = native.last_flush_entries;
    stats.last_flush_bytes = native.last_flush_bytes;
    stats.last_flush_complete_monotonic_us = native.last_flush_complete_monotonic_us;
    stats.last_force_flush_wait_us = native.last_force_flush_wait_us;
    return stats;
}

void RingEnginePy::set_null_mode(bool enabled) {
    ring::set_ring_null_mode(enabled);
}

void RingEnginePy::push_step(StepContext* ctx, std::vector<TensorMeta>& metas) {
    impl_->fifo.push_step(ctx, metas);
}

// ---------------------------------------------------------------------------
// hook_no_notify -- unconditional producer launch.
//
// No condition gating.  Space is guaranteed by the pre-forward capacity
// check in Python.  The kernel just does D2D copy + task_publish.
// ---------------------------------------------------------------------------
void RingEnginePy::hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                                  uint32_t hook_type,
                                  uint64_t stream_handle)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    RING_DBG("[hook_no_notify] idx=%u nbytes=%lu\n",
            impl_->current_hook_idx, (unsigned long)nbytes);

    impl_->current_hook_idx++;

    ring::launch_producer(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, hook_type, stream);
}

void RingEnginePy::notify_drain() {
    impl_->engine.drain_thread().notify();
}

// ---------------------------------------------------------------------------
// prepare_step -- single Python->C++ call for pre-forward capacity check.
//
// Fast path (STEP_RING_OK): reads two uint64_t counters, returns immediately.
// No stream resolution, no sync, no flush.
//
// Slow path (STEP_RING_FLUSHED / STEP_CPU_DIRECT): resolves the current CUDA
// stream via at::cuda::getCurrentCUDAStream(), synchronises it, then asks the
// drain thread to flush all pending entries.
// ---------------------------------------------------------------------------
int RingEnginePy::prepare_step(uint64_t step_total_bytes,
                               uint32_t num_hooks)
{
    impl_->current_hook_idx = 0;

    const uint64_t pcap = impl_->engine.payload_cap();
    const uint64_t scap = impl_->engine.staging_cap();
    const uint64_t effective_cap = std::min(pcap, scap);

    auto& drain = impl_->engine.drain_thread();

    if (step_total_bytes <= effective_cap) {
        // Case A: step fits in both ring AND staging.
        const uint64_t available = pcap -
            (drain.cpu_payload_head() - drain.cpu_payload_tail_committed());
        if (step_total_bytes <= available) {
            // Pre-allocate: advance CPU-side head counters so the next
            // prepare_step sees correct available space.
            drain.reserve(step_total_bytes, num_hooks);
            return STEP_RING_OK;  // fast path -- no CUDA or thread interaction
        }
        // Ring currently full from prior steps.  Sync main stream so all
        // producer kernels finish writing, then flush to free ring space.
        cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(ms);
        drain.force_flush_and_wait();
        drain.reserve(step_total_bytes, num_hooks);
        return STEP_RING_FLUSHED;
    }

    // Case B: step exceeds effective capacity.  Sync + flush, then caller
    // must set cpu_direct=True so all hooks use the eager .cpu() path.
    // No reserve -- cpu_direct hooks don't use the ring.
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    drain.force_flush_and_wait();
    return STEP_CPU_DIRECT;
}

void RingEnginePy::submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes) {
    impl_->engine.drain_thread().submit_cpu_direct(std::move(cpu_tensor), tensor_bytes);
}

// ---------------------------------------------------------------------------
// Capacity queries (startup only, not per-step)
// ---------------------------------------------------------------------------
uint64_t RingEnginePy::payload_cap() const {
    return impl_->engine.payload_cap();
}

uint64_t RingEnginePy::staging_cap() const {
    return impl_->engine.staging_cap();
}

}  // namespace ring_py
