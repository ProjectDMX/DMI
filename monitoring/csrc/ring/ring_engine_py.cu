// ring/ring_engine_py.cu — Pimpl implementation of RingEnginePy.
// Compiled with nvcc so it can instantiate ring::RingEngine (needs CUDA).

#include "ring_engine_py.h"
#include "ring/ring_engine.h"
#include "ring/drain_thread.h"
#include "ring/ring_state.h"
#include "ring/ring_config.h"
#include "ring/tensor_meta.h"
#include "ring/ring_torch_op.h"

// Forward-declare symbols from producer.cu so we don't pull in producer.cuh.
namespace ring {
void launch_producer_with_notify(
    const RingState& ring, const uint8_t* d_src, uint64_t src_bytes,
    uint64_t logical_task_id, uint32_t hook_type, uint32_t hook_id,
    cudaHostFn_t notify_fn, void* notify_arg, cudaStream_t stream);
void launch_producer(
    const RingState& ring, const uint8_t* d_src, uint64_t src_bytes,
    uint64_t logical_task_id, uint32_t hook_type, uint32_t hook_id,
    cudaStream_t stream);
void set_ring_null_mode(bool enabled);
void diag_reset_hook_counters();
void diag_print_hook_counters();
}  // namespace ring

namespace ring_py {

// ---------------------------------------------------------------------------
struct RingEnginePy::Impl {
    TensorMetaFifo   fifo;
    ring::RingEngine engine;

    Impl(ring::RingConfig cfg, SubmitFn sf)
        : engine(std::move(cfg), fifo, std::move(sf))
    {}
};

// ---------------------------------------------------------------------------
static ring::RingConfig convert(const RingConfig& c) {
    ring::RingConfig r{};
    r.task_ring_entries           = c.task_ring_entries;
    r.payload_ring_bytes          = c.payload_ring_bytes;
    r.chunk_bytes                 = c.chunk_bytes;
    r.pinned_staging_bytes        = c.pinned_staging_bytes;
    r.wait_policy                 = static_cast<ring::WaitPolicy>(c.wait_policy);
    r.no_progress_timeout_cycles  = c.no_progress_timeout_cycles;
    r.drop_reporting              = static_cast<ring::DropReporting>(c.drop_reporting);
    r.drain_poll_timeout_us       = c.drain_poll_timeout_us;
    r.drain_notify_on_forward     = c.drain_notify_on_forward;
    r.drain_flush.task_ratio      = c.drain_flush_task_ratio;
    r.drain_flush.payload_ratio   = c.drain_flush_payload_ratio;
    r.drain_flush.entry_threshold = c.drain_flush_entry_threshold;
    r.drain_flush.byte_threshold  = c.drain_flush_byte_threshold;
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
    ring::diag_reset_hook_counters();
    ring_diag_reset_host_counters();
    impl_->engine.start();
}

void RingEnginePy::stop() {
    impl_->engine.stop();
    ring_diag_print_host_counters();
    ring::diag_print_hook_counters();
}

void RingEnginePy::flush(uint64_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Phase 1: wait for all producer kernels on this stream to complete.
    cudaStreamSynchronize(stream);

    // Phase 2: wake the drain thread and let it flush everything.
    impl_->engine.drain_thread().notify();

    // Phase 3: stop + restart to wait for all processing.
    // TODO: implement a proper flush barrier without stop/start cycle.
    // For now, stop() blocks until drain + p2p are done.
    impl_->engine.stop();
    impl_->engine.start();
}

void RingEnginePy::set_null_mode(bool enabled) {
    ring::set_ring_null_mode(enabled);
}

void RingEnginePy::push_meta(TensorMeta meta) {
    impl_->fifo.push(std::move(meta));
}

void RingEnginePy::pop_last_meta() {
    TensorMeta discard;
    impl_->fifo.pop_last(discard);
}

void RingEnginePy::hook(uint64_t d_ptr, uint64_t nbytes,
                        uint64_t logical_task_id,
                        uint32_t hook_type, uint32_t hook_id,
                        uint64_t stream_handle)
{
    launch_producer_with_notify(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, logical_task_id, hook_type, hook_id,
        ring::DrainThread::hostfunc_cb,
        &impl_->engine.drain_thread(),
        reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                                  uint64_t logical_task_id,
                                  uint32_t hook_type, uint32_t hook_id,
                                  uint64_t stream_handle)
{
    ring::launch_producer(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, logical_task_id, hook_type, hook_id,
        reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::notify_drain() {
    impl_->engine.drain_thread().notify();
}

}  // namespace ring_py
