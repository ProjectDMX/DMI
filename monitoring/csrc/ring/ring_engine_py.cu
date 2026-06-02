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
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <set>
#include <unordered_map>
#include <vector>
#include <mutex>

// Forward-declare symbols from producer.cu
namespace ring {
cudaError_t set_ring_null_mode(bool enabled);
}  // namespace ring

namespace ring_py {

// ---------------------------------------------------------------------------
struct RingEnginePy::Impl {
    TensorMetaFifo   fifo;
    ring::RingEngine engine;
    uint32_t         current_hook_idx{0};

    // --- node-toggle registry (Phase B) ---
    struct RegEntry { int hook_type; int layer_no; cudaGraphNode_t node;
                      bool last_enabled{true}; };  // device default after instantiate
    std::unordered_map<cudaGraph_t, std::vector<RegEntry>> reg_nodes;  // per captured graph
    std::unordered_map<cudaGraph_t, cudaGraphExec_t>       reg_exec;   // graph -> exec
    std::set<std::pair<int,int>>                           registered_hooks;  // (ht,layer) with a captured node
    std::set<std::pair<int,int>>                           enabled_hooks;  // single source
    bool                                                   toggle_active{false};
    bool                                                   toggle_capture{false};
    uint64_t                                               last_apply_count{0};
    mutable std::mutex                                     toggle_mu;

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
    // NOTE: this branch's DrainThread exposes only `void force_flush_and_wait()`
    // (no timed variant / FlushStats accessor that this file was written against
    // -- a pre-existing mismatch, see docs/node_toggle_impl_plan_local.md Phase A).
    // Stub the timing return as 0 (only feeds a diagnostic _flush_ms in
    // generate.py); behaviour/correctness of the flush itself is unchanged.
    impl_->engine.drain_thread().force_flush_and_wait();
    return 0;
}

RingFlushStats RingEnginePy::get_stats() const {
    // See note in flush_and_wait: DrainThread has no get_stats()/FlushStats on
    // this branch. Return zeroed stats (ring get_stats is bound but unused from
    // Python). TODO: reconcile drain timed-flush + stats API across branches.
    RingFlushStats stats{};
    return stats;
}

void RingEnginePy::set_null_mode(bool enabled) {
    // null_mode flips a __device__ flag via cudaMemcpyToSymbol, which goes
    // through the legacy default stream and does NOT synchronize with PyTorch's
    // non-blocking compute streams.  Sync before (drain pending producer kernels
    // that still need the old value) and after (make the new value visible
    // before the next launch).  Ported from ring_full_tp d0a99a5.
    //
    // NOTE: this is the discipline for the *null_mode device-flag* path only.
    // Post-capture node-toggle has a DIFFERENT requirement (prior replay must be
    // complete before mutating the exec graph + meta-FIFO lockstep), not this
    // default-stream sync -- see docs/node_toggle_design_notes.md §0.
    //
    // This carries race-fix semantics, so every step is checked: a silent
    // failure here would corrupt capture without any Python-visible signal.
    auto ck = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("set_null_mode: ") + what + ": " +
                                     cudaGetErrorString(e));
    };
    ck(cudaDeviceSynchronize(),         "pre-sync");
    ck(ring::set_ring_null_mode(enabled), "cudaMemcpyToSymbol(g_ring_null_mode)");
    ck(cudaDeviceSynchronize(),         "post-sync");
}

// --- Runtime node-toggle (Phase B) ----------------------------------------
void RingEnginePy::enable_toggle_capture(bool enabled) {
    { std::lock_guard<std::mutex> lk(impl_->toggle_mu); impl_->toggle_capture = enabled; }
    ring_set_toggle_capture(enabled);   // tell the producer op to record nodes
}

bool RingEnginePy::toggle_capture_enabled() const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    return impl_->toggle_capture;
}

void RingEnginePy::register_capture_node(uint64_t graph, int hook_type, int layer_no, uint64_t node) {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    impl_->reg_nodes[reinterpret_cast<cudaGraph_t>(graph)].push_back(
        Impl::RegEntry{hook_type, layer_no, reinterpret_cast<cudaGraphNode_t>(node)});
    impl_->registered_hooks.insert({hook_type, layer_no});
}

void RingEnginePy::bind_graph_exec(uint64_t graph, uint64_t exec) {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    impl_->reg_exec[reinterpret_cast<cudaGraph_t>(graph)] =
        reinterpret_cast<cudaGraphExec_t>(exec);
}

void RingEnginePy::set_enabled_hooks(const std::vector<std::pair<int,int>>& enabled) {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    impl_->enabled_hooks.clear();
    for (const auto& p : enabled) impl_->enabled_hooks.insert(p);
    impl_->toggle_active = true;
}

int RingEnginePy::apply_toggle() {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    cudaError_t first = cudaSuccess;
    uint64_t applied = 0;
    for (auto& kv : impl_->reg_exec) {
        cudaGraphExec_t exec = kv.second;
        auto it = impl_->reg_nodes.find(kv.first);
        if (it == impl_->reg_nodes.end()) continue;
        for (auto& e : it->second) {
            const bool on = impl_->enabled_hooks.count({e.hook_type, e.layer_no}) > 0;
            // Diff: only touch nodes whose desired state changed since the last
            // apply (last_enabled tracks the exec's actual state -- DMI is the
            // only toggler of its nodes). Adaptive flips of a few hooks then cost
            // O(#changed) SetEnabled calls instead of O(#nodes).
            if (on == e.last_enabled) continue;
            cudaError_t err = cudaGraphNodeSetEnabled(exec, e.node, on ? 1u : 0u);
            if (err != cudaSuccess) { if (first == cudaSuccess) first = err; continue; }
            e.last_enabled = on;   // update only on success
            ++applied;
        }
    }
    impl_->last_apply_count = applied;
    return static_cast<int>(first);
}

uint64_t RingEnginePy::bound_graph_count() const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    return impl_->reg_exec.size();
}

uint64_t RingEnginePy::last_apply_count() const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    return impl_->last_apply_count;
}

bool RingEnginePy::toggle_registry_uniform() const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    // True iff every captured graph registered the same set of (hook_type,layer).
    // The meta gate keys on (ht,layer) globally; if graphs differ, a hook present
    // in one graph but absent from the one replayed this step would push a meta
    // with no payload -> desync. vLLM captures all hooks in every graph (uniform).
    const std::set<std::pair<int,int>>* ref = nullptr;
    std::set<std::pair<int,int>> ref_set;
    for (const auto& kv : impl_->reg_nodes) {
        std::set<std::pair<int,int>> s;
        for (const auto& e : kv.second) s.insert({e.hook_type, e.layer_no});
        if (!ref) { ref_set = std::move(s); ref = &ref_set; }
        else if (s != ref_set) return false;
    }
    return true;
}

bool RingEnginePy::is_hook_enabled(int hook_type, int layer_no) const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    if (!impl_->toggle_active) return true;   // toggle inactive -> all hooks on
    // Desync guard (#14): the meta gate returns true only if the hook is BOTH
    // enabled AND actually registered (has a captured producer node). Otherwise
    // an enabled-but-uncaptured hook would push a meta with no matching payload
    // -> p2p desync. With this, "meta pushed" <=> "a producer fires" in all cases.
    const std::pair<int,int> k{hook_type, layer_no};
    return impl_->enabled_hooks.count(k) > 0 && impl_->registered_hooks.count(k) > 0;
}

uint64_t RingEnginePy::toggle_node_count() const {
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    uint64_t n = 0;
    for (const auto& kv : impl_->reg_nodes) n += kv.second.size();
    return n;
}

void RingEnginePy::clear_toggle_registry() {
    {
        std::lock_guard<std::mutex> lk(impl_->toggle_mu);
        impl_->reg_nodes.clear();
        impl_->reg_exec.clear();
        impl_->registered_hooks.clear();
        // Full reset: also drop the enabled set + deactivate, so a stale gate
        // (is_hook_enabled) can't skip every meta after a clear (#2). Disable
        // capture too so a later capture doesn't record into a cleared engine (#3).
        impl_->enabled_hooks.clear();
        impl_->toggle_active = false;
        impl_->toggle_capture = false;
        impl_->last_apply_count = 0;
    }
    ring_set_toggle_capture(false);
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
    const uint64_t tcap = impl_->engine.task_cap();

    auto& drain = impl_->engine.drain_thread();

    // Case B: single step exceeds capacity (payload OR task entries).
    // Must fall back to cpu_direct.
    if (step_total_bytes > effective_cap || num_hooks > tcap) {
        cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(ms);
        drain.force_flush_and_wait();
        return STEP_CPU_DIRECT;
    }

    // Case A: step fits.  Check available space for BOTH payload AND tasks.
    const uint64_t payload_avail = pcap -
        (drain.cpu_payload_head() - drain.cpu_payload_tail_committed());
    const uint64_t task_avail = tcap -
        (drain.cpu_task_head() - drain.cpu_task_tail_committed());

    if (step_total_bytes <= payload_avail && num_hooks <= task_avail) {
        drain.reserve(step_total_bytes, num_hooks);
        return STEP_RING_OK;  // fast path -- no CUDA or thread interaction
    }

    // Either payload or task ring full from prior steps.  Sync main
    // stream so all producer kernels finish writing, then flush.
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    drain.force_flush_and_wait();
    drain.reserve(step_total_bytes, num_hooks);
    return STEP_RING_FLUSHED;
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

uint64_t RingEnginePy::task_cap() const {
    return impl_->engine.task_cap();
}

}  // namespace ring_py
