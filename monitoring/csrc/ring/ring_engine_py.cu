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
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>

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

    // Snapshot of the device-side actual_bytes_counter as of the last
    // prepare_step call.  Used to compute the per-step delta of bytes the
    // producer actually wrote, for reclamation accounting when a step's
    // reservation overshoots its actual writes.  Consumed by future
    // GPU-side-strip flows where the producer's src_bytes is set from a
    // device tensor at execution time and the CPU can't know it upfront.
    uint64_t         last_counter_read{0};

    // Cached torch.Tensor view of the payload buffer.  Built once at
    // engine init; returned by payload_tensor().  Used as the
    // Tensor(a!) mutation alias passed to every producer op call.
    at::Tensor       payload_view;

    // --- node-toggle registry (Phase B) ---
    struct RegEntry { int hook_type; int layer_no; cudaGraphNode_t node;
                      bool last_enabled{true}; };  // device default after instantiate
    std::unordered_map<cudaGraph_t, std::vector<RegEntry>> reg_nodes;  // per captured graph
    std::unordered_map<cudaGraph_t, cudaGraphExec_t>       reg_exec;   // graph -> exec
    std::set<std::pair<int,int>>                           registered_hooks;  // (ht,layer) captured
    std::set<std::pair<int,int>>                           enabled_hooks;     // single source
    bool                                                   toggle_active{false};
    bool                                                   toggle_capture{false};
    uint64_t                                               last_apply_count{0};
    // --- Phase 4: lazy per-graph apply ---
    // target_version bumps on every enabled-set change; each graph tracks the
    // version it was last applied to. ensure_graph_current() applies the current
    // enabled set to ONE graph only when stale (just before that graph replays),
    // so a reconfigure costs O(changed x graphs-actually-used). replay_event[g]
    // is recorded after each replay so ensure can wait for the prior replay to
    // finish before mutating the exec (SetEnabled on an executing exec is UB).
    uint64_t                                               target_version{0};
    std::unordered_map<cudaGraph_t, uint64_t>              applied_version;
    std::unordered_map<cudaGraph_t, cudaEvent_t>           replay_event;
    mutable std::mutex                                     toggle_mu;

    Impl(ring::RingConfig cfg, SubmitFn sf)
        : engine(std::move(cfg), fifo, std::move(sf))
    {
        const auto& state = engine.ring_state();
        int dev_idx = 0;
        cudaGetDevice(&dev_idx);
        payload_view = at::from_blob(
            state.payload_buf,
            {static_cast<int64_t>(state.payload_cap)},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA, dev_idx));
    }
    ~Impl() {
        for (auto& kv : replay_event) if (kv.second) cudaEventDestroy(kv.second);
    }
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

void RingEnginePy::set_null_mode(bool enabled) {
    // cudaMemcpyToSymbol goes through the legacy default stream, which does
    // NOT synchronize with PyTorch's non-blocking compute streams.  Sync
    // before to drain pending producer kernels that need the old value,
    // and after to ensure the new value is visible before the next launch.
    cudaDeviceSynchronize();
    ring::set_ring_null_mode(enabled);
    cudaDeviceSynchronize();
}



void RingEnginePy::push_step(StepContext* ctx, std::vector<TensorMeta>& metas) {
    impl_->fifo.push_step(ctx, metas);
}

RingFlushStats RingEnginePy::get_stats() const {
    // Reserve-invariant probe: expose the live drain-thread head/tail counters
    // so a test can confirm the ring fully drains (head == tail_committed) after
    // a flush -> reserve() matched actual producer writes.
    RingFlushStats stats{};
    auto& drain = impl_->engine.drain_thread();
    stats.cpu_payload_head           = drain.cpu_payload_head();
    stats.cpu_payload_tail_committed = drain.cpu_payload_tail_committed();
    stats.cpu_task_head              = drain.cpu_task_head();
    stats.cpu_task_tail_committed    = drain.cpu_task_tail_committed();
    return stats;
}

// --- Runtime node-toggle (Phase B) -----------------------------------------
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
    ++impl_->target_version;   // mark all graphs stale for lazy ensure (no-op for eager)
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

std::vector<int> RingEnginePy::effective_enabled_mask(
    const std::vector<std::pair<int,int>>& query) const {
    // Batched is_hook_enabled (same semantics, one lock, one pybind crossing):
    // toggle inactive -> all on; else enabled AND registered (#14 guard).
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    const bool inactive = !impl_->toggle_active;
    std::vector<int> mask;
    mask.reserve(query.size());
    for (const auto& k : query) {
        mask.push_back((inactive ||
                        (impl_->enabled_hooks.count(k) > 0 &&
                         impl_->registered_hooks.count(k) > 0)) ? 1 : 0);
    }
    return mask;
}

int RingEnginePy::ensure_graph_current(uint64_t graph) {
    // Phase 4 lazy: apply the current enabled set to ONE graph, only if it is
    // stale (applied_version != target_version). Called just before that graph
    // replays. Same per-node diff as apply_toggle, but scoped to one graph.
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    auto av = impl_->applied_version.find(g);
    const uint64_t cur = (av == impl_->applied_version.end()) ? 0 : av->second;
    if (cur == impl_->target_version) return 0;   // current -> fast no-op (common path)

    auto ex = impl_->reg_exec.find(g);
    if (ex == impl_->reg_exec.end()) return 0;     // unregistered/unbound graph -> skip
    auto it = impl_->reg_nodes.find(g);
    if (it == impl_->reg_nodes.end()) return 0;

    // Event guard: the prior replay of THIS exec must be complete before we
    // mutate it (host can run ahead of the GPU). Wait on its last-replay event.
    auto ev = impl_->replay_event.find(g);
    if (ev != impl_->replay_event.end() && ev->second) cudaEventSynchronize(ev->second);

    cudaError_t first = cudaSuccess;
    for (auto& e : it->second) {
        const bool on = impl_->enabled_hooks.count({e.hook_type, e.layer_no}) > 0;
        if (on == e.last_enabled) continue;
        cudaError_t err = cudaGraphNodeSetEnabled(ex->second, e.node, on ? 1u : 0u);
        if (err != cudaSuccess) { if (first == cudaSuccess) first = err; continue; }
        e.last_enabled = on;
    }
    // (#2) Mark this graph current ONLY on full success. If any SetEnabled
    // failed, leave applied_version stale so a later ensure retries the
    // still-unapplied nodes (last_enabled already tracks per-node success, so
    // the retry is a no-op for the ones that did flip). The caller (replay
    // hook) treats a nonzero return as a FATAL desync and terminates, but the
    // stale version is the correct state regardless.
    if (first == cudaSuccess)
        impl_->applied_version[g] = impl_->target_version;
    return static_cast<int>(first);
}

void RingEnginePy::record_replay_event(uint64_t graph) {
    // Record a (timing-disabled) event on the current stream right after a
    // graph's replay, so a later ensure_graph_current() can wait for it before
    // mutating that exec.
    std::lock_guard<std::mutex> lk(impl_->toggle_mu);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    if (impl_->reg_exec.find(g) == impl_->reg_exec.end()) return;  // only track bound graphs
    cudaEvent_t& ev = impl_->replay_event[g];
    if (ev == nullptr) {
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return;
    }
    cudaEventRecord(ev, at::cuda::getCurrentCUDAStream().stream());
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
        // Phase 4 lazy state: reset versions + destroy per-graph replay events.
        impl_->target_version = 0;
        impl_->applied_version.clear();
        for (auto& kv : impl_->replay_event) if (kv.second) cudaEventDestroy(kv.second);
        impl_->replay_event.clear();
    }
    ring_set_toggle_capture(false);
}

// ---------------------------------------------------------------------------
// hook_no_notify (3 variants) -- unconditional producer launches.
//
// No condition gating.  Space is guaranteed by the pre-forward capacity
// check in Python.  Each variant maps to one torch op.
// ---------------------------------------------------------------------------
void RingEnginePy::hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                                  uint32_t hook_type,
                                  uint64_t stream_handle)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_static] idx=%u nbytes=%lu\n",
            impl_->current_hook_idx, (unsigned long)nbytes);
    impl_->current_hook_idx++;
    ring::launch_producer_static(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, hook_type, stream);
}

void RingEnginePy::hook_no_notify_prefix(uint64_t d_ptr, uint64_t nbytes_upper,
                                          uint64_t row_count_dev_ptr,
                                          uint64_t row_bytes,
                                          uint32_t hook_type,
                                          uint64_t stream_handle)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_prefix] idx=%u nbytes_upper=%lu row_bytes=%lu\n",
            impl_->current_hook_idx, (unsigned long)nbytes_upper,
            (unsigned long)row_bytes);
    impl_->current_hook_idx++;
    ring::launch_producer_prefix(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes_upper,
        reinterpret_cast<const int64_t*>(row_count_dev_ptr),
        row_bytes,
        hook_type, stream);
}

void RingEnginePy::hook_no_notify_chunked(uint64_t d_ptr, uint64_t nbytes_upper,
                                           uint64_t chunk_bytes_dev_ptr,
                                           uint32_t K,
                                           uint32_t hook_type,
                                           uint64_t stream_handle)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_chunked] idx=%u nbytes_upper=%lu K=%u\n",
            impl_->current_hook_idx, (unsigned long)nbytes_upper, K);
    impl_->current_hook_idx++;
    ring::launch_producer_chunked(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes_upper,
        reinterpret_cast<const int64_t*>(chunk_bytes_dev_ptr),
        K,
        hook_type, stream);
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
// Slow path (STEP_RING_FLUSHED / STEP_OVERSIZED): resolves the current CUDA
// stream via at::cuda::getCurrentCUDAStream(), synchronises it, then asks the
// drain thread to flush all pending entries.
// ---------------------------------------------------------------------------
int RingEnginePy::prepare_step(uint64_t step_total_bytes,
                               uint32_t num_hooks)
{
    impl_->current_hook_idx = 0;

    // actual_bytes_counter reclamation: DISABLED for now (see below).
    //
    // The counter exists to reclaim ring space when a step's reservation
    // OVER-estimates what the producer actually writes.  That only happens for
    // producers whose written byte count the CPU cannot size up front -- i.e.
    // variable-byte / EP "chunked" producers that reserve an upper bound.  No
    // hook currently uses that path: the vLLM adapter only wires the prefix
    // producer (CPU-known actual_q_len * row_bytes) and the basic producer
    // (CPU-known x.nbytes()), both of which reserve exactly what they write and
    // need no reclamation.  The reclamation consumer was also never landed, so
    // the delta is unused.
    //
    // Reading the counter here is NOT free: it is a host dereference of a
    // cudaMallocManaged page whose preferred location is the GPU and which the
    // producer writes every step, so the read forces a UVM coherence stall
    // (measured ~430 us/step on Llama-8B -- effectively a per-step implicit GPU
    // sync, despite no explicit cudaStreamSynchronize).  Keep it commented out
    // until a chunked-style producer AND a reclamation consumer actually exist;
    // when they do, read the counter OFF the prepare_step critical path (e.g.
    // on the drain thread, which already synchronizes) rather than here.
    //
    // const uint64_t counter_cur = *impl_->engine.ring_state().actual_bytes_counter;
    // const uint64_t counter_delta = counter_cur - impl_->last_counter_read;
    // impl_->last_counter_read = counter_cur;

    const uint64_t pcap = impl_->engine.payload_cap();
    const uint64_t scap = impl_->engine.staging_cap();
    const uint64_t effective_cap = std::min(pcap, scap);
    const uint64_t tcap = impl_->engine.task_cap();

    auto& drain = impl_->engine.drain_thread();

    // Case B: single step exceeds capacity (payload OR task entries).
    // Caller falls back to the per-hook safety net (force_eager + eager
    // dispatch).  We still flush so the ring is empty when the safety
    // net starts firing.
    if (step_total_bytes > effective_cap || num_hooks > tcap) {
        cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(ms);
        drain.force_flush_and_wait();
        return STEP_OVERSIZED;
    }

    // Case A: step fits.  Check available space for BOTH payload AND tasks.
    // Clamp `used` to [0, cap] on BOTH ends (node-toggle reserve-invariant):
    //   - head < tail: a small constant skew exists because producers that fire
    //     during CUDA-graph capture get drained (advancing the tail) with no
    //     matching reserve() (which only runs per real step) -> the ring is in
    //     fact drained, so used = 0 (avail = cap). Computing head-tail here would
    //     underflow uint64 to a huge value -> avail = 0 -> a spurious ring-full
    //     flush (main-stream sync) every step.
    //   - head-tail > cap: over-reserve drift; fail safe with avail = 0 rather
    //     than underflowing to a huge "available" that disables the check.
    const uint64_t ph = drain.cpu_payload_head();
    const uint64_t pt = drain.cpu_payload_tail_committed();
    const uint64_t payload_used = (ph > pt) ? (ph - pt) : 0;
    const uint64_t payload_avail = (payload_used >= pcap) ? 0 : pcap - payload_used;
    const uint64_t th = drain.cpu_task_head();
    const uint64_t tt = drain.cpu_task_tail_committed();
    const uint64_t task_used = (th > tt) ? (th - tt) : 0;
    const uint64_t task_avail = (task_used >= tcap) ? 0 : tcap - task_used;

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

at::Tensor RingEnginePy::payload_tensor() const {
    return impl_->payload_view;
}

// ---------------------------------------------------------------------------
// Runtime queries / actions used by the safety-net branch in
// HookPoint.forward.  All three are called only when force_eager is active
// (eager mode); never run during CUDA-graph capture or replay.
//
// Thread safety of the check-and-reserve pattern used by the safety net:
//
//   if nbytes <= available_capacity():
//       reserve_one(nbytes)
//
// The main thread (this thread) is the only writer of cpu_payload_head_
// (it advances only through reserve / reserve_one calls).  The drain
// thread only ever advances cpu_payload_tail_committed_ forward as it
// frees ring space.  Between the check and the reserve:
//   - tail may move forward (drain freed more): actual available at
//     reserve time is >= what we observed.
//   - head is unchanged (single-threaded writer).
// So the check's "fits" decision remains valid at reserve time.  No extra
// locking around the pair is required.
//
// Within available_capacity(), the two accessor calls happen under
// separate mutex acquires (drain.cpu_payload_head() and
// drain.cpu_payload_tail_committed() each take mgmt_mu_ internally).
// The observed snapshot is non-atomic: if drain advances tail between
// the two reads, available_observed = pcap - head + tail_later, which
// is >= the true available at the time of the head read.  That is, the
// non-atomicity errs on the "over-estimate available" side -- the
// reserve will still succeed because the actual ring state has at least
// as much room as we computed.
// ---------------------------------------------------------------------------

uint64_t RingEnginePy::available_capacity() const {
    auto& drain = impl_->engine.drain_thread();
    const uint64_t pcap = impl_->engine.payload_cap();
    return pcap - (drain.cpu_payload_head() - drain.cpu_payload_tail_committed());
}

// Per-hook reservation: claim nbytes of payload + 1 task entry for an
// upcoming producer kernel launch.  Caller must have checked
// available_capacity() first.  drain.reserve takes mgmt_mu_ internally.
void RingEnginePy::reserve_one(uint64_t nbytes) {
    impl_->engine.drain_thread().reserve(nbytes, 1);
}

// Synchronise the current CUDA stream so all queued producer kernels
// finish writing, then force the drain thread to flush all outstanding
// task entries through the consumer pipeline.  Blocking call; the
// Python binding releases the GIL.
void RingEnginePy::flush_and_wait() {
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    impl_->engine.drain_thread().force_flush_and_wait();
}

}  // namespace ring_py
