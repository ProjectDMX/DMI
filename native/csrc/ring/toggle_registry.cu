// ring/toggle_registry.cu -- runtime CUDA-graph node toggling (see the header).

#include "toggle_registry.h"

#include <atomic>
#include <cstring>

namespace ring_py {

ToggleRegistry::~ToggleRegistry() {
    for (auto& kv : replay_event_) if (kv.second) cudaEventDestroy(kv.second);
    if (dev_pool_) cudaFree(dev_pool_);
}

void ToggleRegistry::set_capture(bool enabled) {
    std::lock_guard<std::mutex> lk(mu_);
    toggle_capture_ = enabled;
    if (enabled && device_mode_ && dev_pool_ == nullptr) {
        // Outside any capture (allocation is not capturable).  Managed
        // memory: the host appends entries while a graph is being captured
        // (no stream operation, so it is legal under any capture mode), and
        // the planner's first replay migrates the pages to the device.
        // Zeroed slots (n = 0) make a planner a no-op.
        const size_t bytes = sizeof(ToggleDevPool);
        if (cudaMallocManaged(reinterpret_cast<void**>(&dev_pool_), bytes) != cudaSuccess) {
            dev_pool_ = nullptr;
            ++anomaly_;
        } else {
            std::memset(dev_pool_, 0, bytes);
        }
    }
    ++version_;
}

bool ToggleRegistry::capture_enabled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return toggle_capture_;
}

void ToggleRegistry::register_node(uint64_t graph, int hook_type, int layer_no, uint64_t node) {
    std::lock_guard<std::mutex> lk(mu_);
    RegEntry re{hook_type, layer_no, reinterpret_cast<cudaGraphNode_t>(node)};
    re.device = dev_nodes_.count(re.node) > 0;
    reg_nodes_[reinterpret_cast<cudaGraph_t>(graph)].push_back(re);
    registered_hooks_.insert({hook_type, layer_no});
    ++version_;
}

void ToggleRegistry::note_anomaly() {
    std::lock_guard<std::mutex> lk(mu_);
    ++anomaly_;
    ++version_;
}

uint64_t ToggleRegistry::anomaly_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return anomaly_;
}

void ToggleRegistry::bind_exec(uint64_t graph, uint64_t exec) {
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    cudaGraphExec_t ex = reinterpret_cast<cudaGraphExec_t>(exec);
    auto prev = reg_exec_.find(g);
    if (prev != reg_exec_.end() && prev->second != ex) {
        // A new exec for a retained graph (re-instantiation).  Nothing
        // recorded for the old exec describes the new one: its nodes start in
        // whatever state instantiation gave them.  Mark the graph stale and
        // every node's state unknown, so the next apply_all/ensure_current
        // sets each host-managed node explicitly from the enabled set (a
        // cached "already disabled" would otherwise skip a node the new exec
        // still runs).  The replay event and upload record belonged to the
        // old exec; a later exec may reuse the old handle value, so forget it.
        applied_version_.erase(g);
        auto nodes = reg_nodes_.find(g);
        if (nodes != reg_nodes_.end())
            for (auto& e : nodes->second) e.state_known = false;
        auto ev = replay_event_.find(g);
        if (ev != replay_event_.end()) {
            if (ev->second) cudaEventDestroy(ev->second);
            replay_event_.erase(ev);
        }
        dev_uploaded_.erase(prev->second);
    }
    reg_exec_[g] = ex;
    // A graph whose producers are planner-managed must be uploaded before its
    // first launch (once).  Binding runs after the capture, with no replay in
    // flight.
    if (upload_device_graph_locked(g, reinterpret_cast<cudaGraphExec_t>(exec))
            != cudaSuccess)
        ++anomaly_;   // fail closed at activation
    ++version_;
}

cudaError_t ToggleRegistry::upload_device_graph_locked(cudaGraph_t g,
                                                       cudaGraphExec_t exec) {
    if (dev_entries_.count(g) == 0 || dev_uploaded_.count(exec) > 0)
        return cudaSuccess;
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaGraphUpload(exec, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (s) cudaStreamDestroy(s);
    if (e == cudaSuccess) dev_uploaded_.insert(exec);
    return e;
}

int ToggleRegistry::upload_device_graph(uint64_t graph, uint64_t exec) {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(upload_device_graph_locked(
        reinterpret_cast<cudaGraph_t>(graph),
        reinterpret_cast<cudaGraphExec_t>(exec)));
}

uint64_t ToggleRegistry::version() const {
    std::lock_guard<std::mutex> lk(mu_);
    return version_;
}

void ToggleRegistry::set_enabled(const std::vector<std::pair<int,int>>& enabled) {
    std::lock_guard<std::mutex> lk(mu_);
    enabled_hooks_.clear();
    for (const auto& p : enabled) enabled_hooks_.insert(p);
    toggle_active_ = true;
    ++target_version_;   // marks every graph stale for the lazy path
}

int ToggleRegistry::apply_all() {
    std::lock_guard<std::mutex> lk(mu_);
    cudaError_t first = cudaSuccess;
    uint64_t applied = 0;
    for (auto& kv : reg_exec_) {
        cudaGraphExec_t exec = kv.second;
        auto it = reg_nodes_.find(kv.first);
        if (it == reg_nodes_.end()) continue;
        for (auto& e : it->second) {
            if (e.device) continue;   // planner-managed, never host-toggled
            const bool on = entry_on(e);
            // Diff: only touch nodes whose desired state changed since the
            // last apply (last_enabled tracks the exec's actual state; DMI is
            // the only toggler of its nodes), so flipping a few hooks costs
            // O(changed) SetEnabled calls, not O(nodes).
            if (entry_current(e, on)) continue;
            cudaError_t err = cudaGraphNodeSetEnabled(exec, e.node, on ? 1u : 0u);
            if (err != cudaSuccess) { if (first == cudaSuccess) first = err; continue; }
            e.last_enabled = on;   // update only on success
            e.state_known = true;
            ++applied;
        }
    }
    // Every bound graph now holds the current set.
    if (first == cudaSuccess)
        for (const auto& kv : reg_exec_) applied_version_[kv.first] = target_version_;
    last_apply_count_ = applied;
    return static_cast<int>(first);
}

uint64_t ToggleRegistry::bound_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return reg_exec_.size();
}

uint64_t ToggleRegistry::node_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t n = 0;
    for (const auto& kv : reg_nodes_) n += kv.second.size();
    return n;
}

uint64_t ToggleRegistry::node_count_for(uint64_t graph) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reg_nodes_.find(reinterpret_cast<cudaGraph_t>(graph));
    return it == reg_nodes_.end() ? 0 : it->second.size();
}

uint64_t ToggleRegistry::last_apply_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_apply_count_;
}

bool ToggleRegistry::uniform() const {
    std::lock_guard<std::mutex> lk(mu_);
    // True iff every captured graph registered the same (hook_type, layer)
    // set.  The metadata gate keys on (hook_type, layer) globally: a hook
    // present in one graph but absent from the graph replayed this step would
    // push metadata with no payload.
    bool have_ref = false;
    std::set<std::pair<int,int>> ref;
    for (const auto& kv : reg_nodes_) {
        std::set<std::pair<int,int>> s;
        for (const auto& e : kv.second) s.insert({e.hook_type, e.layer_no});
        if (!have_ref) { ref = std::move(s); have_ref = true; }
        else if (s != ref) return false;
    }
    return true;
}

bool ToggleRegistry::complete() const {
    std::lock_guard<std::mutex> lk(mu_);
    // Exact key-set match: every captured graph bound and every bound graph
    // captured.  Equal sizes plus inclusion of one in the other is a
    // bijection (both are keyed by unique cudaGraph_t).
    if (reg_nodes_.size() != reg_exec_.size()) return false;
    for (const auto& kv : reg_nodes_)
        if (reg_exec_.count(kv.first) == 0) return false;
    return true;
}

bool ToggleRegistry::is_ready(uint64_t graph) const {
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    return reg_nodes_.count(g) > 0 && reg_exec_.count(g) > 0;
}

bool ToggleRegistry::is_enabled(int hook_type, int layer_no) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!toggle_active_) return true;   // toggle inactive -> every hook fires
    // A hook is enabled only if it is BOTH in the enabled set AND registered
    // (has a captured producer node).  An enabled but uncaptured hook would
    // push metadata with no payload.  So "metadata pushed" <=> "a producer
    // fires".
    const std::pair<int,int> k{hook_type, layer_no};
    return enabled_hooks_.count(k) > 0 && registered_hooks_.count(k) > 0;
}

std::vector<int> ToggleRegistry::effective_mask(
    const std::vector<std::pair<int,int>>& query) const {
    // Batched is_enabled: one lock, one binding crossing.
    std::lock_guard<std::mutex> lk(mu_);
    const bool inactive = !toggle_active_;
    std::vector<int> mask;
    mask.reserve(query.size());
    for (const auto& k : query) {
        mask.push_back((inactive ||
                        (enabled_hooks_.count(k) > 0 &&
                         registered_hooks_.count(k) > 0)) ? 1 : 0);
    }
    return mask;
}

int ToggleRegistry::ensure_current(uint64_t graph) {
    // Lazy: apply the current enabled set to ONE graph, only when stale.
    // Called just before that graph replays.  Same per-node diff as
    // apply_all, scoped to one graph.
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    auto av = applied_version_.find(g);
    const uint64_t cur = (av == applied_version_.end()) ? 0 : av->second;
    if (cur == target_version_) return 0;   // current: the common path

    auto ex = reg_exec_.find(g);
    if (ex == reg_exec_.end()) return 0;     // unregistered/unbound graph
    auto it = reg_nodes_.find(g);
    if (it == reg_nodes_.end()) return 0;

    // Test fault injection: fail before any mutation or version bump.
    if (force_apply_error_ != 0) return force_apply_error_;

    // A graph whose producers are all planner-managed needs no host apply:
    // the in-graph planner derives each node's state from its control row at
    // every replay.
    bool host_nodes = false;
    for (const auto& e : it->second) host_nodes |= !e.device;
    if (!host_nodes) { applied_version_[g] = target_version_; return 0; }

    // Optional pre-apply wait for this exec's previous replay.  If the wait
    // fails, the previous replay cannot be confirmed finished: touch nothing,
    // leave the version stale and return the error (the caller treats it as
    // fatal).
    auto ev = pre_apply_sync_ ? replay_event_.find(g) : replay_event_.end();
    if (ev != replay_event_.end() && ev->second) {
        cudaError_t se = cudaEventSynchronize(ev->second);
        if (se != cudaSuccess) return static_cast<int>(se);
    }

    cudaError_t first = cudaSuccess;
    for (auto& e : it->second) {
        if (e.device) continue;
        const bool on = entry_on(e);
        if (entry_current(e, on)) continue;
        cudaError_t err = cudaGraphNodeSetEnabled(ex->second, e.node, on ? 1u : 0u);
        if (err != cudaSuccess) { if (first == cudaSuccess) first = err; continue; }
        e.last_enabled = on;
        e.state_known = true;
    }
    // Current only on full success: a failed node leaves the version stale so
    // a later ensure retries it (nodes that did flip are already recorded in
    // last_enabled, so the retry skips them).
    if (first == cudaSuccess)
        applied_version_[g] = target_version_;
    return static_cast<int>(first);
}

int ToggleRegistry::record_replay_event(uint64_t graph, cudaStream_t stream) {
    // Record an event right after a graph's replay so a later ensure_current()
    // can wait for it before mutating that exec.  Returns the CUDA error: a
    // failed create or record would leave a stale or missing event, so the
    // caller treats nonzero as fatal.  No-op unless the pre-apply wait is on
    // (the event's only reader).
    std::lock_guard<std::mutex> lk(mu_);
    if (!pre_apply_sync_) return 0;
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    if (reg_exec_.find(g) == reg_exec_.end()) return 0;  // not bound
    cudaEvent_t& ev = replay_event_[g];
    if (ev == nullptr) {
        cudaError_t ce = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (ce != cudaSuccess) { ev = nullptr; return static_cast<int>(ce); }
    }
    return static_cast<int>(cudaEventRecord(ev, stream));
}

void ToggleRegistry::set_pre_apply_sync(bool enabled) {
    std::lock_guard<std::mutex> lk(mu_);
    pre_apply_sync_ = enabled;
}

bool ToggleRegistry::pre_apply_sync() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pre_apply_sync_;
}

void ToggleRegistry::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    // Restore device state before dropping the registry: re-enable every node
    // we disabled, so a graph that keeps replaying after the clear does not
    // run with producers off while the (now inactive) host gate pushes every
    // metadata entry.  Best effort, bound execs only; the call sites are
    // quiescent (before a capture, at teardown).  To return to all-on while
    // serving, prefer an all-on enabled set over clear().
    for (auto& kv : reg_nodes_) {
        auto ex = reg_exec_.find(kv.first);
        if (ex == reg_exec_.end()) continue;
        for (auto& e : kv.second) {
            if ((e.state_known && e.last_enabled) || e.device) continue;
            if (cudaGraphNodeSetEnabled(ex->second, e.node, 1u) == cudaSuccess) {
                e.last_enabled = true;
                e.state_known = true;
            }
        }
    }
    reg_nodes_.clear();
    reg_exec_.clear();
    registered_hooks_.clear();
    // Full reset: drop the enabled set and deactivate, so a stale gate cannot
    // skip every metadata entry after a clear; close the capture window so a
    // later capture does not record into a cleared registry.
    enabled_hooks_.clear();
    toggle_active_ = false;
    toggle_capture_ = false;
    last_apply_count_ = 0;
    anomaly_ = 0;
    target_version_ = 0;
    applied_version_.clear();
    for (auto& kv : replay_event_) if (kv.second) cudaEventDestroy(kv.second);
    replay_event_.clear();
    // Device pool: forget the open group but never recycle or zero a slot.
    // An exec that outlives the clear keeps its planner, which keeps gating
    // each node by its own control row (always correct); zeroing a slot would
    // freeze device-disabled nodes off while their rows ask them to fire.
    dev_group_slot_ = -1;
    dev_group_rows_.clear();
    dev_entries_.clear();
    dev_nodes_.clear();
    dev_uploaded_.clear();
    // Device-side toggling is opted into per capture window.  The pool stays
    // allocated for reuse.
    device_mode_ = false;
    ++version_;
}

bool ToggleRegistry::set_device_mode(bool enabled) {
    std::lock_guard<std::mutex> lk(mu_);
#if DMX_HAS_DEVICE_GRAPH_UPDATE
    if (enabled) {
        int device = 0;
        int driver = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDriverGetVersion(&driver) != cudaSuccess || driver < 12040) {
            device_mode_ = false;
            return false;
        }
    }
    device_mode_ = enabled;
    ++version_;
    return device_mode_ == enabled;
#else
    device_mode_ = false;
    return !enabled;
#endif
}

bool ToggleRegistry::device_mode() const {
    std::lock_guard<std::mutex> lk(mu_);
    return device_mode_;
}

bool ToggleRegistry::device_armed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return device_mode_ && dev_pool_ != nullptr;
}

const ToggleDevPool* ToggleRegistry::device_pool() const {
    std::lock_guard<std::mutex> lk(mu_);
    return dev_pool_;
}

int ToggleRegistry::device_slot(uint64_t graph, const int64_t* control_row, bool* fresh) {
    // One slot per captured FORWARD ("group"), not per graph: a piecewise
    // forward is captured as many consecutive segment graphs holding one or
    // two producers each.  The group's first producer gets the planner node
    // (in its graph); that planner gates every producer of the group,
    // including producers in later segment graphs of the same forward, which
    // replay after it on the same stream.  Each producer fires once per
    // forward, so a repeated control row starts the next group.
    std::lock_guard<std::mutex> lk(mu_);
    (void)graph;
    *fresh = false;
    if (!device_mode_ || dev_pool_ == nullptr) return -1;
    if (dev_group_slot_ >= 0 && dev_group_rows_.count(control_row) == 0) {
        dev_group_rows_.insert(control_row);
        return dev_group_slot_;
    }
    if (dev_slots_used_ >= TOGGLE_DEV_MAX_GRAPHS) return -1;
    const int s = dev_slots_used_++;
    dev_pool_->slot[s].start = dev_entries_used_;
    dev_pool_->slot[s].n = 0;
    dev_group_slot_ = s;
    dev_group_rows_.clear();
    dev_group_rows_.insert(control_row);
    *fresh = true;
    ++version_;
    return s;
}

void ToggleRegistry::register_device_entry(int slot_index, uint64_t graph, uint64_t node,
                                           const int64_t* control_row,
                                           cudaGraphDeviceNode_t dev) {
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    // A group's entries are contiguous in the arena (captures are
    // sequential): the previous append must belong to this slot.
    const bool ok = dev_pool_ != nullptr && slot_index >= 0
        && slot_index < dev_slots_used_
        && dev_pool_->slot[slot_index].start + dev_pool_->slot[slot_index].n == dev_entries_used_;
    if (!ok || dev == nullptr || dev_entries_used_ >= TOGGLE_DEV_MAX_ENTRIES) {
        // Fail closed.  The node stays enabled (the device default), which
        // is still correct (it early-exits on a -1 row), but activation
        // refuses an anomalous registry.
        ++anomaly_;
        ++version_;
        return;
    }
    dev_entries_[g].push_back(ToggleDevEntry{control_row, dev});
    dev_nodes_.insert(reinterpret_cast<cudaGraphNode_t>(node));
    // Publish now (host write into managed memory; no graph of this group can
    // replay before its capture ends).  Entry before count.
    ToggleDevSlot& slot = dev_pool_->slot[slot_index];
    dev_pool_->entry[dev_entries_used_++] = ToggleDevEntry{control_row, dev};
    std::atomic_thread_fence(std::memory_order_release);
    slot.n += 1;
    ++version_;
}

uint64_t ToggleRegistry::device_node_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t n = 0;
    for (const auto& kv : dev_entries_) n += kv.second.size();
    return n;
}

uint64_t ToggleRegistry::device_errors() const {
    std::lock_guard<std::mutex> lk(mu_);
    // Managed memory: host-readable once the planner's launches complete.
    return dev_pool_ ? dev_pool_->errors : 0;
}

int ToggleRegistry::upload_exec(uint64_t exec) {
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(exec), s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (s) cudaStreamDestroy(s);
    return static_cast<int>(e);
}

void ToggleRegistry::force_apply_error(int code) {
    std::lock_guard<std::mutex> lk(mu_);
    force_apply_error_ = code;
}

bool ToggleRegistry::applied_current(uint64_t graph) const {
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    auto av = applied_version_.find(g);
    const uint64_t cur = (av == applied_version_.end()) ? 0 : av->second;
    return cur == target_version_;
}

}  // namespace ring_py

namespace ring {

// One thread per producer node: enable iff its control row is not -1.  Runs
// upstream of every producer of the same graph launch, so the update applies
// to this launch (device graph update semantics).
__global__ void toggle_planner_kernel(const ring_py::ToggleDevPool* pool, int slot)
{
#if DMX_HAS_DEVICE_GRAPH_UPDATE
    const ring_py::ToggleDevSlot s = pool->slot[slot];
    for (int i = threadIdx.x; i < s.n; i += blockDim.x) {
        const ring_py::ToggleDevEntry e = pool->entry[s.start + i];
        // A failure leaves the node in its previous state (still correct: a
        // node left enabled early-exits on a -1 row).  Failures are counted.
        const bool on = e.control_row[0] != -1;
        const cudaError_t r = cudaGraphKernelNodeSetEnabled(e.node, on);
        unsigned char* fl = const_cast<unsigned char*>(&pool->flags[s.start + i]);
        if (r == cudaSuccess) { if (!on && !(*fl & 1)) *fl |= 1; }
        else if (!(*fl & 2)) *fl |= 2;
        if (r != cudaSuccess) {
            atomicAdd(const_cast<unsigned long long*>(&pool->errors), 1ull);
            ring_py::ToggleDevPool* p = const_cast<ring_py::ToggleDevPool*>(pool);
            p->last_error = static_cast<int>(r);
            p->last_slot = slot;
            p->last_entry = i;
            p->last_enable = on;
        }
    }
#else
    (void)pool; (void)slot;
#endif
}

void launch_toggle_planner(const ring_py::ToggleDevPool* pool, int slot, cudaStream_t stream)
{
    toggle_planner_kernel<<<1, 128, 0, stream>>>(pool, slot);
}

}  // namespace ring
