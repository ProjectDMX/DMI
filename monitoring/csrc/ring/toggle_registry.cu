#include "toggle_registry.h"

namespace ring_py {

ToggleRegistry::~ToggleRegistry() {
    for (auto& kv : replay_event_) if (kv.second) cudaEventDestroy(kv.second);
}

void ToggleRegistry::set_capture(bool enabled) {
    std::lock_guard<std::mutex> lk(mu_);
    toggle_capture_ = enabled;
    ++version_;
}

bool ToggleRegistry::capture_enabled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return toggle_capture_;
}

void ToggleRegistry::register_node(uint64_t graph, int hook_type, int layer_no, uint64_t node) {
    std::lock_guard<std::mutex> lk(mu_);
    reg_nodes_[reinterpret_cast<cudaGraph_t>(graph)].push_back(
        RegEntry{hook_type, layer_no, reinterpret_cast<cudaGraphNode_t>(node)});
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
    reg_exec_[reinterpret_cast<cudaGraph_t>(graph)] =
        reinterpret_cast<cudaGraphExec_t>(exec);
    ++version_;
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
    ++target_version_;   // mark all graphs stale for lazy ensure (no-op for eager)
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
            const bool on = enabled_hooks_.count({e.hook_type, e.layer_no}) > 0;
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

uint64_t ToggleRegistry::last_apply_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_apply_count_;
}

bool ToggleRegistry::uniform() const {
    std::lock_guard<std::mutex> lk(mu_);
    // True iff every captured graph registered the same set of (hook_type,layer).
    // The meta gate keys on (ht,layer) globally; if graphs differ, a hook present
    // in one graph but absent from the one replayed this step would push a meta
    // with no payload -> desync.
    const std::set<std::pair<int,int>>* ref = nullptr;
    std::set<std::pair<int,int>> ref_set;
    for (const auto& kv : reg_nodes_) {
        std::set<std::pair<int,int>> s;
        for (const auto& e : kv.second) s.insert({e.hook_type, e.layer_no});
        if (!ref) { ref_set = std::move(s); ref = &ref_set; }
        else if (s != ref_set) return false;
    }
    return true;
}

bool ToggleRegistry::complete() const {
    std::lock_guard<std::mutex> lk(mu_);
    // Exact key-set match: every captured graph bound AND every bound graph
    // captured. Equal sizes + every reg_nodes key present in reg_exec ⟹
    // bijection (both keyed by unique cudaGraph_t).
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
    if (!toggle_active_) return true;   // toggle inactive -> all hooks on
    // Desync guard: the meta gate returns true only if the hook is BOTH
    // enabled AND actually registered (has a captured producer node). Otherwise
    // an enabled-but-uncaptured hook would push a meta with no matching payload
    // -> p2p desync. With this, "meta pushed" <=> "a producer fires" in all cases.
    const std::pair<int,int> k{hook_type, layer_no};
    return enabled_hooks_.count(k) > 0 && registered_hooks_.count(k) > 0;
}

std::vector<int> ToggleRegistry::effective_mask(
    const std::vector<std::pair<int,int>>& query) const {
    // Batched is_enabled (same semantics, one lock, one pybind crossing):
    // toggle inactive -> all on; else enabled AND registered.
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
    // Lazy: apply the current enabled set to ONE graph, only if it is
    // stale (applied_version != target_version). Called just before that graph
    // replays. Same per-node diff as apply_all, but scoped to one graph.
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    auto av = applied_version_.find(g);
    const uint64_t cur = (av == applied_version_.end()) ? 0 : av->second;
    if (cur == target_version_) return 0;   // current -> fast no-op (common path)

    auto ex = reg_exec_.find(g);
    if (ex == reg_exec_.end()) return 0;     // unregistered/unbound graph -> skip
    auto it = reg_nodes_.find(g);
    if (it == reg_nodes_.end()) return 0;

    // Test fault injection: fail before any mutation / version bump.
    if (force_apply_error_ != 0) return force_apply_error_;

    // Event guard: the prior replay of THIS exec must be complete before we
    // mutate it (host can run ahead of the GPU). Wait on its last-replay event.
    // If the wait itself fails we CANNOT confirm the prior replay finished,
    // so do NOT touch the exec -- return the error; the caller treats it as a
    // FATAL desync risk. No nodes mutated, version left stale.
    auto ev = replay_event_.find(g);
    if (ev != replay_event_.end() && ev->second) {
        cudaError_t se = cudaEventSynchronize(ev->second);
        if (se != cudaSuccess) return static_cast<int>(se);
    }

    cudaError_t first = cudaSuccess;
    for (auto& e : it->second) {
        const bool on = enabled_hooks_.count({e.hook_type, e.layer_no}) > 0;
        if (on == e.last_enabled) continue;
        cudaError_t err = cudaGraphNodeSetEnabled(ex->second, e.node, on ? 1u : 0u);
        if (err != cudaSuccess) { if (first == cudaSuccess) first = err; continue; }
        e.last_enabled = on;
    }
    // Mark this graph current ONLY on full success. If any SetEnabled
    // failed, leave applied_version stale so a later ensure retries the
    // still-unapplied nodes (last_enabled already tracks per-node success, so
    // the retry is a no-op for the ones that did flip). The caller (replay
    // hook) treats a nonzero return as a FATAL desync and terminates, but the
    // stale version is the correct state regardless.
    if (first == cudaSuccess)
        applied_version_[g] = target_version_;
    return static_cast<int>(first);
}

int ToggleRegistry::record_replay_event(uint64_t graph, cudaStream_t stream) {
    // Record a (timing-disabled) event on the given stream right after a
    // graph's replay, so a later ensure_current() can wait for it before
    // mutating that exec. Returns the CUDA error: if event create OR record
    // fails, the stored event would be stale/missing and a later ensure could
    // wrongly believe a prior replay finished -> mutate an executing exec (UB).
    // The caller treats nonzero as FATAL so we never reach that state.
    std::lock_guard<std::mutex> lk(mu_);
    cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph);
    if (reg_exec_.find(g) == reg_exec_.end()) return 0;  // not bound -> nothing to track
    cudaEvent_t& ev = replay_event_[g];
    if (ev == nullptr) {
        cudaError_t ce = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (ce != cudaSuccess) { ev = nullptr; return static_cast<int>(ce); }
    }
    cudaError_t re = cudaEventRecord(ev, stream);
    return static_cast<int>(re);
}

void ToggleRegistry::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    // Restore device state BEFORE dropping the registry: re-enable any node
    // we disabled, so a graph that keeps replaying after a clear does not run
    // with producers OFF while the (now-inactive) host gate pushes all metas
    // -> desync. Best-effort, only for bound execs; safe at the quiescent
    // call sites (warmup-start / teardown, no replay in flight). To revert to
    // all-on DURING serving, prefer an all-on enabled set (event-guarded)
    // over clear.
    for (auto& kv : reg_nodes_) {
        auto ex = reg_exec_.find(kv.first);
        if (ex == reg_exec_.end()) continue;
        for (auto& e : kv.second) {
            if (e.last_enabled) continue;
            if (cudaGraphNodeSetEnabled(ex->second, e.node, 1u) == cudaSuccess)
                e.last_enabled = true;
        }
    }
    reg_nodes_.clear();
    reg_exec_.clear();
    registered_hooks_.clear();
    // Full reset: also drop the enabled set + deactivate, so a stale gate
    // (is_enabled) can't skip every meta after a clear. Disable capture too
    // so a later capture doesn't record into a cleared registry.
    enabled_hooks_.clear();
    toggle_active_ = false;
    toggle_capture_ = false;
    last_apply_count_ = 0;
    anomaly_ = 0;
    // Lazy state: reset versions + destroy per-graph replay events.
    target_version_ = 0;
    applied_version_.clear();
    for (auto& kv : replay_event_) if (kv.second) cudaEventDestroy(kv.second);
    replay_event_.clear();
    ++version_;
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
