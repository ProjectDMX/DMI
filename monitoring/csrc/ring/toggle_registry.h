// ToggleRegistry -- all state and CUDA-graph operations for runtime
// node-toggle, extracted from the ring engine so the engine only delegates.
//
// Owns: the capture-time node registry (graph -> producer kernel nodes), the
// graph->exec bindings, the enabled (hook_type, layer_no) set (SINGLE SOURCE
// OF TRUTH for host meta gate + device node-enable), the lazy per-graph apply
// state (target/applied versions + per-graph replay events), the capture
// anomaly counter, and the registry version every mutation bumps (the host
// caches guard verdicts keyed on it).
//
// Thread-safe: every public method takes the internal mutex. Depends only on
// the CUDA runtime (the caller passes streams in) -- this header contains CUDA
// types, so it is included by .cu/.cpp translation units only, never by the
// CUDA-free ring_engine_py.h.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ring_py {

class ToggleRegistry {
public:
    ~ToggleRegistry();

    // -- capture window ----------------------------------------------------
    void set_capture(bool enabled);
    bool capture_enabled() const;
    // Called from the producer op during capture (single kernel just enqueued).
    void register_node(uint64_t graph, int hook_type, int layer_no, uint64_t node);
    // Called when a capture-time recording could not be validated (non-kernel
    // tail, unreadable capture state, unsupported producer) -> fail-closed.
    void note_anomaly();
    uint64_t anomaly_count() const;

    // -- binding / introspection --------------------------------------------
    void bind_exec(uint64_t graph, uint64_t exec);
    uint64_t version() const;          // bumped by EVERY mutation
    uint64_t bound_count() const;
    uint64_t node_count() const;
    uint64_t last_apply_count() const;
    bool uniform() const;              // all graphs share the same hook set
    bool complete() const;             // reg_nodes keys == reg_exec keys
    bool is_ready(uint64_t graph) const;   // registered AND bound

    // -- enabled set + device apply ------------------------------------------
    void set_enabled(const std::vector<std::pair<int, int>>& enabled);
    int  apply_all();                  // eager: every bound exec, diff-based
    bool is_enabled(int hook_type, int layer_no) const;
    std::vector<int> effective_mask(
        const std::vector<std::pair<int, int>>& query) const;
    int  ensure_current(uint64_t graph);                       // lazy, one graph
    int  record_replay_event(uint64_t graph, cudaStream_t stream);
    void clear();

    // -- test-only fault injection / introspection ---------------------------
    void force_apply_error(int code);
    bool applied_current(uint64_t graph) const;

private:
    struct RegEntry { int hook_type; int layer_no; cudaGraphNode_t node;
                      bool last_enabled{true}; };  // device default after instantiate
    std::unordered_map<cudaGraph_t, std::vector<RegEntry>> reg_nodes_;  // per captured graph
    std::unordered_map<cudaGraph_t, cudaGraphExec_t>       reg_exec_;   // graph -> exec
    std::set<std::pair<int,int>>                           registered_hooks_;  // (ht,layer) captured
    std::set<std::pair<int,int>>                           enabled_hooks_;     // single source
    bool                                                   toggle_active_{false};
    bool                                                   toggle_capture_{false};
    uint64_t                                               last_apply_count_{0};
    // Lazy per-graph apply: target_version_ bumps on every enabled-set change;
    // each graph tracks the version it was last applied to. ensure_current()
    // applies the current enabled set to ONE graph only when stale (just
    // before that graph replays), so a reconfigure costs
    // O(changed x graphs-actually-used). replay_event_[g] is recorded after
    // each replay so ensure can wait for the prior replay to finish before
    // mutating the exec (SetEnabled on an executing exec is UB).
    uint64_t                                               target_version_{0};
    std::unordered_map<cudaGraph_t, uint64_t>              applied_version_;
    std::unordered_map<cudaGraph_t, cudaEvent_t>           replay_event_;
    // Test-only fault injection: when nonzero, ensure_current returns this
    // code as if a device apply failed (no SetEnabled, no version bump).
    int                                                    force_apply_error_{0};
    // Count of capture-time recordings whose tail-dependency node was NOT a
    // kernel node (so it could not be the producer kernel). Nonzero => the
    // registry may be misaligned; activation fails loud (fail-closed).
    uint64_t                                               anomaly_{0};
    // Bumped by EVERY registry mutation (capture flag, node recording,
    // anomaly, bind, clear), so a host-cached "valid" guard verdict can never
    // outlive a mutation.
    uint64_t                                               version_{0};
    mutable std::mutex                                     mu_;
};

}  // namespace ring_py
