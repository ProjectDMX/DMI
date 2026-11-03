// ring/toggle_registry.h -- state and CUDA-graph operations for runtime
// node toggling, kept out of the ring engine so the engine only delegates.
//
// Owns: the capture-time node registry (graph -> producer kernel nodes), the
// graph -> exec bindings, the enabled (hook_type, layer_no) set (the single
// source of truth for the host metadata gate and the device node state), the
// lazy per-graph apply state (target/applied versions and per-graph replay
// events), the capture anomaly counter, the device-side toggle pool, and the
// registry version every mutation bumps (host caches key their guard verdicts
// on it).
//
// Thread-safe: every public method takes the internal mutex.  Depends only on
// the CUDA runtime (callers pass streams in).  This header contains CUDA
// types, so only .cu/.cpp translation units include it, never the CUDA-free
// ring_engine_py.h.
#pragma once

#include <cuda_runtime.h>
#include "toggle_device.h"

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
    // A second, different key recorded for the node just registered (a fused
    // producer) is attached to that node instead of adding an entry.
    void register_node(uint64_t graph, int hook_type, int layer_no, uint64_t node);
    // Called when a capture-time recording could not be validated (non-kernel
    // tail, unreadable capture state, unsupported producer) -> fail closed.
    void note_anomaly();
    uint64_t anomaly_count() const;

    // -- binding / introspection --------------------------------------------
    // Bind (or rebind) a graph to an instantiated exec.  Binding a different
    // exec than the one bound before resets every per-exec record for the
    // graph: the applied version, each node's cached state (unknown until the
    // next apply sets every host-managed node explicitly), the replay event
    // and the previous exec's upload record.  Rebinding the same exec is a
    // no-op apart from the version bump.
    void bind_exec(uint64_t graph, uint64_t exec);
    uint64_t version() const;          // bumped by every registry mutation
    uint64_t bound_count() const;
    uint64_t node_count() const;
    uint64_t node_count_for(uint64_t graph) const;   // nodes recorded for one graph
    uint64_t last_apply_count() const;
    bool uniform() const;              // all graphs registered the same hook set
    bool complete() const;             // registered graphs == bound graphs
    bool is_ready(uint64_t graph) const;   // registered AND bound

    // -- enabled set + host apply ---------------------------------------------
    void set_enabled(const std::vector<std::pair<int, int>>& enabled);
    int  apply_all();                  // eager: every bound exec, diff-based
    bool is_enabled(int hook_type, int layer_no) const;
    std::vector<int> effective_mask(
        const std::vector<std::pair<int, int>>& query) const;
    int  ensure_current(uint64_t graph);                       // lazy, one graph
    int  record_replay_event(uint64_t graph, cudaStream_t stream);
    // Wait for a graph's previous replay before mutating its exec (see
    // pre_apply_sync_).  Off by default.
    void set_pre_apply_sync(bool enabled);
    bool pre_apply_sync() const;
    void clear();

    // -- device-side toggle (toggle_device.h) --------------------------------
    // Opt in before the capture window opens.  Ignored (returns false) when
    // the build or the runtime lacks device graph updates.  clear() turns the
    // mode off again.
    bool set_device_mode(bool enabled);
    bool device_mode() const;
    // The pool exists (allocated when the first capture window opens in
    // device mode).
    bool device_armed() const;
    // Slot of the forward ("group") being captured (-1 = pool exhausted).
    // *fresh = true for the group's first producer: the caller then enqueues
    // the planner node right before it.
    int device_slot(uint64_t graph, const int64_t* control_row, bool* fresh);
    const ToggleDevPool* device_pool() const;
    // Called right after a device-updatable producer was captured.
    void register_device_entry(int slot, uint64_t graph, uint64_t node,
                               const int64_t* control_row,
                               cudaGraphDeviceNode_t dev);
    uint64_t device_node_count() const;
    uint64_t device_errors() const;
    // cudaGraphUpload + synchronize: required before the first launch of an
    // exec holding device-updatable nodes.
    int upload_exec(uint64_t exec);
    // Upload `exec` iff `graph` captured device-updatable producers and the
    // exec was not uploaded yet: each such exec is uploaded at most once,
    // whether it is bound (bind_exec) or not (a capture outside the window).
    // Returns the CUDA error (0 also when nothing needed uploading).
    int upload_device_graph(uint64_t graph, uint64_t exec);

    // -- test-only fault injection / introspection ---------------------------
    void force_apply_error(int code);
    bool applied_current(uint64_t graph) const;

private:
    struct RegEntry {
        int hook_type;
        int layer_no;
        cudaGraphNode_t node;
        bool last_enabled{true};   // device default after instantiate
        // False when the bound exec changed since last_enabled was written:
        // the next apply sets the node unconditionally.
        bool state_known{true};
        bool device{false};        // toggled by the in-graph planner, never by the host
        // A fused producer node carries two keys (rows + heads); it is
        // enabled iff either key is enabled.  -1 = no second key.
        int alt_hook_type{-1};
        int alt_layer_no{-1};
    };
    cudaError_t upload_device_graph_locked(cudaGraph_t g, cudaGraphExec_t exec);
    // The apply loops skip a node only when its exec state is known to match.
    static bool entry_current(const RegEntry& e, bool on) {
        return e.state_known && on == e.last_enabled;
    }
    bool entry_on(const RegEntry& e) const {
        return enabled_hooks_.count({e.hook_type, e.layer_no}) > 0
            || (e.alt_hook_type >= 0 &&
                enabled_hooks_.count({e.alt_hook_type, e.alt_layer_no}) > 0);
    }
    std::unordered_map<cudaGraph_t, std::vector<RegEntry>> reg_nodes_;  // per captured graph
    std::unordered_map<cudaGraph_t, cudaGraphExec_t>       reg_exec_;   // graph -> exec
    std::set<std::pair<int,int>>                           registered_hooks_;  // (ht, layer) captured
    std::set<std::pair<int,int>>                           enabled_hooks_;     // single source
    bool                                                   toggle_active_{false};
    bool                                                   toggle_capture_{false};
    uint64_t                                               last_apply_count_{0};
    // Lazy per-graph apply: target_version_ bumps on every enabled-set change;
    // each graph tracks the version it was last applied to. ensure_current()
    // applies the current enabled set to ONE graph only when stale (just
    // before that graph replays), so a reconfigure costs
    // O(changed x graphs actually replayed).
    //
    // cudaGraphNodeSetEnabled "only affects future launches of hGraphExec.
    // Already enqueued or running launches of hGraphExec are not affected",
    // so mutating an exec whose previous launch is still queued is defined
    // and the in-flight launch keeps the node state it was enqueued with.
    // The pre-apply wait (an event recorded after each replay and
    // synchronized before the first mutation of a stale exec) is therefore
    // off by default: it serializes the host with the GPU on every set
    // change and defeats schedulers that run one step ahead of the GPU.
    // set_pre_apply_sync(true) restores it; with it off,
    // record_replay_event() is a no-op.
    bool                                                   pre_apply_sync_{false};
    uint64_t                                               target_version_{0};
    std::unordered_map<cudaGraph_t, uint64_t>              applied_version_;
    std::unordered_map<cudaGraph_t, cudaEvent_t>           replay_event_;
    // Test-only fault injection: when nonzero, ensure_current returns this
    // code as if a device apply failed (no SetEnabled, no version bump).
    int                                                    force_apply_error_{0};
    // Capture-time recordings that could not be validated.  Nonzero means the
    // registry may be misaligned, so activation fails closed.
    uint64_t                                               anomaly_{0};
    uint64_t                                               version_{0};
    // Device-side toggle state.  The pool is allocated at the first
    // set_capture(true) in device mode (outside any capture).
    bool                                                   device_mode_{false};
    ToggleDevPool*                                         dev_pool_{nullptr};
    int                                                    dev_slots_used_{0};
    int                                                    dev_entries_used_{0};
    int                                                    dev_group_slot_{-1};
    std::set<const int64_t*>                               dev_group_rows_;
    std::unordered_map<cudaGraph_t, std::vector<ToggleDevEntry>> dev_entries_;
    std::set<cudaGraphNode_t>                              dev_nodes_;
    std::set<cudaGraphExec_t>                              dev_uploaded_;
    mutable std::mutex                                     mu_;
};

}  // namespace ring_py
