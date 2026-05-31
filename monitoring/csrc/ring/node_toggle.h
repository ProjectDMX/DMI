// ring/node_toggle.h -- runtime "toggle list" controller for post-capture
// producer node enable/disable (Axis-A #1/#4).
//
// Purpose: make the lockstep invariant STRUCTURAL.  A single enabled-set drives
// BOTH (a) which producer graph nodes fire -- via cudaGraphNodeSetEnabled -- and
// (b) which hook metadata the host pushes -- via enabled_in_capture_order().
// Because both come from the same source, there is no way to enable a node
// without its meta being in the push list, or vice versa.  This removes the
// hand-coordination that, if violated, desyncs the positional meta<->payload
// matching in p2p_thread (see docs/node_toggle_design_notes.md §1).
//
// Lifecycle (design-notes §1, must be honored by the caller):
//   1. capture time:        register_node(hook, node) for each producer, in
//                           capture order.
//   2. between steps only:  with the prior replay COMPLETE, set the enabled-set,
//                           apply(exec), push metas for enabled_in_capture_order(),
//                           then launch.  Never mutate while the exec is running.
//
// Header-only, no ATen / no engine deps -- usable from the native backend and
// from standalone tests alike.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace ring {

// Hook identity = (hook_type, layer_no), matching ring_py::TensorMeta.  Kept as
// plain ints so this header has no dependency on tensor_meta.h.
struct HookId {
    int hook_type;
    int layer_no;
    bool operator==(const HookId& o) const {
        return hook_type == o.hook_type && layer_no == o.layer_no;
    }
};

class NodeToggleController {
public:
    // Register, in capture order, the graph node that produces `hook`.
    void register_node(HookId hook, cudaGraphNode_t node) {
        entries_.push_back(Entry{hook, node, true});  // default: enabled
    }

    size_t size() const { return entries_.size(); }

    void set_all(bool enabled) {
        for (auto& e : entries_) e.enabled = enabled;
    }

    // Enable exactly the hooks for which pred(hook) is true; disable the rest.
    template <class Pred>
    void set_enabled_if(Pred pred) {
        for (auto& e : entries_) e.enabled = static_cast<bool>(pred(e.hook));
    }

    // Apply the current enabled bits to `exec`.  The CALLER must guarantee the
    // prior replay has completed (design-notes §1 #2) before calling this.
    // Returns the first CUDA error encountered, or cudaSuccess.
    cudaError_t apply(cudaGraphExec_t exec) const {
        for (auto& e : entries_) {
            cudaError_t err = cudaGraphNodeSetEnabled(exec, e.node, e.enabled ? 1u : 0u);
            if (err != cudaSuccess) return err;
        }
        return cudaSuccess;
    }

    // The enabled hooks, in capture order -- THE single source of truth for the
    // host meta-push.  Pushing metas for exactly this list (and only this list)
    // is what keeps the host meta lane in lockstep with apply().
    std::vector<HookId> enabled_in_capture_order() const {
        std::vector<HookId> out;
        out.reserve(entries_.size());
        for (const auto& e : entries_) if (e.enabled) out.push_back(e.hook);
        return out;
    }

private:
    struct Entry { HookId hook; cudaGraphNode_t node; bool enabled; };
    std::vector<Entry> entries_;
};

}  // namespace ring
