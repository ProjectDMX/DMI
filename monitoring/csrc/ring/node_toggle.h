// ring/node_toggle.h -- runtime "toggle list" controller for post-capture
// producer node enable/disable (Axis-A #1/#4).
//
// Purpose: keep the lockstep invariant on a SINGLE SOURCE OF TRUTH.  One
// enabled-set drives BOTH (a) which producer graph nodes fire -- via apply() ->
// cudaGraphNodeSetEnabled -- and (b) which hook metadata the host pushes -- via
// enabled_in_capture_order().  This removes the hand-coordination that, if
// violated, desyncs the positional meta<->payload matching in p2p_thread.
//
// IMPORTANT -- this is NOT unconditional "lockstep by construction".  It holds
// only when the caller honors all of:
//   * Single-threaded, step-boundary ownership.  The controller is plain mutable
//     state with no internal locking.  Mutate + apply + read the enabled list
//     from ONE consistent snapshot.  Prefer apply_and_get_enabled(), which
//     applies and returns the enabled list in a single call so no mutation can
//     interleave between the two lanes.
//   * Registration in PRODUCER PUBLISH ORDER.  p2p matches the i-th drained
//     payload to the i-th popped meta POSITIONALLY.  The meta-push order is the
//     order entries were register_node()'d.  Therefore entries MUST be
//     registered in the order producers actually fire (graph capture / launch
//     order), NOT sorted by hook identity.  Having the same enabled *set* is not
//     enough -- the *order* must match, or FIFO positions mismatch and desync.
//     The controller cannot verify this (it has no independent notion of capture
//     order); it is a caller contract.  validate() checks the things it CAN:
//     null/duplicate nodes and duplicate hook ids.
//
// Lifecycle (design-notes §1): register at capture time, in publish order;
// between steps only, with the prior replay COMPLETE, set the enabled-set,
// apply(exec) (or apply_and_get_enabled), push metas for the returned list,
// then launch.  Never mutate while the exec is running.
//
// Header-only, no ATen / no engine deps -- usable from the native backend and
// from standalone tests alike.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
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
    // Register, IN PRODUCER PUBLISH ORDER, the graph node that produces `hook`.
    // (See class doc: order matters, not just identity.)
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

    // Check the registry is well-formed: no null node handles, no duplicate node
    // handles, no duplicate hook ids.  Does NOT (cannot) verify that registration
    // order equals producer publish order -- that is a caller contract.  Returns
    // false and sets *reason (if given) on the first problem.  O(n^2); n is small.
    bool validate(std::string* reason = nullptr) const {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].node == nullptr) {
                if (reason) *reason = "null node handle"; return false;
            }
            for (size_t j = i + 1; j < entries_.size(); ++j) {
                if (entries_[i].node == entries_[j].node) {
                    if (reason) *reason = "duplicate node handle"; return false;
                }
                if (entries_[i].hook == entries_[j].hook) {
                    if (reason) *reason = "duplicate hook id"; return false;
                }
            }
        }
        return true;
    }

    // Apply the current enabled bits to `exec`.  The CALLER must guarantee the
    // prior replay has completed (design-notes §1 #2) before calling this.
    // Returns the first CUDA error encountered, or cudaSuccess.
    cudaError_t apply(cudaGraphExec_t exec) const {
        for (const auto& e : entries_) {
            cudaError_t err = cudaGraphNodeSetEnabled(exec, e.node, e.enabled ? 1u : 0u);
            if (err != cudaSuccess) return err;
        }
        return cudaSuccess;
    }

    // The enabled hooks, in capture order -- the meta-push list.
    std::vector<HookId> enabled_in_capture_order() const {
        std::vector<HookId> out;
        out.reserve(entries_.size());
        for (const auto& e : entries_) if (e.enabled) out.push_back(e.hook);
        return out;
    }

    // PREFERRED entry point: apply the enabled bits AND return the matching
    // meta-push list, from ONE snapshot of the state, in a single call.  Because
    // no caller mutation can interleave between the two lanes, this is the safe
    // way to drive a step (still single-threaded; see class doc).
    cudaError_t apply_and_get_enabled(cudaGraphExec_t exec,
                                      std::vector<HookId>& out_enabled) const {
        cudaError_t err = apply(exec);
        if (err != cudaSuccess) return err;
        out_enabled = enabled_in_capture_order();
        return cudaSuccess;
    }

private:
    struct Entry { HookId hook; cudaGraphNode_t node; bool enabled; };
    std::vector<Entry> entries_;
};

}  // namespace ring
