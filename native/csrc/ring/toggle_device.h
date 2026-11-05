// ring/toggle_device.h -- device-side CUDA-graph node toggling.
//
// Host-side node toggling (ToggleRegistry::apply_all / ensure_current) calls
// cudaGraphNodeSetEnabled on an instantiated exec between replays.  The
// device-side mode moves that decision into the graph: a producer launched as
// a device-updatable kernel node (cudaLaunchAttributeDeviceUpdatableKernelNode)
// is paired with a one-word control row in device memory, and a planner kernel
// captured upstream of the producers calls
// cudaGraphKernelNodeSetEnabled(node, row[0] != -1) at every replay.
//
// The control-row contract is the one the producers themselves honour: a row
// whose first word is -1 makes the producer return without a reservation, a
// task or a payload.  A disabled node is therefore the same observation as an
// early-exit launch, so the planner adds no host/device lockstep and the host
// never toggles these nodes.
//
// CUDA device graph updates (CUDA 12.4+, toolkit and driver): an update made
// by an upstream node of a launch applies to the downstream nodes of the same
// launch and persists to later launches until overwritten.  An exec holding
// device-updatable nodes must be uploaded (cudaGraphUpload) before its first
// launch.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Device graph updates need CUDA >= 12.4.  Older toolkits build with the
// feature compiled out; -DDMX_NO_DEVICE_GRAPH_UPDATE forces the same.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040 && !defined(DMX_NO_DEVICE_GRAPH_UPDATE)
#define DMX_HAS_DEVICE_GRAPH_UPDATE 1
#else
#define DMX_HAS_DEVICE_GRAPH_UPDATE 0
#endif
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12040
typedef void* cudaGraphDeviceNode_t;   // absent from pre-12.4 headers
#endif

// cudaStreamGetCaptureInfo gained the edge-data out-parameter in CUDA 13 (it
// was cudaStreamGetCaptureInfo_v3 before); one portable call site.
static inline cudaError_t dmx_capture_info(cudaStream_t s, cudaStreamCaptureStatus* st,
                                           unsigned long long* id, cudaGraph_t* g,
                                           const cudaGraphNode_t** deps, size_t* nd) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edges = nullptr;
    return cudaStreamGetCaptureInfo(s, st, id, g, deps, &edges, nd);
#else
    return cudaStreamGetCaptureInfo(s, st, id, g, deps, nd);
#endif
}

namespace ring_py {

// One slot per captured forward ("group"), one shared append-only entry
// arena.  A piecewise forward is captured as many consecutive segment graphs
// holding a few producers each, so slots are (start, n) views of the arena.
constexpr int TOGGLE_DEV_MAX_GRAPHS  = 16384;
constexpr int TOGGLE_DEV_MAX_ENTRIES = 65536;

struct ToggleDevEntry {
    const int64_t*        control_row;  // the producer's control row (fixed at capture)
    cudaGraphDeviceNode_t node;         // device handle of its kernel node
};

struct ToggleDevSlot {
    int32_t n;                       // entries published (0 -> planner no-op)
    int32_t start;                   // first entry in the arena
};

struct ToggleDevPool {
    unsigned long long errors;       // device-side cudaGraphKernelNodeSetEnabled failures
    int last_error, last_slot, last_entry, last_enable;   // most recent failure
    ToggleDevSlot  slot[TOGGLE_DEV_MAX_GRAPHS];
    ToggleDevEntry entry[TOGGLE_DEV_MAX_ENTRIES];
    // Per-entry sticky flags (diagnostic, idempotent stores): 1 = a disable
    // succeeded at least once, 2 = an update failed at least once.
    unsigned char  flags[TOGGLE_DEV_MAX_ENTRIES];
};

}  // namespace ring_py

namespace ring {
// Enqueue the planner kernel (captured into the graph) for one slot.
void launch_toggle_planner(const ring_py::ToggleDevPool* pool, int slot, cudaStream_t stream);
}  // namespace ring
