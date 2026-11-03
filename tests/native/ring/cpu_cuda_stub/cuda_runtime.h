// CPU-only stand-in for the CUDA runtime API used by the host half of
// ring/toggle_registry.cu, ring/ring_torch_op.cpp's producer ops and the
// host-side P2P handoff (ring/p2p_thread.cpp, ring/pinned_staging.h).  No
// CUDA library is linked or initialized: each call records what a real exec
// would hold so tests can compare the registry's view with the exec's actual
// node state.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

using cudaError_t = int;
using cudaGraph_t = void*;
using cudaGraphNode_t = void*;
using cudaGraphExec_t = void*;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaStreamCaptureStatus = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorUnknown = 999;
constexpr cudaStreamCaptureStatus cudaStreamCaptureStatusNone = 0;
constexpr cudaStreamCaptureStatus cudaStreamCaptureStatusActive = 1;
using cudaGraphNodeType = int;
constexpr cudaGraphNodeType cudaGraphNodeTypeKernel = 0;
constexpr cudaGraphNodeType cudaGraphNodeTypeMemcpy = 1;
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDisableTiming = 2;

// (exec, node) -> enabled.  A node missing from the map is enabled: CUDA
// instantiates every kernel node enabled.
inline std::map<std::pair<uint64_t, uint64_t>, bool> fake_enabled;
inline unsigned fake_set_enabled_calls = 0;
inline int fake_set_enabled_error = 0;
inline std::multiset<uint64_t> fake_uploads;
inline unsigned fake_events_live = 0;

inline bool fake_node_on(uint64_t exec, uint64_t node) {
    auto it = fake_enabled.find({exec, node});
    return it == fake_enabled.end() ? true : it->second;
}

inline int cudaGraphNodeSetEnabled(cudaGraphExec_t ex, cudaGraphNode_t node,
                                   unsigned on) {
    ++fake_set_enabled_calls;
    if (fake_set_enabled_error) return fake_set_enabled_error;
    fake_enabled[{reinterpret_cast<uint64_t>(ex),
                  reinterpret_cast<uint64_t>(node)}] = on != 0;
    return 0;
}
// Stream capture as the producer ops observe it.  fake_capture_graph != null
// means "the stream is capturing into that graph"; every query then reports
// one tail dependency, a fresh kernel node (fake_tail_type says otherwise).
inline cudaGraph_t fake_capture_graph = nullptr;
inline cudaGraphNodeType fake_tail_type = cudaGraphNodeTypeKernel;
inline uintptr_t fake_next_node = 0x1000;
inline cudaGraphNode_t fake_tail_node = nullptr;
inline int cudaStreamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus* st,
                                    unsigned long long* id, cudaGraph_t* g,
                                    const cudaGraphNode_t** deps, size_t* nd) {
    if (fake_capture_graph == nullptr) {
        if (st) *st = cudaStreamCaptureStatusNone;
        return 0;
    }
    fake_tail_node = reinterpret_cast<cudaGraphNode_t>(fake_next_node++);
    if (st) *st = cudaStreamCaptureStatusActive;
    if (id) *id = 1;
    if (g) *g = fake_capture_graph;
    if (deps) *deps = &fake_tail_node;
    if (nd) *nd = 1;
    return 0;
}
inline int cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType* type) {
    *type = fake_tail_type;
    return 0;
}
inline int cudaMallocManaged(void** p, size_t n) {
    *p = std::malloc(n);
    return *p ? 0 : 1;
}
inline int cudaFree(void* p) { std::free(p); return 0; }
// Pinned host memory (ring/pinned_staging.h).  Declared only, with the C
// linkage of the real runtime API: a host-only test that reaches them must
// define them itself (or never call them, and drop them at link time).
constexpr unsigned cudaHostAllocDefault = 0;
extern "C" cudaError_t cudaHostAlloc(void** p, size_t n, unsigned flags);
extern "C" cudaError_t cudaFreeHost(void* p);
inline int cudaStreamCreateWithFlags(cudaStream_t* p, unsigned) {
    *p = reinterpret_cast<void*>(1);
    return 0;
}
inline int cudaGraphUpload(cudaGraphExec_t ex, cudaStream_t) {
    fake_uploads.insert(reinterpret_cast<uint64_t>(ex));
    return 0;
}
inline int cudaStreamSynchronize(cudaStream_t) { return 0; }
inline int cudaStreamDestroy(cudaStream_t) { return 0; }
inline int cudaEventSynchronize(cudaEvent_t) { return 0; }
inline int cudaEventCreateWithFlags(cudaEvent_t* p, unsigned) {
    ++fake_events_live;
    *p = reinterpret_cast<void*>(2);
    return 0;
}
inline int cudaEventRecord(cudaEvent_t, cudaStream_t) { return 0; }
inline int cudaEventDestroy(cudaEvent_t) { --fake_events_live; return 0; }
