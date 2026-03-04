// ring/ring_alloc.h — CPU-side RAII owner of all ring device memory.
//
// AllocatedRing allocates and initialises all buffers for one ring pair:
//   - task entry array          (cudaMalloc — device-only)
//   - payload byte buffer       (cudaMalloc — device-only)
//   - head/tail counters        (cudaMallocManaged — GPU writes, CPU reads)
//   - consumer_heartbeat        (cudaMallocManaged — CPU writes, GPU reads)
//
// Usage:
//   AllocatedRing ar(cfg);
//   ar.init();                     // memset entries to SENTINEL, zero counters
//   RingState rs = ar.state();     // capture-safe snapshot of pointers
//   launch_producer(rs, ...);      // pass rs into kernel
//
// Must be compiled with nvcc (requires __CUDACC__ for task_ring_init).

#pragma once
#include "ring_state.h"
#include "task_ring.cuh"   // task_ring_init (needs __CUDACC__)

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ring {

class AllocatedRing {
public:
    explicit AllocatedRing(const RingConfig& cfg) : cfg_(cfg) { allocate(); }
    ~AllocatedRing() noexcept { free_all(); }

    AllocatedRing(const AllocatedRing&)            = delete;
    AllocatedRing& operator=(const AllocatedRing&) = delete;

    // Zero all counters and set every task entry byte to 0xFF (SENTINEL).
    // Call once on host before the first graph capture.
    void init(cudaStream_t stream = 0) {
        task_ring_init(state_.task_entries, cfg_.task_ring_entries, stream);
        *state_.task_head          = 0;
        *state_.task_tail          = 0;
        *state_.payload_head       = 0;
        *state_.payload_tail       = 0;
        *state_.consumer_heartbeat = 0;
    }

    RingState&       state()        { return state_; }
    const RingState& state()  const { return state_; }
    const RingConfig& config() const { return cfg_; }

private:
    RingConfig cfg_;
    RingState  state_{};

    static void chk(cudaError_t e, const char* ctx) {
        if (e != cudaSuccess)
            throw std::runtime_error(
                std::string("AllocatedRing ") + ctx + ": " + cudaGetErrorString(e));
    }

    void allocate() {
        chk(cudaMalloc(&state_.task_entries,
                       cfg_.task_ring_entries * sizeof(TaskEntry)),
            "cudaMalloc task_entries");
        chk(cudaMalloc(&state_.payload_buf, cfg_.payload_ring_bytes),
            "cudaMalloc payload_buf");

        auto mg = [&](uint64_t** pp, const char* name) {
            chk(cudaMallocManaged(pp, sizeof(uint64_t)), name);
        };
        mg(&state_.task_head,          "task_head");
        mg(&state_.task_tail,          "task_tail");
        mg(&state_.payload_head,       "payload_head");
        mg(&state_.payload_tail,       "payload_tail");
        mg(&state_.consumer_heartbeat, "consumer_heartbeat");

        state_.task_cap    = cfg_.task_ring_entries;
        state_.payload_cap = cfg_.payload_ring_bytes;
        state_.cfg         = cfg_;
    }

    void free_all() noexcept {
        cudaFree(state_.task_entries);
        cudaFree(state_.payload_buf);
        cudaFree(state_.task_head);
        cudaFree(state_.task_tail);
        cudaFree(state_.payload_head);
        cudaFree(state_.payload_tail);
        cudaFree(state_.consumer_heartbeat);
    }
};

}  // namespace ring
