// ring/ring_alloc.h -- CPU-side RAII owner of all ring device memory.
//
// AllocatedRing allocates and initialises all buffers for one ring pair:
//   - task entry array          (cudaMallocManaged -- GPU writes, CPU drain reads)
//   - payload byte buffer       (cudaMalloc -- device-only, D2H via copy engine)
//   - head counters             (cudaMallocManaged -- GPU writes heads)
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
#include "ring_config.h"
#include "task_ring.cuh"   // task_ring_init (needs __CUDACC__)

#include <cuda_runtime.h>
#include <cstring>
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
        *state_.task_head           = 0;
        *state_.payload_head        = 0;
        *state_.actual_bytes_counter = 0;
        // Trigger page migration: move counter pages to GPU HBM and
        // task_entry pages to CPU RAM, then synchronise.
        int dev = 0;
        cudaGetDevice(&dev);
        const size_t          entries_sz = cfg_.task_ring_entries * sizeof(TaskEntry);
        const cudaMemLocation gpu_loc    = {cudaMemLocationTypeDevice, dev};
        const cudaMemLocation cpu_loc    = {cudaMemLocationTypeHost,   0};
        cudaMemPrefetchAsync(state_.task_entries,         entries_sz,        cpu_loc, 0, stream);
        cudaMemPrefetchAsync(state_.task_head,            sizeof(uint64_t),  gpu_loc, 0, stream);
        cudaMemPrefetchAsync(state_.payload_head,         sizeof(uint64_t),  gpu_loc, 0, stream);
        cudaMemPrefetchAsync(state_.actual_bytes_counter, sizeof(uint64_t),  gpu_loc, 0, stream);
        chk(cudaDeviceSynchronize(), "cudaDeviceSynchronize after prefetch");
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
        const size_t entries_sz = cfg_.task_ring_entries * sizeof(TaskEntry);
        chk(cudaMallocManaged(&state_.task_entries, entries_sz),
            "cudaMallocManaged task_entries");
        chk(cudaMalloc(&state_.payload_buf, cfg_.payload_ring_bytes),
            "cudaMalloc payload_buf");

        auto mg = [&](uint64_t** pp, const char* name) {
            chk(cudaMallocManaged(pp, sizeof(uint64_t)), name);
        };
        mg(&state_.task_head,            "task_head");
        mg(&state_.payload_head,         "payload_head");
        mg(&state_.actual_bytes_counter, "actual_bytes_counter");

        state_.task_cap    = cfg_.task_ring_entries;
        state_.payload_cap = cfg_.payload_ring_bytes;

        // Move head counters to GPU HBM so the producer reads them at L2/HBM
        // speed.  CPU writes (drain thread) use PCIe posted writes.
        // Task entries stay on CPU for fast drain-thread polling.
        int dev = 0;
        chk(cudaGetDevice(&dev), "cudaGetDevice");
        const cudaMemLocation gpu_loc  = {cudaMemLocationTypeDevice, dev};
        const cudaMemLocation cpu_loc  = {cudaMemLocationTypeHost,   0};
        auto advise_gpu = [&](void* ptr) {
            chk(cudaMemAdvise(ptr, sizeof(uint64_t),
                              cudaMemAdviseSetPreferredLocation, gpu_loc),
                "cudaMemAdvise SetPreferredLocation counter");
            chk(cudaMemAdvise(ptr, sizeof(uint64_t),
                              cudaMemAdviseSetAccessedBy, gpu_loc),
                "cudaMemAdvise SetAccessedBy counter");
        };
        advise_gpu(state_.task_head);
        advise_gpu(state_.payload_head);
        advise_gpu(state_.actual_bytes_counter);

        chk(cudaMemAdvise(state_.task_entries, entries_sz,
                          cudaMemAdviseSetPreferredLocation, cpu_loc),
            "cudaMemAdvise SetPreferredLocation task_entries CPU");
        chk(cudaMemAdvise(state_.task_entries, entries_sz,
                          cudaMemAdviseSetAccessedBy, gpu_loc),
            "cudaMemAdvise SetAccessedBy task_entries GPU");
    }

    void free_all() noexcept {
        cudaFree(state_.task_entries);
        cudaFree(state_.payload_buf);
        cudaFree(state_.task_head);
        cudaFree(state_.payload_head);
        cudaFree(state_.actual_bytes_counter);
    }
};

}  // namespace ring
