// Microbenchmark: D2D producer copy time under the previous size-tier grid
// (1/4/16/64 blocks at 64 KB / 4 MB / 32 MB) and the proportional grid
// (ring::producer_grid_blocks).  Not part of `make run`; build and run with
//   make -C tests/native/ring bench_producer_grid
//
// Each configuration replays a CUDA graph of 64 back-to-back producer
// launches so launch overhead is amortized the same way it is in a captured
// decode step; the reported time is per launch.

#include "../../../native/csrc/ring/producer.cu"
#include "ring/ring_alloc.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t error_ = (expr);                                        \
        if (error_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(error_));    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

static uint32_t previous_tier_blocks(uint64_t nbytes) {
    if (nbytes <= 64ull * 1024) return 1;
    if (nbytes <= 4ull * 1024 * 1024) return 4;
    if (nbytes <= 32ull * 1024 * 1024) return 16;
    return 64;
}

static constexpr int kLaunchesPerGraph = 64;
static constexpr int kReplays = 50;

// Returns microseconds per producer launch.
static double time_static(ring::RingState& state, const uint8_t* src,
                          uint64_t nbytes, uint32_t blocks,
                          cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < kLaunchesPerGraph; ++i) {
        ring::producer_static_kernel<<<blocks, ring::PRODUCER_BLOCK_DIM, 0,
                                       stream>>>(state, src, nbytes, 0);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Warm up, then reset the ring heads between replays from the host
    // (outside the timed region) so every replay writes the same slots.
    for (int warm = 0; warm < 3; ++warm) {
        *state.task_head = 0;
        *state.payload_head = 0;
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    float total_ms = 0.0f;
    for (int replay = 0; replay < kReplays; ++replay) {
        *state.task_head = 0;
        *state.payload_head = 0;
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return 1000.0 * total_ms / (kReplays * kLaunchesPerGraph);
}

int main() {
    setbuf(stdout, nullptr);
    ring::set_ring_null_mode(false);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    const std::vector<uint64_t> sizes = {
        8ull * 1024,           // one decode row of a 4096-wide bf16 hidden state
        64ull * 1024,          // 8 rows
        256ull * 1024,         // 32 rows
        1200ull * 1024,        // ~150 rows
        4ull * 1024 * 1024,    // 512 rows
        16ull * 1024 * 1024,   // a 2048-token prefill chunk
    };
    uint64_t largest = sizes.back();
    ring::RingConfig cfg;
    cfg.task_ring_entries = 128;
    cfg.payload_ring_bytes = largest * (kLaunchesPerGraph + 1);
    cfg.pinned_staging_bytes = 4096;
    ring::AllocatedRing allocated(cfg);
    allocated.init();
    ring::RingState& state = allocated.state();

    uint8_t* src = nullptr;
    CUDA_CHECK(cudaMalloc(&src, largest));
    CUDA_CHECK(cudaMemset(src, 0x5A, largest));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("%12s %10s %12s %10s %12s %8s\n", "bytes", "old_blocks",
                "old_us", "new_blocks", "new_us", "speedup");
    for (uint64_t nbytes : sizes) {
        const uint32_t old_blocks = previous_tier_blocks(nbytes);
        const uint32_t new_blocks = ring::producer_grid_blocks(nbytes);
        const double old_us = time_static(state, src, nbytes, old_blocks, stream);
        const double new_us = time_static(state, src, nbytes, new_blocks, stream);
        std::printf("%12llu %10u %12.2f %10u %12.2f %7.2fx  (%.0f -> %.0f GB/s)\n",
                    static_cast<unsigned long long>(nbytes), old_blocks, old_us,
                    new_blocks, new_us, old_us / new_us,
                    nbytes / (old_us * 1e3), nbytes / (new_us * 1e3));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(src));
    return 0;
}
