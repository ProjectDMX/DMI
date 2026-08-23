// CUDA tests for producer null mode with the current producer variants.

#include "ring/producer.cuh"
#include "ring/ring_alloc.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t error_ = (expr);                                        \
        if (error_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(error_));    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define EXPECT(condition)                                                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                      \
                         __FILE__, __LINE__, #condition);                    \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

static ring::RingConfig make_config() {
    ring::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = 4096;
    cfg.pinned_staging_bytes = 4096;
    return cfg;
}

static uint8_t* upload_bytes(uint64_t size) {
    std::vector<uint8_t> source(size);
    for (uint64_t i = 0; i < size; ++i) {
        source[i] = static_cast<uint8_t>(i * 13);
    }
    uint8_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, size));
    CUDA_CHECK(cudaMemcpy(device, source.data(), size, cudaMemcpyHostToDevice));
    return device;
}

static int64_t* upload_counts(const std::vector<int64_t>& counts) {
    int64_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, counts.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(device, counts.data(),
                          counts.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    return device;
}

static void set_null_mode(bool enabled) {
    CUDA_CHECK(cudaDeviceSynchronize());
    ring::set_ring_null_mode(enabled);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void expect_empty(const ring::RingState& state) {
    EXPECT(*state.task_head == 0);
    EXPECT(*state.payload_head == 0);
    EXPECT(*state.actual_bytes_counter == 0);
    for (uint64_t i = 0; i < state.task_cap; ++i) {
        EXPECT(state.task_entries[i].ready_seq == ring::READY_SEQ_SENTINEL);
    }
}

static void test_all_variants_are_noops() {
    std::printf("[ TEST ] all producer variants are no-ops\n");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    uint8_t* source = upload_bytes(256);
    int64_t* row_count = upload_counts({3});
    int64_t* chunk_counts = upload_counts({16, 32, 0, 8});
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    set_null_mode(true);
    ring::launch_producer_static(state, source, 256, 0, stream);
    ring::launch_producer_prefix(state, source, 256, row_count, 32, 0, stream);
    ring::launch_producer_chunked(state, source, 256, chunk_counts, 4, 0,
                                  stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    expect_empty(state);

    set_null_mode(false);
    CUDA_CHECK(cudaFree(chunk_counts));
    CUDA_CHECK(cudaFree(row_count));
    CUDA_CHECK(cudaFree(source));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

static void test_disabling_restores_delivery() {
    std::printf("[ TEST ] disabling null mode restores delivery\n");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();
    uint8_t* source = upload_bytes(128);

    set_null_mode(true);
    ring::launch_producer_static(state, source, 128, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_empty(state);

    set_null_mode(false);
    ring::launch_producer_static(state, source, 128, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT(*state.task_head == 1);
    EXPECT(*state.payload_head == 128);
    EXPECT(*state.actual_bytes_counter == 128);
    EXPECT(state.task_entries[0].ready_seq == 0);
    EXPECT(state.task_entries[0].tensor_total_bytes == 128);

    CUDA_CHECK(cudaFree(source));
}

static void test_toggle_preserves_sequence() {
    std::printf("[ TEST ] toggling preserves producer sequence\n");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();
    uint8_t* source = upload_bytes(64);

    set_null_mode(false);
    ring::launch_producer_static(state, source, 64, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    set_null_mode(true);
    ring::launch_producer_static(state, source, 64, 0);
    ring::launch_producer_static(state, source, 64, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    set_null_mode(false);
    ring::launch_producer_static(state, source, 64, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT(*state.task_head == 2);
    EXPECT(*state.payload_head == 128);
    EXPECT(*state.actual_bytes_counter == 128);
    EXPECT(state.task_entries[0].ready_seq == 0);
    EXPECT(state.task_entries[1].ready_seq == 1);

    CUDA_CHECK(cudaFree(source));
}

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_null_mode (current producer variants)\n");

    test_all_variants_are_noops();
    test_disabling_restores_delivery();
    test_toggle_preserves_sequence();
    set_null_mode(false);

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
