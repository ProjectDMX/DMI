// CUDA integration tests for producer -> drain -> pinned-staging delivery.

#include "ring/drain_thread.h"
#include "ring/pinned_staging.h"
#include "ring/producer.cuh"
#include "ring/ring_alloc.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

static void banner(const char* name) {
    std::printf("[ TEST ] %s\n", name);
}

static ring::RingConfig make_config(uint64_t capacity = 4096) {
    ring::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = capacity;
    cfg.pinned_staging_bytes = capacity;
    cfg.drain_poll_timeout_us = 100;
    cfg.drain_flush.task_ratio = 0.0f;
    cfg.drain_flush.payload_ratio = 0.0f;
    cfg.drain_flush.entry_threshold = 0;
    cfg.drain_flush.byte_threshold = 0;
    cfg.drain_flush.timeout_us = 0;
    return cfg;
}

static std::vector<uint8_t> pattern(uint64_t size, uint8_t seed) {
    std::vector<uint8_t> result(size);
    for (uint64_t i = 0; i < size; ++i) {
        result[i] = static_cast<uint8_t>(seed + i * 29);
    }
    return result;
}

static uint8_t* upload(const std::vector<uint8_t>& source,
                       cudaStream_t stream) {
    if (source.empty()) {
        return nullptr;
    }
    uint8_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, source.size()));
    CUDA_CHECK(cudaMemcpyAsync(device, source.data(), source.size(),
                               cudaMemcpyHostToDevice, stream));
    return device;
}

static std::vector<uint8_t> task_bytes(const ring::DrainTask& task) {
    std::vector<uint8_t> result(task.tensor_total_bytes);
    if (task.data_len1 > 0) {
        std::memcpy(result.data(), task.data_ptr1, task.data_len1);
    }
    if (task.data_len2 > 0) {
        std::memcpy(result.data() + task.data_len1,
                    task.data_ptr2, task.data_len2);
    }
    return result;
}

class DrainHarness {
public:
    explicit DrainHarness(const ring::RingConfig& config)
        : cfg(config), allocated(cfg) {
        allocated.init();
        staging.init(cfg.effective_staging_bytes());
        drain = std::make_unique<ring::DrainThread>(
            allocated.state(), staging, cfg);
        CUDA_CHECK(cudaStreamCreate(&stream));
        drain->start();
    }

    ~DrainHarness() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        drain->stop();
        drain->signal_p2p_stop();
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    DrainHarness(const DrainHarness&) = delete;
    DrainHarness& operator=(const DrainHarness&) = delete;

    ring::DrainTask flush_one() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        drain->force_flush_and_wait();
        const uint64_t count = drain->wait_for_tasks();
        EXPECT(count == 1);
        std::vector<ring::DrainTask> tasks;
        drain->pop_tasks(count, tasks);
        EXPECT(tasks.size() == 1);
        return std::move(tasks.front());
    }

    void release(const ring::DrainTask& task) {
        drain->notify_staging_freed_bytes(task.alloc_bytes);
    }

    ring::RingConfig cfg;
    ring::AllocatedRing allocated;
    ring::PinnedStaging staging;
    std::unique_ptr<ring::DrainThread> drain;
    cudaStream_t stream{};
};

static void test_static_force_flush() {
    banner("static producer drains exact data");
    DrainHarness harness(make_config());
    const std::vector<uint8_t> source = pattern(333, 7);
    uint8_t* device = upload(source, harness.stream);

    const uint64_t reserved = ring::align_up(source.size(), ring::PAYLOAD_ALIGN);
    harness.drain->reserve(reserved, 1);
    ring::launch_producer_static(harness.allocated.state(), device,
                                 source.size(), 0, harness.stream);
    ring::DrainTask task = harness.flush_one();

    EXPECT(task.tensor_total_bytes == source.size());
    EXPECT(task.alloc_bytes == reserved);
    EXPECT(task.data_len1 + task.data_len2 == source.size());
    EXPECT(task_bytes(task) == source);
    EXPECT(harness.drain->cpu_task_head() == 1);
    EXPECT(harness.drain->cpu_task_tail_committed() == 1);
    EXPECT(harness.drain->cpu_payload_head() == reserved);
    EXPECT(harness.drain->cpu_payload_tail_committed() == reserved);

    harness.release(task);
    CUDA_CHECK(cudaFree(device));
}

static void test_prefix_force_flush() {
    banner("prefix producer drains selected rows");
    DrainHarness harness(make_config());
    constexpr uint64_t row_bytes = 32;
    const std::vector<uint8_t> source = pattern(8 * row_bytes, 23);
    uint8_t* device = upload(source, harness.stream);
    int64_t row_count = 3;
    int64_t* device_count = nullptr;
    CUDA_CHECK(cudaMalloc(&device_count, sizeof(row_count)));
    CUDA_CHECK(cudaMemcpyAsync(device_count, &row_count, sizeof(row_count),
                               cudaMemcpyHostToDevice, harness.stream));

    constexpr uint64_t actual = 3 * row_bytes;
    harness.drain->reserve(actual, 1);
    ring::launch_producer_prefix(harness.allocated.state(), device,
                                 source.size(), device_count, row_bytes, 0,
                                 harness.stream);
    ring::DrainTask task = harness.flush_one();

    std::vector<uint8_t> expected(source.begin(), source.begin() + actual);
    EXPECT(task.tensor_total_bytes == actual);
    EXPECT(task.alloc_bytes == actual);
    EXPECT(task_bytes(task) == expected);
    EXPECT(harness.drain->cpu_payload_tail_committed() == actual);

    harness.release(task);
    CUDA_CHECK(cudaFree(device_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_repeated_wrap_delivery() {
    banner("repeated flushes preserve data across GPU and staging wraps");
    constexpr uint64_t capacity = 256;
    constexpr uint64_t bytes_per_tensor = 80;
    DrainHarness harness(make_config(capacity));

    for (uint64_t sequence = 0; sequence < 4; ++sequence) {
        const std::vector<uint8_t> source =
            pattern(bytes_per_tensor, static_cast<uint8_t>(50 + sequence));
        uint8_t* device = upload(source, harness.stream);
        harness.drain->reserve(bytes_per_tensor, 1);
        ring::launch_producer_static(harness.allocated.state(), device,
                                     source.size(), 0, harness.stream);
        ring::DrainTask task = harness.flush_one();

        EXPECT(task.tensor_total_bytes == bytes_per_tensor);
        EXPECT(task_bytes(task) == source);
        if (sequence == 3) {
            EXPECT(task.data_len1 == 16);
            EXPECT(task.data_len2 == 64);
        }
        harness.release(task);
        CUDA_CHECK(cudaFree(device));
    }

    EXPECT(*harness.allocated.state().task_head == 4);
    EXPECT(*harness.allocated.state().payload_head == 320);
    EXPECT(harness.drain->cpu_task_tail_committed() == 4);
    EXPECT(harness.drain->cpu_payload_tail_committed() == 320);
    EXPECT(harness.staging.head() == 320);
    EXPECT(harness.staging.tail() == 320);
}

static void test_zero_byte_delivery() {
    banner("zero-byte tensor still publishes one task");
    DrainHarness harness(make_config());
    harness.drain->reserve(0, 1);
    ring::launch_producer_static(harness.allocated.state(), nullptr, 0, 0,
                                 harness.stream);
    ring::DrainTask task = harness.flush_one();

    EXPECT(task.tensor_total_bytes == 0);
    EXPECT(task.alloc_bytes == 0);
    EXPECT(task.data_ptr1 == nullptr);
    EXPECT(task.data_len1 == 0);
    EXPECT(task.data_ptr2 == nullptr);
    EXPECT(task.data_len2 == 0);
    EXPECT(harness.drain->cpu_task_tail_committed() == 1);
    EXPECT(harness.drain->cpu_payload_tail_committed() == 0);
    harness.release(task);
}

int main() {
    setbuf(stdout, nullptr);
    ring::set_ring_null_mode(false);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("test_ring_engine (current drain pipeline)\n");
    test_static_force_flush();
    test_prefix_force_flush();
    test_repeated_wrap_delivery();
    test_zero_byte_delivery();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
