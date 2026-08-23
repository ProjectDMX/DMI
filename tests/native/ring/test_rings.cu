// Standalone CUDA tests for the current one-tensor-per-entry ring primitives.

#include "ring/payload_ring.cuh"
#include "ring/ring_config.h"
#include "ring/task_entry.h"
#include "ring/task_ring.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

static void test_payload_accounting() {
    banner("payload accounting and alignment");
    using namespace ring;

    EXPECT(align_up(0, PAYLOAD_ALIGN) == 0);
    EXPECT(align_up(1, PAYLOAD_ALIGN) == 16);
    EXPECT(align_up(31, PAYLOAD_ALIGN) == 32);
    EXPECT(payload_free_bytes(0, 0, 1024) == 1024);
    EXPECT(payload_free_bytes(512, 0, 1024) == 512);
    EXPECT(payload_free_bytes(1100, 100, 1024) == 24);
    EXPECT(payload_free_bytes(800, 300, 1024) == 524);

    uint64_t head = 10;
    uint64_t tail = 3;
    payload_advance_head(head, 7);
    payload_release(tail, 5);
    EXPECT(head == 17);
    EXPECT(tail == 8);
}

static void test_payload_spans() {
    banner("payload single-span and wrap descriptors");
    using namespace ring;

    TwoSpan span = payload_compute_spans(100, 256, 50);
    EXPECT(span.off1 == 100);
    EXPECT(span.len1 == 50);
    EXPECT(span.off2 == 0);
    EXPECT(span.len2 == 0);

    span = payload_compute_spans(250, 256, 32);
    EXPECT(span.off1 == 250);
    EXPECT(span.len1 == 6);
    EXPECT(span.off2 == 0);
    EXPECT(span.len2 == 26);
    EXPECT(payload_chunk_bytes(span.len1, span.len2) == 32);

    span = payload_compute_spans(512, 256, 256);
    EXPECT(span.off1 == 0);
    EXPECT(span.len1 == 256);
    EXPECT(span.len2 == 0);
}

static void test_task_accounting_and_layout() {
    banner("task accounting and 64-byte layout");
    using namespace ring;

    EXPECT(task_free_slots(0, 0, 16) == 16);
    EXPECT(task_free_slots(8, 0, 16) == 8);
    EXPECT(task_free_slots(16, 0, 16) == 0);
    EXPECT(task_free_slots(1024, 1020, 16) == 12);

    EXPECT(sizeof(TaskEntry) == 64);
    EXPECT(alignof(TaskEntry) == 64);
    EXPECT(offsetof(TaskEntry, ready_seq) == 0);
    EXPECT(offsetof(TaskEntry, tensor_total_bytes) == 8);
    EXPECT(offsetof(TaskEntry, payload_off1) == 16);
    EXPECT(offsetof(TaskEntry, payload_len1) == 24);
    EXPECT(offsetof(TaskEntry, payload_off2) == 32);
    EXPECT(offsetof(TaskEntry, payload_len2) == 40);
}

static void test_config_defaults() {
    banner("RingConfig defaults");
    using namespace ring;

    RingConfig cfg;
    EXPECT(cfg.task_ring_entries == 1024);
    EXPECT(cfg.payload_ring_bytes == 256ULL * 1024 * 1024);
    EXPECT(cfg.pinned_staging_bytes == 0);
    EXPECT(cfg.drain_poll_timeout_us == 100);
    EXPECT(cfg.drain_flush.payload_ratio == 0.5f);
    EXPECT(cfg.effective_staging_bytes() == cfg.payload_ring_bytes);

    cfg.pinned_staging_bytes = 4096;
    EXPECT(cfg.effective_staging_bytes() == 4096);
}

__global__ void publish_range(ring::TaskEntry* entries, uint64_t capacity,
                              uint64_t start, uint64_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t sequence = start + i;
        ring::TaskEntry entry{};
        entry.tensor_total_bytes = 1000 + sequence;
        entry.payload_off1 = sequence * 16;
        entry.payload_len1 = 900 + sequence;
        entry.payload_off2 = 0;
        entry.payload_len2 = 100;
        ring::task_publish(entries, capacity, sequence, entry);
    }
}

static ring::TaskEntry* allocate_entries(uint64_t capacity) {
    ring::TaskEntry* entries = nullptr;
    CUDA_CHECK(cudaMallocManaged(&entries, capacity * sizeof(ring::TaskEntry)));
    ring::task_ring_init(entries, capacity);
    CUDA_CHECK(cudaDeviceSynchronize());
    return entries;
}

static void test_task_fifo_and_ready_lifecycle() {
    banner("task FIFO and ready lifecycle");
    using namespace ring;

    constexpr uint64_t capacity = 64;
    constexpr uint64_t count = 50;
    TaskEntry* entries = allocate_entries(capacity);

    EXPECT(!task_cpu_ready(entries, capacity, 0));
    publish_range<<<1, 1>>>(entries, capacity, 0, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (uint64_t sequence = 0; sequence < count; ++sequence) {
        EXPECT(task_cpu_ready(entries, capacity, sequence));
        const TaskEntry& entry = entries[sequence % capacity];
        EXPECT(entry.ready_seq == sequence);
        EXPECT(entry.tensor_total_bytes == 1000 + sequence);
        EXPECT(entry.payload_off1 == sequence * 16);
        EXPECT(entry.payload_len1 + entry.payload_len2 == 1000 + sequence);
        task_release_cpu(entries, capacity, sequence);
        EXPECT(!task_cpu_ready(entries, capacity, sequence));
        EXPECT(entry.ready_seq == READY_SEQ_SENTINEL);
    }
    EXPECT(!task_cpu_ready(entries, capacity, count));
    CUDA_CHECK(cudaFree(entries));
}

static void test_task_wrap_reuse() {
    banner("task slot reuse across wraps");
    using namespace ring;

    constexpr uint64_t capacity = 4;
    TaskEntry* entries = allocate_entries(capacity);

    for (uint64_t sequence = 0; sequence < capacity * 3; ++sequence) {
        publish_range<<<1, 1>>>(entries, capacity, sequence, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT(task_cpu_ready(entries, capacity, sequence));
        EXPECT(entries[sequence % capacity].tensor_total_bytes == 1000 + sequence);
        task_release_cpu(entries, capacity, sequence);
        EXPECT(entries[sequence % capacity].ready_seq == READY_SEQ_SENTINEL);
    }
    CUDA_CHECK(cudaFree(entries));
}

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_rings (current ring primitives)\n");

    test_payload_accounting();
    test_payload_spans();
    test_task_accounting_and_layout();
    test_config_defaults();
    test_task_fifo_and_ready_lifecycle();
    test_task_wrap_reuse();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
