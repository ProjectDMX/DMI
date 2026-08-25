// tests/ring/test_ring_engine.cu — End-to-end tests for the ring drain pipeline.
//
// Tests the full path:
//   GPU producer kernel  →  drain thread (batch D2H into pinned staging)
//   →  task queue  →  verify DrainTask contents
//
// NOTE: The new drain pipeline (batch D2H + pinned staging + p2p thread)
// requires ATen linkage for DrainTask (contains at::Tensor).  The p2p thread
// and full tensor assembly are tested via the Python E2E tests
// (tests/test_e2e_correctness_vs_hf.py) which exercise the entire pipeline
// including metadata FIFO, reshape, slicing, and ClickHouse submission.
//
// These C++ tests verify the drain-side pipeline: producer → drain thread →
// task queue, confirming that staged data matches the original GPU tensor.
//
// Build: make test_ring_engine  (requires PyTorch for ATen headers)

#include "ring/ring_alloc.h"
#include "ring/producer.cuh"
#include "ring/drain_thread.h"
#include "ring/pinned_staging.h"

#include <cassert>
#include <cstring>
#include <numeric>
#include <stdio.h>
#include <vector>

using namespace ring;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Allocate and upload a host vector to GPU device memory.
static uint8_t* upload(const std::vector<uint8_t>& h, cudaStream_t s = 0) {
    uint8_t* d;
    cudaMalloc(&d, h.size());
    cudaMemcpyAsync(d, h.data(), h.size(), cudaMemcpyHostToDevice, s);
    return d;
}

// Verify staging data for a single-chunk tensor matches source.
static void verify_staging_data(const DrainTask& task,
                                const std::vector<uint8_t>& src) {
    uint64_t data_len = task.data_len1 + task.data_len2;
    assert(data_len == src.size());
    // Copy from staging (may wrap)
    std::vector<uint8_t> got(data_len);
    if (task.data_len1 > 0)
        memcpy(got.data(), task.data_ptr1, task.data_len1);
    if (task.data_len2 > 0)
        memcpy(got.data() + task.data_len1, task.data_ptr2, task.data_len2);
    assert(memcmp(got.data(), src.data(), src.size()) == 0);
}

// ---------------------------------------------------------------------------
// test_single_chunk
//   One tensor that fits in a single chunk.
//   Verifies drain task fields and staged data match the original.
// ---------------------------------------------------------------------------
static void test_single_chunk() {
    printf("  test_single_chunk ... ");

    const uint64_t data_bytes = 128 * 1024;  // 128 KB < chunk_bytes (256 KB)
    RingConfig cfg{};
    cfg.task_ring_entries  = 64;
    cfg.payload_ring_bytes = 4 * 1024 * 1024;
    cfg.chunk_bytes        = 256 * 1024;
    cfg.wait_policy        = WaitPolicy::INFINITE;
    cfg.drop_reporting     = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    PinnedStaging staging;
    staging.init(cfg.payload_ring_bytes);

    DrainThread dt(ar.state(), staging, cfg);
    dt.start();

    std::vector<uint8_t> src(data_bytes);
    std::iota(src.begin(), src.end(), uint8_t(0));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/1, /*hook_type=*/0, /*hook_id=*/0,
                                DrainThread::hostfunc_cb, &dt, stream);

    // Stop drain to trigger final flush
    cudaStreamSynchronize(stream);
    dt.stop();

    // Now collect tasks (drain has pushed them before joining)
    // Signal p2p stop so wait_for_tasks returns
    dt.signal_p2p_stop();
    uint64_t n = dt.wait_for_tasks();
    assert(n > 0);
    std::vector<DrainTask> tasks;
    dt.pop_tasks(n, tasks);
    assert(tasks.size() == 1);

    const auto& t = tasks[0];
    assert(t.logical_task_id == 1);
    assert(t.tensor_total_bytes == data_bytes);
    assert(t.flags & TASK_FLAG_IS_FIRST);
    assert(t.flags & TASK_FLAG_IS_LAST);
    verify_staging_data(t, src);

    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_multi_chunk
//   One large tensor that spans multiple chunks.
//   Verifies all chunks arrive with correct metadata.
// ---------------------------------------------------------------------------
static void test_multi_chunk() {
    printf("  test_multi_chunk ... ");

    const uint64_t chunk_bytes = 64 * 1024;           // 64 KB
    const uint64_t data_bytes  = 3 * chunk_bytes + 48; // 3 full + 1 partial (aligned to 16)
    RingConfig cfg{};
    cfg.task_ring_entries  = 64;
    cfg.payload_ring_bytes = 2 * 1024 * 1024;
    cfg.chunk_bytes        = chunk_bytes;
    cfg.wait_policy        = WaitPolicy::INFINITE;
    cfg.drop_reporting     = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    PinnedStaging staging;
    staging.init(cfg.payload_ring_bytes);

    DrainThread dt(ar.state(), staging, cfg);
    dt.start();

    std::vector<uint8_t> src(data_bytes);
    for (uint64_t i = 0; i < data_bytes; ++i) src[i] = uint8_t(i * 7 + 3);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/2, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    cudaStreamSynchronize(stream);
    dt.stop();
    dt.signal_p2p_stop();

    uint64_t n = dt.wait_for_tasks();
    assert(n > 0);
    std::vector<DrainTask> tasks;
    dt.pop_tasks(n, tasks);
    assert(tasks.size() == 4);  // 3 full + 1 partial

    // Reassemble from staging
    std::vector<uint8_t> assembled(data_bytes);
    for (const auto& t : tasks) {
        if (t.data_len1 > 0)
            memcpy(assembled.data() + t.chunk_offset_bytes, t.data_ptr1, t.data_len1);
        if (t.data_len2 > 0)
            memcpy(assembled.data() + t.chunk_offset_bytes + t.data_len1,
                   t.data_ptr2, t.data_len2);
    }
    assert(memcmp(assembled.data(), src.data(), data_bytes) == 0);

    assert(tasks[0].flags & TASK_FLAG_IS_FIRST);
    assert(tasks[3].flags & TASK_FLAG_IS_LAST);
    for (const auto& t : tasks) {
        assert(t.tensor_total_bytes == data_bytes);
        assert(t.logical_task_id == 2);
    }

    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_drop
//   Fill the ring so the producer times out and emits a DROP marker.
// ---------------------------------------------------------------------------
static void test_drop() {
    printf("  test_drop ... ");

    RingConfig cfg{};
    cfg.task_ring_entries          = 4;
    cfg.payload_ring_bytes         = 128 * 1024;
    cfg.chunk_bytes                = 64 * 1024;
    cfg.wait_policy                = WaitPolicy::TIMEOUT_DROP;
    cfg.no_progress_timeout_cycles = 1000000ULL;  // ~0.4 ms at 2.5 GHz
    cfg.drop_reporting             = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    // Completely fill the payload ring so producer immediately drops.
    *ar.state().payload_head = cfg.payload_ring_bytes;
    *ar.state().payload_tail = 0;  // free = 0

    PinnedStaging staging;
    staging.init(cfg.payload_ring_bytes);

    DrainThread dt(ar.state(), staging, cfg);
    dt.start();

    const uint64_t data_bytes = 32 * 1024;
    std::vector<uint8_t> src(data_bytes, 0xBE);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/4, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    cudaStreamSynchronize(stream);
    dt.stop();
    dt.signal_p2p_stop();

    uint64_t n = dt.wait_for_tasks();
    assert(n > 0);
    std::vector<DrainTask> tasks;
    dt.pop_tasks(n, tasks);
    assert(tasks.size() == 1);
    assert(tasks[0].flags & TASK_FLAG_IS_DROP);

    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_multiple_tensors
//   Submit N independent tensors back-to-back.
//   Verifies drain handles queued notifications and data correctness.
// ---------------------------------------------------------------------------
static void test_multiple_tensors() {
    printf("  test_multiple_tensors ... ");

    const int      N          = 8;
    const uint64_t data_bytes = 32 * 1024;  // 32 KB each
    RingConfig cfg{};
    cfg.task_ring_entries  = 64;
    cfg.payload_ring_bytes = 4 * 1024 * 1024;
    cfg.chunk_bytes        = 64 * 1024;
    cfg.wait_policy        = WaitPolicy::INFINITE;
    cfg.drop_reporting     = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    PinnedStaging staging;
    staging.init(cfg.payload_ring_bytes);

    DrainThread dt(ar.state(), staging, cfg);
    dt.start();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<uint8_t*> d_srcs(N);
    std::vector<std::vector<uint8_t>> srcs(N);
    for (int i = 0; i < N; ++i) {
        srcs[i].resize(data_bytes);
        for (uint64_t j = 0; j < data_bytes; ++j) srcs[i][j] = uint8_t(i + j);
        d_srcs[i] = upload(srcs[i], stream);
        launch_producer_with_notify(ar.state(), d_srcs[i], data_bytes,
                                    /*logical_task_id=*/uint64_t(10 + i), 0, 0,
                                    DrainThread::hostfunc_cb, &dt, stream);
    }

    cudaStreamSynchronize(stream);
    dt.stop();
    dt.signal_p2p_stop();

    // Collect all tasks
    std::vector<DrainTask> all_tasks;
    while (true) {
        uint64_t n = dt.wait_for_tasks();
        if (n == 0) break;
        std::vector<DrainTask> batch;
        dt.pop_tasks(n, batch);
        for (auto& t : batch) all_tasks.push_back(std::move(t));
    }
    assert(static_cast<int>(all_tasks.size()) == N);

    // Each should be a single-chunk tensor (32KB < 64KB chunk)
    for (int i = 0; i < N; ++i) {
        const auto& t = all_tasks[i];
        assert(!(t.flags & TASK_FLAG_IS_DROP));
        assert(t.flags & TASK_FLAG_IS_FIRST);
        assert(t.flags & TASK_FLAG_IS_LAST);
        assert(t.tensor_total_bytes == data_bytes);

        int idx = static_cast<int>(t.logical_task_id) - 10;
        assert(idx >= 0 && idx < N);
        verify_staging_data(t, srcs[idx]);
    }

    for (auto d : d_srcs) cudaFree(d);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_zero_byte_tensor
//   A tensor with src_bytes=0 produces one task entry with no payload.
// ---------------------------------------------------------------------------
static void test_zero_byte_tensor() {
    printf("  test_zero_byte_tensor ... ");

    RingConfig cfg{};
    cfg.task_ring_entries  = 64;
    cfg.payload_ring_bytes = 4 * 1024 * 1024;
    cfg.chunk_bytes        = 256 * 1024;
    cfg.wait_policy        = WaitPolicy::INFINITE;
    cfg.drop_reporting     = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    PinnedStaging staging;
    staging.init(cfg.payload_ring_bytes);

    DrainThread dt(ar.state(), staging, cfg);
    dt.start();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    launch_producer_with_notify(ar.state(), nullptr, 0,
                                /*logical_task_id=*/99, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    cudaStreamSynchronize(stream);
    dt.stop();
    dt.signal_p2p_stop();

    uint64_t n = dt.wait_for_tasks();
    assert(n > 0);
    std::vector<DrainTask> tasks;
    dt.pop_tasks(n, tasks);
    assert(tasks.size() == 1);

    const auto& t = tasks[0];
    assert(!(t.flags & TASK_FLAG_IS_DROP));
    assert(t.logical_task_id == 99);
    assert(t.tensor_total_bytes == 0);
    assert(t.data_len1 == 0);
    assert(t.data_len2 == 0);

    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
int main() {
    printf("test_ring_engine (batch drain pipeline)\n");
    test_single_chunk();
    test_multi_chunk();
    test_drop();
    test_multiple_tensors();
    test_zero_byte_tensor();
    printf("All 5 tests passed.\n");
    return 0;
}
