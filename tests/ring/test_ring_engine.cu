// tests/ring/test_ring_engine.cu — End-to-end tests for the ring drain pipeline.
//
// Tests the full path:
//   GPU producer kernel  →  D2H copy engine  →  CPU drain thread
//   →  chunk assembler  →  AssembledTensor callback
//
// Build: see tests/ring/Makefile (target: test_ring_engine)
// Run:   ./build/test_ring_engine

#include "ring/ring_alloc.h"
#include "ring/producer.cuh"
#include "ring/pinned_pool.h"
#include "ring/drain_thread.h"
#include "ring/chunk_assembler.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdio.h>
#include <vector>

using namespace ring;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
struct ResultCollector {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<AssembledTensor> tensors;

    void push(AssembledTensor&& t) {
        std::lock_guard<std::mutex> lk(mu);
        tensors.push_back(std::move(t));
        cv.notify_all();
    }

    // Wait until at least `n` tensors have been delivered, up to timeout_ms.
    bool wait_for(int n, int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                           [&] { return (int)tensors.size() >= n; });
    }
};

// Build a RingConfig suitable for most tests.
static RingConfig make_cfg(uint64_t chunk_bytes = 256 * 1024,
                           uint64_t payload_bytes = 4 * 1024 * 1024,
                           uint32_t task_slots = 64,
                           uint32_t pinned_blocks = 8)
{
    RingConfig cfg{};
    cfg.task_ring_entries   = task_slots;
    cfg.payload_ring_bytes  = payload_bytes;
    cfg.chunk_bytes         = chunk_bytes;
    cfg.pinned_pool_blocks  = pinned_blocks;
    cfg.wait_policy         = WaitPolicy::INFINITE;
    cfg.drop_reporting      = DropReporting::DROP_TASK;
    return cfg;
}

// Allocate and upload a host vector to GPU device memory.
static uint8_t* upload(const std::vector<uint8_t>& h, cudaStream_t s = 0) {
    uint8_t* d;
    cudaMalloc(&d, h.size());
    cudaMemcpyAsync(d, h.data(), h.size(), cudaMemcpyHostToDevice, s);
    return d;
}

// ---------------------------------------------------------------------------
// test_single_chunk
//   One tensor that fits in a single chunk.
//   Verifies the assembled data matches the original.
// ---------------------------------------------------------------------------
static void test_single_chunk() {
    printf("  test_single_chunk ... ");

    const uint64_t data_bytes = 128 * 1024;  // 128 KB < chunk_bytes (256 KB)
    RingConfig cfg = make_cfg();

    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    std::vector<uint8_t> src(data_bytes);
    std::iota(src.begin(), src.end(), uint8_t(0));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/1, /*hook_type=*/0, /*hook_id=*/0,
                                DrainThread::hostfunc_cb, &dt, stream);

    assert(rc.wait_for(1));

    const auto& t = rc.tensors[0];
    assert(!t.is_drop);
    assert(t.logical_task_id == 1);
    assert(t.data.size() == data_bytes);
    assert(memcmp(t.data.data(), src.data(), data_bytes) == 0);

    dt.stop();
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_multi_chunk
//   One large tensor that spans three chunks.
//   Verifies the assembler concatenates them correctly.
// ---------------------------------------------------------------------------
static void test_multi_chunk() {
    printf("  test_multi_chunk ... ");

    const uint64_t chunk_bytes = 64 * 1024;          // 64 KB
    const uint64_t data_bytes  = 3 * chunk_bytes + 7; // 3 full + 1 partial
    RingConfig cfg = make_cfg(chunk_bytes, 2 * 1024 * 1024);

    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    std::vector<uint8_t> src(data_bytes);
    for (uint64_t i = 0; i < data_bytes; ++i) src[i] = uint8_t(i * 7 + 3);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/2, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    assert(rc.wait_for(1));

    const auto& t = rc.tensors[0];
    assert(!t.is_drop);
    assert(t.data.size() == data_bytes);
    assert(memcmp(t.data.data(), src.data(), data_bytes) == 0);

    dt.stop();
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_data_correctness_wrap
//   Force the payload ring to wrap around mid-tensor so the two-span path
//   is exercised.  payload ring is 512 KB, chunk is 384 KB, we pre-advance
//   payload_head to 480 KB so the first chunk wraps.
// ---------------------------------------------------------------------------
static void test_wrap() {
    printf("  test_wrap ... ");

    const uint64_t payload_cap  = 512 * 1024;
    const uint64_t chunk_bytes  = 384 * 1024;  // > half ring → forces wrap
    RingConfig cfg = make_cfg(chunk_bytes, payload_cap, 32, 4);

    AllocatedRing ar(cfg);
    ar.init();

    // Pre-advance head and tail so head is near the end of the ring.
    // Free space = cap - (head - tail) = 512K - (480K - 480K + slack)
    // Set head=480K, tail=0 → free = 512K - 480K = 32K < 384K → deadlock.
    // Set head=480K, tail=480K - (512K - 384K) = 480K - 128K = 352K
    // → free = 512K - (480K - 352K) = 512K - 128K = 384K  ✓
    const uint64_t pre_head = 480 * 1024;
    const uint64_t pre_tail = pre_head - (payload_cap - chunk_bytes);
    *ar.state().payload_head = pre_head;
    *ar.state().payload_tail = pre_tail;

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    // Sync drain's local payload_tail with our pre-set value.
    dt.start();

    std::vector<uint8_t> src(chunk_bytes);
    for (uint64_t i = 0; i < chunk_bytes; ++i) src[i] = uint8_t(i ^ 0xA5);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, chunk_bytes,
                                /*logical_task_id=*/3, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    assert(rc.wait_for(1));

    const auto& t = rc.tensors[0];
    assert(!t.is_drop);
    assert(t.data.size() == chunk_bytes);
    assert(memcmp(t.data.data(), src.data(), chunk_bytes) == 0);

    dt.stop();
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_drop
//   Fill the ring so the producer times out and emits a DROP marker.
//   The drain thread should deliver a drop AssembledTensor.
// ---------------------------------------------------------------------------
static void test_drop() {
    printf("  test_drop ... ");

    RingConfig cfg = make_cfg(64 * 1024, 128 * 1024, 4, 4);
    cfg.wait_policy             = WaitPolicy::TIMEOUT_DROP;
    cfg.no_progress_timeout_cycles = 1000000ULL;  // ~0.4 ms at 2.5 GHz
    cfg.drop_reporting          = DropReporting::DROP_TASK;

    AllocatedRing ar(cfg);
    ar.init();

    // Completely fill the payload ring so producer immediately drops.
    *ar.state().payload_head = cfg.payload_ring_bytes;
    *ar.state().payload_tail = 0;  // free = 0

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    const uint64_t data_bytes = 32 * 1024;
    std::vector<uint8_t> src(data_bytes, 0xBE);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/4, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    assert(rc.wait_for(1));
    assert(rc.tensors[0].is_drop);

    dt.stop();
    cudaStreamSynchronize(stream);
    cudaFree(d_src);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_multiple_tensors
//   Submit N independent tensors back-to-back on the same stream.
//   Verifies the drain thread handles queued notifications correctly.
// ---------------------------------------------------------------------------
static void test_multiple_tensors() {
    printf("  test_multiple_tensors ... ");

    const int      N          = 8;
    const uint64_t data_bytes = 32 * 1024;  // 32 KB each
    RingConfig cfg = make_cfg(64 * 1024, 4 * 1024 * 1024, 64, 16);

    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
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

    assert(rc.wait_for(N));
    assert((int)rc.tensors.size() == N);
    for (int i = 0; i < N; ++i) {
        const auto& t = rc.tensors[i];
        assert(!t.is_drop);
        assert(t.data.size() == data_bytes);
        // Find the matching source by logical_task_id.
        int idx = static_cast<int>(t.logical_task_id) - 10;
        assert(idx >= 0 && idx < N);
        assert(memcmp(t.data.data(), srcs[idx].data(), data_bytes) == 0);
    }

    dt.stop();
    cudaStreamSynchronize(stream);
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

    RingConfig cfg = make_cfg();
    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(cfg.pinned_pool_blocks, cfg.chunk_bytes);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // src_bytes=0: producer emits one entry with zero-length payload.
    launch_producer_with_notify(ar.state(), nullptr, 0,
                                /*logical_task_id=*/99, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);

    assert(rc.wait_for(1));
    const auto& t = rc.tensors[0];
    assert(!t.is_drop);
    assert(t.logical_task_id == 99);
    assert(t.data.empty());

    dt.stop();
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
int main() {
    printf("test_ring_engine\n");
    test_single_chunk();
    test_multi_chunk();
    test_wrap();
    test_drop();
    test_multiple_tensors();
    test_zero_byte_tensor();
    printf("All tests passed.\n");
    return 0;
}
