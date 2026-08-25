// tests/ring/test_null_mode.cu — Tests for producer null mode.
//
// Verifies that set_ring_null_mode(true) makes producer_kernel launch with
// the same parameters but skip all ring writes, while
// set_ring_null_mode(false) restores normal delivery.
//
// Build: see tests/ring/Makefile (target: test_null_mode)
// Run:   ./build/test_null_mode

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
// Shared test harness (same as test_ring_engine.cu)
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

    bool wait_for(int n, int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                           [&] { return (int)tensors.size() >= n; });
    }

    int count() {
        std::lock_guard<std::mutex> lk(mu);
        return (int)tensors.size();
    }
};

static constexpr uint64_t PINNED_BYTES = 8 * 256 * 1024;  // 8 chunks worth

static RingConfig make_cfg() {
    RingConfig cfg{};
    cfg.task_ring_entries  = 64;
    cfg.payload_ring_bytes = 4 * 1024 * 1024;
    cfg.chunk_bytes        = 256 * 1024;
    cfg.pinned_pool_bytes  = PINNED_BYTES;
    cfg.wait_policy        = WaitPolicy::INFINITE;
    cfg.drop_reporting     = DropReporting::DROP_TASK;
    return cfg;
}

static uint8_t* upload(const std::vector<uint8_t>& h, cudaStream_t s) {
    uint8_t* d;
    cudaMalloc(&d, h.size());
    cudaMemcpyAsync(d, h.data(), h.size(), cudaMemcpyHostToDevice, s);
    return d;
}

// ---------------------------------------------------------------------------
// test_null_mode_no_delivery
//   With null mode ON, producer kernel launches but emits nothing to the ring.
//   The drain thread should receive no tensors within the timeout window.
// ---------------------------------------------------------------------------
static void test_null_mode_no_delivery() {
    printf("  test_null_mode_no_delivery ... ");

    set_ring_null_mode(true);

    RingConfig cfg = make_cfg();
    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(PINNED_BYTES);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    const uint64_t data_bytes = 64 * 1024;
    std::vector<uint8_t> src(data_bytes, 0xAB);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* d_src = upload(src, stream);

    launch_producer_with_notify(ar.state(), d_src, data_bytes,
                                /*logical_task_id=*/1, 0, 0,
                                DrainThread::hostfunc_cb, &dt, stream);
    cudaStreamSynchronize(stream);

    // Give drain thread time to process any spurious notification.
    bool got = rc.wait_for(1, /*timeout_ms=*/300);
    assert(!got && "null mode: no tensor should arrive");
    assert(rc.count() == 0);

    // task_head must remain 0 — kernel wrote nothing.
    assert(*ar.state().task_head == 0);

    dt.stop();
    cudaFree(d_src);
    cudaStreamDestroy(stream);
    set_ring_null_mode(false);  // restore for subsequent tests
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_null_mode_off_delivers
//   With null mode OFF (default), producer works normally.
// ---------------------------------------------------------------------------
static void test_null_mode_off_delivers() {
    printf("  test_null_mode_off_delivers ... ");

    set_ring_null_mode(false);

    RingConfig cfg = make_cfg();
    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(PINNED_BYTES);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    const uint64_t data_bytes = 64 * 1024;
    std::vector<uint8_t> src(data_bytes);
    std::iota(src.begin(), src.end(), uint8_t(7));
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
// test_toggle
//   Submit N kernels in null mode (none arrive), then toggle off and submit
//   M kernels in real mode (all M arrive, with correct data).
//   Verifies that toggling leaves the ring in a consistent state.
// ---------------------------------------------------------------------------
static void test_toggle() {
    printf("  test_toggle ... ");

    RingConfig cfg = make_cfg();
    AllocatedRing ar(cfg);
    ar.init();

    ResultCollector rc;
    PinnedPool pool;
    pool.init(PINNED_BYTES);

    ChunkAssembler asm_(pool, [&](AssembledTensor&& t) { rc.push(std::move(t)); });
    DrainThread dt(ar.state(), pool, [&](DrainedChunk&& c) { asm_.push(std::move(c)); });
    dt.start();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const uint64_t data_bytes = 32 * 1024;
    const int N_null = 5;
    const int N_real = 4;

    // --- null mode phase ---
    set_ring_null_mode(true);
    std::vector<uint8_t*> null_ptrs(N_null);
    for (int i = 0; i < N_null; ++i) {
        std::vector<uint8_t> src(data_bytes, uint8_t(0xCC));
        null_ptrs[i] = upload(src, stream);
        launch_producer_with_notify(ar.state(), null_ptrs[i], data_bytes,
                                    uint64_t(100 + i), 0, 0,
                                    DrainThread::hostfunc_cb, &dt, stream);
    }
    cudaStreamSynchronize(stream);
    // Brief wait: drain thread must NOT deliver anything.
    bool got = rc.wait_for(1, /*timeout_ms=*/300);
    assert(!got && "toggle: null-mode tensors must not arrive");
    assert(*ar.state().task_head == 0);

    // --- real mode phase ---
    set_ring_null_mode(false);
    std::vector<std::vector<uint8_t>> real_srcs(N_real);
    std::vector<uint8_t*> real_ptrs(N_real);
    for (int i = 0; i < N_real; ++i) {
        real_srcs[i].resize(data_bytes);
        for (uint64_t j = 0; j < data_bytes; ++j) real_srcs[i][j] = uint8_t(i * 13 + j);
        real_ptrs[i] = upload(real_srcs[i], stream);
        launch_producer_with_notify(ar.state(), real_ptrs[i], data_bytes,
                                    uint64_t(200 + i), 0, 0,
                                    DrainThread::hostfunc_cb, &dt, stream);
    }

    assert(rc.wait_for(N_real));
    assert(rc.count() == N_real);
    for (int i = 0; i < N_real; ++i) {
        const auto& t = rc.tensors[i];
        assert(!t.is_drop);
        assert(t.data.size() == data_bytes);
        int idx = static_cast<int>(t.logical_task_id) - 200;
        assert(idx >= 0 && idx < N_real);
        assert(memcmp(t.data.data(), real_srcs[idx].data(), data_bytes) == 0);
    }

    dt.stop();
    cudaStreamSynchronize(stream);
    for (auto p : null_ptrs) cudaFree(p);
    for (auto p : real_ptrs) cudaFree(p);
    cudaStreamDestroy(stream);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// test_null_multiple_launches
//   Launch many kernels in null mode; ring state (task_head, payload_head)
//   must be completely unchanged — no partial writes.
// ---------------------------------------------------------------------------
static void test_null_ring_state_unchanged() {
    printf("  test_null_ring_state_unchanged ... ");

    set_ring_null_mode(true);

    RingConfig cfg = make_cfg();
    AllocatedRing ar(cfg);
    ar.init();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int N = 10;
    const uint64_t data_bytes = 128 * 1024;
    std::vector<uint8_t*> ptrs(N);
    for (int i = 0; i < N; ++i) {
        std::vector<uint8_t> src(data_bytes, uint8_t(i));
        ptrs[i] = upload(src, stream);
        // launch_producer (no hostfunc needed; we just check ring state)
        launch_producer(ar.state(), ptrs[i], data_bytes, uint64_t(i), 0, 0, stream);
    }
    cudaStreamSynchronize(stream);

    assert(*ar.state().task_head    == 0 && "null mode: task_head must be 0");
    assert(*ar.state().payload_head == 0 && "null mode: payload_head must be 0");

    for (auto p : ptrs) cudaFree(p);
    cudaStreamDestroy(stream);
    set_ring_null_mode(false);
    printf("PASS\n");
}

// ---------------------------------------------------------------------------
int main() {
    printf("test_null_mode\n");
    test_null_mode_no_delivery();
    test_null_mode_off_delivers();
    test_toggle();
    test_null_ring_state_unchanged();
    printf("All tests passed.\n");
    return 0;
}
