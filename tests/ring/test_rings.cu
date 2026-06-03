// tests/ring/test_rings.cu — Standalone CUDA unit tests for ring primitives.
//
// Covers:
//   - Payload ring accounting + wrap correctness (two-span)
//   - Task ring free-slot accounting
//   - TaskEntry layout (current 64-byte struct)
//   - RingConfig defaults
//
// NOTE: the task-ring FIFO / ready_seq-lifecycle / wrap-reuse GPU tests and the
// DROP-marker tests were removed — they exercised the old 128-byte TaskEntry
// (per-entry seq_no/hook_id/flags/reason) and the chunking + drop-marker
// subsystems, all of which were removed (identity now flows via TensorMetaFifo,
// one tensor = one entry, no chunking). They need a rewrite against the current
// design, not a field rename. See docs/node_toggle_local_perf.md.
//
// Build with: make -C tests/ring
// Run with:   ./tests/ring/test_rings

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <cuda_runtime.h>

// Include the ring headers under test.
#include "../../monitoring/csrc/ring/task_entry.h"
#include "../../monitoring/csrc/ring/ring_config.h"
#include "../../monitoring/csrc/ring/payload_ring.cuh"
#include "../../monitoring/csrc/ring/task_ring.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t _e = (expr);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond)                                                      \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "  FAIL %s:%d  " #cond "\n",                 \
                    __FILE__, __LINE__);                                  \
            g_fail++;                                                     \
        } else {                                                          \
            g_pass++;                                                     \
        }                                                                 \
    } while (0)

static void banner(const char* name) {
    printf("[ TEST ] %s\n", name);
    fflush(stdout);
}

// ===========================================================================
// HOST-ONLY TESTS (pure arithmetic — __host__ __device__ functions)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: payload_free_bytes basic accounting
// ---------------------------------------------------------------------------
static void test_payload_free_bytes() {
    banner("payload_free_bytes — basic accounting");
    using namespace ring;

    uint64_t cap = 1024;

    // Empty ring.
    EXPECT(payload_free_bytes(0, 0, cap) == cap);

    // Half full.
    EXPECT(payload_free_bytes(512, 0, cap) == 512);

    // Producer wrapped around (head > cap).
    EXPECT(payload_free_bytes(1024, 0, cap) == 0);      // full
    EXPECT(payload_free_bytes(1100, 100, cap) == 24);   // 1024 - (1100-100)

    // Consumer advanced.
    EXPECT(payload_free_bytes(800, 300, cap) == 524);
}

// ---------------------------------------------------------------------------
// Test 2: payload_compute_spans — single span (no wrap)
// ---------------------------------------------------------------------------
static void test_payload_single_span() {
    banner("payload_compute_spans — single span (no wrap)");
    using namespace ring;

    uint64_t cap  = 256;
    uint64_t head = 0;

    // From the beginning.
    TwoSpan s = payload_compute_spans(head, cap, 100);
    EXPECT(s.off1 == 0);
    EXPECT(s.len1 == 100);
    EXPECT(s.len2 == 0);

    // From the middle, still fits before end.
    head = 100;
    s = payload_compute_spans(head, cap, 50);
    EXPECT(s.off1 == 100);
    EXPECT(s.len1 == 50);
    EXPECT(s.len2 == 0);

    // Exactly fills to the end.
    head = 200;
    s = payload_compute_spans(head, cap, 56);
    EXPECT(s.off1 == 200);
    EXPECT(s.len1 == 56);
    EXPECT(s.len2 == 0);
}

// ---------------------------------------------------------------------------
// Test 3: payload_compute_spans — two-span (wrap-around)
// ---------------------------------------------------------------------------
static void test_payload_two_span_wrap() {
    banner("payload_compute_spans — two-span (wrap-around)");
    using namespace ring;

    uint64_t cap = 64;

    // Head at offset 50 in the ring; request 20 bytes that must wrap.
    // span1: [50, 14) → bytes 50..63
    // span2: [0,  6)  → bytes 0..5
    uint64_t head = 50;  // physical off = 50 % 64 = 50
    TwoSpan s = payload_compute_spans(head, cap, 20);
    EXPECT(s.off1 == 50);
    EXPECT(s.len1 == 14);
    EXPECT(s.off2 == 0);
    EXPECT(s.len2 == 6);
    EXPECT(s.len1 + s.len2 == 20);

    // Head has wrapped (head = 130 → phys = 2); request 5 bytes, no wrap.
    head = 130;  // phys = 130 % 64 = 2
    s = payload_compute_spans(head, cap, 5);
    EXPECT(s.off1 == 2);
    EXPECT(s.len1 == 5);
    EXPECT(s.len2 == 0);

    // Head at physical position 60; request 10 bytes → wraps at boundary.
    head = 60;  // phys = 60
    s = payload_compute_spans(head, cap, 10);
    EXPECT(s.off1 == 60);
    EXPECT(s.len1 == 4);
    EXPECT(s.off2 == 0);
    EXPECT(s.len2 == 6);

    // Exactly at boundary (off == 0 after wrap), request full capacity.
    head = 64;  // phys = 0
    s = payload_compute_spans(head, cap, 64);
    EXPECT(s.off1 == 0);
    EXPECT(s.len1 == 64);
    EXPECT(s.len2 == 0);
}

// ---------------------------------------------------------------------------
// Test 4: payload ring simulation — multiple reservations with tail advance
// ---------------------------------------------------------------------------
static void test_payload_ring_simulation() {
    banner("payload_ring_simulation — multi-reservation + tail advance");
    using namespace ring;

    uint64_t cap   = 64;
    uint64_t head  = 0;
    uint64_t tail  = 0;

    // Reserve 50 bytes.
    EXPECT(payload_free_bytes(head, tail, cap) == 64);
    TwoSpan s1 = payload_compute_spans(head, cap, 50);
    EXPECT(s1.off1 == 0 && s1.len1 == 50 && s1.len2 == 0);
    payload_advance_head(head, 50);
    EXPECT(head == 50);

    // Reserve 10 more bytes (still fits).
    EXPECT(payload_free_bytes(head, tail, cap) == 14);
    TwoSpan s2 = payload_compute_spans(head, cap, 10);
    EXPECT(s2.off1 == 50 && s2.len1 == 10 && s2.len2 == 0);
    payload_advance_head(head, 10);
    EXPECT(head == 60);

    // Ring is nearly full (4 bytes free); need 30 — wait until consumer frees.
    EXPECT(payload_free_bytes(head, tail, cap) == 4);

    // Consumer releases first reservation (50 bytes).
    payload_release(tail, 50);
    EXPECT(tail == 50);
    EXPECT(payload_free_bytes(head, tail, cap) == 54);

    // Now reserve 30 bytes — wraps: off=60, len1=4, off2=0, len2=26.
    TwoSpan s3 = payload_compute_spans(head, cap, 30);
    EXPECT(s3.off1 == 60);
    EXPECT(s3.len1 == 4);
    EXPECT(s3.off2 == 0);
    EXPECT(s3.len2 == 26);
    payload_advance_head(head, 30);
    EXPECT(head == 90);  // 60 + 30 (unwrapped)

    // Consumer releases the second reservation (10 bytes).
    payload_release(tail, 10);
    EXPECT(tail == 60);
    EXPECT(payload_free_bytes(head, tail, cap) == 34);  // cap - (90 - 60)
}

// ---------------------------------------------------------------------------
// Test 5: task_free_slots basic accounting
// ---------------------------------------------------------------------------
static void test_task_free_slots() {
    banner("task_free_slots — basic accounting");
    using namespace ring;

    uint64_t cap = 16;
    EXPECT(task_free_slots(0, 0, cap) == 16);
    EXPECT(task_free_slots(8, 0, cap) == 8);
    EXPECT(task_free_slots(16, 0, cap) == 0);   // full
    EXPECT(task_free_slots(16, 16, cap) == 16);  // empty again after consumer drains
    EXPECT(task_free_slots(1024, 1020, cap) == 12);
}

// ---------------------------------------------------------------------------
// Test 6: TaskEntry layout (size, alignment, field offsets)
// ---------------------------------------------------------------------------
static void test_task_entry_layout() {
    banner("TaskEntry — size and field offsets");
    using namespace ring;

    EXPECT(sizeof(TaskEntry) == 64);
    EXPECT(alignof(TaskEntry) == 64);
    EXPECT(offsetof(TaskEntry, ready_seq)          == 0);
    EXPECT(offsetof(TaskEntry, tensor_total_bytes) == 8);
    EXPECT(offsetof(TaskEntry, payload_off1)       == 16);
    EXPECT(offsetof(TaskEntry, payload_len1)       == 24);
    EXPECT(offsetof(TaskEntry, payload_off2)       == 32);
    EXPECT(offsetof(TaskEntry, payload_len2)       == 40);
}

// ---------------------------------------------------------------------------
// Test 8: RingConfig defaults are sane
// ---------------------------------------------------------------------------
static void test_ring_config_defaults() {
    banner("RingConfig — default values");
    using namespace ring;

    RingConfig cfg;
    EXPECT(cfg.task_ring_entries    == 1024);
    EXPECT(cfg.payload_ring_bytes   == 256ULL * 1024 * 1024);
    EXPECT(cfg.drain_poll_timeout_us > 0);
    EXPECT(cfg.pinned_staging_bytes == 0);  // 0 => defaults to payload_ring_bytes
    EXPECT(cfg.effective_staging_bytes() == cfg.payload_ring_bytes);
}

// ===========================================================================
// GPU TESTS — task ring publish/consume protocol (task_ring.cuh)
// ===========================================================================
//
// One kernel publishes a contiguous range of sequence numbers with task_publish()
// (the exact device op the producer uses); the CPU side uses task_cpu_ready() /
// task_release_cpu() (the exact ops the drain uses). Entries live in managed
// memory so both sides touch the same slots. Each seq s encodes recognizable data
// (tensor_total_bytes = 1000+s, payload_off1 = 16*s) so round-trip and ordering
// can be checked. Single-threaded publish keeps the FIFO order deterministic.

__global__ void kernel_publish_range(ring::TaskEntry* entries, uint64_t capacity,
                                     uint64_t start, uint64_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < n; ++i) {
            uint64_t s = start + i;
            ring::TaskEntry src{};
            src.tensor_total_bytes = 1000 + s;
            src.payload_off1       = s * 16;
            src.payload_len1       = 1000 + s;
            ring::task_publish(entries, capacity, s, src);
        }
    }
}

// Test 9: FIFO — publish N (<= capacity) in order, consume in order.
static void test_task_ring_fifo_gpu() {
    banner("task ring FIFO — publish N, consume in order");
    using namespace ring;
    const uint64_t cap = 64, n = 50;
    TaskEntry* e = nullptr;
    CUDA_CHECK(cudaMallocManaged(&e, cap * sizeof(TaskEntry)));
    task_ring_init(e, cap);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_publish_range<<<1, 1>>>(e, cap, 0, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool order_ok = true, data_ok = true;
    for (uint64_t tail = 0; tail < n; ++tail) {
        if (!task_cpu_ready(e, cap, tail)) { order_ok = false; break; }
        const TaskEntry& s = e[tail % cap];
        if (s.tensor_total_bytes != 1000 + tail ||
            s.payload_off1 != tail * 16 ||
            s.payload_len1 != 1000 + tail) data_ok = false;
        task_release_cpu(e, cap, tail);
    }
    EXPECT(order_ok);                     // each slot published in seq order
    EXPECT(data_ok);                      // payload round-tripped intact
    EXPECT(!task_cpu_ready(e, cap, n));   // seq n was never published
    CUDA_CHECK(cudaFree(e));
}

// Test 10: ready_seq guard — an unpublished slot is not consumable; publish then
// release flips it ready then back to unpublished.
static void test_ready_seq_lifecycle_gpu() {
    banner("ready_seq guard — unpublished slot not consumable");
    using namespace ring;
    const uint64_t cap = 8, seq = 0, idx = seq % cap;  // idx: keep '%' out of EXPECT()
    TaskEntry* e = nullptr;
    CUDA_CHECK(cudaMallocManaged(&e, cap * sizeof(TaskEntry)));
    task_ring_init(e, cap);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT(e[idx].ready_seq == READY_SEQ_SENTINEL);
    EXPECT(!task_cpu_ready(e, cap, seq));   // before publish: not ready

    kernel_publish_range<<<1, 1>>>(e, cap, seq, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT(task_cpu_ready(e, cap, seq));    // after publish: ready
    EXPECT(e[idx].tensor_total_bytes == 1000 + seq);

    task_release_cpu(e, cap, seq);
    EXPECT(!task_cpu_ready(e, cap, seq));   // after release: not ready again
    CUDA_CHECK(cudaFree(e));
}

// Test 11: wrap-around — publish/consume 3x capacity one at a time; slot
// (seq % capacity) is reused correctly across wraps.
static void test_task_ring_wrap_reuse_gpu() {
    banner("task ring wrap — slot reuse across capacity");
    using namespace ring;
    const uint64_t cap = 4, n = 3 * cap;   // 3 full wraps
    TaskEntry* e = nullptr;
    CUDA_CHECK(cudaMallocManaged(&e, cap * sizeof(TaskEntry)));
    task_ring_init(e, cap);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    for (uint64_t s = 0; s < n; ++s) {
        kernel_publish_range<<<1, 1>>>(e, cap, s, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!task_cpu_ready(e, cap, s)) { ok = false; break; }   // reused slot republished
        if (e[s % cap].tensor_total_bytes != 1000 + s) { ok = false; break; }
        task_release_cpu(e, cap, s);
        if (task_cpu_ready(e, cap, s)) { ok = false; break; }    // freed for next wrap
    }
    EXPECT(ok);
    CUDA_CHECK(cudaFree(e));
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    // Disable stdout buffering so output is visible even if the process is
    // killed mid-run (helps debug hanging GPU tests).
    setbuf(stdout, nullptr);

    printf("=== Ring Unit Tests ===\n\n");

    // Print device info.
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (CC %d.%d)\n\n", prop.name,
           prop.major, prop.minor);

    // Host tests (pure arithmetic).
    test_payload_free_bytes();
    test_payload_single_span();
    test_payload_two_span_wrap();
    test_payload_ring_simulation();
    test_task_free_slots();
    test_task_entry_layout();
    test_ring_config_defaults();

    // GPU tests — task ring publish/consume protocol (single stream).
    test_task_ring_fifo_gpu();
    test_ready_seq_lifecycle_gpu();
    test_task_ring_wrap_reuse_gpu();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
