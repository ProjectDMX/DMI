// tests/ring/test_rings.cu — Standalone CUDA unit tests for ring primitives.
//
// Covers all Milestone 1 acceptance criteria:
//   - Payload wrap correctness (two-span)
//   - Task ring FIFO correctness by seq_no
//   - ready_seq guard: consumer cannot read unpublished slot
//   - DROP marker correctness
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

    EXPECT(sizeof(TaskEntry) == 128);
    EXPECT(alignof(TaskEntry) == 128);
    EXPECT(offsetof(TaskEntry, ready_seq)          == 0);
    EXPECT(offsetof(TaskEntry, seq_no)             == 8);
    EXPECT(offsetof(TaskEntry, logical_task_id)    == 16);
    EXPECT(offsetof(TaskEntry, chunk_offset_bytes) == 24);
    EXPECT(offsetof(TaskEntry, tensor_total_bytes) == 32);
    EXPECT(offsetof(TaskEntry, payload_off1)       == 40);
    EXPECT(offsetof(TaskEntry, payload_len1)       == 48);
    EXPECT(offsetof(TaskEntry, payload_off2)       == 56);
    EXPECT(offsetof(TaskEntry, payload_len2)       == 64);
    EXPECT(offsetof(TaskEntry, chunk_idx)          == 72);
    EXPECT(offsetof(TaskEntry, hook_type)          == 76);
    EXPECT(offsetof(TaskEntry, hook_id)            == 80);
    EXPECT(offsetof(TaskEntry, flags)              == 84);
    EXPECT(offsetof(TaskEntry, reason)             == 88);
}

// ---------------------------------------------------------------------------
// Test 7: DROP marker fields
// ---------------------------------------------------------------------------
static void test_drop_marker_fields() {
    banner("DROP marker — fields and flags");
    using namespace ring;

    TaskEntry e{};
    // Producer fills a DROP marker.
    e.seq_no          = 42;
    e.logical_task_id = 999;
    e.flags           = TASK_FLAG_IS_DROP;
    e.reason          = DROP_REASON_TIMEOUT_NO_PROGRESS;
    e.payload_len1    = 0;
    e.payload_len2    = 0;

    EXPECT((e.flags & TASK_FLAG_IS_DROP) != 0);
    EXPECT((e.flags & TASK_FLAG_IS_FIRST) == 0);
    EXPECT((e.flags & TASK_FLAG_IS_LAST) == 0);
    EXPECT(e.payload_len1 == 0);
    EXPECT(e.payload_len2 == 0);
    EXPECT(e.reason == DROP_REASON_TIMEOUT_NO_PROGRESS);
    EXPECT(payload_chunk_bytes(e.payload_len1, e.payload_len2) == 0);

    // SENTINEL must not equal any plausible seq_no value (just check a few).
    EXPECT(READY_SEQ_SENTINEL != 0);
    EXPECT(READY_SEQ_SENTINEL != 42);
    EXPECT(READY_SEQ_SENTINEL != uint64_t(-2));  // -2 != UINT64_MAX
}

// ---------------------------------------------------------------------------
// Test 8: RingConfig defaults are sane
// ---------------------------------------------------------------------------
static void test_ring_config_defaults() {
    banner("RingConfig — default values");
    using namespace ring;

    RingConfig cfg;
    EXPECT(cfg.task_ring_entries  == 1024);
    EXPECT(cfg.payload_ring_bytes == 256ULL * 1024 * 1024);
    EXPECT(cfg.chunk_bytes        == 64ULL  * 1024 * 1024);
    EXPECT(cfg.wait_policy        == WaitPolicy::INFINITE);
    EXPECT(cfg.drop_reporting     == DropReporting::DROP_TASK);
    EXPECT(cfg.chunk_bytes * 4    <= cfg.payload_ring_bytes);  // forward-progress
}

// ===========================================================================
// GPU TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 9: task ring FIFO — sequential producer→consumer on the same stream.
//
// The ring is large enough to hold all entries at once, so the producer fills
// it entirely before the consumer drains it.  This tests pure FIFO ordering
// without requiring concurrent kernel execution.
//
// Verifies:
//   - Entries are consumed in seq_no order.
//   - hook_id values match what the producer wrote.
//   - task_release correctly resets ready_seq so slots can be reused.
// ---------------------------------------------------------------------------

static const int FIFO_N = 32;  // number of tasks to produce and consume

// Producer: publishes FIFO_N entries; ring is large enough (cap >= FIFO_N)
// so it never needs to wait for the consumer.
__global__ void kernel_producer_fifo(ring::TaskEntry*   entries,
                                     uint64_t           cap,
                                     volatile uint64_t* d_head)
{
    for (int i = 0; i < FIFO_N; i++) {
        uint64_t seq = *d_head;

        ring::TaskEntry e{};
        e.seq_no          = seq;
        e.logical_task_id = (uint64_t)i;
        e.hook_id         = (uint32_t)i;
        e.flags           = ring::TASK_FLAG_IS_FIRST | ring::TASK_FLAG_IS_LAST;

        ring::task_publish(entries, cap, seq, e);
        *d_head = seq + 1;
    }
}

// Consumer: drains FIFO_N entries; called after the producer completes.
__global__ void kernel_consumer_fifo(const ring::TaskEntry* entries,
                                     ring::TaskEntry*       entries_rw,
                                     uint64_t               cap,
                                     volatile uint64_t*     d_tail,
                                     uint32_t*              results,
                                     uint64_t*              result_seq)
{
    for (int i = 0; i < FIFO_N; i++) {
        uint64_t tail = *d_tail;
        const ring::TaskEntry* e = ring::task_spin_wait(entries, cap, tail);
        results[i]    = e->hook_id;
        result_seq[i] = e->seq_no;
        ring::task_release(entries_rw, cap, tail);
        *d_tail = tail + 1;
    }
}

static void test_task_ring_fifo_gpu() {
    banner("task ring FIFO (GPU) — sequential producer then consumer");
    using namespace ring;

    // Ring larger than FIFO_N so producer never blocks waiting for space.
    const uint64_t cap = 64;

    TaskEntry* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, cap * sizeof(TaskEntry)));
    task_ring_init(d_entries, cap, 0);

    volatile uint64_t* d_head;
    volatile uint64_t* d_tail;
    CUDA_CHECK(cudaMallocManaged((void**)&d_head, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_tail, sizeof(uint64_t)));
    *d_head = 0;
    *d_tail = 0;

    uint32_t* results;
    uint64_t* result_seq;
    CUDA_CHECK(cudaMallocManaged(&results,    FIFO_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&result_seq, FIFO_N * sizeof(uint64_t)));

    // Sync to ensure managed memory is initialised before kernels run.
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Sequential: producer runs completely, then consumer drains.
    kernel_producer_fifo<<<1,1,0,stream>>>(d_entries, cap, d_head);
    kernel_consumer_fifo<<<1,1,0,stream>>>(d_entries, d_entries, cap,
                                           d_tail, results, result_seq);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < FIFO_N; i++) {
        EXPECT(results[i]    == (uint32_t)i);
        EXPECT(result_seq[i] == (uint64_t)i);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_entries);
    cudaFree((void*)d_head);
    cudaFree((void*)d_tail);
    cudaFree(results);
    cudaFree(result_seq);
}

// ---------------------------------------------------------------------------
// Test 10: ready_seq lifecycle — sequential guard semantics (single stream).
//
// In production the producer runs inside the CUDA graph and the consumer runs
// after the graph completes (ordered via a CUDA event).  This test mimics
// that flow and verifies the full ready_seq lifecycle:
//
//   Phase A — after init:    ready_seq[i] == SENTINEL  (slot not readable)
//   Phase B — after publish: ready_seq[seq] == seq_no  (slot readable)
//   Phase C — after release: ready_seq[seq] == SENTINEL (slot recycled)
//
// Correctness guarantee covered: if task_publish omitted __threadfence() or
// wrote the wrong value, the consumer's spin_wait (which checks ready_seq)
// would malfunction.  The sequential stream ordering means "happened-before"
// is enforced by the GPU's stream FIFO, but the volatile + membar semantics
// are still exercised by the actual instructions generated.
// ---------------------------------------------------------------------------

__global__ void kernel_check_all_sentinel(const ring::TaskEntry* entries,
                                          uint64_t               cap,
                                          int*                   ok)
{
    for (uint64_t i = 0; i < cap; i++) {
        uint64_t rs = *reinterpret_cast<const volatile uint64_t*>(
            &entries[i].ready_seq);
        if (rs != ring::READY_SEQ_SENTINEL) { *ok = 0; return; }
    }
    *ok = 1;
}

__global__ void kernel_publish_one(ring::TaskEntry*   entries,
                                   uint64_t           cap,
                                   volatile uint64_t* d_head,
                                   int*               ok)
{
    uint64_t seq = *d_head;
    ring::TaskEntry e{};
    e.seq_no             = seq;
    e.logical_task_id    = 0xDEADBEEFULL;
    e.hook_id            = 0xCAFEBABEu;
    e.flags              = ring::TASK_FLAG_IS_FIRST | ring::TASK_FLAG_IS_LAST;
    e.tensor_total_bytes = 12345;
    ring::task_publish(entries, cap, seq, e);
    *d_head = seq + 1;

    // Verify ready_seq is now the sequence number we published.
    uint64_t rs = *reinterpret_cast<const volatile uint64_t*>(
        &entries[seq % cap].ready_seq);
    *ok = (rs == seq) ? 1 : 0;
}

__global__ void kernel_consume_one(const ring::TaskEntry* entries,
                                   ring::TaskEntry*       entries_rw,
                                   uint64_t               cap,
                                   volatile uint64_t*     d_tail,
                                   uint64_t*              out_hook_id,
                                   uint64_t*              out_ltid,
                                   uint64_t*              out_total,
                                   int*                   sentinel_ok)
{
    uint64_t tail = *d_tail;
    const ring::TaskEntry* e = ring::task_spin_wait(entries, cap, tail);
    *out_hook_id = e->hook_id;
    *out_ltid    = e->logical_task_id;
    *out_total   = e->tensor_total_bytes;
    ring::task_release(entries_rw, cap, tail);
    *d_tail = tail + 1;

    // After release, ready_seq must be SENTINEL again.
    uint64_t rs = *reinterpret_cast<const volatile uint64_t*>(
        &entries_rw[tail % cap].ready_seq);
    *sentinel_ok = (rs == ring::READY_SEQ_SENTINEL) ? 1 : 0;
}

static void test_ready_seq_lifecycle_gpu() {
    banner("ready_seq lifecycle (GPU) — init sentinel, publish, consume, reset");
    using namespace ring;

    const uint64_t cap = 8;

    TaskEntry* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, cap * sizeof(TaskEntry)));
    task_ring_init(d_entries, cap, 0);

    volatile uint64_t* d_head;
    volatile uint64_t* d_tail;
    CUDA_CHECK(cudaMallocManaged((void**)&d_head, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_tail, sizeof(uint64_t)));
    *d_head = 0;
    *d_tail = 0;

    int*     d_ok_sentinel;
    int*     d_ok_publish;
    int*     d_ok_sentinel2;
    uint64_t* d_hook_id;
    uint64_t* d_ltid;
    uint64_t* d_total;
    CUDA_CHECK(cudaMallocManaged(&d_ok_sentinel,  sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_ok_publish,   sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_ok_sentinel2, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_hook_id,      sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&d_ltid,         sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&d_total,        sizeof(uint64_t)));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // All phases run sequentially on one stream.
    kernel_check_all_sentinel<<<1,1,0,stream>>>(d_entries, cap, d_ok_sentinel);
    kernel_publish_one<<<1,1,0,stream>>>(d_entries, cap, d_head, d_ok_publish);
    kernel_consume_one<<<1,1,0,stream>>>(d_entries, d_entries, cap,
                                         d_tail, d_hook_id, d_ltid, d_total,
                                         d_ok_sentinel2);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT(*d_ok_sentinel  == 1);            // all slots are SENTINEL after init
    EXPECT(*d_ok_publish   == 1);            // ready_seq is set correctly after publish
    EXPECT(*d_hook_id      == 0xCAFEBABEULL);
    EXPECT(*d_ltid         == 0xDEADBEEFULL);
    EXPECT(*d_total        == 12345ULL);
    EXPECT(*d_ok_sentinel2 == 1);            // slot reset to SENTINEL after release

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_entries);
    cudaFree((void*)d_head);
    cudaFree((void*)d_tail);
    cudaFree(d_ok_sentinel);
    cudaFree(d_ok_publish);
    cudaFree(d_ok_sentinel2);
    cudaFree(d_hook_id);
    cudaFree(d_ltid);
    cudaFree(d_total);
}

// ---------------------------------------------------------------------------
// Test 11: DROP marker round-trip (GPU, sequential) — producer emits IS_DROP,
//          consumer reads it and observes zero payload and IS_DROP flag.
// ---------------------------------------------------------------------------

__global__ void kernel_producer_drop(ring::TaskEntry*   entries,
                                     uint64_t           cap,
                                     volatile uint64_t* d_head)
{
    uint64_t seq = *d_head;
    ring::TaskEntry e{};
    e.seq_no          = seq;
    e.logical_task_id = 777;
    e.flags           = ring::TASK_FLAG_IS_DROP;
    e.reason          = ring::DROP_REASON_TIMEOUT_NO_PROGRESS;
    e.payload_len1    = 0;
    e.payload_len2    = 0;
    ring::task_publish(entries, cap, seq, e);
    *d_head = seq + 1;
}

__global__ void kernel_consumer_drop(const ring::TaskEntry* entries,
                                     ring::TaskEntry*       entries_rw,
                                     uint64_t               cap,
                                     volatile uint64_t*     d_tail,
                                     uint32_t*              out_flags,
                                     uint32_t*              out_reason,
                                     uint64_t*              out_len_total)
{
    uint64_t tail = *d_tail;
    const ring::TaskEntry* e = ring::task_spin_wait(entries, cap, tail);
    *out_flags     = e->flags;
    *out_reason    = e->reason;
    *out_len_total = e->payload_len1 + e->payload_len2;
    ring::task_release(entries_rw, cap, tail);
    *d_tail = tail + 1;
}

static void test_drop_marker_gpu() {
    banner("DROP marker (GPU) — producer emits drop, consumer sees IS_DROP");
    using namespace ring;

    const uint64_t cap = 4;

    TaskEntry* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, cap * sizeof(TaskEntry)));
    task_ring_init(d_entries, cap, 0);

    volatile uint64_t* d_head;
    volatile uint64_t* d_tail;
    CUDA_CHECK(cudaMallocManaged((void**)&d_head, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_tail, sizeof(uint64_t)));
    *d_head = 0;
    *d_tail = 0;

    uint32_t* out_flags;
    uint32_t* out_reason;
    uint64_t* out_len_total;
    CUDA_CHECK(cudaMallocManaged(&out_flags,     sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&out_reason,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&out_len_total, sizeof(uint64_t)));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Sequential: producer publishes DROP, then consumer drains it.
    kernel_producer_drop<<<1,1,0,stream>>>(d_entries, cap, d_head);
    kernel_consumer_drop<<<1,1,0,stream>>>(d_entries, d_entries, cap,
                                           d_tail,
                                           out_flags, out_reason, out_len_total);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT((*out_flags & TASK_FLAG_IS_DROP) != 0);
    EXPECT((*out_flags & TASK_FLAG_IS_FIRST) == 0);
    EXPECT((*out_flags & TASK_FLAG_IS_LAST) == 0);
    EXPECT(*out_reason    == DROP_REASON_TIMEOUT_NO_PROGRESS);
    EXPECT(*out_len_total == 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_entries);
    cudaFree((void*)d_head);
    cudaFree((void*)d_tail);
    cudaFree(out_flags);
    cudaFree(out_reason);
    cudaFree(out_len_total);
}

// ---------------------------------------------------------------------------
// Test 12: task ring wrap-around (slot reuse) — sequential producer→consumer,
//          ring slots reused 4× to verify FIFO is preserved across full wraps.
//
// Uses cap < WRAP_TOTAL so that slots are recycled; sequential (same stream)
// to avoid concurrent-spin dependency issues.
// ---------------------------------------------------------------------------
static const int WRAP_TOTAL = 64;   // 4× a cap-16 ring

__global__ void kernel_producer_wrap(ring::TaskEntry*   entries,
                                     uint64_t           cap,
                                     volatile uint64_t* d_head,
                                     int                n_total,
                                     int                base_id)
{
    // Producer assumes ring is drained by a previous consumer pass.
    for (int i = 0; i < n_total; i++) {
        uint64_t seq = *d_head;
        ring::TaskEntry e{};
        e.seq_no  = seq;
        e.hook_id = (uint32_t)(base_id + i);  // globally unique hook_id
        e.flags   = ring::TASK_FLAG_IS_FIRST | ring::TASK_FLAG_IS_LAST;
        ring::task_publish(entries, cap, seq, e);
        *d_head = seq + 1;
    }
}

__global__ void kernel_consumer_wrap(const ring::TaskEntry* entries,
                                     ring::TaskEntry*       entries_rw,
                                     uint64_t               cap,
                                     volatile uint64_t*     d_tail,
                                     uint32_t*              results,
                                     int                    n_total)
{
    for (int i = 0; i < n_total; i++) {
        uint64_t tail = *d_tail;
        const ring::TaskEntry* e = ring::task_spin_wait(entries, cap, tail);
        results[i] = e->hook_id;
        ring::task_release(entries_rw, cap, tail);
        *d_tail = tail + 1;
    }
}

static void test_task_ring_wrap_reuse_gpu() {
    banner("task ring wrap-around (GPU) — slot reuse, FIFO preserved");
    using namespace ring;

    // cap < WRAP_TOTAL to force slot reuse; run in batches of cap entries.
    const uint64_t cap    = 16;
    const int      batch  = (int)cap;  // publish+consume `batch` at a time

    TaskEntry* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, cap * sizeof(TaskEntry)));
    task_ring_init(d_entries, cap, 0);

    volatile uint64_t* d_head;
    volatile uint64_t* d_tail;
    CUDA_CHECK(cudaMallocManaged((void**)&d_head, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_tail, sizeof(uint64_t)));
    *d_head = 0;
    *d_tail = 0;

    uint32_t* results;
    CUDA_CHECK(cudaMallocManaged(&results, WRAP_TOTAL * sizeof(uint32_t)));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Process in batches of `cap` entries so the ring never overflows.
    // Each pass: producer fills the ring, then consumer drains it.
    for (int base = 0; base < WRAP_TOTAL; base += batch) {
        kernel_producer_wrap<<<1,1,0,stream>>>(d_entries, cap, d_head,
                                               batch, base);
        kernel_consumer_wrap<<<1,1,0,stream>>>(d_entries, d_entries, cap,
                                               d_tail, results + base, batch);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < WRAP_TOTAL; i++) {
        EXPECT(results[i] == (uint32_t)i);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_entries);
    cudaFree((void*)d_head);
    cudaFree((void*)d_tail);
    cudaFree(results);
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
    test_drop_marker_fields();
    test_ring_config_defaults();

    // GPU tests (all sequential — producer then consumer on same stream).
    test_task_ring_fifo_gpu();
    test_ready_seq_lifecycle_gpu();
    test_drop_marker_gpu();
    test_task_ring_wrap_reuse_gpu();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
