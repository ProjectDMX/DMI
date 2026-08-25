// tests/ring/test_producer.cu — M2 producer kernel correctness tests.
//
// Build:  make -C tests/ring test_producer
// Run:    ./tests/ring/build/test_producer
//
// All GPU tests run sequentially on a single stream (cudaStreamDefault) because
// this GPU serialises kernels from the same process even across streams.

#include "ring/ring_alloc.h"
#include "ring/producer.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------
static int g_total = 0;
static int g_fail  = 0;

#define ASSERT(cond) do {                                         \
    ++g_total;                                                    \
    if (!(cond)) {                                                \
        ++g_fail;                                                 \
        fprintf(stderr, "  FAIL  %s:%d  %s\n",                   \
                __FILE__, __LINE__, #cond);                       \
    }                                                             \
} while (0)

#define CUDA_CHECK(e) do {                                        \
    cudaError_t _e = (e);                                        \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(_e));      \
        std::exit(1);                                            \
    }                                                            \
} while (0)

static void banner(const char* name) {
    printf("[ TEST ] %s\n", name);
    fflush(stdout);
}

// Read task entries from device into a host-side vector.
static std::vector<ring::TaskEntry> read_entries(ring::TaskEntry* d_entries,
                                                  uint64_t n_slots)
{
    std::vector<ring::TaskEntry> v(n_slots);
    CUDA_CHECK(cudaMemcpy(v.data(), d_entries,
                          n_slots * sizeof(ring::TaskEntry),
                          cudaMemcpyDeviceToHost));
    return v;
}

// Read payload buffer (or subset) from device.
static std::vector<uint8_t> read_payload(uint8_t* d_buf, uint64_t nbytes)
{
    std::vector<uint8_t> v(nbytes);
    CUDA_CHECK(cudaMemcpy(v.data(), d_buf, nbytes, cudaMemcpyDeviceToHost));
    return v;
}

// Allocate a device buffer and fill it with a simple byte pattern.
static uint8_t* make_src(uint64_t nbytes)
{
    uint8_t* d;
    CUDA_CHECK(cudaMalloc(&d, nbytes));
    std::vector<uint8_t> h(nbytes);
    for (uint64_t i = 0; i < nbytes; ++i) h[i] = static_cast<uint8_t>(i & 0xFF);
    CUDA_CHECK(cudaMemcpy(d, h.data(), nbytes, cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
// Test 1 — host-only: RingConfig defaults
// ---------------------------------------------------------------------------
static void test_config_defaults()
{
    banner("config_defaults");
    ring::RingConfig cfg;
    ASSERT(cfg.task_ring_entries   == 1024);
    ASSERT(cfg.payload_ring_bytes  == 256ULL * 1024 * 1024);
    ASSERT(cfg.chunk_bytes         == 64ULL  * 1024 * 1024);
    ASSERT(cfg.wait_policy         == ring::WaitPolicy::INFINITE);
    ASSERT(cfg.drop_reporting      == ring::DropReporting::DROP_TASK);
}

// ---------------------------------------------------------------------------
// Test 2 — single-chunk: tensor fits in one chunk
// ---------------------------------------------------------------------------
static void test_single_chunk()
{
    banner("single_chunk");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 16;
    cfg.payload_ring_bytes = 16 * 1024;   // 16 KiB
    cfg.chunk_bytes        = 4 * 1024;    // 4 KiB — tensor fits in one chunk

    ring::AllocatedRing ar(cfg);
    ar.init();

    const uint64_t src_bytes = 256;
    uint8_t* d_src = make_src(src_bytes);

    launch_producer(ar.state(), d_src, src_bytes,
                    /*logical_task_id=*/42,
                    /*hook_type=*/1, /*hook_id=*/7);
    CUDA_CHECK(cudaDeviceSynchronize());

    // task_head should now be 1
    ASSERT(*ar.state().task_head == 1);

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    const auto& e = entries[0];

    ASSERT(e.ready_seq         == 0);
    ASSERT(e.seq_no            == 0);
    ASSERT(e.logical_task_id   == 42);
    ASSERT(e.chunk_idx         == 0);
    ASSERT(e.hook_type         == 1);
    ASSERT(e.hook_id           == 7);
    ASSERT(e.tensor_total_bytes == src_bytes);
    ASSERT(e.chunk_offset_bytes == 0);
    ASSERT((e.flags & ring::TASK_FLAG_IS_FIRST) != 0);
    ASSERT((e.flags & ring::TASK_FLAG_IS_LAST)  != 0);
    ASSERT((e.flags & ring::TASK_FLAG_IS_DROP)  == 0);
    ASSERT(e.payload_len1 == src_bytes);
    ASSERT(e.payload_len2 == 0);
    ASSERT(e.payload_off1 == 0);  // payload_head started at 0

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 3 — multi-chunk: tensor spans multiple chunks
// ---------------------------------------------------------------------------
static void test_multi_chunk()
{
    banner("multi_chunk");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 16;
    cfg.payload_ring_bytes = 64 * 1024;   // 64 KiB
    cfg.chunk_bytes        = 1024;         // 1 KiB per chunk

    ring::AllocatedRing ar(cfg);
    ar.init();

    // 3.5 × chunk_bytes → 4 chunks (3×1024 + 512)
    const uint64_t src_bytes = 3 * 1024 + 512;
    uint8_t* d_src = make_src(src_bytes);

    launch_producer(ar.state(), d_src, src_bytes, 99, 0, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT(*ar.state().task_head == 4);

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);

    // chunk 0 — IS_FIRST
    ASSERT((entries[0].flags & ring::TASK_FLAG_IS_FIRST) != 0);
    ASSERT((entries[0].flags & ring::TASK_FLAG_IS_LAST)  == 0);
    ASSERT(entries[0].chunk_idx          == 0);
    ASSERT(entries[0].chunk_offset_bytes == 0);
    ASSERT(entries[0].payload_len1       == 1024);

    // chunk 1 — middle
    ASSERT((entries[1].flags & ring::TASK_FLAG_IS_FIRST) == 0);
    ASSERT((entries[1].flags & ring::TASK_FLAG_IS_LAST)  == 0);
    ASSERT(entries[1].chunk_idx          == 1);
    ASSERT(entries[1].chunk_offset_bytes == 1024);

    // chunk 2 — middle
    ASSERT(entries[2].chunk_idx          == 2);
    ASSERT(entries[2].chunk_offset_bytes == 2 * 1024);

    // chunk 3 — IS_LAST, smaller
    ASSERT((entries[3].flags & ring::TASK_FLAG_IS_FIRST) == 0);
    ASSERT((entries[3].flags & ring::TASK_FLAG_IS_LAST)  != 0);
    ASSERT(entries[3].chunk_idx          == 3);
    ASSERT(entries[3].chunk_offset_bytes == 3 * 1024);
    ASSERT(entries[3].payload_len1       == 512);

    // All entries share same logical_task_id and tensor_total_bytes
    for (int i = 0; i < 4; ++i) {
        ASSERT(entries[i].logical_task_id    == 99);
        ASSERT(entries[i].tensor_total_bytes == src_bytes);
        ASSERT(entries[i].hook_id            == 5);
    }

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 4 — payload wrap: chunk crosses end of ring buffer → two-span entry
// ---------------------------------------------------------------------------
static void test_wrap()
{
    banner("wrap_two_span");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 8;
    cfg.payload_ring_bytes = 4096;
    cfg.chunk_bytes        = 512;

    ring::AllocatedRing ar(cfg);
    ar.init();

    // Simulate a ring state where physical head is near the end.
    // head and tail must be PAYLOAD_ALIGN-aligned (16 bytes) since
    // vectorized uint4 D2D copies require aligned offsets.
    // head=3904, tail=3584: free = 4096 - (3904-3584) = 3776 >= 512.
    // Physical head position: 3904 % 4096 = 3904.
    // bytes_to_end = 4096 - 3904 = 192 → two spans: len1=192, len2=320.
    const uint64_t head       = 3904;
    const uint64_t tail       = 3584;
    const uint64_t src_bytes  = 512;
    *ar.state().payload_head  = head;
    *ar.state().payload_tail  = tail;

    uint8_t* d_src = make_src(src_bytes);

    launch_producer(ar.state(), d_src, src_bytes, 1, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    const auto& e = entries[0];

    uint64_t bytes_to_end = 4096 - (head % 4096);  // 192
    ASSERT(e.payload_len1 == bytes_to_end);
    ASSERT(e.payload_len2 == src_bytes - bytes_to_end);
    ASSERT(e.payload_off1 == head % 4096);
    ASSERT(e.payload_off2 == 0);
    ASSERT(e.payload_len1 + e.payload_len2 == src_bytes);

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 5 — data correctness: verify payload_buf contains the right bytes
// ---------------------------------------------------------------------------
static void test_data_correctness()
{
    banner("data_correctness");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 8;
    cfg.payload_ring_bytes = 8 * 1024;
    cfg.chunk_bytes        = 4 * 1024;

    ring::AllocatedRing ar(cfg);
    ar.init();

    const uint64_t src_bytes = 512;
    uint8_t* d_src = make_src(src_bytes);  // pattern: byte[i] = i % 256

    launch_producer(ar.state(), d_src, src_bytes, 0, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    const auto& e = entries[0];
    ASSERT(e.payload_len1 == src_bytes && e.payload_len2 == 0);

    auto payload = read_payload(ar.state().payload_buf, cfg.payload_ring_bytes);
    bool ok = true;
    for (uint64_t i = 0; i < src_bytes; ++i) {
        if (payload[e.payload_off1 + i] != static_cast<uint8_t>(i & 0xFF)) {
            ok = false;
            fprintf(stderr, "  byte[%llu] expected %u got %u\n",
                    (unsigned long long)i,
                    (unsigned)static_cast<uint8_t>(i & 0xFF),
                    (unsigned)payload[e.payload_off1 + i]);
            break;
        }
    }
    ASSERT(ok);

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 6 — data correctness across wrap: two-span payload bytes are correct
// ---------------------------------------------------------------------------
static void test_data_correctness_wrap()
{
    banner("data_correctness_wrap");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 8;
    cfg.payload_ring_bytes = 4096;
    cfg.chunk_bytes        = 512;

    ring::AllocatedRing ar(cfg);
    ar.init();

    // Same setup as test_wrap: head=3904, tail=3584 → two-span (192 + 320).
    // head must be PAYLOAD_ALIGN-aligned (16 bytes) for vectorized uint4 copies.
    const uint64_t src_bytes  = 512;
    *ar.state().payload_head  = 3904;
    *ar.state().payload_tail  = 3584;

    uint8_t* d_src = make_src(src_bytes);

    launch_producer(ar.state(), d_src, src_bytes, 0, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    const auto& e = entries[0];

    auto payload = read_payload(ar.state().payload_buf, cfg.payload_ring_bytes);

    bool ok = true;
    // Span 1 bytes
    for (uint64_t i = 0; i < e.payload_len1 && ok; ++i) {
        if (payload[e.payload_off1 + i] != static_cast<uint8_t>(i & 0xFF)) {
            ok = false;
            fprintf(stderr, "  span1 byte[%llu] mismatch\n", (unsigned long long)i);
        }
    }
    // Span 2 bytes
    for (uint64_t i = 0; i < e.payload_len2 && ok; ++i) {
        uint64_t src_i = e.payload_len1 + i;
        if (payload[e.payload_off2 + i] != static_cast<uint8_t>(src_i & 0xFF)) {
            ok = false;
            fprintf(stderr, "  span2 byte[%llu] mismatch\n", (unsigned long long)i);
        }
    }
    ASSERT(ok);

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 7 — timeout-drop: frozen heartbeat + nearly-full payload → DROP marker
// ---------------------------------------------------------------------------
static void test_timeout_drop()
{
    banner("timeout_drop");

    ring::RingConfig cfg;
    cfg.task_ring_entries          = 16;
    cfg.payload_ring_bytes         = 1024;
    cfg.chunk_bytes                = 512;
    cfg.wait_policy                = ring::WaitPolicy::TIMEOUT_DROP;
    cfg.drop_reporting             = ring::DropReporting::DROP_TASK;
    cfg.no_progress_timeout_cycles = 200000;  // ~0.1 ms at 2 GHz

    ring::AllocatedRing ar(cfg);
    ar.init();

    // Simulate a nearly-full payload ring: only 16 bytes free.
    // The producer needs 512 bytes but only 16 are available.
    // consumer_heartbeat stays 0 → no progress → timeout.
    *ar.state().payload_head = cfg.payload_ring_bytes - 16;
    *ar.state().payload_tail = 0;
    // task ring is empty (head=tail=0), so 1 task slot is available for DROP.

    const uint64_t src_bytes = 512;
    uint8_t* d_src = make_src(src_bytes);

    launch_producer(ar.state(), d_src, src_bytes, 77, 2, 9);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Producer should have published exactly 1 DROP entry.
    ASSERT(*ar.state().task_head == 1);

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    const auto& e = entries[0];

    ASSERT((e.flags & ring::TASK_FLAG_IS_DROP)  != 0);
    ASSERT((e.flags & ring::TASK_FLAG_IS_LAST)  != 0);
    ASSERT((e.flags & ring::TASK_FLAG_IS_FIRST) != 0);  // chunk 0 was first
    ASSERT(e.reason          == ring::DROP_REASON_TIMEOUT_NO_PROGRESS);
    ASSERT(e.payload_len1    == 0);
    ASSERT(e.payload_len2    == 0);
    ASSERT(e.logical_task_id == 77);
    ASSERT(e.hook_type       == 2);
    ASSERT(e.hook_id         == 9);

    cudaFree(d_src);
}

// ---------------------------------------------------------------------------
// Test 8 — backpressure release: producer spins, consumer advances tail
// ---------------------------------------------------------------------------
// Simulated by running two sequential kernels on the same stream:
//   kernel A fills ring, then we manually advance tails, then kernel B runs.
static void test_backpressure_release()
{
    banner("backpressure_release");

    ring::RingConfig cfg;
    cfg.task_ring_entries  = 4;            // small ring
    cfg.payload_ring_bytes = 4 * 1024;     // 4 KiB
    cfg.chunk_bytes        = 1 * 1024;     // 1 KiB

    ring::AllocatedRing ar(cfg);
    ar.init();

    // Fill the ring: 4 chunks × 1KiB = 4KiB (exactly fills payload ring).
    const uint64_t src_bytes_a = 4 * 1024;
    uint8_t* d_src_a = make_src(src_bytes_a);
    launch_producer(ar.state(), d_src_a, src_bytes_a, 10, 0, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT(*ar.state().task_head == 4);

    // Now simulate consumer draining slot 0: advance task_tail and payload_tail.
    // This frees 1 task slot + 1 KiB of payload.
    *ar.state().task_tail    = 1;
    *ar.state().payload_tail = 1024;
    ++(*ar.state().consumer_heartbeat);

    // Producer should now be able to fit 1 more chunk.
    const uint64_t src_bytes_b = 512;
    uint8_t* d_src_b = make_src(src_bytes_b);
    launch_producer(ar.state(), d_src_b, src_bytes_b, 20, 0, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // task_head should now be 5 (4 from A + 1 from B).
    ASSERT(*ar.state().task_head == 5);

    auto entries = read_entries(ar.state().task_entries, cfg.task_ring_entries);
    // Slot 4 % 4 = 0 (reused after tail advanced).
    const auto& e = entries[4 % 4];
    ASSERT(e.logical_task_id   == 20);
    ASSERT(e.tensor_total_bytes == src_bytes_b);
    ASSERT(e.seq_no            == 4);

    cudaFree(d_src_a);
    cudaFree(d_src_b);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    setbuf(stdout, nullptr);

    printf("=== test_producer ===\n");

    test_config_defaults();
    test_single_chunk();
    test_multi_chunk();
    test_wrap();
    test_data_correctness();
    test_data_correctness_wrap();
    test_timeout_drop();
    test_backpressure_release();

    printf("\n%d / %d assertions passed\n", g_total - g_fail, g_total);
    if (g_fail > 0) {
        fprintf(stderr, "%d FAILURES\n", g_fail);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
