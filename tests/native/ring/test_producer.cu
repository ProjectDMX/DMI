// CUDA data-path tests for the current static, prefix, and chunked producers.

#include "ring/producer.cuh"
#include "ring/ring_alloc.h"

#include <cuda_runtime.h>

#include <algorithm>
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

static void banner(const char* name) {
    std::printf("[ TEST ] %s\n", name);
}

static ring::RingConfig make_config(uint64_t payload_bytes = 4096) {
    ring::RingConfig cfg;
    cfg.task_ring_entries = 16;
    cfg.payload_ring_bytes = payload_bytes;
    cfg.pinned_staging_bytes = payload_bytes;
    return cfg;
}

static std::vector<uint8_t> pattern(uint64_t size, uint8_t seed = 0) {
    std::vector<uint8_t> result(size);
    for (uint64_t i = 0; i < size; ++i) {
        result[i] = static_cast<uint8_t>(seed + i * 17);
    }
    return result;
}

static uint8_t* upload(const std::vector<uint8_t>& source) {
    if (source.empty()) {
        return nullptr;
    }
    uint8_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, source.size()));
    CUDA_CHECK(cudaMemcpy(device, source.data(), source.size(),
                          cudaMemcpyHostToDevice));
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

static std::vector<uint8_t> read_payload(const ring::RingState& state,
                                         const ring::TaskEntry& entry) {
    std::vector<uint8_t> result(entry.tensor_total_bytes);
    if (entry.payload_len1 > 0) {
        CUDA_CHECK(cudaMemcpy(result.data(),
                              state.payload_buf + entry.payload_off1,
                              entry.payload_len1, cudaMemcpyDeviceToHost));
    }
    if (entry.payload_len2 > 0) {
        CUDA_CHECK(cudaMemcpy(result.data() + entry.payload_len1,
                              state.payload_buf + entry.payload_off2,
                              entry.payload_len2, cudaMemcpyDeviceToHost));
    }
    return result;
}

static void expect_entry(const ring::RingState& state, uint64_t sequence,
                         uint64_t expected_bytes) {
    const ring::TaskEntry& entry = state.task_entries[sequence % state.task_cap];
    EXPECT(entry.ready_seq == sequence);
    EXPECT(entry.tensor_total_bytes == expected_bytes);
    EXPECT(entry.payload_len1 + entry.payload_len2 == expected_bytes);
}

static void test_static_copy() {
    banner("static producer copies exact bytes");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    const std::vector<uint8_t> source = pattern(333, 7);
    uint8_t* device = upload(source);
    ring::launch_producer_static(state, device, source.size(), 3);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_entry(state, 0, source.size());
    const ring::TaskEntry& entry = state.task_entries[0];
    EXPECT(entry.payload_off1 == 0);
    EXPECT(entry.payload_len2 == 0);
    EXPECT(*state.task_head == 1);
    EXPECT(*state.payload_head == ring::align_up(source.size(), ring::PAYLOAD_ALIGN));
    EXPECT(*state.actual_bytes_counter == source.size());
    EXPECT(read_payload(state, entry) == source);

    CUDA_CHECK(cudaFree(device));
}

static void test_static_wrap_copy() {
    banner("static producer copies across payload wrap");
    ring::AllocatedRing allocated(make_config(512));
    allocated.init();
    ring::RingState& state = allocated.state();
    *state.payload_head = 480;

    const std::vector<uint8_t> source = pattern(80, 19);
    uint8_t* device = upload(source);
    ring::launch_producer_static(state, device, source.size(), 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_entry(state, 0, source.size());
    const ring::TaskEntry& entry = state.task_entries[0];
    EXPECT(entry.payload_off1 == 480);
    EXPECT(entry.payload_len1 == 32);
    EXPECT(entry.payload_off2 == 0);
    EXPECT(entry.payload_len2 == 48);
    EXPECT(*state.payload_head == 560);
    EXPECT(read_payload(state, entry) == source);

    CUDA_CHECK(cudaFree(device));
}

static void test_prefix_copy() {
    banner("prefix producer reads device row count");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint64_t row_bytes = 32;
    const std::vector<uint8_t> source = pattern(8 * row_bytes, 31);
    uint8_t* device = upload(source);
    int64_t* row_count = upload_counts({3});

    ring::launch_producer_prefix(state, device, source.size(), row_count,
                                 row_bytes, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr uint64_t expected_bytes = 3 * row_bytes;
    expect_entry(state, 0, expected_bytes);
    std::vector<uint8_t> expected(source.begin(), source.begin() + expected_bytes);
    EXPECT(read_payload(state, state.task_entries[0]) == expected);
    EXPECT(*state.payload_head == expected_bytes);
    EXPECT(*state.actual_bytes_counter == expected_bytes);

    CUDA_CHECK(cudaFree(row_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_prefix_bounds() {
    banner("prefix producer clamps negative and oversized counts");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint64_t row_bytes = 32;
    const std::vector<uint8_t> source = pattern(8 * row_bytes, 43);
    uint8_t* device = upload(source);
    int64_t* row_count = upload_counts({-4});

    ring::launch_producer_prefix(state, device, source.size(), row_count,
                                 row_bytes, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_entry(state, 0, 0);
    EXPECT(*state.payload_head == 0);

    const int64_t oversized = 100;
    CUDA_CHECK(cudaMemcpy(row_count, &oversized, sizeof(oversized),
                          cudaMemcpyHostToDevice));
    ring::launch_producer_prefix(state, device, source.size(), row_count,
                                 row_bytes, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_entry(state, 1, source.size());
    EXPECT(read_payload(state, state.task_entries[1]) == source);
    EXPECT(*state.payload_head == source.size());
    EXPECT(*state.actual_bytes_counter == source.size());

    CUDA_CHECK(cudaFree(row_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_chunked_packed_copy() {
    banner("chunked producer packs device-selected prefixes");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint32_t chunks = 4;
    constexpr uint64_t chunk_input_bytes = 64;
    const std::vector<int64_t> counts = {16, 32, 0, 8};
    std::vector<uint8_t> source(chunks * chunk_input_bytes);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        for (uint64_t i = 0; i < chunk_input_bytes; ++i) {
            source[chunk * chunk_input_bytes + i] =
                static_cast<uint8_t>(chunk * 53 + i);
        }
    }
    std::vector<uint8_t> expected;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto begin = source.begin() + chunk * chunk_input_bytes;
        expected.insert(expected.end(), begin, begin + counts[chunk]);
    }

    uint8_t* device = upload(source);
    int64_t* device_counts = upload_counts(counts);
    ring::launch_producer_chunked(state, device, source.size(), device_counts,
                                  chunks, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_entry(state, 0, expected.size());
    EXPECT(read_payload(state, state.task_entries[0]) == expected);
    EXPECT(*state.actual_bytes_counter == expected.size());
    EXPECT(*state.payload_head == ring::align_up(source.size(), ring::PAYLOAD_ALIGN));

    CUDA_CHECK(cudaFree(device_counts));
    CUDA_CHECK(cudaFree(device));
}

static void test_serialized_static_launches() {
    banner("serialized launches preserve FIFO and payload offsets");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    const std::vector<uint64_t> sizes = {17, 64, 129};
    std::vector<std::vector<uint8_t>> sources;
    std::vector<uint8_t*> devices;
    for (uint64_t i = 0; i < sizes.size(); ++i) {
        sources.push_back(pattern(sizes[i], static_cast<uint8_t>(70 + i)));
        devices.push_back(upload(sources.back()));
        ring::launch_producer_static(state, devices.back(), sizes[i], 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t expected_head = 0;
    uint64_t expected_actual = 0;
    for (uint64_t sequence = 0; sequence < sizes.size(); ++sequence) {
        expect_entry(state, sequence, sizes[sequence]);
        const ring::TaskEntry& entry = state.task_entries[sequence];
        EXPECT(entry.payload_off1 == expected_head);
        EXPECT(read_payload(state, entry) == sources[sequence]);
        expected_head += ring::align_up(sizes[sequence], ring::PAYLOAD_ALIGN);
        expected_actual += sizes[sequence];
    }
    EXPECT(*state.task_head == sizes.size());
    EXPECT(*state.payload_head == expected_head);
    EXPECT(*state.actual_bytes_counter == expected_actual);

    for (uint8_t* device : devices) {
        CUDA_CHECK(cudaFree(device));
    }
}

int main() {
    setbuf(stdout, nullptr);
    ring::set_ring_null_mode(false);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("test_producer (current producer variants)\n");
    test_static_copy();
    test_static_wrap_copy();
    test_prefix_copy();
    test_prefix_bounds();
    test_chunked_packed_copy();
    test_serialized_static_launches();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
