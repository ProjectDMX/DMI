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

static int32_t* upload_gate(int32_t value) {
    int32_t* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, sizeof(value)));
    CUDA_CHECK(cudaMemcpy(device, &value, sizeof(value), cudaMemcpyHostToDevice));
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

static void test_record_chunked_compacts_payload_head() {
    banner("record chunked producer advances by actual aligned bytes");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint32_t chunks = 4;
    constexpr uint64_t input_chunk = 64;
    const std::vector<int64_t> selected = {16, 32, 0, 8};
    const std::vector<uint8_t> source = pattern(chunks * input_chunk, 91);
    uint8_t* device = upload(source);
    int64_t* device_selected = upload_counts(selected);

    ring::launch_record_producer_chunked(
        state, device, source.size(), device_selected, chunks,
        nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr uint64_t actual = 56;
    expect_entry(state, 0, actual);
    EXPECT(*state.payload_head == ring::align_up(actual, ring::PAYLOAD_ALIGN));

    std::vector<uint8_t> expected;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto begin = source.begin() + chunk * input_chunk;
        expected.insert(expected.end(), begin, begin + selected[chunk]);
    }
    EXPECT(read_payload(state, state.task_entries[0]) == expected);

    CUDA_CHECK(cudaFree(device_selected));
    CUDA_CHECK(cudaFree(device));
}

static void test_record_sequence_and_segmented_pack() {
    banner("record row packers preserve declared row order");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    // Source is [S=3, B=2, feature=4 bytes], one distinct byte pattern per row.
    constexpr uint64_t feature_bytes = 4;
    std::vector<uint8_t> source(3 * 2 * feature_bytes);
    for (uint64_t row = 0; row < 6; ++row) {
        std::fill(source.begin() + row * feature_bytes,
                  source.begin() + (row + 1) * feature_bytes,
                  static_cast<uint8_t>(10 + row));
    }
    uint8_t* device = upload(source);
    int64_t* valid_count = upload_counts({2, 1});
    int64_t* valid_prefix = upload_counts({0, 2, 3});
    ring::launch_record_producer_seq_prefix_pack(
        state, device, source.size(), valid_count, valid_prefix, 2,
        feature_bytes, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // [sample0 token0, sample0 token1, sample1 token0] maps to source rows
    // [0, 2, 1] in contiguous [S, B, ...] storage.
    std::vector<uint8_t> expected_sequence;
    for (uint64_t row : {uint64_t(0), uint64_t(2), uint64_t(1)}) {
        expected_sequence.insert(
            expected_sequence.end(), source.begin() + row * feature_bytes,
            source.begin() + (row + 1) * feature_bytes);
    }
    expect_entry(state, 0, expected_sequence.size());
    EXPECT(read_payload(state, state.task_entries[0]) == expected_sequence);

    int64_t* starts = upload_counts({1, 4});
    int64_t* ends = upload_counts({3, 6});
    ring::launch_record_producer_segmented_pack(
        state, device, source.size(), starts, ends, 2, feature_bytes,
        nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> expected_segments;
    for (uint64_t row : {uint64_t(1), uint64_t(2), uint64_t(4), uint64_t(5)}) {
        expected_segments.insert(
            expected_segments.end(), source.begin() + row * feature_bytes,
            source.begin() + (row + 1) * feature_bytes);
    }
    expect_entry(state, 1, expected_segments.size());
    EXPECT(read_payload(state, state.task_entries[1]) == expected_segments);

    CUDA_CHECK(cudaFree(ends));
    CUDA_CHECK(cudaFree(starts));
    CUDA_CHECK(cudaFree(valid_prefix));
    CUDA_CHECK(cudaFree(valid_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_record_device_gate_publishes_zero_byte_entries() {
    banner("false record gates publish zero-byte entries in FIFO order");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();
    const std::vector<uint8_t> source = pattern(64, 123);
    uint8_t* device = upload(source);
    int32_t* gate = upload_gate(0);
    constexpr uint64_t initial_payload_head = 32;
    constexpr uint64_t initial_actual_bytes = 17;
    *state.payload_head = initial_payload_head;
    *state.actual_bytes_counter = initial_actual_bytes;
    const std::vector<uint8_t> payload_before(state.payload_cap, 0xA5);
    CUDA_CHECK(cudaMemcpy(state.payload_buf, payload_before.data(),
                          payload_before.size(), cudaMemcpyHostToDevice));

    ring::launch_record_producer_static(
        state, device, source.size(), gate, 1);
    ring::launch_record_producer_prefix(
        state, device, source.size(), nullptr, 16, gate, 1);
    ring::launch_record_producer_chunked(
        state, device, source.size(), nullptr, 2, gate, 1);
    ring::launch_record_producer_seq_prefix_pack(
        state, device, source.size(), nullptr, nullptr, 2, 8, gate, 1);
    ring::launch_record_producer_segmented_pack(
        state, device, source.size(), nullptr, nullptr, 2, 8, gate, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr uint64_t gated_variants = 5;
    for (uint64_t sequence = 0; sequence < gated_variants; ++sequence) {
        expect_entry(state, sequence, 0);
    }
    EXPECT(*state.task_head == gated_variants);
    EXPECT(*state.payload_head == initial_payload_head);
    EXPECT(*state.actual_bytes_counter == initial_actual_bytes);
    std::vector<uint8_t> payload_after(state.payload_cap);
    CUDA_CHECK(cudaMemcpy(payload_after.data(), state.payload_buf,
                          payload_after.size(), cudaMemcpyDeviceToHost));
    EXPECT(payload_after == payload_before);

    const int32_t enabled = 1;
    CUDA_CHECK(cudaMemcpy(gate, &enabled, sizeof(enabled),
                          cudaMemcpyHostToDevice));
    ring::launch_record_producer_static(
        state, device, source.size(), gate, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_entry(state, gated_variants, source.size());
    EXPECT(read_payload(state, state.task_entries[gated_variants]) == source);
    EXPECT(*state.task_head == gated_variants + 1);
    EXPECT(*state.payload_head == initial_payload_head +
           ring::align_up(source.size(), ring::PAYLOAD_ALIGN));
    EXPECT(*state.actual_bytes_counter == initial_actual_bytes + source.size());

    CUDA_CHECK(cudaFree(gate));
    CUDA_CHECK(cudaFree(device));
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
    test_record_chunked_compacts_payload_head();
    test_record_sequence_and_segmented_pack();
    test_record_device_gate_publishes_zero_byte_entries();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
