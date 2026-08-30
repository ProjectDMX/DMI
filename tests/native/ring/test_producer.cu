// CUDA data-path tests for the current static, prefix, and chunked producers.

#include "ring/producer.cuh"
#include "ring/payload_ring.cuh"
#include "ring/publication_word.h"
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

struct DeviceBytes {
    uint8_t* allocation = nullptr;
    uint8_t* data = nullptr;
};

static DeviceBytes upload_with_offset(const std::vector<uint8_t>& source,
                                      uint64_t offset) {
    DeviceBytes result;
    CUDA_CHECK(cudaMalloc(&result.allocation, source.size() + offset));
    result.data = result.allocation + offset;
    CUDA_CHECK(cudaMemcpy(result.data, source.data(), source.size(),
                          cudaMemcpyHostToDevice));
    return result;
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
                                         uint64_t logical_start,
                                         uint64_t actual_bytes) {
    const ring::TwoSpan spans = ring::payload_compute_spans(
        logical_start, state.payload_cap, actual_bytes);
    std::vector<uint8_t> result(actual_bytes);
    if (spans.len1 > 0) {
        CUDA_CHECK(cudaMemcpy(result.data(),
                              state.payload_buf + spans.off1,
                              spans.len1, cudaMemcpyDeviceToHost));
    }
    if (spans.len2 > 0) {
        CUDA_CHECK(cudaMemcpy(result.data() + spans.len1,
                              state.payload_buf + spans.off2,
                              spans.len2, cudaMemcpyDeviceToHost));
    }
    return result;
}

static void expect_publication(const ring::RingState& state, uint64_t sequence,
                               uint64_t expected_bytes) {
    const uint64_t word = __atomic_load_n(
        &state.publication_slots[sequence % state.task_cap], __ATOMIC_ACQUIRE);
    EXPECT((word & ring::PUBLICATION_READY) != 0);
    EXPECT((word & ring::PUBLICATION_SIZE_MASK) == expected_bytes);
}

enum class ProducerVariant {
    Static,
    Prefix,
    Chunked,
    RecordStatic,
    RecordPrefix,
    RecordChunked,
    RecordSequencePrefix,
    RecordSegmented,
};

struct ProducerVariantCase {
    ProducerVariant variant;
    const char* name;
};

static const ProducerVariantCase kProducerVariants[] = {
    {ProducerVariant::Static, "static"},
    {ProducerVariant::Prefix, "prefix"},
    {ProducerVariant::Chunked, "chunked"},
    {ProducerVariant::RecordStatic, "record static"},
    {ProducerVariant::RecordPrefix, "record prefix"},
    {ProducerVariant::RecordChunked, "record chunked"},
    {ProducerVariant::RecordSequencePrefix, "record sequence-prefix"},
    {ProducerVariant::RecordSegmented, "record segmented"},
};

static std::vector<uint8_t> expected_for_variant(
        ProducerVariant variant, const std::vector<uint8_t>& source) {
    if (variant != ProducerVariant::Chunked &&
        variant != ProducerVariant::RecordChunked) {
        return source;
    }

    // Both chunked cases select 32 bytes from each 48-byte input chunk.
    std::vector<uint8_t> expected;
    expected.insert(expected.end(), source.begin(), source.begin() + 32);
    expected.insert(expected.end(), source.begin() + 48,
                    source.begin() + 80);
    return expected;
}

static void launch_alignment_case(ProducerVariant variant,
                                  const ring::RingState& state,
                                  const uint8_t* source,
                                  uint64_t source_bytes) {
    int64_t* aux1 = nullptr;
    int64_t* aux2 = nullptr;

    switch (variant) {
        case ProducerVariant::Static:
            ring::launch_producer_static(state, source, source_bytes, 0);
            break;
        case ProducerVariant::Prefix:
            aux1 = upload_counts({3});
            ring::launch_producer_prefix(
                state, source, source_bytes, aux1, 32, 0);
            break;
        case ProducerVariant::Chunked:
            aux1 = upload_counts({32, 32});
            ring::launch_producer_chunked(
                state, source, source_bytes, aux1, 2, 0);
            break;
        case ProducerVariant::RecordStatic:
            ring::launch_record_producer_static(
                state, source, source_bytes, nullptr, 0);
            break;
        case ProducerVariant::RecordPrefix:
            aux1 = upload_counts({3});
            ring::launch_record_producer_prefix(
                state, source, source_bytes, aux1, 32, nullptr, 0);
            break;
        case ProducerVariant::RecordChunked:
            aux1 = upload_counts({32, 32});
            ring::launch_record_producer_chunked(
                state, source, source_bytes, aux1, 2, nullptr, 0);
            break;
        case ProducerVariant::RecordSequencePrefix:
            aux1 = upload_counts({3});
            aux2 = upload_counts({0, 3});
            ring::launch_record_producer_seq_prefix_pack(
                state, source, source_bytes, aux1, aux2, 1, 32, nullptr, 0);
            break;
        case ProducerVariant::RecordSegmented:
            aux1 = upload_counts({0});
            aux2 = upload_counts({3});
            ring::launch_record_producer_segmented_pack(
                state, source, source_bytes, aux1, aux2, 1, 32, nullptr, 0);
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (aux2 != nullptr) CUDA_CHECK(cudaFree(aux2));
    if (aux1 != nullptr) CUDA_CHECK(cudaFree(aux1));
}

static void test_all_producers_handle_unaligned_addresses() {
    constexpr uint64_t source_bytes = 96;
    const std::vector<uint8_t> source = pattern(source_bytes, 149);

    for (const ProducerVariantCase& producer : kProducerVariants) {
        for (uint64_t source_offset : {uint64_t(1), uint64_t(0)}) {
            const bool test_source = source_offset != 0;
            std::printf("[ TEST ] %s producer with unaligned %s address\n",
                        producer.name, test_source ? "source" : "destination");

            ring::AllocatedRing allocated(make_config());
            allocated.init();
            ring::RingState& state = allocated.state();
            const uint64_t logical_start = test_source ? 0 : 1;
            *state.payload_head = logical_start;

            const DeviceBytes device_source =
                upload_with_offset(source, source_offset);
            EXPECT((reinterpret_cast<uintptr_t>(device_source.data) &
                    (ring::PAYLOAD_ALIGN - 1)) ==
                   (test_source ? 1 : 0));
            EXPECT((reinterpret_cast<uintptr_t>(state.payload_buf +
                                                logical_start) &
                    (ring::PAYLOAD_ALIGN - 1)) ==
                   (test_source ? 0 : 1));

            launch_alignment_case(producer.variant, state,
                                  device_source.data, source.size());

            const std::vector<uint8_t> expected =
                expected_for_variant(producer.variant, source);
            expect_publication(state, 0, expected.size());
            EXPECT(read_payload(state, logical_start, expected.size()) ==
                   expected);

            CUDA_CHECK(cudaFree(device_source.allocation));
        }
    }
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

    expect_publication(state, 0, source.size());
    EXPECT(*state.task_head == 1);
    EXPECT(*state.payload_head == ring::align_up(source.size(), ring::PAYLOAD_ALIGN));
    EXPECT(*state.actual_bytes_counter == source.size());
    EXPECT(read_payload(state, 0, source.size()) == source);

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

    expect_publication(state, 0, source.size());
    EXPECT(*state.payload_head == 560);
    EXPECT(read_payload(state, 480, source.size()) == source);

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
    expect_publication(state, 0, expected_bytes);
    std::vector<uint8_t> expected(source.begin(), source.begin() + expected_bytes);
    EXPECT(read_payload(state, 0, expected_bytes) == expected);
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
    expect_publication(state, 0, 0);
    EXPECT(*state.payload_head == 0);

    const int64_t oversized = 100;
    CUDA_CHECK(cudaMemcpy(row_count, &oversized, sizeof(oversized),
                          cudaMemcpyHostToDevice));
    ring::launch_producer_prefix(state, device, source.size(), row_count,
                                 row_bytes, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_publication(state, 1, source.size());
    EXPECT(read_payload(state, 0, source.size()) == source);
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

    expect_publication(state, 0, expected.size());
    EXPECT(read_payload(state, 0, expected.size()) == expected);
    EXPECT(*state.actual_bytes_counter == expected.size());
    EXPECT(*state.payload_head ==
           ring::align_up(expected.size(), ring::PAYLOAD_ALIGN));

    CUDA_CHECK(cudaFree(device_counts));
    CUDA_CHECK(cudaFree(device));
}

static void test_chunked_unaligned_internal_boundaries() {
    banner("legacy chunked producer checks effective chunk addresses");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint32_t chunks = 2;
    constexpr uint64_t input_chunk = 33;
    const std::vector<int64_t> selected = {17, 17};
    const std::vector<uint8_t> source = pattern(chunks * input_chunk, 103);
    std::vector<uint8_t> expected;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto begin = source.begin() + chunk * input_chunk;
        expected.insert(expected.end(), begin, begin + selected[chunk]);
    }

    uint8_t* device = upload(source);
    int64_t* device_selected = upload_counts(selected);
    ring::launch_producer_chunked(
        state, device, source.size(), device_selected, chunks, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_publication(state, 0, expected.size());
    EXPECT(read_payload(state, 0, expected.size()) == expected);
    EXPECT(*state.payload_head ==
           ring::align_up(expected.size(), ring::PAYLOAD_ALIGN));

    CUDA_CHECK(cudaFree(device_selected));
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
        expect_publication(state, sequence, sizes[sequence]);
        EXPECT(read_payload(state, expected_head, sizes[sequence]) ==
               sources[sequence]);
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
    expect_publication(state, 0, actual);
    EXPECT(*state.payload_head == ring::align_up(actual, ring::PAYLOAD_ALIGN));

    std::vector<uint8_t> expected;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto begin = source.begin() + chunk * input_chunk;
        expected.insert(expected.end(), begin, begin + selected[chunk]);
    }
    EXPECT(read_payload(state, 0, actual) == expected);

    CUDA_CHECK(cudaFree(device_selected));
    CUDA_CHECK(cudaFree(device));
}

static void test_record_chunked_unaligned_boundaries() {
    banner("record chunked producer copies unaligned chunk boundaries");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    constexpr uint32_t chunks = 2;
    constexpr uint64_t input_chunk = 33;
    const std::vector<int64_t> selected = {17, 17};
    const std::vector<uint8_t> source = pattern(chunks * input_chunk, 107);
    std::vector<uint8_t> expected;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const auto begin = source.begin() + chunk * input_chunk;
        expected.insert(expected.end(), begin, begin + selected[chunk]);
    }

    uint8_t* device = upload(source);
    int64_t* device_selected = upload_counts(selected);
    ring::launch_record_producer_chunked(
        state, device, source.size(), device_selected, chunks,
        nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    expect_publication(state, 0, expected.size());
    EXPECT(read_payload(state, 0, expected.size()) == expected);
    EXPECT(*state.actual_bytes_counter == expected.size());
    EXPECT(*state.payload_head ==
           ring::align_up(expected.size(), ring::PAYLOAD_ALIGN));

    CUDA_CHECK(cudaFree(device_selected));
    CUDA_CHECK(cudaFree(device));
}

static void test_record_sequence_and_segmented_pack() {
    banner("record row packers preserve declared row order");
    ring::AllocatedRing allocated(make_config());
    allocated.init();
    ring::RingState& state = allocated.state();

    // Source is [S=3, B=2, feature=20 bytes], one distinct byte pattern per row.
    constexpr uint64_t feature_bytes = 20;
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
    expect_publication(state, 0, expected_sequence.size());
    EXPECT(read_payload(state, 0, expected_sequence.size()) == expected_sequence);

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
    expect_publication(state, 1, expected_segments.size());
    const uint64_t second_start =
        ring::align_up(expected_sequence.size(), ring::PAYLOAD_ALIGN);
    EXPECT(read_payload(state, second_start, expected_segments.size()) ==
           expected_segments);

    CUDA_CHECK(cudaFree(ends));
    CUDA_CHECK(cudaFree(starts));
    CUDA_CHECK(cudaFree(valid_prefix));
    CUDA_CHECK(cudaFree(valid_count));
    CUDA_CHECK(cudaFree(device));
}

static void test_record_device_gate_rejects_without_publication() {
    banner("false record gates leave ring state unchanged");
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
    EXPECT(*state.task_head == 0);
    for (uint64_t sequence = 0; sequence < gated_variants; ++sequence) {
        EXPECT(state.publication_slots[sequence] == 0);
    }
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
    expect_publication(state, 0, source.size());
    EXPECT(read_payload(state, initial_payload_head, source.size()) == source);
    EXPECT(*state.task_head == 1);
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
    test_chunked_unaligned_internal_boundaries();
    test_serialized_static_launches();
    test_record_chunked_compacts_payload_head();
    test_record_chunked_unaligned_boundaries();
    test_record_sequence_and_segmented_pack();
    test_record_device_gate_rejects_without_publication();
    test_all_producers_handle_unaligned_addresses();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
