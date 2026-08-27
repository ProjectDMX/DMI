#include "clickhouse_record_sink.h"

#include <ATen/ATen.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(condition)                                                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                    \
                         __FILE__, __LINE__, #condition);                    \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

static ring::PayloadSlice tensor_slice(
    std::uint64_t offset,
    std::optional<std::uint64_t> length,
    std::vector<std::int64_t> shape,
    std::int32_t dynamic_dim = -1) {
    ring::PayloadSlice slice;
    slice.offset_bytes = offset;
    slice.length_bytes = length;
    slice.materialization = ring::PayloadMaterialization::TENSOR;
    slice.dtype = static_cast<std::int32_t>(at::kFloat);
    slice.logical_shape = std::move(shape);
    slice.inferred_dynamic_dim = dynamic_dim;
    return slice;
}

static ring::PayloadSlice scalar_slice(
    std::uint64_t offset,
    std::uint64_t length,
    ring::PayloadMaterialization materialization,
    at::ScalarType dtype) {
    ring::PayloadSlice slice;
    slice.offset_bytes = offset;
    slice.length_bytes = length;
    slice.materialization = materialization;
    slice.dtype = static_cast<std::int32_t>(dtype);
    return slice;
}

static void test_materializes_only_inside_clickhouse_adapter() {
    std::printf("[ TEST ] ClickHouse adapter materializes raw envelopes\n");
    std::vector<dmx_host::GenericRecordRow> rows;
    std::vector<std::uint64_t> sizes;
    int flushes = 0;
    ring::RecordSink::Duration observed_timeout{0};
    dmx_host::ClickHouseRecordSink sink(
        [&](dmx_host::GenericRecordRow row, std::uint64_t size) {
            rows.push_back(std::move(row));
            sizes.push_back(size);
        },
        [&](ring::RecordSink::Duration timeout) {
            ++flushes;
            observed_timeout = timeout;
            return true;
        },
        [] {});

    ring::RecordDescriptor descriptor;
    descriptor.layout = "layout_a";
    descriptor.rows = {{std::vector<ring::EncodedRecordCell>{
        std::string("first"), std::int32_t(7), std::int64_t(11), 2.5,
        std::vector<std::int64_t>{3, 4},
        tensor_slice(0, std::nullopt, {-1, 2}, 0)}}};
    at::Tensor payload = at::tensor(
        {1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
        at::TensorOptions().dtype(at::kFloat)).view(at::kByte).clone();
    at::Tensor payload_alias = payload;

    sink.submit(ring::RecordEnvelope{
        std::move(descriptor), std::move(payload)});
    payload_alias.zero_();

    EXPECT(rows.size() == 1);
    EXPECT(rows[0].layout == "layout_a");
    EXPECT(std::get<std::string>(rows[0].cells[0]) == "first");
    EXPECT(std::get<std::int32_t>(rows[0].cells[1]) == 7);
    EXPECT(std::get<std::int64_t>(rows[0].cells[2]) == 11);
    EXPECT(std::get<double>(rows[0].cells[3]) == 2.5);
    EXPECT(std::get<std::vector<std::int64_t>>(rows[0].cells[4]) ==
           std::vector<std::int64_t>({3, 4}));
    const at::Tensor tensor = std::get<at::Tensor>(rows[0].cells[5]);
    EXPECT(tensor.sizes().vec() == std::vector<std::int64_t>({3, 2}));
    EXPECT(at::equal(
        tensor,
        at::tensor({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}).reshape({3, 2})));
    EXPECT(sizes == std::vector<std::uint64_t>({24}));
    EXPECT(sink.flush_and_wait(std::chrono::milliseconds(37)));
    EXPECT(flushes == 1);
    EXPECT(observed_timeout == std::chrono::milliseconds(37));
}

static void test_scalar_materialization_and_failure_delegation() {
    std::printf("[ TEST ] scalar materialization and sink failure delegation\n");
    std::vector<dmx_host::GenericRecordRow> rows;
    bool fail = false;
    dmx_host::ClickHouseRecordSink sink(
        [&](dmx_host::GenericRecordRow row, std::uint64_t) {
            rows.push_back(std::move(row));
        },
        [](ring::RecordSink::Duration) { return true; },
        [&] {
            if (fail) throw std::runtime_error("host failure");
        });

    ring::RecordDescriptor descriptor;
    descriptor.layout = "scalars";
    descriptor.rows = {{std::vector<ring::EncodedRecordCell>{
        scalar_slice(0, sizeof(float),
                     ring::PayloadMaterialization::FLOAT_SCALAR, at::kFloat),
        scalar_slice(8, sizeof(std::int64_t),
                     ring::PayloadMaterialization::INT_SCALAR, at::kLong)}}};
    at::Tensor payload = at::zeros(
        {16}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    const float floating = 3.25f;
    const std::int64_t integer = 41;
    std::memcpy(payload.data_ptr<std::uint8_t>(), &floating, sizeof(floating));
    std::memcpy(payload.data_ptr<std::uint8_t>() + 8, &integer,
                sizeof(integer));

    sink.submit(ring::RecordEnvelope{
        std::move(descriptor), std::move(payload)});
    EXPECT(rows.size() == 1);
    EXPECT(std::get<double>(rows[0].cells[0]) == 3.25);
    EXPECT(std::get<std::int64_t>(rows[0].cells[1]) == 41);

    fail = true;
    bool rethrew = false;
    try {
        sink.rethrow_if_failed();
    } catch (const std::runtime_error&) {
        rethrew = true;
    }
    EXPECT(rethrew);
}

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_clickhouse_record_sink\n");
    test_materializes_only_inside_clickhouse_adapter();
    test_scalar_materialization_and_failure_delegation();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
