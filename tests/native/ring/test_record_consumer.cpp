#include "ring/record_consumer.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static ring::PayloadSlice tensor_slice(uint64_t offset,
                                       std::optional<uint64_t> length,
                                       std::vector<int64_t> shape,
                                       int32_t dynamic_dim = -1) {
    ring::PayloadSlice slice;
    slice.offset_bytes = offset;
    slice.length_bytes = length;
    slice.materialization = ring::PayloadMaterialization::TENSOR;
    slice.dtype = static_cast<int32_t>(at::kFloat);
    slice.logical_shape = std::move(shape);
    slice.inferred_dynamic_dim = dynamic_dim;
    return slice;
}

static at::Tensor byte_payload(const std::vector<float>& values) {
    at::Tensor floats = at::tensor(values, at::TensorOptions().dtype(at::kFloat));
    return floats.view(at::kByte).clone();
}

static ring::PayloadSlice scalar_slice(
    uint64_t offset,
    uint64_t length,
    ring::PayloadMaterialization materialization,
    at::ScalarType dtype) {
    ring::PayloadSlice slice;
    slice.offset_bytes = offset;
    slice.length_bytes = length;
    slice.materialization = materialization;
    slice.dtype = static_cast<int32_t>(dtype);
    return slice;
}

static void test_fifo_and_dynamic_tensor_materialization() {
    std::printf("[ TEST ] record descriptor FIFO and dynamic tensor materialization\n");
    std::vector<dmx_host::GenericRecordRow> submitted;
    ring::RecordConsumer consumer(
        [&](dmx_host::GenericRecordRow row, uint64_t) {
            submitted.push_back(std::move(row));
        });

    ring::RecordDescriptor first;
    first.layout = "layout_a";
    first.rows = {{std::vector<ring::EncodedRecordCell>{
        std::string("first"), int32_t(7), int64_t(11), 2.5,
        std::vector<int64_t>{3, 4},
        tensor_slice(0, std::nullopt, {-1, 2}, 0)}}};

    ring::RecordDescriptor second;
    second.layout = "layout_b";
    second.rows = {{std::vector<ring::EncodedRecordCell>{
        std::string("second"), tensor_slice(0, 8, {2})}}};

    consumer.push_descriptors({first, second});
    consumer.consume_payload(byte_payload({1, 2, 3, 4, 5, 6}));
    consumer.consume_payload(byte_payload({9, 10}));
    consumer.finish();

    EXPECT(submitted.size() == 2);
    EXPECT(submitted[0].layout == "layout_a");
    EXPECT(submitted[1].layout == "layout_b");
    EXPECT(std::get<std::string>(submitted[0].cells[0]) == "first");
    EXPECT(std::get<int32_t>(submitted[0].cells[1]) == 7);
    EXPECT(std::get<int64_t>(submitted[0].cells[2]) == 11);
    EXPECT(std::get<double>(submitted[0].cells[3]) == 2.5);
    const at::Tensor first_tensor =
        std::get<at::Tensor>(submitted[0].cells[5]);
    EXPECT(first_tensor.sizes().vec() == std::vector<int64_t>({3, 2}));
    EXPECT(at::equal(first_tensor,
                     at::tensor({1.f, 2.f, 3.f, 4.f, 5.f, 6.f})
                         .reshape({3, 2})));
    const at::Tensor second_tensor =
        std::get<at::Tensor>(submitted[1].cells[1]);
    EXPECT(at::equal(second_tensor, at::tensor({9.f, 10.f})));
}

static void test_zero_row_descriptor_consumes_zero_byte_task() {
    std::printf("[ TEST ] zero-row descriptor consumes its zero-byte task\n");
    std::vector<dmx_host::GenericRecordRow> submitted;
    ring::RecordConsumer consumer(
        [&](dmx_host::GenericRecordRow row, uint64_t) {
            submitted.push_back(std::move(row));
        });

    ring::RecordDescriptor empty;
    empty.layout = "filtered";
    consumer.push_descriptor(std::move(empty));
    consumer.consume_payload(at::empty(
        {0}, at::TensorOptions().dtype(at::kByte).device(at::kCPU)));
    consumer.finish();

    EXPECT(submitted.empty());
}

static void test_scalar_materialization() {
    std::printf("[ TEST ] floating and integer scalar materialization\n");
    std::vector<dmx_host::GenericRecordRow> submitted;
    ring::RecordConsumer consumer(
        [&](dmx_host::GenericRecordRow row, uint64_t) {
            submitted.push_back(std::move(row));
        });

    ring::RecordDescriptor descriptor;
    descriptor.layout = "scalars";
    descriptor.rows = {{std::vector<ring::EncodedRecordCell>{
        scalar_slice(0, sizeof(float),
                     ring::PayloadMaterialization::FLOAT_SCALAR, at::kFloat),
        scalar_slice(8, sizeof(int64_t),
                     ring::PayloadMaterialization::INT_SCALAR, at::kLong)}}};
    consumer.push_descriptor(std::move(descriptor));

    at::Tensor payload = at::zeros(
        {16}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    const float floating = 3.25f;
    const int64_t integer = 41;
    std::memcpy(payload.data_ptr<uint8_t>(), &floating, sizeof(floating));
    std::memcpy(payload.data_ptr<uint8_t>() + 8, &integer, sizeof(integer));
    consumer.consume_payload(payload);
    consumer.finish();

    EXPECT(submitted.size() == 1);
    EXPECT(std::get<double>(submitted[0].cells[0]) == 3.25);
    EXPECT(std::get<int64_t>(submitted[0].cells[1]) == 41);
}

static void test_exact_association_failures() {
    std::printf("[ TEST ] descriptor/payload association failures\n");
    ring::RecordConsumer missing(
        [](dmx_host::GenericRecordRow, uint64_t) {});
    bool missing_failed = false;
    try {
        missing.consume_payload(byte_payload({1}));
    } catch (const std::runtime_error&) {
        missing_failed = true;
    }
    EXPECT(missing_failed);

    ring::RecordConsumer leftover(
        [](dmx_host::GenericRecordRow, uint64_t) {});
    ring::RecordDescriptor descriptor;
    descriptor.layout = "leftover";
    leftover.push_descriptor(std::move(descriptor));
    bool leftover_failed = false;
    try {
        leftover.finish();
    } catch (const std::runtime_error&) {
        leftover_failed = true;
    }
    EXPECT(leftover_failed);
}

int main() {
    setbuf(stdout, nullptr);
    std::printf("test_record_consumer\n");
    test_fifo_and_dynamic_tensor_materialization();
    test_zero_row_descriptor_consumes_zero_byte_task();
    test_scalar_materialization();
    test_exact_association_failures();
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
