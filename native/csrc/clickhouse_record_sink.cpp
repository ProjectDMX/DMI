#include "clickhouse_record_sink.h"

#include <ATen/ATen.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace dmx_host {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("clickhouse record sink: " + message);
}

std::uint64_t checked_product(const std::vector<std::int64_t>& shape,
                              std::int32_t skip_dimension) {
    std::uint64_t result = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<std::int32_t>(i) == skip_dimension) continue;
        if (shape[i] < 0) invalid("negative logical tensor dimension");
        const auto dimension = static_cast<std::uint64_t>(shape[i]);
        if (dimension != 0 &&
            result > std::numeric_limits<std::uint64_t>::max() / dimension) {
            invalid("logical tensor shape overflows uint64");
        }
        result *= dimension;
    }
    return result;
}

at::ScalarType checked_dtype(std::int32_t encoded) {
    if (encoded < 0 ||
        encoded >= static_cast<std::int32_t>(at::ScalarType::NumOptions)) {
        invalid("unsupported encoded dtype");
    }
    const auto dtype = static_cast<at::ScalarType>(encoded);
    if (at::elementSize(dtype) <= 0) {
        invalid("dtype has no materializable element size");
    }
    return dtype;
}

std::uint64_t resolve_slice_length(const ring::PayloadSlice& slice,
                                   std::uint64_t payload_bytes) {
    if (slice.offset_bytes > payload_bytes) {
        invalid("payload-slice offset exceeds actual payload bytes");
    }
    const std::uint64_t available = payload_bytes - slice.offset_bytes;
    const std::uint64_t length = slice.length_bytes.value_or(available);
    if (length > available) {
        invalid("payload-slice length exceeds actual payload bytes");
    }
    return length;
}

std::vector<std::int64_t> resolve_shape(const ring::PayloadSlice& slice,
                                        std::uint64_t length_bytes,
                                        at::ScalarType dtype) {
    std::vector<std::int64_t> shape = slice.logical_shape;
    if (shape.empty()) invalid("tensor payload requires a logical shape");

    const auto element_bytes =
        static_cast<std::uint64_t>(at::elementSize(dtype));
    if (length_bytes % element_bytes != 0) {
        invalid("payload-slice bytes are not divisible by dtype size");
    }
    const std::uint64_t elements = length_bytes / element_bytes;

    if (slice.inferred_dynamic_dim >= 0) {
        if (slice.inferred_dynamic_dim >=
            static_cast<std::int32_t>(shape.size())) {
            invalid("inferred dynamic dimension is out of range");
        }
        const std::uint64_t fixed =
            checked_product(shape, slice.inferred_dynamic_dim);
        if (fixed == 0 || elements % fixed != 0) {
            invalid("payload bytes cannot infer the requested tensor dimension");
        }
        const std::uint64_t inferred = elements / fixed;
        if (inferred >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
            invalid("inferred tensor dimension exceeds int64");
        }
        shape[static_cast<std::size_t>(slice.inferred_dynamic_dim)] =
            static_cast<std::int64_t>(inferred);
    } else if (checked_product(shape, -1) != elements) {
        invalid("fixed tensor shape does not match payload-slice bytes");
    }
    return shape;
}

at::Tensor materialize_tensor(const at::Tensor& payload,
                              const ring::PayloadSlice& slice,
                              std::uint64_t length_bytes,
                              at::ScalarType dtype) {
    const std::vector<std::int64_t> shape =
        resolve_shape(slice, length_bytes, dtype);
    const auto element_bytes =
        static_cast<std::uint64_t>(at::elementSize(dtype));
    if (slice.offset_bytes % element_bytes != 0) {
        invalid("payload-slice offset is not aligned to its dtype size");
    }
    at::Tensor bytes = payload.narrow(
        0, static_cast<std::int64_t>(slice.offset_bytes),
        static_cast<std::int64_t>(length_bytes));

    // The ClickHouse host queue may outlive this envelope.  Other sinks can
    // consume the owned payload directly without paying this per-cell copy.
    return bytes.view(dtype).reshape(shape).clone();
}

}  // namespace

ClickHouseRecordSink::ClickHouseRecordSink(SubmitRowFn submit_row,
                                           FlushFn flush,
                                           RethrowFn rethrow)
    : submit_row_(std::move(submit_row)),
      flush_(std::move(flush)),
      rethrow_(std::move(rethrow)) {
    if (!submit_row_ || !flush_ || !rethrow_) {
        throw std::invalid_argument(
            "ClickHouseRecordSink requires submit, flush, and failure callbacks");
    }
}

void ClickHouseRecordSink::submit(ring::RecordEnvelope envelope) {
    const auto& descriptor = envelope.descriptor;
    const auto& payload = envelope.payload;
    if (descriptor.layout.empty()) invalid("record layout is empty");

    const auto payload_bytes = static_cast<std::uint64_t>(payload.numel());
    for (const ring::EncodedRecordRow& encoded_row : descriptor.rows) {
        GenericRecordRow row;
        row.layout = descriptor.layout;
        row.cells.reserve(encoded_row.cells.size());

        std::uint64_t accounted_bytes = 0;
        for (const ring::EncodedRecordCell& encoded_cell : encoded_row.cells) {
            if (const auto* value = std::get_if<std::string>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value =
                           std::get_if<std::int32_t>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value =
                           std::get_if<std::int64_t>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value = std::get_if<double>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value =
                           std::get_if<std::vector<std::int64_t>>(
                               &encoded_cell)) {
                row.cells.emplace_back(*value);
            } else {
                const auto& slice = std::get<ring::PayloadSlice>(encoded_cell);
                const std::uint64_t length =
                    resolve_slice_length(slice, payload_bytes);
                accounted_bytes += length;
                const at::ScalarType dtype = checked_dtype(slice.dtype);

                switch (slice.materialization) {
                    case ring::PayloadMaterialization::TENSOR:
                        row.cells.emplace_back(materialize_tensor(
                            payload, slice, length, dtype));
                        break;
                    case ring::PayloadMaterialization::FLOAT_SCALAR: {
                        if (!c10::isFloatingType(dtype)) {
                            invalid(
                                "floating scalar requires a floating-point dtype");
                        }
                        if (length != static_cast<std::uint64_t>(
                                          at::elementSize(dtype))) {
                            invalid(
                                "floating scalar slice must contain exactly one element");
                        }
                        if (slice.offset_bytes %
                                static_cast<std::uint64_t>(
                                    at::elementSize(dtype)) !=
                            0) {
                            invalid(
                                "floating scalar offset is not dtype-aligned");
                        }
                        at::Tensor scalar = payload.narrow(
                            0, static_cast<std::int64_t>(slice.offset_bytes),
                            static_cast<std::int64_t>(length)).view(dtype);
                        row.cells.emplace_back(scalar.item<double>());
                        break;
                    }
                    case ring::PayloadMaterialization::INT_SCALAR: {
                        if (!c10::isIntegralType(
                                dtype, /*includeBool=*/false)) {
                            invalid(
                                "integer scalar requires an integral dtype");
                        }
                        if (length != static_cast<std::uint64_t>(
                                          at::elementSize(dtype))) {
                            invalid(
                                "integer scalar slice must contain exactly one element");
                        }
                        if (slice.offset_bytes %
                                static_cast<std::uint64_t>(
                                    at::elementSize(dtype)) !=
                            0) {
                            invalid("integer scalar offset is not dtype-aligned");
                        }
                        at::Tensor scalar = payload.narrow(
                            0, static_cast<std::int64_t>(slice.offset_bytes),
                            static_cast<std::int64_t>(length)).view(dtype);
                        row.cells.emplace_back(scalar.item<std::int64_t>());
                        break;
                    }
                }
            }
        }

        submit_row_(std::move(row), accounted_bytes);
    }
}

bool ClickHouseRecordSink::flush_and_wait(Duration timeout) {
    return flush_(timeout);
}

void ClickHouseRecordSink::rethrow_if_failed() const {
    rethrow_();
}

}  // namespace dmx_host
