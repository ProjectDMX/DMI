#include "record_consumer.h"

#include <ATen/ATen.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ring {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("record consumer: " + message);
}

uint64_t checked_product(const std::vector<int64_t>& shape,
                         int32_t skip_dimension) {
    uint64_t result = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int32_t>(i) == skip_dimension) continue;
        if (shape[i] < 0) invalid("negative logical tensor dimension");
        const uint64_t dimension = static_cast<uint64_t>(shape[i]);
        if (dimension != 0 &&
            result > std::numeric_limits<uint64_t>::max() / dimension) {
            invalid("logical tensor shape overflows uint64");
        }
        result *= dimension;
    }
    return result;
}

at::ScalarType checked_dtype(int32_t encoded) {
    if (encoded < 0 || encoded >= static_cast<int32_t>(at::ScalarType::NumOptions)) {
        invalid("unsupported encoded dtype");
    }
    const auto dtype = static_cast<at::ScalarType>(encoded);
    const auto size = at::elementSize(dtype);
    if (size <= 0) invalid("dtype has no materializable element size");
    return dtype;
}

uint64_t resolve_slice_length(const PayloadSlice& slice,
                              uint64_t payload_bytes) {
    if (slice.offset_bytes > payload_bytes) {
        invalid("payload-slice offset exceeds actual payload bytes");
    }
    const uint64_t available = payload_bytes - slice.offset_bytes;
    const uint64_t length = slice.length_bytes.value_or(available);
    if (length > available) {
        invalid("payload-slice length exceeds actual payload bytes");
    }
    return length;
}

std::vector<int64_t> resolve_shape(const PayloadSlice& slice,
                                   uint64_t length_bytes,
                                   at::ScalarType dtype) {
    std::vector<int64_t> shape = slice.logical_shape;
    if (shape.empty()) invalid("tensor payload requires a logical shape");

    const uint64_t element_bytes = static_cast<uint64_t>(at::elementSize(dtype));
    if (length_bytes % element_bytes != 0) {
        invalid("payload-slice bytes are not divisible by dtype size");
    }
    const uint64_t elements = length_bytes / element_bytes;

    if (slice.inferred_dynamic_dim >= 0) {
        if (slice.inferred_dynamic_dim >= static_cast<int32_t>(shape.size())) {
            invalid("inferred dynamic dimension is out of range");
        }
        const uint64_t fixed = checked_product(shape, slice.inferred_dynamic_dim);
        if (fixed == 0 || elements % fixed != 0) {
            invalid("payload bytes cannot infer the requested tensor dimension");
        }
        const uint64_t inferred = elements / fixed;
        if (inferred > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            invalid("inferred tensor dimension exceeds int64");
        }
        shape[static_cast<size_t>(slice.inferred_dynamic_dim)] =
            static_cast<int64_t>(inferred);
    } else {
        const uint64_t expected = checked_product(shape, -1);
        if (expected != elements) {
            invalid("fixed tensor shape does not match payload-slice bytes");
        }
    }
    return shape;
}

at::Tensor materialize_tensor(const at::Tensor& payload,
                              const PayloadSlice& slice,
                              uint64_t length_bytes,
                              at::ScalarType dtype) {
    const std::vector<int64_t> shape = resolve_shape(slice, length_bytes, dtype);
    const uint64_t element_bytes = static_cast<uint64_t>(at::elementSize(dtype));
    if (slice.offset_bytes % element_bytes != 0) {
        invalid("payload-slice offset is not aligned to its dtype size");
    }
    at::Tensor bytes = payload.narrow(
        0, static_cast<int64_t>(slice.offset_bytes),
        static_cast<int64_t>(length_bytes));

    // Clone so a submitted row never aliases the reusable pageable payload
    // buffer owned by the p2p worker.
    return bytes.view(dtype).reshape(shape).clone();
}

}  // namespace

RecordConsumer::RecordConsumer(SubmitFn submit_fn)
    : submit_fn_(std::move(submit_fn)) {}

void RecordConsumer::push_descriptor(RecordDescriptor descriptor) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) std::rethrow_exception(failure_);
    descriptors_.push_back(std::move(descriptor));
}

void RecordConsumer::push_descriptors(std::vector<RecordDescriptor> descriptors) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) std::rethrow_exception(failure_);
    for (auto& descriptor : descriptors) {
        descriptors_.push_back(std::move(descriptor));
    }
}

void RecordConsumer::consume_payload(const at::Tensor& payload) {
    RecordDescriptor descriptor;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_) std::rethrow_exception(failure_);
        if (descriptors_.empty()) {
            failure_ = std::make_exception_ptr(std::runtime_error(
                "record consumer: physical payload arrived without an encoded descriptor"));
            std::rethrow_exception(failure_);
        }
        descriptor = std::move(descriptors_.front());
        descriptors_.pop_front();
        ++active_payloads_;
    }

    try {
        if (!payload.defined() || payload.device().type() != at::kCPU ||
            payload.scalar_type() != at::kByte || payload.dim() != 1 ||
            !payload.is_contiguous()) {
            invalid("physical payload must be a contiguous one-dimensional CPU byte tensor");
        }
        materialize_and_submit(descriptor, payload);
        {
            std::lock_guard<std::mutex> lock(mu_);
            --active_payloads_;
        }
        idle_cv_.notify_all();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            --active_payloads_;
        }
        idle_cv_.notify_all();
        record_failure(std::current_exception());
        throw;
    }
}

void RecordConsumer::record_failure(std::exception_ptr failure) noexcept {
    if (!failure) return;
    try {
        std::lock_guard<std::mutex> lock(mu_);
        if (!failure_) failure_ = std::move(failure);
        idle_cv_.notify_all();
    } catch (...) {
        // Failure reporting must not terminate a worker while unwinding.
    }
}

bool RecordConsumer::wait_until_idle(
    std::chrono::milliseconds timeout) const {
    if (timeout.count() < 0) {
        throw std::invalid_argument("record consumer timeout must be non-negative");
    }
    std::unique_lock<std::mutex> lock(mu_);
    const bool ready = idle_cv_.wait_for(lock, timeout, [this] {
        return failure_ || (descriptors_.empty() && active_payloads_ == 0);
    });
    if (failure_) std::rethrow_exception(failure_);
    return ready;
}

void RecordConsumer::rethrow_if_failed() const {
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(mu_);
        failure = failure_;
    }
    if (failure) std::rethrow_exception(failure);
}

void RecordConsumer::finish() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) std::rethrow_exception(failure_);
    if (!descriptors_.empty()) {
        invalid("durable completion found leftover encoded descriptors");
    }
    if (active_payloads_ != 0) {
        invalid("durable completion found active record materialization");
    }
}

size_t RecordConsumer::pending_descriptors() const {
    std::lock_guard<std::mutex> lock(mu_);
    return descriptors_.size();
}

void RecordConsumer::materialize_and_submit(
    const RecordDescriptor& descriptor,
    const at::Tensor& payload) {
    if (descriptor.layout.empty()) invalid("record layout is empty");
    if (!submit_fn_ && !descriptor.rows.empty()) {
        invalid("record submit callback is not configured");
    }

    const uint64_t payload_bytes = static_cast<uint64_t>(payload.numel());

    for (const EncodedRecordRow& encoded_row : descriptor.rows) {
        dmx_host::GenericRecordRow row;
        row.layout = descriptor.layout;
        row.cells.reserve(encoded_row.cells.size());

        uint64_t accounted_bytes = 0;
        for (const EncodedRecordCell& encoded_cell : encoded_row.cells) {
            if (const auto* value = std::get_if<std::string>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value = std::get_if<int32_t>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value = std::get_if<int64_t>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value = std::get_if<double>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else if (const auto* value =
                           std::get_if<std::vector<int64_t>>(&encoded_cell)) {
                row.cells.emplace_back(*value);
            } else {
                const auto& slice = std::get<PayloadSlice>(encoded_cell);
                const uint64_t length = resolve_slice_length(slice, payload_bytes);
                accounted_bytes += length;
                const at::ScalarType dtype = checked_dtype(slice.dtype);

                switch (slice.materialization) {
                    case PayloadMaterialization::TENSOR:
                        row.cells.emplace_back(materialize_tensor(
                            payload, slice, length, dtype));
                        break;
                    case PayloadMaterialization::FLOAT_SCALAR: {
                        if (!c10::isFloatingType(dtype)) {
                            invalid("floating scalar requires a floating-point dtype");
                        }
                        if (length != static_cast<uint64_t>(at::elementSize(dtype))) {
                            invalid("floating scalar slice must contain exactly one element");
                        }
                        if (slice.offset_bytes %
                                static_cast<uint64_t>(at::elementSize(dtype)) != 0) {
                            invalid("floating scalar offset is not dtype-aligned");
                        }
                        at::Tensor scalar = payload.narrow(
                            0, static_cast<int64_t>(slice.offset_bytes),
                            static_cast<int64_t>(length)).view(dtype);
                        row.cells.emplace_back(scalar.item<double>());
                        break;
                    }
                    case PayloadMaterialization::INT_SCALAR: {
                        if (!c10::isIntegralType(dtype, /*includeBool=*/false)) {
                            invalid("integer scalar requires an integral dtype");
                        }
                        if (length != static_cast<uint64_t>(at::elementSize(dtype))) {
                            invalid("integer scalar slice must contain exactly one element");
                        }
                        if (slice.offset_bytes %
                                static_cast<uint64_t>(at::elementSize(dtype)) != 0) {
                            invalid("integer scalar offset is not dtype-aligned");
                        }
                        at::Tensor scalar = payload.narrow(
                            0, static_cast<int64_t>(slice.offset_bytes),
                            static_cast<int64_t>(length)).view(dtype);
                        row.cells.emplace_back(scalar.item<int64_t>());
                        break;
                    }
                }
            }
        }

        submit_fn_(std::move(row), accounted_bytes);
    }
}

}  // namespace ring
