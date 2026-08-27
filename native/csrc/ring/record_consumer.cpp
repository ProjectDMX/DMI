#include "record_consumer.h"

#include <ATen/ATen.h>

#include <stdexcept>
#include <utility>

namespace ring {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("record consumer: " + message);
}

}  // namespace

RecordConsumer::RecordConsumer(std::shared_ptr<RecordSink> sink)
    : sink_(std::move(sink)) {}

void RecordConsumer::push_descriptor(RecordDescriptor descriptor) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) std::rethrow_exception(failure_);
    descriptors_.push_back(std::move(descriptor));
}

void RecordConsumer::push_descriptors(
    std::vector<RecordDescriptor> descriptors) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) std::rethrow_exception(failure_);
    for (auto& descriptor : descriptors) {
        descriptors_.push_back(std::move(descriptor));
    }
}

void RecordConsumer::consume_payload(at::Tensor payload) {
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
            invalid(
                "physical payload must be a contiguous one-dimensional CPU byte tensor");
        }
        if (descriptor.layout.empty()) invalid("record layout is empty");
        if (!descriptor.rows.empty()) {
            if (!sink_) invalid("record sink is not configured");
            sink_->submit(RecordEnvelope{
                std::move(descriptor), std::move(payload)});
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            --active_payloads_;
        }
        idle_cv_.notify_all();
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        {
            // Latch failure and retire the active payload atomically.  A flush
            // waiter must never observe an idle consumer before the submit
            // failure becomes visible.
            std::lock_guard<std::mutex> lock(mu_);
            if (!failure_) failure_ = failure;
            --active_payloads_;
        }
        idle_cv_.notify_all();
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
        throw std::invalid_argument(
            "record consumer timeout must be non-negative");
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
        invalid("durable completion found active sink submission");
    }
}

size_t RecordConsumer::pending_descriptors() const {
    std::lock_guard<std::mutex> lock(mu_);
    return descriptors_.size();
}

}  // namespace ring
