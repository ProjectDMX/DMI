#include "record_consumer.h"

#include <ATen/ATen.h>

#include <stdexcept>
#include <utility>

namespace ring {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("record consumer: " + message);
}

std::string describe_failure(const std::exception_ptr& failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown record failure";
    }
}

}  // namespace

RecordConsumer::RecordConsumer(
    std::shared_ptr<RecordSink> sink, RecordFailurePolicy policy)
    : sink_(std::move(sink)), policy_(policy) {}

void RecordConsumer::latch_locked(std::exception_ptr failure) {
    if (!failure_) failure_ = std::move(failure);
    if (policy_ != RecordFailurePolicy::kDisableCapture) return;
    // Their payloads will be discarded on arrival, so no descriptor queued
    // at the latch can ever be paired again.
    discarded_descriptors_ += descriptors_.size();
    descriptors_.clear();
}

void RecordConsumer::push_descriptor(RecordDescriptor descriptor) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) {
        if (policy_ != RecordFailurePolicy::kDisableCapture) {
            std::rethrow_exception(failure_);
        }
        ++discarded_descriptors_;
        return;
    }
    descriptors_.push_back(std::move(descriptor));
}

void RecordConsumer::push_descriptors(
    std::vector<RecordDescriptor> descriptors) {
    std::lock_guard<std::mutex> lock(mu_);
    if (failure_) {
        if (policy_ != RecordFailurePolicy::kDisableCapture) {
            std::rethrow_exception(failure_);
        }
        discarded_descriptors_ += descriptors.size();
        return;
    }
    for (auto& descriptor : descriptors) {
        descriptors_.push_back(std::move(descriptor));
    }
}

void RecordConsumer::consume_payload(at::Tensor payload) {
    const bool disable = policy_ == RecordFailurePolicy::kDisableCapture;
    RecordDescriptor descriptor;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_) {
            if (!disable) std::rethrow_exception(failure_);
            // Capture is off: the drain still delivers what the forward
            // produced, and the payload is dropped here, never submitted.
            ++discarded_payloads_;
            return;
        }
        if (descriptors_.empty()) {
            latch_locked(std::make_exception_ptr(std::runtime_error(
                "record consumer: physical payload arrived without an encoded descriptor")));
            if (!disable) std::rethrow_exception(failure_);
            ++discarded_payloads_;
            idle_cv_.notify_all();
            return;
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
            latch_locked(failure);
            --active_payloads_;
            if (disable) ++discarded_payloads_;
        }
        idle_cv_.notify_all();
        // Under kDisableCapture the refusal stops capture, not the worker:
        // it is reported at flush, in the snapshot and at close.
        if (!disable) throw;
    }
}

void RecordConsumer::record_failure(std::exception_ptr failure) noexcept {
    if (!failure) return;
    try {
        std::lock_guard<std::mutex> lock(mu_);
        latch_locked(std::move(failure));
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

bool RecordConsumer::failed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<bool>(failure_);
}

RecordConsumerSnapshot RecordConsumer::snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    RecordConsumerSnapshot snapshot;
    snapshot.policy = policy_;
    snapshot.failed = static_cast<bool>(failure_);
    if (failure_) snapshot.failure = describe_failure(failure_);
    snapshot.discarded_descriptors = discarded_descriptors_;
    snapshot.discarded_payloads = discarded_payloads_;
    return snapshot;
}

}  // namespace ring
