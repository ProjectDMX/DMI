#include "record_consumer.h"

#include <ATen/ATen.h>

#include <stdexcept>
#include <string>
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
    // Every later payload is discarded by the latch itself.  Inert -- each
    // read of the count comes after a failure_ check -- but reset so it never
    // describes payloads the latch's own discard already accounts for.
    payloads_to_discard_ = 0;
}

void RecordConsumer::note_discarded_step_locked(uint64_t step) {
    if (last_discarded_step_ && *last_discarded_step_ == step) return;
    last_discarded_step_ = step;
    ++steps_with_discards_;
}

bool RecordConsumer::take_unwanted_payload_locked() {
    if (failure_) {
        // Under kRaiseAtProducer consume_payload raises instead.
        if (policy_ != RecordFailurePolicy::kDisableCapture) return false;
        // Capture is off: the drain still delivers what the forward
        // produced, and the payload is dropped here, never submitted.
        ++discarded_payloads_;
        return true;
    }
    if (payloads_to_discard_ == 0) return false;
    // A skipped step's record: its descriptor was dropped by a discard
    // window.
    --payloads_to_discard_;
    ++discarded_payloads_;
    if (payloads_to_discard_ == 0) idle_cv_.notify_all();
    return true;
}

bool RecordConsumer::discard_next_payload_if_unwanted() {
    std::lock_guard<std::mutex> lock(mu_);
    return take_unwanted_payload_locked();
}

void RecordConsumer::begin_step() {
    std::lock_guard<std::mutex> lock(mu_);
    ++step_;
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
    if (discarding_) {
        ++discarded_descriptors_;
        ++payloads_to_discard_;
        note_discarded_step_locked(step_);
        return;
    }
    descriptors_.push_back({std::move(descriptor), step_});
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
    if (discarding_) {
        discarded_descriptors_ += descriptors.size();
        payloads_to_discard_ += descriptors.size();
        if (!descriptors.empty()) note_discarded_step_locked(step_);
        return;
    }
    for (auto& descriptor : descriptors) {
        descriptors_.push_back({std::move(descriptor), step_});
    }
}

void RecordConsumer::consume_payload(at::Tensor payload) {
    const bool disable = policy_ == RecordFailurePolicy::kDisableCapture;
    RecordDescriptor descriptor;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_ && !disable) std::rethrow_exception(failure_);
        if (take_unwanted_payload_locked()) return;
        if (descriptors_.empty()) {
            latch_locked(std::make_exception_ptr(std::runtime_error(
                "record consumer: physical payload arrived without an encoded descriptor")));
            if (!disable) std::rethrow_exception(failure_);
            ++discarded_payloads_;
            idle_cv_.notify_all();
            return;
        }
        descriptor = std::move(descriptors_.front().descriptor);
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
        return failure_ || (descriptors_.empty() && active_payloads_ == 0 &&
                            payloads_to_discard_ == 0);
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
    if (!descriptors_.empty() || payloads_to_discard_ != 0) {
        invalid("durable completion found " +
                std::to_string(descriptors_.size()) +
                " leftover encoded descriptors and " +
                std::to_string(payloads_to_discard_) +
                " payloads still owed to a discard window");
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
    snapshot.steps_with_discards = steps_with_discards_;
    return snapshot;
}

void RecordConsumer::begin_discard_window() {
    std::lock_guard<std::mutex> lock(mu_);
    discarding_ = true;
    for (const QueuedDescriptor& queued : descriptors_) {
        note_discarded_step_locked(queued.step);
    }
    discarded_descriptors_ += descriptors_.size();
    payloads_to_discard_ += descriptors_.size();
    descriptors_.clear();
}

void RecordConsumer::end_discard_window() {
    std::lock_guard<std::mutex> lock(mu_);
    discarding_ = false;
}

}  // namespace ring
