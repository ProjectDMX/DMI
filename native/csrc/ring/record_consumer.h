// FIFO descriptor-to-payload association at the backend-neutral sink boundary.

#pragma once

#include "record_descriptor.h"
#include "record_failure_policy.h"
#include "record_sink.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ring {

struct RecordConsumerSnapshot {
    RecordFailurePolicy policy{RecordFailurePolicy::kRaiseAtProducer};
    bool failed{false};
    std::string failure;
    // Descriptors dropped by the latch: those still queued when it happened
    // plus every push after it.  kDisableCapture only.
    uint64_t discarded_descriptors{0};
    // Payloads not stored because of the latch: the one whose submission
    // failed, plus every payload delivered after it.  kDisableCapture only.
    uint64_t discarded_payloads{0};
};

class RecordConsumer {
public:
    explicit RecordConsumer(
        std::shared_ptr<RecordSink> sink,
        RecordFailurePolicy policy = RecordFailurePolicy::kRaiseAtProducer);

    RecordConsumer(const RecordConsumer&) = delete;
    RecordConsumer& operator=(const RecordConsumer&) = delete;

    // Publish descriptors before the corresponding producer launch/replay.
    void push_descriptor(RecordDescriptor descriptor);
    void push_descriptors(std::vector<RecordDescriptor> descriptors);

    // Consume exactly one descriptor for one physical payload.  The payload
    // must be a contiguous CPU byte tensor containing the actual produced
    // bytes from its ready publication.  Throws on failure only under
    // kRaiseAtProducer; under kDisableCapture it latches and returns.
    void consume_payload(at::Tensor payload);

    // Latch an asynchronous worker failure.  The first failure is retained.
    void record_failure(std::exception_ptr failure) noexcept;

    // Checked durable-completion helpers.
    void rethrow_if_failed() const;
    bool wait_until_idle(std::chrono::milliseconds timeout) const;
    void finish() const;
    size_t pending_descriptors() const;
    bool failed() const;
    RecordConsumerSnapshot snapshot() const;

private:
    // Retain the first failure; under kDisableCapture also drop the queued
    // descriptors.  Caller holds mu_.
    void latch_locked(std::exception_ptr failure);

    std::shared_ptr<RecordSink> sink_;
    const RecordFailurePolicy policy_;

    mutable std::mutex mu_;
    mutable std::condition_variable idle_cv_;
    std::deque<RecordDescriptor> descriptors_;
    std::exception_ptr failure_;
    size_t active_payloads_{0};
    uint64_t discarded_descriptors_{0};
    uint64_t discarded_payloads_{0};
};

}  // namespace ring
