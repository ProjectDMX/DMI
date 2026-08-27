// FIFO descriptor-to-payload association at the backend-neutral sink boundary.

#pragma once

#include "record_descriptor.h"
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

class RecordConsumer {
public:
    explicit RecordConsumer(std::shared_ptr<RecordSink> sink);

    RecordConsumer(const RecordConsumer&) = delete;
    RecordConsumer& operator=(const RecordConsumer&) = delete;

    // Publish descriptors before the corresponding producer launch/replay.
    void push_descriptor(RecordDescriptor descriptor);
    void push_descriptors(std::vector<RecordDescriptor> descriptors);

    // Consume exactly one descriptor for one physical payload.  The payload
    // must be a contiguous CPU byte tensor containing the actual produced
    // bytes from its ready TaskEntry.
    void consume_payload(at::Tensor payload);

    // Latch an asynchronous worker failure.  The first failure is retained.
    void record_failure(std::exception_ptr failure) noexcept;

    // Checked durable-completion helpers.
    void rethrow_if_failed() const;
    bool wait_until_idle(std::chrono::milliseconds timeout) const;
    void finish() const;
    size_t pending_descriptors() const;

private:
    std::shared_ptr<RecordSink> sink_;

    mutable std::mutex mu_;
    mutable std::condition_variable idle_cv_;
    std::deque<RecordDescriptor> descriptors_;
    std::exception_ptr failure_;
    size_t active_payloads_{0};
};

}  // namespace ring
