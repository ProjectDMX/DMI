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
#include <optional>
#include <string>
#include <vector>

namespace ring {

struct RecordConsumerSnapshot {
    RecordFailurePolicy policy{RecordFailurePolicy::kRaiseAtProducer};
    bool failed{false};
    std::string failure;
    // Descriptors dropped without being stored: by a kDisableCapture latch
    // (those still queued when it happened, plus every push after it) and,
    // under either policy, by a discard window (a skipped step).
    uint64_t discarded_descriptors{0};
    // Payloads not stored: under kDisableCapture the one whose submission
    // failed and every payload delivered after the latch; under either
    // policy the payloads of the descriptors a discard window dropped.
    uint64_t discarded_payloads{0};
    // Distinct steps (see begin_step) that lost at least one record to a
    // discard window.  A window drops every record still queued, from any
    // step, so this can exceed the number of windows.  A latch's discards
    // are not counted here.
    uint64_t steps_with_discards{0};
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

    // Under kDisableCapture after a latch, or while payloads owed to a
    // discard window remain: account the next payload as discarded, exactly
    // as consume_payload would, and return true, so the caller can skip
    // copying bytes nobody will store.  Otherwise change nothing and return
    // false; the caller then delivers the payload to consume_payload.
    bool discard_next_payload_if_unwanted();

    // Start a new step: descriptors pushed from now on are tagged with it,
    // so steps_with_discards can count the steps a discard window reaches.
    void begin_step();

    // Skip the rest of a step without latching (a spent stall budget).
    // begin_discard_window() drops every descriptor still queued -- the
    // step's own and those of earlier steps whose records the sink has not
    // taken yet, which the stall bound needs out of the drain's way -- and
    // every descriptor pushed until end_discard_window() is dropped on
    // arrival.  The payloads of all of them are discarded as the drain
    // delivers them, so the ring's descriptor/payload pairing is kept.
    // Counted in discarded_* and steps_with_discards.  A payload already in
    // the sink is not recalled.
    void begin_discard_window();
    void end_discard_window();

private:
    // A queued descriptor and the step it was pushed in.
    struct QueuedDescriptor {
        RecordDescriptor descriptor;
        uint64_t step{0};
    };

    // Retain the first failure; under kDisableCapture also drop the queued
    // descriptors.  Caller holds mu_.
    void latch_locked(std::exception_ptr failure);
    // Count `step` in steps_with_discards_ unless it already is.  Windows
    // see steps in non-decreasing order (FIFO, and a window drops all of
    // its own step's pushes), so the last counted step is enough.  Caller
    // holds mu_.
    void note_discarded_step_locked(uint64_t step);
    // The discard branches shared by consume_payload and
    // discard_next_payload_if_unwanted; false under kRaiseAtProducer after
    // a latch.  Caller holds mu_.
    bool take_unwanted_payload_locked();

    std::shared_ptr<RecordSink> sink_;
    const RecordFailurePolicy policy_;

    mutable std::mutex mu_;
    mutable std::condition_variable idle_cv_;
    std::deque<QueuedDescriptor> descriptors_;
    uint64_t step_{0};
    std::exception_ptr failure_;
    size_t active_payloads_{0};
    // Discard window: while open, pushes are dropped on arrival.  Every
    // dropped descriptor still owes the consumer one payload, and those
    // payloads arrive before any descriptor queued after the window
    // (FIFO), so a count is enough to keep the pairing.
    bool discarding_{false};
    uint64_t payloads_to_discard_{0};
    uint64_t discarded_descriptors_{0};
    uint64_t discarded_payloads_{0};
    uint64_t steps_with_discards_{0};
    std::optional<uint64_t> last_discarded_step_;
};

}  // namespace ring
