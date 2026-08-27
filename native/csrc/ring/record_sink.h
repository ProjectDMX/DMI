// Backend-neutral ownership and completion boundary for generic records.

#pragma once

#include "record_descriptor.h"

#include <ATen/ATen.h>

#include <chrono>

namespace ring {

// One descriptor paired with its owned, contiguous host payload.  submit()
// transfers both objects to the sink; the ring never accesses them again.
struct RecordEnvelope {
    RecordDescriptor descriptor;
    at::Tensor payload;
};

// Native sinks may enqueue work asynchronously, but they must expose one
// checked, non-closing durability barrier.  The exact durability boundary is
// a property of the concrete sink (for example, ClickHouse acknowledgement,
// local spool commit, or remote object-store commit).
//
// The ring invokes submit() from its single record p2p worker.  It invokes
// flush_and_wait() only after every earlier submit() has returned, so a sink
// owns any additional internal concurrency and backpressure policy.
class RecordSink {
public:
    using Duration = std::chrono::milliseconds;

    virtual ~RecordSink() = default;

    virtual void submit(RecordEnvelope envelope) = 0;
    virtual bool flush_and_wait(Duration timeout) = 0;
    virtual void rethrow_if_failed() const = 0;
};

}  // namespace ring
