// Adapter from backend-neutral record envelopes to ClickHouse host rows.

#pragma once

#include "dmx_host_utils.h"
#include "ring/record_sink.h"

#include <functional>

namespace dmx_host {

class ClickHouseRecordSink final : public ring::RecordSink {
public:
    using SubmitRowFn =
        std::function<void(GenericRecordRow, std::uint64_t)>;
    using FlushFn = std::function<bool(Duration)>;
    using RethrowFn = std::function<void()>;

    ClickHouseRecordSink(SubmitRowFn submit_row,
                         FlushFn flush,
                         RethrowFn rethrow);

    void submit(ring::RecordEnvelope envelope) override;
    bool flush_and_wait(Duration timeout) override;
    void rethrow_if_failed() const override;

private:
    SubmitRowFn submit_row_;
    FlushFn flush_;
    RethrowFn rethrow_;
};

}  // namespace dmx_host
