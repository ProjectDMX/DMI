// Explicit reference-only bridge from RecordEnvelope to Python capture packs.

#pragma once

#include "ring/record_sink.h"

#include <Python.h>

#include <atomic>
#include <mutex>

namespace dmi_capture {

class ReferencePythonCaptureSink final : public ring::RecordSink {
public:
    explicit ReferencePythonCaptureSink(PyObject* target);
    ~ReferencePythonCaptureSink() override;

    ReferencePythonCaptureSink(const ReferencePythonCaptureSink&) = delete;
    ReferencePythonCaptureSink& operator=(
        const ReferencePythonCaptureSink&) = delete;

    void submit(ring::RecordEnvelope envelope) override;
    bool flush_and_wait(Duration timeout) override;
    void rethrow_if_failed() const override;

    void attach_target();
    void detach_target();
    void release_target();
    bool attached() const noexcept { return attached_.load(); }

private:
    std::atomic<PyObject*> target_{nullptr};
    std::atomic<bool> attached_{false};
    mutable std::mutex lifecycle_mu_;
};

}  // namespace dmi_capture
