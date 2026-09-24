// Explicit reference-only bridge from RecordEnvelope to Python capture packs.

#pragma once

#include "ring/record_sink.h"

#include <Python.h>

#include <optional>
#include <string>

namespace dmi_capture {

class ReferencePythonCaptureSink final : public ring::RecordSink {
public:
    // `admission_bound` is what the target promises about its submit: the
    // longest one call can take. The bridge cannot know it, so it has none
    // unless the caller says so (tests that stand in for a bounded sink).
    ReferencePythonCaptureSink(
        PyObject* target, std::string layout,
        std::optional<Duration> admission_bound = std::nullopt);
    ~ReferencePythonCaptureSink() override;

    ReferencePythonCaptureSink(const ReferencePythonCaptureSink&) = delete;
    ReferencePythonCaptureSink& operator=(
        const ReferencePythonCaptureSink&) = delete;

    void submit(ring::RecordEnvelope envelope) override;
    bool flush_and_wait(Duration timeout) override;
    void rethrow_if_failed() const override;
    std::optional<Duration> admission_bound() const override {
        return admission_bound_;
    }

protected:
    void on_engine_acquire() override;
    void on_engine_release() noexcept override;

private:
    PyObject* target_;
    const std::string layout_;
    const std::optional<Duration> admission_bound_;
};

}  // namespace dmi_capture
