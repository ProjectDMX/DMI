// Explicit reference-only bridge from RecordEnvelope to Python capture packs.

#pragma once

#include "ring/record_sink.h"

#include <Python.h>

#include <string>

namespace dmi_capture {

class ReferencePythonCaptureSink final : public ring::RecordSink {
public:
    ReferencePythonCaptureSink(PyObject* target, std::string layout);
    ~ReferencePythonCaptureSink() override;

    ReferencePythonCaptureSink(const ReferencePythonCaptureSink&) = delete;
    ReferencePythonCaptureSink& operator=(
        const ReferencePythonCaptureSink&) = delete;

    void submit(ring::RecordEnvelope envelope) override;
    bool flush_and_wait(Duration timeout) override;
    void rethrow_if_failed() const override;

protected:
    void on_engine_acquire() override;
    void on_engine_release() noexcept override;

private:
    PyObject* target_;
    const std::string layout_;
};

}  // namespace dmi_capture
