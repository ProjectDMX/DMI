#include "reference_python_capture_sink.h"

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace dmi_capture {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("reference Python capture sink: " + message);
}

at::ScalarType checked_dtype(std::int32_t encoded) {
    if (encoded < 0 ||
        encoded >= static_cast<std::int32_t>(at::ScalarType::NumOptions)) {
        invalid("unsupported encoded dtype");
    }
    const auto dtype = static_cast<at::ScalarType>(encoded);
    if (at::elementSize(dtype) <= 0) invalid("dtype has no byte representation");
    return dtype;
}

std::uint64_t checked_elements(const std::vector<std::int64_t>& shape) {
    std::uint64_t elements = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension < 0) invalid("dynamic payload shapes are not supported");
        const auto value = static_cast<std::uint64_t>(dimension);
        if (value != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / value) {
            invalid("payload shape overflows uint64");
        }
        elements *= value;
    }
    return elements;
}

std::uint64_t declared_payload_bytes(const ring::PayloadSlice& slice) {
    if (slice.materialization != ring::PayloadMaterialization::TENSOR) {
        invalid("payload cell must use tensor materialization");
    }
    if (slice.inferred_dynamic_dim >= 0) {
        invalid("payload shape must be fully resolved before storage");
    }
    const at::ScalarType dtype = checked_dtype(slice.dtype);
    const auto element_bytes = static_cast<std::uint64_t>(at::elementSize(dtype));
    const std::uint64_t elements = checked_elements(slice.logical_shape);
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
        invalid("payload dtype and shape overflow uint64 bytes");
    }
    const std::uint64_t expected = elements * element_bytes;
    if (slice.length_bytes && *slice.length_bytes != expected) {
        invalid("payload dtype and shape do not match declared bytes");
    }
    return expected;
}

at::Tensor checked_payload_view(const at::Tensor& payload,
                                const ring::PayloadSlice& slice) {
    const std::uint64_t expected = declared_payload_bytes(slice);
    const auto payload_bytes = static_cast<std::uint64_t>(payload.numel());
    if (slice.offset_bytes > payload_bytes) {
        invalid("payload-slice offset exceeds physical payload");
    }
    const std::uint64_t available = payload_bytes - slice.offset_bytes;
    const std::uint64_t length = slice.length_bytes.value_or(available);
    if (length > available) invalid("payload slice exceeds physical payload");

    const at::ScalarType dtype = checked_dtype(slice.dtype);
    const auto element_bytes = static_cast<std::uint64_t>(at::elementSize(dtype));
    if (slice.offset_bytes % element_bytes != 0) {
        invalid("payload-slice offset is not dtype-aligned");
    }
    if (expected != length) {
        invalid("payload dtype and shape do not match physical bytes");
    }

    return payload
        .narrow(0, static_cast<std::int64_t>(slice.offset_bytes),
                static_cast<std::int64_t>(length))
        .view(dtype)
        .reshape(slice.logical_shape);
}

py::object target_method(PyObject* target, const char* name) {
    if (target == nullptr) invalid("Python target is unavailable");
    return py::reinterpret_borrow<py::object>(target).attr(name);
}

}  // namespace

ReferencePythonCaptureSink::ReferencePythonCaptureSink(
    PyObject* target, std::string layout)
    : target_(target), layout_(std::move(layout)) {
    if (target == nullptr || target == Py_None) {
        throw std::invalid_argument(
            "ReferencePythonCaptureSink requires a Python target");
    }
    if (layout_.empty()) {
        throw std::invalid_argument(
            "ReferencePythonCaptureSink requires a record layout");
    }
    Py_INCREF(target);
}

ReferencePythonCaptureSink::~ReferencePythonCaptureSink() {
    if (!Py_IsInitialized()) {
        // Interpreter teardown cannot safely run arbitrary decref callbacks.
        return;
    }
    const PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(target_);
    PyGILState_Release(state);
}

void ReferencePythonCaptureSink::submit(ring::RecordEnvelope envelope) {
    if (!engine_owned()) invalid("sink is not attached to a RingEngine");
    const auto& descriptor = envelope.descriptor;
    if (descriptor.layout != layout_) invalid("unexpected record layout");
    if (descriptor.rows.empty()) invalid("descriptor must contain at least one row");

    for (const auto& row : descriptor.rows) {
        const auto& cells = row.cells;
        if (cells.size() != 2) invalid("descriptor row must contain two cells");

        const auto* metadata_json = std::get_if<std::string>(&cells[0]);
        const auto* payload_slice = std::get_if<ring::PayloadSlice>(&cells[1]);
        if (metadata_json == nullptr || payload_slice == nullptr) {
            invalid(
                "descriptor requires metadata JSON followed by one payload slice");
        }
        at::Tensor typed_payload =
            checked_payload_view(envelope.payload, *payload_slice);

        py::gil_scoped_acquire gil;
        try {
            target_method(target_, "_submit_capture")(
                *metadata_json, std::move(typed_payload));
        } catch (const py::error_already_set& error) {
            // RecordConsumer stores submit failures in an exception_ptr and
            // may rethrow it on every checked flush.  A pybind exception can
            // only restore its Python error once, so latch a stable native
            // exception instead of carrying Python error state across threads.
            throw std::runtime_error(
                "reference Python capture callback failed: " +
                std::string(error.what()));
        }
    }
}

bool ReferencePythonCaptureSink::flush_and_wait(Duration timeout) {
    if (!engine_owned()) invalid("sink is not attached to a RingEngine");
    py::gil_scoped_acquire gil;
    const double timeout_s = std::chrono::duration<double>(timeout).count();
    return py::cast<bool>(
        target_method(target_, "_flush_capture")(timeout_s));
}

void ReferencePythonCaptureSink::rethrow_if_failed() const {
    if (!engine_owned()) invalid("sink is not attached to a RingEngine");
    py::gil_scoped_acquire gil;
    target_method(target_, "_rethrow_capture")();
}

void ReferencePythonCaptureSink::on_engine_acquire() {
    py::gil_scoped_acquire gil;
    target_method(target_, "_attach")();
}

void ReferencePythonCaptureSink::on_engine_release() noexcept {
    if (!Py_IsInitialized()) return;
    try {
        py::gil_scoped_acquire gil;
        target_method(target_, "_detach")();
    } catch (py::error_already_set& error) {
        error.discard_as_unraisable("ReferencePythonCaptureSink._detach");
    } catch (...) {
        // Engine teardown cannot leave a worker alive merely because an
        // observability-only detach callback failed.
    }
}

}  // namespace dmi_capture
