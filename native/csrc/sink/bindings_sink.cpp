// Standalone torch-CPU test module for the native capture sink.
//
// _dmi_native_sink exposes NativePackSink (a ring::RecordSink) with a
// synthetic-envelope entry point so the pytest suite can drive the full
// adapter path — descriptor validation, ATen dtype mapping, slicing,
// submission, flush — with torch CPU tensors and no ring engine, CUDA, or
// ClickHouse. Production wiring (create_record_runtime) takes the same
// object through the real engine lease.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <chrono>
#include <memory>
#include <string>

#include "../ring/record_sink.h"
#include "native_pack_sink.h"
#include "pack_sink.h"

namespace py = pybind11;

namespace {

ring::PayloadSlice ParseSlice(const py::dict& row) {
  ring::PayloadSlice slice;
  slice.offset_bytes = row["offset"].cast<uint64_t>();
  if (!row["length"].is_none()) {
    slice.length_bytes = row["length"].cast<uint64_t>();
  }
  slice.materialization = ring::PayloadMaterialization::TENSOR;
  slice.dtype = row["dtype"].cast<int32_t>();
  slice.logical_shape = row["shape"].cast<std::vector<int64_t>>();
  slice.inferred_dynamic_dim = -1;
  return slice;
}

}  // namespace

namespace {

// Whether the main _native_backend extension has registered ring::RecordSink
// and ring::RecordSinkLease in pybind11's shared (cross-module) registry.
//
// Both extensions are built against the same pybind11 (torch's), so they
// share one internals table, and a C++ type may be registered in it ONCE.
// The engine's contract is with the MAIN module's types: create_record_runtime
// checks `isinstance(record_sink, _native_backend.RecordSink)` and then
// calls the inherited `_acquire_engine()`. So this module must not
// register its own RecordSink -- neither globally (the "already registered"
// ImportError when both load) nor module-locally (imports fine, but
// NativePackSink then derives from a DIFFERENT Python class: isinstance is
// False, `_acquire_engine` is absent, and the engine refuses the sink with
// "record_sink must be a native RecordSink"). It has to derive from the
// main module's registration, which pybind11 resolves through the shared
// registry as long as the main module has been imported first.
//
// The Python loader (native_sink.py) imports the main backend before this
// module wherever it is built. Here, the base registration is looked up
// directly: if the main module has not been loaded yet, it is imported by
// module name when importable, and only when it is not available at all
// (the torch-CPU test hosts, which build this module without CUDA) does
// this module fall back to module-local stand-ins. In that fallback there
// is no engine to attach to, so the type relationship is moot.
bool RingTypesRegistered() {
  return py::detail::get_type_info(typeid(ring::RecordSink)) != nullptr &&
         py::detail::get_type_info(typeid(ring::RecordSinkLease)) != nullptr;
}

void EnsureRingTypes(py::module_& m) {
  if (RingTypesRegistered()) return;
  // The main backend lives beside the dmi package and is loaded from its
  // file path by dmi.transport.native, never by bare module name. Load it
  // through that loader so that a plain `import _dmi_native_sink` (the
  // pytest suites import the built .so directly, at collection, before any
  // engine exists) still derives from the real RecordSink on a host that
  // has the backend -- otherwise whichever module initialised first in the
  // process would decide the type relationship for every later attachment.
  try {
    py::module_::import("dmi.transport.native").attr("_load_extension")();
  } catch (const py::error_already_set&) {
    // No dmi package on sys.path, or no full backend built: a host without
    // an engine to attach to.
  }
  if (RingTypesRegistered()) return;
  py::class_<ring::RecordSinkLease, std::shared_ptr<ring::RecordSinkLease>>(
      m, "RecordSinkLease", py::module_local());
  py::class_<ring::RecordSink, std::shared_ptr<ring::RecordSink>>(
      m, "RecordSink", py::module_local())
      .def("_acquire_engine",
           [](std::shared_ptr<ring::RecordSink> sink) {
             return ring::RecordSinkLease::acquire(std::move(sink));
           });
  m.attr("RING_TYPES_ARE_STANDINS") = true;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RING_TYPES_ARE_STANDINS") = false;
  EnsureRingTypes(m);

  // Derives from ring::RecordSink AS REGISTERED BY THE MAIN MODULE (see
  // EnsureRingTypes): NativePackSink is then a Python subclass of
  // _native_backend.RecordSink, passes the engine's isinstance check and
  // inherits its `_acquire_engine`. Dropping the base here, or registering
  // a second RecordSink, breaks the attachment even though the module
  // imports.
  py::class_<dmi_sink::NativePackSink, ring::RecordSink,
             std::shared_ptr<dmi_sink::NativePackSink>>(
      m, "NativePackSink")
      .def(py::init([](const std::string& spool_root,
                       const std::string& layout, int num_workers,
                       uint64_t max_queue_records, uint64_t max_queue_bytes,
                       uint64_t max_pack_bytes, uint64_t max_pack_records,
                       uint64_t max_linger_ns, uint64_t spool_max_bytes) {
             dmi_sink::SinkConfig config;
             config.spool_root = spool_root;
             config.spool_max_bytes = spool_max_bytes;
             config.num_workers = num_workers;
             config.max_queue_records = max_queue_records;
             config.max_queue_bytes = max_queue_bytes;
             config.max_pack_bytes = max_pack_bytes;
             config.max_pack_records = max_pack_records;
             config.max_linger_ns = max_linger_ns;
             auto sink = std::make_unique<dmi_sink::PackSink>(config);
             return std::make_shared<dmi_sink::NativePackSink>(
                 std::move(sink), layout);
           }),
           py::arg("spool_root"), py::arg("layout"),
           py::arg("num_workers") = 1,
           py::arg("max_queue_records") = 256,
           py::arg("max_queue_bytes") = 16 * 1024 * 1024,
           py::arg("max_pack_bytes") = 128ull * 1024 * 1024,
           py::arg("max_pack_records") = 10'000,
           py::arg("max_linger_ns") = 1'000'000'000,
           py::arg("spool_max_bytes") = 1ull << 40)
      .def("attach",
           [](std::shared_ptr<dmi_sink::NativePackSink> self) {
             // Simulates engine ownership for tests (the real engine takes
             // the lease in create_record_runtime). Keep the lease alive.
             return ring::RecordSinkLease::acquire(self);
           })
      .def("submit_envelope",
           [](dmi_sink::NativePackSink& self, const std::string& layout,
              const py::list& rows, const torch::Tensor& payload) {
             ring::RecordEnvelope envelope;
             envelope.descriptor.layout = layout;
             envelope.payload = payload;
             for (const auto& item : rows) {
               const py::dict row = item.cast<py::dict>();
               ring::EncodedRecordRow record_row;
               record_row.cells.emplace_back(
                   row["metadata_json"].cast<std::string>());
               record_row.cells.emplace_back(ParseSlice(row));
               envelope.descriptor.rows.push_back(std::move(record_row));
             }
             self.submit(std::move(envelope));
           })
      .def("flush_and_wait",
           [](dmi_sink::NativePackSink& self, double timeout_s) {
             return self.flush_and_wait(
                 std::chrono::duration_cast<ring::RecordSink::Duration>(
                     std::chrono::duration<double>(timeout_s)));
           })
      .def("rethrow_if_failed", &dmi_sink::NativePackSink::rethrow_if_failed)
      .def_property_readonly("layout", &dmi_sink::NativePackSink::layout)
      .def("snapshot", [](dmi_sink::NativePackSink& self) {
        const dmi_sink::SinkSnapshot snapshot = self.sink().Snapshot();
        py::dict out;
        out["submitted_records"] = snapshot.submitted_records;
        out["admitted_records"] = snapshot.admitted_records;
        out["persisted_records"] = snapshot.persisted_records;
        out["packs_persisted"] = snapshot.packs_persisted;
        out["dropped_records"] = snapshot.dropped_records;
        out["duplicate_records"] = snapshot.duplicate_records;
        out["oversized_records"] = snapshot.oversized_records;
        out["failures"] = snapshot.failures;
        return out;
      });
}
