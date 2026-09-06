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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ring::RecordSinkLease, std::shared_ptr<ring::RecordSinkLease>>(
      m, "RecordSinkLease");

  py::class_<dmi_sink::NativePackSink,
             std::shared_ptr<dmi_sink::NativePackSink>>(
      m, "NativePackSink")
      .def(py::init([](const std::string& spool_root,
                       const std::string& layout, int num_workers,
                       uint64_t max_queue_records, uint64_t max_queue_bytes,
                       uint64_t max_pack_bytes, uint64_t max_pack_records,
                       uint64_t max_linger_ns) {
             dmi_sink::SinkConfig config;
             config.spool_root = spool_root;
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
           py::arg("max_linger_ns") = 1'000'000'000)
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
