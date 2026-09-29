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
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "../ring/record_sink.h"
#include "native_pack_sink.h"
#include "pack_sink.h"

namespace py = pybind11;

namespace {

// Parks every stage of a sink's spool until Open(), or for max_hold at
// most, so a test can wedge the sink's pipeline without a hung filesystem
// -- and still gets it back if what it tests never lets go.
struct StageGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool open = false;
  std::chrono::nanoseconds max_hold{0};

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait_for(lock, max_hold, [this] { return open; });
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      open = true;
    }
    cv.notify_all();
  }
};

ring::PayloadSlice ParseSlice(const py::dict& row) {
  ring::PayloadSlice slice;
  slice.offset_bytes = row["offset"].cast<uint64_t>();
  if (!row["length"].is_none()) {
    slice.length_bytes = row["length"].cast<uint64_t>();
  }
  slice.materialization = ring::PayloadMaterialization::TENSOR;
  slice.dtype = row["dtype"].cast<int32_t>();
  slice.logical_shape = row["shape"].cast<std::vector<int64_t>>();
  // Optional, default -1 ("no dynamic dimension"). The production encoder
  // (bindings.cpp) sets this from the -1 it finds in the Python slice's
  // shape, so pinning it here to -1 left the whole dynamic-dim branch of
  // the adapter unreachable from this driver. A caller drives it the way
  // production presents it: `shape=[-1, 4], inferred_dynamic_dim=0`.
  slice.inferred_dynamic_dim = -1;
  if (row.contains("inferred_dynamic_dim") &&
      !row["inferred_dynamic_dim"].is_none()) {
    slice.inferred_dynamic_dim = row["inferred_dynamic_dim"].cast<int32_t>();
  }
  return slice;
}

dmi_sink::Overload ParseOverload(const std::string& name) {
  if (name == "block") return dmi_sink::Overload::kBlock;
  if (name == "drop_newest") return dmi_sink::Overload::kDropNewest;
  throw py::value_error("overload must be 'block' or 'drop_newest', got '" +
                        name + "'");
}

const char* OverloadName(dmi_sink::Overload overload) {
  return overload == dmi_sink::Overload::kBlock ? "block" : "drop_newest";
}

// None waits without bound under block, as SinkConfig's -1 does.
double ParseAdmissionTimeout(const std::optional<double>& timeout_s) {
  if (!timeout_s.has_value()) return -1.0;
  if (!std::isfinite(*timeout_s) || *timeout_s < 0.0) {
    throw py::value_error(
        "admission_timeout_s must be None or a finite, non-negative number");
  }
  return *timeout_s;
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

// RecordSink.admission_bound_s: the sink's admission bound in seconds, or
// None when it has none.
std::optional<double> AdmissionBoundSeconds(const ring::RecordSink& sink) {
  const auto bound = sink.admission_bound();
  if (!bound) return std::nullopt;
  return std::chrono::duration<double>(*bound).count();
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
           })
      .def_property_readonly("admission_bound_s", &AdmissionBoundSeconds);
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
                       uint64_t max_linger_ns, uint64_t spool_max_bytes,
                       const std::string& overload,
                       std::optional<double> admission_timeout_s,
                       double release_flush_timeout_s,
                       const std::string& owner_lock,
                       bool allow_shared_filesystem,
                       bool charge_dead_siblings) {
             dmi_sink::SinkConfig config;
             if (!dmi_store::ParseOwnerLock(owner_lock,
                                            &config.spool_owner_lock)) {
               throw py::value_error(
                   "owner_lock must be 'take' or 'held_by_caller', got '" +
                   owner_lock + "'");
             }
             config.spool_allow_shared_filesystem = allow_shared_filesystem;
             config.spool_charge_dead_siblings = charge_dead_siblings;
             config.overload = ParseOverload(overload);
             config.admission_timeout_s =
                 ParseAdmissionTimeout(admission_timeout_s);
             config.spool_root = spool_root;
             config.spool_max_bytes = spool_max_bytes;
             config.num_workers = num_workers;
             config.max_queue_records = max_queue_records;
             config.max_queue_bytes = max_queue_bytes;
             config.max_pack_bytes = max_pack_bytes;
             config.max_pack_records = max_pack_records;
             config.max_linger_ns = max_linger_ns;
             if (!std::isfinite(release_flush_timeout_s) ||
                 release_flush_timeout_s < 0.0) {
               throw py::value_error(
                   "release_flush_timeout_s must be a finite, non-negative "
                   "number");
             }
             auto sink = std::make_unique<dmi_sink::PackSink>(config);
             return std::make_shared<dmi_sink::NativePackSink>(
                 std::move(sink), layout,
                 std::chrono::ceil<ring::RecordSink::Duration>(
                     std::chrono::duration<double>(release_flush_timeout_s)));
           }),
           py::arg("spool_root"), py::arg("layout"),
           py::arg("num_workers") = 1,
           py::arg("max_queue_records") = 256,
           py::arg("max_queue_bytes") = 16 * 1024 * 1024,
           py::arg("max_pack_bytes") = 128ull * 1024 * 1024,
           py::arg("max_pack_records") = 10'000,
           py::arg("max_linger_ns") = 1'000'000'000,
           py::arg("spool_max_bytes") = 1ull << 40,
           // SinkConfig's own defaults: the Python NativeSinkConfig, which
           // the ring-fed sink is built from, picks block with 2 s.
           py::arg("overload") = "drop_newest",
           py::arg("admission_timeout_s") = py::none(),
           // How long releasing the sink from its engine waits for the open
           // pack to reach the spool; 0 turns that flush off.
           py::arg("release_flush_timeout_s") =
               std::chrono::duration<double>(
                   dmi_sink::NativePackSink::kDefaultReleaseFlushTimeout)
                   .count(),
           // The spool directory's owner lock (store/spool.h): "take" owns
           // it for the sink's life; "held_by_caller" when the caller holds
           // a SpoolOwnerLock on it, as the engine does around its sink and
           // storage service.
           py::arg("owner_lock") = "take",
           py::arg("allow_shared_filesystem") = false,
           // spool_root is a rank directory of the spool layout, and the
           // dead incarnations' packs beside it count against
           // spool_max_bytes, as the engine's claimed directory does.
           py::arg("charge_dead_siblings") = false)
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
      // Test seam, not API: from now on each stage of the sink's spool
      // waits, before it writes anything, until the returned function is
      // called or max_hold_s has passed. Wedges the pipeline the way a hung
      // filesystem would (tests/test_native_sink_release.py).
      .def("_hold_stages_for_testing",
           [](dmi_sink::NativePackSink& self, double max_hold_s) {
             if (!(max_hold_s > 0.0 && max_hold_s <= 3600.0)) {
               throw py::value_error("max_hold_s must be in (0, 3600]");
             }
             auto gate = std::make_shared<StageGate>();
             gate->max_hold =
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::duration<double>(max_hold_s));
             self.sink_for_testing().SpoolForTesting().SetStageHookForTesting(
                 [gate] { gate->Wait(); });
             return py::cpp_function([gate] { gate->Open(); });
           },
           py::arg("max_hold_s"))
      .def_property_readonly("layout", &dmi_sink::NativePackSink::layout)
      .def_property_readonly(
          "overload",
          [](const dmi_sink::NativePackSink& self) {
            return std::string(OverloadName(self.sink().config().overload));
          })
      .def_property_readonly(
          "release_flush_timeout_s",
          [](const dmi_sink::NativePackSink& self) {
            return std::chrono::duration<double>(self.release_flush_timeout())
                .count();
          })
      .def_property_readonly(
          "admission_timeout_s",
          [](const dmi_sink::NativePackSink& self) -> std::optional<double> {
            const double timeout_s = self.sink().config().admission_timeout_s;
            if (timeout_s < 0) return std::nullopt;
            return timeout_s;
          })
      .def("snapshot", [](dmi_sink::NativePackSink& self) {
        const dmi_sink::SinkSnapshot snapshot = self.sink().Snapshot();
        py::dict out;
        out["submitted_records"] = snapshot.submitted_records;
        out["admitted_records"] = snapshot.admitted_records;
        out["persisted_records"] = snapshot.persisted_records;
        out["packs_persisted"] = snapshot.packs_persisted;
        out["dropped_records"] = snapshot.dropped_records;
        out["timed_out_records"] = snapshot.timed_out_records;
        out["rejected_closed_records"] = snapshot.rejected_closed_records;
        out["duplicate_records"] = snapshot.duplicate_records;
        out["oversized_records"] = snapshot.oversized_records;
        out["failures"] = snapshot.failures;
        return out;
      });
}
