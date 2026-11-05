// Pybind11 module — ring transport + ClickHouse pipeline bindings

#ifdef DMI_HOST_ONLY
#include <torch/csrc/utils/pybind.h>
#else
#include <torch/extension.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <climits>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
namespace py = pybind11;

#include "clickhouse_client.h"
#include "dmx_host_engine.h"
#ifndef DMI_HOST_ONLY
#include "clickhouse_record_sink.h"
#include "reference_python_capture_sink.h"
#include "ring/batch_queue.h"
#include "ring/capture_lease.h"
#include "ring/drop_notice.h"
#include "ring/ring_engine_py.h"
#include "ring/ring_torch_op.h"
#include "ring/slice_queue.h"
#include "ring/tensor_meta.h"
#endif

namespace {

std::string EnumValueString(const py::handle& value) {
  if (py::hasattr(value, "value")) {
    return py::cast<std::string>(value.attr("value"));
  }
  return py::cast<std::string>(value);
}

dmx_host::RecordCellType ParseRecordCellType(const py::handle& value) {
  const std::string name = EnumValueString(value);
  if (name == "string") return dmx_host::RecordCellType::STRING;
  if (name == "int32") return dmx_host::RecordCellType::INT32;
  if (name == "int64") return dmx_host::RecordCellType::INT64;
  if (name == "float64") return dmx_host::RecordCellType::FLOAT64;
  if (name == "int64_array") return dmx_host::RecordCellType::INT64_ARRAY;
  if (name == "tensor") return dmx_host::RecordCellType::TENSOR;
  throw py::value_error("unsupported RecordCellType: " + name);
}

std::string OptionalStringAttr(const py::handle& object, const char* name) {
  py::object value = object.attr(name);
  return value.is_none() ? std::string{} : py::cast<std::string>(value);
}

dmx_host::RecordSchema CopyRecordSchema(const py::handle& schema_py) {
  dmx_host::RecordSchema schema;
  schema.index_granularity =
      py::cast<int>(schema_py.attr("index_granularity"));
  for (const py::handle layout_py : schema_py.attr("layouts")) {
    dmx_host::RecordLayout layout;
    layout.name = py::cast<std::string>(layout_py.attr("name"));
    layout.table = py::cast<std::string>(layout_py.attr("table"));
    layout.primary_key = py::cast<std::vector<std::string>>(
        layout_py.attr("primary_key"));
    layout.order_by = py::cast<std::vector<std::string>>(
        layout_py.attr("order_by"));
    for (const py::handle column_py : layout_py.attr("columns")) {
      dmx_host::RecordColumn column;
      column.name = py::cast<std::string>(column_py.attr("name"));
      column.type = ParseRecordCellType(column_py.attr("type"));
      column.dtype_column = OptionalStringAttr(column_py, "dtype_column");
      column.shape_column = OptionalStringAttr(column_py, "shape_column");
      column.bytes_column = OptionalStringAttr(column_py, "bytes_column");
      layout.columns.push_back(std::move(column));
    }
    schema.layouts.push_back(std::move(layout));
  }
  dmx_host::ValidateRecordSchema(schema);
  return schema;
}

dmx_host::RecordValue CopyLiteralRecordValue(
    const py::handle& value, dmx_host::RecordCellType type) {
  switch (type) {
    case dmx_host::RecordCellType::STRING:
      return py::cast<std::string>(value);
    case dmx_host::RecordCellType::INT32: {
      const int64_t parsed = py::cast<int64_t>(value);
      if (parsed < std::numeric_limits<int32_t>::min() ||
          parsed > std::numeric_limits<int32_t>::max()) {
        throw py::value_error("record INT32 value is out of range");
      }
      return static_cast<int32_t>(parsed);
    }
    case dmx_host::RecordCellType::INT64:
      return py::cast<int64_t>(value);
    case dmx_host::RecordCellType::FLOAT64:
      return py::cast<double>(value);
    case dmx_host::RecordCellType::INT64_ARRAY:
      return py::cast<std::vector<int64_t>>(value);
    case dmx_host::RecordCellType::TENSOR:
      return py::cast<at::Tensor>(value);
  }
  throw py::value_error("unsupported record cell type");
}

#ifndef DMI_HOST_ONLY
ring::PayloadMaterialization ParsePayloadMaterialization(
    const py::handle& slice_py, dmx_host::RecordCellType column_type) {
  const int storage = py::cast<int>(slice_py.attr("storage"));
  if (column_type == dmx_host::RecordCellType::TENSOR && storage == 0)
    return ring::PayloadMaterialization::TENSOR;
  if (column_type == dmx_host::RecordCellType::FLOAT64 && storage == 1)
    return ring::PayloadMaterialization::FLOAT_SCALAR;
  if (column_type == dmx_host::RecordCellType::INT64 && storage == 2)
    return ring::PayloadMaterialization::INT_SCALAR;
  throw py::type_error(
      "PayloadSlice storage does not match its record column type");
}

bool IsPayloadSlice(const py::handle& value) {
  return py::hasattr(value, "offset_bytes") &&
         py::hasattr(value, "nbytes") &&
         py::hasattr(value, "storage") &&
         py::hasattr(value, "dtype") &&
         py::hasattr(value, "shape");
}

std::shared_ptr<dmx_host::ClickHouseRecordSink> MakeClickHouseRecordSink(
    std::shared_ptr<dmx_host::DMXHostEngine> host) {
  if (!host) {
    throw std::invalid_argument("ClickHouse record sink requires a host engine");
  }
  return std::make_shared<dmx_host::ClickHouseRecordSink>(
      [host](dmx_host::GenericRecordRow row, uint64_t payload_bytes) {
        host->submit_record(std::move(row), payload_bytes);
      },
      [host](ring::RecordSink::Duration timeout) {
        const double timeout_s =
            std::chrono::duration<double>(timeout).count();
        return host->flush_and_wait(
            dmx_host::DMXHostEngine::Duration(timeout_s));
      },
      [host] { host->raise_if_failed(); });
}

// RecordSink.admission_bound_s: the sink's admission bound in seconds, or
// None when it has none.
std::optional<double> AdmissionBoundSeconds(const ring::RecordSink& sink) {
  const auto bound = sink.admission_bound();
  if (!bound) return std::nullopt;
  return std::chrono::duration<double>(*bound).count();
}

template <typename... Args>
std::shared_ptr<ring_py::RingEnginePy> MakeRingEngine(Args&&... args) {
  // Ring destruction may join a worker that is completing a callback.  Never
  // hold the Python GIL across that join, regardless of the concrete sink.
  return std::shared_ptr<ring_py::RingEnginePy>(
      new ring_py::RingEnginePy(std::forward<Args>(args)...),
      [](ring_py::RingEnginePy* engine) {
        if (Py_IsInitialized() && PyGILState_Check()) {
          py::gil_scoped_release release;
          delete engine;
        } else {
          delete engine;
        }
      });
}

ring::RecordFailurePolicy ParseRecordFailurePolicy(const std::string& name) {
  if (name == "raise") return ring::RecordFailurePolicy::kRaiseAtProducer;
  if (name == "disable_capture") {
    return ring::RecordFailurePolicy::kDisableCapture;
  }
  throw py::value_error(
      "failure_policy must be 'raise' or 'disable_capture', got '" + name +
      "'");
}

const char* RecordFailurePolicyName(ring::RecordFailurePolicy policy) {
  return policy == ring::RecordFailurePolicy::kDisableCapture
             ? "disable_capture"
             : "raise";
}

ring::RecordDescriptor CopyRecordDescriptor(
    const py::handle& descriptor_py,
    const dmx_host::RecordSchema& schema) {
  ring::RecordDescriptor descriptor;
  descriptor.layout = py::cast<std::string>(descriptor_py.attr("layout"));
  const auto& layout = dmx_host::FindRecordLayout(schema, descriptor.layout);

  for (const py::handle row_py : descriptor_py.attr("rows")) {
    const py::sequence row = py::reinterpret_borrow<py::sequence>(row_py);
    if (static_cast<size_t>(py::len(row)) != layout.columns.size()) {
      throw py::value_error(
          "record descriptor row width does not match its layout");
    }
    ring::EncodedRecordRow encoded_row;
    encoded_row.cells.reserve(layout.columns.size());
    for (size_t i = 0; i < layout.columns.size(); ++i) {
      py::handle value = row[i];
      const auto column_type = layout.columns[i].type;
      if (IsPayloadSlice(value)) {
        ring::PayloadSlice slice;
        slice.offset_bytes = py::cast<uint64_t>(value.attr("offset_bytes"));
        py::object nbytes = value.attr("nbytes");
        if (!nbytes.is_none()) {
          slice.length_bytes = py::cast<uint64_t>(nbytes);
        }
        slice.materialization =
            ParsePayloadMaterialization(value, column_type);
        py::object dtype = value.attr("dtype");
        if (dtype.is_none()) {
          throw py::value_error("PayloadSlice dtype is required");
        }
        slice.dtype = static_cast<int32_t>(dtype.cast<at::ScalarType>());
        slice.logical_shape = py::cast<std::vector<int64_t>>(
            value.attr("shape"));
        for (size_t dim = 0; dim < slice.logical_shape.size(); ++dim) {
          if (slice.logical_shape[dim] == -1) {
            if (slice.inferred_dynamic_dim >= 0) {
              throw py::value_error(
                  "PayloadSlice supports at most one dynamic dimension");
            }
            slice.inferred_dynamic_dim = static_cast<int32_t>(dim);
          }
        }
        encoded_row.cells.emplace_back(std::move(slice));
        continue;
      }

      dmx_host::RecordValue literal =
          CopyLiteralRecordValue(value, column_type);
      std::visit(
          [&](auto&& item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, at::Tensor>) {
              throw py::type_error(
                  "record tensor columns require a PayloadSlice");
            } else {
              encoded_row.cells.emplace_back(std::forward<decltype(item)>(item));
            }
          },
          std::move(literal));
    }
    descriptor.rows.push_back(std::move(encoded_row));
  }
  return descriptor;
}

std::vector<ring::RecordDescriptor> CopyRecordDescriptors(
    const py::handle& descriptors_py,
    const py::handle& schema_py) {
  const auto schema = CopyRecordSchema(schema_py);
  std::vector<ring::RecordDescriptor> descriptors;
  for (const py::handle descriptor_py : descriptors_py) {
    descriptors.push_back(CopyRecordDescriptor(descriptor_py, schema));
  }
  return descriptors;
}
#endif

#ifndef DMI_HOST_ONLY
// DropNoticeBuffer::Drained -> Python dict, shared by SliceQueue and
// BatchQueue so both shed paths report the same shape:
//   {"notices": [(source, hook, layer_no, step_seq, start_token, end_token,
//                 nbytes, ts_us, [req_id, ...], rank[, capture_lease]), ...],
//    "overflow": int,   # notices lost since the last poll
//    "sticky":   bool,  # any notice lost in this window
//    "pending":  int}   # notices still buffered (max_items truncation)
// An empty "notices" with sticky False means nothing was lost.  rank 255 is
// the unranked slice path; the lease is appended only when the step had one.
py::dict drop_notices_to_py(ring_py::DropNoticeBuffer::Drained&& d) {
  py::list notices;
  for (auto& n : d.notices) {
    py::list rids;
    for (auto& r : n.req_ids) rids.append(py::str(r));
    py::tuple notice = py::make_tuple(py::str(n.source), py::str(n.hook), n.layer_no,
                                      n.step_seq, n.start_token, n.end_token,
                                      n.nbytes, n.ts_us, rids, static_cast<int>(n.rank));
    if (n.capture_lease.present())
      notice = py::cast<py::tuple>(notice.attr("__add__")(
          py::make_tuple(n.capture_lease.fork())));
    notices.append(notice);
  }
  py::dict out;
  out["notices"]  = notices;
  out["overflow"] = d.overflow;
  out["sticky"]   = d.sticky;
  out["pending"]  = d.pending;
  return out;
}

ring_py::HookBatchItem make_batch_item(
    const std::string& model_id, int32_t shard_rank, const std::string& act_name,
    int32_t layer_no, uint64_t step_seq, bool last_in_step, at::Tensor tensor,
    std::vector<std::string> req_ids, at::Tensor segments,
    ring_py::CaptureLease capture_lease, int64_t capture_ts_us,
    bool is_attn = false) {
  ring_py::HookBatchItem item;
  item.model_id      = model_id;
  item.shard_rank    = shard_rank;
  item.act_name      = act_name;
  item.layer_no      = layer_no;
  item.node_id       = 0;   // in-process
  item.step_seq      = step_seq;
  item.last_in_step  = last_in_step;
  item.is_attn       = is_attn;
  // Logical bytes; BatchQueue::push charges the allocation the tensor keeps
  // alive (pooled_block.h), not this length.
  item.nbytes        = static_cast<uint64_t>(tensor.nbytes());
  item.tensor        = std::move(tensor);
  item.req_ids       = std::move(req_ids);
  item.segments      = std::move(segments);
  item.capture_lease = std::move(capture_lease);
  item.capture_ts_us = capture_ts_us;
  return item;
}

using HookRowsPtr = std::shared_ptr<const std::vector<ring_py::RequestMeta>>;

// One hook's request layout for push_all_metas.  Two encodings:
//   * a list of (req_id, start, end, dim0_offset, kv_offset, orig_position)
//     tuples;
//   * a tuple of five int64 arrays (index into the step's request list,
//     start, end, dim0_offset, orig_position), which avoids building one
//     Python tuple per row.
HookRowsPtr parse_hook_rows(py::handle rows_obj,
                            const std::vector<ring_py::RequestMeta>& requests) {
  auto parsed = std::make_shared<std::vector<ring_py::RequestMeta>>();
  if (py::isinstance<py::tuple>(rows_obj)) {
    auto fields = py::reinterpret_borrow<py::tuple>(rows_obj);
    if (py::len(fields) != 5)
      throw py::value_error("hook request arrays need five fields");
    using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
    I64 idx = fields[0].cast<I64>(), lo = fields[1].cast<I64>();
    I64 hi = fields[2].cast<I64>(), off = fields[3].cast<I64>();
    I64 pos = fields[4].cast<I64>();
    if (idx.ndim() != 1 || lo.ndim() != 1 || hi.ndim() != 1 ||
        off.ndim() != 1 || pos.ndim() != 1)
      throw py::value_error("hook request arrays must be one-dimensional");
    const py::ssize_t n = idx.shape(0);
    if (lo.shape(0) != n || hi.shape(0) != n || off.shape(0) != n ||
        pos.shape(0) != n)
      throw py::value_error("hook request arrays differ in length");
    auto pi = idx.unchecked<1>();
    auto pl = lo.unchecked<1>();
    auto ph = hi.unchecked<1>();
    auto po = off.unchecked<1>();
    auto pp = pos.unchecked<1>();
    const int64_t n_req = static_cast<int64_t>(requests.size());
    auto int32_field = [](int64_t v) {
      if (v < INT32_MIN || v > INT32_MAX)
        throw py::value_error("hook request field exceeds int32");
      return static_cast<int32_t>(v);
    };
    parsed->reserve(static_cast<size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
      const int64_t r = pi(i);
      if (r < 0 || r >= n_req)
        throw py::value_error("hook request index outside the step's requests");
      ring_py::RequestMeta rm;
      rm.req_id = requests[static_cast<size_t>(r)].req_id;
      rm.start_token = int32_field(pl(i));
      rm.end_token = int32_field(ph(i));
      rm.dim0_offset = po(i);
      rm.kv_offset = 0;
      rm.orig_position = int32_field(pp(i));
      if (rm.end_token < rm.start_token || rm.dim0_offset < 0)
        throw py::value_error("invalid hook request range or offset");
      parsed->push_back(std::move(rm));
    }
    return parsed;
  }
  auto rows = rows_obj.cast<py::list>();
  parsed->reserve(py::len(rows));
  for (auto row : rows) {
    auto fields = row.cast<py::tuple>();
    if (py::len(fields) != 6)
      throw py::value_error("hook request row needs six fields");
    ring_py::RequestMeta rm;
    rm.req_id = fields[0].cast<std::string>();
    rm.start_token = fields[1].cast<int32_t>();
    rm.end_token = fields[2].cast<int32_t>();
    rm.dim0_offset = fields[3].cast<int64_t>();
    rm.kv_offset = fields[4].cast<int32_t>();
    rm.orig_position = fields[5].cast<int32_t>();
    if (rm.end_token < rm.start_token || rm.dim0_offset < 0)
      throw py::value_error("invalid hook request range or offset");
    parsed->push_back(std::move(rm));
  }
  return parsed;
}
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef DMI_HOST_ONLY
  // ---- Hook definitions (native ABI table; mirrored by dmi/hooks/catalog.py) ----
  // Expose as list of (id, act_name, short_name, per_layer, group, tp_sharded,
  //                     shape_class, pp_stage) tuples — all ints except act_name/short_name.
  // Python auto-derives all mappings from this at import time.
  {
    py::list defs;
    for (int i = 0; i < ring_py::HOOK_DEFS_COUNT; i++) {
      const auto& d = ring_py::HOOK_DEFS[i];
      defs.append(py::make_tuple(d.id, d.act_name, d.short_name, d.per_layer,
                                 d.group, d.tp_sharded, d.shape_class, d.pp_stage));
    }
    m.attr("HOOK_DEFS") = defs;
    m.attr("HOOK_TYPE_COUNT") = (int)ring_py::HOOK_TYPE_COUNT;
  }
#endif
  // ---- ClickHouseClientConfig (config only; stage is C++-only) ----
  py::class_<dmx_host::ClickHouseClientConfig>(m, "ClickHouseClientConfig")
      .def(py::init<>())

      .def_readwrite("host", &dmx_host::ClickHouseClientConfig::host)
      .def_readwrite("port", &dmx_host::ClickHouseClientConfig::port)
      .def_readwrite("username", &dmx_host::ClickHouseClientConfig::username)
      .def_readwrite("password", &dmx_host::ClickHouseClientConfig::password)
      .def_readwrite("database", &dmx_host::ClickHouseClientConfig::database)
      .def_readwrite("table", &dmx_host::ClickHouseClientConfig::table)
      .def_readwrite("secure", &dmx_host::ClickHouseClientConfig::secure)

      .def_readwrite("create_database_if_missing",
                     &dmx_host::ClickHouseClientConfig::create_database_if_missing)
      .def_readwrite("drop_existing_database",
                     &dmx_host::ClickHouseClientConfig::drop_existing_database)
      .def_readwrite("client_side_compress",
                     &dmx_host::ClickHouseClientConfig::client_side_compress)
      .def_readwrite("index_granularity",
                     &dmx_host::ClickHouseClientConfig::index_granularity)
      .def_readwrite("connect_timeout_ms",
                     &dmx_host::ClickHouseClientConfig::connect_timeout_ms)
      .def_readwrite("receive_timeout_ms",
                     &dmx_host::ClickHouseClientConfig::receive_timeout_ms)
      .def_readwrite("send_timeout_ms",
                     &dmx_host::ClickHouseClientConfig::send_timeout_ms)

      // Expose client_settings as a dict, store internally as unordered_map<string, variant<...>>.
      // This avoids requiring <pybind11/stl_variant.h>.
      .def_property(
          "client_settings",
          [](const dmx_host::ClickHouseClientConfig& self) {
            py::dict d;
            for (const auto& kv : self.client_settings) {
              const auto& key = kv.first;
              const auto& val = kv.second;
              if (std::holds_alternative<bool>(val)) {
                d[py::str(key)] = py::bool_(std::get<bool>(val));
              } else if (std::holds_alternative<std::int64_t>(val)) {
                d[py::str(key)] = py::int_(std::get<std::int64_t>(val));
              } else {
                d[py::str(key)] = py::str(std::get<std::string>(val));
              }
            }
            return d;
          },
          [](dmx_host::ClickHouseClientConfig& self, py::object obj) {
            self.client_settings.clear();
            if (obj.is_none()) return;

            py::dict d = obj.cast<py::dict>();
            for (auto item : d) {
              std::string key = py::cast<std::string>(item.first);
              py::handle v = item.second;

              // bool must be checked before int (Python bool is an int subclass)
              if (py::isinstance<py::bool_>(v)) {
                self.client_settings.emplace(std::move(key), py::cast<bool>(v));
              } else if (py::isinstance<py::int_>(v)) {
                self.client_settings.emplace(
                    std::move(key),
                    static_cast<std::int64_t>(py::cast<long long>(v)));
              } else if (py::isinstance<py::str>(v)) {
                self.client_settings.emplace(std::move(key), py::cast<std::string>(v));
              } else {
                throw py::type_error("client_settings values must be bool/int/str (or None)");
              }
            }
          });

  // ---- dmx_host StageConfig + DMXHostEngine ----
  using DMXHostEngine = dmx_host::DMXHostEngine;
  using StageConfig = DMXHostEngine::StageConfig;
  using ThreadFailure = DMXHostEngine::ThreadFailure;
  using QueueT = DMXHostEngine::QueueT;
  using QueueConfig = DMXHostEngine::QueueConfig;
  using EnqueuePolicy = DMXHostEngine::EnqueuePolicy;
  using Duration = DMXHostEngine::Duration;

  py::class_<dmx_host::ClickHouseWorkerMetrics>(m, "ClickHouseWorkerMetrics")
      .def_readonly("worker_index", &dmx_host::ClickHouseWorkerMetrics::worker_index)
      .def_readonly("batches", &dmx_host::ClickHouseWorkerMetrics::batches)
      .def_readonly("rows", &dmx_host::ClickHouseWorkerMetrics::rows)
      .def_readonly("logical_bytes", &dmx_host::ClickHouseWorkerMetrics::logical_bytes)
      .def_readonly("insert_seconds", &dmx_host::ClickHouseWorkerMetrics::insert_seconds);

  py::class_<dmx_host::ClickHouseMetricsSnapshot>(m, "ClickHouseMetricsSnapshot")
      .def_readonly("expected_workers", &dmx_host::ClickHouseMetricsSnapshot::expected_workers)
      .def_readonly("ready_workers", &dmx_host::ClickHouseMetricsSnapshot::ready_workers)
      .def_readonly("active_inserts", &dmx_host::ClickHouseMetricsSnapshot::active_inserts)
      .def_readonly("peak_active_inserts", &dmx_host::ClickHouseMetricsSnapshot::peak_active_inserts)
      .def_readonly("batches", &dmx_host::ClickHouseMetricsSnapshot::batches)
      .def_readonly("rows", &dmx_host::ClickHouseMetricsSnapshot::rows)
      .def_readonly("logical_bytes", &dmx_host::ClickHouseMetricsSnapshot::logical_bytes)
      .def_readonly("insert_seconds", &dmx_host::ClickHouseMetricsSnapshot::insert_seconds)
      .def_readonly("workers", &dmx_host::ClickHouseMetricsSnapshot::workers);

  py::enum_<dmx_host::OnFullPolicy>(m, "OnFullPolicy")
      .value("RAISE", dmx_host::OnFullPolicy::RAISE)
      .value("DROP", dmx_host::OnFullPolicy::DROP)
      .value("RETRY", dmx_host::OnFullPolicy::RETRY)
      .value("ABORT", dmx_host::OnFullPolicy::ABORT)
      .export_values();

  py::enum_<dmx_host::OnClosedPolicy>(m, "OnClosedPolicy")
      .value("RAISE", dmx_host::OnClosedPolicy::RAISE)
      .value("DROP", dmx_host::OnClosedPolicy::DROP)
      .export_values();

  py::class_<QueueConfig>(m, "QueueConfig")
      .def(py::init<>())
      .def_readwrite("min_batch_items", &QueueConfig::min_batch_items)
      .def_readwrite("min_batch_size", &QueueConfig::min_batch_size)
      .def_property(
          "max_linger_s",
          [](const QueueConfig& q) -> std::optional<double> {
            if (!q.max_linger) return std::nullopt;
            return q.max_linger->count();
          },
          [](QueueConfig& q, std::optional<double> v) {
            if (v) q.max_linger = Duration(*v);
            else q.max_linger.reset();
          })
      .def_readwrite("max_batch_items", &QueueConfig::max_batch_items)
      .def_readwrite("max_batch_size", &QueueConfig::max_batch_size)
      .def_readwrite("high_watermark_items", &QueueConfig::high_watermark_items)
      .def_readwrite("high_watermark_size", &QueueConfig::high_watermark_size);

  py::class_<EnqueuePolicy>(m, "EnqueuePolicy")
      .def(py::init<>())
      .def_readwrite("block", &EnqueuePolicy::block)
      .def_property(
          "timeout_s",
          [](const EnqueuePolicy& p) -> std::optional<double> {
            if (!p.timeout) return std::nullopt;
            return p.timeout->count();
          },
          [](EnqueuePolicy& p, std::optional<double> v) {
            if (v) p.timeout = Duration(*v);
            else p.timeout.reset();
          })
      .def_readwrite("on_full", &EnqueuePolicy::on_full)
      .def_readwrite("max_retries", &EnqueuePolicy::max_retries)
      .def_property(
          "retry_backoff_s",
          [](const EnqueuePolicy& p) { return p.retry_backoff.count(); },
          [](EnqueuePolicy& p, double v) { p.retry_backoff = Duration(v); })
      .def_readwrite("on_closed", &EnqueuePolicy::on_closed)
      .def_readwrite("drop_if_stopping", &EnqueuePolicy::drop_if_stopping);

  py::class_<ThreadFailure>(m, "ThreadFailure")
      .def_readonly("stage", &ThreadFailure::stage)
      .def_readonly("thread_name", &ThreadFailure::thread_name)
      .def_readonly("where", &ThreadFailure::where)
      .def_readonly("exc_type", &ThreadFailure::exc_type)
      .def_readonly("exc_what", &ThreadFailure::exc_what);

  py::class_<StageConfig>(m, "StageConfig")
      .def(py::init<>())
      .def_readwrite("name", &StageConfig::name)
      .def_readwrite("parallelism", &StageConfig::parallelism)
      .def_readwrite("input_queue", &StageConfig::input_queue)
      .def_readwrite("ingress_policy", &StageConfig::ingress_policy)
      .def_property(
          "thread_name_prefix",
          [](const StageConfig& s) { return s.thread_name_prefix; },
          [](StageConfig& s, std::optional<std::string> v) { s.thread_name_prefix = std::move(v); })
      // ClickHouse insert stage (the only stage in DMXHostEngine)
      .def_static(
          "clickhouse_insert",
          [](const dmx_host::ClickHouseClientConfig& ch_cfg, int parallelism, std::string name) {
            StageConfig cfg;
            cfg.name = std::move(name);
            cfg.parallelism = parallelism;
            cfg.input_queue.min_batch_items.reset();
            cfg.input_queue.min_batch_size = 16ULL * 1024 * 1024;
            cfg.input_queue.max_linger = Duration(0.05);
            cfg.input_queue.max_batch_items = 10000;
            cfg.input_queue.high_watermark_items = 20000;
            cfg.input_queue.high_watermark_size = 512ULL * 1024 * 1024;
            cfg.process_fn = [](std::vector<dmx_host::dmx_host_queue_item> batch, QueueT* next_q) {
              return dmx_host::ClickHouseInsertStage::ProcessFn<QueueT>(std::move(batch), next_q);
            };
            auto thread_cfg = ch_cfg;
            cfg.thread_init_config = std::move(thread_cfg);
            cfg.thread_init = &dmx_host::ClickHouseInsertStage::ThreadInitAny;
            cfg.thread_cleanup = &dmx_host::ClickHouseInsertStage::ThreadCleanupAny;
            return cfg;
          },
          py::arg("clickhouse_config"),
          py::arg("parallelism") = 1,
          py::arg("name") = "clickhouse_insert")
      .def_static(
          "clickhouse_records",
          [](const dmx_host::ClickHouseClientConfig& ch_cfg,
             py::object schema_py, int parallelism, std::string name) {
            if (parallelism <= 0) {
              throw py::value_error("parallelism must be positive");
            }
            StageConfig cfg;
            cfg.name = std::move(name);
            cfg.parallelism = parallelism;
            cfg.input_queue.min_batch_items.reset();
            cfg.input_queue.min_batch_size = 16ULL * 1024 * 1024;
            cfg.input_queue.max_linger = Duration(0.05);
            cfg.input_queue.max_batch_items = 10000;
            cfg.input_queue.high_watermark_items = 20000;
            cfg.input_queue.high_watermark_size = 512ULL * 1024 * 1024;
            cfg.process_fn = [](
                std::vector<dmx_host::dmx_host_queue_item> batch,
                QueueT* next_q) {
              return dmx_host::ClickHouseRecordInsertStage::ProcessFn<QueueT>(
                  std::move(batch), next_q);
            };
            dmx_host::ClickHouseRecordStageConfig stage_cfg;
            stage_cfg.client = ch_cfg;
            stage_cfg.schema = CopyRecordSchema(schema_py);
            cfg.thread_init_config = std::move(stage_cfg);
            cfg.thread_init =
                &dmx_host::ClickHouseRecordInsertStage::ThreadInitAny;
            cfg.thread_cleanup =
                &dmx_host::ClickHouseRecordInsertStage::ThreadCleanupAny;
            return cfg;
          },
          py::arg("clickhouse_config"), py::arg("schema"),
          py::arg("parallelism") = 1,
          py::arg("name") = "clickhouse_records");

  py::class_<DMXHostEngine, std::shared_ptr<DMXHostEngine>>(m, "DMXHostEngine")
      .def(py::init<StageConfig>(), py::arg("insert_stage"))
      .def("start", &DMXHostEngine::start)
      .def("wait_until_ready",
           [](DMXHostEngine& self, double timeout_s) {
             if (!std::isfinite(timeout_s)) {
               throw std::invalid_argument("timeout_s must be finite and non-negative");
             }
             return self.wait_until_ready(DMXHostEngine::Duration(timeout_s));
           },
           py::arg("timeout_s"),
           py::call_guard<py::gil_scoped_release>())
      .def("clickhouse_metrics", &DMXHostEngine::clickhouse_metrics)
      .def("stop",
           [](DMXHostEngine& self, bool graceful, std::optional<double> timeout_s) {
             if (timeout_s) {
               if (!std::isfinite(*timeout_s)) {
                 throw std::invalid_argument("timeout_s must be finite and non-negative");
               }
               return self.stop(graceful, DMXHostEngine::Duration(*timeout_s));
             }
             return self.stop(graceful, std::nullopt);
           },
           py::arg("graceful") = true,
           py::arg("timeout_s") = std::optional<double>(),
           py::call_guard<py::gil_scoped_release>())
      .def("close_input", &DMXHostEngine::close_input)
      .def("request_abort", &DMXHostEngine::request_abort)
      .def("join",
           [](DMXHostEngine& self, std::optional<double> timeout_s) {
             if (timeout_s) {
               if (!std::isfinite(*timeout_s)) {
                 throw std::invalid_argument("timeout_s must be finite and non-negative");
               }
               return self.join(DMXHostEngine::Duration(*timeout_s));
             }
             return self.join(std::nullopt);
           },
           py::arg("timeout_s") = std::optional<double>(),
           py::call_guard<py::gil_scoped_release>())
      .def("failures", &DMXHostEngine::failures)
      .def("raise_if_failed", &DMXHostEngine::raise_if_failed)
      .def("flush_and_wait",
           [](DMXHostEngine& self, double timeout_s) {
             if (!std::isfinite(timeout_s) || timeout_s <= 0.0) {
               throw std::invalid_argument(
                   "timeout_s must be finite and positive");
             }
             return self.flush_and_wait(DMXHostEngine::Duration(timeout_s));
           },
           py::arg("timeout_s") = 600.0,
           py::call_guard<py::gil_scoped_release>())
      .def("submit_record",
           [](DMXHostEngine& self, const std::string& layout,
              py::sequence cells, py::sequence cell_types,
              uint64_t nbytes) {
             if (py::len(cells) != py::len(cell_types)) {
               throw py::value_error(
                   "cells and cell_types must have the same length");
             }
             dmx_host::GenericRecordRow row;
             row.layout = layout;
             row.cells.reserve(static_cast<size_t>(py::len(cells)));
             uint64_t accounted = nbytes;
             for (py::ssize_t i = 0; i < py::len(cells); ++i) {
               const auto type = ParseRecordCellType(cell_types[i]);
               dmx_host::RecordValue value =
                   CopyLiteralRecordValue(cells[i], type);
               if (auto* tensor = std::get_if<at::Tensor>(&value)) {
                 if (!tensor->device().is_cpu()) {
                   throw py::value_error(
                       "submit_record tensor cells must already be on CPU");
                 }
                 if (!tensor->is_contiguous()) *tensor = tensor->contiguous();
                 if (nbytes == 0) {
                   accounted += static_cast<uint64_t>(tensor->nbytes());
                 }
               }
               row.cells.push_back(std::move(value));
             }
             py::gil_scoped_release release;
             self.submit_record(std::move(row), accounted);
           },
           py::arg("layout"), py::arg("cells"), py::arg("cell_types"),
           py::arg("nbytes") = uint64_t{0})
      // Submit a pre-formatted ClickHouseRow directly to the insert stage.
      // Called from the ring transport drain callback after format processing.
      .def("submit_direct",
           [](DMXHostEngine& self,
              const std::string& model_id, int32_t shard_rank,
              const std::string& req_id, const std::string& act_name,
              int32_t layer_no, int32_t start_token, int32_t end_token,
              at::Tensor tensor) {
             at::Tensor t = tensor.is_contiguous() ? tensor : tensor.contiguous();
             uint64_t nbytes = static_cast<uint64_t>(t.nbytes());
             dmx_host::ClickHouseRow row;
             row.push_back(model_id);
             row.push_back(req_id);
             row.push_back(act_name);
             row.push_back(layer_no);
             row.push_back(shard_rank);
             row.push_back(start_token);
             row.push_back(end_token);
             row.push_back(std::move(t));
             self.submit_direct(std::move(row), nbytes);
           },
           py::arg("model_id"), py::arg("shard_rank"),
           py::arg("req_id"), py::arg("act_name"),
           py::arg("layer_no"), py::arg("start_token"), py::arg("end_token"),
           py::arg("tensor"),
           py::call_guard<py::gil_scoped_release>());

  m.def(
      "_validate_record_host_schema",
      [](DMXHostEngine& host_engine, py::object schema) {
        host_engine.validate_record_schema(CopyRecordSchema(schema));
      },
      py::arg("host_engine"), py::arg("schema"));

  // -------------------------------------------------------------------------
  // Ring offload engine
  // -------------------------------------------------------------------------
#ifndef DMI_HOST_ONLY
  // ---- Capture leases (capture_lease.h) -----------------------------------
  py::class_<ring_py::CaptureLease>(m, "CaptureLease")
      .def(py::init<>())
      .def(py::init<uint64_t>(), py::arg("id"))
      .def(py::init<uint64_t,
                    const std::vector<std::tuple<std::string, int32_t, int>>&>(),
           py::arg("id"), py::arg("ranks"))
      .def(py::init<uint64_t, const ring_py::CaptureLease&>(),
           py::arg("id"), py::arg("policy"))
      .def_property_readonly("id", &ring_py::CaptureLease::id)
      .def_property_readonly("present", &ring_py::CaptureLease::present)
      .def_property_readonly("idle", &ring_py::CaptureLease::idle)
      .def("same_as", &ring_py::CaptureLease::same_as, py::arg("other"))
      .def("rank_for", &ring_py::CaptureLease::rank_for,
           py::arg("act_name"), py::arg("layer_no"), py::arg("fallback"))
      .def("fork", &ring_py::CaptureLease::fork);

  // ---- SliceQueue: per-slice in-process consumer (slice_queue.h) ---------
  py::class_<ring_py::SliceQueue, std::shared_ptr<ring_py::SliceQueue>>(m, "SliceQueue")
      .def(py::init<size_t, size_t, uint64_t>(), py::arg("max_items") = 200000,
           py::arg("drop_notice_cap") = 4096, py::arg("max_bytes") = uint64_t{0})
      // Normally pushed by the P2P thread; exposed for tests.
      .def("push", &ring_py::SliceQueue::push,
           py::arg("model_id"), py::arg("shard_rank"), py::arg("req_id"),
           py::arg("act_name"), py::arg("layer_no"), py::arg("start_token"),
           py::arg("end_token"), py::arg("slice"),
           py::arg("capture_lease") = ring_py::CaptureLease{},
           py::arg("capture_ts_us") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &ring_py::SliceQueue::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("wake", &ring_py::SliceQueue::wake,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dropped", &ring_py::SliceQueue::dropped)
      .def_property_readonly("qsize", &ring_py::SliceQueue::size)
      .def_property_readonly("retained_bytes", &ring_py::SliceQueue::retained_bytes)
      .def_property_readonly("max_bytes", &ring_py::SliceQueue::max_bytes)
      .def("drain_drop_notices",
           [](ring_py::SliceQueue& self, size_t max_items) {
               ring_py::DropNoticeBuffer::Drained d;
               {
                   py::gil_scoped_release rel;
                   d = self.drain_drop_notices(max_items);
               }
               return drop_notices_to_py(std::move(d));
           },
           py::arg("max_items") = 4096)
      .def_property_readonly("drop_notices_pending",
           [](const ring_py::SliceQueue& self) { return self.drop_notices().pending(); })
      .def_property_readonly("drop_notices_recorded",
           [](const ring_py::SliceQueue& self) { return self.drop_notices().recorded(); })
      .def_property_readonly("drop_notices_lost",
           [](const ring_py::SliceQueue& self) { return self.drop_notices().overflow_total(); })
      .def_property_readonly("drop_notice_cap",
           [](const ring_py::SliceQueue& self) { return self.drop_notices().cap(); })
      // Wait up to timeout_ms with the GIL released; each item is
      // (model_id, shard_rank, req_id, act_name, layer_no, start_token,
      //  end_token, tensor[, None, capture_lease, capture_ts_us]).
      .def("drain_slices",
           [](ring_py::SliceQueue& self, size_t max_items, int timeout_ms) {
               std::vector<ring_py::SliceItem> batch;
               {
                   py::gil_scoped_release rel;
                   batch = self.drain(max_items, timeout_ms);
               }
               py::list out;
               for (auto& it : batch) {
                   py::tuple item = py::make_tuple(it.model_id, it.shard_rank, it.req_id,
                                                   it.act_name, it.layer_no, it.start_token,
                                                   it.end_token, it.slice);
                   if (it.capture_lease.present())
                       item = py::cast<py::tuple>(item.attr("__add__")(py::make_tuple(
                           py::none(), it.capture_lease.fork(), it.capture_ts_us)));
                   out.append(item);
               }
               return out;
           },
           py::arg("max_items") = 256, py::arg("timeout_ms") = 50);

  // ---- BatchQueue: whole-hook batch handoff (batch_queue.h) --------------
  py::class_<ring_py::BatchQueue, std::shared_ptr<ring_py::BatchQueue>>(m, "BatchQueue")
      .def(py::init<uint64_t, size_t>(),
           py::arg("max_bytes") = uint64_t{512} * 1024 * 1024,
           py::arg("drop_notice_cap") = 4096)
      .def("stop", &ring_py::BatchQueue::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("wake", &ring_py::BatchQueue::wake,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dropped", &ring_py::BatchQueue::dropped)
      .def_property_readonly("dropped_bytes", &ring_py::BatchQueue::dropped_bytes)
      .def_property_readonly("qsize", &ring_py::BatchQueue::size)
      .def_property_readonly("retained_bytes", &ring_py::BatchQueue::retained_bytes)
      .def_property_readonly("pushed", &ring_py::BatchQueue::pushed)
      .def_property_readonly("max_bytes", &ring_py::BatchQueue::max_bytes)
      .def("set_layer_class_map", &ring_py::BatchQueue::set_layer_class_map,
           py::arg("entries"),
           py::arg("default_rank") = static_cast<int>(ring_py::kRankBestEffort),
           py::call_guard<py::gil_scoped_release>())
      .def("set_class_eviction", &ring_py::BatchQueue::set_class_eviction,
           py::arg("on"), py::call_guard<py::gil_scoped_release>())
      .def("rank_for", &ring_py::BatchQueue::rank_for,
           py::arg("act_name"), py::arg("layer_no"))
      .def_property_readonly("dropped_by_rank", &ring_py::BatchQueue::dropped_by_rank)
      .def_property_readonly("dropped_bytes_by_rank",
                             &ring_py::BatchQueue::dropped_bytes_by_rank)
      .def_property_readonly("qsize_by_rank", &ring_py::BatchQueue::qsize_by_rank)
      .def_property_readonly("evicted", &ring_py::BatchQueue::evicted)
      .def_property_readonly("class_eviction", &ring_py::BatchQueue::class_eviction)
      .def_property_readonly("class_map_set", &ring_py::BatchQueue::class_map_set)
      .def_property_readonly("class_map_size", &ring_py::BatchQueue::class_map_size)
      .def_property_readonly("class_default_rank",
                             &ring_py::BatchQueue::class_default_rank)
      .def("drain_drop_notices",
           [](ring_py::BatchQueue& self, size_t max_items) {
               ring_py::DropNoticeBuffer::Drained d;
               {
                   py::gil_scoped_release rel;
                   d = self.drain_drop_notices(max_items);
               }
               return drop_notices_to_py(std::move(d));
           },
           py::arg("max_items") = 4096)
      .def_property_readonly("drop_notices_pending",
           [](const ring_py::BatchQueue& self) { return self.drop_notices().pending(); })
      .def_property_readonly("drop_notices_recorded",
           [](const ring_py::BatchQueue& self) { return self.drop_notices().recorded(); })
      .def_property_readonly("drop_notices_lost",
           [](const ring_py::BatchQueue& self) { return self.drop_notices().overflow_total(); })
      .def_property_readonly("drop_notice_cap",
           [](const ring_py::BatchQueue& self) { return self.drop_notices().cap(); })
      // Normally pushed by the P2P thread's batch sink; exposed for tests and
      // benchmarks that feed prebuilt items.
      .def("push",
           [](ring_py::BatchQueue& self,
              const std::string& model_id, int32_t shard_rank,
              const std::string& act_name, int32_t layer_no,
              uint64_t step_seq, bool last_in_step,
              at::Tensor tensor, std::vector<std::string> req_ids,
              at::Tensor segments, ring_py::CaptureLease capture_lease,
              int64_t capture_ts_us) {
               self.push(make_batch_item(model_id, shard_rank, act_name, layer_no,
                                         step_seq, last_in_step, std::move(tensor),
                                         std::move(req_ids), std::move(segments),
                                         std::move(capture_lease), capture_ts_us));
           },
           py::arg("model_id"), py::arg("shard_rank"), py::arg("act_name"),
           py::arg("layer_no"), py::arg("step_seq"), py::arg("last_in_step"),
           py::arg("tensor"), py::arg("req_ids"), py::arg("segments"),
           py::arg("capture_lease") = ring_py::CaptureLease{},
           py::arg("capture_ts_us") = 0,
           py::call_guard<py::gil_scoped_release>())
      // Wait up to timeout_ms with the GIL released; each item is
      // (model_id, shard_rank, act_name, layer_no, node_id, step_seq,
      //  last_in_step, tensor, req_ids, segments[, capture_lease,
      //  capture_ts_us]).
      .def("drain_batches",
           [](ring_py::BatchQueue& self, size_t max_items, int timeout_ms) {
               std::vector<ring_py::HookBatchItem> batch;
               {
                   py::gil_scoped_release rel;
                   batch = self.drain(max_items, timeout_ms);
               }
               py::list out;
               for (auto& it : batch) {
                   py::list rids;
                   for (auto& r : it.req_ids) rids.append(py::str(r));
                   py::tuple item = py::make_tuple(
                       it.model_id, it.shard_rank, it.act_name, it.layer_no,
                       it.node_id, it.step_seq, it.last_in_step,
                       it.tensor, rids, it.segments);
                   if (it.capture_lease.present())
                       item = py::cast<py::tuple>(item.attr("__add__")(
                           py::make_tuple(it.capture_lease.fork(), it.capture_ts_us)));
                   out.append(item);
               }
               return out;
           },
           py::arg("max_items") = 64, py::arg("timeout_ms") = 50);

  // ---- RingFlushStats: host-only telemetry snapshot ----------------------
  py::class_<ring_py::RingFlushStats>(m, "RingFlushStats")
      .def_readonly("cpu_payload_head", &ring_py::RingFlushStats::cpu_payload_head)
      .def_readonly("cpu_payload_tail_committed", &ring_py::RingFlushStats::cpu_payload_tail_committed)
      .def_readonly("cpu_task_head", &ring_py::RingFlushStats::cpu_task_head)
      .def_readonly("cpu_task_tail_committed", &ring_py::RingFlushStats::cpu_task_tail_committed)
      .def_readonly("prepare_ok_calls", &ring_py::RingFlushStats::prepare_ok_calls)
      .def_readonly("prepare_flushed_calls", &ring_py::RingFlushStats::prepare_flushed_calls)
      .def_readonly("prepare_oversized_calls", &ring_py::RingFlushStats::prepare_oversized_calls)
      .def_readonly("prepare_sync_calls", &ring_py::RingFlushStats::prepare_sync_calls)
      .def_readonly("prepare_sync_ns", &ring_py::RingFlushStats::prepare_sync_ns)
      .def_readonly("prepare_sync_errors", &ring_py::RingFlushStats::prepare_sync_errors)
      .def_readonly("prepare_drain_calls", &ring_py::RingFlushStats::prepare_drain_calls)
      .def_readonly("prepare_drain_ns", &ring_py::RingFlushStats::prepare_drain_ns)
      .def_readonly("prepare_drain_errors", &ring_py::RingFlushStats::prepare_drain_errors)
      .def_readonly("toggle_wait_calls", &ring_py::RingFlushStats::toggle_wait_calls)
      .def_readonly("toggle_wait_ns", &ring_py::RingFlushStats::toggle_wait_ns)
      .def_readonly("toggle_wait_errors", &ring_py::RingFlushStats::toggle_wait_errors)
      .def_readonly("staging_wait_calls", &ring_py::RingFlushStats::staging_wait_calls)
      .def_readonly("staging_wait_ns", &ring_py::RingFlushStats::staging_wait_ns)
      .def_readonly("staging_wait_errors", &ring_py::RingFlushStats::staging_wait_errors)
      .def_readonly("ready_commit_tasks", &ring_py::RingFlushStats::ready_commit_tasks)
      .def_readonly("ready_commit_us", &ring_py::RingFlushStats::ready_commit_us)
      .def_readonly("ready_commit_invalid", &ring_py::RingFlushStats::ready_commit_invalid)
      .def_readonly("d2h_copy_errors", &ring_py::RingFlushStats::d2h_copy_errors)
      .def_readonly("d2h_sync_errors", &ring_py::RingFlushStats::d2h_sync_errors)
      .def_readonly("pageable_copy_calls", &ring_py::RingFlushStats::pageable_copy_calls)
      .def_readonly("pageable_copy_ns", &ring_py::RingFlushStats::pageable_copy_ns)
      .def_readonly("pageable_copy_bytes", &ring_py::RingFlushStats::pageable_copy_bytes);

  py::class_<ring_py::RingConfig>(m, "RingConfig")
      .def(py::init<>())
      .def_readwrite("task_ring_entries",         &ring_py::RingConfig::task_ring_entries)
      .def_readwrite("payload_ring_bytes",        &ring_py::RingConfig::payload_ring_bytes)
      .def_readwrite("pinned_staging_bytes",      &ring_py::RingConfig::pinned_staging_bytes)
      .def_readwrite("drain_poll_timeout_us",     &ring_py::RingConfig::drain_poll_timeout_us)
      .def_readwrite("drain_flush_task_ratio",     &ring_py::RingConfig::drain_flush_task_ratio)
      .def_readwrite("drain_flush_payload_ratio",  &ring_py::RingConfig::drain_flush_payload_ratio)
      .def_readwrite("drain_flush_entry_threshold", &ring_py::RingConfig::drain_flush_entry_threshold)
      .def_readwrite("drain_flush_byte_threshold",  &ring_py::RingConfig::drain_flush_byte_threshold)
      .def_readwrite("drain_flush_timeout_us",     &ring_py::RingConfig::drain_flush_timeout_us)
      .def_readwrite("clone_slices",              &ring_py::RingConfig::clone_slices)
      .def_readwrite("p2p_pool_cap_bytes",        &ring_py::RingConfig::p2p_pool_cap_bytes)
      .def_readwrite("batch_submit",              &ring_py::RingConfig::batch_submit)
      .def_readwrite("insert_queue_max_bytes",    &ring_py::RingConfig::insert_queue_max_bytes)
      .def_readwrite("insert_queue_max_items",    &ring_py::RingConfig::insert_queue_max_items)
      .def_readwrite("drain_policy",              &ring_py::RingConfig::drain_policy);

  // Production sinks remain native-only.  The explicitly named reference
  // bridge below is opt-in and intentionally pays the Python GIL/copy cost.
  py::class_<ring::RecordSinkLease,
             std::shared_ptr<ring::RecordSinkLease>>(
      m, "_RecordSinkLease");
  py::class_<ring::RecordSink, std::shared_ptr<ring::RecordSink>>(
      m, "RecordSink")
      .def("_acquire_engine",
           [](std::shared_ptr<ring::RecordSink> sink) {
             return ring::RecordSinkLease::acquire(std::move(sink));
           })
      .def_property_readonly("admission_bound_s", &AdmissionBoundSeconds);
  py::class_<dmx_host::ClickHouseRecordSink, ring::RecordSink,
             std::shared_ptr<dmx_host::ClickHouseRecordSink>>(
      m, "ClickHouseRecordSink")
      .def(py::init([](std::shared_ptr<dmx_host::DMXHostEngine> host) {
             return MakeClickHouseRecordSink(std::move(host));
           }),
           py::arg("host_engine"));
  py::class_<dmi_capture::ReferencePythonCaptureSink, ring::RecordSink,
             std::shared_ptr<dmi_capture::ReferencePythonCaptureSink>>(
      m, "ReferencePythonCaptureSink")
      .def(py::init([](py::object target, std::string layout,
                       std::optional<double> admission_bound_s) {
             std::optional<ring::RecordSink::Duration> bound;
             if (admission_bound_s) {
               if (!std::isfinite(*admission_bound_s) ||
                   *admission_bound_s < 0) {
                 throw py::value_error(
                     "admission_bound_s must be None or a finite, "
                     "non-negative number");
               }
               bound = std::chrono::ceil<ring::RecordSink::Duration>(
                   std::chrono::duration<double>(*admission_bound_s));
             }
             return std::make_shared<
                 dmi_capture::ReferencePythonCaptureSink>(
                     target.ptr(), std::move(layout), bound);
           }),
           py::arg("target"), py::arg("layout"),
           py::arg("admission_bound_s") = py::none())
      .def_property_readonly(
          "attached", &dmi_capture::ReferencePythonCaptureSink::engine_owned);

  py::class_<ring_py::RingEnginePy, std::shared_ptr<ring_py::RingEnginePy>>(m, "RingEngine")
      .def(py::init([](ring_py::RingConfig cfg, py::object host_engine_obj) {
             ring_py::SubmitFn submit_fn;
             ring_py::BatchSink batch_sink;
             if (host_engine_obj.is_none()) {
                 // No consumer: captures are drained and dropped (null and
                 // benchmark runs).
             } else if (py::isinstance<ring_py::BatchQueue>(host_engine_obj)) {
                 // Batch handoff: one item per (hook x step), pushed in pure
                 // C++ (no GIL).  Needs cfg.batch_submit for packed steps;
                 // a consumer drains it with BatchQueue.drain_batches().
                 // HF batched steps (token and attention hooks) arrive as
                 // one-request items, one per request slice (p2p_thread.h).
                 if (!cfg.batch_submit) {
                     throw py::value_error(
                         "a BatchQueue consumer needs RingConfig.batch_submit = True");
                 }
                 auto bq = host_engine_obj.cast<std::shared_ptr<ring_py::BatchQueue>>();
                 batch_sink = [bq](const std::string& model_id, int32_t shard_rank,
                                   const std::string& act_name, int32_t layer_no,
                                   uint64_t step_seq, bool last_in_step,
                                   at::Tensor tensor, std::vector<std::string> req_ids,
                                   at::Tensor segments, ring_py::CaptureLease capture_lease,
                                   int64_t capture_ts_us, bool is_attn) {
                     bq->push(make_batch_item(model_id, shard_rank, act_name, layer_no,
                                              step_seq, last_in_step, std::move(tensor),
                                              std::move(req_ids), std::move(segments),
                                              std::move(capture_lease), capture_ts_us,
                                              is_attn));
                 };
             } else if (py::isinstance<ring_py::SliceQueue>(host_engine_obj)) {
                 // Per-slice queue, pushed in pure C++ (no GIL); a consumer
                 // drains it with SliceQueue.drain_slices().
                 auto sq = host_engine_obj.cast<std::shared_ptr<ring_py::SliceQueue>>();
                 submit_fn = [sq](const std::string& model_id, int32_t shard_rank,
                                  const std::string& req_id, const std::string& act_name,
                                  int32_t layer_no, int32_t start_token, int32_t end_token,
                                  at::Tensor slice, int64_t capture_ts_us,
                                  ring_py::CaptureLease capture_lease) {
                     sq->push(model_id, shard_rank, req_id, act_name, layer_no,
                              start_token, end_token, std::move(slice),
                              std::move(capture_lease), capture_ts_us);
                 };
             } else if (py::isinstance<dmx_host::DMXHostEngine>(host_engine_obj)) {
                 auto host = host_engine_obj.cast<std::shared_ptr<dmx_host::DMXHostEngine>>();
                 submit_fn = [host](const std::string& model_id, int32_t shard_rank,
                                    const std::string& req_id, const std::string& act_name,
                                    int32_t layer_no, int32_t start_token, int32_t end_token,
                                    at::Tensor slice, int64_t /*capture_ts_us*/,
                                    ring_py::CaptureLease /*capture_lease*/) {
                     dmx_host::ClickHouseRow row;
                     row.emplace_back(model_id);
                     row.emplace_back(req_id);
                     row.emplace_back(act_name);
                     row.emplace_back(layer_no);
                     row.emplace_back(shard_rank);
                     row.emplace_back(start_token);
                     row.emplace_back(end_token);
                     uint64_t nbytes = static_cast<uint64_t>(slice.nbytes());
                     row.emplace_back(std::move(slice));
                     host->submit_direct(std::move(row), nbytes);
                 };
             } else {
                 // An in-process consumer: any Python callable taking
                 // (model_id, shard_rank, req_id, act_name, layer_no,
                 //  start_token, end_token, tensor), plus (capture_lease,
                 //  capture_ts_us) when the step carried a lease.  The P2P
                 // thread calls it without the GIL, so take it for the call.
                 // An exception reaches the P2P thread's consumer-failure
                 // accounting (logged once, counted, slice dropped).
                 if (!PyCallable_Check(host_engine_obj.ptr())) {
                     throw py::type_error(
                         "RingEngine host_engine must be a DMXHostEngine, a "
                         "BatchQueue, a SliceQueue, a callable consumer, or None");
                 }
                 auto fn = std::shared_ptr<py::object>(
                     new py::object(host_engine_obj),
                     [](py::object* held) {
                         // The last reference can drop on the P2P thread.
                         py::gil_scoped_acquire gil;
                         delete held;
                     });
                 submit_fn = [fn](const std::string& model_id, int32_t shard_rank,
                                  const std::string& req_id, const std::string& act_name,
                                  int32_t layer_no, int32_t start_token, int32_t end_token,
                                  at::Tensor slice, int64_t capture_ts_us,
                                  ring_py::CaptureLease capture_lease) {
                     py::gil_scoped_acquire gil;
                     try {
                         if (capture_lease.present()) {
                             (*fn)(model_id, shard_rank, req_id, act_name, layer_no,
                                   start_token, end_token, std::move(slice),
                                   capture_lease.fork(), capture_ts_us);
                         } else {
                             (*fn)(model_id, shard_rank, req_id, act_name, layer_no,
                                   start_token, end_token, std::move(slice));
                         }
                     } catch (py::error_already_set& error) {
                         std::string what = error.what();
                         throw std::runtime_error(what);
                     }
                 };
             }
             return MakeRingEngine(std::move(cfg), std::move(submit_fn),
                                   std::move(batch_sink));
           }),
           py::arg("config"), py::arg("host_engine") = py::none())
      .def_static(
          "create_record",
          [](ring_py::RingConfig cfg, py::object sink_or_host,
             const std::string& failure_policy,
             uint64_t step_stall_budget_ms) {
            ring_py::RecordRuntimeOptions options;
            options.failure_policy = ParseRecordFailurePolicy(failure_policy);
            options.step_stall_budget_ms = step_stall_budget_ms;
            std::shared_ptr<ring::RecordSinkLease> lease;
            if (!sink_or_host.is_none()) {
              if (py::isinstance<ring::RecordSinkLease>(sink_or_host)) {
                lease = sink_or_host.cast<
                    std::shared_ptr<ring::RecordSinkLease>>();
              } else if (py::isinstance<ring::RecordSink>(sink_or_host)) {
                lease = ring::RecordSinkLease::acquire(
                    sink_or_host.cast<std::shared_ptr<ring::RecordSink>>());
              } else {
                auto host = sink_or_host.cast<
                    std::shared_ptr<dmx_host::DMXHostEngine>>();
                lease = ring::RecordSinkLease::acquire(
                    MakeClickHouseRecordSink(std::move(host)));
              }
            }
            return MakeRingEngine(std::move(cfg), std::move(lease), options);
          },
          py::arg("config"), py::arg("sink_or_host") = py::none(),
          py::arg("failure_policy") = "raise",
          py::arg("step_stall_budget_ms") = uint64_t{0})
      .def("init",  &ring_py::RingEnginePy::init,
           py::arg("stream_handle") = uint64_t{0})
      .def("start", &ring_py::RingEnginePy::start)
      .def("stop",  &ring_py::RingEnginePy::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_step",
           &ring_py::RingEnginePy::prepare_step,
           py::arg("step_total_bytes"),
           py::arg("num_hooks"),
           py::arg("reserve") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("reserve_record",
           &ring_py::RingEnginePy::reserve_record,
           py::arg("reservation_items"),
           py::call_guard<py::gil_scoped_release>())
      .def("begin_record_step", &ring_py::RingEnginePy::begin_record_step)
      .def("allow_flush_step", &ring_py::RingEnginePy::allow_flush_step)
      .def("capture_status",
           [](const ring_py::RingEnginePy& self) {
             const ring_py::RecordCaptureStatus status =
                 self.record_capture_status();
             py::dict out;
             out["failure_policy"] =
                 RecordFailurePolicyName(status.failure_policy);
             out["failed"] = status.failed;
             out["failure"] = status.failure;
             out["discarded_descriptors"] = status.discarded_descriptors;
             out["discarded_payloads"] = status.discarded_payloads;
             out["step_stall_budget_ms"] = status.step_stall_budget_ms;
             out["stall_budget_exhaustions"] = status.stall_budget_exhaustions;
             out["skipped_steps"] = status.skipped_steps;
             out["steps_with_discards"] = status.steps_with_discards;
             out["reserve_wait_ns"] = status.reserve_wait_ns;
             out["max_step_wait_ns"] = status.max_step_wait_ns;
             return out;
           })
      .def("push_record_descriptors",
           [](ring_py::RingEnginePy& self, py::sequence descriptors,
              py::object schema) {
             auto encoded = CopyRecordDescriptors(descriptors, schema);
             py::gil_scoped_release release;
             self.push_record_descriptors(std::move(encoded));
           },
           py::arg("descriptors"), py::arg("schema"))
      .def("submit_record_cpu_direct",
           [](ring_py::RingEnginePy& self, at::Tensor cpu_tensor,
              uint64_t tensor_bytes) {
             if (!cpu_tensor.device().is_cpu()) {
               throw py::value_error(
                   "record CPU-direct tensor must already be on CPU");
             }
             at::Tensor contiguous = cpu_tensor.is_contiguous()
                 ? std::move(cpu_tensor) : cpu_tensor.contiguous();
             if (tensor_bytes != static_cast<uint64_t>(contiguous.nbytes())) {
               throw py::value_error(
                   "record CPU-direct byte count does not match tensor");
             }
             py::gil_scoped_release release;
             self.submit_record_cpu_direct(
                 std::move(contiguous), tensor_bytes);
           },
           py::arg("cpu_tensor"), py::arg("tensor_bytes"))
      .def("flush_records_and_wait",
           [](ring_py::RingEnginePy& self, double timeout_s) {
             if (!std::isfinite(timeout_s) || timeout_s <= 0.0) {
               throw std::invalid_argument(
                   "timeout_s must be finite and positive");
             }
             const double milliseconds = std::ceil(timeout_s * 1000.0);
             if (milliseconds >
                 static_cast<double>(std::numeric_limits<int64_t>::max())) {
               throw std::invalid_argument("timeout_s is too large");
             }
             return self.flush_records_and_wait(
                 static_cast<uint64_t>(milliseconds));
           },
           py::arg("timeout_s") = 600.0,
           py::call_guard<py::gil_scoped_release>())
      .def("submit_cpu_direct",
           [](ring_py::RingEnginePy& self, at::Tensor cpu_tensor) {
               uint64_t nbytes = static_cast<uint64_t>(cpu_tensor.nbytes());
               self.submit_cpu_direct(std::move(cpu_tensor), nbytes);
           },
           py::arg("cpu_tensor"),
           py::call_guard<py::gil_scoped_release>())
      .def("payload_cap", &ring_py::RingEnginePy::payload_cap)
      .def("staging_cap", &ring_py::RingEnginePy::staging_cap)
      .def("task_cap",    &ring_py::RingEnginePy::task_cap)
      .def("payload_tensor", &ring_py::RingEnginePy::payload_tensor)
      // Safety-net surface (eager only).  available_capacity(),
      // available_task_slots() and reserve_one() are CPU-only and fast --
      // no GIL release needed.
      // flush_and_wait() blocks on cudaStreamSynchronize + drain flush --
      // GIL released so other Python threads aren't blocked.
      .def("available_capacity", &ring_py::RingEnginePy::available_capacity)
      .def("available_task_slots",
           &ring_py::RingEnginePy::available_task_slots)
      .def("reserve_one",
           &ring_py::RingEnginePy::reserve_one,
           py::arg("nbytes"))
      .def("flush_and_wait",
           &ring_py::RingEnginePy::flush_and_wait,
           py::call_guard<py::gil_scoped_release>())
      .def("drain_to_sink_and_wait",
           &ring_py::RingEnginePy::drain_to_sink_and_wait,
           py::arg("timeout_ms") = uint32_t{0},
           py::call_guard<py::gil_scoped_release>())
      .def("request_flush",
           &ring_py::RingEnginePy::request_flush,
           py::call_guard<py::gil_scoped_release>())
      .def("get_stats", &ring_py::RingEnginePy::get_stats,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("submit_exceptions",
           &ring_py::RingEnginePy::submit_exceptions)
      .def_property_readonly("last_sink_error",
           &ring_py::RingEnginePy::last_sink_error)
      .def_property_readonly("sink_failed",
           &ring_py::RingEnginePy::sink_failed)
      .def("set_abort_on_sink_error",
           &ring_py::RingEnginePy::set_abort_on_sink_error,
           py::arg("enabled"))
      .def("push_all_metas",
           [](ring_py::RingEnginePy& self,
              py::list hook_types_py,
              py::list layer_nos_py,
              py::list shapes_py,
              py::list dtypes_py,
              py::list flags_py,
              const std::string& model_id,
              int32_t tp_rank,
              int32_t dp_rank,
              int32_t ep_rank,
              int32_t pp_rank,
              bool flattened,
              py::list req_ids_py,
              py::list token_ranges_py,
              py::list dim0_offsets_py,
              py::list kv_offsets_py,
              py::list orig_positions_py,
              py::list hook_requests_py,
              ring_py::CaptureLease capture_lease) {
               const size_t n_hooks = static_cast<size_t>(py::len(hook_types_py));
               if (py::len(hook_requests_py) &&
                   static_cast<size_t>(py::len(hook_requests_py)) != n_hooks)
                   throw py::value_error("hook_requests must have one entry per hook");
               // Build step context (heap-allocated, ownership to FIFO/p2p)
               auto owned_ctx = std::make_unique<ring_py::StepContext>();
               auto* ctx = owned_ctx.get();
               ctx->capture_lease = std::move(capture_lease);
               ctx->model_id  = model_id;
               ctx->tp_rank   = tp_rank;
               ctx->dp_rank   = dp_rank;
               ctx->ep_rank   = ep_rank;
               ctx->pp_rank   = pp_rank;
               ctx->flattened = flattened;
               ctx->requests.reserve(static_cast<size_t>(py::len(req_ids_py)));
               for (size_t i = 0; i < static_cast<size_t>(py::len(req_ids_py)); ++i) {
                   ring_py::RequestMeta rm;
                   rm.req_id      = py::cast<std::string>(req_ids_py[i]);
                   py::tuple tr   = token_ranges_py[i].cast<py::tuple>();
                   rm.start_token = py::cast<int32_t>(tr[0]);
                   rm.end_token   = py::cast<int32_t>(tr[1]);
                   rm.dim0_offset = py::cast<int64_t>(dim0_offsets_py[i]);
                   if (i < static_cast<size_t>(py::len(kv_offsets_py)))
                       rm.kv_offset = py::cast<int32_t>(kv_offsets_py[i]);
                   // Only when the integration filtered the request list
                   // (empty list = unfiltered; -1 = use the list index).
                   if (i < static_cast<size_t>(py::len(orig_positions_py)))
                       rm.orig_position = py::cast<int32_t>(orig_positions_py[i]);
                   ctx->requests.push_back(std::move(rm));
               }
               // Build per-hook metas
               size_t n = static_cast<size_t>(py::len(hook_types_py));
               std::vector<ring_py::TensorMeta> metas;
               metas.reserve(n);
               // Equal per-hook layouts are passed as the same Python object:
               // parse each once while the arguments keep it alive, and never
               // retain Python objects in the FIFO.
               std::unordered_map<PyObject*, HookRowsPtr> layouts;
               for (size_t i = 0; i < n; ++i) {
                   ring_py::TensorMeta meta;
                   meta.hook_type    = py::cast<int>(hook_types_py[i]);
                   meta.layer_no     = py::cast<int>(layer_nos_py[i]);
                   meta.dtype        = static_cast<int>(dtypes_py[i].cast<at::ScalarType>());
                   meta.last_in_step = (i == n - 1);
                   meta.flags        = static_cast<uint8_t>(py::cast<int>(flags_py[i]));
                   py::list shape    = shapes_py[i].cast<py::list>();
                   for (auto d : shape)
                       meta.shape.push_back(py::cast<int64_t>(d));
                   if (py::len(hook_requests_py) && !hook_requests_py[i].is_none()) {
                       py::handle rows = hook_requests_py[i];
                       auto found = layouts.find(rows.ptr());
                       if (found != layouts.end()) {
                           meta.requests = found->second;
                       } else {
                           auto parsed = parse_hook_rows(rows, ctx->requests);
                           meta.requests = parsed;
                           layouts.emplace(rows.ptr(), std::move(parsed));
                       }
                   }
                   metas.push_back(std::move(meta));
               }
               // Release GIL, push context + metas in single lock
               py::gil_scoped_release release;
               self.push_step(owned_ctx.release(), metas);
           },
           py::arg("hook_types"), py::arg("layer_nos"),
           py::arg("shapes"), py::arg("dtypes"), py::arg("flags"),
           py::arg("model_id"),
           py::arg("tp_rank"), py::arg("dp_rank"),
           py::arg("ep_rank"), py::arg("pp_rank"),
           py::arg("flattened"),
           py::arg("req_ids"), py::arg("token_ranges"),
           py::arg("dim0_offsets"),
           py::arg("kv_offsets") = py::list(),
           py::arg("orig_positions") = py::list(),
           py::arg("hook_requests") = py::list(),
           py::arg("capture_lease") = ring_py::CaptureLease{})
      .def_static("supports_hook_request_arrays", []() { return true; })
      .def("set_gather_host_mirror",
           &ring_py::RingEnginePy::set_gather_host_mirror,
           py::arg("dev_base"), py::arg("host_base"), py::arg("row_bytes"),
           py::arg("rows"))
      .def("gather_host_skips", &ring_py::RingEnginePy::gather_host_skips)
      .def("head_launches", &ring_py::RingEnginePy::head_launches)
      .def("set_null_mode",
           &ring_py::RingEnginePy::set_null_mode,
           py::arg("enabled"),
           py::call_guard<py::gil_scoped_release>())
      .def("notify_drain",
           &ring_py::RingEnginePy::notify_drain,
           py::call_guard<py::gil_scoped_release>())
      // ---- Runtime CUDA-graph node toggling -------------------------------
      // Graph, node and exec handles are the integers returned by
      // torch.cuda.CUDAGraph.raw_cuda_graph() / raw_cuda_graph_exec().
      .def("enable_toggle_capture",
           &ring_py::RingEnginePy::enable_toggle_capture, py::arg("enabled"))
      .def("toggle_capture_enabled",
           &ring_py::RingEnginePy::toggle_capture_enabled)
      .def("capture_anomaly_count",
           &ring_py::RingEnginePy::capture_anomaly_count)
      .def("bind_graph_exec", &ring_py::RingEnginePy::bind_graph_exec,
           py::arg("graph"), py::arg("exec"))
      .def("toggle_registry_version",
           &ring_py::RingEnginePy::toggle_registry_version)
      .def("set_enabled_hooks", &ring_py::RingEnginePy::set_enabled_hooks,
           py::arg("enabled"))
      .def("apply_toggle", &ring_py::RingEnginePy::apply_toggle)
      .def("is_hook_enabled", &ring_py::RingEnginePy::is_hook_enabled,
           py::arg("hook_type"), py::arg("layer_no"))
      .def("effective_enabled_mask",
           &ring_py::RingEnginePy::effective_enabled_mask, py::arg("query"))
      .def("ensure_graph_current",
           &ring_py::RingEnginePy::ensure_graph_current, py::arg("graph"))
      .def("record_replay_event",
           &ring_py::RingEnginePy::record_replay_event, py::arg("graph"))
      .def("set_toggle_pre_apply_sync",
           &ring_py::RingEnginePy::set_toggle_pre_apply_sync,
           py::arg("enabled"))
      .def("toggle_node_count", &ring_py::RingEnginePy::toggle_node_count)
      .def("toggle_node_count_for_graph",
           &ring_py::RingEnginePy::toggle_node_count_for_graph,
           py::arg("graph"))
      .def("producer_launch_count",
           &ring_py::RingEnginePy::producer_launch_count)
      .def("bound_graph_count", &ring_py::RingEnginePy::bound_graph_count)
      .def("last_apply_count", &ring_py::RingEnginePy::last_apply_count)
      .def("toggle_registry_uniform",
           &ring_py::RingEnginePy::toggle_registry_uniform)
      .def("is_graph_ready", &ring_py::RingEnginePy::is_graph_ready,
           py::arg("graph"))
      .def("toggle_registry_complete",
           &ring_py::RingEnginePy::toggle_registry_complete)
      .def("clear_toggle_registry",
           &ring_py::RingEnginePy::clear_toggle_registry)
      .def("enable_device_toggle",
           &ring_py::RingEnginePy::enable_device_toggle, py::arg("enabled"))
      .def("toggle_device_mode", &ring_py::RingEnginePy::toggle_device_mode)
      .def("toggle_device_node_count",
           &ring_py::RingEnginePy::toggle_device_node_count)
      .def("toggle_device_errors",
           &ring_py::RingEnginePy::toggle_device_errors)
      .def("upload_graph_exec", &ring_py::RingEnginePy::upload_graph_exec,
           py::arg("graph"), py::arg("exec"),
           py::call_guard<py::gil_scoped_release>())
      .def("_test_force_apply_error",
           &ring_py::RingEnginePy::_test_force_apply_error, py::arg("code"))
      .def("_test_applied_current",
           &ring_py::RingEnginePy::_test_applied_current, py::arg("graph"));

  // Register the active ring engine pointer so C++ ring_producer_impl can
  // call it during CUDA graph capture.  The raw pointer is valid as long as
  // Python holds the shared_ptr (i.e. while the RingTransport is active).
  m.def("ring_set_active_engine",
        [](std::shared_ptr<ring_py::RingEnginePy> engine) {
            ring_set_active_engine(engine.get());
        },
        py::arg("engine"));

  m.def("ring_clear_active_engine",
        []() { ring_set_active_engine(nullptr); });

  // CPU-only view of push_all_metas' per-hook layout parsing: returns the
  // parsed (req_id, start, end, dim0_offset, kv_offset, orig_position) rows.
  m.def("parse_hook_requests",
        [](py::list req_ids_py, py::handle rows) {
          std::vector<ring_py::RequestMeta> requests;
          requests.reserve(py::len(req_ids_py));
          for (auto rid : req_ids_py) {
            ring_py::RequestMeta rm;
            rm.req_id = rid.cast<std::string>();
            requests.push_back(std::move(rm));
          }
          py::list out;
          const HookRowsPtr parsed = parse_hook_rows(rows, requests);
          for (const auto& rm : *parsed) {
            out.append(py::make_tuple(rm.req_id, rm.start_token, rm.end_token,
                                      rm.dim0_offset, rm.kv_offset,
                                      rm.orig_position));
          }
          return out;
        },
        py::arg("req_ids"), py::arg("rows"));
#endif
}
