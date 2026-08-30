// Pybind11 module — ring transport + ClickHouse pipeline bindings

#ifdef DMI_HOST_ONLY
#include <torch/csrc/utils/pybind.h>
#else
#include <torch/extension.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
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
#include "ring/ring_engine_py.h"
#include "ring/ring_torch_op.h"
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
      .def_readwrite("insert_queue_max_bytes",    &ring_py::RingConfig::insert_queue_max_bytes)
      .def_readwrite("insert_queue_max_items",    &ring_py::RingConfig::insert_queue_max_items);

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
           });
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
      .def(py::init([](py::object target, std::string layout) {
             return std::make_shared<
                 dmi_capture::ReferencePythonCaptureSink>(
                     target.ptr(), std::move(layout));
           }),
           py::arg("target"), py::arg("layout"))
      .def_property_readonly(
          "attached", &dmi_capture::ReferencePythonCaptureSink::engine_owned);

  py::class_<ring_py::RingEnginePy, std::shared_ptr<ring_py::RingEnginePy>>(m, "RingEngine")
      .def(py::init([](ring_py::RingConfig cfg, py::object host_engine_obj) {
             ring_py::SubmitFn submit_fn;
             if (!host_engine_obj.is_none()) {
                 auto host = host_engine_obj.cast<std::shared_ptr<dmx_host::DMXHostEngine>>();
                 submit_fn = [host](const std::string& model_id, int32_t shard_rank,
                                    const std::string& req_id, const std::string& act_name,
                                    int32_t layer_no, int32_t start_token, int32_t end_token,
                                    at::Tensor slice) {
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
             }
             return MakeRingEngine(
                 std::move(cfg), std::move(submit_fn));
           }),
           py::arg("config"), py::arg("host_engine") = py::none())
      .def_static(
          "create_record",
          [](ring_py::RingConfig cfg, py::object sink_or_host) {
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
            return MakeRingEngine(std::move(cfg), std::move(lease));
          },
          py::arg("config"), py::arg("sink_or_host") = py::none())
      .def("init",  &ring_py::RingEnginePy::init,
           py::arg("stream_handle") = uint64_t{0})
      .def("start", &ring_py::RingEnginePy::start)
      .def("stop",  &ring_py::RingEnginePy::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_step",
           &ring_py::RingEnginePy::prepare_step,
           py::arg("step_total_bytes"),
           py::arg("num_hooks"),
           py::call_guard<py::gil_scoped_release>())
      .def("reserve_record",
           &ring_py::RingEnginePy::reserve_record,
           py::arg("reservation_items"),
           py::call_guard<py::gil_scoped_release>())
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
      // Safety-net surface (eager only).  available_capacity() and
      // reserve_one() are CPU-only and fast -- no GIL release needed.
      // flush_and_wait() blocks on cudaStreamSynchronize + drain flush --
      // GIL released so other Python threads aren't blocked.
      .def("available_capacity", &ring_py::RingEnginePy::available_capacity)
      .def("reserve_one",
           &ring_py::RingEnginePy::reserve_one,
           py::arg("nbytes"))
      .def("flush_and_wait",
           &ring_py::RingEnginePy::flush_and_wait,
           py::call_guard<py::gil_scoped_release>())
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
              py::list kv_offsets_py) {
               // Build step context (heap-allocated, ownership to FIFO/p2p)
               auto* ctx = new ring_py::StepContext();
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
                   ctx->requests.push_back(std::move(rm));
               }
               // Build per-hook metas
               size_t n = static_cast<size_t>(py::len(hook_types_py));
               std::vector<ring_py::TensorMeta> metas;
               metas.reserve(n);
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
                   metas.push_back(std::move(meta));
               }
               // Release GIL, push context + metas in single lock
               py::gil_scoped_release release;
               self.push_step(ctx, metas);
           },
           py::arg("hook_types"), py::arg("layer_nos"),
           py::arg("shapes"), py::arg("dtypes"), py::arg("flags"),
           py::arg("model_id"),
           py::arg("tp_rank"), py::arg("dp_rank"),
           py::arg("ep_rank"), py::arg("pp_rank"),
           py::arg("flattened"),
           py::arg("req_ids"), py::arg("token_ranges"),
           py::arg("dim0_offsets"),
           py::arg("kv_offsets") = py::list())
      .def("set_null_mode",
           &ring_py::RingEnginePy::set_null_mode,
           py::arg("enabled"),
           py::call_guard<py::gil_scoped_release>())
      .def("notify_drain",
           &ring_py::RingEnginePy::notify_drain,
           py::call_guard<py::gil_scoped_release>());

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
#endif
}
