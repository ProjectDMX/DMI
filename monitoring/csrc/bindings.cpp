// Pybind11 module — ring transport + ClickHouse pipeline bindings

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "clickhouse_client.h"
#include "dmx_host_engine.h"
#include "ring/ring_engine_py.h"
#include "ring/ring_torch_op.h"
#include "ring/tensor_meta.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- Hook definitions (single source of truth from C++ HOOK_DEFS table) ----
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
            cfg.process_fn = [](std::vector<dmx_host::dmx_host_queue_item> batch, QueueT* next_q) {
              return dmx_host::ClickHouseInsertStage::ProcessFn<QueueT>(std::move(batch), next_q);
            };
            // Stored by value in std::any; ClickHouseInsertStage::ThreadInitAny will any_cast it.
            cfg.thread_init_config = ch_cfg;
            cfg.thread_init = &dmx_host::ClickHouseInsertStage::ThreadInitAny;
            cfg.thread_cleanup = &dmx_host::ClickHouseInsertStage::ThreadCleanupAny;
            return cfg;
          },
          py::arg("clickhouse_config"),
          py::arg("parallelism") = 1,
          py::arg("name") = "clickhouse_insert");

  py::class_<DMXHostEngine, std::shared_ptr<DMXHostEngine>>(m, "DMXHostEngine")
      .def(py::init<StageConfig>(), py::arg("insert_stage"))
      .def("start", &DMXHostEngine::start)
      .def("stop",
           [](DMXHostEngine& self, bool graceful, std::optional<double> timeout_s) {
             if (timeout_s) {
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
             if (timeout_s) return self.join(DMXHostEngine::Duration(*timeout_s));
             return self.join(std::nullopt);
           },
           py::arg("timeout_s") = std::optional<double>(),
           py::call_guard<py::gil_scoped_release>())
      .def("failures", &DMXHostEngine::failures)
      .def("raise_if_failed", &DMXHostEngine::raise_if_failed)
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

  // -------------------------------------------------------------------------
  // Ring offload engine
  // -------------------------------------------------------------------------
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
             return std::make_shared<ring_py::RingEnginePy>(
                 std::move(cfg), std::move(submit_fn));
           }),
           py::arg("config"), py::arg("host_engine") = py::none())
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
      .def("push_all_metas",
           [](ring_py::RingEnginePy& self,
              py::list hook_types_py,
              py::list layer_nos_py,
              py::list shapes_py,
              py::list dtypes_py,
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
           py::arg("shapes"), py::arg("dtypes"),
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

  m.def("ring_set_cpu_direct",
        [](bool enabled) { ring_set_cpu_direct(enabled); },
        py::arg("enabled"));
}
