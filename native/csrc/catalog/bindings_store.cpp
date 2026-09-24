// _dmi_native_store: the native capture storage path's Python surface.
//
//   StorageService   spool -> object store -> catalog, on a background thread
//   CaptureReader    search / select / hydrate against the catalog + store
//
// Pure C++ plus libcurl and libcrypto. It uses pybind11's headers but links
// nothing from torch, and registers no ring types, so it loads beside
// _native_backend and _dmi_native_sink without type-registry conflicts.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "catalog/hydration.h"
#include "catalog/reader.h"
#include "catalog/storage_service.h"

namespace py = pybind11;
namespace dc = dmi_catalog;

namespace {

template <typename T>
T get(const py::dict& d, const char* key, T fallback) {
  return d.contains(key) && !d[key].is_none() ? d[key].cast<T>() : fallback;
}

dmi_store::S3Config s3_config(const py::dict& d) {
  dmi_store::S3Config c;
  c.endpoint = get<std::string>(d, "s3_endpoint", "");
  c.bucket = get<std::string>(d, "s3_bucket", "");
  c.region = get<std::string>(d, "s3_region", c.region);
  c.access_key = get<std::string>(d, "s3_access_key", "");
  c.secret_key = get<std::string>(d, "s3_secret_key", "");
  c.session_token = get<std::string>(d, "s3_session_token", "");
  c.allow_insecure_http = get<bool>(d, "s3_allow_insecure_http", false);
  c.connect_timeout_s = get<int>(d, "s3_connect_timeout_s", c.connect_timeout_s);
  c.read_timeout_s = get<int>(d, "s3_read_timeout_s", c.read_timeout_s);
  c.max_attempts = get<int>(d, "s3_max_attempts", c.max_attempts);
  if (c.endpoint.empty() || c.bucket.empty()) {
    throw py::value_error("s3_endpoint and s3_bucket are required");
  }
  return c;
}

dc::ClickHouseTimeouts clickhouse_timeouts(const py::dict& d) {
  dc::ClickHouseTimeouts t;
  t.connect_s = get<double>(d, "clickhouse_connect_timeout_s", t.connect_s);
  t.request_s = get<double>(d, "clickhouse_request_timeout_s", t.request_s);
  return t;
}

dc::StorageServiceConfig service_config(const py::dict& d) {
  dc::StorageServiceConfig c;
  c.spool_root = get<std::string>(d, "spool_root", "");
  c.spool_max_bytes = get<uint64_t>(d, "spool_max_bytes", c.spool_max_bytes);
  c.s3 = s3_config(d);
  c.uploader.store_id = get<std::string>(d, "store_id", c.uploader.store_id);
  c.uploader.max_workers = get<int>(d, "uploader_max_workers", c.uploader.max_workers);
  c.uploader.max_in_flight_bytes =
      get<uint64_t>(d, "uploader_max_in_flight_bytes", c.uploader.max_in_flight_bytes);
  c.uploader.max_attempts = get<int>(d, "uploader_max_attempts", c.uploader.max_attempts);
  c.clickhouse_host = get<std::string>(d, "clickhouse_host", c.clickhouse_host);
  c.clickhouse_port = get<uint16_t>(d, "clickhouse_port", c.clickhouse_port);
  c.clickhouse_timeouts = clickhouse_timeouts(d);
  c.max_index_attempts = get<int>(d, "max_index_attempts", c.max_index_attempts);
  c.writer.database = get<std::string>(d, "database", "default");
  c.writer.table_prefix = get<std::string>(d, "table_prefix", "dmi");
  c.writer.lease_ttl_ns = get<uint64_t>(d, "lease_ttl_ns", c.writer.lease_ttl_ns);
  c.writer.publish_timeout_ns =
      get<uint64_t>(d, "publish_timeout_ns", c.writer.publish_timeout_ns);
  c.writer.clock_skew_ns = get<uint64_t>(d, "clock_skew_ns", c.writer.clock_skew_ns);
  c.indexer.max_packs = get<int>(d, "indexer_max_packs", c.indexer.max_packs);
  c.indexer.max_estimated_bytes =
      get<uint64_t>(d, "indexer_max_estimated_bytes", c.indexer.max_estimated_bytes);
  c.holder = get<std::string>(d, "holder", "");
  c.reconcile_prefix = get<std::string>(d, "reconcile_prefix", c.reconcile_prefix);
  c.poll_interval_ns = get<uint64_t>(d, "poll_interval_ns", c.poll_interval_ns);
  c.max_backoff_ns = get<uint64_t>(d, "max_backoff_ns", c.max_backoff_ns);
  c.reconcile_interval_ns =
      get<uint64_t>(d, "reconcile_interval_ns", c.reconcile_interval_ns);
  c.sweep_spool_on_start = get<bool>(d, "sweep_spool_on_start", c.sweep_spool_on_start);
  c.reconcile_on_start = get<bool>(d, "reconcile_on_start", c.reconcile_on_start);
  return c;
}

py::dict snapshot_dict(const dc::StorageServiceSnapshot& s) {
  py::dict out;
  out["running"] = s.running;
  out["cycles"] = s.cycles;
  out["uploaded_packs"] = s.uploaded_packs;
  out["uploaded_bytes"] = s.uploaded_bytes;
  out["upload_failures"] = s.upload_failures;
  out["indexed_packs"] = s.indexed_packs;
  out["indexed_rows"] = s.indexed_rows;
  out["index_failures"] = s.index_failures;
  out["batch_splits"] = s.batch_splits;
  out["reconcile_passes"] = s.reconcile_passes;
  out["reconciled_packs"] = s.reconciled_packs;
  out["reconcile_skipped_objects"] = s.reconcile_skipped_objects;
  out["lease_renewals"] = s.lease_renewals;
  out["swept_on_start"] = s.swept_on_start;
  out["pending_index"] = s.pending_index;
  out["rejected_packs"] = s.rejected_packs;
  out["last_error"] = s.last_error;
  return out;
}

dc::SearchFilters filters_from(const py::dict& d) {
  dc::SearchFilters f;
  auto opt_str = [&d](const char* key) -> std::optional<std::string> {
    if (!d.contains(key) || d[key].is_none()) return std::nullopt;
    return d[key].cast<std::string>();
  };
  auto opt_u64 = [&d](const char* key) -> std::optional<uint64_t> {
    if (!d.contains(key) || d[key].is_none()) return std::nullopt;
    return d[key].cast<uint64_t>();
  };
  f.tenant_id = opt_str("tenant_id");
  f.experiment_id = opt_str("experiment_id");
  f.run_id = opt_str("run_id");
  f.session_id = opt_str("session_id");
  f.model_id = opt_str("model_id");
  f.cursor = opt_str("cursor");
  f.captured_after_ns = opt_u64("captured_after_ns");
  f.captured_before_ns = opt_u64("captured_before_ns");
  if (d.contains("hook_names") && !d["hook_names"].is_none()) {
    f.hook_names = d["hook_names"].cast<std::vector<std::string>>();
  }
  if (d.contains("layer_numbers") && !d["layer_numbers"].is_none()) {
    f.layer_numbers = d["layer_numbers"].cast<std::vector<int64_t>>();
  }
  f.limit = get<int>(d, "limit", f.limit);
  return f;
}

dc::Selection selection_from(const py::dict& d) {
  dc::Selection s;
  s.selection_id = d["selection_id"].cast<std::string>();
  s.capture_ids = d["capture_ids"].cast<std::vector<std::string>>();
  s.catalog_watermark = d["catalog_watermark"].cast<std::string>();
  s.filter_hash = d["filter_hash"].cast<std::string>();
  s.tenant_id = d["tenant_id"].cast<std::string>();
  return s;
}

// The catalog and store handles a reader needs, owned together.
class CaptureReader {
 public:
  explicit CaptureReader(const py::dict& d)
      : s3_(s3_config(d)),
        client_(std::make_shared<const dc::ClickHouseClient>(
            get<std::string>(d, "clickhouse_host", "127.0.0.1"),
            get<uint16_t>(d, "clickhouse_port", 8123),
            clickhouse_timeouts(d))),
        config_{get<std::string>(d, "database", "default"),
                get<std::string>(d, "table_prefix", "dmi")},
        catalog_(client_, config_),
        reader_(&s3_, s3_.config().bucket, client_, config_,
                get<int64_t>(d, "max_coalesce_gap_bytes", 4096)) {}

  py::dict search(const py::dict& filters) {
    const dc::SearchFilters f = filters_from(filters);
    dc::SearchPage page;
    {
      py::gil_scoped_release release;
      page = catalog_.search(f);
    }
    py::dict out;
    out["items"] = page.items;
    out["watermark"] = page.watermark;
    out["next_cursor"] = page.next_cursor ? py::cast(*page.next_cursor) : py::none();
    return out;
  }

  py::dict select(const py::dict& filters) {
    const dc::SearchFilters f = filters_from(filters);
    dc::Selection s;
    {
      py::gil_scoped_release release;
      s = reader_.select(f);
    }
    py::dict out;
    out["selection_id"] = s.selection_id;
    out["capture_ids"] = s.capture_ids;
    out["catalog_watermark"] = s.catalog_watermark;
    out["filter_hash"] = s.filter_hash;
    out["tenant_id"] = s.tenant_id;
    return out;
  }

  py::list resolve(const py::dict& selection) {
    const dc::Selection s = selection_from(selection);
    std::vector<std::vector<std::string>> items;
    {
      py::gil_scoped_release release;
      items = reader_.resolve(s);
    }
    return py::cast(items);
  }

  // Payload bytes in selection order, as `bytes` -- never `str`, which would
  // UTF-8-decode tensor data.
  py::list hydrate(const py::dict& selection, int64_t byte_limit,
                   int64_t request_limit) {
    // Checked here as well as in the reader core, and passed through as
    // int64: an unsigned cast of a negative limit reads as no limit at all.
    if (byte_limit < 0) throw py::value_error("byte_limit must be non-negative");
    if (request_limit <= 0) throw py::value_error("request_limit must be positive");
    const dc::Selection s = selection_from(selection);
    std::vector<std::string> payloads;
    {
      py::gil_scoped_release release;
      payloads = reader_.hydrate(s, byte_limit, request_limit);
    }
    py::list out;
    for (const std::string& p : payloads) out.append(py::bytes(p));
    return out;
  }

 private:
  dmi_store::S3Client s3_;
  std::shared_ptr<const dc::ClickHouseClient> client_;
  dc::ReaderConfig config_;
  dc::NativeCaptureCatalog catalog_;
  dc::NativeCaptureReader reader_;
};

}  // namespace

PYBIND11_MODULE(_dmi_native_store, m) {
  m.doc() = "Native capture storage path: storage service and catalog reader.";
  m.attr("SEARCH_ITEM_COLUMNS") = py::tuple(py::cast(dc::search_item_columns()));

  py::class_<dc::CaptureStorageService>(m, "StorageService")
      .def(py::init([](const py::dict& d) {
             return std::make_unique<dc::CaptureStorageService>(service_config(d));
           }),
           py::arg("config"))
      .def("start", &dc::CaptureStorageService::start,
           py::call_guard<py::gil_scoped_release>())
      .def("flush", &dc::CaptureStorageService::flush, py::arg("timeout_s"),
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &dc::CaptureStorageService::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("snapshot",
           [](const dc::CaptureStorageService& s) { return snapshot_dict(s.snapshot()); })
      .def("rethrow_if_failed", &dc::CaptureStorageService::rethrow_if_failed);

  py::class_<CaptureReader>(m, "CaptureReader")
      .def(py::init<const py::dict&>(), py::arg("config"))
      .def("search", &CaptureReader::search, py::arg("filters"))
      .def("select", &CaptureReader::select, py::arg("filters"))
      .def("resolve", &CaptureReader::resolve, py::arg("selection"))
      .def("hydrate", &CaptureReader::hydrate, py::arg("selection"),
           py::arg("byte_limit"), py::arg("request_limit") = 1024);
}
