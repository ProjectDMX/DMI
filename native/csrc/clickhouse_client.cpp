// clickhouse_client.cpp
#include "clickhouse_client.h"
// For DMXHostEngine::QueueT (the queue type actually used in bindings.cpp)
#include "dmx_host_engine.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <clickhouse/columns/array.h>
#include <clickhouse/columns/string.h>
#include <clickhouse/columns/numeric.h>
#include <clickhouse/types/types.h>

namespace dmx_host {

ClickHouseRuntimeMetrics::ClickHouseRuntimeMetrics(int expected_workers)
    : expected_workers_(expected_workers),
      ready_(static_cast<std::size_t>(std::max(0, expected_workers)), false),
      workers_(static_cast<std::size_t>(std::max(0, expected_workers))) {
  if (expected_workers < 0) {
    throw std::invalid_argument("expected_workers must be non-negative");
  }
  for (int i = 0; i < expected_workers; ++i) workers_[i].worker_index = i;
}

void ClickHouseRuntimeMetrics::WorkerReady(int worker_index) {
  std::lock_guard<std::mutex> lock(mu_);
  if (worker_index < 0 || worker_index >= expected_workers_) {
    throw std::out_of_range("ClickHouse worker index is out of range");
  }
  if (!ready_[worker_index]) {
    ready_[worker_index] = true;
    ++ready_workers_;
    ready_cv_.notify_all();
  }
}

bool ClickHouseRuntimeMetrics::WaitUntilReady(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mu_);
  return ready_cv_.wait_for(lock, timeout, [&] {
    return ready_workers_ == expected_workers_;
  });
}

void ClickHouseRuntimeMetrics::BeginInsert() {
  std::lock_guard<std::mutex> lock(mu_);
  ++active_inserts_;
  peak_active_inserts_ = std::max(peak_active_inserts_, active_inserts_);
}

void ClickHouseRuntimeMetrics::EndInsert(int worker_index, std::uint64_t rows,
                                         std::uint64_t logical_bytes,
                                         double seconds) {
  std::lock_guard<std::mutex> lock(mu_);
  if (active_inserts_ > 0) --active_inserts_;
  ++batches_;
  rows_ += rows;
  logical_bytes_ += logical_bytes;
  insert_seconds_ += seconds;
  if (worker_index >= 0 && worker_index < expected_workers_) {
    auto& worker = workers_[worker_index];
    ++worker.batches;
    worker.rows += rows;
    worker.logical_bytes += logical_bytes;
    worker.insert_seconds += seconds;
  }
}

ClickHouseMetricsSnapshot ClickHouseRuntimeMetrics::Snapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  ClickHouseMetricsSnapshot snapshot;
  snapshot.expected_workers = expected_workers_;
  snapshot.ready_workers = ready_workers_;
  snapshot.active_inserts = active_inserts_;
  snapshot.peak_active_inserts = peak_active_inserts_;
  snapshot.batches = batches_;
  snapshot.rows = rows_;
  snapshot.logical_bytes = logical_bytes_;
  snapshot.insert_seconds = insert_seconds_;
  snapshot.workers = workers_;
  return snapshot;
}

namespace {

thread_local std::unique_ptr<clickhouse::Client> tl_client;
thread_local bool tl_inited = false;
thread_local bool tl_cleaned = false;
thread_local std::string tl_db;
thread_local std::string tl_table;
thread_local int tl_worker_index = -1;
thread_local std::shared_ptr<ClickHouseRuntimeMetrics> tl_runtime_metrics;

// --------------------- SQL helpers ---------------------

std::string QuoteIdent(const std::string& ident) {
  std::string out;
  out.reserve(ident.size() + 2);
  out.push_back('`');
  for (char c : ident) {
    if (c == '`') { out.push_back('`'); out.push_back('`'); }
    else { out.push_back(c); }
  }
  out.push_back('`');
  return out;
}

std::string QuoteStringLiteral(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\\' || c == '\'') out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

void ValidateSettingName(const std::string& name) {
  if (name.empty()) throw std::invalid_argument("client_settings contains an empty key");
  for (unsigned char uc : name) {
    const char c = static_cast<char>(uc);
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.')) {
      throw std::invalid_argument(
          "client_settings key contains unsupported character: " + name +
          " (allowed: [A-Za-z0-9_.])");
    }
  }
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string QualifiedTableNameQuoted(const std::string& db, const std::string& table) {
  return QuoteIdent(db) + "." + QuoteIdent(table);
}

std::string ClickHouseInitRemediation(const ClickHouseClientConfig& cfg,
                                      const char* error) {
  const std::string err = error ? error : "";
  const bool database_missing =
      err.find("Database ") != std::string::npos
      && err.find("does not exist") != std::string::npos;
  if (err.find("Database default does not exist") != std::string::npos) {
    return "Please create the ClickHouse database 'default' before starting DMI.";
  }
  if (database_missing && !cfg.create_database_if_missing) {
    return "Please set create_database_if_missing=true, or create the configured "
           "ClickHouse database '" + cfg.database + "' manually before "
           "starting DMI.";
  }
  if (database_missing) {
    return "Please create the missing ClickHouse database manually, or verify the "
           "configured database name and permissions.";
  }
  return "Please check ClickHouse connectivity, credentials, database/table "
         "configuration, and server logs.";
}

void LogClickHouseInitFailure(const ClickHouseClientConfig& cfg,
                              const char* error) noexcept {
  static std::once_flag log_once;
  std::call_once(log_once, [&] {
    try {
      const std::string remediation = ClickHouseInitRemediation(cfg, error);
      std::cerr
          << "[DMI][ClickHouse] ERROR: failed to initialize ClickHouse insert "
             "stage. "
          << remediation
          << " host=" << cfg.host
          << " port=" << cfg.port
          << " database=" << cfg.database
          << " table=" << cfg.table
          << " create_database_if_missing="
          << (cfg.create_database_if_missing ? "true" : "false")
          << " error=\"" << (error ? error : "unknown") << "\""
          << std::endl;
    } catch (...) {
    }
  });
}

// --------------------- clickhouse-cpp compat: compression ---------------------

template <typename OptionsT>
auto SetCompressionCompatImpl(OptionsT& opts, clickhouse::CompressionMethod m, int)
    -> decltype(opts.SetCompressionMethod(m), void()) {
  opts.SetCompressionMethod(m);
}

template <typename OptionsT>
auto SetCompressionCompatImpl(OptionsT& opts, clickhouse::CompressionMethod m, long)
    -> decltype((void)(opts.compression_method = m), void()) {
  opts.compression_method = m;
}

template <typename OptionsT>
void SetCompressionCompat(OptionsT& opts, clickhouse::CompressionMethod m) {
  SetCompressionCompatImpl(opts, m, 0);
}

// --------------------- parsing helpers ---------------------

int32_t TakeInt32(ClickHouseValue&& v, const char* field) {
  if (!std::holds_alternative<int32_t>(v)) {
    throw std::invalid_argument(std::string("Expected int32_t for ") + field);
  }
  return std::get<int32_t>(v);
}

std::string TakeString(ClickHouseValue&& v, const char* field) {
  if (!std::holds_alternative<std::string>(v)) {
    throw std::invalid_argument(std::string("Expected string for ") + field);
  }
  return std::move(std::get<std::string>(v));
}

at::Tensor TakeTensor(ClickHouseValue&& v, const char* field) {
  if (!std::holds_alternative<at::Tensor>(v)) {
    throw std::invalid_argument(std::string("Expected at::Tensor for ") + field);
  }
  return std::move(std::get<at::Tensor>(v));
}

// --------------------- tensor dtype mapping (Torch strings) ---------------------

// Must match strings from your Python mapping, e.g. "torch.float", "torch.long", "torch.cfloat", ...
const char* ScalarTypeToTorchDtypeString(at::ScalarType t) {
  switch (t) {
    // float types (choose aliases that your TORCH_DTYPES_TYPE2NAME will produce)
    case at::kFloat:    return "torch.float";    // torch.float32 type -> "torch.float"
    case at::kDouble:   return "torch.double";   // torch.float64 type -> "torch.double"
    case at::kHalf:     return "torch.half";     // torch.float16 type -> "torch.half"
    case at::kBFloat16: return "torch.bfloat16";

    // integer/bool (choose aliases)
    case at::kByte:   return "torch.uint8";
    case at::kChar:   return "torch.int8";
    case at::kShort:  return "torch.short";      // int16 -> "torch.short"
    case at::kInt:    return "torch.int";        // int32 -> "torch.int"
    case at::kLong:   return "torch.long";       // int64 -> "torch.long"
    case at::kBool:   return "torch.bool";

    // complex (choose aliases)
#if defined(ATen_CORE_ScalarType_H) || 1
    case at::kComplexHalf:   return "torch.chalf";   // complex32 -> "torch.chalf"
    case at::kComplexFloat:  return "torch.cfloat";  // complex64 -> "torch.cfloat"
    case at::kComplexDouble: return "torch.cdouble"; // complex128 -> "torch.cdouble"
#endif

    default:
      return nullptr;  // unsupported (e.g. float8 variants)
  }
}

// --------------------- tensor encoding ---------------------

struct EncodedTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  at::Tensor cpu_contig;         // owns bytes
  std::string_view bytes_view;   // view into cpu_contig storage
};

EncodedTensor EncodeTensorToColumns(const at::Tensor& tin) {
  if (!tin.defined()) throw std::invalid_argument("Tensor is undefined");

  at::Tensor t = tin;
  if (!t.device().is_cpu()) t = t.cpu();
  if (!t.is_contiguous()) t = t.contiguous();

  const char* dt = ScalarTypeToTorchDtypeString(t.scalar_type());
  if (!dt) {
    throw std::invalid_argument("Unsupported tensor scalar_type for offload (dtype string mapping missing)");
  }

  EncodedTensor out;
  out.dtype = dt;
  out.shape.assign(t.sizes().begin(), t.sizes().end());
  out.cpu_contig = t;

  const size_t nbytes =
      static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());

  if (nbytes == 0) {
    out.bytes_view = std::string_view{};
    return out;
  }

  const void* ptr = t.data_ptr();
  if (!ptr) throw std::runtime_error("tensor.data_ptr() is null but nbytes > 0");

  out.bytes_view = std::string_view(static_cast<const char*>(ptr), nbytes);
  return out;
}

// --------------------- client/session helpers ---------------------

clickhouse::Client& ClientOrThrow() {
  if (!tl_inited || tl_cleaned || !tl_client) {
    throw std::runtime_error(
        "ClickHouse client not initialized in this thread. "
        "Call ClickHouseInsertStage::ThreadInit() first.");
  }
  return *tl_client;
}

void ApplySessionSettings(clickhouse::Client& client, const ClickHouseClientConfig& cfg) {
  if (cfg.client_settings.empty()) return;

  for (const auto& kv : cfg.client_settings) {
    const std::string& key = kv.first;
    ValidateSettingName(key);

    const auto& v = kv.second;
    std::string sql_value;

    if (std::holds_alternative<bool>(v)) {
      sql_value = std::get<bool>(v) ? "1" : "0";
    } else if (std::holds_alternative<std::int64_t>(v)) {
      sql_value = std::to_string(std::get<std::int64_t>(v));
    } else if (std::holds_alternative<std::string>(v)) {
      sql_value = QuoteStringLiteral(std::get<std::string>(v));
    } else {
      throw std::invalid_argument("client_settings value must be str/int/bool");
    }

    client.Execute("SET " + key + " = " + sql_value);
  }
}

void RunSchemaInitOnce(const ClickHouseClientConfig& cfg, const clickhouse::ClientOptions& opts) {
  auto client = std::make_unique<clickhouse::Client>(opts);

  const std::string db_q = QuoteIdent(cfg.database);
  const std::string table_q = QuoteIdent(cfg.table);
  const std::string fq_table_q = db_q + "." + table_q;

  if (cfg.drop_existing_database) {
    client->Execute("DROP DATABASE IF EXISTS " + db_q);
  }

  if (cfg.create_database_if_missing || cfg.drop_existing_database) {
    client->Execute("CREATE DATABASE IF NOT EXISTS " + db_q);
  }

  // Schema:
  // dtype: String
  // shape: Array(Int64)
  // bytes: String (binary-safe)
  std::ostringstream ddl;
  ddl << "CREATE TABLE IF NOT EXISTS " << fq_table_q << " ("
      << QuoteIdent("model_id") << " String, "
      << QuoteIdent("request_id") << " String, "
      << QuoteIdent("act_name") << " String, "
      << QuoteIdent("layer_no") << " Int32, "
      << QuoteIdent("shard_rank") << " Int32, "
      << QuoteIdent("start_token_idx") << " Int32, "
      << QuoteIdent("end_token_idx") << " Int32, "
      << QuoteIdent("dtype") << " String, "
      << QuoteIdent("shape") << " Array(Int64), "
      << QuoteIdent("bytes") << " String"
      << ") ENGINE = MergeTree "
      << "PRIMARY KEY ("
      << QuoteIdent("model_id") << ", "
      << QuoteIdent("request_id") << ", "
      << QuoteIdent("act_name") << ", "
      << QuoteIdent("layer_no") << ", "
      << QuoteIdent("shard_rank") << ", "
      << QuoteIdent("start_token_idx") << ", "
      << QuoteIdent("end_token_idx")
      << ") "
      << "ORDER BY ("
      << QuoteIdent("model_id") << ", "
      << QuoteIdent("request_id") << ", "
      << QuoteIdent("act_name") << ", "
      << QuoteIdent("layer_no") << ", "
      << QuoteIdent("shard_rank") << ", "
      << QuoteIdent("start_token_idx") << ", "
      << QuoteIdent("end_token_idx")
      << ") "
      << "SETTINGS index_granularity = " << cfg.index_granularity;

  client->Execute(ddl.str());
}

// --------------------- staging ---------------------

struct StagedRow {
  std::string model_id;
  std::string request_id;
  std::string act_name;
  int32_t layer_no = 0;
  int32_t shard_rank = 0;
  int32_t start_token_idx = 0;
  int32_t end_token_idx = 0;

  std::string dtype;
  std::vector<int64_t> shape;

  at::Tensor bytes_tensor;       // owns bytes (contig CPU)
  std::string_view bytes_view;   // view into bytes_tensor storage
};

StagedRow StageOneRow(ClickHouseRow&& row) {
  // Contract: exactly 8 fields.
  // [model_id, request_id, act_name, layer_no, shard_rank, start_token_idx, end_token_idx, tensor]
  if (row.size() != 8) {
    throw std::invalid_argument(
        "ClickHouseRow must have exactly 8 cells: "
        "[model_id,request_id,act_name,layer_no,shard_rank,start_token_idx,end_token_idx,tensor]");
  }

  StagedRow r;
  r.model_id = TakeString(std::move(row[0]), "model_id");
  r.request_id = TakeString(std::move(row[1]), "request_id");
  r.act_name = TakeString(std::move(row[2]), "act_name");
  r.layer_no = TakeInt32(std::move(row[3]), "layer_no");
  r.shard_rank = TakeInt32(std::move(row[4]), "shard_rank");
  r.start_token_idx = TakeInt32(std::move(row[5]), "start_token_idx");
  r.end_token_idx = TakeInt32(std::move(row[6]), "end_token_idx");

  at::Tensor t = TakeTensor(std::move(row[7]), "tensor");

  EncodedTensor enc = EncodeTensorToColumns(t);
  r.dtype = std::move(enc.dtype);
  r.shape = std::move(enc.shape);
  r.bytes_tensor = std::move(enc.cpu_contig);
  r.bytes_view = enc.bytes_view;

  return r;
}

}  // namespace

// ===================== Stage API =====================

void ClickHouseInsertStage::ThreadInit(int thread_idx, const ClickHouseClientConfig& cfg) {
  if (tl_inited) {
    throw std::runtime_error("ClickHouseInsertStage::ThreadInit() called more than once in the same thread");
  }
  if (tl_cleaned) {
    throw std::runtime_error("ClickHouseInsertStage::ThreadInit() called after ThreadCleanup() in the same thread");
  }

  tl_db = cfg.database;
  tl_table = cfg.table;

  clickhouse::ClientOptions opts;
  opts.SetHost(cfg.host);
  opts.SetPort(static_cast<uint16_t>(cfg.port));
  opts.SetUser(cfg.username);
  opts.SetPassword(cfg.password);
  if (cfg.connect_timeout_ms < 0 || cfg.receive_timeout_ms < 0 ||
      cfg.send_timeout_ms < 0) {
    throw std::invalid_argument("ClickHouse socket timeouts must be non-negative");
  }
  opts.SetConnectionConnectTimeout(std::chrono::milliseconds(cfg.connect_timeout_ms));
  opts.SetConnectionRecvTimeout(std::chrono::milliseconds(cfg.receive_timeout_ms));
  opts.SetConnectionSendTimeout(std::chrono::milliseconds(cfg.send_timeout_ms));

  // stable default database for handshake
  opts.SetDefaultDatabase("default");

  if (cfg.secure) {
    opts.SetSSLOptions(clickhouse::ClientOptions::SSLOptions{});
  }

  // Compression: "none" | "lz4" | "zstd" | "true" | "false"
  {
    const std::string s = ToLower(cfg.client_side_compress);
    if (s == "lz4" || s == "true" || s == "1") {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::LZ4);
    } else if (s == "zstd") {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::ZSTD);
    } else if (s == "none" || s == "false" || s == "0" || s.empty()) {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::None);
    } else {
      throw std::invalid_argument("client_side_compress must be one of: 'none','lz4','zstd','true','false'");
    }
  }

  try {
    std::call_once(*cfg.schema_once,
                   [cfg, opts]() { RunSchemaInitOnce(cfg, opts); });

    // Per-thread client
    tl_client = std::make_unique<clickhouse::Client>(opts);
    tl_client->Ping();

    tl_client->Execute("USE " + QuoteIdent(cfg.database));
    ApplySessionSettings(*tl_client, cfg);
  } catch (const std::exception& e) {
    tl_client.reset();
    LogClickHouseInitFailure(cfg, e.what());
    throw;
  } catch (...) {
    tl_client.reset();
    LogClickHouseInitFailure(cfg, "unknown non-std exception");
    throw;
  }

  tl_worker_index = thread_idx;
  tl_runtime_metrics = cfg.runtime_metrics;
  tl_inited = true;
  tl_runtime_metrics->WorkerReady(thread_idx);
}

void ClickHouseInsertStage::ThreadCleanup() noexcept {
  if (!tl_inited || tl_cleaned) return;
  tl_cleaned = true;
  try { tl_client.reset(); } catch (...) {}
  tl_runtime_metrics.reset();
  tl_worker_index = -1;
}

void ClickHouseInsertStage::InsertBatch(std::vector<dmx_host_queue_item>&& batch) {
  clickhouse::Client& client = ClientOrThrow();
  if (batch.empty()) return;

  std::uint64_t logical_bytes = 0;
  for (const auto& item : batch) logical_bytes += item.item_size;
  const std::uint64_t row_count = batch.size();

  // Stage rows (keeps tensor memory alive for AppendNoManagedLifetime)
  std::vector<StagedRow> rows;
  rows.reserve(batch.size());
  for (auto& row : batch) {
    rows.emplace_back(StageOneRow(std::move(row.core)));
  }

  const std::string fq_table = QualifiedTableNameQuoted(tl_db, tl_table);

  // Build columns
  auto col_model_id = std::make_shared<clickhouse::ColumnString>();
  auto col_request_id = std::make_shared<clickhouse::ColumnString>();
  auto col_act_name = std::make_shared<clickhouse::ColumnString>();
  auto col_layer_no = std::make_shared<clickhouse::ColumnInt32>();
  auto col_shard_rank = std::make_shared<clickhouse::ColumnInt32>();
  auto col_start_token_idx = std::make_shared<clickhouse::ColumnInt32>();
  auto col_end_token_idx = std::make_shared<clickhouse::ColumnInt32>();

  auto col_dtype = std::make_shared<clickhouse::ColumnString>();

  // Array(Int64) = nested values column + offsets column (cumulative sizes)
  auto shape_values  = std::make_shared<clickhouse::ColumnInt64>();
  auto shape_offsets = std::make_shared<clickhouse::ColumnUInt64>();
  auto col_shape     = std::make_shared<clickhouse::ColumnArray>(shape_values, shape_offsets);
 

  auto col_bytes = std::make_shared<clickhouse::ColumnString>();

  uint64_t shape_off = 0;
  for (const auto& r : rows) {
    col_model_id->Append(r.model_id);
    col_request_id->Append(r.request_id);
    col_act_name->Append(r.act_name);

    col_layer_no->Append(r.layer_no);
    col_shard_rank->Append(r.shard_rank);
    col_start_token_idx->Append(r.start_token_idx);
    col_end_token_idx->Append(r.end_token_idx);

    col_dtype->Append(r.dtype);

    for (int64_t d : r.shape) {
      shape_values->Append(d);
    }
    shape_off += static_cast<uint64_t>(r.shape.size());
    shape_offsets->Append(shape_off);

    // Avoid copying large binary payloads (requires bytes_view lifetime until Insert returns)
    col_bytes->AppendNoManagedLifetime(r.bytes_view);
  }

  clickhouse::Block block;
  block.AppendColumn("model_id", col_model_id);
  block.AppendColumn("request_id", col_request_id);
  block.AppendColumn("act_name", col_act_name);
  block.AppendColumn("layer_no", col_layer_no);
  block.AppendColumn("shard_rank", col_shard_rank);
  block.AppendColumn("start_token_idx", col_start_token_idx);
  block.AppendColumn("end_token_idx", col_end_token_idx);
  block.AppendColumn("dtype", col_dtype);
  block.AppendColumn("shape", col_shape);
  block.AppendColumn("bytes", col_bytes);

  const auto started = std::chrono::steady_clock::now();
  tl_runtime_metrics->BeginInsert();
  try {
    client.Insert(fq_table, block);
  } catch (...) {
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    tl_runtime_metrics->EndInsert(tl_worker_index, row_count, logical_bytes, seconds);
    throw;
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  tl_runtime_metrics->EndInsert(tl_worker_index, row_count, logical_bytes, seconds);
}

// Force emission of the specialization referenced from bindings.cpp:
template std::optional<std::vector<dmx_host_queue_item>>
ClickHouseInsertStage::ProcessFn<DMXHostEngine::QueueT>(
    std::vector<dmx_host_queue_item>,
    DMXHostEngine::QueueT*);

}  // namespace dmx_host
