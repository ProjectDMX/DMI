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
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
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
thread_local std::optional<RecordSchema> tl_record_schema;

std::mutex g_schema_mutex;
std::set<std::string> g_schema_inited_dbs;
std::set<std::string> g_schema_inited_tables;

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

void RunDbInitOnce(const ClickHouseClientConfig& cfg, const clickhouse::ClientOptions& opts) {
  auto client = std::make_unique<clickhouse::Client>(opts);
  const std::string db_q = QuoteIdent(cfg.database);

  if (cfg.drop_existing_database) {
    client->Execute("DROP DATABASE IF EXISTS " + db_q);
  }

  if (cfg.create_database_if_missing || cfg.drop_existing_database) {
    client->Execute("CREATE DATABASE IF NOT EXISTS " + db_q);
  }
}

void RunTableInitOnce(const ClickHouseClientConfig& cfg, const clickhouse::ClientOptions& opts) {
  auto client = std::make_unique<clickhouse::Client>(opts);

  const std::string db_q = QuoteIdent(cfg.database);
  const std::string table_q = QuoteIdent(cfg.table);
  const std::string fq_table_q = db_q + "." + table_q;

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

bool IsSafeIdentifier(const std::string& value) {
  if (value.empty()) return false;
  const auto first = static_cast<unsigned char>(value.front());
  if (!(std::isalpha(first) || value.front() == '_')) return false;
  return std::all_of(value.begin() + 1, value.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_';
  });
}

const char* ClickHouseTypeName(RecordCellType type) {
  switch (type) {
    case RecordCellType::STRING: return "String";
    case RecordCellType::INT32: return "Int32";
    case RecordCellType::INT64: return "Int64";
    case RecordCellType::FLOAT64: return "Float64";
    case RecordCellType::INT64_ARRAY: return "Array(Int64)";
    case RecordCellType::TENSOR: break;
  }
  throw std::invalid_argument("TENSOR expands to three physical columns");
}

std::vector<std::pair<std::string, std::string>> PhysicalColumns(
    const RecordLayout& layout) {
  std::vector<std::pair<std::string, std::string>> result;
  for (const auto& column : layout.columns) {
    if (column.type == RecordCellType::TENSOR) {
      result.emplace_back(column.dtype_column, "String");
      result.emplace_back(column.shape_column, "Array(Int64)");
      result.emplace_back(column.bytes_column, "String");
    } else {
      result.emplace_back(column.name, ClickHouseTypeName(column.type));
    }
  }
  return result;
}

std::string JoinQuotedIdentifiers(const std::vector<std::string>& names) {
  std::ostringstream out;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i != 0) out << ", ";
    out << QuoteIdent(names[i]);
  }
  return out.str();
}

std::string RecordLayoutSignature(const RecordLayout& layout,
                                  int index_granularity) {
  std::ostringstream out;
  for (const auto& [name, type] : PhysicalColumns(layout)) {
    out << name << '\0' << type << '\0';
  }
  out << "pk\0";
  for (const auto& name : layout.primary_key) out << name << '\0';
  out << "order\0";
  for (const auto& name : layout.order_by) out << name << '\0';
  out << index_granularity;
  return out.str();
}

void AppendIdentityField(std::string& identity, std::string_view value) {
  identity.append(std::to_string(value.size()));
  identity.push_back(':');
  identity.append(value.data(), value.size());
}

std::string RecordSchemaIdentityImpl(const RecordSchema& schema) {
  std::vector<const RecordLayout*> layouts;
  layouts.reserve(schema.layouts.size());
  for (const auto& layout : schema.layouts) layouts.push_back(&layout);
  std::sort(layouts.begin(), layouts.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->name < rhs->name;
  });

  std::string identity = "dmi-record-schema-v1";
  AppendIdentityField(identity, std::to_string(schema.index_granularity));
  AppendIdentityField(identity, std::to_string(layouts.size()));
  for (const auto* layout : layouts) {
    AppendIdentityField(identity, layout->name);
    AppendIdentityField(identity, layout->table);
    AppendIdentityField(identity, std::to_string(layout->columns.size()));
    for (const auto& column : layout->columns) {
      AppendIdentityField(identity, column.name);
      AppendIdentityField(
          identity,
          std::to_string(static_cast<unsigned int>(column.type)));
      if (column.type == RecordCellType::TENSOR) {
        AppendIdentityField(identity, column.dtype_column);
        AppendIdentityField(identity, column.shape_column);
        AppendIdentityField(identity, column.bytes_column);
      }
    }
    AppendIdentityField(identity, std::to_string(layout->primary_key.size()));
    for (const auto& name : layout->primary_key) {
      AppendIdentityField(identity, name);
    }
    AppendIdentityField(identity, std::to_string(layout->order_by.size()));
    for (const auto& name : layout->order_by) {
      AppendIdentityField(identity, name);
    }
  }
  return identity;
}

void ValidateRecordSchemaImpl(const RecordSchema& schema) {
  if (schema.layouts.empty()) {
    throw std::invalid_argument("RecordSchema must contain at least one layout");
  }
  if (schema.index_granularity <= 0) {
    throw std::invalid_argument("RecordSchema index_granularity must be positive");
  }

  std::set<std::string> layout_names;
  std::unordered_map<std::string, std::string> table_signatures;
  for (const auto& layout : schema.layouts) {
    if (!IsSafeIdentifier(layout.name)) {
      throw std::invalid_argument("record layout name is not a safe identifier: " +
                                  layout.name);
    }
    if (!layout_names.insert(layout.name).second) {
      throw std::invalid_argument("duplicate record layout name: " + layout.name);
    }
    if (!IsSafeIdentifier(layout.table)) {
      throw std::invalid_argument("record table is not a safe identifier: " +
                                  layout.table);
    }
    if (layout.columns.empty()) {
      throw std::invalid_argument("record layout has no columns: " + layout.name);
    }
    if (layout.primary_key.empty() || layout.order_by.empty()) {
      throw std::invalid_argument(
          "record layout requires non-empty primary_key and order_by: " +
          layout.name);
    }

    std::set<std::string> logical_names;
    std::set<std::string> physical_names;
    for (const auto& column : layout.columns) {
      if (!IsSafeIdentifier(column.name)) {
        throw std::invalid_argument("record column is not a safe identifier: " +
                                    column.name);
      }
      if (!logical_names.insert(column.name).second) {
        throw std::invalid_argument("duplicate logical record column: " +
                                    column.name);
      }
      if (column.type == RecordCellType::TENSOR) {
        for (const auto* name : {&column.dtype_column, &column.shape_column,
                                 &column.bytes_column}) {
          if (!IsSafeIdentifier(*name)) {
            throw std::invalid_argument(
                "tensor physical column is not a safe identifier: " + *name);
          }
          if (!physical_names.insert(*name).second) {
            throw std::invalid_argument("duplicate physical record column: " +
                                        *name);
          }
        }
      } else if (!physical_names.insert(column.name).second) {
        throw std::invalid_argument("duplicate physical record column: " +
                                    column.name);
      }
    }

    for (const auto& key : layout.primary_key) {
      if (!physical_names.count(key)) {
        throw std::invalid_argument("primary-key column is not declared: " + key);
      }
    }
    for (const auto& key : layout.order_by) {
      if (!physical_names.count(key)) {
        throw std::invalid_argument("ordering-key column is not declared: " + key);
      }
    }

    const std::string signature =
        RecordLayoutSignature(layout, schema.index_granularity);
    const auto [it, inserted] =
        table_signatures.emplace(layout.table, signature);
    if (!inserted && it->second != signature) {
      throw std::invalid_argument(
          "record layouts targeting the same table must declare the same "
          "physical schema and keys: " + layout.table);
    }
  }
}

const RecordLayout& FindRecordLayoutImpl(const RecordSchema& schema,
                                         const std::string& name) {
  const auto it = std::find_if(
      schema.layouts.begin(), schema.layouts.end(),
      [&](const RecordLayout& layout) { return layout.name == name; });
  if (it == schema.layouts.end()) {
    throw std::invalid_argument("unknown record layout: " + name);
  }
  return *it;
}

std::optional<std::vector<std::string>> ParseIdentifierKey(
    std::string_view expression) {
  std::vector<std::string> names;
  std::size_t cursor = 0;
  while (cursor < expression.size()) {
    while (cursor < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[cursor]))) {
      ++cursor;
    }
    if (cursor == expression.size()) break;

    std::string name;
    if (expression[cursor] == '`') {
      ++cursor;
      bool closed = false;
      while (cursor < expression.size()) {
        const char character = expression[cursor++];
        if (character != '`') {
          name.push_back(character);
          continue;
        }
        if (cursor < expression.size() && expression[cursor] == '`') {
          name.push_back('`');
          ++cursor;
          continue;
        }
        closed = true;
        break;
      }
      if (!closed) return std::nullopt;
    } else {
      const std::size_t begin = cursor;
      while (cursor < expression.size()) {
        const unsigned char character =
            static_cast<unsigned char>(expression[cursor]);
        if (!std::isalnum(character) && expression[cursor] != '_') break;
        ++cursor;
      }
      if (cursor == begin) return std::nullopt;
      name.assign(expression.substr(begin, cursor - begin));
    }
    if (!IsSafeIdentifier(name)) return std::nullopt;
    names.push_back(std::move(name));

    while (cursor < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[cursor]))) {
      ++cursor;
    }
    if (cursor == expression.size()) break;
    if (expression[cursor] != ',') return std::nullopt;
    ++cursor;
    std::size_t next = cursor;
    while (next < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[next]))) {
      ++next;
    }
    if (next == expression.size()) return std::nullopt;
  }
  return names;
}

std::optional<int> ParsePositiveIntegerSetting(
    std::string_view text, std::string_view setting) {
  std::size_t search_from = 0;
  while (search_from < text.size()) {
    const std::size_t found = text.find(setting, search_from);
    if (found == std::string_view::npos) return std::nullopt;
    const std::size_t after_name = found + setting.size();
    const bool left_boundary =
        found == 0 || (!std::isalnum(static_cast<unsigned char>(text[found - 1])) &&
                       text[found - 1] != '_');
    const bool right_boundary =
        after_name == text.size() ||
        (!std::isalnum(static_cast<unsigned char>(text[after_name])) &&
         text[after_name] != '_');
    if (!left_boundary || !right_boundary) {
      search_from = after_name;
      continue;
    }

    std::size_t cursor = after_name;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor]))) {
      ++cursor;
    }
    if (cursor == text.size() || text[cursor] != '=') {
      search_from = after_name;
      continue;
    }
    ++cursor;
    while (cursor < text.size() &&
           std::isspace(static_cast<unsigned char>(text[cursor]))) {
      ++cursor;
    }
    const char* begin = text.data() + cursor;
    const char* end = text.data() + text.size();
    int parsed = 0;
    const auto [ptr, error] = std::from_chars(begin, end, parsed);
    const char* suffix = ptr;
    while (suffix < end &&
           std::isspace(static_cast<unsigned char>(*suffix))) {
      ++suffix;
    }
    if (error == std::errc{} && ptr != begin && parsed > 0 &&
        (suffix == end || *suffix == ',')) {
      return parsed;
    }
    throw std::runtime_error(
        "live ClickHouse table has an invalid index_granularity setting");
  }
  return std::nullopt;
}

int DefaultIndexGranularity(clickhouse::Client& client) {
  std::optional<int> granularity;
  client.Select(
      "SELECT toString(value) FROM system.merge_tree_settings "
      "WHERE name = 'index_granularity'",
      [&](const clickhouse::Block& block) {
        if (block.GetRowCount() == 0 || block.GetColumnCount() == 0) {
          return;
        }
        if (block.GetColumnCount() != 1) {
          throw std::runtime_error(
              "ClickHouse merge-tree settings query returned an unexpected "
              "column count");
        }
        const auto values =
            block[0]->AsStrict<clickhouse::ColumnString>();
        for (std::size_t i = 0; i < block.GetRowCount(); ++i) {
          if (granularity) {
            throw std::runtime_error(
                "ClickHouse returned duplicate index_granularity settings");
          }
          const std::string value(values->At(i));
          int parsed = 0;
          const auto [ptr, error] = std::from_chars(
              value.data(), value.data() + value.size(), parsed);
          if (error != std::errc{} || ptr != value.data() + value.size() ||
              parsed <= 0) {
            throw std::runtime_error(
                "ClickHouse returned an invalid default index_granularity");
          }
          granularity = parsed;
        }
      });
  if (!granularity) {
    throw std::runtime_error(
        "ClickHouse did not return a default index_granularity");
  }
  return *granularity;
}

void ValidateLiveRecordTable(
    clickhouse::Client& client, const std::string& database,
    const RecordLayout& layout, int index_granularity) {
  const std::string qualified_table =
      QualifiedTableNameQuoted(database, layout.table);
  std::vector<std::pair<std::string, std::string>> actual;
  client.Select("DESCRIBE TABLE " + qualified_table,
                [&](const clickhouse::Block& block) {
    if (block.GetRowCount() == 0 || block.GetColumnCount() == 0) {
      return;
    }
    if (block.GetColumnCount() < 2) {
      throw std::runtime_error(
          "ClickHouse DESCRIBE returned fewer than two columns");
    }
    const auto names = block[0]->AsStrict<clickhouse::ColumnString>();
    const auto types = block[1]->AsStrict<clickhouse::ColumnString>();
    for (std::size_t i = 0; i < block.GetRowCount(); ++i) {
      actual.emplace_back(std::string(names->At(i)),
                          std::string(types->At(i)));
    }
  });

  const auto expected = PhysicalColumns(layout);
  if (actual != expected) {
    throw std::runtime_error(
        "live ClickHouse columns/types do not match record layout '" +
        layout.name + "'");
  }

  std::optional<std::string> primary_key;
  std::optional<std::string> sorting_key;
  std::optional<std::string> engine_full;
  const std::string metadata_query =
      "SELECT toString(primary_key), toString(sorting_key), "
      "toString(engine_full) FROM system.tables "
      "WHERE database = " +
      QuoteStringLiteral(database) + " AND name = " +
      QuoteStringLiteral(layout.table);
  client.Select(metadata_query, [&](const clickhouse::Block& block) {
    if (block.GetRowCount() == 0 || block.GetColumnCount() == 0) {
      return;
    }
    if (block.GetColumnCount() != 3) {
      throw std::runtime_error(
          "ClickHouse table metadata query returned an unexpected column "
          "count");
    }
    const auto primary =
        block[0]->AsStrict<clickhouse::ColumnString>();
    const auto sorting =
        block[1]->AsStrict<clickhouse::ColumnString>();
    const auto engine =
        block[2]->AsStrict<clickhouse::ColumnString>();
    for (std::size_t i = 0; i < block.GetRowCount(); ++i) {
      if (primary_key) {
        throw std::runtime_error(
            "ClickHouse returned duplicate live-table metadata rows");
      }
      primary_key = std::string(primary->At(i));
      sorting_key = std::string(sorting->At(i));
      engine_full = std::string(engine->At(i));
    }
  });
  if (!primary_key || !sorting_key || !engine_full) {
    throw std::runtime_error(
        "ClickHouse did not return metadata for record table '" +
        layout.table + "'");
  }

  const auto parsed_primary_key = ParseIdentifierKey(*primary_key);
  if (!parsed_primary_key || *parsed_primary_key != layout.primary_key) {
    throw std::runtime_error(
        "live ClickHouse primary key does not match record layout '" +
        layout.name + "'");
  }
  const auto parsed_sorting_key = ParseIdentifierKey(*sorting_key);
  if (!parsed_sorting_key || *parsed_sorting_key != layout.order_by) {
    throw std::runtime_error(
        "live ClickHouse ORDER BY does not match record layout '" +
        layout.name + "'");
  }
  const auto explicit_granularity =
      ParsePositiveIntegerSetting(*engine_full, "index_granularity");
  const int live_granularity = explicit_granularity
      ? *explicit_granularity
      : DefaultIndexGranularity(client);
  if (live_granularity != index_granularity) {
    throw std::runtime_error(
        "live ClickHouse index_granularity does not match record schema");
  }
}

void InitializeAndValidateRecordTables(
    const ClickHouseClientConfig& cfg, const RecordSchema& schema,
    const clickhouse::ClientOptions& opts) {
  auto client = std::make_unique<clickhouse::Client>(opts);
  std::set<std::string> visited_tables;
  for (const auto& layout : schema.layouts) {
    if (!visited_tables.insert(layout.table).second) continue;

    const std::string fq_table =
        QualifiedTableNameQuoted(cfg.database, layout.table);
    const auto physical_columns = PhysicalColumns(layout);

    std::ostringstream ddl;
    ddl << "CREATE TABLE IF NOT EXISTS " << fq_table << " (";
    for (std::size_t i = 0; i < physical_columns.size(); ++i) {
      if (i != 0) ddl << ", ";
      ddl << QuoteIdent(physical_columns[i].first) << " "
          << physical_columns[i].second;
    }
    ddl << ") ENGINE = MergeTree PRIMARY KEY ("
        << JoinQuotedIdentifiers(layout.primary_key) << ") ORDER BY ("
        << JoinQuotedIdentifiers(layout.order_by)
        << ") SETTINGS index_granularity = " << schema.index_granularity;

    client->Execute(ddl.str());
    ValidateLiveRecordTable(
        *client, cfg.database, layout, schema.index_granularity);
  }
}

using StagedRecordValue =
    std::variant<std::string, int32_t, int64_t, double,
                 std::vector<int64_t>, EncodedTensor>;

struct StagedRecordRow {
  std::vector<StagedRecordValue> cells;
};

StagedRecordRow StageGenericRecordRow(GenericRecordRow&& row,
                                      const RecordLayout& layout) {
  if (row.cells.size() != layout.columns.size()) {
    throw std::invalid_argument(
        "generic record cell count does not match layout '" + layout.name +
        "'");
  }

  StagedRecordRow staged;
  staged.cells.reserve(row.cells.size());
  for (std::size_t i = 0; i < row.cells.size(); ++i) {
    auto& value = row.cells[i];
    const auto type = layout.columns[i].type;
    switch (type) {
      case RecordCellType::STRING:
        if (!std::holds_alternative<std::string>(value))
          throw std::invalid_argument("record STRING cell has wrong value type");
        staged.cells.emplace_back(
            std::move(std::get<std::string>(value)));
        break;
      case RecordCellType::INT32:
        if (!std::holds_alternative<int32_t>(value))
          throw std::invalid_argument("record INT32 cell has wrong value type");
        staged.cells.emplace_back(std::get<int32_t>(value));
        break;
      case RecordCellType::INT64:
        if (!std::holds_alternative<int64_t>(value))
          throw std::invalid_argument("record INT64 cell has wrong value type");
        staged.cells.emplace_back(std::get<int64_t>(value));
        break;
      case RecordCellType::FLOAT64:
        if (!std::holds_alternative<double>(value))
          throw std::invalid_argument("record FLOAT64 cell has wrong value type");
        staged.cells.emplace_back(std::get<double>(value));
        break;
      case RecordCellType::INT64_ARRAY:
        if (!std::holds_alternative<std::vector<int64_t>>(value))
          throw std::invalid_argument(
              "record INT64_ARRAY cell has wrong value type");
        staged.cells.emplace_back(
            std::move(std::get<std::vector<int64_t>>(value)));
        break;
      case RecordCellType::TENSOR:
        if (!std::holds_alternative<at::Tensor>(value))
          throw std::invalid_argument("record TENSOR cell has wrong value type");
        staged.cells.emplace_back(
            EncodeTensorToColumns(std::get<at::Tensor>(value)));
        break;
    }
  }
  return staged;
}

}  // namespace

void ValidateRecordSchema(const RecordSchema& schema) {
  ValidateRecordSchemaImpl(schema);
}

std::string RecordSchemaIdentity(const RecordSchema& schema) {
  ValidateRecordSchemaImpl(schema);
  return RecordSchemaIdentityImpl(schema);
}

const RecordLayout& FindRecordLayout(const RecordSchema& schema,
                                     const std::string& name) {
  return FindRecordLayoutImpl(schema, name);
}

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
    // DDL init: DB-level ops guarded once per (host, port, secure, database),
    // table-level ops guarded once per (host, port, secure, database, table).
    // Null-byte separators avoid collisions with identifiers that contain dots.
    // The endpoint prefix ensures engines targeting different servers are not
    // incorrectly treated as already initialised.
    {
      const std::string endpoint_prefix =
          cfg.host + '\0' + std::to_string(cfg.port) + '\0' +
          (cfg.secure ? "1" : "0") + '\0';
      const std::string db_key = endpoint_prefix + cfg.database;
      const std::string table_key = db_key + '\0' + cfg.table;

      std::lock_guard<std::mutex> lock(g_schema_mutex);
      if (g_schema_inited_dbs.find(db_key) == g_schema_inited_dbs.end()) {
        RunDbInitOnce(cfg, opts);
        // Only mark as initialised when DB-level DDL was (or would be) executed.
        // If neither flag is set, no DDL runs; skip the mark so a later engine
        // that does set one of these flags is not incorrectly suppressed.
        if (cfg.create_database_if_missing || cfg.drop_existing_database) {
          g_schema_inited_dbs.insert(db_key);
        }
      }
      if (g_schema_inited_tables.find(table_key) == g_schema_inited_tables.end()) {
        RunTableInitOnce(cfg, opts);
        g_schema_inited_tables.insert(table_key);
      }
    }

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
  tl_record_schema.reset();
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
    if (!std::holds_alternative<ClickHouseRow>(row.row)) {
      throw std::invalid_argument(
          "fixed ClickHouse insert stage received a generic record row");
    }
    rows.emplace_back(
        StageOneRow(std::move(std::get<ClickHouseRow>(row.row))));
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

void ClickHouseRecordInsertStage::ThreadInit(
    int thread_idx, const ClickHouseRecordStageConfig& stage_cfg) {
  if (tl_inited) {
    throw std::runtime_error(
        "ClickHouseRecordInsertStage::ThreadInit() called more than once in "
        "the same thread");
  }
  if (tl_cleaned) {
    throw std::runtime_error(
        "ClickHouseRecordInsertStage::ThreadInit() called after "
        "ThreadCleanup() in the same thread");
  }

  ValidateRecordSchema(stage_cfg.schema);
  const ClickHouseClientConfig& cfg = stage_cfg.client;
  tl_db = cfg.database;
  tl_table.clear();

  clickhouse::ClientOptions opts;
  opts.SetHost(cfg.host);
  opts.SetPort(static_cast<uint16_t>(cfg.port));
  opts.SetUser(cfg.username);
  opts.SetPassword(cfg.password);
  if (cfg.connect_timeout_ms < 0 || cfg.receive_timeout_ms < 0 ||
      cfg.send_timeout_ms < 0) {
    throw std::invalid_argument("ClickHouse socket timeouts must be non-negative");
  }
  opts.SetConnectionConnectTimeout(
      std::chrono::milliseconds(cfg.connect_timeout_ms));
  opts.SetConnectionRecvTimeout(
      std::chrono::milliseconds(cfg.receive_timeout_ms));
  opts.SetConnectionSendTimeout(
      std::chrono::milliseconds(cfg.send_timeout_ms));
  opts.SetDefaultDatabase("default");
  if (cfg.secure) {
    opts.SetSSLOptions(clickhouse::ClientOptions::SSLOptions{});
  }
  {
    const std::string compression = ToLower(cfg.client_side_compress);
    if (compression == "lz4" || compression == "true" ||
        compression == "1") {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::LZ4);
    } else if (compression == "zstd") {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::ZSTD);
    } else if (compression == "none" || compression == "false" ||
               compression == "0" || compression.empty()) {
      SetCompressionCompat(opts, clickhouse::CompressionMethod::None);
    } else {
      throw std::invalid_argument(
          "client_side_compress must be one of: "
          "'none','lz4','zstd','true','false'");
    }
  }

  try {
    const std::string endpoint_prefix =
        cfg.host + '\0' + std::to_string(cfg.port) + '\0' +
        (cfg.secure ? "1" : "0") + '\0';
    const std::string db_key = endpoint_prefix + cfg.database;

    {
      std::lock_guard<std::mutex> lock(g_schema_mutex);
      if (g_schema_inited_dbs.find(db_key) == g_schema_inited_dbs.end()) {
        RunDbInitOnce(cfg, opts);
        if (cfg.create_database_if_missing || cfg.drop_existing_database) {
          g_schema_inited_dbs.insert(db_key);
        }
      }

      // This is deliberately a live startup check, not a process-lifetime
      // schema cache: a table may be dropped or recreated between engine
      // instances in the same process.
      InitializeAndValidateRecordTables(cfg, stage_cfg.schema, opts);
    }

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

  tl_record_schema = stage_cfg.schema;
  tl_worker_index = thread_idx;
  tl_runtime_metrics = cfg.runtime_metrics;
  tl_inited = true;
  tl_runtime_metrics->WorkerReady(thread_idx);
}

void ClickHouseRecordInsertStage::ThreadCleanup() noexcept {
  ClickHouseInsertStage::ThreadCleanup();
}

void ClickHouseRecordInsertStage::InsertBatch(
    std::vector<dmx_host_queue_item>&& batch) {
  clickhouse::Client& client = ClientOrThrow();
  if (batch.empty()) return;
  if (!tl_record_schema) {
    throw std::runtime_error("record schema is not initialized in this worker");
  }

  struct LayoutBatch {
    const RecordLayout* layout = nullptr;
    std::vector<StagedRecordRow> rows;
    std::uint64_t logical_bytes = 0;
  };
  std::map<std::string, LayoutBatch> grouped;

  for (auto& item : batch) {
    if (!std::holds_alternative<GenericRecordRow>(item.row)) {
      throw std::invalid_argument(
          "schema-driven ClickHouse stage received a fixed inference row");
    }
    GenericRecordRow record =
        std::move(std::get<GenericRecordRow>(item.row));
    const RecordLayout& layout =
        FindRecordLayout(*tl_record_schema, record.layout);
    auto& group = grouped[layout.name];
    group.layout = &layout;
    group.logical_bytes += item.item_size;
    group.rows.emplace_back(
        StageGenericRecordRow(std::move(record), layout));
  }

  for (auto& [layout_name, group] : grouped) {
    (void)layout_name;
    const RecordLayout& layout = *group.layout;
    clickhouse::Block block;

    for (std::size_t column_index = 0;
         column_index < layout.columns.size(); ++column_index) {
      const RecordColumn& definition = layout.columns[column_index];
      switch (definition.type) {
        case RecordCellType::STRING: {
          auto column = std::make_shared<clickhouse::ColumnString>();
          for (const auto& row : group.rows)
            column->Append(std::get<std::string>(row.cells[column_index]));
          block.AppendColumn(definition.name, column);
          break;
        }
        case RecordCellType::INT32: {
          auto column = std::make_shared<clickhouse::ColumnInt32>();
          for (const auto& row : group.rows)
            column->Append(std::get<int32_t>(row.cells[column_index]));
          block.AppendColumn(definition.name, column);
          break;
        }
        case RecordCellType::INT64: {
          auto column = std::make_shared<clickhouse::ColumnInt64>();
          for (const auto& row : group.rows)
            column->Append(std::get<int64_t>(row.cells[column_index]));
          block.AppendColumn(definition.name, column);
          break;
        }
        case RecordCellType::FLOAT64: {
          auto column = std::make_shared<clickhouse::ColumnFloat64>();
          for (const auto& row : group.rows)
            column->Append(std::get<double>(row.cells[column_index]));
          block.AppendColumn(definition.name, column);
          break;
        }
        case RecordCellType::INT64_ARRAY: {
          auto values = std::make_shared<clickhouse::ColumnInt64>();
          auto offsets = std::make_shared<clickhouse::ColumnUInt64>();
          auto column = std::make_shared<clickhouse::ColumnArray>(values, offsets);
          std::uint64_t offset = 0;
          for (const auto& row : group.rows) {
            const auto& array =
                std::get<std::vector<int64_t>>(row.cells[column_index]);
            for (const auto value : array) values->Append(value);
            offset += static_cast<std::uint64_t>(array.size());
            offsets->Append(offset);
          }
          block.AppendColumn(definition.name, column);
          break;
        }
        case RecordCellType::TENSOR: {
          auto dtype = std::make_shared<clickhouse::ColumnString>();
          auto shape_values = std::make_shared<clickhouse::ColumnInt64>();
          auto shape_offsets = std::make_shared<clickhouse::ColumnUInt64>();
          auto shape = std::make_shared<clickhouse::ColumnArray>(
              shape_values, shape_offsets);
          auto bytes = std::make_shared<clickhouse::ColumnString>();
          std::uint64_t shape_offset = 0;
          for (const auto& row : group.rows) {
            const auto& tensor =
                std::get<EncodedTensor>(row.cells[column_index]);
            dtype->Append(tensor.dtype);
            for (const auto dimension : tensor.shape)
              shape_values->Append(dimension);
            shape_offset += static_cast<std::uint64_t>(tensor.shape.size());
            shape_offsets->Append(shape_offset);
            bytes->AppendNoManagedLifetime(tensor.bytes_view);
          }
          block.AppendColumn(definition.dtype_column, dtype);
          block.AppendColumn(definition.shape_column, shape);
          block.AppendColumn(definition.bytes_column, bytes);
          break;
        }
      }
    }

    const std::string table =
        QualifiedTableNameQuoted(tl_db, layout.table);
    const std::uint64_t row_count = group.rows.size();
    const auto started = std::chrono::steady_clock::now();
    tl_runtime_metrics->BeginInsert();
    try {
      client.Insert(table, block);
    } catch (...) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      tl_runtime_metrics->EndInsert(tl_worker_index, row_count,
                                    group.logical_bytes, seconds);
      throw;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    tl_runtime_metrics->EndInsert(tl_worker_index, row_count,
                                  group.logical_bytes, seconds);
  }
}

// Force emission of the specialization referenced from bindings.cpp:
template std::optional<std::vector<dmx_host_queue_item>>
ClickHouseInsertStage::ProcessFn<DMXHostEngine::QueueT>(
    std::vector<dmx_host_queue_item>,
    DMXHostEngine::QueueT*);

template std::optional<std::vector<dmx_host_queue_item>>
ClickHouseRecordInsertStage::ProcessFn<DMXHostEngine::QueueT>(
    std::vector<dmx_host_queue_item>,
    DMXHostEngine::QueueT*);

}  // namespace dmx_host
