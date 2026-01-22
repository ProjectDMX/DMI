#include "clickhouse_client.h"

#include <iostream>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>


namespace dmx_host {
namespace py = pybind11;

namespace {

// Per-thread client (clickhouse::Client is NOT thread-safe). :contentReference[oaicite:1]{index=1}
thread_local std::unique_ptr<clickhouse::Client> tl_client;
thread_local bool tl_inited = false;
thread_local bool tl_cleaned = false;
thread_local std::string tl_db;
thread_local std::string tl_table;

// Only one thread performs DDL init.
std::once_flag g_schema_once;

// ---------- helpers ----------

std::string QuoteIdent(const std::string& ident) {
    std::string out;
    out.reserve(ident.size() + 2);
    out.push_back('`');
    for (char c : ident) {
        if (c == '`') {
            out.push_back('`');  // escape backtick by doubling
            out.push_back('`');
        } else {
            out.push_back(c);
        }
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
    if (name.empty()) {
        throw py::value_error("client_settings contains an empty key");
    }
    for (unsigned char uc : name) {
        char c = static_cast<char>(uc);
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.')) {
            throw py::value_error(
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

// Try to set compression in a way that works across clickhouse-cpp versions.
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

// Append bytes to ColumnString with minimal extra copying where possible.
template <typename ColumnT>
auto AppendBytesImpl(ColumnT& col, const char* data, size_t size, int)
    -> decltype(col.Append(data, size), void()) {
    col.Append(data, size);
}

template <typename ColumnT>
void AppendBytesImpl(ColumnT& col, const char* data, size_t size, long) {
    // Fallback: temporary std::string (may still be moved if an rvalue overload exists).
    col.Append(std::string(data, size));
}

template <typename ColumnT>
void AppendBytes(ColumnT& col, const char* data, size_t size) {
    AppendBytesImpl(col, data, size, 0);
}

struct BufferView {
    const char* ptr = nullptr;
    size_t size = 0;
};

// Expects a 1-D byte buffer (memoryview cast('B') from your torch_encode()).
BufferView GetByteBufferView(const py::handle& obj) {
    py::buffer buf = py::reinterpret_borrow<py::buffer>(obj);
    py::buffer_info info = buf.request();  // requires GIL

    if (info.itemsize != 1) {
        throw py::value_error("Expected a byte buffer (itemsize=1) for memoryview");
    }
    if (info.ndim != 1) {
        throw py::value_error("Expected a 1-D byte buffer (memoryview cast('B'))");
    }
    if (!info.strides.empty() && info.strides[0] != 1) {
        throw py::value_error("Expected a contiguous byte buffer (stride=1)");
    }

    BufferView v;
    v.ptr = static_cast<const char*>(info.ptr);
    v.size = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
    return v;
}

std::string QualifiedTableNameQuoted(const std::string& db, const std::string& table) {
    return QuoteIdent(db) + "." + QuoteIdent(table);
}

clickhouse::Client& ClientOrThrow() {
    if (!tl_inited || tl_cleaned || !tl_client) {
        throw std::runtime_error("ClickHouse client not initialized in this thread. "
                                "Call clickhouse_init(thread_idx, cfg) first.");
    }
    return *tl_client;
}

void RunSchemaInitOnce(const ClickHouseClientConfig& cfg) {
    // Executed only by the first thread entering clickhouse_init().
    // Use that thread's client.
    clickhouse::Client& client = ClientOrThrow();

    const std::string db_q = QuoteIdent(cfg.database);
    const std::string table_q = QuoteIdent(cfg.table);
    const std::string fq_table_q = db_q + "." + table_q;

    if (cfg.drop_existing_database) {
        client.Execute("DROP DATABASE IF EXISTS " + db_q);
    }

    if (cfg.create_database_if_missing || cfg.drop_existing_database) {
        client.Execute("CREATE DATABASE IF NOT EXISTS " + db_q);
    }

    // Hard-coded schema as requested.
    // Store byte blobs in String (ClickHouse String is binary-safe).
    std::ostringstream ddl;
    ddl
        << "CREATE TABLE IF NOT EXISTS " << fq_table_q << " ("
        << QuoteIdent("model_id") << " String, "
        << QuoteIdent("request_id") << " String, "
        << QuoteIdent("act_name") << " String, "
        << QuoteIdent("layer_no") << " Int32, "
        << QuoteIdent("start_token_idx") << " Int32, "
        << QuoteIdent("end_token_idx") << " Int32, "
        << QuoteIdent("json") << " String, "
        << QuoteIdent("bytes") << " String"
        << ") ENGINE = MergeTree "
        << "PRIMARY KEY ("
        << QuoteIdent("model_id") << ", "
        << QuoteIdent("request_id") << ", "
        << QuoteIdent("act_name") << ", "
        << QuoteIdent("layer_no") << ", "
        << QuoteIdent("start_token_idx") << ", "
        << QuoteIdent("end_token_idx")
        << ") "
        << "ORDER BY ("
        << QuoteIdent("model_id") << ", "
        << QuoteIdent("request_id") << ", "
        << QuoteIdent("act_name") << ", "
        << QuoteIdent("layer_no") << ", "
        << QuoteIdent("start_token_idx") << ", "
        << QuoteIdent("end_token_idx")
        << ") "
        << "SETTINGS index_granularity = " << cfg.index_granularity;

    client.Execute(ddl.str());
    std::cout << "DATABASE INIT DONE" << std::endl;
}

void ApplySessionSettings(clickhouse::Client& client, const ClickHouseClientConfig& cfg) {
    if (cfg.client_settings.is_none()) {
        return;
    }

    py::dict d = cfg.client_settings.cast<py::dict>();
    for (auto item : d) {
        std::string key = py::cast<std::string>(item.first);
        ValidateSettingName(key);

        py::handle v = item.second;
        std::string sql_value;

        if (py::isinstance<py::bool_>(v)) {
            bool bv = py::cast<bool>(v);
            sql_value = bv ? "1" : "0";
        } else if (py::isinstance<py::int_>(v)) {
            // bool is also int in Python, so bool is handled above first.
            long long iv = py::cast<long long>(v);
            sql_value = std::to_string(iv);
        } else if (py::isinstance<py::str>(v)) {
            std::string sv = py::cast<std::string>(v);
            sql_value = QuoteStringLiteral(sv);
        } else {
            throw py::value_error("client_settings value must be str/int/bool");
        }

        client.Execute("SET " + key + " = " + sql_value);
    }
}

}  // namespace

// ---------- exported functions ----------

void clickhouse_init(int thread_idx, const ClickHouseClientConfig& cfg) {
    (void)thread_idx;

    if (tl_inited) {
        throw std::runtime_error("clickhouse_init() called more than once in the same thread");
    }
    if (tl_cleaned) {
        throw std::runtime_error("clickhouse_init() called after clickhouse_cleanup() in the same thread");
    }

    tl_db = cfg.database;
    tl_table = cfg.table;

    clickhouse::ClientOptions opts;
    opts.SetHost(cfg.host);
    opts.SetPort(static_cast<uint16_t>(cfg.port));
    opts.SetUser(cfg.username);
    opts.SetPassword(cfg.password);

    // Use a stable default database for handshake; we always fully-qualify table names anyway.
    // (If you prefer, you can change this to cfg.database once you're sure it exists.)
    opts.SetDefaultDatabase("default");

    if (cfg.secure) {
        // Enable SSL/TLS (requires clickhouse-cpp built with OpenSSL). :contentReference[oaicite:2]{index=2}
        opts.SetSSLOptions(clickhouse::ClientOptions::SSLOptions{});
    }

    // Client-side compression config: False / True / "lz4" / "zstd" / "none"
    if (!cfg.client_side_compress.is_none()) {
        if (py::isinstance<py::bool_>(cfg.client_side_compress)) {
            bool enabled = cfg.client_side_compress.cast<bool>();
            if (enabled) {
                SetCompressionCompat(opts, clickhouse::CompressionMethod::LZ4);
            }
        } else if (py::isinstance<py::str>(cfg.client_side_compress)) {
            std::string s = ToLower(cfg.client_side_compress.cast<std::string>());
            if (s == "lz4") {
                SetCompressionCompat(opts, clickhouse::CompressionMethod::LZ4);
            } else if (s == "zstd") {
                SetCompressionCompat(opts, clickhouse::CompressionMethod::ZSTD);
            } else if (s == "none" || s == "false" || s == "0") {
                SetCompressionCompat(opts, clickhouse::CompressionMethod::None);
            } else {
                throw py::value_error("client_side_compress must be False/True or one of: 'lz4', 'zstd', 'none'");
            }
        } else {
            throw py::value_error("client_side_compress must be False/True or a string");
        }
    }

    // Construct client (network operations below release the GIL).
    tl_client = std::make_unique<clickhouse::Client>(opts);

    {
        py::gil_scoped_release release;
        tl_client->Ping();
    }

    tl_inited = true;

    // DDL init done once globally.
    std::call_once(g_schema_once, [&cfg]() {
        py::gil_scoped_release release;
        RunSchemaInitOnce(cfg);
    });

    // Validate / set session database for this thread's client (throws if DB missing).
    {
        py::gil_scoped_release release;
        tl_client->Execute("USE " + QuoteIdent(cfg.database));
        ApplySessionSettings(*tl_client, cfg);
    }
}

py::list clickhouse_insert(py::sequence items) {
    clickhouse::Client& client = ClientOrThrow();

    const size_t n = static_cast<size_t>(py::len(items));
    if (n == 0) {
        return py::list();
    }

    // 1. Create Columns
    auto col_model_id = std::make_shared<clickhouse::ColumnString>();
    auto col_request_id = std::make_shared<clickhouse::ColumnString>();
    auto col_act_name = std::make_shared<clickhouse::ColumnString>();
    auto col_layer_no = std::make_shared<clickhouse::ColumnInt32>();
    auto col_start_token_idx = std::make_shared<clickhouse::ColumnInt32>();
    auto col_end_token_idx = std::make_shared<clickhouse::ColumnInt32>();
    auto col_json = std::make_shared<clickhouse::ColumnString>();
    auto col_bytes = std::make_shared<clickhouse::ColumnString>();

    // 2. Populate Columns (Loop)
    // NOTE: We do NOT build the block here. We must fill the columns first.
    for (py::handle h_item : items) {
        py::object item = py::reinterpret_borrow<py::object>(h_item);

        // Expect StageTwoItemForDB with attribute _row (tuple of 8 fields).
        py::tuple row = item.attr("_row").cast<py::tuple>();
        if (row.size() != 8) {
            throw py::value_error("StageTwoItemForDB._row must be a tuple of length 8");
        }

        // (str, str, str, int, int, int, memoryview, memoryview)
        const std::string model_id = row[0].cast<std::string>();
        const std::string request_id = row[1].cast<std::string>();
        const std::string act_name = row[2].cast<std::string>();
        const int32_t layer_no = row[3].cast<int32_t>();
        const int32_t start_token_idx = row[4].cast<int32_t>();
        const int32_t end_token_idx = row[5].cast<int32_t>();

        // Bytes fields: assume C-contiguous 1-D byte buffers.
        py::object json_obj = py::reinterpret_borrow<py::object>(row[6]);
        py::object bytes_obj = py::reinterpret_borrow<py::object>(row[7]);

        BufferView json_view = GetByteBufferView(json_obj);
        BufferView bytes_view = GetByteBufferView(bytes_obj);

        col_model_id->Append(model_id);
        col_request_id->Append(request_id);
        col_act_name->Append(act_name);
        col_layer_no->Append(layer_no);
        col_start_token_idx->Append(start_token_idx);
        col_end_token_idx->Append(end_token_idx);

        // Append binary-safe.
        AppendBytes(*col_json, json_view.ptr ? json_view.ptr : "", json_view.size);
        AppendBytes(*col_bytes, bytes_view.ptr ? bytes_view.ptr : "", bytes_view.size);
    }

    // 3. Build block (MOVED HERE)
    // Constructing the block now ensures it detects the correct row count (n > 0).
    clickhouse::Block block;
    block.AppendColumn("model_id", col_model_id);
    block.AppendColumn("request_id", col_request_id);
    block.AppendColumn("act_name", col_act_name);
    block.AppendColumn("layer_no", col_layer_no);
    block.AppendColumn("start_token_idx", col_start_token_idx);
    block.AppendColumn("end_token_idx", col_end_token_idx);
    block.AppendColumn("json", col_json);
    block.AppendColumn("bytes", col_bytes);

    // 4. Insert: release GIL for network IO.
    const std::string fq_table = QualifiedTableNameQuoted(tl_db, tl_table);
    {
        py::gil_scoped_release release;
        client.Insert(fq_table, block);
    }

    return py::list();
}

void clickhouse_cleanup() {
    if (!tl_inited) {
        throw std::runtime_error("clickhouse_cleanup() called before clickhouse_init() in this thread");
    }
    if (tl_cleaned) {
        throw std::runtime_error("clickhouse_cleanup() called more than once in the same thread");
    }

    tl_cleaned = true;

    // Destruct client (close connections) without holding GIL.
    {
        py::gil_scoped_release release;
        tl_client.reset();
    }
}

// ---------- module definition ----------

PYBIND11_MODULE(clickhouse_client, m) {
    m.doc() = "DMX ClickHouse interface (thread-local clickhouse-cpp client)";

    py::class_<ClickHouseClientConfig>(m, "ClickHouseClientConfig")
        .def(py::init<>())
        .def_readwrite("host", &ClickHouseClientConfig::host)
        .def_readwrite("port", &ClickHouseClientConfig::port)
        .def_readwrite("username", &ClickHouseClientConfig::username)
        .def_readwrite("password", &ClickHouseClientConfig::password)
        .def_readwrite("database", &ClickHouseClientConfig::database)
        .def_readwrite("table", &ClickHouseClientConfig::table)
        .def_readwrite("secure", &ClickHouseClientConfig::secure)
        .def_readwrite("client_settings", &ClickHouseClientConfig::client_settings)
        .def_readwrite("create_database_if_missing", &ClickHouseClientConfig::create_database_if_missing)
        .def_readwrite("drop_existing_database", &ClickHouseClientConfig::drop_existing_database)
        .def_readwrite("client_side_compress", &ClickHouseClientConfig::client_side_compress)
        .def_readwrite("index_granularity", &ClickHouseClientConfig::index_granularity);

    m.def("clickhouse_init", &clickhouse_init,
          py::arg("thread_idx"), py::arg("thread_init_config"),
          "Initialize a thread-local ClickHouse client and (once) create DB/table schema.");

    m.def("clickhouse_insert", &clickhouse_insert,
          py::arg("items"),
          "Insert StageTwoItemForDB rows into ClickHouse. Returns [] (empty list).");

    m.def("clickhouse_cleanup", &clickhouse_cleanup,
          "Cleanup thread-local ClickHouse client. Must be called once per thread.");
}

}  // namespace dmx_host
