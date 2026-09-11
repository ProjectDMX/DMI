#include "clickhouse_client.h"

#include <cstdio>

#include "../common/curl_init.h"
#include "sql_escape.h"

namespace dmi_catalog {

namespace {

size_t write_body(char* ptr, size_t size, size_t nmemb, void* userp) {
  static_cast<std::string*>(userp)->append(ptr, size * nmemb);
  return size * nmemb;
}

std::string url_encode(const std::string& value) {
  char* escaped = curl_easy_escape(nullptr, value.c_str(),
                                   static_cast<int>(value.size()));
  if (escaped == nullptr) {
    // curl_easy_escape returns null on allocation failure; constructing
    // a std::string from null is undefined behavior.
    throw ClickHouseError("libcurl failed to escape a URL component");
  }
  const std::string out(escaped);
  curl_free(escaped);
  return out;
}

}  // namespace

std::string substitute(const std::string& query, const Params& params) {
  // ONE left-to-right pass, appending to an output buffer, because
  // clickhouse-driver's substitution is `query % escaped`
  // (`clickhouse_driver/client.py`) and `%` is a single pass: rendered
  // text is never looked at again.
  //
  // Rendering one parameter at a time across the whole statement is NOT
  // that, however carefully each pass advances past its own replacement:
  // the next parameter's pass starts at the top again and reaches into
  // text an earlier parameter already inserted. A lease holder that
  // literally says "%(ttl_ns)s" is a legal 11-byte string — only its
  // length is checked, on both sides — and was stored as the TTL, while
  // the driver stored the holder.
  std::string out;
  out.reserve(query.size());
  size_t at = 0;
  while (at < query.size()) {
    if (query.compare(at, 2, "%(") != 0) {
      out.push_back(query[at++]);
      continue;
    }
    const size_t close = query.find(")s", at + 2);
    const auto found =
        close == std::string::npos
            ? params.end()
            : params.find(query.substr(at + 2, close - at - 2));
    if (found == params.end()) {
      // Nothing this call can fill: an unknown name, or a `%(` with no
      // `)s` after it. Both go through as the text they are, and the
      // scan resumes just past the `%(` so that a well-formed
      // placeholder further along is still found.
      out += "%(";
      at += 2;
      continue;
    }
    const Param& value = found->second;
    if (auto* i = std::get_if<int64_t>(&value)) {
      out += std::to_string(*i);
    } else if (auto* u = std::get_if<uint64_t>(&value)) {
      out += std::to_string(*u);
    } else {
      out += sql_quote(std::get<std::string>(value));
    }
    at = close + 2;
  }
  return out;
}

uint64_t parse_u64_field(const std::string& text, const char* what) {
  if (text.empty()) return 0;
  uint64_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      throw ClickHouseError(std::string("invalid ") + what + ": " + text);
    }
    const uint64_t digit = static_cast<uint64_t>(c - '0');
    // A field too wide for UInt64 is refused, through the same error a
    // non-digit already raises: the accumulator had no bound, so it wrapped
    // modulo 2**64 and answered a plausible-looking number for a field it
    // could not represent. Every server-side caller reads a UInt64/UInt32
    // column, which renders in at most 20 digits, so from a well-behaved
    // ClickHouse this arm is unreachable and the bound is defence in depth
    // -- but `get_by_ids` hands this function a CALLER's watermark string,
    // gated only on "non-empty and all digits", and 2**64 wrapped to 0.
    // The bound is checked BEFORE the multiply, as it is in json.cpp,
    // record_row.cpp and reader.cpp: after it, the value is already gone.
    if (value > (UINT64_MAX - digit) / 10) {
      throw ClickHouseError(std::string(what) + " does not fit UInt64: " +
                            text);
    }
    value = value * 10 + digit;
  }
  return value;
}

std::map<std::string, std::string> deciding_read() {
  return {{"select_sequential_consistency", "1"}};
}

ClickHouseClient::ClickHouseClient(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port) {
  // Process-lifetime, not per-object: the matching curl_global_cleanup used
  // to run in the destructor below, which tore libcurl down for the WHOLE
  // process while the uploader's worker threads were inside
  // curl_easy_perform. See common/curl_init.h.
  dmi_common::EnsureCurlGlobalInit();
}

ClickHouseClient::~ClickHouseClient() = default;

std::vector<Row> ClickHouseClient::execute(
    const std::string& query, const Params& params,
    const std::map<std::string, std::string>& settings) const {
  const std::string statement = substitute(query, params);

  // Settings ride as URL parameters; the statement is the POST body
  // (GET-with-query is evaluated as readonly — writes are refused).
  std::string url = "http://" + host_ + ":" + std::to_string(port_) + "/?";
  for (const auto& [key, value] : settings) {
    url += url_encode(key) + "=" + url_encode(value) + "&";
  }
  url.pop_back();

  CURL* curl = curl_easy_init();
  if (curl == nullptr) throw ClickHouseError("libcurl init failed");
  std::string body;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  // The body is a LENGTH, not a C string. Without an explicit size
  // libcurl measures the POST body with strlen, so a NUL anywhere in the
  // statement silently drops everything after it and the server answers
  // the prefix as if that were the whole query. The escaper never emits a
  // raw NUL, but statement text assembled outside it still can, so the
  // length is its own guard rather than a consequence of the escaping.
  //
  // Order is free for CURLOPT_POSTFIELDS, which only borrows the buffer;
  // anyone switching to CURLOPT_COPYPOSTFIELDS must set the size FIRST,
  // because that option copies using the size known at the time.
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, statement.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE,
                   static_cast<curl_off_t>(statement.size()));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
  const CURLcode code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_cleanup(curl);
  if (code != CURLE_OK) {
    throw ClickHouseError(std::string("curl: ") +
                          curl_easy_strerror(code));
  }
  if (status != 200) {
    throw ClickHouseError("clickhouse " + std::to_string(status) + ": " +
                          body.substr(0, 4096));
  }

  std::vector<Row> rows;
  size_t start = 0;
  while (start < body.size()) {
    size_t end = body.find('\n', start);
    if (end == std::string::npos) end = body.size();
    const std::string line = body.substr(start, end - start);
    start = end + 1;
    if (line.empty()) continue;
    Row row;
    size_t field_start = 0;
    while (field_start <= line.size()) {
      size_t field_end = line.find('\t', field_start);
      if (field_end == std::string::npos) field_end = line.size();
      row.push_back(line.substr(field_start, field_end - field_start));
      field_start = field_end + 1;
      if (field_end == line.size()) break;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace dmi_catalog
