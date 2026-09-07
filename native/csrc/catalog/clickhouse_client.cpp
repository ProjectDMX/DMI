#include "clickhouse_client.h"

#include <cstdio>

namespace dmi_catalog {

namespace {

size_t write_body(char* ptr, size_t size, size_t nmemb, void* userp) {
  static_cast<std::string*>(userp)->append(ptr, size * nmemb);
  return size * nmemb;
}

// clickhouse-driver's client-side `%(name)s` substitution: the same
// placeholders in the ported statements, the same server-side text.
std::string substitute(std::string query, const Params& params) {
  for (const auto& [name, value] : params) {
    std::string rendered;
    if (auto* i = std::get_if<int64_t>(&value)) {
      rendered = std::to_string(*i);
    } else if (auto* u = std::get_if<uint64_t>(&value)) {
      rendered = std::to_string(*u);
    } else {
      const auto& s = std::get<std::string>(value);
      rendered.push_back('\'');
      for (const char c : s) {
        if (c == '\\' || c == '\'') rendered.push_back('\\');
        if (c == '\n') {
          rendered += "\\n";
          continue;
        }
        if (c == '\t') {
          rendered += "\\t";
          continue;
        }
        rendered.push_back(c);
      }
      rendered.push_back('\'');
    }
    const std::string needle = "%(" + name + ")s";
    // The search advances PAST each replacement: a rendered value that
    // itself contains the placeholder (a holder that literally says
    // "%(holder)s" is a legal 11-byte string) must not be re-matched
    // inside the inserted text — restarting at the top grew without
    // bound and hung the driver.
    // The scan restarts at position 0 for each parameter: `from` persisting
    // across parameters would skip a real occurrence of this placeholder
    // that sits BEFORE an earlier parameter's replacement. Within one
    // parameter's loop, `from` advances past each replacement so a
    // self-referential value (a rendered string containing its own
    // placeholder) cannot loop.
    size_t from = 0;
    size_t at;
    while ((at = query.find(needle, from)) != std::string::npos) {
      query.replace(at, needle.size(), rendered);
      from = at + rendered.size();
    }
  }
  return query;
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

uint64_t parse_u64_field(const std::string& text, const char* what) {
  if (text.empty()) return 0;
  uint64_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      throw ClickHouseError(std::string("invalid ") + what + ": " + text);
    }
    value = value * 10 + static_cast<uint64_t>(c - '0');
  }
  return value;
}

std::map<std::string, std::string> deciding_read() {
  return {{"select_sequential_consistency", "1"}};
}

ClickHouseClient::ClickHouseClient(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
}

ClickHouseClient::~ClickHouseClient() { curl_global_cleanup(); }

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
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, statement.c_str());
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
