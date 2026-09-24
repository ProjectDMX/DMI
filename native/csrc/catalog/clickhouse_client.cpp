#include "clickhouse_client.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

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

bool has_header_breaking_byte(const std::string& value) {
  return value.find_first_of(std::string("\r\n\0", 3)) != std::string::npos;
}

// The host goes into the URL verbatim, so anything that is not a host
// would change what the URL means: `user:pw@host` sends credentials in the
// URL (and into every log that records it), `host:port` or `host/path`
// reaches a different endpoint than the port field says. Only a bare name
// or address is admitted, and the message never repeats the value -- the
// likeliest mistake is a URL with a password in it.
void validate_host(const std::string& host) {
  const auto refuse = [](const std::string& why) {
    throw ClickHouseError(
        "clickhouse host " + why + ": pass a bare host name or address; "
        "the scheme, port and credentials each have their own option");
  };
  if (host.empty()) refuse("is empty");
  if (host.find('@') != std::string::npos) {
    refuse("must not carry userinfo (user:password@)");
  }
  if (host.front() == '[') {
    const bool closed = host.size() > 2 && host.back() == ']';
    const bool inner_ok =
        closed && std::all_of(host.begin() + 1, host.end() - 1, [](char c) {
          return std::isxdigit(static_cast<unsigned char>(c)) || c == ':' ||
                 c == '.';
        });
    if (!inner_ok) refuse("is not a bracketed IPv6 address");
    return;
  }
  const bool name_ok = std::all_of(host.begin(), host.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' ||
           c == '.' || c == '_';
  });
  if (!name_ok) refuse("is not a bare host name or address");
}

// Failures where the request never left this process: repeating the
// statement cannot run it twice, whatever it is.
bool never_connected(CURLcode code) {
  return code == CURLE_COULDNT_CONNECT || code == CURLE_COULDNT_RESOLVE_HOST ||
         code == CURLE_COULDNT_RESOLVE_PROXY;
}

// Failures after connecting that a repeat can plausibly cure: the
// connection broke, or the answer was cut short. Timeouts are deliberately
// absent (see execute()), as are TLS verification failures, which a repeat
// cannot cure.
bool transient_transport(CURLcode code) {
  return code == CURLE_SEND_ERROR || code == CURLE_RECV_ERROR ||
         code == CURLE_GOT_NOTHING || code == CURLE_PARTIAL_FILE;
}

struct Attempt {
  CURLcode code = CURLE_OK;
  long status = 0;
  std::string body;
  std::string detail;  // libcurl's error buffer: says WHICH certificate check
};

struct SlistFree {
  void operator()(curl_slist* list) const { curl_slist_free_all(list); }
};
using Headers = std::unique_ptr<curl_slist, SlistFree>;

void append_header(Headers* headers, const std::string& line) {
  curl_slist* grown = curl_slist_append(headers->get(), line.c_str());
  if (grown == nullptr) throw ClickHouseError("libcurl failed to add a header");
  (void)headers->release();
  headers->reset(grown);
}

std::vector<Row> parse_tsv(const std::string& body) {
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

}  // namespace

void validate(const ClickHouseConnection& c) {
  if (c.scheme != "http" && c.scheme != "https") {
    throw ClickHouseError("clickhouse scheme must be \"http\" or \"https\"");
  }
  validate_host(c.host);
  if (c.port == 0) throw ClickHouseError("clickhouse port must be in 1..65535");
  if (has_header_breaking_byte(c.user)) {
    throw ClickHouseError("clickhouse user must not contain CR, LF or NUL");
  }
  if (has_header_breaking_byte(c.password)) {
    throw ClickHouseError("clickhouse password must not contain CR, LF or NUL");
  }
  if (!c.password.empty() && c.user.empty()) {
    throw ClickHouseError(
        "clickhouse password is set without a clickhouse user; name the "
        "account it belongs to");
  }
  if (c.scheme == "http") {
    if (!c.ca_file.empty() || !c.ca_path.empty()) {
      throw ClickHouseError(
          "clickhouse ca_file/ca_path need the https scheme; over http they "
          "would verify nothing");
    }
    if (!c.password.empty() && !c.allow_insecure_http) {
      throw ClickHouseError(
          "clickhouse password over plain http is refused: use https, or set "
          "allow_insecure_http to send it in the clear");
    }
  } else if (c.allow_insecure_http) {
    throw ClickHouseError(
        "clickhouse allow_insecure_http admits plain http and never "
        "downgrades TLS; leave it unset for https");
  }
  if (!(c.timeouts.connect_s > 0) || !(c.timeouts.request_s > 0)) {
    throw ClickHouseError("clickhouse timeouts must be positive");
  }
  if (c.max_attempts < 1) {
    throw ClickHouseError("clickhouse max_attempts must be at least 1");
  }
}

bool is_read_statement(const std::string& statement) {
  size_t at = 0;
  while (at < statement.size() &&
         std::isspace(static_cast<unsigned char>(statement[at]))) {
    ++at;
  }
  std::string keyword;
  while (at < statement.size() &&
         std::isalpha(static_cast<unsigned char>(statement[at]))) {
    keyword.push_back(static_cast<char>(
        std::toupper(static_cast<unsigned char>(statement[at]))));
    ++at;
  }
  return keyword == "SELECT" || keyword == "WITH" || keyword == "SHOW" ||
         keyword == "DESCRIBE" || keyword == "DESC" || keyword == "EXISTS" ||
         keyword == "CHECK";
}

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

ClickHouseClient::ClickHouseClient(ClickHouseConnection connection)
    : connection_(std::move(connection)) {
  validate(connection_);
  // Process-lifetime, not per-object: the matching curl_global_cleanup used
  // to run in the destructor below, which tore libcurl down for the WHOLE
  // process while the uploader's worker threads were inside
  // curl_easy_perform. See common/curl_init.h.
  dmi_common::EnsureCurlGlobalInit();
}

ClickHouseClient::ClickHouseClient(std::string host, uint16_t port,
                                   ClickHouseTimeouts timeouts)
    : ClickHouseClient([&] {
        ClickHouseConnection c;
        c.host = std::move(host);
        c.port = port;
        c.timeouts = timeouts;
        return c;
      }()) {}

ClickHouseClient::~ClickHouseClient() = default;

std::vector<Row> ClickHouseClient::execute(
    const std::string& query, const Params& params,
    const std::map<std::string, std::string>& settings) const {
  const std::string statement = substitute(query, params);
  const bool read = is_read_statement(statement);

  // Settings ride as URL parameters; the statement is the POST body
  // (GET-with-query is evaluated as readonly — writes are refused). A
  // caller's own wait_end_of_query wins over the default added here.
  std::map<std::string, std::string> url_settings = settings;
  if (read) url_settings.emplace("wait_end_of_query", "1");
  std::string url = connection_.scheme + "://" + connection_.host + ":" +
                    std::to_string(connection_.port) + "/?";
  for (const auto& [key, value] : url_settings) {
    url += url_encode(key) + "=" + url_encode(value) + "&";
  }
  url.pop_back();

  // Credentials as headers: ClickHouse reads X-ClickHouse-User/-Key, and
  // unlike URL parameters or userinfo they do not end up in access logs,
  // proxies or the error text below. libcurl follows no redirect (no
  // CURLOPT_FOLLOWLOCATION), so they go to this host and nowhere else.
  Headers headers;
  if (!connection_.user.empty()) {
    append_header(&headers, "X-ClickHouse-User: " + connection_.user);
    if (!connection_.password.empty()) {
      append_header(&headers, "X-ClickHouse-Key: " + connection_.password);
    }
  }

  const auto perform = [&]() {
    Attempt attempt;
    CURL* curl = curl_easy_init();
    if (curl == nullptr) throw ClickHouseError("libcurl init failed");
    char error_buffer[CURL_ERROR_SIZE] = {0};
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
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &attempt.body);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers.get());
    if (connection_.scheme == "https") {
      // libcurl's defaults, stated: https without verification is not an
      // option this client offers.
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
      if (!connection_.ca_file.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, connection_.ca_file.c_str());
      }
      if (!connection_.ca_path.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAPATH, connection_.ca_path.c_str());
      }
    }
    // NOSIGNAL: timeouts must not use SIGALRM in a multi-threaded process.
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(connection_.timeouts.connect_s * 1000));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,
                     static_cast<long>(connection_.timeouts.request_s * 1000));
    attempt.code = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &attempt.status);
    curl_easy_cleanup(curl);
    attempt.detail = error_buffer;
    return attempt;
  };

  for (int number = 1;; ++number) {
    const Attempt attempt = perform();
    if (attempt.code == CURLE_OK && attempt.status == 200) {
      return parse_tsv(attempt.body);
    }
    std::string error;
    bool retry = false;
    if (attempt.code != CURLE_OK) {
      error = std::string("curl: ") + curl_easy_strerror(attempt.code);
      if (!attempt.detail.empty()) error += ": " + attempt.detail;
      retry = never_connected(attempt.code) ||
              (read && transient_transport(attempt.code));
    } else {
      error = "clickhouse " + std::to_string(attempt.status) + ": " +
              attempt.body.substr(0, 4096);
      retry = read && attempt.status >= 500 && attempt.status < 600;
    }
    if (!retry || number >= connection_.max_attempts) {
      if (number > 1) {
        error += " (after " + std::to_string(number) + " attempts)";
      }
      throw ClickHouseError(error);
    }
    // 100 ms, doubling, capped at 1 s: enough for a restarting server or a
    // flapping connection, short beside the request timeout it adds to.
    const int shift = std::min(number - 1, 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(
        std::min(100 << shift, 1000)));
  }
}

}  // namespace dmi_catalog
