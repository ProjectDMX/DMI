#include "s3_client.h"

#include <curl/curl.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <thread>

#include "../common/curl_init.h"
#include "s3_sign.h"

namespace dmi_store {

namespace {

std::string AmzDate(std::time_t when) {
  char buf[17];
  std::tm tm_utc{};
  gmtime_r(&when, &tm_utc);
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm_utc);
  return buf;
}

std::string Datestamp(const std::string& amz_date) {
  return amz_date.substr(0, 8);
}

size_t WriteBody(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* body = static_cast<std::string*>(userdata);
  body->append(ptr, size * nmemb);
  return size * nmemb;
}

size_t WriteHeaders(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);
  std::string line(ptr, size * nmemb);
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
    line.pop_back();
  }
  const size_t colon = line.find(':');
  if (colon == std::string::npos) return size * nmemb;  // status line
  std::string name = LowerHeader(line.substr(0, colon));
  size_t v = colon + 1;
  while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
  (*headers)[name] = line.substr(v);
  return size * nmemb;
}

struct UploadCursor {
  const uint8_t* data = nullptr;
  size_t remaining = 0;
};

size_t ReadBody(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* cursor = static_cast<UploadCursor*>(userdata);
  const size_t want = size * nmemb;
  const size_t n = want < cursor->remaining ? want : cursor->remaining;
  std::memcpy(ptr, cursor->data, n);
  cursor->data += n;
  cursor->remaining -= n;
  return n;
}

bool IsRetryableCurl(CURLcode code) {
  return code == CURLE_OPERATION_TIMEDOUT || code == CURLE_COULDNT_CONNECT ||
         code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_GOT_NOTHING ||
         code == CURLE_PARTIAL_FILE || code == CURLE_RECV_ERROR ||
         code == CURLE_SEND_ERROR || code == CURLE_SEND_FAIL_REWIND ||
         code == CURLE_HTTP2;
}

bool IsRetryableStatus(long status) {
  return status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

void Backoff(int attempt) {
  // 0.2s * 2^attempt, capped at 5s. Deterministic: the fault-matrix tests
  // assert attempt counts, not wall time, so no jitter.
  using namespace std::chrono;
  const int64_t ms = std::min<int64_t>(5000, 200LL << attempt);
  std::this_thread::sleep_for(milliseconds(ms));
}

std::string XmlEscape(const std::string& value) {
  std::string out;
  for (char c : value) {
    switch (c) {
      case '&': out.append("&amp;"); break;
      case '<': out.append("&lt;"); break;
      case '>': out.append("&gt;"); break;
      case '"': out.append("&quot;"); break;
      default: out.push_back(c);
    }
  }
  return out;
}

// First occurrence of <tag>...</tag> (no namespaces in S3 XML).
std::string XmlTag(const std::string& xml, const std::string& tag) {
  const std::string open = "<" + tag + ">";
  const std::string close = "</" + tag + ">";
  const size_t at = xml.find(open);
  if (at == std::string::npos) return "";
  const size_t start = at + open.size();
  const size_t end = xml.find(close, start);
  if (end == std::string::npos) return "";
  return xml.substr(start, end - start);
}

}  // namespace

S3Client::S3Client(S3Config config) : config_(std::move(config)) {
  // Explicit, rather than leaning on the implicit init inside
  // curl_easy_init: that implicit path carries libcurl's thread-safety
  // caveat, and Exchange() runs on the uploader's worker threads. Once per
  // process, and never torn down. See common/curl_init.h.
  dmi_common::EnsureCurlGlobalInit();
  // endpoint := scheme://host[:port]; bucket and key are appended per call
  // (path style, matching the Python store's addressing_style="path").
  std::string rest = config_.endpoint;
  if (rest.compare(0, 7, "http://") == 0) {
    scheme_ = "http";
    rest = rest.substr(7);
  } else if (rest.compare(0, 8, "https://") == 0) {
    scheme_ = "https";
    is_https_ = true;
    rest = rest.substr(8);
  } else {
    scheme_ = "http";
  }
  while (!rest.empty() && rest.back() == '/') rest.pop_back();
  host_ = rest;
  if (is_https_ && config_.allow_insecure_http) {
    // Refused at construction: silently downgrading TLS was never the intent
    // of the flag — it gates plain http:// endpoints for local Garage.
    host_.clear();
  }
}

S3Client::~S3Client() = default;

S3Response S3Client::Exchange(
    const std::string& method, const std::string& key,
    const std::vector<std::pair<std::string, std::string>>& query,
    const std::map<std::string, std::string>& extra_headers,
    const uint8_t* body, size_t body_len, const std::string& body_hash_hex) {
  S3Response response;
  if (host_.empty()) {
    response.error = "https endpoint with allow_insecure_http is refused";
    return response;
  }
  const std::string amz_date = AmzDate(std::time(nullptr));
  const std::string datestamp = Datestamp(amz_date);

  const std::string encoded_resource = "/" + config_.bucket + "/" + key;
  const std::string encoded_path = UriEncode(encoded_resource, true);

  std::map<std::string, std::string> headers;
  headers["host"] = host_;
  headers["x-amz-date"] = amz_date;
  headers["x-amz-content-sha256"] = body_hash_hex;
  if (!config_.session_token.empty()) {
    headers["x-amz-security-token"] = config_.session_token;
  }
  for (const auto& [name, value] : extra_headers) {
    headers[LowerHeader(name)] = value;
  }
  const std::string authz = AuthorizationHeader(
      config_.access_key, config_.secret_key, datestamp, amz_date,
      config_.region, "s3", method, encoded_path, query, headers,
      body_hash_hex);

  // Canonical query string for the URL (same encoding the signer used).
  std::string query_text;
  {
    std::vector<std::pair<std::string, std::string>> sorted = query;
    for (auto& [name, value] : sorted) {
      name = UriEncode(name, false);
      value = UriEncode(value, false);
    }
    std::sort(sorted.begin(), sorted.end());
    bool first = true;
    for (const auto& [name, value] : sorted) {
      if (!first) query_text.push_back('&');
      query_text.append(name);
      query_text.push_back('=');
      query_text.append(value);
      first = false;
    }
  }
  const std::string url = scheme_ + "://" + host_ + encoded_path +
                          (query_text.empty() ? "" : "?" + query_text);

  last_attempts_ = 0;
  for (int attempt = 0; attempt < config_.max_attempts; ++attempt) {
    ++last_attempts_;
    CURL* curl = curl_easy_init();
    if (!curl) {
      response.error = "curl_easy_init failed";
      return response;
    }
    struct curl_slist* chunk = nullptr;
    chunk = curl_slist_append(chunk, ("Authorization: " + authz).c_str());
    // libcurl sets Host itself, but the signed value must be byte-identical:
    // re-send every signed header explicitly except host (curl owns it) and
    // content-length (curl computes it from the body size).
    for (const auto& [name, value] : headers) {
      if (name == "host") continue;
      chunk = curl_slist_append(chunk, (name + ": " + value).c_str());
    }
    if (!config_.user_agent.empty()) {
      curl_easy_setopt(curl, CURLOPT_USERAGENT, config_.user_agent.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout_s);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.read_timeout_s);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    if (!is_https_) {
      // Plain http only by explicit opt-in (local Garage); https always
      // verifies (no CURLOPT_SSL_VERIFYPEER toggle exists anywhere here).
      if (!config_.allow_insecure_http) {
        response.error = "plain http endpoint requires allow_insecure_http";
        curl_slist_free_all(chunk);
        curl_easy_cleanup(curl);
        return response;
      }
    }
    std::string response_body;
    std::map<std::string, std::string> response_headers;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteBody);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, WriteHeaders);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response_headers);

    UploadCursor cursor{body, body_len};
    if (method == "PUT" || method == "POST") {
      curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
      curl_easy_setopt(curl, CURLOPT_READFUNCTION, ReadBody);
      curl_easy_setopt(curl, CURLOPT_READDATA, &cursor);
      curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE,
                       static_cast<curl_off_t>(body_len));
      if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
      }
    } else if (method == "HEAD") {
      curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    } else if (method == "DELETE") {
      curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    const CURLcode code = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(chunk);
    curl_easy_cleanup(curl);

    if (code != CURLE_OK) {
      response.error = std::string("curl: ") + curl_easy_strerror(code);
      if (IsRetryableCurl(code) && attempt + 1 < config_.max_attempts) {
        Backoff(attempt);
        continue;
      }
      return response;
    }
    response.ok = true;
    response.http_status = status;
    response.headers = std::move(response_headers);
    response.body = std::move(response_body);
    if (IsRetryableStatus(status) && attempt + 1 < config_.max_attempts) {
      Backoff(attempt);
      continue;
    }
    return response;
  }
  response.ok = false;
  if (response.error.empty()) {
    response.error = "exhausted retries without a response";
  }
  return response;
}

ObjectHead S3Client::HeadObject(const std::string& key, std::string* error) {
  ObjectHead head;
  S3Response response =
      Exchange("HEAD", key, {}, {}, nullptr, 0, Sha256Hex(""));
  if (!response.ok) {
    if (error) *error = response.error;
    return head;
  }
  if (response.http_status == 404) return head;  // {found=false}
  if (response.http_status != 200) {
    if (error) {
      *error = "HeadObject returned HTTP " +
               std::to_string(response.http_status);
    }
    return head;
  }
  head.found = true;
  const auto size = response.headers.find("content-length");
  if (size != response.headers.end()) {
    head.size = std::strtoull(size->second.c_str(), nullptr, 10);
  }
  const auto etag = response.headers.find("etag");
  if (etag != response.headers.end()) {
    head.etag = etag->second;
    while (!head.etag.empty() &&
           (head.etag.front() == '"' || head.etag.front() == ' ')) {
      head.etag.erase(head.etag.begin());
    }
    while (!head.etag.empty() &&
           (head.etag.back() == '"' || head.etag.back() == ' ')) {
      head.etag.pop_back();
    }
  }
  for (const auto& [name, value] : response.headers) {
    if (name.compare(0, 11, "x-amz-meta-") == 0) {
      head.metadata[name.substr(11)] = value;
    }
  }
  return head;
}

bool S3Client::GetRange(const std::string& key, uint64_t offset,
                        uint64_t length, std::vector<uint8_t>* out,
                        std::string* error) {
  out->clear();
  if (length == 0) return true;
  std::map<std::string, std::string> headers;
  headers["Range"] = "bytes=" + std::to_string(offset) + "-" +
                     std::to_string(offset + length - 1);
  S3Response response = Exchange("GET", key, {}, headers, nullptr, 0,
                                 Sha256Hex(""));
  if (!response.ok) {
    if (error) *error = response.error;
    return false;
  }
  if (response.http_status != 200 && response.http_status != 206) {
    if (error) {
      *error = "GetObject returned HTTP " +
               std::to_string(response.http_status);
    }
    return false;
  }
  // A short or oversized body is corruption or a server bug, never a
  // truncation: the caller sized the request from the pack footer.
  if (response.body.size() != length) {
    if (error) {
      *error = "GetObject returned " +
               std::to_string(response.body.size()) + " bytes for a " +
               std::to_string(length) + "-byte range";
    }
    return false;
  }
  out->assign(response.body.begin(), response.body.end());
  return true;
}

bool S3Client::PutSingle(
    const std::string& key, const uint8_t* data, size_t n,
    const std::map<std::string, std::string>& metadata,
    const std::string& content_type, std::string* etag_out,
    std::string* error) {
  std::map<std::string, std::string> headers;
  headers["Content-Type"] = content_type;
  for (const auto& [name, value] : metadata) {
    headers["x-amz-meta-" + LowerHeader(name)] = value;
  }
  S3Response response =
      Exchange("PUT", key, {}, headers, data, n, Sha256Hex(data, n));
  if (!response.ok) {
    if (error) *error = response.error;
    return false;
  }
  if (response.http_status != 200) {
    if (error) {
      *error = "PutObject returned HTTP " +
               std::to_string(response.http_status) + ": " +
               response.body.substr(0, 512);
    }
    return false;
  }
  const auto etag = response.headers.find("etag");
  *etag_out = etag != response.headers.end() ? etag->second : "";
  return true;
}

bool S3Client::PutMultipart(
    const std::string& key, const uint8_t* data, size_t n,
    const std::map<std::string, std::string>& metadata,
    const std::string& content_type, std::string* etag_out,
    std::string* error) {
  std::map<std::string, std::string> headers;
  headers["Content-Type"] = content_type;
  for (const auto& [name, value] : metadata) {
    headers["x-amz-meta-" + LowerHeader(name)] = value;
  }
  // 1. Create.
  S3Response created =
      Exchange("POST", key, {{"uploads", ""}}, headers, nullptr, 0,
               Sha256Hex(""));
  if (!created.ok || created.http_status != 200) {
    if (error) {
      *error = "CreateMultipartUpload failed: " +
               (created.ok ? "HTTP " + std::to_string(created.http_status)
                           : created.error);
    }
    return false;
  }
  const std::string upload_id = XmlTag(created.body, "UploadId");
  if (upload_id.empty()) {
    if (error) *error = "CreateMultipartUpload returned no UploadId";
    return false;
  }
  // 2. Parts.
  const size_t chunk = static_cast<size_t>(config_.multipart_chunk_bytes);
  std::vector<std::pair<int, std::string>> parts;
  size_t offset = 0;
  int part_number = 0;
  std::string abort_error;
  while (offset < n) {
    const size_t len = std::min(chunk, n - offset);
    ++part_number;
    S3Response part = Exchange(
        "PUT", key,
        {{"partNumber", std::to_string(part_number)},
         {"uploadId", upload_id}},
        {}, data + offset, len, Sha256Hex(data + offset, len));
    if (!part.ok || part.http_status != 200) {
      abort_error =
          "UploadPart failed: " +
          (part.ok ? "HTTP " + std::to_string(part.http_status) : part.error);
      break;
    }
    const auto etag = part.headers.find("etag");
    if (etag == part.headers.end()) {
      abort_error = "UploadPart returned no ETag";
      break;
    }
    parts.emplace_back(part_number, etag->second);
    offset += len;
  }
  if (!abort_error.empty()) {
    Exchange("DELETE", key, {{"uploadId", upload_id}}, {}, nullptr, 0,
             Sha256Hex(""));
    if (error) *error = abort_error;
    return false;
  }
  // 3. Complete.
  std::string xml =
      "<CompleteMultipartUpload xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">";
  for (const auto& [number, etag] : parts) {
    xml.append("<Part><PartNumber>");
    xml.append(std::to_string(number));
    xml.append("</PartNumber><ETag>");
    xml.append(XmlEscape(etag));
    xml.append("</ETag></Part>");
  }
  xml.append("</CompleteMultipartUpload>");
  S3Response done =
      Exchange("POST", key, {{"uploadId", upload_id}},
               {{"Content-Type", "application/xml"}},
               reinterpret_cast<const uint8_t*>(xml.data()), xml.size(),
               Sha256Hex(reinterpret_cast<const uint8_t*>(xml.data()),
                         xml.size()));
  if (!done.ok || done.http_status != 200) {
    if (error) {
      *error = "CompleteMultipartUpload failed: " +
               (done.ok ? "HTTP " + std::to_string(done.http_status) + ": " +
                              done.body.substr(0, 512)
                         : done.error);
    }
    return false;
  }
  const auto etag = done.headers.find("etag");
  *etag_out = etag != done.headers.end() ? etag->second : "";
  return true;
}

bool S3Client::PutObject(const std::string& key, const uint8_t* data, size_t n,
                         const std::map<std::string, std::string>& metadata,
                         const std::string& content_type,
                         std::string* etag_out, std::string* error) {
  if (n >= config_.multipart_threshold_bytes) {
    return PutMultipart(key, data, n, metadata, content_type, etag_out, error);
  }
  return PutSingle(key, data, n, metadata, content_type, etag_out, error);
}

bool S3Client::DeleteObject(const std::string& key, std::string* error) {
  S3Response response =
      Exchange("DELETE", key, {}, {}, nullptr, 0, Sha256Hex(""));
  if (!response.ok) {
    if (error) *error = response.error;
    return false;
  }
  if (response.http_status != 200 && response.http_status != 204) {
    if (error) {
      *error = "DeleteObject returned HTTP " +
               std::to_string(response.http_status);
    }
    return false;
  }
  return true;
}

bool S3Client::ListObjects(const std::string& prefix,
                           const std::string& delimiter, int max_keys,
                           const std::string& continuation, ListResult* out,
                           std::string* error) {
  out->objects.clear();
  std::vector<std::pair<std::string, std::string>> query = {
      {"list-type", "2"},
      {"prefix", prefix},
      {"max-keys", std::to_string(max_keys)},
  };
  if (!delimiter.empty()) query.emplace_back("delimiter", delimiter);
  if (!continuation.empty()) {
    query.emplace_back("continuation-token", continuation);
  }
  S3Response response =
      Exchange("GET", "", query, {}, nullptr, 0, Sha256Hex(""));
  if (!response.ok) {
    if (error) *error = response.error;
    return false;
  }
  if (response.http_status != 200) {
    if (error) {
      *error = "ListObjectsV2 returned HTTP " +
               std::to_string(response.http_status);
    }
    return false;
  }
  size_t at = 0;
  while (true) {
    const size_t open = response.body.find("<Contents>", at);
    if (open == std::string::npos) break;
    const size_t close = response.body.find("</Contents>", open);
    if (close == std::string::npos) break;
    const std::string item =
        response.body.substr(open, close - open + 11);
    ListedObject object;
    object.key = XmlTag(item, "Key");
    object.etag = XmlTag(item, "ETag");
    object.size = std::strtoull(XmlTag(item, "Size").c_str(), nullptr, 10);
    out->objects.push_back(std::move(object));
    at = close + 11;
  }
  out->truncated = XmlTag(response.body, "IsTruncated") == "true";
  out->next_token = XmlTag(response.body, "NextContinuationToken");
  return true;
}

}  // namespace dmi_store
