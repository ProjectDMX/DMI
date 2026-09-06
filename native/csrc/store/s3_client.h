// S3-compatible object client over libcurl, with SigV4 request signing.
//
// Mirrors the operations S3PackStore needs (s3.py): HeadObject, GetObject
// (full + Range), PutObject (single + multipart), DeleteObject,
// ListObjectsV2, plus the S3PackStore-level semantics documented per method:
// preflight HEAD with metadata comparison, post-upload visibility check,
// short/oversized range refusal. Retry policy mirrors botocore "standard"
// mode: retry transport errors and 5xx/429, never 4xx, exponential backoff.
//
// Garaging note: verified against the same behaviors the Python store pins
// (preflight, metadata assertions, checksummed streams). Live Garage tests
// need DMI_S3_ENDPOINT and stay behind the `garage` marker.

#ifndef DMI_STORE_S3_CLIENT_H_
#define DMI_STORE_S3_CLIENT_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dmi_store {

struct S3Config {
  std::string endpoint;   // "http://host:port" or "https://host" (no bucket, no trailing /)
  std::string bucket;
  std::string region = "us-east-1";
  std::string access_key;
  std::string secret_key;
  std::string session_token;  // empty when unused
  bool allow_insecure_http = false;
  int connect_timeout_s = 5;
  int read_timeout_s = 120;
  int max_attempts = 4;
  uint64_t multipart_threshold_bytes = 64ull * 1024 * 1024;
  uint64_t multipart_chunk_bytes = 16ull * 1024 * 1024;
  std::string user_agent = "dmi-native-store/1";
};

// Outcome of one HTTP exchange. `ok` means a response was received (any
// status); transport failure sets ok=false with `error` naming the cause.
struct S3Response {
  bool ok = false;
  long http_status = 0;
  std::map<std::string, std::string> headers;  // lowercased names
  std::string body;
  std::string error;
};

// Parsed HEAD metadata the DMI pack layout stores per object.
struct ObjectHead {
  bool found = false;
  uint64_t size = 0;
  std::map<std::string, std::string> metadata;  // dmi-* names, lowercased
  std::string etag;
};

struct ListedObject {
  std::string key;
  uint64_t size = 0;
  std::string etag;
};

struct ListResult {
  bool truncated = false;
  std::string next_token;
  std::vector<ListedObject> objects;
};

class S3Client {
 public:
  explicit S3Client(S3Config config);
  ~S3Client();

  S3Client(const S3Client&) = delete;
  S3Client& operator=(const S3Client&) = delete;

  const S3Config& config() const { return config_; }
  // Attempts actually made by the last call (1 + retries), for tests.
  int last_attempts() const { return last_attempts_; }

  // HEAD /bucket/key. 404 → {found=false}, no error.
  ObjectHead HeadObject(const std::string& key, std::string* error);

  // GET /bucket/key, optionally Range: bytes=offset-(offset+length-1).
  // length==0 returns empty without a request (matches read_range).
  // Short/oversized bodies are errors, not truncations.
  bool GetRange(const std::string& key, uint64_t offset, uint64_t length,
                std::vector<uint8_t>* out, std::string* error);

  // PUT /bucket/key with x-amz-content-sha256 over the exact bytes plus the
  // DMI metadata headers. Over multipart_threshold_bytes the call becomes
  // Create + UploadPart* + Complete automatically. Returns the ETag.
  bool PutObject(const std::string& key, const uint8_t* data, size_t n,
                 const std::map<std::string, std::string>& metadata,
                 const std::string& content_type, std::string* etag_out,
                 std::string* error);

  bool DeleteObject(const std::string& key, std::string* error);

  // ListObjectsV2 with prefix/delimiter/max-keys/continuation. Used by the
  // reconciler (A2b) and the fault matrix.
  bool ListObjects(const std::string& prefix, const std::string& delimiter,
                   int max_keys, const std::string& continuation,
                   ListResult* out, std::string* error);

  // Exposed for the fault-matrix tests: one raw signed exchange.
  S3Response Exchange(const std::string& method, const std::string& key,
                      const std::vector<std::pair<std::string, std::string>>& query,
                      const std::map<std::string, std::string>& extra_headers,
                      const uint8_t* body, size_t body_len,
                      const std::string& body_hash_hex);

 private:
  S3Config config_;
  std::string host_;    // endpoint host (with :port when non-default)
  std::string scheme_;
  bool is_https_ = false;
  int last_attempts_ = 0;

  // Multipart primitives (single PUT when under threshold).
  bool PutSingle(const std::string& key, const uint8_t* data, size_t n,
                 const std::map<std::string, std::string>& metadata,
                 const std::string& content_type, std::string* etag_out,
                 std::string* error);
  bool PutMultipart(const std::string& key, const uint8_t* data, size_t n,
                    const std::map<std::string, std::string>& metadata,
                    const std::string& content_type, std::string* etag_out,
                    std::string* error);
};

}  // namespace dmi_store

#endif  // DMI_STORE_S3_CLIENT_H_
