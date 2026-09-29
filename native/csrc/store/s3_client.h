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

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cancel.h"

namespace dmi_store {

struct S3Config {
  std::string endpoint;   // "http://host:port" or "https://host" (no bucket, no trailing /)
  std::string bucket;
  std::string region = "us-east-1";
  std::string access_key;
  std::string secret_key;
  std::string session_token;  // empty when unused
  bool allow_insecure_http = false;
  // https only: trust a private CA. ca_file is a PEM bundle
  // (CURLOPT_CAINFO), ca_path an OpenSSL-hashed certificate directory
  // (CURLOPT_CAPATH). Both empty uses libcurl's default trust store. Either
  // way https always verifies the peer and the host name.
  std::string ca_file;
  std::string ca_path;
  int connect_timeout_s = 5;
  int read_timeout_s = 120;
  int max_attempts = 4;
  uint64_t multipart_threshold_bytes = 64ull * 1024 * 1024;
  // The size of every part but the last. S3 refuses a smaller non-final
  // part (EntityTooSmall), so the client refuses one under kMinMultipartPartBytes.
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
  // The client's Cancellation cut the exchange short (ok is false): before
  // an attempt, during its transfer, or in the backoff before a retry.
  // Never retried.
  bool cancelled = false;
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

// S3's minimum size for every part of a multipart upload but the last.
inline constexpr uint64_t kMinMultipartPartBytes = 5ull * 1024 * 1024;

// The whole-request bound, connect included, on the AbortMultipartUpload a
// cancel leads to: one attempt, since whoever cancelled is waiting on it.
inline constexpr int kAbortAfterCancelTimeoutS = 5;

class S3Client {
 public:
  // An invalid config (see ValidateConfig) does not throw: the client
  // refuses every request with the reason, before anything goes out.
  explicit S3Client(S3Config config);

  // Empty when `config` is usable; otherwise why not. Callers with an error
  // channel of their own (the Python bindings) check it at construction.
  static std::string ValidateConfig(const S3Config& config);
  ~S3Client();

  S3Client(const S3Client&) = delete;
  S3Client& operator=(const S3Client&) = delete;

  const S3Config& config() const { return config_; }

  // Every request from now on honours `cancel` (nullptr: none): none is
  // sent once it is cancelled, a transfer in flight is aborted (libcurl's
  // progress callback asks at least once a second, connecting included),
  // and a retry's backoff wakes for it. A cancelled request fails with the
  // error "request cancelled" and is not retried; a multipart upload it cut
  // short is aborted with a request of its own, which the cancel does not
  // cut (bounded by kAbortAfterCancelTimeoutS instead). Set it before the
  // client is shared, and keep `cancel` alive as long as the client.
  void set_cancellation(const Cancellation* cancel) { cancel_ = cancel; }
  bool cancelled() const { return cancel_ != nullptr && cancel_->cancelled(); }
  // Attempts actually made by the last call (1 + retries), for tests.
  int last_attempts() const { return last_attempts_.load(std::memory_order_relaxed); }

  // HEAD /bucket/key. 404 → {found=false}, no error.
  //
  // On failure, *cancelled (when given, here and on GetRange and
  // PutObject) says whether the Cancellation cut the call short -- before
  // an attempt, in its transfer or in a retry's backoff -- rather than the
  // store failing it: a cancel that merely comes in while a failure is
  // reported does not count.
  ObjectHead HeadObject(const std::string& key, std::string* error,
                        bool* cancelled = nullptr);

  // GET /bucket/key, optionally Range: bytes=offset-(offset+length-1).
  // length==0 returns empty without a request (matches read_range).
  // Short/oversized bodies are errors, not truncations. On failure,
  // *unavailable (when given) says whether the store never answered for
  // the object: a transport error or timeout, a retryable status on every
  // attempt, or a cancel -- as against an answer about it (a 404, a 403, a
  // short body), which says something about the object itself.
  bool GetRange(const std::string& key, uint64_t offset, uint64_t length,
                std::vector<uint8_t>* out, std::string* error,
                bool* unavailable = nullptr, bool* cancelled = nullptr);

  // PUT /bucket/key with x-amz-content-sha256 over the exact bytes plus the
  // DMI metadata headers. Over multipart_threshold_bytes the call becomes
  // Create + UploadPart* + Complete automatically. Returns the ETag.
  bool PutObject(const std::string& key, const uint8_t* data, size_t n,
                 const std::map<std::string, std::string>& metadata,
                 const std::string& content_type, std::string* etag_out,
                 std::string* error, bool* cancelled = nullptr);

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
  std::string config_error_;  // non-empty: every request is refused
  std::string host_;    // endpoint host (with :port when non-default)
  std::string scheme_;
  bool is_https_ = false;
  // Atomic because SpoolUploader shares one client across its worker
  // threads, and every request writes this; a plain int was a data race.
  std::atomic<int> last_attempts_{0};
  const Cancellation* cancel_ = nullptr;

  // How one exchange departs from the config: whether the Cancellation
  // applies, and (when positive) its own attempt count and whole-request
  // timeout in seconds.
  struct ExchangeOptions {
    bool cancellable = true;
    int max_attempts = 0;
    int timeout_s = 0;
  };
  S3Response ExchangeWith(
      const std::string& method, const std::string& key,
      const std::vector<std::pair<std::string, std::string>>& query,
      const std::map<std::string, std::string>& extra_headers,
      const uint8_t* body, size_t body_len, const std::string& body_hash_hex,
      const ExchangeOptions& options);
  // Aborts a multipart upload: as configured, or -- after a cancel -- once,
  // uncancelled, within kAbortAfterCancelTimeoutS.
  void AbortMultipart(const std::string& key, const std::string& upload_id,
                      bool after_cancel);

  // Multipart primitives (single PUT when under threshold).
  bool PutSingle(const std::string& key, const uint8_t* data, size_t n,
                 const std::map<std::string, std::string>& metadata,
                 const std::string& content_type, std::string* etag_out,
                 std::string* error, bool* cancelled_out);
  bool PutMultipart(const std::string& key, const uint8_t* data, size_t n,
                    const std::map<std::string, std::string>& metadata,
                    const std::string& content_type, std::string* etag_out,
                    std::string* error, bool* cancelled_out);
};

}  // namespace dmi_store

#endif  // DMI_STORE_S3_CLIENT_H_
