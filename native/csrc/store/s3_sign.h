// AWS Signature Version 4 for S3, dependency-free except OpenSSL (HMAC/SHA-256).
//
// The contract is botocore's SigV4Auth: the differential test
// (tests/test_native_s3_sign.py) signs a battery of generated requests with
// botocore and with this code and requires identical Authorization headers.
// No live endpoint needed.

#ifndef DMI_STORE_S3_SIGN_H_
#define DMI_STORE_S3_SIGN_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dmi_store {

// SigV4 URI encoding: unreserved marks (A-Z a-z 0-9 - _ . ~) and, for paths
// only, '/' pass through; everything else is %XX uppercase hex.
std::string UriEncode(const std::string& value, bool is_path);

// Lowercase header name copy (values are handled separately).
std::string LowerHeader(const std::string& name);

// Canonical request for method + path + query + headers + payload hash.
// `query` is (name, value) pairs; empty value renders as "name=" (S3
// subresources such as ?uploads). `headers` are raw (name, value); names are
// lowercased, values trimmed with internal runs collapsed to one space.
std::string CanonicalRequest(const std::string& method,
                             const std::string& encoded_path,
                             const std::vector<std::pair<std::string, std::string>>& query,
                             const std::map<std::string, std::string>& headers,
                             const std::string& payload_hash_hex);

// "name1;name2" over the sorted lowercase header names.
std::string SignedHeaders(const std::map<std::string, std::string>& headers);

// Scope "YYYYMMDD/region/service/aws4_request" and the full string to sign.
std::string CredentialScope(const std::string& datestamp,
                            const std::string& region,
                            const std::string& service);
std::string StringToSign(const std::string& amz_date,
                         const std::string& scope,
                         const std::string& hashed_canonical_hex);

// Signing key chain (HMAC-SHA256); 32 bytes out.
void SigningKey(const std::string& secret_key, const std::string& datestamp,
                const std::string& region, const std::string& service,
                uint8_t out[32]);

// Hex HMAC-SHA256 of the string to sign under the signing key.
std::string Signature(const uint8_t signing_key[32],
                      const std::string& string_to_sign);

// Full Authorization header value for one request. `headers` must already
// contain host, x-amz-date, x-amz-content-sha256 (and x-amz-security-token
// when a session token is in play); those names are exactly what gets
// signed. `amz_date` is "YYYYMMDDTHHMMSSZ".
std::string AuthorizationHeader(
    const std::string& access_key, const std::string& secret_key,
    const std::string& datestamp, const std::string& amz_date,
    const std::string& region, const std::string& service,
    const std::string& method, const std::string& encoded_path,
    const std::vector<std::pair<std::string, std::string>>& query,
    const std::map<std::string, std::string>& headers,
    const std::string& payload_hash_hex);

// Helpers also needed by the client.
std::string Sha256Hex(const uint8_t* data, size_t n);
std::string Sha256Hex(const std::string& text);
std::string HmacSha256Hex(const uint8_t* key, size_t key_len,
                           const std::string& message);

}  // namespace dmi_store

#endif  // DMI_STORE_S3_SIGN_H_
