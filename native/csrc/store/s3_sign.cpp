#include "s3_sign.h"

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace dmi_store {

namespace {

void AppendHexByte(std::string* out, unsigned char c) {
  static const char* kHex = "0123456789ABCDEF";
  out->push_back('%');
  out->push_back(kHex[(c >> 4) & 0xF]);
  out->push_back(kHex[c & 0xF]);
}

bool IsUnreserved(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~';
}

std::string TrimCollapse(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  bool pending_space = false;
  bool started = false;
  for (char ch : value) {
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
      if (started) pending_space = true;
      continue;
    }
    if (pending_space) {
      out.push_back(' ');
      pending_space = false;
    }
    started = true;
    out.push_back(ch);
  }
  return out;
}

void Hmac(const uint8_t* key, size_t key_len, const uint8_t* data, size_t n,
          uint8_t out[32]) {
  unsigned int len = 0;
  HMAC(EVP_sha256(), key, static_cast<int>(key_len), data, n, out, &len);
}

}  // namespace

std::string UriEncode(const std::string& value, bool is_path) {
  std::string out;
  out.reserve(value.size() + value.size() / 4);
  for (unsigned char c : value) {
    if (IsUnreserved(c) || (is_path && c == '/')) {
      out.push_back(static_cast<char>(c));
    } else {
      AppendHexByte(&out, c);
    }
  }
  return out;
}

std::string LowerHeader(const std::string& name) {
  std::string out = name;
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string SignedHeaders(const std::map<std::string, std::string>& headers) {
  std::string out;
  bool first = true;
  for (const auto& [name, value] : headers) {
    (void)value;
    if (!first) out.push_back(';');
    out.append(LowerHeader(name));
    first = false;
  }
  return out;
}

std::string CanonicalRequest(
    const std::string& method, const std::string& encoded_path,
    const std::vector<std::pair<std::string, std::string>>& query,
    const std::map<std::string, std::string>& headers,
    const std::string& payload_hash_hex) {
  // Query pairs are sorted by encoded name, then encoded value (botocore
  // sorts on the encoded form; for S3 subresources there are no duplicates).
  std::vector<std::pair<std::string, std::string>> sorted = query;
  for (auto& [name, value] : sorted) {
    name = UriEncode(name, false);
    value = UriEncode(value, false);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
              return a.first != b.first ? a.first < b.first
                                        : a.second < b.second;
            });
  std::string out;
  out.append(method);
  out.push_back('\n');
  out.append(encoded_path.empty() ? "/" : encoded_path);
  out.push_back('\n');
  bool first = true;
  for (const auto& [name, value] : sorted) {
    if (!first) out.push_back('&');
    out.append(name);
    out.push_back('=');
    out.append(value);
    first = false;
  }
  out.push_back('\n');
  for (const auto& [name, value] : headers) {
    out.append(LowerHeader(name));
    out.push_back(':');
    out.append(TrimCollapse(value));
    out.push_back('\n');
  }
  out.push_back('\n');
  out.append(SignedHeaders(headers));
  out.push_back('\n');
  out.append(payload_hash_hex);
  return out;
}

std::string CredentialScope(const std::string& datestamp,
                            const std::string& region,
                            const std::string& service) {
  return datestamp + "/" + region + "/" + service + "/aws4_request";
}

std::string Sha256Hex(const uint8_t* data, size_t n) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(data, n, digest);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

std::string Sha256Hex(const std::string& text) {
  return Sha256Hex(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

std::string HmacSha256Hex(const uint8_t* key, size_t key_len,
                           const std::string& message) {
  uint8_t mac[32];
  Hmac(key, key_len, reinterpret_cast<const uint8_t*>(message.data()),
       message.size(), mac);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[mac[i] >> 4];
    out[2 * i + 1] = kHex[mac[i] & 0xF];
  }
  return out;
}

std::string StringToSign(const std::string& amz_date, const std::string& scope,
                         const std::string& hashed_canonical_hex) {
  return "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" +
         hashed_canonical_hex;
}

void SigningKey(const std::string& secret_key, const std::string& datestamp,
                const std::string& region, const std::string& service,
                uint8_t out[32]) {
  const std::string aws4 = "AWS4" + secret_key;
  uint8_t k_date[32], k_region[32], k_service[32];
  Hmac(reinterpret_cast<const uint8_t*>(aws4.data()), aws4.size(),
       reinterpret_cast<const uint8_t*>(datestamp.data()), datestamp.size(),
       k_date);
  Hmac(k_date, 32, reinterpret_cast<const uint8_t*>(region.data()),
       region.size(), k_region);
  Hmac(k_region, 32, reinterpret_cast<const uint8_t*>(service.data()),
       service.size(), k_service);
  Hmac(k_service, 32, reinterpret_cast<const uint8_t*>("aws4_request"), 12,
       out);
}

std::string Signature(const uint8_t signing_key[32],
                      const std::string& string_to_sign) {
  return HmacSha256Hex(signing_key, 32, string_to_sign);
}

std::string AuthorizationHeader(
    const std::string& access_key, const std::string& secret_key,
    const std::string& datestamp, const std::string& amz_date,
    const std::string& region, const std::string& service,
    const std::string& method, const std::string& encoded_path,
    const std::vector<std::pair<std::string, std::string>>& query,
    const std::map<std::string, std::string>& headers,
    const std::string& payload_hash_hex) {
  const std::string canonical =
      CanonicalRequest(method, encoded_path, query, headers, payload_hash_hex);
  const std::string scope = CredentialScope(datestamp, region, service);
  const std::string to_sign =
      StringToSign(amz_date, scope, Sha256Hex(canonical));
  uint8_t key[32];
  SigningKey(secret_key, datestamp, region, service, key);
  return "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + scope +
         ", SignedHeaders=" + SignedHeaders(headers) +
         ", Signature=" + Signature(key, to_sign);
}

}  // namespace dmi_store
