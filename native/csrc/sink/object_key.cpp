#include "object_key.h"

#include <openssl/sha.h>

#include <cstdio>
#include <ctime>

namespace dmi_sink {

namespace {

void AppendQuotedByte(std::string* out, unsigned char c) {
  static const char* kHex = "0123456789ABCDEF";
  out->push_back('%');
  out->push_back(kHex[(c >> 4) & 0xF]);
  out->push_back(kHex[c & 0xF]);
}

bool IsKeySafe(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
         c == '=';
}

std::string Sha256Hex(const std::string& text) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(),
         digest);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}


}  // namespace

std::string KeyComponent(const std::string& value) {
  // quote(value, safe="-_.="): unreserved + the safe set pass through;
  // everything else (including '~', which quote() would spare per RFC 3986)
  // is %XX uppercase.
  std::string encoded;
  encoded.reserve(value.size());
  for (unsigned char c : value) {
    if (IsKeySafe(c)) {
      encoded.push_back(static_cast<char>(c));
    } else if (c == '~') {
      encoded.append("%7E");
    } else {
      AppendQuotedByte(&encoded, c);
    }
  }
  if (encoded.size() <= 160 && encoded.compare(0, 7, "sha256-") != 0) {
    return encoded;
  }
  // Overlong, or already shaped like the reserved digest: collapse to
  // "sha256-" + sha256(identifier), keeping the mapping one-to-one
  // (see key_component's own comment for the collision this closes).
  return "sha256-" + Sha256Hex(value);
}

std::string DateSegment(uint64_t captured_at_ns) {
  const std::time_t seconds =
      static_cast<std::time_t>(captured_at_ns / 1'000'000'000);
  std::tm tm_utc{};
  gmtime_r(&seconds, &tm_utc);
  char buf[11];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_utc);
  return buf;
}

std::string ObjectKeyFor(const std::string& tenant_id,
                         const std::string& session_id,
                         uint64_t producer_rank, uint64_t captured_at_ns,
                         const std::string& pack_id) {
  return "v1/tenant=" + KeyComponent(tenant_id) + "/date=" +
         DateSegment(captured_at_ns) + "/session=" +
         KeyComponent(session_id) + "/rank=" + std::to_string(producer_rank) +
         "/" + pack_id + ".dmi-pack";
}

}  // namespace dmi_sink
