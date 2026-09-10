// SigV4 conformance driver: stdin/stdout JSON protocol so the pytest suite can
// differential-test this signer against botocore without pybind.
//
// Request:
//   {"op":"sign","method":"PUT","path":"/bucket/some key","query":[["uploads",""]],
//    "headers":{"host":"endpoint:3900","x-amz-date":"...","x-amz-content-sha256":"..."},
//    "payload_hash":"<hex>","access":"AK","secret":"SK","token":"..."|null,
//    "region":"us-east-1","service":"s3","datestamp":"20260905",
//    "amz_date":"20260905T120000Z"}
// The driver adds x-amz-security-token when "token" is non-null, URI-encodes
// the path (preserving '/'), encodes query pairs, and returns:
//   {"ok":true,"authorization":"...","canonical":"..."}
// Errors: {"ok":false,"what":"..."}.

#include "s3_sign.h"

#include "../common/json.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>



namespace jc = dmi_common;

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    const bool is_sign = line.find("\"sign\"") != std::string::npos;
    if (!is_sign) {
      std::cout << "{\"ok\":false,\"what\":\"unknown op\"}\n";
      continue;
    }
    const std::string method = jc::FindString(line, "method");
    const std::string path = jc::FindString(line, "path");
    const std::string payload_hash = jc::FindString(line, "payload_hash");
    const std::string access = jc::FindString(line, "access");
    const std::string secret = jc::FindString(line, "secret");
    // token: null or string — detect literally.
    std::string token;
    {
      size_t at = line.find("\"token\"");
      if (at != std::string::npos) {
        size_t q = line.find(':', at) + 1;
        while (q < line.size() && line[q] == ' ') ++q;
        if (line.compare(q, 4, "null") != 0 && line[q] == '"') {
          ++q;
          token = jc::Unescape(line, q);
        }
      }
    }
    const std::string region = jc::FindString(line, "region");
    const std::string service = jc::FindString(line, "service");
    const std::string datestamp = jc::FindString(line, "datestamp");
    const std::string amz_date = jc::FindString(line, "amz_date");

    std::vector<std::pair<std::string, std::string>> query;
    {
      const std::string arr = jc::FindArray(line, "query");
      for (const auto& item : jc::SplitElements(jc::Unwrap(arr))) {
        // item is ["name","value"] — split into its two literals.
        const auto parts = jc::SplitElements(jc::Unwrap(item));
        if (parts.size() == 2) {
          query.emplace_back(jc::ParseLiteral(parts[0]), jc::ParseLiteral(parts[1]));
        }
      }
    }
    std::map<std::string, std::string> headers;
    {
      const std::string obj = jc::FindObject(line, "headers");
      size_t q = 0;
      while (q < obj.size()) {
        if (obj[q] != '"') {
          ++q;
          continue;
        }
        ++q;
        const std::string name = jc::Unescape(obj, q);
        ++q;  // closing quote
        while (q < obj.size() && (obj[q] == ' ' || obj[q] == ':')) ++q;
        std::string value;
        if (q < obj.size() && obj[q] == '"') {
          ++q;
          value = jc::Unescape(obj, q);
          ++q;
        }
        if (!name.empty()) headers[name] = value;
      }
    }
    if (!token.empty()) {
      headers["x-amz-security-token"] = token;
    }

    const std::string encoded_path = dmi_store::UriEncode(path, true);
    const std::string canonical = dmi_store::CanonicalRequest(
        method, encoded_path, query, headers, payload_hash);
    const std::string authz = dmi_store::AuthorizationHeader(
        access, secret, datestamp, amz_date, region, service, method,
        encoded_path, query, headers, payload_hash);

    std::string out = "{\"ok\":true,\"authorization\":";
    jc::EscapeJson(authz, &out);
    out.append(",\"canonical\":");
    jc::EscapeJson(canonical, &out);
    out.append("}\n");
    std::cout << out;
  }
  return 0;
}
