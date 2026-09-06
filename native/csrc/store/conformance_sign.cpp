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

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

// JSON string unescaper shared with the pack conformance driver pattern:
// short forms back to bytes, \uXXXX to UTF-8.
std::string Unescape(const std::string& text, size_t& q) {
  std::string raw;
  while (q < text.size() && text[q] != '"') {
    if (text[q] == '\\' && q + 1 < text.size()) {
      ++q;
      switch (text[q]) {
        case 'n': raw.push_back('\n'); ++q; break;
        case 't': raw.push_back('\t'); ++q; break;
        case 'r': raw.push_back('\r'); ++q; break;
        case 'b': raw.push_back('\b'); ++q; break;
        case 'f': raw.push_back('\f'); ++q; break;
        case 'u': {
          unsigned v = 0;
          for (int k = 1; k <= 4 && q + k < text.size(); ++k) {
            const char h = text[q + k];
            v = v * 16 + (h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
          }
          q += 5;
          if (v < 0x80) {
            raw.push_back(static_cast<char>(v));
          } else if (v < 0x800) {
            raw.push_back(static_cast<char>(0xC0 | (v >> 6)));
            raw.push_back(static_cast<char>(0x80 | (v & 0x3F)));
          } else {
            raw.push_back(static_cast<char>(0xE0 | (v >> 12)));
            raw.push_back(static_cast<char>(0x80 | ((v >> 6) & 0x3F)));
            raw.push_back(static_cast<char>(0x80 | (v & 0x3F)));
          }
          break;
        }
        default: raw.push_back(text[q]); ++q; break;
      }
    } else {
      raw.push_back(text[q++]);
    }
  }
  return raw;
}

// Find "key" (either separator style) and return its string value.
std::string FindString(const std::string& text, const std::string& key,
                       size_t from = 0) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep + "\"";
    const size_t at = text.find(needle, from);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    std::string out = Unescape(text, q);
    return out;
  }
  return "";
}

// Raw slice of the value for "key" (object or array), for nested parsing.
std::string FindRaw(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    while (q < text.size() && text[q] == ' ') ++q;
    if (q >= text.size()) return "";
    const char open = text[q];
    const char close = (open == '{') ? '}' : (open == '[' ? ']' : 0);
    if (!close) return "";
    int depth = 0;
    bool in_str = false;
    const size_t start = q;
    for (; q < text.size(); ++q) {
      if (in_str) {
        if (text[q] == '\\') ++q;
        else if (text[q] == '"') in_str = false;
        continue;
      }
      if (text[q] == '"') in_str = true;
      else if (text[q] == open) ++depth;
      else if (text[q] == close) {
        if (--depth == 0) return text.substr(start, q - start + 1);
      }
    }
  }
  return "";
}

// Split the comma-separated top-level elements of `inside`, which is the
// content of an array or object with the outer brackets already stripped.
std::vector<std::string> SplitElements(const std::string& inside) {
  std::vector<std::string> items;
  int depth = 0;
  bool in_str = false;
  size_t start = 0;
  for (size_t i = 0; i <= inside.size(); ++i) {
    const char c = (i < inside.size()) ? inside[i] : ',';
    if (in_str) {
      if (c == '\\') ++i;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      in_str = true;
    } else if (c == '[' || c == '{') {
      ++depth;
    } else if (c == ']' || c == '}') {
      --depth;
    } else if (c == ',' && depth == 0) {
      items.push_back(inside.substr(start, i - start));
      start = i + 1;
    }
  }
  return items;
}

// Strip one layer of [] or {} (already validated by FindRaw). Leading and
// trailing spaces are trimmed first: after a top-level ", " separator an
// item starts with a space, and blind substr would eat the wrong ends.
std::string Unwrap(const std::string& wrapped) {
  size_t start = 0, end = wrapped.size();
  while (start < end && wrapped[start] == ' ') ++start;
  while (end > start && wrapped[end - 1] == ' ') --end;
  if (end - start >= 2) return wrapped.substr(start + 1, end - start - 2);
  return "";
}

// Parse a "..." JSON string literal (with quotes) into raw text.
std::string ParseLiteral(const std::string& literal) {
  size_t q = 0;
  while (q < literal.size() && literal[q] != '"') ++q;
  if (q >= literal.size()) return "";
  ++q;
  return Unescape(literal, q);
}

void EscapeJson(const std::string& value, std::string* out) {
  out->push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '"': out->append("\\\""); break;
      case '\\': out->append("\\\\"); break;
      case '\n': out->append("\\n"); break;
      case '\r': out->append("\\r"); break;
      case '\t': out->append("\\t"); break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out->append(buf);
        } else {
          out->push_back(static_cast<char>(c));
        }
    }
  }
  out->push_back('"');
}

}  // namespace

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    const bool is_sign = line.find("\"sign\"") != std::string::npos;
    if (!is_sign) {
      std::cout << "{\"ok\":false,\"what\":\"unknown op\"}\n";
      continue;
    }
    const std::string method = FindString(line, "method");
    const std::string path = FindString(line, "path");
    const std::string payload_hash = FindString(line, "payload_hash");
    const std::string access = FindString(line, "access");
    const std::string secret = FindString(line, "secret");
    // token: null or string — detect literally.
    std::string token;
    {
      size_t at = line.find("\"token\"");
      if (at != std::string::npos) {
        size_t q = line.find(':', at) + 1;
        while (q < line.size() && line[q] == ' ') ++q;
        if (line.compare(q, 4, "null") != 0 && line[q] == '"') {
          ++q;
          token = Unescape(line, q);
        }
      }
    }
    const std::string region = FindString(line, "region");
    const std::string service = FindString(line, "service");
    const std::string datestamp = FindString(line, "datestamp");
    const std::string amz_date = FindString(line, "amz_date");

    std::vector<std::pair<std::string, std::string>> query;
    {
      const std::string arr = FindRaw(line, "query");
      for (const auto& item : SplitElements(Unwrap(arr))) {
        // item is ["name","value"] — split into its two literals.
        const auto parts = SplitElements(Unwrap(item));
        if (parts.size() == 2) {
          query.emplace_back(ParseLiteral(parts[0]), ParseLiteral(parts[1]));
        }
      }
    }
    std::map<std::string, std::string> headers;
    {
      const std::string obj = FindRaw(line, "headers");
      size_t q = 0;
      while (q < obj.size()) {
        if (obj[q] != '"') {
          ++q;
          continue;
        }
        ++q;
        const std::string name = Unescape(obj, q);
        ++q;  // closing quote
        while (q < obj.size() && (obj[q] == ' ' || obj[q] == ':')) ++q;
        std::string value;
        if (q < obj.size() && obj[q] == '"') {
          ++q;
          value = Unescape(obj, q);
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
    EscapeJson(authz, &out);
    out.append(",\"canonical\":");
    EscapeJson(canonical, &out);
    out.append("}\n");
    std::cout << out;
  }
  return 0;
}
