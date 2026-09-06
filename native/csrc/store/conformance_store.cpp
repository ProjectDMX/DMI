// Store conformance driver: stdin/stdout JSON protocol for the pytest suite.
// No pybind, no network beyond the endpoint the test points at.
//
// Ops (one JSON object per line, one JSON response per line):
//   {"op":"put","endpoint":"...","bucket":"...","region":"...",
//    "access":"...","secret":"...","token":null,"insecure":true,
//    "key":"...","data_b64":"...","metadata":{...},"content_type":"...",
//    "multipart_threshold":N,"multipart_chunk":N,"max_attempts":N,
//    "connect_timeout":N,"read_timeout":N}
//     -> {"ok":true,"etag":"...","attempts":N}
//   {"op":"get",...,"offset":N,"length":N} -> {"ok":true,"data_b64":"...","attempts":N}
//   {"op":"head",...} -> {"ok":true,"found":bool,"size":N,"metadata":{...},
//                         "etag":"...","attempts":N}
//   {"op":"delete",...} -> {"ok":true,"attempts":N}
//   {"op":"list",...,"prefix":"...","delimiter":"...","max_keys":N,
//    "continuation":"..."} -> {"ok":true,"truncated":bool,"next_token":"...",
//    "objects":[{"key":"...","size":N,"etag":"..."}...],"attempts":N}
// Errors: {"ok":false,"what":"..."}.

#include "s3_client.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

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
          if (v < 0x80) raw.push_back(static_cast<char>(v));
          else if (v < 0x800) {
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

std::string FindString(const std::string& text, const std::string& key,
                       size_t from = 0) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep + "\"";
    const size_t at = text.find(needle, from);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    return Unescape(text, q);
  }
  return "";
}

int64_t FindInt(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    while (q < text.size() && text[q] == ' ') ++q;
    int64_t v = 0;
    bool neg = false;
    if (q < text.size() && text[q] == '-') { neg = true; ++q; }
    while (q < text.size() && text[q] >= '0' && text[q] <= '9') {
      v = v * 10 + (text[q] - '0');
      ++q;
    }
    return neg ? -v : v;
  }
  return 0;
}

bool FindBool(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    return text.compare(q, 4, "true") == 0;
  }
  return false;
}

// metadata object: {"k":"v",...} — string values only.
std::map<std::string, std::string> FindMetadata(const std::string& text) {
  std::map<std::string, std::string> out;
  size_t at = text.find("\"metadata\"");
  if (at == std::string::npos) return out;
  size_t q = text.find('{', at);
  if (q == std::string::npos) return out;
  ++q;
  while (q < text.size() && text[q] != '}') {
    while (q < text.size() && (text[q] == ' ' || text[q] == ',')) ++q;
    if (q >= text.size() || text[q] != '"') break;
    ++q;
    const std::string name = Unescape(text, q);
    ++q;  // closing quote
    while (q < text.size() && (text[q] == ' ' || text[q] == ':')) ++q;
    std::string value;
    if (q < text.size() && text[q] == '"') {
      ++q;
      value = Unescape(text, q);
      ++q;
    }
    if (!name.empty()) out[name] = value;
  }
  return out;
}

bool DecodeBase64(const std::string& in, std::vector<uint8_t>* out) {
  auto nib = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  out->clear();
  uint32_t acc = 0;
  int bits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    const int v = nib(c);
    if (v < 0) return false;
    acc = acc << 6 | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(static_cast<uint8_t>(acc >> bits));
    }
  }
  return true;
}

void EncodeBase64(const std::vector<uint8_t>& data, std::string* out) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  out->clear();
  out->reserve((data.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < data.size(); i += 3) {
    const uint32_t v = data[i] << 16 | data[i + 1] << 8 | data[i + 2];
    out->push_back(kAlpha[(v >> 18) & 63]);
    out->push_back(kAlpha[(v >> 12) & 63]);
    out->push_back(kAlpha[(v >> 6) & 63]);
    out->push_back(kAlpha[v & 63]);
  }
  while (i < data.size()) {
    uint32_t v = data[i++] << 16;
    bool two = false;
    if (i < data.size()) {
      v |= data[i++] << 8;
      two = true;
    }
    out->push_back(kAlpha[(v >> 18) & 63]);
    out->push_back(kAlpha[(v >> 12) & 63]);
    out->push_back(two ? kAlpha[(v >> 6) & 63] : '=');
    out->push_back('=');
  }
}

void EscapeJson(const std::string& value, std::string* out) {
  out->push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '"': out->append("\\\""); break;
      case '\\': out->append("\\\\"); break;
      case '\n': out->append("\\n"); break;
      default: out->push_back(static_cast<char>(c));
    }
  }
  out->push_back('"');
}

dmi_store::S3Config ReadConfig(const std::string& line) {
  dmi_store::S3Config config;
  config.endpoint = FindString(line, "endpoint");
  config.bucket = FindString(line, "bucket");
  config.region = FindString(line, "region");
  config.access_key = FindString(line, "access");
  config.secret_key = FindString(line, "secret");
  // token: null or string.
  {
    const size_t at = line.find("\"token\"");
    if (at != std::string::npos) {
      size_t q = line.find(':', at) + 1;
      while (q < line.size() && line[q] == ' ') ++q;
      if (line.compare(q, 4, "null") != 0 && line[q] == '"') {
        ++q;
        config.session_token = Unescape(line, q);
      }
    }
  }
  config.allow_insecure_http = FindBool(line, "insecure");
  config.connect_timeout_s = static_cast<int>(FindInt(line, "connect_timeout"));
  if (config.connect_timeout_s <= 0) config.connect_timeout_s = 5;
  config.read_timeout_s = static_cast<int>(FindInt(line, "read_timeout"));
  if (config.read_timeout_s <= 0) config.read_timeout_s = 120;
  config.max_attempts = static_cast<int>(FindInt(line, "max_attempts"));
  if (config.max_attempts <= 0) config.max_attempts = 4;
  const int64_t threshold = FindInt(line, "multipart_threshold");
  if (threshold > 0) {
    config.multipart_threshold_bytes = static_cast<uint64_t>(threshold);
  }
  const int64_t chunk = FindInt(line, "multipart_chunk");
  if (chunk > 0) config.multipart_chunk_bytes = static_cast<uint64_t>(chunk);
  return config;
}

}  // namespace

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    const std::string op = FindString(line, "op");
    const std::string key = FindString(line, "key");
    dmi_store::S3Client client(ReadConfig(line));
    std::string error;
    std::string out = "{\"ok\":";
    if (op == "put") {
      std::vector<uint8_t> data;
      DecodeBase64(FindString(line, "data_b64"), &data);
      std::string etag;
      const bool ok = client.PutObject(
          key, data.data(), data.size(), FindMetadata(line),
          FindString(line, "content_type"), &etag, &error);
      out += ok ? "true" : "false";
      if (ok) {
        out += ",\"etag\":";
        EscapeJson(etag, &out);
      } else {
        out += ",\"what\":";
        EscapeJson(error, &out);
      }
    } else if (op == "get") {
      std::vector<uint8_t> data;
      const bool ok = client.GetRange(
          key, static_cast<uint64_t>(FindInt(line, "offset")),
          static_cast<uint64_t>(FindInt(line, "length")), &data, &error);
      out += ok ? "true" : "false";
      if (ok) {
        std::string b64;
        EncodeBase64(data, &b64);
        out += ",\"data_b64\":";
        EscapeJson(b64, &out);
      } else {
        out += ",\"what\":";
        EscapeJson(error, &out);
      }
    } else if (op == "head") {
      const dmi_store::ObjectHead head = client.HeadObject(key, &error);
      // HeadObject reports transport errors only via error string; a missing
      // object is ok:true with found:false.
      if (!error.empty() && !head.found) {
        out += "false,\"what\":";
        EscapeJson(error, &out);
      } else {
        out += "true,\"found\":";
        out += head.found ? "true" : "false";
        out += ",\"size\":" + std::to_string(head.size);
        out += ",\"etag\":";
        EscapeJson(head.etag, &out);
        out += ",\"metadata\":{";
        bool first = true;
        for (const auto& [name, value] : head.metadata) {
          if (!first) out.push_back(',');
          EscapeJson(name, &out);
          out.push_back(':');
          EscapeJson(value, &out);
          first = false;
        }
        out.push_back('}');
      }
    } else if (op == "delete") {
      const bool ok = client.DeleteObject(key, &error);
      out += ok ? "true" : "false";
      if (!ok) {
        out += ",\"what\":";
        EscapeJson(error, &out);
      }
    } else if (op == "list") {
      dmi_store::ListResult result;
      const bool ok = client.ListObjects(
          FindString(line, "prefix"), FindString(line, "delimiter"),
          static_cast<int>(FindInt(line, "max_keys")),
          FindString(line, "continuation"), &result, &error);
      out += ok ? "true" : "false";
      if (ok) {
        out += ",\"truncated\":";
        out += result.truncated ? "true" : "false";
        out += ",\"next_token\":";
        EscapeJson(result.next_token, &out);
        out += ",\"objects\":[";
        bool first = true;
        for (const auto& obj : result.objects) {
          if (!first) out.push_back(',');
          out += "{\"key\":";
          EscapeJson(obj.key, &out);
          out += ",\"size\":" + std::to_string(obj.size) + ",\"etag\":";
          EscapeJson(obj.etag, &out);
          out += "}";
          first = false;
        }
        out.push_back(']');
      } else {
        out += ",\"what\":";
        EscapeJson(error, &out);
      }
    } else {
      out += "false,\"what\":\"unknown op\"";
    }
    out += ",\"attempts\":" + std::to_string(client.last_attempts()) + "}\n";
    std::cout << out;
  }
  return 0;
}
