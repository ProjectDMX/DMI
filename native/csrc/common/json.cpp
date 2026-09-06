#include "json.h"

namespace dmi_common {

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

std::string FindString(const std::string& text, const std::string& key,
                       size_t from) {
  for (const char* sep : {": \"", ":\""}) {
    // NOTE: sep includes the opening quote; search for key+sep.
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle, from);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size() - 1;  // back up onto the opening quote
    ++q;
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
    int64_t v = 0;
    bool neg = false;
    if (q < text.size() && text[q] == '-') {
      neg = true;
      ++q;
    }
    bool any = false;
    while (q < text.size() && text[q] >= '0' && text[q] <= '9') {
      v = v * 10 + (text[q] - '0');
      ++q;
      any = true;
    }
    if (!any) continue;
    return neg ? -v : v;
  }
  return -1;
}

bool HasKey(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    if (text.find("\"" + key + "\"" + sep) != std::string::npos) return true;
  }
  return false;
}

bool FindBool(const std::string& text, const std::string& key) {  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    return text.compare(at + needle.size(), 4, "true") == 0;
  }
  return false;
}

bool FindNull(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    while (q < text.size() && text[q] == ' ') ++q;
    if (text.compare(q, 4, "null") == 0) return true;
  }
  return false;
}

namespace {

std::string FindBracketed(const std::string& text, const std::string& key,
                           char open) {
  const char close = (open == '{') ? '}' : ']';
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep;
    size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size();
    while (q < text.size() && text[q] == ' ') ++q;
    if (q >= text.size() || text[q] != open) continue;
    const size_t start = q;
    int depth = 0;
    bool in_str = false;
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

}  // namespace

std::string FindObject(const std::string& text, const std::string& key) {
  return FindBracketed(text, key, '{');
}

std::string FindArray(const std::string& text, const std::string& key) {
  return FindBracketed(text, key, '[');
}

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

std::string Unwrap(const std::string& wrapped) {
  size_t start = 0, end = wrapped.size();
  while (start < end && wrapped[start] == ' ') ++start;
  while (end > start && wrapped[end - 1] == ' ') --end;
  if (end - start >= 2) return wrapped.substr(start + 1, end - start - 2);
  return "";
}

std::string ParseLiteral(const std::string& literal) {
  size_t q = 0;
  while (q < literal.size() && literal[q] != '"') ++q;
  if (q >= literal.size()) return "";
  ++q;
  return Unescape(literal, q);
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
    if (c == '=') continue;
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

void EncodeBase64(const uint8_t* data, size_t n, std::string* out) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  out->clear();
  out->reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    const uint32_t v = static_cast<uint32_t>(data[i]) << 16 |
                       static_cast<uint32_t>(data[i + 1]) << 8 | data[i + 2];
    out->push_back(kAlpha[(v >> 18) & 63]);
    out->push_back(kAlpha[(v >> 12) & 63]);
    out->push_back(kAlpha[(v >> 6) & 63]);
    out->push_back(kAlpha[v & 63]);
  }
  uint32_t tail = 0;
  size_t left = 0;
  while (i < n) {
    tail = tail << 8 | data[i++];
    ++left;
  }
  if (left == 1) {
    tail <<= 16;
    out->push_back(kAlpha[(tail >> 18) & 63]);
    out->push_back(kAlpha[(tail >> 12) & 63]);
    out->append("==");
  } else if (left == 2) {
    tail <<= 8;
    out->push_back(kAlpha[(tail >> 18) & 63]);
    out->push_back(kAlpha[(tail >> 12) & 63]);
    out->push_back(kAlpha[(tail >> 6) & 63]);
    out->push_back('=');
  }
}

void EncodeBase64(const std::vector<uint8_t>& data, std::string* out) {
  EncodeBase64(data.data(), data.size(), out);
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

}  // namespace dmi_conformance
