// Spool conformance driver: stdin/stdout JSON protocol for the pytest suite.
//
// Ops (stateful: one Spool per process, opened by the first op carrying it):
//   {"op":"stage","root":"...","max_bytes":N,"pack_id":"...",
//    "created_at_ns":N,"record_count":N,"checksum":"...",
//    "object_key":"...","data_b64":"..."}
//     -> {"ok":true,"staged":{...StagedPack...}}
//   {"op":"recover","root":"...","max_bytes":N}
//     -> {"ok":true,"staged":[{...},...]}
//   {"op":"remove","root":"...","max_bytes":N,"staged":{...}}
//     -> {"ok":true}
//   {"op":"snapshot","root":"...","max_bytes":N}
//     -> {"ok":true,"snapshot":{"entries":N,"bytes":N,"peak_bytes":N,"max_bytes":N}}
// Errors: {"ok":false,"status":"...","what":"..."}.

#include "spool.h"

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
    int64_t v = 0;
    while (q < text.size() && text[q] >= '0' && text[q] <= '9') {
      v = v * 10 + (text[q] - '0');
      ++q;
    }
    return v;
  }
  return 0;
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

void EmitStaged(const dmi_store::StagedPack& s, std::string* out) {
  out->append("{\"pack_id\":");
  EscapeJson(s.pack_id, out);
  out->append(",\"created_at_ns\":" + std::to_string(s.created_at_ns));
  out->append(",\"record_count\":" + std::to_string(s.record_count));
  out->append(",\"checksum\":");
  EscapeJson(s.checksum, out);
  out->append(",\"object_key\":");
  EscapeJson(s.object_key, out);
  out->append(",\"path\":");
  EscapeJson(s.path, out);
  out->append(",\"object_bytes\":" + std::to_string(s.object_bytes) + "}");
}

// A nested "staged":{...} value, located by brace matching.
std::string FindObject(const std::string& text, const std::string& key) {
  for (const char* sep : {": ", ":"}) {
    const std::string needle = "\"" + key + "\"" + sep + "{";
    const size_t at = text.find(needle);
    if (at == std::string::npos) continue;
    size_t q = at + needle.size() - 1;
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
      else if (text[q] == '{') ++depth;
      else if (text[q] == '}') {
        if (--depth == 0) return text.substr(start, q - start + 1);
      }
    }
  }
  return "";
}

dmi_store::StagedPack ParseStaged(const std::string& obj) {
  dmi_store::StagedPack s;
  s.pack_id = FindString(obj, "pack_id");
  s.created_at_ns = static_cast<uint64_t>(FindInt(obj, "created_at_ns"));
  s.record_count = static_cast<uint64_t>(FindInt(obj, "record_count"));
  s.checksum = FindString(obj, "checksum");
  s.object_key = FindString(obj, "object_key");
  s.path = FindString(obj, "path");
  s.object_bytes = static_cast<uint64_t>(FindInt(obj, "object_bytes"));
  return s;
}

}  // namespace

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  // One spool per process would force one driver per root; instead the
  // driver re-opens per op (cheap: a directory scan) which also exercises
  // the constructor accounting path every call.
  while (std::getline(std::cin, line)) {
    const std::string op = FindString(line, "op");
    const std::string root = FindString(line, "root");
    dmi_store::SpoolConfig config;
    config.root = root;
    config.max_bytes = static_cast<uint64_t>(FindInt(line, "max_bytes"));
    if (config.max_bytes == 0) config.max_bytes = 1ull << 40;
    dmi_store::Spool spool;
    std::string error;
    if (dmi_store::Spool::Open(config, &spool, &error) !=
        dmi_store::SpoolStatus::kOk) {
      std::string out = "{\"ok\":false,\"status\":\"open\",\"what\":";
      EscapeJson(error, &out);
      out += "}\n";
      std::cout << out;
      continue;
    }
    // Every op reports the post-op snapshot: the driver re-opens per op, so
    // the constructor accounting (not live counters) is what gets verified.
    std::string body;
    bool ok = true;
    if (op == "stage") {
      std::vector<uint8_t> data;
      DecodeBase64(FindString(line, "data_b64"), &data);
      dmi_store::StagedPack staged;
      const dmi_store::SpoolStatus st = spool.Stage(
          FindString(line, "pack_id"),
          static_cast<uint64_t>(FindInt(line, "created_at_ns")),
          static_cast<uint64_t>(FindInt(line, "record_count")),
          FindString(line, "checksum"), FindString(line, "object_key"),
          data.data(), data.size(), &staged, &error);
      ok = (st == dmi_store::SpoolStatus::kOk);
      if (ok) {
        body += "\"staged\":";
        EmitStaged(staged, &body);
      } else {
        body += "\"status\":";
        EscapeJson(dmi_store::SpoolStatusName(st), &body);
        body += ",\"what\":";
        EscapeJson(error, &body);
      }
    } else if (op == "recover") {
      std::vector<dmi_store::StagedPack> staged;
      const dmi_store::SpoolStatus st = spool.Recover(&staged, &error);
      ok = (st == dmi_store::SpoolStatus::kOk);
      if (ok) {
        body += "\"staged\":[";
        bool first = true;
        for (const auto& s : staged) {
          if (!first) body.push_back(',');
          EmitStaged(s, &body);
          first = false;
        }
        body.push_back(']');
      } else {
        body += "\"what\":";
        EscapeJson(error, &body);
      }
    } else if (op == "remove") {
      const dmi_store::StagedPack staged = ParseStaged(FindObject(line, "staged"));
      const dmi_store::SpoolStatus st = spool.Remove(staged, &error);
      ok = (st == dmi_store::SpoolStatus::kOk);
      if (!ok) {
        body += "\"what\":";
        EscapeJson(error, &body);
      }
    } else if (op == "snapshot") {
      const dmi_store::SpoolSnapshot snap = spool.Snapshot();
      std::string out = "{\"ok\":true,\"snapshot\":{\"entries\":" +
                        std::to_string(snap.entries) + ",\"bytes\":" +
                        std::to_string(snap.bytes) + ",\"peak_bytes\":" +
                        std::to_string(snap.peak_bytes) + ",\"max_bytes\":" +
                        std::to_string(snap.max_bytes) + "}}\n";
      std::cout << out;
      continue;
    } else {
      ok = false;
      body += "\"what\":\"unknown op\"";
    }
    const dmi_store::SpoolSnapshot snap = spool.Snapshot();
    std::string out = std::string("{\"ok\":") + (ok ? "true" : "false");
    if (!body.empty()) {
      out.push_back(',');
      out += body;
    }
    out += ",\"snapshot\":{\"entries\":" + std::to_string(snap.entries) +
           ",\"bytes\":" + std::to_string(snap.bytes) + "}}\n";
    std::cout << out;
  }
  return 0;
}
