// A1 conformance driver: builds packs from a JSON stdin protocol so the Python
// test suite can drive the native writer and compare bytes directly.
//
// Protocol (newline-delimited JSON requests, one JSON response per request):
//   {"op":"build","pack_id":"...","created_at_ns":N,"max_pack_bytes":N,
//    "max_records":N?,
//    "records":[{"metadata":{...canonical field names...},
//                "payload_b64":"..."}, ...]}
//   -> {"ok":true,"pack_id":"...","created_at_ns":N,"record_count":N,
//       "footer_offset":N,"checksum":"...","data_sha256":"...","data_b64":"..."}
//   {"op":"crc32","data_b64":"..."} -> {"ok":true,"crc32":"xxxxxxxx"}
//   {"op":"ping"} -> {"ok":true}
// Errors: {"ok":false,"status":"...","what":"..."}
//
// base64 payload keeps the protocol text-safe; the conformance test decodes and
// hashes on the Python side.

#include "pack_builder.h"

#include <openssl/sha.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace {

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
  if (i + 1 == data.size()) {
    const uint32_t v = data[i] << 16;
    out->push_back(kAlpha[(v >> 18) & 63]);
    out->push_back(kAlpha[(v >> 12) & 63]);
    out->push_back('=');
    out->push_back('=');
  } else if (i + 2 == data.size()) {
    const uint32_t v = data[i] << 16 | data[i + 1] << 8;
    out->push_back(kAlpha[(v >> 18) & 63]);
    out->push_back(kAlpha[(v >> 12) & 63]);
    out->push_back(kAlpha[(v >> 6) & 63]);
    out->push_back('=');
  }
}

}  // namespace

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    // Python json.dumps renders separators as ", " / ": "; accept both that
    // and the compact form when matching the op tag. The build request also
    // carries "crc32"-shaped text nowhere, so order is: ping, crc32, build.
    auto has_op = [&line](const char* op) {
      const std::string spaced = std::string("\"op\": \"") + op + "\"";
      const std::string compact = std::string("\"op\":\"") + op + "\"";
      return line.find(spaced) != std::string::npos ||
             line.find(compact) != std::string::npos;
    };
    if (has_op("ping")) {
      std::cout << "{\"ok\":true}\n";
      continue;
    }
    if (has_op("crc32")) {
      // Extract "data_b64" with either separator style.
      auto find_value = [](const std::string& text, const char* k) -> std::string {
        for (const char* sep : {": ", ":"}) {
          const std::string needle = std::string("\"") + k + "\"" + sep + "\"";
          const size_t at = text.find(needle);
          if (at == std::string::npos) continue;
          const size_t start = at + needle.size();
          const size_t end = text.find('"', start);
          return text.substr(start, end - start);
        }
        return "";
      };
      const std::string b64 = find_value(line, "data_b64");
      std::vector<uint8_t> data;
      if (!DecodeBase64(b64, &data)) {
        std::cout << "{\"ok\":false,\"what\":\"bad base64\"}\n";
        continue;
      }
      const uint32_t crc = dmi_pack::Crc32(data.data(), data.size());
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%08x", crc);
      std::cout << "{\"ok\":true,\"crc32\":\"" << buf << "\"}\n";
      continue;
    }
    if (has_op("build")) {
      // fall through to the build path below
    } else {
      std::cout << "{\"ok\":false,\"what\":\"unknown op\"}\n";
      continue;
    }

    // The build request is parsed by the key scanners below (extract_*,
    // meta_*, find_array), which handle both JSON separator styles. No
    // general JSON parser is needed — or wanted — in a conformance driver.
    auto extract_string = [&](const std::string& k) -> std::string {
      const std::string needle = "\"" + k + "\":";
      const size_t at = line.find(needle);
      if (at == std::string::npos) return "";
      size_t j = at + needle.size();
      while (j < line.size() && line[j] == ' ') ++j;
      if (j >= line.size() || line[j] != '"') return "";
      ++j;
      std::string out;
      while (j < line.size() && line[j] != '"') {
        if (line[j] == '\\' && j + 1 < line.size()) { out.push_back(line[++j]); ++j; }
        else out.push_back(line[j++]);
      }
      return out;
    };
    auto extract_int = [&](const std::string& k) -> int64_t {
      const std::string needle = "\"" + k + "\":";
      const size_t at = line.find(needle);
      if (at == std::string::npos) return -1;
      size_t j = at + needle.size();
      while (j < line.size() && line[j] == ' ') ++j;
      int64_t v = 0;
      while (j < line.size() && line[j] >= '0' && line[j] <= '9') {
        v = v * 10 + (line[j] - '0'); ++j;
      }
      return v;
    };

    // Parse the records array: "records": [ {...}, ... ] — with either
    // separator style. Extract the payload array first, then rebuild a
    // compact text we can scan reliably.
    auto find_array = [](const std::string& text, const char* k) -> std::string {
      for (const char* sep : {": ", ":"}) {
        const std::string needle = std::string("\"") + k + "\"" + sep + "[";
        const size_t at = text.find(needle);
        if (at == std::string::npos) continue;
        size_t j = at + needle.size();
        int depth = 1;
        bool instr = false;
        for (; j < text.size(); ++j) {
          if (instr) {
            if (text[j] == '\\') ++j;
            else if (text[j] == '"') instr = false;
            continue;
          }
          if (text[j] == '"') instr = true;
          else if (text[j] == '[') ++depth;
          else if (text[j] == ']') {
            if (--depth == 0) return text.substr(at + needle.size(), j - (at + needle.size()));
          }
        }
      }
      return "";
    };
    const std::string records_text = find_array(line, "records");
    if (records_text.empty()) {
      std::cout << "{\"ok\":false,\"what\":\"missing records\"}\n";
      continue;
    }

    // Split top-level record objects.
    std::vector<std::string> record_texts;
    {
      int d = 0; bool instr = false; size_t start = 0;
      for (size_t k = 0; k < records_text.size(); ++k) {
        const char c = records_text[k];
        if (instr) {
          if (c == '\\') ++k;
          else if (c == '"') instr = false;
          continue;
        }
        if (c == '"') instr = true;
        else if (c == '{') { if (d == 0) start = k; ++d; }
        else if (c == '}') { --d; if (d == 0) record_texts.push_back(records_text.substr(start, k - start + 1)); }
      }
    }

    dmi_pack::PackBuilder builder(
        extract_string("pack_id"),
        static_cast<uint64_t>(extract_int("created_at_ns")),
        static_cast<uint64_t>(extract_int("max_pack_bytes")),
        static_cast<uint64_t>(extract_int("max_records") > 0
                                  ? extract_int("max_records")
                                  : 1'000'000));

    auto meta_string = [&](const std::string& obj, const std::string& k) {
      const std::string needle = "\"" + k + "\":";
      const size_t at = obj.find(needle);
      if (at == std::string::npos) return std::string();
      size_t q = at + needle.size();
      while (q < obj.size() && obj[q] == ' ') ++q;
      if (q >= obj.size() || obj[q] != '"') return std::string();
      ++q;
      // The value is JSON-ESCAPED text. Unescape the short forms back to raw
      // bytes (\n -> 0x0A etc.) and \uXXXX to UTF-8, so the writer re-encodes
      // them through its own canonical escaper. Without this, the six-char
      // escape sequence "\u0001" flows through as literal text and the JSON
      // written into the footer diverges from the reference.
      std::string raw;
      while (q < obj.size() && obj[q] != '"') {
        if (obj[q] == '\\' && q + 1 < obj.size()) {
          ++q;
          switch (obj[q]) {
            case 'n': raw.push_back('\n'); ++q; break;
            case 't': raw.push_back('\t'); ++q; break;
            case 'r': raw.push_back('\r'); ++q; break;
            case 'b': raw.push_back('\b'); ++q; break;
            case 'f': raw.push_back('\f'); ++q; break;
            case 'u': {
              if (q + 4 < obj.size()) {
                unsigned v = 0;
                for (int k2 = 1; k2 <= 4; ++k2) {
                  const char h = obj[q + k2];
                  v = v * 16 + (h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
                }
                q += 5;
                // Encode as UTF-8 (BMP only; surrogates cannot appear here
                // because json.dumps produced them from real code points and
                // this driver receives exactly its output).
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
              }
              break;
            }
            default: raw.push_back(obj[q]); ++q; break;  // \" \\ / etc.
          }
        } else {
          raw.push_back(obj[q++]);
        }
      }
      return raw;
    };
    auto meta_null = [&](const std::string& obj, const std::string& k) {
      const std::string needle = "\"" + k + "\":";
      const size_t at = obj.find(needle);
      if (at == std::string::npos) return false;
      size_t q = at + needle.size();
      while (q < obj.size() && obj[q] == ' ') ++q;
      return obj.compare(q, 4, "null") == 0;
    };
    auto meta_int = [&](const std::string& obj, const std::string& k) -> int64_t {
      const std::string needle = "\"" + k + "\":";
      const size_t at = obj.find(needle);
      if (at == std::string::npos) return 0;
      size_t q = at + needle.size();
      int64_t v = 0; bool neg = false;
      while (q < obj.size() && obj[q] == ' ') ++q;
      if (q < obj.size() && obj[q] == '-') { neg = true; ++q; }
      while (q < obj.size() && obj[q] >= '0' && obj[q] <= '9') {
        v = v * 10 + (obj[q] - '0'); ++q;
      }
      return neg ? -v : v;
    };
    auto meta_shape = [&](const std::string& obj) {
      std::vector<uint32_t> shape;
      // Either separator style after the key.
      for (const char* sep : {": ", ":"}) {
        const std::string needle = std::string("\"shape\"") + sep + "[";
        const size_t at = obj.find(needle);
        if (at == std::string::npos) continue;
        size_t q = at + needle.size();
        while (q < obj.size() && obj[q] != ']') {
          if (obj[q] >= '0' && obj[q] <= '9') {
            uint32_t v = 0;
            while (q < obj.size() && obj[q] >= '0' && obj[q] <= '9') {
              v = v * 10 + static_cast<uint32_t>(obj[q] - '0');
              ++q;
            }
            shape.push_back(v);
          } else {
            ++q;
          }
        }
        return shape;
      }
      return shape;
    };
    auto meta_payload = [&](const std::string& obj) {
      std::vector<uint8_t> payload;
      // Either separator style; a base64 value can be empty ("").
      for (const char* sep : {": ", ":"}) {
        const std::string needle =
            std::string("\"payload_b64\"") + sep + "\"";
        const size_t at = obj.find(needle);
        if (at == std::string::npos) continue;
        size_t q = at + needle.size();
        const size_t end = obj.find('"', q);
        if (end == std::string::npos) return payload;
        DecodeBase64(obj.substr(q, end - q), &payload);
        return payload;
      }
      return payload;
    };

    std::string fail_status;
    std::string fail_what;
    for (const auto& rt : record_texts) {
      dmi_pack::PackRecord rec;
      rec.metadata.capture_id = meta_string(rt, "capture_id");
      rec.metadata.tenant_id = meta_string(rt, "tenant_id");
      rec.metadata.experiment_id = meta_string(rt, "experiment_id");
      rec.metadata.run_id = meta_string(rt, "run_id");
      rec.metadata.session_id = meta_string(rt, "session_id");
      rec.metadata.request_id = meta_string(rt, "request_id");
      rec.metadata.sequence_id = meta_string(rt, "sequence_id");
      rec.metadata.model_id = meta_string(rt, "model_id");
      rec.metadata.model_revision = meta_string(rt, "model_revision");
      if (!meta_null(rt, "adapter_revision")) {
        rec.metadata.adapter_revision = meta_string(rt, "adapter_revision");
      }
      rec.metadata.capture_policy_version = meta_string(rt, "capture_policy_version");
      rec.metadata.hook_name = meta_string(rt, "hook_name");
      rec.metadata.layer_number = meta_int(rt, "layer_number");
      rec.metadata.producer_rank = static_cast<uint64_t>(meta_int(rt, "producer_rank"));
      rec.metadata.step_number = static_cast<uint64_t>(meta_int(rt, "step_number"));
      rec.metadata.token_start = static_cast<uint64_t>(meta_int(rt, "token_start"));
      rec.metadata.token_end = static_cast<uint64_t>(meta_int(rt, "token_end"));
      rec.metadata.batch_position = static_cast<uint64_t>(meta_int(rt, "batch_position"));
      rec.metadata.dtype = meta_string(rt, "dtype");
      rec.metadata.shape = meta_shape(rt);
      rec.metadata.captured_at_ns = static_cast<uint64_t>(meta_int(rt, "captured_at_ns"));
      // payload
      static std::vector<std::vector<uint8_t>> payload_storage;
      payload_storage.push_back(meta_payload(rt));
      rec.payload = payload_storage.back().data();
      rec.payload_bytes = payload_storage.back().size();
      const dmi_pack::Status st = builder.Append(rec);
      if (st != dmi_pack::Status::kOk) {
        fail_status = dmi_pack::StatusName(st);
        fail_what = rec.metadata.capture_id;
        break;
      }
    }

    if (!fail_status.empty()) {
      std::cout << "{\"ok\":false,\"status\":\"" << fail_status
                << "\",\"what\":\"" << fail_what << "\"}\n";
      continue;
    }

    dmi_pack::SealedPack sealed;
    const dmi_pack::Status st = builder.Seal(&sealed);
    if (st != dmi_pack::Status::kOk) {
      std::cout << "{\"ok\":false,\"status\":\"" << dmi_pack::StatusName(st)
                << "\",\"what\":\"seal\"}\n";
      continue;
    }

    std::string b64;
    EncodeBase64(sealed.data, &b64);
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(sealed.data.data(), sealed.data.size(), digest);
    char hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
      std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    }
    std::cout << "{\"ok\":true,\"pack_id\":\"" << sealed.pack_id
              << "\",\"created_at_ns\":" << sealed.created_at_ns
              << ",\"record_count\":" << sealed.record_count
              << ",\"footer_offset\":" << sealed.footer_offset
              << ",\"checksum\":\"" << sealed.checksum
              << "\",\"data_sha256\":\"" << hex
              << "\",\"data_b64\":\"" << b64 << "\"}\n";
  }
  return 0;
}
