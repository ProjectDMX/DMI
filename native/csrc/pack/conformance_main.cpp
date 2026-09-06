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

#include "../conformance/json_scan.h"

#include <openssl/sha.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace jc = dmi_conformance;

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
      const std::string b64 = jc::FindString(line, "data_b64");
      std::vector<uint8_t> data;
      if (!jc::DecodeBase64(b64, &data)) {
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
      return jc::FindString(line, k);
    };
    auto extract_int = [&](const std::string& k) -> int64_t {
      return jc::FindInt(line, k);
    };

    // Parse the records array: "records": [ {...}, ... ] — with either
    // separator style.
    const std::string records_text =
        jc::Unwrap(jc::FindArray(line, "records"));
    if (records_text.empty()) {
      std::cout << "{\"ok\":false,\"what\":\"missing records\"}\n";
      continue;
    }

    // Split top-level record objects.
    std::vector<std::string> record_texts;
    for (const auto& item : jc::SplitElements(records_text)) {
      // Items are {...} objects (whitespace around them is harmless).
      size_t start = item.find('{');
      if (start != std::string::npos) record_texts.push_back(item);
    }

    dmi_pack::PackBuilder builder(
        extract_string("pack_id"),
        static_cast<uint64_t>(extract_int("created_at_ns")),
        static_cast<uint64_t>(extract_int("max_pack_bytes")),
        static_cast<uint64_t>(extract_int("max_records") > 0
                                  ? extract_int("max_records")
                                  : 1'000'000));

    auto meta_string = [&](const std::string& obj, const std::string& k) {
      // Shared FindString unescapes JSON escapes back to raw bytes, so the
      // writer re-encodes them through its own canonical escaper.
      return jc::FindString(obj, k);
    };
    auto meta_null = [&](const std::string& obj, const std::string& k) {
      return jc::FindNull(obj, k);
    };
    auto meta_int = [&](const std::string& obj, const std::string& k) -> int64_t {
      return jc::FindInt(obj, k);
    };
    auto meta_shape = [&](const std::string& obj) {
      std::vector<uint32_t> shape;
      for (const auto& item : jc::SplitElements(jc::Unwrap(jc::FindArray(obj, "shape")))) {
        size_t q = 0;
        while (q < item.size() && item[q] == ' ') ++q;
        uint32_t v = 0;
        bool any = false;
        while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
          v = v * 10 + static_cast<uint32_t>(item[q] - '0');
          ++q;
          any = true;
        }
        if (any) shape.push_back(v);
      }
      return shape;
    };
    auto meta_payload = [&](const std::string& obj) {
      std::vector<uint8_t> payload;
      jc::DecodeBase64(jc::FindString(obj, "payload_b64"), &payload);
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
    jc::EncodeBase64(sealed.data, &b64);
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
