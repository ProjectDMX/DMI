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

#include "../common/json.h"
#include "spool.h"
#include "uploader.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>



namespace jc = dmi_common;

// The key of the first integer literal on this line that does not fit in the
// 64-bit union, empty when there was none. One latch per line is enough: the
// driver is single-threaded and handles one op per line.
//
// The since-removed FindInt wrapper reported such a literal as -1, and -1 is
// a legal answer for none of
// these fields. The fields tested for "> 0" as a stand-in for "was it given"
// silently fell back to their defaults -- 5s, 120s, 4 attempts, 4 workers,
// and the multipart sizes. The fields cast straight to uint64_t became
// 18446744073709551615 instead: a range offset or length, a StagedPack
// counter, and the spool's max_bytes (whose 1 TiB fallback is keyed on ZERO,
// which -1 is not, so the limit became no limit). In every case the caller's
// own bound was replaced by a different one and the op reported success.
std::string g_out_of_range;

int64_t Integer(const std::string& text, const char* key) {
  int64_t value = 0;
  const jc::IntFind found = jc::FindIntChecked(text, key, &value);
  if (found == jc::IntFind::kOutOfRange && g_out_of_range.empty()) {
    g_out_of_range = key;
  }
  // kAbsent keeps answering -1: an absent timeout still means "default", and
  // upload_pending's limit of -1 means "no limit".
  return found == jc::IntFind::kOk ? value : -1;
}

// metadata object: {"k":"v",...} — string values only.
std::map<std::string, std::string> FindMetadata(const std::string& text) {
  std::map<std::string, std::string> out;
  for (const auto& item :
       jc::SplitElements(jc::Unwrap(jc::FindObject(text, "metadata")))) {
    const size_t colon = item.find(':');
    if (colon == std::string::npos) continue;
    const std::string name = jc::ParseLiteral(item.substr(0, colon));
    const std::string value = jc::ParseLiteral(item.substr(colon + 1));
    if (!name.empty()) out[name] = value;
  }
  return out;
}

dmi_store::S3Config ReadConfig(const std::string& line) {
  dmi_store::S3Config config;
  config.endpoint = jc::FindString(line, "endpoint");
  config.bucket = jc::FindString(line, "bucket");
  config.region = jc::FindString(line, "region");
  config.access_key = jc::FindString(line, "access");
  config.secret_key = jc::FindString(line, "secret");
  // token: null or string.
  {
    const size_t at = line.find("\"token\"");
    if (at != std::string::npos) {
      size_t q = line.find(':', at) + 1;
      while (q < line.size() && line[q] == ' ') ++q;
      if (line.compare(q, 4, "null") != 0 && line[q] == '"') {
        ++q;
        config.session_token = jc::Unescape(line, q);
      }
    }
  }
  config.allow_insecure_http = jc::FindBool(line, "insecure");
  const int64_t connect_timeout = Integer(line, "connect_timeout");
  config.connect_timeout_s =
      static_cast<int>(connect_timeout > 0 ? connect_timeout : 5);
  const int64_t read_timeout = Integer(line, "read_timeout");
  config.read_timeout_s =
      static_cast<int>(read_timeout > 0 ? read_timeout : 120);
  const int64_t max_attempts = Integer(line, "max_attempts");
  config.max_attempts =
      static_cast<int>(max_attempts > 0 ? max_attempts : 4);
  const int64_t threshold = Integer(line, "multipart_threshold");
  if (threshold > 0) {
    config.multipart_threshold_bytes = static_cast<uint64_t>(threshold);
  }
  const int64_t chunk = Integer(line, "multipart_chunk");
  if (chunk > 0) config.multipart_chunk_bytes = static_cast<uint64_t>(chunk);
  return config;
}

dmi_store::StagedPack ParseStaged(const std::string& obj) {
  dmi_store::StagedPack s;
  s.pack_id = jc::FindString(obj, "pack_id");
  s.created_at_ns = static_cast<uint64_t>(Integer(obj, "created_at_ns"));
  s.record_count = static_cast<uint64_t>(Integer(obj, "record_count"));
  s.checksum = jc::FindString(obj, "checksum");
  s.object_key = jc::FindString(obj, "object_key");
  s.path = jc::FindString(obj, "path");
  s.object_bytes = static_cast<uint64_t>(Integer(obj, "object_bytes"));
  return s;
}

void EmitRef(const dmi_store::PackRef& ref, std::string* out) {
  out->append("{\"pack_id\":");
  jc::EscapeJson(ref.pack_id, out);
  out->append(",\"store_id\":");
  jc::EscapeJson(ref.store_id, out);
  out->append(",\"object_key\":");
  jc::EscapeJson(ref.object_key, out);
  out->append(",\"object_bytes\":" + std::to_string(ref.object_bytes));
  out->append(",\"checksum\":");
  jc::EscapeJson(ref.checksum, out);
  out->append(",\"record_count\":" + std::to_string(ref.record_count) + "}");
}

dmi_store::UploaderConfig ReadUploaderConfig(const std::string& line) {
  dmi_store::UploaderConfig config;
  const int64_t workers = Integer(line, "max_workers");
  config.max_workers = static_cast<int>(workers > 0 ? workers : 4);
  const int64_t in_flight = Integer(line, "max_in_flight_bytes");
  if (in_flight > 0) {
    config.max_in_flight_bytes = static_cast<uint64_t>(in_flight);
  }
  // Uploader-level attempts are keyed separately from the transport's
  // "max_attempts": the transport absorbs single 5xx inside one upload
  // attempt (like botocore), and only its exhaustion surfaces here.
  const int64_t attempts = Integer(line, "upload_max_attempts");
  if (attempts > 0) config.max_attempts = static_cast<int>(attempts);
  config.store_id = jc::FindString(line, "store_id");
  if (config.store_id.empty()) config.store_id = "s3";
  return config;
}

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  // The refusal for an integer literal wider than 64 bits, in the ok:false /
  // what shape this driver documents for every error. Emitted on its own,
  // without the trailing "attempts", because nothing was ever attempted.
  const auto refuse_out_of_range = [] {
    std::string out = "{\"ok\":false,\"what\":";
    jc::EscapeJson("integer field " + g_out_of_range + " is out of range",
                   &out);
    std::cout << out << "}\n";
  };
  while (std::getline(std::cin, line)) {
    g_out_of_range.clear();
    const std::string op = jc::FindString(line, "op");
    const std::string key = jc::FindString(line, "key");
    dmi_store::S3Client client(ReadConfig(line));
    // Before any request goes out: a timeout or attempt count that cannot be
    // represented must not be replaced by the default.
    if (!g_out_of_range.empty()) {
      refuse_out_of_range();
      continue;
    }
    std::string error;
    std::string out = "{\"ok\":";
    if (op == "put") {
      std::vector<uint8_t> data;
      jc::DecodeBase64(jc::FindString(line, "data_b64"), &data);
      std::string etag;
      const bool ok = client.PutObject(
          key, data.data(), data.size(), FindMetadata(line),
          jc::FindString(line, "content_type"), &etag, &error);
      out += ok ? "true" : "false";
      if (ok) {
        out += ",\"etag\":";
        jc::EscapeJson(etag, &out);
      } else {
        out += ",\"what\":";
        jc::EscapeJson(error, &out);
      }
    } else if (op == "get") {
      std::vector<uint8_t> data;
      // Parsed before the call, not inside its argument list: a range built
      // from -1 asks the server for 18446744073709551615 bytes.
      const uint64_t offset = static_cast<uint64_t>(Integer(line, "offset"));
      const uint64_t length = static_cast<uint64_t>(Integer(line, "length"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      const bool ok =
          client.GetRange(key, offset, length, &data, &error);
      out += ok ? "true" : "false";
      if (ok) {
        std::string b64;
        jc::EncodeBase64(data, &b64);
        out += ",\"data_b64\":";
        jc::EscapeJson(b64, &out);
      } else {
        out += ",\"what\":";
        jc::EscapeJson(error, &out);
      }
    } else if (op == "head") {
      const dmi_store::ObjectHead head = client.HeadObject(key, &error);
      // HeadObject reports transport errors only via error string; a missing
      // object is ok:true with found:false.
      if (!error.empty() && !head.found) {
        out += "false,\"what\":";
        jc::EscapeJson(error, &out);
      } else {
        out += "true,\"found\":";
        out += head.found ? "true" : "false";
        out += ",\"size\":" + std::to_string(head.size);
        out += ",\"etag\":";
        jc::EscapeJson(head.etag, &out);
        out += ",\"metadata\":{";
        bool first = true;
        for (const auto& [name, value] : head.metadata) {
          if (!first) out.push_back(',');
          jc::EscapeJson(name, &out);
          out.push_back(':');
          jc::EscapeJson(value, &out);
          first = false;
        }
        out.push_back('}');
      }
    } else if (op == "delete") {
      const bool ok = client.DeleteObject(key, &error);
      out += ok ? "true" : "false";
      if (!ok) {
        out += ",\"what\":";
        jc::EscapeJson(error, &out);
      }
    } else if (op == "list") {
      dmi_store::ListResult result;
      const int64_t max_keys = Integer(line, "max_keys");
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      const bool ok = client.ListObjects(
          jc::FindString(line, "prefix"), jc::FindString(line, "delimiter"),
          static_cast<int>(max_keys), jc::FindString(line, "continuation"),
          &result, &error);
      out += ok ? "true" : "false";
      if (ok) {
        out += ",\"truncated\":";
        out += result.truncated ? "true" : "false";
        out += ",\"next_token\":";
        jc::EscapeJson(result.next_token, &out);
        out += ",\"objects\":[";
        bool first = true;
        for (const auto& obj : result.objects) {
          if (!first) out.push_back(',');
          out += "{\"key\":";
          jc::EscapeJson(obj.key, &out);
          out += ",\"size\":" + std::to_string(obj.size) + ",\"etag\":";
          jc::EscapeJson(obj.etag, &out);
          out += "}";
          first = false;
        }
        out.push_back(']');
      } else {
        out += ",\"what\":";
        jc::EscapeJson(error, &out);
      }
    } else if (op == "upload_one") {
      // Upload one staged entry (parsed from the nested "staged" object).
      dmi_store::SpoolConfig spool_config;
      spool_config.root = jc::FindString(line, "root");
      spool_config.max_bytes =
          static_cast<uint64_t>(Integer(line, "spool_max_bytes"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      if (spool_config.max_bytes == 0) spool_config.max_bytes = 1ull << 40;
      dmi_store::Spool spool;
      std::string spool_error;
      if (dmi_store::Spool::Open(spool_config, &spool, &spool_error) !=
          dmi_store::SpoolStatus::kOk) {
        out += "false,\"what\":";
        jc::EscapeJson("spool open: " + spool_error, &out);
      } else {
        const dmi_store::UploaderConfig uploader_config =
            ReadUploaderConfig(line);
        const dmi_store::StagedPack staged =
            ParseStaged(jc::FindObject(line, "staged"));
        // Both parses land before the upload: a staged counter read as -1
        // becomes 18446744073709551615 and fails the checksum gate for the
        // wrong reason.
        if (!g_out_of_range.empty()) {
          refuse_out_of_range();
          continue;
        }
        dmi_store::SpoolUploader uploader(&spool, &client, uploader_config);
        dmi_store::PackRef ref;
        int attempts = 0;
        std::string error;
        const bool ok = uploader.UploadOne(staged, &ref, &attempts, &error);
        out += ok ? "true" : "false";
        if (ok) {
          out += ",\"ref\":";
          EmitRef(ref, &out);
        } else {
          out += ",\"what\":";
          jc::EscapeJson(error, &out);
        }
        out += ",\"upload_attempts\":" + std::to_string(attempts);
      }
    } else if (op == "upload_pending") {
      dmi_store::SpoolConfig spool_config;
      spool_config.root = jc::FindString(line, "root");
      spool_config.max_bytes =
          static_cast<uint64_t>(Integer(line, "spool_max_bytes"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      if (spool_config.max_bytes == 0) spool_config.max_bytes = 1ull << 40;
      dmi_store::Spool spool;
      std::string spool_error;
      if (dmi_store::Spool::Open(spool_config, &spool, &spool_error) !=
          dmi_store::SpoolStatus::kOk) {
        out += "false,\"what\":";
        jc::EscapeJson("spool open: " + spool_error, &out);
      } else {
        const dmi_store::UploaderConfig uploader_config =
            ReadUploaderConfig(line);
        const int64_t limit = Integer(line, "limit");
        // Before the batch runs: a worker count or a batch limit that cannot
        // be represented must not become the default.
        if (!g_out_of_range.empty()) {
          refuse_out_of_range();
          continue;
        }
        dmi_store::SpoolUploader uploader(&spool, &client, uploader_config);
        const dmi_store::UploadBatchResult result =
            uploader.UploadPending(limit < 0 ? -1 : static_cast<int>(limit));
        out += "true,\"refs\":[";
        bool first = true;
        for (const auto& ref : result.refs) {
          if (!first) out.push_back(',');
          EmitRef(ref, &out);
          first = false;
        }
        out += "],\"failures\":[";
        first = true;
        for (const auto& failure : result.failures) {
          if (!first) out.push_back(',');
          out += "{\"pack_id\":";
          jc::EscapeJson(failure.pack_id, &out);
          out += ",\"object_key\":";
          jc::EscapeJson(failure.object_key, &out);
          out += ",\"attempts\":" + std::to_string(failure.attempts);
          out += ",\"error\":";
          jc::EscapeJson(failure.error, &out);
          out += "}";
          first = false;
        }
        const auto& snap = result.snapshot;
        out += "],\"snapshot\":{\"attempted_packs\":" +
               std::to_string(snap.attempted_packs) + ",\"uploaded_packs\":" +
               std::to_string(snap.uploaded_packs) + ",\"uploaded_bytes\":" +
               std::to_string(snap.uploaded_bytes) + ",\"failed_packs\":" +
               std::to_string(snap.failed_packs) + ",\"retries\":" +
               std::to_string(snap.retries) + ",\"peak_active_uploads\":" +
               std::to_string(snap.peak_active_uploads) +
               ",\"peak_in_flight_bytes\":" +
               std::to_string(snap.peak_in_flight_bytes) +
               ",\"duration_count\":" +
               std::to_string(snap.duration_count) + "}";
      }
    } else {
      out += "false,\"what\":\"unknown op\"";
    }
    out += ",\"attempts\":" + std::to_string(client.last_attempts()) + "}\n";
    std::cout << out;
  }
  return 0;
}
