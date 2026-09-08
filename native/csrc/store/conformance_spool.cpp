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

#include "../common/json.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>



namespace jc = dmi_common;

// The key of the first integer literal on this line that does not fit in 64
// bits, empty when there was none. One latch per line is enough: the driver
// is single-threaded and handles one op per line.
//
// FindInt reported such a literal as -1, which is a legal value here for
// nothing at all -- `max_bytes` read it as "not given" and took the 1 TiB
// default, and the StagedPack counters cast it to 18446744073709551615. So
// the op has to refuse, which means knowing WHICH field did it.
std::string g_out_of_range;

int64_t Integer(const std::string& text, const char* key) {
  int64_t value = 0;
  const jc::IntFind found = jc::FindIntChecked(text, key, &value);
  if (found == jc::IntFind::kOutOfRange && g_out_of_range.empty()) {
    g_out_of_range = key;
  }
  // kAbsent keeps FindInt's -1: an absent max_bytes still means "default".
  return found == jc::IntFind::kOk ? value : -1;
}

void EmitStaged(const dmi_store::StagedPack& s, std::string* out) {
  out->append("{\"pack_id\":");
  jc::EscapeJson(s.pack_id, out);
  out->append(",\"created_at_ns\":" + std::to_string(s.created_at_ns));
  out->append(",\"record_count\":" + std::to_string(s.record_count));
  out->append(",\"checksum\":");
  jc::EscapeJson(s.checksum, out);
  out->append(",\"object_key\":");
  jc::EscapeJson(s.object_key, out);
  out->append(",\"path\":");
  jc::EscapeJson(s.path, out);
  out->append(",\"object_bytes\":" + std::to_string(s.object_bytes) + "}");
}

dmi_store::StagedPack ParseStaged(const std::string& obj) {  dmi_store::StagedPack s;
  s.pack_id = jc::FindString(obj, "pack_id");
  s.created_at_ns = static_cast<uint64_t>(Integer(obj, "created_at_ns"));
  s.record_count = static_cast<uint64_t>(Integer(obj, "record_count"));
  s.checksum = jc::FindString(obj, "checksum");
  s.object_key = jc::FindString(obj, "object_key");
  s.path = jc::FindString(obj, "path");
  s.object_bytes = static_cast<uint64_t>(Integer(obj, "object_bytes"));
  return s;
}

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  // One spool per process would force one driver per root; instead the
  // driver re-opens per op (cheap: a directory scan) which also exercises
  // the constructor accounting path every call.
  // The refusal for an integer literal wider than 64 bits, emitted in the
  // same shape (and without the post-op snapshot) as a failed open, which is
  // the driver's other refusal from before the op runs.
  const auto refuse_out_of_range = [] {
    std::string out = "{\"ok\":false,\"what\":";
    jc::EscapeJson("integer field " + g_out_of_range + " is out of range",
                   &out);
    out += "}\n";
    std::cout << out;
  };
  while (std::getline(std::cin, line)) {
    g_out_of_range.clear();
    const std::string op = jc::FindString(line, "op");
    const std::string root = jc::FindString(line, "root");
    dmi_store::SpoolConfig config;
    config.root = root;
    config.max_bytes = static_cast<uint64_t>(Integer(line, "max_bytes"));
    if (!g_out_of_range.empty()) {
      refuse_out_of_range();
      continue;
    }
    if (config.max_bytes == 0) config.max_bytes = 1ull << 40;
    dmi_store::Spool spool;
    std::string error;
    if (dmi_store::Spool::Open(config, &spool, &error) !=
        dmi_store::SpoolStatus::kOk) {
      std::string out = "{\"ok\":false,\"status\":\"open\",\"what\":";
      jc::EscapeJson(error, &out);
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
      jc::DecodeBase64(jc::FindString(line, "data_b64"), &data);
      dmi_store::StagedPack staged;
      // Parsed before the call, not inside its argument list: an out-of-range
      // literal has to stop the staging rather than be discovered after the
      // ready file is already on disk.
      const uint64_t created_at_ns =
          static_cast<uint64_t>(Integer(line, "created_at_ns"));
      const uint64_t record_count =
          static_cast<uint64_t>(Integer(line, "record_count"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      const dmi_store::SpoolStatus st = spool.Stage(
          jc::FindString(line, "pack_id"), created_at_ns, record_count,
          jc::FindString(line, "checksum"), jc::FindString(line, "object_key"),
          data.data(), data.size(), &staged, &error);
      ok = (st == dmi_store::SpoolStatus::kOk);
      if (ok) {
        body += "\"staged\":";
        EmitStaged(staged, &body);
      } else {
        body += "\"status\":";
        jc::EscapeJson(dmi_store::SpoolStatusName(st), &body);
        body += ",\"what\":";
        jc::EscapeJson(error, &body);
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
        jc::EscapeJson(error, &body);
      }
    } else if (op == "remove") {
      const dmi_store::StagedPack staged = ParseStaged(jc::FindObject(line, "staged"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      const dmi_store::SpoolStatus st = spool.Remove(staged, &error);
      ok = (st == dmi_store::SpoolStatus::kOk);
      if (!ok) {
        body += "\"what\":";
        jc::EscapeJson(error, &body);
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
