// Sink conformance driver: stdin/stdout JSON protocol for the pytest suite.
//
// One PackSink per process, configured by the first "open":
//   {"op":"open","root":"...","max_bytes":N,"max_queue_records":N,
//    "max_queue_bytes":N,"max_pack_bytes":N,"max_pack_records":N,
//    "max_linger_ns":N,"overload":"block"|"drop_newest",
//    "admission_timeout":-1}
//     -> {"ok":true}
//   {"op":"submit","metadata":{...canonical field names...},"payload_b64":"..."}
//     -> {"ok":true,"admission":"accepted"|...}
//   {"op":"flush","timeout":-1} -> {"ok":true}|{"ok":false,"what":"timeout"|...}
//   {"op":"close","timeout":30} -> {"ok":true,"snapshot":{...}}
//   {"op":"snapshot"} -> {"ok":true,"snapshot":{...}}
//   {"op":"object_key","tenant_id":"...","session_id":"...","producer_rank":N,
//    "captured_at_ns":N,"pack_id":"..."} -> {"ok":true,"object_key":"..."}
// Errors: {"ok":false,"what":"..."}.

#include "pack_sink.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../conformance/json_scan.h"
#include "object_key.h"

namespace jc = dmi_conformance;

namespace {

dmi_pack::RecordMetadata ParseMetadata(const std::string& obj) {
  dmi_pack::RecordMetadata m;
  m.capture_id = jc::FindString(obj, "capture_id");
  m.tenant_id = jc::FindString(obj, "tenant_id");
  m.experiment_id = jc::FindString(obj, "experiment_id");
  m.run_id = jc::FindString(obj, "run_id");
  m.session_id = jc::FindString(obj, "session_id");
  m.request_id = jc::FindString(obj, "request_id");
  m.sequence_id = jc::FindString(obj, "sequence_id");
  m.model_id = jc::FindString(obj, "model_id");
  m.model_revision = jc::FindString(obj, "model_revision");
  if (!jc::FindNull(obj, "adapter_revision")) {
    m.adapter_revision = jc::FindString(obj, "adapter_revision");
  }
  m.capture_policy_version = jc::FindString(obj, "capture_policy_version");
  m.hook_name = jc::FindString(obj, "hook_name");
  m.layer_number = jc::FindInt(obj, "layer_number");
  m.producer_rank = static_cast<uint64_t>(jc::FindInt(obj, "producer_rank"));
  m.step_number = static_cast<uint64_t>(jc::FindInt(obj, "step_number"));
  m.token_start = static_cast<uint64_t>(jc::FindInt(obj, "token_start"));
  m.token_end = static_cast<uint64_t>(jc::FindInt(obj, "token_end"));
  m.batch_position =
      static_cast<uint64_t>(jc::FindInt(obj, "batch_position"));
  m.dtype = jc::FindString(obj, "dtype");
  for (const auto& item : jc::SplitElements(
           jc::Unwrap(jc::FindArray(obj, "shape")))) {
    size_t q = 0;
    while (q < item.size() && item[q] == ' ') ++q;
    uint32_t v = 0;
    bool any = false;
    while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
      v = v * 10 + static_cast<uint32_t>(item[q] - '0');
      ++q;
      any = true;
    }
    if (any) m.shape.push_back(v);
  }
  m.captured_at_ns =
      static_cast<uint64_t>(jc::FindInt(obj, "captured_at_ns"));
  return m;
}

void EmitSnapshot(const dmi_sink::SinkSnapshot& s, std::string* out) {
  out->append("{\"submitted_records\":" + std::to_string(s.submitted_records));
  const uint64_t* fields[] = {
      &s.admitted_records, &s.admitted_bytes, &s.dropped_records,
      &s.timed_out_records, &s.oversized_records, &s.duplicate_records,
      &s.rejected_closed_records, &s.persisted_records, &s.packs_persisted,
      &s.packed_bytes, &s.flush_size, &s.flush_records, &s.flush_linger,
      &s.flush_session, &s.flush_manual, &s.flush_shutdown, &s.failures,
      &s.queue_records, &s.queue_bytes, &s.queue_peak_records,
      &s.queue_peak_bytes,
  };
  static const char* names[] = {
      "admitted_records", "admitted_bytes", "dropped_records",
      "timed_out_records", "oversized_records", "duplicate_records",
      "rejected_closed_records", "persisted_records", "packs_persisted",
      "packed_bytes", "flush_size", "flush_records", "flush_linger",
      "flush_session", "flush_manual", "flush_shutdown", "failures",
      "queue_records", "queue_bytes", "queue_peak_records",
      "queue_peak_bytes",
  };
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    out->append(",\"");
    out->append(names[i]);
    out->append("\":" + std::to_string(*fields[i]));
  }
  out->push_back('}');
}

}  // namespace

int main() {
  std::string line;
  std::ios::sync_with_stdio(false);
  std::unique_ptr<dmi_sink::PackSink> sink;
  while (std::getline(std::cin, line)) {
    const std::string op = jc::FindString(line, "op");
    if (op == "open") {
      dmi_sink::SinkConfig config;
      config.spool_root = jc::FindString(line, "root");
      config.spool_max_bytes =
          static_cast<uint64_t>(jc::FindInt(line, "max_bytes"));
      config.max_queue_records =
          static_cast<uint64_t>(jc::FindInt(line, "max_queue_records"));
      config.max_queue_bytes =
          static_cast<uint64_t>(jc::FindInt(line, "max_queue_bytes"));
      config.max_pack_bytes =
          static_cast<uint64_t>(jc::FindInt(line, "max_pack_bytes"));
      config.max_pack_records =
          static_cast<uint64_t>(jc::FindInt(line, "max_pack_records"));
      config.max_linger_ns =
          static_cast<uint64_t>(jc::FindInt(line, "max_linger_ns"));
      config.overload =
          jc::FindString(line, "overload") == "block"
              ? dmi_sink::Overload::kBlock
              : dmi_sink::Overload::kDropNewest;
      // admission_timeout arrives as a JSON number (possibly -1 or 0.5);
      // parse the raw text to keep the fraction.
      {
        const size_t at = line.find("\"admission_timeout\"");
        if (at != std::string::npos) {
          size_t q = line.find(':', at) + 1;
          config.admission_timeout_s = std::stod(line.substr(q));
        }
      }
      sink = std::make_unique<dmi_sink::PackSink>(config);
      std::string spool_error;
      const std::string err = sink->Start(&spool_error);
      if (!err.empty()) {
        std::string out = "{\"ok\":false,\"what\":";
        jc::EscapeJson(err, &out);
        std::cout << out << "}\n";
        sink.reset();
        continue;
      }
      std::cout << "{\"ok\":true}\n";
      continue;
    }
    if (op == "object_key") {
      const std::string key = dmi_sink::ObjectKeyFor(
          jc::FindString(line, "tenant_id"),
          jc::FindString(line, "session_id"),
          static_cast<uint64_t>(jc::FindInt(line, "producer_rank")),
          static_cast<uint64_t>(jc::FindInt(line, "captured_at_ns")),
          jc::FindString(line, "pack_id"));
      std::string out = "{\"ok\":true,\"object_key\":";
      jc::EscapeJson(key, &out);
      std::cout << out << "}\n";
      continue;
    }
    if (!sink) {
      std::cout << "{\"ok\":false,\"what\":\"sink is not open\"}\n";
      continue;
    }
    if (op == "submit") {
      const dmi_pack::RecordMetadata metadata =
          ParseMetadata(jc::FindObject(line, "metadata"));
      std::vector<uint8_t> payload;
      jc::DecodeBase64(jc::FindString(line, "payload_b64"), &payload);
      const dmi_sink::Admission admission =
          sink->Submit(metadata, payload.data(), payload.size());
      std::string out = "{\"ok\":true,\"admission\":\"";
      out += dmi_sink::AdmissionName(admission);
      std::cout << out << "\"}\n";
      continue;
    }
    if (op == "flush") {
      double timeout = -1.0;
      {
        const size_t at = line.find("\"timeout\"");
        if (at != std::string::npos) {
          size_t q = line.find(':', at) + 1;
          timeout = std::stod(line.substr(q));
        }
      }
      std::string error;
      const bool ok = sink->Flush(timeout, &error);
      if (ok) {
        std::cout << "{\"ok\":true}\n";
      } else if (error.empty()) {
        std::cout << "{\"ok\":false,\"what\":\"timeout\"}\n";
      } else {
        std::string out = "{\"ok\":false,\"what\":";
        jc::EscapeJson(error, &out);
        std::cout << out << "}\n";
      }
      continue;
    }
    if (op == "close") {
      double timeout = -1.0;
      {
        const size_t at = line.find("\"timeout\"");
        if (at != std::string::npos) {
          size_t q = line.find(':', at) + 1;
          timeout = std::stod(line.substr(q));
        }
      }
      std::string error;
      const dmi_sink::SinkSnapshot snapshot = sink->Close(timeout, &error);
      sink.reset();
      std::string out = "{\"ok\":true,\"snapshot\":";
      EmitSnapshot(snapshot, &out);
      if (!error.empty()) {
        out += ",\"error\":";
        jc::EscapeJson(error, &out);
      }
      std::cout << out << "}\n";
      continue;
    }
    if (op == "snapshot") {
      std::string out = "{\"ok\":true,\"snapshot\":";
      EmitSnapshot(sink->Snapshot(), &out);
      std::cout << out << "}\n";
      continue;
    }
    std::cout << "{\"ok\":false,\"what\":\"unknown op\"}\n";
  }
  return 0;
}
