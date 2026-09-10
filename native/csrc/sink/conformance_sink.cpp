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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../common/json.h"
#include "object_key.h"
#include "record_row.h"

namespace jc = dmi_common;

namespace {

// The key of the first integer literal on this line that does not fit in the
// 64-bit union, empty when there was none. One latch per line is enough: the
// driver is single-threaded and handles one op per line.
//
// The since-removed FindInt wrapper reported such a literal as -1, and every
// site here read that as
// something legal: the unsigned `open` limits cast it to
// 18446744073709551615, `num_workers` fell back to 1, ObjectKeyFor rendered
// rank=18446744073709551615, and a metadata counter wrapped modulo 2**64 and
// was admitted. SubmitRow's own parse already refuses (record_row.cpp); these
// are the driver's remaining decodes.
std::string g_out_of_range;

// A field the oracle types as SIGNED must pass IntDomain::kSigned: the
// union's two's-complement bit pattern is a legal-looking negative to such a
// field, and 18446744073709551615 arrived at layer_number as -1 -- its own
// "no layer" sentinel -- and was admitted.
int64_t Integer(const std::string& text, const char* key,
                jc::IntDomain domain = jc::IntDomain::kUnion) {
  int64_t value = 0;
  const jc::IntFind found = jc::FindIntChecked(text, key, &value, domain);
  if (found == jc::IntFind::kOutOfRange && g_out_of_range.empty()) {
    g_out_of_range = key;
  }
  // kAbsent keeps answering -1: layer_number == -1 is legal, and an absent
  // num_workers still means "one".
  return found == jc::IntFind::kOk ? value : -1;
}

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
  m.layer_number = Integer(obj, "layer_number", jc::IntDomain::kSigned);
  m.producer_rank = static_cast<uint64_t>(Integer(obj, "producer_rank"));
  m.step_number = static_cast<uint64_t>(Integer(obj, "step_number"));
  m.token_start = static_cast<uint64_t>(Integer(obj, "token_start"));
  m.token_end = static_cast<uint64_t>(Integer(obj, "token_end"));
  m.batch_position =
      static_cast<uint64_t>(Integer(obj, "batch_position"));
  m.dtype = jc::FindString(obj, "dtype");
  for (const auto& item : jc::SplitElements(
           jc::Unwrap(jc::FindArray(obj, "shape")))) {
    size_t q = 0;
    while (q < item.size() && item[q] == ' ') ++q;
    // Bounded like the scalars above, and for the same reason: this
    // accumulator had none, so a dimension over 2**32 wrapped INTO range and
    // [2**32 + 1] was admitted and persisted as (1,) where CaptureMetadata
    // raises "shape dimensions must be integers in [0, 2^31 - 1]". The bound
    // is the ACCUMULATOR's, not the field's -- it refuses a literal with no
    // uint32 to hold it and leaves 0 .. 2**31 - 1 to the pack layer, exactly
    // as FindIntChecked bounds at 64 bits and leaves the field's range to
    // ValidateMetadata. The report rides the existing out-of-range latch, so
    // it lands before Submit the way every other refusal here does.
    uint32_t v = 0;
    bool any = false;
    bool over = false;
    while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
      const uint32_t digit = static_cast<uint32_t>(item[q] - '0');
      // "v * 10 + digit > UINT32_MAX", rearranged to not overflow itself.
      if (v > (~uint32_t{0} - digit) / 10) over = true;
      if (!over) v = v * 10 + digit;
      ++q;
      any = true;
    }
    if (over && g_out_of_range.empty()) g_out_of_range = "shape";
    if (any) m.shape.push_back(v);
  }
  m.captured_at_ns =
      static_cast<uint64_t>(Integer(obj, "captured_at_ns"));
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
      &s.queue_peak_bytes, &s.stage_packs, &s.stage_bytes,
      &s.stage_peak_packs, &s.stage_peak_bytes,
  };
  static const char* names[] = {
      "admitted_records", "admitted_bytes", "dropped_records",
      "timed_out_records", "oversized_records", "duplicate_records",
      "rejected_closed_records", "persisted_records", "packs_persisted",
      "packed_bytes", "flush_size", "flush_records", "flush_linger",
      "flush_session", "flush_manual", "flush_shutdown", "failures",
      "queue_records", "queue_bytes", "queue_peak_records",
      "queue_peak_bytes", "stage_packs", "stage_bytes", "stage_peak_packs",
      "stage_peak_bytes",
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
  // The refusal for an integer literal wider than 64 bits, in the same
  // ok:false/what shape the driver already answers "unknown op" and
  // "sink is not open" with.
  const auto refuse_out_of_range = [] {
    std::string out = "{\"ok\":false,\"what\":";
    jc::EscapeJson("integer field " + g_out_of_range + " is out of range",
                   &out);
    std::cout << out << "}\n";
  };
  while (std::getline(std::cin, line)) {
    g_out_of_range.clear();
    const std::string op = jc::FindString(line, "op");
    if (op == "open") {
      dmi_sink::SinkConfig config;
      config.spool_root = jc::FindString(line, "root");
      config.spool_max_bytes =
          static_cast<uint64_t>(Integer(line, "max_bytes"));
      config.max_queue_records =
          static_cast<uint64_t>(Integer(line, "max_queue_records"));
      config.max_queue_bytes =
          static_cast<uint64_t>(Integer(line, "max_queue_bytes"));
      config.max_pack_bytes =
          static_cast<uint64_t>(Integer(line, "max_pack_bytes"));
      config.max_pack_records =
          static_cast<uint64_t>(Integer(line, "max_pack_records"));
      config.max_linger_ns =
          static_cast<uint64_t>(Integer(line, "max_linger_ns"));
      config.overload =
          jc::FindString(line, "overload") == "block"
              ? dmi_sink::Overload::kBlock
              : dmi_sink::Overload::kDropNewest;
      const int64_t workers = Integer(line, "num_workers");
      config.num_workers = static_cast<int>(workers > 0 ? workers : 1);
      // Before the sink exists: a limit that cannot be represented must not
      // be silently replaced by UINT64_MAX or by the one-worker fallback.
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
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
      // Parsed before the call, not inside its argument list: a key built
      // from a wrapped rank is a key Python's builder never mints.
      const uint64_t producer_rank =
          static_cast<uint64_t>(Integer(line, "producer_rank"));
      const uint64_t captured_at_ns =
          static_cast<uint64_t>(Integer(line, "captured_at_ns"));
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      const std::string key = dmi_sink::ObjectKeyFor(
          jc::FindString(line, "tenant_id"),
          jc::FindString(line, "session_id"), producer_rank, captured_at_ns,
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
      // CaptureMetadata raises on these, so the mapping path refuses them
      // too rather than admit a counter that wrapped modulo 2**64. This is
      // the same refusal SubmitRow already makes on the row path.
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      std::vector<uint8_t> payload;
      jc::DecodeBase64(jc::FindString(line, "payload_b64"), &payload);
      const dmi_sink::Admission admission =
          sink->Submit(metadata, payload.data(), payload.size());
      std::string out = "{\"ok\":true,\"admission\":\"";
      out += dmi_sink::AdmissionName(admission);
      std::cout << out << "\"}\n";
      continue;
    }
    if (op == "submit_row") {
      // Row path (metadata JSON string + declared envelope dtype/shape),
      // exactly what the torch adapter feeds SubmitRow per envelope row.
      dmi_sink::RowInput input;
      input.metadata_json = jc::FindString(line, "metadata_json");
      std::vector<uint8_t> payload;
      jc::DecodeBase64(jc::FindString(line, "payload_b64"), &payload);
      input.payload = payload.data();
      input.payload_bytes = payload.size();
      input.dtype_name = jc::FindString(line, "dtype");
      for (const auto& item : jc::SplitElements(
               jc::Unwrap(jc::FindArray(line, "shape")))) {
        size_t q = 0;
        while (q < item.size() && item[q] == ' ') ++q;
        // The fourth accumulation of this class, and the only one over a
        // SIGNED accumulator: it wraps modulo 2**64, so the 32-bit witnesses
        // that catch the three uint32 sites do not reach it -- 2**32 + 1
        // fits an int64 and correctly mismatches the metadata. What aliases
        // is a literal past 2**63 - 1, and it aliases ONTO the metadata's own
        // dimension, which is exactly what carried it past the
        // envelope/metadata agreement check in SubmitRow and into a pack:
        // envelope [2**64 + 1] against metadata shape [1] answered ok and
        // persisted the record. Signed overflow is undefined besides, so the
        // bound has to be tested before the multiply rather than after.
        int64_t v = 0;
        bool any = false;
        bool over = false;
        while (q < item.size() && item[q] >= '0' && item[q] <= '9') {
          const int64_t digit = static_cast<int64_t>(item[q] - '0');
          // "v * 10 + digit > INT64_MAX", rearranged to not overflow itself.
          if (v > (INT64_MAX - digit) / 10) over = true;
          if (!over) v = v * 10 + digit;
          ++q;
          any = true;
        }
        if (over && g_out_of_range.empty()) g_out_of_range = "shape";
        if (any) input.shape.push_back(v);
      }
      // Before SubmitRow: an envelope dimension with no int64 to hold it is
      // not a shape mismatch to report, it is a literal the protocol cannot
      // carry, and nothing may be admitted on it.
      if (!g_out_of_range.empty()) {
        refuse_out_of_range();
        continue;
      }
      std::string detail;
      const dmi_sink::RowStatus status =
          dmi_sink::SubmitRow(*sink, input, &detail);
      std::string out = "{\"ok\":";
      out += (status == dmi_sink::RowStatus::kOk) ? "true" : "false";
      if (status == dmi_sink::RowStatus::kOk) {
        out += "}";
      } else {
        out += ",\"status\":";
        jc::EscapeJson(dmi_sink::RowStatusName(status), &out);
        out += ",\"what\":";
        jc::EscapeJson(detail, &out);
        out += "}";
      }
      std::cout << out << "\n";
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
