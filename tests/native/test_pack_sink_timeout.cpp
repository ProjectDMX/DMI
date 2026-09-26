// The kBlock admission timeout, exercised without racing the packer.
//
// The refusal a timeout guards is "the queue is full and nobody is
// draining it". Filling the queue with a live packer thread is a race:
// the packer pops as fast as we push. So the pipeline is wedged bottom-up
// instead, through the same seam test_spool_reservations.cpp uses:
//
//   Spool::SetStageHookForTesting blocks the STAGER inside Stage()      (1)
//   the stage queue (stage_queue_packs = 1) then fills with one pack    (2)
//   the packer seals its next pack and blocks in SealForStager          (3)
//   the admission queue (max_queue_records = 1) fills behind it         (4)
//
// After (4) no admission can succeed until the hook is released: the only
// thread that could free queue space is the packer, the packer can only
// move once the stager pops, and the stager is parked in the hook. A
// further kBlock submit with a finite admission_timeout_s must therefore
// return kTimedOut and count timed_out_records — deterministically, with
// no sleep-based guessing about where the packer is.
//
// Every earlier submit is admitted on its FIRST bounds check (the test
// waits for the pipeline to visibly quiesce between submits), so the
// finite timeout never touches them.
//
// A second case pins the ENVELOPE deadline: the ring hands the sink one
// envelope of N rows per submit, and its admission as a whole must be bounded
// by one admission_timeout_s -- not one timeout per row, which let a steadily
// slow sink hold the record worker (and, through it, the forward) for N
// timeouts. There the stager is slowed rather than wedged, so each row after
// the pipeline fills waits a little under one timeout for room.
//
// Built and run by tests/test_native_pack_sink_timeout.py.

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sink/pack_sink.h"
#include "sink/record_row.h"

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

dmi_pack::RecordMetadata Metadata(int n) {
  dmi_pack::RecordMetadata m;
  m.capture_id = "capture-" + std::to_string(n);
  m.tenant_id = "tenant-a";
  m.experiment_id = "exp-1";
  m.run_id = "run-1";
  m.session_id = "session-1";
  m.request_id = "req-1";
  m.sequence_id = "seq-1";
  m.model_id = "model-1";
  m.model_revision = "rev-1";
  m.capture_policy_version = "policy-1";
  m.hook_name = "hook.0";
  m.dtype = "float32";
  m.shape = {4};
  m.captured_at_ns = 1700000000000000000ull + n;
  return m;
}

dmi_sink::Admission SubmitRecord(dmi_sink::PackSink& sink, int n) {
  const uint8_t payload[16] = {static_cast<uint8_t>(n)};
  return sink.Submit(Metadata(n), payload, sizeof(payload));
}

// Poll the snapshot until `done` holds. Bounded patience, not a race: the
// state waited for is reached by threads that are runnable, and failing to
// reach it within 30s is itself a pipeline bug worth failing on.
bool WaitForSnapshot(
    dmi_sink::PackSink& sink,
    const std::function<bool(const dmi_sink::SinkSnapshot&)>& done) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    if (done(sink.Snapshot())) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void TestBlockedPipelineTimesOutAndCountsIt() {
  const char* base = std::getenv("SPOOL_TEST_ROOT");
  const std::string root =
      std::string(base != nullptr ? base : "/tmp") + "/sink-timeout";
  fs::remove_all(root);

  dmi_sink::SinkConfig config;
  config.spool_root = root;
  config.num_workers = 1;
  config.max_queue_records = 1;   // (4) one record fills admission
  config.max_pack_records = 1;    // every record seals into its own pack
  config.stage_queue_packs = 1;   // (2) one sealed pack fills the stage
  config.max_linger_ns = 3600ull * 1000 * 1000 * 1000;  // linger never fires
  config.overload = dmi_sink::Overload::kBlock;
  config.admission_timeout_s = 0.2;

  dmi_sink::PackSink sink(config);
  std::string start_error = sink.Start();
  CHECK(start_error.empty());
  if (!start_error.empty()) {
    std::cerr << "start failed: " << start_error << "\n";
    return;
  }

  std::mutex mutex;
  std::condition_variable cv;
  bool in_hook = false, release = false;
  sink.SpoolForTesting().SetStageHookForTesting([&] {
    std::unique_lock<std::mutex> lock(mutex);
    in_hook = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  });

  // R1 seals into pack 1; the stager pops it and parks in the hook (1).
  CHECK(SubmitRecord(sink, 1) == dmi_sink::Admission::kAccepted);
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return in_hook; });
  }
  CHECK(WaitForSnapshot(sink, [](const dmi_sink::SinkSnapshot& s) {
    return s.queue_records == 0 && s.stage_packs == 0;
  }));

  // R2 seals into pack 2, which fills the stage queue (2).
  CHECK(SubmitRecord(sink, 2) == dmi_sink::Admission::kAccepted);
  CHECK(WaitForSnapshot(sink, [](const dmi_sink::SinkSnapshot& s) {
    return s.queue_records == 0 && s.stage_packs == 1;
  }));

  // R3 seals into pack 3; the packer blocks handing it over (3). Once the
  // packer has popped R3 it cannot pop again before pushing pack 3, and it
  // cannot push until the wedged stager pops — so waiting for the pop is
  // enough; no guess about scheduler timing decides the outcome.
  CHECK(SubmitRecord(sink, 3) == dmi_sink::Admission::kAccepted);
  CHECK(WaitForSnapshot(sink, [](const dmi_sink::SinkSnapshot& s) {
    return s.queue_records == 0;
  }));

  // R4 fills the admission queue behind the wedged packer (4).
  CHECK(SubmitRecord(sink, 4) == dmi_sink::Admission::kAccepted);

  // R5 must wait, and the wait can never be satisfied: kTimedOut.
  CHECK(SubmitRecord(sink, 5) == dmi_sink::Admission::kTimedOut);
  {
    const dmi_sink::SinkSnapshot snap = sink.Snapshot();
    CHECK(snap.timed_out_records == 1);
    CHECK(snap.submitted_records == 5);
    CHECK(snap.admitted_records == 4);
    CHECK(snap.dropped_records == 0);
    CHECK(snap.oversized_records == 0);
    CHECK(snap.queue_records == 1);  // R4 still waiting on the packer
  }

  // Release the stager and drain: every ADMITTED record still lands, the
  // timed-out one stays counted, and close neither hangs nor latches.
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
    cv.notify_all();
  }
  std::string close_error;
  const dmi_sink::SinkSnapshot final_snap = sink.Close(-1.0, &close_error);
  CHECK(close_error.empty());
  if (!close_error.empty()) std::cerr << close_error << "\n";
  CHECK(final_snap.timed_out_records == 1);
  CHECK(final_snap.admitted_records == 4);
  CHECK(final_snap.persisted_records == 4);
  CHECK(final_snap.packs_persisted == 4);
  CHECK(final_snap.failures == 0);
  CHECK(final_snap.queue_records == 0);
  CHECK(final_snap.stage_packs == 0);
}

std::string MetadataJson(int n) {
  return std::string("{\"capture_id\": \"envelope-") + std::to_string(n) +
         "\", \"tenant_id\": \"tenant-a\", \"experiment_id\": \"exp-1\", "
         "\"run_id\": \"run-1\", \"session_id\": \"session-1\", "
         "\"request_id\": \"req-1\", \"sequence_id\": \"seq-1\", "
         "\"model_id\": \"model-1\", \"model_revision\": \"rev-1\", "
         "\"adapter_revision\": null, \"capture_policy_version\": "
         "\"policy-1\", \"hook_name\": \"hook.0\", \"layer_number\": 0, "
         "\"producer_rank\": 0, \"step_number\": " + std::to_string(n) +
         ", \"token_start\": 0, \"token_end\": 1, \"batch_position\": 0, "
         "\"dtype\": \"float32\", \"shape\": [4], \"captured_at_ns\": " +
         std::to_string(1700000000000000000ull + n) + "}";
}

void TestAnEnvelopeSharesOneAdmissionDeadline() {
  const char* base = std::getenv("SPOOL_TEST_ROOT");
  const std::string root =
      std::string(base != nullptr ? base : "/tmp") + "/sink-envelope";
  fs::remove_all(root);

  constexpr double kTimeoutS = 0.5;
  // Each pack takes a little over half a timeout to stage, so every row
  // that has to wait for room waits well under one timeout: row by row,
  // none of them would ever time out.
  constexpr auto kStage = std::chrono::milliseconds(300);
  constexpr int kRows = 10;

  dmi_sink::SinkConfig config;
  config.spool_root = root;
  config.num_workers = 1;
  config.max_queue_records = 1;
  config.max_pack_records = 1;
  config.stage_queue_packs = 1;
  config.max_linger_ns = 3600ull * 1000 * 1000 * 1000;
  config.overload = dmi_sink::Overload::kBlock;
  config.admission_timeout_s = kTimeoutS;

  dmi_sink::PackSink sink(config);
  const std::string start_error = sink.Start();
  CHECK(start_error.empty());
  if (!start_error.empty()) return;
  sink.SpoolForTesting().SetStageHookForTesting(
      [&] { std::this_thread::sleep_for(kStage); });

  const uint8_t payload[16] = {};
  std::vector<std::string> metadata;
  for (int n = 0; n < kRows; ++n) metadata.push_back(MetadataJson(n));

  const auto started = std::chrono::steady_clock::now();
  dmi_sink::EnvelopeAdmission envelope(sink);
  dmi_sink::RowStatus status = dmi_sink::RowStatus::kOk;
  std::string detail;
  int admitted = 0;
  for (int n = 0; n < kRows; ++n) {
    dmi_sink::RowInput row;
    row.metadata_json = metadata[n];
    row.payload = payload;
    row.payload_bytes = sizeof(payload);
    row.dtype_name = "float32";
    row.shape = {4};
    status = envelope.SubmitRow(row, &detail);
    if (status != dmi_sink::RowStatus::kOk) break;
    ++admitted;
  }
  const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();

  // One timeout for the whole envelope: a row times out once the
  // envelope's deadline passes, well before every row got in.
  CHECK(status == dmi_sink::RowStatus::kNotAccepted);
  CHECK(detail == "timed_out");
  CHECK(admitted < kRows);
  CHECK(elapsed_s >= kTimeoutS - 0.05);
  CHECK(elapsed_s < kTimeoutS + 0.25);
  if (status != dmi_sink::RowStatus::kNotAccepted ||
      elapsed_s >= kTimeoutS + 0.25) {
    std::cerr << "envelope: " << admitted << " rows admitted in "
              << elapsed_s << " s (status " << RowStatusName(status)
              << ")\n";
  }
  CHECK(sink.Snapshot().timed_out_records == 1);

  std::string close_error;
  const dmi_sink::SinkSnapshot final_snap = sink.Close(-1.0, &close_error);
  CHECK(close_error.empty());
  CHECK(final_snap.persisted_records == static_cast<uint64_t>(admitted));
}

}  // namespace

int main() {
  TestBlockedPipelineTimesOutAndCountsIt();
  TestAnEnvelopeSharesOneAdmissionDeadline();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
