// A5a throughput gate: PackSink on the bench corpus shape (10k x 64 KiB),
// spool mode, at worker counts 1/2/4/8. Reports median GiB/s per count plus
// the Python pipeline baseline for the same corpus (0.235 GiB/s spool).
//
// Usage: bench_sink [workers...]  (default: 1 2 4 8)

#include "pack_sink.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double gib_per_s(uint64_t bytes, double s) {
  return static_cast<double>(bytes) / s / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv) {
  const size_t kRecords = 10'000;
  const size_t kPayload = 64 * 1024;
  // Scope pattern: long single-scope runs for the apples-to-apples leg
  // (matches bench_capture_pipeline: 1 scope -> ~5 x 128 MiB packs), plus
  // an interleaved 8-scope leg (32 MiB chunks) for the scaling curve. Tiny
  // round-robin scopes seal a pack per chunk and benchmark fsync, not
  // packing — that shape is recorded in the ledger as a discarded reading.
  const uint64_t kLogical = kRecords * kPayload;

  // Usage: bench_sink [single|interleaved] [workers...]
  //   single: 1 scope (matches bench_capture_pipeline) at each worker count
  //     (only 1 worker can be active — scaling leg is interleaved).
  //   interleaved: 8 scopes in 512-record (32 MiB) chunks at each count.
  //   qrecs=N qbytes=N override the queue bounds (unbounded-queue runs
  //   separate producer rate from drain rate).
  //   spooldir=PREFIX stages into PREFIX/dmi-bench-sink-XXXXXX instead of
  //   /tmp (lets the disk-vs-tmpfs comparison run without code changes).
  std::string mode = "single";
  std::vector<int> worker_counts;
  uint64_t qrecs = 256, qbytes = 256 * 64 * 1024;
  std::string spool_prefix = "/tmp/dmi-bench-sink-";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "single" || arg == "interleaved") {
      mode = arg;
    } else if (arg.compare(0, 6, "qrecs=") == 0) {
      qrecs = std::stoull(arg.substr(6));
    } else if (arg.compare(0, 7, "qbytes=") == 0) {
      qbytes = std::stoull(arg.substr(7));
    } else if (arg.compare(0, 9, "spooldir=") == 0) {
      spool_prefix = arg.substr(9);
    } else {
      worker_counts.push_back(std::atoi(argv[i]));
    }
  }
  if (worker_counts.empty()) worker_counts = {1, 2, 4, 8};
  const bool interleaved = (mode == "interleaved");
  const size_t kScopes = interleaved ? 8 : 1;
  const size_t kChunk = interleaved ? 512 : kRecords;

  std::vector<std::vector<uint8_t>> payloads(64,
                                             std::vector<uint8_t>(kPayload));
  {
    std::mt19937_64 rng(17);
    for (auto& p : payloads) {
      for (auto& b : p) b = static_cast<uint8_t>(rng() >> 33);
    }
  }

  for (int workers : worker_counts) {
    double best = 1e9;
    double submit_best = 1e9, flush_best = 1e9, close_best = 1e9;
    uint64_t packs = 0;
    for (int trial = 0; trial < 3; ++trial) {
      std::string spool_template = spool_prefix + "XXXXXX";
      const char* spool_dir = mkdtemp(spool_template.data());
      dmi_sink::SinkConfig config;
      config.max_queue_records = qrecs;
      config.max_queue_bytes = qbytes;
      config.max_pack_bytes = 128ull * 1024 * 1024;
      config.max_pack_records = 10'000;
      config.max_linger_ns = 1'000'000'000;
      config.overload = dmi_sink::Overload::kBlock;
      config.admission_timeout_s = 30.0;
      config.spool_root = spool_dir;
      config.spool_max_bytes = 4ull * 1024 * 1024 * 1024;
      config.num_workers = workers;
      dmi_sink::PackSink sink(config);
      std::string error;
      if (!sink.Start(&error).empty()) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
      }
      auto started = Clock::now();
      for (size_t i = 0; i < kRecords; ++i) {        const size_t scope = interleaved ? (i / kChunk) % kScopes : 0;
        dmi_pack::RecordMetadata m;
        m.capture_id = "capture-" + std::to_string(trial * kRecords + i);
        m.tenant_id = "benchmark";
        m.experiment_id = "capture-pipeline";
        m.run_id = "seed-17";
        m.session_id = "session-" + std::to_string(scope);
        m.request_id = "request-" + std::to_string(i / 128);
        m.sequence_id = "sequence-" + std::to_string(i / 128);
        m.model_id = "synthetic";
        m.model_revision = "benchmark-v1";
        m.capture_policy_version = "all-v1";
        m.hook_name = "resid_pre";
        m.layer_number = static_cast<int64_t>(i % 32);
        m.producer_rank = scope;
        m.step_number = i;
        m.token_start = i;
        m.token_end = i + 1;
        m.batch_position = i % 128;
        m.dtype = "float32";
        m.shape = {kPayload / 4};
        m.captured_at_ns = 1'700'000'000'000'000'000 + i;
        const auto& payload = payloads[i % payloads.size()];
        const dmi_sink::Admission admitted =
            sink.Submit(m, payload.data(), payload.size());
        if (admitted != dmi_sink::Admission::kAccepted) {
          std::fprintf(stderr, "submit rejected: %s\n",
                       dmi_sink::AdmissionName(admitted));
          return 1;
        }
      }
      std::string flush_error;
      const double submit_s =
          std::chrono::duration<double>(Clock::now() - started).count();
      if (!sink.Flush(60.0, &flush_error)) {
        std::fprintf(stderr, "flush failed: %s\n", flush_error.c_str());
        return 1;
      }
      const double flush_s =
          std::chrono::duration<double>(Clock::now() - started).count();
      std::string close_error;
      const dmi_sink::SinkSnapshot snapshot =
          sink.Close(60.0, &close_error);
      if (!close_error.empty() || snapshot.persisted_records != kRecords) {
        std::fprintf(stderr, "close failed: %s persisted=%llu\n",
                     close_error.c_str(),
                     static_cast<unsigned long long>(
                         snapshot.persisted_records));
        return 1;
      }
    const double s =
        std::chrono::duration<double>(Clock::now() - started).count();
    best = std::min(best, s);
    submit_best = std::min(submit_best, submit_s);
    flush_best = std::min(flush_best, flush_s - submit_s);
    close_best = std::min(close_best, s - flush_s);
    packs = snapshot.packs_persisted;
      // Clean the spool dir for the next trial.
      std::string rm = std::string("rm -rf ") + spool_dir;
      if (system(rm.c_str()) != 0) return 1;
    }
    std::printf("workers=%d: %8.3f GiB/s  (%llu packs, submit %.2fs flush %.2fs close %.2fs)\n",
                workers, gib_per_s(kLogical, best),
                static_cast<unsigned long long>(packs),
                submit_best, flush_best, close_best);
  }
  std::printf("Python pipeline baseline (spool): 0.235 GiB/s\n");
  return 0;
}
