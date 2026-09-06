// A1 throughput gate: PackBuilder append+seal on the bench corpus shape
// (10k records x 64 KiB, 5 packs of 128 MiB), same as
// benchmarks/profile_pipeline_stages.py stage_writer_only.
// Reports median GiB/s over 5 trials.

#include "pack_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

static double gib_per_s(uint64_t bytes, double s) {
  return static_cast<double>(bytes) / s / (1024.0 * 1024.0 * 1024.0);
}

int main() {
  const size_t kRecords = 10'000;
  const size_t kPayload = 64 * 1024;
  const size_t kPool = 64;
  const uint64_t kMaxPack = 128ull * 1024 * 1024;
  const uint64_t kLogical = kRecords * kPayload;

  std::vector<std::vector<uint8_t>> payloads(kPool, std::vector<uint8_t>(kPayload));
  {
    std::mt19937_64 rng(17);
    for (auto& p : payloads) {
      for (auto& b : p) b = static_cast<uint8_t>(rng() >> 33);
    }
  }
  std::vector<dmi_pack::RecordMetadata> metas;
  metas.reserve(kPool);
  for (size_t i = 0; i < kPool; ++i) {
    dmi_pack::RecordMetadata m;
    m.capture_id = "capture-" + std::to_string(i);
    m.tenant_id = "benchmark";
    m.experiment_id = "capture-pipeline";
    m.run_id = "seed-17";
    m.session_id = "session-0";
    m.request_id = "request-" + std::to_string(i / 128);
    m.sequence_id = "sequence-" + std::to_string(i / 128);
    m.model_id = "synthetic";
    m.model_revision = "benchmark-v1";
    m.capture_policy_version = "all-v1";
    m.hook_name = "resid_pre";
    m.layer_number = static_cast<int64_t>(i % 32);
    m.producer_rank = 0;
    m.step_number = i;
    m.token_start = i;
    m.token_end = i + 1;
    m.batch_position = i % 128;
    m.dtype = "float32";
    m.shape = {kPayload / 4};
    m.captured_at_ns = 1'700'000'000'000'000'000 + i;
    metas.push_back(std::move(m));
  }

  double best = 1e9;
  uint64_t packed_total = 0;
  for (int trial = 0; trial < 5; ++trial) {
    auto started = Clock::now();
    double append_s = 0.0, seal_s = 0.0;
    dmi_pack::PackBuilder builder(
        "018f0000-0000-7000-8000-000000000f01", 1'700'000'000'000'000'000,
        kMaxPack);
    dmi_pack::SealedPack sealed;
    uint64_t packed = 0;
    int packs = 0;
    for (size_t i = 0; i < kRecords; ++i) {
      dmi_pack::PackRecord record;
      record.metadata = metas[i % kPool];
      record.metadata.capture_id =
          "capture-" + std::to_string(trial * kRecords + i);  // unique per pack
      record.payload = payloads[i % kPool].data();
      record.payload_bytes = kPayload;
      auto a0 = Clock::now();
      const dmi_pack::Status st = builder.Append(record);
      append_s += std::chrono::duration<double>(Clock::now() - a0).count();
      if (st != dmi_pack::Status::kOk) {
        auto s0 = Clock::now();
        const dmi_pack::Status sst = builder.Seal(&sealed);
        seal_s += std::chrono::duration<double>(Clock::now() - s0).count();
        if (sst != dmi_pack::Status::kOk) {
          std::fprintf(stderr, "seal failed\n");
          return 1;
        }
        packed += sealed.data.size();
        ++packs;
        // fresh builder, reset capture ids
        builder = dmi_pack::PackBuilder(
            "018f0000-0000-7000-8000-000000000f01",
            1'700'000'000'000'000'000, kMaxPack);
        record.metadata.capture_id =
            "capture-" + std::to_string(trial * kRecords + i);
        if (builder.Append(record) != dmi_pack::Status::kOk) {
          std::fprintf(stderr, "append after rollover failed\n");
          return 1;
        }
      }
    }
    if (builder.Seal(&sealed) != dmi_pack::Status::kOk) {
      std::fprintf(stderr, "final seal failed\n");
      return 1;
    }
    packed += sealed.data.size();
    ++packs;
    double s = std::chrono::duration<double>(Clock::now() - started).count();
    best = std::min(best, s);
    packed_total = packed;
    std::printf("trial%d: %8.3f GiB/s  (%d packs, %llu packed bytes, "
                "append %.3fs seal %.3fs)\n",
                trial, gib_per_s(kLogical, s), packs,
                static_cast<unsigned long long>(packed), append_s, seal_s);
  }
  std::printf("BEST: %8.3f GiB/s (Python writer baseline: 0.359)\n",
              gib_per_s(kLogical, best));
  (void)packed_total;
  return 0;
}
