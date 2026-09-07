#include "uploader.h"

#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <random>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace dmi_store {

namespace {

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string Sha256HexBytes(const uint8_t* data, size_t n) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_CTX ctx;
  SHA256_Init(&ctx);
  SHA256_Update(&ctx, data, n);
  SHA256_Final(digest, &ctx);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(64);
  for (int i = 0; i < 32; ++i) {
    out[2 * i] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0xF];
  }
  return out;
}

std::vector<uint8_t> ReadFile(const std::string& path, std::string* error) {
  std::vector<uint8_t> data;
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error) *error = "cannot open " + path;
    return data;
  }
  constexpr size_t kChunk = 1 << 20;
  std::vector<uint8_t> chunk(kChunk);
  for (;;) {
    const ssize_t n = ::read(fd, chunk.data(), chunk.size());
    if (n < 0) {
      if (error) *error = "cannot read " + path;
      ::close(fd);
      return {};
    }
    if (n == 0) break;
    data.insert(data.end(), chunk.begin(), chunk.begin() + n);
  }
  ::close(fd);
  return data;
}

void SleepBackoff(const UploaderConfig& config, int attempt,
                  std::mt19937_64* rng) {
  double wait = config.base_backoff_s * (1 << std::min(attempt, 20));
  wait = std::min(wait, config.max_backoff_s);
  if (config.jitter_ratio > 0) {
    std::uniform_real_distribution<double> jitter(1.0 - config.jitter_ratio,
                                                  1.0 + config.jitter_ratio);
    wait *= jitter(*rng);
  }
  if (wait > 0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(wait));
  }
}

std::map<std::string, std::string> PackMetadata(const StagedPack& staged) {
  return {
      {"dmi-format", "dmi-pack-v1"},
      {"dmi-pack-id", staged.pack_id},
      {"dmi-sha256", staged.checksum},
      {"dmi-record-count", std::to_string(staged.record_count)},
      {"dmi-created-at-ns", std::to_string(staged.created_at_ns)},
  };
}

}  // namespace

SpoolUploader::SpoolUploader(Spool* spool, S3Client* client,
                             UploaderConfig config)
    : spool_(spool), client_(client), config_(std::move(config)) {}

bool SpoolUploader::UploadOne(const StagedPack& staged, PackRef* ref,
                              int* attempts_out, std::string* error) {
  const std::string& key = staged.object_key;
  std::mt19937_64 rng(
      static_cast<uint64_t>(std::hash<std::string>{}(staged.pack_id)));
  std::string last_error;
  int attempts = 0;
  for (int attempt = 0; attempt < config_.max_attempts; ++attempt) {
    ++attempts;
    if (attempt > 0) SleepBackoff(config_, attempt - 1, &rng);

    // 1. Preflight: an object already carrying this pack is re-read and
    // re-hashed before it is blessed — metadata alone is not proof.
    {
      std::string head_error;
      const ObjectHead head = client_->HeadObject(key, &head_error);
      if (!head_error.empty()) {
        last_error = head_error;
        continue;  // transport-level: retryable below
      }
      if (head.found) {
        const auto meta = head.metadata.find("dmi-sha256");
        if (head.size == staged.object_bytes && meta != head.metadata.end() &&
            meta->second == staged.checksum) {
          std::vector<uint8_t> existing;
          std::string get_error;
          if (!client_->GetRange(key, 0, staged.object_bytes, &existing,
                                 &get_error) ||
              Sha256HexBytes(existing.data(), existing.size()) !=
                  staged.checksum) {
            last_error = get_error.empty()
                             ? "pre-existing object failed verification"
                             : get_error;
            continue;
          }
          std::string remove_error;
          if (spool_->Remove(staged, &remove_error) != SpoolStatus::kOk) {
            last_error = remove_error;
            continue;
          }
          ref->pack_id = staged.pack_id;
          ref->store_id = config_.store_id;
          ref->object_key = key;
          ref->object_bytes = staged.object_bytes;
          ref->checksum = staged.checksum;
          ref->record_count = staged.record_count;
          if (attempts_out) *attempts_out = attempts;
          return true;
        }
        // The object exists but its size or checksum disagrees with the
        // staged pack: a DIFFERENT pack already owns this key. The
        // previous form fell through to the PUT below, which replaced the
        // existing object, removed the local pack, and reported success —
        // destroying someone else's immutable object and losing the local
        // record in one step. HEAD already detected the conflict before
        // this branch, so refusing is not a race: the conflict is with
        // durable state, and retrying would PUT over it again.
        last_error =
            "pack conflict: the object store already holds a different "
            "object at " + key + " (its size or dmi-sha256 disagrees with "
            "the staged pack " + staged.pack_id + "). The staged pack is "
            "retained in the spool for inspection; do not overwrite the "
            "existing object.";
        return false;  // NOT retryable
      }
    }

    // 2. Read + hash the staged bytes (the spool validated at stage time;
    // this confirms what the upload stream will carry).
    std::string read_error;
    const std::vector<uint8_t> data = ReadFile(staged.path, &read_error);
    if (!read_error.empty()) {
      last_error = read_error;
      continue;
    }
    if (data.size() != staged.object_bytes ||
        Sha256HexBytes(data.data(), data.size()) != staged.checksum) {
      // Corrupt staged bytes: no retry can fix local corruption, but report
      // it as the failure rather than uploading garbage.
      last_error = "staged bytes do not match the staged checksum";
      break;
    }
    std::string etag;
    std::string put_error;
    if (!client_->PutObject(key, data.data(), data.size(),
                            PackMetadata(staged), config_.content_type, &etag,
                            &put_error)) {
      last_error = put_error;
      continue;
    }
    // 3. Post-upload visibility: the object must be there.
    {
      std::string head_error;
      const ObjectHead head = client_->HeadObject(key, &head_error);
      if (!head_error.empty() || !head.found ||
          head.size != staged.object_bytes) {
        last_error = head_error.empty()
                         ? "uploaded object is not visible to HeadObject"
                         : head_error;
        continue;
      }
    }
    std::string remove_error;
    if (spool_->Remove(staged, &remove_error) != SpoolStatus::kOk) {
      last_error = remove_error;
      continue;
    }
    ref->pack_id = staged.pack_id;
    ref->store_id = config_.store_id;
    ref->object_key = key;
    ref->object_bytes = staged.object_bytes;
    ref->checksum = staged.checksum;
    ref->record_count = staged.record_count;
    if (attempts_out) *attempts_out = attempts;
    return true;
  }
  if (attempts_out) *attempts_out = attempts;
  if (error) *error = last_error;
  return false;
}

UploadBatchResult SpoolUploader::UploadPending(int limit) {
  UploadBatchResult result;
  if (limit != -1 && limit <= 0) return result;  // invalid limit: empty result
  std::vector<StagedPack> pending;
  {
    std::string error;
    if (spool_->Recover(&pending, &error) != SpoolStatus::kOk) {
      return result;
    }
  }
  if (limit != -1 && static_cast<size_t>(limit) < pending.size()) {
    pending.resize(static_cast<size_t>(limit));
  }
  // Both vectors are positional from the start: sized to the recover()
  // order up front, oversized refusals written into their own slot, and
  // workers below fill the rest by index.
  result.refs.assign(pending.size(), PackRef{});
  result.failures.assign(pending.size(), UploadFailure{});
  for (size_t i = 0; i < pending.size(); ++i) {
    if (pending[i].object_bytes > config_.max_in_flight_bytes) {
      // A pack that can never be admitted must fail the batch loudly, not
      // stall it: same rule as the Python uploader's up-front refusal.
      result.failures[i] = {pending[i].pack_id, pending[i].object_key, 0,
                            "pack exceeds the in-flight byte limit"};
      ++result.snapshot.attempted_packs;
      ++result.snapshot.failed_packs;
    }
  }

  struct Slot {
    size_t index = 0;
  };
  std::deque<Slot> remaining;
  for (size_t i = 0; i < pending.size(); ++i) {
    if (pending[i].object_bytes <= config_.max_in_flight_bytes) {
      Slot slot;
      slot.index = i;
      remaining.push_back(slot);
    }
  }

  std::mutex mutex;
  std::condition_variable cv;
  uint64_t in_flight = 0;
  uint64_t active = 0;
  // Outcomes by position (never by pack id): two ready files may share a
  // pack id, and keying by id could report a failed upload as succeeded.
  std::vector<bool> finished(pending.size(), false);
  for (size_t i = 0; i < pending.size(); ++i) {
    if (pending[i].object_bytes > config_.max_in_flight_bytes) {
      finished[i] = true;
    }
  }

  auto worker = [&] {
    for (;;) {
      Slot slot;
      const StagedPack* staged = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex);
        // Wait until there is nothing left or some remaining pack fits
        // under the in-flight cap alongside current work. in_flight only
        // shrinks under this mutex with a notify, so no wakeup is missed;
        // the scan after the wait always finds the pack the predicate saw.
        cv.wait(lock, [&] {
          if (remaining.empty()) return true;
          for (const auto& candidate : remaining) {
            if (in_flight +
                    pending[candidate.index].object_bytes <=
                config_.max_in_flight_bytes) {
              return true;
            }
          }
          return false;
        });
        if (remaining.empty()) return;
        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
          if (in_flight + pending[it->index].object_bytes <=
              config_.max_in_flight_bytes) {
            slot = *it;
            remaining.erase(it);
            break;
          }
        }
        staged = &pending[slot.index];
        in_flight += staged->object_bytes;
        ++active;
        result.snapshot.peak_active_uploads =
            std::max(result.snapshot.peak_active_uploads, active);
        result.snapshot.peak_in_flight_bytes =
            std::max(result.snapshot.peak_in_flight_bytes, in_flight);
      }
      const int64_t started = NowNs();
      PackRef ref;
      std::string error;
      int attempts = 0;
      const bool ok = UploadOne(*staged, &ref, &attempts, &error);
      const int64_t elapsed = NowNs() - started;
      {
        std::lock_guard<std::mutex> lock(mutex);
        in_flight -= staged->object_bytes;
        --active;
        ++result.snapshot.attempted_packs;
        result.snapshot.duration_count += 1;
        result.snapshot.duration_total_ns += static_cast<uint64_t>(elapsed);
        result.snapshot.duration_max_ns = std::max(
            result.snapshot.duration_max_ns, static_cast<uint64_t>(elapsed));
        if (ok) {
          result.refs[slot.index] = std::move(ref);
          result.failures[slot.index] = {"", "", 0, ""};
          ++result.snapshot.uploaded_packs;
          result.snapshot.uploaded_bytes += staged->object_bytes;
          if (attempts > 1) {
            result.snapshot.retries += static_cast<uint64_t>(attempts - 1);
          }
        } else {
          result.refs[slot.index] = PackRef{};
          result.failures[slot.index] = {staged->pack_id, staged->object_key,
                                         attempts, error};
          ++result.snapshot.failed_packs;
          if (attempts > 1) {
            result.snapshot.retries += static_cast<uint64_t>(attempts - 1);
          }
        }
        finished[slot.index] = true;
      }
      cv.notify_all();
    }
  };

  const int workers =
      std::max(1, std::min(config_.max_workers,
                           static_cast<int>(remaining.size() + 1)));
  std::vector<std::thread> threads;
  for (int i = 0; i < workers; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) thread.join();
  return result;
}

}  // namespace dmi_store
