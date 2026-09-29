// Bounded parallel uploader: staged packs → object store → spool removal.
//
// Mirrors ParallelSpoolUploader + SpoolUploader (spool.py): recover the
// spool, admit packs under an in-flight byte cap, upload over N workers with
// per-pack retry and backoff, remove each pack only after its object is
// confirmed, and report per-position outcomes (never keyed by pack id).
//
// Upload integrity per pack (same as the Python put()):
//   1. Preflight HEAD: an object already carrying this pack's size and
//      checksum is re-read and re-hashed before it is blessed (metadata
//      alone is an assertion, not proof). Match → no upload, remove staged.
//   2. The upload stream itself is hashed as curl reads it; a source whose
//      bytes contradict the staged checksum deletes the upload and fails.
//   3. Post-upload HEAD must show the object, else the upload is refused.
//
// Cancellation (set_cancellation): the spool listing stops between packs,
// no pack starts once cancelled, and a pack's retries and their backoff end
// at once. A cancelled pack stays in the spool and is reported as
// cancelled, not failed.

#ifndef DMI_STORE_UPLOADER_H_
#define DMI_STORE_UPLOADER_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "cancel.h"
#include "s3_client.h"
#include "spool.h"

namespace dmi_store {

struct UploaderConfig {
  int max_workers = 4;
  uint64_t max_in_flight_bytes = 256ull * 1024 * 1024;
  int max_attempts = 4;
  // Backoff between attempts: base * 2^attempt, capped. Jitter matches the
  // Python contract shape (uniform [1-jitter, 1+jitter]); tests pin outcomes,
  // not sleep durations.
  double base_backoff_s = 0.25;
  double max_backoff_s = 10.0;
  double jitter_ratio = 0.2;
  std::string store_id = "s3";
  std::string content_type = "application/vnd.dmi.pack";
};

struct PackRef {
  std::string pack_id;
  std::string store_id;
  std::string object_key;
  uint64_t object_bytes = 0;
  std::string checksum;
  uint64_t record_count = 0;
};

struct UploadFailure {
  std::string pack_id;
  std::string object_key;
  int attempts = 0;
  std::string error;
  // Cut short by a cancel: before it started, between its attempts, or in
  // a request the cancel cut. The pack is still staged. Not counted in
  // failed_packs. Attempts that ran out on failures of their own stay
  // failures, even when a cancel came in meanwhile.
  bool cancelled = false;
};

struct UploadSnapshot {
  uint64_t attempted_packs = 0;
  uint64_t uploaded_packs = 0;
  uint64_t uploaded_bytes = 0;
  uint64_t failed_packs = 0;
  uint64_t cancelled_packs = 0;  // left staged by a cancel
  uint64_t retries = 0;
  uint64_t peak_active_uploads = 0;
  uint64_t peak_in_flight_bytes = 0;
  uint64_t duration_count = 0;
  uint64_t duration_total_ns = 0;
  uint64_t duration_max_ns = 0;
};

struct UploadBatchResult {
  // refs[i] pairs with failures[i]: exactly one is set, in recover() order.
  std::vector<PackRef> refs;
  std::vector<UploadFailure> failures;
  UploadSnapshot snapshot;
  // A cancel cut the spool listing short, before any upload started: the
  // batch is empty whatever the spool holds.
  bool listing_cancelled = false;
};

class SpoolUploader {
 public:
  SpoolUploader(Spool* spool, S3Client* client, UploaderConfig config);

  SpoolUploader(const SpoolUploader&) = delete;
  SpoolUploader& operator=(const SpoolUploader&) = delete;

  // From now on, once `cancel` is cancelled, UploadPending stops listing
  // the spool (listing_cancelled, if it had not listed it all) and starts
  // no pack, and UploadOne stops retrying (its backoff wakes for it); the
  // pack stays staged. A transfer in flight is cut only when the S3 client
  // has the same Cancellation (S3Client::set_cancellation). nullptr: none.
  // Keep `cancel` alive as long as the uploader.
  void set_cancellation(const Cancellation* cancel) { cancel_ = cancel; }

  // Recover the spool and upload every entry (or the first `limit`, which
  // must be positive when set). A pack over max_in_flight_bytes is recorded
  // as a failure at its own position and the rest of the batch still
  // uploads; the return carries no batch-level error. Note this DIVERGES
  // from the Python uploader, which refuses the whole batch up front and
  // uploads nothing — see the comment at the byte gate in uploader.cpp.
  UploadBatchResult UploadPending(int limit = -1);

  // Upload one staged entry with retry. Public for tests. *cancelled_out
  // (when given) says whether a cancel ended it.
  bool UploadOne(const StagedPack& staged, PackRef* ref, int* attempts_out,
                 std::string* error, bool* cancelled_out = nullptr);

 private:
  Spool* spool_;
  S3Client* client_;
  UploaderConfig config_;
  const Cancellation* cancel_ = nullptr;
};

}  // namespace dmi_store

#endif  // DMI_STORE_UPLOADER_H_
