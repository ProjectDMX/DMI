// B1a: the publisher lease coordinator, ported from
// src/dmi/storage/capture/clickhouse_lease.py.
//
// The SQL statements are the Python module's, textually: the same
// placeholders (`%(name)s`), substituted client-side by the HTTP client
// exactly as clickhouse-driver substitutes them, so the server receives
// byte-identical text from both implementations. The claim protocol is a
// sole-claimant read-back and only sound while the statements it runs are
// the audited ones — drift here re-opens the #119 surface the plan warns
// about. Comments in the Python module carry the derivations (why the
// release tombstone sits at the holder's own term, why the fence adds the
// skew bound); they are not duplicated here.

#ifndef DMI_CATALOG_LEASE_COORDINATOR_H
#define DMI_CATALOG_LEASE_COORDINATOR_H

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "clickhouse_client.h"

namespace dmi_catalog {

// The deadline on the publisher lease's requests.
//
// A lease row lives lease_ttl_ns from when the server stamps it, and a rival
// whose clock runs clock_skew_ns ahead sees it expire that much early. The
// server stamps it no earlier than the claim INSERT was sent, so a writer can
// count on its row keeping rivals out until
//
//   lease deadline = claim INSERT sent + lease_ttl_ns - clock_skew_ns - 0.1 s
//
// on its own steady clock, the 0.1 s (kLeaseDeadlineMarginNs) covering the
// time between a request failing and the writer saying so. Every request
// made while the lease is held has to be answered by then: the lease's own
// (a renewal's head read, claim INSERT and read-back, the release tombstone)
// here, and in the storage service every catalog request made under its
// lease lock (storage_service.h), since the lease cannot renew until that
// lock is let go. A renewal that cannot finish in time therefore fails, and
// quarantines the writer, while its row still keeps rivals out -- however
// the time is spread across its requests, so one slow but healthy request
// may use all of it. The deadline moves with each confirmed renewal, a
// publish's included, since each claim that stamps a row restarts it.
//
// A claim made without a lease has no row to protect yet. Each of its
// requests is bounded by min(the client's request timeout, lease_ttl_ns /
// 3): long enough for a slow catalog, short enough that a claim which hangs
// fails well inside a TTL. So is a request made under a lease whose deadline
// has already passed (run() says why it is still sent). From its INSERT on,
// a claim is also bounded by the deadline of the lease it takes: a claim
// confirmed after that would hand its holder a lease it could not use.
//
// Each attempt of a lease INSERT carries the time the client gives it -- the
// time left before its deadline, recomputed for a retry -- to the server as
// max_execution_time and lock_acquire_timeout, to the millisecond, and caps
// a quorum wait by it, so the server abandons a claim when the client does
// (up to the time the request took to reach the server).
constexpr uint64_t kLeaseDeadlineMarginNs = 100'000'000ull;

// The deadline of a lease whose claim INSERT was sent at `sent_ns` (steady
// clock); `sent_ns` itself when the skew and margin leave nothing.
uint64_t lease_deadline_ns(uint64_t sent_ns, uint64_t lease_ttl_ns,
                           uint64_t clock_skew_ns);

// How long a renewal has between when it starts, at the latest, and the
// lease deadline it must finish by. The storage service renews a third of
// the TTL after the claim that stamped the row was sent and looks every
// sixth, so a renewal starts within lease_ttl_ns / 2 of that send:
//
//   window = lease_ttl_ns / 2 - clock_skew_ns - kLeaseDeadlineMarginNs
//
// 0 when the skew leaves no time at all.
uint64_t renewal_window_ns(uint64_t lease_ttl_ns, uint64_t clock_skew_ns);

// The storage service refuses a skew whose renewal window is shorter than
// this (storage_service.cpp): below it a renewal cannot be expected to
// finish against a real server.
constexpr uint64_t kMinimumRenewalWindowNs = 200'000'000ull;

// What bounds a request made under the lease, for a timeout's message.
extern const char* const kLeaseDeadlineBound;

struct LeaseConfig {
  std::string database;
  std::string table_prefix;
  uint64_t lease_ttl_ns = 30'000'000'000ull;
  uint64_t publish_timeout_ns = 5'000'000'000ull;
  uint64_t clock_skew_ns = 0;
  // Empty = no quorum (single node). Mirrors `insert_quorum: int | None`.
  std::optional<uint64_t> insert_quorum;
};

struct PublisherLease {
  uint64_t term = 0;
  std::string lease_id;
  std::string holder;
  uint64_t acquired_at_ns = 0;
  uint64_t expires_at_ns = 0;
  // steady_clock ns: when the claim INSERT that stamped this row was sent,
  // and the lease deadline that follows from it (lease_deadline_ns).
  uint64_t sent_ns = 0;
  uint64_t deadline_ns = 0;
};

struct LeaseHead {
  uint64_t term = 0;
  size_t claimants = 0;
  std::string lease_id;
  std::string holder;
  uint64_t expires_at_ns = 0;
  uint64_t live_until_ns = 0;
  uint64_t now_ns = 0;
  std::vector<std::string> lease_ids;  // every claimant at the head term
};

class CatalogError : public std::runtime_error {
 public:
  enum class Kind {
    kValue,
    kHeld,
    kLease,
    kAllocation,
    kPublishRace,
    kPublishConflict,
    kQuarantined,
    kSchema,
    // index() refused a batch over max_estimated_bytes. Its own kind so a
    // caller that can split the batch need not match message text.
    kBatchTooLarge
  };

  CatalogError(Kind kind, const std::string& what)
      : std::runtime_error(what), kind_(kind) {}

  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

std::string new_uuid_v4();

class LeaseCoordinator {
 public:
  LeaseCoordinator(std::shared_ptr<const ClickHouseClient> client,
                   LeaseConfig config);

  const PublisherLease* lease() const {
    return lease_.has_value() ? &*lease_ : nullptr;
  }
  uint64_t ttl_ns() const { return config_.lease_ttl_ns; }
  // The bound on each request of a claim made without a live lease:
  // min(request timeout, lease_ttl_ns / 3).
  uint64_t claim_bound_ns() const { return claim_bound_ns_; }
  // Whether the last claim's INSERT may have reached the server. One that
  // failed before it -- its head read timed out, say -- or whose INSERT
  // never connected, or was not sent because its deadline had passed,
  // wrote nothing, whatever it failed with.
  bool claim_insert_sent() const { return claim_insert_sent_; }
  // Whether the last refusal (kHeld) came from claim rows this coordinator
  // itself inserted, and nobody else's: a claim whose request gave up but
  // which the server still executed, late. Such a row expires one TTL after
  // it landed and is no rival publisher.
  bool refused_by_own_claims() const { return refused_by_own_claims_; }

  PublisherLease acquire(const std::string& holder);
  PublisherLease renew();
  void release();
  void discard_local_lease() { lease_.reset(); }
  PublisherLease claim(const std::string& holder, const std::string& lease_id);
  // The claim protocol with a rival row inserted between the claim INSERT
  // and its read-back — the contested-claim scenario the Python live
  // suite drives through a client wrapper.
  PublisherLease claim_contested(const std::string& holder,
                                 const std::string& rival_lease_id);
  LeaseHead head() const;
  std::string fence() const;
  bool fence_eval(const std::string& lease_id, uint64_t publish_timeout_ns,
                  uint64_t clock_skew_ns) const;
  void reject_if_gone() const;
  std::string release_statement() const;

 private:
  PublisherLease claim_with_rival(
      const std::string& holder, const std::string& lease_id,
      std::optional<std::string> rival_lease_id);
  void insert(uint64_t term, const std::string& lease_id,
              const std::string& holder,
              std::optional<uint64_t> ttl_ns = std::nullopt) const;
  void reject_live(const LeaseHead& head, const std::string& lease_id);
  // Runs one lease statement under its deadline (see above): the held
  // lease's while that is still ahead, the claim bound from now otherwise.
  // A lease INSERT (`write`) also carries the time left to the server.
  std::vector<Row> run(const std::string& query, const Params& params,
                       std::map<std::string, std::string> settings,
                       bool write) const;
  // The server-side caps on a lease INSERT attempt given attempt_ms.
  void add_write_caps(uint64_t attempt_ms,
                      std::map<std::string, std::string>* settings) const;

  std::shared_ptr<const ClickHouseClient> client_;
  LeaseConfig config_;
  uint64_t claim_bound_ns_ = 0;
  std::string claim_bound_text_;  // claim_bound_ns_, for a timeout's message
  bool claim_insert_sent_ = false;
  std::optional<PublisherLease> lease_;
  std::string table_;
  // The lease_ids of the most recent claim INSERTs this coordinator sent,
  // whether or not they were confirmed: a claim that timed out may still
  // land. Only a claim from the last TTL or so can still be live, so a
  // short history covers it -- each id once, so the renewals of one lease,
  // which all claim the same id, cannot crowd the others out.
  std::deque<std::string> claimed_ids_;
  bool refused_by_own_claims_ = false;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_LEASE_COORDINATOR_H
