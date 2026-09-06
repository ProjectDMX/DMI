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
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "clickhouse_client.h"

namespace dmi_catalog {

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
};

struct LeaseHead {
  uint64_t term = 0;
  size_t claimants = 0;
  std::string lease_id;
  std::string holder;
  uint64_t expires_at_ns = 0;
  uint64_t live_until_ns = 0;
  uint64_t now_ns = 0;
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
    kSchema
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
  std::map<std::string, std::string> quorum_write() const;

  std::shared_ptr<const ClickHouseClient> client_;
  LeaseConfig config_;
  std::optional<PublisherLease> lease_;
  std::string table_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_LEASE_COORDINATOR_H
