#include "lease_coordinator.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>

namespace dmi_catalog {

namespace {

}  // namespace

std::string new_uuid_v4() {
  static std::mt19937_64 rng(std::random_device{}());
  uint64_t a = rng(), b = rng();
  a = (a & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;   // version 4
  b = (b & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;   // variant 10
  char out[37];
  std::snprintf(out, sizeof(out),
                "%08x-%04x-%04x-%04x-%012llx",
                static_cast<unsigned>(a >> 32),
                static_cast<unsigned>((a >> 16) & 0xFFFF),
                static_cast<unsigned>(a & 0xFFFF),
                static_cast<unsigned>(b >> 48),
                static_cast<unsigned long long>(b & 0xFFFFFFFFFFFFull));
  return out;
}

LeaseCoordinator::LeaseCoordinator(
    std::shared_ptr<const ClickHouseClient> client, LeaseConfig config)
    : client_(std::move(client)), config_(std::move(config)) {
  table_ = "`" + config_.database + "`.`" + config_.table_prefix +
           "_publisher_lease`";
}

std::map<std::string, std::string> LeaseCoordinator::quorum_write() const {
  if (!config_.insert_quorum.has_value()) return {};
  return {{"insert_quorum", std::to_string(*config_.insert_quorum)},
          {"insert_quorum_parallel", "0"},
          {"insert_quorum_timeout",
           std::to_string(config_.publish_timeout_ns / 1'000'000)}};
}

PublisherLease LeaseCoordinator::acquire(const std::string& holder) {
  // Bounded in BYTES, which is what the column stores and what the
  // message promises (the Python port validates the same way).
  if (holder.empty() || holder.size() > 256) {
    throw CatalogError(CatalogError::Kind::kValue,
                       "holder must be a non-empty string of at most "
                       "256 bytes");
  }
  const PublisherLease* held = lease();
  return claim(holder, held != nullptr ? held->lease_id : new_uuid_v4());
}

PublisherLease LeaseCoordinator::renew() {
  const PublisherLease* held = lease();
  if (held == nullptr) {
    throw CatalogError(
        CatalogError::Kind::kLease,
        "no publisher lease is held; call acquire_publisher_lease() "
        "before publishing. Only the lease holder can make a snapshot "
        "visible, and the check rides inside the publish statement, so "
        "publishing without one writes nothing.");
  }
  return claim(held->holder, held->lease_id);
}

void LeaseCoordinator::release() {
  const PublisherLease* held = lease();
  if (held == nullptr) return;
  client_->execute(release_statement(),
                   {{"term", held->term},
                    {"lease_id", held->lease_id},
                    {"holder", held->holder}},
                   // A deciding WRITE like the claim: the successor's head
                   // read is what this row is written for.
                   quorum_write());
  lease_.reset();
}

std::string LeaseCoordinator::release_statement() const {
  return (
      "INSERT INTO " + table_ + " "
      "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
      "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
      "now_ns, now_ns "
      "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)");
}

PublisherLease LeaseCoordinator::claim(const std::string& holder,
                                       const std::string& lease_id) {
  return claim_with_rival(holder, lease_id, std::nullopt);
}

PublisherLease LeaseCoordinator::claim_contested(
    const std::string& holder, const std::string& rival_lease_id) {
  return claim_with_rival(holder, new_uuid_v4(), rival_lease_id);
}

PublisherLease LeaseCoordinator::claim_with_rival(
    const std::string& holder, const std::string& lease_id,
    std::optional<std::string> rival_lease_id) {
  const LeaseHead current = head();
  reject_live(current, lease_id);
  const uint64_t term = current.term + 1;
  insert(term, lease_id, holder);
  if (rival_lease_id.has_value()) {
    // The contested-claim scenario: a rival row lands between the
    // claimant's INSERT and its read-back, the way the Python live suite
    // injects it through a client wrapper. Same term, long TTL.
    insert(term, *rival_lease_id, "rival", 600'000'000'000ull);
  }
  const std::vector<Row> rows = client_->execute(
      "SELECT toString(lease_id), acquired_at_ns, expires_at_ns "
      "FROM " + table_ + " WHERE term = %(term)s",
      {{"term", term}}, deciding_read());
  std::set<std::string> owners;
  for (const Row& row : rows) owners.insert(row[0]);
  if (owners == std::set<std::string>{lease_id}) {
    lease_ = PublisherLease{
        term, lease_id, holder,
        parse_u64_field(rows[0][1], "lease acquisition"),
        parse_u64_field(rows[0][2], "lease expiry")};
    return *lease_;
  }
  lease_.reset();
  reject_live(head(), lease_id);
  throw CatalogError(CatalogError::Kind::kLease,
                     "publisher lease claim was not recorded");
}

LeaseHead LeaseCoordinator::head() const {
  const std::string table = table_;
  const std::vector<Row> rows = client_->execute(
      "SELECT term, toString(lease_id), any(holder), min(expires_at_ns), "
      "toUnixTimestamp64Nano(now64(9)) FROM " + table + " "
      "WHERE term = (SELECT max(term) FROM " + table + ") "
      "GROUP BY term, lease_id ORDER BY lease_id DESC",
      {}, deciding_read());
  if (rows.empty()) return LeaseHead{};
  LeaseHead out;
  out.term = parse_u64_field(rows[0][0], "lease term");
  out.claimants = rows.size();
  out.lease_id = rows[0][1];
  out.holder = rows[0][2];
  out.expires_at_ns = parse_u64_field(rows[0][3], "lease expiry");
  out.now_ns = parse_u64_field(rows[0][4], "server clock");
  for (const Row& row : rows) {
    out.live_until_ns = std::max(
        out.live_until_ns, parse_u64_field(row[3], "lease expiry"));
  }
  return out;
}

std::string LeaseCoordinator::fence() const {
  const std::string table = table_;
  return (
      "(SELECT count() FROM ("
      "SELECT lease_id, min(expires_at_ns) AS expires_at_ns "
      "FROM " + table + " "
      "WHERE term = (SELECT max(term) FROM " + table + ") "
      "GROUP BY lease_id ORDER BY lease_id DESC LIMIT 1"
      ") WHERE lease_id = toUUID(%(lease_id)s) "
      "AND expires_at_ns > "
      "toUInt64(toUnixTimestamp64Nano(now64(9))) "
      "+ toUInt64(%(publish_timeout_ns)s) "
      "+ toUInt64(%(clock_skew_ns)s)) = 1");
}

bool LeaseCoordinator::fence_eval(const std::string& lease_id,
                                  uint64_t publish_timeout_ns,
                                  uint64_t clock_skew_ns) const {
  // A deciding read: the answer decides whether the fence admits, and the
  // fence's head subquery read from a replica behind on the lease table
  // would admit or deny on stale state.
  const std::vector<Row> rows = client_->execute(
      "SELECT " + fence(),
      {{"lease_id", lease_id},
       {"publish_timeout_ns", publish_timeout_ns},
       {"clock_skew_ns", clock_skew_ns}},
      deciding_read());
  return !rows.empty() && rows[0][0] == "1";
}

void LeaseCoordinator::reject_if_gone() const {
  const PublisherLease* held = lease();
  if (held == nullptr) {
    throw CatalogError(CatalogError::Kind::kLease, "no lease is held");
  }
  const LeaseHead current = head();
  if (current.lease_id == held->lease_id) return;
  throw CatalogError(
      CatalogError::Kind::kLease,
      "publisher lease " + held->lease_id + " (term " +
          std::to_string(held->term) + ") no longer stands at the head of "
          "`" + config_.database + "`.`" + config_.table_prefix +
          "_publisher_lease`, which is now term " +
          std::to_string(current.term) + " held by '" + current.holder +
          "': the publish was fenced out and made no snapshot visible.");
}

void LeaseCoordinator::insert(uint64_t term, const std::string& lease_id,
                              const std::string& holder,
                              std::optional<uint64_t> ttl_ns) const {
  client_->execute(
      "INSERT INTO " + table_ + " "
      "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
      "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
      "now_ns, now_ns + toUInt64(%(ttl_ns)s) "
      "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
      {{"term", term},
       {"lease_id", lease_id},
       {"holder", holder},
       {"ttl_ns", ttl_ns.value_or(config_.lease_ttl_ns)}},
      quorum_write());
}

void LeaseCoordinator::reject_live(const LeaseHead& head,
                                   const std::string& lease_id) {
  if (head.live_until_ns <= head.now_ns ||
      (head.claimants == 1 && head.lease_id == lease_id)) {
    return;
  }
  lease_.reset();
  if (head.claimants > 1) {
    throw CatalogError(
        CatalogError::Kind::kHeld,
        "publisher lease term " + std::to_string(head.term) + " on "
            "`" + config_.database + "`.`" + config_.table_prefix +
            "_*` is contested and remains unavailable for another " +
            std::to_string(head.live_until_ns - head.now_ns) +
            " ns. A claimant may have completed its ownership read-back "
            "before the competing row arrived, so no higher term is safe "
            "until every claim at this term expires.");
  }
  throw CatalogError(
      CatalogError::Kind::kHeld,
      "publisher lease on `" + config_.database + "`.`" +
          config_.table_prefix +
          "_*` is held by '" + head.holder + "' (lease " + head.lease_id +
          ", term " + std::to_string(head.term) + ") for another " +
          std::to_string(head.expires_at_ns - head.now_ns) +
          " ns. Only one publisher may make snapshots visible; wait for "
          "it to expire or stop it.");
}

}  // namespace dmi_catalog
