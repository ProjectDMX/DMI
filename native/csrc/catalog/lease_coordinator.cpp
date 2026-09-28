#include "lease_coordinator.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>

namespace dmi_catalog {

namespace {

// A Seconds setting from whole milliseconds. Whole seconds wherever the
// value allows -- what every server parses, and what publish_timeout_ns
// already requires -- rounded DOWN, so the server's cap never exceeds the
// client's deadline. Only a time under a second goes out as a fraction,
// which current servers parse, for max_execution_time and
// lock_acquire_timeout alike (checked on 25.12). Never "0": ClickHouse reads
// a zero max_execution_time as no limit at all.
std::string seconds_setting(uint64_t ms) {
  if (ms >= 1000) return std::to_string(ms / 1000);
  ms = std::max<uint64_t>(ms, 1);
  char out[8];
  std::snprintf(out, sizeof(out), "0.%03u", static_cast<unsigned>(ms));
  std::string text(out);
  while (text.back() == '0') text.pop_back();
  return text;
}

}  // namespace

const char* const kLeaseDeadlineBound =
    "the lease deadline (lease_ttl_s, less clock_skew_s and a 0.1 s margin, "
    "after the claim that stamped the lease row was sent; "
    "clickhouse_request_timeout_s caps each request as well)";

uint64_t lease_deadline_ns(uint64_t sent_ns, uint64_t lease_ttl_ns,
                           uint64_t clock_skew_ns) {
  const uint64_t spent = clock_skew_ns + kLeaseDeadlineMarginNs;
  return lease_ttl_ns > spent ? sent_ns + (lease_ttl_ns - spent) : sent_ns;
}

uint64_t renewal_window_ns(uint64_t lease_ttl_ns, uint64_t clock_skew_ns) {
  const uint64_t spent = clock_skew_ns + kLeaseDeadlineMarginNs;
  const uint64_t half = lease_ttl_ns / 2;
  return half > spent ? half - spent : 0;
}

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
  const double request_ns = client_->request_timeout_s() * 1e9;
  claim_bound_ns_ = std::max<uint64_t>(
      request_ns < static_cast<double>(config_.lease_ttl_ns / 3)
          ? static_cast<uint64_t>(request_ns)
          : config_.lease_ttl_ns / 3,
      1'000'000);
  claim_bound_text_ =
      "the bound on each request of a claim made without a live lease, "
      "min(clickhouse_request_timeout_s, lease_ttl_s / 3) = " +
      std::to_string(claim_bound_ns_ / 1'000'000) + " ms";
}

std::vector<Row> LeaseCoordinator::run(
    const std::string& query, const Params& params,
    std::map<std::string, std::string> settings, bool write) const {
  // Held, and its deadline still ahead: that deadline, shared by every
  // request until the lease renews. Otherwise each request gets the claim
  // bound from when it starts -- with no lease, and with one whose deadline
  // has passed, whose row can no longer be counted on to keep rivals out.
  // A request made under such a lease is a claim like any other, decided by
  // the protocol's own reads: a renewal that meets a successor is refused,
  // and a publish is fenced out. Refusing to send it at all would turn those
  // known outcomes into unknown ones. What must not happen is a holder going
  // on as though it still held the lease, and the storage service abandons
  // one before it makes another request (storage_service.h).
  const uint64_t now = steady_now_ns();
  const bool live = lease_.has_value() && lease_->deadline_ns > now;
  const RequestDeadline deadline(
      live ? lease_->deadline_ns : now + claim_bound_ns_,
      live ? std::string(kLeaseDeadlineBound) : claim_bound_text_);
  if (write) add_write_caps(&settings);
  return client_->execute(query, params, settings);
}

void LeaseCoordinator::add_write_caps(
    std::map<std::string, std::string>* settings) const {
  // A lease INSERT the client gave up on (a timeout, so an unknown outcome)
  // must not land afterwards: a claim row stamped then outlives the
  // quarantine meant to cover it. So the server gets the time left before
  // the request's deadline -- the tightest in force, as the client computes
  // it -- and abandons the statement then: max_execution_time for running
  // it, lock_acquire_timeout for waiting on the table lock before it starts,
  // and throw, not break, since break would insert what had been read so
  // far. The quorum wait is bounded by insert_quorum_timeout alone, so that
  // is capped too, as well as by publish_timeout as before: past the
  // deadline nobody is listening.
  //
  // What the caps do not cover: ClickHouse checks max_execution_time only at
  // designated points while the pipeline runs, so the part commit can
  // overrun it, and its clock starts when the server starts the query, not
  // when the client sent it -- a request held up in transit can still land
  // up to that delay after the client gave up.
  const uint64_t deadline = RequestDeadline::current();  // run() set one
  const uint64_t now = steady_now_ns();
  // 0 when the deadline has passed; execute() then sends nothing.
  const uint64_t left_ms = deadline > now ? (deadline - now) / 1'000'000 : 0;
  if (config_.insert_quorum.has_value()) {
    (*settings)["insert_quorum"] = std::to_string(*config_.insert_quorum);
    (*settings)["insert_quorum_parallel"] = "0";
    (*settings)["insert_quorum_timeout"] = std::to_string(std::max<uint64_t>(
        std::min<uint64_t>(config_.publish_timeout_ns / 1'000'000, left_ms),
        1));
  }
  (*settings)["max_execution_time"] = seconds_setting(left_ms);
  (*settings)["lock_acquire_timeout"] = seconds_setting(left_ms);
  (*settings)["timeout_overflow_mode"] = "throw";
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
  // A deciding WRITE like the claim: the successor's head read is what this
  // row is written for.
  run(release_statement(),
      {{"term", held->term},
       {"lease_id", held->lease_id},
       {"holder", held->holder}},
      {}, true);
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
  claim_insert_sent_ = false;
  const LeaseHead current = head();
  reject_live(current, lease_id);
  const uint64_t term = current.term + 1;
  // Taken before the INSERT goes out, so never after the server stamps the
  // row: the new lease's deadline counts from here.
  const uint64_t sent_ns = steady_now_ns();
  claim_insert_sent_ = true;
  try {
    insert(term, lease_id, holder);
  } catch (const ClickHouseError& exc) {
    // Nothing reached the server when no attempt connected, or when the
    // deadline had passed before the INSERT could go out.
    claim_insert_sent_ = exc.sent();
    throw;
  }
  if (rival_lease_id.has_value()) {
    // The contested-claim scenario: a rival row lands between the
    // claimant's INSERT and its read-back, the way the Python live suite
    // injects it through a client wrapper. Same term, long TTL.
    insert(term, *rival_lease_id, "rival", 600'000'000'000ull);
  }
  // Under the old lease's deadline on a renewal: its row is what keeps
  // rivals out until this one is confirmed.
  const std::vector<Row> rows = run(
      "SELECT toString(lease_id), acquired_at_ns, expires_at_ns "
      "FROM " + table_ + " WHERE term = %(term)s",
      {{"term", term}}, deciding_read(), false);
  std::set<std::string> owners;
  for (const Row& row : rows) owners.insert(row[0]);
  if (owners == std::set<std::string>{lease_id}) {
    // rows[0] is this claim's own row, unordered and ungrouped though the
    // read is: every row at this term carries our lease_id, and we wrote one
    // (the client repeats an INSERT only when its connection was never made,
    // so nothing reached the server, and CatalogWriter quarantines after one
    // of unknown outcome rather than retrying). The expiry rule is
    // the minimum expires_at_ns under (term, lease_id), which differs from
    // any one row only once a release's tombstone shares the key, and none
    // can here: a release writes at the holder's own term and a claim always
    // goes to head + 1, so no claim lands on a term that already holds its
    // own tombstone.
    lease_ = PublisherLease{
        term, lease_id, holder,
        parse_u64_field(rows[0][1], "lease acquisition"),
        parse_u64_field(rows[0][2], "lease expiry"),
        sent_ns,
        lease_deadline_ns(sent_ns, config_.lease_ttl_ns,
                          config_.clock_skew_ns)};
    return *lease_;
  }
  lease_.reset();
  reject_live(head(), lease_id);
  throw CatalogError(CatalogError::Kind::kLease,
                     "publisher lease claim was not recorded");
}

LeaseHead LeaseCoordinator::head() const {
  const std::string table = table_;
  const std::vector<Row> rows = run(
      "SELECT term, toString(lease_id), any(holder), min(expires_at_ns), "
      "toUnixTimestamp64Nano(now64(9)) FROM " + table + " "
      "WHERE term = (SELECT max(term) FROM " + table + ") "
      "GROUP BY term, lease_id ORDER BY lease_id DESC",
      {}, deciding_read(), false);
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
  const std::vector<Row> rows = run(
      "SELECT " + fence(),
      {{"lease_id", lease_id},
       {"publish_timeout_ns", publish_timeout_ns},
       {"clock_skew_ns", clock_skew_ns}},
      deciding_read(), false);
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
  run("INSERT INTO " + table_ + " "
      "(term, lease_id, holder, acquired_at_ns, expires_at_ns) "
      "SELECT toUInt64(%(term)s), toUUID(%(lease_id)s), %(holder)s, "
      "now_ns, now_ns + toUInt64(%(ttl_ns)s) "
      "FROM (SELECT toUnixTimestamp64Nano(now64(9)) AS now_ns)",
      {{"term", term},
       {"lease_id", lease_id},
       {"holder", holder},
       {"ttl_ns", ttl_ns.value_or(config_.lease_ttl_ns)}},
      {}, true);
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
